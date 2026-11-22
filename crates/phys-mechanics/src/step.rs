//! One mechanics step: interacting objects solved together by
//! minimising an incremental potential — once per stage of the integrator.

use core_math::{Mat3, Pose, Quat, Vec3};
use core_numerics::{BlockMatrix, Newton, NewtonReport, NewtonSettings, Objective, Sum, Triplets};
use core_runtime::{Pool, Profile};
use core_world::{ObjectId, Probe, ProbeReading, Representation, World};

use crate::contact::{Contact, Impact};
use crate::dofs::{Attach, Frame, Layout, RigidFrame, HELD};
use crate::elastic::{Elastic, SolidPlastic};
use crate::gravity::{MutualGravity, MutualGravitySettings};
use crate::inertia::{Gravity, Inertia};
use crate::joints::Joints;
use crate::rigid::{contact_points, free_rotation, gyration_radius, second_moment, RigidBodies};
use crate::term::{Context, Term};

#[path = "adjoint.rs"]
mod adjoint;
#[path = "conservation.rs"]
mod conservation;
#[path = "energy.rs"]
mod energy;
#[path = "islands.rs"]
mod islands;
#[path = "sleep.rs"]
mod sleep;
pub use adjoint::{AdjointGradients, AdjointLoss};
pub use energy::{EnergyLedger, MechanicalEnergy};

/// How positions and velocities advance. Every stage of every scheme
/// evaluates forces at the stage's end positions, so contact barriers stay
/// valid throughout.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Integrator {
    /// First order; damps all oscillation strongly (a 5 Hz bar at 60 steps a
    /// second keeps a quarter of its amplitude per period).
    BackwardEuler,
    /// Second order, two-step BDF; damps resolved oscillation noticeably
    /// (that bar keeps 88% per period).
    Bdf2,
    /// Average acceleration (trapezoidal); second order and conserves the
    /// energy of linear oscillation — but never damps modes too fast for
    /// the step, so stiff contact keeps ringing.
    Newmark,
    /// A trapezoidal stage then a BDF2 stage (Bank et al. 1985): second
    /// order, keeps resolved oscillation (that bar keeps 99.7% per period)
    /// and damps unresolved modes away (L-stable). Two solves per step.
    TrBdf2,
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Settings {
    /// Smooth-motion integrator. With the impact law enabled, active contact
    /// and its release use uniform backward-Euler stages within each island.
    pub integrator: Integrator,
    /// Contact precision ε, m; `None` for a thousandth of the smallest
    /// moving object's diagonal, capped at twice its thickness.
    pub contact_precision: Option<f64>,
    /// Newton stops when no unknown moves more than this fraction of ε.
    pub tolerance: f64,
    /// Newton iterations before a solve counts as failed (and its step is
    /// halved): a solve that converges takes well under ten.
    pub max_iterations: usize,
    /// Times per stage that friction's normal forces are refreshed from the
    /// solution and the stage solved again. Friction lagged from the start
    /// of a stage misses the force of a contact still settling (a box set
    /// down sliding brakes at 8% of μg its first step); a refresh fixes
    /// that. A refresh that does not converge — a jammed contact can read as
    /// a huge normal force — is discarded, keeping the solve before it.
    pub friction_updates: usize,
    /// Use contact-stage restarts and Newton's impact law for pairs that
    /// strike. Off only to study the smooth barrier without impact handling.
    pub impact_law: bool,
    /// Non-incident surface primitives of one deformable solid collide.
    /// Enabled by default; disable only for controlled numerical comparisons.
    pub self_contact: bool,
    /// Solve independent swept-contact components separately. Disable for
    /// comparison against one global coupled solve.
    pub islands: bool,
    /// Evaluate initial elastic/gravity acceleration for second-order smooth
    /// startup. Active contact keeps the dissipative contact rule.
    pub force_startup: bool,
    /// Consecutive FEM blocks per positive dense preconditioning patch (1..=16).
    /// One preserves nodal block Jacobi; rigid groups remain coupled by four.
    pub solid_block_size: usize,
    /// Optional accepted-stage work accounting; disabled to avoid extra
    /// energy quadrature in throughput runs. It never changes physical state.
    pub energy_accounting: bool,
    /// Dispatch independent islands on the persistent pool.
    pub parallel_islands: bool,
    /// Relative local angular-error budget for free islands; smaller steps
    /// recover accuracy. Anchors/active fixed contacts are excluded.
    pub angular_tolerance: Option<f64>,
    /// Reuse bitwise stationary discrete equilibria while inputs are unchanged.
    /// No approximate speed threshold is used.
    pub sleeping: bool,
    /// Put an island to sleep once it has settled: when neither its velocity
    /// nor its acceleration would move any of its points by more than the
    /// contact precision within this many seconds. Its velocities and
    /// acceleration become exactly zero and it costs nothing until something
    /// reaches it, a setting changes or its state is edited. Decided from
    /// checkpointed state only, so replay stays exact. Islands only (it needs
    /// [`Settings::islands`]); `None` keeps only exact stationary reuse.
    pub settle_time: Option<f64>,
    /// Solve the three constant translation modes exactly in free islands.
    pub translation_coarse: bool,
    /// Maximum halvings for an enabled angular-error audit (0..=12).
    pub accuracy_subdivisions: usize,
    /// Optional long-range mutual Newtonian gravitation over dynamic masses.
    pub mutual_gravity: Option<MutualGravitySettings>,
    /// Optional maximum substep duration (seconds) for an active island when
    /// [`Settings::islands`] is enabled (`None` to advance in one outer step).
    pub max_substep: Option<f64>,
    /// Optional proactive multi-rate CFL and joint-frequency subcycling factor
    /// for active islands when [`Settings::islands`] is enabled (`None` to
    /// disable adaptive subcycling).
    pub cfl_subcycling: Option<f64>,
    /// Outer Augmented-Lagrangian multiplier update passes over bilateral
    /// joints within each stage (`0` disables multiplier tightening).
    pub augmented_lagrangian_passes: usize,
}

impl Default for Settings {
    fn default() -> Self {
        Self {
            integrator: Integrator::TrBdf2,
            contact_precision: None,
            tolerance: 1e-3,
            max_iterations: 100,
            friction_updates: 1,
            impact_law: true,
            self_contact: true,
            islands: true,
            force_startup: true,
            solid_block_size: 1,
            energy_accounting: false,
            parallel_islands: true,
            angular_tolerance: None,
            sleeping: true,
            settle_time: Some(SETTLE_TIME),
            translation_coarse: true,
            accuracy_subdivisions: 8,
            mutual_gravity: None,
            max_substep: None,
            cfl_subcycling: None,
            augmented_lagrangian_passes: 0,
        }
    }
}

/// What a step did.
#[derive(Clone, Copy, Debug, Default)]
pub struct StepReport {
    /// The last solve's report.
    pub newton: NewtonReport,
    /// Newton solves over all stages.
    pub solves: usize,
    /// Newton iterations over all solves.
    pub iterations: usize,
    /// CG iterations across every attempted Newton solve, including retries.
    pub linear_iterations: usize,
    /// Sum of local block count × CG iterations, including discarded work.
    /// A size-weighted work indicator, not a floating-point operation count.
    pub linear_block_iterations: usize,
    /// Matrix-vector products including true-residual checks.
    pub linear_products: usize,
    /// Local blocks × matrix-vector products, including rejected work.
    pub linear_block_products: usize,
    /// Frictional contacts at the end.
    pub grips: usize,
    /// Whether any solve failed to converge.
    pub unconverged: bool,
    /// Pairs that struck and had the impact law applied.
    pub impacts: usize,
    /// Smooth stages repeated with a consistent Euler rule for active contact.
    /// Their attempted work is included in the solve counters.
    pub contact_restarts: usize,
    /// Times the step was halved after a failed solve.
    pub splits: usize,
    /// Friction refreshes that did not converge and were dropped.
    pub refreshes_discarded: usize,
    /// A stage could not start: its state was not a valid one.
    pub invalid: bool,
    /// Components in the final partition (fixed supports do not join them).
    pub islands: usize,
    /// Largest local solve, in three-component unknown blocks.
    pub largest_island_blocks: usize,
    /// Partition retries after an evaluated path found additional coupling.
    pub island_rebuilds: usize,
    /// Local full-interval attempts, including discarded partition attempts.
    pub island_solves: usize,
    /// Physical energy and discrete work over the accepted interval only.
    pub energy: Option<EnergyLedger>,
    /// A temporal angular-error budget was not met; the interval is rolled back.
    pub accuracy_failed: bool,
    /// Discarded interval attempts that exceeded the angular-error budget.
    pub accuracy_retries: usize,
    /// Sum of accepted local angular-error magnitudes, kg m²/s. On failure,
    /// describes the rejected attempt rather than an accepted trajectory.
    pub angular_error: f64,
    /// Anchors or active fixed contacts make the free-island audit inapplicable.
    pub boundary_forces: bool,
    /// Certified unchanged islands advanced without another Newton solve.
    pub sleeping_islands: usize,
    /// Largest proactive substep count taken by any island during this step.
    pub max_island_substeps: usize,
}

impl StepReport {
    /// Whether the requested interval was rejected and rolled back.
    pub fn failed(&self) -> bool {
        self.unconverged || self.invalid || self.accuracy_failed || self.newton.not_finite
    }
}

/// One implicit stage: how `x̃`, `β` and the new velocity follow from the
/// stage's start (`x₀`, `v₀`, `a₀`).
#[derive(Clone, Copy, Debug)]
enum Stage {
    /// `x = x₀ + d v`, `v = v₀ + d a`.
    Euler,
    /// `x = x₀ + d (v₀ + v)/2`, `v = v₀ + d (a₀ + a)/2`.
    Trapezoid,
    /// `x = (1+q) x₀ − q x₋ + c H v`, likewise for `v` with `a` — BDF2 over
    /// the earlier state `x₋` (in [`Mechanics::earlier`]) and the stage's
    /// start.
    Bdf2 { q: f64, c: f64, span: f64 },
}

/// The dynamic state a step changes, to undo it.
#[derive(Clone, Debug)]
pub struct MechanicsState {
    pub(crate) time: f64,
    pub(crate) nodes: Vec<(Vec<Vec3>, Vec<Vec3>)>,
    pub(crate) rigids: Vec<(core_math::Pose, Vec3, Vec3)>,
    pub(crate) acceleration: Option<Vec<f64>>,
    pub(crate) earlier: Option<(Vec<f64>, Vec<f64>)>,
    pub(crate) restart: Vec<bool>,
    pub(crate) affine: Vec<Option<(Quat, Mat3)>>,
    pub(crate) gravity_seen: Option<Vec3>,
    pub(crate) previous_h: Option<f64>,
    pub(crate) unloading: Vec<f64>,
    pub(crate) plastic: Vec<SolidPlastic>,
}

/// Newton iterations allowed a friction refresh.
const REFRESH_ITERATIONS: usize = 30;

/// How many times over a step may be halved after a failed solve.
pub const MAX_SPLITS: usize = 4;

/// The default [`Settings::settle_time`], seconds.
pub const SETTLE_TIME: f64 = 0.1;

const GAMMA: f64 = 2.0 - std::f64::consts::SQRT_2;

/// A stage is solved again with refreshed friction only if a friction
/// bound changed by more than this fraction.
const FRICTION_CHANGE: f64 = 0.05;

/// The mechanics of a world.
pub struct Mechanics {
    pub settings: Settings,
    /// Where the time went, by term and phase; cleared by the caller.
    pub profile: Profile,
    layout: Layout,
    frame: Frame,
    built_for: (usize, usize),
    inertia: Inertia,
    gravity: Gravity,
    mutual_gravity: MutualGravity,
    elastic: Elastic,
    rigid: RigidBodies,
    joints: Joints,
    contact: Contact,
    hessian: Sum<BlockMatrix, BlockMatrix>,
    triplets: Triplets,
    newton: Newton,
    /// Per block, at the start of the stage being solved.
    velocity: Vec<f64>,
    /// Per block, the acceleration at the end of the last stage, if any.
    acceleration: Option<Vec<f64>>,
    /// Positions and velocities a BDF2 stage reaches back to.
    earlier: Option<(Vec<f64>, Vec<f64>)>,
    /// `β²` per block for the stage being solved.
    beta2: Vec<f64>,
    /// Impact restart flags, retained per block for checkpoint compatibility.
    /// Any flag restarts the entire coupled solve with the Euler rule; mixing
    /// per-node rules breaks momentum exchange through internal forces.
    restart: Vec<bool>,
    stage: Stage,
    /// The length of the step being taken (across its stages).
    step_length: f64,
    /// Per rigid body: the matrix `A` the last stage ended with, and the
    /// rotation written to the world for it. A stage starts from `A` itself
    /// (always a valid state) unless the world's pose was changed since.
    affine: Vec<Option<(Quat, Mat3)>>,
    /// Per rigid body: solver-only contact potential (barrier plus
    /// orthogonality penalty, J) stored by its last resolved impact and not
    /// yet prevented from returning as normal separation. See
    /// [`Mechanics::restitute`]. Cleared once the body has no contact.
    unloading: Vec<f64>,
    plastic: Vec<SolidPlastic>,
    pub(super) dissipative_energy: Vec<Vec<f64>>,
    pub(super) dissipative_power: Vec<Vec<f64>>,
    x: Vec<f64>,
    energy_positions: Vec<f64>,
    terms_ready: bool,
    energy_ready: bool,
    workers: Vec<islands::Worker>,
    sweep: islands::Sweep,
    solver_groups: Vec<(usize, usize)>,
    fixed_point: bool,
    gravity_seen: Option<Vec3>,
    previous_h: Option<f64>,
    translation_weights: Vec<(usize, f64)>,
    audit_scale: f64,
}

impl Mechanics {
    pub fn new(settings: Settings) -> Self {
        Self {
            settings,
            profile: Profile::default(),
            layout: Layout::default(),
            frame: Frame::default(),
            built_for: (usize::MAX, usize::MAX),
            inertia: Inertia::default(),
            gravity: Gravity::default(),
            mutual_gravity: MutualGravity::default(),
            elastic: Elastic::default(),
            rigid: RigidBodies::default(),
            joints: Joints::default(),
            contact: Contact::default(),
            hessian: Sum(BlockMatrix::default(), BlockMatrix::default()),
            triplets: Triplets::default(),
            newton: Newton::default(),
            velocity: Vec::new(),
            acceleration: None,
            earlier: None,
            beta2: Vec::new(),
            restart: Vec::new(),
            stage: Stage::Euler,
            step_length: 0.0,
            affine: Vec::new(),
            unloading: Vec::new(),
            plastic: Vec::new(),
            dissipative_energy: Vec::new(),
            dissipative_power: Vec::new(),
            x: Vec::new(),
            energy_positions: Vec::new(),
            terms_ready: false,
            energy_ready: false,
            workers: Vec::new(),
            sweep: islands::Sweep::default(),
            solver_groups: Vec::new(),
            fixed_point: false,
            gravity_seen: None,
            previous_h: None,
            translation_weights: Vec::new(),
            audit_scale: 0.0,
        }
    }

    /// Physical strain energy using cached quadrature, layout and scratch.
    /// Does not rebuild elastic Hessian buffers or modify integration history.
    pub fn strain_energy(&mut self, pool: &Pool, world: &World) -> f64 {
        if self.built_for != (world.objects().len(), world.links().len()) {
            self.build(pool, world);
        }
        if !self.energy_ready {
            let ctx = Context {
                pool,
                world,
                layout: &self.layout,
                frame: &self.frame,
            };
            self.elastic = Elastic::for_energy(&ctx);
            self.energy_ready = true;
        }
        self.elastic.sync_state(world, &self.layout, &self.plastic);
        self.energy_positions.resize(3 * self.layout.blocks, 0.0);
        for dofs in &self.layout.solids {
            let Representation::Solid(s) = &world.objects()[dofs.object].representation else {
                unreachable!()
            };
            for (node, &b) in dofs.block.iter().enumerate() {
                if b != HELD {
                    s.nodes.position[node].write(&mut self.energy_positions, b as usize);
                }
            }
        }
        let frame = Frame::default();
        let ctx = Context {
            pool,
            world,
            layout: &self.layout,
            frame: &frame,
        };
        self.elastic.physical_energy(&ctx, &self.energy_positions)
    }

    /// Block and rigid counts used by the integration-history encoding.
    pub fn integration_shape(&self) -> (usize, usize) {
        (self.layout.blocks, self.layout.rigids.len())
    }

    /// Append fixed-width integration history for learning one-step dynamics.
    /// Order: acceleration present + 3*blocks values; earlier present + two
    /// 3*blocks vectors; restart flags; per rigid affine present + quaternion
    /// xyzw + column-major matrix; gravity present + xyz; previous interval
    /// present + seconds; per rigid post-impact unloading budget, J. Absent
    /// values are zero filled.
    pub fn append_integration_state(&self, out: &mut Vec<f64>) {
        let n = self.layout.blocks * 3;
        out.push(f64::from(self.acceleration.is_some()));
        if let Some(a) = &self.acceleration {
            out.extend_from_slice(a);
        } else {
            out.resize(out.len() + n, 0.0);
        }
        out.push(f64::from(self.earlier.is_some()));
        if let Some((x, v)) = &self.earlier {
            out.extend_from_slice(x);
            out.extend_from_slice(v);
        } else {
            out.resize(out.len() + 2 * n, 0.0);
        }
        out.extend(self.restart.iter().map(|&b| f64::from(b)));
        for a in &self.affine {
            out.push(f64::from(a.is_some()));
            if let Some((q, m)) = a {
                out.extend([q.x, q.y, q.z, q.w]);
                for c in m.cols {
                    out.extend([c.x, c.y, c.z]);
                }
            } else {
                out.resize(out.len() + 13, 0.0);
            }
        }
        out.push(f64::from(self.gravity_seen.is_some()));
        out.extend(self.gravity_seen.unwrap_or(Vec3::ZERO).to_array());
        out.push(f64::from(self.previous_h.is_some()));
        out.push(self.previous_h.unwrap_or(0.0));
        out.extend_from_slice(&self.unloading);
    }

    /// Minimum contact precision across the world, m. With automatic scaling,
    /// an island uses its own smallest moving object; unrelated tiny objects
    /// do not tighten its nonlinear tolerance.
    pub fn contact_precision(&self) -> f64 {
        self.contact.precision
    }

    /// Time-averaged dissipative thermal heat source (frictional + plastic +
    /// cohesive damage work, in W = J/s) distributed onto each node of `object`
    /// during the most recent successful mechanical step.
    pub fn dissipative_thermal_power(&self, object: ObjectId) -> Option<Vec<f64>> {
        if let Some(&crate::dofs::Owner::Solid(s)) = self.layout.objects.get(object.0 as usize) {
            self.dissipative_power.get(s).cloned()
        } else {
            None
        }
    }

    /// Evaluates all virtual instrumentation probes attached to `world`.
    pub fn sample_probes(&mut self, pool: &Pool, world: &World) -> Vec<ProbeReading> {
        if self.built_for != (world.objects().len(), world.links().len()) {
            self.build(pool, world);
        }
        let use_workers = self.settings.islands && !self.workers.is_empty();
        world
            .probes()
            .iter()
            .map(|probe| match *probe {
                Probe::JointState { link } => ProbeReading::JointState {
                    coordinate: world.joint_coordinate(link).unwrap_or(0.0),
                    velocity: world.joint_velocity(link).unwrap_or(0.0),
                },
                Probe::Wrench { link } => {
                    let (force, torque) = if use_workers {
                        self.workers
                            .iter_mut()
                            .find_map(|w| w.sample_wrench(pool, world, link.0 as usize))
                            .unwrap_or_else(|| self.link_wrench_for(pool, world, link.0 as usize))
                    } else {
                        self.link_wrench_for(pool, world, link.0 as usize)
                    };
                    ProbeReading::Wrench { force, torque }
                }
                Probe::Tactile {
                    object,
                    local_origin,
                    local_normal,
                    radius,
                } => {
                    let pose = match &world.objects()[object.0 as usize].representation {
                        Representation::Fixed(f) => f.pose,
                        Representation::Rigid(r) => r.pose,
                        Representation::Solid(_) => Pose::IDENTITY,
                    };
                    let world_origin = pose.transform_point(local_origin);
                    let world_normal = pose.rotation.rotate(local_normal);
                    let (normal_force, total_force, center_of_pressure, active_contacts) =
                        if use_workers {
                            let mut n_sum = 0.0;
                            let mut f_sum = Vec3::ZERO;
                            let mut cop_sum = Vec3::ZERO;
                            let mut c_sum = 0usize;
                            for w in &mut self.workers {
                                if let Some((nf, tf, cop, cnt)) = w.sample_tactile(
                                    pool,
                                    object.0 as usize,
                                    world_origin,
                                    world_normal,
                                    radius,
                                ) {
                                    n_sum += nf;
                                    f_sum += tf;
                                    if let Some(p) = cop {
                                        cop_sum += p * nf;
                                    }
                                    c_sum += cnt;
                                }
                            }
                            let cop = (n_sum > 0.0).then(|| cop_sum / n_sum);
                            (n_sum, f_sum, cop, c_sum)
                        } else {
                            self.tactile_for(
                                pool,
                                world,
                                object.0 as usize,
                                world_origin,
                                world_normal,
                                radius,
                            )
                        };
                    ProbeReading::Tactile {
                        normal_force,
                        total_force,
                        center_of_pressure,
                        active_contacts,
                    }
                }
                Probe::Imu { object, local } => {
                    let idx = object.0 as usize;
                    match (
                        &world.objects()[idx].representation,
                        self.layout.objects.get(idx).copied(),
                    ) {
                        (Representation::Rigid(r), Some(crate::dofs::Owner::Rigid(g))) => {
                            let b = self.layout.rigids[g].block;
                            let a_cm = self
                                .acceleration
                                .as_ref()
                                .map_or(world.gravity, |a| Vec3::read(a, b));
                            let r_world = r.pose.rotation.rotate(local);
                            let a_pt = a_cm + r.spin.cross(r.spin.cross(r_world));
                            ProbeReading::Imu {
                                linear_acceleration: r
                                    .pose
                                    .rotation
                                    .rotate_inverse(a_pt - world.gravity),
                                angular_velocity: r.pose.rotation.rotate_inverse(r.spin),
                            }
                        }
                        (Representation::Fixed(f), _) => ProbeReading::Imu {
                            linear_acceleration: f.pose.rotation.rotate_inverse(-world.gravity),
                            angular_velocity: Vec3::ZERO,
                        },
                        (Representation::Solid(_), Some(crate::dofs::Owner::Solid(i))) => {
                            let mut a_sum = Vec3::ZERO;
                            let mut n = 0usize;
                            if let Some(acc) = &self.acceleration {
                                for &b in &self.layout.solids[i].block {
                                    if b != HELD {
                                        a_sum += Vec3::read(acc, b as usize);
                                        n += 1;
                                    }
                                }
                            }
                            let a_avg = if n > 0 {
                                a_sum / (n as f64)
                            } else {
                                world.gravity
                            };
                            ProbeReading::Imu {
                                linear_acceleration: a_avg - world.gravity,
                                angular_velocity: Vec3::ZERO,
                            }
                        }
                        _ => ProbeReading::Imu {
                            linear_acceleration: -world.gravity,
                            angular_velocity: Vec3::ZERO,
                        },
                    }
                }
            })
            .collect()
    }

    pub(super) fn link_wrench_for(
        &mut self,
        pool: &Pool,
        world: &World,
        link_index: usize,
    ) -> (Vec3, Vec3) {
        if self.layout.blocks == 0 || link_index >= world.links().len() {
            return (Vec3::ZERO, Vec3::ZERO);
        }
        self.ensure_terms(pool, world);
        if self.frame.start.len() != 3 * self.layout.blocks {
            self.begin(world, Stage::Euler, self.previous_h.unwrap_or(1.0));
            let ctx = Context {
                pool,
                world,
                layout: &self.layout,
                frame: &self.frame,
            };
            self.joints.begin_step(&ctx);
        }
        let x = if self.x.len() == 3 * self.layout.blocks {
            self.x.clone()
        } else {
            self.state(world).0
        };
        let ctx = Context {
            pool,
            world,
            layout: &self.layout,
            frame: &self.frame,
        };
        self.joints.link_wrench(&ctx, world, link_index, &x)
    }

    pub(super) fn tactile_for(
        &mut self,
        pool: &Pool,
        world: &World,
        object: usize,
        world_origin: Vec3,
        world_normal: Vec3,
        radius: f64,
    ) -> (f64, Vec3, Option<Vec3>, usize) {
        if !self.terms_ready || self.frame.step <= 0.0 || object >= world.objects().len() {
            return (0.0, Vec3::ZERO, None, 0);
        }
        let x = if self.x.len() == 3 * self.layout.blocks {
            self.x.clone()
        } else {
            self.state(world).0
        };
        let ctx = Context {
            pool,
            world,
            layout: &self.layout,
            frame: &self.frame,
        };
        self.contact
            .tactile_patch(&ctx, &x, object, world_origin, world_normal, radius)
    }

    /// Forget everything derived from the world's layout (call after adding
    /// or removing objects/links, or changing meshes, material data or fixed
    /// geometry). Moving bodies advance without invalidating the layout.
    pub fn invalidate(&mut self) {
        self.built_for = (usize::MAX, usize::MAX);
    }

    fn build(&mut self, pool: &Pool, world: &World) {
        self.layout = Layout::new(world);
        self.release_terms();
        let ctx = Context {
            pool,
            world,
            layout: &self.layout,
            frame: &self.frame,
        };
        self.contact.precision = self
            .settings
            .contact_precision
            .unwrap_or_else(|| Contact::default_precision(&ctx));
        self.workers.clear();
        self.sweep = islands::Sweep::default();
        self.acceleration = None;
        self.earlier = None;
        self.gravity_seen = Some(world.gravity);
        self.previous_h = None;
        self.restart = vec![false; self.layout.blocks];
        self.affine = vec![None; self.layout.rigids.len()];
        self.unloading = vec![0.0; self.layout.rigids.len()];
        self.plastic = self
            .layout
            .solids
            .iter()
            .map(|d| {
                let obj = &world.objects()[d.object];
                let Representation::Solid(s) = &obj.representation else {
                    unreachable!()
                };
                let mat = world.material(obj.material);
                let pts = if s.mesh.is_quadratic() { 4 } else { 1 };
                let n = s.mesh.tets.len() * pts;
                SolidPlastic {
                    fp_inv: if mat.plasticity.is_some() {
                        vec![Mat3::IDENTITY; n]
                    } else {
                        Vec::new()
                    },
                    eps_p: if mat.plasticity.is_some() {
                        vec![0.0; n]
                    } else {
                        Vec::new()
                    },
                    damage: if mat.fracture.is_some() {
                        vec![0.0; n]
                    } else {
                        Vec::new()
                    },
                }
            })
            .collect();
        self.dissipative_energy = self
            .layout
            .solids
            .iter()
            .map(|d| {
                let Representation::Solid(s) = &world.objects()[d.object].representation else {
                    unreachable!()
                };
                vec![0.0; s.nodes.position.len()]
            })
            .collect();
        self.dissipative_power = self.dissipative_energy.clone();
        self.built_for = (world.objects().len(), world.links().len());
    }

    fn release_terms(&mut self) {
        self.frame = Frame::default();
        self.inertia = Inertia::default();
        self.gravity = Gravity::default();
        self.mutual_gravity = MutualGravity::default();
        self.elastic = Elastic::default();
        self.rigid = RigidBodies::default();
        self.joints = Joints::default();
        self.contact = Contact::default();
        self.hessian = Sum(BlockMatrix::default(), BlockMatrix::default());
        self.triplets = Triplets::default();
        self.newton = Newton::default();
        self.velocity = Vec::new();
        self.beta2 = Vec::new();
        self.x = Vec::new();
        self.energy_positions = Vec::new();
        self.terms_ready = false;
        self.energy_ready = false;
    }

    /// Allocate derivative and sparse solve storage only for an actual solve.
    fn ensure_terms(&mut self, pool: &Pool, world: &World) {
        let ctx = Context {
            pool,
            world,
            layout: &self.layout,
            frame: &self.frame,
        };
        let precision = self
            .settings
            .contact_precision
            .unwrap_or_else(|| Contact::default_precision(&ctx));
        if !self.terms_ready {
            self.elastic = Elastic::new(&ctx);
            self.elastic.sync_state(world, &self.layout, &self.plastic);
            self.joints = Joints::new(&ctx);
            let mut entries = Vec::new();
            self.inertia.pattern(&ctx, &mut entries);
            self.elastic.pattern(&ctx, &mut entries);
            self.rigid.pattern(&ctx, &mut entries);
            self.joints.pattern(&ctx, &mut entries);
            let fixed = BlockMatrix::with_pattern(self.layout.blocks, entries);
            self.inertia.bind(&ctx, &fixed);
            self.elastic.bind(&ctx, &fixed);
            self.rigid.bind(&ctx, &fixed);
            self.joints.bind(&ctx, &fixed);
            self.hessian = Sum(fixed, BlockMatrix::default());
        }
        if !self.terms_ready || self.contact.precision != precision {
            self.contact = Contact::new(&ctx, precision);
        }
        self.joints.sync_world(world);
        self.mutual_gravity.settings = self.settings.mutual_gravity;
        self.contact.self_contact = self.settings.self_contact;
        self.solver_groups.clear();
        let per_chunk = core_numerics::vector::CHUNK / 3;
        let coupled_pairs = self.joints.coupled_rigid_pairs();
        let mut r = 0usize;
        while r < self.layout.rigid_blocks.len() {
            let (b_start, mut count) = self.layout.rigid_blocks[r];
            let mut end_r = r;
            while end_r + 1 < self.layout.rigid_blocks.len() {
                let next_r = end_r + 1;
                let (b_next, next_count) = self.layout.rigid_blocks[next_r];
                let merged = count + next_count;
                let same_chunk = (b_start / per_chunk) == ((b_start + merged - 1) / per_chunk);
                let connected = (r..=end_r)
                    .any(|prev_r| coupled_pairs.binary_search(&(prev_r, next_r)).is_ok());
                if b_next == b_start + count && merged <= 32 && same_chunk && connected {
                    count = merged;
                    end_r = next_r;
                } else {
                    break;
                }
            }
            self.solver_groups.push((b_start, count));
            r = end_r + 1;
        }
        if self.settings.solid_block_size > 1 {
            for solid in &self.layout.solids {
                let free: Vec<_> = solid.block.iter().copied().filter(|&b| b != HELD).collect();
                if let Some(&first) = free.first() {
                    let mut b = first as usize;
                    let end = b + free.len();
                    while b < end {
                        let n = self
                            .settings
                            .solid_block_size
                            .min(end - b)
                            .min(per_chunk - b % per_chunk);
                        if n > 1 {
                            self.solver_groups.push((b, n));
                        }
                        b += n;
                    }
                }
            }
        }
        self.solver_groups.sort_unstable();
        self.terms_ready = true;
        self.energy_ready = true;
    }

    /// Capture physical state and all integration history for exact replay.
    /// Call after preparing this world's layout (or after its first step).
    pub fn checkpoint(&self, world: &World) -> MechanicsState {
        self.snapshot(world)
    }

    /// Prepare a world without advancing its clock.
    pub fn prepare(&mut self, pool: &Pool, world: &World) {
        self.build(pool, world);
    }

    /// Restore a checkpoint belonging to the same world and settings.
    pub fn restore_checkpoint(&mut self, world: &mut World, state: &MechanicsState) {
        self.restore(world, state.clone());
    }

    /// Advances `world` by `h` seconds, or preserves state on failure.
    pub fn step(&mut self, pool: &Pool, world: &mut World, h: f64) -> StepReport {
        if !h.is_finite()
            || h <= 0.0
            || !world.time.is_finite()
            || !(world.time + h).is_finite()
            || world.time + h <= world.time
            || !self.settings.tolerance.is_finite()
            || self.settings.tolerance <= 0.0
            || !(1..=16).contains(&self.settings.solid_block_size)
            || self.settings.accuracy_subdivisions > 12
            || self
                .settings
                .angular_tolerance
                .is_some_and(|t| !t.is_finite() || t <= 0.0)
            || self
                .settings
                .contact_precision
                .is_some_and(|p| !p.is_finite() || p <= 0.0)
            || self
                .settings
                .settle_time
                .is_some_and(|t| !t.is_finite() || t <= 0.0)
            || self
                .settings
                .max_substep
                .is_some_and(|t| !t.is_finite() || t <= 0.0)
            || self
                .settings
                .cfl_subcycling
                .is_some_and(|c| !c.is_finite() || c <= 0.0)
            || self.settings.mutual_gravity.is_some_and(|g| {
                !g.constant.is_finite()
                    || g.constant < 0.0
                    || !g.softening.is_finite()
                    || g.softening < 0.0
                    || !g.theta.is_finite()
                    || g.theta < 0.0
            })
        {
            return StepReport {
                invalid: true,
                ..StepReport::default()
            };
        }
        if self.built_for != (world.objects().len(), world.links().len()) {
            self.build(pool, world);
        }
        for v in &mut self.dissipative_energy {
            v.fill(0.0);
        }
        for v in &mut self.dissipative_power {
            v.fill(0.0);
        }
        if self.settings.islands {
            self.step_islands(pool, world, h)
        } else {
            self.workers.clear();
            self.ensure_terms(pool, world);
            let time = world.time;
            let mut report = self.step_within(pool, world, h, 0);
            if !report.failed() {
                let inv_h = 1.0 / h;
                for (pow, eng) in self
                    .dissipative_power
                    .iter_mut()
                    .zip(&self.dissipative_energy)
                {
                    for (p, &e) in pow.iter_mut().zip(eng) {
                        *p = e * inv_h;
                    }
                }
                self.previous_h = Some(h);
                self.gravity_seen = Some(world.gravity);
                world.time = time + h;
            }
            report.islands = usize::from(self.layout.blocks > 0);
            report.largest_island_blocks = self.layout.blocks;
            report.island_solves = report.islands;
            report.max_island_substeps = report.islands;
            report
        }
    }

    /// A step of `h`; if a solve fails, the step is undone and taken as two
    /// halves instead (at most [`MAX_SPLITS`] times over), so a hard moment
    /// costs smaller steps rather than a bad state.
    fn step_within(&mut self, pool: &Pool, world: &mut World, h: f64, depth: usize) -> StepReport {
        let saved = self.snapshot(world);
        let saved_diss = self.dissipative_energy.clone();
        let audit = self
            .settings
            .angular_tolerance
            .map(|_| self.angular_state(world));
        if depth == 0 {
            if let Some(before) = &audit {
                // Elastic energy can become motion during impact. Freeze the
                // interval's characteristic scale instead of tightening it as
                // kinetic energy temporarily becomes stored strain energy.
                let elastic = self.strain_energy(pool, world).max(0.0);
                self.audit_scale = before.scale.max(if before.mass > 0.0 {
                    before.mass_radius * (2.0 * elastic / before.mass).sqrt()
                } else {
                    0.0
                });
            }
        }
        let mut report = self.step_once(pool, world, h);
        if !report.failed() && !report.boundary_forces {
            if let Some(before) = audit {
                let after = self.angular_state(world);
                let error = (after.momentum - before.momentum).length();
                let floor = 8.0
                    * (before.mass_radius + after.mass_radius)
                    * self.settings.tolerance
                    * self.contact.precision
                    / h;
                let budget = self.settings.angular_tolerance.unwrap() * self.audit_scale
                    / (1usize << depth) as f64
                    + floor;
                report.angular_error = error;
                if error > budget {
                    report.accuracy_failed = true;
                    report.accuracy_retries += 1;
                }
            }
        }
        if !report.failed() {
            self.fixed_point = self.stationary_match(world, &saved);
            return report;
        }
        self.fixed_point = false;
        report.energy = None;
        self.restore(world, saved.clone());
        self.dissipative_energy = saved_diss.clone();
        let limit = if report.accuracy_failed {
            self.settings.accuracy_subdivisions
        } else {
            MAX_SPLITS
        };
        if depth >= limit || report.invalid {
            return report;
        }
        if self.settings.integrator == Integrator::Bdf2 {
            self.earlier = None;
        }
        let mut first = self.step_within(pool, world, 0.5 * h, depth + 1);
        first.solves += report.solves;
        first.iterations += report.iterations;
        first.linear_iterations += report.linear_iterations;
        first.linear_block_iterations += report.linear_block_iterations;
        first.linear_products += report.linear_products;
        first.linear_block_products += report.linear_block_products;
        first.contact_restarts += report.contact_restarts;
        first.accuracy_retries += report.accuracy_retries;
        first.refreshes_discarded += report.refreshes_discarded;
        first.splits += 1;
        if first.failed() {
            first.energy = None;
            self.restore(world, saved);
            self.dissipative_energy = saved_diss;
            return first;
        }
        let second = self.step_within(pool, world, 0.5 * h, depth + 1);
        first.solves += second.solves;
        first.iterations += second.iterations;
        first.linear_iterations += second.linear_iterations;
        first.linear_block_iterations += second.linear_block_iterations;
        first.linear_products += second.linear_products;
        first.linear_block_products += second.linear_block_products;
        first.impacts += second.impacts;
        first.contact_restarts += second.contact_restarts;
        first.accuracy_retries += second.accuracy_retries;
        first.angular_error += second.angular_error;
        first.boundary_forces |= second.boundary_forces;
        first.accuracy_failed = second.accuracy_failed;
        first.splits += second.splits;
        first.refreshes_discarded += second.refreshes_discarded;
        first.unconverged = second.unconverged;
        first.invalid = second.invalid;
        first.grips = second.grips;
        first.newton = second.newton;
        EnergyLedger::serial(&mut first.energy, second.energy);
        if first.failed() {
            first.energy = None;
            self.restore(world, saved);
            self.dissipative_energy = saved_diss;
            self.fixed_point = false;
        } else {
            if self.settings.integrator == Integrator::Bdf2 {
                // The next outer step has a different length from either half.
                self.earlier = None;
            }
            self.fixed_point = self.stationary_match(world, &saved);
        }
        first
    }

    fn snapshot(&self, world: &World) -> MechanicsState {
        MechanicsState {
            time: world.time,
            nodes: self
                .layout
                .solids
                .iter()
                .map(|d| match &world.objects()[d.object].representation {
                    Representation::Solid(s) => {
                        (s.nodes.position.clone(), s.nodes.velocity.clone())
                    }
                    _ => unreachable!(),
                })
                .collect(),
            rigids: self
                .layout
                .rigids
                .iter()
                .map(|d| match &world.objects()[d.object].representation {
                    Representation::Rigid(r) => (r.pose, r.velocity, r.spin),
                    _ => unreachable!(),
                })
                .collect(),
            acceleration: self.acceleration.clone(),
            earlier: self.earlier.clone(),
            restart: self.restart.clone(),
            affine: self.affine.clone(),
            gravity_seen: self.gravity_seen,
            previous_h: self.previous_h,
            unloading: self.unloading.clone(),
            plastic: self.plastic.clone(),
        }
    }

    fn restore(&mut self, world: &mut World, saved: MechanicsState) {
        world.time = saved.time;
        for (d, (position, velocity)) in self.layout.solids.iter().zip(saved.nodes) {
            let Representation::Solid(s) = &mut world.objects_mut()[d.object].representation else {
                unreachable!()
            };
            s.nodes.position = position;
            s.nodes.velocity = velocity;
        }
        for (d, (pose, velocity, spin)) in self.layout.rigids.iter().zip(saved.rigids) {
            let Representation::Rigid(r) = &mut world.objects_mut()[d.object].representation else {
                unreachable!()
            };
            r.pose = pose;
            r.velocity = velocity;
            r.spin = spin;
        }
        self.acceleration = saved.acceleration;
        self.earlier = saved.earlier;
        self.restart = saved.restart;
        self.affine = saved.affine;
        self.gravity_seen = saved.gravity_seen;
        self.previous_h = saved.previous_h;
        self.unloading = saved.unloading;
        self.plastic = saved.plastic;
        Elastic::sync_plastic_columns(&self.layout, &self.plastic, world);
        if self.terms_ready || self.energy_ready {
            self.elastic.sync_state(world, &self.layout, &self.plastic);
        }
    }

    fn step_once(&mut self, pool: &Pool, world: &mut World, h: f64) -> StepReport {
        self.step_length = h;
        if let Some(old) = self.gravity_seen {
            if !sleep::same_vec(old, world.gravity) {
                if let Some(a) = &mut self.acceleration {
                    let change = world.gravity - old;
                    for solid in &self.layout.solids {
                        for &b in &solid.block {
                            if b != HELD {
                                (Vec3::read(a, b as usize) + change).write(a, b as usize);
                            }
                        }
                    }
                    for rigid in &self.layout.rigids {
                        (Vec3::read(a, rigid.block) + change).write(a, rigid.block);
                    }
                }
                if self.settings.integrator == Integrator::Bdf2 {
                    self.earlier = None;
                }
            }
        }
        self.gravity_seen = Some(world.gravity);
        let mut report = StepReport {
            boundary_forces: self.layout.solids.iter().any(|s| s.block.contains(&HELD)),
            ..StepReport::default()
        };
        let started = self.state(world);
        if self.acceleration.is_none()
            && self.settings.force_startup
            && self.settings.integrator != Integrator::BackwardEuler
        {
            self.initialize_acceleration(pool, world, h);
        }
        let have_acceleration = self.acceleration.is_some();
        let method = if self.settings.integrator == Integrator::Bdf2
            && self.earlier.is_none()
            && have_acceleration
            && self.settings.force_startup
        {
            Integrator::TrBdf2
        } else {
            self.settings.integrator
        };
        match method {
            _ if !have_acceleration => self.stage(pool, world, Stage::Euler, h, &mut report),
            Integrator::BackwardEuler => self.stage(pool, world, Stage::Euler, h, &mut report),
            Integrator::Newmark => self.stage(pool, world, Stage::Trapezoid, h, &mut report),
            Integrator::Bdf2 => match self.earlier.is_some() {
                true => self.stage(
                    pool,
                    world,
                    {
                        let ratio = h / self.previous_h.unwrap_or(h);
                        Stage::Bdf2 {
                            q: ratio * ratio / (1.0 + 2.0 * ratio),
                            c: (1.0 + ratio) / (1.0 + 2.0 * ratio),
                            span: h,
                        }
                    },
                    h,
                    &mut report,
                ),
                false => self.stage(pool, world, Stage::Euler, h, &mut report),
            },
            Integrator::TrBdf2 => {
                self.stage(pool, world, Stage::Trapezoid, GAMMA * h, &mut report);
                if report.failed() {
                    return report;
                }
                self.earlier = Some(started.clone());
                let q = (1.0 - GAMMA) * (1.0 - GAMMA) / (GAMMA * (2.0 - GAMMA));
                let c = (1.0 - GAMMA) / (2.0 - GAMMA);
                self.stage(
                    pool,
                    world,
                    Stage::Bdf2 { q, c, span: h },
                    (1.0 - GAMMA) * h,
                    &mut report,
                );
            }
        }
        // A plain BDF2 step reaches back one whole step.
        self.earlier = Some(started);
        self.previous_h = Some(h);
        report
    }

    /// A₀ = M⁻¹(f_gravity - ∇U(x₀)) for smooth starts. Barriers are
    /// nonsmooth at impact and must never be extrapolated into startup.
    fn initialize_acceleration(&mut self, pool: &Pool, world: &World, h: f64) {
        self.step_length = h;
        self.begin(world, Stage::Euler, h);
        self.elastic.sync_state(world, &self.layout, &self.plastic);
        let ctx = Context {
            pool,
            world,
            layout: &self.layout,
            frame: &self.frame,
        };
        let x = &self.frame.start;
        if !self.elastic.physical_energy(&ctx, x).is_finite()
            || !self.contact.energy(&ctx, x).is_finite()
            || !self.contact.impacts(&ctx, x).is_empty()
        {
            return;
        }
        let mut grad = vec![0.0; 3 * self.layout.blocks];
        self.elastic.sync_state(world, &self.layout, &self.plastic);
        self.elastic.physical_gradient(&ctx, x, &mut grad);
        if !self.terms_ready {
            self.joints = Joints::new(&ctx);
        }
        self.joints.begin_step(&ctx);
        self.joints.startup_gradient(&ctx, x, &mut grad);
        self.mutual_gravity.begin_step(&ctx);
        self.mutual_gravity.physical_gradient(&ctx, x, &mut grad);
        let mut a = vec![0.0; 3 * self.layout.blocks];
        for &(b, mass) in &self.gravity.weights {
            (world.gravity - Vec3::read(&grad, b) / mass).write(&mut a, b);
        }
        if a.iter().all(|v| v.is_finite()) {
            self.acceleration = Some(a);
        }
    }

    /// Positions and velocities of the unknowns, from the world.
    fn state(&self, world: &World) -> (Vec<f64>, Vec<f64>) {
        let n = 3 * self.layout.blocks;
        let (mut x, mut v) = (vec![0.0; n], vec![0.0; n]);
        for dofs in &self.layout.solids {
            let Representation::Solid(s) = &world.objects()[dofs.object].representation else {
                unreachable!()
            };
            for (node, &b) in dofs.block.iter().enumerate() {
                if b != HELD {
                    s.nodes.position[node].write(&mut x, b as usize);
                    s.nodes.velocity[node].write(&mut v, b as usize);
                }
            }
        }
        for dofs in &self.layout.rigids {
            let Representation::Rigid(r) = &world.objects()[dofs.object].representation else {
                unreachable!()
            };
            r.pose.position.write(&mut x, dofs.block);
            let rot = r.pose.rotation.to_mat3();
            for k in 0..3 {
                rot.col(k).write(&mut x, dofs.block + 1 + k);
            }
            r.velocity.write(&mut v, dofs.block);
        }
        (x, v)
    }

    /// Solves one stage of length `d` and writes it back to the world.
    fn stage(
        &mut self,
        pool: &Pool,
        world: &mut World,
        stage: Stage,
        d: f64,
        report: &mut StepReport,
    ) {
        // All mutually coupled unknowns need the same integration rule.
        // The scheduler supplies one independent island to this solve.
        // Restart the whole island so internal forces exchange equal impulses.
        let stage = if self.restart.iter().any(|&r| r) {
            Stage::Euler
        } else {
            stage
        };
        let energy_before = self
            .settings
            .energy_accounting
            .then(|| self.sample_energy(pool, world));
        let before = Velocities::of(world, &self.layout);
        self.begin(world, stage, d);
        self.elastic.sync_state(world, &self.layout, &self.plastic);
        {
            let ctx = Context {
                pool,
                world,
                layout: &self.layout,
                frame: &self.frame,
            };
            let terms: [&mut dyn Term; 7] = [
                &mut self.inertia,
                &mut self.gravity,
                &mut self.mutual_gravity,
                &mut self.elastic,
                &mut self.rigid,
                &mut self.contact,
                &mut self.joints,
            ];
            for term in terms {
                term.begin_step(&ctx);
            }
        }
        report.boundary_forces |= self.joints.touches_boundary();
        if self.settings.angular_tolerance.is_some() {
            let ctx = Context {
                pool,
                world,
                layout: &self.layout,
                frame: &self.frame,
            };
            report.boundary_forces |= self.contact.boundary_contact(&ctx, &self.frame.start);
        }
        self.newton.settings = NewtonSettings {
            tolerance: self.settings.tolerance * self.contact.precision,
            max_iterations: self.settings.max_iterations,
            ..NewtonSettings::default()
        };
        let mut x = std::mem::take(&mut self.x);
        x.clear();
        x.extend_from_slice(&self.frame.start);
        for pass in 0..=self.settings.friction_updates {
            let previous_grips = (pass > 0).then(|| self.contact.friction_state());
            let change = {
                let ctx = Context {
                    pool,
                    world,
                    layout: &self.layout,
                    frame: &self.frame,
                };
                self.contact.grip(&ctx, &x)
            };
            // Solve again only if friction's bounds moved: a re-solve over a
            // narrow stick–slip kink costs many iterations for nothing.
            if pass > 0 && change < FRICTION_CHANGE {
                self.contact.restore_friction(previous_grips.unwrap());
                break;
            }
            let mut system = System {
                world,
                layout: &self.layout,
                frame: &self.frame,
                terms: [
                    &mut self.inertia,
                    &mut self.gravity,
                    &mut self.mutual_gravity,
                    &mut self.elastic,
                    &mut self.rigid,
                    &mut self.contact,
                    &mut self.joints,
                ],
                hessian: &mut self.hessian,
                triplets: &mut self.triplets,
                profile: &mut self.profile,
                sweep: &mut self.sweep,
                groups: &self.solver_groups,
                translation: &self.translation_weights,
            };
            if !system.energy(pool, &x).is_finite() {
                // Not a state any solve can start from (and not one any
                // solve may end in): report it, leave the world as it is.
                report.invalid = true;
                self.x = x;
                return;
            }
            let started = std::time::Instant::now();
            let before = (pass > 0).then(|| x.clone());
            // A refresh only improves a converged solve: it gets a small
            // budget, and is dropped if that is not enough.
            self.newton.settings.max_iterations = if pass > 0 {
                REFRESH_ITERATIONS.min(self.settings.max_iterations)
            } else {
                self.settings.max_iterations
            };
            let r = self.newton.minimize(pool, &mut system, &mut x);
            self.profile.add("solve (all)", started.elapsed());
            report.solves += 1;
            report.iterations += r.iterations;
            report.linear_iterations += r.linear_iterations;
            report.linear_block_iterations += r.linear_iterations * self.layout.blocks;
            report.linear_products += r.linear_products;
            report.linear_block_products += r.linear_products * self.layout.blocks;
            if let (Some(before), false) = (before, r.converged) {
                // A failed refresh: keep the converged solve before it.
                x = before;
                self.contact.restore_friction(previous_grips.unwrap());
                report.refreshes_discarded += 1;
                break;
            }
            report.newton = r;
            if !r.not_finite && !matches!(stage, Stage::Euler) && self.settings.impact_law {
                let ctx = Context {
                    pool,
                    world,
                    layout: &self.layout,
                    frame: &self.frame,
                };
                if !self.contact.impacts(&ctx, &x).is_empty() {
                    // The world has not been changed yet. Active barriers can
                    // carry a nonsmooth impact even below the restitution speed
                    // threshold, or stall a smooth solve before convergence.
                    // Repeat the physical solve with one uniform rule before
                    // refreshing friction or splitting time.
                    self.x = x;
                    report.contact_restarts += 1;
                    self.stage(pool, world, Stage::Euler, d, report);
                    return;
                }
            }
            report.unconverged |= !r.converged;
            if report.failed() {
                self.x = x;
                return;
            }
        }
        for _al in 0..self.settings.augmented_lagrangian_passes {
            if report.failed() {
                break;
            }
            let err = {
                let ctx = Context {
                    pool,
                    world,
                    layout: &self.layout,
                    frame: &self.frame,
                };
                self.joints.update_multipliers(&ctx, &x)
            };
            if err <= 1e-12 {
                break;
            }
            let mut system = System {
                world,
                layout: &self.layout,
                frame: &self.frame,
                terms: [
                    &mut self.inertia,
                    &mut self.gravity,
                    &mut self.mutual_gravity,
                    &mut self.elastic,
                    &mut self.rigid,
                    &mut self.contact,
                    &mut self.joints,
                ],
                hessian: &mut self.hessian,
                triplets: &mut self.triplets,
                profile: &mut self.profile,
                sweep: &mut self.sweep,
                groups: &self.solver_groups,
                translation: &self.translation_weights,
            };
            self.newton.settings.max_iterations = self.settings.max_iterations;
            let r = self.newton.minimize(pool, &mut system, &mut x);
            report.solves += 1;
            report.iterations += r.iterations;
            report.linear_iterations += r.linear_iterations;
            report.linear_block_iterations += r.linear_iterations * self.layout.blocks;
            report.linear_products += r.linear_products;
            report.linear_block_products += r.linear_products * self.layout.blocks;
            report.newton = r;
            report.unconverged |= !r.converged;
            if report.failed() {
                self.x = x;
                return;
            }
        }
        report.grips = self.contact.grips();
        if self.settings.angular_tolerance.is_some() {
            let ctx = Context {
                pool,
                world,
                layout: &self.layout,
                frame: &self.frame,
            };
            report.boundary_forces |= self.contact.boundary_contact(&ctx, &x);
        }
        if !self.dissipative_energy.is_empty() {
            let ctx = Context {
                pool,
                world,
                layout: &self.layout,
                frame: &self.frame,
            };
            let mut fric_power: Vec<Vec<f64>> = self
                .dissipative_energy
                .iter()
                .map(|v| vec![0.0; v.len()])
                .collect();
            self.contact.friction_power(&ctx, &x, &mut fric_power);
            for (dst, src) in self.dissipative_energy.iter_mut().zip(&fric_power) {
                for (a, &b) in dst.iter_mut().zip(src) {
                    *a += b * d;
                }
            }
        }
        let work = energy_before.map(|_| {
            let ctx = Context {
                pool,
                world,
                layout: &self.layout,
                frame: &self.frame,
            };
            let (barrier, friction) = self.contact.discrete_work(&ctx, &x);
            let damping = self.joints.damping_work(&ctx, &x);
            (barrier, friction, damping)
        });
        self.finish(pool, world, &x, d);
        let kinetic_before_impulse = energy_before.map(|_| self.kinetic_energy(world));
        self.x = x;
        if self.settings.impact_law {
            report.impacts += self.restitute(pool, world, &before);
        }
        let (plastic_work, fracture_work) = {
            let ctx = Context {
                pool,
                world,
                layout: &self.layout,
                frame: &self.frame,
            };
            let pw = self.elastic.update_plasticity(
                &ctx,
                &mut self.plastic,
                &self.x,
                Some(&mut self.dissipative_energy),
            );
            let fw = self.elastic.update_damage(
                &ctx,
                &mut self.plastic,
                &self.x,
                Some(&mut self.dissipative_energy),
            );
            (pw, fw)
        };
        Elastic::sync_plastic_columns(&self.layout, &self.plastic, world);
        world.time += d;
        if let Some(before) = energy_before {
            let after = self.sample_energy(pool, world);
            let (barrier_work, friction_work, damping_work) = work.unwrap();
            let mut ledger = EnergyLedger {
                before,
                after,
                barrier_work,
                friction_work,
                damping_work,
                plastic_work,
                fracture_work,
                restitution_work: after.kinetic - kinetic_before_impulse.unwrap(),
                ..EnergyLedger::default()
            };
            ledger.finish();
            EnergyLedger::serial(&mut report.energy, Some(ledger));
        }
    }

    /// For tests: sets up a backward-Euler step of `h` from the world's
    /// state, moves the unknowns by `perturb(block)`, and returns the
    /// largest error of the step's gradient against central differences of
    /// its energy, relative to the gradient's size, with the energy there.
    #[doc(hidden)]
    pub fn gradient_check(
        &mut self,
        pool: &Pool,
        world: &World,
        h: f64,
        perturb: impl Fn(usize) -> Vec3,
    ) -> (f64, f64) {
        if self.built_for != (world.objects().len(), world.links().len()) {
            self.build(pool, world);
        }
        self.ensure_terms(pool, world);
        self.step_length = h;
        self.begin(world, Stage::Euler, h);
        self.elastic.sync_state(world, &self.layout, &self.plastic);
        let mut x = self.frame.start.clone();
        for b in 0..self.layout.blocks {
            (Vec3::read(&x, b) + perturb(b)).write(&mut x, b);
        }
        {
            let ctx = Context {
                pool,
                world,
                layout: &self.layout,
                frame: &self.frame,
            };
            self.joints.begin_step(&ctx);
            self.mutual_gravity.begin_step(&ctx);
            let _ = self.contact.grip(&ctx, &self.frame.start.clone());
        }
        let step = 1e-4 * self.contact.precision;
        let mut system = System {
            world,
            layout: &self.layout,
            frame: &self.frame,
            terms: [
                &mut self.inertia,
                &mut self.gravity,
                &mut self.mutual_gravity,
                &mut self.elastic,
                &mut self.rigid,
                &mut self.contact,
                &mut self.joints,
            ],
            hessian: &mut self.hessian,
            triplets: &mut self.triplets,
            profile: &mut self.profile,
            sweep: &mut self.sweep,
            groups: &self.solver_groups,
            translation: &self.translation_weights,
        };
        let mut gradient = vec![0.0; x.len()];
        system.derivatives(pool, &x, &mut gradient);
        let energy = system.energy(pool, &x);
        let error =
            core_numerics::check::gradient_error(|p| system.energy(pool, p), &x, &gradient, step);
        (error, energy)
    }

    /// Sets up a stage's constants: starting unknowns, predictions, targets.
    fn begin(&mut self, world: &World, stage: Stage, d: f64) {
        // Whatever contact found belongs to the stage that is ending.
        self.contact.forget();
        let blocks = self.layout.blocks;
        let (start_translation, velocity) = self.state(world);
        let mut start = start_translation;
        self.velocity = velocity;
        let weight = match stage {
            Stage::Euler | Stage::Trapezoid => d * d,
            Stage::Bdf2 { span, .. } => span * span,
        };
        let mut mass = vec![Mat3::ZERO; blocks];
        let mut weights = Vec::new();
        let mut rigids = Vec::with_capacity(self.layout.rigids.len());
        self.beta2 = vec![1.0; blocks];

        for dofs in &self.layout.solids {
            let Representation::Solid(s) = &world.objects()[dofs.object].representation else {
                unreachable!()
            };
            for (node, &b) in dofs.block.iter().enumerate() {
                if b != HELD {
                    mass[b as usize] = Mat3::scalar(s.nodes.mass[node]);
                    weights.push((b as usize, s.nodes.mass[node]));
                }
            }
        }
        let mut target = vec![0.0; 3 * blocks];
        let mut is_rotation = vec![false; blocks];
        for (index, dofs) in self.layout.rigids.iter().enumerate() {
            let Representation::Rigid(r) = &world.objects()[dofs.object].representation else {
                unreachable!()
            };
            let radius = gyration_radius(r.mass, &r.inertia);
            let predicted = free_rotation(r.pose.rotation, r.angular_momentum(), &r.inertia, d);
            let rot = predicted.to_mat3();
            let world_inertia = rot * r.inertia * rot.transpose();
            let points = contact_points(&r.shape);
            let reach = points
                .iter()
                .map(|(p, radius)| p.length() + radius)
                .fold(0.0, f64::max);
            // The matrix part starts where the body is (A = R); its course,
            // in [`RigidBodies`], is the free motion R̃. The centre's
            // inertia is the integrator's, like any point's.
            let rotation_now = match self.affine[index] {
                Some((written, a)) if written == r.pose.rotation => a,
                _ => r.pose.rotation.to_mat3(),
            };
            for k in 0..3 {
                (rotation_now.col(k) * radius).write(&mut start, dofs.block + 1 + k);
                is_rotation[dofs.block + 1 + k] = true;
            }
            mass[dofs.block] = Mat3::scalar(r.mass);
            weights.push((dofs.block, r.mass));
            rigids.push(RigidFrame {
                mass: r.mass,
                inertia: r.inertia,
                moment: second_moment(&r.inertia),
                radius,
                centre_start: r.pose.position,
                rotation_start: r.pose.rotation,
                matrix_start: rotation_now,
                predicted,
                world_inertia,
                shape: r.shape,
                points,
                reach,
            });
        }

        for b in 0..blocks {
            if is_rotation[b] {
                continue;
            }
            let (x0, v0) = (Vec3::read(&start, b), Vec3::read(&self.velocity, b));
            let (t, beta2) = match stage {
                Stage::Euler => (x0 + v0 * d, d * d / weight),
                Stage::Trapezoid => {
                    let a0 = self
                        .acceleration
                        .as_ref()
                        .map_or(Vec3::ZERO, |a| Vec3::read(a, b));
                    (x0 + v0 * d + a0 * (0.25 * d * d), 0.25)
                }
                Stage::Bdf2 { q, c, span, .. } => {
                    let (xe, ve) = self.earlier.as_ref().expect("BDF2 needs an earlier state");
                    (
                        x0 + (x0 - Vec3::read(xe, b)) * q
                            + (v0 + (v0 - Vec3::read(ve, b)) * q) * (c * span),
                        c * c,
                    )
                }
            };
            t.write(&mut target, b);
            mass[b] *= 1.0 / beta2;
            self.beta2[b] = beta2;
        }
        self.restart.iter_mut().for_each(|r| *r = false);
        self.translation_weights.clear();
        if self.settings.translation_coarse
            && !self.layout.solids.iter().any(|s| s.block.contains(&HELD))
            && !self.joints.touches_boundary()
        {
            self.translation_weights
                .extend(weights.iter().map(|&(b, _)| (b, mass[b][(0, 0)])));
        }
        self.frame = Frame {
            h: d,
            weight,
            step: self.step_length,
            start,
            rigids,
        };
        self.inertia.target = target;
        self.inertia.mass = mass;
        self.gravity.weights = weights.clone();
        self.mutual_gravity.weights = weights;
        self.mutual_gravity.settings = self.settings.mutual_gravity;
        self.stage = stage;
    }

    /// Writes a stage's solution back: positions, and velocities and
    /// accelerations by the stage's rule.
    fn finish(&mut self, pool: &Pool, world: &mut World, x: &[f64], d: f64) {
        let blocks = self.layout.blocks;
        // Contact derivatives identify touched rigid bodies. Their angular
        // impulse below comes from the finite variational inertia identity.
        let mut contact = vec![0.0; 3 * blocks];
        if !self.layout.rigids.is_empty() {
            let ctx = Context {
                pool,
                world,
                layout: &self.layout,
                frame: &self.frame,
            };
            self.contact.gradient(&ctx, x, &mut contact);
        }
        let start = &self.frame.start;
        let weight = self.frame.weight;
        let mut velocity = vec![0.0; 3 * blocks];
        let mut acceleration = vec![0.0; 3 * blocks];
        for b in 0..blocks {
            // Affine orientation blocks use their own torque-free/torque
            // update; they have no translational mass or acceleration history.
            if self.inertia.mass[b] == Mat3::ZERO {
                continue;
            }
            let (x1, x0, v0) = (
                Vec3::read(x, b),
                Vec3::read(start, b),
                Vec3::read(&self.velocity, b),
            );
            // Every stage is M (x − x̃) / (β² W) = force: the acceleration.
            let a1 = (x1 - Vec3::read(&self.inertia.target, b)) / (self.beta2[b] * weight);
            let v1 = match self.stage {
                Stage::Euler => (x1 - x0) / d,
                Stage::Trapezoid => {
                    let a0 = self
                        .acceleration
                        .as_ref()
                        .map_or(Vec3::ZERO, |a| Vec3::read(a, b));
                    v0 + (a0 + a1) * (0.5 * d)
                }
                Stage::Bdf2 { q, c, span, .. } => {
                    let xe = &self.earlier.as_ref().unwrap().0;
                    ((x1 - x0) - (x0 - Vec3::read(xe, b)) * q) / (c * span)
                }
            };
            a1.write(&mut acceleration, b);
            v1.write(&mut velocity, b);
        }

        for dofs in &self.layout.solids {
            let Representation::Solid(s) = &mut world.objects_mut()[dofs.object].representation
            else {
                unreachable!()
            };
            for (node, &b) in dofs.block.iter().enumerate() {
                if b == HELD {
                    s.nodes.position[node] = dofs.held_at[node];
                    s.nodes.velocity[node] = Vec3::ZERO;
                } else {
                    s.nodes.position[node] = Vec3::read(x, b as usize);
                    s.nodes.velocity[node] = Vec3::read(&velocity, b as usize);
                }
            }
        }
        for (i, dofs) in self.layout.rigids.iter().enumerate() {
            let f = &self.frame.rigids[i];
            let Representation::Rigid(r) = &mut world.objects_mut()[dofs.object].representation
            else {
                unreachable!()
            };
            let a = Mat3::from_cols(
                Vec3::read(x, dofs.block + 1),
                Vec3::read(x, dofs.block + 2),
                Vec3::read(x, dofs.block + 3),
            ) * (1.0 / f.radius);
            // The pose takes the rotation nearest A.
            let rotation = Quat::from_mat3(&a.polar().0);
            self.affine[i] = Some((rotation, a));
            // Rotational stationarity gives the contact angular impulse
            // Σ a_k × [(A-Ã) J]_k / d. The rigidity penalty has zero torque.
            // This finite-increment identity avoids a small-angle approximation
            // and avoids amplifying a tiny nonlinear residual into a large
            // contact-force torque when the barrier is extremely stiff.
            let touched = (0..4).any(|k| Vec3::read(&contact, dofs.block + k) != Vec3::ZERO)
                || self.joints.touches_rigid(i);
            let mut momentum = r.angular_momentum();
            if touched {
                let inertial = (a - f.predicted.to_mat3()) * f.moment;
                for k in 0..3 {
                    momentum += a.col(k).cross(inertial.col(k)) * (1.0 / d);
                }
            }
            r.pose.position = Vec3::read(x, dofs.block);
            r.pose.rotation = rotation;
            r.velocity = Vec3::read(&velocity, dofs.block);
            let rot = rotation.to_mat3();
            r.spin = (rot * r.inertia * rot.transpose()).inverse() * momentum;
        }
        self.acceleration = Some(acceleration);
    }
}

/// Velocities of everything that moves, as they were at some moment.
struct Velocities {
    /// Per solid in the layout, per node.
    nodes: Vec<Vec<Vec3>>,
    /// Per rigid in the layout: linear and angular.
    rigids: Vec<(Vec3, Vec3)>,
}

impl Velocities {
    fn of(world: &World, layout: &Layout) -> Self {
        let nodes = layout
            .solids
            .iter()
            .map(|d| match &world.objects()[d.object].representation {
                Representation::Solid(s) => s.nodes.velocity.clone(),
                _ => unreachable!(),
            })
            .collect();
        let rigids = layout
            .rigids
            .iter()
            .map(|d| match &world.objects()[d.object].representation {
                Representation::Rigid(r) => (r.velocity, r.spin),
                _ => unreachable!(),
            })
            .collect();
        Self { nodes, rigids }
    }
}

impl Mechanics {
    /// Newton's impact law: every pair in contact that closed faster than
    /// the contact can resolve leaves at `e` times its closing speed. The
    /// step itself stopped it (perfectly inelastic); this gives back what
    /// the coefficient of restitution says, by impulses along the normal,
    /// pair by pair in a fixed order (projected Gauss–Seidel), pushing only.
    ///
    /// A resolved rigid impact also leaves solver-only potential in its
    /// barrier and in the orthogonality penalty of the compressed matrix
    /// `A`. Rigid bodies store no physical strain energy, so on later stages
    /// that potential must not return as extra normal separation on top of
    /// the impact law. While a struck rigid body stays in contact, a pair
    /// that did not strike again is held to its own starting normal speed
    /// (never below it, and never closing). The kinetic energy this removes
    /// is limited by the potential stored at the impact and by what is still
    /// stored now, so the hold expires and cannot act as general adhesion.
    /// Pairs with deformable nodes are excluded: their rebound is physical.
    fn restitute(&mut self, pool: &Pool, world: &mut World, before: &Velocities) -> usize {
        let impacts = {
            let ctx = Context {
                pool,
                world,
                layout: &self.layout,
                frame: &self.frame,
            };
            let x = std::mem::take(&mut self.x);
            let impacts = self.contact.impacts(&ctx, &x);
            self.x = x;
            impacts
        };
        // Keep contact forces out of the next smooth stage even if these
        // contacts are resting, below the restitution speed threshold. The
        // first stage after separation must start without contact memory too.
        if !impacts.is_empty() || self.contact.grips() > 0 {
            self.restart.fill(true);
        }
        // Closing slower than this is resting, not striking.
        let threshold = 4.0 * self.contact.precision / self.step_length;
        let layout = &self.layout;
        let n = layout.rigids.len();
        let previous = std::mem::take(&mut self.unloading);
        // Rigid bodies on each side, or none if a deformable node takes part:
        // deformable rebound stores physical strain energy and is never held.
        let rigid_sides = |i: &Impact| -> Option<Vec<usize>> {
            let mut out = Vec::new();
            for &(a, _) in &i.sides[..i.count] {
                match a {
                    Attach::Node { .. } => return None,
                    Attach::Rigid { rigid, .. } if !out.contains(&(rigid as usize)) => {
                        out.push(rigid as usize)
                    }
                    _ => {}
                }
            }
            Some(out)
        };
        let mut stored = vec![0.0; n];
        let mut touched = vec![false; n];
        let mut struck_body = vec![false; n];
        let mut hold_count = vec![0usize; n];
        let mut rows: Vec<Row> = Vec::new();
        let mut pending_holds: Vec<(Impact, f64, f64, Vec<usize>)> = Vec::new();
        for impact in impacts {
            let closing = relative_velocity(&impact, |a| attach_velocity(world, layout, before, a));
            let rigids = rigid_sides(&impact).unwrap_or_default();
            for &b in &rigids {
                touched[b] = true;
                stored[b] += impact.potential / rigids.len() as f64;
            }
            // Positions, normals and inverse masses stay fixed throughout
            // the impulse sweeps; compute the coupled response only once.
            if closing < -threshold {
                for &b in &rigids {
                    struck_body[b] = true;
                }
                rows.push(Row {
                    target: -impact.restitution.max(0.0) * closing,
                    compliance: impact_inverse_mass(world, layout, &impact),
                    impact,
                    hold: None,
                });
            } else if rigids.iter().any(|&b| previous[b] > 0.0) {
                for &b in &rigids {
                    hold_count[b] += 1;
                }
                pending_holds.push((
                    impact,
                    closing.max(0.0),
                    impact_inverse_mass(world, layout, &impact),
                    rigids,
                ));
            }
        }
        for (impact, target, compliance, rigids) in pending_holds {
            let available: f64 = rigids
                .iter()
                .map(|&b| previous[b] / hold_count[b].max(1) as f64)
                .sum();
            if available > 0.0 {
                rows.push(Row {
                    target,
                    compliance,
                    impact,
                    hold: Some((rigids, available)),
                });
            }
        }
        {
            let ctx = Context {
                pool,
                world,
                layout,
                frame: &self.frame,
            };
            for b in (0..n).filter(|&b| touched[b]) {
                stored[b] += RigidBodies::rigidity_potential(&ctx, b, &self.x);
            }
        }
        // Contact stages have already been solved with a uniform Euler rule.
        // Its nodal velocities satisfy the same momentum equations as the
        // positions. Only equal-and-opposite impulses may change them here.
        // Restart all coupled unknowns together at the next stage.
        let mut pushed = vec![0.0; rows.len()];
        for _ in 0..20 {
            let mut largest = 0.0f64;
            for (k, row) in rows.iter().enumerate() {
                let current =
                    relative_velocity(&row.impact, |a| current_attach_velocity(world, layout, a));
                if row.compliance <= 0.0 {
                    continue;
                }
                let raw = pushed[k] + (row.target - current) / row.compliance;
                let next = match &row.hold {
                    None => raw.max(0.0),
                    Some((_, available)) => raw.min(0.0).max(-row.hold_limit(*available)),
                };
                let delta = next - pushed[k];
                pushed[k] = next;
                largest = largest.max(delta.abs() * row.compliance);
                for j in 0..row.impact.count {
                    let (attach, weight) = row.impact.sides[j];
                    apply_impulse(world, layout, attach, row.impact.normal * (delta * weight));
                }
            }
            if largest < 1e-6 * threshold {
                break;
            }
        }
        // What a struck body may hold back later is what its impact stored.
        // Afterwards it can only shrink: by what holding has removed, and to
        // what is still stored now (so it is gone once the contact unloads).
        let mut remaining = previous;
        remaining.resize(n, 0.0);
        for (row, &j) in rows.iter().zip(&pushed) {
            if let Some((rigids, _)) = &row.hold {
                let used = row.removed(j) / rigids.len() as f64;
                for &b in rigids {
                    remaining[b] = (remaining[b] - used).max(0.0);
                }
            }
        }
        self.unloading = (0..n)
            .map(|b| match (struck_body[b], touched[b]) {
                (true, _) => stored[b],
                (false, true) => remaining[b].min(stored[b]),
                (false, false) => 0.0,
            })
            .collect();
        rows.iter().filter(|r| r.hold.is_none()).count()
    }
}

/// One pair in the impulse sweep: a striking impact (`hold` absent, pushes
/// only), or a post-impact hold over the struck rigid bodies with the
/// solver potential, J, it may still keep from returning as motion.
struct Row {
    impact: Impact,
    /// Relative normal speed the sweep moves the pair toward, m/s.
    target: f64,
    /// Effective inverse mass along the normal, 1/kg.
    compliance: f64,
    hold: Option<(Vec<usize>, f64)>,
}

impl Row {
    /// Kinetic energy removed by a pulling impulse `j ≤ 0` from a pair that
    /// ends at speed `target ≥ 0`: ½ c j² + target |j|, J.
    fn removed(&self, j: f64) -> f64 {
        0.5 * self.compliance * j * j + self.target * j.abs()
    }

    /// Largest pulling impulse whose removed energy stays within `available`.
    fn hold_limit(&self, available: f64) -> f64 {
        let (c, t) = (self.compliance, self.target);
        ((t * t + 2.0 * c * available).sqrt() - t) / c
    }
}

fn relative_velocity(impact: &Impact, velocity: impl Fn(Attach) -> Vec3) -> f64 {
    (0..impact.count)
        .map(|k| impact.sides[k].1 * impact.normal.dot(velocity(impact.sides[k].0)))
        .sum()
}

fn attach_velocity(world: &World, layout: &Layout, v: &Velocities, attach: Attach) -> Vec3 {
    match attach {
        Attach::Node { solid, node } => v.nodes[solid as usize][node as usize],
        Attach::Rigid { rigid, local } => {
            let Representation::Rigid(r) =
                &world.objects()[layout.rigids[rigid as usize].object].representation
            else {
                unreachable!()
            };
            let (linear, angular) = v.rigids[rigid as usize];
            linear + angular.cross(r.pose.rotation.rotate(local))
        }
        Attach::Still(_) => Vec3::ZERO,
    }
}

/// Read only the material points touched by this impulse. A full velocity
/// snapshot here would copy every node for every contact on every sweep.
fn current_attach_velocity(world: &World, layout: &Layout, attach: Attach) -> Vec3 {
    match attach {
        Attach::Node { solid, node } => {
            let Representation::Solid(s) =
                &world.objects()[layout.solids[solid as usize].object].representation
            else {
                unreachable!()
            };
            s.nodes.velocity[node as usize]
        }
        Attach::Rigid { rigid, local } => {
            let Representation::Rigid(r) =
                &world.objects()[layout.rigids[rigid as usize].object].representation
            else {
                unreachable!()
            };
            r.velocity + r.spin.cross(r.pose.rotation.rotate(local))
        }
        Attach::Still(_) => Vec3::ZERO,
    }
}

/// Effective inverse mass of the complete weighted contact Jacobian.
/// Endpoints of one rigid edge are correlated through the same six DOFs;
/// summing per-point inverse masses would omit their cross terms.
fn impact_inverse_mass(world: &World, layout: &Layout, impact: &Impact) -> f64 {
    let mut total = 0.0;
    for k in 0..impact.count {
        let (attach, _) = impact.sides[k];
        match attach {
            Attach::Node { solid, node } => {
                if impact.sides[..k].iter().any(|&(a, _)| a == attach) {
                    continue;
                }
                let d = &layout.solids[solid as usize];
                if d.block[node as usize] == HELD {
                    continue;
                }
                let Representation::Solid(s) = &world.objects()[d.object].representation else {
                    unreachable!()
                };
                let w: f64 = impact.sides[..impact.count]
                    .iter()
                    .filter(|&&(a, _)| a == attach)
                    .map(|&(_, w)| w)
                    .sum();
                total += w * w / s.nodes.mass[node as usize];
            }
            Attach::Rigid { rigid, .. } => {
                if impact.sides[..k]
                    .iter()
                    .any(|&(a, _)| matches!(a,Attach::Rigid{rigid:r,..} if r==rigid))
                {
                    continue;
                }
                let Representation::Rigid(r) =
                    &world.objects()[layout.rigids[rigid as usize].object].representation
                else {
                    unreachable!()
                };
                let mut linear = 0.0;
                let mut angular = Vec3::ZERO;
                for &(a, w) in &impact.sides[..impact.count] {
                    if let Attach::Rigid {
                        rigid: other,
                        local,
                    } = a
                    {
                        if other == rigid {
                            linear += w;
                            angular += r.pose.rotation.rotate(local).cross(impact.normal) * w;
                        }
                    }
                }
                total +=
                    linear * linear / r.mass + angular.dot(r.world_inertia().inverse() * angular);
            }
            Attach::Still(_) => {}
        }
    }
    total
}

fn apply_impulse(world: &mut World, layout: &Layout, attach: Attach, impulse: Vec3) {
    match attach {
        Attach::Node { solid, node } => {
            let d = &layout.solids[solid as usize];
            if d.block[node as usize] == HELD {
                return;
            }
            let Representation::Solid(s) = &mut world.objects_mut()[d.object].representation else {
                unreachable!()
            };
            let m = s.nodes.mass[node as usize];
            s.nodes.velocity[node as usize] += impulse / m;
        }
        Attach::Rigid { rigid, local } => {
            let Representation::Rigid(r) =
                &mut world.objects_mut()[layout.rigids[rigid as usize].object].representation
            else {
                unreachable!()
            };
            let arm = r.pose.rotation.rotate(local);
            r.velocity += impulse / r.mass;
            r.spin += r.world_inertia().inverse() * arm.cross(impulse);
        }
        Attach::Still(_) => {}
    }
}

/// The step's objective: the sum of the terms.
struct System<'a> {
    world: &'a World,
    layout: &'a Layout,
    frame: &'a Frame,
    terms: [&'a mut dyn Term; 7],
    hessian: &'a mut Sum<BlockMatrix, BlockMatrix>,
    triplets: &'a mut Triplets,
    profile: &'a mut Profile,
    sweep: &'a mut islands::Sweep,
    groups: &'a [(usize, usize)],
    translation: &'a [(usize, f64)],
}

/// Profile labels: (energy, derivatives, step limit) per term.
fn labels(term: &str) -> (&'static str, &'static str, &'static str) {
    match term {
        "inertia" => (
            "inertia: energy",
            "inertia: derivatives",
            "inertia: step limit",
        ),
        "gravity" => (
            "gravity: energy",
            "gravity: derivatives",
            "gravity: step limit",
        ),
        "mutual gravity" => (
            "mutual gravity: energy",
            "mutual gravity: derivatives",
            "mutual gravity: step limit",
        ),
        "elasticity" => (
            "elasticity: energy",
            "elasticity: derivatives",
            "elasticity: step limit",
        ),
        "contact" => (
            "contact: energy",
            "contact: derivatives",
            "contact: step limit",
        ),
        "rigid bodies" => (
            "rigid bodies: energy",
            "rigid bodies: derivatives",
            "rigid bodies: step limit",
        ),
        "joints" => (
            "joints: energy",
            "joints: derivatives",
            "joints: step limit",
        ),
        _ => ("other: energy", "other: derivatives", "other: step limit"),
    }
}

impl Objective for System<'_> {
    type Hessian = Sum<BlockMatrix, BlockMatrix>;

    fn blocks(&self) -> usize {
        self.layout.blocks
    }

    fn energy(&mut self, pool: &Pool, x: &[f64]) -> f64 {
        let ctx = Context {
            pool,
            world: self.world,
            layout: self.layout,
            frame: self.frame,
        };
        self.sweep.record(&ctx, x);
        let mut total = 0.0;
        for term in self.terms.iter_mut() {
            let label = labels(term.name()).0;
            total += self.profile.time(label, || term.energy(&ctx, x));
            if total == f64::INFINITY {
                break;
            }
        }
        total
    }

    fn derivatives(&mut self, pool: &Pool, x: &[f64], gradient: &mut [f64]) {
        let ctx = Context {
            pool,
            world: self.world,
            layout: self.layout,
            frame: self.frame,
        };
        self.hessian.0.clear_values();
        self.triplets.reset(self.layout.blocks);
        for term in self.terms.iter_mut() {
            let label = labels(term.name()).1;
            let (hessian, triplets) = (&mut self.hessian.0, &mut *self.triplets);
            self.profile.time(label, || {
                term.derivatives(&ctx, x, gradient, hessian, triplets)
            });
        }
        let (triplets, changing) = (&mut *self.triplets, &mut self.hessian.1);
        self.profile
            .time("contact: build matrix", || triplets.assemble_into(changing));
    }

    fn hessian(&self) -> &Self::Hessian {
        self.hessian
    }

    fn coupled(&self) -> &[(usize, usize)] {
        self.groups
    }

    fn polish_step(&mut self, pool: &Pool, x: &[f64], gradient: &[f64], step: &mut [f64]) {
        if self.translation.is_empty() {
            return;
        }
        let ctx = Context {
            pool,
            world: self.world,
            layout: self.layout,
            frame: self.frame,
        };
        if self.terms[6].external_forces(&ctx, x)
            || (!self.layout.fixed.is_empty() && self.terms[5].external_forces(&ctx, x))
        {
            return;
        }
        // Internal force/Hessian terms annihilate uniform translation.
        // Σ H δx = Σ M δx, so these three coarse equations can be solved
        // analytically without projecting the final physical velocities.
        let mut residual = Vec3::ZERO;
        let mut total_mass = 0.0;
        for &(b, m) in self.translation {
            residual += Vec3::read(gradient, b) + Vec3::read(step, b) * m;
            total_mass += m;
        }
        let correction = residual * (-1.0 / total_mass);
        if !correction.is_finite() {
            return;
        }
        for &(b, _) in self.translation {
            (Vec3::read(step, b) + correction).write(step, b);
        }
    }

    fn step_limit(&mut self, pool: &Pool, x: &[f64], dx: &[f64]) -> f64 {
        let ctx = Context {
            pool,
            world: self.world,
            layout: self.layout,
            frame: self.frame,
        };
        let mut limit = 1.0f64;
        for term in self.terms.iter_mut() {
            let label = labels(term.name()).2;
            limit = limit.min(self.profile.time(label, || term.step_limit(&ctx, x, dx)));
        }
        limit
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn rigid_edge_endpoints_share_one_effective_contact_mass() {
        let mut w = World::new();
        w.gravity = Vec3::ZERO;
        let m = w.add_material(core_world::library::rubber());
        for y in [-1.0, 1.0] {
            w.add_rigid(
                "bar",
                m,
                core_space::Shape::cuboid(Vec3::ONE),
                core_math::Pose::at(Vec3::Y * y),
            );
        }
        let layout = Layout::new(&w);
        let sides = [
            (
                Attach::Rigid {
                    rigid: 0,
                    local: -Vec3::X * 0.5,
                },
                0.5,
            ),
            (
                Attach::Rigid {
                    rigid: 0,
                    local: Vec3::X * 0.5,
                },
                0.5,
            ),
            (
                Attach::Rigid {
                    rigid: 1,
                    local: -Vec3::X * 0.5,
                },
                -0.5,
            ),
            (
                Attach::Rigid {
                    rigid: 1,
                    local: Vec3::X * 0.5,
                },
                -0.5,
            ),
        ];
        let i = Impact {
            sides,
            count: 4,
            normal: Vec3::Y,
            restitution: 1.0,
            potential: 0.0,
        };
        let expected = 2.0 / w.objects()[0].mass();
        assert!((impact_inverse_mass(&w, &layout, &i) / expected - 1.0).abs() < 1e-14);
        for &(a, weight) in &sides {
            apply_impulse(&mut w, &layout, a, Vec3::Y * weight);
        }
        let response = relative_velocity(&i, |a| current_attach_velocity(&w, &layout, a));
        assert!((response / expected - 1.0).abs() < 1e-14);
    }
}
