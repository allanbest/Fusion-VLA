//! Fusion L3 — the material point method.
//!
//! PLAN.md §11 makes MPM the spine of Fusion Advanced, and §29 makes the
//! substrate the whole of v0.8. This crate is that substrate: material points
//! carry mass, momentum and deformation history; a background grid does the
//! momentum solve; the two exchange information twice a step.
//!
//! ```text
//!     P2G      particle mass, momentum and stress onto the grid
//!     grid     one explicit velocity update, plus boundary conditions
//!     G2P      grid velocity back to the particles, and F with it
//! ```
//!
//! ## Why MPM and not FEM
//!
//! §11 gives four reasons and the first is decisive for a solo project: no
//! meshing. Large deformation, tearing and splashing need no remeshing and no
//! special cases, because the only persistent structure is the point cloud and
//! points do not have connectivity to invalidate. The grid is rebuilt from
//! scratch every step, which sounds wasteful and is the entire trick.
//!
//! ## MLS-MPM, and what APIC is for
//!
//! The transfer here is the moving-least-squares form (Hu et al. 2018), which
//! folds the force term into the same affine matrix APIC already carries. That
//! is not a micro-optimization — it removes the separate gradient-of-weight
//! loop, which is the part of a naive implementation most likely to be subtly
//! wrong and hardest to notice, because a wrong force gradient produces a
//! solid that behaves plausibly and has the wrong stiffness.
//!
//! APIC (the affine `C` on each point) is what makes the transfer stable
//! *and* non-dissipative. The two older choices each fail one way: PIC
//! transfers only velocity and loses the local rotation every step, so a
//! spinning block grinds to a halt in about a second; FLIP transfers the
//! velocity *difference* and keeps every bit of noise the grid ever had.
//! Carrying the first moment of the velocity field is what avoids choosing.
//!
//! ## The accuracy tier, said in advance
//!
//! §9 requires a solver to declare what it is. This one is explicit, which
//! means conditionally stable: information must not cross a cell in one step.
//! [`SolidMaterial::max_stable_dt`](fusion_core::SolidMaterial::max_stable_dt)
//! computes the bound, [`MpmSolver::substeps_for`] turns it into a substep
//! count, and [`MpmReport::unstable`] says when the scene asked for something
//! the timestep cannot deliver. **Real steel is out of reach** and no amount
//! of substepping brings it in: 200 GPa needs a microsecond step. The material
//! library ships analogues at stiffnesses that run, and says so.
//!
//! ## What this crate does not do
//!
//! No thermal field, no phase change, no damage, no fracture — all v0.9 in
//! §29's table, all fields on this same grid when they come. No self-collision
//! between separate solids beyond what sharing a grid gives for free (which is
//! quite a lot: two solids that meet on the same grid node cannot pass through
//! each other, and that is the method working rather than a feature).

#![deny(unsafe_code)]
#![deny(missing_docs)]

pub mod grid;
pub mod linear;
pub mod model;
pub mod pressure;
pub mod thermal;
pub mod unified;

pub use grid::{weights, Grid, MAX_NODES};
pub use model::{
    energy_density, hardened, principal_stress, return_map, stress, stress_with_damage,
};
pub use thermal::{diffuse, plastic_heat, thermal_stretch};
pub use unified::{UnifiedReport, UnifiedSolver};

use fusion_core::{BodySet, BodyType, SolidSet};
use fusion_math::{outer_product, to_f32, to_f64, DVec3, Isometry, Mat3, Quat, Vec3};

/// What one MPM step did, for diagnostics and for the caller to report.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct MpmReport {
    /// Material points stepped.
    pub points: usize,
    /// Grid nodes that ended the step carrying mass.
    pub active_nodes: usize,
    /// Substeps actually taken.
    pub substeps: u32,
    /// True where the material's wave speed needed a shorter step than the
    /// substep budget allowed.
    ///
    /// Not an error and not a failure — the step still ran. It is the
    /// statement that the answer is outside the method's domain of
    /// applicability, which §2's third principle says has to be legible rather
    /// than inferred from the picture looking wrong.
    pub unstable: bool,
    /// True where the grid a solid asked for was larger than [`MAX_NODES`], so
    /// the step was refused. Only reachable by a solid that has already left.
    pub refused: bool,
    /// Largest `|det F - 1|` after the step.
    pub peak_strain: f32,
    /// The smallest volume ratio anywhere, `min(det F)` (v0.9).
    ///
    /// One is undeformed and zero is a point crushed flat. Reported because
    /// [`MpmReport::peak_strain`] takes an absolute value and cannot tell a
    /// stretched point from a crushed one, and only the crushed one is a state
    /// the material may be unable to come back from — see
    /// [`Constitutive::FixedCorotated`](fusion_core::solid::Constitutive::FixedCorotated).
    pub least_volume: f32,
    /// Hottest material point after the step, Kelvin (v0.9).
    pub peak_temperature: f32,
    /// Fraction of the material that has melted, by point count.
    pub melted: f32,
    /// Diffusion sub-cycles the thermal solve needed.
    pub thermal_cycles: u32,
    /// Worst damage anywhere, in `[0, 1]` (v0.9).
    pub peak_damage: f32,
    /// Fraction of material points that have fully parted.
    pub broken: f32,
    /// Material points pulled back from inversion this step (v1.0).
    ///
    /// See [`model::uninvert`]. Non-zero means the step changed a point's
    /// state by something other than the constitutive model — the answer is
    /// still an answer, and it is no longer only the model's. Reported rather
    /// than repaired quietly, for the same reason [`MpmReport::unstable`] is.
    pub uninverted: usize,
}

impl Default for MpmReport {
    /// Every counter at zero, and `least_volume` at **one**.
    ///
    /// Written out rather than derived for that one field: a derived zero
    /// means "crushed flat", so the report of a scene with no continuum in it
    /// would read as a collapsed one the moment anything looked at it.
    fn default() -> Self {
        Self {
            points: 0,
            active_nodes: 0,
            substeps: 0,
            unstable: false,
            refused: false,
            peak_strain: 0.0,
            least_volume: 1.0,
            peak_temperature: 0.0,
            melted: 0.0,
            thermal_cycles: 0,
            peak_damage: 0.0,
            broken: 0.0,
            uninverted: 0,
        }
    }
}

impl MpmReport {
    /// True where some of the material has been crushed to nearly nothing.
    ///
    /// A tenth of its rest volume, which no real solid reaches elastically and
    /// which a soft one under a corotated law can fall into and stay in. It is
    /// a *statement about the answer* rather than an error: the step ran, and
    /// what it produced is outside the model's useful range. §2's third
    /// principle wants that legible rather than inferred from a picture of a
    /// pancake.
    pub fn collapsed(&self) -> bool {
        self.points > 0 && self.least_volume < 0.1
    }
}

/// The material point method over one or more solids.
///
/// Holds the grid and nothing else that survives a step, which is the point:
/// every piece of state that matters lives on [`SolidSet`] where the checkpoint
/// can see it. That is not tidiness — §17's rewind needs the solver's working
/// set to be derivable from the world at a step boundary, and the cheapest way
/// to guarantee that is for the working set to be rebuilt every step anyway.
#[derive(Clone, Debug)]
pub struct MpmSolver {
    /// The background grid, reused across steps and cleared by each.
    pub grid: Grid,
    /// Ceiling on substeps per call, so a stiff material degrades into a wrong
    /// answer that is reported rather than a frame that never returns.
    pub max_substeps: u32,
    /// Coulomb friction between the continuum and rigid geometry.
    pub friction: f32,
    /// Whether a solid pushes back on the bodies it touches.
    ///
    /// The roadmap asks only for "rigid-body coupling as boundary conditions",
    /// which is one-way: a body moves the material and feels nothing. Two-way
    /// is one accumulation more — the impulse the boundary condition removed
    /// from the grid is the impulse it should hand the body — and without it a
    /// solid cannot rest on a see-saw or float a boat, which is most of what
    /// coupling is for.
    pub two_way: bool,
    /// Diffusion sub-cycles the last substep needed, for the report.
    cycles: u32,
    /// Points [`model::uninvert`] pulled back, accumulated over a step.
    uninverted: usize,
    /// Bodies within reach of the material this substep. Scratch.
    near: Vec<usize>,
    /// Whether the thermal field is solved at all (v0.9).
    ///
    /// On, because every material in the library carries thermal properties
    /// and the inert default does nothing visible: heat diffuses through a
    /// material with no expansion, no melting point and no softening and
    /// changes none of its behaviour. A scene pays for a field it is not using
    /// only in the transfer, which is one pass over the points.
    pub thermal: bool,
}

impl Default for MpmSolver {
    fn default() -> Self {
        Self::new()
    }
}

impl MpmSolver {
    /// A solver with an empty grid and the default dials.
    pub fn new() -> Self {
        Self {
            grid: Grid::new(),
            max_substeps: 24,
            friction: 0.4,
            two_way: true,
            cycles: 0,
            uninverted: 0,
            near: Vec::new(),
            thermal: true,
        }
    }

    /// Drop the grid, keeping the dials.
    ///
    /// The grid is scratch and holds nothing between steps, so this changes no
    /// answer — it exists so a reset returns the allocation rather than
    /// carrying a grid sized for a scene that no longer exists.
    pub fn reset_grid(&mut self) {
        self.grid = Grid::new();
        self.cycles = 0;
    }

    /// How many substeps a solid needs to stay inside its CFL bound, capped by
    /// [`MpmSolver::max_substeps`].
    ///
    /// Returned rather than applied silently so that the caller can report the
    /// cap being hit. A solver that quietly runs 24 substeps because somebody
    /// typed a large Young's modulus is a solver that turns a material mistake
    /// into a performance mystery.
    pub fn substeps_for(&self, solid: &SolidSet, dt: f32) -> (u32, bool) {
        self.substeps_with_bodies(solid, 0.0, dt)
    }

    fn substeps_with_bodies(&self, solid: &SolidSet, max_body_speed: f32, dt: f32) -> (u32, bool) {
        let cell = solid.params.cell_size;
        let wave = solid.params.material.max_stable_dt(cell);
        // Include rigid bodies sweeping through the solid so a fast projectile
        // advances at most half a grid cell per MPM substep instead of tunneling.
        let particle_speed = (0..solid.len())
            .map(|i| solid.particles.velocity[i].length())
            .fold(0.0f32, f32::max);
        let speed = particle_speed.max(max_body_speed);
        let advection = if speed > 1e-6 {
            0.5 * cell / speed
        } else {
            f32::INFINITY
        };
        let limit = wave.min(advection);
        if !limit.is_finite() || limit <= 0.0 {
            return (1, false);
        }
        let wanted = (dt / limit).ceil().max(1.0);
        if wanted > self.max_substeps as f32 {
            (self.max_substeps, true)
        } else {
            (wanted as u32, false)
        }
    }

    /// Advance one solid by `dt`, substepping as its material requires.
    pub fn step(
        &mut self,
        solid: &mut SolidSet,
        bodies: &mut BodySet,
        gravity: Vec3,
        dt: f32,
    ) -> MpmReport {
        let mut report = MpmReport {
            points: solid.len(),
            ..Default::default()
        };
        if solid.is_empty() || dt <= 0.0 {
            return report;
        }

        let mut solid_box = fusion_math::Aabb {
            min: Vec3::splat(f32::INFINITY),
            max: Vec3::splat(f32::NEG_INFINITY),
        };
        let mut fastest_particle = 0.0f32;
        for (p, v) in solid
            .particles
            .position
            .iter()
            .zip(&solid.particles.velocity)
        {
            let p = to_f32(*p);
            solid_box.min = solid_box.min.min(p);
            solid_box.max = solid_box.max.max(p);
            fastest_particle = fastest_particle.max(v.length());
        }
        let solid_reach = solid_box.expanded(solid.params.cell_size + fastest_particle * dt);

        struct SweptBody {
            index: usize,
            start_pos: DVec3,
            end_pos: DVec3,
            start_rot: Quat,
            end_rot: Quat,
        }
        let mut swept: Vec<SweptBody> = Vec::new();
        let mut max_body_speed = 0.0f32;
        for b in 0..bodies.len() {
            if bodies.is_removed(b) || bodies.body_type[b] == BodyType::Static {
                continue;
            }
            let lin = bodies.linear_velocity[b];
            let ang = bodies.angular_velocity[b];
            if lin == Vec3::ZERO && ang == Vec3::ZERO {
                continue;
            }
            let end_pos = bodies.position[b];
            let end_rot = bodies.orientation[b];
            let start_pos = end_pos - to_f64(lin * dt);
            let w_len = ang.length();
            let start_rot = if w_len > 1e-6 {
                (Quat::from_axis_angle(ang / w_len, -w_len * dt) * end_rot).normalize()
            } else {
                end_rot
            };
            let end_aabb = bodies.shape[b].world_aabb(&Isometry::new(to_f32(end_pos), end_rot));
            let start_aabb =
                bodies.shape[b].world_aabb(&Isometry::new(to_f32(start_pos), start_rot));
            let union_aabb = fusion_math::Aabb {
                min: end_aabb.min.min(start_aabb.min),
                max: end_aabb.max.max(start_aabb.max),
            };
            if union_aabb.overlaps(&solid_reach) {
                max_body_speed = max_body_speed.max(lin.length());
                swept.push(SweptBody {
                    index: b,
                    start_pos,
                    end_pos,
                    start_rot,
                    end_rot,
                });
            }
        }

        let (substeps, unstable) = self.substeps_with_bodies(solid, max_body_speed, dt);
        self.uninverted = 0;
        report.substeps = substeps;
        report.unstable = unstable;
        let h = dt / substeps as f32;

        let mut prev_poses: Vec<Isometry> = (0..bodies.len()).map(|b| bodies.pose(b)).collect();
        for s in &swept {
            prev_poses[s.index] = Isometry::new(to_f32(s.start_pos), s.start_rot);
        }

        for step_idx in 0..substeps {
            let t_curr = (step_idx + 1) as f32 / substeps as f32;
            for s in &swept {
                if step_idx > 0 {
                    prev_poses[s.index] = bodies.pose(s.index);
                }
                let pos = s.start_pos + (s.end_pos - s.start_pos) * t_curr as f64;
                let rot = nlerp_quat(s.start_rot, s.end_rot, t_curr);
                bodies.set_pose(s.index, Isometry::new(to_f32(pos), rot));
            }
            if !self.substep_with_prev(solid, bodies, &prev_poses, gravity, h) {
                report.refused = true;
                break;
            }
            report.thermal_cycles = report.thermal_cycles.max(self.cycles);
        }
        for s in &swept {
            bodies.position[s.index] = s.end_pos;
            bodies.orientation[s.index] = s.end_rot;
        }
        report.active_nodes = (0..self.grid.len())
            .filter(|n| self.grid.has_mass(*n))
            .count();
        report.peak_strain = solid.peak_strain();
        report.least_volume = solid.least_volume();
        report.peak_temperature = solid.peak_temperature();
        report.melted = solid.melted_fraction();
        report.peak_damage = solid.peak_damage();
        report.broken = solid.broken_fraction();
        report.uninverted = self.uninverted;
        report
    }

    /// One transfer-solve-transfer cycle. False where the grid was refused.
    fn substep_with_prev(
        &mut self,
        solid: &mut SolidSet,
        bodies: &mut BodySet,
        prev_poses: &[Isometry],
        gravity: Vec3,
        h: f32,
    ) -> bool {
        if !self.grid.rebuild(
            &solid.particles.position,
            solid.params.cell_size,
            solid.params.point_mass(),
        ) {
            return false;
        }
        self.particles_to_grid(solid, h);
        self.solve_grid(bodies, gravity, h);
        // Between the two transfers, because the stencil the points were at
        // during P2G is still the stencil they are at: `grid_to_particles`
        // below is what moves them. Diffusing after the move would carry heat
        // with a set of weights the masses on the grid no longer match.
        if self.thermal {
            self.cycles = thermal::diffuse(solid, &mut self.grid, h);
        }
        // The bodies the material can reach this substep, for the
        // point-level pass in `grid_to_particles`.
        let mut reach = fusion_math::Aabb {
            min: Vec3::splat(f32::INFINITY),
            max: Vec3::splat(f32::NEG_INFINITY),
        };
        let mut fastest = 0.0f32;
        for (p, v) in solid
            .particles
            .position
            .iter()
            .zip(&solid.particles.velocity)
        {
            let p = to_f32(*p);
            reach.min = reach.min.min(p);
            reach.max = reach.max.max(p);
            fastest = fastest.max(v.length());
        }
        let reach = reach.expanded(solid.params.cell_size + fastest * h);
        self.near.clear();
        self.near.extend(
            (0..bodies.len()).filter(|&b| !bodies.is_removed(b) && bodies.aabb(b).overlaps(&reach)),
        );
        let near = std::mem::take(&mut self.near);
        self.grid_to_particles(solid, bodies, prev_poses, &near, h);
        self.near = near;
        true
    }

    /// Mass, momentum and internal stress onto the grid.
    fn particles_to_grid(&mut self, solid: &SolidSet, h: f32) {
        let cell = self.grid.cell();
        let material = solid.params.material;
        let (mu0, lambda0) = material.lame();
        let thermal_coupled = self.thermal;
        let breakable = material.fracture.breaks();
        let mass = solid.params.point_mass();
        // The inverse of the quadratic B-spline's second moment, which for
        // this kernel is `(h²/4) I` — so its inverse is the scalar below and
        // the matrix never has to be formed.
        let d_inv = 4.0 / (cell * cell);

        for p in 0..solid.len() {
            let local = self.grid.to_local(solid.particles.position[p]);
            let (base, fx) = self.grid.stencil(local);
            if !self.grid.holds(base) {
                continue;
            }
            let (wx, wy, wz) = (weights(fx.x), weights(fx.y), weights(fx.z));

            let f = solid.deformation[p];
            let (temperature, melted) = material.thermal.state_of(solid.enthalpy[p]);
            // Two thermal couplings, and they are different in kind. Softening
            // changes how hard the material *pushes back*; expansion changes
            // what counts as undeformed. Only the second touches `F`.
            let (mu, lambda) = if thermal_coupled {
                let (mu, lambda) = material.lame_at(temperature as f32, melted as f32);
                hardened(material.model, mu, lambda, solid.plastic[p])
            } else {
                hardened(material.model, mu0, lambda0, solid.plastic[p])
            };
            let elastic = if thermal_coupled && material.thermal.expansion != 0.0 {
                // A block that has merely grown is not under stress, so the
                // thermal stretch is divided out before the model sees it.
                // Without this a heated free block develops a stress field
                // from nothing, which is the ghost force that makes thermal
                // expansion look like a bug in the solver.
                let stretch = thermal::thermal_stretch(
                    material.thermal.expansion,
                    temperature as f32,
                    solid.temperature_at_rest,
                );
                f * (1.0 / stretch)
            } else {
                f
            };
            let p_stress = if breakable {
                stress_with_damage(material.model, elastic, mu, lambda, solid.damage[p])
            } else {
                stress(material.model, elastic, mu, lambda)
            };
            // MLS-MPM's one idea: the force term and the affine velocity enter
            // the transfer as the same matrix, so the loop below has no
            // gradient-of-weight in it at all.
            let force = p_stress * elastic.transpose() * (-h * solid.volume[p] * d_inv);
            let affine = force + solid.affine[p] * mass;
            let velocity = solid.particles.velocity[p];

            for (k, wk) in wz.iter().enumerate() {
                for (j, wj) in wy.iter().enumerate() {
                    for (i, wi) in wx.iter().enumerate() {
                        let weight = wi * wj * wk;
                        let offset = Vec3::new(i as f32, j as f32, k as f32) - fx;
                        let dpos = offset * cell;
                        let node = self.grid.index(base[0] + i, base[1] + j, base[2] + k);
                        self.grid.mass[node] += weight * mass;
                        self.grid.velocity[node] += (velocity * mass + affine * dpos) * weight;
                    }
                }
            }
        }
    }

    /// One explicit velocity update on the grid, then the boundaries.
    fn solve_grid(&mut self, bodies: &mut BodySet, gravity: Vec3, h: f32) {
        let dims = self.grid.dims();
        let cell = self.grid.cell();
        for k in 0..dims[2] {
            for j in 0..dims[1] {
                for i in 0..dims[0] {
                    let node = self.grid.index(i, j, k);
                    if !self.grid.has_mass(node) {
                        self.grid.velocity[node] = Vec3::ZERO;
                        continue;
                    }
                    let mass = self.grid.mass[node];
                    let mut velocity = self.grid.velocity[node] / mass + gravity * h;
                    let position = self.grid.node_position(i, j, k);
                    for body in 0..bodies.len() {
                        if bodies.is_removed(body) {
                            continue;
                        }
                        // The node is a point and the body is a shape, so the
                        // test is the shape's own SDF — the same one §8
                        // requires of every shape and the same one the fluid's
                        // boundary pass uses. No new collision code.
                        let pose = bodies.pose(body);
                        if !bodies.aabb(body).expanded(cell).contains_point(position) {
                            continue;
                        }
                        let shape = &bodies.shape[body];
                        let local = pose.inverse_transform_point(position);
                        let distance = shape.signed_distance_local(local);
                        // Half a cell of stand-off, because a node exactly on
                        // the surface still owns the material inside it.
                        if distance > 0.5 * cell {
                            continue;
                        }
                        let normal = pose.rotation.rotate(sdf_normal(shape, local));
                        let surface = bodies.velocity_at_point(body, position);
                        let relative = velocity - surface;
                        let approach = relative.dot(normal);
                        if approach >= 0.0 {
                            continue;
                        }
                        // Separate the normal component and let Coulomb take
                        // whatever it can from the tangential one. Written as
                        // a scale on the tangent rather than a subtraction so
                        // that friction can stop a tangent but never reverse
                        // it — the sign flip is the classic way a friction
                        // model turns into a motor.
                        let tangent = relative - normal * approach;
                        let speed = tangent.length();
                        let scale = if speed > 1e-6 {
                            (1.0 + self.friction * approach / speed).max(0.0)
                        } else {
                            0.0
                        };
                        let before = velocity;
                        velocity = surface + tangent * scale;
                        if self.two_way && bodies.is_dynamic(body) {
                            bodies.apply_impulse_at(body, (before - velocity) * mass, position);
                        }
                    }

                    self.grid.velocity[node] = velocity;
                }
            }
        }
    }

    /// Grid velocity back to the points, and the deformation with it.
    fn grid_to_particles(
        &mut self,
        solid: &mut SolidSet,
        bodies: &mut BodySet,
        prev_poses: &[Isometry],
        near: &[usize],
        h: f32,
    ) {
        let cell = self.grid.cell();
        let d_inv = 4.0 / (cell * cell);
        let model = solid.params.material.model;
        let point_mass = solid.params.point_mass();
        let fracture = solid.params.material.fracture;
        let heating = if self.thermal {
            solid.params.material.thermal.plastic_heating
        } else {
            0.0
        };

        for p in 0..solid.len() {
            let local = self.grid.to_local(solid.particles.position[p]);
            let (base, fx) = self.grid.stencil(local);
            if !self.grid.holds(base) {
                continue;
            }
            let (wx, wy, wz) = (weights(fx.x), weights(fx.y), weights(fx.z));

            let mut velocity = Vec3::ZERO;
            let mut affine = Mat3::ZERO;
            for (k, wk) in wz.iter().enumerate() {
                for (j, wj) in wy.iter().enumerate() {
                    for (i, wi) in wx.iter().enumerate() {
                        let weight = wi * wj * wk;
                        let offset = Vec3::new(i as f32, j as f32, k as f32) - fx;
                        let dpos = offset * cell;
                        let node = self.grid.index(base[0] + i, base[1] + j, base[2] + k);
                        let nodal = self.grid.velocity[node];
                        velocity += nodal * weight;
                        affine += outer_product(nodal * weight, dpos) * d_inv;
                    }
                }
            }

            // `F ← (I + h ∇v) F`. The velocity gradient is exactly the affine
            // matrix APIC is already carrying, which is the second thing
            // MLS-MPM gets for free.
            let mut updated = (Mat3::IDENTITY + affine * h) * solid.deformation[p];
            // Before anything reads it. The models below all have a term in
            // `1/σ`, and a point that has been turned inside out is a question
            // none of them answers — see `model::uninvert`.
            if let Some(pulled) = model::uninvert(updated) {
                updated = pulled;
                self.uninverted += 1;
            }
            let (elastic, plastic) = return_map(model, updated);
            if fracture.breaks() {
                // Evaluated here, from the deformation this substep produced,
                // and used by the *next* substep's stress. A thousandth of a
                // second of lag, against a crack that takes tens of
                // milliseconds to run — and it is what keeps the whole damage
                // model out of the transfer, which is read-only over the
                // points.
                let (_, sigma, _) = elastic.svd();
                let stretch = sigma.x.abs().max(sigma.y.abs()).max(sigma.z.abs());
                // Monotone, because a material does not heal. Without the max
                // a crack would close as the pieces settled against each other
                // and the solid would knit itself back together.
                solid.damage[p] = solid.damage[p].max(fracture.damage_at(stretch));
            }
            if heating > 0.0 {
                // What the return map took away is the plastic increment, and
                // the work the stress did through it is the heat. Nothing here
                // is authored: where the bar gets hot is wherever it yielded,
                // and how hot is however much work that took.
                let increment = updated - elastic;
                if increment.double_dot(&increment) > 0.0 {
                    let (temperature, melted) =
                        solid.params.material.thermal.state_of(solid.enthalpy[p]);
                    let (mu, lambda) = solid
                        .params
                        .material
                        .lame_at(temperature as f32, melted as f32);
                    let (mu, lambda) = hardened(model, mu, lambda, solid.plastic[p]);
                    let p_stress = stress(model, elastic, mu, lambda);
                    solid.enthalpy[p] += thermal::plastic_heat(
                        p_stress,
                        increment,
                        solid.volume[p],
                        point_mass,
                        heating,
                    );
                }
            }
            solid.deformation[p] = elastic;
            // Accumulated, and bounded.
            //
            // The published formulation multiplies this every substep and
            // says nothing about its range, which is fine on paper and is not
            // a program: a point that yields in the same direction on every
            // substep multiplies a number slightly above one about five
            // thousand times a second, and it reaches infinity in seconds.
            // The demo reported `yielded inf%` within four seconds, which is
            // how this was found.
            //
            // A factor of twenty either way is far past anything a material
            // means by "permanently compacted" — the hardening exponent it
            // feeds is clamped at the same place — so the bound costs nothing
            // real and removes an unbounded accumulation from the middle of
            // the solver.
            solid.plastic[p] = (solid.plastic[p] * plastic).clamp(0.05, 20.0);
            solid.affine[p] = affine;
            solid.particles.velocity[p] = velocity;
            solid.particles.position[p] += to_f64(velocity * h);
            if !near.is_empty() {
                keep_out_swept(
                    &mut solid.particles.position[p],
                    &mut solid.particles.velocity[p],
                    point_mass,
                    bodies,
                    prev_poses,
                    near,
                    self.friction,
                    self.two_way,
                    velocity * h,
                );
            }
        }
    }
}

/// Put a material point that has ended up inside a rigid body back on its
/// surface, and take away the part of its velocity still going in.
///
/// The grid's boundary condition acts on nodes within half a cell of a
/// surface, which keeps material out as long as a cell is small against what
/// it hits. At the editor's default sampling a cell is a sixth of the
/// object's radius — 17 cm for a metre ball — and a soft ball landing at
/// 12 m/s on a box put a sixth of its points inside the box: the nodes
/// stopped, and the points they were interpolated to did not. Checked here,
/// where each point has just moved, the way most MPM codes do it: the grid
/// for the momentum, the point for where it may be.
///
/// The body is pushed back as the grid's condition pushes it, so the
/// exchange keeps its momentum.
///
/// Out the way it came in, not out of the nearest face. Where two bodies
/// meet — a tank's wall standing on its floor — a point resting on the floor
/// that moved into the wall was nearer the wall's bottom face than its side,
/// was "pushed out" onto the floor's own surface, and slid on inside the
/// wall: a dam break lost a third of its water into the walls in twelve
/// seconds. The point's last step, `travel`, says which face it crossed;
/// the nearest face is the answer only for a point that was already inside.
///
/// And a hair outside, not on the surface. A point at distance exactly zero
/// is outside every body whose face it is on — where a wall's end meets the
/// next wall's face, both — and slid along that shared plane into the wall
/// it was supposed to be outside of.
fn nlerp_quat(a: Quat, b: Quat, t: f32) -> Quat {
    let dot = a.x * b.x + a.y * b.y + a.z * b.z + a.w * b.w;
    let b = if dot < 0.0 {
        Quat::from_xyzw(-b.x, -b.y, -b.z, -b.w)
    } else {
        b
    };
    Quat::from_xyzw(
        a.x + (b.x - a.x) * t,
        a.y + (b.y - a.y) * t,
        a.z + (b.z - a.z) * t,
        a.w + (b.w - a.w) * t,
    )
    .normalize()
}

#[allow(clippy::too_many_arguments)]
pub(crate) fn keep_out(
    position: &mut fusion_math::DVec3,
    velocity: &mut Vec3,
    mass: f32,
    bodies: &mut BodySet,
    near: &[usize],
    friction: f32,
    two_way: bool,
    travel: Vec3,
) {
    keep_out_swept(
        position,
        velocity,
        mass,
        bodies,
        &[],
        near,
        friction,
        two_way,
        travel,
    );
}

#[allow(clippy::too_many_arguments)]
fn keep_out_swept(
    position: &mut fusion_math::DVec3,
    velocity: &mut Vec3,
    mass: f32,
    bodies: &mut BodySet,
    prev_poses: &[Isometry],
    near: &[usize],
    friction: f32,
    two_way: bool,
    travel: Vec3,
) {
    /// How far outside the surface a point is put back, m.
    const SKIN: f32 = 1e-4;
    // Where the point was before this substep, which every body is asked
    // about: a push out of one body can leave it in another one already
    // looked at — out of a wall's bottom face into the floor it stands on —
    // so the bodies are visited again until none holds it.
    let start = to_f32(*position) - travel;
    for _ in 0..3 {
        let mut inside = false;
        for &body in near {
            let point = to_f32(*position);
            if !bodies.aabb(body).contains_point(point) {
                continue;
            }
            let pose = bodies.pose(body);
            let prev_pose = prev_poses.get(body).copied().unwrap_or(pose);
            let shape = &bodies.shape[body];
            let local = pose.inverse_transform_point(point);
            let distance = shape.signed_distance_local(local);
            if distance >= 0.0 {
                continue;
            }
            inside = true;
            let from = prev_pose.inverse_transform_point(start);
            let local_normal = if shape.signed_distance_local(from) > 0.0 {
                // Where along the step it went in: the distance changes sign
                // once on a convex shape, and a few halvings find it well
                // inside a millimetre.
                let (mut outside, mut inside) = (0.0f32, 1.0f32);
                for _ in 0..10 {
                    let mid = 0.5 * (outside + inside);
                    if shape.signed_distance_local(from + (local - from) * mid) > 0.0 {
                        outside = mid;
                    } else {
                        inside = mid;
                    }
                }
                let entry = from + (local - from) * outside;
                // Read just outside, where the face crossed is the nearest.
                let normal = sdf_normal(shape, entry);
                let mut back = local - normal * (normal.dot(local - entry) - SKIN);
                if shape.signed_distance_local(back) < 0.0 {
                    // Not convex there: the nearest face after all.
                    let d = shape.signed_distance_local(back);
                    back -= sdf_normal(shape, back) * (d - SKIN);
                }
                *position += to_f64(pose.rotation.rotate(back - local));
                normal
            } else {
                let normal = sdf_normal(shape, local);
                *position += to_f64(pose.rotation.rotate(normal * (SKIN - distance)));
                normal
            };
            let normal = pose.rotation.rotate(local_normal);
            let surface = bodies.velocity_at_point(body, point);
            let relative = *velocity - surface;
            let approach = relative.dot(normal);
            if approach >= 0.0 {
                continue;
            }
            let tangent = relative - normal * approach;
            let speed = tangent.length();
            let scale = if speed > 1e-6 {
                (1.0 + friction * approach / speed).max(0.0)
            } else {
                0.0
            };
            let before = *velocity;
            *velocity = surface + tangent * scale;
            if two_way && bodies.is_dynamic(body) {
                bodies.apply_impulse_at(body, (before - *velocity) * mass, point);
            }
        }
        if !inside {
            return;
        }
    }
    // Still inside after three rounds: caught between two faces that touch,
    // each pushing it into the other — a wall's end against the next wall's
    // side. Where it started is the one place known to be clear, if it was.
    let clear = |p: Vec3, bodies: &BodySet| {
        near.iter().all(|&body| {
            !bodies.aabb(body).contains_point(p) || {
                let local = bodies.pose(body).inverse_transform_point(p);
                bodies.shape[body].signed_distance_local(local) >= 0.0
            }
        })
    };
    if !clear(fusion_math::to_f32(*position), bodies) && clear(start, bodies) {
        *position = to_f64(start);
    }
}

/// Outward normal of a shape at a local point, by central differences on its
/// signed distance.
///
/// Numerical rather than analytic, and deliberately: §8 requires every shape
/// to carry an SDF and requires nothing about its gradient, so differencing is
/// the only thing that works for *every* shape including the convex hulls.
/// The epsilon is absolute because the shapes are in metres and a relative one
/// would vanish at the origin.
pub(crate) fn sdf_normal(shape: &fusion_core::Shape, local: Vec3) -> Vec3 {
    const E: f32 = 1e-4;
    let at = |p: Vec3| shape.signed_distance_local(p);
    let gradient = Vec3::new(
        at(local + Vec3::X * E) - at(local - Vec3::X * E),
        at(local + Vec3::Y * E) - at(local - Vec3::Y * E),
        at(local + Vec3::Z * E) - at(local - Vec3::Z * E),
    );
    let length = gradient.length();
    if length > 1e-12 {
        gradient / length
    } else {
        // Dead centre of a symmetric shape, where the distance field has no
        // gradient at all. Any direction is as wrong as any other; up is the
        // one that lets a buried point climb out.
        Vec3::Y
    }
}

/// Where a continuum keeps its energy, in joules.
///
/// PLAN.md §29 puts "conservation diagnostics" in v0.9, and this is the half
/// of them the continuum needed: a solid stores energy in a place no rigid
/// body has. Kinetic and potential are the two terms the rest of the engine
/// accounts for; **strain** is the third, and without it a bouncing elastic
/// block appears to lose all its energy at the bottom of the bounce and make
/// it again on the way up.
///
/// Heat is carried separately rather than added in. It is conserved by its own
/// solve and it is a *sink* for the mechanical budget — plastic work moves
/// joules from strain to heat and they do not come back — so a single total
/// would hide the one transfer worth watching.
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct ContinuumEnergy {
    /// Kinetic energy of the material points, J.
    pub kinetic: f64,
    /// Gravitational potential energy, J, on the same convention the ledger
    /// uses: `-m g . x`.
    pub potential: f64,
    /// Elastic strain energy stored in the deformation gradient, J.
    pub strain: f64,
    /// Thermal energy, J. See the note above on why it is not in the total.
    pub heat: f64,
}

impl ContinuumEnergy {
    /// Read every term out of one solid.
    ///
    /// Summed in `f64` throughout — a diagnostic has to be finer than the
    /// thing it is measuring, which is the same argument the ledger makes in
    /// its own module docs. The Lamé parameters are the ones the solver would
    /// use at this instant, thermal softening and plastic hardening included,
    /// because an energy computed from different parameters than the force is
    /// not the potential of that force.
    pub fn of(solid: &SolidSet, gravity: Vec3) -> Self {
        let material = solid.params.material;
        let mass = solid.params.point_mass() as f64;
        let mut energy = Self::default();
        for i in 0..solid.len() {
            let v = solid.particles.velocity[i];
            energy.kinetic += 0.5 * mass * v.length_squared() as f64;
            let p = solid.particles.position[i];
            energy.potential -=
                (gravity.x as f64 * p.x + gravity.y as f64 * p.y + gravity.z as f64 * p.z) * mass;

            let (temperature, melted) = material.thermal.state_of(solid.enthalpy[i]);
            let (mu, lambda) = material.lame_at(temperature as f32, melted as f32);
            let (mu, lambda) = hardened(material.model, mu, lambda, solid.plastic[i]);
            let density = model::energy_at(material.model, solid.deformation[i], mu, lambda);
            energy.strain += density as f64 * solid.volume[i] as f64;
        }
        energy.heat = solid.heat();
        // The elastic parameters above ignore damage, and that is deliberate:
        // the energy a cracked point *stores* is the energy its remaining
        // stiffness will give back, but the work that opened the crack has
        // left the mechanical budget for good. Fracture is a sink, like
        // plasticity, and neither is accounted here — see `docs/STATUS.md`.
        energy
    }

    /// Kinetic, potential and strain: the mechanical budget, J.
    #[inline]
    pub fn mechanical(&self) -> f64 {
        self.kinetic + self.potential + self.strain
    }
}

impl std::ops::Add for ContinuumEnergy {
    type Output = Self;
    fn add(self, other: Self) -> Self {
        Self {
            kinetic: self.kinetic + other.kinetic,
            potential: self.potential + other.potential,
            strain: self.strain + other.strain,
            heat: self.heat + other.heat,
        }
    }
}
