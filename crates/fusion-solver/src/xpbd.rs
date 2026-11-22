//! XPBD — extended position-based dynamics.
//!
//! PLAN.md §10 picks XPBD as Fusion Basic's spine for one reason above all:
//! one mechanism covers contacts, joints, cloth and soft bodies, so there are
//! fewer moving parts to own. This module is the orchestration; the physics
//! lives in `constraint`, `contact` and `joint`, all speaking the same two
//! primitives.
//!
//! The honest caveat, stated in the plan and repeated in [`SolverInfo::caveat`]:
//! XPBD does not conserve energy exactly. Invisible in games and VFX,
//! disqualifying as a scientific integrator — which is why Fusion Advanced
//! gets its own integrators rather than reusing this one. Solver reuse must
//! not become an accuracy leak across the mode boundary.
//!
//! ## The step
//!
//! ```text
//! for each substep h = dt / substeps:
//!     integrate velocities      gravity, damping, gyroscopic term
//!     capture approach speeds   restitution is defined against these
//!     predict positions         x += v*h, ignoring all constraints
//!     solve constraints         contacts and joints, iterated, position level
//!     solve fluids              neighbours, then the density constraint
//!     recover velocities        v += correction / h
//!     solve velocities          restitution and dynamic friction
//!     fluid post-pass           vorticity confinement, then XSPH viscosity
//! ```
//!
//! Contacts are detected once per *step* but solved every *substep*, using the
//! body-local anchors stored in the manifold. Re-running collision detection
//! per substep would be more accurate and several times more expensive; local
//! anchors keep the contact geometry honest as the bodies move.

use fusion_collide::contact::{ContactManifold, WorldContact, MAX_CONTACT_POINTS};
use fusion_math::{Quat, Vec3};

use crate::contact::{self, ContactParameters, ContactScratch};
use crate::fluid::{self, FluidScratch};
use crate::joint::{self, JointScratch};
use crate::ledger::{EnergyLedger, Stage};
use crate::soft;
use crate::{
    integrate, AccuracyTier, Conserves, RigidSolver, SolverInfo, SolverScene, Stability,
    StepContext,
};

/// How far an anchor may move and still be recognised as the same contact
/// point for warm-starting.
///
/// Tighter than `CollisionPipeline::ANCHOR_MATCH_DISTANCE`, which decides
/// whether a contact is the *same contact*. This decides whether it is safe to
/// resume its force, and resuming the wrong one pushes on a surface that is no
/// longer there — so the two thresholds answer different questions and should
/// not be shared.
const WARM_START_MATCH_DISTANCE: f32 = 0.01;

/// What the solver remembers between steps.
///
/// Everything else in [`XpbdWorkingSet`] is scratch that `prepare` rebuilds
/// from the state at the top of every step. This is the exception, and it has
/// to travel in a `Checkpoint` (see `fusion-sim`) or replay stops being replay —
/// a restored run would solve its contacts from zero for one step while the
/// original resumed them, and diverge on the spot.
///
/// Opaque on purpose: callers move it around and a state file saves it, but
/// nothing outside the solver reads it for meaning.
#[derive(Clone, Debug, Default)]
pub struct SolverMemory {
    contacts: Vec<CachedManifold>,
}

/// One pair's remembered contacts, as [`SolverMemory::contacts`] lists them:
/// for each point, its anchor on B, the multiplier it converged to and where
/// it last stuck on A.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct RememberedContact {
    /// The pair, lower index first.
    pub body_a: u32,
    /// The other body.
    pub body_b: u32,
    /// `(anchor on B, multiplier, stuck at on A)`, the first `count` of them.
    pub points: [(Vec3, f32, Vec3); MAX_CONTACT_POINTS],
    /// How many of `points` are real.
    pub count: u8,
}

impl SolverMemory {
    /// Every remembered pair, sorted by pair.
    pub fn contacts(&self) -> impl Iterator<Item = RememberedContact> + '_ {
        self.contacts.iter().map(|c| RememberedContact {
            body_a: c.body_a,
            body_b: c.body_b,
            points: std::array::from_fn(|k| (c.local_b[k], c.lambda[k], c.stick[k])),
            count: c.count,
        })
    }

    /// Memory holding exactly these pairs. Counts above
    /// [`MAX_CONTACT_POINTS`] are cut to it.
    pub fn from_contacts(contacts: impl IntoIterator<Item = RememberedContact>) -> Self {
        let mut out: Vec<CachedManifold> = contacts
            .into_iter()
            .map(|c| CachedManifold {
                body_a: c.body_a,
                body_b: c.body_b,
                local_b: c.points.map(|p| p.0),
                lambda: c.points.map(|p| p.1),
                stick: c.points.map(|p| p.2),
                count: c.count.min(MAX_CONTACT_POINTS as u8),
            })
            .collect();
        out.sort_unstable_by_key(|c| (c.body_a, c.body_b));
        Self { contacts: out }
    }
}

/// One manifold's converged normal multipliers, kept for the next step.
///
/// Lives in the solver's working set rather than in the manifold, on the
/// boundary `fusion_collide::ContactPoint` draws in its own documentation: a
/// contact point is a fact about where two surfaces meet, not a scratch pad
/// for whichever method happens to be running. XPBD wants the multipliers; a
/// future solver would want something else.
#[derive(Clone, Copy, Debug, Default)]
struct CachedManifold {
    body_a: u32,
    body_b: u32,
    /// Anchor on B, matched against the new manifold's anchors. The pipeline
    /// already carries anchors across steps, so a contact that persisted has
    /// the *same* anchor and matches exactly; the distance test is for the
    /// case where it re-anchored.
    local_b: [Vec3; MAX_CONTACT_POINTS],
    lambda: [f32; MAX_CONTACT_POINTS],
    /// Where each contact last stuck — `ContactScratch::stick_local_a`. The
    /// reason static friction needs this cache at all: a reference that does
    /// not survive the step boundary is re-seeded every step, which is the
    /// "since the substep" reference that let a stack random-walk.
    stick: [Vec3; MAX_CONTACT_POINTS],
    count: u8,
}

/// Separation at detection below which a contact counts as touching for the
/// heat solve, metres: a millimetre, around the penetration slop.
const TOUCHING: f32 = 1e-3;

/// One rigid contact point as the step left it. See
/// [`XpbdWorkingSet::contact_reports`].
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct ContactReport {
    /// First body.
    pub body_a: u32,
    /// Second body.
    pub body_b: u32,
    /// The contact on A, in A's frame.
    pub local_a: Vec3,
    /// The contact on B, in B's frame.
    pub local_b: Vec3,
    /// Contact normal in A's frame, from A towards B.
    pub local_normal_a: Vec3,
    /// The positional normal multiplier of the last substep, kg m. Divided by
    /// the substep squared it is the normal force.
    pub normal_lambda: f32,
    /// Whether the surfaces were actually pressed together, as opposed to a
    /// speculative point that was only near.
    pub touching: bool,
    /// Kinetic energy removed over the step, joules. See
    /// `ContactScratch::dissipated`.
    pub dissipated: f32,
    /// Impulse the contact applied to B over the step, world frame, N s; A
    /// received the opposite. See `ContactScratch::impulse`.
    pub impulse: Vec3,
    /// Whether the contact closed faster than an impact speed at some point
    /// in the step: a landing or a collision rather than a resting load.
    pub impacting: bool,
    /// The part of `impulse`, along the normal, that stopped an impact rather
    /// than held a load, N s. See `ContactScratch::impact_impulse`.
    pub impact_impulse: f32,
    /// Fastest the contact closed during the step, m/s.
    pub impact_speed: f32,
    /// The pair's coefficient of restitution.
    pub restitution: f32,
}

/// Per-step scratch. Reused across steps so a running simulation does not
/// allocate.
#[derive(Clone, Debug, Default)]
pub struct XpbdWorkingSet {
    delta_position: Vec<Vec3>,
    delta_rotation: Vec<Vec3>,

    /// The manifolds being solved this step.
    pub manifolds: Vec<ContactManifold>,
    /// Flat per-point solver state; manifold `m` starts at `offsets[m]`.
    contact_scratch: Vec<ContactScratch>,
    offsets: Vec<u32>,
    joint_scratch: Vec<JointScratch>,

    /// Deformable scratch: per-particle correction, and one multiplier per
    /// distance and bending constraint.
    particle_delta: Vec<Vec3>,
    distance_lambda: Vec<f32>,
    bending_lambda: Vec<f32>,

    /// Per-fluid scratch: neighbour lists, densities, multipliers. One entry
    /// per body of fluid in the scene.
    pub fluid: Vec<FluidScratch>,

    /// Total normal multiplier applied per manifold over the step. The
    /// beginnings of the contact-force reporting that impact events and audio
    /// will need.
    pub impulses: Vec<f32>,

    /// Per-stage energy attribution (PLAN.md §23). Inert unless
    /// [`StepContext::energy_ledger`] is set.
    pub ledger: EnergyLedger,

    /// Last step's converged contact multipliers, sorted by body pair so the
    /// lookup is a binary search.
    contact_cache: Vec<CachedManifold>,

    /// Particle-body contacts resolved in the final pass over each system.
    /// Overwritten rather than accumulated, so what survives the step is a
    /// count of the contacts that were active at the end of it — the same
    /// question `contact_count` answers for the rigid bodies. Summing over
    /// iterations instead would report contact *solves*, which is iterations
    /// times larger and means nothing on its own.
    cloth_contacts: usize,
    fluid_contacts: usize,

    /// Each manifold's two orientations when its points were last found:
    /// what [`XpbdSolver::refresh_contacts`] measures how stale they are by.
    detected: Vec<[Quat; 2]>,
    /// Each manifold's pair offset when its points were last found, and its
    /// narrowest gap then — positive for a pair not touching yet.
    detected_offset: Vec<Vec3>,
    detected_gap: Vec<f32>,
    /// Points a re-detection dropped part-way through the step, as they
    /// stood. What they pushed and dissipated before they went is still this
    /// step's, and the heat and stress solves are owed it.
    retired: Vec<ContactReport>,
    refresh_buffer: Vec<WorldContact>,
    refreshes: usize,
    /// Each body's distance from the ground through the contacts, and the
    /// manifolds in that order. See [`contact::propagate_shock`].
    levels: Vec<u32>,
    shock_order: Vec<u32>,
    supports: Vec<bool>,
}

impl XpbdWorkingSet {
    /// Empty scratch. Keep one per simulation and reuse it — it holds the
    /// contact cache that warm-starting reads, so a fresh one every step
    /// would silently turn warm-starting off.
    pub fn new() -> Self {
        Self::default()
    }

    /// Rigid-rigid contact points prepared for this step.
    pub fn contact_count(&self) -> usize {
        self.manifolds.iter().map(|m| m.len()).sum()
    }

    /// Make the contacts' reported dissipation add up to `taken`, the energy
    /// the contact work took out of the bodies this step, J.
    ///
    /// Each pass counts the energy it removes where it can, and some cannot:
    /// a contact resting under a load counts nothing, by design, so that a
    /// block sitting on a floor does not warm from the solver's own
    /// bookkeeping — and that same rule counted nothing at the floor under a
    /// cube a 25 t ball landed on, though the floor stopped the lot. Shock
    /// propagation counts nothing either. Measured, 40–60% of a heavy
    /// landing's energy went nowhere, and the heat solve had nothing to
    /// warm with. The bodies' energy is the total that cannot be argued
    /// with; the passes' own counts say where it went, and so the total is
    /// shared out in proportion to them (by impulse where none counted).
    /// A step that took nothing reports nothing.
    pub fn close_dissipation(&mut self, taken: f64) {
        let mut slots: Vec<(usize, f32)> = Vec::new();
        for (m_index, manifold) in self.manifolds.iter().enumerate() {
            let base = self.offsets[m_index] as usize;
            for k in 0..manifold.len() {
                slots.push((base + k, self.contact_scratch[base + k].dissipated));
            }
        }
        let weight_of = |d: f32, impulse: f32, by_impulse: bool| {
            if by_impulse {
                impulse
            } else {
                d.max(0.0)
            }
        };
        let counted: f64 = slots.iter().map(|&(_, d)| d.max(0.0) as f64).sum::<f64>()
            + self
                .retired
                .iter()
                .map(|r| r.dissipated.max(0.0) as f64)
                .sum::<f64>();
        let by_impulse = counted <= 0.0;
        let total: f64 = if by_impulse {
            slots
                .iter()
                .map(|&(i, _)| self.contact_scratch[i].impulse.length() as f64)
                .sum::<f64>()
                + self
                    .retired
                    .iter()
                    .map(|r| r.impulse.length() as f64)
                    .sum::<f64>()
        } else {
            counted
        };
        let share = if taken > 0.0 && total > 0.0 {
            taken / total
        } else {
            0.0
        };
        for &(i, d) in &slots {
            let s = &mut self.contact_scratch[i];
            s.dissipated = (weight_of(d, s.impulse.length(), by_impulse) as f64 * share) as f32;
        }
        for r in &mut self.retired {
            r.dissipated =
                (weight_of(r.dissipated, r.impulse.length(), by_impulse) as f64 * share) as f32;
        }
    }

    /// Manifolds re-detected part-way through the last step. See
    /// [`XpbdSolver::refresh_contacts`].
    pub fn refreshed_manifolds(&self) -> usize {
        self.refreshes
    }

    /// Particle-body contacts active at the end of the step: cloth, soft
    /// bodies and fluid alike, against rigid bodies.
    ///
    /// Not included, because they are not contacts with anything: a fluid
    /// particle reflecting off a domain wall, which has no body to push back.
    pub fn particle_contact_count(&self) -> usize {
        self.cloth_contacts + self.fluid_contacts
    }

    /// The part of this working set that survives a step, for a checkpoint to
    /// carry.
    ///
    /// Not the whole working set: the neighbour lists a fluid builds are
    /// megabytes and are rebuilt every substep regardless. What is here is
    /// what `prepare` cannot reconstruct — and the guarantee that the list is
    /// complete is `replay_from_a_checkpoint_is_bit_identical`, which caught
    /// this field going missing the day it was added.
    pub fn memory(&self) -> SolverMemory {
        SolverMemory {
            contacts: self.contact_cache.clone(),
        }
    }

    /// Forget the contact multipliers remembered for a body, whose shape has
    /// just changed. See `CollisionPipeline::forget_body`.
    pub fn forget_body(&mut self, body: u32) {
        self.contact_cache
            .retain(|m| m.body_a != body && m.body_b != body);
    }

    /// Put back the part of the working set a checkpoint carried. See
    /// [`XpbdWorkingSet::memory`].
    pub fn restore_memory(&mut self, memory: &SolverMemory) {
        self.contact_cache.clone_from(&memory.contacts);
    }

    /// Every rigid contact point of the last step: where it was, how hard it
    /// pressed, and how much kinetic energy it removed. What the heat solve
    /// reads — and, later, the stress solve.
    pub fn contact_reports(&self) -> impl Iterator<Item = ContactReport> + '_ {
        self.manifolds
            .iter()
            .enumerate()
            .flat_map(move |(m_index, manifold)| {
                let base = self.offsets[m_index] as usize;
                manifold.points().iter().enumerate().map(move |(k, point)| {
                    let s = &self.contact_scratch[base + k];
                    ContactReport {
                        body_a: manifold.body_a,
                        body_b: manifold.body_b,
                        local_a: point.local_a,
                        local_b: point.local_b,
                        local_normal_a: point.local_normal_a,
                        normal_lambda: s.normal_lambda,
                        // Pushed on, or geometrically closed. The second
                        // matters for a settled pile: a sleeping pair is
                        // not solved, so it pushes on nothing, and its
                        // surfaces are no less pressed together for that.
                        touching: s.active || s.normal_lambda > 0.0 || point.separation <= TOUCHING,
                        dissipated: s.dissipated,
                        impulse: s.impulse,
                        impacting: s.impacting || s.speculative_impact > contact::IMPACT_SPEED,
                        impact_impulse: s.impact_impulse,
                        impact_speed: s.impact_speed.max(s.speculative_impact),
                        restitution: manifold.restitution,
                    }
                })
            })
            .chain(self.retired.iter().copied())
    }

    /// Largest normal multiplier currently held by any contact point. Used by
    /// sleeping to tell a settled pile from a bodies-barely-touching one.
    pub fn peak_normal_lambda(&self) -> f32 {
        self.contact_scratch
            .iter()
            .fold(0.0f32, |m, s| m.max(s.normal_lambda))
    }

    /// Worst mean compression across the fluids, as a fraction of rest
    /// density. The one number that says whether the density solve converged.
    pub fn fluid_compression(&self) -> f32 {
        self.fluid
            .iter()
            .fold(0.0f32, |m, f| m.max(f.compression_error()))
    }

    /// Neighbour lists built across every fluid, since this working set was
    /// created. See [`XpbdSolver::fluid_neighbour_tolerance`].
    pub fn fluid_neighbour_builds(&self) -> u64 {
        self.fluid.iter().map(|f| f.builds()).sum()
    }

    /// Largest neighbour count any fluid particle currently has. Rises sharply
    /// just before a fluid solve gets into trouble.
    pub fn peak_fluid_neighbours(&self) -> usize {
        self.fluid
            .iter()
            .map(|f| f.peak_neighbours())
            .max()
            .unwrap_or(0)
    }
}

/// Extended position-based dynamics: the solver everything in Basic runs on.
///
/// One numerical stack for contacts, joints, cloth, soft bodies and fluid.
/// It declares the real-time accuracy tier (§9) and means it: stable and
/// plausible at any stiffness, and not a measurement.
#[derive(Clone, Copy, Debug)]
pub struct XpbdSolver {
    /// Substeps per frame. PLAN.md §10 makes substepping default-on: the
    /// stability gain is disproportionate to the cost, because each substep
    /// re-linearizes the contact rather than iterating a stale one.
    pub substeps: u32,
    /// Constraint iterations per substep.
    pub iterations: u32,
    /// Sweeps of the velocity pass per substep. One sweep only propagates
    /// information one contact deep, which is not enough for a tall stack:
    /// the box at the bottom learns about the box at the top a sweep at a
    /// time.
    pub velocity_iterations: u32,

    /// Contact compliance, m/N. Zero is a perfectly rigid contact; XPBD stays
    /// stable there, unlike a stiffness-based formulation.
    pub contact_compliance: f32,
    /// Contacts shallower than this are left alone. Solving to exactly zero
    /// makes resting bodies jitter between touching and separated.
    pub penetration_slop: f32,
    /// Fraction of the remaining penetration corrected per iteration.
    pub correction_rate: f32,
    /// Ceiling on how fast a contact may push overlapping bodies apart, m/s.
    ///
    /// Without it, a deep overlap is undone in a single substep, and since
    /// velocity is recovered from the correction divided by that substep, the
    /// bodies leave at hundreds of metres per second. The linear part of that
    /// is caught by the velocity pass; the *angular* part is not, and a
    /// correction applied off-centre becomes `I^-1 (r x p) / h` — hundreds of
    /// radians per second for a 5 cm overlap at a 1 ms substep.
    pub max_depenetration_speed: f32,
    /// Impacts slower than this do not bounce, whatever the material says.
    pub restitution_threshold: f32,
    /// Tangential speed above which a contact is sliding rather than stuck.
    pub sliding_threshold: f32,
    /// Speculative contacts: stop fast bodies at surfaces rather than letting
    /// them tunnel. PLAN.md §10 calls CCD "a real correctness tax"; this form
    /// of it costs one comparison per contact point.
    pub speculative: bool,
    /// Solve static friction at the position level as well as the velocity
    /// level.
    ///
    /// **On by default, and known to be imperfect.**
    ///
    /// Velocity-level Coulomb friction alone cannot hold a tall stack together
    /// laterally — it only ever removes velocity that already exists, so a
    /// stack creeps. The positional constraint is what should stop the slide
    /// before it starts, and with it enabled a ten-box tower holds to 0.14 m
    /// over twelve seconds instead of coming down.
    ///
    /// But the slip it measures is contaminated. Slip is taken as the change
    /// in the anchor-to-anchor offset over a substep, projected into the
    /// tangent plane — and the normal solve, applying its corrections one
    /// contact point at a time, tilts the body slightly and moves that offset
    /// tangentially without anything having slid. Friction then "corrects"
    /// depenetration, and a box dropped perfectly flat slides 0.39 m sideways
    /// on landing.
    ///
    /// It is on regardless, because the alternative is worse: with only
    /// velocity-level friction, nothing is ever held — blocks creep down
    /// slopes that should hold them, and stacks come apart. A landing that
    /// slides too far is a bounded error in one situation; friction that
    /// cannot hold anything is wrong everywhere.
    ///
    /// The fix is to measure slip against anchors that persist across steps
    /// rather than against a within-substep delta — the same contact
    /// persistence that warm-starting needs, and the piece of v0.2 still
    /// outstanding.
    pub static_friction: bool,
    /// See `ContactParameters::budget_depenetration`.
    pub budget_depenetration: bool,
    /// Velocity decay for particles, per second. Cloth needs more damping than
    /// rigid bodies: a sheet with none of it never stops rippling, because
    /// nothing else in the model dissipates the energy that bending stores.
    pub cloth_damping: f32,
    /// Friction between cloth and the geometry it drapes over.
    pub cloth_friction: f32,
    /// Fraction of the previous solve's contact impulse re-applied before the
    /// first iteration of each substep.
    ///
    /// Zero disables warm-starting and restores the pre-v0.5 solver.
    ///
    /// **What follows predates shock propagation** (`shock_propagation`),
    /// which finishes each substep's contact solve from the ground up. Since
    /// then a tower creeps under five microns in ten seconds at every fraction
    /// from zero to one, so the table below no longer separates the settings;
    /// it is kept as the record of why 0.8 was chosen. One re-applied the
    /// whole remembered impulse and was then **the worst setting available**.
    ///
    /// Measured as lateral creep over ten seconds, across three stack depths
    /// and seven initial leans — twenty-one configurations, each compared
    /// against the same scene with warm-starting off. A ratio below one is an
    /// improvement. `cargo run --release -p fusion-demos --example warm_table`
    /// re-measures it.
    ///
    /// | fraction | median | geomean | worst case | improved | worst drift |
    /// |---|---|---|---|---|---|
    /// | 0.50 | 0.500 | 0.513 | 1.68x | 20/21 | 1.05 mm |
    /// | 0.70 | 0.300 | 0.286 | 0.55x | 21/21 | 0.34 mm |
    /// | **0.80** | **0.194** | **0.183** | **0.58x** | **21/21** | **0.36 mm** |
    /// | 0.90 | 0.098 | 0.100 | 0.26x | 21/21 | 0.16 mm |
    /// | 0.95 | 0.051 | 0.054 | 0.15x | 21/21 | 0.07 mm |
    /// | 1.00 | 0.958 | 0.746 | 1.68x | 12/21 | 0.41 mm |
    ///
    /// Against warm-starting off, which itself drifts 0.93 mm at worst.
    ///
    /// There is a cliff between 0.95 and 1.0: past it the re-applied impulse
    /// overshoots, the iterations spend themselves taking it back, and the
    /// stack is left no better than if nothing had been remembered.
    ///
    /// It used to be much worse and sit lower. Before each manifold's normal
    /// impulses were solved as a converged block (`contact::solve_velocities`),
    /// the same table read 0.267 median and 3.1 mm worst at 0.80, fell off the
    /// cliff between 0.90 and 0.95 — 22 mm worst at 0.95 — and at 1.00 was
    /// worse than no warm start in 15 of 21 cases, 28 mm at worst. The default
    /// was chosen then as the best *worst* case with a wide margin from the
    /// cliff, and 0.80 still improves all twenty-one. On that same rule 0.90
    /// is now the better choice; it has not been changed, because it changes
    /// every contact-rich scene's answer and deserves its own measurement.
    ///
    /// **The perfectly symmetric tower is not a valid test of this.** The
    /// first measurement taken here used one, reported a 25x improvement, and
    /// was measuring a coincidence — a symmetric column is a degenerate
    /// unstable equilibrium, and which way it falls is chaotic in the solver
    /// parameters. Adding a lean of half a millimetre turned the same
    /// comparison into a 10x *regression*. Every number above is over a spread
    /// of leans for that reason.
    pub contact_warm_start: f32,
    /// Density-constraint iterations per substep, per fluid.
    ///
    /// Fewer than the rigid `iterations` on purpose. The published method runs
    /// four of these per *frame*; substepping already re-linearizes the
    /// constraint eight times, so two per substep is sixteen per frame. It is
    /// also the single most expensive number in a fluid scene — each iteration
    /// is a full pass over every particle's neighbourhood.
    /// How many times per step the fluid is solved, at most.
    ///
    /// A fluid does not need the rigid solver's substep rate. Substepping
    /// exists to re-linearize *contacts* — a stack of eight boxes needs the
    /// contact geometry refreshed eight times before the bottom one knows what
    /// is on top of it. A density constraint has no such geometry to go stale;
    /// PBF is unconditionally stable and the published method runs it once per
    /// frame at 1/60 s. Running it eight times at 1/1920 s costs eight times as
    /// much and buys nothing, and it actively hurts: velocity is recovered as
    /// `correction / h`, so a smaller `h` turns the same residual correction
    /// into a larger spurious velocity.
    ///
    /// Rounded down to a divisor of `substeps`, so the fluid always solves on
    /// substep boundaries and the bodies it is pushing against are in a state
    /// the rigid solver has finished with.
    ///
    /// Cost is linear in this number and quality is not, which is the whole
    /// reason to tune it. Measured on a 3528-particle pool with a floating
    /// cube of density 300, at the default two iterations:
    ///
    /// | fluid substeps | relative cost | residual compression | cube submerged |
    /// |---|---|---|---|
    /// | 1 | 1.0x | 0.538% | 10% |
    /// | 2 | 2.0x | 0.132% | 16% |
    /// | **4** | **4.0x** | **0.034%** | **27%** |
    /// | 8 | 8.0x | 0.010% | 29% |
    ///
    /// Archimedes says 30%. Four is the knee: it recovers almost all of the
    /// buoyancy, and doubling again buys two percentage points for twice the
    /// cost. Below it the fluid is not converged enough to hold a body up —
    /// note that the compression at one substep is still only half a percent,
    /// which looks fine and floats nothing. Compression alone is not a
    /// sufficient quality measure for a fluid that has to push on something.
    pub fluid_substeps: u32,
    /// Density-solve iterations per fluid substep.
    pub fluid_iterations: u32,
    /// How far a particle may move before its neighbour list is rebuilt, as a
    /// fraction of the smoothing radius. Zero rebuilds every substep.
    ///
    /// The neighbour build is the largest single cost in a fluid scene — 38%
    /// of a frame of the 3528-particle tank, against 30% for the density solve
    /// it feeds — and at four fluid substeps per step it ran sixteen times per
    /// output frame. Most of those rebuilds produced the list that was already
    /// there: between substeps at 1/960 s a particle moves well under a
    /// millimetre, and the smoothing radius is a hundred.
    ///
    /// See [`FluidScratch::neighbours_are_stale`] for what a reused list gets
    /// wrong and why the answer is "the pairs whose weight is nearly zero".
    /// 1/50 of the radius is conservative against that argument by a wide
    /// margin; it is not tuned to a benchmark, because the quantity it trades
    /// against is bounded analytically rather than measured.
    pub fluid_neighbour_tolerance: f32,
    /// Fraction of the computed density correction actually applied, per
    /// iteration.
    ///
    /// An under-relaxation, because the density solve is Jacobi (see
    /// `fluid::solve_density` for why it has to be): every particle in a
    /// compressed region applies its full correction as though its neighbours
    /// were staying put, and they are not.
    ///
    /// Measured on a 3528-particle pool with a floating cube of density 300.
    /// Buoyancy is the sensitive column — it is the *difference* between the
    /// pressure above and below a body, so it degrades long before anything
    /// looks wrong:
    ///
    /// | iterations x rate | ms/step | residual compression | cube submerged |
    /// |---|---|---|---|
    /// | 1 x 0.3 | 13.5 | 0.109% | 18% |
    /// | 1 x 0.5 | 13.5 | 0.065% | 23% |
    /// | 1 x 1.0 | 13.1 | 0.055% | 18% |
    /// | 2 x 0.25 | 15.5 | 0.065% | 27% |
    /// | **2 x 0.5** | **16.1** | **0.033%** | **31%** |
    /// | 2 x 1.0 | 15.6 | 0.020% | 24% |
    /// | 4 x 0.5 | 21.1 | 0.016% | 25% |
    ///
    /// Archimedes says 30%. Two iterations at half strength is the entry that
    /// agrees with him, and more iterations do not improve on it — past this
    /// point the residual compression keeps falling while the buoyancy does
    /// not, because what is left is the resolution of the geometric coupling
    /// rather than the convergence of the solve.
    ///
    /// **None of these diverge**, which is worth saying because an earlier
    /// version of this table had four entries marked as divergent and a
    /// paragraph explaining that the product of the two numbers was the
    /// stability parameter. That was real and it was not the method: the
    /// iterations were not re-reading the particle positions, so a second
    /// iteration recomputed an identical correction and applied it twice
    /// instead of converging. Two-at-half really was one-at-full, the solver
    /// really did diverge past a total of one, and the explanation fitted the
    /// measurements perfectly while being wrong about the cause. Fixing the
    /// refresh in `solve_density` removed the cliff, halved the cost of the
    /// settings that had been needed to stay clear of it, and improved the
    /// answer.
    pub fluid_relaxation: f32,
    /// Ceiling on fluid particle speed, m/s. See [`fluid::clamp_speed`] for
    /// why a fluid needs one and a rigid body does not.
    pub max_fluid_speed: f32,
    /// Friction between fluid particles and the geometry that contains them.
    ///
    /// Low, and low on purpose: the boundary is free-slip in reality and there
    /// is no boundary layer here to make it otherwise. A small value stops
    /// water sliding across a container floor as if it were ice, without
    /// pretending to model wall shear.
    pub fluid_friction: f32,
    /// Integrate the gyroscopic term of Euler's equations.
    ///
    /// Physically mandatory for bodies whose principal moments of inertia
    /// differ, and free for those whose do not — the integrator skips it when
    /// the inertia is isotropic. Exposed so a scene that would rather have
    /// placid rotation than correct rotation can say so explicitly.
    pub gyroscopic: bool,
    /// Re-detect a pair part-way through the step once either body has
    /// turned far enough that the contacts found at the top of it are wrong
    /// by more than [`REFRESH_ERROR`].
    ///
    /// Contacts are found once a step and carried by body-local anchors,
    /// which is exact for a body that slides and wrong for one that spins:
    /// an anchor is a material point, and a quarter turn carries it a
    /// quarter of the way round the body while the corner that turns into
    /// the surface has no contact at all. Two boxes meeting corner-first at
    /// a combined 60 m/s come apart at a hundred radians a second — the
    /// right answer; an off-centre impulse does that — and with contacts
    /// from the start of the step they went 200 mm into each other. A
    /// spinning ball's anchor rises off the floor as it turns, and the ball
    /// sank by as much.
    pub refresh_contacts: bool,
    /// Finish each substep's contact solve with one pass from the ground up
    /// in which the body underneath does not move. What keeps a light board
    /// under a heavy block out of the floor. See [`contact::propagate_shock`].
    pub shock_propagation: bool,
}

/// How far a contact may be off, from the turning of its bodies since it
/// was found, before the pair is detected again, metres. A body of radius
/// `r` turned through `θ` has moved a material point `r (1 - cos θ)`, about
/// `r θ² / 2`, off the line the contact was measured along.
pub const REFRESH_ERROR: f32 = 1e-3;

impl Default for XpbdSolver {
    fn default() -> Self {
        Self {
            // Measured against a ten-box resting tower over twelve seconds of
            // simulated time. Lateral drift at these settings is 0.14 m and
            // falling; at four iterations it is 5.7 m and rising, and the
            // tower comes down. Static friction is what holds a stack
            // sideways, and it needs iterations to converge — which is exactly
            // the "iteration-count problem" PLAN.md §10 names, and exactly
            // what contact warm-starting would relieve.
            substeps: 8,
            iterations: 8,
            velocity_iterations: 2,
            contact_compliance: 0.0,
            penetration_slop: 5e-4,
            correction_rate: 0.8,
            max_depenetration_speed: 2.0,
            restitution_threshold: 0.5,
            sliding_threshold: 0.05,
            speculative: true,
            static_friction: true,
            budget_depenetration: true,
            cloth_damping: 0.4,
            cloth_friction: 0.5,
            contact_warm_start: 0.8,
            fluid_substeps: 4,
            fluid_iterations: 2,
            fluid_neighbour_tolerance: 0.02,
            fluid_relaxation: 0.5,
            max_fluid_speed: 30.0,
            fluid_friction: 0.05,
            gyroscopic: true,
            refresh_contacts: true,
            shock_propagation: true,
        }
    }
}

impl XpbdSolver {
    /// A solver with this many substeps and iterations, and defaults for
    /// everything else. Both are clamped to at least one.
    pub fn new(substeps: u32, iterations: u32) -> Self {
        Self {
            substeps: substeps.max(1),
            iterations: iterations.max(1),
            ..Default::default()
        }
    }

    fn contact_parameters(&self, substep: f32) -> ContactParameters {
        ContactParameters {
            substep,
            compliance: self.contact_compliance,
            penetration_slop: self.penetration_slop,
            correction_rate: self.correction_rate,
            max_depenetration_speed: self.max_depenetration_speed,
            restitution_threshold: self.restitution_threshold,
            sliding_threshold: self.sliding_threshold,
            speculative: self.speculative,
            static_friction: self.static_friction,
            budget_depenetration: self.budget_depenetration,
        }
    }
}

impl RigidSolver for XpbdSolver {
    type WorkingSet = XpbdWorkingSet;

    fn prepare(
        &self,
        scene: &SolverScene<'_>,
        manifolds: &[ContactManifold],
        working: &mut XpbdWorkingSet,
    ) {
        let bodies = &*scene.bodies;
        working.delta_position.clear();
        working.delta_position.resize(bodies.len(), Vec3::ZERO);
        working.delta_rotation.clear();
        working.delta_rotation.resize(bodies.len(), Vec3::ZERO);

        working.manifolds.clear();
        working.manifolds.extend_from_slice(manifolds);
        working.offsets.clear();
        working.offsets.reserve(manifolds.len());

        // Room for a full manifold each, so a pair re-detected part-way
        // through the step can grow without moving every point after it.
        for m_index in 0..manifolds.len() {
            working.offsets.push((m_index * MAX_CONTACT_POINTS) as u32);
        }
        working.contact_scratch.clear();
        working.contact_scratch.resize(
            manifolds.len() * MAX_CONTACT_POINTS,
            ContactScratch::default(),
        );
        working.detected.clear();
        working.detected.extend(manifolds.iter().map(|m| {
            [
                bodies.orientation[m.body_a as usize],
                bodies.orientation[m.body_b as usize],
            ]
        }));
        working.detected_offset.clear();
        working.detected_offset.extend(
            manifolds
                .iter()
                .map(|m| bodies.relative_translation(m.body_a as usize, m.body_b as usize)),
        );
        working.detected_gap.clear();
        working.detected_gap.extend(manifolds.iter().map(gap_of));
        working.retired.clear();
        working.refreshes = 0;
        if self.shock_propagation {
            contact::support_levels(bodies, manifolds, &mut working.levels);
            // A pair of moving bodies at different heights, and the ground
            // contacts of any body that holds another up. Against something
            // that does not move a contact has no mass ratio to lose to, but
            // a board's contact with the floor is where the block's weight
            // has to end: solved only in the iterations, the board kept the
            // block's fall and carried it into the floor.
            let levels = &working.levels;
            let mut supports = std::mem::take(&mut working.supports);
            supports.clear();
            supports.resize(bodies.len(), false);
            let stacked = |m: &ContactManifold| {
                let (a, b) = (m.body_a as usize, m.body_b as usize);
                levels[a] != levels[b]
                    && levels[a].min(levels[b]) != u32::MAX
                    && bodies.is_movable(a)
                    && bodies.is_movable(b)
                    && m.points().iter().any(|p| p.separation <= 0.0)
            };
            for m in manifolds.iter().filter(|m| stacked(m)) {
                let (a, b) = (m.body_a as usize, m.body_b as usize);
                supports[if levels[a] < levels[b] { a } else { b }] = true;
            }
            working.shock_order.clear();
            working
                .shock_order
                .extend((0..manifolds.len() as u32).filter(|&i| {
                    let m = &manifolds[i as usize];
                    let (a, b) = (m.body_a as usize, m.body_b as usize);
                    stacked(m)
                        || (levels[a] != levels[b]
                            && m.points().iter().any(|p| p.separation <= 0.0)
                            && ((bodies.is_movable(a) && supports[a])
                                || (bodies.is_movable(b) && supports[b])))
                }));
            working.shock_order.sort_by_key(|&i| {
                let m = &manifolds[i as usize];
                (levels[m.body_a as usize].min(levels[m.body_b as usize]), i)
            });
            working.supports = supports;
        } else {
            working.shock_order.clear();
        }

        // Seed each point with what the same contact converged to last step.
        // `capture_approach_velocities` rotates this into the warm-start slot
        // on the first substep, so the seeding path and the substep-to-substep
        // path are the same code from here on.
        //
        // The same match also recovers where each contact last stuck, which
        // is friction state rather than a warm start, so it runs whatever
        // `contact_warm_start` says. A contact with no history is held where
        // it was born, which for a new contact is where it touched.
        for (m_index, manifold) in manifolds.iter().enumerate() {
            let base = working.offsets[m_index] as usize;
            for (k, point) in manifold.points().iter().enumerate() {
                working.contact_scratch[base + k].stick_local_a = point.local_a;
            }
        }
        {
            for (m_index, manifold) in manifolds.iter().enumerate() {
                let base = working.offsets[m_index] as usize;
                let Ok(cached) = working
                    .contact_cache
                    .binary_search_by_key(&(manifold.body_a, manifold.body_b), |c| {
                        (c.body_a, c.body_b)
                    })
                else {
                    continue;
                };
                let cached = working.contact_cache[cached];
                for (k, point) in manifold.points().iter().enumerate() {
                    if !point.inherited {
                        continue;
                    }
                    let mut best = f32::INFINITY;
                    let mut lambda = 0.0;
                    let mut stick = point.local_a;
                    for i in 0..cached.count as usize {
                        let d = cached.local_b[i].distance_squared(point.local_b);
                        if d < best {
                            best = d;
                            lambda = cached.lambda[i];
                            stick = cached.stick[i];
                        }
                    }
                    if best <= WARM_START_MATCH_DISTANCE * WARM_START_MATCH_DISTANCE {
                        let scratch = &mut working.contact_scratch[base + k];
                        if self.contact_warm_start > 0.0 {
                            scratch.normal_lambda = lambda;
                        }
                        scratch.stick_local_a = stick;
                    }
                }
            }
        }

        working.impulses.clear();
        working.impulses.resize(manifolds.len(), 0.0);

        working.joint_scratch.clear();
        working
            .joint_scratch
            .resize(scene.joints.len(), JointScratch::default());

        working.particle_delta.clear();
        working
            .particle_delta
            .resize(scene.particles.len(), Vec3::ZERO);
        working.distance_lambda.clear();
        working.distance_lambda.resize(scene.distance.len(), 0.0);
        working.bending_lambda.clear();
        working.bending_lambda.resize(scene.bending.len(), 0.0);

        // Fluid scratch is resized rather than cleared: the neighbour lists
        // and the calibrated rest density are the expensive part, and a fluid
        // does not change resolution between steps.
        working
            .fluid
            .resize_with(scene.fluids.len(), FluidScratch::new);
        for (scratch, fluid) in working.fluid.iter_mut().zip(scene.fluids.iter()) {
            scratch.prepare(fluid);
        }
    }

    fn step(&self, scene: &mut SolverScene<'_>, working: &mut XpbdWorkingSet, ctx: &StepContext) {
        let h = ctx.dt / self.substeps as f32;
        let params = self.contact_parameters(h);
        let joints = scene.joints;
        let bodies = &mut *scene.bodies;

        // Which substeps the fluid runs on, and with what timestep. See
        // `fluid_substeps`.
        let fluid_stride = (self.substeps / self.fluid_substeps.max(1)).max(1);
        let fluid_steps = self.substeps.div_ceil(fluid_stride);
        let fluid_h = ctx.dt / fluid_steps as f32;

        working.ledger.set_enabled(ctx.energy_ledger);
        working
            .ledger
            .begin_step(bodies, scene.particles, scene.fluids, ctx.gravity);
        // One macro rather than ten near-identical five-line calls. The
        // borrow dance is the reason it is a macro and not a closure: a
        // closure would have to capture `bodies` and `working` for its whole
        // lifetime, and the solver needs both mutably in between samples.
        macro_rules! ledger {
            ($stage:expr) => {
                working
                    .ledger
                    .sample($stage, bodies, scene.particles, scene.fluids, ctx.gravity);
            };
        }

        // See `close_dissipation`. Only where every change of the bodies'
        // energy over the contact work is the contacts' doing: a joint's
        // motor, a cloth draped over a body or a fluid pushing one also move
        // energy in and out of them, and that is not heat.
        let closing = ctx.close_dissipation
            && joints.is_empty()
            && scene.particles.is_empty()
            && scene.fluids.iter().all(|f| f.is_empty());
        let mut taken = 0.0f64;

        for substep in 0..self.substeps {
            let fluid_now = substep % fluid_stride == 0;
            if self.refresh_contacts && substep > 0 {
                refresh_contacts(bodies, working, ctx, h, ctx.dt - substep as f32 * h);
            }
            integrate::integrate_velocities(
                bodies,
                ctx.gravity,
                ctx.linear_damping,
                ctx.angular_damping,
                h,
                self.gyroscopic,
            );
            soft::integrate_particles(scene.particles, ctx.gravity, self.cloth_damping, h);
            // Fluids take gravity through the same integrator and no damping.
            // Cloth needs damping because bending stores energy nothing else
            // dissipates; a fluid has XSPH viscosity for that job, and adding
            // a second global decay on top would make the viscosity dial mean
            // two different things at once.
            if fluid_now {
                for f in scene.fluids.iter_mut() {
                    soft::integrate_particles(&mut f.particles, ctx.gravity, 0.0, fluid_h);
                }
            }
            ledger!(Stage::Integrate);

            crate::aero::apply_to_bodies(bodies, &ctx.air, h);
            crate::aero::apply_to_cloth(scene.particles, scene.triangles, &ctx.air, h);
            ledger!(Stage::Aerodynamics);

            // Restitution is defined against the velocity a contact arrives
            // with, so it has to be recorded before the position solve
            // replaces it. This also resets the per-substep multipliers: the
            // constraint is re-linearized at the new positions each substep,
            // so the previous substep's multiplier is not a valid start.
            contact::capture_approach_velocities(
                bodies,
                &working.manifolds,
                &working.offsets,
                &mut working.contact_scratch,
            );
            for s in &mut working.joint_scratch {
                *s = JointScratch::default();
            }
            working.distance_lambda.iter_mut().for_each(|l| *l = 0.0);
            working.bending_lambda.iter_mut().for_each(|l| *l = 0.0);

            integrate::predict_positions(bodies, h);
            soft::predict_particles(scene.particles, h);
            if fluid_now {
                for f in scene.fluids.iter_mut() {
                    soft::predict_particles(&mut f.particles, fluid_h);
                }
            }
            working
                .particle_delta
                .iter_mut()
                .for_each(|d| *d = Vec3::ZERO);
            working
                .delta_position
                .iter_mut()
                .for_each(|d| *d = Vec3::ZERO);
            working
                .delta_rotation
                .iter_mut()
                .for_each(|d| *d = Vec3::ZERO);
            working
                .particle_delta
                .iter_mut()
                .for_each(|d| *d = Vec3::ZERO);
            if !scene.particles.is_empty() {
                soft::sweep_particles(
                    scene.particles,
                    bodies,
                    &mut working.particle_delta,
                    &mut working.delta_position,
                    &mut working.delta_rotation,
                    h,
                );
            }
            let before = if closing {
                crate::ledger::body_energy(bodies, ctx.gravity)
            } else {
                0.0
            };
            contact::warm_start(
                bodies,
                &working.manifolds,
                &working.offsets,
                &mut working.contact_scratch,
                &params,
                self.contact_warm_start,
                &mut working.delta_position,
                &mut working.delta_rotation,
                &mut working.impulses,
            );
            ledger!(Stage::Predict);

            for _ in 0..self.iterations {
                contact::solve_positions(
                    bodies,
                    &working.manifolds,
                    &working.offsets,
                    &mut working.contact_scratch,
                    &params,
                    h,
                    &mut working.delta_position,
                    &mut working.delta_rotation,
                    &mut working.impulses,
                );
                ledger!(Stage::Contacts);
                joint::solve_positions(
                    bodies,
                    joints,
                    &mut working.joint_scratch,
                    h,
                    &mut working.delta_position,
                    &mut working.delta_rotation,
                );
                ledger!(Stage::Joints);
                soft::solve_distance(
                    scene.particles,
                    scene.distance,
                    &mut working.distance_lambda,
                    &mut working.particle_delta,
                    h,
                );
                soft::solve_bending(
                    scene.particles,
                    scene.bending,
                    &mut working.bending_lambda,
                    &mut working.particle_delta,
                    h,
                );
                working.cloth_contacts = soft::solve_particle_body_contacts(
                    scene.particles,
                    bodies,
                    &mut working.particle_delta,
                    &mut working.delta_position,
                    &mut working.delta_rotation,
                    params.penetration_slop,
                    self.cloth_friction,
                    h,
                );
                ledger!(Stage::Deformables);
            }
            if !working.shock_order.is_empty() {
                contact::propagate_shock(
                    bodies,
                    &working.manifolds,
                    &working.offsets,
                    &mut working.contact_scratch,
                    &params,
                    h,
                    &working.levels,
                    &working.shock_order,
                    &mut working.delta_position,
                    &mut working.delta_rotation,
                );
                ledger!(Stage::Contacts);
            }

            // --- fluids ---
            //
            // After the rigid solve, not before it and not interleaved with
            // it. The boundary pass pushes particles out of bodies that have
            // already been moved this substep, so the container a fluid is
            // measured against is the one it will actually be in at the end of
            // the substep rather than the one it was in at the start. Solving
            // the fluid first put a sloshing tank half a centimetre ahead of
            // its own walls and leaked particles through the corners.
            //
            // `fluids_touching` collects each fluid's last-iteration boundary
            // contact count, so what is reported is every fluid in the scene
            // as it stood at the end of the substep.
            let mut fluids_touching = 0;
            for (index, fluid) in scene.fluids.iter_mut().enumerate() {
                if !fluid_now {
                    break;
                }
                let scratch = &mut working.fluid[index];
                scratch.begin_substep();
                // Only where they have gone stale. The boundary pass below is
                // rebuilt regardless: it is a fortieth of the cost and it
                // reads body poses, which the rigid solve has just moved.
                let tolerance = self.fluid_neighbour_tolerance * fluid.params.smoothing_radius;
                if scratch.neighbours_are_stale(fluid, tolerance) {
                    fluid::build_neighbours(fluid, scratch);
                }
                fluid::build_boundary_density(fluid, bodies, scratch);
                soft::sweep_particles(
                    &mut fluid.particles,
                    bodies,
                    scratch.accumulated_mut(),
                    &mut working.delta_position,
                    &mut working.delta_rotation,
                    fluid_h,
                );
                let mut resolved = 0;
                for _ in 0..self.fluid_iterations {
                    fluid::solve_density(fluid, scratch, self.fluid_relaxation);
                    resolved = fluid::solve_boundaries(
                        fluid,
                        bodies,
                        scratch.accumulated_mut(),
                        &mut working.delta_position,
                        &mut working.delta_rotation,
                        params.penetration_slop,
                        self.fluid_friction,
                        fluid_h,
                    );
                }
                fluids_touching += resolved;
            }
            // Only where the fluid actually ran. Fluids are solved on a
            // stride, so the last substep of a step is usually not one of
            // theirs, and assigning unconditionally overwrote a real count
            // with the zero of a substep that never looked.
            if fluid_now {
                working.fluid_contacts = fluids_touching;
            }
            ledger!(Stage::Fluid);

            integrate::recover_velocities(
                bodies,
                &working.delta_position,
                &working.delta_rotation,
                h,
            );
            soft::recover_particle_velocities(scene.particles, &working.particle_delta, h);
            for (index, fluid) in scene.fluids.iter_mut().enumerate() {
                if !fluid_now {
                    break;
                }
                soft::recover_particle_velocities(
                    &mut fluid.particles,
                    working.fluid[index].accumulated(),
                    fluid_h,
                );
            }
            ledger!(Stage::Recover);

            // Vorticity and viscosity are velocity-level and belong after
            // recovery: applied before it, the position solve's own correction
            // would overwrite them wholesale.
            //
            // Split into its own loop rather than folded into the one above so
            // the ledger can tell them apart. It matters here more than
            // anywhere: vorticity confinement adds energy by construction, and
            // a breakdown that hid that inside `recover` would make the one
            // stage with a licence to gain energy invisible.
            for (index, fluid) in scene.fluids.iter_mut().enumerate() {
                if !fluid_now {
                    break;
                }
                let scratch = &mut working.fluid[index];
                fluid::apply_viscosity_and_vorticity(fluid, scratch, fluid_h);
            }
            ledger!(Stage::FluidPost);

            for fluid in scene.fluids.iter_mut() {
                if !fluid_now {
                    break;
                }
                fluid::clamp_speed(fluid, self.max_fluid_speed);
            }
            ledger!(Stage::SpeedClamp);

            contact::record_impact_dissipation(
                bodies,
                &working.manifolds,
                &working.offsets,
                &mut working.contact_scratch,
                h,
            );
            for _ in 0..self.velocity_iterations {
                contact::solve_velocities(
                    bodies,
                    &working.manifolds,
                    &working.offsets,
                    &mut working.contact_scratch,
                    &params,
                    h,
                );
            }
            if !working.shock_order.is_empty() {
                contact::propagate_shock_velocities(
                    bodies,
                    &working.manifolds,
                    &working.offsets,
                    &mut working.contact_scratch,
                    h,
                    &working.levels,
                    &working.shock_order,
                );
            }
            if closing {
                taken += before - crate::ledger::body_energy(bodies, ctx.gravity);
            }
            ledger!(Stage::VelocityPass);
        }

        if closing {
            working.close_dissipation(taken);
        }
        save_contact_cache(working);
    }

    fn info(&self) -> SolverInfo {
        SolverInfo {
            name: "XPBD",
            order: 1,
            stability: Stability::PositionBased,
            // Momentum is preserved by construction: every correction applies
            // equal and opposite impulses. Energy is not.
            conserves: Conserves {
                momentum: true,
                energy: false,
                mass: true,
            },
            tier: AccuracyTier::RealTime,
            caveat: "Position-based contacts are stable and plausible but not \
                     energy-conserving; results are not measurements.",
        }
    }
}

/// Re-detect each pair whose bodies have turned enough since its points were
/// found. See [`XpbdSolver::refresh_contacts`].
///
/// A point found again keeps its slot — and with it what the step has
/// already recorded there, the impulse, the dissipation, whether it was an
/// impact — and takes the fresh anchors. One not found again is retired with
/// its record intact; one new to this detection starts clean.
fn refresh_contacts(
    bodies: &fusion_core::BodySet,
    working: &mut XpbdWorkingSet,
    ctx: &StepContext,
    h: f32,
    remaining: f32,
) {
    use fusion_collide::{anchor_reach, collide_pair, inherit_anchors, sweep_of};
    for m_index in 0..working.manifolds.len() {
        let old = working.manifolds[m_index];
        let (ia, ib) = (old.body_a as usize, old.body_b as usize);
        let off = |i: usize, then: Quat| {
            let w = (bodies.orientation[i] * then.conjugate()).w.abs().min(1.0);
            let turned = 2.0 * w.acos();
            0.5 * bodies.shape[i].bounding_radius() * turned * turned
        };
        // How far turning carries a body's surface in one substep. Past the
        // same millimetre, a corner swinging in can be a centimetre inside
        // before the drift test above notices — at a hundred radians a second
        // a 30 cm box's corner moves 13 mm a substep — so the pair is found
        // again every substep, and its points chosen for the next one only.
        let spin = |i: usize| {
            if bodies.is_movable(i) || bodies.is_kinematic(i) {
                bodies.angular_velocity[i].length() * bodies.shape[i].bounding_radius() * h
            } else {
                0.0
            }
        };
        let fast = spin(ia).max(spin(ib)) > REFRESH_ERROR;
        let [then_a, then_b] = working.detected[m_index];
        // A pair not touching yet whose gap has halved since it was found:
        // the features that will meet are chosen from where the pair is,
        // and across a 25 cm gap at 60 m/s the face and corner chosen were
        // not the ones that met. Found again as the gap closes — at most
        // once a substep — and they are.
        let gap = working.detected_gap[m_index];
        let moved =
            (bodies.relative_translation(ia, ib) - working.detected_offset[m_index]).length();
        let closing = gap > 0.0 && moved > (0.5 * gap).max(REFRESH_ERROR);
        if !fast && !closing && off(ia, then_a).max(off(ib, then_b)) <= REFRESH_ERROR {
            continue;
        }
        let horizon = if fast { h.min(remaining) } else { remaining };
        let reach = if fast {
            (2.0 * h).min(remaining)
        } else {
            remaining
        };
        working.detected[m_index] = [bodies.orientation[ia], bodies.orientation[ib]];
        working.detected_offset[m_index] = bodies.relative_translation(ia, ib);
        working.refreshes += 1;

        let tolerance = ctx.contact_tolerance
            + sweep_of(bodies, ia, reach, ctx.max_sweep_distance)
            + sweep_of(bodies, ib, reach, ctx.max_sweep_distance);
        let mut fresh = collide_pair(
            bodies,
            ia,
            ib,
            tolerance,
            ctx.contact_tolerance,
            horizon,
            &mut working.refresh_buffer,
        )
        .unwrap_or_else(|| {
            let mut empty =
                ContactManifold::new(old.body_a, old.body_b, old.local_normal_a, old.world_normal);
            empty.friction = old.friction;
            empty.restitution = old.restitution;
            empty
        });
        let matches = inherit_anchors(
            &old,
            &mut fresh,
            bodies,
            anchor_reach(bodies, ia, ib),
            false,
        );

        let base = working.offsets[m_index] as usize;
        let slots = &mut working.contact_scratch[base..base + MAX_CONTACT_POINTS];
        let before: [ContactScratch; MAX_CONTACT_POINTS] = std::array::from_fn(|k| slots[k]);
        let mut kept = [false; MAX_CONTACT_POINTS];
        for (k, slot) in slots.iter_mut().enumerate() {
            *slot = match (fresh.points().get(k), matches[k]) {
                (Some(_), Some(j)) => {
                    kept[j] = true;
                    before[j]
                }
                _ => ContactScratch::default(),
            };
            // Friction holds a contact to where it stuck. The point found
            // now is a new pair of anchors, so it is stuck where it is.
            if let Some(point) = fresh.points().get(k) {
                slot.stick_local_a = point.local_a;
            }
        }
        for (j, point) in old.points().iter().enumerate() {
            let s = &before[j];
            if kept[j] || (s.impulse == Vec3::ZERO && s.dissipated == 0.0 && !s.impacting) {
                continue;
            }
            working.retired.push(ContactReport {
                body_a: old.body_a,
                body_b: old.body_b,
                local_a: point.local_a,
                local_b: point.local_b,
                local_normal_a: point.local_normal_a,
                normal_lambda: s.normal_lambda,
                touching: false,
                dissipated: s.dissipated,
                impulse: s.impulse,
                impacting: s.impacting || s.speculative_impact > contact::IMPACT_SPEED,
                impact_impulse: s.impact_impulse,
                impact_speed: s.impact_speed.max(s.speculative_impact),
                restitution: old.restitution,
            });
        }
        working.detected_gap[m_index] = gap_of(&fresh);
        working.manifolds[m_index] = fresh;
    }
}

/// The narrowest a manifold's points were apart when found: negative where
/// they overlap, infinite for an empty manifold.
fn gap_of(m: &ContactManifold) -> f32 {
    m.points()
        .iter()
        .fold(f32::INFINITY, |g, p| g.min(p.separation))
}

/// Store what each contact converged to, for the next step to resume from.
///
/// Sorted by body pair, which makes the lookup in `prepare` a binary search
/// and — more importantly — makes it independent of the order the broad phase
/// happened to emit pairs in. An order-dependent cache would make the warm
/// start, and therefore the answer, depend on something the scene has no
/// control over, and would take bit-identical replay (PLAN.md §34) with it.
fn save_contact_cache(working: &mut XpbdWorkingSet) {
    working.contact_cache.clear();
    working.contact_cache.reserve(working.manifolds.len());
    for (m_index, manifold) in working.manifolds.iter().enumerate() {
        let base = working.offsets[m_index] as usize;
        let mut entry = CachedManifold {
            body_a: manifold.body_a,
            body_b: manifold.body_b,
            ..Default::default()
        };
        for (k, point) in manifold.points().iter().enumerate() {
            let lambda = working.contact_scratch[base + k].normal_lambda;
            // A point that ended the step pushing on nothing has nothing worth
            // remembering, and storing a zero would only cost the next step a
            // failed match.
            if lambda <= 0.0 {
                continue;
            }
            let i = entry.count as usize;
            entry.local_b[i] = point.local_b;
            entry.lambda[i] = lambda;
            entry.stick[i] = working.contact_scratch[base + k].stick_local_a;
            entry.count += 1;
        }
        if entry.count > 0 {
            working.contact_cache.push(entry);
        }
    }
    working
        .contact_cache
        .sort_unstable_by_key(|c| (c.body_a, c.body_b));
}
