//! The energy ledger: which stage of the step changed the energy, and by how
//! much.
//!
//! PLAN.md §23 calls this "probably the highest value-per-effort feature in the
//! entire plan", and explains why in one line: physics regressions are silent.
//! Nothing crashes; stacks just get less stable. A total energy figure tells
//! you *that* something is wrong several seconds after it started. A per-stage
//! breakdown tells you *what*:
//!
//! ```text
//! integrate  -0.000 J | contacts  +1.412 J <-- | recover  -0.031 J
//! ```
//!
//! The arrow is the bug. Without the ledger that same defect is "the stack
//! looks a bit lively", found by eye, a week later.
//!
//! ## Reading it correctly
//!
//! XPBD is a *position*-based method, and that makes the attribution less
//! obvious than it looks. A stage that pushes a body upwards changes its
//! potential energy immediately, and the kinetic energy that same push implies
//! does not appear until `recover_velocities` turns the accumulated position
//! correction into velocity. So a position-solve stage and `Recover` are two
//! halves of one physical event and have to be read as a pair.
//!
//! What *is* exact, and is the number worth trusting on its own, is the sum:
//! every joule the step gained or lost is attributed to exactly one stage, and
//! [`EnergyLedger::total`] equals the step's actual energy change. Individual
//! rows locate a problem; the total measures it.
//!
//! ## Why the sums are `f64`
//!
//! The engine works in `f32` and this module does not, deliberately. A pit of
//! two hundred bodies holds about 7 x 10^6 J, and the per-stage deltas worth
//! seeing are hundredths of a joule. Single precision has about seven decimal
//! digits, so at that magnitude the spacing between representable values is
//! roughly 0.5 J — twenty times larger than the signal. Measured in `f32` the
//! ledger reports its own rounding error and nothing else.
//!
//! This is the one place in Fusion where a diagnostic needs more precision
//! than the thing it is diagnosing, and it is not a contradiction: the
//! simulation only needs energy to be *approximately* conserved, while the
//! ledger needs to measure the small amount by which it is not.

use fusion_core::{BodySet, FluidSet, ParticleSet};
use fusion_math::Vec3;

/// One stage of the step, in the order the solver runs them.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub enum Stage {
    /// Gravity, damping and the gyroscopic term.
    Integrate,
    /// Drag, lift and the Magnus effect.
    Aerodynamics,
    /// `x += v h`, before any constraint has an opinion.
    Predict,
    /// Non-penetration and static friction.
    Contacts,
    /// Ball, hinge, slider, distance, fixed.
    Joints,
    /// Cloth and soft bodies: distance, bending, and particle-vs-body.
    Deformables,
    /// The fluid density constraint and its boundaries.
    Fluid,
    /// Position corrections becoming velocity.
    Recover,
    /// Restitution and dynamic friction.
    VelocityPass,
    /// Vorticity confinement and XSPH viscosity.
    FluidPost,
    /// The fluid speed clamp.
    ///
    /// Its own stage because it is not physics and does not pretend to be —
    /// it is the safety net that stops one ejected particle taking the
    /// neighbour grid's bounds with it. Folded into `FluidPost` it hid the one
    /// stage with a licence to *add* energy behind a mechanism that only ever
    /// removes it, and a violent splash showed the pair netting negative while
    /// vorticity confinement was pumping energy in.
    ///
    /// A non-zero row here is a signal in its own right: the clamp firing at
    /// all means the solve produced a velocity nobody intended.
    SpeedClamp,
}

impl Stage {
    /// Every stage, in the order a step runs them. `StepReport::energy_stages`
    /// is indexed by position in this array.
    pub const ALL: [Stage; 11] = [
        Stage::Integrate,
        Stage::Aerodynamics,
        Stage::Predict,
        Stage::Contacts,
        Stage::Joints,
        Stage::Deformables,
        Stage::Fluid,
        Stage::Recover,
        Stage::VelocityPass,
        Stage::FluidPost,
        Stage::SpeedClamp,
    ];

    /// Short lowercase name, for tables and reports.
    pub fn name(self) -> &'static str {
        match self {
            Stage::Integrate => "integrate",
            Stage::Aerodynamics => "aero",
            Stage::Predict => "predict",
            Stage::Contacts => "contacts",
            Stage::Joints => "joints",
            Stage::Deformables => "deformables",
            Stage::Fluid => "fluid",
            Stage::Recover => "recover",
            Stage::VelocityPass => "velocity",
            Stage::FluidPost => "fluid post",
            Stage::SpeedClamp => "speed clamp",
        }
    }

    #[inline]
    fn index(self) -> usize {
        self as usize
    }

    /// Whether this stage is capable of adding energy in a correct simulation.
    ///
    /// Only two stages are not, and the list is short on purpose — a
    /// diagnostic that cries wolf is worse than no diagnostic, and the first
    /// version of this function was wrong in exactly that way.
    ///
    /// It listed every constraint solve as unable to gain, on the reasoning
    /// that a constraint removes a degree of freedom. Pointed at a ten-box
    /// tower it duly accused `contacts` of gaining 18.8 kJ over four seconds
    /// — and `contacts` was innocent. A non-penetration constraint holding a
    /// box up against gravity *raises its potential energy*; that is what
    /// holding something up means. The matching row was `predict` at
    /// -19.0 kJ, gravity pulling the same boxes down again, and the pair
    /// nearly cancels. Reading either alone says nothing.
    ///
    /// What remains unambiguous is the two stages that touch only velocity and
    /// only ever subtract from it: the Coulomb velocity pass, and the fluid
    /// speed clamp. Either of those going positive is a real defect with no
    /// benign reading. For everything else, the number to watch is the step
    /// total — see [`EnergyLedger::net_gain`].
    fn may_add_energy(self) -> bool {
        !matches!(self, Stage::VelocityPass | Stage::SpeedClamp)
    }
}

/// Per-stage energy attribution for one step.
///
/// Disabled by default. Enabled, it costs a full pass over every body and
/// particle after each of ten stages of every substep — roughly a 40% step
/// overhead on a rigid scene — which is a fine price for a debugging session
/// and not one to pay in a shipped frame loop.
#[derive(Clone, Debug, Default)]
pub struct EnergyLedger {
    enabled: bool,
    /// Energy at the last sample. `f64` for the reason in the module docs.
    previous: f64,
    /// Energy at the top of the step.
    step_start: f64,
    totals: [f64; Stage::ALL.len()],
    samples: u64,
}

impl EnergyLedger {
    /// A ledger that is off. Enabling it costs a full energy sum per stage
    /// per substep; see the module docs.
    pub fn new() -> Self {
        Self::default()
    }

    /// Turn attribution on or off. Off, every sample is a branch and
    /// nothing else.
    pub fn set_enabled(&mut self, enabled: bool) {
        self.enabled = enabled;
    }

    /// Whether this ledger is recording.
    #[inline]
    pub fn is_enabled(&self) -> bool {
        self.enabled
    }

    /// Start a step. Clears the per-stage totals and takes the baseline.
    pub fn begin_step(
        &mut self,
        bodies: &BodySet,
        particles: &ParticleSet,
        fluids: &[FluidSet],
        gravity: Vec3,
    ) {
        if !self.enabled {
            return;
        }
        self.totals = [0.0; Stage::ALL.len()];
        self.previous = mechanical_energy(bodies, particles, fluids, gravity);
        self.step_start = self.previous;
        self.samples = 0;
    }

    /// Attribute everything since the last sample to `stage`.
    ///
    /// A no-op when disabled, which is the whole of the cost story: one
    /// predictable branch per stage per substep.
    #[inline]
    pub fn sample(
        &mut self,
        stage: Stage,
        bodies: &BodySet,
        particles: &ParticleSet,
        fluids: &[FluidSet],
        gravity: Vec3,
    ) {
        if !self.enabled {
            return;
        }
        let now = mechanical_energy(bodies, particles, fluids, gravity);
        self.totals[stage.index()] += now - self.previous;
        self.previous = now;
        self.samples += 1;
    }

    /// Joules attributed to one stage over the whole step.
    pub fn stage(&self, stage: Stage) -> f32 {
        self.totals[stage.index()] as f32
    }

    /// Total energy change across the step.
    ///
    /// Exactly the sum of the stages, and exactly the step's real energy
    /// change — the two agree by construction, which is what makes the
    /// breakdown trustworthy even where an individual row is hard to read.
    pub fn total(&self) -> f32 {
        (self.previous - self.step_start) as f32
    }

    /// Energy at the end of the last sampled stage.
    pub fn energy(&self) -> f32 {
        self.previous as f32
    }

    /// The stage that added the most energy, and how much — or `None` if no
    /// stage added any.
    pub fn worst_gain(&self) -> Option<(Stage, f32)> {
        Stage::ALL
            .iter()
            .map(|&s| (s, self.stage(s)))
            .filter(|&(_, joules)| joules > 0.0)
            .max_by(|a, b| a.1.total_cmp(&b.1))
    }

    /// Stages that added energy when they had no business doing so.
    ///
    /// A stage listed here is a defect with no benign reading. Only the
    /// velocity pass and the speed clamp can appear — see
    /// `Stage::may_add_energy` for why the constraint solves cannot, and for
    /// the false accusation that taught the difference.
    pub fn suspects(&self, threshold: f32) -> Vec<(Stage, f32)> {
        Stage::ALL
            .iter()
            .filter(|s| !s.may_add_energy())
            .map(|&s| (s, self.stage(s)))
            .filter(|&(_, joules)| joules > threshold)
            .collect()
    }

    /// Whether the step as a whole gained energy, and how much.
    ///
    /// The number to watch when no individual row is damning. A position-based
    /// solver moves large amounts of energy between stages every substep — a
    /// resting ten-box tower churns 108 kJ through `recover` and takes 126 kJ
    /// back out at the velocity pass — and the only quantity with an
    /// unambiguous sign is what is left at the end.
    pub fn net_gain(&self) -> Option<f32> {
        let total = self.total();
        (total > 0.0).then_some(total)
    }

    /// One line per stage that did anything, widest first.
    pub fn summary(&self) -> String {
        if !self.enabled {
            return "energy ledger: disabled".into();
        }
        let mut rows: Vec<(Stage, f32)> = Stage::ALL
            .iter()
            .map(|&s| (s, self.stage(s)))
            .filter(|&(_, j)| j != 0.0)
            .collect();
        rows.sort_by(|a, b| b.1.abs().total_cmp(&a.1.abs()));
        let mut out = String::new();
        for (stage, joules) in rows {
            let flag = if !stage.may_add_energy() && joules > 0.0 {
                "  <-- gained"
            } else {
                ""
            };
            out.push_str(&format!("{:>12} {:>+12.6} J{flag}\n", stage.name(), joules));
        }
        out.push_str(&format!("{:>12} {:>+12.6} J\n", "total", self.total()));
        out
    }
}

/// Total mechanical energy of everything in a scene, in double precision.
///
/// Kahan summation is deliberately *not* used. It would buy another few digits
/// and cost four times the arithmetic, and the thing being measured is a
/// difference between two sums taken microseconds apart over an almost
/// identical set of numbers — the terms are the same, so their rounding is
/// nearly the same, and it cancels in the subtraction. Plain `f64` is already
/// past the point where the ledger's noise is smaller than anything worth
/// looking at.
pub fn mechanical_energy(
    bodies: &BodySet,
    particles: &ParticleSet,
    fluids: &[FluidSet],
    gravity: Vec3,
) -> f64 {
    let mut total = body_energy(bodies, gravity);
    total += particle_energy(particles, gravity);
    for fluid in fluids {
        total += particle_energy(&fluid.particles, gravity);
    }
    total
}

/// Kinetic, rotational and potential energy of the bodies that move, J.
pub(crate) fn body_energy(bodies: &BodySet, gravity: Vec3) -> f64 {
    let mut total = 0.0f64;
    for i in 0..bodies.len() {
        if !bodies.is_movable(i) {
            continue;
        }
        let inv_mass = bodies.inv_mass[i] as f64;
        if inv_mass <= 0.0 {
            continue;
        }
        let mass = 1.0 / inv_mass;
        let v = bodies.linear_velocity[i];
        total += 0.5 * mass * v.length_squared() as f64;

        // Rotational kinetic energy, which the f32 helpers on `BodySet` omit.
        // Omitting it is defensible for a drift *ratio* and not for a ledger:
        // a body spun up by a bad contact stores the evidence entirely in its
        // angular velocity, and a ledger blind to that would report the
        // energy vanishing into nowhere.
        let w = bodies.angular_velocity[i];
        let inertia = bodies.inertia_local[i];
        let local = bodies.orientation[i].rotate_inverse(w);
        total += 0.5
            * ((inertia.x * local.x * local.x
                + inertia.y * local.y * local.y
                + inertia.z * local.z * local.z) as f64);

        // Potential, from the double-precision position rather than the
        // narrowed one — the whole reason positions are `f64` is that the
        // narrowing loses metres of resolution far from the origin, and a
        // ledger built on the narrowed value would inherit that.
        let p = bodies.position[i];
        total -=
            (gravity.x as f64 * p.x + gravity.y as f64 * p.y + gravity.z as f64 * p.z) / inv_mass;
    }
    total
}

fn particle_energy(particles: &ParticleSet, gravity: Vec3) -> f64 {
    let mut total = 0.0f64;
    for i in 0..particles.len() {
        let inv_mass = particles.inv_mass[i] as f64;
        if inv_mass <= 0.0 {
            continue;
        }
        let mass = 1.0 / inv_mass;
        total += 0.5 * mass * particles.velocity[i].length_squared() as f64;
        let p = particles.position[i];
        total -= (gravity.x as f64 * p.x + gravity.y as f64 * p.y + gravity.z as f64 * p.z) * mass;
    }
    total
}
