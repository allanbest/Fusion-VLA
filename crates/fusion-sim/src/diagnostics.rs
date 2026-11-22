//! Step diagnostics.
//!
//! PLAN.md §23 calls the energy ledger the highest value-per-effort feature in
//! the whole plan, and points out that it serves Basic development as much as
//! Advanced research: physics regressions are silent, nothing crashes, stacks
//! just get less stable.
//!
//! What ships in v0.1 is the measurement, not yet the per-stage attribution.
//! Total mechanical energy before and after the step already catches the
//! failure that matters most — a contact solver quietly injecting energy —
//! and the per-stage breakdown (`integration: -0.02 J | contacts: +1.4 J`)
//! becomes a matter of sampling at more points inside the same step.

use std::time::Duration;

/// Why a step was rejected, whichever engine was running it.
///
/// One list for both engines, so a caller reports a failure the same way
/// without knowing which engine produced it. The Unified continuum's own
/// reasons come from [`fusion_mpm::unified::StepFailure`]; the Fusion Core
/// reasons are defined here, because the MPM crate knows nothing about
/// Fusion Core.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum StepFailure {
    /// Invalid timestep, grid spacing or solver configuration, or a scene
    /// the selected engine cannot run.
    InvalidConfiguration,
    /// Inconsistent arrays, non-finite state or inadmissible material data.
    InvalidState,
    /// Completing the interval would exceed the configured work budget.
    SubstepBudget,
    /// The sparse grid would exceed the configured block budget.
    GridBudget,
    /// Arithmetic produced a non-finite state.
    NonFiniteUpdate,
    /// Strict mode encountered a numerical repair or clamp.
    RepairRequired,
    /// Fusion Core: the mechanics solve exhausted its convergence and retry
    /// budget.
    Convergence,
    /// Fusion Core: the scene or its output views were edited without a
    /// reset, so the prepared runtime no longer matches them.
    SceneChanged,
}

impl From<fusion_mpm::unified::StepFailure> for StepFailure {
    fn from(failure: fusion_mpm::unified::StepFailure) -> Self {
        use fusion_mpm::unified::StepFailure as Mpm;
        match failure {
            Mpm::InvalidConfiguration => Self::InvalidConfiguration,
            Mpm::InvalidState => Self::InvalidState,
            Mpm::SubstepBudget => Self::SubstepBudget,
            Mpm::GridBudget => Self::GridBudget,
            Mpm::NonFiniteUpdate => Self::NonFiniteUpdate,
            Mpm::RepairRequired => Self::RepairRequired,
        }
    }
}

impl std::fmt::Display for StepFailure {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(match self {
            Self::InvalidConfiguration => "invalid solver settings",
            Self::InvalidState => "invalid material state",
            Self::SubstepBudget => "more internal steps are needed",
            Self::GridBudget => "grid memory limit reached",
            Self::NonFiniteUpdate => "numerical overflow",
            Self::RepairRequired => "accuracy mode rejected a numerical repair",
            Self::Convergence => "mechanics solve did not converge; reduce the timestep",
            Self::SceneChanged => "scene changed; reset to rebuild Fusion Core",
        })
    }
}

impl std::error::Error for StepFailure {}

/// What one step did, and what it cost.
///
/// Read-only: `#[non_exhaustive]` because this type grows. v0.7 added
/// `particle_contacts` to it, and under the API freeze (`docs/API.md`) that
/// has to be a compatible change rather than a breaking one.
#[derive(Clone, Copy, Debug, Default)]
#[non_exhaustive]
pub struct StepReport {
    /// Which solver step this was, counting from t = 0.
    pub step_index: u64,
    /// Simulated time at the end of this step, seconds.
    pub time: f64,
    /// Length of the step, seconds. The internal timestep, not the frame.
    pub dt: f32,
    /// Why the step was rejected, if it was. A rejected step preserves the
    /// world's state and clock, and reports `dt = 0`; no partial interval is
    /// accepted.
    pub failure: Option<StepFailure>,

    /// Pairs the broad phase thought worth a closer look.
    pub broad_phase_pairs: usize,
    /// Pairs that turned out to be touching, one manifold each. The ratio of
    /// this to `broad_phase_pairs` is how well the broad phase is doing.
    pub manifolds: usize,
    /// Rigid-rigid contact points prepared for this step. Particles are not
    /// in here — see [`Self::particle_contacts`], which is a different
    /// question with a different answer.
    pub contact_points: usize,
    /// Manifolds detected again part-way through the step because their
    /// bodies had turned too far for the contacts found at the top of it.
    /// See `XpbdSolver::refresh_contacts`.
    pub contact_refreshes: usize,
    /// Particle-body contacts active at the end of the step: cloth, soft
    /// bodies and fluid alike, resting on or pressing into rigid bodies.
    ///
    /// Reporting rigid contacts alone made every cloth and fluid scene look
    /// contact-free — a dam break against a wall of boxes reported zero, which
    /// is the one number a person would check to find out whether the fluid
    /// had noticed the wall. It was a listed defect through v0.6.
    ///
    /// A fluid particle bouncing off a *domain wall* is not counted, because
    /// there is no body there to push back; that mechanism is a reflection,
    /// not a contact.
    pub particle_contacts: usize,
    /// Sum of normal Lagrange multipliers applied this step. Proportional to
    /// total contact impulse, and the quantity an impact event would threshold.
    pub total_normal_impulse: f32,

    /// Kinetic and potential energy of *everything* — bodies, cloth, soft
    /// bodies and fluid alike.
    pub kinetic_energy: f32,
    /// Potential energy of the same, against this world's gravity.
    pub potential_energy: f32,
    /// Elastic strain energy stored in the continuum, J (v0.9).
    ///
    /// The third term of a solid's mechanical budget, and the one nothing in
    /// the engine could see before: a block falling on the floor converts its
    /// gravitational energy into *this* and hands it back on the way up.
    /// Without the term, `energy_delta` reported a continuum scene creating
    /// and destroying joules at the bottom of every bounce.
    pub strain_energy: f32,
    /// Thermal energy the continuum holds, J (v0.9).
    ///
    /// Not part of [`Self::total_energy`], deliberately. Heat is conserved by
    /// its own solve and is a one-way *sink* for the mechanical budget —
    /// plastic work moves joules into it and none come back — so the two
    /// totals are worth more apart than added together.
    pub heat: f32,
    /// Change in total mechanical energy across the step. Gravity moving a
    /// body trades one for the other and leaves this near zero; a contact
    /// solve that adds energy shows up here as a positive spike.
    pub energy_delta: f32,

    /// Wall-clock spent finding contacts.
    pub collision_time: Duration,
    /// Wall-clock spent solving them, and everything else the step solved.
    pub solve_time: Duration,

    /// Bodies currently asleep, and how many islands the awake ones form.
    pub sleeping_bodies: usize,
    /// Connected groups the awake bodies form.
    pub islands: usize,

    /// Fluid particles in the scene.
    pub fluid_particles: usize,
    /// Largest neighbourhood any fluid particle had this step. A number that
    /// rises well before a fluid solve gets into trouble, and the first place
    /// to look when one does.
    pub peak_fluid_neighbours: usize,
    /// Mean density excess over interior fluid particles, as a fraction of
    /// rest. This is the residual the density constraint did *not* remove, and
    /// it is the honest measure of how incompressible the fluid actually was —
    /// the accuracy claim of a position-based fluid lives or dies here.
    pub fluid_compression: f32,

    /// Per-stage energy attribution, in [`fusion_solver::Stage::ALL`] order.
    /// `None` unless the ledger was enabled for this step.
    ///
    /// This is the answer to "the stack is getting lively and I do not know
    /// why". `energy_delta` says the step gained a joule; this says which of
    /// the ten stages gained it.
    pub energy_stages: Option<[f32; 11]>,
}

impl StepReport {
    /// Kinetic plus potential plus strain — the quantity `energy_delta`
    /// differences.
    pub fn total_energy(&self) -> f32 {
        self.kinetic_energy + self.potential_energy + self.strain_energy
    }

    /// The stage that gained the most energy this step, if the ledger ran.
    pub fn worst_stage(&self) -> Option<(fusion_solver::Stage, f32)> {
        let stages = self.energy_stages?;
        fusion_solver::Stage::ALL
            .iter()
            .enumerate()
            .map(|(i, &s)| (s, stages[i]))
            .filter(|&(_, joules)| joules > 0.0)
            .max_by(|a, b| a.1.total_cmp(&b.1))
    }

    /// One-line summary, for the console and for CI logs.
    pub fn summary(&self) -> String {
        let fluid = if self.fluid_particles > 0 {
            format!(
                "  fluid={} compression={:>6.3}%",
                self.fluid_particles,
                self.fluid_compression * 100.0
            )
        } else {
            String::new()
        };
        format!(
            "step {:>6}  t={:7.3}s  pairs={:<5} contacts={:<5} asleep={:<5} \
             E={:>10.3} J  dE={:>+9.4} J  collide={:>6.2}ms solve={:>6.2}ms{fluid}",
            self.step_index,
            self.time,
            self.broad_phase_pairs,
            self.contact_points,
            self.sleeping_bodies,
            self.total_energy(),
            self.energy_delta,
            self.collision_time.as_secs_f64() * 1e3,
            self.solve_time.as_secs_f64() * 1e3,
        )
    }
}

/// Running totals over a run. The seed of the conservation diagnostics in
/// PLAN.md §23.
#[derive(Clone, Copy, Debug, Default)]
#[non_exhaustive]
pub struct RunDiagnostics {
    /// Steps observed.
    pub steps: u64,
    /// Simulated seconds covered by them.
    pub simulated_time: f64,
    /// Total mechanical energy at the first observed step.
    pub energy_at_start: f32,
    /// And at the most recent one. The pair is the drift over a run, which
    /// §23 calls the highest value-per-effort diagnostic in the plan.
    pub energy_now: f32,
    /// Largest single-step energy increase seen. The number that matters: a
    /// slow bleed downwards is damping, a spike upwards is a bug.
    pub worst_energy_gain: f32,
    /// Most rigid contact points any one step had to solve.
    pub peak_contact_points: usize,
    /// Wall-clock in collision detection over the run.
    pub total_collision_time: Duration,
    /// Wall-clock in the solver over the run.
    pub total_solve_time: Duration,
    /// Worst fluid compression seen over the run. A dam break peaks on impact
    /// and settles; a solve that is diverging never comes back down.
    pub worst_fluid_compression: f32,
}

impl RunDiagnostics {
    /// Fold one step's report into the running totals.
    pub fn observe(&mut self, report: &StepReport) {
        if self.steps == 0 {
            self.energy_at_start = report.total_energy();
        }
        self.steps += 1;
        self.simulated_time = report.time;
        self.energy_now = report.total_energy();
        self.worst_energy_gain = self.worst_energy_gain.max(report.energy_delta);
        self.peak_contact_points = self.peak_contact_points.max(report.contact_points);
        self.worst_fluid_compression = self.worst_fluid_compression.max(report.fluid_compression);
        self.total_collision_time += report.collision_time;
        self.total_solve_time += report.solve_time;
    }

    /// Total energy change since the run began, as a fraction of the starting
    /// energy. The quantity the drift regression test asserts on.
    pub fn relative_drift(&self) -> f32 {
        if self.energy_at_start.abs() < 1e-6 {
            return 0.0;
        }
        (self.energy_now - self.energy_at_start) / self.energy_at_start.abs()
    }
}
