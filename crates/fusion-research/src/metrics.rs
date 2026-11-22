//! Conservation diagnostics: momentum, energy and mass, tracked per run.
//!
//! §23 asks for these three "tracked and plotted per run, always available",
//! and the phrasing is the specification. Not a test you run when you suspect
//! something — a record every run carries, so that the question "when did this
//! start leaking momentum" has an answer without re-running anything.
//!
//! ## What each one is worth
//!
//! - **Momentum** is the sharp one, and it needs one correction to be worth
//!   anything. Nothing in the *solver* may change total momentum — XPBD's
//!   corrections are equal and opposite by construction — but gravity is an
//!   external force and changes it every step, by exactly `M g Δt`. Measuring
//!   drift against the starting value therefore reports 100% on every scene
//!   that has gravity and starts at rest, which is most of them, and a
//!   diagnostic that always screams is not a diagnostic. What is tracked here
//!   is the momentum the solver invented: the difference between the momentum
//!   the world has and the momentum gravity accounts for. That number has no
//!   honest reason to move at all.
//!
//!   The same correction applies to angular momentum, where the torque is
//!   `M R × g` and depends on where the centre of mass is, so it has to be
//!   integrated over the run rather than computed in closed form.
//!
//!   And it only means anything in a **closed** scene. A body resting on a
//!   static floor is held up by that floor, and an immovable body supplies
//!   whatever impulse it likes — so in any scene with static geometry there is
//!   no momentum invariant to check, and a number claiming otherwise is
//!   noise wearing a percentage sign. Correcting for gravity and reporting the
//!   remainder on a tank of water gave 1456%, which is the tool being wrong
//!   rather than the engine. Closedness is detected and the check is reported
//!   as inapplicable where it does not apply, which is §2's third principle —
//!   make the domain of applicability legible — applied to a diagnostic.
//! - **Mass** is trivially conserved in Basic — nothing is created or
//!   destroyed — and is tracked anyway, because the day Advanced starts moving
//!   material between representations it stops being trivial, and a diagnostic
//!   introduced at that point has no history to compare against.
//! - **Energy** is the blunt one, and the honest one. XPBD does not conserve
//!   it and says so (§9). What matters is the *shape*: a slow bleed downward
//!   is damping doing its job, and a spike upward is a solver injecting
//!   energy. `fusion_solver::EnergyLedger` says which stage; this says when.

use fusion_math::Vec3;
use fusion_sim::Simulation;
use serde::{Deserialize, Serialize};

/// The conserved quantities at one instant.
#[derive(Clone, Copy, Debug, Default, PartialEq, Serialize, Deserialize)]
pub struct Conserved {
    /// Simulated time, seconds.
    pub time: f64,
    /// Solver steps taken to get here.
    pub step: u64,
    /// Total linear momentum, kg m/s.
    pub linear_momentum: [f32; 3],
    /// Total angular momentum about the world origin, kg m²/s.
    pub angular_momentum: [f32; 3],
    /// Total mechanical energy, J — the continuum's strain energy included.
    ///
    /// In double, for the reason `fusion_solver::ledger` gives at length: a
    /// pit of two hundred bodies holds about 7 x 10^6 J and the per-step gains
    /// worth catching are hundredths of a joule, which is four orders below
    /// the spacing between `f32` values up there. Measured in single precision
    /// this channel reported its own rounding.
    pub energy: f64,
    /// Total mass of everything that can move, kg.
    pub mass: f32,
    /// Centre of mass, m. Recorded because the gravitational torque about the
    /// origin is `M R × g`, and correcting angular momentum for gravity means
    /// integrating that over the run.
    pub centre_of_mass: [f32; 3],
    /// The gravity this sample was taken under, m/s².
    pub gravity: [f32; 3],
    /// True where every body can move. In a scene with static geometry the
    /// momentum checks do not apply — see the module docs.
    pub closed: bool,
}

impl Conserved {
    /// Read the three quantities out of a simulation as it stands.
    pub fn of(sim: &Simulation) -> Self {
        Self {
            time: sim.time(),
            step: sim.step_index(),
            linear_momentum: sim.world.linear_momentum().to_array(),
            angular_momentum: sim.world.angular_momentum().to_array(),
            energy: sim.mechanical_energy(),
            mass: sim.world.total_mass(),
            centre_of_mass: centre_of_mass(&sim.world).to_array(),
            gravity: sim.world.gravity.to_array(),
            closed: (0..sim.world.len()).all(|i| sim.world.bodies.is_dynamic(i))
                && (0..sim.world.particles.len()).all(|i| sim.world.particles.is_movable(i)),
        }
    }

    fn momentum(&self) -> Vec3 {
        Vec3::from_array(self.linear_momentum)
    }

    fn angular(&self) -> Vec3 {
        Vec3::from_array(self.angular_momentum)
    }
}

/// Centre of mass of everything that can move.
fn centre_of_mass(world: &fusion_core::World) -> Vec3 {
    let mut weighted = Vec3::ZERO;
    let mut total = 0.0f32;
    for i in 0..world.len() {
        if world.bodies.is_dynamic(i) {
            let m = world.bodies.mass(i);
            weighted += fusion_math::to_f32(world.bodies.position[i]) * m;
            total += m;
        }
    }
    let mut add_particles = |set: &fusion_core::ParticleSet| {
        for i in 0..set.len() {
            if set.is_movable(i) {
                let m = 1.0 / set.inv_mass[i];
                weighted += set.translation(i) * m;
                total += m;
            }
        }
    };
    add_particles(&world.particles);
    for fluid in &world.fluids {
        add_particles(&fluid.particles);
    }
    for solid in &world.solids {
        add_particles(&solid.particles);
    }
    if total > 0.0 {
        weighted / total
    } else {
        Vec3::ZERO
    }
}

/// Conservation over a whole run: the samples, and what they add up to.
#[derive(Clone, Debug, Default, Serialize, Deserialize)]
pub struct ConservationTrace {
    /// One sample per recorded step, in order.
    pub samples: Vec<Conserved>,
}

impl ConservationTrace {
    /// An empty trace.
    pub fn new() -> Self {
        Self::default()
    }

    /// Record where the simulation stands now.
    pub fn observe(&mut self, sim: &Simulation) {
        self.samples.push(Conserved::of(sim));
    }

    /// The summary a results store keeps and a person reads.
    ///
    /// `None` for a trace with nothing in it — a run that recorded no samples
    /// has no conservation to report, and inventing zeros for it would put a
    /// perfect row in a results table for a run that never happened.
    pub fn summary(&self) -> Option<ConservationSummary> {
        let first = self.samples.first()?;
        let last = self.samples.last()?;

        // Drift measured against the *scale of the run*, not against the
        // starting value. A scene whose momentum starts at zero — which is
        // most of them, since things usually start at rest — has no starting
        // value to take a percentage of, and dividing by it produces either
        // infinity or a silent zero depending on the rounding.
        let scale = self
            .samples
            .iter()
            .map(|s| s.momentum().length())
            .fold(0.0f32, f32::max)
            .max(1e-6);
        let angular_scale = self
            .samples
            .iter()
            .map(|s| s.angular().length())
            .fold(0.0f32, f32::max)
            .max(1e-6);

        // What gravity accounts for, accumulated across the samples.
        //
        // Linear: `M g Δt`, exact, since the mass does not change. Angular:
        // `M R × g Δt` by trapezoid, because the torque depends on where the
        // centre of mass is and that moves. Both are subtracted before any
        // drift is reported, so what is left is what the solver did.
        let mut expected_linear = first.momentum();
        let mut expected_angular = first.angular();
        let mut accounted: Vec<(Vec3, Vec3)> = Vec::with_capacity(self.samples.len());
        accounted.push((expected_linear, expected_angular));
        for pair in self.samples.windows(2) {
            let dt = (pair[1].time - pair[0].time) as f32;
            let g = Vec3::from_array(pair[1].gravity);
            let mass = pair[1].mass;
            expected_linear += g * mass * dt;
            let torque_start = Vec3::from_array(pair[0].centre_of_mass).cross(g * pair[0].mass);
            let torque_end = Vec3::from_array(pair[1].centre_of_mass).cross(g * mass);
            expected_angular += (torque_start + torque_end) * 0.5 * dt;
            accounted.push((expected_linear, expected_angular));
        }

        let mut worst_momentum = 0.0f32;
        let mut worst_angular = 0.0f32;
        let mut worst_mass = 0.0f32;
        let mut worst_energy_gain = 0.0f64;
        let mut worst_energy_step = 0u64;
        for pair in self.samples.windows(2) {
            let gain = pair[1].energy - pair[0].energy;
            if gain > worst_energy_gain {
                worst_energy_gain = gain;
                worst_energy_step = pair[1].step;
            }
        }
        for (s, (linear, angular)) in self.samples.iter().zip(&accounted) {
            worst_momentum = worst_momentum.max((s.momentum() - *linear).length());
            worst_angular = worst_angular.max((s.angular() - *angular).length());
            worst_mass = worst_mass.max((s.mass - first.mass).abs());
        }

        Some(ConservationSummary {
            closed: self.samples.iter().all(|s| s.closed),
            steps: last.step.saturating_sub(first.step),
            simulated_time: last.time - first.time,
            momentum_drift: worst_momentum,
            momentum_drift_relative: worst_momentum / scale,
            angular_drift: worst_angular,
            angular_drift_relative: worst_angular / angular_scale,
            mass_drift: worst_mass,
            energy_start: first.energy,
            energy_end: last.energy,
            worst_energy_gain,
            worst_energy_gain_step: worst_energy_step,
        })
    }
}

/// What a run's conservation came to, in one row.
#[derive(Clone, Copy, Debug, Default, PartialEq, Serialize, Deserialize)]
pub struct ConservationSummary {
    /// True where the scene had no static geometry and no pinned particles,
    /// which is the only case in which the momentum figures below are a check
    /// on anything. Elsewhere they record what happened, and an immovable body
    /// supplied the difference.
    pub closed: bool,
    /// Steps covered by the trace.
    pub steps: u64,
    /// Simulated seconds covered.
    pub simulated_time: f64,
    /// Largest momentum the solver invented: how far total linear momentum
    /// got from the value gravity accounts for. Should be zero.
    pub momentum_drift: f32,
    /// The same as a fraction of the largest momentum seen during the run.
    pub momentum_drift_relative: f32,
    /// The same for angular momentum, against the gravitational torque
    /// integrated over the run.
    pub angular_drift: f32,
    /// The same, relative to the largest seen.
    pub angular_drift_relative: f32,
    /// Largest departure of total mass from where it started. Should be zero.
    pub mass_drift: f32,
    /// Total mechanical energy at the first sample.
    pub energy_start: f64,
    /// And at the last.
    pub energy_end: f64,
    /// Largest single-step energy *increase*. The number that matters: a slow
    /// bleed downward is damping, a spike upward is a solver bug.
    pub worst_energy_gain: f64,
    /// Which step that happened on, so the ledger can be pointed at it.
    pub worst_energy_gain_step: u64,
}

impl ConservationSummary {
    /// Total energy change over the run, negative for a run that lost energy.
    pub fn energy_delta(&self) -> f64 {
        self.energy_end - self.energy_start
    }
}
