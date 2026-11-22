//! Fusion's runtime: a [`Simulation`], and the engine that advances it.
//!
//! A simulation turns a `World` at time *t* into the same world at *t + dt*,
//! on one of two engines ([`EngineKind`]):
//!
//! - **Standard** — f32 XPBD rigid bodies, cloth and fluids, with an MPM
//!   continuum for deformable solids (Classic or Unified). The loop is in
//!   `standard.rs`, and it is the only engine that runs
//!   scripts, joints, heat, stress and failure.
//! - **Fusion Core** — f64 rigid bodies and finite-element solids in one
//!   implicit solve, from `crates/core/`. The adapter in [`core_engine`] owns
//!   the authoritative state and writes the `World` as an output view.
//!
//! Everything else here — checkpoints, timelines, state files, determinism
//! hashes, controls and diagnostics — is shared by both, so a front-end
//! never needs to know which engine is running.
//!
//! **Headless by construction.** PLAN.md §25 makes this the one platform
//! decision that constrains the architecture: nothing in this crate or below
//! may depend on the renderer or on a window. Cluster sweeps, CI runs and
//! remote boxes with no display are the entire research workflow of §23, and
//! the dependency is free to avoid now and expensive to untangle later.
//! `cargo tree -p fusion-sim` is the check — no `wgpu`, no `winit`.
//!
//! ```
//! use fusion_core::{BodyDesc, Material, World};
//! use fusion_math::{vec3, Vec3};
//! use fusion_sim::Simulation;
//!
//! let mut world = World::new();
//! world.add(BodyDesc::cuboid(vec3(20.0, 1.0, 20.0)).at(vec3(0.0, -0.5, 0.0)).fixed());
//! let ball = world.add(
//!     BodyDesc::sphere(0.5)
//!         .at(vec3(0.0, 6.0, 0.0))
//!         .with_material(Material::RUBBER),
//! );
//!
//! let mut sim = Simulation::new(world);
//! sim.run_frames(120, |_, _| {});
//!
//! // It landed, and it is on the floor rather than through it.
//! let y = sim.world.bodies.translation(ball.index()).y;
//! assert!(y > 0.4 && y < 6.0, "the ball ended at {y}");
//! assert!(sim.is_stable());
//! ```

#![deny(unsafe_code)]
// v0.7 freezes the public API (PLAN.md §29), and an undocumented public item
// is one nobody agreed to freeze. See `docs/API.md` for what the freeze
// promises and what it deliberately does not.
#![deny(missing_docs)]

pub mod checkpoint;
pub mod control;
pub mod core_engine;
pub mod determinism;
pub mod diagnostics;
pub mod failure;
pub mod heat;
pub mod islands;
pub mod script;
mod standard;
pub mod state;
pub mod stress;
pub mod surface;

use fusion_collide::{CollisionPipeline, ContactManifold};
use fusion_core::World;
use fusion_mpm::{MpmReport, MpmSolver};
use fusion_solver::{RigidSolver, SolverInfo, XpbdSolver, XpbdWorkingSet};

pub use checkpoint::{Checkpoint, SeekResult, Timeline};
pub use core_engine::{
    import_urdf_xml, CoreIntegrator, CoreRunner, CoreSettings, CoreStateField, CoreStepReport,
    ImportedUrdf, ImportedUrdfJoint, ImportedUrdfLink, MutualGravitySettings,
};
pub use determinism::{state_hash, Determinism, DeterminismTier};
pub use diagnostics::{RunDiagnostics, StepFailure, StepReport};
pub use failure::{FailureEvent, Outcome};
pub use fusion_mpm::unified::FluidModel;
pub use fusion_solver::{AccuracyTier, XpbdSolver as Xpbd};
pub use heat::HeatSettings;
pub use islands::{IslandSet, SleepPolicy};
pub use script::BodyScript;
pub use state::{StateError, STATE_VERSION};
pub use stress::StressSettings;
pub use surface::{SurfaceMesh, SurfaceSettings, Surfacer};

/// Which profile a simulation runs under (PLAN.md §4). Two profiles over one
/// core, not two codebases: the mode selects registered solvers, materials and
/// defaults.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Default)]
pub enum Mode {
    /// Real-time, FP32, XPBD spine. Behaves correctly.
    #[default]
    Basic,
    /// Offline-acceptable, MPM spine, conserves what it claims to. Arrives at
    /// v0.8; declaring it is rejected until then rather than silently ignored.
    Advanced,
}

/// Most failures [`Simulation::take_failures`] keeps between calls.
pub const MAX_KEPT_FAILURES: usize = 1024;

/// Which complete mechanics engine advances a scene.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum EngineKind {
    /// XPBD rigid bodies, cloth and fluids, plus an MPM continuum
    /// ([`ContinuumSolver`]). Runs every feature a scene can declare.
    #[default]
    Standard,
    /// Fusion Core: coupled rigid and finite-element mechanics in double
    /// precision. Runs rigid boxes and spheres and elastic solids; anything
    /// else is refused when it is selected. See [`core_engine`].
    Core,
}

/// Which solver advances the continua on the Standard engine — the
/// deformable solids and the fluids. Rigid bodies are on the rigid solver
/// either way.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash)]
pub enum ContinuumSolver {
    /// Each solid on its own MLS-MPM grid, and fluids by position-based
    /// fluids (PBF). What Fusion has always run.
    #[default]
    Classic,
    /// Every solid and fluid on one shared, sparse MLS-MPM grid, in parallel:
    /// fluids are material points, solids can carry fibres, and they all push
    /// on each other. See [`fusion_mpm::unified`]. Project files spell it
    /// `"fusion"`, its name while it was a preview.
    Unified,
}

impl ContinuumSolver {
    /// The name [`ContinuumSolver::Unified`] had while it was a preview.
    #[allow(non_upper_case_globals)]
    #[deprecated(note = "renamed to `ContinuumSolver::Unified`")]
    pub const Fusion: Self = Self::Unified;
}

/// A world plus everything needed to advance it in time.
///
/// The world holds the scene and the state; the simulation holds the clock,
/// the solver, and the scratch. That split is why `reset` is cheap and why a
/// project file will be enough to reproduce a run (§22).
pub struct Simulation {
    /// Which engine steps the world. Select Fusion Core with
    /// [`Simulation::use_core_engine`] at t = 0.
    engine: EngineKind,
    /// Fusion Core's runtime, while it is the selected engine.
    core: Option<core_engine::CoreEngine>,
    /// Persistent numerical settings for Fusion Core (`crates/core/`).
    pub core_settings: CoreSettings,
    /// The scene and its state. On the Standard engine it can be edited
    /// between steps. On Fusion Core it is an output view: edit the project
    /// and rebuild, or reset, to change its scene or initial state.
    pub world: World,
    /// The continuum solver, and its grid.
    ///
    /// Beside the XPBD solver rather than inside it, because they are two
    /// methods rather than two settings — §9 makes the solver an interface and
    /// this is the second implementation of it. A scene with no solids in it
    /// pays for an empty `Vec`.
    pub mpm: MpmSolver,
    /// What the last step's continuum solve did.
    mpm_report: MpmReport,
    /// Which continuum solver runs the solids and fluids.
    pub continuum: ContinuumSolver,
    /// The unified one, used when [`Simulation::continuum`] says so.
    pub unified: fusion_mpm::UnifiedSolver,
    /// What the last unified step did.
    unified_report: fusion_mpm::UnifiedReport,
    /// The solver that advances it, and every dial on it. Swappable, because
    /// §9 makes the solver an interface rather than a fixed choice.
    pub solver: XpbdSolver,
    /// Internal solver timestep. PLAN.md §17: frames are an output sampling
    /// choice, not the engine's clock. Basic runs at 1/240 s by default and
    /// records at whatever rate the caller asks for.
    pub solver_dt: f32,
    /// Output sampling rate, frames per second of simulated time.
    pub output_rate: f64,

    /// When bodies are allowed to stop being simulated.
    pub sleep: SleepPolicy,
    /// The air the scene moves through. Disabled by default — a scene that has
    /// not asked for aerodynamics should not pay for them.
    pub air: fusion_solver::AirModel,
    /// Ceiling on how far ahead speculative contacts look, metres. Without a
    /// cap, a very fast body can overlap many otherwise distant candidates.
    /// Applied per body; resting bodies keep the base margin.
    pub max_speculative_distance: f32,
    /// Attribute each step's energy change to the stage that caused it
    /// (PLAN.md §23). Off by default — see [`fusion_solver::EnergyLedger`] for
    /// what it costs and how to read it.
    pub energy_ledger: bool,
    /// Temperature in rigid bodies: friction and impact heating, conduction,
    /// cooling to the air. Off by default — see [`heat`].
    pub heat: HeatSettings,
    heat_working: heat::HeatWorking,
    /// Stress in rigid bodies, from the contacts, their own acceleration and
    /// thermal strain. Off by default — see [`stress`]. A body that can break
    /// or deform has its stress measured regardless.
    pub stress: StressSettings,
    stress_working: stress::StressWorking,
    /// Bodies that broke or deformed, in the order they did, since the last
    /// time a caller took them.
    failures: Vec<FailureEvent>,
    contact_reports: Vec<fusion_solver::ContactReport>,

    /// Behaviour scripts, one per scripted body.
    ///
    /// On the simulation rather than on the `World` because a script is a
    /// thing that *runs*, and `World` is state the solver reads and writes.
    /// Putting them here also keeps `fusion-core` free of any dependency on
    /// the language.
    scripts: Vec<script::BodyScript>,
    /// Step-indexed actions; retained when resetting or seeking a run.
    pub inputs: Vec<control::InputAction>,
    /// Resolved, stateless components evaluated before scripts each step.
    pub components: Vec<control::BodyComponents>,
    probe_samples: std::collections::VecDeque<control::ProbeReading>,
    /// Failures since a caller last took them.
    script_errors: Vec<(usize, fusion_script::ScriptError)>,

    pipeline: CollisionPipeline,
    manifolds: Vec<ContactManifold>,
    /// Deformable bodies as they were before this step's rigid solve. See
    /// `failure::Before`.
    before_solve: Vec<failure::Before>,
    working: XpbdWorkingSet,
    islands: IslandSet,

    time: f64,
    step_index: u64,
    frame_index: u64,
    /// Left-over simulated time from the last `advance` call, so that variable
    /// requests never desynchronize the fixed internal clock.
    accumulator: f64,

    last_energy: Option<f32>,
    /// Run-level counters: what has been stepped, and anything the engine
    /// wants to admit to afterwards.
    pub diagnostics: RunDiagnostics,
}

impl Simulation {
    /// A simulation of `world`, in Basic mode, with default solver settings.
    pub fn new(world: World) -> Self {
        Self {
            engine: EngineKind::Standard,
            core: None,
            core_settings: CoreSettings::default(),
            world,
            mpm: MpmSolver::new(),
            mpm_report: MpmReport::default(),
            continuum: ContinuumSolver::default(),
            unified: fusion_mpm::UnifiedSolver::new(),
            unified_report: fusion_mpm::UnifiedReport::default(),
            solver: XpbdSolver::default(),
            solver_dt: 1.0 / 240.0,
            output_rate: 60.0,
            sleep: SleepPolicy::default(),
            air: fusion_solver::AirModel::default(),
            max_speculative_distance: 2.0,
            energy_ledger: false,
            heat: HeatSettings::default(),
            heat_working: heat::HeatWorking::default(),
            stress: StressSettings::default(),
            stress_working: stress::StressWorking::default(),
            failures: Vec::new(),
            contact_reports: Vec::new(),
            scripts: Vec::new(),
            inputs: Vec::new(),
            components: Vec::new(),
            probe_samples: std::collections::VecDeque::new(),
            script_errors: Vec::new(),
            pipeline: CollisionPipeline::new(),
            manifolds: Vec::new(),
            before_solve: Vec::new(),
            working: XpbdWorkingSet::new(),
            islands: IslandSet::new(),
            time: 0.0,
            step_index: 0,
            frame_index: 0,
            accumulator: 0.0,
            last_energy: None,
            diagnostics: RunDiagnostics::default(),
        }
    }

    /// The same, in a named mode.
    ///
    /// [`Mode::Advanced`] is an error rather than a fallback. §9 attaches an
    /// accuracy tier to every solver, and quietly running Basic when Advanced
    /// was asked for would report the wrong one.
    pub fn with_mode(world: World, mode: Mode) -> Result<Self, &'static str> {
        match mode {
            Mode::Basic => Ok(Self::new(world)),
            Mode::Advanced => Err(
                "Fusion Advanced (MPM continuum) is not implemented before v0.8. \
                 Silently falling back to Basic would misreport the accuracy tier.",
            ),
        }
    }

    /// Builder form: use this solver.
    pub fn with_solver(mut self, solver: XpbdSolver) -> Self {
        self.solver = solver;
        self
    }

    /// Builder form: use this internal timestep, in seconds. See
    /// [`Simulation::solver_dt`].
    pub fn with_solver_dt(mut self, dt: f32) -> Self {
        self.solver_dt = dt;
        self
    }

    /// Simulated seconds elapsed. PLAN.md §19: the timeline is in sim-seconds,
    /// and wall-clock cost is reported separately rather than confused with it.
    pub fn time(&self) -> f64 {
        self.time
    }

    /// Solver steps taken since t = 0. Advances at `1 / solver_dt` per
    /// simulated second, whatever the output rate is.
    pub fn step_index(&self) -> u64 {
        self.step_index
    }

    /// Output frames emitted since t = 0, at [`Simulation::output_rate`].
    pub fn frame_index(&self) -> u64 {
        self.frame_index
    }

    /// What the solver declares about itself — order, accuracy tier, and what
    /// it does and does not conserve (§9).
    pub fn solver_info(&self) -> SolverInfo {
        match self.engine {
            EngineKind::Standard => self.solver.info(),
            EngineKind::Core => core_engine::solver_info(self.solver.info()),
        }
    }

    /// Attach a compiled script to a body.
    ///
    /// Replaces whatever was on that body, so recompiling after an edit is
    /// the same call — and clears the failure state, which is what makes
    /// fixing a broken script bring it back to life.
    pub fn attach_script(&mut self, body: usize, program: fusion_script::Program) {
        let script = script::BodyScript::new(body, program);
        match self.scripts.iter().position(|s| s.body == body) {
            Some(index) => {
                self.scripts[index] = script;
                self.script_errors.retain(|(failed, _)| *failed != index);
            }
            None => self.scripts.push(script),
        }
    }

    /// Remove the script on a body, if there is one.
    pub fn detach_script(&mut self, body: usize) {
        if let Some(removed) = self.scripts.iter().position(|script| script.body == body) {
            self.scripts.remove(removed);
            self.script_errors.retain_mut(|(index, _)| {
                if *index == removed {
                    return false;
                }
                if *index > removed {
                    *index -= 1;
                }
                true
            });
        }
    }

    /// Every script attached to this run.
    pub fn scripts(&self) -> &[script::BodyScript] {
        &self.scripts
    }

    /// Script failures since the last time they were taken.
    ///
    /// Drained rather than read, so a caller that reports them cannot report
    /// the same one twice. A script only fails once anyway — it is switched
    /// off at the first error — but a caller that polls would otherwise see
    /// the backlog on every frame until something cleared it.
    pub fn take_script_errors(&mut self) -> Vec<(usize, fusion_script::ScriptError)> {
        std::mem::take(&mut self.script_errors)
    }

    /// Drive a named action from outside, for the step about to run.
    ///
    /// The runtime half of §22's input contract: a game loop reads a keyboard,
    /// a gamepad or a network packet and pushes the result in before calling
    /// [`Self::step`]. Returns false for an unknown action or a value outside
    /// [-1, 1], and true when the take changed.
    ///
    /// **Pushed, not a callback.** A stored closure would have to be cloned
    /// into every checkpoint, could not be serialized with the scene, and would
    /// run user code inside the step where a panic has nowhere to go. A value
    /// is data, so it checkpoints, saves and replays like the rest of the take.
    ///
    /// **And it writes into the take**, which is the whole point rather than a
    /// side effect: a live session is its own recording, so the run reproduces
    /// afterwards without the caller doing anything. Seeking backwards and
    /// driving again truncates what came after, which is what live play means.
    pub fn set_action(&mut self, name: &str, value: f32) -> bool {
        let step = self.step_index;
        self.inputs
            .iter_mut()
            .find(|action| action.name == name)
            .is_some_and(|action| action.record(step, value))
    }

    /// What a named action reads on the step about to run.
    pub fn action_value(&self, name: &str) -> Option<f32> {
        control::value(&self.inputs, name, self.step_index)
    }

    /// Every action this scene defines, in scene order.
    pub fn action_names(&self) -> impl Iterator<Item = &str> {
        self.inputs.iter().map(|action| action.name.as_str())
    }

    /// The contacts the last step found, one manifold per touching pair.
    pub fn contact_manifolds(&self) -> &[ContactManifold] {
        &self.manifolds
    }

    /// Every rigid contact point of the last step: where it was, the impulse
    /// it applied over the step, and the kinetic energy it removed. Divide an
    /// impulse by [`Simulation::solver_dt`] for the mean force.
    pub fn contact_reports(&self) -> &[fusion_solver::ContactReport] {
        &self.contact_reports
    }

    /// Mean residual compression of the worst fluid, as a fraction of rest
    /// density. Zero would be exact incompressibility, which no finite number
    /// of iterations reaches.
    pub fn fluid_compression(&self) -> f32 {
        self.working.fluid_compression()
    }

    /// Largest neighbourhood any fluid particle had in the last step. It
    /// rises well before a fluid solve gets into trouble, which makes it the
    /// first number to look at when one does.
    pub fn peak_fluid_neighbours(&self) -> usize {
        self.working.peak_fluid_neighbours()
    }

    /// Mean temperature of a rigid body, Kelvin.
    ///
    /// Its starting temperature until the heat solve has run; see
    /// [`Simulation::heat`].
    pub fn body_temperature(&self, body: fusion_core::BodyHandle) -> f32 {
        self.world.bodies.heat[body.index()].mean() as f32
    }

    /// Coldest and hottest point of a rigid body, Kelvin.
    pub fn body_temperature_range(&self, body: fusion_core::BodyHandle) -> (f32, f32) {
        let (lo, hi) = self.world.bodies.heat[body.index()].range();
        (lo as f32, hi as f32)
    }

    /// Temperature of a rigid body at a point in the world, Kelvin: the
    /// nearest cell of its grid inside the body.
    pub fn body_temperature_at(
        &self,
        body: fusion_core::BodyHandle,
        point: fusion_math::Vec3,
    ) -> f32 {
        let i = body.index();
        let local = self.world.bodies.pose(i).inverse_transform_point(point);
        self.world.bodies.heat[i].at_local(local) as f32
    }

    /// Set a rigid body to one temperature throughout, Kelvin.
    pub fn set_body_temperature(&mut self, body: fusion_core::BodyHandle, kelvin: f32) {
        self.world.bodies.heat[body.index()].set_uniform(kelvin as f64);
    }

    /// Largest von Mises stress anywhere in a rigid body, Pa. Zero for a body
    /// whose stress has not been measured; see [`Simulation::stress`].
    pub fn body_stress(&self, body: fusion_core::BodyHandle) -> f32 {
        self.world.bodies.stress[body.index()].peak_von_mises
    }

    /// Von Mises stress in a rigid body at a point in the world, Pa: its
    /// nearest cell's.
    pub fn body_stress_at(&self, body: fusion_core::BodyHandle, point: fusion_math::Vec3) -> f32 {
        let i = body.index();
        let local = self.world.bodies.pose(i).inverse_transform_point(point);
        self.world.bodies.stress[i].von_mises_at_local(local)
    }

    /// How close a rigid body is to giving way: its largest ratio of load to
    /// strength, at its temperature. One is where it first yields or cracks.
    pub fn body_utilisation(&self, body: fusion_core::BodyHandle) -> f32 {
        self.world.bodies.stress[body.index()].utilisation
    }

    /// Bodies that broke or deformed since the last call, oldest first — the
    /// last [`MAX_KEPT_FAILURES`] of them.
    ///
    /// Drained, like [`Simulation::take_script_errors`], so a caller that
    /// reports them cannot report one twice.
    pub fn take_failures(&mut self) -> Vec<FailureEvent> {
        std::mem::take(&mut self.failures)
    }

    /// Conjugate-gradient iterations the hardest stress solve took last step,
    /// and how many bodies were solved.
    pub fn stress_effort(&self) -> (u32, usize) {
        (self.stress_working.iterations, self.stress_working.solved)
    }

    /// Heat held by every rigid body, joules above absolute zero. Its change
    /// over a run is what the run turned into heat. See [`heat::thermal_energy`].
    pub fn rigid_thermal_energy(&self) -> f64 {
        heat::thermal_energy(&self.world.bodies)
    }

    /// Total mechanical energy of the whole world, continuum included, J.
    ///
    /// [`World::mechanical_energy`] counts kinetic and potential; this adds
    /// the elastic strain energy the continuum stores, which is the term that
    /// makes a solid's budget balance. Summed in double, because a diagnostic
    /// that is no finer than the thing it measures reports its own rounding —
    /// see `fusion_solver::ledger`.
    pub fn mechanical_energy(&self) -> f64 {
        if let Some(core) = &self.core {
            return core.mechanical_energy();
        }
        self.standard_mechanical_energy(&self.standard_continuum_energy())
    }

    /// Where the continuum keeps its energy, summed over every solid (v0.9).
    pub fn continuum_energy(&self) -> fusion_mpm::ContinuumEnergy {
        if let Some(core) = &self.core {
            return core.continuum_energy();
        }
        self.standard_continuum_energy()
    }

    /// What the last step's continuum solve did — points, substeps, and
    /// whether the material asked for a timestep it could not have.
    pub fn mpm_report(&self) -> MpmReport {
        self.mpm_report
    }

    /// What the last step's unified continuum solve did, when it ran.
    pub fn unified_report(&self) -> fusion_mpm::UnifiedReport {
        self.unified_report
    }

    /// Fluid neighbour lists built since this simulation was created.
    ///
    /// Four fluid substeps a step and four steps an output frame is sixteen
    /// solves a frame; this counts how many of those had to rebuild the lists
    /// rather than reuse them.
    pub fn fluid_neighbour_builds(&self) -> u64 {
        self.working.fluid_neighbour_builds()
    }

    /// The last step's per-stage energy attribution (PLAN.md §23).
    ///
    /// Inert until [`Simulation::energy_ledger`] is set, because it costs a
    /// pass over every body and particle ten times per substep.
    ///
    /// ```
    /// use fusion_core::{BodyDesc, World};
    /// use fusion_math::{vec3, Vec3};
    /// use fusion_sim::Simulation;
    ///
    /// let mut world = World::new();
    /// world.add(BodyDesc::cuboid(vec3(20.0, 1.0, 20.0)).at(vec3(0.0, -0.5, 0.0)).fixed());
    /// world.add(BodyDesc::cuboid(Vec3::ONE).at(vec3(0.0, 0.5, 0.0)));
    ///
    /// let mut sim = Simulation::new(world);
    /// sim.energy_ledger = true;
    /// sim.step();
    ///
    /// // Every joule the step moved is attributed to exactly one stage.
    /// println!("{}", sim.ledger().summary());
    /// assert!(sim.ledger().is_enabled());
    /// ```
    pub fn ledger(&self) -> &fusion_solver::EnergyLedger {
        &self.working.ledger
    }

    /// Per-particle density of one fluid, kg/m³ — the kernel estimate scaled
    /// back into physical units. Empty until the first step has run.
    pub fn fluid_density(&self, fluid: fusion_core::FluidHandle) -> Vec<f32> {
        let Some(scratch) = self.working.fluid.get(fluid.index()) else {
            return Vec::new();
        };
        let mass = self.world.fluids[fluid.index()].params.particle_mass();
        scratch.density().iter().map(|d| d * mass).collect()
    }

    /// Back to t = 0 by rebuilding the state from the scene. A simulation is
    /// causal — there is no seek — so this is the only backwards operation
    /// v0.1 offers. Checkpoint-and-replay rewind arrives with determinism at
    /// v0.4 (§17).
    pub fn reset(&mut self) {
        self.probe_samples.clear();
        for script in &mut self.scripts {
            script.locals.clear();
            script.failed = None;
        }
        self.script_errors.clear();
        self.failures.clear();
        self.world.reset();
        self.mpm.reset_grid();
        self.mpm_report = MpmReport::default();
        self.manifolds.clear();
        self.working = XpbdWorkingSet::new();
        self.islands = IslandSet::new();
        self.time = 0.0;
        self.step_index = 0;
        self.frame_index = 0;
        self.accumulator = 0.0;
        self.last_energy = None;
        self.diagnostics = RunDiagnostics::default();
        self.core = None;
        if self.engine == EngineKind::Core {
            // If callers changed the scene to unsupported physics, the next
            // step refuses it instead of running the previous world's core.
            let _ = self.use_core_engine();
        }
    }

    /// One solver step. This is the loop from PLAN.md §31.
    pub fn step(&mut self) -> StepReport {
        let dt = self.solver_dt;
        if !dt.is_finite() || dt <= 0.0 {
            self.unified_report = fusion_mpm::UnifiedReport {
                failure: Some(fusion_mpm::unified::StepFailure::InvalidConfiguration),
                refused: true,
                ..Default::default()
            };
            return StepReport {
                step_index: self.step_index,
                time: self.time,
                failure: Some(StepFailure::InvalidConfiguration),
                ..Default::default()
            };
        }
        match self.engine {
            EngineKind::Standard => self.step_standard(dt),
            EngineKind::Core => self.step_core(),
        }
    }

    /// Advance by `seconds` of simulated time in whole solver steps, keeping
    /// the remainder for next time. Returns the report of the last step run.
    pub fn advance(&mut self, seconds: f64) -> Option<StepReport> {
        self.advance_with_hook(seconds, |_, _| {})
    }

    /// Advance by `seconds` of simulated time, running `pre_step(sim, dt)`
    /// right before each fixed `solver_dt` step (used by interactive runtime dragging).
    pub fn advance_with_hook(
        &mut self,
        seconds: f64,
        mut pre_step: impl FnMut(&mut Self, f32),
    ) -> Option<StepReport> {
        self.accumulator += seconds;
        let mut last = None;
        // Cap the catch-up so a long pause cannot spiral into an unbounded
        // number of steps.
        let max_steps = 240;
        let mut taken = 0;
        // The comparison is slack by a millionth of a step. `1/240` rounded to
        // f32 is slightly larger than the exact value, so four of them exceed
        // `1/60` and an exact comparison would run three steps per frame
        // forever.
        let dt = self.solver_dt as f64;
        let threshold = dt * (1.0 - 1e-6);
        while self.accumulator >= threshold && taken < max_steps {
            pre_step(self, self.solver_dt);
            let report = self.step();
            let rejected = report.failure.is_some();
            last = Some(report);
            if rejected {
                break; // Keep the requested time available for a corrected retry.
            }
            self.accumulator -= dt;
            taken += 1;
        }
        if taken == max_steps
            && self.engine == EngineKind::Standard
            && self.continuum == ContinuumSolver::Classic
        {
            self.accumulator = 0.0;
        }
        last
    }

    /// Advance by exactly one output frame at [`Simulation::output_rate`].
    pub fn advance_frame(&mut self) -> Option<StepReport> {
        self.frame_index += 1;
        self.advance(1.0 / self.output_rate)
    }

    /// Run `frames` output frames, handing each frame's last step report to
    /// `on_frame`. The plain form of the loop in the target API of §20.
    pub fn run_frames(&mut self, frames: u64, mut on_frame: impl FnMut(&Self, &StepReport)) {
        for _ in 0..frames {
            if let Some(report) = self.advance_frame() {
                on_frame(self, &report);
                if report.failure.is_some() {
                    break;
                }
            }
        }
    }

    /// True while every value in the state is finite. The cheapest possible
    /// "did this explode" check, and what the stacking smoke test asserts.
    ///
    /// Covers particles and fluids as well as bodies. Testing the rigid arrays
    /// alone was a listed defect through v0.4: a NaN anywhere in a cloth or a
    /// fluid reported a perfectly stable scene, which is precisely backwards
    /// for the two systems most able to produce one.
    pub fn is_stable(&self) -> bool {
        self.world.bodies.is_finite()
            && self.world.particles.is_finite()
            && self.world.fluids.iter().all(|f| {
                f.particles.is_finite()
                    && f.affine.iter().all(|m| {
                        m.x_axis.is_finite() && m.y_axis.is_finite() && m.z_axis.is_finite()
                    })
                    && f.volume_ratio.iter().all(|j| j.is_finite() && *j > 0.0)
            })
            && self.world.solids.iter().all(|s| {
                s.particles.is_finite()
                    && s.deformation.iter().chain(&s.affine).all(|m| {
                        m.x_axis.is_finite() && m.y_axis.is_finite() && m.z_axis.is_finite()
                    })
                    && s.plastic.iter().all(|j| j.is_finite() && *j > 0.0)
                    && s.damage.iter().all(|d| d.is_finite())
                    && s.enthalpy.iter().all(|h| h.is_finite())
            })
    }
}
