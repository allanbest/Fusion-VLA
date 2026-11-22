//! The on-disk schema, and its translation to and from the engine's types.
//!
//! Every type here is a *file format*, not a view of an engine struct. They
//! look almost the same today and the reason to keep them apart is that they
//! will not always: a field renamed in [`fusion_core`] must not silently
//! rename itself in files people have saved, and a field removed from the
//! engine must still be readable from files that have it.
//!
//! The rule that follows from that, and the reason `#[serde(default)]` is on
//! nearly everything: **a missing field is the default, and an unknown field
//! is ignored.** An older Fusion reading a newer file gets a scene it can
//! actually run rather than an error about a field it does not care about, and
//! a newer Fusion reading an older file fills in what was added since. Only a
//! change in what an existing field *means* justifies a format bump.

use fusion_core::{
    BodyDesc, BodyType, ClothDesc, Constitutive, FluidDesc, FluidParams, Joint, JointKind,
    Material, Shape, SoftBodyDesc, SolidDesc, SolidMaterial, SolidParams, World,
};
use fusion_math::{vec3, Isometry, Quat, Vec3};
use fusion_sim::{
    CoreIntegrator, CoreSettings, MutualGravitySettings, Simulation, SleepPolicy, Xpbd,
};
use fusion_solver::AirModel;
use serde::{Deserialize, Serialize};

use crate::{ProjectError, FORMAT_VERSION};

// ---------------------------------------------------------------- provenance

/// Which Fusion wrote this, and which format it wrote.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct Provenance {
    /// Format version. See [`crate::FORMAT_VERSION`].
    pub format: u32,
    /// The `fusion-project` package version that wrote the file.
    pub version: String,
    /// A hash identifying the solver build, when one was supplied.
    ///
    /// PLAN.md §22 asks for "ideally a content hash of the relevant solver
    /// code", and this is that slot — filled from `FUSION_SOLVER_HASH` at
    /// compile time if the build sets it, and left empty otherwise. It is
    /// empty in an ordinary `cargo build`, which is the honest answer: the
    /// version alone does not pin down a working tree with uncommitted changes
    /// in the solver, and pretending it does would be worse than admitting the
    /// gap. Wiring it up is a build-script job for the release pipeline.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub solver_hash: Option<String>,
}

impl Provenance {
    /// Provenance for a file written by this build, now.
    pub fn current() -> Self {
        Self {
            format: FORMAT_VERSION,
            version: env!("CARGO_PKG_VERSION").to_string(),
            solver_hash: option_env!("FUSION_SOLVER_HASH").map(str::to_string),
        }
    }
}

// --------------------------------------------------------------------- state

/// Where the run's state lives, for a manifest that describes a run in
/// progress rather than a scene at t = 0 (v1.0, format 2).
///
/// The manifest stays the manifest: a scene of 22 000 fluid particles is 1.6 KB
/// of readable TOML because it says "a block of water this big at this
/// spacing" rather than listing them. A *checkpoint* of the same scene is the
/// particles, and there is no version of that which is readable text — decimal
/// f64 is three times the bytes and is not the same number when it comes back.
/// §22 says so directly: "a human-readable manifest ... referencing binary
/// blobs for meshes, point data and large initial fields".
///
/// Everything here except `path` is a check rather than a payload. A manifest
/// and a blob that have drifted apart — an old state file left beside an
/// edited scene, a copy that stopped halfway — is the failure worth catching,
/// because its symptom is a run that starts and is wrong.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct StateRef {
    /// File holding the state, relative to the manifest.
    pub path: String,
    /// Solver step the state was taken after.
    pub step: u64,
    /// Simulated time at that step, seconds.
    pub time: f64,
    /// Size of the blob, bytes.
    pub bytes: u64,
    /// [`fusion_sim::state_hash`] of the simulation the blob was taken from.
    ///
    /// Proves the blob is intact and complete: it decoded into the state that
    /// was written, every field of it.
    pub hash: u64,
    /// Hash of the manifest's own text, with this section removed.
    ///
    /// Proves the *scene* is the scene, which the field above cannot. A state
    /// blob carries what a step changes and nothing else, so restoring it
    /// overwrites every number `state_hash` looks at — change gravity, or a
    /// material stiffness, or a body's shape, and the restored state
    /// fingerprints identically while the run continues under different
    /// physics. Measured by writing the test first: a scene whose gravity had
    /// been retuned from −9.81 to −3.0 resumed happily, with both the size and
    /// the state fingerprint agreeing.
    ///
    /// The text rather than the values, because the text is what the format
    /// is: two projects that serialize identically are the same scene by
    /// definition, and one that does not round-trip through its own writer has
    /// a bigger problem than this check.
    pub scene: u64,
}

/// FNV-1a over the manifest text.
///
/// A hash, not a checksum with a threat model: what it is up against is a
/// person editing a scene and forgetting the checkpoint beside it, and 64 bits
/// of FNV is a great deal more than that needs. Hand-rolled because
/// `fusion-project`'s only dependencies are serde and toml (§5), and it is
/// eight lines.
pub fn text_hash(text: &str) -> u64 {
    let mut h: u64 = 0xcbf2_9ce4_8422_2325;
    for byte in text.as_bytes() {
        h ^= *byte as u64;
        h = h.wrapping_mul(0x100_0000_01b3);
    }
    h
}

// -------------------------------------------------------------------- config

/// Clock and sampling. PLAN.md §17: frames are an output choice, not the
/// engine's clock, so the two rates are recorded separately.
#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
pub struct RunConfig {
    /// Internal solver timestep, seconds.
    pub solver_dt: f32,
    /// Output frames per simulated second.
    pub output_rate: f64,
    /// Ceiling on speculative contact distance, metres.
    #[serde(default = "default_speculative")]
    pub max_speculative_distance: f32,
}

fn default_speculative() -> f32 {
    2.0
}

impl Default for RunConfig {
    fn default() -> Self {
        Self {
            solver_dt: 1.0 / 240.0,
            output_rate: 60.0,
            max_speculative_distance: 2.0,
        }
    }
}

/// Everything about how the answer is computed, as opposed to what is being
/// computed. §22 lists it as its own line in the tuple, and it is: the same
/// scene at four iterations and at eight is two different runs.
#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
#[serde(default)]
pub struct SolverConfig {
    /// Which engine: `"xpbd"`, the Standard engine, or `"core"`, Fusion Core.
    /// The settings below are the Standard engine's; Fusion Core ignores them.
    pub kind: SolverKind,
    /// Substeps per step.
    pub substeps: u32,
    /// Constraint iterations per substep.
    pub iterations: u32,
    /// Velocity-pass sweeps per substep.
    pub velocity_iterations: u32,
    /// Contact compliance, m/N. Zero is rigid.
    pub contact_compliance: f32,
    /// Overlap left uncorrected, metres.
    pub penetration_slop: f32,
    /// Fraction of remaining overlap removed per iteration.
    pub correction_rate: f32,
    /// Ceiling on depenetration speed, m/s.
    pub max_depenetration_speed: f32,
    /// Relative speed below which a bounce is treated as a rest.
    pub restitution_threshold: f32,
    /// Tangential speed below which friction is treated as static.
    pub sliding_threshold: f32,
    /// Whether speculative contacts are enabled.
    pub speculative: bool,
    /// Whether static friction is also solved at the position level.
    pub static_friction: bool,
    /// Whether depenetration is budgeted rather than applied in full.
    pub budget_depenetration: bool,
    /// Whether the gyroscopic term is integrated.
    pub gyroscopic: bool,
    /// Cloth velocity damping per second.
    pub cloth_damping: f32,
    /// Friction between cloth particles and bodies.
    pub cloth_friction: f32,
    /// Fluid substeps per solver substep.
    pub fluid_substeps: u32,
    /// Density iterations per fluid substep.
    pub fluid_iterations: u32,
    /// Under-relaxation of the density correction.
    pub fluid_relaxation: f32,
    /// Friction between fluid particles and bodies.
    pub fluid_friction: f32,
    /// Speed clamp for fluid particles, m/s.
    pub max_fluid_speed: f32,
    /// When bodies are allowed to sleep.
    pub sleep: SleepConfig,
    /// The air the scene moves through.
    pub air: AirConfig,
    /// Temperature in rigid bodies. Absent is off, so a file written before
    /// rigid bodies had a temperature reads — and hashes — as it always did.
    #[serde(default, skip_serializing_if = "HeatConfig::is_default")]
    pub heat: HeatConfig,
    /// Stress in rigid bodies. Absent is off, for the same reason.
    #[serde(default, skip_serializing_if = "StressConfig::is_default")]
    pub stress: StressConfig,
    /// Which solver runs the deformable solids and the fluids. Absent is the
    /// classic one.
    #[serde(default, skip_serializing_if = "ContinuumEntry::is_classic")]
    pub continuum: ContinuumEntry,
    /// How the Unified continuum gives its fluids pressure. Absent is weakly
    /// compressible, as every file before format 10 ran.
    #[serde(default, skip_serializing_if = "FluidModelEntry::is_default")]
    pub fluid_model: FluidModelEntry,
    /// Whether the Unified continuum solves solids too stiff for the step
    /// implicitly rather than taking substeps for them. Absent is off.
    #[serde(default, skip_serializing_if = "std::ops::Not::not")]
    pub implicit_solids: bool,
    /// Numerical settings for the Fusion Core engine (`crates/core/`).
    #[serde(default, skip_serializing_if = "CoreConfig::is_default")]
    pub core: CoreConfig,
}

/// Time-stepping scheme used by Fusion Core (`phys_mechanics::Integrator`).
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum CoreIntegratorEntry {
    /// First-order A-stable implicit Euler scheme.
    BackwardEuler,
    /// Second-order backward differentiation formula.
    Bdf2,
    /// Conservative Newmark time integration.
    Newmark,
    /// Second-order L-stable TR-BDF2 hybrid scheme (default).
    #[default]
    TrBdf2,
}

impl From<CoreIntegratorEntry> for CoreIntegrator {
    fn from(entry: CoreIntegratorEntry) -> Self {
        match entry {
            CoreIntegratorEntry::BackwardEuler => Self::BackwardEuler,
            CoreIntegratorEntry::Bdf2 => Self::Bdf2,
            CoreIntegratorEntry::Newmark => Self::Newmark,
            CoreIntegratorEntry::TrBdf2 => Self::TrBdf2,
        }
    }
}

/// Core V1 numerical, island, sleeping, joint, and multi-rate settings persisted with a scene.
#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
#[serde(default)]
pub struct CoreConfig {
    /// Implicit time-stepping scheme.
    pub integrator: CoreIntegratorEntry,
    /// Partition disconnected contact/joint components into independent islands.
    pub islands: bool,
    /// Solve independent islands across worker threads.
    pub parallel_islands: bool,
    /// Deactivate settled connected islands cleanly until disturbed.
    pub sleeping: bool,
    /// Continuous quiet duration in seconds before an island sleeps (`None` = disabled).
    #[serde(skip_serializing_if = "Option::is_none")]
    pub settle_time: Option<f64>,
    /// Maximum Newton iterations per substep.
    pub max_iterations: usize,
    /// Maximum tangential friction lag updates per solve.
    pub friction_updates: usize,
    /// Use restitution/impact law during discrete collision resolution.
    pub impact_law: bool,
    /// Enable self-contact on deformable Q2 FEM bodies.
    pub self_contact: bool,
    /// Augmented Lagrangian polish passes on articulated joints.
    pub augmented_lagrangian_passes: usize,
    /// Optional CFL-guided multi-rate subcycling factor (`None` = disabled).
    #[serde(skip_serializing_if = "Option::is_none")]
    pub cfl_subcycling: Option<f64>,
    /// Optional N-body gravitational constant `G` in SI units (`None` = disabled).
    #[serde(skip_serializing_if = "Option::is_none")]
    pub mutual_gravity_g: Option<f64>,
}

impl Default for CoreConfig {
    fn default() -> Self {
        let d = CoreSettings::default();
        Self {
            integrator: CoreIntegratorEntry::TrBdf2,
            islands: d.islands,
            parallel_islands: d.parallel_islands,
            sleeping: d.sleeping,
            settle_time: d.settle_time,
            max_iterations: d.max_iterations,
            friction_updates: d.friction_updates,
            impact_law: d.impact_law,
            self_contact: d.self_contact,
            augmented_lagrangian_passes: d.augmented_lagrangian_passes,
            cfl_subcycling: d.cfl_subcycling,
            mutual_gravity_g: None,
        }
    }
}

impl CoreConfig {
    /// Whether this table matches the default settings so clean scenes omit it.
    pub fn is_default(&self) -> bool {
        *self == Self::default()
    }

    /// Write these dials into `CoreSettings`.
    pub fn apply(&self, s: &mut CoreSettings) {
        s.integrator = self.integrator.into();
        s.islands = self.islands;
        s.parallel_islands = self.parallel_islands;
        s.sleeping = self.sleeping;
        s.settle_time = self.settle_time;
        s.max_iterations = self.max_iterations.max(1);
        s.friction_updates = self.friction_updates;
        s.impact_law = self.impact_law;
        s.self_contact = self.self_contact;
        s.augmented_lagrangian_passes = self.augmented_lagrangian_passes;
        s.cfl_subcycling = self.cfl_subcycling;
        s.mutual_gravity = self.mutual_gravity_g.map(|g| MutualGravitySettings {
            constant: g,
            ..MutualGravitySettings::default()
        });
    }
}

/// [`fusion_sim::FluidModel`], as a file spells it.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum FluidModelEntry {
    /// Pressure from each point's compression, through a stiff equation of
    /// state.
    #[default]
    WeaklyCompressible,
    /// Pressure from a projection on the grid: divergence-free, no sound
    /// speed in the step.
    Incompressible,
}

impl FluidModelEntry {
    /// Whether it is the default.
    pub fn is_default(&self) -> bool {
        *self == Self::WeaklyCompressible
    }
}

impl From<fusion_sim::FluidModel> for FluidModelEntry {
    fn from(m: fusion_sim::FluidModel) -> Self {
        match m {
            fusion_sim::FluidModel::WeaklyCompressible => Self::WeaklyCompressible,
            fusion_sim::FluidModel::Incompressible => Self::Incompressible,
        }
    }
}

impl From<FluidModelEntry> for fusion_sim::FluidModel {
    fn from(m: FluidModelEntry) -> Self {
        match m {
            FluidModelEntry::WeaklyCompressible => Self::WeaklyCompressible,
            FluidModelEntry::Incompressible => Self::Incompressible,
        }
    }
}

/// [`fusion_sim::ContinuumSolver`], as a file spells it.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum ContinuumEntry {
    /// Each solid on its own grid, and fluids by PBF.
    #[default]
    Classic,
    /// Every solid and fluid on one shared MPM grid. Written `"fusion"`, its
    /// name as a preview, so files stay readable by earlier builds; both
    /// spellings are read.
    #[serde(rename = "fusion", alias = "unified")]
    Unified,
}

impl ContinuumEntry {
    /// The name [`ContinuumEntry::Unified`] had while it was a preview.
    #[allow(non_upper_case_globals)]
    #[deprecated(note = "renamed to `ContinuumEntry::Unified`")]
    pub const Fusion: Self = Self::Unified;

    /// Whether it is the default.
    pub fn is_classic(&self) -> bool {
        *self == Self::Classic
    }
}

impl From<fusion_sim::ContinuumSolver> for ContinuumEntry {
    fn from(c: fusion_sim::ContinuumSolver) -> Self {
        match c {
            fusion_sim::ContinuumSolver::Classic => Self::Classic,
            fusion_sim::ContinuumSolver::Unified => Self::Unified,
        }
    }
}

impl From<ContinuumEntry> for fusion_sim::ContinuumSolver {
    fn from(c: ContinuumEntry) -> Self {
        match c {
            ContinuumEntry::Classic => Self::Classic,
            ContinuumEntry::Unified => Self::Unified,
        }
    }
}

/// Which engine a project runs on: [`fusion_sim::EngineKind`], as a file
/// spells it.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum SolverKind {
    /// The Standard engine: XPBD rigid bodies with a Classic or Unified MPM
    /// continuum. Written `"xpbd"`, the name every earlier file uses.
    #[default]
    #[serde(rename = "xpbd", alias = "standard")]
    Standard,
    /// Fusion Core: coupled f64 rigid and finite-element mechanics.
    Core,
}

impl From<fusion_sim::EngineKind> for SolverKind {
    fn from(kind: fusion_sim::EngineKind) -> Self {
        match kind {
            fusion_sim::EngineKind::Standard => Self::Standard,
            fusion_sim::EngineKind::Core => Self::Core,
        }
    }
}

impl Default for SolverConfig {
    fn default() -> Self {
        Self::from_solver(
            &Xpbd::default(),
            &SleepPolicy::default(),
            &AirModel::default(),
        )
    }
}

impl SolverConfig {
    /// Switch to the Unified continuum as a new choice is made: with the
    /// incompressible fluid and implicit stiff solids, which measured the
    /// fastest of its four combinations in every scene tried — water alone
    /// 1.6 ms a step against 2.5, a porcelain block 3.2 against 9.1, water
    /// with a rubber block 11.7 against 12.7. A file that names the Unified
    /// continuum without them still runs as it was written; this is for
    /// scenes being set up, not ones being read.
    pub fn choose_unified(&mut self) {
        self.kind = SolverKind::Standard;
        if self.continuum != ContinuumEntry::Unified {
            self.continuum = ContinuumEntry::Unified;
            self.fluid_model = FluidModelEntry::Incompressible;
            self.implicit_solids = true;
        }
    }

    /// Capture the solver, sleep policy and air model as they stand.
    pub fn from_solver(solver: &Xpbd, sleep: &SleepPolicy, air: &AirModel) -> Self {
        Self {
            kind: SolverKind::Standard,
            substeps: solver.substeps,
            iterations: solver.iterations,
            velocity_iterations: solver.velocity_iterations,
            contact_compliance: solver.contact_compliance,
            penetration_slop: solver.penetration_slop,
            correction_rate: solver.correction_rate,
            max_depenetration_speed: solver.max_depenetration_speed,
            restitution_threshold: solver.restitution_threshold,
            sliding_threshold: solver.sliding_threshold,
            speculative: solver.speculative,
            static_friction: solver.static_friction,
            budget_depenetration: solver.budget_depenetration,
            gyroscopic: solver.gyroscopic,
            cloth_damping: solver.cloth_damping,
            cloth_friction: solver.cloth_friction,
            fluid_substeps: solver.fluid_substeps,
            fluid_iterations: solver.fluid_iterations,
            fluid_relaxation: solver.fluid_relaxation,
            fluid_friction: solver.fluid_friction,
            max_fluid_speed: solver.max_fluid_speed,
            sleep: SleepConfig {
                enabled: sleep.enabled,
                linear_threshold: sleep.linear_threshold,
                angular_threshold: sleep.angular_threshold,
                time_to_sleep: sleep.time_to_sleep,
                max_sleeping_penetration: sleep.max_sleeping_penetration,
            },
            air: AirConfig {
                enabled: air.enabled,
                density: air.density,
                wind: air.wind.to_array(),
                drag_scale: air.drag_scale,
                magnus: air.magnus,
                cloth_drag: air.cloth_drag,
                cloth_lift: air.cloth_lift,
            },
            heat: HeatConfig::default(),
            stress: StressConfig::default(),
            continuum: ContinuumEntry::default(),
            fluid_model: FluidModelEntry::default(),
            implicit_solids: false,
            core: CoreConfig::default(),
        }
    }

    /// Apply this configuration to a simulation.
    pub fn apply(&self, sim: &mut Simulation) -> Result<(), ProjectError> {
        let s = &mut sim.solver;
        s.substeps = self.substeps.max(1);
        s.iterations = self.iterations.max(1);
        s.velocity_iterations = self.velocity_iterations;
        s.contact_compliance = self.contact_compliance;
        s.penetration_slop = self.penetration_slop;
        s.correction_rate = self.correction_rate;
        s.max_depenetration_speed = self.max_depenetration_speed;
        s.restitution_threshold = self.restitution_threshold;
        s.sliding_threshold = self.sliding_threshold;
        s.speculative = self.speculative;
        s.static_friction = self.static_friction;
        s.budget_depenetration = self.budget_depenetration;
        s.gyroscopic = self.gyroscopic;
        s.cloth_damping = self.cloth_damping;
        s.cloth_friction = self.cloth_friction;
        s.fluid_substeps = self.fluid_substeps.max(1);
        s.fluid_iterations = self.fluid_iterations.max(1);
        s.fluid_relaxation = self.fluid_relaxation;
        s.fluid_friction = self.fluid_friction;
        s.max_fluid_speed = self.max_fluid_speed;

        sim.sleep = SleepPolicy {
            enabled: self.sleep.enabled,
            linear_threshold: self.sleep.linear_threshold,
            angular_threshold: self.sleep.angular_threshold,
            time_to_sleep: self.sleep.time_to_sleep,
            max_sleeping_penetration: self.sleep.max_sleeping_penetration,
        };
        sim.air = AirModel {
            enabled: self.air.enabled,
            density: self.air.density,
            wind: Vec3::from_array(self.air.wind),
            drag_scale: self.air.drag_scale,
            magnus: self.air.magnus,
            cloth_drag: self.air.cloth_drag,
            cloth_lift: self.air.cloth_lift,
        };
        sim.heat = self.heat.settings();
        sim.stress = self.stress.settings();
        sim.continuum = self.continuum.into();
        sim.unified.fluid_model = self.fluid_model.into();
        sim.unified.implicit_solids = self.implicit_solids;
        self.core.apply(&mut sim.core_settings);
        match self.kind {
            SolverKind::Core => sim.use_core_engine().map_err(ProjectError::Unresolved)?,
            SolverKind::Standard => sim
                .use_standard_engine()
                .map_err(ProjectError::Unresolved)?,
        }
        Ok(())
    }
}

/// How heat moves in rigid bodies. The numbers are
/// [`fusion_sim::HeatSettings`]'s, and the mapping is the identity.
#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
#[serde(default)]
pub struct HeatConfig {
    /// Whether rigid bodies carry a temperature field.
    pub enabled: bool,
    /// Air temperature, K.
    pub air_temperature: f32,
    /// Convective coefficient to the air, W/(m^2 K).
    pub convection: f32,
    /// Whether surfaces radiate.
    pub radiation: bool,
    /// Interface conductance between bodies in contact, W/(m^2 K).
    pub contact_conductance: f32,
}

impl Default for HeatConfig {
    fn default() -> Self {
        Self::from(&fusion_sim::HeatSettings::default())
    }
}

impl From<&fusion_sim::HeatSettings> for HeatConfig {
    fn from(h: &fusion_sim::HeatSettings) -> Self {
        Self {
            enabled: h.enabled,
            air_temperature: h.air_temperature,
            convection: h.convection,
            radiation: h.radiation,
            contact_conductance: h.contact_conductance,
        }
    }
}

impl HeatConfig {
    /// Whether this is the default, so the writer can leave it out.
    pub fn is_default(&self) -> bool {
        *self == Self::default()
    }

    /// The simulation settings this describes.
    pub fn settings(&self) -> fusion_sim::HeatSettings {
        fusion_sim::HeatSettings {
            enabled: self.enabled,
            air_temperature: self.air_temperature,
            convection: self.convection,
            radiation: self.radiation,
            contact_conductance: self.contact_conductance,
        }
    }
}

/// How rigid bodies carry load. The numbers are
/// [`fusion_sim::StressSettings`]'s, and the mapping is the identity.
#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
#[serde(default)]
pub struct StressConfig {
    /// Whether every rigid body's stress is measured. A body that can break or
    /// deform is measured regardless.
    pub enabled: bool,
    /// Whether thermal strain counts.
    pub thermal: bool,
    /// Whether an impact's load is shortened to the time it really lasts.
    pub impact_duration: bool,
}

impl Default for StressConfig {
    fn default() -> Self {
        Self::from(&fusion_sim::StressSettings::default())
    }
}

impl From<&fusion_sim::StressSettings> for StressConfig {
    fn from(s: &fusion_sim::StressSettings) -> Self {
        Self {
            enabled: s.enabled,
            thermal: s.thermal,
            impact_duration: s.impact_duration,
        }
    }
}

impl StressConfig {
    /// Whether this is the default, so the writer can leave it out.
    pub fn is_default(&self) -> bool {
        *self == Self::default()
    }

    /// The simulation settings this describes.
    pub fn settings(&self) -> fusion_sim::StressSettings {
        fusion_sim::StressSettings {
            enabled: self.enabled,
            thermal: self.thermal,
            impact_duration: self.impact_duration,
        }
    }
}

/// What a rigid object does when it is loaded past its strength. See
/// [`fusion_core::Failure`]; the mapping is the identity.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum FailureEntry {
    /// Stays whole.
    #[default]
    Never,
    /// Breaks into rigid pieces.
    Breaks,
    /// Deforms and melts, as a continuum.
    Deforms,
}

impl FailureEntry {
    /// Whether this is the default, so the writer can leave it out.
    pub fn is_never(&self) -> bool {
        *self == Self::Never
    }
}

impl From<fusion_core::Failure> for FailureEntry {
    fn from(f: fusion_core::Failure) -> Self {
        match f {
            fusion_core::Failure::Never => Self::Never,
            fusion_core::Failure::Breaks => Self::Breaks,
            fusion_core::Failure::Deforms => Self::Deforms,
        }
    }
}

impl From<FailureEntry> for fusion_core::Failure {
    fn from(f: FailureEntry) -> Self {
        match f {
            FailureEntry::Never => Self::Never,
            FailureEntry::Breaks => Self::Breaks,
            FailureEntry::Deforms => Self::Deforms,
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
#[serde(default)]
pub struct SleepConfig {
    pub enabled: bool,
    pub linear_threshold: f32,
    pub angular_threshold: f32,
    pub time_to_sleep: f32,
    pub max_sleeping_penetration: f32,
}

impl Default for SleepConfig {
    fn default() -> Self {
        let p = SleepPolicy::default();
        Self {
            enabled: p.enabled,
            linear_threshold: p.linear_threshold,
            angular_threshold: p.angular_threshold,
            time_to_sleep: p.time_to_sleep,
            max_sleeping_penetration: p.max_sleeping_penetration,
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
#[serde(default)]
pub struct AirConfig {
    pub enabled: bool,
    pub density: f32,
    pub wind: [f32; 3],
    pub drag_scale: f32,
    pub magnus: f32,
    pub cloth_drag: f32,
    pub cloth_lift: f32,
}

impl Default for AirConfig {
    fn default() -> Self {
        let a = AirModel::default();
        Self {
            enabled: a.enabled,
            density: a.density,
            wind: a.wind.to_array(),
            drag_scale: a.drag_scale,
            magnus: a.magnus,
            cloth_drag: a.cloth_drag,
            cloth_lift: a.cloth_lift,
        }
    }
}

/// World-level parameters and the seed.
#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
#[serde(default)]
pub struct WorldConfig {
    /// Uniform acceleration, m/s^2.
    pub gravity: [f32; 3],
    /// Linear velocity decay per second.
    pub linear_damping: f32,
    /// Angular velocity decay per second.
    pub angular_damping: f32,
    /// PLAN.md §22: "RNG seeds, explicitly logged, never implicit."
    pub seed: u64,
    /// Domain origin and unit scales (§7). SI unless a scene says otherwise.
    pub domain: DomainConfig,
}

impl Default for WorldConfig {
    fn default() -> Self {
        Self::from_world(&World::new())
    }
}

impl WorldConfig {
    /// Capture a world's global settings.
    pub fn from_world(world: &World) -> Self {
        Self {
            gravity: world.gravity.to_array(),
            linear_damping: world.linear_damping,
            angular_damping: world.angular_damping,
            seed: world.seed,
            domain: DomainConfig {
                origin: world.domain.origin.to_array(),
                length_scale: world.domain.length_scale,
                time_scale: world.domain.time_scale,
                mass_scale: world.domain.mass_scale,
            },
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
#[serde(default)]
pub struct DomainConfig {
    pub origin: [f64; 3],
    pub length_scale: f64,
    pub time_scale: f64,
    pub mass_scale: f64,
}

impl Default for DomainConfig {
    fn default() -> Self {
        Self {
            origin: [0.0; 3],
            length_scale: 1.0,
            time_scale: 1.0,
            mass_scale: 1.0,
        }
    }
}

/// Free-form experiment metadata (§22). Nothing here affects the simulation,
/// and that is the point — it is where the human context goes so that it
/// travels with the run instead of in a chat message.
#[derive(Clone, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(default)]
pub struct Metadata {
    /// Free-text note from whoever saved the file.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub note: Option<String>,
    /// Identifier for the run this file came from, if it had one.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub run_id: Option<String>,
    /// The step a bug appears at. §22's "a `.fusion` file plus a step number".
    #[serde(skip_serializing_if = "Option::is_none")]
    pub step_of_interest: Option<u64>,
    /// Sweep parameters, as free key-value pairs.
    #[serde(skip_serializing_if = "Vec::is_empty")]
    pub tags: Vec<(String, String)>,
}

impl Metadata {
    /// Whether there is anything to write. An empty `[metadata]` table in
    /// every file is noise in a format meant to be read.
    pub fn is_empty(&self) -> bool {
        *self == Self::default()
    }
}

/// Project-specific starting values for newly authored objects.
/// These do not alter objects already in the scene.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(default)]
pub struct CreationDefaults {
    /// Material for ordinary primitives and imports. Absent selects Solid.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub material: Option<String>,
    /// Include a dynamic rigidbody; otherwise new rigid objects are static.
    pub rigidbody: bool,
    /// Initial linear velocity, metres per second.
    pub velocity: [f32; 3],
    /// Initial angular velocity, radians per second, for dynamic rigid objects.
    pub spin: [f32; 3],
}

impl Default for CreationDefaults {
    fn default() -> Self {
        Self {
            material: None,
            rigidbody: true,
            velocity: [0.0; 3],
            spin: [0.0; 3],
        }
    }
}

impl CreationDefaults {
    /// Whether the default table can be omitted from a scene file.
    pub fn is_default(&self) -> bool {
        *self == Self::default()
    }

    /// Whether the initial motion contains only finite numbers.
    pub fn is_valid(&self) -> bool {
        self.velocity
            .iter()
            .chain(&self.spin)
            .all(|n| n.is_finite())
    }
}

// --------------------------------------------------------------------- scene

/// How a material is solved (PLAN.md §35, format 3).
///
/// **This is the field that used to be the menu the user picked from.** A cube
/// and a jelly cube are the same object; the difference between them is not a
/// difference in what somebody is making, it is a difference in how it is
/// solved, and choosing a solver is not an authoring decision.
///
/// It lives on the *material* rather than on the object because that is where
/// the answer actually comes from. Steel is rigid because steel's stiffness is
/// four orders of magnitude past what an explicit method resolves at a
/// graphics timestep (§9); jelly is deformable because deforming is the whole
/// of what jelly does. An object made of steel and an object made of jelly
/// differ in one dropdown, and the cost of changing it — a pose becoming a
/// hundred material points — is something the editor reports rather than
/// hides.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum Representation {
    /// A pose and a velocity, solved by XPBD. The default, and what "Solid",
    /// the default material, is.
    #[default]
    Rigid,
    /// Material points on a background grid, solved by MPM. Carries a
    /// deformation gradient, and optionally heat, plasticity and damage.
    Deformable,
}

impl Representation {
    /// Whether this is the default, so the writer can leave it out.
    pub fn is_rigid(&self) -> bool {
        matches!(self, Self::Rigid)
    }
}

/// What a material looks like, as opposed to what it does.
///
/// §35 puts the two halves of a material side by side because a user does not
/// think of them as different subjects. Nothing here reaches the solver: an
/// object's colour cannot change where it lands, and the separation is worth
/// keeping strict for exactly that reason — a scene file's physics section
/// should be readable without wondering whether `metallic` is load-bearing.
#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
#[serde(default)]
pub struct Appearance {
    /// Base colour, linear RGB in `[0, 1]`.
    pub color: [f32; 3],
    /// Metallic in `[0, 1]`.
    pub metallic: f32,
    /// Roughness in `[0, 1]`. Zero is a mirror.
    pub roughness: f32,
    /// Emission, as a multiplier on the base colour.
    pub emission: f32,
}

impl Default for Appearance {
    /// The grey §35 calls the default material's: neutral, non-metallic and
    /// matte, so a scene nobody has styled reads as geometry rather than as a
    /// styling decision somebody made badly.
    fn default() -> Self {
        Self {
            color: [0.72, 0.72, 0.74],
            metallic: 0.0,
            roughness: 0.55,
            emission: 0.0,
        }
    }
}

impl Appearance {
    /// Whether this is the default, so the writer can leave it out.
    pub fn is_default(&self) -> bool {
        *self == Self::default()
    }
}

/// The physics a deformable material needs and a rigid one has no use for.
///
/// Its own table rather than fields on [`MaterialEntry`] so that a rigid
/// material — which is most of them — writes four lines and not twenty. The
/// numbers are exactly [`SolidMaterial`]'s, and the mapping between them is
/// the identity; this type exists to give the file a shape, not to reinterpret
/// anything.
#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
#[serde(default)]
pub struct DeformableEntry {
    /// Young's modulus, Pa. The number that sets the timestep.
    pub youngs_modulus: f32,
    /// Poisson's ratio.
    pub poissons_ratio: f32,
    /// The constitutive law.
    pub model: ConstitutiveEntry,
    /// Compression past which snow yields, as a fraction.
    pub critical_compression: f32,
    /// Stretch past which snow yields, as a fraction.
    pub critical_stretch: f32,
    /// How sharply packing stiffens snow.
    pub hardening: f32,
    /// Specific heat capacity, J/(kg K).
    pub specific_heat: f32,
    /// Thermal conductivity, W/(m K).
    pub conductivity: f32,
    /// Linear thermal expansion, 1/K.
    pub expansion: f32,
    /// Melting point, K. Absent means it does not melt.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub melt_point: Option<f32>,
    /// Latent heat of fusion, J/kg.
    pub latent_heat: f32,
    /// Fraction of stiffness lost at the melting point.
    pub softening: f32,
    /// Fraction of plastic work that becomes heat.
    pub plastic_heating: f32,
    /// Surface emissivity, in `[0, 1]`.
    pub emissivity: f32,
    /// Stretch at which damage starts. Absent means it does not break.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub fracture_onset: Option<f32>,
    /// Stretch at which a point has fully parted.
    pub fracture_failure: f32,
    /// Fibres: how much stiffer it is pulled along them than its Young's
    /// modulus says, as a multiple of it. Absent is no fibres. Which way they
    /// run is the object's grain. Read by the unified continuum solver.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub fibre: Option<f32>,
}

impl Default for DeformableEntry {
    fn default() -> Self {
        Self::from(&SolidMaterial::JELLY)
    }
}

/// How a rigid material stores, conducts and sheds heat.
///
/// Its own table for the same reason [`DeformableEntry`] is: a material that
/// keeps the neutral defaults writes nothing, and one that is steel writes the
/// eight numbers that make it steel. The numbers are [`fusion_core::Thermal`]'s
/// and the mapping is the identity. Missing keys are the neutral default's.
#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
#[serde(default)]
pub struct ThermalEntry {
    /// Specific heat capacity, J/(kg K).
    pub specific_heat: f32,
    /// Thermal conductivity, W/(m K).
    pub conductivity: f32,
    /// Linear thermal expansion, 1/K.
    pub expansion: f32,
    /// Melting point, K. Absent means it does not melt.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub melt_point: Option<f32>,
    /// Latent heat of fusion, J/kg.
    pub latent_heat: f32,
    /// Fraction of stiffness lost at the melting point.
    pub softening: f32,
    /// Fraction of plastic work that becomes heat.
    pub plastic_heating: f32,
    /// Surface emissivity, in `[0, 1]`.
    pub emissivity: f32,
}

impl Default for ThermalEntry {
    fn default() -> Self {
        Self::from(&fusion_core::Thermal::GENERIC)
    }
}

impl From<&fusion_core::Thermal> for ThermalEntry {
    fn from(t: &fusion_core::Thermal) -> Self {
        Self {
            specific_heat: t.specific_heat,
            conductivity: t.conductivity,
            expansion: t.expansion,
            melt_point: t.melt_point.is_finite().then_some(t.melt_point),
            latent_heat: t.latent_heat,
            softening: t.softening,
            plastic_heating: t.plastic_heating,
            emissivity: t.emissivity,
        }
    }
}

impl From<&ThermalEntry> for fusion_core::Thermal {
    fn from(e: &ThermalEntry) -> Self {
        Self {
            specific_heat: e.specific_heat,
            conductivity: e.conductivity,
            expansion: e.expansion,
            melt_point: e.melt_point.unwrap_or(f32::INFINITY),
            latent_heat: e.latent_heat,
            softening: e.softening,
            plastic_heating: e.plastic_heating,
            emissivity: e.emissivity,
        }
    }
}

/// How stiff a rigid material is and how much it carries before it fails.
///
/// The numbers are [`fusion_core::Strength`]'s and the mapping is the
/// identity, with an infinite strength written as absent. Missing keys are the
/// neutral default's.
#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
#[serde(default)]
pub struct StrengthEntry {
    /// Young's modulus, Pa.
    pub youngs_modulus: f32,
    /// Poisson's ratio.
    pub poissons_ratio: f32,
    /// Von Mises stress at which it yields, Pa. Absent means brittle.
    ///
    /// Each of these three defaults to absent on its own, rather than to the
    /// neutral material's number: an infinite strength is written by leaving
    /// it out, so reading a missing one as the neutral plastic's 40 MPa turned
    /// every saved concrete ductile.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub yield_stress: Option<f32>,
    /// Ultimate tensile strength, Pa. Absent means unbreakable in tension.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub tensile: Option<f32>,
    /// Compressive strength, Pa. Absent means it does not crush.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub compressive: Option<f32>,
    /// The strengths across a grain, for a material that has one — wood.
    /// Absent is the same every way, and the three above are then the
    /// strengths along the grain.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub grain: Option<GrainEntry>,
    /// How it comes apart when it breaks. Absent is wood's for a material
    /// with a grain, a metal's for one that yields, and a crack otherwise.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub pattern: Option<PatternEntry>,
}

/// [`fusion_core::Pattern`], as a file spells it.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum PatternEntry {
    /// `"crack"`.
    Crack,
    /// `"shatter"`.
    Shatter,
    /// `"chip"`.
    Chip,
    /// `"splinter"`.
    Splinter,
    /// `"tear"`.
    Tear,
}

impl From<fusion_core::Pattern> for PatternEntry {
    fn from(p: fusion_core::Pattern) -> Self {
        use fusion_core::Pattern;
        match p {
            Pattern::Crack => Self::Crack,
            Pattern::Shatter => Self::Shatter,
            Pattern::Chip => Self::Chip,
            Pattern::Splinter => Self::Splinter,
            Pattern::Tear => Self::Tear,
        }
    }
}

impl From<PatternEntry> for fusion_core::Pattern {
    fn from(p: PatternEntry) -> Self {
        match p {
            PatternEntry::Crack => Self::Crack,
            PatternEntry::Shatter => Self::Shatter,
            PatternEntry::Chip => Self::Chip,
            PatternEntry::Splinter => Self::Splinter,
            PatternEntry::Tear => Self::Tear,
        }
    }
}

/// [`fusion_core::Grain`]: strengths across a grain, Pa.
#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
pub struct GrainEntry {
    /// Tension across the grain before the fibres pull apart.
    pub tensile_across: f32,
    /// Compression across the grain before it crushes.
    pub compressive_across: f32,
    /// Shear along the grain before the fibres slide.
    pub shear: f32,
}

impl Default for StrengthEntry {
    fn default() -> Self {
        Self::from(&fusion_core::Strength::GENERIC)
    }
}

impl From<&fusion_core::Strength> for StrengthEntry {
    fn from(s: &fusion_core::Strength) -> Self {
        let finite = |v: f32| v.is_finite().then_some(v);
        Self {
            youngs_modulus: s.youngs_modulus,
            poissons_ratio: s.poissons_ratio,
            yield_stress: finite(s.yield_stress),
            tensile: finite(s.tensile),
            compressive: finite(s.compressive),
            grain: s.grain.map(|g| GrainEntry {
                tensile_across: g.tensile_across,
                compressive_across: g.compressive_across,
                shear: g.shear,
            }),
            pattern: Some(s.pattern.into()),
        }
    }
}

impl From<&StrengthEntry> for fusion_core::Strength {
    fn from(e: &StrengthEntry) -> Self {
        Self {
            youngs_modulus: e.youngs_modulus,
            poissons_ratio: e.poissons_ratio,
            yield_stress: e.yield_stress.unwrap_or(f32::INFINITY),
            tensile: e.tensile.unwrap_or(f32::INFINITY),
            compressive: e.compressive.unwrap_or(f32::INFINITY),
            grain: e.grain.map(|g| fusion_core::Grain {
                tensile_across: g.tensile_across,
                compressive_across: g.compressive_across,
                shear: g.shear,
            }),
            pattern: e.pattern.map(Into::into).unwrap_or_else(|| {
                let yields =
                    e.yield_stress.unwrap_or(f32::INFINITY) < e.tensile.unwrap_or(f32::INFINITY);
                fusion_core::Pattern::implied_by(e.grain.is_some(), yields)
            }),
        }
    }
}

/// A named material, referenced by objects rather than repeated in them.
///
/// Format 3 widened this from "density, friction, restitution" to the whole of
/// what a material is, because §35 makes the material the component that
/// decides how an object is simulated. The widening is additive: every field
/// added has a default, so a format-2 file's four-line material still reads,
/// and a rigid material still writes four lines.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct MaterialEntry {
    /// Name objects refer to it by.
    pub name: String,
    /// How it is solved (§35). Absent is rigid.
    #[serde(default, skip_serializing_if = "Representation::is_rigid")]
    pub representation: Representation,
    /// kg/m^3.
    pub density: f32,
    /// Coulomb friction coefficient.
    pub friction: f32,
    /// Restitution in [0, 1].
    pub restitution: f32,
    /// What it looks like. Absent is the neutral grey.
    #[serde(default, skip_serializing_if = "Appearance::is_default")]
    pub look: Appearance,
    /// The continuum parameters, for a deformable material.
    ///
    /// Absent on a rigid one, and absent on a deformable one that wants
    /// [`SolidMaterial::JELLY`]'s numbers — the same rule as everywhere else
    /// in this format: a missing field is the default.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub deformable: Option<DeformableEntry>,
    /// How a rigid object of this material takes and loses heat.
    ///
    /// Absent means the deformable table's thermal numbers when there is one,
    /// and [`fusion_core::Thermal::GENERIC`] otherwise — so a scene written
    /// before rigid bodies had a temperature reads exactly as it did, and a
    /// jelly that is solved rigid still conducts like jelly.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub thermal: Option<ThermalEntry>,
    /// How stiff a rigid object of this material is and how much load it
    /// carries before it yields or cracks. Absent means
    /// [`fusion_core::Strength::GENERIC`], so a scene written before rigid
    /// bodies had a stress reads as it did.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub strength: Option<StrengthEntry>,
}

impl MaterialEntry {
    /// Check physical and appearance parameters before building a scene or
    /// installing a material pack. Missing optional strengths and melting
    /// points retain their documented infinite/no-transition meaning.
    pub fn validate(&self) -> Result<(), String> {
        let valid = |v: f32, min: f32, max: f32| v.is_finite() && v >= min && v <= max;
        if !self.density.is_finite()
            || self.density <= 0.0
            || !valid(self.friction, 0.0, f32::MAX)
            || !valid(self.restitution, 0.0, 1.0)
            || !self.look.color.iter().all(|v| valid(*v, 0.0, 1.0))
            || !valid(self.look.metallic, 0.0, 1.0)
            || !valid(self.look.roughness, 0.0, 1.0)
            || !valid(self.look.emission, 0.0, f32::MAX)
        {
            return Err(format!(
                "{} has invalid density, surface or appearance values",
                self.name
            ));
        }
        if let Some(d) = self.deformable {
            if !valid(d.youngs_modulus, f32::MIN_POSITIVE, f32::MAX)
                || !valid(d.poissons_ratio, -0.999, 0.499)
                || !valid(d.specific_heat, f32::MIN_POSITIVE, f32::MAX)
                || !valid(d.conductivity, 0.0, f32::MAX)
                || !d.expansion.is_finite()
                || !valid(d.emissivity, 0.0, 1.0)
                || d.fibre.is_some_and(|v| !valid(v, 0.0, f32::MAX))
                || !valid(d.latent_heat, 0.0, f32::MAX)
                || !valid(d.softening, 0.0, 1.0)
                || !valid(d.plastic_heating, 0.0, 1.0)
                || !valid(d.hardening, 0.0, f32::MAX)
                || !valid(d.critical_compression, 0.0, 0.999)
                || !valid(d.critical_stretch, 0.0, f32::MAX)
                || d.melt_point
                    .is_some_and(|v| !valid(v, f32::MIN_POSITIVE, f32::MAX))
                || d.fracture_onset.is_some_and(|v| {
                    !valid(v, 0.0, f32::MAX)
                        || !d.fracture_failure.is_finite()
                        || d.fracture_failure <= v
                })
                || d.fracture_failure.is_nan()
                || d.fracture_failure < 0.0
            {
                return Err(format!("{} has invalid continuum parameters", self.name));
            }
        }
        if let Some(t) = self.thermal {
            if !valid(t.specific_heat, f32::MIN_POSITIVE, f32::MAX)
                || !valid(t.conductivity, 0.0, f32::MAX)
                || !t.expansion.is_finite()
                || !valid(t.latent_heat, 0.0, f32::MAX)
                || !valid(t.softening, 0.0, 1.0)
                || !valid(t.plastic_heating, 0.0, 1.0)
                || !valid(t.emissivity, 0.0, 1.0)
                || t.melt_point
                    .is_some_and(|v| !valid(v, f32::MIN_POSITIVE, f32::MAX))
            {
                return Err(format!("{} has invalid thermal parameters", self.name));
            }
        }
        if let Some(s) = self.strength {
            if !valid(s.youngs_modulus, f32::MIN_POSITIVE, f32::MAX)
                || !s.poissons_ratio.is_finite()
                || s.poissons_ratio <= -1.0
                || s.poissons_ratio >= 0.5
                || [s.yield_stress, s.tensile, s.compressive]
                    .iter()
                    .flatten()
                    .any(|&v| !valid(v, 0.0, f32::MAX))
                || s.grain.is_some_and(|g| {
                    [g.tensile_across, g.compressive_across, g.shear]
                        .iter()
                        .any(|&v| !valid(v, 0.0, f32::MAX))
                })
            {
                return Err(format!("{} has invalid strength parameters", self.name));
            }
        }
        Ok(())
    }

    /// The default authoring material: neutral grey and indestructible.
    /// It uses rigid mechanics, so it allocates no stress or thermal state.
    pub fn solid() -> Self {
        Self::rigid("Solid", &Material::GENERIC)
    }

    /// A rigid material with the default appearance.
    pub fn rigid(name: impl Into<String>, material: &Material) -> Self {
        Self {
            name: name.into(),
            representation: Representation::Rigid,
            density: material.density,
            friction: material.friction,
            restitution: material.restitution,
            look: Appearance::default(),
            deformable: None,
            thermal: (material.thermal != fusion_core::Thermal::GENERIC)
                .then(|| ThermalEntry::from(&material.thermal)),
            strength: (material.strength != fusion_core::Strength::GENERIC)
                .then(|| StrengthEntry::from(&material.strength)),
        }
    }

    /// Builder: the same material, looking like this. Linear RGB.
    pub fn looking(mut self, color: [f32; 3], metallic: f32, roughness: f32) -> Self {
        self.look.color = color;
        self.look.metallic = metallic;
        self.look.roughness = roughness;
        self
    }

    /// The rigid half of this material, which every representation has.
    ///
    /// A deformable object still needs a density and still rubs against rigid
    /// geometry, so friction and restitution are not rigid-only fields — they
    /// are the properties of a *surface*, which every object has.
    pub fn rigid_part(&self) -> Material {
        let thermal = match (&self.thermal, &self.deformable) {
            (Some(t), _) => fusion_core::Thermal::from(t),
            (None, Some(d)) => SolidMaterial::from(d).thermal,
            (None, None) => fusion_core::Thermal::GENERIC,
        };
        Material {
            density: self.density,
            friction: self.friction,
            restitution: self.restitution,
            thermal,
            strength: self
                .strength
                .as_ref()
                .map(fusion_core::Strength::from)
                .unwrap_or(fusion_core::Strength::GENERIC),
        }
    }

    /// The continuum parameters this material implies.
    pub fn solid_part(&self) -> SolidMaterial {
        let d = self.deformable.unwrap_or_default();
        let mut material = SolidMaterial::from(&d);
        material.density = self.density;
        material
    }
}

/// Shape, tagged so a reader can tell a sphere from a box without guessing
/// from which fields are present.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(tag = "kind", rename_all = "lowercase")]
pub enum ShapeEntry {
    /// A sphere.
    Sphere {
        /// Radius, metres.
        radius: f32,
    },
    /// Full size, not half-extents. The engine stores half-extents and the
    /// file stores what a person would type — `size = [1, 1, 1]` is a metre
    /// cube, and having to remember which convention a file uses is exactly
    /// the sort of thing a text format should not ask.
    Cuboid {
        /// Full extents, metres — not half, so a file reads the way a person
        /// would describe the box.
        size: [f32; 3],
    },
    /// A convex hull, by its points.
    ///
    /// **The points, not a path to the mesh they came from.** PLAN.md §21
    /// makes the scene the single source of truth, and a file that says
    /// `mesh = "chair.obj"` is a file that stops being one the moment the
    /// `.obj` moves — the same failure as a texture path in a game engine,
    /// with a physics answer that silently changes instead of a missing
    /// texture. A hull is tens of vertices; storing them costs a kilobyte and
    /// buys a scene that still simulates in five years.
    ///
    /// It is the *hull's* vertices rather than the mesh's, for the same
    /// reason: the shape the solver used is the shape the file records.
    Hull {
        /// Vertices, metres, in the body's own frame.
        points: Vec<[f32; 3]>,
    },
}

impl From<&Shape> for ShapeEntry {
    fn from(shape: &Shape) -> Self {
        match shape {
            Shape::Sphere { radius } => Self::Sphere { radius: *radius },
            Shape::Cuboid { half_extents } => Self::Cuboid {
                size: (*half_extents * 2.0).to_array(),
            },
            Shape::Hull(hull) => Self::Hull {
                points: hull.vertices().iter().map(|v| v.to_array()).collect(),
            },
        }
    }
}

impl From<ShapeEntry> for Shape {
    fn from(entry: ShapeEntry) -> Self {
        match entry {
            ShapeEntry::Sphere { radius } => Shape::sphere(radius),
            ShapeEntry::Cuboid { size } => Shape::cuboid(Vec3::from_array(size)),
            // A hull that will not rebuild becomes a small box rather than a
            // failed load. The alternative is a scene that refuses to open
            // because one body's points were degenerate, which is a worse
            // trade for a file somebody has been working in — and `verify`
            // reports the substitution rather than hiding it.
            ShapeEntry::Hull { points } => {
                let points: Vec<Vec3> = points.iter().map(|p| Vec3::from_array(*p)).collect();
                Shape::hull(&points).unwrap_or_else(|| Shape::cuboid(Vec3::splat(0.2)))
            }
        }
    }
}

/// Whether the solver may move a body.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum BodyKindEntry {
    /// Moved by forces and constraints.
    #[default]
    Dynamic,
    /// Immovable.
    Static,
    /// Immovable by the solver, and moved by its script.
    ///
    /// Written as `kinematic` in the file. Loading a project that predates
    /// this reads every body as dynamic or static exactly as before, because
    /// nothing that was ever written says `kinematic`.
    Kinematic,
}

impl BodyKindEntry {
    /// Whether this is the default, so the writer can leave it out.
    pub fn is_dynamic(&self) -> bool {
        matches!(self, Self::Dynamic)
    }
}

/// One rigid body in a project file.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct BodyEntry {
    /// Its collision shape.
    pub shape: ShapeEntry,
    /// Position, metres.
    #[serde(default, skip_serializing_if = "is_zero3")]
    pub position: [f32; 3],
    /// Orientation as `[x, y, z, w]`.
    #[serde(default = "identity_quat", skip_serializing_if = "is_identity_quat")]
    pub orientation: [f32; 4],
    /// Initial linear velocity, m/s.
    #[serde(default, skip_serializing_if = "is_zero3")]
    pub linear_velocity: [f32; 3],
    /// Initial angular velocity, rad/s.
    #[serde(default, skip_serializing_if = "is_zero3")]
    pub angular_velocity: [f32; 3],
    /// Whether the solver may move it.
    #[serde(default, skip_serializing_if = "is_dynamic")]
    pub body_type: BodyKindEntry,
    /// Name of a `[[material]]` entry.
    pub material: String,
    /// Overrides the mass the shape and density imply.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub mass: Option<f32>,
    /// A behaviour script, in the language `fusion-script` defines.
    ///
    /// Stored as the source text, not as anything compiled. A `.fusion` file
    /// is a thing people read and edit in a text editor, and a scene whose
    /// behaviour was an opaque blob would stop being that. It also means a
    /// script survives a version of the compiler that has forgotten how to
    /// parse it: the text is still there to be fixed.
    ///
    /// Written as a TOML multi-line string, so indentation — which this
    /// language makes structural — survives the round trip exactly.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub script: Option<String>,
}

fn identity_quat() -> [f32; 4] {
    [0.0, 0.0, 0.0, 1.0]
}

/// Written only when it is not the default.
///
/// A scene of forty resting boxes otherwise spends three quarters of its lines
/// stating that nothing is moving and nothing is rotated. Every one of these
/// fields carries `#[serde(default)]` on the way in, so omitting it on the way
/// out loses nothing and is the difference between a file somebody reads and a
/// file somebody greps.
fn is_zero3(v: &[f32; 3]) -> bool {
    *v == [0.0, 0.0, 0.0]
}

fn is_identity_quat(q: &[f32; 4]) -> bool {
    *q == [0.0, 0.0, 0.0, 1.0]
}

fn is_zero(v: &f32) -> bool {
    *v == 0.0
}

fn is_dynamic(kind: &BodyKindEntry) -> bool {
    matches!(kind, BodyKindEntry::Dynamic)
}

fn is_true(v: &bool) -> bool {
    *v
}

/// Which degrees of freedom a joint removes.
#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
#[serde(tag = "kind", rename_all = "lowercase")]
pub enum JointKindEntry {
    /// Anchors coincide, rotation free.
    Ball,
    /// Anchors and orientation both locked.
    Fixed,
    /// Anchors coincide, rotation about the shared +X axis only.
    Hinge,
    /// Anchors held between `min` and `max` apart.
    Distance {
        /// Closest the anchors may come, metres.
        min: f32,
        /// Furthest they may separate, metres.
        max: f32,
    },
    /// Translation along the shared +X axis only, between `min` and `max`.
    Slider {
        /// Lower travel limit, metres.
        min: f32,
        /// Upper travel limit, metres.
        max: f32,
    },
}

/// One joint in a project file. Bodies are referred to by position in
/// `bodies`, so deleting a body renumbers the joints that follow it.
#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
pub struct JointEntry {
    /// Indices into `[[body]]`, in the order they appear in the file.
    pub body_a: u32,
    /// Index of the second body.
    pub body_b: u32,
    /// Anchor on A, in A's local frame.
    pub anchor_a: [f32; 3],
    /// Anchor on B, in B's local frame.
    pub anchor_b: [f32; 3],
    /// Joint frame on A, as a quaternion.
    #[serde(default = "identity_quat", skip_serializing_if = "is_identity_quat")]
    pub frame_a: [f32; 4],
    /// Joint frame on B.
    #[serde(default = "identity_quat", skip_serializing_if = "is_identity_quat")]
    pub frame_b: [f32; 4],
    /// Which kind of joint it is.
    pub joint: JointKindEntry,
    /// Positional compliance, m/N.
    #[serde(default, skip_serializing_if = "is_zero")]
    pub compliance: f32,
    /// Angular compliance.
    #[serde(default, skip_serializing_if = "is_zero")]
    pub angular_compliance: f32,
    /// Whether the joint is active.
    #[serde(default = "yes", skip_serializing_if = "is_true")]
    pub enabled: bool,
    /// Whether the two bodies may collide with each other.
    #[serde(default, skip_serializing_if = "std::ops::Not::not")]
    pub collide_connected: bool,
}

fn yes() -> bool {
    true
}

/// One cloth sheet in a project file, stored as a description rather than
/// as particles.
#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
pub struct ClothEntry {
    /// Particles across.
    pub columns: usize,
    /// Particles down.
    pub rows: usize,
    /// Distance between neighbouring particles, metres.
    pub spacing: f32,
    /// Total mass of the sheet, kg.
    pub mass: f32,
    /// Centre position, metres.
    #[serde(default, skip_serializing_if = "is_zero3")]
    pub position: [f32; 3],
    /// Orientation, as a quaternion.
    #[serde(default = "identity_quat", skip_serializing_if = "is_identity_quat")]
    pub orientation: [f32; 4],
    /// Compliance along the weave, m/N.
    #[serde(default)]
    pub stretch_compliance: f32,
    /// Compliance along the diagonals, which is shear.
    #[serde(default)]
    pub shear_compliance: f32,
    /// Compliance against folding.
    #[serde(default)]
    pub bend_compliance: f32,
    /// Contact radius per particle, metres.
    pub thickness: f32,
}

impl From<&ClothDesc> for ClothEntry {
    fn from(d: &ClothDesc) -> Self {
        Self {
            columns: d.columns,
            rows: d.rows,
            spacing: d.spacing,
            mass: d.mass,
            position: d.pose.translation.to_array(),
            orientation: d.pose.rotation.to_array(),
            stretch_compliance: d.stretch_compliance,
            shear_compliance: d.shear_compliance,
            bend_compliance: d.bend_compliance,
            thickness: d.thickness,
        }
    }
}

impl From<&ClothEntry> for ClothDesc {
    fn from(e: &ClothEntry) -> Self {
        Self {
            columns: e.columns,
            rows: e.rows,
            spacing: e.spacing,
            mass: e.mass,
            pose: isometry(e.position, e.orientation),
            stretch_compliance: e.stretch_compliance,
            shear_compliance: e.shear_compliance,
            bend_compliance: e.bend_compliance,
            thickness: e.thickness,
        }
    }
}

/// One soft body in a project file.
#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
pub struct SoftBodyEntry {
    /// Full extents of the block, metres.
    pub size: [f32; 3],
    /// Particles along each axis.
    pub resolution: [usize; 3],
    /// Total mass, kg.
    pub mass: f32,
    /// Centre position, metres.
    #[serde(default, skip_serializing_if = "is_zero3")]
    pub position: [f32; 3],
    /// Orientation, as a quaternion.
    #[serde(default = "identity_quat", skip_serializing_if = "is_identity_quat")]
    pub orientation: [f32; 4],
    /// Lattice compliance, m/N.
    #[serde(default)]
    pub compliance: f32,
    /// Contact radius per particle, metres.
    pub thickness: f32,
}

impl From<&SoftBodyDesc> for SoftBodyEntry {
    fn from(d: &SoftBodyDesc) -> Self {
        Self {
            size: d.size.to_array(),
            resolution: [d.resolution.0, d.resolution.1, d.resolution.2],
            mass: d.mass,
            position: d.pose.translation.to_array(),
            orientation: d.pose.rotation.to_array(),
            compliance: d.compliance,
            thickness: d.thickness,
        }
    }
}

impl From<&SoftBodyEntry> for SoftBodyDesc {
    fn from(e: &SoftBodyEntry) -> Self {
        Self {
            size: Vec3::from_array(e.size),
            resolution: (e.resolution[0], e.resolution[1], e.resolution[2]),
            mass: e.mass,
            pose: isometry(e.position, e.orientation),
            compliance: e.compliance,
            thickness: e.thickness,
        }
    }
}

/// A body of fluid.
///
/// Nine numbers and a shape for any number of particles, which is the whole
/// argument for storing the scene rather than the state: `fill` regenerates
/// the lattice, and it regenerates the same one every time (see
/// `fusion_core::fill`).
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct FluidEntry {
    /// Shape the fluid fills.
    pub shape: ShapeEntry,
    /// Centre position, metres.
    #[serde(default, skip_serializing_if = "is_zero3")]
    pub position: [f32; 3],
    /// Orientation, as a quaternion.
    #[serde(default = "identity_quat", skip_serializing_if = "is_identity_quat")]
    pub orientation: [f32; 4],
    /// Initial velocity of every particle, m/s.
    #[serde(default, skip_serializing_if = "is_zero3")]
    pub velocity: [f32; 3],
    /// Rest density, kg/m^3.
    pub rest_density: f32,
    /// Distance between particles on the fill lattice, metres.
    pub spacing: f32,
    /// SPH smoothing radius, metres.
    pub smoothing_radius: f32,
    /// Constraint relaxation.
    #[serde(default)]
    pub relaxation: f32,
    /// XSPH viscosity.
    #[serde(default)]
    pub viscosity: f32,
    /// Vorticity confinement strength.
    #[serde(default)]
    pub vorticity: f32,
    /// Artificial-pressure strength, which looks like surface tension and is
    /// not derived from one.
    #[serde(default)]
    pub surface_tension: f32,
    /// Exponent of the artificial-pressure term.
    #[serde(default)]
    pub tension_exponent: f32,
    /// Reference distance for it, as a fraction of the smoothing radius.
    #[serde(default)]
    pub tension_reference: f32,
    /// Dynamic viscosity, Pa·s, for the Unified continuum. Absent, it maps the
    /// `viscosity` dial onto one, as it always has.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub dynamic_viscosity: Option<f32>,
}

/// Which constitutive law a solid is solved under, as a tagged value.
///
/// A tag rather than the enum itself so the file stays readable and stays
/// extensible: `model = "fixed_corotated"` in a `.fusion` file is something a
/// person can change by hand, and a law that grows a parameter grows a field
/// here rather than changing its representation.
#[derive(Clone, Copy, Debug, Default, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ConstitutiveEntry {
    /// Corotated elasticity.
    #[default]
    FixedCorotated,
    /// Compressible neo-Hookean elasticity.
    NeoHookean,
    /// Elastoplastic snow.
    Snow,
    /// Von Mises plasticity: metal. Its yield strain is the entry's
    /// `critical_stretch`.
    VonMises,
}

/// **One object.** PLAN.md §35, format 3.
///
/// A cube is an object; a jelly cube is the same object with a different
/// material; a puddle will be the same object with a fluid material. There is
/// no separate list for each, because the difference between them is not a
/// difference in what somebody is making.
///
/// ## Where the fields that only apply to one representation went
///
/// They are here, and they are `Option`, and that is deliberate rather than
/// lazy. The alternative — an enum with a variant per representation, so that
/// a rigid object cannot carry a sampling resolution — puts the thing the user
/// changes (the material) inside the thing that decides the type, so changing
/// jelly to steel would have to *destroy and rebuild* the object rather than
/// edit one field. It would also lose the resolution on the way through, so
/// changing it back would not come back to where it started.
///
/// A `None` here means "the default for whatever this material turns out to
/// be", and the defaults are resolved at build time by
/// [`Project::build_world`](crate::Project::build_world). Editing the material never invalidates the object.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct ObjectEntry {
    /// What the outliner calls it. Absent means the editor names it from its
    /// shape and index, which is what it did before objects had names.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub name: Option<String>,
    /// Its shape: a primitive, or the hull of an imported mesh.
    pub shape: ShapeEntry,
    /// Position, metres.
    #[serde(default)]
    pub position: [f32; 3],
    /// Orientation as `[x, y, z, w]`.
    #[serde(default = "identity_quat", skip_serializing_if = "is_identity_quat")]
    pub orientation: [f32; 4],
    /// Initial linear velocity, m/s. For a deformable object this is the
    /// velocity every material point starts with.
    #[serde(default, skip_serializing_if = "is_zero3")]
    pub linear_velocity: [f32; 3],
    /// Initial angular velocity, rad/s. Rigid only — a continuum has no single
    /// angular velocity, and giving one a spin is `fill` plus a velocity
    /// field, which is not something the scene format expresses yet.
    #[serde(default, skip_serializing_if = "is_zero3")]
    pub angular_velocity: [f32; 3],
    /// Name of a `[[material]]` entry. **The field that decides how this
    /// object is simulated** (§35).
    pub material: String,
    /// Whether the solver may move it. Rigid only.
    #[serde(default, skip_serializing_if = "BodyKindEntry::is_dynamic")]
    pub body_type: BodyKindEntry,
    /// Overrides the mass the shape and density imply. Rigid only.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub mass: Option<f32>,
    /// Distance between material points on the fill lattice, metres.
    /// Deformable only; absent is [`DEFAULT_RESOLUTION`] of the shape's
    /// bounding radius.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub resolution: Option<f32>,
    /// Background grid cell size, metres. Deformable only; absent is twice the
    /// resolution, which puts eight material points in a cell.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub cell_size: Option<f32>,
    /// Temperature it starts at, K: every material point of a deformable
    /// object, the whole of a rigid one. Absent is the material's own resting
    /// temperature. A rigid object's is read only when heat is on.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub temperature: Option<f32>,
    /// A behaviour script, in the language `fusion-script` defines. Rigid only
    /// — a script writes a pose, and a continuum does not have one.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub script: Option<String>,
    /// Reusable rigid-body behaviours, evaluated on the solver clock.
    #[serde(
        default,
        skip_serializing_if = "fusion_sim::control::Components::is_empty"
    )]
    pub components: fusion_sim::control::Components,
    /// What a rigid object does when it is loaded past its strength. Absent is
    /// stays whole.
    #[serde(default, skip_serializing_if = "FailureEntry::is_never")]
    pub failure: FailureEntry,
    /// Which way a rigid object's grain runs, in its own frame. Absent is
    /// along its longest side. Read only for a material with a grain.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub grain: Option<[f32; 3]>,
    /// Which tier a rigid object is solved in. Absent is Advanced: stress,
    /// breaking and temperature when the scene has them on. Basic is a plain
    /// rigid body, for the props that only need to fall and collide.
    #[serde(default, skip_serializing_if = "TierEntry::is_advanced")]
    pub tier: TierEntry,
}

/// Basic or Advanced, as a file spells it. See [`fusion_core::BodyDesc::advanced`].
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum TierEntry {
    /// A plain rigid body: gravity, contacts, joints.
    Basic,
    /// Everything the scene has on: stress, breaking, temperature.
    #[default]
    Advanced,
}

impl TierEntry {
    /// Whether it is the default.
    pub fn is_advanced(&self) -> bool {
        *self == Self::Advanced
    }
}

/// Sampling resolution a deformable object gets when its file does not say:
/// this fraction of the shape's bounding radius.
///
/// A sixth, which is **six material points across a cube whatever its size** —
/// 216 of them — because the bounding radius scales with the shape. That
/// matters more than the absolute number: switching a 3 m block to jelly
/// should not produce a million points, and switching a 10 cm one should not
/// produce eight.
///
/// Six across is coarse, and deliberately: it is interactive the instant
/// somebody changes a dropdown, and it is well clear of the under-sampled
/// lattice that comes apart into one clump per column (see
/// `fusion-core/tests/sampling.rs`). Anyone who wants the answer rather than
/// the feel sets `resolution` and pays for it.
pub const DEFAULT_RESOLUTION: f32 = 1.0 / 6.0;

impl ObjectEntry {
    /// An object of `material`, with everything else defaulted.
    pub fn new(shape: ShapeEntry, material: impl Into<String>) -> Self {
        Self {
            name: None,
            shape,
            position: [0.0; 3],
            orientation: identity_quat(),
            linear_velocity: [0.0; 3],
            angular_velocity: [0.0; 3],
            material: material.into(),
            body_type: BodyKindEntry::default(),
            mass: None,
            resolution: None,
            cell_size: None,
            temperature: None,
            script: None,
            components: Default::default(),
            failure: Default::default(),
            grain: None,
            tier: Default::default(),
        }
    }

    /// The sampling parameters a deformable object builds with.
    ///
    /// Resolved here rather than written into every file, so that a file which
    /// says nothing about resolution gets a sensible one for its own size —
    /// and so that scaling an object scales its sampling with it instead of
    /// leaving a 3 m block sampled for a 30 cm one.
    pub fn solid_params(&self, material: &SolidMaterial) -> SolidParams {
        let shape = Shape::from(self.shape.clone());
        let spacing = self
            .resolution
            .unwrap_or_else(|| shape.bounding_radius() * DEFAULT_RESOLUTION)
            .max(1e-4);
        SolidParams {
            spacing,
            cell_size: self.cell_size.unwrap_or(2.0 * spacing).max(spacing),
            material: *material,
            fibre: None,
        }
    }
}

/// A body of continuum, as a description rather than as material points.
///
/// The same trade §16 makes everywhere: a 0.3 m block at 2 cm spacing is 3375
/// points carrying two matrices each, and this is under two hundred bytes of
/// it.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct SolidEntry {
    /// Shape the solid fills.
    pub shape: ShapeEntry,
    /// Centre position, metres.
    #[serde(default, skip_serializing_if = "is_zero3")]
    pub position: [f32; 3],
    /// Orientation, as a quaternion.
    #[serde(default = "identity_quat", skip_serializing_if = "is_identity_quat")]
    pub orientation: [f32; 4],
    /// Initial velocity of every material point, m/s.
    #[serde(default, skip_serializing_if = "is_zero3")]
    pub velocity: [f32; 3],
    /// Distance between material points on the fill lattice, metres.
    pub spacing: f32,
    /// Background grid cell size, metres.
    pub cell_size: f32,
    /// Rest density, kg/m^3.
    pub density: f32,
    /// Young's modulus, Pa.
    pub youngs_modulus: f32,
    /// Poisson's ratio.
    pub poissons_ratio: f32,
    /// The constitutive law.
    #[serde(default)]
    pub model: ConstitutiveEntry,
    /// Compression past which snow yields, as a fraction.
    #[serde(default)]
    pub critical_compression: f32,
    /// Stretch past which snow yields, as a fraction.
    #[serde(default)]
    pub critical_stretch: f32,
    /// How sharply packing stiffens snow.
    #[serde(default)]
    pub hardening: f32,
    /// Temperature every point starts at, Kelvin (v0.9). Absent means the
    /// material's own natural temperature.
    #[serde(default)]
    pub temperature: Option<f32>,
    /// Specific heat capacity, J/(kg K).
    #[serde(default = "water_specific_heat")]
    pub specific_heat: f32,
    /// Thermal conductivity, W/(m K).
    #[serde(default)]
    pub conductivity: f32,
    /// Linear thermal expansion, 1/K.
    #[serde(default)]
    pub expansion: f32,
    /// Melting point, Kelvin. Absent or non-finite for a material that does
    /// not melt.
    #[serde(default)]
    pub melt_point: Option<f32>,
    /// Latent heat of fusion, J/kg.
    #[serde(default)]
    pub latent_heat: f32,
    /// Fraction of stiffness lost by the melting point.
    #[serde(default)]
    pub softening: f32,
    /// Heat generated per joule of plastic work.
    #[serde(default)]
    pub plastic_heating: f32,
    /// Surface emissivity, in `[0, 1]`.
    #[serde(default = "water_emissivity")]
    pub emissivity: f32,
    /// Tensile stretch at which damage begins. Absent means it never breaks.
    #[serde(default)]
    pub fracture_onset: Option<f32>,
    /// Tensile stretch at which it has fully parted.
    #[serde(default)]
    pub fracture_failure: f32,
}

fn water_specific_heat() -> f32 {
    fusion_core::Thermal::INERT.specific_heat
}

fn water_emissivity() -> f32 {
    fusion_core::Thermal::INERT.emissivity
}

impl From<&SolidMaterial> for DeformableEntry {
    fn from(m: &SolidMaterial) -> Self {
        let (model, critical_compression, critical_stretch, hardening) = match m.model {
            Constitutive::FixedCorotated => (ConstitutiveEntry::FixedCorotated, 0.0, 0.0, 0.0),
            Constitutive::NeoHookean => (ConstitutiveEntry::NeoHookean, 0.0, 0.0, 0.0),
            Constitutive::Snow {
                critical_compression,
                critical_stretch,
                hardening,
            } => (
                ConstitutiveEntry::Snow,
                critical_compression,
                critical_stretch,
                hardening,
            ),
            Constitutive::VonMises { yield_strain } => {
                (ConstitutiveEntry::VonMises, 0.0, yield_strain, 0.0)
            }
        };
        Self {
            youngs_modulus: m.youngs_modulus,
            poissons_ratio: m.poissons_ratio,
            model,
            critical_compression,
            critical_stretch,
            hardening,
            specific_heat: m.thermal.specific_heat,
            conductivity: m.thermal.conductivity,
            expansion: m.thermal.expansion,
            // `f32::INFINITY` has no TOML spelling, so "does not melt" is the
            // absence of the key rather than a magic number in it.
            melt_point: m
                .thermal
                .melt_point
                .is_finite()
                .then_some(m.thermal.melt_point),
            latent_heat: m.thermal.latent_heat,
            softening: m.thermal.softening,
            plastic_heating: m.thermal.plastic_heating,
            emissivity: m.thermal.emissivity,
            fracture_onset: m.fracture.breaks().then_some(m.fracture.onset),
            fracture_failure: m.fracture.failure,
            fibre: None,
        }
    }
}

impl From<&DeformableEntry> for SolidMaterial {
    /// Density is not here — it is [`MaterialEntry`]'s, because every
    /// representation has one — so this leaves the default in place and
    /// [`MaterialEntry::solid_part`] fills it in.
    fn from(e: &DeformableEntry) -> Self {
        Self {
            density: SolidMaterial::JELLY.density,
            youngs_modulus: e.youngs_modulus,
            poissons_ratio: e.poissons_ratio,
            thermal: fusion_core::Thermal {
                specific_heat: e.specific_heat,
                conductivity: e.conductivity,
                expansion: e.expansion,
                melt_point: e.melt_point.unwrap_or(f32::INFINITY),
                latent_heat: e.latent_heat,
                softening: e.softening,
                plastic_heating: e.plastic_heating,
                emissivity: e.emissivity,
            },
            fracture: match e.fracture_onset {
                Some(onset) => fusion_core::Fracture {
                    onset,
                    failure: e.fracture_failure,
                },
                None => fusion_core::Fracture::TOUGH,
            },
            model: match e.model {
                ConstitutiveEntry::FixedCorotated => Constitutive::FixedCorotated,
                ConstitutiveEntry::NeoHookean => Constitutive::NeoHookean,
                ConstitutiveEntry::Snow => Constitutive::Snow {
                    critical_compression: e.critical_compression,
                    critical_stretch: e.critical_stretch,
                    hardening: e.hardening,
                },
                ConstitutiveEntry::VonMises => Constitutive::VonMises {
                    yield_strain: e.critical_stretch,
                },
            },
        }
    }
}

impl From<&SolidDesc> for SolidEntry {
    fn from(d: &SolidDesc) -> Self {
        let m = d.params.material;
        let (model, critical_compression, critical_stretch, hardening) = match m.model {
            Constitutive::FixedCorotated => (ConstitutiveEntry::FixedCorotated, 0.0, 0.0, 0.0),
            Constitutive::NeoHookean => (ConstitutiveEntry::NeoHookean, 0.0, 0.0, 0.0),
            Constitutive::Snow {
                critical_compression,
                critical_stretch,
                hardening,
            } => (
                ConstitutiveEntry::Snow,
                critical_compression,
                critical_stretch,
                hardening,
            ),
            Constitutive::VonMises { yield_strain } => {
                (ConstitutiveEntry::VonMises, 0.0, yield_strain, 0.0)
            }
        };
        Self {
            shape: ShapeEntry::from(&d.shape),
            position: d.pose.translation.to_array(),
            orientation: d.pose.rotation.to_array(),
            velocity: d.velocity.to_array(),
            spacing: d.params.spacing,
            cell_size: d.params.cell_size,
            density: m.density,
            youngs_modulus: m.youngs_modulus,
            poissons_ratio: m.poissons_ratio,
            model,
            critical_compression,
            critical_stretch,
            hardening,
            temperature: d.temperature,
            specific_heat: m.thermal.specific_heat,
            conductivity: m.thermal.conductivity,
            expansion: m.thermal.expansion,
            // `f32::INFINITY` has no TOML spelling, so "does not melt" is the
            // absence of the key rather than a magic number in it.
            melt_point: m
                .thermal
                .melt_point
                .is_finite()
                .then_some(m.thermal.melt_point),
            latent_heat: m.thermal.latent_heat,
            softening: m.thermal.softening,
            plastic_heating: m.thermal.plastic_heating,
            emissivity: m.thermal.emissivity,
            fracture_onset: m.fracture.breaks().then_some(m.fracture.onset),
            fracture_failure: m.fracture.failure,
        }
    }
}

impl From<&SolidEntry> for SolidDesc {
    fn from(e: &SolidEntry) -> Self {
        Self {
            shape: Shape::from(e.shape.clone()),
            pose: isometry(e.position, e.orientation),
            velocity: Vec3::from_array(e.velocity),
            temperature: e.temperature,
            params: SolidParams {
                spacing: e.spacing,
                cell_size: e.cell_size,
                material: SolidMaterial {
                    density: e.density,
                    youngs_modulus: e.youngs_modulus,
                    poissons_ratio: e.poissons_ratio,
                    thermal: fusion_core::Thermal {
                        specific_heat: e.specific_heat,
                        conductivity: e.conductivity,
                        expansion: e.expansion,
                        melt_point: e.melt_point.unwrap_or(f32::INFINITY),
                        latent_heat: e.latent_heat,
                        softening: e.softening,
                        plastic_heating: e.plastic_heating,
                        emissivity: e.emissivity,
                    },
                    fracture: match e.fracture_onset {
                        Some(onset) => fusion_core::Fracture {
                            onset,
                            failure: e.fracture_failure,
                        },
                        None => fusion_core::Fracture::TOUGH,
                    },
                    model: match e.model {
                        ConstitutiveEntry::FixedCorotated => Constitutive::FixedCorotated,
                        ConstitutiveEntry::NeoHookean => Constitutive::NeoHookean,
                        ConstitutiveEntry::Snow => Constitutive::Snow {
                            critical_compression: e.critical_compression,
                            critical_stretch: e.critical_stretch,
                            hardening: e.hardening,
                        },
                        ConstitutiveEntry::VonMises => Constitutive::VonMises {
                            yield_strain: e.critical_stretch,
                        },
                    },
                },
                fibre: None,
            },
        }
    }
}

impl From<&FluidDesc> for FluidEntry {
    fn from(d: &FluidDesc) -> Self {
        Self {
            shape: ShapeEntry::from(&d.shape),
            position: d.pose.translation.to_array(),
            orientation: d.pose.rotation.to_array(),
            velocity: d.velocity.to_array(),
            rest_density: d.params.rest_density,
            spacing: d.params.spacing,
            smoothing_radius: d.params.smoothing_radius,
            relaxation: d.params.relaxation,
            viscosity: d.params.viscosity,
            vorticity: d.params.vorticity,
            surface_tension: d.params.surface_tension,
            tension_exponent: d.params.tension_exponent,
            tension_reference: d.params.tension_reference,
            dynamic_viscosity: d.params.dynamic_viscosity,
        }
    }
}

impl From<&FluidEntry> for FluidDesc {
    fn from(e: &FluidEntry) -> Self {
        Self {
            shape: Shape::from(e.shape.clone()),
            pose: isometry(e.position, e.orientation),
            velocity: Vec3::from_array(e.velocity),
            params: FluidParams {
                rest_density: e.rest_density,
                spacing: e.spacing,
                smoothing_radius: e.smoothing_radius,
                relaxation: e.relaxation,
                viscosity: e.viscosity,
                vorticity: e.vorticity,
                surface_tension: e.surface_tension,
                tension_exponent: e.tension_exponent,
                tension_reference: e.tension_reference,
                dynamic_viscosity: e.dynamic_viscosity,
            },
        }
    }
}

// ---------------------------------------------------------------- conversion

fn isometry(position: [f32; 3], orientation: [f32; 4]) -> Isometry {
    Isometry::new(
        Vec3::from_array(position),
        Quat::from_xyzw(
            orientation[0],
            orientation[1],
            orientation[2],
            orientation[3],
        ),
    )
}

/// Collect a world's bodies, interning their materials by name.
///
/// Materials are deduplicated by value and named after the library entry they
/// match, falling back to `material_N`. A file where forty boxes all say
/// `material = "wood"` is one somebody can edit; forty copies of three
/// floating-point numbers is one they can only regenerate.
/// Every object in a world, with the materials they share (§35).
///
/// Rigid bodies first and then solids, which is the order `World` builds them
/// in and therefore the order that reproduces the run it came from.
pub fn objects_from_world(world: &World) -> (Vec<MaterialEntry>, Vec<ObjectEntry>) {
    let mut materials: Vec<MaterialEntry> = Vec::new();
    let mut objects = Vec::new();

    for desc in world.scene() {
        let material = intern_material(&mut materials, &desc.material);
        objects.push(ObjectEntry {
            name: None,
            shape: ShapeEntry::from(&desc.shape),
            position: desc.position.to_array(),
            orientation: desc.orientation.to_array(),
            linear_velocity: desc.linear_velocity.to_array(),
            angular_velocity: desc.angular_velocity.to_array(),
            material,
            body_type: match desc.body_type {
                BodyType::Static => BodyKindEntry::Static,
                BodyType::Kinematic => BodyKindEntry::Kinematic,
                BodyType::Dynamic => BodyKindEntry::Dynamic,
            },
            mass: desc.mass,
            resolution: None,
            cell_size: None,
            temperature: desc.temperature,
            // A `World` carries no scripts — they live on the `Simulation`
            // that runs it, because a script is a thing that *runs* rather
            // than state the solver reads.
            script: None,
            components: Default::default(),
            failure: desc.failure.into(),
            grain: desc.grain.map(|g| g.to_array()),
            tier: if desc.advanced {
                TierEntry::Advanced
            } else {
                TierEntry::Basic
            },
        });
    }

    for desc in world.solid_scene() {
        let material = intern_solid_material(&mut materials, &desc.params.material);
        objects.push(ObjectEntry {
            name: None,
            shape: ShapeEntry::from(&desc.shape),
            position: desc.pose.translation.to_array(),
            orientation: desc.pose.rotation.to_array(),
            linear_velocity: desc.velocity.to_array(),
            angular_velocity: [0.0; 3],
            material,
            body_type: BodyKindEntry::Dynamic,
            mass: None,
            // Stated rather than left to the default: the world it came from
            // was sampled at these, and a round trip that re-resolved them
            // would hand back a different number of material points.
            resolution: Some(desc.params.spacing),
            cell_size: Some(desc.params.cell_size),
            temperature: desc.temperature,
            script: None,
            components: Default::default(),
            failure: Default::default(),
            grain: None,
            tier: Default::default(),
        });
    }

    (materials, objects)
}

/// Names for the library materials, so a saved scene reads as `"wood"` rather
/// than `"material_3"` when it is in fact wood.
const NAMED: &[(&str, Material)] = &[
    ("generic", Material::GENERIC),
    ("steel", Material::STEEL),
    ("aluminium", Material::ALUMINIUM),
    ("concrete", Material::CONCRETE),
    ("wood", Material::WOOD),
    ("rubber", Material::RUBBER),
    ("ice", Material::ICE),
    ("glass", Material::GLASS),
    ("stone", Material::STONE),
    ("water", Material::WATER),
];

fn intern_material(into: &mut Vec<MaterialEntry>, material: &Material) -> String {
    if let Some(existing) = into
        .iter()
        .find(|m| m.representation == Representation::Rigid && m.rigid_part() == *material)
    {
        return existing.name.clone();
    }
    let name = NAMED
        .iter()
        .find(|(_, m)| m == material)
        .map(|(name, _)| (*name).to_string())
        .unwrap_or_else(|| format!("material_{}", into.len()));
    into.push(MaterialEntry::rigid(name.clone(), material));
    name
}

/// The same, for a continuum material.
///
/// Two materials with the same *rigid* half are not the same material once one
/// of them is jelly, so this matches on the whole of it. Named from
/// [`SolidMaterial::LIBRARY`] where it can, for the same reason the rigid one
/// is: a saved scene should say `"jelly"` when it is in fact jelly.
pub(crate) fn intern_solid_material(
    into: &mut Vec<MaterialEntry>,
    material: &SolidMaterial,
) -> String {
    let deformable = DeformableEntry::from(material);
    if let Some(existing) = into.iter().find(|m| {
        m.representation == Representation::Deformable
            && m.density == material.density
            && m.deformable == Some(deformable)
    }) {
        return existing.name.clone();
    }
    let name = SolidMaterial::LIBRARY
        .iter()
        .find(|(_, m)| m == material)
        .map(|(name, _)| name.to_lowercase())
        .unwrap_or_else(|| format!("material_{}", into.len()));
    // A name a rigid material already has would collide; two different
    // materials sharing one name is a scene that builds the wrong thing, and
    // "rubber" is in both libraries.
    let name = if into.iter().any(|m| m.name == name) {
        format!("{name}_deformable")
    } else {
        name
    };
    into.push(MaterialEntry {
        name: name.clone(),
        representation: Representation::Deformable,
        density: material.density,
        // A continuum still rubs against rigid geometry, and the MPM solver
        // has one friction coefficient for all of it — see `MpmSolver::friction`
        // — so these are the surface's properties rather than dead fields.
        friction: 0.4,
        restitution: 0.0,
        look: Appearance::default(),
        deformable: Some(deformable),
        thermal: None,
        strength: None,
    });
    name
}

pub fn joints_from_world(world: &World) -> Vec<JointEntry> {
    world
        .joints()
        .iter()
        .map(|j| JointEntry {
            body_a: j.body_a.0,
            body_b: j.body_b.0,
            anchor_a: j.local_anchor_a.to_array(),
            anchor_b: j.local_anchor_b.to_array(),
            frame_a: j.local_frame_a.to_array(),
            frame_b: j.local_frame_b.to_array(),
            joint: match j.kind {
                JointKind::Ball => JointKindEntry::Ball,
                JointKind::Fixed => JointKindEntry::Fixed,
                JointKind::Hinge => JointKindEntry::Hinge,
                JointKind::Distance { min, max } => JointKindEntry::Distance { min, max },
                JointKind::Slider { min, max } => JointKindEntry::Slider { min, max },
            },
            compliance: j.compliance,
            angular_compliance: j.angular_compliance,
            enabled: j.enabled,
            collide_connected: j.collide_connected,
        })
        .collect()
}

/// The named material, or the error that says which object wanted it.
fn find_material<'a>(
    materials: &'a [MaterialEntry],
    name: &str,
    object: usize,
) -> Result<&'a MaterialEntry, ProjectError> {
    materials.iter().find(|m| m.name == name).ok_or_else(|| {
        ProjectError::Unresolved(format!(
            "object {object} is made of \"{name}\", which has no [[material]] entry"
        ))
    })
}

/// A rigid object's body description.
pub fn object_body_desc(entry: &ObjectEntry, material: &MaterialEntry) -> BodyDesc {
    let mut desc = BodyDesc::new(Shape::from(entry.shape.clone()))
        .at(Vec3::from_array(entry.position))
        .rotated(Quat::from_xyzw(
            entry.orientation[0],
            entry.orientation[1],
            entry.orientation[2],
            entry.orientation[3],
        ))
        .with_velocity(Vec3::from_array(entry.linear_velocity))
        .with_angular_velocity(Vec3::from_array(entry.angular_velocity))
        .with_material(material.rigid_part());
    desc = match entry.body_type {
        BodyKindEntry::Dynamic => desc,
        BodyKindEntry::Static => desc.fixed(),
        BodyKindEntry::Kinematic => desc.driven(),
    };
    if let Some(mass) = entry.mass {
        desc = desc.with_mass(mass);
    }
    if let Some(kelvin) = entry.temperature {
        desc = desc.with_temperature(kelvin);
    }
    let desc = desc.with_failure(entry.failure.into());
    let desc = match entry.grain {
        Some(g) => desc.with_grain(Vec3::from_array(g)),
        None => desc,
    };
    match entry.tier {
        TierEntry::Basic => desc.basic(),
        TierEntry::Advanced => desc,
    }
}

/// A deformable object's solid description.
pub fn object_solid_desc(entry: &ObjectEntry, material: &MaterialEntry) -> SolidDesc {
    let solid = material.solid_part();
    let shape = Shape::from(entry.shape.clone());
    let pose = isometry(entry.position, entry.orientation);
    let mut params = entry.solid_params(&solid);
    // Fibres along the object's grain, or its longest side, turned with the
    // object into the frame the material is sampled in.
    params.fibre = material
        .deformable
        .and_then(|d| d.fibre)
        .filter(|k| *k > 0.0)
        .map(|stiffness| {
            let local = entry
                .grain
                .map(Vec3::from_array)
                .filter(|g| g.length_squared() > 1e-8)
                .map(|g| g.normalize())
                .unwrap_or_else(|| fusion_core::body::default_grain(&shape));
            fusion_core::Fibre {
                direction: pose.rotation.rotate(local).normalize_or_zero(),
                stiffness,
            }
        });
    SolidDesc {
        shape,
        pose,
        velocity: Vec3::from_array(entry.linear_velocity),
        temperature: entry.temperature,
        params,
    }
}

/// One body entry as the descriptor that builds it.
///
/// Public, and that is the point. A caller that wants to add a single body to
/// a world that is already running — the sandbox, when you press "+ box" while
/// the simulation is playing — has to produce exactly the body a rebuild would
/// have produced, or the two paths drift and a scene stops meaning one thing.
/// §21's rule is that there is one path from scene data to a world; this is
/// that path, factored out so there can go on being one.
pub fn body_desc(entry: &BodyEntry, materials: &[MaterialEntry]) -> Result<BodyDesc, ProjectError> {
    let material = materials
        .iter()
        .find(|m| m.name == entry.material)
        .ok_or_else(|| {
            ProjectError::Unresolved(format!(
                "material \"{}\" is used by a body but has no [[material]] entry",
                entry.material
            ))
        })?;
    let mut desc = BodyDesc::new(Shape::from(entry.shape.clone()))
        .at(Vec3::from_array(entry.position))
        .rotated(Quat::from_xyzw(
            entry.orientation[0],
            entry.orientation[1],
            entry.orientation[2],
            entry.orientation[3],
        ))
        .with_velocity(Vec3::from_array(entry.linear_velocity))
        .with_angular_velocity(Vec3::from_array(entry.angular_velocity))
        .with_material(material.rigid_part());
    desc = match entry.body_type {
        BodyKindEntry::Dynamic => desc,
        BodyKindEntry::Static => desc.fixed(),
        BodyKindEntry::Kinematic => desc.driven(),
    };
    if let Some(mass) = entry.mass {
        desc = desc.with_mass(mass);
    }
    Ok(desc)
}

/// Build the world a project describes.
pub fn world_from_project(project: &crate::Project) -> Result<World, ProjectError> {
    let mut names = std::collections::HashSet::new();
    for material in &project.materials {
        if material.name.trim().is_empty() || !names.insert(&material.name) {
            return Err(ProjectError::Unresolved(format!(
                "Empty or duplicate material name: {}",
                material.name
            )));
        }
        material.validate().map_err(ProjectError::Unresolved)?;
    }

    let mut world = World::new()
        .with_gravity(Vec3::from_array(project.world.gravity))
        .with_seed(project.world.seed);
    world.linear_damping = project.world.linear_damping;
    world.angular_damping = project.world.angular_damping;
    world.domain = fusion_core::Domain {
        origin: fusion_math::DVec3::from_array(project.world.domain.origin),
        length_scale: project.world.domain.length_scale,
        time_scale: project.world.domain.time_scale,
        mass_scale: project.world.domain.mass_scale,
    };

    // The objects, in file order, dispatched on what their material says they
    // are (§35). File order rather than "rigid ones first": the order bodies
    // are added in is part of the answer — it decides the sequence the
    // Gauss-Seidel contact solve visits them in — so it has to be a property
    // of the file rather than of how this loop happens to be written.
    //
    // `handles` maps object index to body index, because they are not the same
    // once a deformable object sits between two rigid ones, and joints refer
    // to objects.
    let mut handles: Vec<Option<u32>> = Vec::with_capacity(project.objects.len());
    for (index, entry) in project.objects.iter().enumerate() {
        let material = find_material(&project.materials, &entry.material, index)?;
        match material.representation {
            Representation::Rigid => {
                handles.push(Some(world.len() as u32));
                world.add(object_body_desc(entry, material));
            }
            Representation::Deformable => {
                handles.push(None);
                world.add_solid(&object_solid_desc(entry, material));
            }
        }
    }

    // Format 2's separate lists, still read. A file that has them is a file
    // written before objects existed, and it still describes a scene.
    let legacy_base = world.len() as u32;
    for entry in &project.bodies {
        world.add(body_desc(entry, &project.materials)?);
    }

    let count = world.len() as u32;
    for entry in &project.joints {
        // A joint's indices name *objects* in a format-3 file and bodies in a
        // format-2 one, and the two agree whenever every object is rigid —
        // which is every format-2 file, by construction, since it had no
        // objects at all. `resolve` is the one place that difference lives.
        let resolve = |index: u32| -> Result<u32, ProjectError> {
            match handles.get(index as usize) {
                Some(Some(body)) => Ok(*body),
                Some(None) => Err(ProjectError::Unresolved(format!(
                    "joint refers to object {index}, which is deformable — a \
                     joint connects two rigid bodies, and a continuum has no \
                     single pose to attach one to"
                ))),
                None => {
                    let legacy = legacy_base + (index - handles.len() as u32);
                    if legacy < count {
                        Ok(legacy)
                    } else {
                        Err(ProjectError::Unresolved(format!(
                            "joint refers to body {index}, but the scene has {count}"
                        )))
                    }
                }
            }
        };
        let (body_a, body_b) = (resolve(entry.body_a)?, resolve(entry.body_b)?);
        let mut joint = Joint::new(
            fusion_core::BodyHandle(body_a),
            fusion_core::BodyHandle(body_b),
            match entry.joint {
                JointKindEntry::Ball => JointKind::Ball,
                JointKindEntry::Fixed => JointKind::Fixed,
                JointKindEntry::Hinge => JointKind::Hinge,
                JointKindEntry::Distance { min, max } => JointKind::Distance { min, max },
                JointKindEntry::Slider { min, max } => JointKind::Slider { min, max },
            },
        );
        joint.local_anchor_a = Vec3::from_array(entry.anchor_a);
        joint.local_anchor_b = Vec3::from_array(entry.anchor_b);
        joint.local_frame_a = Quat::from_xyzw(
            entry.frame_a[0],
            entry.frame_a[1],
            entry.frame_a[2],
            entry.frame_a[3],
        );
        joint.local_frame_b = Quat::from_xyzw(
            entry.frame_b[0],
            entry.frame_b[1],
            entry.frame_b[2],
            entry.frame_b[3],
        );
        joint.compliance = entry.compliance;
        joint.angular_compliance = entry.angular_compliance;
        joint.enabled = entry.enabled;
        joint.collide_connected = entry.collide_connected;
        world.add_joint(joint);
    }

    // Order matters and is the file's order: cloth, then soft bodies, then
    // fluids. Particle indices follow from it, and a scene that built them in
    // a different order would be a different scene — same picture, different
    // constraint numbering, different floating-point sum.
    for entry in &project.cloths {
        world.add_cloth(&ClothDesc::from(entry));
    }
    for entry in &project.soft_bodies {
        world.add_soft_body(&SoftBodyDesc::from(entry));
    }
    for entry in &project.fluids {
        world.add_fluid(&FluidDesc::from(entry));
    }
    for entry in &project.solids {
        world.add_solid(&SolidDesc::from(entry));
    }

    Ok(world)
}

/// Unused today; kept so the `vec3` import documents the coordinate
/// convention the file format uses.
#[allow(dead_code)]
fn _convention() -> Vec3 {
    vec3(0.0, 1.0, 0.0)
}
