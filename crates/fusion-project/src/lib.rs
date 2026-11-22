//! Fusion L5 — the project format.
//!
//! PLAN.md §22 defines a project as the tuple that makes a run reproducible:
//!
//! ```text
//! S = (initial_state, parameters, models, solvers, software_version, seed)
//! ```
//!
//! and states what it is for: **the project file is also the bug report.** A
//! `.fusion` file plus a step number reproduces anything exactly, and every
//! bug fixed becomes a regression test file. That is the design goal this
//! crate is measured against, and [`Project::verify`] is it stated as code —
//! save a run, load it, step it forward, and check the state hash matches.
//!
//! ## Why the scene and not the state
//!
//! A project stores the *scene* — the declarative description of the world at
//! t = 0 — and never the state. PLAN.md §16 draws that line and this is what
//! it buys: a dam break of forty thousand particles is one `[[fluid]]` table
//! of nine numbers, because `fill` regenerates the particles from the shape.
//! Storing the state would be megabytes, would need binary blobs, and would
//! not be readable or diffable, which is most of what makes a file useful as
//! a bug report.
//!
//! §22 anticipates blobs for "meshes, point data and large initial fields".
//! None of those exist yet — Fusion has no triangle meshes (§8) and no
//! imported fields — so format v1 is a single self-contained text file and
//! says so in its own header. The blob reference arrives with the first thing
//! that needs one.
//!
//! ## Why TOML, and why a hand-written schema
//!
//! §22 says "TOML or JSON — avoid YAML's footguns for anything
//! reproducibility-sensitive". TOML, because a project file is meant to be
//! read and edited by a person.
//!
//! The schema types here are written out rather than derived from
//! [`fusion_core`]'s. That is more code, and it is the difference between a
//! file format and an accident: derived serialization makes every internal
//! field rename a silent format break, and the one thing a format promising
//! to reproduce a run cannot do is change its mind about what a field means.
//! The conversion functions are the place where a v1 file meets whatever the
//! engine currently calls things.
//!
//! ```
//! use fusion_core::{BodyDesc, World};
//! use fusion_math::{vec3, Vec3};
//! use fusion_project::Project;
//! use fusion_sim::Simulation;
//!
//! let mut world = World::new();
//! world.add(BodyDesc::cuboid(vec3(20.0, 1.0, 20.0)).at(vec3(0.0, -0.5, 0.0)).fixed());
//! world.add(BodyDesc::cuboid(Vec3::ONE).at(vec3(0.0, 3.0, 0.0)));
//!
//! let sim = Simulation::new(world);
//! let project = Project::from_simulation(&sim).with_note("a box, falling");
//!
//! // A few kilobytes of TOML, whatever the particle count.
//! let text = project.to_toml().unwrap();
//! assert!(text.contains("[[object]]"));
//!
//! // And it reproduces the run it describes, which is the whole claim.
//! let report = project.verify(200).unwrap();
//! assert!(report.reproduces(), "{report}");
//! ```

#![deny(unsafe_code)]
// The public API freeze (PLAN.md §29, `docs/API.md`): an undocumented public
// item is one nobody agreed to freeze.
#![deny(missing_docs)]

mod schema;

use std::fmt;
use std::path::Path;

use fusion_core::World;
use fusion_sim::Simulation;

pub mod mesh;
pub mod pack;
pub use pack::MaterialPack;

pub use mesh::{hull_of, parse_obj_points, read_obj_hull, read_obj_points, MeshError};
pub use schema::{
    body_desc, object_body_desc, object_solid_desc, Appearance, BodyEntry, BodyKindEntry,
    ClothEntry, ConstitutiveEntry, ContinuumEntry, CoreConfig, CoreIntegratorEntry,
    CreationDefaults, DeformableEntry, FailureEntry, FluidEntry, FluidModelEntry, GrainEntry,
    HeatConfig, JointEntry, JointKindEntry, MaterialEntry, Metadata, ObjectEntry, PatternEntry,
    Provenance, Representation, RunConfig, ShapeEntry, SoftBodyEntry, SolidEntry, SolverConfig,
    SolverKind, StateRef, StrengthEntry, StressConfig, ThermalEntry, TierEntry, WorldConfig,
    DEFAULT_RESOLUTION,
};

/// The format version written into every file.
///
/// Bumped when a change would make an older reader misinterpret a newer file,
/// which is not the same as when a field is added — a reader that ignores an
/// unknown field is still correct, and a reader that misreads a changed one is
/// not.
///
/// **2** is v1.0's, and [`StateRef`] is why it is a bump rather than a field.
/// Every other addition to this format has been safe to ignore: a v1 reader
/// meeting a v2 scene it does not fully understand still loads a scene. A v1
/// reader meeting a manifest with `[state]` in it would ignore the one line
/// that says "this is a run at step 4000, not a scene at t = 0", start from
/// the beginning, and produce a plausible run that is not the one in the file.
/// That is the exact case the rule above is about.
///
/// **4** adds named input takes and physical components. Older readers must
/// refuse these files rather than silently omit the forces. Existing v3 run
/// manifests retain their version so their scene fingerprints remain valid.
///
/// **5** adds the take archive. A format-4 reader meeting one would load
/// `[[input]]` — the live take — and quietly drop every other recording in the
/// file, so the first save that stores a second take is a bump. A scene with
/// one take still writes as 4 and still opens in a v4 editor.
///
/// **6** adds temperature in rigid bodies. A format-5 reader meeting a scene
/// with `[solver.heat]` switched on would drop the table, run the scene cold,
/// and report a steel block that slid a hundred metres as still at room
/// temperature — a different run, quietly. A scene with heat off writes no
/// heat table, and a run manifest saved by an older build keeps its version
/// and its fingerprint.
///
/// **7** adds stress in rigid bodies and failure: objects that break into
/// pieces or deform and melt. A format-6 reader would drop `failure` from
/// every object and run a scene where the glass is supposed to shatter with
/// glass that cannot, so a scene that uses either is a bump. Materials gain a
/// `[material.strength]` table, which an older reader ignores harmlessly —
/// nothing it runs reads it.
///
/// **8** adds the unified continuum solver, `[solver] continuum =
/// "unified"`, and fibres in deformable materials, which only it reads. A
/// format-7 reader would run either on the classic solver.
///
/// **9** adds the per-object tier, `tier = "basic"`: a plain rigid body,
/// with none of stress, breaking or temperature. A format-8 reader would
/// solve the object Advanced.
///
/// **10** adds the incompressible fluid model, `[solver] fluid_model =
/// "incompressible"`, and implicit stiff solids, `implicit_solids = true`. A
/// format-9 reader would give the fluid its pressure from compression
/// instead, and take substeps for the solids.
///
/// **11** adds the choice of engine, `[solver] kind = "core"` for Fusion
/// Core. A format-10 reader does not know the word and would refuse the file
/// with a parse error rather than a version message, so a Core scene always
/// writes 11.
pub const FORMAT_VERSION: u32 = 11;

/// A complete, reproducible description of a run.
#[derive(Clone, Debug, PartialEq, serde::Serialize, serde::Deserialize)]
pub struct Project {
    /// Which Fusion wrote this file, and which format version it speaks.
    pub fusion: Provenance,
    /// Timestep and output rate: the clock the run was made on.
    #[serde(default)]
    pub run: RunConfig,
    /// Every solver dial, recorded so a run reproduces rather than merely
    /// resembles.
    #[serde(default)]
    pub solver: SolverConfig,
    /// Gravity, damping and the seed.
    #[serde(default)]
    pub world: WorldConfig,
    /// Named materials, so a scene can say `material = "wood"` once rather
    /// than repeating three numbers per body. §22 asks for "material
    /// definitions and references"; this is both halves.
    #[serde(default, rename = "material", skip_serializing_if = "Vec::is_empty")]
    pub materials: Vec<MaterialEntry>,
    /// **The scene** (§35, format 3): one list, one kind of entry, and a
    /// material on each that decides how it is simulated.
    #[serde(default, rename = "object", skip_serializing_if = "Vec::is_empty")]
    pub objects: Vec<ObjectEntry>,
    /// Rigid bodies, format 2's separate list.
    ///
    /// Still read, never written. [`Project::unify`] folds them into
    /// [`Project::objects`], and `load` does that on the way in, so a
    /// format-2 file becomes a format-3 project as it is opened and nothing
    /// downstream has to know which it was.
    #[serde(default, rename = "body", skip_serializing_if = "Vec::is_empty")]
    pub bodies: Vec<BodyEntry>,
    /// Joints, which refer to bodies by position in `bodies`.
    #[serde(default, rename = "joint", skip_serializing_if = "Vec::is_empty")]
    pub joints: Vec<JointEntry>,
    /// Cloth sheets, as descriptions rather than as particles.
    #[serde(default, rename = "cloth", skip_serializing_if = "Vec::is_empty")]
    pub cloths: Vec<ClothEntry>,
    /// Soft bodies, likewise.
    #[serde(default, rename = "soft_body", skip_serializing_if = "Vec::is_empty")]
    pub soft_bodies: Vec<SoftBodyEntry>,
    /// Bodies of fluid, likewise — which is why 22 000 particles are 1.6 KB.
    #[serde(default, rename = "fluid", skip_serializing_if = "Vec::is_empty")]
    pub fluids: Vec<FluidEntry>,
    /// Bodies of continuum, likewise (v0.8).
    #[serde(default, rename = "solid", skip_serializing_if = "Vec::is_empty")]
    pub solids: Vec<SolidEntry>,
    /// Free-form experiment notes: sweep parameters, a run ID, what was being
    /// investigated. §22 asks for it and deliberately does not say what goes
    /// in it.
    #[serde(default, skip_serializing_if = "Metadata::is_empty")]
    pub metadata: Metadata,
    /// Starting values for newly created objects; existing physics is unchanged.
    #[serde(default, skip_serializing_if = "CreationDefaults::is_default")]
    pub creation: CreationDefaults,
    /// Named controls and their recorded solver-step events (format 4).
    ///
    /// The events here are the *live* take: the one the solver samples and the
    /// one recording writes. [`Project::takes`] is the archive around it.
    #[serde(default, rename = "input", skip_serializing_if = "Vec::is_empty")]
    pub inputs: Vec<fusion_sim::control::InputAction>,
    /// Stored takes, including the live one (format 5).
    ///
    /// `takes[active_take]` and [`Project::inputs`] hold the same events; the
    /// duplication buys every existing reader of `inputs` staying correct, and
    /// is maintained at the one place a take changes.
    #[serde(default, rename = "take", skip_serializing_if = "Vec::is_empty")]
    pub takes: Vec<fusion_sim::control::Take>,
    /// Which stored take is live. Out-of-range is corrected on load.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub active_take: Option<usize>,
    /// Where the state lives, when this manifest describes a run in progress
    /// rather than a scene (v1.0, format 2).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub state: Option<StateRef>,
}

/// What went wrong loading or building a project.
#[derive(Debug)]
pub enum ProjectError {
    /// The file could not be read or written.
    Io(std::io::Error),
    /// The file is not valid TOML, or does not match the schema.
    Parse(toml::de::Error),
    /// The project could not be turned into TOML.
    Serialize(toml::ser::Error),
    /// A file written by a version of the format this build cannot read.
    UnsupportedFormat {
        /// Version found in the file.
        found: u32,
        /// Newest version this build understands.
        supported: u32,
    },
    /// A reference — a material name, a body index — that does not resolve.
    Unresolved(String),
    /// The mode the project asks for is not implemented.
    Mode(&'static str),
    /// The state blob beside the manifest could not be decoded.
    State(fusion_sim::StateError),
    /// The state blob decoded, and is not the state this manifest claims.
    ///
    /// Its own error rather than a warning, because what it catches is a run
    /// that would otherwise *start*: an old checkpoint left beside an edited
    /// scene loads cleanly and continues from a state that never existed.
    StaleState {
        /// What disagrees — "size" or "fingerprint".
        what: &'static str,
        /// What the manifest says.
        expected: u64,
        /// What the blob actually is.
        found: u64,
    },
}

impl fmt::Display for ProjectError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Io(e) => write!(f, "{e}"),
            Self::Parse(e) => write!(f, "not a valid .fusion file: {e}"),
            Self::Serialize(e) => write!(f, "could not write project: {e}"),
            Self::UnsupportedFormat { found, supported } => write!(
                f,
                "project uses format version {found}; this build reads {supported}. \
                 A newer file is not readable by an older Fusion, which is the \
                 point of the version — reading it anyway would reproduce the \
                 wrong run and say nothing."
            ),
            Self::Unresolved(what) => write!(f, "unresolved reference: {what}"),
            Self::Mode(m) => write!(f, "{m}"),
            Self::State(e) => write!(f, "{e}"),
            Self::StaleState {
                what,
                expected,
                found,
            } => write!(
                f,
                "the state file beside this project has a different {what}: the \
                 manifest says {expected} and the file is {found}. They have \
                 come apart, and continuing would resume a run that never \
                 happened."
            ),
        }
    }
}

impl std::error::Error for ProjectError {}

impl From<std::io::Error> for ProjectError {
    fn from(e: std::io::Error) -> Self {
        Self::Io(e)
    }
}

impl Project {
    /// Capture a simulation's scene and configuration.
    ///
    /// Takes the scene, not the state, so this is valid at any point in a run
    /// and always describes t = 0. Saving a project mid-run records how the
    /// run *started*, which is what reproducing it needs.
    pub fn from_simulation(sim: &Simulation) -> Self {
        let mut project = Self::from_world(&sim.world);
        project.run = RunConfig {
            solver_dt: sim.solver_dt,
            output_rate: sim.output_rate,
            max_speculative_distance: sim.max_speculative_distance,
        };
        project.inputs = sim.inputs.clone();
        for entry in &sim.components {
            if let Some(object) = project.objects.get_mut(entry.body) {
                object.components = entry.components.clone();
            }
        }
        for script in sim.scripts() {
            if let Some(object) = project.objects.get_mut(script.body) {
                object.script = Some(script.program.source().into());
            }
        }
        project.solver = SolverConfig::from_solver(&sim.solver, &sim.sleep, &sim.air);
        project.solver.kind = sim.engine_kind().into();
        project.solver.heat = HeatConfig::from(&sim.heat);
        project.solver.stress = StressConfig::from(&sim.stress);
        project.solver.continuum = sim.continuum.into();
        project.solver.fluid_model = sim.unified.fluid_model.into();
        project.solver.implicit_solids = sim.unified.implicit_solids;
        // A simulation carries only the live take — the archive is authoring
        // data the solver never reads — so give the captured scene the one take
        // it can prove exists.
        project.adopt_live_take();
        project
    }

    /// Capture a world alone, with default run and solver configuration.
    pub fn from_world(world: &World) -> Self {
        let (materials, objects) = schema::objects_from_world(world);
        Self {
            fusion: Provenance::current(),
            run: RunConfig::default(),
            solver: SolverConfig::default(),
            world: WorldConfig::from_world(world),
            materials,
            objects,
            bodies: Vec::new(),
            joints: schema::joints_from_world(world),
            cloths: world.cloth_scene().iter().map(ClothEntry::from).collect(),
            soft_bodies: world.soft_scene().iter().map(SoftBodyEntry::from).collect(),
            fluids: world.fluid_scene().iter().map(FluidEntry::from).collect(),
            // Folded into `objects` above, along with the bodies.
            solids: Vec::new(),
            metadata: Metadata::default(),
            creation: CreationDefaults::default(),
            inputs: Vec::new(),
            takes: Vec::new(),
            active_take: None,
            state: None,
        }
    }

    /// Build the world this project describes, at t = 0.
    pub fn build_world(&self) -> Result<World, ProjectError> {
        schema::world_from_project(self)
    }

    /// Build a simulation ready to run: world, solver, clock, all as recorded.
    pub fn build(&self) -> Result<Simulation, ProjectError> {
        if !self.run.solver_dt.is_finite() || self.run.solver_dt <= 0.0 {
            return Err(ProjectError::Unresolved(
                "solver_dt must be finite and positive".into(),
            ));
        }
        if !self.run.output_rate.is_finite() || self.run.output_rate <= 0.0 {
            return Err(ProjectError::Unresolved(
                "output_rate must be finite and positive".into(),
            ));
        }
        if !self.run.max_speculative_distance.is_finite() || self.run.max_speculative_distance < 0.0
        {
            return Err(ProjectError::Unresolved(
                "max_speculative_distance must be finite and nonnegative".into(),
            ));
        }

        fusion_sim::control::validate_actions(&self.inputs).map_err(ProjectError::Unresolved)?;
        fusion_sim::control::validate_takes(&self.takes).map_err(ProjectError::Unresolved)?;
        let world = self.build_world()?;
        let mut sim = Simulation::new(world);
        sim.inputs = self.inputs.clone();
        sim.solver_dt = self.run.solver_dt;
        sim.output_rate = self.run.output_rate;
        sim.max_speculative_distance = self.run.max_speculative_distance;
        self.attach_scripts(&mut sim)?;
        self.solver.apply(&mut sim)?;
        Ok(sim)
    }

    /// Compile every body's script and attach it to the run.
    ///
    /// A script that will not compile fails the *build*, rather than being
    /// dropped or attached broken. A scene is a reproducible artefact (§22),
    /// and one that quietly runs without a behaviour it declares is a scene
    /// that does something different from what the file says — which is worse
    /// than one that refuses to start and names the line.
    fn attach_scripts(&self, sim: &mut Simulation) -> Result<(), ProjectError> {
        // Scripts attach by *body* index, and an object's index is not its
        // body's once a deformable object sits between two rigid ones — so
        // the count is walked the same way `world_from_project` walks it.
        let mut body_index = 0usize;
        for (index, object) in self.objects.iter().enumerate() {
            let deformable = self
                .materials
                .iter()
                .find(|m| m.name == object.material)
                .is_some_and(|m| m.representation == Representation::Deformable);
            object
                .components
                .validate(&self.inputs)
                .map_err(|e| ProjectError::Unresolved(format!("object {index}: {e}")))?;
            if deformable {
                if object.script.is_some() || !object.components.is_empty() {
                    return Err(ProjectError::Unresolved(format!(
                        "object {index} is deformable and carries a script — a \
                         script writes a pose, and a continuum does not have one"
                    )));
                }
                continue;
            }
            let here = body_index;
            body_index += 1;
            if (object.components.controller.is_some() || object.components.force.is_some())
                && object.body_type != BodyKindEntry::Dynamic
            {
                return Err(ProjectError::Unresolved(format!(
                    "object {index}: a force/controller requires a dynamic Rigidbody"
                )));
            }
            if !object.components.is_empty() {
                sim.components.push(fusion_sim::control::BodyComponents {
                    body: here,
                    components: object.components.clone(),
                });
            }
            let Some(source) = object.script.as_deref() else {
                continue;
            };
            if source.trim().is_empty() {
                continue;
            }
            let program = fusion_script::Program::compile(source).map_err(|e| {
                ProjectError::Unresolved(format!(
                    "the script on object {index} does not compile: {e}"
                ))
            })?;
            sim.attach_script(here, program);
        }
        for (index, body) in self.bodies.iter().enumerate() {
            let Some(source) = body.script.as_deref() else {
                continue;
            };
            if source.trim().is_empty() {
                continue;
            }
            let program = fusion_script::Program::compile(source).map_err(|e| {
                ProjectError::Unresolved(format!(
                    "the script on body {index} does not compile: {e}"
                ))
            })?;
            sim.attach_script(body_index + index, program);
        }
        Ok(())
    }

    /// Builder: attach a human note, which §22 wants a project to carry.
    pub fn with_note(mut self, note: impl Into<String>) -> Self {
        self.metadata.note = Some(note.into());
        self
    }

    /// Record the step a bug shows up at, so the file is the whole report.
    ///
    /// §22: "A `.fusion` file plus a step number reproduces anything exactly."
    /// The step number belongs *in* the file, because the two travelling
    /// separately is how a bug report arrives without one.
    pub fn with_step_of_interest(mut self, step: u64) -> Self {
        self.metadata.step_of_interest = Some(step);
        self
    }

    /// The hash that decides whether a checkpoint still belongs to this scene.
    ///
    /// Deliberately blind to the take *archive*. Only the live take reaches the
    /// solver, and that is `inputs`, which is hashed: selecting a different
    /// take changes `inputs` and so invalidates a checkpoint, correctly, while
    /// renaming or deleting some other recording does not. Keeping the archive
    /// out also means a file written before takes existed still hashes to what
    /// its saved run recorded.
    fn scene_hash(&self) -> Result<u64, ProjectError> {
        let mut bare = self.clone();
        bare.state = None;
        bare.takes = Vec::new();
        bare.active_take = None;
        Ok(schema::text_hash(&bare.to_toml()?))
    }

    /// Reconcile the take archive with the live take after reading a file.
    ///
    /// `inputs` is authoritative, because it is what the solver samples and
    /// what every format-4 file carries. A format-4 scene gains the archive it
    /// never had; an inconsistent or hand-edited index is corrected rather than
    /// refused, since the events themselves are still perfectly legible.
    fn adopt_live_take(&mut self) {
        let live_has_events = self.inputs.iter().any(|a| !a.events.is_empty());
        if self.takes.is_empty() {
            self.active_take = None;
            if live_has_events {
                self.takes
                    .push(fusion_sim::control::Take::capture("Take 1", &self.inputs));
                self.active_take = Some(0);
            }
            return;
        }
        let active = self.active_take.unwrap_or(0).min(self.takes.len() - 1);
        self.active_take = Some(active);
        // Re-park the live events under the active take's *name*, so the two
        // agree from here on whatever the file claimed.
        let name = self.takes[active].name.clone();
        self.takes[active] = fusion_sim::control::Take::capture(name, &self.inputs);
    }

    /// Serialize to TOML.
    ///
    /// Compact rather than "pretty". The pretty writer puts every array
    /// element on its own line, which turns `gravity = [0.0, -9.81, 0.0]` into
    /// five lines and a scene of forty bodies into two thousand. A format
    /// whose stated purpose is that a person can read it and change one number
    /// does not get to be that shape.
    pub fn to_toml(&self) -> Result<String, ProjectError> {
        let mut output = self.clone();
        if output.solver.kind == schema::SolverKind::Core {
            // Fusion Core: no earlier reader can run it at all.
            output.fusion.format = 11;
        } else if !output.solver.fluid_model.is_default() || output.solver.implicit_solids {
            // A projected fluid, or implicit solids: a format-9 reader would
            // run the scene the old way, which is a different run.
            output.fusion.format = 10;
        } else if output.objects.iter().any(|o| !o.tier.is_advanced()) {
            // A Basic object: a format-8 reader would solve it Advanced —
            // stressed, breakable, heated — which is a different run.
            output.fusion.format = 9;
        } else if !output.solver.continuum.is_classic()
            || output
                .materials
                .iter()
                .any(|m| m.deformable.is_some_and(|d| d.fibre.is_some()))
        {
            // The unified solver, or a material with fibres only it reads:
            // a format-7 reader would run the scene on the classic solver,
            // which is a different run.
            output.fusion.format = 8;
        } else if !output.solver.stress.is_default()
            || output.objects.iter().any(|o| !o.failure.is_never())
        {
            // Failure changes what the run does, so a reader that would
            // ignore it must refuse the file instead.
            output.fusion.format = 7;
        } else if output.solver.heat.enabled {
            // Heat switched on is the one thing a format-5 reader would run
            // differently by ignoring. Heat off writes no table at all.
            output.fusion.format = 6;
        } else if output.takes.len() > 1 {
            // Only a second take is unreadable by a format-4 editor. One take
            // writes its `[[take]]` block anyway — an old reader ignores it and
            // loses the take's *name*, which is cosmetic, not a different run.
            output.fusion.format = 5;
        } else if !output.inputs.is_empty()
            || output.objects.iter().any(|o| !o.components.is_empty())
        {
            output.fusion.format = 4;
        }
        let body = toml::to_string(&output).map_err(ProjectError::Serialize)?;
        Ok(format!("{HEADER}{body}"))
    }

    /// Parse a project from TOML text.
    pub fn from_toml(text: &str) -> Result<Self, ProjectError> {
        let mut project: Self = toml::from_str(text).map_err(ProjectError::Parse)?;
        if project.fusion.format > FORMAT_VERSION {
            return Err(ProjectError::UnsupportedFormat {
                found: project.fusion.format,
                supported: FORMAT_VERSION,
            });
        }
        // A format-2 file becomes a format-3 project as it is read, so nothing
        // downstream — not the editor, not the exporters, not `verify` — has
        // to know which one it was. The scene it describes is unchanged: see
        // `Project::unify`.
        if !project.creation.is_valid() {
            return Err(ProjectError::Mode(
                "Object defaults require finite velocity and spin",
            ));
        }
        project.unify();
        project.adopt_live_take();
        // Preserve v3 manifests: changing their provenance would invalidate saved-run hashes.
        project.fusion.format = project.fusion.format.max(3);
        Ok(project)
    }

    /// Write this project to `path` as TOML.
    pub fn save(&self, path: impl AsRef<Path>) -> Result<(), ProjectError> {
        std::fs::write(path, self.to_toml()?)?;
        Ok(())
    }

    /// Fold format 2's separate lists into one list of objects (§35).
    ///
    /// The migration, and it is a fold rather than a rewrite: a body becomes
    /// an object whose material is the one it already named, and a solid
    /// becomes an object whose material is a new `[[material]]` entry carrying
    /// the continuum parameters that used to be repeated on every solid.
    ///
    /// **Order is preserved exactly**, bodies first and then solids, because
    /// order is part of the answer — it decides the sequence the contact solve
    /// visits bodies in, and the order material points are built in. A
    /// migration that produced the same picture in a different order would
    /// produce a different run, and `Project::fingerprint_at` would say so.
    ///
    /// Idempotent: a project that has already been unified is left alone.
    pub fn unify(&mut self) {
        if self.bodies.is_empty() && self.solids.is_empty() {
            return;
        }
        let mut objects = Vec::with_capacity(self.bodies.len() + self.solids.len());
        for body in self.bodies.drain(..) {
            objects.push(ObjectEntry {
                name: None,
                shape: body.shape,
                position: body.position,
                orientation: body.orientation,
                linear_velocity: body.linear_velocity,
                angular_velocity: body.angular_velocity,
                material: body.material,
                body_type: body.body_type,
                mass: body.mass,
                resolution: None,
                cell_size: None,
                temperature: None,
                script: body.script,
                components: Default::default(),
                failure: Default::default(),
                grain: None,
                tier: Default::default(),
            });
        }
        for solid in self.solids.drain(..) {
            let desc = fusion_core::SolidDesc::from(&solid);
            let name = schema::intern_solid_material(&mut self.materials, &desc.params.material);
            objects.push(ObjectEntry {
                name: None,
                shape: solid.shape,
                position: solid.position,
                orientation: solid.orientation,
                linear_velocity: solid.velocity,
                angular_velocity: [0.0; 3],
                material: name,
                body_type: schema::BodyKindEntry::Dynamic,
                mass: None,
                // Written out rather than left to the default, because these
                // came from a file that stated them and a migration that
                // quietly re-resolved them would change the answer.
                resolution: Some(solid.spacing),
                cell_size: Some(solid.cell_size),
                temperature: solid.temperature,
                script: None,
                components: Default::default(),
                failure: Default::default(),
                grain: None,
                tier: Default::default(),
            });
        }
        // Appended rather than assigned: a file with both is a file somebody
        // hand-edited, and the objects it already had come first.
        self.objects.append(&mut objects);
    }

    /// Read a project from `path`.
    pub fn load(path: impl AsRef<Path>) -> Result<Self, ProjectError> {
        Self::from_toml(&std::fs::read_to_string(path)?)
    }

    /// Write this project *and the state of a run* — the manifest at `path`,
    /// the state in a blob beside it (v1.0, format 2).
    ///
    /// §23 asks for checkpoint/resume as "basic hygiene for anything expensive
    /// enough to crash mid-run", and this is the pair of calls that is: this
    /// one and [`Project::resume`].
    ///
    /// The blob is named after the manifest — `study.fusion` gets
    /// `study.fusionstate` — rather than given a name of its own, because the
    /// two have to travel together and a name that matches is the cheapest way
    /// to make that obvious to whoever is copying files.
    ///
    /// The scene written is this project, which is the caller's: a scene
    /// edited after the run started saves the *edited* scene beside the old
    /// state, and the fingerprint recorded here is what catches it on the way
    /// back in.
    pub fn save_run(
        &self,
        sim: &fusion_sim::Simulation,
        path: impl AsRef<Path>,
    ) -> Result<(), ProjectError> {
        let path = path.as_ref();
        let blob = sim.checkpoint().to_bytes();
        let name = format!(
            "{}.fusionstate",
            path.file_stem()
                .map(|s| s.to_string_lossy())
                .unwrap_or_else(|| "state".into())
        );
        let mut with_state = self.clone();
        with_state.state = None;
        let scene = self.scene_hash()?;
        with_state.state = Some(StateRef {
            path: name.clone(),
            step: sim.step_index(),
            time: sim.time(),
            bytes: blob.len() as u64,
            hash: fusion_sim::state_hash(sim),
            scene,
        });
        std::fs::write(path.with_file_name(&name), &blob)?;
        with_state.save(path)?;
        Ok(())
    }

    /// Build the run this manifest describes, resuming from its state blob if
    /// it has one.
    ///
    /// `manifest` is the path the project was loaded from — the blob is
    /// relative to it — and a project with no `[state]` builds at t = 0, so
    /// this is safe to use everywhere [`Project::build`] is.
    ///
    /// A blob that is the wrong size, will not decode, or fingerprints
    /// differently is an **error**. Continuing from a state file that does not
    /// belong to this scene is the failure mode worth being loud about: it
    /// produces a run rather than a message.
    pub fn resume(
        &self,
        manifest: impl AsRef<Path>,
    ) -> Result<fusion_sim::Simulation, ProjectError> {
        let mut sim = self.build()?;
        let Some(state) = &self.state else {
            return Ok(sim);
        };
        // The scene first, because it is the check that says *why*: a blob
        // whose fingerprint disagrees might be truncated or from another
        // machine, and a scene that disagrees is somebody having edited the
        // file since the checkpoint, which is a different sentence to read.
        let scene = self.scene_hash()?;
        if scene != state.scene {
            return Err(ProjectError::StaleState {
                what: "scene",
                expected: state.scene,
                found: scene,
            });
        }
        let blob = std::fs::read(manifest.as_ref().with_file_name(&state.path))?;
        if blob.len() as u64 != state.bytes {
            return Err(ProjectError::StaleState {
                what: "size",
                expected: state.bytes,
                found: blob.len() as u64,
            });
        }
        let restored = fusion_sim::Checkpoint::from_bytes(&blob, &sim.checkpoint())
            .map_err(ProjectError::State)?;
        sim.restore(&restored);
        let hash = fusion_sim::state_hash(&sim);
        if hash != state.hash {
            return Err(ProjectError::StaleState {
                what: "fingerprint",
                expected: state.hash,
                found: hash,
            });
        }
        Ok(sim)
    }

    /// Load a manifest and resume the run it describes, in one call.
    pub fn load_run(
        path: impl AsRef<Path>,
    ) -> Result<(Self, fusion_sim::Simulation), ProjectError> {
        let path = path.as_ref();
        let project = Self::load(path)?;
        let sim = project.resume(path)?;
        Ok((project, sim))
    }

    /// Run this project for `steps` and return the state fingerprint.
    ///
    /// The operation §22 is actually asking for. A project file and a step
    /// number identify a state; this is the function that produces it, and
    /// comparing two of them is how "it reproduces" stops being a claim.
    ///
    /// Tier (a) only — see [`fusion_sim::determinism`]. Two machines will
    /// disagree about this number while agreeing about the physics.
    pub fn fingerprint_at(&self, steps: u64) -> Result<u64, ProjectError> {
        let mut sim = self.build()?;
        for _ in 0..steps {
            sim.step();
        }
        Ok(fusion_sim::state_hash(&sim))
    }

    /// Check that this project round-trips through text unchanged and
    /// reproduces the same run afterwards.
    ///
    /// Not a test helper — a *user-facing* check, because the failure it
    /// catches is one that only appears at the moment the file is needed. A
    /// project saved with a field the writer forgot loads into a subtly
    /// different scene, and nobody finds out until a year later when the bug
    /// it was reporting cannot be reproduced.
    pub fn verify(&self, steps: u64) -> Result<VerifyReport, ProjectError> {
        let text = self.to_toml()?;
        let reloaded = Self::from_toml(&text)?;
        let original = self.fingerprint_at(steps)?;
        let round_tripped = reloaded.fingerprint_at(steps)?;
        Ok(VerifyReport {
            identical_text: reloaded == *self,
            original,
            round_tripped,
            steps,
            bytes: text.len(),
        })
    }

    /// Count deformable materials that need adaptation for the Fusion Core solver
    /// (which currently translates Neo-Hookean elasticity with inert thermal,
    /// no continuum fracture, and no fibres).
    pub fn core_incompatible_deformable_count(&self) -> usize {
        self.materials
            .iter()
            .filter(|m| m.representation == Representation::Deformable)
            .filter(|m| {
                let d = m.deformable.unwrap_or_default();
                let solid = m.solid_part();
                d.model != ConstitutiveEntry::NeoHookean
                    || d.fracture_onset.is_some()
                    || d.fibre.is_some()
                    || solid.thermal != fusion_core::Thermal::INERT
            })
            .count()
    }

    /// Adapt deformable materials in place so they build cleanly on the Fusion
    /// Core solver adapter (Neo-Hookean, inert thermal, no fracture or fibres).
    /// Returns the number of materials modified.
    pub fn adapt_deformables_for_core(&mut self) -> usize {
        let inert = DeformableEntry::from(&fusion_core::SolidMaterial {
            thermal: fusion_core::Thermal::INERT,
            ..fusion_core::SolidMaterial::JELLY
        });
        let mut changed = 0;
        for m in &mut self.materials {
            if m.representation != Representation::Deformable {
                continue;
            }
            let solid = m.solid_part();
            let d = m.deformable.get_or_insert_with(Default::default);
            let incompatible = d.model != ConstitutiveEntry::NeoHookean
                || d.fracture_onset.is_some()
                || d.fibre.is_some()
                || solid.thermal != fusion_core::Thermal::INERT;
            if incompatible {
                *d = DeformableEntry {
                    youngs_modulus: d.youngs_modulus,
                    poissons_ratio: d.poissons_ratio,
                    model: ConstitutiveEntry::NeoHookean,
                    fracture_onset: None,
                    fibre: None,
                    ..inert
                };
                changed += 1;
            }
        }
        changed
    }
}

/// The result of [`Project::verify`].
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct VerifyReport {
    /// Whether the project parsed back into exactly the same description.
    pub identical_text: bool,
    /// State hash of the original run at the end of `steps`.
    pub original: u64,
    /// State hash of the run rebuilt from the saved file.
    pub round_tripped: u64,
    /// How many steps both were advanced before hashing.
    pub steps: u64,
    /// Size of the saved file, bytes.
    pub bytes: usize,
}

impl VerifyReport {
    /// Whether the file reproduces the run.
    pub fn reproduces(&self) -> bool {
        self.identical_text && self.original == self.round_tripped
    }
}

impl fmt::Display for VerifyReport {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        if self.reproduces() {
            write!(
                f,
                "reproduces: {} bytes, {} steps, state {:#018x}",
                self.bytes, self.steps, self.original
            )
        } else if !self.identical_text {
            write!(
                f,
                "does NOT reproduce: the file did not parse back into the same description"
            )
        } else {
            write!(
                f,
                "does NOT reproduce: state {:#018x} became {:#018x} after {} steps",
                self.original, self.round_tripped, self.steps
            )
        }
    }
}

const HEADER: &str = "\
# A Fusion project: the tuple that makes a run reproducible (PLAN.md §22).
#
# This is a scene, not a state — it describes the world at t = 0, and the
# simulation is whatever follows from it. That is why a fluid of forty
# thousand particles is nine numbers here: the particles are regenerated by
# filling the shape, not stored.
#
# Together with a step number this file identifies one exact state, on the
# same build and machine. `Project::verify` checks that it still does.

";
