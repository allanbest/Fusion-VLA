//! A checkpoint as bytes, so a run can outlive the process that made it.
//!
//! PLAN.md §23 asks for "checkpoint / resume — worth pulling early; basic
//! hygiene for anything expensive enough to crash mid-run", and §22 wants the
//! project format to reference "binary blobs for meshes, point data and large
//! initial fields" rather than inlining them. A million material points is
//! 100 MB of state and would be 400 MB of decimal TOML that no longer holds
//! the same numbers; this is the other half of the format.
//!
//! ## What is in the file, and what is not
//!
//! **In:** everything a step *changes* — body poses, velocities and sleep
//! state, cloth and soft-body particles, every fluid, and every material
//! point's position, velocity, deformation gradient, affine velocity, plastic
//! ratio, damage and enthalpy — plus the clock, script locals, script failures
//! and pending script error reports. Script source/body identity is checked
//! against the scene on load.
//!
//! **Not in:** anything the scene already says. Shapes, masses, materials,
//! joints, gravity and every solver dial live in the manifest beside this
//! file, which is what keeps a checkpoint of a large scene the size of its
//! state rather than the size of its state plus its description.
//!
//! **Also in, since version 6:** the contact history — the anchors the
//! collision pipeline carries between steps, which are what static friction
//! holds each resting contact to, and the multipliers the solver warm-starts
//! from. They are not physics, and they are not free of consequence either:
//! without them a resumed pile starts its contacts from nothing, and a box
//! balanced most of the way over an edge slid off it. [`crate::checkpoint`]'s
//! module documentation has the older measurement: a world-only restore
//! diverges after one step at five of the eight checkpoint times tried.
//!
//! **Still not in:** the remaining scratch — island assignment, fluid
//! neighbour lists, the continuum grid — which the first step rebuilds. So the
//! guarantee this file makes is narrower than an in-process [`Checkpoint`]'s,
//! and stated rather than hoped for:
//!
//! - The physical state is restored **bit for bit**. Every number in the list
//!   above comes back exactly as it was written.
//! - A run *continued* from a restored file is close to, and not promised to
//!   be bit-identical with, the run it was cut from. See
//!   `a_resumed_run_is_the_run_it_was_cut_from`, which finds it bit-identical.
//!
//! Which is the right trade for what a checkpoint file is for — a run too
//! expensive to lose — and the wrong one for reproducing a bug exactly, where
//! §22's answer is the one to use: the scene file plus a step number, from
//! t = 0, which is bit-exact by [`crate::determinism`].
//!
//! ## The encoding
//!
//! Little-endian, fixed-width, no compression and no framing beyond a header:
//! a `Vec<f32>` is its bytes. Hand-rolled rather than pulled from a crate
//! because `fusion-sim` is L4 and §5 does not let it take a dependency, and
//! because the format is a dozen arrays — the serialization library would be
//! larger than the thing it serialized.

use fusion_core::solid::{Constitutive, Fracture, SolidMaterial, SolidParams, Thermal};
use fusion_core::{BodyType, Failure, ParticleSet, Shape, SolidSet};
use fusion_math::{DVec3, Mat3, Quat, Vec3};

use crate::checkpoint::Checkpoint;

/// Eight bytes at the front of every state file.
const MAGIC: [u8; 8] = *b"FUSIONST";

/// The state encoding's own version, which is not the project format's.
///
/// Bumped when the *bytes* change: a field added to a material point moves
/// this and does not necessarily move `fusion_project::FORMAT_VERSION`, and a
/// manifest field moves that one and not this.
// Version 2 adds behaviour state. Version 1 remains readable for scenes without
// scripts; it cannot safely resume a script whose locals were never recorded.
// Version 3 adds rigid-body temperature. Versions 1 and 2 remain readable:
// every body is left at the temperature the scene starts it at.
// Version 4 adds what failure changes: bodies broken into pieces carry their
// shapes and masses, removed bodies are marked, and solids a body turned into
// carry their material. Earlier versions remain readable.
// Version 5 adds the unified continuum solver's state: each solid's fibres,
// and each fluid's affine field and volume ratios. Earlier versions remain
// readable: a fluid then starts the unified solver at rest.
// Version 6 adds the contact history: the anchors the collision pipeline
// carries between steps and the multipliers the solver warm-starts from.
// Earlier versions remain readable, and resume with no history.
// Version 7 ends the file with a flag for Fusion Core state, followed when it
// is set by the core's own encoding of its state and integration history.
// Versions 1..=6 remain readable, on the Standard engine only.
pub const STATE_VERSION: u32 = 7;

/// Why a state file could not be read.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum StateError {
    /// The file does not start with the Fusion state signature.
    NotState,
    /// Structurally invalid script state or an incomplete legacy script checkpoint.
    Invalid(&'static str),
    /// Written by a newer Fusion than this one.
    Version {
        /// What the file says it is.
        found: u32,
        /// The newest this build reads.
        supported: u32,
    },
    /// The file ended in the middle of a field.
    Truncated {
        /// Bytes wanted at the point it ran out.
        wanted: usize,
        /// Bytes left.
        left: usize,
    },
    /// The file holds a differently-shaped scene than the one it is being
    /// restored into.
    ///
    /// A state file is not a scene: it is the state *of* one, and the
    /// simulation it is loaded into has to be the simulation it came from.
    /// Reported rather than asserted, because unlike an in-process restore
    /// this one is reading a file a user chose.
    Shape {
        /// What is being counted.
        what: &'static str,
        /// How many the file holds.
        found: usize,
        /// How many the simulation has.
        wanted: usize,
    },
}

impl std::fmt::Display for StateError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::NotState => write!(f, "not a Fusion state file"),
            Self::Invalid(reason) => write!(f, "invalid state: {reason}"),
            Self::Version { found, supported } => write!(
                f,
                "state file is version {found} and this build reads {supported}"
            ),
            Self::Truncated { wanted, left } => {
                write!(
                    f,
                    "state file ended early: wanted {wanted} bytes, {left} left"
                )
            }
            Self::Shape {
                what,
                found,
                wanted,
            } => write!(
                f,
                "state file holds {found} {what} and this simulation has {wanted} — \
                 a state file belongs to the scene it was taken from"
            ),
        }
    }
}

impl std::error::Error for StateError {}

// ------------------------------------------------------------------- writing

/// A little-endian byte sink. Nothing clever: the point of the encoding is
/// that there is nothing to get wrong on the way back.
struct Writer(Vec<u8>);

impl Writer {
    fn text(&mut self, text: &str) {
        self.u64(text.len() as u64);
        self.0.extend_from_slice(text.as_bytes());
    }
    fn error(&mut self, error: &fusion_script::ScriptError) {
        self.u32(error.line);
        self.text(&error.message);
    }
    fn scripts(&mut self, scripts: &[crate::script::BodyScript]) {
        self.u32(scripts.len() as u32);
        for script in scripts {
            self.u64(script.body as u64);
            self.text(script.program.source());
            let mut locals: Vec<_> = script.locals.iter().collect();
            locals.sort_unstable_by(|a, b| a.0.cmp(b.0));
            self.u32(locals.len() as u32);
            for (name, value) in locals {
                self.text(name);
                match value {
                    fusion_script::Value::Nothing => self.u8(0),
                    fusion_script::Value::Number(value) => {
                        self.u8(1);
                        self.f32(*value);
                    }
                    fusion_script::Value::Bool(value) => {
                        self.u8(2);
                        self.u8(u8::from(*value));
                    }
                    fusion_script::Value::Vector(value) => {
                        self.u8(3);
                        self.vec3(*value);
                    }
                }
            }
            self.u8(u8::from(script.failed.is_some()));
            if let Some(error) = &script.failed {
                self.error(error);
            }
        }
    }
    fn u8(&mut self, v: u8) {
        self.0.push(v);
    }
    fn u32(&mut self, v: u32) {
        self.0.extend_from_slice(&v.to_le_bytes());
    }
    fn u64(&mut self, v: u64) {
        self.0.extend_from_slice(&v.to_le_bytes());
    }
    fn f32(&mut self, v: f32) {
        self.0.extend_from_slice(&v.to_le_bytes());
    }
    fn f64(&mut self, v: f64) {
        self.0.extend_from_slice(&v.to_le_bytes());
    }
    fn dvec3(&mut self, v: DVec3) {
        self.f64(v.x);
        self.f64(v.y);
        self.f64(v.z);
    }
    fn vec3(&mut self, v: Vec3) {
        self.f32(v.x);
        self.f32(v.y);
        self.f32(v.z);
    }
    fn quat(&mut self, q: Quat) {
        for v in q.to_array() {
            self.f32(v);
        }
    }
    fn mat3(&mut self, m: Mat3) {
        for column in [m.x_axis, m.y_axis, m.z_axis] {
            self.vec3(column);
        }
    }
    fn particles(&mut self, p: &ParticleSet) {
        self.u32(p.len() as u32);
        for i in 0..p.len() {
            self.dvec3(p.position[i]);
            self.vec3(p.velocity[i]);
        }
    }
    fn shape(&mut self, shape: &Shape) {
        match shape {
            Shape::Sphere { radius } => {
                self.u8(0);
                self.f32(*radius);
            }
            Shape::Cuboid { half_extents } => {
                self.u8(1);
                self.vec3(*half_extents);
            }
            Shape::Hull(hull) => {
                self.u8(2);
                self.u32(hull.vertices().len() as u32);
                for v in hull.vertices() {
                    self.vec3(*v);
                }
            }
        }
    }
    fn solid_params(&mut self, s: &SolidSet) {
        let p = &s.params;
        self.f32(p.spacing);
        self.f32(p.cell_size);
        self.f32(s.temperature_at_rest);
        let m = &p.material;
        self.f32(m.density);
        self.f32(m.youngs_modulus);
        self.f32(m.poissons_ratio);
        match m.model {
            Constitutive::FixedCorotated => {
                self.u8(0);
                self.f32(0.0);
                self.f32(0.0);
                self.f32(0.0);
            }
            Constitutive::NeoHookean => {
                self.u8(1);
                self.f32(0.0);
                self.f32(0.0);
                self.f32(0.0);
            }
            Constitutive::Snow {
                critical_compression,
                critical_stretch,
                hardening,
            } => {
                self.u8(2);
                self.f32(critical_compression);
                self.f32(critical_stretch);
                self.f32(hardening);
            }
            Constitutive::VonMises { yield_strain } => {
                self.u8(3);
                self.f32(yield_strain);
                self.f32(0.0);
                self.f32(0.0);
            }
        }
        let t = &m.thermal;
        for v in [
            t.specific_heat,
            t.conductivity,
            t.expansion,
            t.melt_point,
            t.latent_heat,
            t.softening,
            t.plastic_heating,
            t.emissivity,
        ] {
            self.f32(v);
        }
        self.f32(m.fracture.onset);
        self.f32(m.fracture.failure);
        match s.params.fibre {
            Some(fibre) => {
                self.u8(1);
                self.vec3(fibre.direction);
                self.f32(fibre.stiffness);
            }
            None => self.u8(0),
        }
    }
    fn solid(&mut self, s: &SolidSet) {
        self.particles(&s.particles);
        for i in 0..s.len() {
            self.mat3(s.deformation[i]);
            self.mat3(s.affine[i]);
            self.f32(s.volume[i]);
            self.f32(s.plastic[i]);
            self.f32(s.damage[i]);
            self.f64(s.enthalpy[i]);
        }
    }
}

// ------------------------------------------------------------------- reading

struct Reader<'a> {
    bytes: &'a [u8],
    at: usize,
}

impl<'a> Reader<'a> {
    fn text(&mut self) -> Result<&'a str, StateError> {
        let length =
            usize::try_from(self.u64()?).map_err(|_| StateError::Invalid("text is too large"))?;
        std::str::from_utf8(self.take(length)?)
            .map_err(|_| StateError::Invalid("text is not UTF-8"))
    }
    fn boolean(&mut self) -> Result<bool, StateError> {
        match self.u8()? {
            0 => Ok(false),
            1 => Ok(true),
            _ => Err(StateError::Invalid("invalid boolean tag")),
        }
    }
    fn error(&mut self) -> Result<fusion_script::ScriptError, StateError> {
        let line = self.u32()?;
        let message = self.text()?.to_owned();
        Ok(fusion_script::ScriptError::new(line, message))
    }
    fn scripts(&mut self, scripts: &mut [crate::script::BodyScript]) -> Result<(), StateError> {
        self.count("scripts", scripts.len())?;
        for script in scripts {
            if self.u64()? != script.body as u64 || self.text()? != script.program.source() {
                return Err(StateError::Invalid(
                    "script source or attached body differs from the scene",
                ));
            }
            let count = self.u32()?;
            script.locals.clear();
            for _ in 0..count {
                let name = self.text()?.to_owned();
                let value = match self.u8()? {
                    0 => fusion_script::Value::Nothing,
                    1 => fusion_script::Value::Number(self.f32()?),
                    2 => fusion_script::Value::Bool(self.boolean()?),
                    3 => fusion_script::Value::Vector(self.vec3()?),
                    _ => return Err(StateError::Invalid("unknown script value tag")),
                };
                if script.locals.insert(name, value).is_some() {
                    return Err(StateError::Invalid("duplicate script variable"));
                }
            }
            script.failed = if self.boolean()? {
                Some(self.error()?)
            } else {
                None
            };
        }
        Ok(())
    }
    fn take(&mut self, n: usize) -> Result<&'a [u8], StateError> {
        let left = self.bytes.len() - self.at;
        if left < n {
            return Err(StateError::Truncated { wanted: n, left });
        }
        let slice = &self.bytes[self.at..self.at + n];
        self.at += n;
        Ok(slice)
    }
    fn u8(&mut self) -> Result<u8, StateError> {
        Ok(self.take(1)?[0])
    }
    fn u32(&mut self) -> Result<u32, StateError> {
        Ok(u32::from_le_bytes(self.take(4)?.try_into().unwrap()))
    }
    fn u64(&mut self) -> Result<u64, StateError> {
        Ok(u64::from_le_bytes(self.take(8)?.try_into().unwrap()))
    }
    fn f32(&mut self) -> Result<f32, StateError> {
        Ok(f32::from_le_bytes(self.take(4)?.try_into().unwrap()))
    }
    fn f64(&mut self) -> Result<f64, StateError> {
        Ok(f64::from_le_bytes(self.take(8)?.try_into().unwrap()))
    }
    fn dvec3(&mut self) -> Result<DVec3, StateError> {
        Ok(DVec3::new(self.f64()?, self.f64()?, self.f64()?))
    }
    fn vec3(&mut self) -> Result<Vec3, StateError> {
        Ok(Vec3::new(self.f32()?, self.f32()?, self.f32()?))
    }
    fn quat(&mut self) -> Result<Quat, StateError> {
        let (x, y, z, w) = (self.f32()?, self.f32()?, self.f32()?, self.f32()?);
        Ok(Quat::from_xyzw(x, y, z, w))
    }
    fn mat3(&mut self) -> Result<Mat3, StateError> {
        Ok(Mat3::from_cols(self.vec3()?, self.vec3()?, self.vec3()?))
    }
    fn count(&mut self, what: &'static str, wanted: usize) -> Result<(), StateError> {
        let found = self.u32()? as usize;
        if found != wanted {
            return Err(StateError::Shape {
                what,
                found,
                wanted,
            });
        }
        Ok(())
    }
    fn particles(&mut self, what: &'static str, p: &mut ParticleSet) -> Result<(), StateError> {
        self.count(what, p.len())?;
        for i in 0..p.len() {
            p.position[i] = self.dvec3()?;
            p.velocity[i] = self.vec3()?;
        }
        Ok(())
    }
    fn shape(&mut self) -> Result<Shape, StateError> {
        match self.u8()? {
            0 => Ok(Shape::sphere(self.f32()?)),
            1 => Ok(Shape::Cuboid {
                half_extents: self.vec3()?,
            }),
            2 => {
                let n = self.u32()? as usize;
                if n > 1 << 20 {
                    return Err(StateError::Invalid("a hull with too many vertices"));
                }
                let mut points = Vec::with_capacity(n);
                for _ in 0..n {
                    points.push(self.vec3()?);
                }
                Shape::hull(&points).ok_or(StateError::Invalid("a broken piece has no volume"))
            }
            _ => Err(StateError::Invalid("unknown shape")),
        }
    }
    fn solid_params(&mut self, version: u32) -> Result<SolidSet, StateError> {
        let spacing = self.f32()?;
        let cell_size = self.f32()?;
        let at_rest = self.f32()?;
        let density = self.f32()?;
        let youngs_modulus = self.f32()?;
        let poissons_ratio = self.f32()?;
        let tag = self.u8()?;
        let (a, b, c) = (self.f32()?, self.f32()?, self.f32()?);
        let model = match tag {
            0 => Constitutive::FixedCorotated,
            1 => Constitutive::NeoHookean,
            2 => Constitutive::Snow {
                critical_compression: a,
                critical_stretch: b,
                hardening: c,
            },
            3 => {
                let _ = (b, c);
                Constitutive::VonMises { yield_strain: a }
            }
            _ => return Err(StateError::Invalid("unknown constitutive model")),
        };
        let mut t = [0.0f32; 8];
        for v in &mut t {
            *v = self.f32()?;
        }
        let thermal = Thermal {
            specific_heat: t[0],
            conductivity: t[1],
            expansion: t[2],
            melt_point: t[3],
            latent_heat: t[4],
            softening: t[5],
            plastic_heating: t[6],
            emissivity: t[7],
        };
        let fracture = Fracture {
            onset: self.f32()?,
            failure: self.f32()?,
        };
        let fibre = if version >= 5 && self.u8()? == 1 {
            Some(fusion_core::Fibre {
                direction: self.vec3()?,
                stiffness: self.f32()?,
            })
        } else {
            None
        };
        let params = SolidParams {
            spacing,
            cell_size,
            material: SolidMaterial {
                density,
                youngs_modulus,
                poissons_ratio,
                model,
                thermal,
                fracture,
            },
            fibre,
        };
        let mut solid = SolidSet::new(params);
        solid.temperature_at_rest = at_rest;
        Ok(solid)
    }
    fn solid(&mut self, s: &mut SolidSet) -> Result<(), StateError> {
        self.particles("material points", &mut s.particles)?;
        for i in 0..s.len() {
            s.deformation[i] = self.mat3()?;
            s.affine[i] = self.mat3()?;
            s.volume[i] = self.f32()?;
            s.plastic[i] = self.f32()?;
            s.damage[i] = self.f32()?;
            s.enthalpy[i] = self.f64()?;
        }
        Ok(())
    }
}

/// Stable ordering for checkpoint encoding and simulation fingerprints.
pub(crate) fn script_state_bytes(scripts: &[crate::script::BodyScript]) -> Vec<u8> {
    let mut writer = Writer(Vec::new());
    writer.scripts(scripts);
    writer.0
}

impl Checkpoint {
    /// Encode the physical state and the clock.
    ///
    /// See the module documentation for what is deliberately left out and what
    /// that costs.
    pub fn to_bytes(&self) -> Vec<u8> {
        let mut w = Writer(Vec::with_capacity(self.size_bytes()));
        w.0.extend_from_slice(&MAGIC);
        w.u32(STATE_VERSION);

        w.f64(self.time());
        w.u64(self.step_index());
        w.u64(self.frame_index_of());
        w.f64(self.accumulator_of());

        let bodies = self.bodies_of();
        w.u32(bodies.len() as u32);
        for i in 0..bodies.len() {
            w.dvec3(bodies.position[i]);
            w.quat(bodies.orientation[i]);
            w.vec3(bodies.linear_velocity[i]);
            w.vec3(bodies.angular_velocity[i]);
            w.u8(bodies.sleeping[i] as u8);
            w.f32(bodies.sleep_timer[i]);
            // What failure changed. A reshaped body carries what the scene can
            // no longer say about it: its shape, its mass, which way its grain
            // runs in it, and which scene body it came from, whose material
            // it has.
            w.u8(bodies.removed[i] as u8 | (bodies.reshaped[i] as u8) << 1);
            w.u8(match bodies.failure[i] {
                Failure::Never => 0,
                Failure::Breaks => 1,
                Failure::Deforms => 2,
            });
            if bodies.reshaped[i] {
                w.u32(bodies.source[i]);
                w.f32(bodies.inv_mass[i]);
                w.vec3(bodies.inv_inertia_local[i]);
                w.shape(&bodies.shape[i]);
                w.vec3(bodies.grain[i]);
            }
        }

        w.particles(self.particles_of());

        let fluids = self.fluids_of();
        w.u32(fluids.len() as u32);
        for fluid in fluids {
            w.particles(&fluid.particles);
        }

        let solids = self.solids_of();
        w.u32(solids.len() as u32);
        for solid in solids {
            // Every solid's material and resolution, so one a body turned into
            // can be rebuilt. A scene's own solids write the same numbers the
            // manifest has, which the reader ignores for them.
            w.solid_params(solid);
            w.solid(solid);
        }
        w.scripts(self.scripts_of());
        w.u32(self.script_errors_of().len() as u32);
        for (index, error) in self.script_errors_of() {
            w.u64(*index as u64);
            w.error(error);
        }
        // Rigid-body temperature. The grid's geometry is not written: it is a
        // function of the shape, which the scene already has, so the reader
        // rebuilds it and only the numbers travel.
        for heat in &bodies.heat {
            w.f64(heat.uniform);
            w.u32(heat.temperature.len() as u32);
            for t in &heat.temperature {
                w.f64(*t);
            }
        }
        // The unified solver's fluid state: empty for a fluid it has not run.
        for fluid in self.fluids_of() {
            let n = if fluid.affine.len() == fluid.len() && fluid.volume_ratio.len() == fluid.len()
            {
                fluid.len()
            } else {
                0
            };
            w.u32(n as u32);
            for i in 0..n {
                w.mat3(fluid.affine[i]);
                w.f32(fluid.volume_ratio[i]);
            }
        }
        // The contact history: what friction holds each resting contact to,
        // and what the solver resumes each one from.
        let remembered = self.pipeline_of().remembered();
        w.u32(remembered.len() as u32);
        for m in remembered {
            w.u32(m.body_a);
            w.u32(m.body_b);
            w.vec3(m.local_normal_a);
            w.vec3(m.world_normal);
            w.f32(m.friction);
            w.f32(m.restitution);
            w.u8(m.len() as u8);
            for p in m.points() {
                w.vec3(p.local_a);
                w.vec3(p.local_b);
                w.vec3(p.local_normal_a);
                w.f32(p.separation);
                w.u8(p.inherited as u8);
            }
        }
        let memory: Vec<_> = self.solver_memory_of().contacts().collect();
        w.u32(memory.len() as u32);
        for c in &memory {
            w.u32(c.body_a);
            w.u32(c.body_b);
            w.u8(c.count);
            for &(local_b, lambda, stick) in &c.points[..c.count as usize] {
                w.vec3(local_b);
                w.f32(lambda);
                w.vec3(stick);
            }
        }
        w.u32(u32::from(self.core.is_some()));
        if let Some(core) = &self.core {
            let bytes = core.to_bytes();
            w.u64(bytes.len() as u64);
            w.0.extend_from_slice(&bytes);
        }
        w.0
    }

    /// Decode into a checkpoint shaped like `template`.
    ///
    /// `template` supplies everything the file does not carry — the scene, the
    /// joints, and the solver caches — so the natural way to call this is with
    /// a checkpoint of the simulation the file is about to be restored into.
    /// A file from a differently-shaped scene is a [`StateError::Shape`]
    /// rather than a panic, because unlike an in-process restore this one is
    /// reading a file somebody chose.
    pub fn from_bytes(bytes: &[u8], template: &Checkpoint) -> Result<Checkpoint, StateError> {
        let mut r = Reader { bytes, at: 0 };
        if r.take(8)? != MAGIC {
            return Err(StateError::NotState);
        }
        let version = r.u32()?;
        if version == 0 || version > STATE_VERSION {
            return Err(StateError::Version {
                found: version,
                supported: STATE_VERSION,
            });
        }

        let mut out = template.clone();
        let (time, step, frame, accumulator) = (r.f64()?, r.u64()?, r.u64()?, r.f64()?);
        out.set_clock(time, step, frame, accumulator);

        {
            let bodies = out.bodies_mut();
            let scene = bodies.len();
            let count = if version >= 4 {
                // At least the scene's bodies: more where something broke.
                let found = r.u32()? as usize;
                if found < scene || found > scene + (1 << 20) {
                    return Err(StateError::Shape {
                        what: "bodies",
                        found,
                        wanted: scene,
                    });
                }
                found
            } else {
                r.count("bodies", scene)?;
                scene
            };
            for i in 0..count {
                let position = r.dvec3()?;
                let orientation = r.quat()?;
                let linear = r.vec3()?;
                let angular = r.vec3()?;
                let sleeping = r.u8()? != 0;
                let timer = r.f32()?;
                let (flags, failure) = if version >= 4 {
                    (r.u8()?, r.u8()?)
                } else {
                    (0, 0)
                };
                let reshaped = flags & 2 != 0;
                if i >= scene && !reshaped {
                    // More bodies than the scene has, and not pieces of any
                    // of them: a file from another scene.
                    return Err(StateError::Shape {
                        what: "bodies",
                        found: count,
                        wanted: scene,
                    });
                }
                if reshaped {
                    let source = r.u32()?;
                    let inv_mass = r.f32()?;
                    let inv_inertia = r.vec3()?;
                    let shape = r.shape()?;
                    let grain = r.vec3()?;
                    if source as usize >= scene {
                        return Err(StateError::Invalid(
                            "a broken piece names a body that is not in the scene",
                        ));
                    }
                    let material = bodies.material[source as usize];
                    if i >= scene {
                        bodies.push(
                            fusion_math::Isometry::new(fusion_math::Vec3::ZERO, orientation),
                            linear,
                            angular,
                            inv_mass,
                            inv_inertia,
                            shape,
                            material,
                            BodyType::Dynamic,
                        );
                    } else {
                        bodies.inv_mass[i] = inv_mass;
                        bodies.inv_inertia_local[i] = inv_inertia;
                        bodies.inertia_local[i] = fusion_math::Vec3::new(
                            if inv_inertia.x > 0.0 {
                                1.0 / inv_inertia.x
                            } else {
                                0.0
                            },
                            if inv_inertia.y > 0.0 {
                                1.0 / inv_inertia.y
                            } else {
                                0.0
                            },
                            if inv_inertia.z > 0.0 {
                                1.0 / inv_inertia.z
                            } else {
                                0.0
                            },
                        );
                        bodies.shape[i] = shape;
                    }
                    bodies.source[i] = source;
                    bodies.reshaped[i] = true;
                    bodies.grain[i] = grain;
                    bodies.stress[i] = fusion_core::BodyStress::default();
                }
                bodies.position[i] = position;
                bodies.orientation[i] = orientation;
                bodies.linear_velocity[i] = linear;
                bodies.angular_velocity[i] = angular;
                bodies.failure[i] = match failure {
                    0 => Failure::Never,
                    1 => Failure::Breaks,
                    2 => Failure::Deforms,
                    _ => return Err(StateError::Invalid("unknown failure mode")),
                };
                if flags & 1 != 0 {
                    bodies.remove(i);
                }
                bodies.sleeping[i] = sleeping;
                bodies.sleep_timer[i] = timer;
            }
        }
        r.particles("particles", out.particles_mut())?;
        {
            let fluids = out.fluids_mut();
            r.count("fluids", fluids.len())?;
            for fluid in fluids.iter_mut() {
                r.particles("fluid particles", &mut fluid.particles)?;
            }
        }
        {
            let solids = out.solids_mut();
            let scene = solids.len();
            let count = if version >= 4 {
                let found = r.u32()? as usize;
                if found < scene || found > scene + (1 << 16) {
                    return Err(StateError::Shape {
                        what: "solids",
                        found,
                        wanted: scene,
                    });
                }
                found
            } else {
                r.count("solids", scene)?;
                scene
            };
            for k in 0..count {
                if version >= 4 {
                    let described = r.solid_params(version)?;
                    if k >= scene {
                        // A solid a body became. Its points are in the file;
                        // made here with placeholder positions to read into.
                        let mut solid = described;
                        let at = r.at;
                        let points = r.u32()? as usize;
                        r.at = at;
                        if points > 1 << 26 {
                            return Err(StateError::Invalid("a solid with too many points"));
                        }
                        for _ in 0..points {
                            solid.push(fusion_math::Vec3::ZERO);
                        }
                        solids.push(solid);
                    }
                }
                r.solid(&mut solids[k])?;
            }
        }
        if version == 1 {
            if !out.scripts_of().is_empty() {
                return Err(StateError::Invalid("legacy checkpoints do not contain script memory; restart this scripted scene from its initial state"));
            }
        } else {
            r.scripts(out.scripts_mut())?;
            let count = r.u32()? as usize;
            let script_count = out.scripts_of().len();
            let errors = out.script_errors_mut();
            errors.clear();
            for _ in 0..count {
                let index = r.u64()?;
                if index >= script_count as u64 {
                    return Err(StateError::Invalid(
                        "script error refers to a missing script",
                    ));
                }
                errors.push((index as usize, r.error()?));
            }
        }
        if version >= 3 {
            let bodies = out.bodies_mut();
            for i in 0..bodies.len() {
                let uniform = r.f64()?;
                let cells = r.u32()? as usize;
                let shape = bodies.shape[i].clone();
                let heat = &mut bodies.heat[i];
                *heat = fusion_core::BodyHeat::uniform(uniform as f32);
                heat.uniform = uniform;
                if cells > 0 {
                    heat.build(&shape);
                    if heat.temperature.len() != cells {
                        return Err(StateError::Invalid(
                            "a body's temperature grid does not match its shape",
                        ));
                    }
                    for t in heat.temperature.iter_mut() {
                        *t = r.f64()?;
                    }
                }
            }
        }
        if version >= 5 {
            for fluid in out.fluids_mut() {
                let n = r.u32()? as usize;
                if n != 0 && n != fluid.len() {
                    return Err(StateError::Invalid(
                        "a fluid's unified state does not match its particles",
                    ));
                }
                fluid.affine.clear();
                fluid.volume_ratio.clear();
                for _ in 0..n {
                    fluid.affine.push(r.mat3()?);
                    fluid.volume_ratio.push(r.f32()?);
                }
            }
        }
        if version >= 6 {
            let bodies = out.bodies_of().len() as u32;
            let pair = |a: u32, b: u32| {
                if a < b && b < bodies {
                    Ok(())
                } else {
                    Err(StateError::Invalid("contact history names a missing body"))
                }
            };
            let count = r.u32()? as usize;
            if count > 1 << 24 {
                return Err(StateError::Invalid("too much contact history"));
            }
            let mut manifolds = Vec::with_capacity(count);
            for _ in 0..count {
                let (a, b) = (r.u32()?, r.u32()?);
                pair(a, b)?;
                let (local_normal, world_normal) = (r.vec3()?, r.vec3()?);
                let mut m = fusion_collide::ContactManifold::new(a, b, local_normal, world_normal);
                m.friction = r.f32()?;
                m.restitution = r.f32()?;
                if !(local_normal.is_finite()
                    && world_normal.is_finite()
                    && m.friction.is_finite()
                    && m.restitution.is_finite())
                {
                    return Err(StateError::Invalid("contact history is not finite"));
                }
                let points = r.u8()? as usize;
                if points > fusion_collide::MAX_CONTACT_POINTS {
                    return Err(StateError::Invalid("a contact with too many points"));
                }
                for _ in 0..points {
                    let p = fusion_collide::ContactPoint {
                        local_a: r.vec3()?,
                        local_b: r.vec3()?,
                        local_normal_a: r.vec3()?,
                        separation: r.f32()?,
                        inherited: r.boolean()?,
                    };
                    if !(p.local_a.is_finite()
                        && p.local_b.is_finite()
                        && p.local_normal_a.is_finite()
                        && p.separation.is_finite())
                    {
                        return Err(StateError::Invalid("contact history is not finite"));
                    }
                    m.push(p);
                }
                manifolds.push(m);
            }
            out.pipeline_mut().remember(manifolds);
            let count = r.u32()? as usize;
            if count > 1 << 24 {
                return Err(StateError::Invalid("too much contact history"));
            }
            let mut memory = Vec::with_capacity(count);
            for _ in 0..count {
                let (a, b) = (r.u32()?, r.u32()?);
                pair(a, b)?;
                let n = r.u8()?;
                if n as usize > fusion_collide::MAX_CONTACT_POINTS {
                    return Err(StateError::Invalid("a contact with too many points"));
                }
                let mut points =
                    [(Vec3::ZERO, 0.0, Vec3::ZERO); fusion_collide::MAX_CONTACT_POINTS];
                for p in points.iter_mut().take(n as usize) {
                    *p = (r.vec3()?, r.f32()?, r.vec3()?);
                    if !(p.0.is_finite() && p.1.is_finite() && p.1 >= 0.0 && p.2.is_finite()) {
                        return Err(StateError::Invalid("contact history is not finite"));
                    }
                }
                memory.push(fusion_solver::RememberedContact {
                    body_a: a,
                    body_b: b,
                    points,
                    count: n,
                });
            }
            *out.solver_memory_mut() = fusion_solver::SolverMemory::from_contacts(memory);
        }
        if version >= 7 {
            match (r.u32()?, &template.core) {
                (0, None) => {}
                (1, Some(core)) => {
                    let length = usize::try_from(r.u64()?)
                        .map_err(|_| StateError::Invalid("core state too large"))?;
                    let decoded = phys_mechanics::MechanicsState::from_bytes(r.take(length)?, core)
                        .map_err(StateError::Invalid)?;
                    if decoded.time() != time {
                        return Err(StateError::Invalid("core checkpoint clocks disagree"));
                    }
                    out.core = Some(decoded);
                }
                _ => {
                    return Err(StateError::Invalid(
                        "checkpoint belongs to a different mechanics engine",
                    ))
                }
            }
        } else if template.core.is_some() {
            return Err(StateError::Invalid(
                "a state file before version 7 has no Fusion Core integration history",
            ));
        }
        if r.at != bytes.len() {
            return Err(StateError::Invalid("unexpected bytes after checkpoint"));
        }
        Ok(out)
    }
}
