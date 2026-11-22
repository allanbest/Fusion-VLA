//! Fusion Core behind [`Simulation`]: the f64 variational mechanics engine
//! from `crates/core/`, adapted to the same interface the Standard engine
//! runs behind.
//!
//! The core owns the authoritative f64 state. The `fusion_core::World` that
//! front-ends read is an output view written after every step and never read
//! back, which would round away the precision the core provides. Edits to it
//! are detected and refused ([`StepFailure::SceneChanged`]) rather than
//! silently discarded.
//!
//! [`CoreRunner`] is the same engine without the editor's views, for
//! throughput: training-data rollouts and benchmarks.
use crate::{EngineKind, Simulation, StepFailure, StepReport, SurfaceMesh};
use core_math::{Pose, Quat, Vec3};
use core_runtime::Pool;
use core_space::Shape;
use core_world::{JointDrive, Link, LinkPoint, Material, ObjectId, Representation, TetMesh, World};
pub use phys_mechanics::{
    Integrator as CoreIntegrator, MutualGravitySettings, Settings as CoreSettings,
    StepReport as CoreStepReport,
};
use phys_mechanics::{Mechanics, MechanicsState};

/// What Fusion Core declares about itself, in the Standard engine's terms.
pub(crate) fn solver_info(mut info: fusion_solver::SolverInfo) -> fusion_solver::SolverInfo {
    info.name = "Fusion Core V1";
    info.order = 2;
    info.stability = fusion_solver::Stability::Implicit;
    info.conserves = fusion_solver::Conserves {
        mass: true,
        momentum: false,
        energy: false,
    };
    info.tier = fusion_solver::AccuracyTier::RealTime;
    info.caveat =
        "Coupled f64 rigid/FEM mechanics with joints, springs, welds, islands & sleeping; \
                   TR-BDF2 / BDF2 / BackwardEuler / Newmark.";
    info
}

pub(crate) struct CoreEngine {
    world: World,
    mechanics: Mechanics,
    bodies: Vec<ObjectId>,
    solids: Vec<ObjectId>,
    /// Strain energy at the state last stepped to or restored, J. Cached
    /// because the uncached sum rebuilds every element's quadrature, and the
    /// editor reads the energy every frame. [`CoreRunner`] does not keep it
    /// current: it never reports energy.
    strain: f64,
    pub(crate) last: phys_mechanics::StepReport,
}

fn v(a: fusion_math::Vec3) -> Vec3 {
    Vec3::new(a.x as f64, a.y as f64, a.z as f64)
}
fn back(a: Vec3) -> fusion_math::Vec3 {
    fusion_math::vec3(a.x as f32, a.y as f32, a.z as f32)
}
fn q(a: fusion_math::Quat) -> Quat {
    Quat::from_xyzw(a.x as f64, a.y as f64, a.z as f64, a.w as f64).normalize()
}
fn shape(a: &fusion_core::Shape) -> Result<Shape, String> {
    match a {
        fusion_core::Shape::Sphere { radius } if radius.is_finite() && *radius > 0.0 => {
            Ok(Shape::Sphere {
                radius: *radius as f64,
            })
        }
        fusion_core::Shape::Cuboid { half_extents }
            if half_extents.is_finite() && half_extents.min_element() > 0.0 =>
        {
            Ok(Shape::Cuboid {
                half: v(*half_extents),
            })
        }
        _ => Err("Fusion Core supports positive-size boxes and spheres; \
                  imported hulls are not supported yet"
            .into()),
    }
}
fn dvec(a: Vec3) -> fusion_math::DVec3 {
    fusion_math::DVec3::new(a.x, a.y, a.z)
}
fn quat_back(a: Quat) -> fusion_math::Quat {
    fusion_math::Quat::from_xyzw(a.x as f32, a.y as f32, a.z as f32, a.w as f32)
}
fn material(
    name: &str,
    density: f64,
    youngs: f64,
    poisson: f64,
    friction: f64,
    restitution: f64,
) -> Result<Material, String> {
    if ![density, youngs, poisson, friction, restitution]
        .iter()
        .all(|x| x.is_finite())
        || density <= 0.0
        || youngs < 0.0
        || !(0.0..0.5).contains(&poisson)
        || friction < 0.0
        || !(0.0..=1.0).contains(&restitution)
    {
        return Err(format!(
            "{name}: invalid density, stiffness, Poisson ratio, friction or restitution"
        ));
    }
    Ok(Material::new(
        name,
        density,
        youngs,
        poisson,
        friction,
        restitution,
    ))
}

impl CoreEngine {
    fn build(sim: &Simulation) -> Result<Self, String> {
        sim.validate_core_features()?;
        let source = &sim.world;
        if source.solids.len() != source.solid_scene().len() {
            return Err("Reset continuum state before selecting Fusion Core".into());
        }
        let mut world = World::new();
        world.gravity = v(source.gravity);
        if !source.gravity.is_finite() || !source.bodies.is_finite() {
            return Err("Fusion Core requires finite initial state".into());
        }
        let mut bodies = Vec::new();
        for i in 0..source.bodies.len() {
            let b = &source.bodies;
            if b.is_removed(i) || b.is_kinematic(i) || b.failure[i] != fusion_core::Failure::Never {
                return Err(format!(
                    "body {i}: Fusion Core does not yet support kinematic bodies, \
                     fragmentation or representation switching"
                ));
            }
            let s = shape(&b.shape[i])?;
            let m = b.material[i];
            let name = format!("body {i}");
            let material = world.add_material(material(
                &name,
                m.density as f64,
                m.strength.youngs_modulus as f64,
                m.strength.poissons_ratio as f64,
                m.friction as f64,
                m.restitution as f64,
            )?);
            if !b.orientation[i].is_finite() || b.orientation[i].length_squared() < 0.5 {
                return Err(format!("body {i}: invalid orientation"));
            }
            let p = b.position[i];
            let pose = Pose::new(Vec3::new(p.x, p.y, p.z), q(b.orientation[i]));
            let id = if b.is_dynamic(i) {
                let id = world.add_rigid(&name, material, s, pose);
                let Representation::Rigid(r) = &mut world.object_mut(id).representation else {
                    unreachable!()
                };
                if !b.mass(i).is_finite()
                    || b.mass(i) <= 0.0
                    || !b.inertia_local[i].is_finite()
                    || b.inertia_local[i].min_element() <= 0.0
                {
                    return Err(format!(
                        "body {i}: mass and inertia must be finite and positive"
                    ));
                }
                r.mass = b.mass(i) as f64;
                r.inertia = core_math::Mat3::from_diagonal(v(b.inertia_local[i]));
                r.velocity = v(b.linear_velocity[i]);
                r.spin = v(b.angular_velocity[i]);
                id
            } else {
                world.add_fixed(&name, material, s, pose)
            };
            bodies.push(id);
        }
        let mut solids = Vec::new();
        for (i, desc) in source.solid_scene().iter().enumerate() {
            if !desc.pose.translation.is_finite()
                || !desc.pose.rotation.is_finite()
                || desc.pose.rotation.length_squared() < 0.5
                || !desc.velocity.is_finite()
            {
                return Err(format!("solid {i}: invalid initial pose or velocity"));
            }
            let p = desc.params;
            let m = p.material;
            if m.model != fusion_core::Constitutive::NeoHookean
                || m.fracture.breaks()
                || p.fibre.is_some()
                || m.thermal != fusion_core::Thermal::INERT
            {
                return Err(format!(
                    "solid {i}: Fusion Core currently supports isotropic neo-Hookean \
                     elasticity without fracture or thermal evolution"
                ));
            }
            let s = shape(&desc.shape)?;
            let edge = p.cell_size as f64;
            if !edge.is_finite() || edge <= 0.0 {
                return Err(format!(
                    "solid {i}: mesh cell size must be finite and positive"
                ));
            }
            let size = s.local_bounds().size();
            let cells = [size.x, size.y, size.z].map(|x| (x / edge).round().max(2.0));
            if cells.iter().product::<f64>() > 6000.0 {
                return Err(format!(
                    "solid {i}: mesh exceeds the 6000-cell preview budget; increase its cell size"
                ));
            }
            let mesh = match s {
                Shape::Cuboid { half } => TetMesh::cuboid(half * 2.0, edge),
                Shape::Sphere { radius } => TetMesh::ball(radius, edge),
                _ => unreachable!(),
            }
            .quadratic();
            let name = format!("solid {i}");
            let material = world.add_material(material(
                &name,
                m.density as f64,
                m.youngs_modulus as f64,
                m.poissons_ratio as f64,
                0.3,
                1.0,
            )?);
            let id = world.add_solid(
                &name,
                material,
                mesh,
                Pose::new(v(desc.pose.translation), q(desc.pose.rotation)),
            );
            world.set_velocity(id, v(desc.velocity), Vec3::ZERO);
            solids.push(id);
        }
        for joint in source.joints() {
            if !joint.enabled {
                continue;
            }
            let Some(&id_a) = bodies.get(joint.body_a.index()) else {
                return Err(format!(
                    "joint refers to body {}, out of range",
                    joint.body_a.index()
                ));
            };
            let Some(&id_b) = bodies.get(joint.body_b.index()) else {
                return Err(format!(
                    "joint refers to body {}, out of range",
                    joint.body_b.index()
                ));
            };
            let la = v(joint.local_anchor_a);
            let lb = v(joint.local_anchor_b);
            let fa = q(joint.local_frame_a);
            let fb = q(joint.local_frame_b);
            let rest_rotation = (fb * fa.conjugate()).normalize();
            let k_lin = if joint.compliance <= 0.0 || !joint.compliance.is_finite() {
                1.0e5
            } else {
                (1.0 / (joint.compliance as f64)).clamp(20.0, 1.0e5)
            };
            let k_rot = if joint.angular_compliance <= 0.0 || !joint.angular_compliance.is_finite()
            {
                2.0e4
            } else {
                (1.0 / (joint.angular_compliance as f64)).clamp(10.0, 2.0e4)
            };
            match joint.kind {
                fusion_core::JointKind::Ball => {
                    world.add_ball_joint(
                        LinkPoint::Body {
                            object: id_a,
                            local: la,
                        },
                        LinkPoint::Body {
                            object: id_b,
                            local: lb,
                        },
                        k_lin,
                    );
                }
                fusion_core::JointKind::Fixed => {
                    world.add_weld(id_a, Some(id_b), la, lb, rest_rotation, k_lin, k_rot);
                }
                fusion_core::JointKind::Hinge => {
                    let axis_a = fa.rotate(Vec3::X).normalize();
                    world.add_hinge(
                        id_a,
                        Some(id_b),
                        la,
                        lb,
                        axis_a,
                        rest_rotation,
                        k_lin,
                        k_rot,
                        None,
                        JointDrive::NONE,
                    );
                }
                fusion_core::JointKind::Slider { min, max } => {
                    let axis_b = fb.rotate(Vec3::X).normalize();
                    world.add_slider(
                        id_a,
                        Some(id_b),
                        la,
                        lb,
                        axis_b,
                        rest_rotation,
                        k_lin,
                        k_rot,
                        Some((min as f64, max as f64)),
                        JointDrive::NONE,
                    );
                }
                fusion_core::JointKind::Distance { min, max } => {
                    let rest = 0.5 * (min as f64 + max as f64);
                    let stiffness = if joint.compliance > 0.0 {
                        k_lin.min(5.0e3)
                    } else {
                        k_lin
                    };
                    let damping = if joint.compliance > 0.0 { 4.0 } else { 0.5 };
                    world.add_spring(
                        LinkPoint::Body {
                            object: id_a,
                            local: la,
                        },
                        LinkPoint::Body {
                            object: id_b,
                            local: lb,
                        },
                        rest.max(0.0),
                        stiffness,
                        damping,
                    );
                }
            }
        }
        let mut mechanics = Mechanics::new(sim.core_settings);
        mechanics.prepare(Pool::shared(), &world);
        let strain = mechanics.strain_energy(Pool::shared(), &world);
        Ok(Self {
            world,
            mechanics,
            bodies,
            solids,
            strain,
            last: Default::default(),
        })
    }
    // Views are read-only while Core is running. Reject changed topology or
    // physical inputs rather than silently discarding an external edit.
    fn matches_output(&self, out: &fusion_core::World) -> bool {
        let b = &out.bodies;
        if b.len() != self.bodies.len() || out.solids.len() != self.solids.len() {
            return false;
        }
        for (i, &id) in self.bodies.iter().enumerate() {
            let o = self.world.object(id);
            let m = self.world.material(o.material);
            if b.is_removed(i)
                || b.is_kinematic(i)
                || b.failure[i] != fusion_core::Failure::Never
                || b.material[i].density as f64 != m.density
                || b.material[i].friction as f64 != m.friction
                || b.material[i].restitution as f64 != m.restitution
            {
                return false;
            }
            let (s, p) = match &o.representation {
                Representation::Rigid(r) => {
                    let qr = r.pose.rotation;
                    if !b.is_dynamic(i)
                        || b.mass(i) as f64 != r.mass
                        || core_math::Mat3::from_diagonal(v(b.inertia_local[i])) != r.inertia
                        || b.linear_velocity[i] != back(r.velocity)
                        || b.angular_velocity[i] != back(r.spin)
                        || b.orientation[i] != quat_back(qr)
                    {
                        return false;
                    }
                    (r.shape, r.pose)
                }
                Representation::Fixed(f) => {
                    if b.is_dynamic(i) || q(b.orientation[i]) != f.pose.rotation {
                        return false;
                    }
                    (f.shape, f.pose)
                }
                _ => unreachable!(),
            };
            if shape(&b.shape[i]).ok() != Some(s) || b.position[i] != dvec(p.position) {
                return false;
            }
        }
        for (i, &id) in self.solids.iter().enumerate() {
            let Representation::Solid(s) = &self.world.object(id).representation else {
                unreachable!()
            };
            let view = &out.solids[i];
            if view.len() != s.nodes.len() || view.params != out.solid_scene()[i].params {
                return false;
            }
            for n in 0..s.nodes.len() {
                if view.particles.position[n] != dvec(s.nodes.position[n])
                    || view.particles.velocity[n] != back(s.nodes.velocity[n])
                    || view.particles.inv_mass[n] != (1.0 / s.nodes.mass[n]) as f32
                {
                    return false;
                }
            }
        }
        true
    }
    fn state_into(&self, out: &mut Vec<f64>) {
        out.clear();
        for o in self.world.objects() {
            match &o.representation {
                Representation::Rigid(r) => {
                    out.extend([
                        r.pose.position.x,
                        r.pose.position.y,
                        r.pose.position.z,
                        r.pose.rotation.x,
                        r.pose.rotation.y,
                        r.pose.rotation.z,
                        r.pose.rotation.w,
                        r.velocity.x,
                        r.velocity.y,
                        r.velocity.z,
                        r.spin.x,
                        r.spin.y,
                        r.spin.z,
                    ]);
                }
                Representation::Solid(s) => {
                    for &p in s.nodes.position.iter().chain(&s.nodes.velocity) {
                        out.extend([p.x, p.y, p.z]);
                    }
                }
                Representation::Fixed(_) => {}
            }
        }
        self.mechanics.append_integration_state(out);
    }
    /// One interval of `h` seconds. A failed solve leaves state and time as
    /// they were.
    fn advance(&mut self, pool: &Pool, h: f64, linear_damping: f64, angular_damping: f64) {
        self.last = self.mechanics.step(pool, &mut self.world, h);
        if self.last.failed() {
            return;
        }
        for object in self.world.objects_mut() {
            if let Representation::Rigid(r) = &mut object.representation {
                r.velocity /= 1.0 + linear_damping * h;
                r.spin /= 1.0 + angular_damping * h;
            }
        }
    }
    pub(crate) fn apply_impulse_at(
        &mut self,
        body: usize,
        impulse: fusion_math::Vec3,
        at: fusion_math::Vec3,
        out: &mut fusion_core::World,
    ) -> bool {
        let Some(&id) = self.bodies.get(body) else {
            return false;
        };
        let j = v(impulse);
        let p = v(at);
        if !j.is_finite() || !p.is_finite() {
            return false;
        }
        let applied = {
            let Representation::Rigid(r) = &mut self.world.object_mut(id).representation else {
                return false;
            };
            if r.mass <= 0.0 || !r.mass.is_finite() {
                return false;
            }
            r.velocity += j / r.mass;
            if let Some(inv_local) = r.inertia.try_inverse() {
                let rot = r.pose.rotation;
                let tau_world = (p - r.pose.position).cross(j);
                r.spin += rot.rotate(inv_local * rot.conjugate().rotate(tau_world));
            }
            true
        };
        if applied {
            self.sync(out);
        }
        applied
    }

    pub(crate) fn pull_solid_near(
        &mut self,
        solid: usize,
        at: fusion_math::Vec3,
        delta_v: fusion_math::Vec3,
        radius: f32,
        out: &mut fusion_core::World,
    ) -> bool {
        let Some(&id) = self.solids.get(solid) else {
            return false;
        };
        let target = v(at);
        let dv = v(delta_v);
        let r = (radius as f64).max(0.05);
        if !target.is_finite() || !dv.is_finite() {
            return false;
        }
        let applied = {
            let Representation::Solid(s) = &mut self.world.object_mut(id).representation else {
                return false;
            };
            if s.nodes.is_empty() {
                return false;
            }
            let r2 = r * r;
            let mut touched = 0usize;
            for (pos, vel) in s.nodes.position.iter().zip(s.nodes.velocity.iter_mut()) {
                let d2 = (*pos - target).length_squared();
                if d2 <= r2 {
                    let w = 1.0 - (d2.sqrt() / r);
                    *vel += dv * (0.25 + 0.75 * w);
                    touched += 1;
                }
            }
            if touched == 0 {
                if let Some((best, _)) =
                    s.nodes
                        .position
                        .iter()
                        .enumerate()
                        .min_by(|(_, a), (_, b)| {
                            (**a - target)
                                .length_squared()
                                .total_cmp(&(**b - target).length_squared())
                        })
                {
                    s.nodes.velocity[best] += dv;
                }
            }
            true
        };
        if applied {
            self.sync(out);
        }
        applied
    }

    pub(crate) fn checkpoint(&self) -> MechanicsState {
        self.mechanics.checkpoint(&self.world)
    }
    pub(crate) fn restore(&mut self, state: &MechanicsState, out: &mut fusion_core::World) {
        self.mechanics.restore_checkpoint(&mut self.world, state);
        self.last = Default::default();
        self.strain = self.mechanics.strain_energy(Pool::shared(), &self.world);
        self.sync(out);
    }
    fn sync(&self, out: &mut fusion_core::World) {
        for (i, &id) in self.bodies.iter().enumerate() {
            if let Representation::Rigid(r) = &self.world.object(id).representation {
                out.bodies.position[i] = dvec(r.pose.position);
                out.bodies.orientation[i] = quat_back(r.pose.rotation);
                out.bodies.linear_velocity[i] = back(r.velocity);
                out.bodies.angular_velocity[i] = back(r.spin);
                out.bodies.sleeping[i] = false;
                out.bodies.sleep_timer[i] = 0.0;
            }
        }
        for (i, &id) in self.solids.iter().enumerate() {
            let Representation::Solid(s) = &self.world.object(id).representation else {
                unreachable!()
            };
            let out = &mut out.solids[i];
            if out.len() != s.nodes.len() {
                *out = fusion_core::SolidSet::new(out.params);
                for &p in &s.nodes.position {
                    out.push(back(p));
                }
            }
            let deformation = phys_mechanics::nodal_deformation(s);
            for (n, f) in deformation.iter().enumerate() {
                out.particles.position[n] = dvec(s.nodes.position[n]);
                out.particles.velocity[n] = back(s.nodes.velocity[n]);
                out.particles.inv_mass[n] = (1.0 / s.nodes.mass[n]) as f32;
                out.deformation[n] =
                    fusion_math::Mat3::from_cols(back(f.cols[0]), back(f.cols[1]), back(f.cols[2]));
                out.volume[n] = (s.nodes.mass[n] / out.params.material.density as f64) as f32;
            }
        }
    }
    pub(crate) fn continuum_energy(&self) -> fusion_mpm::ContinuumEnergy {
        let mut energy = fusion_mpm::ContinuumEnergy {
            strain: self.strain,
            ..Default::default()
        };
        for &id in &self.solids {
            let o = self.world.object(id);
            energy.kinetic += o.kinetic_energy();
            energy.potential += o.gravitational_energy(self.world.gravity);
        }
        energy
    }
    pub(crate) fn mechanical_energy(&self) -> f64 {
        let t = self.world.totals();
        t.kinetic_energy + t.gravitational_energy + self.strain
    }
    fn surface(&self) -> SurfaceMesh {
        let mut out = SurfaceMesh::default();
        for &id in &self.solids {
            let Representation::Solid(s) = &self.world.object(id).representation else {
                unreachable!()
            };
            let offset = out.positions.len() as u32;
            out.positions
                .extend(s.nodes.position.iter().map(|&p| back(p)));
            out.indices
                .extend(s.surface.faces.iter().flatten().map(|&n| n + offset));
        }
        out.normals
            .resize(out.positions.len(), fusion_math::Vec3::ZERO);
        for t in out.indices.chunks_exact(3) {
            let [a, b, c] = [t[0] as usize, t[1] as usize, t[2] as usize];
            let n =
                (out.positions[b] - out.positions[a]).cross(out.positions[c] - out.positions[a]);
            for i in [a, b, c] {
                out.normals[i] += n;
            }
        }
        for n in &mut out.normals {
            *n = n.normalize_or_zero();
        }
        out
    }
}

impl Simulation {
    /// Flatten the authoritative state and integration history, matching CoreRunner.
    /// Returns false on the Standard engine.
    pub fn core_state_into(&self, out: &mut Vec<f64>) -> bool {
        if let Some(core) = &self.core {
            core.state_into(out);
            true
        } else {
            out.clear();
            false
        }
    }
    /// The complete mechanics backend currently selected.
    pub fn engine_kind(&self) -> EngineKind {
        self.engine
    }
    fn validate_core_features(&self) -> Result<(), String> {
        if !self.world.gravity.is_finite()
            || !self.world.linear_damping.is_finite()
            || self.world.linear_damping < 0.0
            || !self.world.angular_damping.is_finite()
            || self.world.angular_damping < 0.0
        {
            return Err(
                "Fusion Core requires finite gravity and nonnegative finite damping".into(),
            );
        }
        let active_components = self
            .components
            .iter()
            .any(|c| c.components.controller.is_some() || c.components.force.is_some());
        if !self.world.fluids.is_empty()
            || !self.world.particles.is_empty()
            || !self.scripts.is_empty()
            || active_components
            || self.heat.enabled
            || self.stress.enabled
            || self.air.enabled
        {
            return Err(
                "Fusion Core supports rigid bodies, joints/springs/welds, and elastic solids. \
                 Fluids, cloth, scripts, force controllers, air, heat and rigid stress need the \
                 Standard engine."
                    .into(),
            );
        }
        Ok(())
    }
    /// Select and prepare the new core at t=0. Unsupported physics returns a
    /// descriptive error; the previous runtime remains selected on failure.
    pub fn use_core_engine(&mut self) -> Result<(), String> {
        if self.time != 0.0 {
            return Err("Reset the simulation before changing its engine".into());
        }
        let core = CoreEngine::build(self)?;
        core.sync(&mut self.world);
        self.core = Some(core);
        self.engine = EngineKind::Core;
        Ok(())
    }
    /// Return to the Standard engine at t = 0, rebuilding its initial state.
    pub fn use_standard_engine(&mut self) -> Result<(), String> {
        if self.engine == EngineKind::Standard {
            return Ok(());
        }
        if self.time != 0.0 {
            return Err("Reset the simulation before changing its engine".into());
        }
        self.engine = EngineKind::Standard;
        self.reset();
        Ok(())
    }
    /// Fusion Core's numerical settings; `None` on the Standard engine.
    pub fn core_settings_mut(&mut self) -> Option<&mut CoreSettings> {
        self.core.as_ref().map(|_| &mut self.core_settings)
    }
    /// Apply a world-space runtime impulse at `at` (e.g. interactive mouse drag),
    /// waking sleeping islands seamlessly on both Standard and Fusion Core.
    pub fn apply_impulse_at(
        &mut self,
        body: usize,
        impulse: fusion_math::Vec3,
        at: fusion_math::Vec3,
    ) -> bool {
        if body >= self.world.bodies.len()
            || !impulse.is_finite()
            || !at.is_finite()
            || !self.world.bodies.is_dynamic(body)
        {
            return false;
        }
        match self.engine {
            EngineKind::Standard => {
                self.world.bodies.wake(body);
                self.world.bodies.apply_impulse_at(body, impulse, at);
                true
            }
            EngineKind::Core => match self.core.as_mut() {
                Some(core) => core.apply_impulse_at(body, impulse, at, &mut self.world),
                None => false,
            },
        }
    }
    /// Apply a localized runtime velocity pull on deformable solid nodes near `at`
    /// across both Standard MPM and Fusion Core Q2 FEM solids.
    pub fn pull_solid_near(
        &mut self,
        solid: usize,
        at: fusion_math::Vec3,
        delta_v: fusion_math::Vec3,
        radius: f32,
    ) -> bool {
        if !at.is_finite() || !delta_v.is_finite() || !radius.is_finite() {
            return false;
        }
        match self.engine {
            EngineKind::Standard => {
                let Some(set) = self.world.solids.get_mut(solid) else {
                    return false;
                };
                if set.is_empty() {
                    return false;
                }
                let r = radius.max(0.05) as f64;
                let r2 = r * r;
                let target = fusion_math::DVec3::new(at.x as f64, at.y as f64, at.z as f64);
                let mut touched = 0usize;
                for (pos, vel) in set
                    .particles
                    .position
                    .iter()
                    .zip(set.particles.velocity.iter_mut())
                {
                    let d2 = (*pos - target).length_squared();
                    if d2 <= r2 {
                        let w = (1.0 - (d2.sqrt() / r)) as f32;
                        *vel += delta_v * (0.25 + 0.75 * w);
                        touched += 1;
                    }
                }
                if touched == 0 {
                    if let Some((best, _)) =
                        set.particles
                            .position
                            .iter()
                            .enumerate()
                            .min_by(|(_, a), (_, b)| {
                                (**a - target)
                                    .length_squared()
                                    .total_cmp(&(**b - target).length_squared())
                            })
                    {
                        set.particles.velocity[best] += delta_v;
                    }
                }
                true
            }
            EngineKind::Core => match self.core.as_mut() {
                Some(core) => core.pull_solid_near(solid, at, delta_v, radius, &mut self.world),
                None => false,
            },
        }
    }
    /// Exact finite-element boundary mesh, rather than a particle isosurface.
    pub fn core_surface(&self) -> Option<SurfaceMesh> {
        self.core
            .as_ref()
            .filter(|_| self.engine == EngineKind::Core)
            .map(CoreEngine::surface)
    }
    /// Work and convergence of the last new-core mechanics step.
    pub fn core_report(&self) -> Option<phys_mechanics::StepReport> {
        self.core.as_ref().map(|c| c.last)
    }
    /// A step report for an interval that was refused: nothing advanced.
    fn refused(&self, failure: StepFailure) -> StepReport {
        StepReport {
            time: self.time,
            step_index: self.step_index,
            failure: Some(failure),
            ..Default::default()
        }
    }
    pub(crate) fn step_core(&mut self) -> StepReport {
        let start = std::time::Instant::now();
        if self.validate_core_features().is_err() {
            return self.refused(StepFailure::InvalidConfiguration);
        }
        let settings = self.core_settings;
        let Some(c) = self.core.as_mut() else {
            return self.refused(StepFailure::InvalidConfiguration);
        };
        if !c.matches_output(&self.world) {
            return self.refused(StepFailure::SceneChanged);
        }
        c.mechanics.settings = settings;
        c.world.gravity = v(self.world.gravity);
        c.advance(
            Pool::shared(),
            self.solver_dt as f64,
            self.world.linear_damping as f64,
            self.world.angular_damping as f64,
        );
        if c.last.failed() {
            return StepReport {
                solve_time: start.elapsed(),
                ..self.refused(StepFailure::Convergence)
            };
        }
        c.sync(&mut self.world);
        self.time = c.world.time;
        self.step_index += 1;
        let totals = c.world.totals();
        c.strain = c.mechanics.strain_energy(Pool::shared(), &c.world);
        let strain = c.strain;
        let total = (totals.kinetic_energy + totals.gravitational_energy + strain) as f32;
        let delta = self.last_energy.map_or(0.0, |previous| total - previous);
        self.last_energy = Some(total);
        let report = StepReport {
            step_index: self.step_index,
            time: self.time,
            dt: self.solver_dt,
            kinetic_energy: totals.kinetic_energy as f32,
            potential_energy: totals.gravitational_energy as f32,
            strain_energy: strain as f32,
            energy_delta: delta,
            solve_time: start.elapsed(),
            contact_points: c.last.grips,
            ..Default::default()
        };
        self.diagnostics.observe(&report);
        report
    }
}

/// A named contiguous field in the f64 training-state vector.
#[derive(Clone, Debug)]
pub struct CoreStateField {
    /// Meaning of the field; object indices refer to `CoreRunner::world`.
    pub name: String,
    /// First scalar in the flattened vector.
    pub offset: usize,
    /// Number of f64 scalars.
    pub width: usize,
}

/// The app's core physics without editor views, surface fields or per-step
/// diagnostic energy. Each runner owns its pool, so independent trajectories
/// can run in parallel without competing for a shared executor.
pub struct CoreRunner {
    core: CoreEngine,
    pool: Pool,
    initial: MechanicsState,
    dt: f64,
    linear_damping: f64,
    angular_damping: f64,
}
impl CoreRunner {
    /// Read the authoritative f64 state and static geometry/materials.
    pub fn world(&self) -> &World {
        &self.core.world
    }
    /// Internal step duration, seconds.
    pub fn timestep(&self) -> f64 {
        self.dt
    }
    /// Advance one interval. Failure preserves time and state; do not emit
    /// a training target for a failed result.
    pub fn step(&mut self) -> phys_mechanics::StepReport {
        self.core.advance(
            &self.pool,
            self.dt,
            self.linear_damping,
            self.angular_damping,
        );
        self.core.last
    }
    /// Restore the initial trajectory without rebuilding meshes or solvers.
    /// Add independent, seeded velocity perturbations to each moving object.
    /// Each velocity component is uniform in [-speed,speed] m/s; each spin
    /// component is uniform in [-spin,spin] rad/s. Zero retains initial motion.
    pub fn reset(&mut self, seed: u64, speed: f64, spin: f64) -> Result<(), String> {
        if !speed.is_finite() || speed < 0.0 || !spin.is_finite() || spin < 0.0 {
            return Err("velocity/spin scales must be finite and nonnegative".into());
        }
        self.core
            .mechanics
            .restore_checkpoint(&mut self.core.world, &self.initial);
        let mut rng = core_math::Rng::new(seed);
        for o in self.core.world.objects_mut() {
            let dv = Vec3::new(
                rng.range(-speed, speed),
                rng.range(-speed, speed),
                rng.range(-speed, speed),
            );
            let dw = Vec3::new(
                rng.range(-spin, spin),
                rng.range(-spin, spin),
                rng.range(-spin, spin),
            );
            match &mut o.representation {
                Representation::Rigid(r) => {
                    r.velocity += dv;
                    r.spin += dw;
                }
                Representation::Solid(s) => {
                    let mass: f64 = s.nodes.mass.iter().sum();
                    let centre = s
                        .nodes
                        .position
                        .iter()
                        .zip(&s.nodes.mass)
                        .map(|(&p, &m)| p * m)
                        .sum::<Vec3>()
                        / mass;
                    for (v, &p) in s.nodes.velocity.iter_mut().zip(&s.nodes.position) {
                        *v += dv + dw.cross(p - centre);
                    }
                }
                Representation::Fixed(_) => {}
            }
        }
        self.core.last = Default::default();
        Ok(())
    }
    /// Flatten physical state and integration history into a reused buffer.
    /// Including history avoids treating TR-BDF2 as a memoryless method.
    pub fn state_into(&self, out: &mut Vec<f64>) {
        self.core.state_into(out);
    }
    /// Scalar ranges for each object field and the fixed-width history suffix.
    pub fn state_layout(&self) -> Vec<CoreStateField> {
        let mut fields = Vec::new();
        let mut offset = 0;
        for (i, o) in self.core.world.objects().iter().enumerate() {
            let parts: Vec<(&str, usize)> = match &o.representation {
                Representation::Rigid(_) => vec![
                    ("position", 3),
                    ("orientation_xyzw", 4),
                    ("velocity", 3),
                    ("spin", 3),
                ],
                Representation::Solid(s) => vec![
                    ("node_position_xyz", 3 * s.nodes.len()),
                    ("node_velocity_xyz", 3 * s.nodes.len()),
                ],
                Representation::Fixed(_) => Vec::new(),
            };
            for (name, width) in parts {
                fields.push(CoreStateField {
                    name: format!("object.{i}.{name}"),
                    offset,
                    width,
                });
                offset += width;
            }
        }
        let (blocks, rigids) = self.core.mechanics.integration_shape();
        for (name, width) in [
            ("history.acceleration_present", 1),
            ("history.acceleration", 3 * blocks),
            ("history.earlier_present", 1),
            ("history.earlier_position", 3 * blocks),
            ("history.earlier_velocity", 3 * blocks),
            ("history.restart", blocks),
        ] {
            fields.push(CoreStateField {
                name: name.into(),
                offset,
                width,
            });
            offset += width;
        }
        for i in 0..rigids {
            for (name, width) in [
                ("present", 1),
                ("orientation_xyzw", 4),
                ("matrix_column_major", 9),
            ] {
                fields.push(CoreStateField {
                    name: format!("history.affine.{i}.{name}"),
                    offset,
                    width,
                });
                offset += width;
            }
        }
        for (name, width) in [
            ("history.gravity_present", 1),
            ("history.gravity_xyz", 3),
            ("history.previous_step_present", 1),
            ("history.previous_step_seconds", 1),
        ] {
            fields.push(CoreStateField {
                name: name.into(),
                offset,
                width,
            });
            offset += width;
        }
        for i in 0..rigids {
            fields.push(CoreStateField {
                name: format!("history.contact_unloading.{i}.joules"),
                offset,
                width: 1,
            });
            offset += 1;
        }
        fields
    }
    /// Enable optional section timing (disabled by default for throughput).
    pub fn set_profiling(&mut self, enabled: bool) {
        self.core.mechanics.profile.set_enabled(enabled);
    }
    /// Clear accumulated profiling before a measured interval.
    pub fn clear_profile(&mut self) {
        self.core.mechanics.profile.clear();
    }
    /// Timing by engine section. Entries overlap (the whole solve includes terms).
    pub fn profile(&self) -> String {
        self.core.mechanics.profile.report()
    }
}
impl Simulation {
    /// Move a freshly built Core simulation into the throughput runtime.
    /// The same mechanics, timestep, damping and precision are retained.
    pub fn into_core_runner(mut self, threads: usize) -> Result<CoreRunner, String> {
        if self.engine != EngineKind::Core || self.time != 0.0 || threads == 0 {
            return Err("a Core runner needs a fresh Core scene and at least one thread".into());
        }
        let mut core = self.core.take().ok_or("Core runtime is not prepared")?;
        core.mechanics.profile.set_enabled(false);
        let initial = core.checkpoint();
        Ok(CoreRunner {
            core,
            pool: Pool::new(threads),
            initial,
            dt: self.solver_dt as f64,
            linear_damping: self.world.linear_damping as f64,
            angular_damping: self.world.angular_damping as f64,
        })
    }
}

/// One rigid link extracted from a parsed URDF model.
#[derive(Clone, Debug)]
pub struct ImportedUrdfLink {
    /// Link name from the URDF `<link>` tag.
    pub name: String,
    /// Bounding rigid shape converted into editor coordinates.
    pub shape: fusion_core::Shape,
    /// World-space link origin position in metres.
    pub position: [f32; 3],
    /// World-space link orientation `[x, y, z, w]`.
    pub orientation: [f32; 4],
    /// True if the link is static/fixed (`Representation::Fixed`).
    pub fixed: bool,
    /// Mass in kg (`None` for fixed links).
    pub mass: Option<f32>,
}

/// One mechanical joint extracted from a parsed URDF model.
#[derive(Clone, Debug)]
pub struct ImportedUrdfJoint {
    /// Index into [`ImportedUrdf::links`] for body A.
    pub body_a: usize,
    /// Index into [`ImportedUrdf::links`] for body B.
    pub body_b: usize,
    /// Joint anchor in body A's local frame.
    pub anchor_a: [f32; 3],
    /// Joint anchor in body B's local frame.
    pub anchor_b: [f32; 3],
    /// Joint frame orientation `[x, y, z, w]` on body A.
    pub frame_a: [f32; 4],
    /// Joint frame orientation `[x, y, z, w]` on body B.
    pub frame_b: [f32; 4],
    /// Mapped Fusion joint kind.
    pub kind: fusion_core::JointKind,
}

/// Parsed URDF robot description ready to spawn into a scene or project.
#[derive(Clone, Debug)]
pub struct ImportedUrdf {
    /// Robot name from the `<robot name="...">` attribute.
    pub name: String,
    /// Rigid links in spawn order.
    pub links: Vec<ImportedUrdfLink>,
    /// Joints linking adjacent [`ImportedUrdf::links`].
    pub joints: Vec<ImportedUrdfJoint>,
}

fn quat_from_x_to(target: Vec3) -> Quat {
    let dir = target.normalize();
    let dot = Vec3::X.dot(dir);
    if dot > 1.0 - 1e-10 {
        Quat::IDENTITY
    } else if dot < -1.0 + 1e-10 {
        Quat::from_axis_angle(Vec3::Y, std::f64::consts::PI)
    } else {
        let axis = Vec3::X.cross(dir).normalize();
        Quat::from_axis_angle(axis, dot.clamp(-1.0, 1.0).acos())
    }
}

/// Parse URDF XML via `core_world::World::load_urdf` into rigid links and joints.
pub fn import_urdf_xml(xml: &str) -> Result<ImportedUrdf, String> {
    let mut world = World::new();
    let mat = world.add_material(Material::new("urdf", 1000.0, 1.0e5, 0.3, 0.5, 0.0));
    let model = world.load_urdf(xml, mat, Pose::IDENTITY, true)?;

    let mut links = Vec::with_capacity(world.objects().len());
    for obj in world.objects() {
        let (shape, pose, fixed, mass) = match &obj.representation {
            Representation::Fixed(f) => (f.shape, f.pose, true, None),
            Representation::Rigid(r) => (r.shape, r.pose, false, Some(r.mass as f32)),
            Representation::Solid(_) => continue,
        };
        let converted = match shape {
            Shape::Sphere { radius } => fusion_core::Shape::Sphere {
                radius: radius as f32,
            },
            Shape::Cuboid { half } => fusion_core::Shape::Cuboid {
                half_extents: back(half),
            },
            Shape::Capsule {
                radius,
                half_length,
            } => fusion_core::Shape::Cuboid {
                half_extents: fusion_math::vec3(
                    radius as f32,
                    (radius + half_length) as f32,
                    radius as f32,
                ),
            },
            Shape::HalfSpace { .. } => fusion_core::Shape::Cuboid {
                half_extents: fusion_math::vec3(10.0, 0.5, 10.0),
            },
        };
        links.push(ImportedUrdfLink {
            name: obj.name.clone(),
            shape: converted,
            position: [
                pose.position.x as f32,
                pose.position.y as f32,
                pose.position.z as f32,
            ],
            orientation: [
                pose.rotation.x as f32,
                pose.rotation.y as f32,
                pose.rotation.z as f32,
                pose.rotation.w as f32,
            ],
            fixed,
            mass,
        });
    }

    let mut joints = Vec::new();
    for link in world.links() {
        match link {
            Link::BallJoint {
                a:
                    LinkPoint::Body {
                        object: a,
                        local: la,
                    },
                b:
                    LinkPoint::Body {
                        object: b,
                        local: lb,
                    },
                ..
            } => {
                joints.push(ImportedUrdfJoint {
                    body_a: a.0 as usize,
                    body_b: b.0 as usize,
                    anchor_a: [la.x as f32, la.y as f32, la.z as f32],
                    anchor_b: [lb.x as f32, lb.y as f32, lb.z as f32],
                    frame_a: [0.0, 0.0, 0.0, 1.0],
                    frame_b: [0.0, 0.0, 0.0, 1.0],
                    kind: fusion_core::JointKind::Ball,
                });
            }
            Link::Weld {
                a,
                b: Some(b),
                local_a,
                local_b,
                rest_rotation,
                ..
            } => {
                let fb = *rest_rotation;
                joints.push(ImportedUrdfJoint {
                    body_a: a.0 as usize,
                    body_b: b.0 as usize,
                    anchor_a: [local_a.x as f32, local_a.y as f32, local_a.z as f32],
                    anchor_b: [local_b.x as f32, local_b.y as f32, local_b.z as f32],
                    frame_a: [0.0, 0.0, 0.0, 1.0],
                    frame_b: [fb.x as f32, fb.y as f32, fb.z as f32, fb.w as f32],
                    kind: fusion_core::JointKind::Fixed,
                });
            }
            Link::Hinge {
                a,
                b: Some(b),
                local_a,
                local_b,
                axis_a,
                rest_rotation,
                ..
            } => {
                let fa = quat_from_x_to(*axis_a);
                let fb = (*rest_rotation * fa).normalize();
                joints.push(ImportedUrdfJoint {
                    body_a: a.0 as usize,
                    body_b: b.0 as usize,
                    anchor_a: [local_a.x as f32, local_a.y as f32, local_a.z as f32],
                    anchor_b: [local_b.x as f32, local_b.y as f32, local_b.z as f32],
                    frame_a: [fa.x as f32, fa.y as f32, fa.z as f32, fa.w as f32],
                    frame_b: [fb.x as f32, fb.y as f32, fb.z as f32, fb.w as f32],
                    kind: fusion_core::JointKind::Hinge,
                });
            }
            Link::Slider {
                a,
                b: Some(b),
                local_a,
                local_b,
                axis_b,
                rest_rotation,
                limits,
                ..
            } => {
                let fb = quat_from_x_to(*axis_b);
                let fa = (rest_rotation.conjugate() * fb).normalize();
                let (min, max) = limits
                    .map(|(lo, hi)| (lo as f32, hi as f32))
                    .unwrap_or((-1.0, 1.0));
                joints.push(ImportedUrdfJoint {
                    body_a: a.0 as usize,
                    body_b: b.0 as usize,
                    anchor_a: [local_a.x as f32, local_a.y as f32, local_a.z as f32],
                    anchor_b: [local_b.x as f32, local_b.y as f32, local_b.z as f32],
                    frame_a: [fa.x as f32, fa.y as f32, fa.z as f32, fa.w as f32],
                    frame_b: [fb.x as f32, fb.y as f32, fb.z as f32, fb.w as f32],
                    kind: fusion_core::JointKind::Slider { min, max },
                });
            }
            _ => {}
        }
    }

    Ok(ImportedUrdf {
        name: model.name,
        links,
        joints,
    })
}
