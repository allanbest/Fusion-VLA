//! The world: a scene, and the state built from it.
//!
//! PLAN.md §16 makes the central user-facing distinction:
//!
//! ```text
//! Scene (declarative, small, saved) -> build -> State (SoA arrays, huge, transient)
//! ```
//!
//! The scene describes the world at t = 0; the state is what the solver chews
//! on. `reset` throws the state away and rebuilds it from the scene, which is
//! also why a project file will be enough to reproduce a run exactly (§22).

use fusion_math::{Isometry, Quat, Vec3, GRAVITY};

use crate::body::{BodyHandle, BodySet, BodyType};
use crate::cloth::{build_cloth, build_soft_body, ClothDesc, ClothHandles, SoftBodyDesc};
use crate::domain::Domain;
use crate::fluid::{build_fluid, FluidDesc, FluidHandle, FluidSet};
use crate::joint::{Joint, JointHandle, JointKind};
use crate::material::Material;
use crate::particle::{
    BendingConstraint, DistanceConstraint, ParticleHandle, ParticleSet, SurfaceTriangle,
};
use crate::shape::Shape;
use crate::solid::{build_solid, SolidDesc, SolidHandle, SolidSet};

/// Declarative description of one body — a line in the scene, not a line in
/// the state.
#[derive(Clone, Debug)]
pub struct BodyDesc {
    /// The collision shape.
    pub shape: Shape,
    /// Density, friction and restitution. Mass comes from density and volume
    /// unless [`BodyDesc::with_mass`] overrides it.
    pub material: Material,
    /// Where to put it.
    pub position: Vec3,
    /// How it is oriented.
    pub orientation: Quat,
    /// Initial linear velocity, m/s.
    pub linear_velocity: Vec3,
    /// Initial angular velocity, rad/s.
    pub angular_velocity: Vec3,
    /// Whether the solver may move it.
    pub body_type: BodyType,
    /// Overrides the mass implied by shape volume and material density.
    pub mass: Option<f32>,
    /// Starting temperature, Kelvin. Absent means the material's resting
    /// temperature — room temperature, or just below melting for something
    /// that melts below it. Only the heat solve reads it.
    pub temperature: Option<f32>,
    /// What it does when it is loaded past its strength. Stays whole unless
    /// told otherwise.
    pub failure: crate::Failure,
    /// Which way its grain runs, in its own frame. Absent means along its
    /// longest side; see [`crate::body::default_grain`]. Read only for a
    /// material with a grain.
    pub grain: Option<Vec3>,
    /// Which tier it is solved in: Advanced (the default) gets stress,
    /// breaking and temperature when the scene has them on; Basic gets none
    /// of them, whatever the scene says, and costs a plain rigid body. See
    /// `docs/FUSION_SOLVER.md`, "Basic and Advanced".
    pub advanced: bool,
}

impl BodyDesc {
    /// A dynamic body of this shape at the origin, of generic material.
    pub fn new(shape: Shape) -> Self {
        Self {
            shape,
            material: Material::GENERIC,
            position: Vec3::ZERO,
            orientation: Quat::IDENTITY,
            linear_velocity: Vec3::ZERO,
            angular_velocity: Vec3::ZERO,
            body_type: BodyType::Dynamic,
            mass: None,
            temperature: None,
            failure: crate::Failure::Never,
            grain: None,
            advanced: true,
        }
    }

    /// Shorthand for a sphere of `radius`.
    pub fn sphere(radius: f32) -> Self {
        Self::new(Shape::sphere(radius))
    }

    /// Box from its full size.
    pub fn cuboid(size: Vec3) -> Self {
        Self::new(Shape::cuboid(size))
    }

    /// Builder: place it here.
    pub fn at(mut self, position: Vec3) -> Self {
        self.position = position;
        self
    }

    /// Builder: orient it like this.
    pub fn rotated(mut self, orientation: Quat) -> Self {
        self.orientation = orientation;
        self
    }

    /// Builder: give it this material.
    pub fn with_material(mut self, material: Material) -> Self {
        self.material = material;
        self
    }

    /// Builder: start it moving.
    pub fn with_velocity(mut self, v: Vec3) -> Self {
        self.linear_velocity = v;
        self
    }

    /// Builder: start it spinning.
    pub fn with_angular_velocity(mut self, w: Vec3) -> Self {
        self.angular_velocity = w;
        self
    }

    /// Builder: start it at this temperature, Kelvin.
    pub fn with_temperature(mut self, kelvin: f32) -> Self {
        self.temperature = Some(kelvin);
        self
    }

    /// Builder: let it break, or deform and melt, when it is overloaded. See
    /// [`crate::Failure`].
    pub fn with_failure(mut self, failure: crate::Failure) -> Self {
        self.failure = failure;
        self
    }

    /// Builder: solve it in the Basic tier — a plain rigid body, with no
    /// stress, breaking or temperature, whatever the scene has switched on.
    pub fn basic(mut self) -> Self {
        self.advanced = false;
        self
    }

    /// Builder: run its grain along `direction`, in its own frame.
    pub fn with_grain(mut self, direction: Vec3) -> Self {
        self.grain = Some(direction.normalize_or_zero());
        self
    }

    /// Builder: override the mass the density would have implied, in kg.
    /// Inertia is rescaled with it, so the shape still spins like itself.
    pub fn with_mass(mut self, mass: f32) -> Self {
        self.mass = Some(mass);
        self
    }

    /// Immovable: infinite mass, ignores gravity and every correction.
    pub fn fixed(mut self) -> Self {
        self.body_type = BodyType::Static;
        self
    }

    /// Driven from outside: infinite mass, but its pose is written each step
    /// and its velocity is derived from the motion that produced it.
    pub fn driven(mut self) -> Self {
        self.body_type = BodyType::Kinematic;
        self
    }

    /// Position and orientation together.
    pub fn pose(&self) -> Isometry {
        Isometry::new(self.position, self.orientation)
    }

    /// Inverse mass and inverse inertia this description implies.
    fn inverses(&self) -> (f32, Vec3) {
        // Kinematic bodies are infinite-mass too. The difference between the
        // two is what maintains their velocity, not what the solver does with
        // their mass — see [`BodyType::Kinematic`].
        if !self.body_type.is_dynamic() {
            return (0.0, Vec3::ZERO);
        }
        let mut props = self.shape.mass_properties(self.material.density);
        if let Some(m) = self.mass {
            // Scale the tensor with the mass — inertia is linear in mass for a
            // fixed shape, so an override rescales rather than invalidates it.
            let scale = if props.mass > 0.0 {
                m / props.mass
            } else {
                0.0
            };
            props.inertia_diagonal *= scale;
            props.mass = m;
        }
        props.to_inverses()
    }
}

/// A simulation world: parameters, the scene, and the state built from it.
#[derive(Clone, Debug)]
pub struct World {
    /// Uniform acceleration on everything with mass, m/s^2.
    pub gravity: Vec3,
    /// Per-second velocity retention factors. v0.1 applies damping globally;
    /// per-material damping arrives with materials-as-data in v0.2.
    pub linear_damping: f32,
    /// Angular velocity decay per second, same form.
    pub angular_damping: f32,
    /// Units and origin for this world (PLAN.md §7). SI unless changed.
    pub domain: Domain,
    /// Seed for every generator this world hands out.
    ///
    /// PLAN.md §22 puts the seed in the reproducibility tuple and requires it
    /// to be explicit. It lives on the world rather than in a thread-local so
    /// that it is impossible to draw a random number without one having been
    /// recorded — a generator reached through the environment is a generator
    /// the project file cannot describe.
    pub seed: u64,

    scene: Vec<BodyDesc>,
    /// The state. Public because solvers work on the arrays directly; there is
    /// no accessor layer between the solver and its data.
    pub bodies: BodySet,
    /// Joints are scene *and* state: unlike contacts, they are authored rather
    /// than discovered, and they survive a reset unchanged.
    joints: Vec<Joint>,

    /// Deformables. A cloth or a soft body is a set of point masses plus the
    /// constraints between them — no new system, just more work for the same
    /// solver (PLAN.md §5).
    pub particles: ParticleSet,
    /// Distance constraints between particles: the weave of a cloth, the
    /// edges of a soft body.
    pub distance_constraints: Vec<DistanceConstraint>,
    /// Bending constraints, which resist folding rather than stretching.
    pub bending_constraints: Vec<BendingConstraint>,
    /// Triangles over the particles, used by the aerodynamic model and by
    /// surface reconstruction.
    pub triangles: Vec<SurfaceTriangle>,

    /// Bodies of fluid. One entry per `add_fluid` call, each with its own rest
    /// density and resolution.
    ///
    /// Kept apart from `particles` rather than merged into it, even though
    /// both are point masses. Two fluids at different rest densities are two
    /// different constraints, and a fluid particle and a cloth particle share
    /// no constraint at all — flattening them into one array would mean
    /// carrying a per-particle tag and branching on it inside the innermost
    /// loop in the engine.
    pub fluids: Vec<FluidSet>,

    /// Bodies of continuum, solved by MPM (PLAN.md §11).
    ///
    /// Kept apart from `fluids` for the same reason `fluids` is kept apart
    /// from `particles`: a material point carries a deformation gradient and a
    /// fluid particle does not, and merging them would put a tag in the
    /// innermost loop of both solvers to tell them apart.
    pub solids: Vec<SolidSet>,

    /// The declarative half of the deformables and the fluids.
    ///
    /// Kept for the same reason `scene` is kept for bodies, and it was missing
    /// until v0.4: without it `reset` rebuilt the rigid bodies and left the
    /// cloth wherever the last run had draped it, so "reset" produced a state
    /// the scene had never described. It also makes a `World` fully
    /// serializable — PLAN.md §16's whole point is that the scene is small and
    /// the state is not, and that only holds if the scene is complete.
    cloth_scene: Vec<ClothDesc>,
    soft_scene: Vec<SoftBodyDesc>,
    fluid_scene: Vec<FluidDesc>,
    solid_scene: Vec<SolidDesc>,
}

impl Default for World {
    fn default() -> Self {
        Self::new()
    }
}

impl World {
    /// An empty world at standard gravity.
    pub fn new() -> Self {
        Self {
            gravity: GRAVITY,
            linear_damping: 0.0,
            angular_damping: 0.05,
            domain: Domain::SI,
            seed: 0,
            scene: Vec::new(),
            bodies: BodySet::new(),
            joints: Vec::new(),
            particles: ParticleSet::new(),
            distance_constraints: Vec::new(),
            bending_constraints: Vec::new(),
            triangles: Vec::new(),
            solids: Vec::new(),
            fluids: Vec::new(),
            cloth_scene: Vec::new(),
            soft_scene: Vec::new(),
            fluid_scene: Vec::new(),
            solid_scene: Vec::new(),
        }
    }

    /// Builder: use this gravity.
    pub fn with_gravity(mut self, g: Vec3) -> Self {
        self.gravity = g;
        self
    }

    /// Builder: use this RNG seed. §22 wants seeds explicit and logged, never
    /// implicit.
    pub fn with_seed(mut self, seed: u64) -> Self {
        self.seed = seed;
        self
    }

    /// A generator on an independent stream, seeded from this world.
    ///
    /// Give each system that wants randomness its own stream rather than
    /// sharing one generator; see [`crate::Rng::with_stream`] for why sharing makes
    /// every system's numbers depend on how many the others drew.
    pub fn rng(&self, stream: u64) -> crate::Rng {
        crate::Rng::with_stream(self.seed, stream)
    }

    /// Add a body to the scene and build its state. Returns a handle valid for
    /// both.
    pub fn add(&mut self, desc: BodyDesc) -> BodyHandle {
        let (inv_mass, inv_inertia) = desc.inverses();
        let handle = self.bodies.push(
            desc.pose(),
            desc.linear_velocity,
            desc.angular_velocity,
            inv_mass,
            inv_inertia,
            desc.shape.clone(),
            desc.material,
            desc.body_type,
        );
        if let Some(t) = desc.temperature {
            self.bodies.heat[handle.index()] = crate::BodyHeat::uniform(t);
        }
        self.bodies.failure[handle.index()] = desc.failure;
        if let Some(g) = desc.grain.filter(|g| g.length_squared() > 0.5) {
            self.bodies.grain[handle.index()] = g;
        }
        self.bodies.advanced[handle.index()] = desc.advanced;
        self.scene.push(desc);
        handle
    }

    /// Add a dynamic sphere.
    pub fn add_sphere(&mut self, radius: f32, position: Vec3, material: Material) -> BodyHandle {
        self.add(
            BodyDesc::sphere(radius)
                .at(position)
                .with_material(material),
        )
    }

    /// Add a dynamic box, sized by its full extents rather than half.
    pub fn add_box(&mut self, size: Vec3, position: Vec3, material: Material) -> BodyHandle {
        self.add(BodyDesc::cuboid(size).at(position).with_material(material))
    }

    /// Add an immovable box — a floor, a wall, a container.
    pub fn add_static_box(&mut self, size: Vec3, position: Vec3) -> BodyHandle {
        self.add(
            BodyDesc::cuboid(size)
                .at(position)
                .with_material(Material::CONCRETE)
                .fixed(),
        )
    }

    /// The scene — the declarative half. Read-only: changing a description
    /// after build would silently disagree with the state until the next reset.
    pub fn scene(&self) -> &[BodyDesc] {
        &self.scene
    }

    /// Add a joint between two bodies already in the world.
    pub fn add_joint(&mut self, joint: Joint) -> JointHandle {
        let handle = JointHandle(self.joints.len() as u32);
        self.joints.push(joint);
        handle
    }

    /// Connect two bodies at a world-space point, using their current poses to
    /// derive the local anchors.
    pub fn connect(
        &mut self,
        a: BodyHandle,
        b: BodyHandle,
        point: Vec3,
        kind: JointKind,
    ) -> JointHandle {
        let pose_a = self.bodies.pose(a.index());
        let pose_b = self.bodies.pose(b.index());
        self.add_joint(Joint::at_world_point(a, &pose_a, b, &pose_b, point, kind))
    }

    /// Connect two bodies at a world point about a world axis — the form a
    /// hinge or a slider wants.
    pub fn connect_axis(
        &mut self,
        a: BodyHandle,
        b: BodyHandle,
        point: Vec3,
        axis: Vec3,
        kind: JointKind,
    ) -> JointHandle {
        let pose_a = self.bodies.pose(a.index());
        let pose_b = self.bodies.pose(b.index());
        self.add_joint(Joint::at_world_point_with_axis(
            a, &pose_a, b, &pose_b, point, axis, kind,
        ))
    }

    /// Add a sheet of cloth, returning handles into its particle grid.
    pub fn add_cloth(&mut self, desc: &ClothDesc) -> ClothHandles {
        self.cloth_scene.push(*desc);
        build_cloth(
            desc,
            &mut self.particles,
            &mut self.distance_constraints,
            &mut self.bending_constraints,
            &mut self.triangles,
        )
    }

    /// Add a deformable block, returning its particle handles.
    pub fn add_soft_body(&mut self, desc: &SoftBodyDesc) -> Vec<ParticleHandle> {
        self.soft_scene.push(*desc);
        build_soft_body(desc, &mut self.particles, &mut self.distance_constraints)
    }

    /// The declarative descriptions this world was built from. Read-only for
    /// the reason [`World::scene`] is: editing one after the build would
    /// disagree with the state until the next reset.
    pub fn cloth_scene(&self) -> &[ClothDesc] {
        &self.cloth_scene
    }

    /// The soft bodies as they were described, which is what a rebuild and a
    /// project file both need.
    pub fn soft_scene(&self) -> &[SoftBodyDesc] {
        &self.soft_scene
    }

    /// The fluids as they were described.
    pub fn fluid_scene(&self) -> &[FluidDesc] {
        &self.fluid_scene
    }

    /// The solids as described, for serialization.
    pub fn solid_scene(&self) -> &[SolidDesc] {
        &self.solid_scene
    }

    /// Fill a region with fluid.
    ///
    /// The count is not obvious from the description — a 1 m cube at the
    /// default 10 cm spacing is a thousand particles and at 5 cm it is eight
    /// thousand — so [`FluidDesc::particle_count`] answers that before the
    /// memory is committed.
    pub fn add_fluid(&mut self, desc: &FluidDesc) -> FluidHandle {
        let handle = FluidHandle(self.fluids.len() as u32);
        self.fluids.push(build_fluid(desc));
        self.fluid_scene.push(desc.clone());
        handle
    }

    /// One body of fluid, by handle.
    pub fn fluid(&self, handle: FluidHandle) -> &FluidSet {
        &self.fluids[handle.index()]
    }

    /// The same, mutably.
    pub fn fluid_mut(&mut self, handle: FluidHandle) -> &mut FluidSet {
        &mut self.fluids[handle.index()]
    }

    /// Total fluid particles across every body of fluid.
    pub fn fluid_particle_count(&self) -> usize {
        self.fluids.iter().map(|f| f.len()).sum()
    }

    /// Add a body of continuum, sampled from its description.
    pub fn add_solid(&mut self, desc: &SolidDesc) -> SolidHandle {
        let handle = SolidHandle(self.solids.len() as u32);
        self.solids.push(build_solid(desc));
        self.solid_scene.push(desc.clone());
        handle
    }

    /// One solid, by handle.
    pub fn solid(&self, handle: SolidHandle) -> &SolidSet {
        &self.solids[handle.index()]
    }

    /// The same, mutably.
    pub fn solid_mut(&mut self, handle: SolidHandle) -> &mut SolidSet {
        &mut self.solids[handle.index()]
    }

    /// Total material points across every solid.
    pub fn solid_point_count(&self) -> usize {
        self.solids.iter().map(|s| s.len()).sum()
    }

    /// Every joint in the world.
    pub fn joints(&self) -> &[Joint] {
        &self.joints
    }

    /// The same, mutably.
    pub fn joints_mut(&mut self) -> &mut [Joint] {
        &mut self.joints
    }

    /// The joint storage itself, so a caller can move it aside for the
    /// duration of a step. Used by the simulation loop to hand the solver both
    /// the joints and a mutable body set at the same time.
    #[doc(hidden)]
    pub fn joints_storage_mut(&mut self) -> &mut Vec<Joint> {
        &mut self.joints
    }

    /// Total linear momentum of everything in the world: bodies, cloth and
    /// soft-body particles, and every body of fluid.
    ///
    /// One of §23's three conserved quantities. Nothing in the solver may
    /// change it without an external force, so a run where it drifts has a
    /// bug in it — which is the whole reason to track it per run rather than
    /// per test.
    pub fn linear_momentum(&self) -> Vec3 {
        self.bodies.linear_momentum()
            + self.particles.linear_momentum()
            + self
                .fluids
                .iter()
                .map(|f| f.particles.linear_momentum())
                .fold(Vec3::ZERO, |a, b| a + b)
            + self
                .solids
                .iter()
                .map(|s| s.particles.linear_momentum())
                .fold(Vec3::ZERO, |a, b| a + b)
    }

    /// Total angular momentum about the world origin, bodies only.
    ///
    /// Particles carry `m r × v` about the origin as well; they are included.
    pub fn angular_momentum(&self) -> Vec3 {
        let particle_moment = |set: &crate::ParticleSet| {
            let mut l = Vec3::ZERO;
            for i in 0..set.len() {
                if set.is_movable(i) {
                    let r = set.translation(i);
                    l += r.cross(set.velocity[i] / set.inv_mass[i]);
                }
            }
            l
        };
        self.bodies.angular_momentum_total()
            + particle_moment(&self.particles)
            + self
                .fluids
                .iter()
                .map(|f| particle_moment(&f.particles))
                .fold(Vec3::ZERO, |a, b| a + b)
            + self
                .solids
                .iter()
                .map(|s| particle_moment(&s.particles))
                .fold(Vec3::ZERO, |a, b| a + b)
    }

    /// Total mass of everything that can move.
    ///
    /// Mass is conserved by construction in Basic — nothing is created or
    /// destroyed — so a run where this changes has found something worse than
    /// a solver bug. It is tracked because §23 asks for it and because the
    /// day Advanced starts moving material between representations, it stops
    /// being trivially true.
    pub fn total_mass(&self) -> f32 {
        self.bodies.total_mass()
            + self.particles.total_mass()
            + self.fluids.iter().map(|f| f.mass()).sum::<f32>()
            + self.solids.iter().map(|s| s.mass()).sum::<f32>()
    }

    /// How many rigid bodies there are. Particles and fluid are counted
    /// separately — see [`World::fluid_particle_count`].
    pub fn len(&self) -> usize {
        self.bodies.len()
    }

    /// True where there are no rigid bodies.
    pub fn is_empty(&self) -> bool {
        self.bodies.is_empty()
    }

    /// Discard the state and rebuild it from the scene. This is what "reset"
    /// means in PLAN.md §17 — a simulation is causal, so the only way back to
    /// t = 0 is to construct t = 0 again.
    pub fn reset(&mut self) {
        self.fluids.clear();
        self.fluids.reserve(self.fluid_scene.len());
        for desc in &self.fluid_scene {
            self.fluids.push(build_fluid(desc));
        }

        // Rebuilt rather than restored, and for a solid the difference is
        // sharper than it is anywhere else: a material point's deformation
        // gradient is its whole history, so "put it back where it started"
        // without resetting `F` would leave a block sitting in its rest
        // position under the stress of the squash it just took.
        self.solids.clear();
        self.solids.reserve(self.solid_scene.len());
        for desc in &self.solid_scene {
            self.solids.push(build_solid(desc));
        }

        // Deformables are rebuilt from their descriptions rather than having
        // their positions restored, so the constraint rest lengths are the
        // ones the description implies rather than the ones a previous run
        // happened to stretch them to.
        self.particles.clear();
        self.distance_constraints.clear();
        self.bending_constraints.clear();
        self.triangles.clear();
        for desc in &self.cloth_scene {
            build_cloth(
                desc,
                &mut self.particles,
                &mut self.distance_constraints,
                &mut self.bending_constraints,
                &mut self.triangles,
            );
        }
        for desc in &self.soft_scene {
            build_soft_body(desc, &mut self.particles, &mut self.distance_constraints);
        }

        self.bodies.clear();
        self.bodies.reserve(self.scene.len());
        for desc in &self.scene {
            let (inv_mass, inv_inertia) = desc.inverses();
            self.bodies.push(
                desc.pose(),
                desc.linear_velocity,
                desc.angular_velocity,
                inv_mass,
                inv_inertia,
                desc.shape.clone(),
                desc.material,
                desc.body_type,
            );
            if let Some(t) = desc.temperature {
                *self.bodies.heat.last_mut().expect("just pushed") = crate::BodyHeat::uniform(t);
            }
            *self.bodies.failure.last_mut().expect("just pushed") = desc.failure;
            if let Some(g) = desc.grain.filter(|g| g.length_squared() > 0.5) {
                *self.bodies.grain.last_mut().expect("just pushed") = g;
            }
            *self.bodies.advanced.last_mut().expect("just pushed") = desc.advanced;
        }
    }

    /// Total mechanical energy, bodies and particles alike. Reported by the
    /// diagnostics in `fusion-sim` and asserted on by the energy-drift
    /// regression test.
    ///
    /// **Kinetic and potential only.** A continuum's third term — the elastic
    /// strain energy stored in its deformation gradient — is a question for
    /// its constitutive model, which lives three layers up in `fusion-mpm`. A
    /// scene with a solid in it wants
    /// [`Simulation::mechanical_energy`](../../fusion_sim/struct.Simulation.html#method.mechanical_energy),
    /// which adds that term; this one counts the material points as if they
    /// were sand.
    pub fn mechanical_energy(&self) -> f32 {
        self.bodies.kinetic_energy()
            + self.bodies.potential_energy(self.gravity)
            + self.particles.kinetic_energy()
            + self.particles.potential_energy(self.gravity)
            + self
                .fluids
                .iter()
                .map(|f| f.particles.kinetic_energy() + f.particles.potential_energy(self.gravity))
                .sum::<f32>()
            + self
                .solids
                .iter()
                .map(|s| s.particles.kinetic_energy() + s.particles.potential_energy(self.gravity))
                .sum::<f32>()
    }
}
