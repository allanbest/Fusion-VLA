//! Rigid body state, stored struct-of-arrays.
//!
//! PLAN.md §6 and §30: parallel arrays behind a `u32` handle, and deliberately
//! **not** an ECS. An archetype system generalized from a single example would
//! be generalized along the wrong axis; that decision waits until three real
//! systems have shown what they share.
//!
//! The layout is the six arrays the v0.1 spec names, plus shape, material and
//! body type. The split is by access pattern: the first six are read and
//! written every substep, the last three are read once per step during
//! collision detection.

use fusion_math::{to_f32, to_f64, Aabb, DVec3, Isometry, Mat3, Quat, Vec3};

use crate::heat::BodyHeat;
use crate::material::Material;
use crate::shape::Shape;
use crate::stress::{BodyStress, Failure};

/// Reference to a body. An index, not a pointer — the arrays move.
///
/// v0.1 has no body removal, so a bare index is sound. When removal lands, this
/// gains a generation counter and `BodySet` gains a free list; every call site
/// already goes through this type, so that change stays local.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct BodyHandle(pub u32);

impl BodyHandle {
    /// The index this handle refers to.
    #[inline]
    pub fn index(self) -> usize {
        self.0 as usize
    }
}

/// Whether a body is moved by the solver.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Default)]
pub enum BodyType {
    /// Moved by forces and constraints like anything else.
    #[default]
    Dynamic,
    /// Immovable. Represented as zero inverse mass, so the solver needs no
    /// branch: corrections scaled by `inv_mass` simply vanish.
    Static,
    /// Immovable by the solver, but moved by something else — a script, or a
    /// caller writing its pose each step.
    ///
    /// Physically this is a body of infinite mass that happens to be going
    /// somewhere: it pushes everything it meets and nothing pushes back, which
    /// is what a piston, a lift, a conveyor or a wave paddle is. It carries
    /// the same zero inverse mass as [`Self::Static`], so every guard in the
    /// solver already treats it correctly and none of them needed a new
    /// branch.
    ///
    /// What separates it from static is not how the solver treats it but that
    /// the *velocity* on it is meaningful. A static body's velocity is
    /// nominally zero and nobody maintains it; a kinematic body's is derived
    /// from how far it moved this step, and the contact solve reads it — which
    /// is what makes a rising platform carry a crate instead of sliding out
    /// from under it. See `fusion_sim::Simulation::step`.
    Kinematic,
}

impl BodyType {
    /// True where the solver integrates it. Both of the other two are
    /// immovable as far as a constraint is concerned.
    pub fn is_dynamic(self) -> bool {
        matches!(self, BodyType::Dynamic)
    }
}

/// The simulation state of every rigid body in a world.
#[derive(Clone, Debug, Default)]
pub struct BodySet {
    // --- hot: touched every substep ---
    /// Centre of mass, world space, in **double precision**.
    ///
    /// Everything else in the engine is f32, and this one array is not,
    /// because position is the only unbounded accumulation in a rigid body
    /// simulation. Velocities, inertias and impulses are all bounded by the
    /// physics; a coordinate is bounded only by how big the scene is.
    ///
    /// In f32 the spacing at 200 m is 15 microns, while one substep of a
    /// resting body moves it under 3 — so `position += velocity * h` rounds to
    /// no change, and a contact correction of the same size disappears
    /// entirely. The result is not merely imprecise: corrections that vanish
    /// translationally still apply their torque, so a box resting quietly a
    /// couple of hundred metres from the origin rocks itself off the floor.
    /// The usable radius of an all-f32 engine turns out to be about 20 m,
    /// which is not a physics engine.
    ///
    /// PLAN.md §9 says precision is a per-solver choice rather than a global
    /// switch. This is that idea at the finest grain available: one array,
    /// widened, where the accumulation actually demands it.
    pub position: Vec<DVec3>,
    /// Orientation, one per body.
    pub orientation: Vec<Quat>,
    /// Linear velocity, m/s.
    pub linear_velocity: Vec<Vec3>,
    /// Angular velocity, rad/s, world frame.
    pub angular_velocity: Vec<Vec3>,
    /// Inverse mass. Zero means immovable, which is what makes static bodies
    /// free of branches in the solver.
    pub inv_mass: Vec<f32>,
    /// Diagonal of the inverse inertia tensor, body frame.
    pub inv_inertia_local: Vec<Vec3>,
    /// Diagonal of the inertia tensor itself, body frame. Derived from the
    /// inverse, and kept because the gyroscopic term in the integrator needs
    /// `I` rather than `I^-1` every substep, and a division per component per
    /// body per substep is not worth repeating.
    pub inertia_local: Vec<Vec3>,

    /// Whether a body is asleep: excluded from integration and from the
    /// constraint solve until something disturbs it.
    ///
    /// PLAN.md §26 lists sleeping first among the adaptive-resolution levers,
    /// and it is the cheapest one by a wide margin — a settled pile costs
    /// nothing at all rather than costing a full solve to confirm it has not
    /// moved.
    pub sleeping: Vec<bool>,
    /// How long this body has been below the motion thresholds, seconds.
    pub sleep_timer: Vec<f32>,

    // --- warm: read once per step ---
    /// Collision shape.
    pub shape: Vec<Shape>,
    /// Material: density, friction, restitution.
    pub material: Vec<Material>,
    /// Whether the solver may move it.
    pub body_type: Vec<BodyType>,

    // --- cold: read only by the heat solve ---
    /// Temperature inside each body. One number per body until the heat solve
    /// builds a grid for it; see [`BodyHeat`].
    pub heat: Vec<BodyHeat>,

    // --- cold: read by the stress solve and by failure ---
    /// Stress inside each body. Empty until the body is first loaded; see
    /// [`BodyStress`].
    pub stress: Vec<BodyStress>,
    /// What each body does when its stress is more than its material carries.
    pub failure: Vec<Failure>,
    /// Bodies that have left the rigid world: turned into a continuum, which
    /// carries on as a solid in their place.
    ///
    /// Marked rather than removed, because a body's index is how everything
    /// else refers to it — the editor's object list, scripts, joints, a
    /// recorded take — and removing one would renumber every body after it. A
    /// removed body is static, has no velocity, and is skipped by collision,
    /// the continuum, the fluids and drawing.
    pub removed: Vec<bool>,
    /// The scene body each body came from: itself, or for a piece broken off
    /// another body, the body that piece was first part of. What lets a
    /// fragment be drawn in its original's colours.
    pub source: Vec<u32>,
    /// Whether a body's shape and mass are no longer the ones its scene gave
    /// it: a piece of something that broke. A saved state has to carry these
    /// bodies' shapes, because the scene cannot say what they are.
    pub reshaped: Vec<bool>,
    /// Which way each body's grain runs, a unit vector in its own frame.
    /// Read only for a material that has one ([`crate::Strength::grain`]);
    /// see [`default_grain`] for where it runs unless told.
    pub grain: Vec<Vec3>,
    /// Whether each body is in the Advanced tier: measured for stress, able
    /// to break, and carrying a temperature when the scene has those on. A
    /// Basic body is a plain rigid body. See [`crate::BodyDesc::advanced`].
    pub advanced: Vec<bool>,
}

/// Where a body's grain runs unless told: along its longest side, the way a
/// plank or a post is cut from a trunk. Upright for a body as long one way as
/// another.
pub fn default_grain(shape: &Shape) -> Vec3 {
    let aabb = shape.local_aabb();
    let e = aabb.max - aabb.min;
    let longest = e.max_element();
    if e.y >= longest * 0.99 {
        Vec3::Y
    } else if e.x >= e.z {
        Vec3::X
    } else {
        Vec3::Z
    }
}

impl BodySet {
    /// An empty set.
    pub fn new() -> Self {
        Self::default()
    }

    /// How many bodies there are.
    #[inline]
    pub fn len(&self) -> usize {
        self.position.len()
    }

    /// True where there are none.
    #[inline]
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// Remove every body, keeping the allocated capacity.
    pub fn clear(&mut self) {
        self.position.clear();
        self.orientation.clear();
        self.linear_velocity.clear();
        self.angular_velocity.clear();
        self.inv_mass.clear();
        self.inv_inertia_local.clear();
        self.inertia_local.clear();
        self.sleeping.clear();
        self.sleep_timer.clear();
        self.shape.clear();
        self.material.clear();
        self.body_type.clear();
        self.heat.clear();
        self.stress.clear();
        self.failure.clear();
        self.removed.clear();
        self.source.clear();
        self.reshaped.clear();
        self.grain.clear();
        self.advanced.clear();
    }

    /// Reserve room for `n` more bodies in every array at once.
    pub fn reserve(&mut self, n: usize) {
        self.position.reserve(n);
        self.orientation.reserve(n);
        self.linear_velocity.reserve(n);
        self.angular_velocity.reserve(n);
        self.inv_mass.reserve(n);
        self.inv_inertia_local.reserve(n);
        self.inertia_local.reserve(n);
        self.sleeping.reserve(n);
        self.sleep_timer.reserve(n);
        self.shape.reserve(n);
        self.material.reserve(n);
        self.body_type.reserve(n);
        self.heat.reserve(n);
        self.stress.reserve(n);
        self.failure.reserve(n);
        self.removed.reserve(n);
        self.source.reserve(n);
        self.reshaped.reserve(n);
        self.grain.reserve(n);
        self.advanced.reserve(n);
    }

    /// Append one body. Callers go through `World`; this stays public so that
    /// state can be built directly in tests and benchmarks.
    ///
    /// The long argument list is the honest shape of the data: nine parallel
    /// arrays need nine values, and bundling them into a struct would just
    /// move the same fields somewhere else (`BodyDesc` already plays that
    /// role, one layer up).
    #[allow(clippy::too_many_arguments)]
    pub fn push(
        &mut self,
        pose: Isometry,
        linear_velocity: Vec3,
        angular_velocity: Vec3,
        inv_mass: f32,
        inv_inertia_local: Vec3,
        shape: Shape,
        material: Material,
        body_type: BodyType,
    ) -> BodyHandle {
        let handle = BodyHandle(self.len() as u32);
        self.position.push(to_f64(pose.translation));
        self.orientation.push(pose.rotation);
        self.linear_velocity.push(linear_velocity);
        self.angular_velocity.push(angular_velocity);
        self.inv_mass.push(inv_mass);
        self.inv_inertia_local.push(inv_inertia_local);
        self.inertia_local.push(Vec3::new(
            inv_or_zero(inv_inertia_local.x),
            inv_or_zero(inv_inertia_local.y),
            inv_or_zero(inv_inertia_local.z),
        ));
        self.sleeping.push(false);
        self.sleep_timer.push(0.0);
        self.grain.push(default_grain(&shape));
        self.shape.push(shape);
        self.material.push(material);
        self.body_type.push(body_type);
        self.heat
            .push(BodyHeat::uniform(material.thermal.resting_temperature()));
        self.stress.push(BodyStress::default());
        self.failure.push(Failure::Never);
        self.removed.push(false);
        self.source.push(handle.0);
        self.reshaped.push(false);
        self.advanced.push(true);
        handle
    }

    /// Centre of mass as single precision. Fine for rendering, bounds and
    /// anything else that does not subtract two of them.
    #[inline]
    pub fn translation(&self, i: usize) -> Vec3 {
        to_f32(self.position[i])
    }

    /// Move a body, in world space.
    #[inline]
    pub fn set_translation(&mut self, i: usize, p: Vec3) {
        self.position[i] = to_f64(p);
    }

    /// Move a body by a single-precision displacement, accumulated in double.
    /// This is the write that has to be exact; the read rarely does.
    #[inline]
    pub fn displace(&mut self, i: usize, delta: Vec3) {
        self.position[i] += to_f64(delta);
    }

    /// `position[b] - position[a]`, differenced in double and returned in
    /// single.
    ///
    /// Every piece of contact and joint code wants this rather than two
    /// absolute positions: the bodies are close together, so the difference is
    /// small and f32 represents it exactly, while the operands are not and it
    /// does not.
    #[inline]
    pub fn relative_translation(&self, a: usize, b: usize) -> Vec3 {
        to_f32(self.position[b] - self.position[a])
    }

    /// Position and orientation together.
    #[inline]
    pub fn pose(&self, i: usize) -> Isometry {
        Isometry::new(self.translation(i), self.orientation[i])
    }

    /// Set both at once.
    #[inline]
    pub fn set_pose(&mut self, i: usize, pose: Isometry) {
        self.set_translation(i, pose.translation);
        self.orientation[i] = pose.rotation;
    }

    /// Whether this body has left the rigid world. See [`BodySet::removed`].
    #[inline]
    pub fn is_removed(&self, i: usize) -> bool {
        self.removed[i]
    }

    /// Take a body out of the rigid world for good: static, still, asleep, and
    /// skipped by everything that looks for bodies. Its index stays valid.
    pub fn remove(&mut self, i: usize) {
        self.removed[i] = true;
        self.body_type[i] = BodyType::Static;
        self.inv_mass[i] = 0.0;
        self.inv_inertia_local[i] = Vec3::ZERO;
        self.inertia_local[i] = Vec3::ZERO;
        self.linear_velocity[i] = Vec3::ZERO;
        self.angular_velocity[i] = Vec3::ZERO;
        self.sleeping[i] = true;
        self.stress[i] = BodyStress::default();
    }

    /// True where the solver may move this body.
    #[inline]
    pub fn is_dynamic(&self, i: usize) -> bool {
        self.inv_mass[i] > 0.0
    }

    /// Whether this body is moved by something outside the solver.
    ///
    /// Distinct from [`Self::is_dynamic`], which asks whether the *solver*
    /// moves it — a kinematic body answers no to that and yes to this.
    #[inline]
    pub fn is_kinematic(&self, i: usize) -> bool {
        self.body_type[i] == BodyType::Kinematic
    }

    /// Whether the solver may move this body at all: dynamic, and awake.
    ///
    /// Every guard in the solver tests this rather than `inv_mass > 0`, which
    /// is what makes sleeping free — a sleeping body is simply one the solver
    /// declines to move, exactly like a static one, with no separate code path
    /// to keep in step.
    #[inline]
    pub fn is_movable(&self, i: usize) -> bool {
        self.inv_mass[i] > 0.0 && !self.sleeping[i]
    }

    /// Wake a body and restart its sleep timer.
    #[inline]
    pub fn wake(&mut self, i: usize) {
        self.sleeping[i] = false;
        self.sleep_timer[i] = 0.0;
    }

    /// Put a body to sleep, zeroing its velocity so it does not resume from a
    /// stale one when woken.
    #[inline]
    pub fn sleep(&mut self, i: usize) {
        self.sleeping[i] = true;
        self.linear_velocity[i] = Vec3::ZERO;
        self.angular_velocity[i] = Vec3::ZERO;
    }

    /// How many bodies are currently asleep.
    pub fn sleeping_count(&self) -> usize {
        self.sleeping.iter().filter(|s| **s).count()
    }

    /// Mass in kilograms. Infinite mass reports zero, since that is what the
    /// stored inverse says.
    #[inline]
    pub fn mass(&self, i: usize) -> f32 {
        if self.inv_mass[i] > 0.0 {
            1.0 / self.inv_mass[i]
        } else {
            f32::INFINITY
        }
    }

    /// Inertia in world space: `R * I_body * R^T`.
    #[inline]
    pub fn inertia_world(&self, i: usize) -> Mat3 {
        Mat3::from_quat(self.orientation[i]).mul_diagonal_mul_transpose(self.inertia_local[i])
    }

    /// World-space angular momentum of one body about its centre of mass.
    /// Conserved exactly in the absence of torque — which is the property the
    /// gyroscopic term in the integrator exists to preserve.
    #[inline]
    pub fn angular_momentum(&self, i: usize) -> Vec3 {
        self.inertia_world(i) * self.angular_velocity[i]
    }

    /// Inverse inertia in world space: `R * I_body^-1 * R^T`. Recomputed from
    /// the orientation rather than cached, because the orientation changes
    /// every substep and the recompute is a handful of multiplies.
    #[inline]
    pub fn inv_inertia_world(&self, i: usize) -> Mat3 {
        Mat3::from_quat(self.orientation[i]).mul_diagonal_mul_transpose(self.inv_inertia_local[i])
    }

    /// Velocity of the material point currently at world position `p`.
    #[inline]
    pub fn velocity_at_point(&self, i: usize, p: Vec3) -> Vec3 {
        self.linear_velocity[i] + self.angular_velocity[i].cross(p - self.translation(i))
    }

    /// World-space bounds of a body's shape at its current pose.
    pub fn aabb(&self, i: usize) -> Aabb {
        self.shape[i].world_aabb(&self.pose(i))
    }

    /// Apply an impulse at a world-space point. The only way v0.1 pokes a
    /// running simulation from outside.
    pub fn apply_impulse_at(&mut self, i: usize, impulse: Vec3, point: Vec3) {
        if !self.is_dynamic(i) {
            return;
        }
        // An impulse is a disturbance; a sleeping body that receives one is no
        // longer settled.
        self.wake(i);
        let r = point - self.translation(i);
        let delta_w = self.inv_inertia_world(i) * r.cross(impulse);
        self.linear_velocity[i] += impulse * self.inv_mass[i];
        self.angular_velocity[i] += delta_w;
    }

    // --- diagnostics: the seed of the energy ledger in PLAN.md §23 ---

    /// Total kinetic energy, translation and rotation, of every body.
    pub fn kinetic_energy(&self) -> f32 {
        let mut total = 0.0;
        for i in 0..self.len() {
            if !self.is_dynamic(i) {
                continue;
            }
            let m = 1.0 / self.inv_mass[i];
            let inertia =
                Mat3::from_quat(self.orientation[i]).mul_diagonal_mul_transpose(Vec3::new(
                    inv_or_zero(self.inv_inertia_local[i].x),
                    inv_or_zero(self.inv_inertia_local[i].y),
                    inv_or_zero(self.inv_inertia_local[i].z),
                ));
            let w = self.angular_velocity[i];
            total += 0.5 * m * self.linear_velocity[i].length_squared() + 0.5 * w.dot(inertia * w);
        }
        total
    }

    /// Gravitational potential energy relative to `datum` height along `up`.
    pub fn potential_energy(&self, gravity: Vec3) -> f32 {
        let mut total = 0.0;
        for i in 0..self.len() {
            if !self.is_dynamic(i) {
                continue;
            }
            total -= (1.0 / self.inv_mass[i]) * gravity.dot(self.translation(i));
        }
        total
    }

    /// Total linear momentum. Conserved by every part of the solver, so a
    /// change in it with nothing external acting is a bug.
    pub fn linear_momentum(&self) -> Vec3 {
        let mut p = Vec3::ZERO;
        for i in 0..self.len() {
            if self.is_dynamic(i) {
                p += self.linear_velocity[i] * (1.0 / self.inv_mass[i]);
            }
        }
        p
    }

    /// Total angular momentum of every dynamic body, about the world origin.
    ///
    /// `Σ (I ω + m r × v)`: the spin of each body plus the moment of its
    /// linear motion. Both terms are needed — a body drifting past the origin
    /// carries angular momentum about it whether or not it is spinning, and a
    /// sum of the spin terms alone is not conserved by anything.
    ///
    /// Conserved in the absence of external torque about that origin, which
    /// makes it one of the three quantities §23 wants tracked per run.
    pub fn angular_momentum_total(&self) -> Vec3 {
        let mut l = Vec3::ZERO;
        for i in 0..self.len() {
            if self.is_dynamic(i) {
                let mass = self.mass(i);
                let r = fusion_math::to_f32(self.position[i]);
                l += self.angular_momentum(i) + r.cross(self.linear_velocity[i] * mass);
            }
        }
        l
    }

    /// Total mass of every dynamic body. Static bodies are infinite and are
    /// left out rather than being counted as zero.
    pub fn total_mass(&self) -> f32 {
        (0..self.len())
            .filter(|&i| self.is_dynamic(i))
            .map(|i| self.mass(i))
            .sum()
    }

    /// True if every value is finite. A simulation that has blown up fails
    /// this, and it is the cheapest possible smoke test.
    pub fn is_finite(&self) -> bool {
        self.position.iter().all(|p| p.is_finite())
            && self.orientation.iter().all(|q| q.is_finite())
            && self.linear_velocity.iter().all(|v| v.is_finite())
            && self.angular_velocity.iter().all(|v| v.is_finite())
    }

    /// Iterate body indices.
    #[inline]
    pub fn iter(&self) -> impl Iterator<Item = usize> {
        0..self.len()
    }
}

#[inline]
fn inv_or_zero(v: f32) -> f32 {
    if v > 0.0 {
        1.0 / v
    } else {
        0.0
    }
}

/// Convenience view of one body, for reading and for debug output. Never used
/// inside a solver loop — that is what the arrays are for.
#[derive(Clone, Debug)]
pub struct BodyView {
    /// Which body this is.
    pub handle: BodyHandle,
    /// Where it is.
    pub pose: Isometry,
    /// How fast it is moving, m/s.
    pub linear_velocity: Vec3,
    /// And rotating, rad/s.
    pub angular_velocity: Vec3,
    /// Its collision shape.
    pub shape: Shape,
    /// Its material.
    pub material: Material,
    /// Whether the solver moves it.
    pub body_type: BodyType,
}

impl BodySet {
    /// One body's state gathered out of the parallel arrays.
    ///
    /// A convenience for reading, not the representation: the arrays are the
    /// representation, and §6 explains why.
    pub fn view(&self, h: BodyHandle) -> BodyView {
        let i = h.index();
        BodyView {
            handle: h,
            pose: self.pose(i),
            linear_velocity: self.linear_velocity[i],
            angular_velocity: self.angular_velocity[i],
            shape: self.shape[i].clone(),
            material: self.material[i],
            body_type: self.body_type[i],
        }
    }
}

/// Orientation-only helper used by demos and tests.
pub fn quat_from_euler_xyz(x: f32, y: f32, z: f32) -> Quat {
    Quat::from_rotation_z(z) * Quat::from_rotation_y(y) * Quat::from_rotation_x(x)
}
