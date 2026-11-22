//! Particles and the constraints between them.
//!
//! PLAN.md §5 lists cloth as `Mesh · Material · Constraint set (distance,
//! bending)` and a soft body as much the same. Neither is a new *system* —
//! both are a bag of point masses plus constraint functions, solved by the
//! same XPBD loop that already handles contacts and joints. That is the payoff
//! §10 promised for choosing XPBD: "one mechanism covers contacts, joints,
//! cloth and soft bodies, so there are fewer moving parts to own."
//!
//! A particle is not a small rigid body. It has no orientation, no inertia
//! tensor and no angular velocity, which is exactly why a cloth of ten
//! thousand of them is affordable.

use fusion_math::{to_f32, to_f64, DVec3, Vec3};

/// Reference to a particle.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct ParticleHandle(pub u32);

impl ParticleHandle {
    /// The index this handle refers to.
    #[inline]
    pub fn index(self) -> usize {
        self.0 as usize
    }
}

/// Point masses, struct-of-arrays, in the same layout and for the same reasons
/// as [`crate::BodySet`].
#[derive(Clone, Debug, Default)]
pub struct ParticleSet {
    /// Double precision, for the reason documented on `BodySet::position`:
    /// position is the one unbounded accumulation in the simulation.
    pub position: Vec<DVec3>,
    /// Velocity, m/s.
    pub velocity: Vec<Vec3>,
    /// Zero pins the particle in place — the same convention static bodies
    /// use, and the same reason: a pin becomes a zero rather than a branch.
    pub inv_mass: Vec<f32>,
    /// Collision radius. Cloth is a surface with no thickness, so this is the
    /// stand-off that keeps it off the geometry it drapes over.
    pub radius: Vec<f32>,
}

impl ParticleSet {
    /// An empty set.
    pub fn new() -> Self {
        Self::default()
    }

    /// How many particles there are.
    #[inline]
    pub fn len(&self) -> usize {
        self.position.len()
    }

    /// True where there are none.
    #[inline]
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// Remove every particle, keeping the allocated capacity.
    pub fn clear(&mut self) {
        self.position.clear();
        self.velocity.clear();
        self.inv_mass.clear();
        self.radius.clear();
    }

    /// Add a particle and return its handle. A mass of zero pins it.
    pub fn push(&mut self, position: Vec3, mass: f32, radius: f32) -> ParticleHandle {
        let handle = ParticleHandle(self.len() as u32);
        self.position.push(to_f64(position));
        self.velocity.push(Vec3::ZERO);
        self.inv_mass
            .push(if mass > 0.0 { 1.0 / mass } else { 0.0 });
        self.radius.push(radius);
        handle
    }

    /// Position as `f32`, for the many places that do not need the width.
    #[inline]
    pub fn translation(&self, i: usize) -> Vec3 {
        to_f32(self.position[i])
    }

    /// Move a particle by `delta`.
    #[inline]
    pub fn displace(&mut self, i: usize, delta: Vec3) {
        self.position[i] += to_f64(delta);
    }

    /// `position[b] - position[a]`, differenced in double and returned in
    /// single — the same exactness argument as `BodySet::relative_translation`.
    #[inline]
    pub fn relative(&self, a: usize, b: usize) -> Vec3 {
        to_f32(self.position[b] - self.position[a])
    }

    /// Offset from a particle to a world point, without building the
    /// particle's absolute position in f32.
    #[inline]
    pub fn offset_to(&self, i: usize, point: Vec3) -> Vec3 {
        to_f32(to_f64(point) - self.position[i])
    }

    /// True where the solver may move it — a pinned particle is not movable.
    #[inline]
    pub fn is_movable(&self, i: usize) -> bool {
        self.inv_mass[i] > 0.0
    }

    /// Pin a particle where it currently is.
    pub fn pin(&mut self, i: usize) {
        self.inv_mass[i] = 0.0;
        self.velocity[i] = Vec3::ZERO;
    }

    /// Total kinetic energy of every particle.
    pub fn kinetic_energy(&self) -> f32 {
        (0..self.len())
            .filter(|&i| self.is_movable(i))
            .map(|i| 0.5 * self.velocity[i].length_squared() / self.inv_mass[i])
            .sum()
    }

    /// Total linear momentum of every movable particle.
    pub fn linear_momentum(&self) -> Vec3 {
        let mut p = Vec3::ZERO;
        for i in 0..self.len() {
            if self.is_movable(i) {
                p += self.velocity[i] / self.inv_mass[i];
            }
        }
        p
    }

    /// Total mass of every movable particle. Pinned particles are infinite
    /// and are left out.
    pub fn total_mass(&self) -> f32 {
        (0..self.len())
            .filter(|&i| self.is_movable(i))
            .map(|i| 1.0 / self.inv_mass[i])
            .sum()
    }

    /// Total potential energy against `gravity`.
    pub fn potential_energy(&self, gravity: Vec3) -> f32 {
        (0..self.len())
            .filter(|&i| self.is_movable(i))
            .map(|i| -gravity.dot(self.translation(i)) / self.inv_mass[i])
            .sum()
    }

    /// True where every position and velocity is finite. What `is_stable`
    /// asks about cloth and soft bodies.
    pub fn is_finite(&self) -> bool {
        self.position.iter().all(|p| p.is_finite()) && self.velocity.iter().all(|v| v.is_finite())
    }
}

/// `|p_b - p_a| = rest`. The workhorse: stretch resistance in cloth, edge
/// length in a soft body, and a rope when the compliance is high.
#[derive(Clone, Copy, Debug)]
pub struct DistanceConstraint {
    /// First particle.
    pub a: u32,
    /// Second particle.
    pub b: u32,
    /// Rest length, metres.
    pub rest: f32,
    /// Inverse stiffness, m/N. Zero is inextensible. XPBD is stable there,
    /// which is the property that makes cloth practical — a stiffness-based
    /// formulation needs a timestep that shrinks with the stiffness.
    pub compliance: f32,
    /// Below 1.0 the constraint resists stretching more than compression, so
    /// cloth can crumple freely while refusing to stretch. Real fabric does
    /// exactly this.
    pub compression_scale: f32,
}

impl DistanceConstraint {
    /// A constraint holding two particles `rest` metres apart, inextensible.
    pub fn new(a: ParticleHandle, b: ParticleHandle, rest: f32) -> Self {
        Self {
            a: a.0,
            b: b.0,
            rest,
            compliance: 0.0,
            compression_scale: 1.0,
        }
    }

    /// Builder: soften it. See [`DistanceConstraint::compliance`].
    pub fn with_compliance(mut self, compliance: f32) -> Self {
        self.compliance = compliance;
        self
    }

    /// Builder: resist compression less than stretch. See
    /// [`DistanceConstraint::compression_scale`].
    pub fn with_compression_scale(mut self, scale: f32) -> Self {
        self.compression_scale = scale;
        self
    }
}

/// Resists folding across the edge shared by two triangles.
///
/// Implemented as a distance constraint between the two *opposite* corners
/// rather than as a dihedral angle. The angle formulation is more faithful and
/// considerably more expensive, and it is singular when the fold reaches 180
/// degrees — which is precisely where cloth spends its time. This form is what
/// most production cloth actually uses.
#[derive(Clone, Copy, Debug)]
pub struct BendingConstraint {
    /// First particle of the pair across the fold.
    pub a: u32,
    /// Second particle of the pair.
    pub b: u32,
    /// Rest distance between them, which encodes the rest angle.
    pub rest: f32,
    /// Inverse stiffness, m/N. Higher is floppier cloth.
    pub compliance: f32,
}

impl BendingConstraint {
    /// A bending constraint over the two particles opposite a shared edge.
    pub fn new(a: ParticleHandle, b: ParticleHandle, rest: f32, compliance: f32) -> Self {
        Self {
            a: a.0,
            b: b.0,
            rest,
            compliance,
        }
    }
}

/// A triangle of the cloth surface. Carried for rendering and for the
/// per-triangle aerodynamics in PLAN.md §13.
#[derive(Clone, Copy, Debug)]
pub struct SurfaceTriangle {
    /// First vertex.
    pub a: u32,
    /// Second vertex.
    pub b: u32,
    /// Third vertex.
    pub c: u32,
}
