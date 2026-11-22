//! Aerodynamics as force models.
//!
//! PLAN.md §13 is blunt about the boundary: "nobody asking for aerodynamics in
//! a physics engine wants CFD." They want things to feel the air, and that is
//! nearly free — drag, lift, wind and the Magnus effect are a handful of
//! algebra each, applied to bodies that already exist.
//!
//! What this is *not* is a fluid solver, and §13 explains why not in terms
//! worth keeping in view. SPH cannot do aerodynamics: gases fill space where
//! liquids do not; weakly-compressible SPH breaks above Mach 0.3; the boundary
//! layer that produces most of the lift and drag is millimetres thick and
//! would demand millimetre particles *everywhere*; and a car at Re ~ 10^7
//! needs about 10^15 cells to resolve directly. Everyone real uses turbulence
//! models instead.
//!
//! So: Fusion will tell you how a parachute behaves. It will not tell you the
//! drag coefficient of a car you just designed — and no particle solver would
//! have.

use fusion_core::{BodySet, ParticleSet, Shape, SurfaceTriangle};
use fusion_math::Vec3;

/// Air, as far as a force model is concerned.
#[derive(Clone, Copy, Debug)]
pub struct AirModel {
    /// Off by default: a scene that has not asked for aerodynamics should
    /// not pay for them.
    pub enabled: bool,
    /// kg/m^3. 1.225 at sea level, 15 C.
    pub density: f32,
    /// Uniform wind velocity. A curl-noise field is the obvious next step
    /// (§13 calls it very cheap and excellent-looking) and slots in here.
    pub wind: Vec3,
    /// Scales every drag coefficient at once, for art direction.
    pub drag_scale: f32,
    /// Magnus coefficient: the sideways force on a spinning body. Zero for a
    /// simulation that does not care; around 0.2 makes ball sports feel right.
    pub magnus: f32,
    /// Normal-projected drag on cloth triangles.
    pub cloth_drag: f32,
    /// Tangential lift on cloth triangles — what makes a flag ripple rather
    /// than simply blow flat.
    pub cloth_lift: f32,
}

impl Default for AirModel {
    fn default() -> Self {
        Self {
            enabled: false,
            density: 1.225,
            wind: Vec3::ZERO,
            drag_scale: 1.0,
            magnus: 0.0,
            cloth_drag: 1.0,
            cloth_lift: 0.2,
        }
    }
}

impl AirModel {
    /// Still air at sea level.
    pub fn still() -> Self {
        Self {
            enabled: true,
            ..Default::default()
        }
    }

    /// Still air of standard density, moving at `velocity`.
    pub fn with_wind(velocity: Vec3) -> Self {
        Self {
            enabled: true,
            wind: velocity,
            ..Default::default()
        }
    }

    /// Wind at a point. A uniform field today; the signature is the one a
    /// spatially varying field needs.
    #[inline]
    pub fn wind_at(&self, _point: Vec3) -> Vec3 {
        self.wind
    }
}

/// Drag coefficient and reference area presented by a shape along `direction`.
///
/// The coefficients are the standard handbook values — 0.47 for a sphere, 1.05
/// for a cube face-on. The box area is the sum of its three faces weighted by
/// how squarely each meets the flow, which is exact for a box and cheap.
fn drag_profile(shape: &Shape, orientation: fusion_math::Quat, direction: Vec3) -> (f32, f32) {
    match shape {
        Shape::Sphere { radius } => (0.47, std::f32::consts::PI * radius * radius),
        // A hull's frontal area, by its support width across the two axes
        // perpendicular to the flow. That is the bounding rectangle of the
        // silhouette rather than the silhouette itself, so it over-states a
        // rounded shape by up to a factor of `4/pi` — stated because it is a
        // drag model's whole content, and because computing the true
        // silhouette area is a projection and a 2D hull per body per substep.
        // The coefficient is a cube's; a hull is a faceted thing.
        Shape::Hull(hull) => {
            let local = orientation.rotate_inverse(direction);
            let (u, v) = perpendicular_pair(local);
            let width = hull.support(u).dot(u) - hull.support(-u).dot(u);
            let height = hull.support(v).dot(v) - hull.support(-v).dot(v);
            (1.05, (width * height).max(0.0))
        }
        Shape::Cuboid { half_extents: h } => {
            let h = *h;
            let local = orientation.rotate_inverse(direction).abs();
            let area = local.x * (2.0 * h.y) * (2.0 * h.z)
                + local.y * (2.0 * h.x) * (2.0 * h.z)
                + local.z * (2.0 * h.x) * (2.0 * h.y);
            (1.05, area)
        }
    }
}

/// Two unit vectors perpendicular to `direction` and to each other.
fn perpendicular_pair(direction: Vec3) -> (Vec3, Vec3) {
    // Cross with whichever axis the direction is least aligned to, so the
    // product is never near zero.
    let axis = if direction.x.abs() < 0.9 {
        Vec3::X
    } else {
        Vec3::Y
    };
    let u = direction.cross(axis).normalize_or_zero();
    (u, direction.cross(u).normalize_or_zero())
}

/// Apply drag and Magnus force to rigid bodies.
///
/// Applied as a velocity change over the substep rather than as an impulse at
/// a contact point: drag acts on the whole body, and the centre of pressure of
/// a tumbling box is not something this model claims to know.
pub fn apply_to_bodies(bodies: &mut BodySet, air: &AirModel, h: f32) {
    if !air.enabled {
        return;
    }
    for i in 0..bodies.len() {
        if !bodies.is_movable(i) {
            continue;
        }
        let relative = bodies.linear_velocity[i] - air.wind_at(bodies.translation(i));
        let (direction, speed) = relative.normalize_and_length();
        if speed < 1e-4 {
            continue;
        }

        let (coefficient, area) = drag_profile(&bodies.shape[i], bodies.orientation[i], direction);
        // F = 1/2 rho v^2 Cd A, opposing the motion.
        let drag =
            -direction * (0.5 * air.density * speed * speed * coefficient * area * air.drag_scale);

        // F ∝ omega x v. One cross product, and it is what makes a sliced
        // ball curve.
        let magnus = if air.magnus > 0.0 {
            bodies.angular_velocity[i].cross(relative) * (air.magnus * air.density)
        } else {
            Vec3::ZERO
        };

        let acceleration = (drag + magnus) * bodies.inv_mass[i];
        // Never let a force model reverse a velocity within one substep; that
        // is a sign the timestep is too coarse for the drag, not a physical
        // event, and it turns into a body that vibrates instead of slowing.
        let change = acceleration * h;
        let limited = if change.length() > speed {
            -relative
        } else {
            change
        };
        bodies.linear_velocity[i] += limited;
    }
}

/// Per-triangle aerodynamics on cloth.
///
/// §13 lists this as the feature that gives flags, sails, parachutes and
/// capes, and it is the reason cloth carries a triangle list at all. Each
/// triangle feels the wind through its own normal: force along the normal
/// proportional to the squared normal component of the relative wind, which
/// vanishes for a triangle edge-on to the flow and peaks for one face-on.
/// That angular dependence is the whole effect — without it cloth simply
/// blows downwind like a solid sheet.
pub fn apply_to_cloth(
    particles: &mut ParticleSet,
    triangles: &[SurfaceTriangle],
    air: &AirModel,
    h: f32,
) {
    if !air.enabled || triangles.is_empty() {
        return;
    }
    for triangle in triangles {
        let (a, b, c) = (
            triangle.a as usize,
            triangle.b as usize,
            triangle.c as usize,
        );
        let edge_ab = particles.relative(a, b);
        let edge_ac = particles.relative(a, c);
        let cross = edge_ab.cross(edge_ac);
        let (normal, twice_area) = cross.normalize_and_length();
        if twice_area < 1e-9 {
            continue;
        }
        let area = twice_area * 0.5;

        let centre = particles.translation(a);
        let velocity =
            (particles.velocity[a] + particles.velocity[b] + particles.velocity[c]) / 3.0;
        let relative = velocity - air.wind_at(centre);
        let normal_speed = relative.dot(normal);

        // Drag opposes the normal component; lift acts along the normal itself,
        // signed so it always pushes the sheet away from the flow.
        let drag = -normal
            * (0.5
                * air.density
                * area
                * normal_speed.abs()
                * normal_speed
                * air.cloth_drag
                * air.drag_scale);
        let tangential = relative.reject_from_unit(normal);
        let lift = -tangential * (0.5 * air.density * area * tangential.length() * air.cloth_lift);

        let force = (drag + lift) / 3.0;
        for &index in &[a, b, c] {
            let inv_mass = particles.inv_mass[index];
            if inv_mass <= 0.0 {
                continue;
            }
            let change = force * inv_mass * h;
            // Same guard as bodies: a force model may slow a particle to a
            // stop, never fling it backwards.
            let speed = particles.velocity[index].length();
            if change.length() > speed.max(1.0) * 2.0 {
                continue;
            }
            particles.velocity[index] += change;
        }
    }
}
