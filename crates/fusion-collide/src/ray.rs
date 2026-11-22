//! Rays against shapes: what is under the pointer.
//!
//! The narrow phase answers "are these two shapes touching"; this answers
//! "what does this line hit first", which is a different question with the
//! same geometry behind it. It exists because an editor cannot let you click
//! on a body without it — and because §8's requirement that every shape carry
//! an SDF makes the same shapes answerable either way.
//!
//! Deliberately not GJK. A ray against a sphere or a box has a closed form
//! that is exact, branch-free and about twenty operations; running an
//! iterative solver for it would be slower and less accurate. When Fusion
//! grows shapes without a closed form — a convex hull is the next one — the
//! fallback is the slab test against its face planes, which a hull already
//! has.

use fusion_math::{Isometry, Vec3};

use fusion_core::Shape;

/// A half-line: everything at `origin + direction * t` for `t >= 0`.
///
/// `direction` is expected to be normalized. Nothing here checks, because the
/// caller building a ray from a camera already normalized it and doing it
/// again per shape is the kind of cost that only shows up at ten thousand
/// bodies.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Ray {
    /// Where the ray starts.
    pub origin: Vec3,
    /// Unit direction.
    pub direction: Vec3,
}

impl Ray {
    /// A ray from a point along a direction, which is normalized here.
    pub fn new(origin: Vec3, direction: Vec3) -> Self {
        Self {
            origin,
            direction: direction.normalize_or_zero(),
        }
    }

    /// The point at distance `t`.
    pub fn at(&self, t: f32) -> Vec3 {
        self.origin + self.direction * t
    }
}

/// Distance along `ray` to the first intersection with `shape` at `pose`, or
/// `None` where it misses.
///
/// A ray starting *inside* a shape hits it at `t = 0`, which is what a picker
/// wants: clicking while the camera is inside a box should select the box
/// rather than see through it.
pub fn ray_vs_shape(ray: &Ray, shape: &Shape, pose: &Isometry) -> Option<f32> {
    // Into the shape's own frame, so every case below is axis-aligned and
    // centred. The same trick the narrow phase uses, and for the same reason:
    // one transform beats a special case per orientation.
    let origin = pose.inverse_transform_point(ray.origin);
    let direction = pose.inverse_transform_direction(ray.direction);

    match shape {
        Shape::Sphere { radius } => sphere(origin, direction, *radius),
        Shape::Cuboid { half_extents } => slab(origin, direction, *half_extents),
        Shape::Hull(hull) => convex(origin, direction, hull),
    }
}

/// Ray against a convex polyhedron: the slab test, generalised.
///
/// A box is the intersection of three pairs of parallel slabs and the test
/// walks them; a hull is the intersection of one half-space per face and the
/// same walk works, keeping the latest entry and the earliest exit. Same
/// twelve lines, any number of faces.
fn convex(origin: Vec3, direction: Vec3, hull: &fusion_core::ConvexHull) -> Option<f32> {
    let mut enter = f32::NEG_INFINITY;
    let mut exit = f32::INFINITY;
    for face in hull.faces() {
        let denominator = direction.dot(face.normal);
        let distance = face.offset - origin.dot(face.normal);
        if denominator.abs() < 1e-8 {
            // Parallel to this face: a miss unless the ray is already on the
            // inside of it.
            if distance < 0.0 {
                return None;
            }
            continue;
        }
        let t = distance / denominator;
        if denominator < 0.0 {
            enter = enter.max(t);
        } else {
            exit = exit.min(t);
        }
        if enter > exit {
            return None;
        }
    }
    if exit < 0.0 {
        return None;
    }
    Some(enter.max(0.0))
}

/// Ray against a sphere at the origin.
fn sphere(origin: Vec3, direction: Vec3, radius: f32) -> Option<f32> {
    // |o + td|² = r², expanded into a quadratic in t. `a` is 1 for a unit
    // direction, so it drops out.
    let b = origin.dot(direction);
    let c = origin.length_squared() - radius * radius;
    if c > 0.0 && b > 0.0 {
        return None; // outside, pointing away
    }
    let discriminant = b * b - c;
    if discriminant < 0.0 {
        return None;
    }
    let root = discriminant.sqrt();
    let near = -b - root;
    // Inside the sphere the near root is negative and the far one is the exit;
    // a ray that starts inside is touching it now.
    Some(if near >= 0.0 { near } else { 0.0 })
}

/// Ray against an axis-aligned box at the origin: the slab test.
fn slab(origin: Vec3, direction: Vec3, half: Vec3) -> Option<f32> {
    let mut enter = f32::NEG_INFINITY;
    let mut exit = f32::INFINITY;

    for k in 0..3 {
        if direction[k].abs() < 1e-8 {
            // Parallel to this pair of faces: a miss unless already between
            // them. Without this the division below produces infinities that
            // happen to work on most inputs and produce NaN on the axis-
            // aligned ones, which are exactly the rays an editor generates
            // when somebody looks straight down.
            if origin[k].abs() > half[k] {
                return None;
            }
            continue;
        }
        let inverse = 1.0 / direction[k];
        let mut t0 = (-half[k] - origin[k]) * inverse;
        let mut t1 = (half[k] - origin[k]) * inverse;
        if t0 > t1 {
            std::mem::swap(&mut t0, &mut t1);
        }
        enter = enter.max(t0);
        exit = exit.min(t1);
        if enter > exit {
            return None;
        }
    }

    if exit < 0.0 {
        return None; // the box is behind
    }
    Some(enter.max(0.0))
}

/// The nearest body a ray hits, and how far away it is.
///
/// Walks every body, which is right for a pointer — one ray per click, against
/// a scene an editor is holding anyway — and wrong for anything that casts
/// thousands. Those want the broad phase, and this deliberately does not
/// pretend to be that.
pub fn pick(ray: &Ray, bodies: &fusion_core::BodySet) -> Option<(usize, f32)> {
    let mut best: Option<(usize, f32)> = None;
    for i in 0..bodies.len() {
        if bodies.is_removed(i) {
            continue;
        }
        // Cheap rejection first: a ray that misses the bounding sphere cannot
        // hit the shape inside it.
        let centre = fusion_math::to_f32(bodies.position[i]);
        let radius = bodies.shape[i].bounding_radius();
        let to_centre = centre - ray.origin;
        let along = to_centre.dot(ray.direction);
        let perpendicular = to_centre.length_squared() - along * along;
        if along < -radius || perpendicular > radius * radius {
            continue;
        }

        if let Some(t) = ray_vs_shape(ray, &bodies.shape[i], &bodies.pose(i)) {
            if best.is_none_or(|(_, best_t)| t < best_t) {
                best = Some((i, t));
            }
        }
    }
    best
}
