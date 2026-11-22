//! Narrow phase.
//!
//! Two paths, deliberately: analytic routines for the shape pairs Fusion
//! actually ships, and GJK/EPA as the general convex fallback that every
//! future shape arrives through.

pub mod epa;
pub mod gjk;
pub mod primitives;

use fusion_core::Shape;
use fusion_math::Isometry;

use crate::contact::WorldContact;

/// Contact between a point of radius `radius` and a shape.
///
/// A particle is a sphere, so this is the existing analytic sphere paths with
/// the arguments arranged the way the particle solvers want them. Returning
/// the normal pointing *away* from the shape matches how a particle wants to
/// be pushed, which is the opposite of the A-to-B convention used between
/// bodies — stated here rather than left to be discovered.
///
/// **Allocation-free**, and it has to be. Cloth calls this once per particle
/// per body per iteration; a fluid of ten thousand particles calls it a
/// million times a step. The obvious spelling — build a `Vec`, run the shared
/// primitive, take the first element — cost 40% of the step time in a fluid
/// scene and every bit of it was `malloc`.
pub fn point_vs_shape(
    point: fusion_math::Vec3,
    radius: f32,
    shape: &Shape,
    pose: &Isometry,
    tolerance: f32,
) -> Option<(fusion_math::Vec3, f32)> {
    use fusion_math::Vec3;
    match shape {
        Shape::Sphere { radius: r } => {
            let delta = point - pose.translation;
            let (direction, distance) = delta.normalize_and_length();
            let separation = distance - (radius + r);
            if separation >= tolerance {
                return None;
            }
            // Coincident centres have no meaningful direction; pick one rather
            // than returning a zero normal the solver would multiply by.
            let normal = if distance > 1e-9 { direction } else { Vec3::Y };
            Some((normal, separation))
        }
        // A hull, by its own face planes: the deepest one is the surface a
        // point inside is closest to, and its normal is the direction out.
        // Exact inside, and outside it under-estimates near an edge — which is
        // the direction that keeps a particle out rather than sucking it in.
        Shape::Hull(hull) => {
            let local = pose.inverse_transform_point(point);
            let mut best = f32::NEG_INFINITY;
            let mut local_normal = Vec3::Y;
            for face in hull.faces() {
                let distance = local.dot(face.normal) - face.offset;
                if distance > best {
                    best = distance;
                    local_normal = face.normal;
                }
            }
            let separation = best - radius;
            if separation >= tolerance {
                return None;
            }
            Some((pose.transform_direction(local_normal), separation))
        }
        Shape::Cuboid { half_extents } => {
            let local = pose.inverse_transform_point(point);
            let closest = local.clamp(-*half_extents, *half_extents);
            let delta = local - closest;
            let dist_sq = delta.length_squared();
            let (local_normal, separation) = if dist_sq > 1e-12 {
                let dist = dist_sq.sqrt();
                (delta / dist, dist - radius)
            } else {
                // Centre inside the box: escape through the nearest face.
                let to_face = *half_extents - local.abs();
                let axis = (-to_face).max_axis();
                let sign = if local[axis] >= 0.0 { 1.0 } else { -1.0 };
                let mut n = Vec3::ZERO;
                n[axis] = sign;
                (n, -(radius + to_face[axis]))
            };
            if separation >= tolerance {
                return None;
            }
            Some((pose.transform_direction(local_normal), separation))
        }
    }
}

/// General convex-convex contact through GJK, then EPA if they overlap.
///
/// Produces a single deepest point. That is enough for a bouncing sphere and
/// not enough for a resting box, which is why `primitives::collide` exists and
/// is what the pipeline calls.
pub fn collide_gjk_epa(
    shape_a: &Shape,
    pose_a: &Isometry,
    shape_b: &Shape,
    pose_b: &Isometry,
    out: &mut Vec<WorldContact>,
) {
    match gjk::intersect(shape_a, pose_a, shape_b, pose_b) {
        gjk::GjkResult::Intersecting(simplex) => {
            if let Some(c) = epa::penetration(shape_a, pose_a, shape_b, pose_b, &simplex) {
                out.push(c);
            }
        }
        gjk::GjkResult::Separated { .. } | gjk::GjkResult::Indeterminate => {}
    }
}
