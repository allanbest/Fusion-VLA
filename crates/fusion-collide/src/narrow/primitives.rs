//! Analytic narrow-phase fast paths.
//!
//! PLAN.md §10 and §32: narrow phase, not broad phase, is the first real
//! bottleneck, and it is where hand-written paths pay. Sphere-sphere and
//! sphere-box are closed-form. Box-box uses SAT for the axis and face clipping
//! for the manifold, because a single deepest point — which is all GJK/EPA can
//! give — makes a resting box rock on one corner. A flat resting contact needs
//! a flat patch of contact points.

use fusion_core::Shape;
use fusion_math::{vec3, Isometry, Mat3, Vec3};

use crate::contact::WorldContact;

/// Contacts closer than this are treated as coincident and merged.
const POINT_MERGE_EPS: f32 = 1e-4;
/// Contacts are kept while the gap is below this, not only while it is
/// negative.
///
/// A box resting on another has four corners at a separation of roughly zero,
/// and rounding decides individually for each whether it counts. The manifold
/// then flickers between four points and one from frame to frame, the stack
/// rocks, and the rocking feeds itself. Keeping near-touching points gives the
/// solver a stable patch; it ignores the ones that are not actually
/// penetrating when it comes to solve them.
///
/// This is also the seed of speculative contacts (v0.3): a constraint that
/// exists slightly *before* the bodies touch is a constraint that can stop
/// them at the surface instead of after they have passed through it.
pub const DEFAULT_CONTACT_TOLERANCE: f32 = 5e-3;

/// Face axes win ties against edge-edge axes by this margin. Without the bias,
/// floating-point noise picks a near-parallel edge axis and the manifold
/// collapses to one point.
const FACE_AXIS_BIAS: f32 = 1.02;

/// Sphere against sphere.
pub fn sphere_sphere(
    center_a: Vec3,
    radius_a: f32,
    center_b: Vec3,
    radius_b: f32,
    tolerance: f32,
    out: &mut Vec<WorldContact>,
) {
    let delta = center_b - center_a;
    let (normal, distance) = delta.normalize_and_length();
    // Concentric spheres have no meaningful normal; pick one so the solver
    // still pushes them apart instead of producing NaNs.
    let normal = if distance > 1e-6 { normal } else { Vec3::Y };
    let separation = distance - radius_a - radius_b;
    if separation >= tolerance {
        return;
    }
    out.push(WorldContact {
        point_a: center_a + normal * radius_a,
        point_b: center_b - normal * radius_b,
        normal,
        separation,
    });
}

/// Sphere against box. The normal points from the sphere towards the box.
pub fn sphere_cuboid(
    center: Vec3,
    radius: f32,
    half_extents: Vec3,
    box_pose: &Isometry,
    tolerance: f32,
    out: &mut Vec<WorldContact>,
) {
    let local_center = box_pose.inverse_transform_point(center);
    let closest = local_center.clamp(-half_extents, half_extents);
    let delta = local_center - closest;
    let dist_sq = delta.length_squared();

    let (local_normal, separation, local_point_on_box) = if dist_sq > 1e-12 {
        // Centre outside the box: the closest surface point is unambiguous.
        let dist = dist_sq.sqrt();
        // `delta` runs box -> sphere, so the sphere -> box normal is its negation.
        (-(delta / dist), dist - radius, closest)
    } else {
        // Centre inside the box: escape through the nearest face.
        let to_face = half_extents - local_center.abs();
        let axis = (-to_face).max_axis();
        let sign = if local_center[axis] >= 0.0 { 1.0 } else { -1.0 };
        let mut n = Vec3::ZERO;
        let mut surface = local_center;
        match axis {
            0 => {
                n.x = -sign;
                surface.x = sign * half_extents.x;
            }
            1 => {
                n.y = -sign;
                surface.y = sign * half_extents.y;
            }
            _ => {
                n.z = -sign;
                surface.z = sign * half_extents.z;
            }
        }
        (n, -(radius + to_face[axis]), surface)
    };

    if separation >= tolerance {
        return;
    }
    let normal = box_pose.transform_direction(local_normal);
    out.push(WorldContact {
        point_a: center + normal * radius,
        point_b: box_pose.transform_point(local_point_on_box),
        normal,
        separation,
    });
}

/// Box against box: separating-axis test for the normal, then face clipping
/// for the manifold.
///
/// Contacts are **appended** to `out`; a separating axis returns without
/// touching what is already there. Anything else would make the routine unsafe
/// to call twice with the same buffer.
///
/// When two axes tie on penetration depth — a corner-on impact, where pushing
/// out sideways and pushing out upwards cost the same — the choice between
/// them is arbitrary, and both are correct minimum translations. GJK/EPA may
/// pick the other one for the same configuration.
/// Box against box, by the separating-axis theorem with face clipping.
///
/// `tolerance` widens every axis test by that much, which is what makes a box
/// capable of a **speculative** contact: a pair reported while still apart,
/// with a positive separation, so the solver can stop them at the surface
/// instead of discovering them through it.
///
/// It did not do that until v1.0, and the way it failed is the interesting
/// part. The parameter was here, it was passed in correctly from a pipeline
/// that sized it from the fastest body in the scene, and it was used — in the
/// last dozen lines, to decide which clipped points were close enough to keep.
/// The separating-axis test above never got that far: it returned on the first
/// axis with any gap at all. So the feature was present at both ends and
/// absent in the middle, and nothing failed, because every test of it put the
/// boxes in contact first.
///
/// What it cost: a box falling at 2.6 m/s covers 11 mm in a step at 240 Hz and
/// was not reported until it was 10 mm inside the floor. The impulse that then
/// pushes it out is a hundred times a resting one, Coulomb's cone is sized
/// from that impulse, and a single step of tidying up was licensed to remove
/// twice the box's whole horizontal speed. The bug reads as friction and lives
/// here.
pub fn cuboid_cuboid(
    half_a: Vec3,
    pose_a: &Isometry,
    half_b: Vec3,
    pose_b: &Isometry,
    tolerance: f32,
    out: &mut Vec<WorldContact>,
) {
    let axes_a = Mat3::from_quat(pose_a.rotation);
    let axes_b = Mat3::from_quat(pose_b.rotation);
    let center_delta = pose_b.translation - pose_a.translation;

    let mut best_depth = f32::INFINITY;
    // The chosen axis's depth before the edge bias, which is the one the
    // contact is built from.
    let mut best_raw = 0.0f32;
    let mut best_normal = Vec3::ZERO;
    // 0..3 = face of A, 3..6 = face of B, 6.. = edge-edge.
    let mut best_kind = usize::MAX;

    // --- face axes ---
    for i in 0..3 {
        for (offset, axes) in [(0usize, &axes_a), (3usize, &axes_b)] {
            let axis = axes.col(i);
            let Some(depth) = axis_overlap(
                axis,
                center_delta,
                half_a,
                &axes_a,
                half_b,
                &axes_b,
                tolerance,
            ) else {
                return; // separating axis found: no contact at all
            };
            if depth < best_depth {
                best_depth = depth;
                best_raw = depth;
                best_normal = axis;
                best_kind = offset + i;
            }
        }
    }

    // --- edge-edge axes ---
    let mut best_edge = (0usize, 0usize);
    for i in 0..3 {
        for j in 0..3 {
            let axis = axes_a.col(i).cross(axes_b.col(j));
            let len_sq = axis.length_squared();
            if len_sq < 1e-8 {
                continue; // parallel edges: already covered by a face axis
            }
            let axis = axis / len_sq.sqrt();
            let Some(depth) = axis_overlap(
                axis,
                center_delta,
                half_a,
                &axes_a,
                half_b,
                &axes_b,
                tolerance,
            ) else {
                return;
            };
            // The bias has to *penalise* the edge axis on both signs of depth.
            // Multiplying works while boxes overlap, where a smaller depth
            // wins; once they are apart the depth is negative and multiplying
            // makes it more negative, which hands every near-miss to an edge.
            let biased = if depth > 0.0 {
                depth * FACE_AXIS_BIAS
            } else {
                depth / FACE_AXIS_BIAS
            };
            if biased < best_depth {
                best_depth = biased;
                best_raw = depth;
                best_normal = axis;
                best_kind = 6 + i * 3 + j;
                best_edge = (i, j);
            }
        }
    }

    if best_kind == usize::MAX {
        return;
    }
    // Orient the chosen axis from A towards B.
    if best_normal.dot(center_delta) < 0.0 {
        best_normal = -best_normal;
    }

    if best_kind >= 6 {
        edge_edge_contact(
            best_edge,
            best_normal,
            best_raw,
            half_a,
            pose_a,
            &axes_a,
            half_b,
            pose_b,
            &axes_b,
            out,
        );
        return;
    }

    // --- face contact: clip the incident face against the reference face ---
    let a_is_reference = best_kind < 3;
    let (ref_half, ref_pose, ref_axes, inc_half, inc_pose, inc_axes) = if a_is_reference {
        (half_a, pose_a, &axes_a, half_b, pose_b, &axes_b)
    } else {
        (half_b, pose_b, &axes_b, half_a, pose_a, &axes_a)
    };
    // Reference normal points out of the reference box, towards the other one.
    let ref_normal = if a_is_reference {
        best_normal
    } else {
        -best_normal
    };
    let ref_axis = best_kind % 3;
    let ref_sign = if ref_axes.col(ref_axis).dot(ref_normal) >= 0.0 {
        1.0
    } else {
        -1.0
    };

    // The incident face is the one most anti-parallel to the reference normal.
    let mut inc_axis = 0;
    let mut inc_sign = 1.0;
    let mut most_negative = f32::INFINITY;
    for k in 0..3 {
        let d = inc_axes.col(k).dot(ref_normal);
        if d < most_negative {
            most_negative = d;
            inc_axis = k;
            inc_sign = 1.0;
        }
        if -d < most_negative {
            most_negative = -d;
            inc_axis = k;
            inc_sign = -1.0;
        }
    }

    let mut polygon: Vec<Vec3> = face_vertices(inc_half, inc_pose, inc_axis, inc_sign).to_vec();

    // Clip against the four side planes of the reference face.
    let side_axes: [usize; 2] = match ref_axis {
        0 => [1, 2],
        1 => [0, 2],
        _ => [0, 1],
    };
    for &s in &side_axes {
        let axis = ref_axes.col(s);
        let offset = axis.dot(ref_pose.translation) + ref_half[s];
        polygon = clip_to_plane(&polygon, axis, offset);
        let offset = -axis.dot(ref_pose.translation) + ref_half[s];
        polygon = clip_to_plane(&polygon, -axis, offset);
        if polygon.is_empty() {
            return;
        }
    }

    // The reference face plane sits one half-extent out along its own axis.
    let plane_point =
        ref_pose.translation + ref_axes.col(ref_axis) * (ref_sign * ref_half[ref_axis]);
    let plane_d = ref_normal.dot(plane_point);

    let mut candidates: Vec<WorldContact> = Vec::with_capacity(polygon.len());
    for p in polygon {
        let separation = ref_normal.dot(p) - plane_d;
        if separation > tolerance {
            continue; // too far away to be part of this patch
        }
        let on_ref = p - ref_normal * separation;
        let (point_a, point_b) = if a_is_reference {
            (on_ref, p)
        } else {
            (p, on_ref)
        };
        candidates.push(WorldContact {
            point_a,
            point_b,
            normal: best_normal,
            separation,
        });
    }

    reduce_manifold(&mut candidates);
    out.extend(candidates);
}

/// Penetration depth along `axis`, or `None` if the axis separates the boxes by
/// more than `tolerance`. `axis` must be unit length.
///
/// A **negative** return is a gap rather than an overlap, and returning one at
/// all is what makes a box capable of a speculative contact. Without the
/// tolerance this returned `None` the instant the boxes stopped touching, and
/// the consequence was not subtle: `cuboid_cuboid` took a `tolerance`
/// parameter, used it only when deciding which clipped points to keep, and
/// never got that far, so **no pair of boxes has ever produced a speculative
/// contact**. Spheres always did, which is why a thrown ball rolled and a
/// thrown box landed in sand — see the note on `cuboid_cuboid`.
fn axis_overlap(
    axis: Vec3,
    center_delta: Vec3,
    half_a: Vec3,
    axes_a: &Mat3,
    half_b: Vec3,
    axes_b: &Mat3,
    tolerance: f32,
) -> Option<f32> {
    let project = |half: Vec3, axes: &Mat3| {
        half.x * axes.col(0).dot(axis).abs()
            + half.y * axes.col(1).dot(axis).abs()
            + half.z * axes.col(2).dot(axis).abs()
    };
    let distance = center_delta.dot(axis).abs();
    let reach = project(half_a, axes_a) + project(half_b, axes_b);
    if distance > reach + tolerance {
        None
    } else {
        Some(reach - distance)
    }
}

/// The four world-space corners of one box face.
fn face_vertices(half: Vec3, pose: &Isometry, axis: usize, sign: f32) -> [Vec3; 4] {
    let (u, v) = match axis {
        0 => (1, 2),
        1 => (0, 2),
        _ => (0, 1),
    };
    let mut out = [Vec3::ZERO; 4];
    for (k, corner) in [(-1.0, -1.0), (1.0, -1.0), (1.0, 1.0), (-1.0, 1.0)]
        .iter()
        .enumerate()
    {
        let mut local = Vec3::ZERO;
        local_set(&mut local, axis, sign * half[axis]);
        local_set(&mut local, u, corner.0 * half[u]);
        local_set(&mut local, v, corner.1 * half[v]);
        out[k] = pose.transform_point(local);
    }
    out
}

#[inline]
fn local_set(v: &mut Vec3, axis: usize, value: f32) {
    match axis {
        0 => v.x = value,
        1 => v.y = value,
        _ => v.z = value,
    }
}

/// Sutherland-Hodgman clip of a convex polygon to the half-space
/// `dot(normal, p) <= offset`.
fn clip_to_plane(polygon: &[Vec3], normal: Vec3, offset: f32) -> Vec<Vec3> {
    let mut out = Vec::with_capacity(polygon.len() + 1);
    if polygon.is_empty() {
        return out;
    }
    let mut prev = *polygon.last().unwrap();
    let mut prev_d = normal.dot(prev) - offset;
    for &curr in polygon {
        let curr_d = normal.dot(curr) - offset;
        if curr_d <= 0.0 {
            if prev_d > 0.0 {
                let t = prev_d / (prev_d - curr_d);
                out.push(prev.lerp(curr, t));
            }
            out.push(curr);
        } else if prev_d <= 0.0 {
            let t = prev_d / (prev_d - curr_d);
            out.push(prev.lerp(curr, t));
        }
        prev = curr;
        prev_d = curr_d;
    }
    out
}

/// Clipping can yield up to eight points; the solver keeps four. Choosing the
/// four deepest would cluster them along one edge and let the box rock, so keep
/// the deepest point and then spread for area.
fn reduce_manifold(points: &mut Vec<WorldContact>) {
    // Merge coincident points first — clipping duplicates shared corners.
    let mut i = 0;
    while i < points.len() {
        let mut j = i + 1;
        while j < points.len() {
            if points[i].point_b.distance_squared(points[j].point_b)
                < POINT_MERGE_EPS * POINT_MERGE_EPS
            {
                if points[j].separation < points[i].separation {
                    points[i] = points[j];
                }
                points.swap_remove(j);
            } else {
                j += 1;
            }
        }
        i += 1;
    }
    if points.len() <= crate::contact::MAX_CONTACT_POINTS {
        return;
    }

    let deepest = (0..points.len())
        .min_by(|&x, &y| points[x].separation.total_cmp(&points[y].separation))
        .unwrap();
    points.swap(0, deepest);

    let farthest = (1..points.len())
        .max_by(|&x, &y| {
            points[0]
                .point_b
                .distance_squared(points[x].point_b)
                .total_cmp(&points[0].point_b.distance_squared(points[y].point_b))
        })
        .unwrap();
    points.swap(1, farthest);

    let area_of = |p: Vec3, a: Vec3, b: Vec3| (b - a).cross(p - a).length();
    let third =
        (2..points.len())
            .max_by(|&x, &y| {
                area_of(points[x].point_b, points[0].point_b, points[1].point_b).total_cmp(
                    &area_of(points[y].point_b, points[0].point_b, points[1].point_b),
                )
            })
            .unwrap();
    points.swap(2, third);

    let fourth = (3..points.len())
        .max_by(|&x, &y| {
            let score = |k: usize| {
                area_of(points[k].point_b, points[0].point_b, points[2].point_b)
                    + area_of(points[k].point_b, points[1].point_b, points[2].point_b)
            };
            score(x).total_cmp(&score(y))
        })
        .unwrap();
    points.swap(3, fourth);

    points.truncate(crate::contact::MAX_CONTACT_POINTS);
}

/// A crossed-edge contact is a single point: the closest pair on the two edges
/// that produced the separating axis.
#[allow(clippy::too_many_arguments)]
fn edge_edge_contact(
    edge: (usize, usize),
    normal: Vec3,
    depth: f32,
    half_a: Vec3,
    pose_a: &Isometry,
    axes_a: &Mat3,
    half_b: Vec3,
    pose_b: &Isometry,
    axes_b: &Mat3,
    out: &mut Vec<WorldContact>,
) {
    let midpoint = |half: Vec3, pose: &Isometry, axes: &Mat3, skip: usize, dir: Vec3| {
        let mut p = pose.translation;
        for k in 0..3 {
            if k == skip {
                continue;
            }
            let axis = axes.col(k);
            let sign = if axis.dot(dir) >= 0.0 { 1.0 } else { -1.0 };
            p += axis * (sign * half[k]);
        }
        p
    };

    let mid_a = midpoint(half_a, pose_a, axes_a, edge.0, normal);
    let mid_b = midpoint(half_b, pose_b, axes_b, edge.1, -normal);
    let dir_a = axes_a.col(edge.0);
    let dir_b = axes_b.col(edge.1);

    let (pa, pb) = closest_points_on_segments(
        mid_a - dir_a * half_a[edge.0],
        mid_a + dir_a * half_a[edge.0],
        mid_b - dir_b * half_b[edge.1],
        mid_b + dir_b * half_b[edge.1],
    );
    out.push(WorldContact {
        point_a: pa,
        point_b: pb,
        normal,
        separation: -depth,
    });
}

/// Closest points between two segments, clamped to their endpoints.
pub fn closest_points_on_segments(p1: Vec3, q1: Vec3, p2: Vec3, q2: Vec3) -> (Vec3, Vec3) {
    let d1 = q1 - p1;
    let d2 = q2 - p2;
    let r = p1 - p2;
    let a = d1.length_squared();
    let e = d2.length_squared();
    let f = d2.dot(r);

    if a <= 1e-12 && e <= 1e-12 {
        return (p1, p2);
    }
    let (mut s, mut t);
    if a <= 1e-12 {
        s = 0.0;
        t = fusion_math::clamp(f / e, 0.0, 1.0);
    } else {
        let c = d1.dot(r);
        if e <= 1e-12 {
            t = 0.0;
            s = fusion_math::clamp(-c / a, 0.0, 1.0);
        } else {
            let b = d1.dot(d2);
            let denom = a * e - b * b;
            s = if denom.abs() > 1e-12 {
                fusion_math::clamp((b * f - c * e) / denom, 0.0, 1.0)
            } else {
                0.0
            };
            t = (b * s + f) / e;
            if t < 0.0 {
                t = 0.0;
                s = fusion_math::clamp(-c / a, 0.0, 1.0);
            } else if t > 1.0 {
                t = 1.0;
                s = fusion_math::clamp((b - c) / a, 0.0, 1.0);
            }
        }
    }
    (p1 + d1 * s, p2 + d2 * t)
}

/// Dispatch to the right routine for a shape pair. Contacts are appended to
/// `out` with the normal pointing from A towards B.
pub fn collide(
    shape_a: &Shape,
    pose_a: &Isometry,
    shape_b: &Shape,
    pose_b: &Isometry,
    tolerance: f32,
    out: &mut Vec<WorldContact>,
) {
    match (shape_a, shape_b) {
        // A hull against anything, in either order. Everything below this is a
        // hand-written pair; a hull is the case that cannot be, because the
        // shape is whatever somebody imported.
        (Shape::Hull(_), _) | (_, Shape::Hull(_)) => {
            hull_contacts(shape_a, pose_a, shape_b, pose_b, tolerance, out);
        }
        (Shape::Sphere { radius: ra }, Shape::Sphere { radius: rb }) => {
            sphere_sphere(
                pose_a.translation,
                *ra,
                pose_b.translation,
                *rb,
                tolerance,
                out,
            );
        }
        (Shape::Sphere { radius }, Shape::Cuboid { half_extents }) => {
            sphere_cuboid(
                pose_a.translation,
                *radius,
                *half_extents,
                pose_b,
                tolerance,
                out,
            );
        }
        (Shape::Cuboid { half_extents }, Shape::Sphere { radius }) => {
            // Solve in the sphere-first form and flip the result, rather than
            // duplicating the routine with every sign inverted.
            let start = out.len();
            sphere_cuboid(
                pose_b.translation,
                *radius,
                *half_extents,
                pose_a,
                tolerance,
                out,
            );
            for c in &mut out[start..] {
                *c = WorldContact {
                    point_a: c.point_b,
                    point_b: c.point_a,
                    normal: -c.normal,
                    separation: c.separation,
                };
            }
        }
        (Shape::Cuboid { half_extents: ha }, Shape::Cuboid { half_extents: hb }) => {
            let start = out.len();
            cuboid_cuboid(*ha, pose_a, *hb, pose_b, tolerance, out);
            // Reaching further than a resting pair does, which is a pair
            // moving fast: the corners on their way in too.
            if tolerance > DEFAULT_CONTACT_TOLERANCE + 1e-3 && out.len() > start {
                approaching_corners(*ha, pose_a, *hb, pose_b, tolerance, start, out);
            }
        }
    }
}

/// Every corner of either box still clear of the other but within
/// `tolerance` of it, each with its own normal, beside the separating-axis
/// contact already in `out[start..]`.
///
/// The separating axis picks one feature, and its contact has one normal.
/// A box that has just hit another corner-first is spinning, and the face
/// that swings into it next is a different feature with a different normal:
/// found only once it was already 25 mm in, at 60 m/s. Clear of the other
/// box, a corner's nearest point on it is exact and so is its normal, so
/// these are safe to add; one already inside is the separating axis's to
/// describe, and is left to it.
fn approaching_corners(
    half_a: Vec3,
    pose_a: &Isometry,
    half_b: Vec3,
    pose_b: &Isometry,
    tolerance: f32,
    start: usize,
    out: &mut Vec<WorldContact>,
) {
    let box_a = Shape::Cuboid {
        half_extents: half_a,
    };
    let box_b = Shape::Cuboid {
        half_extents: half_b,
    };
    let found = out.len();
    let near_existing = |out: &[WorldContact], p: Vec3| {
        out[start..found]
            .iter()
            .any(|c| c.point_a.distance_squared(p) < 1e-4 || c.point_b.distance_squared(p) < 1e-4)
    };
    for corner in corners(half_a, pose_a) {
        if let Some((normal, separation)) =
            crate::narrow::point_vs_shape(corner, 0.0, &box_b, pose_b, tolerance)
        {
            if separation > 0.0 && !near_existing(out, corner) {
                // `normal` runs from B towards the corner, which is A's.
                out.push(WorldContact {
                    point_a: corner,
                    point_b: corner - normal * separation,
                    normal: -normal,
                    separation,
                });
            }
        }
    }
    for corner in corners(half_b, pose_b) {
        if let Some((normal, separation)) =
            crate::narrow::point_vs_shape(corner, 0.0, &box_a, pose_a, tolerance)
        {
            if separation > 0.0 && !near_existing(out, corner) {
                out.push(WorldContact {
                    point_a: corner - normal * separation,
                    point_b: corner,
                    normal,
                    separation,
                });
            }
        }
    }
}

/// Local-frame helper used by tests: the eight corners of a box.
pub fn corners(half: Vec3, pose: &Isometry) -> [Vec3; 8] {
    let mut out = [Vec3::ZERO; 8];
    for (i, slot) in out.iter_mut().enumerate() {
        let s = vec3(
            if i & 1 == 0 { -1.0 } else { 1.0 },
            if i & 2 == 0 { -1.0 } else { 1.0 },
            if i & 4 == 0 { -1.0 } else { 1.0 },
        );
        *slot = pose.transform_point(half.mul_element_wise(s));
    }
    out
}

/// Contacts for a pair where at least one shape is a convex hull.
///
/// ## Why not GJK and EPA alone
///
/// They are the general answer and they produce **one** point — the deepest.
/// One point is enough for a bouncing ball and not enough for anything that
/// has to *rest*: a crate on a floor supported at a single corner rocks, and
/// no number of solver iterations fixes a manifold that never mentions the
/// other three corners. `crate::narrow::collide_gjk_epa` says so itself.
///
/// So the vertices do the work. Each vertex of the hull is tested against the
/// other shape with the same point-vs-shape routine cloth and fluid particles
/// use, and every one within tolerance becomes a contact. A crate resting flat
/// reports four; tipped onto an edge, two; balanced on a corner, one — which
/// is the manifold each of those situations actually has.
///
/// ## What that misses, and what catches it
///
/// Two hulls meeting edge-to-edge — the wedges and splinters a break leaves
/// cross like that all the time — have no vertex inside the other. Where
/// they overlap and no vertex is inside, GJK and EPA give the single deepest
/// point, which is exactly the contact an edge crossing has. Where a vertex
/// is inside, its own face's depth is exact, and EPA's is not to be trusted
/// against a 40 m floor: capped to it, a shard's corner 40 mm into the floor
/// read as half a millimetre and stayed there.
fn hull_contacts(
    shape_a: &Shape,
    pose_a: &Isometry,
    shape_b: &Shape,
    pose_b: &Isometry,
    tolerance: f32,
    out: &mut Vec<WorldContact>,
) {
    let before = out.len();

    // A's vertices against B, then B's against A, with the normal flipped so
    // that every contact points from A towards B whichever side found it.
    if let Shape::Hull(hull) = shape_a {
        for vertex in hull.vertices() {
            let world = pose_a.transform_point(*vertex);
            if let Some((normal, separation)) =
                crate::narrow::point_vs_shape(world, 0.0, shape_b, pose_b, tolerance)
            {
                // `point_vs_shape` returns the normal pointing from the shape
                // towards the point — from B towards A here — and B's surface
                // is the point less that normal times the separation. It was
                // written plus, which put B's anchor on the wrong side of A's:
                // a vertex inside B read as clear of it and one clear of it as
                // inside, so hulls sank into what they lay on and were pushed
                // off what they had not reached.
                out.push(WorldContact {
                    point_a: world,
                    point_b: world - normal * separation,
                    normal: -normal,
                    separation,
                });
            }
        }
    }
    if let Shape::Hull(hull) = shape_b {
        for vertex in hull.vertices() {
            let world = pose_b.transform_point(*vertex);
            if let Some((normal, separation)) =
                crate::narrow::point_vs_shape(world, 0.0, shape_a, pose_a, tolerance)
            {
                out.push(WorldContact {
                    point_a: world - normal * separation,
                    point_b: world,
                    normal,
                    separation,
                });
            }
        }
    }

    // Deepest first: a pair's contacts share the first one's normal, and the
    // deepest vertex's face is the one to push out along.
    if let Some(deepest) =
        (before..out.len()).min_by(|&a, &b| out[a].separation.total_cmp(&out[b].separation))
    {
        out.swap(before, deepest);
    }

    // Edges crossing edges: the overlap vertices cannot see, and EPA's single
    // deepest point is exactly the contact it has. Asked whenever the pair
    // overlaps, not only when no vertex is inside: a shard resting one
    // vertex half a millimetre into a steel block, its edges crossing the
    // block's ten centimetres deep, reported the half millimetre and lay
    // ten centimetres inside it, unseen, for as long as that vertex stayed.
    // Kept where it is deeper than every vertex by more than a resting
    // contact's slop — where the vertices are the whole story, they say it
    // better.
    let vertex_depth = out[before..]
        .iter()
        .fold(0.0f32, |d, c| d.max(-c.separation));
    let mut deepest = Vec::new();
    crate::narrow::collide_gjk_epa(shape_a, pose_a, shape_b, pose_b, &mut deepest);
    if let Some(mtv) = deepest.first().copied() {
        if -mtv.separation > vertex_depth + EDGE_DEPTH_MARGIN && mtv.normal.length_squared() > 0.5 {
            out.insert(before, mtv);
        }
    }
}

/// How much deeper than every vertex an edge crossing must go before
/// [`hull_contacts`] adds it, metres.
const EDGE_DEPTH_MARGIN: f32 = 1e-3;
