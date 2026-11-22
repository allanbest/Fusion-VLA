//! Contacts between a convex hull and anything else: the shape every broken
//! piece is.
//!
//! The solver recovers each contact's separation from its two anchors, so
//! the anchors have to say what the separation says. They did not: B's
//! anchor was put on the wrong side of A's, so a hull vertex inside a floor
//! read as clear of it and one clear of it as inside. Hulls sank into what
//! they lay on and were pushed off what they had not yet reached.

use fusion_collide::narrow::primitives::collide;
use fusion_core::Shape;
use fusion_math::{vec3, Isometry, Quat, Vec3};

fn cube_hull(half: f32) -> Shape {
    let corners: Vec<Vec3> = (0..8)
        .map(|q| {
            Vec3::new(
                if q & 1 == 0 { -half } else { half },
                if q & 2 == 0 { -half } else { half },
                if q & 4 == 0 { -half } else { half },
            )
        })
        .collect();
    Shape::hull(&corners).unwrap()
}

#[test]
fn a_hull_contacts_anchors_agree_with_its_separation() {
    let hull = cube_hull(0.5);
    let floor = Shape::cuboid(vec3(10.0, 1.0, 10.0));
    let floor_pose = Isometry::new(vec3(0.0, -0.5, 0.0), Quat::IDENTITY);
    // 2 cm into the floor, and 1 cm above it; the hull first and second.
    for y in [0.48f32, 0.51] {
        let hull_pose = Isometry::new(vec3(0.0, y, 0.0), Quat::from_rotation_y(0.3));
        for hull_first in [true, false] {
            let mut out = Vec::new();
            if hull_first {
                collide(&hull, &hull_pose, &floor, &floor_pose, 0.05, &mut out);
            } else {
                collide(&floor, &floor_pose, &hull, &hull_pose, 0.05, &mut out);
            }
            assert!(!out.is_empty());
            for c in &out {
                let from_anchors = (c.point_b - c.point_a).dot(c.normal);
                assert!(
                    (from_anchors - c.separation).abs() < 1e-4,
                    "y {y}, hull first {hull_first}: anchors say {from_anchors}, separation {}",
                    c.separation
                );
                assert!((c.separation - (y - 0.5)).abs() < 1e-3, "{c:?}");
            }
        }
    }
}

#[test]
fn two_hulls_crossing_edge_to_edge_still_touch() {
    // Two long bars crossed like an X, one dropped 1 cm into the other: no
    // vertex of either is inside the other, and they still overlap.
    let bar: Vec<Vec3> = (0..8)
        .map(|q| {
            Vec3::new(
                if q & 1 == 0 { -1.0 } else { 1.0 },
                if q & 2 == 0 { -0.05 } else { 0.05 },
                if q & 4 == 0 { -0.05 } else { 0.05 },
            )
        })
        .collect();
    let bar = Shape::hull(&bar).unwrap();
    let below = Isometry::new(
        Vec3::ZERO,
        Quat::from_rotation_x(std::f32::consts::FRAC_PI_4),
    );
    let above = Isometry::new(
        vec3(0.0, 0.0707 * 2.0 - 0.01, 0.0),
        Quat::from_rotation_y(std::f32::consts::FRAC_PI_2)
            * Quat::from_rotation_x(std::f32::consts::FRAC_PI_4),
    );
    let mut out = Vec::new();
    collide(&bar, &below, &bar, &above, 0.0, &mut out);
    assert!(
        out.iter().any(|c| c.separation < -0.005),
        "the crossing is found: {out:?}"
    );
}

/// However two hulls overlap, the narrow phase sees at least as deep as GJK
/// and EPA do. Only the vertices were asked unless none was inside at all,
/// so a shard with one vertex half a millimetre into a steel block, its
/// edges crossing the block's a hundred millimetres deep, reported the half
/// millimetre and lay inside the block, unseen.
#[test]
fn edges_crossing_deep_are_seen_beside_a_shallow_vertex() {
    let shard = Shape::hull(&[
        vec3(0.1, -0.1, 0.05),
        vec3(-0.12, 0.08, -0.1),
        vec3(0.05, 0.13, 0.11),
        vec3(-0.07, -0.11, 0.12),
        vec3(0.14, 0.04, -0.09),
        vec3(-0.02, -0.06, -0.14),
    ])
    .unwrap();
    let block = Shape::cuboid(Vec3::splat(0.3));
    let mut state = 0x2545_f491_4f6c_dd1du64;
    let mut next = || {
        state ^= state << 13;
        state ^= state >> 7;
        state ^= state << 17;
        (state >> 40) as f32 / (1u64 << 24) as f32 * 2.0 - 1.0
    };
    let mut missed = (0.0f32, 0.0f32);
    let mut tried = 0;
    for _ in 0..4000 {
        let mut quat = || Quat::from_xyzw(next(), next(), next(), next()).normalize();
        let (qa, qb) = (quat(), quat());
        let offset = vec3(next(), next(), next()) * 0.3;
        for other in [&block, &shard] {
            let a = Isometry::new(Vec3::ZERO, qa);
            let b = Isometry::new(offset, qb);
            let mut truth = Vec::new();
            fusion_collide::narrow::collide_gjk_epa(other, &a, &shard, &b, &mut truth);
            let Some(epa) = truth.first() else { continue };
            if epa.separation > -0.005 {
                continue;
            }
            tried += 1;
            let mut out = Vec::new();
            collide(other, &a, &shard, &b, 0.0, &mut out);
            let seen = out.iter().fold(0.0f32, |d, c| d.max(-c.separation));
            if -epa.separation - seen > missed.0 - missed.1 {
                missed = (-epa.separation, seen);
            }
        }
    }
    assert!(tried > 500, "only {tried} overlapping poses");
    assert!(
        missed.0 - missed.1 < 1e-3,
        "an overlap {:.1} mm deep was seen as {:.1} mm",
        missed.0 * 1e3,
        missed.1 * 1e3
    );
}
