//! Adversarial narrow-phase tests.
//!
//! The tests in `narrow_phase.rs` check that the right answer comes out for
//! well-posed inputs. These check what happens for inputs that are *not* well
//! posed — degenerate shapes, coincident bodies, huge coordinates, deep
//! overlaps — because those are what a real scene eventually contains, and a
//! NaN produced here propagates silently through the whole solve.

use fusion_collide::narrow::{collide_gjk_epa, primitives};
use fusion_collide::WorldContact;
use fusion_core::Shape;
use fusion_math::{vec3, Isometry, Quat, Vec3};

fn at(p: Vec3) -> Isometry {
    Isometry::from_translation(p)
}

fn boxx(half: Vec3) -> Shape {
    Shape::Cuboid { half_extents: half }
}

fn check_finite(contacts: &[WorldContact], case: &str) {
    for c in contacts {
        assert!(
            c.point_a.is_finite(),
            "{case}: point_a not finite: {:?}",
            c.point_a
        );
        assert!(
            c.point_b.is_finite(),
            "{case}: point_b not finite: {:?}",
            c.point_b
        );
        assert!(
            c.normal.is_finite(),
            "{case}: normal not finite: {:?}",
            c.normal
        );
        assert!(c.separation.is_finite(), "{case}: separation not finite");
        let len = c.normal.length();
        assert!(
            (len - 1.0).abs() < 1e-3,
            "{case}: contact normals must be unit length, got {len}"
        );
    }
}

#[test]
fn coincident_bodies_produce_a_finite_push() {
    // Two bodies spawned at exactly the same place is a scene-authoring
    // mistake, not a physics question — but it must resolve, not NaN.
    let mut out = Vec::new();
    primitives::collide(
        &Shape::sphere(1.0),
        &at(Vec3::ZERO),
        &Shape::sphere(1.0),
        &at(Vec3::ZERO),
        0.0,
        &mut out,
    );
    check_finite(&out, "coincident spheres");
    assert_eq!(out.len(), 1);
    assert!(out[0].separation < 0.0);

    out.clear();
    primitives::collide(
        &boxx(Vec3::ONE),
        &at(Vec3::ZERO),
        &boxx(Vec3::ONE),
        &at(Vec3::ZERO),
        0.0,
        &mut out,
    );
    check_finite(&out, "coincident boxes");
    assert!(
        !out.is_empty(),
        "fully coincident boxes must report contact"
    );

    out.clear();
    primitives::collide(
        &Shape::sphere(0.5),
        &at(Vec3::ZERO),
        &boxx(Vec3::ONE),
        &at(Vec3::ZERO),
        0.0,
        &mut out,
    );
    check_finite(&out, "sphere at box centre");
    assert_eq!(out.len(), 1);
}

#[test]
fn degenerate_shapes_do_not_produce_nan() {
    let cases: Vec<(&str, Shape, Shape)> = vec![
        ("zero sphere", Shape::sphere(0.0), Shape::sphere(1.0)),
        ("zero box", boxx(Vec3::ZERO), boxx(Vec3::ONE)),
        ("flat box", boxx(vec3(1.0, 0.0, 1.0)), boxx(Vec3::ONE)),
        ("needle box", boxx(vec3(1e-4, 2.0, 1e-4)), boxx(Vec3::ONE)),
        ("tiny sphere", Shape::sphere(1e-6), boxx(Vec3::ONE)),
    ];
    for (name, a, b) in cases {
        let mut out = Vec::new();
        primitives::collide(
            &a,
            &at(vec3(0.0, 0.4, 0.0)),
            &b,
            &at(Vec3::ZERO),
            0.0,
            &mut out,
        );
        check_finite(&out, name);
    }
}

#[test]
fn deep_overlap_reports_a_sane_normal() {
    // A box almost entirely inside another. The minimum translation is out
    // through the nearest face, and the depth must not exceed the geometry.
    let big = boxx(Vec3::splat(2.0));
    let small = boxx(Vec3::splat(0.5));
    let mut out = Vec::new();
    primitives::collide(
        &big,
        &at(Vec3::ZERO),
        &small,
        &at(vec3(0.0, 1.2, 0.0)),
        0.0,
        &mut out,
    );
    check_finite(&out, "deep overlap");
    assert!(!out.is_empty());
    for c in &out {
        assert!(
            c.normal.approx_eq(Vec3::Y, 1e-3),
            "shortest way out is up: {:?}",
            c.normal
        );
        assert!(
            -c.separation <= 2.5 + 1e-3,
            "depth {} exceeds the geometry",
            -c.separation
        );
    }
}

#[test]
fn contacts_survive_large_coordinates() {
    // The same contact, evaluated a kilometre from the origin. PLAN.md §7 is
    // about 10^21 m; this is the version of that problem which arrives in an
    // ordinary scene.
    let cube = boxx(Vec3::ONE);
    let reference = {
        let mut out = Vec::new();
        primitives::collide(
            &cube,
            &at(Vec3::ZERO),
            &cube,
            &at(vec3(0.0, 1.9, 0.0)),
            0.0,
            &mut out,
        );
        out
    };
    for offset in [100.0f32, 1_000.0, 10_000.0] {
        let base = Vec3::splat(offset);
        let mut out = Vec::new();
        primitives::collide(
            &cube,
            &at(base),
            &cube,
            &at(base + vec3(0.0, 1.9, 0.0)),
            0.0,
            &mut out,
        );
        check_finite(&out, "far from origin");
        assert_eq!(
            out.len(),
            reference.len(),
            "manifold changed size at {offset} m"
        );
        for (a, b) in out.iter().zip(reference.iter()) {
            assert!(
                a.normal.approx_eq(b.normal, 1e-3),
                "normal drifted at {offset} m"
            );
            // A tolerance that scales with the coordinate, because f32 spacing
            // does too: at 10 km one ULP is about 1 mm.
            let tol = 1e-6 * offset.max(1.0) + 1e-4;
            assert!(
                (a.separation - b.separation).abs() < tol,
                "separation drifted by {} at {offset} m",
                (a.separation - b.separation).abs()
            );
        }
    }
}

#[test]
fn rotating_both_bodies_rotates_the_contact_with_them() {
    // Physics does not have a preferred direction. Rotating a whole
    // configuration must rotate the answer and change nothing else.
    let cube = boxx(Vec3::ONE);
    let pose_a = Isometry::new(vec3(0.0, 0.0, 0.0), Quat::from_rotation_x(0.3));
    let pose_b = Isometry::new(vec3(0.0, 1.9, 0.0), Quat::from_rotation_x(0.3));

    let mut base = Vec::new();
    primitives::collide(&cube, &pose_a, &cube, &pose_b, 0.0, &mut base);
    assert!(!base.is_empty());

    let r = Quat::from_axis_angle(vec3(0.4, 1.0, -0.2), 1.1);
    let rot = |p: &Isometry| Isometry::new(r.rotate(p.translation), r * p.rotation);
    let mut turned = Vec::new();
    primitives::collide(&cube, &rot(&pose_a), &cube, &rot(&pose_b), 0.0, &mut turned);

    assert_eq!(
        turned.len(),
        base.len(),
        "manifold size must be rotation invariant"
    );
    for (a, b) in turned.iter().zip(base.iter()) {
        assert!(
            a.normal.approx_eq(r.rotate(b.normal), 1e-3),
            "normal did not rotate with the scene"
        );
        assert!((a.separation - b.separation).abs() < 1e-4);
    }
}

#[test]
fn the_manifold_normal_always_points_from_a_to_b() {
    // The single most load-bearing convention in the whole contact path. It is
    // checked here over many random configurations rather than one, because a
    // sign error that only shows up for one shape pair in one orientation is
    // exactly the kind that survives to release.
    let mut seed = 0x1234_5678u64;
    let mut rng = move || {
        seed = seed
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        ((seed >> 33) as f32 / (u32::MAX as f32 / 2.0)) - 1.0
    };

    let shapes = [
        Shape::sphere(0.6),
        boxx(Vec3::splat(0.6)),
        boxx(vec3(1.0, 0.3, 0.7)),
    ];
    let mut checked = 0;
    for _ in 0..400 {
        let a = shapes[(rng().abs() * 2.99) as usize].clone();
        let b = shapes[(rng().abs() * 2.99) as usize].clone();
        let pose_a = Isometry::new(
            vec3(rng(), rng(), rng()) * 0.6,
            Quat::from_axis_angle(vec3(rng(), rng(), rng()), rng() * 3.0),
        );
        let pose_b = Isometry::new(
            vec3(rng(), rng(), rng()) * 0.6,
            Quat::from_axis_angle(vec3(rng(), rng(), rng()), rng() * 3.0),
        );
        let mut out = Vec::new();
        primitives::collide(&a, &pose_a, &b, &pose_b, 0.0, &mut out);
        check_finite(&out, "random pair");
        for c in &out {
            // Moving B along the normal must reduce the overlap; that is what
            // "from A to B" means operationally.
            let separation_now = (c.point_b - c.point_a).dot(c.normal);
            assert!(
                (separation_now - c.separation).abs() < 1e-3,
                "witness points disagree with the reported separation"
            );
            checked += 1;
        }
    }
    assert!(
        checked > 200,
        "expected plenty of contacts, only checked {checked}"
    );
}

#[test]
fn gjk_and_the_fast_path_agree_on_boxes() {
    // The general algorithm and the hand-written one must not disagree, or a
    // future shape added through GJK will behave differently from a box.
    let cube = Shape::Cuboid {
        half_extents: Vec3::ONE,
    };
    // Configurations where one axis is a clear winner. An exact tie — equal
    // penetration on two axes, as in a corner-on impact — has two equally
    // correct answers, and the two algorithms are entitled to disagree.
    for (dy, dx) in [(1.7f32, 0.0f32), (1.9, 0.3), (1.4, 1.8), (0.5, 1.8)] {
        let pose_b = at(vec3(dx, dy, 0.0));
        let mut fast = Vec::new();
        primitives::collide(&cube, &at(Vec3::ZERO), &cube, &pose_b, 0.0, &mut fast);
        let mut general = Vec::new();
        collide_gjk_epa(&cube, &at(Vec3::ZERO), &cube, &pose_b, &mut general);

        assert_eq!(
            fast.is_empty(),
            general.is_empty(),
            "disagree on whether ({dx},{dy}) touches"
        );
        if fast.is_empty() {
            continue;
        }
        let deepest = fast.iter().fold(0.0f32, |d, c| d.max(-c.separation));
        let general_depth = -general[0].separation;
        assert!(
            (deepest - general_depth).abs() < 5e-3,
            "depth disagreement at ({dx},{dy}): fast {deepest} vs gjk {general_depth}"
        );
        assert!(
            fast[0].normal.approx_eq(general[0].normal, 1e-2),
            "normal disagreement at ({dx},{dy}): {:?} vs {:?}",
            fast[0].normal,
            general[0].normal
        );
    }
}

#[test]
fn a_non_empty_output_buffer_is_appended_to_not_cleared() {
    // `collide` appends. A routine that clears the caller's buffer works fine
    // until the day someone batches two pairs into one buffer, and then it
    // silently loses contacts.
    let cube = Shape::Cuboid {
        half_extents: Vec3::ONE,
    };
    let mut out = Vec::new();
    primitives::collide(
        &cube,
        &at(Vec3::ZERO),
        &cube,
        &at(vec3(0.0, 1.9, 0.0)),
        0.0,
        &mut out,
    );
    let first = out.len();
    assert!(first > 0);

    // A pair that does not touch must leave the earlier contacts alone.
    primitives::collide(
        &cube,
        &at(Vec3::ZERO),
        &cube,
        &at(vec3(0.0, 50.0, 0.0)),
        0.0,
        &mut out,
    );
    assert_eq!(
        out.len(),
        first,
        "a separated pair discarded another pair's contacts"
    );
}
