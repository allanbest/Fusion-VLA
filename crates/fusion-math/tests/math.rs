use fusion_math::*;

const TOL: f32 = 1e-5;

#[test]
fn vector_algebra_identities() {
    let a = vec3(1.0, 2.0, 3.0);
    let b = vec3(-4.0, 0.5, 2.0);
    assert!(
        (a.cross(b).dot(a)).abs() < TOL,
        "a x b must be perpendicular to a"
    );
    assert!(
        (a.cross(b).dot(b)).abs() < TOL,
        "a x b must be perpendicular to b"
    );
    assert!(
        (a.cross(b) + b.cross(a)).length() < TOL,
        "cross product is antisymmetric"
    );
    assert!((a.normalize().length() - 1.0).abs() < TOL);
    assert_eq!(Vec3::ZERO.normalize_or_zero(), Vec3::ZERO);
    assert!(a.any_orthonormal().dot(a).abs() < TOL);
}

#[test]
fn quaternion_rotation_matches_expectation() {
    let q = Quat::from_rotation_z(std::f32::consts::FRAC_PI_2);
    let r = q.rotate(Vec3::X);
    assert!(
        r.approx_eq(Vec3::Y, TOL),
        "90deg about +Z takes +X to +Y, got {r:?}"
    );

    // Rotating and un-rotating is the identity.
    let v = vec3(0.3, -1.7, 4.0);
    let q = Quat::from_axis_angle(vec3(1.0, 2.0, -0.5), 0.9);
    assert!(q.rotate_inverse(q.rotate(v)).approx_eq(v, TOL));

    // Composition order: (q1 * q2) applies q2 first.
    let q1 = Quat::from_rotation_x(0.4);
    let q2 = Quat::from_rotation_y(-1.1);
    assert!((q1 * q2).rotate(v).approx_eq(q1.rotate(q2.rotate(v)), TOL));
}

#[test]
fn quaternion_matches_its_rotation_matrix() {
    let q = Quat::from_axis_angle(vec3(0.3, 1.0, -0.2), 2.1);
    let m = Mat3::from_quat(q);
    let v = vec3(1.0, -2.0, 0.7);
    assert!((m * v).approx_eq(q.rotate(v), TOL));
    assert!(
        (m.determinant() - 1.0).abs() < TOL,
        "rotation matrices have unit determinant"
    );
}

#[test]
fn integrating_angular_velocity_converges_on_the_exact_rotation() {
    // Spin at 1 rad/s about +Y for 1 second in small steps; expect 1 radian.
    let w = Vec3::Y;
    let mut q = Quat::IDENTITY;
    let steps = 2000;
    let dt = 1.0 / steps as f32;
    for _ in 0..steps {
        q = q.integrate(w, dt);
    }
    assert!(q.approx_eq(Quat::from_rotation_y(1.0), 1e-4), "got {q:?}");
}

#[test]
fn matrix_inverse_round_trips() {
    let m = Mat3::from_cols(
        vec3(2.0, 0.1, 0.0),
        vec3(-0.3, 1.5, 0.2),
        vec3(0.0, 0.4, 3.0),
    );
    let i = m * m.inverse();
    for k in 0..3 {
        assert!(i.col(k).approx_eq(Mat3::IDENTITY.col(k), 1e-4));
    }
    // Singular matrices report zero rather than NaN — the "infinite mass"
    // convention the solver relies on.
    let singular = Mat3::from_cols(Vec3::X, Vec3::X, Vec3::X);
    assert_eq!(singular.inverse(), Mat3::ZERO);
}

#[test]
fn world_space_inertia_transform() {
    // A diagonal body inertia rotated into world space stays symmetric and
    // preserves its trace.
    let d = vec3(1.0, 2.0, 5.0);
    let r = Mat3::from_quat(Quat::from_axis_angle(vec3(1.0, 1.0, 0.0), 0.7));
    let w = r.mul_diagonal_mul_transpose(d);
    let trace = w.x_axis.x + w.y_axis.y + w.z_axis.z;
    assert!((trace - (d.x + d.y + d.z)).abs() < 1e-4);
    for i in 0..3 {
        for j in 0..3 {
            assert!(
                (w.col(i)[j] - w.col(j)[i]).abs() < 1e-5,
                "must stay symmetric"
            );
        }
    }
}

#[test]
fn isometry_round_trips_and_composes() {
    let a = Isometry::new(vec3(1.0, 2.0, 3.0), Quat::from_rotation_y(0.6));
    let b = Isometry::new(vec3(-0.5, 0.0, 2.0), Quat::from_rotation_x(-1.2));
    let p = vec3(0.25, -3.0, 1.0);
    assert!(a
        .inverse_transform_point(a.transform_point(p))
        .approx_eq(p, TOL));
    assert!(a
        .mul(&b)
        .transform_point(p)
        .approx_eq(a.transform_point(b.transform_point(p)), TOL));
    assert!(a.inverse().mul(&a).transform_point(p).approx_eq(p, TOL));
}

#[test]
fn aabb_overlap_and_merge() {
    let a = Aabb::new(vec3(0.0, 0.0, 0.0), vec3(1.0, 1.0, 1.0));
    let b = Aabb::new(vec3(0.9, 0.9, 0.9), vec3(2.0, 2.0, 2.0));
    let c = Aabb::new(vec3(1.1, 0.0, 0.0), vec3(2.0, 1.0, 1.0));
    assert!(a.overlaps(&b));
    assert!(!a.overlaps(&c));
    assert!(
        a.expanded(0.2).overlaps(&c),
        "margin should create the overlap"
    );
    let m = a.merge(&b);
    assert!(m.contains_point(vec3(2.0, 2.0, 2.0)) && m.contains_point(Vec3::ZERO));
    assert!(Aabb::EMPTY.merge(&a) == a, "EMPTY is the merge identity");
}

#[test]
fn f64_variant_exists_and_agrees_with_f32() {
    // PLAN.md §9: precision is a per-solver choice, so both widths ship.
    let q32 = Quat::from_rotation_z(0.5).rotate(Vec3::X);
    let q64 = DQuat::from_rotation_z(0.5).rotate(DVec3::X);
    assert!((q32.x as f64 - q64.x).abs() < 1e-6);
    assert!((q32.y as f64 - q64.y).abs() < 1e-6);
}

/// The decomposition is exact on the cases whose answer is known by hand.
#[test]
fn svd_reproduces_matrices_whose_answer_is_known() {
    let (u, s, v) = Mat3::IDENTITY.svd();
    assert!((s - vec3(1.0, 1.0, 1.0)).length() < TOL);
    assert!((u * v.transpose() - Mat3::IDENTITY).determinant().abs() < 1.0);

    // A pure rotation stretches nothing, and is its own closest rotation.
    let q = Quat::from_rotation_z(0.7) * Quat::from_rotation_x(-0.4);
    let r = Mat3::from_quat(q);
    let (_, s, _) = r.svd();
    assert!(
        (s - vec3(1.0, 1.0, 1.0)).length() < TOL,
        "rotation stretched: {s:?}"
    );
    let (rotation, stretch) = r.polar();
    for i in 0..3 {
        assert!((rotation.col(i) - r.col(i)).length() < TOL);
        assert!((stretch.col(i) - Mat3::IDENTITY.col(i)).length() < TOL);
    }

    // A pure stretch, whose singular values are its diagonal, descending.
    let (_, s, _) = Mat3::from_diagonal(vec3(2.0, 5.0, 0.5)).svd();
    assert!((s - vec3(5.0, 2.0, 0.5)).length() < TOL, "unsorted: {s:?}");
}

/// An inverted deformation gradient keeps its inversion in the singular value
/// rather than losing it to a reflection.
#[test]
fn svd_puts_an_inversion_where_a_material_model_can_see_it() {
    let flipped = Mat3::from_diagonal(vec3(1.0, 1.0, -0.5));
    let (u, s, v) = flipped.svd();
    assert!(s.z < 0.0, "an inverted point reported no inversion: {s:?}");
    assert!((u.determinant() - 1.0).abs() < TOL, "u is not a rotation");
    assert!((v.determinant() - 1.0).abs() < TOL, "v is not a rotation");
    let rebuilt = u * Mat3::from_diagonal(s) * v.transpose();
    for i in 0..3 {
        assert!((rebuilt.col(i) - flipped.col(i)).length() < TOL);
    }
}

/// And on a hundred arbitrary matrices, which is where a decomposition written
/// against three worked examples usually turns out to be wrong.
#[test]
fn svd_rebuilds_whatever_it_was_given() {
    let mut seed = 0x5eed_1234u64;
    let mut next = || {
        seed = seed
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        ((seed >> 33) as f32 / (1u32 << 31) as f32) * 4.0 - 2.0
    };
    for case in 0..100 {
        let m = Mat3::from_cols(
            vec3(next(), next(), next()),
            vec3(next(), next(), next()),
            vec3(next(), next(), next()),
        );
        let (u, s, v) = m.svd();
        assert!(
            (u.determinant() - 1.0).abs() < 1e-4 && (v.determinant() - 1.0).abs() < 1e-4,
            "case {case}: not rotations ({}, {})",
            u.determinant(),
            v.determinant()
        );
        assert!(
            s.x.abs() >= s.y.abs() - TOL && s.y.abs() >= s.z.abs() - TOL,
            "case {case}: singular values out of order: {s:?}"
        );
        let rebuilt = u * Mat3::from_diagonal(s) * v.transpose();
        for i in 0..3 {
            assert!(
                (rebuilt.col(i) - m.col(i)).length() < 1e-4,
                "case {case}: column {i} rebuilt as {:?}, not {:?}",
                rebuilt.col(i),
                m.col(i)
            );
        }

        // The polar factors agree with the decomposition they came from.
        let (r, stretch) = m.polar();
        let product = r * stretch;
        for i in 0..3 {
            assert!(
                (product.col(i) - m.col(i)).length() < 1e-4,
                "case {case}: r * s is not m"
            );
        }
    }
}

/// A collapsed point still yields a rotation, which is what stops a stress
/// evaluation from producing NaN and taking the whole scene with it.
#[test]
fn svd_survives_a_degenerate_deformation() {
    for m in [
        Mat3::ZERO,
        Mat3::from_diagonal(vec3(1.0, 1.0, 0.0)),
        Mat3::from_diagonal(vec3(1.0, 0.0, 0.0)),
        Mat3::from_cols(Vec3::X, Vec3::X, Vec3::X),
    ] {
        let (u, s, v) = m.svd();
        assert!(
            (u.determinant() - 1.0).abs() < 1e-4,
            "u is not a rotation for {m:?}: det {}",
            u.determinant()
        );
        assert!(
            (v.determinant() - 1.0).abs() < 1e-4,
            "v is not a rotation for {m:?}"
        );
        assert!(s.x.is_finite() && s.y.is_finite() && s.z.is_finite());
        let (r, _) = m.polar();
        assert!(
            (r.determinant() - 1.0).abs() < 1e-4,
            "polar gave no rotation for {m:?}"
        );
    }
}
