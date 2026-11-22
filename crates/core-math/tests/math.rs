use core_math::{vec3, Mat3, Pose, Quat, Rng, Vec3};

fn close(a: f64, b: f64, tol: f64) -> bool {
    (a - b).abs() <= tol
}

fn mat_close(a: &Mat3, b: &Mat3, tol: f64) -> bool {
    (*a - *b).frobenius_squared().sqrt() <= tol
}

fn random_mat(rng: &mut Rng) -> Mat3 {
    Mat3::from_cols(
        vec3(rng.normal(), rng.normal(), rng.normal()),
        vec3(rng.normal(), rng.normal(), rng.normal()),
        vec3(rng.normal(), rng.normal(), rng.normal()),
    )
}

#[test]
fn inverse_and_determinant() {
    let mut rng = Rng::new(1);
    for _ in 0..100 {
        let m = random_mat(&mut rng);
        let inverse = m.try_inverse().unwrap();
        assert!(mat_close(&(m * inverse), &Mat3::IDENTITY, 1e-9));
        assert!(close(
            (m * m.transpose()).determinant(),
            m.determinant().powi(2),
            1e-9
        ));
    }
    let singular = Mat3::from_cols(Vec3::X, Vec3::Y, Vec3::X + Vec3::Y);
    assert!(singular.try_inverse().is_none());
}

#[test]
fn indexing_is_row_then_column() {
    let m = Mat3::from_rows(
        vec3(1.0, 2.0, 3.0),
        vec3(4.0, 5.0, 6.0),
        vec3(7.0, 8.0, 9.0),
    );
    assert_eq!(m[(0, 1)], 2.0);
    assert_eq!(m[(1, 0)], 4.0);
    assert_eq!(m * vec3(1.0, 0.0, 0.0), vec3(1.0, 4.0, 7.0));
    assert_eq!(
        Mat3::skew(vec3(1.0, 2.0, 3.0)) * vec3(-2.0, 0.5, 4.0),
        vec3(1.0, 2.0, 3.0).cross(vec3(-2.0, 0.5, 4.0))
    );
}

#[test]
fn symmetric_eigen_reconstructs() {
    let mut rng = Rng::new(2);
    for _ in 0..200 {
        let a = random_mat(&mut rng);
        let s = a + a.transpose();
        let (values, vectors) = s.symmetric_eigen();
        assert!(mat_close(
            &(vectors.transpose() * vectors),
            &Mat3::IDENTITY,
            1e-12
        ));
        let rebuilt = vectors * Mat3::from_diagonal(values) * vectors.transpose();
        assert!(mat_close(
            &rebuilt,
            &s,
            1e-12 * s.frobenius_squared().sqrt().max(1.0)
        ));
    }
    // Repeated eigenvalues.
    let (values, _) = Mat3::scalar(2.0).symmetric_eigen();
    assert_eq!(values, Vec3::splat(2.0));
}

#[test]
fn positive_projection() {
    let s = Mat3::from_diagonal(vec3(3.0, -1.0, 2.0));
    let p = s.project_positive();
    assert!(mat_close(
        &p,
        &Mat3::from_diagonal(vec3(3.0, 0.0, 2.0)),
        1e-14
    ));
    let mut rng = Rng::new(3);
    for _ in 0..50 {
        let a = random_mat(&mut rng);
        let (values, _) = (a + a.transpose()).project_positive().symmetric_eigen();
        assert!(values.min_element() > -1e-12);
    }
}

#[test]
fn quaternions() {
    let mut rng = Rng::new(4);
    for _ in 0..100 {
        let axis = rng.direction();
        let angle = rng.range(0.0, std::f64::consts::PI);
        let q = Quat::from_axis_angle(axis, angle);
        let v = vec3(rng.normal(), rng.normal(), rng.normal());
        assert!((q.to_mat3() * v - q * v).length() < 1e-12);
        assert!(q.angle_to(Quat::from_mat3(&q.to_mat3())) < 1e-7);
        assert!((q.to_rotation_vector() - axis * angle).length() < 1e-9);
        let p = Quat::from_axis_angle(rng.direction(), 1.0);
        assert!(((p * q) * v - p * (q * v)).length() < 1e-12);
    }
    // Integration at a constant spin turns by spin × time.
    let mut q = Quat::IDENTITY;
    for _ in 0..1000 {
        q = q.integrate(vec3(0.0, 0.0, 2.0), 1e-3);
    }
    assert!(close(q.to_rotation_vector().z, 2.0, 1e-10));
}

#[test]
fn poses() {
    let pose = Pose::new(vec3(1.0, 2.0, 3.0), Quat::from_axis_angle(Vec3::Y, 0.7));
    let p = vec3(-0.3, 0.8, 2.0);
    assert!((pose.inverse_transform_point(pose.transform_point(p)) - p).length() < 1e-14);
    assert!((pose.inverse().transform_point(pose.transform_point(p)) - p).length() < 1e-14);
    let other = Pose::new(vec3(0.0, -1.0, 0.5), Quat::from_axis_angle(Vec3::X, -0.4));
    assert!(
        (pose.then(&other).transform_point(p) - pose.transform_point(other.transform_point(p)))
            .length()
            < 1e-14
    );
}

#[test]
fn rng_is_deterministic_and_uniform() {
    let (mut a, mut b) = (Rng::new(42), Rng::new(42));
    for _ in 0..100 {
        assert_eq!(a.next_u64(), b.next_u64());
    }
    let n = 100_000;
    let mean = (0..n).map(|_| a.uniform()).sum::<f64>() / n as f64;
    assert!(close(mean, 0.5, 0.01));
    let variance = (0..n).map(|_| a.normal().powi(2)).sum::<f64>() / n as f64;
    assert!(close(variance, 1.0, 0.02));
}

#[test]
fn singular_value_decomposition() {
    let mut rng = Rng::new(10);
    let mut cases: Vec<Mat3> = (0..300).map(|_| random_mat(&mut rng)).collect();
    // Rotations, a reflection, rank-deficient and repeated cases, tiny scales.
    cases.push(Quat::from_axis_angle(rng.direction(), 2.0).to_mat3());
    cases.push(Mat3::from_diagonal(vec3(1.0, 1.0, -1.0)));
    cases.push(Mat3::outer(vec3(1.0, 2.0, 3.0), vec3(-1.0, 0.5, 2.0)));
    cases.push(Mat3::from_diagonal(vec3(2.0, 2.0, 1e-9)));
    cases.push(Mat3::scalar(3.0));
    cases.push(Mat3::ZERO);
    cases.push(random_mat(&mut rng) * 1e-200);
    for f in cases {
        let (u, s, v) = f.svd();
        let scale = f.frobenius_squared().sqrt().max(f64::MIN_POSITIVE);
        assert!(
            mat_close(
                &(u * Mat3::from_diagonal(s) * v.transpose()),
                &f,
                1e-12 * scale
            ),
            "{f:?}"
        );
        assert!(
            mat_close(&(u.transpose() * u), &Mat3::IDENTITY, 1e-12)
                && (u.determinant() - 1.0).abs() < 1e-12
        );
        assert!(
            mat_close(&(v.transpose() * v), &Mat3::IDENTITY, 1e-12)
                && (v.determinant() - 1.0).abs() < 1e-12
        );
        assert!(
            s.x >= s.y - 1e-12 * scale && s.y >= s.z.abs() - 1e-12 * scale,
            "{s:?}"
        );
        assert!(s.z * f.determinant() >= 0.0);
    }
    let f =
        Quat::from_axis_angle(Vec3::Y, 0.3).to_mat3() * Mat3::from_diagonal(vec3(1.2, 0.9, 1.0));
    let (r, sym) = f.polar();
    assert!(mat_close(&(r * sym), &f, 1e-12) && mat_close(&sym, &sym.transpose(), 1e-12));
}

#[test]
fn rotation_jacobian() {
    let mut rng = Rng::new(11);
    for scale in [1e-6, 0.1, 1.0, 2.5] {
        let v = rng.direction() * scale;
        let j = Quat::left_jacobian(v);
        let h = 1e-7;
        for axis in [Vec3::X, Vec3::Y, Vec3::Z] {
            // exp(v + h a) exp(v)⁻¹ = exp(h J a) to first order.
            let delta = (Quat::from_rotation_vector(v + axis * h)
                * Quat::from_rotation_vector(v).conjugate())
            .to_rotation_vector()
                / h;
            assert!((delta - j * axis).length() < 1e-6, "scale {scale}");
        }
    }
}

#[test]
fn warm_started_eigen_and_svd_agree_with_cold() {
    let mut rng = Rng::new(12);
    for _ in 0..200 {
        let f = Quat::from_axis_angle(rng.direction(), rng.range(0.0, 3.0)).to_mat3()
            * (Mat3::IDENTITY + random_mat(&mut rng) * 0.1);
        let (_, s0, v0) = f.svd();
        // A nearby matrix, warm-started from the first's V.
        let g = f + random_mat(&mut rng) * 1e-3;
        let (u, s, v) = g.svd_from(&v0);
        let (_, s_cold, _) = g.svd();
        assert!((s - s_cold).length() < 1e-12, "{s:?} vs {s_cold:?}");
        assert!(mat_close(
            &(u * Mat3::from_diagonal(s) * v.transpose()),
            &g,
            1e-12
        ));
        assert!((v.determinant() - 1.0).abs() < 1e-12 && (u.determinant() - 1.0).abs() < 1e-12);
        let _ = s0;
        // Symmetric eigen from a poor (non-orthonormal) guess still works.
        let sym = g.transpose() * g;
        let (values, vectors) = sym.symmetric_eigen_from(&(v0 * 1.01));
        assert!(mat_close(
            &(vectors * Mat3::from_diagonal(values) * vectors.transpose()),
            &sym,
            1e-12
        ));
    }
}

#[test]
fn fast_eigen_and_svd_match_jacobi_across_degeneracies() {
    let mut rng = Rng::new(13);
    let mut syms: Vec<Mat3> = (0..200)
        .map(|_| {
            let a = random_mat(&mut rng);
            a + a.transpose()
        })
        .collect();
    syms.push(Mat3::IDENTITY);
    syms.push(Mat3::scalar(-3.5));
    syms.push(Mat3::ZERO);
    // Nearly repeated eigenvalues at an arbitrary rotation.
    for eps in [1e-14, 1e-8, 1e-5, 1e-3] {
        let r = Quat::from_axis_angle(rng.direction(), 1.2).to_mat3();
        let d = Mat3::from_diagonal(vec3(2.0 + eps, 2.0, 1.0));
        syms.push(r * d * r.transpose());
    }
    for s in syms {
        let (values, vectors) = s.symmetric_eigen_fast();
        let scale = s.frobenius_squared().sqrt().max(1.0);
        assert!(mat_close(
            &(vectors.transpose() * vectors),
            &Mat3::IDENTITY,
            1e-11
        ));
        let rebuilt = vectors * Mat3::from_diagonal(values) * vectors.transpose();
        assert!(mat_close(&rebuilt, &s, 1e-11 * scale), "{s:?}");
    }
    let mut mats: Vec<Mat3> = (0..200).map(|_| random_mat(&mut rng)).collect();
    mats.push(Mat3::from_diagonal(vec3(2.0, 2.0, -1.0)));
    mats.push(Mat3::from_diagonal(vec3(1.5, 1.5, 1.5)));
    mats.push(Mat3::ZERO);
    for f in mats {
        let (u, s, v) = f.svd_fast(&Mat3::IDENTITY);
        let scale = f.frobenius_squared().sqrt().max(f64::MIN_POSITIVE);
        assert!(mat_close(
            &(u * Mat3::from_diagonal(s) * v.transpose()),
            &f,
            1e-11 * scale
        ));
        assert!((u.determinant() - 1.0).abs() < 1e-11);
        assert!((v.determinant() - 1.0).abs() < 1e-11);
        assert!(s.z * f.determinant() >= 0.0);
    }
}
