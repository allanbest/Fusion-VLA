use core_math::{Mat3, Quat, Rng, Vec3};
use core_numerics::check;
use phys_mechanics::law::NeoHookean;

fn flat(m: &Mat3) -> [f64; 9] {
    std::array::from_fn(|k| m[(k / 3, k % 3)])
}

fn unflat(x: &[f64]) -> Mat3 {
    let mut m = Mat3::ZERO;
    for k in 0..9 {
        m[(k / 3, k % 3)] = x[k];
    }
    m
}

fn deformations(rng: &mut Rng) -> Vec<Mat3> {
    let mut out = vec![
        Mat3::IDENTITY,
        Quat::from_axis_angle(rng.direction(), 1.0).to_mat3(),
        Mat3::from_diagonal(Vec3::new(1.3, 1.3, 0.8)),
    ];
    for _ in 0..40 {
        let mut f = Mat3::IDENTITY;
        for i in 0..3 {
            for j in 0..3 {
                f[(i, j)] += 0.3 * rng.normal();
            }
        }
        if f.determinant() > 0.1 {
            out.push(Quat::from_axis_angle(rng.direction(), rng.range(0.0, 3.0)).to_mat3() * f);
        }
    }
    out
}

#[test]
fn stress_is_the_energy_gradient() {
    let law = NeoHookean::from_youngs(1e6, 0.4);
    let mut rng = Rng::new(1);
    for f in deformations(&mut rng) {
        let p = law.stress(&f);
        if p.frobenius_squared().sqrt() < 1e-6 * law.mu {
            // Stress-free (at rest, or turned): no gradient to compare to.
            continue;
        }
        let error = check::gradient_error(|x| law.energy(&unflat(x)), &flat(&f), &flat(&p), 1e-6);
        assert!(error < 1e-6, "{error} at {f:?}");
    }
    // At rest: no stress, no energy.
    assert!(law.stress(&Mat3::IDENTITY).frobenius_squared() < 1e-20);
    assert_eq!(law.energy(&Mat3::IDENTITY), 0.0);
    assert_eq!(
        law.energy(&Mat3::from_diagonal(Vec3::new(1.0, 1.0, -1.0))),
        f64::INFINITY
    );
}

#[test]
fn modes_are_the_exact_second_derivative() {
    let law = NeoHookean::from_youngs(1e6, 0.45);
    let mut rng = Rng::new(2);
    for f in deformations(&mut rng) {
        let modes = law.modes(&f);
        // Orthonormal eigenmatrices.
        for a in 0..9 {
            for b in 0..9 {
                let dot = modes.matrices[a].double_dot(&modes.matrices[b]);
                assert!((dot - if a == b { 1.0 } else { 0.0 }).abs() < 1e-10);
            }
        }
        let mut hessian = vec![0.0; 81];
        for k in 0..9 {
            let q = flat(&modes.matrices[k]);
            for i in 0..9 {
                for j in 0..9 {
                    hessian[9 * i + j] += modes.values[k] * q[i] * q[j];
                }
            }
        }
        let error = check::hessian_error(
            |x, g| g.copy_from_slice(&flat(&law.stress(&unflat(x)))),
            &flat(&f),
            &hessian,
            1e-6,
        );
        assert!(error < 1e-6, "{error} at {f:?}");
    }
}

#[test]
fn small_strain_is_hookes_law() {
    // Uniaxial stress σ = E ε for a bar free to contract sideways.
    let (e, nu) = (2e5, 0.3);
    let law = NeoHookean::from_youngs(e, nu);
    let strain = 1e-6;
    let f = Mat3::from_diagonal(Vec3::new(
        1.0 + strain,
        1.0 - nu * strain,
        1.0 - nu * strain,
    ));
    let p = law.stress(&f);
    assert!((p[(0, 0)] / (e * strain) - 1.0).abs() < 1e-4);
    assert!(p[(1, 1)].abs() < 1e-4 * e * strain);
}

#[test]
fn analytic_blocks_with_negative_mode_correction_match_psd_projection() {
    let law = NeoHookean::from_youngs(1e6, 0.45);
    let mut rng = Rng::new(123);
    for f in deformations(&mut rng) {
        let modes = law.modes(&f);
        let j = f.determinant();
        let inverse_t = f.adjugate().transpose() * (1.0 / j);
        for _ in 0..8 {
            let ga = rng.direction();
            let gb = rng.direction();
            let va = inverse_t * ga;
            let vb = inverse_t * gb;
            let mut reference = Mat3::ZERO;
            let mut optimized = Mat3::scalar(law.mu * ga.dot(gb))
                + Mat3::outer(vb, va) * (law.mu - law.lambda * j.ln())
                + Mat3::outer(va, vb) * law.lambda;
            for (value, q) in modes.values.iter().zip(&modes.matrices) {
                if *value > 0.0 {
                    reference += Mat3::outer(*q * ga, *q * gb) * *value;
                } else {
                    optimized -= Mat3::outer(*q * ga, *q * gb) * *value;
                }
            }
            let error = (optimized - reference).frobenius_squared().sqrt() / law.mu;
            assert!(error < 1e-9, "{error} at {f:?}");
        }
    }
}

#[test]
fn tiny_shear_energy_is_positive_and_matches_the_exact_answer_after_rotation() {
    // det(F)=1 for simple shear, so psi=mu*gamma^2/2 exactly, at any rotation.
    // Directly subtracting tr(C)-3 and 2*ln(J) loses this energy near rest.
    for youngs in [1e5, 2e11] {
        let law = NeoHookean::from_youngs(youngs, 0.3);
        for shear in [1e-2, 1e-4, 1e-6, 1e-8, 1e-10] {
            let f = Mat3::IDENTITY + Mat3::outer(Vec3::X, Vec3::Y) * shear;
            let reference = 0.5 * law.mu * shear * shear;
            for angle in [0.0, 0.7, 2.4] {
                let rotation =
                    Quat::from_axis_angle(Vec3::new(1.0, 2.0, 3.0).normalize(), angle).to_mat3();
                let energy = law.energy(&(rotation * f));
                assert!(energy >= 0.0 && energy.is_finite());
                let tolerance = 64.0 * f64::EPSILON / shear + 1e-10;
                assert!(
                    (energy / reference - 1.0).abs() < tolerance,
                    "E={youngs}, shear={shear}, angle={angle}: {energy} vs {reference}"
                );
            }
        }
    }
}

#[test]
fn sign_gated_negative_modes_match_full_modes_projection() {
    let law = NeoHookean::from_youngs(1e6, 0.45);
    let mut rng = Rng::new(456);
    let mut warm = phys_mechanics::law::Warm::default();
    let mut extra = deformations(&mut rng);
    // Include large expansion (mu - lambda*ln(J) < 0) and extreme compression.
    extra.push(Mat3::scalar(1.6));
    extra.push(Mat3::from_diagonal(Vec3::new(0.4, 2.2, 1.5)));
    for f in extra {
        let modes = law.modes(&f);
        let j = f.determinant();
        let inverse_t = f.adjugate().transpose() * (1.0 / j);
        let ga = rng.direction();
        let gb = rng.direction();
        let va = inverse_t * ga;
        let vb = inverse_t * gb;
        let base = Mat3::scalar(law.mu * ga.dot(gb))
            + Mat3::outer(vb, va) * (law.mu - law.lambda * j.ln())
            + Mat3::outer(va, vb) * law.lambda;
        let mut ref_block = base;
        for (value, q) in modes.values.iter().zip(&modes.matrices) {
            if *value < 0.0 {
                ref_block -= Mat3::outer(*q * ga, *q * gb) * *value;
            }
        }
        let mut fast_block = base;
        law.for_each_negative_mode(&f, &mut warm, |value, q| {
            fast_block -= Mat3::outer(q * ga, q * gb) * value;
        });
        let error = (fast_block - ref_block).frobenius_squared().sqrt() / law.mu;
        assert!(error < 1e-10, "{error} at {f:?}");
    }
}
