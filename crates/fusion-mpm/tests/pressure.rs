use fusion_mpm::linear::{IncompleteCholesky, Pcg, SolveFailure, SolveSettings, SparseSymmetric};
use fusion_mpm::pressure::{PressureFace, PressureProjection, ProjectionFailure};

#[test]
fn pcg_solves_a_known_spd_system_and_checks_true_residual() {
    let mut x = [7.0, -3.0];
    let r = Pcg::default()
        .solve(
            &[1.0, 2.0],
            &[4.0, 3.0],
            &mut x,
            SolveSettings::default(),
            |x, y| {
                y[0] = 4.0 * x[0] + x[1];
                y[1] = x[0] + 3.0 * x[1];
            },
        )
        .unwrap();
    assert!((x[0] - 1.0 / 11.0).abs() < 1e-10);
    assert!((x[1] - 7.0 / 11.0).abs() < 1e-10);
    assert!(r.residual < 1e-10);
}

#[test]
fn pcg_failure_does_not_overwrite_the_warm_start() {
    let initial = [2.0, 3.0];
    let mut x = initial;
    let result = Pcg::default().solve(
        &[1.0, 2.0],
        &[1.0, 1.0],
        &mut x,
        SolveSettings {
            max_iterations: 0,
            ..Default::default()
        },
        |x, y| y.copy_from_slice(x),
    );
    assert!(matches!(result, Err(SolveFailure::IterationLimit { .. })));
    assert_eq!(x, initial);
    let result = Pcg::default().solve(
        &[1.0, 2.0],
        &[1.0, 1.0],
        &mut x,
        SolveSettings::default(),
        |x, y| {
            y[0] = -x[0];
            y[1] = -x[1];
        },
    );
    assert_eq!(result, Err(SolveFailure::NonPositiveCurvature));
    assert_eq!(x, initial);
}

#[test]
fn pcg_rejects_nonfinite_operators() {
    let mut x = [0.0];
    let r = Pcg::default().solve(&[1.0], &[1.0], &mut x, SolveSettings::default(), |_, y| {
        y[0] = f64::NAN
    });
    assert_eq!(r, Err(SolveFailure::NonFinite));
    assert_eq!(x, [0.0]);
}

#[test]
fn vertical_water_column_recovers_hydrostatic_pressure() {
    // Unit-area column, cells bottom to top; floor omitted (zero flux),
    // atmospheric pressure at the top face half a cell above the last centre.
    for n in [4, 16, 64] {
        let h = 1.0 / n as f64;
        let dt = 0.01;
        let rho = 1000.0;
        let gravity = -9.81;
        let mut faces: Vec<_> = (0..n)
            .map(|i| PressureFace {
                minus: i,
                plus: (i + 1 < n).then_some(i + 1),
                mobility: dt / (rho * if i + 1 == n { h * 0.5 } else { h }),
                flux: gravity * dt,
            })
            .collect();
        let r = PressureProjection::default()
            .project(
                &vec![h; n],
                &mut faces,
                SolveSettings {
                    relative_tolerance: 1e-11,
                    absolute_tolerance: 1e-13,
                    max_iterations: 4000,
                },
            )
            .unwrap();
        assert!(r.divergence_after < 1e-8, "{n}: {}", r.divergence_after);
        for (i, &p) in r.pressure.iter().enumerate() {
            let expected = rho * -gravity * (1.0 - (i as f64 + 0.5) * h);
            assert!((p - expected).abs() < 1e-5, "{n}: {i}: {p} != {expected}");
        }
        assert!(faces.iter().all(|f| f.flux.abs() < 1e-9));
    }
}

#[test]
fn closed_components_have_independent_gauges_and_preserve_circulation() {
    let mut faces = vec![
        PressureFace {
            minus: 0,
            plus: Some(1),
            mobility: 1.0,
            flux: 2.0,
        },
        PressureFace {
            minus: 1,
            plus: Some(2),
            mobility: 2.0,
            flux: -1.0,
        },
        PressureFace {
            minus: 2,
            plus: Some(0),
            mobility: 0.5,
            flux: 0.4,
        },
        // A disconnected closed pair, plus an isolated cell at index 5.
        PressureFace {
            minus: 3,
            plus: Some(4),
            mobility: 1.0,
            flux: 7.0,
        },
    ];
    let energy = |f: &[PressureFace]| f.iter().map(|f| f.flux * f.flux / f.mobility).sum::<f64>();
    let before = energy(&faces);
    let circulation_before: f64 = faces[..3].iter().map(|f| f.flux / f.mobility).sum();
    let mut solver = PressureProjection::default();
    let r = solver
        .project(&[1.0; 6], &mut faces, SolveSettings::default())
        .unwrap();
    assert!(r.divergence_after < 1e-9);
    assert!(energy(&faces) <= before);
    let circulation_after: f64 = faces[..3].iter().map(|f| f.flux / f.mobility).sum();
    assert!((circulation_after - circulation_before).abs() < 1e-10);
    let once: Vec<_> = faces.iter().map(|f| f.flux).collect();
    solver
        .project(&[1.0; 6], &mut faces, SolveSettings::default())
        .unwrap();
    for (f, q) in faces.iter().zip(once) {
        assert!((f.flux - q).abs() < 1e-10);
    }
}

#[test]
fn invalid_or_unconverged_projection_is_atomic() {
    let original = PressureFace {
        minus: 0,
        plus: None,
        mobility: 1.0,
        flux: 3.0,
    };
    let mut faces = [original];
    let r = PressureProjection::default().project(
        &[1.0],
        &mut faces,
        SolveSettings {
            max_iterations: 0,
            ..Default::default()
        },
    );
    assert!(matches!(
        r,
        Err(ProjectionFailure::Linear(
            SolveFailure::IterationLimit { .. }
        ))
    ));
    assert_eq!(faces[0].flux, original.flux);
    faces[0].plus = Some(99);
    assert!(matches!(
        PressureProjection::default().project(&[1.0], &mut faces, SolveSettings::default()),
        Err(ProjectionFailure::InvalidInput)
    ));
    assert_eq!(faces[0].flux, original.flux);
}

/// A 7-point Laplacian on an `n³` grid, plus `shift` on the diagonal.
fn laplacian(n: usize, shift: f64) -> SparseSymmetric {
    let mut m = SparseSymmetric::default();
    m.starts.push(0);
    let at = |x: usize, y: usize, z: usize| (z * n + y) * n + x;
    for z in 0..n {
        for y in 0..n {
            for x in 0..n {
                let mut row: Vec<(u32, f64)> = vec![(at(x, y, z) as u32, 6.0 + shift)];
                for (dx, dy, dz) in [
                    (-1i64, 0i64, 0i64),
                    (1, 0, 0),
                    (0, -1, 0),
                    (0, 1, 0),
                    (0, 0, -1),
                    (0, 0, 1),
                ] {
                    let (a, b, c) = (x as i64 + dx, y as i64 + dy, z as i64 + dz);
                    if (0..n as i64).contains(&a)
                        && (0..n as i64).contains(&b)
                        && (0..n as i64).contains(&c)
                    {
                        row.push((at(a as usize, b as usize, c as usize) as u32, -1.0));
                    }
                }
                row.sort_by_key(|e| e.0);
                for (c, v) in row {
                    m.columns.push(c);
                    m.values.push(v);
                }
                m.starts.push(m.columns.len());
            }
        }
    }
    m
}

/// A tridiagonal matrix has no fill to drop, so its incomplete factor is its
/// exact one.
#[test]
fn incomplete_cholesky_is_exact_where_there_is_no_fill() {
    let mut m = SparseSymmetric::default();
    m.starts.push(0);
    let n = 12;
    for i in 0..n {
        if i > 0 {
            m.columns.push(i as u32 - 1);
            m.values.push(-1.0);
        }
        m.columns.push(i as u32);
        m.values.push(2.5);
        if i + 1 < n {
            m.columns.push(i as u32 + 1);
            m.values.push(-1.0);
        }
        m.starts.push(m.columns.len());
    }
    let mut factor = IncompleteCholesky::default();
    factor.factor(&m);
    let x: Vec<f64> = (0..n).map(|i| (i as f64 * 0.7).sin()).collect();
    let mut b = vec![0.0; n];
    m.apply(&x, &mut b);
    let mut back = vec![0.0; n];
    factor.apply(&b, &mut back);
    for (a, b) in x.iter().zip(&back) {
        assert!((a - b).abs() < 1e-12, "{a} vs {b}");
    }
}

/// On a Poisson matrix, the incomplete factor reaches the same answer as
/// Jacobi in far fewer iterations — the reason the projection uses it.
#[test]
fn incomplete_cholesky_preconditions_a_poisson_solve_better_than_jacobi() {
    let a = laplacian(12, 1e-3);
    let n = a.len();
    let rhs: Vec<f64> = (0..n)
        .map(|i| ((i * 37 % 101) as f64 / 101.0) - 0.5)
        .collect();
    let diagonal: Vec<f64> = (0..n)
        .map(|r| {
            (a.starts[r]..a.starts[r + 1])
                .find(|&e| a.columns[e] as usize == r)
                .map(|e| a.values[e])
                .unwrap()
        })
        .collect();
    let settings = SolveSettings {
        relative_tolerance: 1e-8,
        absolute_tolerance: 0.0,
        max_iterations: 2000,
    };
    let mut pcg = Pcg::default();
    let mut jacobi = vec![0.0; n];
    let plain = pcg
        .solve(&rhs, &diagonal, &mut jacobi, settings, |x, y| a.apply(x, y))
        .unwrap();
    let mut factor = IncompleteCholesky::default();
    factor.factor(&a);
    let mut ic = vec![0.0; n];
    let better = pcg
        .solve_preconditioned_within_budget(
            &rhs,
            &mut ic,
            settings,
            |x, y| a.apply(x, y),
            |r, z| factor.apply(r, z),
        )
        .unwrap();
    assert!(better.iterations < settings.max_iterations);
    assert!(
        better.iterations * 2 < plain.iterations,
        "IC(0) {} iterations, Jacobi {}",
        better.iterations,
        plain.iterations
    );
    let worst = jacobi
        .iter()
        .zip(&ic)
        .map(|(a, b)| (a - b).abs())
        .fold(0.0, f64::max);
    let scale = jacobi.iter().fold(0.0f64, |m, v| m.max(v.abs()));
    assert!(
        worst < 1e-5 * scale,
        "the same answer: {worst} against {scale}"
    );
}
