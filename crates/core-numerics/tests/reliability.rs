use core_numerics::{conjugate_gradient, CgSettings, CgWorkspace, Identity, LinearOperator};
use core_runtime::Pool;
struct Chain {
    nodes: usize,
    mass: f64,
}
impl LinearOperator for Chain {
    fn dimension(&self) -> usize {
        self.nodes * 3
    }
    fn add_part(&self, x: &[f64], offset: usize, out: &mut [f64]) {
        for (k, y) in out.iter_mut().enumerate() {
            let i = offset + k;
            *y += self.mass * x[i];
            if i >= 3 {
                *y += x[i] - x[i - 3];
            }
            if i + 3 < x.len() {
                *y += x[i] - x[i + 3];
            }
        }
    }
}
#[test]
fn true_equation_residual_controls_acceptance_on_ill_conditioned_systems() {
    let pool = Pool::new(1);
    for mass in [1e-3, 1e-8, 1e-12] {
        let a = Chain { nodes: 80, mass };
        let solution: Vec<_> = (0..a.dimension())
            .map(|i| 0.3 + (i as f64 * 0.1).sin())
            .collect();
        let mut b = vec![0.0; solution.len()];
        a.apply(&pool, &solution, &mut b);
        for tolerance in [1e-8, 1e-13] {
            let mut x = vec![0.0; b.len()];
            let report = conjugate_gradient(
                &pool,
                &a,
                &Identity,
                &b,
                &mut x,
                CgSettings {
                    tolerance,
                    max_iterations: 500,
                },
                &mut CgWorkspace::default(),
            );
            let mut ax = vec![0.0; x.len()];
            a.apply(&pool, &x, &mut ax);
            let truth = (b
                .iter()
                .zip(&ax)
                .map(|(&b, &a)| (b - a).powi(2))
                .sum::<f64>()
                / b.iter().map(|v| v * v).sum::<f64>())
            .sqrt();
            assert!(
                (report.residual - truth).abs() < 1e-12 * truth.max(1e-18),
                "{report:?}, true={truth}"
            );
            assert_eq!(report.converged, truth <= tolerance, "{report:?}");
            assert!(report.residual_checks > 0);
            assert!(!report.not_finite);
        }
    }
}
#[test]
fn verified_cg_remains_bitwise_deterministic_across_vector_chunks() {
    let a = Chain {
        nodes: 1500,
        mass: 0.1,
    };
    let b: Vec<_> = (0..a.dimension())
        .map(|i| (i as f64 * 0.07).cos())
        .collect();
    let solve = |threads| {
        let pool = Pool::new(threads);
        let mut x = vec![0.0; b.len()];
        let r = conjugate_gradient(
            &pool,
            &a,
            &Identity,
            &b,
            &mut x,
            CgSettings {
                tolerance: 1e-9,
                max_iterations: 500,
            },
            &mut CgWorkspace::default(),
        );
        assert!(r.converged, "{r:?}");
        (x, r.residual.to_bits(), r.iterations, r.residual_restarts)
    };
    assert_eq!(solve(1), solve(4));
}
