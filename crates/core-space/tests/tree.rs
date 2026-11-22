use core_math::{Rng, Vec3};
use core_space::{BarnesHut, Interaction};

fn direct_forces(positions: &[Vec3], masses: &[f64], eps: f64) -> Vec<Vec3> {
    let n = positions.len();
    let mut forces = vec![Vec3::ZERO; n];
    for i in 0..n {
        for j in (i + 1)..n {
            let d = positions[j] - positions[i];
            let s = (d.length_squared() + eps * eps).sqrt();
            let f = d * (masses[i] * masses[j] / (s * s * s));
            forces[i] += f;
            forces[j] -= f;
        }
    }
    forces
}

fn tree_forces(
    tree: &BarnesHut,
    positions: &[Vec3],
    masses: &[f64],
    eps: f64,
    theta: f64,
) -> Vec<Vec3> {
    let mut forces = vec![Vec3::ZERO; positions.len()];
    tree.for_each_pair(theta, |interaction| match interaction {
        Interaction::Direct(i, j) => {
            let d = positions[j] - positions[i];
            let s = (d.length_squared() + eps * eps).sqrt();
            let f = d * (masses[i] * masses[j] / (s * s * s));
            forces[i] += f;
            forces[j] -= f;
        }
        Interaction::Cells { a, b } => {
            let ma = tree.cell_mass(a);
            let mb = tree.cell_mass(b);
            let d = tree.cell_com(b) - tree.cell_com(a);
            let s = (d.length_squared() + eps * eps).sqrt();
            let f_ab = d * (ma * mb / (s * s * s));
            for &i in tree.cell_members(a) {
                forces[i] += f_ab * (masses[i] / ma);
            }
            for &j in tree.cell_members(b) {
                forces[j] -= f_ab * (masses[j] / mb);
            }
        }
    });
    forces
}

#[test]
fn zero_theta_matches_exact_all_pairs_and_theta_half_is_accurate_and_conservative() {
    let mut rng = Rng::new(42);
    let n = 80;
    let positions: Vec<Vec3> = (0..n)
        .map(|_| {
            Vec3::new(
                -5.0 + 10.0 * rng.uniform(),
                -5.0 + 10.0 * rng.uniform(),
                -5.0 + 10.0 * rng.uniform(),
            )
        })
        .collect();
    let masses: Vec<f64> = (0..n).map(|_| 0.5 + 2.5 * rng.uniform()).collect();
    let eps = 0.05;

    let tree = BarnesHut::build(&positions, &masses);
    let exact = direct_forces(&positions, &masses, eps);
    let theta0 = tree_forces(&tree, &positions, &masses, eps, 0.0);

    let mut exact_pairs = 0;
    tree.for_each_pair(0.0, |i| {
        assert!(matches!(i, Interaction::Direct(_, _)));
        exact_pairs += 1;
    });
    assert_eq!(exact_pairs, n * (n - 1) / 2);

    for i in 0..n {
        assert!(
            (theta0[i] - exact[i]).length() < 1e-12,
            "theta=0 mismatch at {i}: {:?} vs {:?}",
            theta0[i],
            exact[i]
        );
    }

    let approx = tree_forces(&tree, &positions, &masses, eps, 0.5);
    let err_norm: f64 = approx
        .iter()
        .zip(&exact)
        .map(|(a, b)| (*a - *b).length_squared())
        .sum::<f64>()
        .sqrt();
    let exact_norm: f64 = exact.iter().map(|f| f.length_squared()).sum::<f64>().sqrt();
    assert!(
        err_norm / exact_norm < 0.01,
        "Barnes-Hut relative error too high: {}",
        err_norm / exact_norm
    );

    let net_force: Vec3 = approx.iter().copied().sum();
    assert!(
        net_force.length() < 1e-12,
        "symmetric dual-tree must conserve linear momentum exactly: {net_force:?}"
    );
}

#[test]
fn coincident_points_terminate_without_overflow_or_nans() {
    let positions = vec![Vec3::ZERO; 16];
    let masses = vec![1.0; 16];
    let tree = BarnesHut::build(&positions, &masses);
    let f = tree_forces(&tree, &positions, &masses, 0.1, 0.5);
    assert!(f.iter().all(|v| v.length() == 0.0));
}
