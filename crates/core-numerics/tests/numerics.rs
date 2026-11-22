use core_math::{Mat3, Rng, Vec3};
use core_numerics::{
    check, conjugate_gradient, dense, vector, BlockJacobi, BlockMatrix, BlockOperator, CgSettings,
    CgWorkspace, Dual, LinearOperator, Newton, NewtonSettings, Objective, Sum, Triplets,
};
use core_runtime::Pool;

/// A 3D lattice of `side³` points joined to their neighbours by springs of
/// stiffness `k`, each point also held by a weak spring to its rest place —
/// the shape of an elastic body's Hessian (block Laplacian plus mass).
fn lattice(side: usize, k: f64, hold: f64) -> Triplets {
    let n = side * side * side;
    let id = |i: usize, j: usize, l: usize| i + side * (j + side * l);
    let mut t = Triplets::new(n);
    let mut rng = Rng::new(9);
    for l in 0..side {
        for j in 0..side {
            for i in 0..side {
                let a = id(i, j, l);
                t.add(a, a, Mat3::scalar(hold));
                for (di, dj, dl) in [(1, 0, 0), (0, 1, 0), (0, 0, 1)] {
                    let (ii, jj, ll) = (i + di, j + dj, l + dl);
                    if ii < side && jj < side && ll < side {
                        let b = id(ii, jj, ll);
                        let d = Vec3::new(di as f64, dj as f64, dl as f64) + rng.direction() * 0.3;
                        let block =
                            Mat3::outer(d, d) * (k / d.length_squared()) + Mat3::scalar(0.1 * k);
                        t.add(a, a, block);
                        t.add(b, b, block);
                        t.add_symmetric(a, b, -block);
                    }
                }
            }
        }
    }
    t
}

#[test]
fn triplets_sum_duplicates_and_stay_symmetric() {
    let mut t = Triplets::new(3);
    t.add(0, 0, Mat3::scalar(1.0));
    t.add(0, 0, Mat3::scalar(2.0));
    t.add_symmetric(0, 2, Mat3::outer(Vec3::X, Vec3::Y));
    let m = t.build();
    assert_eq!(m.get(0, 0), Mat3::scalar(3.0));
    assert_eq!(m.get(2, 0), Mat3::outer(Vec3::Y, Vec3::X));
    assert_eq!(m.get(1, 1), Mat3::ZERO);
    assert!(lattice(5, 10.0, 1.0).build().is_symmetric(1e-14));
}

/// Whatever the matrix's size (a small one is summed through a table, a
/// large one row by row) and however the entries arrive (one at a time or
/// as whole lists), every block is the sum of its entries in the order they
/// were added, to the bit, and each row's columns ascend.
#[test]
fn triplets_sum_in_insertion_order_whatever_the_route() {
    for (n, entries) in [(7usize, 400usize), (1100, 6000)] {
        let mut rng = Rng::new(n as u64);
        // Few distinct places, so most entries are duplicates.
        let places: Vec<(usize, usize)> = (0..entries / 6)
            .map(|_| {
                (
                    (rng.uniform() * n as f64) as usize % n,
                    (rng.uniform() * n as f64) as usize % n,
                )
            })
            .collect();
        let list: Vec<(usize, usize, Mat3)> = (0..entries)
            .map(|k| {
                let (r, c) = places[(rng.uniform() * places.len() as f64) as usize % places.len()];
                // Sizes spread over many orders, so the order of a sum shows.
                let scale = 10f64.powi(k as i32 % 13 - 6);
                (r, c, Mat3::outer(rng.direction(), rng.direction()) * scale)
            })
            .collect();
        let mut expected: std::collections::BTreeMap<(usize, usize), Mat3> = Default::default();
        for &(r, c, block) in &list {
            expected
                .entry((r, c))
                .and_modify(|sum| *sum += block)
                .or_insert(block);
        }
        // One at a time; and in three parts, the middle one handed over whole.
        let mut single = Triplets::new(n);
        for &(r, c, block) in &list {
            single.add(r, c, block);
        }
        let mut parts = Triplets::new(n);
        let third = entries / 3;
        for &(r, c, block) in &list[..third] {
            parts.add(r, c, block);
        }
        let mut whole: Vec<_> = list[third..2 * third]
            .iter()
            .map(|&(r, c, block)| (r as u32, c as u32, block))
            .collect();
        parts.append(&mut whole);
        assert!(whole.is_empty());
        for &(r, c, block) in &list[2 * third..] {
            parts.add(r, c, block);
        }
        assert_eq!(parts.len(), entries);
        let mut reused = BlockMatrix::default();
        parts.assemble_into(&mut reused);
        parts.assemble_into(&mut reused);
        for matrix in [single.build(), parts.build(), reused.clone()] {
            assert_eq!(matrix.stored(), expected.len());
            for row in 0..n {
                let columns: Vec<usize> = matrix.row(row).map(|(c, _)| c).collect();
                assert!(columns.windows(2).all(|w| w[0] < w[1]));
            }
            for (&(r, c), sum) in &expected {
                assert_eq!(matrix.get(r, c), *sum, "block ({r}, {c}) of {n}");
            }
        }
        // Emptied and filled again, nothing of the last use remains.
        parts.reset(n);
        assert!(parts.is_empty());
        parts.add(1, 2, Mat3::IDENTITY);
        parts.assemble_into(&mut reused);
        assert_eq!(reused.stored(), 1);
        assert_eq!(reused.get(1, 2), Mat3::IDENTITY);
    }
}

#[test]
fn product_matches_dense() {
    let pool = Pool::new(4);
    let m = lattice(4, 3.0, 0.5).build();
    let n = m.dimension();
    let mut rng = Rng::new(1);
    let x: Vec<f64> = (0..n).map(|_| rng.normal()).collect();
    let mut y = vec![0.0; n];
    m.apply(&pool, &x, &mut y);
    for i in 0..m.size() {
        let expected: Vec3 = m.row(i).map(|(j, b)| *b * Vec3::read(&x, j)).sum();
        assert!((Vec3::read(&y, i) - expected).length() < 1e-12);
    }
}

#[test]
fn conjugate_gradient_solves_and_is_deterministic() {
    let m = lattice(12, 1e4, 1.0).build();
    let n = m.dimension();
    let mut rng = Rng::new(2);
    let truth: Vec<f64> = (0..n).map(|_| rng.normal()).collect();
    let pool = Pool::new(8);
    let mut b = vec![0.0; n];
    m.apply(&pool, &truth, &mut b);

    let solve = |pool: &Pool| {
        let mut x = vec![0.0; n];
        let report = conjugate_gradient(
            pool,
            &m,
            &BlockJacobi::new(pool, &m),
            &b,
            &mut x,
            CgSettings {
                tolerance: 1e-10,
                max_iterations: 5000,
            },
            &mut CgWorkspace::default(),
        );
        (x, report)
    };
    let (x, report) = solve(&pool);
    assert!(report.converged, "{report:?}");
    let error = x
        .iter()
        .zip(&truth)
        .map(|(a, b)| (a - b).abs())
        .fold(0.0, f64::max);
    assert!(error < 1e-5, "error {error}, {report:?}");
    let (x1, _) = solve(&Pool::new(1));
    assert!(
        x.iter().zip(&x1).all(|(a, b)| a.to_bits() == b.to_bits()),
        "one thread and eight differ"
    );
}

#[test]
#[allow(clippy::needless_range_loop)]
fn dense_eigen_and_projection() {
    let mut rng = Rng::new(3);
    let mut a = [[0.0; 12]; 12];
    for i in 0..12 {
        for j in 0..=i {
            let v = rng.normal();
            a[i][j] = v;
            a[j][i] = v;
        }
    }
    let (values, vectors) = dense::symmetric_eigen(&a);
    for i in 0..12 {
        for j in 0..12 {
            let rebuilt: f64 = (0..12)
                .map(|k| vectors[i][k] * values[k] * vectors[j][k])
                .sum();
            assert!((rebuilt - a[i][j]).abs() < 1e-12);
        }
    }
    dense::project_positive(&mut a);
    let (values, _) = dense::symmetric_eigen(&a);
    assert!(values.iter().all(|&v| v > -1e-12));
}

#[test]
fn dual_numbers_differentiate() {
    let [x, y] = Dual::<2>::variables([1.3, -0.4]);
    let f = (x * y).sin() + x.ln() * y.powi(3) - (x / (y * y + 1.0)).sqrt();
    let (xv, yv) = (1.3f64, -0.4f64);
    let value = |x: f64, y: f64| (x * y).sin() + x.ln() * y.powi(3) - (x / (y * y + 1.0)).sqrt();
    assert!((f.value - value(xv, yv)).abs() < 1e-15);
    let error = check::gradient_error(|p| value(p[0], p[1]), &[xv, yv], &f.gradient, 1e-6);
    assert!(error < 1e-8, "{error}");
}

/// Springs between points, pinned at both ends: a small nonlinear energy.
#[derive(Clone)]
struct Chain {
    rest: f64,
    k: f64,
    ends: [Vec3; 2],
    gravity: f64,
    triplets: Triplets,
    hessian: BlockMatrix,
}

impl Chain {
    fn points(&self, x: &[f64]) -> Vec<Vec3> {
        let mut p = vec![self.ends[0]];
        p.extend((0..x.len() / 3).map(|i| Vec3::read(x, i)));
        p.push(self.ends[1]);
        p
    }
}

impl Objective for Chain {
    type Hessian = BlockMatrix;

    fn hessian(&self) -> &BlockMatrix {
        &self.hessian
    }

    fn blocks(&self) -> usize {
        8
    }

    fn energy(&mut self, _: &Pool, x: &[f64]) -> f64 {
        let p = self.points(x);
        let springs: f64 = p
            .windows(2)
            .map(|w| 0.5 * self.k * (w[1].distance(w[0]) - self.rest).powi(2))
            .sum();
        springs
            + p[1..p.len() - 1]
                .iter()
                .map(|q| self.gravity * q.y)
                .sum::<f64>()
    }

    fn derivatives(&mut self, _: &Pool, x: &[f64], gradient: &mut [f64]) {
        let mut hessian = std::mem::take(&mut self.triplets);
        hessian.reset(self.blocks());
        let p = self.points(x);
        let inner = |i: usize| (i >= 1 && i <= self.blocks()).then(|| i - 1);
        for i in 0..p.len() - 1 {
            let d = p[i + 1] - p[i];
            let length = d.length();
            let u = d / length;
            let force = u * (self.k * (length - self.rest));
            // Exact spring Hessian, projected to be positive.
            let h = (Mat3::outer(u, u) * self.k
                + (Mat3::IDENTITY - Mat3::outer(u, u)) * (self.k * (1.0 - self.rest / length)))
                .project_positive();
            if let Some(a) = inner(i) {
                (-force).add_to(gradient, a);
                hessian.add(a, a, h);
            }
            if let Some(b) = inner(i + 1) {
                force.add_to(gradient, b);
                hessian.add(b, b, h);
            }
            if let (Some(a), Some(b)) = (inner(i), inner(i + 1)) {
                hessian.add_symmetric(a, b, -h);
            }
        }
        for i in 0..self.blocks() {
            Vec3::new(0.0, self.gravity, 0.0).add_to(gradient, i);
        }
        hessian.build_into(&mut self.hessian);
        self.triplets = hessian;
    }
}

#[test]
fn newton_hangs_a_chain() {
    let pool = Pool::new(2);
    let mut chain = Chain {
        rest: 0.1,
        k: 500.0,
        ends: [Vec3::ZERO, Vec3::new(1.0, 0.0, 0.0)],
        gravity: 9.81 * 0.05,
        triplets: Triplets::default(),
        hessian: BlockMatrix::default(),
    };
    let mut x: Vec<f64> = (0..8)
        .flat_map(|i| [(i + 1) as f64 / 9.0, 0.0, 0.01 * i as f64])
        .collect();

    // The gradient is right.
    let mut g = vec![0.0; 24];
    chain.derivatives(&pool, &x, &mut g);
    let error = check::gradient_error(|p| chain.clone().energy(&pool, p), &x, &g, 1e-6);
    assert!(error < 1e-7, "gradient error {error}");

    let mut newton = Newton::new(NewtonSettings {
        tolerance: 1e-12,
        ..Default::default()
    });
    let report = newton.minimize(&pool, &mut chain, &mut x);
    assert!(report.converged, "{report:?}");
    // At the minimum the gradient vanishes, and the chain sags.
    let mut g = vec![0.0; 24];
    chain.derivatives(&pool, &x, &mut g);
    assert!(vector::norm_max(&pool, &g) < 1e-8, "{report:?}");
    assert!(x[3 * 4 + 1] < -0.01);
    assert!(
        x.iter().skip(2).step_by(3).all(|z| z.abs() < 1e-9),
        "the chain falls into its plane"
    );
}

#[test]
fn fixed_patterns_fill_in_place() {
    let pool = Pool::new(4);
    let reference = lattice(6, 2.0, 1.0).build();
    let n = reference.size();
    // The same matrix, as a pattern filled row by row in parallel.
    let mut m = BlockMatrix::with_pattern(
        n,
        (0..n)
            .flat_map(|i| reference.row(i).map(move |(j, _)| (i, j)))
            .collect::<Vec<_>>(),
    );
    assert_eq!(m.stored(), reference.stored());
    m.fill_rows(&pool, 7, |row, first_slot, blocks| {
        for (k, (column, value)) in reference.row(row).enumerate() {
            assert_eq!(m_slot(&reference, row, column), first_slot + k);
            blocks[k] = *value;
        }
    });
    let mut rng = Rng::new(4);
    let x: Vec<f64> = (0..3 * n).map(|_| rng.normal()).collect();
    let (mut y0, mut y1) = (vec![0.0; 3 * n], vec![0.0; 3 * n]);
    reference.apply(&pool, &x, &mut y0);
    m.apply(&pool, &x, &mut y1);
    assert_eq!(y0, y1);
    assert_eq!(m.slot(0, n - 1), None);
    m.clear_values();
    assert!(m.values().iter().all(|b| *b == Mat3::ZERO));
}

fn m_slot(m: &BlockMatrix, row: usize, column: usize) -> usize {
    m.slot(row, column).unwrap()
}

#[test]
fn sums_apply_and_precondition_as_one_matrix() {
    let pool = Pool::new(3);
    let a = lattice(5, 2.0, 1.0);
    let mut b = Triplets::new(a.size());
    b.add_symmetric(3, 70, Mat3::scalar(-0.5));
    b.add(3, 3, Mat3::scalar(0.5));
    b.add(70, 70, Mat3::scalar(0.5));
    let mut both = a.clone();
    both.add_symmetric(3, 70, Mat3::scalar(-0.5));
    both.add(3, 3, Mat3::scalar(0.5));
    both.add(70, 70, Mat3::scalar(0.5));
    let (a, b, both) = (a.build(), b.build(), both.build());
    let sum = Sum(&a, &b);
    let mut rng = Rng::new(5);
    let x: Vec<f64> = (0..sum.dimension()).map(|_| rng.normal()).collect();
    let (mut y0, mut y1) = (vec![0.0; x.len()], vec![0.0; x.len()]);
    both.apply(&pool, &x, &mut y0);
    sum.apply(&pool, &x, &mut y1);
    assert!(y0.iter().zip(&y1).all(|(p, q)| (p - q).abs() < 1e-12));
    let (mut d0, mut d1) = (vec![Mat3::ZERO; a.size()], vec![Mat3::ZERO; a.size()]);
    both.add_diagonal(&mut d0);
    sum.add_diagonal(&mut d1);
    assert!(d0
        .iter()
        .zip(&d1)
        .all(|(p, q)| (*p - *q).frobenius_squared() < 1e-24));
}

#[test]
fn not_a_number_is_never_small() {
    let pool = Pool::new(2);
    let mut v = vec![0.0; 10_000];
    v[7777] = f64::NAN;
    assert!(vector::norm_max(&pool, &v).is_nan());
    let m = lattice(3, 1.0, 1.0).build();
    let mut b = vec![1.0; m.dimension()];
    b[4] = f64::NAN;
    let mut x = vec![0.0; m.dimension()];
    let report = conjugate_gradient(
        &pool,
        &m,
        &BlockJacobi::new(&pool, &m),
        &b,
        &mut x,
        CgSettings::default(),
        &mut CgWorkspace::default(),
    );
    assert!(report.not_finite && !report.converged);
}

#[test]
fn coupled_pairs_precondition_together() {
    // Pairs of blocks coupled almost as strongly as they are stiff — a
    // rigid body's position and rotation under stiff contact — on a weak
    // chain: block-Jacobi alone struggles, pairs fix it.
    let n = 400;
    let mut t = Triplets::new(n);
    let mut rng = Rng::new(7);
    for b in (0..n).step_by(2) {
        let d = rng.direction();
        let stiff = Mat3::outer(d, d) * 1e8;
        t.add(b, b, stiff + Mat3::scalar(1.0));
        t.add(b + 1, b + 1, stiff + Mat3::scalar(1.0));
        t.add_symmetric(b, b + 1, -stiff * 0.999_999);
        if b + 2 < n {
            t.add(b + 1, b + 1, Mat3::scalar(0.5));
            t.add(b + 2, b + 2, Mat3::scalar(0.5));
            t.add_symmetric(b + 1, b + 2, Mat3::scalar(-0.5));
        }
    }
    let m = t.build();
    let pool = Pool::new(4);
    let b: Vec<f64> = (0..3 * n).map(|_| rng.normal()).collect();
    let solve = |pairs: &[(usize, usize)]| {
        let mut p = BlockJacobi::default();
        p.update(&pool, &m, pairs);
        let mut x = vec![0.0; 3 * n];
        conjugate_gradient(
            &pool,
            &m,
            &p,
            &b,
            &mut x,
            CgSettings {
                tolerance: 1e-8,
                max_iterations: 20_000,
            },
            &mut CgWorkspace::default(),
        )
    };
    let alone = solve(&[]);
    let pairs: Vec<(usize, usize)> = (0..n).step_by(2).map(|b| (b, 2)).collect();
    let together = solve(&pairs);
    assert!(alone.converged, "{alone:?}");
    assert!(together.converged, "{together:?}");
    // Equilibrated 3x3 factors also improve the ungrouped solve; grouping
    // must still provide a substantial additional reduction in work.
    assert!(
        together.iterations * 2 < alone.iterations,
        "pairs {} vs alone {}",
        together.iterations,
        alone.iterations
    );
}

#[test]
fn zero_start_cg_matches_nonzero_path_and_project_positive_handles_all_regimes() {
    let m = lattice(6, 1e3, 1.0).build();
    let n = m.dimension();
    let mut rng = Rng::new(11);
    let b: Vec<f64> = (0..n).map(|_| rng.normal()).collect();
    let pool = Pool::new(2);
    let pre = BlockJacobi::new(&pool, &m);
    let mut x0 = vec![0.0; n];
    let r0 = conjugate_gradient(
        &pool,
        &m,
        &pre,
        &b,
        &mut x0,
        CgSettings::default(),
        &mut CgWorkspace::default(),
    );
    assert!(r0.converged);
    // Zero-start skips the redundant A*0 evaluation.
    assert_eq!(r0.matrix_products, r0.iterations + r0.residual_checks);
    assert!(r0.residual_checks >= 1);

    // Cholesky fast path preserves SPD matrices identically; fallback projects singular/indefinite.
    let mut spd = [[0.0; 4]; 4];
    for i in 0..4 {
        spd[i][i] = 4.0 + i as f64;
        if i + 1 < 4 {
            spd[i][i + 1] = 0.5;
            spd[i + 1][i] = 0.5;
        }
    }
    let before = spd;
    dense::project_positive(&mut spd);
    assert_eq!(spd, before);

    let mut indefinite = [[2.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, -3.0]];
    dense::project_positive(&mut indefinite);
    assert!((indefinite[0][0] - 2.0).abs() < 1e-12);
    assert!(indefinite[1][1].abs() < 1e-12);
    assert!(indefinite[2][2].abs() < 1e-12);
}

#[test]
fn sparse_cholesky_solves_ill_conditioned_and_direct_newton_systems() {
    // 1. High-contrast (1e8 stiffness ratio) coupled lattice.
    let m = lattice(8, 1e8, 1.0).build();
    let n = m.dimension();
    let mut rng = Rng::new(77);
    let truth: Vec<f64> = (0..n).map(|_| rng.normal()).collect();
    let mut rhs = vec![0.0; n];
    m.apply(&Pool::new(1), &truth, &mut rhs);

    let mut chol = core_numerics::SparseCholesky::default();
    assert!(chol.factor(&m));
    assert!(!chol.was_regularized());
    let mut solved = vec![0.0; n];
    assert!(chol.solve(&rhs, &mut solved));
    let max_err = solved
        .iter()
        .zip(&truth)
        .map(|(a, b)| (a - b).abs())
        .fold(0.0, f64::max);
    assert!(max_err < 1e-8, "SparseCholesky error {max_err}");
    let mut check_rhs = vec![0.0; n];
    m.apply(&Pool::new(1), &solved, &mut check_rhs);
    let rhs_norm = rhs.iter().map(|v| v * v).sum::<f64>().sqrt();
    let res_norm = check_rhs
        .iter()
        .zip(&rhs)
        .map(|(a, b)| (a - b) * (a - b))
        .sum::<f64>()
        .sqrt();
    assert!(
        res_norm / rhs_norm < 1e-12,
        "SparseCholesky residual {}",
        res_norm / rhs_norm
    );

    // 2. Direct Newton mode on the hanging chain converges with 0 CG iterations
    // and reaches the same equilibrium as CG Newton.
    let make_chain = || Chain {
        rest: 0.1,
        k: 500.0,
        ends: [Vec3::ZERO, Vec3::new(1.0, 0.0, 0.0)],
        gravity: 9.81 * 0.05,
        triplets: Triplets::default(),
        hessian: BlockMatrix::default(),
    };
    let x_init: Vec<f64> = (0..8)
        .flat_map(|i| [(i + 1) as f64 / 9.0, 0.0, 0.01 * i as f64])
        .collect();
    let mut x_dir = x_init.clone();
    let mut chain_dir = make_chain();
    let mut newton_dir = Newton::new(NewtonSettings {
        tolerance: 1e-12,
        direct_blocks: chain_dir.blocks(),
        ..Default::default()
    });
    let report = newton_dir.minimize(&Pool::new(1), &mut chain_dir, &mut x_dir);
    assert!(report.converged, "{report:?}");
    assert_eq!(report.linear_iterations, 0);

    let mut x_cg = x_init;
    let mut chain_cg = make_chain();
    let mut newton_cg = Newton::new(NewtonSettings {
        tolerance: 1e-12,
        ..Default::default()
    });
    let report_cg = newton_cg.minimize(&Pool::new(1), &mut chain_cg, &mut x_cg);
    assert!(report_cg.converged);
    let diff = x_dir
        .iter()
        .zip(&x_cg)
        .map(|(a, b)| (a - b).abs())
        .fold(0.0, f64::max);
    assert!(diff < 1e-11, "Direct vs CG equilibrium diff {diff}");
}
