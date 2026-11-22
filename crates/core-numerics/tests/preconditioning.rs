use core_math::{Mat3, Rng, Vec3};
use core_numerics::{
    conjugate_gradient, vector, BlockJacobi, BlockMatrix, CgSettings, CgWorkspace, LinearOperator,
    Preconditioner, Triplets,
};
use core_runtime::Pool;

/// D (I + V V^T) D: well conditioned after equilibration, with wildly
/// different physical units before it. A global pivot cutoff rejects it.
fn scaled_positive(blocks: usize) -> (BlockMatrix, Vec<f64>) {
    let n = blocks * 3;
    let mut rng = Rng::new(991);
    let rows: Vec<[f64; 3]> = (0..n)
        .map(|_| std::array::from_fn(|_| rng.normal() * 0.3))
        .collect();
    let scale: Vec<_> = (0..n)
        .map(|i| 10f64.powf(-80.0 + 160.0 * i as f64 / (n - 1) as f64))
        .collect();
    let mut t = Triplets::new(blocks);
    for a in 0..blocks {
        for b in 0..blocks {
            let mut m = Mat3::ZERO;
            for i in 0..3 {
                for j in 0..3 {
                    let (r, c) = (3 * a + i, 3 * b + j);
                    let value = if r == c { 1.0 } else { 0.0 };
                    let value = value + (0..3).map(|k| rows[r][k] * rows[c][k]).sum::<f64>();
                    m[(i, j)] = (value * scale[r]) * scale[c];
                }
            }
            t.add(a, b, m);
        }
    }
    (t.build(), scale)
}

#[test]
fn equilibrated_blocks_solve_across_mass_scales() {
    for blocks in [1, 4] {
        let (matrix, scale) = scaled_positive(blocks);
        let truth: Vec<_> = scale
            .iter()
            .enumerate()
            .map(|(i, d)| (1.0 + i as f64 * 0.1) / d)
            .collect();
        let mut expected = None;
        for threads in [1, 4] {
            let pool = Pool::new(threads);
            let mut b = vec![0.0; truth.len()];
            matrix.apply(&pool, &truth, &mut b);
            let mut p = BlockJacobi::default();
            let groups = if blocks == 1 {
                vec![]
            } else {
                vec![(0, blocks)]
            };
            p.update(&pool, &matrix, &groups);
            let mut x = vec![0.0; truth.len()];
            let report = conjugate_gradient(
                &pool,
                &matrix,
                &p,
                &b,
                &mut x,
                CgSettings {
                    tolerance: 1e-12,
                    max_iterations: 20,
                },
                &mut CgWorkspace::default(),
            );
            assert!(
                report.converged && report.iterations == 1,
                "blocks={blocks}: {report:?}"
            );
            for i in 0..x.len() {
                assert!(((x[i] - truth[i]) * scale[i]).abs() < 1e-12);
            }
            let mut result = vec![0.0; x.len()];
            matrix.apply(&pool, &x, &mut result);
            for i in 0..x.len() {
                assert!(((result[i] - b[i]) / scale[i]).abs() < 1e-12);
            }
            if let Some(expected) = &expected {
                assert_eq!(&x, expected);
            } else {
                expected = Some(x);
            }
        }
    }
}

#[test]
fn unsafe_coupled_blocks_fall_back_to_positive_local_blocks() {
    let pool = Pool::new(1);
    let mut t = Triplets::new(2);
    t.add(0, 0, Mat3::IDENTITY);
    t.add(1, 1, Mat3::IDENTITY);
    t.add_symmetric(0, 1, Mat3::scalar(2.0)); // invertible, but indefinite
    let matrix = t.build();
    let mut p = BlockJacobi::default();
    p.update(&pool, &matrix, &[(0, 2)]);
    let r = [1.0, 2.0, -3.0, -1.0, -2.0, 3.0];
    let mut z = [0.0; 6];
    p.apply_part(0, &r, &mut z);
    assert_eq!(z, r);
    assert!(vector::dot(&pool, &r, &z) > 0.0);
}

#[test]
fn nonpositive_singular_or_nonfinite_diagonals_do_not_reverse_or_poison_residuals() {
    let pool = Pool::new(1);
    for diagonal in [
        vec3(-1.0, -2.0, -3.0),
        vec3(0.0, 1e-320, 2.0),
        vec3(f64::NAN, f64::INFINITY, 1.0),
    ] {
        let mut t = Triplets::new(1);
        t.add(0, 0, Mat3::from_diagonal(diagonal));
        let p = BlockJacobi::new(&pool, &t.build());
        let r = [1.0, -2.0, 3.0];
        let mut z = [0.0; 3];
        p.apply_part(0, &r, &mut z);
        assert!(z.iter().all(|v| v.is_finite()));
        assert!(r.iter().zip(z).map(|(a, b)| a * b).sum::<f64>() > 0.0);
    }
}

fn vec3(x: f64, y: f64, z: f64) -> Vec3 {
    Vec3::new(x, y, z)
}

#[test]
fn updates_remove_stale_couplings_and_accept_unsorted_groups() {
    let pool = Pool::new(4);
    let mut t = Triplets::new(8);
    for i in 0..8 {
        t.add(i, i, Mat3::scalar(2.0));
    }
    for first in [0, 4] {
        t.add_symmetric(first, first + 1, Mat3::scalar(0.5));
    }
    let matrix = t.build();
    let mut p = BlockJacobi::default();
    p.update(&pool, &matrix, &[(4, 4), (0, 4)]);
    let cloned = p.clone();
    let rhs: Vec<_> = (0..24).map(|i| i as f64 - 12.0).collect();
    let mut z = vec![0.0; 24];
    p.apply_part(0, &rhs, &mut z);
    let mut result = vec![0.0; 24];
    matrix.apply(&pool, &z, &mut result);
    assert!(result.iter().zip(&rhs).all(|(a, b)| (a - b).abs() < 1e-12));
    p.update(&pool, &matrix, &[]);
    p.apply_part(0, &rhs, &mut z);
    assert!(z.iter().zip(&rhs).all(|(a, b)| (a - 0.5 * b).abs() < 1e-12));
    cloned.apply_part(0, &rhs, &mut z);
    matrix.apply(&pool, &z, &mut result);
    assert!(result.iter().zip(&rhs).all(|(a, b)| (a - b).abs() < 1e-12));
}

#[test]
fn groups_at_cg_chunk_boundaries_keep_a_symmetric_positive_preconditioner() {
    let blocks = vector::CHUNK / 3 + 8;
    let first = vector::CHUNK / 3 - 2;
    let pool = Pool::new(4);
    let mut t = Triplets::new(blocks);
    for i in 0..blocks {
        t.add(i, i, Mat3::scalar(2.0));
    }
    t.add_symmetric(first, first + 3, Mat3::scalar(0.9));
    let matrix = t.build();
    let mut p = BlockJacobi::default();
    p.update(&pool, &matrix, &[(first, 4)]);
    let r: Vec<_> = (0..3 * blocks).map(|i| (i as f64).sin()).collect();
    let mut z = vec![0.0; r.len()];
    pool.for_chunks_mut(&mut z, vector::CHUNK, |offset, part| {
        p.apply_part(offset, &r[offset..offset + part.len()], part);
    });
    // Both halves of the straddling group consistently use local blocks.
    assert!(z.iter().zip(&r).all(|(a, b)| (a - 0.5 * b).abs() < 1e-14));
    assert!(vector::dot(&pool, &r, &z) > 0.0);
}
