//! Coupled rigid-body preconditioning: setup, application, and solve cost.
use std::{hint::black_box, time::Instant};

use core_math::{Mat3, Rng, Vec3};
use core_numerics::{
    conjugate_gradient, BlockJacobi, CgSettings, CgWorkspace, LinearOperator, Preconditioner,
    Triplets,
};
use core_runtime::Pool;

fn main() {
    let count = std::env::var("PC_GROUPS")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(512);
    let threads = std::env::var("PC_THREADS")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(1);
    let repeats = 30;
    let pool = Pool::new(threads);
    let mut triplets = Triplets::new(count * 4);
    let mut rng = Rng::new(310);
    for body in 0..count {
        let first = 4 * body;
        for a in 0..4 {
            triplets.add(first + a, first + a, Mat3::scalar(1.0));
        }
        // Normal/friction contact rows couple translation and affine axes.
        for _ in 0..4 {
            let row: [Vec3; 4] = std::array::from_fn(|_| rng.direction());
            for a in 0..4 {
                for b in 0..4 {
                    triplets.add(first + a, first + b, Mat3::outer(row[a], row[b]) * 1e5);
                }
            }
        }
        if body > 0 {
            triplets.add(first, first, Mat3::scalar(0.1));
            triplets.add(first - 4, first - 4, Mat3::scalar(0.1));
            triplets.add_symmetric(first, first - 4, Mat3::scalar(-0.1));
        }
    }
    let matrix = triplets.build();
    let groups: Vec<_> = (0..count).map(|b| (4 * b, 4)).collect();
    let mut preconditioner = BlockJacobi::default();
    preconditioner.update(&pool, &matrix, &groups);
    let start = Instant::now();
    for _ in 0..repeats {
        preconditioner.update(&pool, black_box(&matrix), black_box(&groups));
    }
    let setup = start.elapsed().as_secs_f64() * 1e3 / repeats as f64;
    let rhs: Vec<_> = (0..matrix.dimension()).map(|_| rng.normal()).collect();
    let mut output = vec![0.0; rhs.len()];
    let start = Instant::now();
    for _ in 0..100 {
        pool.for_chunks_mut(&mut output, core_numerics::vector::CHUNK, |offset, part| {
            preconditioner.apply_part(offset, &rhs[offset..offset + part.len()], part);
        });
        black_box(&output);
    }
    let apply = start.elapsed().as_secs_f64() * 1e6 / 100.0;
    let mut solution = vec![0.0; rhs.len()];
    let report = conjugate_gradient(
        &pool,
        &matrix,
        &preconditioner,
        &rhs,
        &mut solution,
        CgSettings {
            tolerance: 1e-8,
            max_iterations: 5000,
        },
        &mut CgWorkspace::default(),
    );
    matrix.apply(&pool, &solution, &mut output);
    let residual = output
        .iter()
        .zip(&rhs)
        .map(|(a, b)| (a - b).powi(2))
        .sum::<f64>()
        .sqrt()
        / rhs.iter().map(|b| b * b).sum::<f64>().sqrt();
    assert!(
        report.converged && residual < 2e-8,
        "{report:?}; true residual {residual}"
    );
    println!("groups={count} threads={threads} setup_ms={setup:.4} apply_us={apply:.3} cg={} true_residual={residual:.3e}", report.iterations);
}
