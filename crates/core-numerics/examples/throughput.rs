//! What the hot paths of an implicit step cost: assembling an
//! element-shaped matrix, a product, and a CG solve.
//!
//! `cargo run --release -p core-numerics --example throughput`

use std::time::Instant;

use core_math::{Mat3, Rng, Vec3};
use core_numerics::{
    conjugate_gradient, BlockJacobi, BlockMatrix, CgSettings, CgWorkspace, LinearOperator, Triplets,
};
use core_runtime::Pool;
use core_world::TetMesh;

fn best<T>(runs: usize, mut f: impl FnMut() -> T) -> (f64, T) {
    let mut best = f64::INFINITY;
    let mut out = None;
    for _ in 0..runs {
        let start = Instant::now();
        let r = f();
        best = best.min(start.elapsed().as_secs_f64());
        out = Some(r);
    }
    (best, out.unwrap())
}

/// Each tetrahedron adds a dense 4×4 block of 3×3 blocks, like an elastic
/// element's Hessian; plus a mass on the diagonal.
fn assemble(mesh: &TetMesh, triplets: &mut Triplets) {
    triplets.reset(mesh.rest.len());
    for (i, _) in mesh.rest.iter().enumerate() {
        triplets.add(i, i, Mat3::scalar(1.0));
    }
    for (k, t) in mesh.tets.iter().enumerate() {
        let d = Vec3::new(1.0, (k % 7) as f64 * 0.1, 0.3);
        let b = Mat3::outer(d, d) + Mat3::scalar(0.5);
        for (a, &i) in t.iter().enumerate() {
            for (c, &j) in t.iter().enumerate() {
                triplets.add(i as usize, j as usize, if a == c { b * 3.0 } else { -b });
            }
        }
    }
}

fn main() {
    let cores = std::thread::available_parallelism().map_or(1, |n| n.get());
    for cells in [10usize, 20, 40] {
        let mesh = TetMesh::cuboid(Vec3::splat(1.0), 1.0 / cells as f64);
        let n = mesh.rest.len();
        println!("\n{n} nodes, {} tetrahedra", mesh.tets.len());
        let mut triplets = Triplets::default();
        let (t, _) = best(5, || assemble(&mesh, &mut triplets));
        println!(
            "  fill triplets ({} entries)      {:>8.2} ms",
            triplets.len(),
            t * 1e3
        );
        let mut matrix = BlockMatrix::default();
        let (t, _) = best(5, || triplets.build_into(&mut matrix));
        println!(
            "  build matrix ({} blocks)        {:>8.2} ms",
            matrix.stored(),
            t * 1e3
        );

        // The same matrix through a fixed pattern: slots found once, then
        // each fill writes values straight in.
        let mut fixed = BlockMatrix::with_pattern(
            n,
            (0..n)
                .flat_map(|i| matrix.row(i).map(move |(j, _)| (i, j)))
                .collect::<Vec<_>>(),
        );
        let slots: Vec<[usize; 16]> = mesh
            .tets
            .iter()
            .map(|t| {
                std::array::from_fn(|k| fixed.slot(t[k / 4] as usize, t[k % 4] as usize).unwrap())
            })
            .collect();
        let diagonal: Vec<usize> = (0..n).map(|i| fixed.slot(i, i).unwrap()).collect();
        let (t, _) = best(5, || {
            fixed.clear_values();
            let values = fixed.values_mut();
            for &s in &diagonal {
                values[s] += Mat3::scalar(1.0);
            }
            for (k, s) in slots.iter().enumerate() {
                let d = Vec3::new(1.0, (k % 7) as f64 * 0.1, 0.3);
                let b = Mat3::outer(d, d) + Mat3::scalar(0.5);
                for (e, &slot) in s.iter().enumerate() {
                    values[slot] += if e / 4 == e % 4 { b * 3.0 } else { -b };
                }
            }
        });
        println!("  fill fixed pattern by slots        {:>8.2} ms", t * 1e3);
        assert!(fixed
            .values()
            .iter()
            .zip(matrix.values())
            .all(|(a, b)| (*a - *b).frobenius_squared() < 1e-20));

        let mut rng = Rng::new(1);
        let x: Vec<f64> = (0..3 * n).map(|_| rng.normal()).collect();
        let mut y = vec![0.0; 3 * n];
        for threads in [1, 4, 8, cores] {
            let pool = Pool::new(threads);
            let (t, _) = best(20, || matrix.apply(&pool, &x, &mut y));
            let flops = matrix.stored() as f64 * 18.0;
            println!(
                "  product, {threads:>2} threads             {:>8.3} ms  ({:.2} GFLOP/s)",
                t * 1e3,
                flops / t * 1e-9
            );
            let mut work = CgWorkspace::default();
            let (t, report) = best(3, || {
                let mut sol = vec![0.0; 3 * n];
                conjugate_gradient(
                    &pool,
                    &matrix,
                    &BlockJacobi::new(&pool, &matrix),
                    &x,
                    &mut sol,
                    CgSettings {
                        tolerance: 1e-6,
                        max_iterations: 10_000,
                    },
                    &mut work,
                )
            });
            println!("  CG to 1e-6, {threads:>2} threads          {:>8.2} ms  ({} iterations, {:.1} µs each)", t * 1e3, report.iterations, t * 1e6 / report.iterations as f64);
        }
    }
}
