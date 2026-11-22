//! What one quadrature point of a quadratic element costs, by part.
//! `cargo run --release -p phys-mechanics --example law_cost`
use std::hint::black_box;
use std::time::Instant;

use core_math::{Mat3, Quat, Rng, Vec3};
use phys_mechanics::law::NeoHookean;

fn main() {
    let law = NeoHookean::from_youngs(1e5, 0.3);
    let mut rng = Rng::new(1);
    let fs: Vec<Mat3> = (0..4096)
        .map(|_| {
            let mut f = Mat3::IDENTITY;
            for i in 0..3 {
                for j in 0..3 {
                    f[(i, j)] += 0.05 * rng.normal();
                }
            }
            Quat::from_axis_angle(rng.direction(), rng.range(0.0, 3.0)).to_mat3() * f
        })
        .collect();
    let g: [Vec3; 10] = std::array::from_fn(|_| rng.direction());
    let time = |name: &str, f: &dyn Fn(&Mat3) -> f64| {
        let mut best = f64::INFINITY;
        for _ in 0..5 {
            let start = Instant::now();
            let mut sink = 0.0;
            for m in &fs {
                sink += f(m);
            }
            black_box(sink);
            best = best.min(start.elapsed().as_secs_f64() / fs.len() as f64);
        }
        println!("{name:>28}: {:7.1} ns", best * 1e9);
    };
    time("svd", &|f| f.svd().1.x);
    time("symmetric eigen (FᵀF)", &|f| {
        (f.transpose() * *f).symmetric_eigen().0.x
    });
    time("stress", &|f| law.stress(f)[(0, 0)]);
    time("modes", &|f| law.modes(f).values[0]);
    time("for_each_negative_mode", &|f| {
        let mut warm = phys_mechanics::law::Warm::default();
        let mut sum = 0.0;
        law.for_each_negative_mode(f, &mut warm, |v, q| sum += v * q[(0, 0)]);
        sum
    });
    time("hessian blocks (55, 9 modes)", &|f| {
        let modes = law.modes(f);
        let mut h = [Mat3::ZERO; 100];
        for (value, q) in modes.values.iter().zip(&modes.matrices) {
            if *value <= 0.0 {
                continue;
            }
            let v = g.map(|gi| *q * gi);
            for a in 0..10 {
                let va = v[a] * *value;
                for b in a..10 {
                    h[10 * a + b] += Mat3::outer(va, v[b]);
                }
            }
        }
        h[11][(0, 0)]
    });
    time("hessian blocks (negative only)", &|f| {
        let (p, _, log_j, inverse_t) = law.stress_and_factors(f);
        let t = g.map(|gi| inverse_t * gi);
        let a_coeff = law.mu - law.lambda * log_j;
        let mut h = [Mat3::ZERO; 100];
        for a in 0..10 {
            for b in a..10 {
                h[10 * a + b] = Mat3::scalar(law.mu * g[a].dot(g[b]))
                    + Mat3::outer(t[b], t[a] * a_coeff)
                    + Mat3::outer(t[a], t[b] * law.lambda);
            }
        }
        let mut warm = phys_mechanics::law::Warm::default();
        law.for_each_negative_mode(f, &mut warm, |value, q| {
            let v = g.map(|gi| q * gi);
            for a in 0..10 {
                let va = v[a] * (-value);
                for b in a..10 {
                    h[10 * a + b] += Mat3::outer(va, v[b]);
                }
            }
        });
        h[11][(0, 0)] + p[(0, 0)]
    });
}
