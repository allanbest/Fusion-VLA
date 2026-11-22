//! Operations on long vectors of `f64`, parallel and deterministic: every
//! sum is taken over the same fixed chunks in the same order, whatever the
//! number of threads, so results are identical to the bit.

use core_runtime::Pool;

use crate::cg::dot as dot_part;

/// Elements per chunk: large enough that a chunk outweighs handing it out,
/// small enough to spread a mid-sized problem over the cores. A multiple of
/// 3, so chunks hold whole blocks.
pub const CHUNK: usize = 3 * 1024;

/// Elements per chunk of a pass that computes each block on its own, with
/// nothing summed across them — a matrix times a vector. Its chunks can be
/// any size without changing a bit of the result; these are small enough
/// that an island of a hundred nodes, whose matrix products are the
/// largest part of its solve, is spread over the threads free to help.
pub const ROWS: usize = 3 * 32;

/// `Σ aᵢ bᵢ`.
pub fn dot(pool: &Pool, a: &[f64], b: &[f64]) -> f64 {
    debug_assert_eq!(a.len(), b.len());
    pool.map_reduce(
        a.len(),
        CHUNK,
        0.0,
        |r| dot_part(&a[r.clone()], &b[r]),
        |x, y| x + y,
    )
}

/// `Σ aᵢ²`.
pub fn norm_squared(pool: &Pool, a: &[f64]) -> f64 {
    dot(pool, a, a)
}

/// `max |aᵢ|`; NaN if any element is NaN (so a broken vector is never
/// mistaken for a small one).
pub fn norm_max(pool: &Pool, a: &[f64]) -> f64 {
    let max = |m: f64, v: f64| {
        if m.is_nan() || v.is_nan() {
            f64::NAN
        } else {
            m.max(v)
        }
    };
    pool.map_reduce(
        a.len(),
        CHUNK,
        0.0,
        |r| a[r].iter().fold(0.0, |m, v| max(m, v.abs())),
        max,
    )
}

/// `y ← y + α x`.
pub fn add_scaled(pool: &Pool, y: &mut [f64], alpha: f64, x: &[f64]) {
    debug_assert_eq!(x.len(), y.len());
    pool.for_chunks_mut(y, CHUNK, |offset, part| {
        for (yi, xi) in part.iter_mut().zip(&x[offset..]) {
            *yi += alpha * xi;
        }
    });
}

/// `y ← x + β y`.
pub fn scale_and_add(pool: &Pool, y: &mut [f64], beta: f64, x: &[f64]) {
    debug_assert_eq!(x.len(), y.len());
    pool.for_chunks_mut(y, CHUNK, |offset, part| {
        for (yi, xi) in part.iter_mut().zip(&x[offset..]) {
            *yi = xi + beta * *yi;
        }
    });
}
