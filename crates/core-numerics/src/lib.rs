//! Layer 2 of the Fusion core: the numerical methods every module solves
//! with.
//!
//! - [`vector`]: parallel, deterministic operations on long vectors.
//! - [`BlockMatrix`], [`Triplets`], [`Sum`]: sparse matrices of 3×3 blocks,
//!   with fixed patterns or changing ones.
//! - [`conjugate_gradient`]: preconditioned CG for symmetric systems.
//! - [`Newton`]: minimising an energy with a line search — the implicit step.
//! - [`dense`]: small symmetric eigenproblems and positive projection.
//! - [`Dual`]: forward-mode automatic differentiation.
//! - [`check`]: derivatives against finite differences.
//!
//! Multigrid, incomplete factorisations and quadrature rules join when a
//! module needs them.

mod block;
mod cg;
pub mod check;
pub mod dense;
mod direct;
mod dual;
mod newton;
pub mod vector;

use core_math::Mat3;
use core_runtime::Pool;

pub use block::{BlockMatrix, Sum, Triplet, Triplets};
pub use cg::{
    conjugate_gradient, BlockJacobi, CgReport, CgSettings, CgWorkspace, Identity, Preconditioner,
};
pub use direct::SparseCholesky;
pub use dual::Dual;
pub use newton::{Newton, NewtonReport, NewtonSettings, Objective};

/// A linear map `y = A x` — a stored matrix, a sum of them, or a product
/// computed on the fly — evaluated a part of `y` at a time, so that callers
/// can fuse it with other work in one parallel pass.
pub trait LinearOperator: Sync {
    fn dimension(&self) -> usize;

    /// `out += (A x)[offset .. offset + out.len()]`; `offset` and the length
    /// are multiples of 3.
    fn add_part(&self, x: &[f64], offset: usize, out: &mut [f64]);

    /// `y = A x`, in parallel. Every row is computed on its own, so the
    /// chunks may be any size: they are small ([`vector::ROWS`]), so that
    /// a system of a few hundred unknowns is still shared between threads.
    fn apply(&self, pool: &Pool, x: &[f64], y: &mut [f64]) {
        pool.for_chunks_mut(y, vector::ROWS, |offset, part| {
            part.fill(0.0);
            self.add_part(x, offset, part);
        });
    }
}

/// A linear operator on 3-blocks that can say what its diagonal blocks are
/// (for preconditioning).
pub trait BlockOperator: LinearOperator {
    /// Adds the diagonal blocks to `diagonal`.
    fn add_diagonal(&self, diagonal: &mut [Mat3]);

    /// Block `(row, column)`.
    fn block(&self, row: usize, column: usize) -> Mat3;

    /// Visits every potentially non-zero `(column, block)` in `row`.
    fn for_each_row_block(&self, row: usize, visit: &mut dyn FnMut(usize, Mat3)) {
        let blocks = self.dimension() / 3;
        for col in 0..blocks {
            let b = self.block(row, col);
            if b != Mat3::ZERO {
                visit(col, b);
            }
        }
    }
}
