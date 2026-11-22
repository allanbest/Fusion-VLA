//! Fusion L3 — GPU implementations of the solver's hot loops.
//!
//! ## Why this is a separate crate from the solver
//!
//! `fusion-solver` has no external dependencies and must run on a machine with
//! no display and no GPU (PLAN.md §25). A GPU implementation of anything in it
//! therefore cannot live in it. This crate depends on
//! [`fusion_compute`](fusion_compute) and nothing in the core depends on this
//! one, so the property is kept by the dependency graph rather than by
//! discipline.
//!
//! ## Why this is a separate crate from `fusion-compute`
//!
//! That one is a general compute layer — buffers, kernels, a command graph —
//! and knows nothing about particles. This one knows about nothing else.
//!
//! ## What is here
//!
//! v0.6 part 3 puts the fluid's neighbour pipeline on the GPU, which the
//! profile in `bench profile` says is the largest single cost in a fluid
//! substep — 42% of one after part 1 made it 1.4x cheaper. It is also the part
//! that vectorizes worst on a CPU and best on a GPU: two hundred independent
//! distance tests per particle, all the same shape.
//!
//! The density solve that consumes those neighbours is still on the CPU. That
//! is a deliberate stopping point rather than a half-finished one — see
//! [`GpuGrid`] on what it costs to hand the lists back across the bus, and
//! `docs/STATUS.md` for the measurement that says a partial GPU pipeline is
//! the wrong shape.

#![forbid(unsafe_code)]

mod grid;
mod scan;

pub use grid::{GpuGrid, MAX_NEIGHBOURS};
pub use scan::{Scan, BLOCK};
