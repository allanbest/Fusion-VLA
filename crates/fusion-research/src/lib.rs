//! Fusion L5 — the research workflow.
//!
//! PLAN.md §23 lists seven things and calls the set "Advanced's differentiator
//! over being just a solver library… mostly orchestration over the project
//! format plus a results store". Three of the seven were already in the
//! engine — checkpoint/resume, the energy ledger, and the determinism
//! guarantee — and this crate is the other four, plus the store they all
//! write to.
//!
//! | §23 asks for | where it is |
//! |---|---|
//! | Parameter sweeps | [`sweep::Sweep`] |
//! | Checkpoint / resume | `fusion_sim::Timeline`, from v0.4 |
//! | Run comparison and sim diffing | [`diff::compare`] |
//! | Conservation diagnostics | [`metrics::ConservationTrace`] |
//! | Convergence studies | [`converge::study`] |
//! | Determinism diff | [`diff::determinism_diff`] |
//! | Energy ledger | `fusion_solver::EnergyLedger`, from v0.5 |
//!
//! ## What makes this research rather than scripting
//!
//! Every one of these could be a loop somebody writes in an afternoon. What
//! they could not write in an afternoon is the part that makes the answers
//! keep: each run records the `state_hash` it ended on and the scene it came
//! from, so a row in a results table is re-runnable rather than remembered.
//! §22 already made the project file enough to reproduce a run; this makes the
//! *table* enough to reproduce a study.
//!
//! The crate sits at L5 alongside `fusion-project` and takes the same two
//! external dependencies for the same reason — serialization is a
//! compatibility surface, and a hand-rolled one corrupts saved work rather
//! than producing a visibly wrong picture. Nothing below L5 depends on it.

#![deny(unsafe_code)]
#![deny(missing_docs)]

pub mod converge;
pub mod diff;
pub mod metrics;
pub mod store;
pub mod sweep;

pub use converge::{study, ConvergenceReport};
pub use diff::{compare, determinism_diff, ComparisonReport, DeterminismReport};
pub use metrics::{ConservationSummary, ConservationTrace, Conserved};
pub use store::{StoreError, Study};
pub use sweep::{Axis, RunRecord, Sweep, SweepResults};
