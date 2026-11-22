//! The interface every mechanical law implements: a contribution to the
//! step's incremental potential
//!
//! ```text
//! E(x) = Σ ½ (x − x̃)ᵀ (M / β²) (x − x̃)  +  h² Σ Ψ(x)  +  contact barriers
//! ```
//!
//! — its value, gradient, a positive semi-definite Hessian, and how far a
//! step may safely go. Each term adds its own scaling (`h²` for physical
//! potentials), so the solver only sums.

use core_numerics::{BlockMatrix, Triplets};
use core_runtime::Pool;
use core_world::World;

use crate::dofs::{Frame, Layout};

/// What a term may read during a step.
pub struct Context<'a> {
    pub pool: &'a Pool,
    pub world: &'a World,
    pub layout: &'a Layout,
    pub frame: &'a Frame,
}

/// A mechanical law.
pub trait Term: Send {
    /// A name for reports and profiles.
    fn name(&self) -> &'static str;

    /// Called at the start of every integrator stage, before anything else.
    /// A term may cache between the evaluations of one stage's solve, but
    /// must forget here: a stage's result depends only on the state it
    /// starts from (so a run resumed from a checkpoint replays exactly).
    fn begin_step(&mut self, _ctx: &Context) {}

    /// The `(row, column)` blocks this term writes into the fixed-pattern
    /// Hessian; asked again only when the layout changes.
    fn pattern(&self, _ctx: &Context, _entries: &mut Vec<(usize, usize)>) {}

    /// The fixed pattern is built: look up slots.
    fn bind(&mut self, _ctx: &Context, _matrix: &BlockMatrix) {}

    /// The term's part of `E(x)`; `+∞` where `x` is not allowed.
    fn energy(&mut self, ctx: &Context, x: &[f64]) -> f64;

    /// Adds the gradient to `gradient`, and a positive semi-definite Hessian
    /// to `fixed` (into its own slots) and/or `changing`.
    fn derivatives(
        &mut self,
        ctx: &Context,
        x: &[f64],
        gradient: &mut [f64],
        fixed: &mut BlockMatrix,
        changing: &mut Triplets,
    );

    /// Whether this term has an active coupling to the fixed world. Used to
    /// decide whether the exact constant-translation coarse solve is valid.
    fn external_forces(&mut self, _ctx: &Context, _x: &[f64]) -> bool {
        false
    }

    /// The largest fraction of `dx` the step may take from `x`.
    fn step_limit(&mut self, _ctx: &Context, _x: &[f64], _dx: &[f64]) -> f64 {
        1.0
    }
}
