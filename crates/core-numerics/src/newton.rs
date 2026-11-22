//! Minimising an energy by Newton's method with a line search — the shape
//! of every implicit mechanics step: find the positions that minimise the
//! step's incremental potential.

use core_runtime::Pool;

use crate::cg::{
    conjugate_gradient, BlockJacobi, CgReport, CgSettings, CgWorkspace, Preconditioner,
};
use crate::direct::SparseCholesky;
use crate::{vector, BlockOperator};

/// A smooth energy over a vector of `3 n` numbers (n points in space).
pub trait Objective {
    /// The Hessian approximation, as the objective stores it.
    type Hessian: BlockOperator;

    /// Number of 3-blocks.
    fn blocks(&self) -> usize;

    /// The energy at `x`; `+∞` where `x` is not allowed (an inverted
    /// element, an interpenetration), which the line search then avoids.
    fn energy(&mut self, pool: &Pool, x: &[f64]) -> f64;

    /// Adds the gradient at `x` to `gradient` (zeroed by the caller), and
    /// brings [`Objective::hessian`] to a positive semi-definite
    /// approximation of the Hessian at `x`.
    fn derivatives(&mut self, pool: &Pool, x: &[f64], gradient: &mut [f64]);

    /// The Hessian from the last [`Objective::derivatives`].
    fn hessian(&self) -> &Self::Hessian;

    /// Groups of consecutive blocks `(first, count)` to precondition
    /// together (a rigid body's unknowns).
    fn coupled(&self) -> &[(usize, usize)] {
        &[]
    }

    /// Optional exact coarse-mode solve on an approximate Newton direction.
    /// The objective must preserve its actual equations and the line search
    /// still checks the resulting direction against the original potential.
    fn polish_step(&mut self, _pool: &Pool, _x: &[f64], _gradient: &[f64], _step: &mut [f64]) {}

    /// The largest fraction of the step `dx` from `x` that is safe to try,
    /// in `(0, 1]` — a barrier that must not be crossed, a free-flight bound.
    fn step_limit(&mut self, _pool: &Pool, _x: &[f64], _dx: &[f64]) -> f64 {
        1.0
    }
}

/// When to stop.
#[derive(Clone, Copy, Debug)]
pub struct NewtonSettings {
    /// Converged when the largest component of a Newton step is below this,
    /// in the units of `x`.
    pub tolerance: f64,
    pub max_iterations: usize,
    /// The linear solve; its tolerance is tightened as Newton converges.
    pub linear: CgSettings,
    /// Use the sparse direct `LDLᵀ` solver whenever `blocks <= direct_blocks`
    /// (`0` to always try preconditioned CG first, keeping `LDLᵀ` as a fallback
    /// for up to 256 blocks if CG fails to produce a descent direction).
    pub direct_blocks: usize,
}

impl Default for NewtonSettings {
    fn default() -> Self {
        Self {
            tolerance: 1e-8,
            max_iterations: 50,
            linear: CgSettings {
                tolerance: 1e-3,
                max_iterations: 1000,
            },
            direct_blocks: 0,
        }
    }
}

/// What a minimisation did.
#[derive(Clone, Copy, Debug, Default)]
pub struct NewtonReport {
    pub iterations: usize,
    pub converged: bool,
    /// The energy at the end.
    pub energy: f64,
    /// Linear-solver iterations, summed.
    pub linear_iterations: usize,
    /// All matrix-vector products, including equation-residual verification.
    pub linear_products: usize,
    /// The largest component of the last Newton step.
    pub last_step: f64,
    /// A line search found no decrease; stopped where it was.
    pub stalled: bool,
    /// The gradient or the step was not finite; stopped where it was.
    pub not_finite: bool,
}

/// Newton's method with reusable memory.
#[derive(Default)]
pub struct Newton {
    pub settings: NewtonSettings,
    gradient: Vec<f64>,
    rhs: Vec<f64>,
    step: Vec<f64>,
    trial: Vec<f64>,
    preconditioner: BlockJacobi,
    cg: CgWorkspace,
    direct: SparseCholesky,
}

impl Newton {
    pub fn new(settings: NewtonSettings) -> Self {
        Self {
            settings,
            ..Default::default()
        }
    }

    /// Moves `x` to a minimiser of `objective`, starting from `x`, which
    /// must have finite energy.
    pub fn minimize(
        &mut self,
        pool: &Pool,
        objective: &mut impl Objective,
        x: &mut [f64],
    ) -> NewtonReport {
        let n = x.len();
        debug_assert_eq!(n, 3 * objective.blocks());
        for v in [
            &mut self.gradient,
            &mut self.rhs,
            &mut self.step,
            &mut self.trial,
        ] {
            v.clear();
            v.resize(n, 0.0);
        }
        let mut report = NewtonReport {
            energy: objective.energy(pool, x),
            ..Default::default()
        };
        assert!(
            report.energy.is_finite(),
            "Newton started where the energy is not finite"
        );
        let mut first_gradient = 0.0;

        for iteration in 0..self.settings.max_iterations {
            self.gradient.fill(0.0);
            objective.derivatives(pool, x, &mut self.gradient);
            let g_norm = vector::norm_squared(pool, &self.gradient).sqrt();
            if !g_norm.is_finite() {
                report.not_finite = true;
                return report;
            }
            if iteration == 0 {
                first_gradient = g_norm.max(f64::MIN_POSITIVE);
            }

            // Solve H dx = −g, to a tolerance that tightens as |g| falls
            // (Eisenstat–Walker style: no point solving a far-off linear
            // model exactly). Relative, so independent of units.
            for (r, g) in self.rhs.iter_mut().zip(&self.gradient) {
                *r = -g;
            }
            self.step.fill(0.0);
            let blocks = objective.blocks();
            let mut cg = CgReport::default();
            let mut direct_solved = false;
            if blocks > 0 && blocks <= self.settings.direct_blocks {
                let hessian = objective.hessian();
                if self.direct.factor(hessian) && self.direct.solve(&self.rhs, &mut self.step) {
                    direct_solved = true;
                    objective.polish_step(pool, x, &self.gradient, &mut self.step);
                }
            }
            if !direct_solved {
                let linear = CgSettings {
                    tolerance: self
                        .settings
                        .linear
                        .tolerance
                        .min((g_norm / first_gradient).sqrt())
                        .max(1e-12),
                    ..self.settings.linear
                };
                let hessian = objective.hessian();
                self.preconditioner
                    .update(pool, hessian, objective.coupled());
                cg = conjugate_gradient(
                    pool,
                    hessian,
                    &self.preconditioner,
                    &self.rhs,
                    &mut self.step,
                    linear,
                    &mut self.cg,
                );
                report.linear_iterations += cg.iterations;
                report.linear_products += cg.matrix_products;
                objective.polish_step(pool, x, &self.gradient, &mut self.step);
            }

            let mut slope = vector::dot(pool, &self.step, &self.gradient);
            let mut fallback = false;
            if !direct_solved
                && g_norm > 0.0
                && (!cg.converged || cg.not_finite || slope.is_nan() || slope >= 0.0)
                && blocks <= 256
            {
                let hessian = objective.hessian();
                if self.direct.factor(hessian) && self.direct.solve(&self.rhs, &mut self.step) {
                    objective.polish_step(pool, x, &self.gradient, &mut self.step);
                    slope = vector::dot(pool, &self.step, &self.gradient);
                    direct_solved = slope.is_finite() && slope < 0.0;
                }
            }
            if !direct_solved && g_norm > 0.0 && (cg.not_finite || slope.is_nan() || slope >= 0.0) {
                // Not a descent direction (a bad Hessian approximation):
                // fall back to preconditioned steepest descent in meters, and
                // never declare convergence without taking a step.
                pool.split_reduce(
                    (&self.rhs[..], &mut self.step[..]),
                    vector::CHUNK,
                    (),
                    |offset, (r, z)| self.preconditioner.apply_part(offset, r, z),
                    |_, _| (),
                );
                slope = vector::dot(pool, &self.step, &self.gradient);
                if slope.is_nan() || slope >= 0.0 {
                    self.step.copy_from_slice(&self.rhs);
                    slope = -g_norm * g_norm;
                }
                fallback = true;
            }
            report.last_step = vector::norm_max(pool, &self.step);
            report.iterations = iteration + 1;
            if report.last_step.is_nan() {
                report.not_finite = true;
                return report;
            }
            if !fallback && report.last_step <= self.settings.tolerance {
                report.converged = true;
                return report;
            }

            // Backtracking line search with a sufficient-decrease condition.
            let limit = objective.step_limit(pool, x, &self.step).min(1.0);
            let mut alpha = limit;
            let mut accepted = false;
            for _ in 0..40 {
                if alpha * report.last_step == 0.0 {
                    break; // no step at all is not progress
                }
                for ((t, xi), di) in self.trial.iter_mut().zip(x.iter()).zip(&self.step) {
                    *t = xi + alpha * di;
                }
                let energy = objective.energy(pool, &self.trial);
                let decreased = energy <= report.energy + 1e-4 * alpha * slope;
                // Near convergence, rounding hides the decrease: accept a
                // step too small to matter if it does not increase the energy.
                let negligible =
                    energy <= report.energy && alpha * report.last_step <= self.settings.tolerance;
                if decreased || negligible {
                    x.copy_from_slice(&self.trial);
                    report.energy = energy;
                    accepted = true;
                    break;
                }
                // Halving: measured better than quadratic interpolation on
                // friction-heavy scenes, where the interpolant cuts steps
                // short and costs Newton iterations.
                alpha *= 0.5;
            }
            if trace() {
                eprintln!(
                    "newton {iteration:>3}: E {:.9e}  |g| {g_norm:.3e}  |dx| {:.3e}  limit {limit:.3e}  α {alpha:.3e}  cg {}{}",
                    report.energy,
                    report.last_step,
                    cg.iterations,
                    if accepted { "" } else { "  no decrease" }
                );
            }
            if !accepted {
                report.stalled = true;
                return report;
            }
        }
        report
    }
}

/// Whether to print every iteration to stderr: set `FUSION_TRACE_NEWTON`.
fn trace() -> bool {
    static TRACE: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *TRACE.get_or_init(|| std::env::var_os("FUSION_TRACE_NEWTON").is_some())
}
