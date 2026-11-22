//! Convergence studies: refine the timestep, confirm the answer settles.
//!
//! §23: "refine Δt or resolution, confirm the result converges as theory
//! predicts". The second half is the part that matters. A simulation that
//! gives a different answer at every timestep is not converging; a simulation
//! that gives *nearly* the same answer is not evidence of anything until you
//! know how fast the difference shrinks.
//!
//! ## What is being measured
//!
//! Halve the timestep and the error against the true answer should fall by
//! `2^p`, where `p` is the order of the method. Semi-implicit Euler is order
//! one, so halving Δt should halve the error; a second-order method would
//! quarter it. The observed order comes out of three refinements:
//!
//! ```text
//! p = log2( |q(h) - q(h/2)| / |q(h/2) - q(h/4)| )
//! ```
//!
//! and needs no knowledge of the true answer, which is the point — the true
//! answer is exactly what is unavailable in the cases worth studying.
//!
//! ## What it will say about XPBD
//!
//! Something disappointing, and honestly. XPBD declares the real-time accuracy
//! tier (§9): stable and plausible, not convergent in the numerical-analysis
//! sense. Contacts are solved by iteration to a tolerance that does not shrink
//! with the timestep, so a contact-heavy scene will show an order well under
//! one. That is not a bug to fix, it is the tier doing what it says, and a
//! convergence study is how you tell the difference between a solver that is
//! wrong and a solver that is honest about being approximate.

use serde::{Deserialize, Serialize};

/// One refinement level and what it produced.
#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
pub struct Level {
    /// The timestep, or whatever was refined.
    pub h: f64,
    /// The measured quantity at that refinement.
    pub value: f64,
}

/// What a convergence study concluded.
#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
pub struct ConvergenceReport {
    /// The refinements, coarsest first.
    pub levels: Vec<Level>,
    /// Observed order between each consecutive triple, coarsest first.
    ///
    /// One number per triple rather than one for the study: an order that
    /// changes as you refine is itself the finding, usually meaning the
    /// coarsest levels were nowhere near the asymptotic regime.
    pub observed_order: Vec<f64>,
    /// The Richardson extrapolation of the finest triple — an estimate of the
    /// answer at zero timestep. `None` where the orders are unusable.
    pub extrapolated: Option<f64>,
}

impl ConvergenceReport {
    /// The order from the finest triple, which is the one to quote.
    pub fn order(&self) -> Option<f64> {
        self.observed_order.last().copied()
    }

    /// True where the observed order is within `tolerance` of `expected`.
    ///
    /// The assertion a convergence test makes. Deliberately not a default:
    /// what counts as converging depends on the method being studied, and a
    /// tolerance chosen after seeing the number is not a test.
    pub fn converges_at(&self, expected: f64, tolerance: f64) -> bool {
        self.order()
            .is_some_and(|p| (p - expected).abs() <= tolerance)
    }
}

/// Refine `h` and watch a measurement settle.
///
/// `levels` is how many refinements to run, each halving `h`. `measure` is
/// given a timestep and returns the quantity of interest.
///
/// Three levels is the minimum that says anything — two points give a
/// difference, three give a *ratio* of differences, and only the ratio is an
/// order.
pub fn study(
    coarsest: f64,
    levels: usize,
    mut measure: impl FnMut(f64) -> f64,
) -> ConvergenceReport {
    let mut report = ConvergenceReport::default();
    let mut h = coarsest;
    for _ in 0..levels.max(1) {
        report.levels.push(Level {
            h,
            value: measure(h),
        });
        h *= 0.5;
    }

    for window in report.levels.windows(3) {
        let (coarse, mid, fine) = (window[0].value, window[1].value, window[2].value);
        let first = (coarse - mid).abs();
        let second = (mid - fine).abs();
        // A ratio needs both differences to be real. Two identical values mean
        // either perfect convergence or a measurement that does not depend on
        // the timestep at all, and reporting an infinite order for the second
        // would be worse than reporting nothing.
        if second > 1e-12 && first > 1e-12 {
            report.observed_order.push((first / second).log2());
        }
    }

    if let (Some(&p), Some(window)) = (
        report.observed_order.last(),
        report.levels.windows(2).last(),
    ) {
        // Richardson: the finer value plus its share of the remaining error.
        if p.is_finite() && p > 0.05 {
            let (coarse, fine) = (window[0].value, window[1].value);
            report.extrapolated = Some(fine + (fine - coarse) / (2f64.powf(p) - 1.0));
        }
    }
    report
}
