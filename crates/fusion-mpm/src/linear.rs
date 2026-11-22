//! Deterministic, matrix-free preconditioned conjugate gradients.
//!
//! Operators must be symmetric positive definite on the solved subspace.
//! A positive diagonal alone does not prove that condition. Callers own
//! boundary conditions and nullspace treatment. Uses f64, with Jacobi scaling
//! or a caller's preconditioner — [`IncompleteCholesky`] for an operator the
//! caller can also assemble.

/// Stopping and work limits, using the infinity norm of the true residual.
#[derive(Clone, Copy, Debug)]
pub struct SolveSettings {
    /// Relative tolerance against the right-hand side norm.
    pub relative_tolerance: f64,
    /// Absolute residual tolerance, in the right-hand side's units.
    pub absolute_tolerance: f64,
    /// Maximum matrix iterations.
    pub max_iterations: usize,
}

impl Default for SolveSettings {
    fn default() -> Self {
        Self {
            relative_tolerance: 1e-8,
            absolute_tolerance: 1e-12,
            max_iterations: 500,
        }
    }
}

/// Successful solve diagnostics; residual is recomputed as `b - A*x`.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct SolveReport {
    /// Iterations used; zero means the initial guess already met tolerance.
    pub iterations: usize,
    /// Infinity norm of the true residual.
    pub residual: f64,
}

/// An unsuccessful solve never overwrites the caller's solution.
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum SolveFailure {
    /// Dimensions, tolerances, diagonal, or starting vectors are invalid.
    InvalidInput,
    /// Non-positive curvature or an inadmissible preconditioned residual.
    NonPositiveCurvature,
    /// An operator application or scalar reduction was non-finite.
    NonFinite,
    /// Work was exhausted before meeting the true-residual tolerance.
    IterationLimit {
        /// Iterations attempted.
        iterations: usize,
        /// Final true residual norm.
        residual: f64,
    },
}

/// Reusable workspace; allocations grow only when the unknown count grows.
#[derive(Clone, Debug, Default)]
pub struct Pcg {
    candidate: Vec<f64>,
    residual: Vec<f64>,
    z: Vec<f64>,
    direction: Vec<f64>,
    product: Vec<f64>,
}

fn dot(a: &[f64], b: &[f64]) -> f64 {
    a.iter().zip(b).map(|(x, y)| x * y).sum()
}
fn norm(a: &[f64]) -> f64 {
    a.iter().fold(0.0_f64, |v, x| v.max(x.abs()))
}
fn finite(a: &[f64]) -> bool {
    a.iter().all(|x| x.is_finite())
}

impl Pcg {
    /// Solve `A*x=b` with a positive Jacobi diagonal. `apply` must overwrite
    /// every output entry, must be linear, and must not modify external state.
    /// The input solution is a warm start and is committed only on convergence.
    pub fn solve(
        &mut self,
        rhs: &[f64],
        diagonal: &[f64],
        solution: &mut [f64],
        settings: SolveSettings,
        apply: impl Fn(&[f64], &mut [f64]),
    ) -> Result<SolveReport, SolveFailure> {
        self.run(rhs, diagonal, solution, settings, false, apply)
    }

    /// As [`Pcg::solve`], except that running out of iterations commits the
    /// best iterate rather than nothing, and says so with
    /// [`SolveFailure::IterationLimit`]'s numbers in an `Ok`. For a solver
    /// with a real-time budget, where a pressure that is 99% converged is a
    /// better answer than none; the residual says how far it got.
    pub fn solve_within_budget(
        &mut self,
        rhs: &[f64],
        diagonal: &[f64],
        solution: &mut [f64],
        settings: SolveSettings,
        apply: impl Fn(&[f64], &mut [f64]),
    ) -> Result<SolveReport, SolveFailure> {
        self.run(rhs, diagonal, solution, settings, true, apply)
    }

    /// As [`Pcg::solve_within_budget`], preconditioned by `precondition`,
    /// which must apply a fixed symmetric positive definite approximation of
    /// the operator's inverse, overwriting every output entry.
    pub fn solve_preconditioned_within_budget(
        &mut self,
        rhs: &[f64],
        solution: &mut [f64],
        settings: SolveSettings,
        apply: impl Fn(&[f64], &mut [f64]),
        precondition: impl Fn(&[f64], &mut [f64]),
    ) -> Result<SolveReport, SolveFailure> {
        if solution.len() != rhs.len() {
            return Err(SolveFailure::InvalidInput);
        }
        self.iterate(rhs, solution, settings, true, apply, precondition)
    }

    fn run(
        &mut self,
        rhs: &[f64],
        diagonal: &[f64],
        solution: &mut [f64],
        settings: SolveSettings,
        commit_partial: bool,
        apply: impl Fn(&[f64], &mut [f64]),
    ) -> Result<SolveReport, SolveFailure> {
        if diagonal.len() != rhs.len() || diagonal.iter().any(|&d| !d.is_finite() || d <= 0.0) {
            return Err(SolveFailure::InvalidInput);
        }
        let jacobi = |r: &[f64], z: &mut [f64]| {
            for ((z, r), d) in z.iter_mut().zip(r).zip(diagonal) {
                *z = r / d;
            }
        };
        self.iterate(rhs, solution, settings, commit_partial, apply, jacobi)
    }

    fn iterate(
        &mut self,
        rhs: &[f64],
        solution: &mut [f64],
        settings: SolveSettings,
        commit_partial: bool,
        apply: impl Fn(&[f64], &mut [f64]),
        precondition: impl Fn(&[f64], &mut [f64]),
    ) -> Result<SolveReport, SolveFailure> {
        let n = rhs.len();
        if solution.len() != n
            || !finite(rhs)
            || !finite(solution)
            || !settings.relative_tolerance.is_finite()
            || settings.relative_tolerance < 0.0
            || !settings.absolute_tolerance.is_finite()
            || settings.absolute_tolerance < 0.0
        {
            return Err(SolveFailure::InvalidInput);
        }
        let tolerance = settings
            .absolute_tolerance
            .max(settings.relative_tolerance * norm(rhs));
        if !tolerance.is_finite() {
            return Err(SolveFailure::InvalidInput);
        }
        self.candidate.resize(n, 0.0);
        self.candidate.copy_from_slice(solution);
        self.residual.resize(n, 0.0);
        self.z.resize(n, 0.0);
        self.direction.resize(n, 0.0);
        self.product.resize(n, 0.0);
        apply(&self.candidate, &mut self.product);
        if !finite(&self.product) {
            return Err(SolveFailure::NonFinite);
        }
        for (i, &value) in rhs.iter().enumerate() {
            self.residual[i] = value - self.product[i];
        }
        if !finite(&self.residual) {
            return Err(SolveFailure::NonFinite);
        }
        let initial = norm(&self.residual);
        if initial <= tolerance {
            return Ok(SolveReport {
                iterations: 0,
                residual: initial,
            });
        }
        precondition(&self.residual, &mut self.z);
        self.direction.copy_from_slice(&self.z);
        let mut rz = dot(&self.residual, &self.z);
        for iteration in 1..=settings.max_iterations {
            apply(&self.direction, &mut self.product);
            let curvature = dot(&self.direction, &self.product);
            if !curvature.is_finite() || !rz.is_finite() || !finite(&self.product) {
                return Err(SolveFailure::NonFinite);
            }
            if curvature <= 0.0 || rz <= 0.0 {
                return Err(SolveFailure::NonPositiveCurvature);
            }
            let alpha = rz / curvature;
            for i in 0..n {
                self.candidate[i] += alpha * self.direction[i];
                self.residual[i] -= alpha * self.product[i];
            }
            if !finite(&self.candidate) || !finite(&self.residual) {
                return Err(SolveFailure::NonFinite);
            }
            // Recheck before claiming success; recurrence residuals can drift.
            // Periodic replacement restarts the search in the true residual.
            let restart = norm(&self.residual) <= tolerance || iteration % 32 == 0;
            if restart {
                apply(&self.candidate, &mut self.product);
                for (i, &value) in rhs.iter().enumerate() {
                    self.residual[i] = value - self.product[i];
                }
                if !finite(&self.residual) {
                    return Err(SolveFailure::NonFinite);
                }
                let residual = norm(&self.residual);
                if residual <= tolerance {
                    solution.copy_from_slice(&self.candidate);
                    return Ok(SolveReport {
                        iterations: iteration,
                        residual,
                    });
                }
            }
            precondition(&self.residual, &mut self.z);
            let next = dot(&self.residual, &self.z);
            let beta = if restart { 0.0 } else { next / rz };
            for i in 0..n {
                self.direction[i] = self.z[i] + beta * self.direction[i];
            }
            rz = next;
        }
        apply(&self.candidate, &mut self.product);
        for (i, &value) in rhs.iter().enumerate() {
            self.residual[i] = value - self.product[i];
        }
        if !finite(&self.residual) {
            return Err(SolveFailure::NonFinite);
        }
        let residual = norm(&self.residual);
        if commit_partial {
            solution.copy_from_slice(&self.candidate);
            return Ok(SolveReport {
                iterations: settings.max_iterations,
                residual,
            });
        }
        Err(SolveFailure::IterationLimit {
            iterations: settings.max_iterations,
            residual,
        })
    }
}

/// A symmetric matrix in compressed rows, each row's columns ascending and
/// its diagonal among them.
#[derive(Clone, Debug, Default)]
pub struct SparseSymmetric {
    /// Where each row starts in `columns` and `values`; one more than rows.
    pub starts: Vec<usize>,
    /// Column of each entry.
    pub columns: Vec<u32>,
    /// Value of each entry.
    pub values: Vec<f64>,
}

impl SparseSymmetric {
    /// Rows.
    pub fn len(&self) -> usize {
        self.starts.len().saturating_sub(1)
    }

    /// Whether it has no rows.
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// `y = A x`.
    pub fn apply(&self, x: &[f64], y: &mut [f64]) {
        for (row, out) in y.iter_mut().enumerate() {
            let range = self.starts[row]..self.starts[row + 1];
            *out = self.columns[range.clone()]
                .iter()
                .zip(&self.values[range])
                .map(|(&c, &v)| v * x[c as usize])
                .sum();
        }
    }
}

/// Incomplete Cholesky with no fill, `A ≈ L Lᵀ` on `A`'s own pattern: the
/// classic preconditioner for a pressure Poisson matrix, several times fewer
/// iterations than Jacobi for one forward and one backward sweep. A pivot
/// that breaks down — the factor's diagonal going non-positive, which an
/// incomplete factor of an SPD matrix can — falls back to that row's own
/// diagonal, so the result is always SPD.
#[derive(Clone, Debug, Default)]
pub struct IncompleteCholesky {
    starts: Vec<usize>,
    columns: Vec<u32>,
    /// Strictly lower entries of `L`, by row.
    lower: Vec<f64>,
    /// `L`'s diagonal.
    pivot: Vec<f64>,
    scratch: std::cell::RefCell<Vec<f64>>,
}

impl IncompleteCholesky {
    /// Factor `a`, reusing this one's storage.
    pub fn factor(&mut self, a: &SparseSymmetric) {
        let n = a.len();
        self.starts.clear();
        self.columns.clear();
        self.lower.clear();
        self.pivot.clear();
        self.starts.push(0);
        let mut diagonal = Vec::with_capacity(n);
        for row in 0..n {
            let mut d = 0.0;
            for at in a.starts[row]..a.starts[row + 1] {
                let column = a.columns[at] as usize;
                if column < row {
                    self.columns.push(column as u32);
                    self.lower.push(a.values[at]);
                } else if column == row {
                    d = a.values[at];
                }
            }
            diagonal.push(d);
            self.starts.push(self.columns.len());
        }
        for (row, &own) in diagonal.iter().enumerate() {
            let (start, end) = (self.starts[row], self.starts[row + 1]);
            for at in start..end {
                let j = self.columns[at] as usize;
                // Σ_k<j L_ik L_jk, over the two rows' shared columns.
                let (mut p, mut q) = (start, self.starts[j]);
                let q_end = self.starts[j + 1];
                let mut sum = 0.0;
                while p < at && q < q_end {
                    let (cp, cq) = (self.columns[p], self.columns[q]);
                    if cp == cq {
                        sum += self.lower[p] * self.lower[q];
                        p += 1;
                        q += 1;
                    } else if cp < cq {
                        p += 1;
                    } else {
                        q += 1;
                    }
                }
                self.lower[at] = (self.lower[at] - sum) / self.pivot[j];
            }
            let squares: f64 = self.lower[start..end].iter().map(|l| l * l).sum();
            let d = own - squares;
            let pivot = if d > 1e-6 * own && d.is_finite() {
                d.sqrt()
            } else {
                // Broken down: this row as Jacobi would see it, and none of
                // its coupling.
                for l in &mut self.lower[start..end] {
                    *l = 0.0;
                }
                own.max(f64::MIN_POSITIVE).sqrt()
            };
            self.pivot.push(pivot);
        }
    }

    /// `z = (L Lᵀ)⁻¹ r`.
    pub fn apply(&self, r: &[f64], z: &mut [f64]) {
        let n = self.pivot.len();
        let mut y = self.scratch.borrow_mut();
        y.clear();
        y.extend_from_slice(&r[..n]);
        for row in 0..n {
            let mut v = y[row];
            for at in self.starts[row]..self.starts[row + 1] {
                v -= self.lower[at] * y[self.columns[at] as usize];
            }
            y[row] = v / self.pivot[row];
        }
        for row in (0..n).rev() {
            let v = y[row] / self.pivot[row];
            z[row] = v;
            for at in self.starts[row]..self.starts[row + 1] {
                y[self.columns[at] as usize] -= self.lower[at] * v;
            }
        }
    }
}
