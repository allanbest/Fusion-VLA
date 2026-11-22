//! Elastic laws: energy per unit rest volume as a function of the
//! deformation gradient `F`, its stress, and a positive semi-definite
//! second derivative for Newton's method.

use core_math::{Mat3, Vec3};

/// Compressible neo-Hookean: `Ψ = μ/2 (tr FᵀF − 3) − μ ln J + λ/2 (ln J)²`.
/// Exact for large rotations, rubber-like at large stretch, Hooke's law at
/// small strain with the same `μ` and `λ`. Infinite for an inverted
/// element, which the line search then refuses to step into.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct NeoHookean {
    /// Shear modulus, Pa.
    pub mu: f64,
    /// Lamé's first parameter, Pa.
    pub lambda: f64,
}

/// What one quadrature point remembers between evaluations of its
/// eigensystem: the last right singular vectors and scaling-mode basis.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Warm {
    pub v: Mat3,
    pub scaling: Mat3,
}

impl Default for Warm {
    fn default() -> Self {
        Self {
            v: Mat3::IDENTITY,
            scaling: Mat3::IDENTITY,
        }
    }
}

/// The second derivative `∂²Ψ/∂F²` as nine eigenpairs: `Σ λₖ Qₖ ⊗ Qₖ`, with
/// orthonormal `Qₖ`. Negative `λₖ` are kept, so a caller can clamp them.
#[derive(Clone, Copy, Debug)]
pub struct Modes {
    pub values: [f64; 9],
    pub matrices: [Mat3; 9],
}

impl NeoHookean {
    /// From Young's modulus and Poisson's ratio.
    pub fn from_youngs(youngs: f64, poisson: f64) -> Self {
        Self {
            mu: youngs / (2.0 * (1.0 + poisson)),
            lambda: youngs * poisson / ((1.0 + poisson) * (1.0 - 2.0 * poisson)),
        }
    }

    /// J/m³.
    pub fn energy(&self, f: &Mat3) -> f64 {
        let j = f.determinant();
        if j <= 0.0 {
            return f64::INFINITY;
        }
        // Near any rotation, the usual O(strain) terms cancel to leave an
        // O(strain^2) energy. Their rounding noise can defeat a line search.
        // Write C = F^T F = I + A and det(C) = 1 + tr(A) + I2(A) + det(A).
        // Then tr(A)-ln(det(C)) = [t-ln(1+t)] - I2(A) - det(A).
        // This is the same constitutive law, with no change to stiffness.
        let a = f.transpose() * *f - Mat3::IDENTITY;
        if a.frobenius_squared() < 0.01 {
            let trace = a[(0, 0)] + a[(1, 1)] + a[(2, 2)];
            let second = a[(0, 0)] * a[(1, 1)] + a[(0, 0)] * a[(2, 2)] + a[(1, 1)] * a[(2, 2)]
                - a[(0, 1)] * a[(1, 0)]
                - a[(0, 2)] * a[(2, 0)]
                - a[(1, 2)] * a[(2, 1)];
            let third = a.determinant();
            let t = trace + second + third;
            let log_c = t.ln_1p();
            let remainder = if t.abs() < 0.01 {
                // Horner series for t-ln(1+t), through t^12. The omitted
                // term is below rounding relative to t^2 in this interval.
                let mut series = 0.0;
                for n in (2..=12).rev() {
                    let coefficient = if n % 2 == 0 { 1.0 } else { -1.0 } / n as f64;
                    series = coefficient + t * series;
                }
                t * t * series
            } else {
                t - log_c
            };
            return 0.5 * self.mu * (remainder - second - third)
                + 0.125 * self.lambda * log_c * log_c;
        }
        let log_j = j.ln();
        0.5 * self.mu * (f.frobenius_squared() - 3.0) - self.mu * log_j
            + 0.5 * self.lambda * log_j * log_j
    }

    /// First Piola–Kirchhoff stress `P = μ (F − F⁻ᵀ) + λ ln J F⁻ᵀ`, Pa.
    pub fn stress(&self, f: &Mat3) -> Mat3 {
        self.stress_and_factors(f).0
    }

    /// Stress together with `(J, ln J, F⁻ᵀ)` so callers avoid recomputing
    /// `determinant()`, `adjugate()` and `ln()`.
    pub fn stress_and_factors(&self, f: &Mat3) -> (Mat3, f64, f64, Mat3) {
        let j = f.determinant();
        let log_j = j.ln();
        let inverse_t = f.adjugate().transpose() * (1.0 / j);
        let stress = (*f - inverse_t) * self.mu + inverse_t * (self.lambda * log_j);
        (stress, j, log_j, inverse_t)
    }

    /// Emits only the strictly negative eigenvalues and orthonormal eigenmatrices
    /// of `∂²Ψ/∂F²` at `F` (with `J > 0`), skipping 3×3 scaling/flip eigen-work
    /// whenever `μ − λ ln J ≥ 0` proves them strictly positive.
    pub fn for_each_negative_mode(
        &self,
        f: &Mat3,
        warm: &mut Warm,
        mut emit: impl FnMut(f64, Mat3),
    ) {
        let (u, s, v) = f.svd_fast(&warm.v);
        warm.v = v;
        let log_j = (s.x * s.y * s.z).ln();
        let a = self.mu - self.lambda * log_j;
        let sigma = [s.x, s.y, s.z];
        let r = std::f64::consts::FRAC_1_SQRT_2;
        if a >= 0.0 {
            // Scaling (μ + a/σᵢ² + λ...) and flip (μ + a/(σᵢσⱼ)) are all > 0.
            for (i, j) in [(0, 1), (0, 2), (1, 2)] {
                let value = self.mu - a / (sigma[i] * sigma[j]);
                if value < 0.0 {
                    let q = (Mat3::outer(u.col(i), v.col(j)) - Mat3::outer(u.col(j), v.col(i))) * r;
                    emit(value, q);
                }
            }
            return;
        }
        // a < 0: twist modes μ - a/(σᵢσⱼ) are all > 0.
        let mut scaling = Mat3::ZERO;
        for i in 0..3 {
            for j in 0..3 {
                scaling[(i, j)] = self.lambda / (sigma[i] * sigma[j])
                    + if i == j {
                        self.mu + a / (sigma[i] * sigma[i])
                    } else {
                        0.0
                    };
            }
        }
        let (scale_values, scale_vectors) = scaling.symmetric_eigen_fast_or(&warm.scaling);
        warm.scaling = scale_vectors;
        for k in 0..3 {
            if scale_values[k] < 0.0 {
                let q = u * Mat3::from_diagonal(scale_vectors.col(k)) * v.transpose();
                emit(scale_values[k], q);
            }
        }
        for (i, j) in [(0, 1), (0, 2), (1, 2)] {
            let value = self.mu + a / (sigma[i] * sigma[j]);
            if value < 0.0 {
                let q = (Mat3::outer(u.col(i), v.col(j)) + Mat3::outer(u.col(j), v.col(i))) * r;
                emit(value, q);
            }
        }
    }

    /// Multiplicative Hencky J2 radial return with linear isotropic hardening:
    /// given trial elastic deformation gradient `F_e^tr = F F_p^-1` and prior
    /// equivalent plastic strain `eps_p`, returns `Some((F_e^new, ΔF_p^-1, eps_p^new))`
    /// if the Mises trial stress exceeds `yield_stress + hardening * eps_p`,
    /// preserving `det(F_e)` and decreasing stored elastic strain energy.
    pub fn hencky_j2_return(
        &self,
        fe_trial: &Mat3,
        eps_p: f64,
        yield_stress: f64,
        hardening: f64,
        warm: &mut Warm,
    ) -> Option<(Mat3, Mat3, f64)> {
        let (u, s, v) = fe_trial.svd_fast(&warm.v);
        warm.v = v;
        if s.x <= 0.0 || s.y <= 0.0 || s.z <= 0.0 {
            return None;
        }
        let eps = Vec3::new(s.x.ln(), s.y.ln(), s.z.ln());
        let mean = (eps.x + eps.y + eps.z) / 3.0;
        let dev = eps - Vec3::splat(mean);
        let q_trial = 6.0_f64.sqrt() * self.mu * dev.length();
        let sigma_y = yield_stress + hardening * eps_p;
        let denom = 3.0 * self.mu + hardening;
        if q_trial <= sigma_y * (1.0 + 1e-12) || q_trial <= 0.0 || denom <= 0.0 {
            return None;
        }
        let dgamma = (q_trial - sigma_y) / denom;
        let scale = (sigma_y + hardening * dgamma) / q_trial;
        let s_new = Vec3::new(
            (mean + dev.x * scale).exp(),
            (mean + dev.y * scale).exp(),
            (mean + dev.z * scale).exp(),
        );
        let ratio = Vec3::new(s_new.x / s.x, s_new.y / s.y, s_new.z / s.z);
        let delta_fp_inv = v * Mat3::from_diagonal(ratio) * v.transpose();
        let fe_new = u * Mat3::from_diagonal(s_new) * v.transpose();
        Some((fe_new, delta_fp_inv, eps_p + dgamma))
    }

    /// The eigensystem of `∂²Ψ/∂F²` at `F` (with `J > 0`), in closed form
    /// from `F`'s singular values (Smith et al. 2019): three scaling modes
    /// from a 3×3 problem, and for each pair of axes a twist and a flip
    /// whose values need no division by a difference of stretches — exact
    /// when stretches repeat, as they do at rest.
    pub fn modes(&self, f: &Mat3) -> Modes {
        self.modes_from(f, &mut Warm::default())
    }

    /// As [`NeoHookean::modes`], warm-started from what the same point
    /// found last time (and updating it): the same answer to rounding, in a
    /// fraction of the eigen-sweeps when the deformation has changed little.
    pub fn modes_from(&self, f: &Mat3, warm: &mut Warm) -> Modes {
        let (u, s, v) = f.svd_from(&warm.v);
        warm.v = v;
        let log_j = (s.x * s.y * s.z).ln();
        let a = self.mu - self.lambda * log_j; // μ − λ ln J
        let sigma = [s.x, s.y, s.z];
        let mut values = [0.0; 9];
        let mut matrices = [Mat3::ZERO; 9];

        // Scaling: ∂²Ψ/∂σᵢ∂σⱼ = δᵢⱼ (μ + a/σᵢ²) + λ/(σᵢσⱼ).
        let mut scaling = Mat3::ZERO;
        for i in 0..3 {
            for j in 0..3 {
                scaling[(i, j)] = self.lambda / (sigma[i] * sigma[j])
                    + if i == j {
                        self.mu + a / (sigma[i] * sigma[i])
                    } else {
                        0.0
                    };
            }
        }
        let (scale_values, scale_vectors) = scaling.symmetric_eigen_from(&warm.scaling);
        warm.scaling = scale_vectors;
        for k in 0..3 {
            values[k] = scale_values[k];
            matrices[k] = u * Mat3::from_diagonal(scale_vectors.col(k)) * v.transpose();
        }

        // Twist (antisymmetric) and flip (symmetric) for each pair of axes.
        let r = std::f64::consts::FRAC_1_SQRT_2;
        for (n, (i, j)) in [(0, 1), (0, 2), (1, 2)].into_iter().enumerate() {
            let product = sigma[i] * sigma[j];
            let (ei, ej) = (unit(i), unit(j));
            let twist = (Mat3::outer(ei, ej) - Mat3::outer(ej, ei)) * r;
            let flip = (Mat3::outer(ei, ej) + Mat3::outer(ej, ei)) * r;
            values[3 + 2 * n] = self.mu - a / product;
            matrices[3 + 2 * n] = u * twist * v.transpose();
            values[4 + 2 * n] = self.mu + a / product;
            matrices[4 + 2 * n] = u * flip * v.transpose();
        }
        Modes { values, matrices }
    }
}

fn unit(i: usize) -> Vec3 {
    let mut e = Vec3::ZERO;
    e[i] = 1.0;
    e
}
