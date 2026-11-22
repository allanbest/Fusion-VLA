use std::ops::{Add, AddAssign, Index, IndexMut, Mul, MulAssign, Neg, Sub, SubAssign};

use crate::Vec3;

/// A 3×3 matrix, stored by columns.
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct Mat3 {
    pub cols: [Vec3; 3],
}

impl Mat3 {
    pub const ZERO: Self = Self::from_cols(Vec3::ZERO, Vec3::ZERO, Vec3::ZERO);
    pub const IDENTITY: Self = Self::from_diagonal(Vec3::ONE);

    #[inline]
    pub const fn from_cols(a: Vec3, b: Vec3, c: Vec3) -> Self {
        Self { cols: [a, b, c] }
    }

    #[inline]
    pub fn from_rows(a: Vec3, b: Vec3, c: Vec3) -> Self {
        Self::from_cols(a, b, c).transpose()
    }

    #[inline]
    pub const fn from_diagonal(d: Vec3) -> Self {
        Self::from_cols(
            Vec3::new(d.x, 0.0, 0.0),
            Vec3::new(0.0, d.y, 0.0),
            Vec3::new(0.0, 0.0, d.z),
        )
    }

    /// `s` times the identity.
    #[inline]
    pub const fn scalar(s: f64) -> Self {
        Self::from_diagonal(Vec3::splat(s))
    }

    /// `a bᵀ`.
    #[inline]
    pub fn outer(a: Vec3, b: Vec3) -> Self {
        Self::from_cols(a * b.x, a * b.y, a * b.z)
    }

    /// The matrix `[v]×` with `[v]× u = v × u`.
    #[inline]
    pub fn skew(v: Vec3) -> Self {
        Self::from_cols(
            Vec3::new(0.0, v.z, -v.y),
            Vec3::new(-v.z, 0.0, v.x),
            Vec3::new(v.y, -v.x, 0.0),
        )
    }

    #[inline]
    pub fn col(&self, j: usize) -> Vec3 {
        self.cols[j]
    }

    #[inline]
    pub fn row(&self, i: usize) -> Vec3 {
        Vec3::new(self.cols[0][i], self.cols[1][i], self.cols[2][i])
    }

    #[inline]
    pub fn diagonal(&self) -> Vec3 {
        Vec3::new(self.cols[0].x, self.cols[1].y, self.cols[2].z)
    }

    #[inline]
    pub fn transpose(&self) -> Self {
        Self::from_cols(self.row(0), self.row(1), self.row(2))
    }

    #[inline]
    pub fn trace(&self) -> f64 {
        self.cols[0].x + self.cols[1].y + self.cols[2].z
    }

    #[inline]
    pub fn determinant(&self) -> f64 {
        self.cols[0].dot(self.cols[1].cross(self.cols[2]))
    }

    /// The transpose of the cofactor matrix: `det(A) A⁻¹`, defined for every
    /// matrix.
    #[inline]
    pub fn adjugate(&self) -> Self {
        let [a, b, c] = self.cols;
        Self::from_rows(b.cross(c), c.cross(a), a.cross(b))
    }

    /// The inverse; not finite for a singular matrix.
    #[inline]
    pub fn inverse(&self) -> Self {
        self.adjugate() * (1.0 / self.determinant())
    }

    /// The inverse, or `None` for a matrix too close to singular to invert
    /// meaningfully.
    pub fn try_inverse(&self) -> Option<Self> {
        let det = self.determinant();
        let scale = self.frobenius_squared().powf(1.5);
        (det.abs() > 1e-14 * scale && det.is_finite()).then(|| self.adjugate() * (1.0 / det))
    }

    /// `Σ aᵢⱼ bᵢⱼ`.
    #[inline]
    pub fn double_dot(&self, o: &Self) -> f64 {
        self.cols[0].dot(o.cols[0]) + self.cols[1].dot(o.cols[1]) + self.cols[2].dot(o.cols[2])
    }

    #[inline]
    pub fn frobenius_squared(&self) -> f64 {
        self.double_dot(self)
    }

    /// `(A + Aᵀ) / 2`.
    #[inline]
    pub fn symmetric_part(&self) -> Self {
        (*self + self.transpose()) * 0.5
    }

    pub fn is_finite(&self) -> bool {
        self.cols.iter().all(|c| c.is_finite())
    }

    /// Eigenvalues and orthonormal eigenvectors (as columns, in the same
    /// order) of a symmetric matrix, by cyclic Jacobi rotations — accurate to
    /// rounding for any symmetric input, including repeated eigenvalues.
    pub fn symmetric_eigen(&self) -> (Vec3, Mat3) {
        self.symmetric_eigen_from(&Mat3::IDENTITY)
    }

    /// As [`Mat3::symmetric_eigen`], starting from `guess`, a rotation that
    /// nearly diagonalises the matrix — the eigenvectors of a matrix close
    /// to this one (the same quadrature point a Newton iteration ago). The
    /// answer is the same to rounding; it takes one or two sweeps instead of
    /// several.
    #[allow(clippy::needless_range_loop)] // rotations read two entries per index
    pub fn symmetric_eigen_from(&self, guess: &Mat3) -> (Vec3, Mat3) {
        // Re-orthonormalised, so a guess reused call after call cannot drift.
        let guess = &if *guess == Mat3::IDENTITY {
            Mat3::IDENTITY
        } else {
            let x = guess.col(0).normalize();
            let y = (guess.col(1) - x * x.dot(guess.col(1))).normalize();
            let rotation = Mat3::from_cols(x, y, x.cross(y));
            if rotation.is_finite() {
                rotation
            } else {
                Mat3::IDENTITY
            }
        };
        let rotated = guess.transpose() * self.symmetric_part() * *guess;
        let mut a = [[0.0; 3]; 3];
        for (i, row) in a.iter_mut().enumerate() {
            for (j, value) in row.iter_mut().enumerate() {
                *value = 0.5 * (rotated[(i, j)] + rotated[(j, i)]);
            }
        }
        let mut v = [
            guess.row(0).to_array(),
            guess.row(1).to_array(),
            guess.row(2).to_array(),
        ];
        for _ in 0..32 {
            let off = a[0][1] * a[0][1] + a[0][2] * a[0][2] + a[1][2] * a[1][2];
            if off == 0.0 {
                break;
            }
            for (p, q, r) in [(0, 1, 2), (0, 2, 1), (1, 2, 0)] {
                let apq = a[p][q];
                // Negligible against both diagonal entries: zero it, so
                // rounding-level remainders do not keep the sweeps going.
                if apq.abs() <= 1e-18 * (a[p][p].abs() + a[q][q].abs()) {
                    a[p][q] = 0.0;
                    a[q][p] = 0.0;
                    continue;
                }
                let theta = (a[q][q] - a[p][p]) / (2.0 * apq);
                let t = if theta == 0.0 {
                    1.0
                } else {
                    theta.signum() / (theta.abs() + (theta * theta + 1.0).sqrt())
                };
                let c = 1.0 / (t * t + 1.0).sqrt();
                let s = t * c;
                // Only the entries the rotation changes.
                a[p][p] -= t * apq;
                a[q][q] += t * apq;
                a[p][q] = 0.0;
                a[q][p] = 0.0;
                let (arp, arq) = (a[r][p], a[r][q]);
                a[r][p] = c * arp - s * arq;
                a[p][r] = a[r][p];
                a[r][q] = s * arp + c * arq;
                a[q][r] = a[r][q];
                for row in v.iter_mut() {
                    let (vp, vq) = (row[p], row[q]);
                    row[p] = c * vp - s * vq;
                    row[q] = s * vp + c * vq;
                }
            }
        }
        let values = Vec3::new(a[0][0], a[1][1], a[2][2]);
        let vectors = Mat3::from_rows(
            Vec3::from_array(v[0]),
            Vec3::from_array(v[1]),
            Vec3::from_array(v[2]),
        );
        (values, vectors)
    }

    /// Analytic trigonometric eigensolver for well-separated 3×3 symmetric
    /// matrices, falling back to warm-started Jacobi when eigenvalues are
    /// nearly repeated or numerically ill-conditioned.
    pub fn symmetric_eigen_fast(&self) -> (Vec3, Mat3) {
        self.symmetric_eigen_fast_or(&Mat3::IDENTITY)
    }

    pub fn symmetric_eigen_fast_or(&self, warm: &Mat3) -> (Vec3, Mat3) {
        let s = self.symmetric_part();
        let frob = s.frobenius_squared();
        if !frob.is_finite() {
            return self.symmetric_eigen_from(warm);
        }
        let p1 = s[(0, 1)] * s[(0, 1)] + s[(0, 2)] * s[(0, 2)] + s[(1, 2)] * s[(1, 2)];
        if p1 <= 1e-30 * frob {
            return (s.diagonal(), Mat3::IDENTITY);
        }
        let q = s.trace() / 3.0;
        let b = s - Mat3::scalar(q);
        let p2 = b[(0, 0)] * b[(0, 0)] + b[(1, 1)] * b[(1, 1)] + b[(2, 2)] * b[(2, 2)] + 2.0 * p1;
        let p = (p2 / 6.0).sqrt();
        if p == 0.0 || !p.is_finite() {
            return self.symmetric_eigen_from(warm);
        }
        let r = 0.5 * (b * (1.0 / p)).determinant();
        let phi = r.clamp(-1.0, 1.0).acos() / 3.0;
        let eig0 = q + 2.0 * p * phi.cos();
        let eig2 = q + 2.0 * p * (phi + 2.0 * std::f64::consts::PI / 3.0).cos();
        let eig1 = 3.0 * q - eig0 - eig2;
        if (eig0 - eig1).abs() < 1e-5 * p || (eig1 - eig2).abs() < 1e-5 * p {
            return self.symmetric_eigen_from(warm);
        }
        let eigenvector = |lambda: f64| -> Option<Vec3> {
            let m = s - Mat3::scalar(lambda);
            let (r0, r1, r2) = (m.row(0), m.row(1), m.row(2));
            let cands = [r0.cross(r1), r0.cross(r2), r1.cross(r2)];
            let mut best = cands[0];
            let mut best_l2 = best.length_squared();
            for c in &cands[1..] {
                let l2 = c.length_squared();
                if l2 > best_l2 {
                    best = *c;
                    best_l2 = l2;
                }
            }
            if best_l2 > 0.0 && best_l2.is_finite() {
                Some(best * (1.0 / best_l2.sqrt()))
            } else {
                None
            }
        };
        let (Some(v0), Some(v2_raw)) = (eigenvector(eig0), eigenvector(eig2)) else {
            return self.symmetric_eigen_from(warm);
        };
        let Some(v1) = v2_raw.cross(v0).try_normalize() else {
            return self.symmetric_eigen_from(warm);
        };
        let v2 = v0.cross(v1);
        let scale = p + q.abs();
        let ok = |v: Vec3, eig: f64| (s * v - v * eig).length() <= 1e-11 * scale;
        if ok(v0, eig0) && ok(v1, eig1) && ok(v2, eig2) {
            (Vec3::new(eig0, eig1, eig2), Mat3::from_cols(v0, v1, v2))
        } else {
            self.symmetric_eigen_from(warm)
        }
    }

    /// The singular value decomposition `self = U diag(σ) Vᵀ` with `U` and
    /// `V` rotations (determinant +1) and `|σ₀| ≥ σ₁ ≥ |σ₂|`; `σ₂` is
    /// negative when the matrix reflects (`det < 0`) — the convention
    /// deformation gradients need, where a flipped element must show as a
    /// negative stretch, not a reflected frame.
    pub fn svd(&self) -> (Mat3, Vec3, Mat3) {
        self.svd_from(&Mat3::IDENTITY)
    }

    /// As [`Mat3::svd`], with `guess` a rotation near `V` (warm start; see
    /// [`Mat3::symmetric_eigen_from`]).
    pub fn svd_from(&self, guess: &Mat3) -> (Mat3, Vec3, Mat3) {
        let (values, vectors) = (self.transpose() * *self).symmetric_eigen_from(guess);
        self.finish_svd(values, vectors)
    }

    /// Faster SVD using [`Mat3::symmetric_eigen_fast_or`], falling back to
    /// warm-started Jacobi when stretches coincide.
    pub fn svd_fast(&self, guess: &Mat3) -> (Mat3, Vec3, Mat3) {
        let (values, vectors) = (self.transpose() * *self).symmetric_eigen_fast_or(guess);
        self.finish_svd(values, vectors)
    }

    fn finish_svd(&self, values: Vec3, vectors: Mat3) -> (Mat3, Vec3, Mat3) {
        // Sort by decreasing eigenvalue.
        let mut order = [0, 1, 2];
        order.sort_by(|&a, &b| values[b].total_cmp(&values[a]));
        let mut v = Mat3::from_cols(
            vectors.col(order[0]),
            vectors.col(order[1]),
            vectors.col(order[2]),
        );
        if v.determinant() < 0.0 {
            v.cols[2] = -v.cols[2];
        }
        let a = *self * v;
        let s0 = a.cols[0].length();
        if s0 == 0.0 {
            return (Mat3::IDENTITY, Vec3::ZERO, v);
        }
        let u0 = a.cols[0] / s0;
        let rest = a.cols[1] - u0 * u0.dot(a.cols[1]);
        let u1 = rest
            .try_normalize()
            .filter(|_| rest.length() > 1e-14 * s0)
            .unwrap_or_else(|| u0.any_perpendicular());
        let u2 = u0.cross(u1);
        let sigma = Vec3::new(s0, u1.dot(a.cols[1]), u2.dot(a.cols[2]));
        (Mat3::from_cols(u0, u1, u2), sigma, v)
    }

    /// The polar decomposition `self = R S`: `R` the rotation nearest to
    /// `self`, `S` symmetric.
    pub fn polar(&self) -> (Mat3, Mat3) {
        let (u, sigma, v) = self.svd();
        (
            u * v.transpose(),
            v * Mat3::from_diagonal(sigma) * v.transpose(),
        )
    }

    /// The nearest positive semi-definite matrix to a symmetric one: its
    /// negative eigenvalues set to zero.
    pub fn project_positive(&self) -> Self {
        let (values, vectors) = self.symmetric_eigen();
        if values.min_element() >= 0.0 {
            return self.symmetric_part();
        }
        let clamped = values.max(Vec3::ZERO);
        vectors * Mat3::from_diagonal(clamped) * vectors.transpose()
    }
}

impl Index<(usize, usize)> for Mat3 {
    type Output = f64;
    /// Row `i`, column `j`.
    #[inline]
    fn index(&self, (i, j): (usize, usize)) -> &f64 {
        &self.cols[j][i]
    }
}

impl IndexMut<(usize, usize)> for Mat3 {
    #[inline]
    fn index_mut(&mut self, (i, j): (usize, usize)) -> &mut f64 {
        &mut self.cols[j][i]
    }
}

impl Mul<Vec3> for Mat3 {
    type Output = Vec3;
    #[inline]
    fn mul(self, v: Vec3) -> Vec3 {
        self.cols[0] * v.x + self.cols[1] * v.y + self.cols[2] * v.z
    }
}

impl Mul for Mat3 {
    type Output = Self;
    #[inline]
    fn mul(self, o: Self) -> Self {
        Self::from_cols(self * o.cols[0], self * o.cols[1], self * o.cols[2])
    }
}

impl Mul<f64> for Mat3 {
    type Output = Self;
    #[inline]
    fn mul(self, s: f64) -> Self {
        Self::from_cols(self.cols[0] * s, self.cols[1] * s, self.cols[2] * s)
    }
}

impl Mul<Mat3> for f64 {
    type Output = Mat3;
    #[inline]
    fn mul(self, m: Mat3) -> Mat3 {
        m * self
    }
}

impl Add for Mat3 {
    type Output = Self;
    #[inline]
    fn add(self, o: Self) -> Self {
        Self::from_cols(
            self.cols[0] + o.cols[0],
            self.cols[1] + o.cols[1],
            self.cols[2] + o.cols[2],
        )
    }
}

impl Sub for Mat3 {
    type Output = Self;
    #[inline]
    fn sub(self, o: Self) -> Self {
        Self::from_cols(
            self.cols[0] - o.cols[0],
            self.cols[1] - o.cols[1],
            self.cols[2] - o.cols[2],
        )
    }
}

impl Neg for Mat3 {
    type Output = Self;
    #[inline]
    fn neg(self) -> Self {
        self * -1.0
    }
}

impl AddAssign for Mat3 {
    #[inline]
    fn add_assign(&mut self, o: Self) {
        *self = *self + o;
    }
}

impl SubAssign for Mat3 {
    #[inline]
    fn sub_assign(&mut self, o: Self) {
        *self = *self - o;
    }
}

impl MulAssign<f64> for Mat3 {
    #[inline]
    fn mul_assign(&mut self, s: f64) {
        *self = *self * s;
    }
}
