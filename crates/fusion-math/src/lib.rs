//! Fusion L0 — foundation math.
//!
//! Every type here is generated twice, once over `f32` and once over `f64`
//! ([`f32x`] and [`f64x`]). PLAN.md §9 makes precision a *per-solver* choice
//! rather than a global switch, so both widths have to exist at the bottom of
//! the stack from the start; adding the second one later would mean touching
//! every call site.
//!
//! The crate root re-exports the `f32` set unqualified (`Vec3`, `Quat`, ...)
//! because Fusion Basic is FP32 throughout. The `f64` set is reachable as
//! `DVec3`, `DQuat`, ... for the long accumulations and stiff solves that
//! Advanced will need.
//!
//! No dependencies, by intent. This crate is the one thing everything else
//! sits on, and it should never be a source of build friction.

#![deny(unsafe_code)]
// The public API freeze (PLAN.md §29, `docs/API.md`): an undocumented public
// item is one nobody agreed to freeze.
#![deny(missing_docs)]

pub mod simd;

macro_rules! define_math {
    ($t:ty) => {
        use core::ops::{
            Add, AddAssign, Div, DivAssign, Index, IndexMut, Mul, MulAssign, Neg, Sub, SubAssign,
        };

        /// The scalar type this module is generated over.
        pub type Scalar = $t;

        // ---------------------------------------------------------------- Vec3

        /// A 3-component vector.
        #[derive(Clone, Copy, Debug, Default, PartialEq)]
        #[repr(C)]
        pub struct Vec3 {
            /// The x component.
            pub x: $t,
            /// The y component.
            pub y: $t,
            /// The z component.
            pub z: $t,
        }

        /// Shorthand constructor: `vec3(1.0, 0.0, 0.0)`.
        #[inline]
        pub const fn vec3(x: $t, y: $t, z: $t) -> Vec3 {
            Vec3 { x, y, z }
        }

        impl Vec3 {
            /// All zeros.
            pub const ZERO: Self = Self {
                x: 0.0,
                y: 0.0,
                z: 0.0,
            };
            /// All ones.
            pub const ONE: Self = Self {
                x: 1.0,
                y: 1.0,
                z: 1.0,
            };
            /// The +x unit vector.
            pub const X: Self = Self {
                x: 1.0,
                y: 0.0,
                z: 0.0,
            };
            /// The +y unit vector.
            pub const Y: Self = Self {
                x: 0.0,
                y: 1.0,
                z: 0.0,
            };
            /// The +z unit vector.
            pub const Z: Self = Self {
                x: 0.0,
                y: 0.0,
                z: 1.0,
            };

            /// A vector from its three components.
            #[inline]
            pub const fn new(x: $t, y: $t, z: $t) -> Self {
                Self { x, y, z }
            }

            /// A vector with every component set to `v`.
            #[inline]
            pub const fn splat(v: $t) -> Self {
                Self { x: v, y: v, z: v }
            }

            /// From an array, in x, y, z order.
            #[inline]
            pub const fn from_array(a: [$t; 3]) -> Self {
                Self {
                    x: a[0],
                    y: a[1],
                    z: a[2],
                }
            }

            /// To an array, in x, y, z order.
            #[inline]
            pub const fn to_array(self) -> [$t; 3] {
                [self.x, self.y, self.z]
            }

            /// Dot product.
            #[inline]
            pub fn dot(self, o: Self) -> $t {
                self.x * o.x + self.y * o.y + self.z * o.z
            }

            /// Cross product, right-handed: `X.cross(Y) == Z`.
            #[inline]
            pub fn cross(self, o: Self) -> Self {
                Self {
                    x: self.y * o.z - self.z * o.y,
                    y: self.z * o.x - self.x * o.z,
                    z: self.x * o.y - self.y * o.x,
                }
            }

            /// Squared length. Prefer it to [`Self::length`] wherever the square root
            /// is about to be squared again — a comparison against a radius, for one.
            #[inline]
            pub fn length_squared(self) -> $t {
                self.dot(self)
            }

            /// Length.
            #[inline]
            pub fn length(self) -> $t {
                self.length_squared().sqrt()
            }

            /// Distance to another point.
            #[inline]
            pub fn distance(self, o: Self) -> $t {
                (self - o).length()
            }

            /// Squared distance to another point.
            #[inline]
            pub fn distance_squared(self, o: Self) -> $t {
                (self - o).length_squared()
            }

            /// Unit vector. Returns garbage on a zero vector — use
            /// [`Vec3::normalize_or_zero`] where the input can degenerate.
            #[inline]
            pub fn normalize(self) -> Self {
                self / self.length()
            }

            /// Unit vector in the same direction, or zero where there is no direction
            /// to speak of. The variant to reach for when the input may be degenerate.
            #[inline]
            pub fn normalize_or_zero(self) -> Self {
                let len_sq = self.length_squared();
                if len_sq > <$t>::EPSILON * <$t>::EPSILON {
                    self / len_sq.sqrt()
                } else {
                    Self::ZERO
                }
            }

            /// Normalized vector paired with the original length, which callers
            /// in the contact code almost always need as well.
            #[inline]
            pub fn normalize_and_length(self) -> (Self, $t) {
                let len = self.length();
                if len > <$t>::EPSILON {
                    (self / len, len)
                } else {
                    (Self::ZERO, 0.0)
                }
            }

            /// True where every component is finite — neither infinite nor NaN.
            #[inline]
            pub fn is_finite(self) -> bool {
                self.x.is_finite() && self.y.is_finite() && self.z.is_finite()
            }

            /// Component-wise absolute value.
            #[inline]
            pub fn abs(self) -> Self {
                Self {
                    x: self.x.abs(),
                    y: self.y.abs(),
                    z: self.z.abs(),
                }
            }

            /// Component-wise sign: -1, 0 or +1 per component.
            #[inline]
            pub fn signum(self) -> Self {
                Self {
                    x: self.x.signum(),
                    y: self.y.signum(),
                    z: self.z.signum(),
                }
            }

            /// Component-wise minimum.
            #[inline]
            pub fn min(self, o: Self) -> Self {
                Self {
                    x: self.x.min(o.x),
                    y: self.y.min(o.y),
                    z: self.z.min(o.z),
                }
            }

            /// Component-wise maximum.
            #[inline]
            pub fn max(self, o: Self) -> Self {
                Self {
                    x: self.x.max(o.x),
                    y: self.y.max(o.y),
                    z: self.z.max(o.z),
                }
            }

            /// Component-wise clamp between two corners.
            #[inline]
            pub fn clamp(self, lo: Self, hi: Self) -> Self {
                self.max(lo).min(hi)
            }

            /// Smallest of the three components.
            #[inline]
            pub fn min_element(self) -> $t {
                self.x.min(self.y).min(self.z)
            }

            /// Largest of the three components.
            #[inline]
            pub fn max_element(self) -> $t {
                self.x.max(self.y).max(self.z)
            }

            /// Index of the largest component. Used by the box support function
            /// and by SAT axis selection.
            #[inline]
            pub fn max_axis(self) -> usize {
                if self.x > self.y {
                    if self.x > self.z {
                        0
                    } else {
                        2
                    }
                } else if self.y > self.z {
                    1
                } else {
                    2
                }
            }

            /// Component-wise product. Not a dot or a cross — the scaling one.
            #[inline]
            pub fn mul_element_wise(self, o: Self) -> Self {
                Self {
                    x: self.x * o.x,
                    y: self.y * o.y,
                    z: self.z * o.z,
                }
            }

            /// Component-wise quotient.
            #[inline]
            pub fn div_element_wise(self, o: Self) -> Self {
                Self {
                    x: self.x / o.x,
                    y: self.y / o.y,
                    z: self.z / o.z,
                }
            }

            /// Component-wise reciprocal.
            #[inline]
            pub fn recip(self) -> Self {
                Self {
                    x: 1.0 / self.x,
                    y: 1.0 / self.y,
                    z: 1.0 / self.z,
                }
            }

            /// Linear interpolation towards `o`; `t = 0` is `self`, `t = 1` is `o`.
            #[inline]
            pub fn lerp(self, o: Self, t: $t) -> Self {
                self + (o - self) * t
            }

            /// Any unit vector perpendicular to `self`, chosen to stay
            /// well-conditioned. Friction bases and EPA seeds need this.
            #[inline]
            pub fn any_orthonormal(self) -> Self {
                let a = if self.x.abs() > 0.9 { Self::Y } else { Self::X };
                self.cross(a).normalize_or_zero()
            }

            /// `self` with any component along `n` removed. `n` must be unit.
            #[inline]
            pub fn reject_from_unit(self, n: Self) -> Self {
                self - n * self.dot(n)
            }

            /// True where every component is within `tol` of the other's.
            #[inline]
            pub fn approx_eq(self, o: Self, tol: $t) -> bool {
                (self - o).abs().max_element() <= tol
            }
        }

        impl Index<usize> for Vec3 {
            type Output = $t;
            #[inline]
            fn index(&self, i: usize) -> &$t {
                match i {
                    0 => &self.x,
                    1 => &self.y,
                    2 => &self.z,
                    _ => panic!("Vec3 index out of range: {i}"),
                }
            }
        }

        impl IndexMut<usize> for Vec3 {
            #[inline]
            fn index_mut(&mut self, i: usize) -> &mut $t {
                match i {
                    0 => &mut self.x,
                    1 => &mut self.y,
                    2 => &mut self.z,
                    _ => panic!("Vec3 index out of range: {i}"),
                }
            }
        }

        impl Add for Vec3 {
            type Output = Self;
            #[inline]
            fn add(self, o: Self) -> Self {
                Self {
                    x: self.x + o.x,
                    y: self.y + o.y,
                    z: self.z + o.z,
                }
            }
        }
        impl Sub for Vec3 {
            type Output = Self;
            #[inline]
            fn sub(self, o: Self) -> Self {
                Self {
                    x: self.x - o.x,
                    y: self.y - o.y,
                    z: self.z - o.z,
                }
            }
        }
        impl Mul<$t> for Vec3 {
            type Output = Self;
            #[inline]
            fn mul(self, s: $t) -> Self {
                Self {
                    x: self.x * s,
                    y: self.y * s,
                    z: self.z * s,
                }
            }
        }
        impl Mul<Vec3> for $t {
            type Output = Vec3;
            #[inline]
            fn mul(self, v: Vec3) -> Vec3 {
                v * self
            }
        }
        impl Div<$t> for Vec3 {
            type Output = Self;
            #[inline]
            fn div(self, s: $t) -> Self {
                let inv = 1.0 / s;
                Self {
                    x: self.x * inv,
                    y: self.y * inv,
                    z: self.z * inv,
                }
            }
        }
        impl Neg for Vec3 {
            type Output = Self;
            #[inline]
            fn neg(self) -> Self {
                Self {
                    x: -self.x,
                    y: -self.y,
                    z: -self.z,
                }
            }
        }
        impl AddAssign for Vec3 {
            #[inline]
            fn add_assign(&mut self, o: Self) {
                *self = *self + o;
            }
        }
        impl SubAssign for Vec3 {
            #[inline]
            fn sub_assign(&mut self, o: Self) {
                *self = *self - o;
            }
        }
        impl MulAssign<$t> for Vec3 {
            #[inline]
            fn mul_assign(&mut self, s: $t) {
                *self = *self * s;
            }
        }
        impl DivAssign<$t> for Vec3 {
            #[inline]
            fn div_assign(&mut self, s: $t) {
                *self = *self / s;
            }
        }

        // ---------------------------------------------------------------- Mat3

        /// Column-major 3x3 matrix. Used for inertia tensors and rotation.
        #[derive(Clone, Copy, Debug, PartialEq)]
        #[repr(C)]
        pub struct Mat3 {
            /// First column, the image of the x axis.
            pub x_axis: Vec3,
            /// Second column, the image of the y axis.
            pub y_axis: Vec3,
            /// Third column, the image of the z axis.
            pub z_axis: Vec3,
        }

        impl Default for Mat3 {
            fn default() -> Self {
                Self::IDENTITY
            }
        }

        impl Mat3 {
            /// All zeros.
            pub const ZERO: Self = Self {
                x_axis: Vec3::ZERO,
                y_axis: Vec3::ZERO,
                z_axis: Vec3::ZERO,
            };
            /// The identity.
            pub const IDENTITY: Self = Self {
                x_axis: Vec3::X,
                y_axis: Vec3::Y,
                z_axis: Vec3::Z,
            };

            /// From three column vectors.
            #[inline]
            pub const fn from_cols(x_axis: Vec3, y_axis: Vec3, z_axis: Vec3) -> Self {
                Self {
                    x_axis,
                    y_axis,
                    z_axis,
                }
            }

            /// A diagonal matrix with `d` down the diagonal.
            #[inline]
            pub const fn from_diagonal(d: Vec3) -> Self {
                Self {
                    x_axis: Vec3::new(d.x, 0.0, 0.0),
                    y_axis: Vec3::new(0.0, d.y, 0.0),
                    z_axis: Vec3::new(0.0, 0.0, d.z),
                }
            }

            /// Column `i`, which panics outside 0..3.
            #[inline]
            pub fn col(&self, i: usize) -> Vec3 {
                match i {
                    0 => self.x_axis,
                    1 => self.y_axis,
                    2 => self.z_axis,
                    _ => panic!("Mat3 column out of range: {i}"),
                }
            }

            /// Row `i`, which panics outside 0..3.
            #[inline]
            pub fn row(&self, i: usize) -> Vec3 {
                Vec3::new(self.x_axis[i], self.y_axis[i], self.z_axis[i])
            }

            /// Transpose. For a rotation this is also the inverse.
            #[inline]
            pub fn transpose(&self) -> Self {
                Self {
                    x_axis: self.row(0),
                    y_axis: self.row(1),
                    z_axis: self.row(2),
                }
            }

            /// Determinant.
            #[inline]
            pub fn determinant(&self) -> $t {
                self.x_axis.dot(self.y_axis.cross(self.z_axis))
            }

            /// Inverse via the adjugate. Returns [`Mat3::ZERO`] if singular,
            /// which the caller treats as "infinite resistance" — the same
            /// convention as an inverse mass of zero.
            pub fn inverse(&self) -> Self {
                let a = self.y_axis.cross(self.z_axis);
                let b = self.z_axis.cross(self.x_axis);
                let c = self.x_axis.cross(self.y_axis);
                let det = self.x_axis.dot(a);
                if det.abs() < <$t>::EPSILON {
                    return Self::ZERO;
                }
                let inv_det = 1.0 / det;
                Self::from_cols(a * inv_det, b * inv_det, c * inv_det).transpose()
            }

            /// Rotation matrix equivalent to a unit quaternion.
            pub fn from_quat(q: Quat) -> Self {
                let (x, y, z, w) = (q.x, q.y, q.z, q.w);
                let (x2, y2, z2) = (x + x, y + y, z + z);
                let (xx, xy, xz) = (x * x2, x * y2, x * z2);
                let (yy, yz, zz) = (y * y2, y * z2, z * z2);
                let (wx, wy, wz) = (w * x2, w * y2, w * z2);
                Self::from_cols(
                    Vec3::new(1.0 - (yy + zz), xy + wz, xz - wy),
                    Vec3::new(xy - wz, 1.0 - (xx + zz), yz + wx),
                    Vec3::new(xz + wy, yz - wx, 1.0 - (xx + yy)),
                )
            }

            /// `self * diag(d) * self^T` — how a body-space inertia tensor is
            /// carried into world space each step.
            #[inline]
            pub fn mul_diagonal_mul_transpose(&self, d: Vec3) -> Self {
                let scaled =
                    Self::from_cols(self.x_axis * d.x, self.y_axis * d.y, self.z_axis * d.z);
                scaled * self.transpose()
            }

            /// Component-wise absolute value. Used on inertia tensors, where the
            /// magnitudes matter and the signs do not.
            #[inline]
            pub fn abs(&self) -> Self {
                Self::from_cols(self.x_axis.abs(), self.y_axis.abs(), self.z_axis.abs())
            }

            /// The matrix `S` with `S * x == v.cross(x)`.
            ///
            /// Needed wherever a cross product has to appear inside a
            /// derivative — the gyroscopic Jacobian in the integrator is the
            /// first such place.
            #[inline]
            pub fn skew_symmetric(v: Vec3) -> Self {
                Self::from_cols(
                    Vec3::new(0.0, v.z, -v.y),
                    Vec3::new(-v.z, 0.0, v.x),
                    Vec3::new(v.y, -v.x, 0.0),
                )
            }

            /// The Frobenius inner product, `sum_ij a_ij b_ij`.
            ///
            /// The double-dot of continuum mechanics. Stress contracted with a
            /// strain increment is a work density, which is what makes this the
            /// operation a plastic-dissipation term is written in.
            #[inline]
            pub fn double_dot(&self, o: &Self) -> $t {
                self.x_axis.dot(o.x_axis) + self.y_axis.dot(o.y_axis) + self.z_axis.dot(o.z_axis)
            }

            /// Singular value decomposition: `self == u * diag(s) * v^T`.
            ///
            /// `u` and `v` are **proper rotations** — determinant `+1`, not
            /// merely orthogonal — and `s` is ordered by descending magnitude
            /// with the last entry allowed to be negative. That convention is
            /// the useful one for a deformation gradient rather than a
            /// mathematical nicety: a material point that has inverted has
            /// `det F < 0`, and putting the sign on the smallest singular
            /// value keeps it visible to a constitutive model instead of
            /// hiding it in a reflection that the model would then mistake for
            /// a rotation and happily leave inside out.
            ///
            /// ## Method
            ///
            /// Cyclic Jacobi on `self^T * self`, which is symmetric positive
            /// semi-definite, then `u_i = F v_i / s_i`. Jacobi rather than a
            /// closed form because the closed forms for a 3x3 go through the
            /// roots of a cubic, and those lose most of their precision on the
            /// case that matters here — nearly equal singular values, which is
            /// what an undeformed material point has. Jacobi is unconditionally
            /// stable, and on a matrix this small its cost is a fixed number of
            /// plane rotations rather than an iteration count that depends on
            /// the answer.
            ///
            /// A fixed sweep count, not a tolerance, because a solver that
            /// converges in a data-dependent number of steps is a solver whose
            /// cost and results both depend on the scene in a way nothing can
            /// reproduce. Eight sweeps is past machine precision for a 3x3;
            /// the off-diagonal shrinks quadratically once the rotations start
            /// hitting the same pairs.
            pub fn svd(&self) -> (Self, Vec3, Self) {
                let at_a = self.transpose() * *self;
                let mut a = [
                    [at_a.x_axis.x, at_a.y_axis.x, at_a.z_axis.x],
                    [at_a.x_axis.y, at_a.y_axis.y, at_a.z_axis.y],
                    [at_a.x_axis.z, at_a.y_axis.z, at_a.z_axis.z],
                ];
                let mut v = [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]];

                for _ in 0..8 {
                    for (p, q) in [(0usize, 1usize), (0, 2), (1, 2)] {
                        if a[p][q].abs() <= <$t>::EPSILON {
                            continue;
                        }
                        // The rotation that zeroes (p, q), from the standard
                        // half-angle form. `theta` is written this way rather
                        // than through an `atan2` because the tangent is what
                        // the update needs and the arctangent would only be
                        // taken to be undone.
                        let theta = (a[q][q] - a[p][p]) / (2.0 * a[p][q]);
                        let sign = if theta < 0.0 { -1.0 } else { 1.0 };
                        let t = sign / (theta.abs() + (theta * theta + 1.0).sqrt());
                        let c = 1.0 / (t * t + 1.0).sqrt();
                        let s = t * c;
                        for k in 0..3 {
                            let (kp, kq) = (a[k][p], a[k][q]);
                            a[k][p] = c * kp - s * kq;
                            a[k][q] = s * kp + c * kq;
                        }
                        for k in 0..3 {
                            let (pk, qk) = (a[p][k], a[q][k]);
                            a[p][k] = c * pk - s * qk;
                            a[q][k] = s * pk + c * qk;
                        }
                        for k in 0..3 {
                            let (kp, kq) = (v[k][p], v[k][q]);
                            v[k][p] = c * kp - s * kq;
                            v[k][q] = s * kp + c * kq;
                        }
                    }
                }

                // Eigenvalues of `F^T F` are the squared singular values. The
                // clamp is not defensive: Jacobi on a matrix built from a
                // near-singular `F` can leave a diagonal entry a few ulps
                // below zero, and the square root of that is a NaN in the
                // middle of a material point's stress.
                let mut order = [0usize, 1, 2];
                let mut sigma = [
                    a[0][0].max(0.0).sqrt(),
                    a[1][1].max(0.0).sqrt(),
                    a[2][2].max(0.0).sqrt(),
                ];
                order.sort_by(|&i, &j| {
                    sigma[j]
                        .partial_cmp(&sigma[i])
                        .unwrap_or(core::cmp::Ordering::Equal)
                });
                let sorted = [sigma[order[0]], sigma[order[1]], sigma[order[2]]];
                sigma = sorted;
                let v_cols = [
                    Vec3::new(v[0][order[0]], v[1][order[0]], v[2][order[0]]),
                    Vec3::new(v[0][order[1]], v[1][order[1]], v[2][order[1]]),
                    Vec3::new(v[0][order[2]], v[1][order[2]], v[2][order[2]]),
                ];
                let mut vm = Self::from_cols(v_cols[0], v_cols[1], v_cols[2]);

                // `u_i` is the direction `F` sends `v_i`, orthonormalized
                // against the columns already settled rather than simply
                // divided by `s_i`.
                //
                // The division is the textbook form and it is not enough. It
                // divides the error in `F v_i` by the singular value, so a
                // matrix with one small singular value — a material point
                // squashed nearly flat, which is the interesting case — hands
                // back a `u` whose determinant is 0.97 rather than 1. That is
                // not a rotation, and a corotated model handed it would report
                // a stress for a deformation that never happened. Found on the
                // 27th of a hundred random matrices; three worked examples all
                // passed.
                //
                // Ordered by descending singular value, so each column is
                // built against the ones that are most reliable, and any that
                // collapses is completed from the others below.
                let tiny = sigma[0] * <$t>::EPSILON.sqrt();
                let mut u_cols = [Vec3::ZERO; 3];
                for i in 0..3 {
                    let mut image = *self * v_cols[i];
                    for settled in u_cols.iter().take(i) {
                        image -= *settled * settled.dot(image);
                    }
                    u_cols[i] = if image.length_squared() > tiny * tiny {
                        image.normalize()
                    } else {
                        Vec3::ZERO
                    };
                }
                for i in 0..3 {
                    if u_cols[i] != Vec3::ZERO {
                        continue;
                    }
                    let (j, k) = ((i + 1) % 3, (i + 2) % 3);
                    let candidate = u_cols[j].cross(u_cols[k]);
                    u_cols[i] = if candidate.length_squared() > <$t>::EPSILON {
                        candidate.normalize()
                    } else {
                        // Two columns gone as well, so there is no basis to
                        // complete against: fall back to an axis, chosen away
                        // from whichever one the surviving column lies along.
                        let along = u_cols[j] + u_cols[k];
                        let axis = if along.x.abs() < 0.9 {
                            Vec3::X
                        } else {
                            Vec3::Y
                        };
                        let candidate = along.cross(axis);
                        if candidate.length_squared() > <$t>::EPSILON {
                            candidate.normalize()
                        } else {
                            axis
                        }
                    };
                }
                let mut um = Self::from_cols(u_cols[0], u_cols[1], u_cols[2]);

                // Make both proper. Negating a column of `u` and the matching
                // singular value leaves `u * diag(s) * v^T` untouched, because
                // the two sign flips are the same diagonal `D` applied twice
                // and `D * D` is the identity.
                if um.determinant() < 0.0 {
                    um.z_axis = -um.z_axis;
                    sigma[2] = -sigma[2];
                }
                if vm.determinant() < 0.0 {
                    vm.z_axis = -vm.z_axis;
                    sigma[2] = -sigma[2];
                }
                (um, Vec3::new(sigma[0], sigma[1], sigma[2]), vm)
            }

            /// Polar decomposition: the rotation `r` and symmetric `s` with
            /// `self == r * s`.
            ///
            /// `r` is the rotation closest to `self` in the least-squares
            /// sense, which is what a corotated constitutive model means by
            /// "the rotation this material point has undergone". Taken from
            /// [`Self::svd`] rather than from a Newton iteration on
            /// `r <- (r + r^-T) / 2`: that iteration is cheaper and diverges on
            /// exactly the input a large-deformation solver eventually
            /// produces, an inverted point with `det F < 0`.
            pub fn polar(&self) -> (Self, Self) {
                let (u, sigma, v) = self.svd();
                let r = u * v.transpose();
                let s = v * Self::from_diagonal(sigma) * v.transpose();
                (r, s)
            }
        }

        impl Mul<Vec3> for Mat3 {
            type Output = Vec3;
            #[inline]
            fn mul(self, v: Vec3) -> Vec3 {
                self.x_axis * v.x + self.y_axis * v.y + self.z_axis * v.z
            }
        }

        impl Mul<Mat3> for Mat3 {
            type Output = Mat3;
            #[inline]
            fn mul(self, o: Mat3) -> Mat3 {
                Mat3::from_cols(self * o.x_axis, self * o.y_axis, self * o.z_axis)
            }
        }

        impl AddAssign<Mat3> for Mat3 {
            #[inline]
            fn add_assign(&mut self, o: Mat3) {
                *self = *self + o;
            }
        }

        impl Neg for Mat3 {
            type Output = Mat3;
            #[inline]
            fn neg(self) -> Mat3 {
                Mat3::from_cols(-self.x_axis, -self.y_axis, -self.z_axis)
            }
        }

        /// The outer product `a bᵀ`, whose columns are `a` scaled by each
        /// component of `b`.
        ///
        /// One of the two matrix constructions a particle method needs — this
        /// one builds the velocity gradient a transfer accumulates — and it
        /// is here rather than written out at each of its call sites because
        /// which vector indexes the rows is exactly the kind of thing that
        /// gets transposed once and stays wrong.
        #[inline]
        pub fn outer_product(a: Vec3, b: Vec3) -> Mat3 {
            Mat3::from_cols(a * b.x, a * b.y, a * b.z)
        }

        impl Mul<$t> for Mat3 {
            type Output = Mat3;
            #[inline]
            fn mul(self, s: $t) -> Mat3 {
                Mat3::from_cols(self.x_axis * s, self.y_axis * s, self.z_axis * s)
            }
        }

        impl Add for Mat3 {
            type Output = Mat3;
            #[inline]
            fn add(self, o: Mat3) -> Mat3 {
                Mat3::from_cols(
                    self.x_axis + o.x_axis,
                    self.y_axis + o.y_axis,
                    self.z_axis + o.z_axis,
                )
            }
        }

        impl Sub for Mat3 {
            type Output = Mat3;
            #[inline]
            fn sub(self, o: Mat3) -> Mat3 {
                Mat3::from_cols(
                    self.x_axis - o.x_axis,
                    self.y_axis - o.y_axis,
                    self.z_axis - o.z_axis,
                )
            }
        }

        // ---------------------------------------------------------------- Quat

        /// Unit quaternion, `(x, y, z)` vector part and `w` scalar part.
        #[derive(Clone, Copy, Debug, PartialEq)]
        #[repr(C)]
        pub struct Quat {
            /// The x component of the vector part.
            pub x: $t,
            /// The y component of the vector part.
            pub y: $t,
            /// The z component of the vector part.
            pub z: $t,
            /// The scalar part. Stored last, so `to_array` is xyzw.
            pub w: $t,
        }

        impl Default for Quat {
            fn default() -> Self {
                Self::IDENTITY
            }
        }

        impl Quat {
            /// The identity rotation.
            pub const IDENTITY: Self = Self {
                x: 0.0,
                y: 0.0,
                z: 0.0,
                w: 1.0,
            };

            /// From four components, in x, y, z, w order.
            #[inline]
            pub const fn from_xyzw(x: $t, y: $t, z: $t, w: $t) -> Self {
                Self { x, y, z, w }
            }

            /// To an array, in x, y, z, w order.
            #[inline]
            pub const fn to_array(self) -> [$t; 4] {
                [self.x, self.y, self.z, self.w]
            }

            /// A rotation of `angle` radians about `axis`, which must be normalized.
            #[inline]
            pub fn from_axis_angle(axis: Vec3, angle: $t) -> Self {
                let (s, c) = (angle * 0.5).sin_cos();
                let a = axis.normalize_or_zero();
                Self {
                    x: a.x * s,
                    y: a.y * s,
                    z: a.z * s,
                    w: c,
                }
            }

            /// Rotation vector form: direction is the axis, magnitude the angle.
            /// This is what integrating angular velocity naturally produces.
            pub fn from_scaled_axis(v: Vec3) -> Self {
                let (axis, angle) = v.normalize_and_length();
                if angle < <$t>::EPSILON {
                    Self::IDENTITY
                } else {
                    Self::from_axis_angle(axis, angle)
                }
            }

            /// A rotation about the x axis, in radians.
            #[inline]
            pub fn from_rotation_x(angle: $t) -> Self {
                Self::from_axis_angle(Vec3::X, angle)
            }
            /// A rotation about the y axis, in radians.
            #[inline]
            pub fn from_rotation_y(angle: $t) -> Self {
                Self::from_axis_angle(Vec3::Y, angle)
            }
            /// A rotation about the z axis, in radians.
            #[inline]
            pub fn from_rotation_z(angle: $t) -> Self {
                Self::from_axis_angle(Vec3::Z, angle)
            }

            /// Euler angles in radians, in the order `R = Rz(z)·Ry(y)·Rx(x)`.
            ///
            /// A quaternion is the right thing to *store* and an impossible
            /// thing to type, so anything with a text field for an orientation
            /// needs this pair. Which of the twelve Euler conventions it is
            /// matters less than that both directions agree, and that the
            /// answer is stated where somebody can read it — here.
            ///
            /// Not unique: several triples name the same rotation, and at the
            /// poles a whole family does. Round-tripping a quaternion through
            /// this and back is exact as a *rotation*; round-tripping the
            /// angles is not, and a caller that rewrites stored angles from a
            /// stored quaternion every frame will watch them jump between
            /// equivalent forms.
            pub fn to_euler(self) -> Vec3 {
                let q = self.normalize();
                let (x, y, z, w) = (q.x, q.y, q.z, q.w);
                // Third row of the rotation matrix, which is all the first two
                // angles need, and the top-left pair for the third.
                let m20 = 2.0 * (x * z - y * w);
                let pitch = (-m20).clamp(-1.0, 1.0).asin();
                // Straight up or straight down: the first and third angles
                // turn about the same axis and only their sum is determined.
                // Putting all of it in the first keeps both numbers small.
                if m20.abs() > 0.99999 {
                    let m12 = 2.0 * (y * z - x * w);
                    let m11 = 1.0 - 2.0 * (x * x + z * z);
                    return Vec3::new((-m12).atan2(m11), pitch, 0.0);
                }
                let m21 = 2.0 * (y * z + x * w);
                let m22 = 1.0 - 2.0 * (x * x + y * y);
                let m10 = 2.0 * (x * y + z * w);
                let m00 = 1.0 - 2.0 * (y * y + z * z);
                Vec3::new(m21.atan2(m22), pitch, m10.atan2(m00))
            }

            /// The inverse of [`Self::to_euler`]: radians back to a rotation.
            pub fn from_euler(angles: Vec3) -> Self {
                Self::from_rotation_z(angles.z)
                    * Self::from_rotation_y(angles.y)
                    * Self::from_rotation_x(angles.x)
            }

            /// The vector part, x, y, z.
            #[inline]
            pub fn vector_part(self) -> Vec3 {
                Vec3::new(self.x, self.y, self.z)
            }

            /// Squared norm. One for any rotation that has not drifted.
            #[inline]
            pub fn length_squared(self) -> $t {
                self.x * self.x + self.y * self.y + self.z * self.z + self.w * self.w
            }

            /// Norm.
            #[inline]
            pub fn length(self) -> $t {
                self.length_squared().sqrt()
            }

            /// The same rotation, renormalized. Integrating a quaternion lets its
            /// norm drift, and a drifted quaternion scales as well as rotates.
            #[inline]
            pub fn normalize(self) -> Self {
                let len = self.length();
                if len < <$t>::EPSILON {
                    return Self::IDENTITY;
                }
                let inv = 1.0 / len;
                Self {
                    x: self.x * inv,
                    y: self.y * inv,
                    z: self.z * inv,
                    w: self.w * inv,
                }
            }

            /// Inverse of a *unit* quaternion.
            #[inline]
            pub fn conjugate(self) -> Self {
                Self {
                    x: -self.x,
                    y: -self.y,
                    z: -self.z,
                    w: self.w,
                }
            }

            /// The inverse rotation. For a unit quaternion this is the conjugate.
            #[inline]
            pub fn inverse(self) -> Self {
                self.conjugate()
            }

            /// Dot product of the four components, which is the cosine of half the
            /// angle between two rotations.
            #[inline]
            pub fn dot(self, o: Self) -> $t {
                self.x * o.x + self.y * o.y + self.z * o.z + self.w * o.w
            }

            /// Rotate a vector by this quaternion.
            #[inline]
            pub fn rotate(self, v: Vec3) -> Vec3 {
                let u = self.vector_part();
                let t = u.cross(v) * 2.0;
                v + t * self.w + u.cross(t)
            }

            /// Rotate by the inverse — world direction into body space.
            #[inline]
            pub fn rotate_inverse(self, v: Vec3) -> Vec3 {
                self.conjugate().rotate(v)
            }

            /// Integrate angular velocity `w` (rad/s, world frame) over `dt`
            /// using the first-order update `q += 0.5 * (w,0) * q * dt`,
            /// renormalized. Cheaper than the exact exponential map and
            /// indistinguishable at solver timesteps.
            #[inline]
            pub fn integrate(self, w: Vec3, dt: $t) -> Self {
                let dq = Quat::from_xyzw(w.x, w.y, w.z, 0.0) * self;
                Self {
                    x: self.x + dq.x * 0.5 * dt,
                    y: self.y + dq.y * 0.5 * dt,
                    z: self.z + dq.z * 0.5 * dt,
                    w: self.w + dq.w * 0.5 * dt,
                }
                .normalize()
            }

            /// True where every component is finite.
            #[inline]
            pub fn is_finite(self) -> bool {
                self.x.is_finite() && self.y.is_finite() && self.z.is_finite() && self.w.is_finite()
            }

            /// True where every component is within `tol` of the other's. Note that a
            /// quaternion and its negation are the same rotation and are not `approx_eq`.
            pub fn approx_eq(self, o: Self, tol: $t) -> bool {
                // q and -q are the same rotation.
                let d = self.dot(o).abs();
                (1.0 - d) <= tol
            }
        }

        impl Mul for Quat {
            type Output = Self;
            #[inline]
            fn mul(self, o: Self) -> Self {
                Self {
                    x: self.w * o.x + self.x * o.w + self.y * o.z - self.z * o.y,
                    y: self.w * o.y - self.x * o.z + self.y * o.w + self.z * o.x,
                    z: self.w * o.z + self.x * o.y - self.y * o.x + self.z * o.w,
                    w: self.w * o.w - self.x * o.x - self.y * o.y - self.z * o.z,
                }
            }
        }

        impl Mul<Vec3> for Quat {
            type Output = Vec3;
            #[inline]
            fn mul(self, v: Vec3) -> Vec3 {
                self.rotate(v)
            }
        }

        // ------------------------------------------------------------ Isometry

        /// Rigid transform: rotation then translation. No scale — a scaled
        /// rigid body is a different shape, not a different transform.
        #[derive(Clone, Copy, Debug, PartialEq)]
        #[repr(C)]
        pub struct Isometry {
            /// The translation.
            pub translation: Vec3,
            /// The rotation, applied before the translation.
            pub rotation: Quat,
        }

        impl Default for Isometry {
            fn default() -> Self {
                Self::IDENTITY
            }
        }

        impl Isometry {
            /// No rotation, no translation.
            pub const IDENTITY: Self = Self {
                translation: Vec3::ZERO,
                rotation: Quat::IDENTITY,
            };

            /// From a translation and a rotation.
            #[inline]
            pub const fn new(translation: Vec3, rotation: Quat) -> Self {
                Self {
                    translation,
                    rotation,
                }
            }

            /// A pure translation.
            #[inline]
            pub const fn from_translation(translation: Vec3) -> Self {
                Self {
                    translation,
                    rotation: Quat::IDENTITY,
                }
            }

            /// Rotate and translate a point.
            #[inline]
            pub fn transform_point(&self, p: Vec3) -> Vec3 {
                self.rotation.rotate(p) + self.translation
            }

            /// Rotate a direction, leaving the translation out of it.
            #[inline]
            pub fn transform_direction(&self, d: Vec3) -> Vec3 {
                self.rotation.rotate(d)
            }

            /// The inverse of [`Self::transform_point`].
            #[inline]
            pub fn inverse_transform_point(&self, p: Vec3) -> Vec3 {
                self.rotation.rotate_inverse(p - self.translation)
            }

            /// The inverse of [`Self::transform_direction`].
            #[inline]
            pub fn inverse_transform_direction(&self, d: Vec3) -> Vec3 {
                self.rotation.rotate_inverse(d)
            }

            /// The inverse transform.
            #[inline]
            pub fn inverse(&self) -> Self {
                let r = self.rotation.conjugate();
                Self {
                    translation: r.rotate(-self.translation),
                    rotation: r,
                }
            }

            /// `self * other` — apply `other` first, then `self`.
            #[inline]
            pub fn mul(&self, o: &Self) -> Self {
                Self {
                    translation: self.transform_point(o.translation),
                    rotation: self.rotation * o.rotation,
                }
            }

            /// Transform taking points expressed in `self` into `other`.
            #[inline]
            pub fn inv_mul(&self, o: &Self) -> Self {
                self.inverse().mul(o)
            }

            /// Column-major 4x4 for graphics APIs, row-major in memory order
            /// as WGSL expects.
            pub fn to_cols_array_4x4(&self) -> [$t; 16] {
                let m = Mat3::from_quat(self.rotation);
                [
                    m.x_axis.x,
                    m.x_axis.y,
                    m.x_axis.z,
                    0.0,
                    m.y_axis.x,
                    m.y_axis.y,
                    m.y_axis.z,
                    0.0,
                    m.z_axis.x,
                    m.z_axis.y,
                    m.z_axis.z,
                    0.0,
                    self.translation.x,
                    self.translation.y,
                    self.translation.z,
                    1.0,
                ]
            }
        }

        // ---------------------------------------------------------------- Aabb

        /// Axis-aligned bounding box. The broad phase's only currency.
        #[derive(Clone, Copy, Debug, PartialEq)]
        #[repr(C)]
        pub struct Aabb {
            /// Lower corner.
            pub min: Vec3,
            /// Upper corner.
            pub max: Vec3,
        }

        impl Aabb {
            /// Inverted bounds — the identity for [`Aabb::merge`].
            pub const EMPTY: Self = Self {
                min: Vec3::splat(<$t>::INFINITY),
                max: Vec3::splat(<$t>::NEG_INFINITY),
            };

            /// From two corners, which are taken as given rather than sorted.
            #[inline]
            pub const fn new(min: Vec3, max: Vec3) -> Self {
                Self { min, max }
            }

            /// From a centre and half-extents.
            #[inline]
            pub fn from_center_half_extents(center: Vec3, half: Vec3) -> Self {
                Self {
                    min: center - half,
                    max: center + half,
                }
            }

            /// The smallest box containing every point. Invalid for an empty slice.
            #[inline]
            pub fn from_points(points: &[Vec3]) -> Self {
                points.iter().fold(Self::EMPTY, |b, &p| b.extended(p))
            }

            /// Centre.
            #[inline]
            pub fn center(&self) -> Vec3 {
                (self.min + self.max) * 0.5
            }

            /// Half the size, per axis.
            #[inline]
            pub fn half_extents(&self) -> Vec3 {
                (self.max - self.min) * 0.5
            }

            /// Size, per axis.
            #[inline]
            pub fn extents(&self) -> Vec3 {
                self.max - self.min
            }

            /// True where no axis has its minimum above its maximum.
            #[inline]
            pub fn is_valid(&self) -> bool {
                self.min.x <= self.max.x && self.min.y <= self.max.y && self.min.z <= self.max.z
            }

            /// The smallest box containing this one and a point.
            #[inline]
            pub fn extended(&self, p: Vec3) -> Self {
                Self {
                    min: self.min.min(p),
                    max: self.max.max(p),
                }
            }

            /// The smallest box containing both.
            #[inline]
            pub fn merge(&self, o: &Self) -> Self {
                Self {
                    min: self.min.min(o.min),
                    max: self.max.max(o.max),
                }
            }

            /// Grown by `m` on every side. Broad-phase margins let contact
            /// pairs persist across frames instead of flickering.
            #[inline]
            pub fn expanded(&self, m: $t) -> Self {
                Self {
                    min: self.min - Vec3::splat(m),
                    max: self.max + Vec3::splat(m),
                }
            }

            /// True where the two overlap, touching included.
            #[inline]
            pub fn overlaps(&self, o: &Self) -> bool {
                self.min.x <= o.max.x
                    && self.max.x >= o.min.x
                    && self.min.y <= o.max.y
                    && self.max.y >= o.min.y
                    && self.min.z <= o.max.z
                    && self.max.z >= o.min.z
            }

            /// True where the point is inside, on the boundary included.
            #[inline]
            pub fn contains_point(&self, p: Vec3) -> bool {
                p.x >= self.min.x
                    && p.x <= self.max.x
                    && p.y >= self.min.y
                    && p.y <= self.max.y
                    && p.z >= self.min.z
                    && p.z <= self.max.z
            }

            /// Total surface area. The cost metric a bounding-volume hierarchy is
            /// built to minimize.
            #[inline]
            pub fn surface_area(&self) -> $t {
                let e = self.extents();
                2.0 * (e.x * e.y + e.y * e.z + e.z * e.x)
            }
        }
    };
}

/// Math over `f32` — the working precision for Fusion Basic.
pub mod f32x {
    define_math!(f32);
}

/// Math over `f64` — long accumulations, stiff solves, reference runs.
pub mod f64x {
    define_math!(f64);
}

pub use f32x::{outer_product, vec3, Aabb, Isometry, Mat3, Quat, Scalar, Vec3};

pub use f64x::{
    outer_product as douter_product, Aabb as DAabb, Isometry as DIsometry, Mat3 as DMat3,
    Quat as DQuat, Scalar as DScalar, Vec3 as DVec3,
};

/// Widen a single-precision vector to double.
#[inline]
pub fn to_f64(v: Vec3) -> DVec3 {
    DVec3::new(v.x as f64, v.y as f64, v.z as f64)
}

/// Narrow a double-precision vector to single.
#[inline]
pub fn to_f32(v: DVec3) -> Vec3 {
    Vec3::new(v.x as f32, v.y as f32, v.z as f32)
}

/// Standard gravity, m/s^2.
pub const GRAVITY: Vec3 = Vec3::new(0.0, -9.81, 0.0);

/// Clamp a scalar between two bounds.
#[inline]
pub fn clamp(v: Scalar, lo: Scalar, hi: Scalar) -> Scalar {
    if v < lo {
        lo
    } else if v > hi {
        hi
    } else {
        v
    }
}

#[cfg(test)]
mod euler_tests {
    use super::*;

    /// Angles in, the same rotation out. Not the same *angles* — the mapping
    /// is many-to-one — but the same orientation, which is the thing that has
    /// to hold.
    #[test]
    fn euler_angles_round_trip_as_rotations() {
        for angles in [
            vec3(0.0, 0.0, 0.0),
            vec3(0.3, 0.0, 0.0),
            vec3(0.0, 0.7, 0.0),
            vec3(0.0, 0.0, -1.1),
            vec3(0.2, -0.6, 1.0),
            // The pole, where the decomposition is not unique.
            vec3(0.0, std::f32::consts::FRAC_PI_2, 0.0),
        ] {
            let q = Quat::from_euler(angles);
            let back = Quat::from_euler(q.to_euler());
            assert!(
                q.approx_eq(back, 1e-5),
                "{angles:?} became a different rotation"
            );
        }
    }

    /// A single-axis rotation comes back on that axis and no other.
    #[test]
    fn one_axis_stays_one_axis() {
        let angles = Quat::from_rotation_y(0.5).to_euler();
        assert!(angles.x.abs() < 1e-5, "{angles:?}");
        assert!((angles.y - 0.5).abs() < 1e-5, "{angles:?}");
        assert!(angles.z.abs() < 1e-5, "{angles:?}");
    }
}
