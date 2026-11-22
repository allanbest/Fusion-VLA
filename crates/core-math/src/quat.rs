use std::ops::Mul;

use crate::{Mat3, Vec3};

/// A rotation, as a unit quaternion `w + xi + yj + zk`.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Quat {
    pub x: f64,
    pub y: f64,
    pub z: f64,
    pub w: f64,
}

impl Default for Quat {
    fn default() -> Self {
        Self::IDENTITY
    }
}

impl Quat {
    pub const IDENTITY: Self = Self {
        x: 0.0,
        y: 0.0,
        z: 0.0,
        w: 1.0,
    };

    #[inline]
    pub const fn from_xyzw(x: f64, y: f64, z: f64, w: f64) -> Self {
        Self { x, y, z, w }
    }

    /// A rotation by `angle` radians about the unit vector `axis`.
    pub fn from_axis_angle(axis: Vec3, angle: f64) -> Self {
        let (s, c) = (0.5 * angle).sin_cos();
        let v = axis * s;
        Self::from_xyzw(v.x, v.y, v.z, c)
    }

    /// A rotation by `|v|` radians about `v` (the exponential map).
    pub fn from_rotation_vector(v: Vec3) -> Self {
        let angle = v.length();
        if angle < 1e-8 {
            // Second-order series, exact to rounding at this size.
            let half = v * 0.5;
            return Self::from_xyzw(half.x, half.y, half.z, 1.0 - 0.5 * half.length_squared())
                .normalize();
        }
        Self::from_axis_angle(v / angle, angle)
    }

    /// The left Jacobian of the exponential map at `v`: for a small `δ`,
    /// `exp(v + δ) ≈ exp(J δ) exp(v)` — how a change in a rotation vector
    /// turns the body, in the world frame.
    pub fn left_jacobian(v: Vec3) -> Mat3 {
        let theta2 = v.length_squared();
        let k = Mat3::skew(v);
        let (a, b) = if theta2 < 1e-10 {
            (0.5 - theta2 / 24.0, 1.0 / 6.0 - theta2 / 120.0)
        } else {
            let theta = theta2.sqrt();
            (
                (1.0 - theta.cos()) / theta2,
                (theta - theta.sin()) / (theta2 * theta),
            )
        };
        Mat3::IDENTITY + k * a + k * k * b
    }

    /// The rotation vector: axis times angle in `[0, π]` (the logarithm).
    pub fn to_rotation_vector(self) -> Vec3 {
        let q = if self.w < 0.0 { -self } else { self };
        let v = q.vector();
        let s = v.length();
        if s < 1e-8 {
            return v * 2.0;
        }
        v * (2.0 * s.atan2(q.w) / s)
    }

    /// The rotation a matrix with orthonormal columns performs.
    pub fn from_mat3(m: &Mat3) -> Self {
        // Shepperd's method: divide by the largest of the four candidates.
        let trace = m.trace();
        let q = if trace > 0.0 {
            let s = 2.0 * (trace + 1.0).sqrt();
            Self::from_xyzw(
                (m[(2, 1)] - m[(1, 2)]) / s,
                (m[(0, 2)] - m[(2, 0)]) / s,
                (m[(1, 0)] - m[(0, 1)]) / s,
                0.25 * s,
            )
        } else if m[(0, 0)] > m[(1, 1)] && m[(0, 0)] > m[(2, 2)] {
            let s = 2.0 * (1.0 + m[(0, 0)] - m[(1, 1)] - m[(2, 2)]).sqrt();
            Self::from_xyzw(
                0.25 * s,
                (m[(0, 1)] + m[(1, 0)]) / s,
                (m[(0, 2)] + m[(2, 0)]) / s,
                (m[(2, 1)] - m[(1, 2)]) / s,
            )
        } else if m[(1, 1)] > m[(2, 2)] {
            let s = 2.0 * (1.0 + m[(1, 1)] - m[(0, 0)] - m[(2, 2)]).sqrt();
            Self::from_xyzw(
                (m[(0, 1)] + m[(1, 0)]) / s,
                0.25 * s,
                (m[(1, 2)] + m[(2, 1)]) / s,
                (m[(0, 2)] - m[(2, 0)]) / s,
            )
        } else {
            let s = 2.0 * (1.0 + m[(2, 2)] - m[(0, 0)] - m[(1, 1)]).sqrt();
            Self::from_xyzw(
                (m[(0, 2)] + m[(2, 0)]) / s,
                (m[(1, 2)] + m[(2, 1)]) / s,
                0.25 * s,
                (m[(1, 0)] - m[(0, 1)]) / s,
            )
        };
        q.normalize()
    }

    pub fn to_mat3(self) -> Mat3 {
        Mat3::from_cols(
            self.rotate(Vec3::X),
            self.rotate(Vec3::Y),
            self.rotate(Vec3::Z),
        )
    }

    #[inline]
    pub fn vector(self) -> Vec3 {
        Vec3::new(self.x, self.y, self.z)
    }

    #[inline]
    pub fn dot(self, o: Self) -> f64 {
        self.x * o.x + self.y * o.y + self.z * o.z + self.w * o.w
    }

    #[inline]
    pub fn length(self) -> f64 {
        self.dot(self).sqrt()
    }

    #[inline]
    pub fn normalize(self) -> Self {
        let s = 1.0 / self.length();
        Self::from_xyzw(self.x * s, self.y * s, self.z * s, self.w * s)
    }

    /// The inverse rotation.
    #[inline]
    pub fn conjugate(self) -> Self {
        Self::from_xyzw(-self.x, -self.y, -self.z, self.w)
    }

    #[inline]
    pub fn rotate(self, v: Vec3) -> Vec3 {
        let u = self.vector();
        let t = u.cross(v) * 2.0;
        v + t * self.w + u.cross(t)
    }

    #[inline]
    pub fn rotate_inverse(self, v: Vec3) -> Vec3 {
        self.conjugate().rotate(v)
    }

    /// The rotation after turning at angular velocity `spin` (rad/s, world
    /// frame) for `dt` seconds.
    pub fn integrate(self, spin: Vec3, dt: f64) -> Self {
        (Self::from_rotation_vector(spin * dt) * self).normalize()
    }

    /// The angle, in `[0, π]`, of the rotation from `self` to `o`.
    pub fn angle_to(self, o: Self) -> f64 {
        2.0 * self.dot(o).abs().min(1.0).acos()
    }

    pub fn is_finite(self) -> bool {
        self.x.is_finite() && self.y.is_finite() && self.z.is_finite() && self.w.is_finite()
    }
}

impl std::ops::Neg for Quat {
    type Output = Self;
    fn neg(self) -> Self {
        Self::from_xyzw(-self.x, -self.y, -self.z, -self.w)
    }
}

impl Mul for Quat {
    type Output = Self;
    /// `self * o` rotates by `o` first, then by `self`.
    #[inline]
    fn mul(self, o: Self) -> Self {
        let (a, b) = (self.vector(), o.vector());
        let v = b * self.w + a * o.w + a.cross(b);
        Self::from_xyzw(v.x, v.y, v.z, self.w * o.w - a.dot(b))
    }
}

impl Mul<Vec3> for Quat {
    type Output = Vec3;
    #[inline]
    fn mul(self, v: Vec3) -> Vec3 {
        self.rotate(v)
    }
}
