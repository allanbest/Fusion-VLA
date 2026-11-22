//! Analytic shapes in their own frame: signed distance, bounds, and mass
//! properties.

use std::f64::consts::PI;

use core_math::{Mat3, Pose, Vec3};

use crate::Aabb;

/// A shape, centred on its own origin (its centre of mass, for a solid of
/// uniform density).
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum Shape {
    /// A ball.
    Sphere { radius: f64 },
    /// A box, by half its size along each axis.
    Cuboid { half: Vec3 },
    /// A cylinder along the local `y` axis with hemispherical ends: the
    /// points within `radius` of the segment from `−half_length` to
    /// `+half_length` on `y`.
    Capsule { radius: f64, half_length: f64 },
    /// Everything on the side of a plane opposite its (unit) `normal`: the
    /// points `p` with `normal · p ≤ offset`. Infinite: for the ground and
    /// walls.
    HalfSpace { normal: Vec3, offset: f64 },
}

/// Mass properties of a shape of unit density (so mass equals volume).
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct UnitMass {
    /// m³.
    pub volume: f64,
    /// Inertia about the shape's origin in its own frame, per unit density,
    /// m⁵.
    pub inertia: Mat3,
}

impl Shape {
    /// A box by its full size.
    pub fn cuboid(size: Vec3) -> Self {
        Self::Cuboid { half: size * 0.5 }
    }

    /// The ground: everything below height `y`.
    pub fn ground(y: f64) -> Self {
        Self::HalfSpace {
            normal: Vec3::Y,
            offset: y,
        }
    }

    /// Whether the shape is finite (so can have a mass).
    pub fn is_bounded(&self) -> bool {
        !matches!(self, Self::HalfSpace { .. })
    }

    /// Signed distance from `p` (in the shape's frame) to the surface —
    /// negative inside — and the outward unit normal of the nearest surface
    /// point (the distance's gradient).
    pub fn distance(&self, p: Vec3) -> (f64, Vec3) {
        match *self {
            Self::Sphere { radius } => {
                let length = p.length();
                let normal = if length > 0.0 { p / length } else { Vec3::Y };
                (length - radius, normal)
            }
            Self::Cuboid { half } => {
                let q = p.abs() - half;
                let sign = |v: f64| if v < 0.0 { -1.0 } else { 1.0 };
                let outside = q.max(Vec3::ZERO);
                let outside_length = outside.length();
                if outside_length > 0.0 {
                    let n = Vec3::new(
                        sign(p.x) * outside.x,
                        sign(p.y) * outside.y,
                        sign(p.z) * outside.z,
                    );
                    (outside_length, n / outside_length)
                } else {
                    let axis = q.max_axis();
                    let mut n = Vec3::ZERO;
                    n[axis] = sign(p[axis]);
                    (q[axis], n)
                }
            }
            Self::Capsule {
                radius,
                half_length,
            } => {
                let closest = Vec3::new(0.0, p.y.clamp(-half_length, half_length), 0.0);
                let d = p - closest;
                let length = d.length();
                let normal = if length > 0.0 { d / length } else { Vec3::X };
                (length - radius, normal)
            }
            Self::HalfSpace { normal, offset } => (normal.dot(p) - offset, normal),
        }
    }

    /// Bounds in the shape's own frame.
    pub fn local_bounds(&self) -> Aabb {
        match *self {
            Self::Sphere { radius } => Aabb::around(Vec3::ZERO, Vec3::splat(radius)),
            Self::Cuboid { half } => Aabb::around(Vec3::ZERO, half),
            Self::Capsule {
                radius,
                half_length,
            } => Aabb::around(Vec3::ZERO, Vec3::new(radius, half_length + radius, radius)),
            Self::HalfSpace { .. } => Aabb::EVERYTHING,
        }
    }

    /// Bounds of the shape placed at `pose`.
    pub fn bounds(&self, pose: &Pose) -> Aabb {
        match *self {
            Self::Sphere { radius } => Aabb::around(pose.position, Vec3::splat(radius)),
            Self::Cuboid { half } => {
                let r = pose.rotation.to_mat3();
                let extent = Vec3::new(
                    r.row(0).abs().dot(half),
                    r.row(1).abs().dot(half),
                    r.row(2).abs().dot(half),
                );
                Aabb::around(pose.position, extent)
            }
            Self::Capsule {
                radius,
                half_length,
            } => {
                let axis = pose.transform_vector(Vec3::new(0.0, half_length, 0.0));
                Aabb::around(pose.position, axis.abs() + Vec3::splat(radius))
            }
            Self::HalfSpace { .. } => Aabb::EVERYTHING,
        }
    }

    /// Bounds after an arbitrary affine map `centre + matrix * local`.
    /// Unlike rotation-only bounds, these remain conservative at intermediate
    /// Newton iterates where a rigid body's matrix is not yet orthogonal.
    pub fn affine_bounds(&self, centre: Vec3, matrix: Mat3) -> Aabb {
        let rows = [matrix.row(0), matrix.row(1), matrix.row(2)];
        let extent = match *self {
            Self::Cuboid { half } => Vec3::new(
                rows[0].abs().dot(half),
                rows[1].abs().dot(half),
                rows[2].abs().dot(half),
            ),
            Self::Sphere { radius } => {
                Vec3::new(rows[0].length(), rows[1].length(), rows[2].length()) * radius
            }
            Self::Capsule {
                radius,
                half_length,
            } => Vec3::new(
                rows[0].length() * radius + rows[0].y.abs() * half_length,
                rows[1].length() * radius + rows[1].y.abs() * half_length,
                rows[2].length() * radius + rows[2].y.abs() * half_length,
            ),
            Self::HalfSpace { .. } => return Aabb::EVERYTHING,
        };
        Aabb::around(centre, extent)
    }

    /// Volume and inertia at unit density; `None` for an infinite shape.
    pub fn unit_mass(&self) -> Option<UnitMass> {
        let (volume, diagonal) = match *self {
            Self::Sphere { radius } => {
                let v = 4.0 / 3.0 * PI * radius.powi(3);
                (v, Vec3::splat(0.4 * v * radius * radius))
            }
            Self::Cuboid { half: h } => {
                let v = 8.0 * h.x * h.y * h.z;
                let s = h.mul_elements(h);
                (v, Vec3::new(s.y + s.z, s.z + s.x, s.x + s.y) * (v / 3.0))
            }
            Self::Capsule {
                radius: r,
                half_length: h,
            } => {
                let cylinder = PI * r * r * 2.0 * h;
                let caps = 4.0 / 3.0 * PI * r.powi(3);
                let axial = cylinder * r * r / 2.0 + caps * 0.4 * r * r;
                let across = cylinder * (r * r / 4.0 + h * h / 3.0)
                    + caps * (0.4 * r * r + h * h + 0.75 * h * r);
                (cylinder + caps, Vec3::new(across, axial, across))
            }
            Self::HalfSpace { .. } => return None,
        };
        Some(UnitMass {
            volume,
            inertia: Mat3::from_diagonal(diagonal),
        })
    }
}
