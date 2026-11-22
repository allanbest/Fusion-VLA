//! Collision and inertia geometry.
//!
//! PLAN.md §8 names three representations: triangle meshes, convex hulls, and
//! SDFs. v0.1 ships only the two analytic primitives the spec calls for. They
//! are not a stepping stone to a general convex type — they are the permanent
//! fast path, and every later shape is added alongside them.

use crate::convex::SharedHull;
use fusion_math::{vec3, Aabb, Isometry, Mat3, Quat, Vec3};

/// Mass, and the inertia tensor about the centre of mass.
///
/// Both primitives here have their principal axes aligned with the body frame,
/// so the tensor is a diagonal and we store only that. A general convex hull
/// would need the full [`Mat3`]; when that lands, this type grows a variant
/// rather than every caller changing.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct MassProperties {
    /// Mass, kg.
    pub mass: f32,
    /// Diagonal of the inertia tensor in body space, kg m^2.
    pub inertia_diagonal: Vec3,
}

impl MassProperties {
    /// Inverse mass and inverse inertia, which is what the solver actually
    /// wants (PLAN.md §10: inverses keep division out of the hot loop, and a
    /// static body is a zero rather than a special case).
    pub fn to_inverses(self) -> (f32, Vec3) {
        if self.mass <= 0.0 {
            return (0.0, Vec3::ZERO);
        }
        let inv_inertia = Vec3::new(
            invert_or_zero(self.inertia_diagonal.x),
            invert_or_zero(self.inertia_diagonal.y),
            invert_or_zero(self.inertia_diagonal.z),
        );
        (1.0 / self.mass, inv_inertia)
    }
}

#[inline]
fn invert_or_zero(v: f32) -> f32 {
    if v > 0.0 {
        1.0 / v
    } else {
        0.0
    }
}

/// A collision shape. §8 requires every shape to carry an SDF, which is
/// what lets the fluid's boundary handling work without boundary particles.
///
/// ## Why this is no longer `Copy`
///
/// A hull carries tens of vertices and their faces, so it is shared behind an
/// `Arc` rather than copied into every body that uses it — a hundred crates of
/// the same imported mesh are a hundred handles and one hull. That costs
/// `Copy`, which cost about forty `*self` patterns across the workspace and
/// nothing else: a shape is read far more often than it is assigned.
#[derive(Clone, Debug, PartialEq)]
pub enum Shape {
    /// A sphere centred on the body origin.
    Sphere {
        /// Radius, metres.
        radius: f32,
    },
    /// Box, half-extents from the centre. Named `Cuboid` because `Box` is
    /// taken by the standard library.
    Cuboid {
        /// Half the extents, from the centre.
        half_extents: Vec3,
    },
    /// An arbitrary convex polyhedron — what an imported mesh becomes.
    ///
    /// PLAN.md §16 lists mesh import as the second of five creation verbs, and
    /// [`ConvexHull`](crate::ConvexHull) has carried the support function, the
    /// face planes and the inertia integral since v0.5. Nothing could reach
    /// it: the shape enum had two variants, so an engine with a working hull
    /// implementation could still only simulate spheres and boxes.
    ///
    /// **A concave mesh is simulated as its convex hull**, which is a real
    /// limitation rather than a rounding error: a teacup holds no water and a
    /// chair has no gap under the seat. `ConvexHull` says so at length.
    Hull(SharedHull),
}

impl Shape {
    /// A sphere of `radius`.
    pub fn sphere(radius: f32) -> Self {
        Shape::Sphere { radius }
    }

    /// Box from its *full* size, which is how people describe boxes.
    pub fn cuboid(size: Vec3) -> Self {
        Shape::Cuboid {
            half_extents: size * 0.5,
        }
    }

    /// A convex hull of a point cloud — the vertices of an imported mesh.
    ///
    /// `None` where the points enclose no volume: fewer than four of them, or
    /// all on one line or plane. A body with no volume has no mass, so that is
    /// a failure to report rather than a shape to build.
    pub fn hull(points: &[Vec3]) -> Option<Self> {
        crate::ConvexHull::from_points(points).map(|h| Shape::Hull(SharedHull::new(h)))
    }

    /// Volume, m^3.
    pub fn volume(&self) -> f32 {
        match self {
            Shape::Sphere { radius } => (4.0 / 3.0) * std::f32::consts::PI * radius.powi(3),
            Shape::Cuboid { half_extents: h } => 8.0 * h.x * h.y * h.z,
            Shape::Hull(hull) => hull.volume(),
        }
    }

    /// Surface area, m^2. What a body exchanges heat with the air through.
    pub fn surface_area(&self) -> f32 {
        match self {
            Shape::Sphere { radius } => 4.0 * std::f32::consts::PI * radius * radius,
            Shape::Cuboid { half_extents: h } => 8.0 * (h.x * h.y + h.y * h.z + h.z * h.x),
            Shape::Hull(hull) => {
                let v = hull.vertices();
                hull.faces()
                    .iter()
                    .map(|f| {
                        let [a, b, c] = f.indices.map(|i| v[i as usize]);
                        0.5 * (b - a).cross(c - a).length()
                    })
                    .sum()
            }
        }
    }

    /// Radius of the smallest origin-centred sphere containing the shape.
    pub fn bounding_radius(&self) -> f32 {
        match self {
            Shape::Sphere { radius } => *radius,
            Shape::Cuboid { half_extents } => half_extents.length(),
            Shape::Hull(hull) => hull.bounding_radius(),
        }
    }

    /// Mass and inertia for this shape at a given density.
    pub fn mass_properties(&self, density: f32) -> MassProperties {
        let mass = self.volume() * density;
        let inertia_diagonal = match self {
            Shape::Sphere { radius } => Vec3::splat(0.4 * mass * radius * radius),
            Shape::Hull(hull) => hull.inertia_diagonal(mass),
            Shape::Cuboid { half_extents: h } => {
                let (x, y, z) = (2.0 * h.x, 2.0 * h.y, 2.0 * h.z);
                let k = mass / 12.0;
                vec3(
                    k * (y * y + z * z),
                    k * (x * x + z * z),
                    k * (x * x + y * y),
                )
            }
        };
        MassProperties {
            mass,
            inertia_diagonal,
        }
    }

    /// Bounds in the body's own frame.
    pub fn local_aabb(&self) -> Aabb {
        match self {
            Shape::Sphere { radius } => {
                Aabb::from_center_half_extents(Vec3::ZERO, Vec3::splat(*radius))
            }
            Shape::Cuboid { half_extents } => {
                Aabb::from_center_half_extents(Vec3::ZERO, *half_extents)
            }
            Shape::Hull(hull) => hull.local_aabb(),
        }
    }

    /// Signed distance to the surface, in body space. Negative inside.
    ///
    /// PLAN.md §8 makes the SDF the second representation every shape carries
    /// alongside its support mapping, and §16 makes it load-bearing for
    /// authoring: `fill` samples a shape's interior, and "that one verb is the
    /// entire authoring story for fluids and continuum bodies." Exact for both
    /// primitives Fusion has; a mesh would carry a sampled grid instead.
    pub fn signed_distance_local(&self, p: Vec3) -> f32 {
        match self {
            Shape::Sphere { radius } => p.length() - radius,
            // Exact inside a convex body, and an under-estimate outside one
            // near an edge — `ConvexHull::signed_distance` says so. Every
            // caller in the engine is asking about the inside.
            Shape::Hull(hull) => hull.signed_distance(p),
            Shape::Cuboid { half_extents } => {
                // The standard box SDF. The first term is the distance to the
                // nearest corner region when outside; the second is negative
                // only when inside, where it is the distance to the nearest
                // face. Adding them is correct in both cases because exactly
                // one of them is non-zero.
                let q = p.abs() - *half_extents;
                q.max(Vec3::ZERO).length() + q.max_element().min(0.0)
            }
        }
    }

    /// World-space AABB. For a rotated box this is the standard
    /// `|R| * half_extents` trick — exact, and cheaper than transforming the
    /// eight corners.
    pub fn world_aabb(&self, pose: &Isometry) -> Aabb {
        match self {
            Shape::Sphere { radius } => {
                Aabb::from_center_half_extents(pose.translation, Vec3::splat(*radius))
            }
            Shape::Cuboid { half_extents } => {
                let abs_r = Mat3::from_quat(pose.rotation).abs();
                Aabb::from_center_half_extents(pose.translation, abs_r * *half_extents)
            }
            // From the vertices rather than from the local box, which would be
            // the local box *rotated* — correct and loose, and loose bounds are
            // broad-phase pairs that go on to fail the narrow phase.
            Shape::Hull(hull) => {
                let mut bounds = Aabb::EMPTY;
                for vertex in hull.vertices() {
                    bounds = bounds.extended(pose.transform_point(*vertex));
                }
                bounds
            }
        }
    }

    /// Support mapping: the point of the shape furthest along `dir`, in body
    /// space. `dir` need not be normalized. This one function is the entire
    /// interface GJK and EPA need, which is why they generalize to any convex
    /// shape added later.
    pub fn support_local(&self, dir: Vec3) -> Vec3 {
        match self {
            Shape::Sphere { radius } => dir.normalize_or_zero() * *radius,
            Shape::Hull(hull) => hull.support(dir),
            Shape::Cuboid { half_extents: h } => vec3(
                copysign_nonzero(h.x, dir.x),
                copysign_nonzero(h.y, dir.y),
                copysign_nonzero(h.z, dir.z),
            ),
        }
    }

    /// Support mapping in world space.
    pub fn support_world(&self, pose: &Isometry, dir: Vec3) -> Vec3 {
        let local_dir = pose.inverse_transform_direction(dir);
        pose.transform_point(self.support_local(local_dir))
    }
}

/// `f32::copysign` treats -0.0 as negative; for a support function on an exact
/// axis we want a deterministic choice instead of a sign-of-zero coin flip.
#[inline]
fn copysign_nonzero(magnitude: f32, sign_of: f32) -> f32 {
    if sign_of < 0.0 {
        -magnitude
    } else {
        magnitude
    }
}

/// The eight corners of a box, world space. Used by the SAT face-clipping path.
pub fn cuboid_corners(half_extents: Vec3, pose: &Isometry) -> [Vec3; 8] {
    let mut out = [Vec3::ZERO; 8];
    for (i, slot) in out.iter_mut().enumerate() {
        let s = vec3(
            if i & 1 == 0 { -1.0 } else { 1.0 },
            if i & 2 == 0 { -1.0 } else { 1.0 },
            if i & 4 == 0 { -1.0 } else { 1.0 },
        );
        *slot = pose.transform_point(half_extents.mul_element_wise(s));
    }
    out
}

/// Rotation of a pose as a matrix — the box axes, one per column.
pub fn axes_of(rotation: Quat) -> Mat3 {
    Mat3::from_quat(rotation)
}
