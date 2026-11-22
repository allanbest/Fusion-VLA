//! Convex hulls: the collision proxy an imported mesh becomes.
//!
//! PLAN.md §16 lists five creation verbs and **mesh import** is the second of
//! them — "rigid body + collision proxy", with `convex_decompose=True` as an
//! option. Until this existed Fusion could simulate spheres and boxes and
//! nothing else, which put a ceiling on the whole engine that had nothing to
//! do with the solver: no amount of fluid or cloth quality matters if the only
//! thing you can drop into a scene is a primitive.
//!
//! ## Why a hull rather than the triangles
//!
//! §8 requires every shape to carry an SDF, and the narrow phase is built on
//! support functions (see `fusion_collide::narrow::gjk`). A convex hull gives
//! both, exactly:
//!
//! - **Support** is `max over vertices of v·d`, and this is the part worth
//!   understanding — the support function of an arbitrary point set is already
//!   the support function of its convex hull. Collision against a mesh is
//!   therefore collision against its hull whether or not anybody computes one.
//!   Computing it explicitly is what makes that *fast* (tens of vertices
//!   instead of tens of thousands) and *honest* (the shape the solver sees is
//!   the shape you can look at).
//! - **The SDF** needs face planes, which the hull has and a point cloud does
//!   not. Inside a convex body the largest signed plane distance is the exact
//!   distance to the surface; outside it under-estimates near an edge or a
//!   corner, which is stated in [`ConvexHull::signed_distance`] rather than
//!   left to be discovered.
//!
//! **A concave mesh is simulated as its convex hull, and that is a real
//! limitation, not a rounding error.** A teacup holds no water here; a chair
//! is a wedge. §16's answer is convex *decomposition* — several hulls per body
//! — and this is the piece that has to exist first either way.
//!
//! ## The algorithm
//!
//! Incremental hull construction, the quickhull family: start from a
//! tetrahedron of extreme points, then repeatedly take the point furthest
//! outside the current hull, delete every face it can see, and stitch a cone
//! of new faces to the horizon. Degenerate inputs — fewer than four points,
//! all collinear, all coplanar — are the whole difficulty, and each is handled
//! explicitly rather than left to produce a hull with no volume.

use std::sync::Arc;

use fusion_math::{to_f32, to_f64, Aabb, DVec3, Vec3};

/// One triangular face of a hull, with the plane it lies in.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct HullFace {
    /// Vertex indices, counter-clockwise seen from outside.
    pub indices: [u32; 3],
    /// Outward unit normal.
    pub normal: Vec3,
    /// Plane offset: the face lies on `normal · p == offset`.
    pub offset: f32,
}

/// A convex polyhedron.
///
/// Built by [`ConvexHull::from_points`] and then immutable. Bodies share one
/// through an [`Arc`], because a scene with two hundred of the same bolt should
/// hold the geometry once.
#[derive(Clone, Debug, PartialEq)]
pub struct ConvexHull {
    vertices: Vec<Vec3>,
    faces: Vec<HullFace>,
    centroid: Vec3,
    volume: f32,
    bounding_radius: f32,
    bounds: Aabb,
}

/// The hull under construction, in double precision.
///
/// Hull building is where `f32` runs out, and for the reason §9 gives about
/// positions: it accumulates. Every face plane is derived from three vertices,
/// every later point is judged against those planes, and over twelve hundred
/// faces the drift reached two millimetres on a seventy-centimetre ball —
/// enough to leave six of six hundred input points outside a hull that is
/// supposed to contain all of them.
///
/// Building in `f64` and storing `f32` costs nothing at run time. This runs
/// once, when a mesh is imported; the solver never sees it.
struct Builder {
    vertices: Vec<DVec3>,
    faces: Vec<BuilderFace>,
    /// A point known to be inside, which is what orients every face.
    interior: DVec3,
}

#[derive(Clone, Copy)]
struct BuilderFace {
    indices: [u32; 3],
    normal: DVec3,
    offset: f64,
}

/// How far apart two points must be to count as different, and how far off a
/// plane a point must be to count as outside it.
///
/// Absolute rather than relative, and in metres: Fusion is an SI engine (§7),
/// and a tenth of a millimetre is below anything the contact solver resolves.
const EPSILON: f32 = 1e-4;

impl ConvexHull {
    /// The convex hull of a point cloud — the vertices of a mesh, usually.
    ///
    /// Returns `None` where the points do not enclose a volume: fewer than
    /// four of them, or all on one line or one plane. A body needs a volume to
    /// have a mass, so a flat hull is a failure to report rather than a shape
    /// to build.
    pub fn from_points(points: &[Vec3]) -> Option<Self> {
        let cloud: Vec<DVec3> = points.iter().copied().map(to_f64).collect();
        let mut builder = Builder::seed(&cloud)?;

        // Furthest point first, repeatedly, until nothing is left outside.
        //
        // Taking them in input order instead is the obvious loop and it leaves
        // points behind: adding a nearly-flat sliver first produces faces so
        // thin that the horizon walk around them fails, and whatever should
        // have been stitched to them ends up outside a hull that claims to
        // contain everything. Furthest-first is the quickhull ordering and it
        // exists for exactly this reason.
        let mut abandoned = vec![false; cloud.len()];
        loop {
            let mut worst = EPSILON as f64;
            let mut furthest = None;
            for (i, &p) in cloud.iter().enumerate() {
                if abandoned[i] {
                    continue;
                }
                let d = builder.distance(p);
                if d > worst {
                    worst = d;
                    furthest = Some(i);
                }
            }
            let Some(i) = furthest else { break };
            if !builder.add_point(cloud[i]) || builder.distance(cloud[i]) > EPSILON as f64 {
                abandoned[i] = true;
            }
        }

        let hull = builder.finish()?;
        // The definition, checked rather than assumed. A hull that does not
        // contain its own input is not a hull, and the failure mode it causes
        // — a body things pass through from one side — is one of the least
        // debuggable in the engine.
        let contained = points
            .iter()
            .all(|&p| hull.signed_distance(p) <= EPSILON * 10.0);
        contained.then_some(hull)
    }

    /// The hull's vertices, in no particular order.
    pub fn vertices(&self) -> &[Vec3] {
        &self.vertices
    }

    /// The hull's triangular faces.
    pub fn faces(&self) -> &[HullFace] {
        &self.faces
    }

    /// Volume, m³.
    pub fn volume(&self) -> f32 {
        self.volume
    }

    /// Centre of volume, in the hull's own frame.
    pub fn centroid(&self) -> Vec3 {
        self.centroid
    }

    /// Distance from the origin to the furthest vertex.
    pub fn bounding_radius(&self) -> f32 {
        self.bounding_radius
    }

    /// Bounds in the hull's own frame.
    pub fn local_aabb(&self) -> Aabb {
        self.bounds
    }

    /// Furthest point of the hull along `dir`.
    ///
    /// Linear in the vertex count, which is why the hull is computed rather
    /// than the mesh kept: a hull of forty vertices answers this in forty
    /// dot products where the mesh it came from might have taken forty
    /// thousand.
    pub fn support(&self, dir: Vec3) -> Vec3 {
        let mut best = self.vertices[0];
        let mut best_dot = best.dot(dir);
        for &v in &self.vertices[1..] {
            let d = v.dot(dir);
            if d > best_dot {
                best_dot = d;
                best = v;
            }
        }
        best
    }

    /// Signed distance to the surface: negative inside, positive outside.
    ///
    /// The largest signed distance to any face plane. **Exact inside**, which
    /// is where the fluid boundary correction and the particle contacts need
    /// it (§8). Outside, near an edge or a corner, it under-reports — the true
    /// distance there is to the edge, not to either plane — and the error is
    /// bounded by the distance to the nearest face. Contacts are generated by
    /// GJK rather than by this, so the approximation costs nothing where it is
    /// wrong.
    pub fn signed_distance(&self, p: Vec3) -> f32 {
        let mut worst = f32::NEG_INFINITY;
        for face in &self.faces {
            worst = worst.max(p.dot(face.normal) - face.offset);
        }
        worst
    }

    /// Inertia tensor diagonal about the centroid, for a solid of this shape
    /// at the given mass.
    ///
    /// The covariance of the hull, accumulated over the tetrahedra formed by
    /// the centroid and each face — the standard polyhedron integral, in the
    /// form that needs no face-by-face special cases.
    pub fn inertia_diagonal(&self, mass: f32) -> Vec3 {
        if self.volume <= 0.0 {
            return Vec3::splat(mass * 0.4);
        }
        // Second moments of the whole solid, per axis, in units of volume.
        let mut moment = Vec3::ZERO;
        for face in &self.faces {
            let a = self.vertices[face.indices[0] as usize] - self.centroid;
            let b = self.vertices[face.indices[1] as usize] - self.centroid;
            let c = self.vertices[face.indices[2] as usize] - self.centroid;
            // Signed volume of the tetrahedron (centroid, a, b, c).
            let det = a.dot(b.cross(c));
            if det.abs() < 1e-12 {
                continue;
            }
            for k in 0..3 {
                // ∫x² over the tetrahedron (origin, a, b, c) is
                // `det/60 · (a² + b² + c² + ab + bc + ca)` in that component.
                moment[k] += det
                    * (a[k] * a[k]
                        + b[k] * b[k]
                        + c[k] * c[k]
                        + a[k] * b[k]
                        + b[k] * c[k]
                        + c[k] * a[k]);
            }
        }
        // Mass per unit volume, then the 1/60 the integral above leaves out.
        // Getting this constant wrong is invisible in every test that checks a
        // shape against itself and obvious against an analytic box, which is
        // why there is one: the first version divided by 120 and every body in
        // the engine would have spun twice as easily as it should.
        let moment = moment * (mass / self.volume / 60.0);
        // I_xx = ∫(y² + z²), and so on around.
        Vec3::new(
            moment.y + moment.z,
            moment.z + moment.x,
            moment.x + moment.y,
        )
    }
}

impl ConvexHull {
    /// The whole inertia tensor about the centroid, for a solid of this shape
    /// at the given mass, as `[xx, yy, zz, xy, yz, zx]`.
    ///
    /// [`ConvexHull::inertia_diagonal`] is enough for a hull built from a
    /// mesh in its own principal frame. A piece broken off another body is in
    /// no such frame, and its products of inertia are what say which way its
    /// principal axes point.
    pub fn inertia_tensor(&self, mass: f32) -> [f32; 6] {
        if self.volume <= 0.0 {
            let d = mass * 0.4;
            return [d, d, d, 0.0, 0.0, 0.0];
        }
        // Covariance ∫ x_i x_j over the solid, in units of volume.
        let mut c = [[0.0f64; 3]; 3];
        for face in &self.faces {
            let p = face
                .indices
                .map(|i| to_f64(self.vertices[i as usize] - self.centroid));
            let [a, b, d] = p;
            let det = a.dot(b.cross(d));
            for i in 0..3 {
                for j in 0..3 {
                    // ∫ x_i x_j over the tetrahedron (origin, a, b, d).
                    c[i][j] += det / 120.0
                        * (2.0 * (a[i] * a[j] + b[i] * b[j] + d[i] * d[j])
                            + a[i] * b[j]
                            + a[j] * b[i]
                            + a[i] * d[j]
                            + a[j] * d[i]
                            + b[i] * d[j]
                            + b[j] * d[i]);
                }
            }
        }
        let k = mass as f64 / self.volume as f64;
        let trace = c[0][0] + c[1][1] + c[2][2];
        [
            (k * (trace - c[0][0])) as f32,
            (k * (trace - c[1][1])) as f32,
            (k * (trace - c[2][2])) as f32,
            (-k * c[0][1]) as f32,
            (-k * c[1][2]) as f32,
            (-k * c[2][0]) as f32,
        ]
    }
}

impl Builder {
    /// Four extreme, non-degenerate points, as a starting tetrahedron.
    ///
    /// `None` where the cloud encloses no volume: fewer than four points, or
    /// all of them on one line or one plane. A body needs a volume to have a
    /// mass, so a flat hull is a failure to report rather than a shape to
    /// build.
    fn seed(points: &[DVec3]) -> Option<Self> {
        if points.len() < 4 {
            return None;
        }
        let eps = EPSILON as f64;

        // The two furthest apart along the most spread axis.
        let (mut min, mut max) = ([0usize; 3], [0usize; 3]);
        for (i, p) in points.iter().enumerate() {
            for k in 0..3 {
                if p[k] < points[min[k]][k] {
                    min[k] = i;
                }
                if p[k] > points[max[k]][k] {
                    max[k] = i;
                }
            }
        }
        let axis = (0..3)
            .max_by(|&a, &b| {
                let sa = points[max[a]][a] - points[min[a]][a];
                let sb = points[max[b]][b] - points[min[b]][b];
                sa.total_cmp(&sb)
            })
            .unwrap();
        let (a, b) = (points[min[axis]], points[max[axis]]);
        if (b - a).length() < eps {
            return None; // every point in the same place
        }

        // Furthest from the line ab, then furthest from the plane abc.
        let dir = (b - a).normalize_or_zero();
        let c = *points
            .iter()
            .max_by(|p, q| {
                let dp = (**p - a).reject_from_unit(dir).length();
                let dq = (**q - a).reject_from_unit(dir).length();
                dp.total_cmp(&dq)
            })
            .unwrap();
        if (c - a).reject_from_unit(dir).length() < eps {
            return None; // collinear
        }

        let normal = (b - a).cross(c - a).normalize_or_zero();
        let d = *points
            .iter()
            .max_by(|p, q| {
                let dp = (**p - a).dot(normal).abs();
                let dq = (**q - a).dot(normal).abs();
                dp.total_cmp(&dq)
            })
            .unwrap();
        if (d - a).dot(normal).abs() < eps {
            return None; // coplanar
        }

        let mut hull = Self {
            vertices: vec![a, b, c, d],
            faces: Vec::new(),
            interior: (a + b + c + d) * 0.25,
        };
        for (i, j, k) in [(0, 1, 2), (0, 1, 3), (0, 2, 3), (1, 2, 3)] {
            if let Some(face) = hull.make_face(i, j, k) {
                hull.faces.push(face);
            }
        }
        (hull.faces.len() == 4).then_some(hull)
    }

    /// Largest signed distance to any face plane: positive outside.
    fn distance(&self, p: DVec3) -> f64 {
        self.faces.iter().fold(f64::NEG_INFINITY, |worst, f| {
            worst.max(p.dot(f.normal) - f.offset)
        })
    }

    /// A face through three vertices, oriented so its normal points away from
    /// the interior.
    fn make_face(&self, a: u32, b: u32, c: u32) -> Option<BuilderFace> {
        let (pa, pb, pc) = (
            self.vertices[a as usize],
            self.vertices[b as usize],
            self.vertices[c as usize],
        );
        let normal = (pb - pa).cross(pc - pa);
        let length = normal.length();
        if length < 1e-14 {
            return None;
        }
        let normal = normal / length;
        let offset = normal.dot(pa);
        if normal.dot(self.interior) - offset > 0.0 {
            Some(BuilderFace {
                indices: [a, c, b],
                normal: -normal,
                offset: -offset,
            })
        } else {
            Some(BuilderFace {
                indices: [a, b, c],
                normal,
                offset,
            })
        }
    }

    /// Add one point, deleting the faces it can see and stitching a cone of
    /// new faces onto the horizon they leave behind.
    ///
    /// Returns false where the point was not added, which the caller uses to
    /// stop offering it.
    fn add_point(&mut self, point: DVec3) -> bool {
        let eps = EPSILON as f64;
        // A point already on the hull is not a new vertex. Without this the
        // same input can be pushed twice — a vertex sits on its own faces, and
        // once the plane arithmetic drifts past the epsilon it reads as
        // outside and gets a second cone of faces built around it. The hull
        // stays convex and every check of the definition still passes; it
        // simply has more vertices than it was given points, and the duplicate
        // faces are each counted again in the volume.
        if self
            .vertices
            .iter()
            .any(|v| (*v - point).length_squared() < eps * eps)
        {
            return false;
        }

        // Visible means "in front of at all", not "in front of by an epsilon".
        //
        // This is the difference between a hull and a nearly-hull. A face the
        // new point sits almost exactly on is not visible under a `> eps`
        // test, so it survives — and then the cone stitched onto the horizon
        // lays a second face across the same ground. The result has more faces
        // than Euler allows, and the vertices around that patch end up
        // fractionally outside the surface that is supposed to contain them.
        // Six of six hundred points on a sphere, two millimetres out.
        //
        // Worth recording precisely because it looks like a precision problem
        // and is not one: building the whole hull in `f64` instead changed
        // nothing at all — same six points, same two millimetres, to the
        // digit. It is a degeneracy in which faces get deleted, and no amount
        // of precision fixes the wrong set.
        let visible: Vec<bool> = self
            .faces
            .iter()
            .map(|f| point.dot(f.normal) - f.offset > 0.0)
            .collect();
        if !visible.iter().any(|&v| v) {
            return false;
        }

        // The horizon: the boundary of the visible region.
        //
        // Found from the visible faces alone. Every directed edge interior to
        // that region appears twice, once each way, because the surface is
        // consistently wound; an edge appearing one way only is on the
        // boundary. Asking instead whether some *hidden* face carries the
        // reversed edge is the same statement for a perfect mesh and not the
        // same at all once one sliver makes a normal unreliable — it drops
        // horizon edges, the cone comes out with holes, and points end up
        // outside a hull that refused to notice.
        let mut edges: Vec<(u32, u32)> = Vec::new();
        for (f, face) in self.faces.iter().enumerate() {
            if visible[f] {
                for e in 0..3 {
                    edges.push((face.indices[e], face.indices[(e + 1) % 3]));
                }
            }
        }
        let horizon: Vec<(u32, u32)> = edges
            .iter()
            .copied()
            .filter(|&(a, b)| !edges.contains(&(b, a)))
            .collect();
        if horizon.is_empty() {
            return false;
        }

        let apex = self.vertices.len() as u32;
        self.vertices.push(point);
        let mut kept: Vec<BuilderFace> = self
            .faces
            .iter()
            .zip(&visible)
            .filter(|(_, &v)| !v)
            .map(|(f, _)| *f)
            .collect();
        for (a, b) in horizon {
            if let Some(face) = self.make_face(a, b, apex) {
                kept.push(face);
            }
        }
        self.faces = kept;
        true
    }

    /// Drop the vertices no face refers to, compute the mass properties, and
    /// hand back the finished single-precision hull.
    fn finish(mut self) -> Option<ConvexHull> {
        // Two faces over the same three vertices are one face, and the volume
        // sum counts whatever is in this list.
        self.faces.sort_unstable_by_key(|f| {
            let mut i = f.indices;
            i.sort_unstable();
            i
        });
        self.faces.dedup_by(|a, b| {
            let (mut x, mut y) = (a.indices, b.indices);
            x.sort_unstable();
            y.sort_unstable();
            x == y
        });

        let mut used = vec![false; self.vertices.len()];
        for face in &self.faces {
            for i in face.indices {
                used[i as usize] = true;
            }
        }
        let mut remap = vec![u32::MAX; self.vertices.len()];
        let mut kept = Vec::with_capacity(self.vertices.len());
        for (i, &keep) in used.iter().enumerate() {
            if keep {
                remap[i] = kept.len() as u32;
                kept.push(self.vertices[i]);
            }
        }

        // Volume and centroid by the divergence theorem, over the tetrahedra
        // from the origin to each face.
        let mut volume = 0.0f64;
        let mut centroid = DVec3::ZERO;
        for face in &self.faces {
            let a = self.vertices[face.indices[0] as usize];
            let b = self.vertices[face.indices[1] as usize];
            let c = self.vertices[face.indices[2] as usize];
            let det = a.dot(b.cross(c));
            volume += det;
            centroid += (a + b + c) * det;
        }
        if volume <= 0.0 {
            return None;
        }

        let faces = self
            .faces
            .iter()
            .map(|f| HullFace {
                indices: [
                    remap[f.indices[0] as usize],
                    remap[f.indices[1] as usize],
                    remap[f.indices[2] as usize],
                ],
                normal: to_f32(f.normal),
                offset: f.offset as f32,
            })
            .collect();
        let vertices: Vec<Vec3> = kept.iter().copied().map(to_f32).collect();
        let bounding_radius = vertices.iter().fold(0.0f32, |m, v| m.max(v.length()));
        let bounds = Aabb::from_points(&vertices);

        Some(ConvexHull {
            vertices,
            faces,
            centroid: to_f32(centroid / (4.0 * volume)),
            volume: (volume / 6.0) as f32,
            bounding_radius,
            bounds,
        })
    }
}

/// A hull ready to be shared between bodies.
pub type SharedHull = Arc<ConvexHull>;

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_full_tensor_agrees_with_the_diagonal_and_rotates() {
        let corners: Vec<Vec3> = (0..8)
            .map(|i| {
                Vec3::new(
                    if i & 1 == 0 { -1.0 } else { 1.0 },
                    if i & 2 == 0 { -0.5 } else { 0.5 },
                    if i & 4 == 0 { -0.25 } else { 0.25 },
                )
            })
            .collect();
        let hull = ConvexHull::from_points(&corners).unwrap();
        let t = hull.inertia_tensor(6.0);
        let d = hull.inertia_diagonal(6.0);
        for k in 0..3 {
            assert!((t[k] - d[k]).abs() < 1e-4, "{t:?} {d:?}");
        }
        assert!(t[3].abs() + t[4].abs() + t[5].abs() < 1e-4, "{t:?}");
        // Turned 45 degrees about z, the x-y product appears.
        let q = fusion_math::Quat::from_rotation_z(std::f32::consts::FRAC_PI_4);
        let turned: Vec<Vec3> = corners.iter().map(|p| q.rotate(*p)).collect();
        let t = ConvexHull::from_points(&turned)
            .unwrap()
            .inertia_tensor(6.0);
        assert!(t[3].abs() > 0.1, "{t:?}");
    }
}
