//! GJK — the generic convex distance and intersection test.
//!
//! GJK needs exactly one thing from a shape: a support function. That is why
//! PLAN.md §8 can promise convex hulls and decompositions later without the
//! narrow phase changing — a hull that answers `support_local` drops straight
//! into this code.
//!
//! PLAN.md §10 and §32 both flag narrow phase as the first real bottleneck, so
//! the analytic fast paths in `primitives` handle the common pairs and this is
//! the general fallback.

use fusion_core::Shape;
use fusion_math::{Isometry, Vec3};

/// A vertex of the Minkowski difference `A - B`, carrying the two world-space
/// points it came from so that witness points survive to the end.
#[derive(Clone, Copy, Debug, Default)]
pub struct SupportPoint {
    /// The Minkowski difference point, `a - b`.
    pub v: Vec3,
    /// The supporting point on A that produced it.
    pub a: Vec3,
    /// The supporting point on B.
    pub b: Vec3,
}

/// Support point of the Minkowski difference along `dir`.
#[inline]
pub fn support(
    shape_a: &Shape,
    pose_a: &Isometry,
    shape_b: &Shape,
    pose_b: &Isometry,
    dir: Vec3,
) -> SupportPoint {
    let a = shape_a.support_world(pose_a, dir);
    let b = shape_b.support_world(pose_b, -dir);
    SupportPoint { v: a - b, a, b }
}

/// Up to four support points bounding the origin.
#[derive(Clone, Copy, Debug, Default)]
pub struct Simplex {
    /// The vertices, of which only the first `len` are meaningful.
    pub points: [SupportPoint; 4],
    /// How many of them there are: 1 for a point, 4 for a tetrahedron.
    pub len: usize,
}

impl Simplex {
    fn push_front(&mut self, p: SupportPoint) {
        self.points.copy_within(0..3, 1);
        self.points[0] = p;
        self.len = (self.len + 1).min(4);
    }

    fn set(&mut self, pts: &[SupportPoint]) {
        self.len = pts.len();
        self.points[..pts.len()].copy_from_slice(pts);
    }
}

/// What one GJK query concluded.
#[derive(Clone, Copy, Debug)]
pub enum GjkResult {
    /// The shapes overlap. The simplex encloses the origin and seeds EPA.
    Intersecting(Simplex),
    /// The shapes are apart, with the closest points found.
    Separated {
        /// Distance between the closest points.
        distance: f32,
        /// Closest point on A.
        point_a: Vec3,
        /// Closest point on B.
        point_b: Vec3,
    },
    /// Neither converged within the iteration budget. Treated as "no contact"
    /// by callers; it should not happen for the shapes v0.1 supports.
    Indeterminate,
}

const MAX_ITERATIONS: usize = 64;
const TOLERANCE: f32 = 1e-6;

/// Boolean-and-simplex GJK. This is the entry point when the caller only needs
/// to know whether the shapes overlap, and wants a simplex for EPA if they do.
pub fn intersect(
    shape_a: &Shape,
    pose_a: &Isometry,
    shape_b: &Shape,
    pose_b: &Isometry,
) -> GjkResult {
    // Any non-degenerate starting direction works; the vector between centres
    // is the one most likely to be nearly right.
    let mut dir = pose_a.translation - pose_b.translation;
    if dir.length_squared() < TOLERANCE {
        dir = Vec3::X;
    }

    let mut simplex = Simplex::default();
    simplex.push_front(support(shape_a, pose_a, shape_b, pose_b, dir));
    dir = -simplex.points[0].v;

    for _ in 0..MAX_ITERATIONS {
        if dir.length_squared() < TOLERANCE * TOLERANCE {
            // The origin sits on the current feature: touching contact.
            return GjkResult::Intersecting(simplex);
        }
        let p = support(shape_a, pose_a, shape_b, pose_b, dir);
        if p.v.dot(dir) < 0.0 {
            // The furthest point along `dir` did not reach the origin, so no
            // point of the Minkowski difference does: the shapes are apart.
            return separated(shape_a, pose_a, shape_b, pose_b);
        }
        simplex.push_front(p);
        if do_simplex(&mut simplex, &mut dir) {
            return GjkResult::Intersecting(simplex);
        }
    }
    GjkResult::Indeterminate
}

/// Reduce the simplex to the feature closest to the origin and update `dir` to
/// point from that feature towards the origin. Returns true when the simplex
/// encloses the origin.
fn do_simplex(s: &mut Simplex, dir: &mut Vec3) -> bool {
    match s.len {
        2 => {
            let (a, b) = (s.points[0], s.points[1]);
            let ab = b.v - a.v;
            let ao = -a.v;
            if ab.dot(ao) > 0.0 {
                *dir = ab.cross(ao).cross(ab);
                if dir.length_squared() < TOLERANCE * TOLERANCE {
                    // Origin lies on the segment.
                    *dir = ab.any_orthonormal();
                }
            } else {
                s.set(&[a]);
                *dir = ao;
            }
            false
        }
        3 => {
            let (a, b, c) = (s.points[0], s.points[1], s.points[2]);
            let (ab, ac, ao) = (b.v - a.v, c.v - a.v, -a.v);
            let abc = ab.cross(ac);

            if abc.cross(ac).dot(ao) > 0.0 {
                if ac.dot(ao) > 0.0 {
                    s.set(&[a, c]);
                    *dir = ac.cross(ao).cross(ac);
                } else {
                    s.set(&[a, b]);
                    return do_simplex(s, dir);
                }
            } else if ab.cross(abc).dot(ao) > 0.0 {
                s.set(&[a, b]);
                return do_simplex(s, dir);
            } else if abc.dot(ao) > 0.0 {
                *dir = abc;
            } else {
                s.set(&[a, c, b]);
                *dir = -abc;
            }
            false
        }
        4 => {
            let (a, b, c, d) = (s.points[0], s.points[1], s.points[2], s.points[3]);
            let ao = -a.v;
            let (ab, ac, ad) = (b.v - a.v, c.v - a.v, d.v - a.v);

            let abc = ab.cross(ac);
            let acd = ac.cross(ad);
            let adb = ad.cross(ab);

            if abc.dot(ao) > 0.0 {
                s.set(&[a, b, c]);
                return do_simplex(s, dir);
            }
            if acd.dot(ao) > 0.0 {
                s.set(&[a, c, d]);
                return do_simplex(s, dir);
            }
            if adb.dot(ao) > 0.0 {
                s.set(&[a, d, b]);
                return do_simplex(s, dir);
            }
            // Origin is behind all three faces, so it is inside.
            true
        }
        _ => false,
    }
}

/// Closest-point GJK, run when the boolean test proves separation. Iterates on
/// the closest point of the simplex to the origin until it stops improving.
fn separated(shape_a: &Shape, pose_a: &Isometry, shape_b: &Shape, pose_b: &Isometry) -> GjkResult {
    let mut dir = pose_a.translation - pose_b.translation;
    if dir.length_squared() < TOLERANCE {
        dir = Vec3::X;
    }
    let mut simplex = Simplex::default();
    simplex.push_front(support(shape_a, pose_a, shape_b, pose_b, dir));
    let mut best = f32::INFINITY;

    for _ in 0..MAX_ITERATIONS {
        let (closest, bary) = closest_point_on_simplex(&simplex);
        let dist = closest.length();
        if dist < TOLERANCE {
            return GjkResult::Intersecting(simplex);
        }
        if best - dist < TOLERANCE {
            let (a, b) = witness_points(&simplex, &bary);
            return GjkResult::Separated {
                distance: dist,
                point_a: a,
                point_b: b,
            };
        }
        best = dist;
        let p = support(shape_a, pose_a, shape_b, pose_b, -closest);
        // Reduce to the supporting feature before adding, so the simplex never
        // accumulates interior points.
        reduce_to_feature(&mut simplex, &bary);
        simplex.push_front(p);
    }
    GjkResult::Indeterminate
}

/// Barycentric weights over the current simplex points.
type Barycentric = [f32; 4];

fn closest_point_on_simplex(s: &Simplex) -> (Vec3, Barycentric) {
    let mut w = [0.0f32; 4];
    match s.len {
        1 => {
            w[0] = 1.0;
            (s.points[0].v, w)
        }
        2 => {
            let (a, b) = (s.points[0].v, s.points[1].v);
            let ab = b - a;
            let t = fusion_math::clamp((-a).dot(ab) / ab.length_squared().max(TOLERANCE), 0.0, 1.0);
            w[0] = 1.0 - t;
            w[1] = t;
            (a + ab * t, w)
        }
        3 => {
            let (p, bary) = closest_on_triangle(s.points[0].v, s.points[1].v, s.points[2].v);
            w[..3].copy_from_slice(&bary);
            (p, w)
        }
        _ => {
            // Closest point on a tetrahedron: take the best of its four faces.
            let idx = [[0, 1, 2], [0, 2, 3], [0, 3, 1], [1, 3, 2]];
            let mut best_p = Vec3::ZERO;
            let mut best_d = f32::INFINITY;
            for face in idx {
                let (p, bary) = closest_on_triangle(
                    s.points[face[0]].v,
                    s.points[face[1]].v,
                    s.points[face[2]].v,
                );
                let d = p.length_squared();
                if d < best_d {
                    best_d = d;
                    best_p = p;
                    w = [0.0; 4];
                    for (k, &fi) in face.iter().enumerate() {
                        w[fi] = bary[k];
                    }
                }
            }
            (best_p, w)
        }
    }
}

/// Closest point to the origin on triangle `abc`, with barycentric weights.
fn closest_on_triangle(a: Vec3, b: Vec3, c: Vec3) -> (Vec3, [f32; 3]) {
    let (ab, ac, ao) = (b - a, c - a, -a);
    let d1 = ab.dot(ao);
    let d2 = ac.dot(ao);
    if d1 <= 0.0 && d2 <= 0.0 {
        return (a, [1.0, 0.0, 0.0]);
    }
    let bo = -b;
    let d3 = ab.dot(bo);
    let d4 = ac.dot(bo);
    if d3 >= 0.0 && d4 <= d3 {
        return (b, [0.0, 1.0, 0.0]);
    }
    let vc = d1 * d4 - d3 * d2;
    if vc <= 0.0 && d1 >= 0.0 && d3 <= 0.0 {
        let v = d1 / (d1 - d3);
        return (a + ab * v, [1.0 - v, v, 0.0]);
    }
    let co = -c;
    let d5 = ab.dot(co);
    let d6 = ac.dot(co);
    if d6 >= 0.0 && d5 <= d6 {
        return (c, [0.0, 0.0, 1.0]);
    }
    let vb = d5 * d2 - d1 * d6;
    if vb <= 0.0 && d2 >= 0.0 && d6 <= 0.0 {
        let w = d2 / (d2 - d6);
        return (a + ac * w, [1.0 - w, 0.0, w]);
    }
    let va = d3 * d6 - d5 * d4;
    if va <= 0.0 && (d4 - d3) >= 0.0 && (d5 - d6) >= 0.0 {
        let w = (d4 - d3) / ((d4 - d3) + (d5 - d6));
        return (b + (c - b) * w, [0.0, 1.0 - w, w]);
    }
    let denom = 1.0 / (va + vb + vc);
    let v = vb * denom;
    let w = vc * denom;
    (a + ab * v + ac * w, [1.0 - v - w, v, w])
}

fn witness_points(s: &Simplex, w: &Barycentric) -> (Vec3, Vec3) {
    let mut a = Vec3::ZERO;
    let mut b = Vec3::ZERO;
    for (point, &weight) in s.points[..s.len].iter().zip(w.iter()) {
        a += point.a * weight;
        b += point.b * weight;
    }
    (a, b)
}

/// Drop simplex vertices with zero weight — they are not part of the closest
/// feature and only degrade conditioning.
fn reduce_to_feature(s: &mut Simplex, w: &Barycentric) {
    let mut kept = [SupportPoint::default(); 4];
    let mut n = 0;
    for (point, &weight) in s.points[..s.len].iter().zip(w.iter()) {
        if weight > 1e-7 && n < 3 {
            kept[n] = *point;
            n += 1;
        }
    }
    if n == 0 {
        kept[0] = s.points[0];
        n = 1;
    }
    s.len = n;
    s.points = kept;
}
