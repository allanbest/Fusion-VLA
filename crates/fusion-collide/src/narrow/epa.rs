//! EPA — expanding polytope algorithm.
//!
//! GJK answers *whether* two convex shapes overlap; EPA answers *by how much*
//! and *in which direction*. It grows the terminating GJK simplex outwards
//! through the Minkowski difference until it reaches the boundary face nearest
//! the origin. That face gives the minimum translation vector.
//!
//! Sign convention throughout Fusion: the contact normal points **from A
//! towards B**, and `separation` is negative when penetrating.

use fusion_core::Shape;
use fusion_math::{Isometry, Vec3};

use super::gjk::{support, Simplex, SupportPoint};
use crate::contact::WorldContact;

const MAX_ITERATIONS: usize = 64;
const MAX_FACES: usize = 256;
const TOLERANCE: f32 = 1e-4;

#[derive(Clone, Copy, Debug)]
struct Face {
    v: [usize; 3],
    /// Outward unit normal.
    normal: Vec3,
    /// Distance from the origin along `normal`.
    distance: f32,
}

/// Penetration depth and witness points for a pair known to overlap.
///
/// `simplex` must be the tetrahedron GJK terminated on. A lower-dimensional
/// simplex (a touching contact) is inflated first.
pub fn penetration(
    shape_a: &Shape,
    pose_a: &Isometry,
    shape_b: &Shape,
    pose_b: &Isometry,
    simplex: &Simplex,
) -> Option<WorldContact> {
    let mut vertices: Vec<SupportPoint> = simplex.points[..simplex.len].to_vec();
    if !inflate_to_tetrahedron(shape_a, pose_a, shape_b, pose_b, &mut vertices) {
        return None;
    }

    let mut faces = initial_faces(&vertices)?;

    for _ in 0..MAX_ITERATIONS {
        // The face nearest the origin is the current best answer.
        let (best_index, best) = faces
            .iter()
            .enumerate()
            .min_by(|a, b| a.1.distance.total_cmp(&b.1.distance))
            .map(|(i, f)| (i, *f))?;

        let p = support(shape_a, pose_a, shape_b, pose_b, best.normal);
        let reach = p.v.dot(best.normal);

        if reach - best.distance < TOLERANCE || faces.len() >= MAX_FACES {
            return Some(contact_from_face(&vertices, &best));
        }

        // Delete every face this new point can see, and re-close the hole.
        let new_index = vertices.len();
        vertices.push(p);
        let horizon = carve(&mut faces, &vertices, p.v);
        if horizon.is_empty() {
            return Some(contact_from_face(
                &vertices,
                &faces[best_index.min(faces.len() - 1)],
            ));
        }
        for (i, j) in horizon {
            if let Some(f) = make_face(&vertices, [i, j, new_index]) {
                faces.push(f);
            }
        }
        if faces.is_empty() {
            return None;
        }
    }
    None
}

/// GJK can terminate on a point, segment or triangle when the shapes merely
/// touch. EPA needs a closed volume, so add support points until there is one.
fn inflate_to_tetrahedron(
    shape_a: &Shape,
    pose_a: &Isometry,
    shape_b: &Shape,
    pose_b: &Isometry,
    v: &mut Vec<SupportPoint>,
) -> bool {
    if v.is_empty() {
        v.push(support(shape_a, pose_a, shape_b, pose_b, Vec3::X));
    }
    if v.len() == 1 {
        for dir in [Vec3::X, -Vec3::X, Vec3::Y, -Vec3::Y, Vec3::Z, -Vec3::Z] {
            let p = support(shape_a, pose_a, shape_b, pose_b, dir);
            if (p.v - v[0].v).length_squared() > TOLERANCE {
                v.push(p);
                break;
            }
        }
    }
    if v.len() == 2 {
        let axis = (v[1].v - v[0].v).normalize_or_zero();
        let seed = axis.any_orthonormal();
        for k in 0..6 {
            let angle = k as f32 * std::f32::consts::FRAC_PI_3;
            let dir = fusion_math::Quat::from_axis_angle(axis, angle).rotate(seed);
            let p = support(shape_a, pose_a, shape_b, pose_b, dir);
            if (p.v - v[0].v).cross(p.v - v[1].v).length_squared() > TOLERANCE {
                v.push(p);
                break;
            }
        }
    }
    if v.len() == 3 {
        let n = (v[1].v - v[0].v).cross(v[2].v - v[0].v).normalize_or_zero();
        for dir in [n, -n] {
            let p = support(shape_a, pose_a, shape_b, pose_b, dir);
            if (p.v - v[0].v).dot(n).abs() > TOLERANCE {
                v.push(p);
                break;
            }
        }
    }
    v.len() == 4
}

/// The four faces of the starting tetrahedron, each wound so its normal points
/// away from the interior.
fn initial_faces(v: &[SupportPoint]) -> Option<Vec<Face>> {
    let mut faces = Vec::with_capacity(8);
    for tri in [[0usize, 1, 2], [0, 3, 1], [0, 2, 3], [1, 3, 2]] {
        faces.push(make_face(v, tri)?);
    }
    Some(faces)
}

/// Build a face with an outward normal. "Outward" is defined against the
/// origin, which is inside the polytope by construction.
fn make_face(v: &[SupportPoint], tri: [usize; 3]) -> Option<Face> {
    let (a, b, c) = (v[tri[0]].v, v[tri[1]].v, v[tri[2]].v);
    let mut normal = (b - a).cross(c - a);
    let len_sq = normal.length_squared();
    if len_sq < 1e-14 {
        return None; // degenerate sliver
    }
    normal /= len_sq.sqrt();
    let mut distance = normal.dot(a);
    let mut tri = tri;
    if distance < 0.0 {
        normal = -normal;
        distance = -distance;
        tri.swap(1, 2);
    }
    Some(Face {
        v: tri,
        normal,
        distance,
    })
}

/// Remove every face visible from `p` and return the horizon: the edges that
/// bounded exactly one removed face.
fn carve(faces: &mut Vec<Face>, v: &[SupportPoint], p: Vec3) -> Vec<(usize, usize)> {
    let mut edges: Vec<(usize, usize)> = Vec::new();
    let mut kept = Vec::with_capacity(faces.len());

    for f in faces.iter() {
        let visible = f.normal.dot(p - v[f.v[0]].v) > 1e-7;
        if !visible {
            kept.push(*f);
            continue;
        }
        for k in 0..3 {
            let e = (f.v[k], f.v[(k + 1) % 3]);
            // An edge shared by two removed faces is interior to the hole and
            // cancels; only unmatched edges form the horizon.
            if let Some(pos) = edges.iter().position(|&(a, b)| a == e.1 && b == e.0) {
                edges.swap_remove(pos);
            } else {
                edges.push(e);
            }
        }
    }
    *faces = kept;
    edges
}

/// Turn the closing face into a world-space contact by interpolating the
/// witness points at the origin's projection onto that face.
fn contact_from_face(v: &[SupportPoint], f: &Face) -> WorldContact {
    let (a, b, c) = (v[f.v[0]], v[f.v[1]], v[f.v[2]]);
    let projected = f.normal * f.distance;
    let w = barycentric(projected, a.v, b.v, c.v);
    let point_a = a.a * w[0] + b.a * w[1] + c.a * w[2];
    let point_b = a.b * w[0] + b.b * w[1] + c.b * w[2];
    WorldContact {
        point_a,
        point_b,
        normal: f.normal,
        separation: -f.distance,
    }
}

fn barycentric(p: Vec3, a: Vec3, b: Vec3, c: Vec3) -> [f32; 3] {
    let (v0, v1, v2) = (b - a, c - a, p - a);
    let d00 = v0.dot(v0);
    let d01 = v0.dot(v1);
    let d11 = v1.dot(v1);
    let d20 = v2.dot(v0);
    let d21 = v2.dot(v1);
    let denom = d00 * d11 - d01 * d01;
    if denom.abs() < 1e-12 {
        return [1.0, 0.0, 0.0];
    }
    let v = (d11 * d20 - d01 * d21) / denom;
    let w = (d00 * d21 - d01 * d20) / denom;
    [1.0 - v - w, v, w]
}
