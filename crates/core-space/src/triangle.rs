//! Distances to triangles.

use crate::closest_segments;
use core_math::Vec3;

/// The point of triangle `abc` closest to `p`, and its barycentric weights
/// `(wa, wb, wc)` (non-negative, summing to one). Works by the Voronoi
/// region of `p` (Ericson, *Real-Time Collision Detection*, 5.1.5), exact for
/// degenerate triangles too.
pub fn closest_on_triangle(p: Vec3, a: Vec3, b: Vec3, c: Vec3) -> (Vec3, [f64; 3]) {
    let ab = b - a;
    let ac = c - a;
    if ab.cross(ac).length_squared() == 0.0 {
        // Voronoi formulas divide by zero for collapsed edges/faces.
        // A degenerate triangle is the union of its three segments.
        let vertices = [a, b, c];
        let mut result = (a, [1.0, 0.0, 0.0]);
        let mut best = f64::INFINITY;
        for [i, j] in [[0, 1], [1, 2], [2, 0]] {
            let (_, q, [_, t]) = closest_segments(p, p, vertices[i], vertices[j]);
            let distance2 = (p - q).length_squared();
            if distance2 < best {
                best = distance2;
                let mut weights = [0.0; 3];
                weights[i] = 1.0 - t;
                weights[j] = t;
                result = (q, weights);
            }
        }
        return result;
    }
    let ap = p - a;
    let d1 = ab.dot(ap);
    let d2 = ac.dot(ap);
    if d1 <= 0.0 && d2 <= 0.0 {
        return (a, [1.0, 0.0, 0.0]);
    }
    let bp = p - b;
    let d3 = ab.dot(bp);
    let d4 = ac.dot(bp);
    if d3 >= 0.0 && d4 <= d3 {
        return (b, [0.0, 1.0, 0.0]);
    }
    let vc = d1 * d4 - d3 * d2;
    if vc <= 0.0 && d1 >= 0.0 && d3 <= 0.0 {
        let v = d1 / (d1 - d3);
        return (a + ab * v, [1.0 - v, v, 0.0]);
    }
    let cp = p - c;
    let d5 = ab.dot(cp);
    let d6 = ac.dot(cp);
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
    let denominator = 1.0 / (va + vb + vc);
    let v = vb * denominator;
    let w = vc * denominator;
    (a + ab * v + ac * w, [1.0 - v - w, v, w])
}
