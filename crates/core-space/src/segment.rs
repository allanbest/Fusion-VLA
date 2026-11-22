//! Closest points on finite segments, including collapsed and parallel ones.

use core_math::Vec3;

/// Closest points on `ab` and `cd`, and their parameters `(s, t)` in `[0, 1]`.
/// Tests the four boundary minima and the interior stationary point of the
/// squared distance. Cross products avoid subtracting nearly equal squared
/// lengths when the segments are almost parallel. No angular threshold
/// discards valid interior minima of long, nearly parallel segments.
pub fn closest_segments(a: Vec3, b: Vec3, c: Vec3, d: Vec3) -> (Vec3, Vec3, [f64; 2]) {
    let u = b - a;
    let v = d - c;
    let project = |p: Vec3, start: Vec3, delta: Vec3| {
        let length2 = delta.length_squared();
        if length2 > 0.0 {
            ((p - start).dot(delta) / length2).clamp(0.0, 1.0)
        } else {
            0.0
        }
    };
    let mut best = (a, c, [0.0, 0.0]);
    let mut distance2 = f64::INFINITY;
    let mut consider = |s: f64, t: f64| {
        let p = a + u * s;
        let q = c + v * t;
        let candidate = (p - q).length_squared();
        if candidate < distance2 {
            best = (p, q, [s, t]);
            distance2 = candidate;
        }
    };
    consider(0.0, project(a, c, v));
    consider(1.0, project(b, c, v));
    consider(project(c, a, u), 0.0);
    consider(project(d, a, u), 1.0);
    let cross = u.cross(v);
    let denominator = cross.length_squared();
    if denominator > 0.0 {
        let w = c - a;
        let s = w.cross(v).dot(cross) / denominator;
        let t = w.cross(u).dot(cross) / denominator;
        if (0.0..=1.0).contains(&s) && (0.0..=1.0).contains(&t) {
            consider(s, t);
        }
    }
    best
}
