//! The hull builder, against shapes whose answers are known without it.
//!
//! A convex hull is the kind of code that looks right and is wrong by a
//! sliver: a face oriented inward, a vertex left outside, a volume that is
//! close enough to pass a glance. Every test here checks against something
//! analytic — a cube's volume, a sphere's radius, the definition of convexity
//! — rather than against a previous run.

use fusion_core::ConvexHull;
use fusion_math::{vec3, Vec3};

/// Deterministic LCG, so a failure is reproducible.
struct Rng(u64);

impl Rng {
    fn unit(&mut self) -> f32 {
        self.0 = self
            .0
            .wrapping_mul(6_364_136_223_846_793_005)
            .wrapping_add(1);
        ((self.0 >> 40) as f32) / (1u32 << 24) as f32
    }
    fn range(&mut self, lo: f32, hi: f32) -> f32 {
        lo + self.unit() * (hi - lo)
    }
    fn point_in_ball(&mut self, radius: f32) -> Vec3 {
        loop {
            let p = vec3(
                self.range(-1.0, 1.0),
                self.range(-1.0, 1.0),
                self.range(-1.0, 1.0),
            );
            if p.length_squared() <= 1.0 && p.length_squared() > 1e-6 {
                return p * radius;
            }
        }
    }
}

fn cube_corners(half: f32) -> Vec<Vec3> {
    let mut out = Vec::new();
    for x in [-half, half] {
        for y in [-half, half] {
            for z in [-half, half] {
                out.push(vec3(x, y, z));
            }
        }
    }
    out
}

/// Every input point is inside the hull, and every face points outward.
///
/// This is the definition of a convex hull, and it is the assertion that
/// catches an inverted face — which otherwise shows up much later as a body
/// that objects fall through from one side.
fn assert_is_a_hull_of(points: &[Vec3], hull: &ConvexHull) {
    for (i, &p) in points.iter().enumerate() {
        assert!(
            hull.signed_distance(p) <= 1e-3,
            "input point {i} at {p:?} is outside the hull by {}",
            hull.signed_distance(p)
        );
    }
    for (f, face) in hull.faces().iter().enumerate() {
        assert!(
            (face.normal.length() - 1.0).abs() < 1e-4,
            "face {f} has a normal of length {}",
            face.normal.length()
        );
        // Outward: the centroid is behind every face.
        assert!(
            hull.centroid().dot(face.normal) - face.offset < 0.0,
            "face {f} points inward"
        );
    }
    assert!(hull.volume() > 0.0, "the hull has no volume");
}

#[test]
fn a_cube_is_a_cube() {
    let points = cube_corners(0.5);
    let hull = ConvexHull::from_points(&points).expect("eight corners make a cube");
    assert_is_a_hull_of(&points, &hull);

    assert_eq!(hull.vertices().len(), 8, "a cube has eight vertices");
    assert_eq!(hull.faces().len(), 12, "and twelve triangles");
    assert!(
        (hull.volume() - 1.0).abs() < 1e-4,
        "a unit cube has unit volume, got {}",
        hull.volume()
    );
    assert!(hull.centroid().length() < 1e-4, "centred on the origin");
    // The SDF is exact inside a convex body: the centre of a unit cube is half
    // a metre from the nearest face.
    assert!(
        (hull.signed_distance(Vec3::ZERO) + 0.5).abs() < 1e-4,
        "centre distance was {}",
        hull.signed_distance(Vec3::ZERO)
    );
}

/// Interior points change nothing. The hull of a cube's corners and the hull
/// of a cube's corners plus a thousand points inside it are the same solid.
#[test]
fn points_inside_do_not_change_the_hull() {
    let mut rng = Rng(7);
    let mut points = cube_corners(0.5);
    for _ in 0..1_000 {
        points.push(vec3(
            rng.range(-0.49, 0.49),
            rng.range(-0.49, 0.49),
            rng.range(-0.49, 0.49),
        ));
    }
    let hull = ConvexHull::from_points(&points).expect("a cube with filling");
    assert_eq!(hull.vertices().len(), 8);
    assert!((hull.volume() - 1.0).abs() < 1e-4);
}

/// A point cloud on a sphere hulls to something that fills most of the sphere
/// and none of the space outside it.
#[test]
fn a_ball_of_points_hulls_to_about_a_ball() {
    let mut rng = Rng(11);
    let radius = 0.7f32;
    let points: Vec<Vec3> = (0..600)
        .map(|_| rng.point_in_ball(1.0).normalize_or_zero() * radius)
        .collect();
    let hull = ConvexHull::from_points(&points).expect("a sphere of points");
    assert_is_a_hull_of(&points, &hull);

    let exact = 4.0 / 3.0 * std::f32::consts::PI * radius.powi(3);
    let ratio = hull.volume() / exact;
    // An inscribed polyhedron on 600 samples is a little under the sphere.
    assert!(
        (0.93..=1.0).contains(&ratio),
        "hull volume is {ratio:.3} of the sphere's"
    );
    assert!(
        (hull.bounding_radius() - radius).abs() < 1e-3,
        "bounding radius {} should be {radius}",
        hull.bounding_radius()
    );
}

/// The support function is the whole reason this type exists: it is what GJK
/// asks, and it has to agree with a brute-force scan over the original points.
#[test]
fn support_agrees_with_a_brute_force_scan() {
    let mut rng = Rng(13);
    let points: Vec<Vec3> = (0..300).map(|_| rng.point_in_ball(0.9)).collect();
    let hull = ConvexHull::from_points(&points).expect("a cloud");

    for _ in 0..200 {
        let dir = rng.point_in_ball(1.0).normalize_or_zero();
        if dir.length_squared() < 0.5 {
            continue;
        }
        let from_hull = hull.support(dir).dot(dir);
        let from_points = points
            .iter()
            .map(|p| p.dot(dir))
            .fold(f32::NEG_INFINITY, f32::max);
        assert!(
            (from_hull - from_points).abs() < 1e-3,
            "support along {dir:?} was {from_hull}, brute force says {from_points}"
        );
    }
}

/// Degenerate clouds are refused rather than turned into a hull with no
/// volume, because a body with no volume has no mass and the failure would
/// surface as a division much later.
#[test]
fn degenerate_clouds_are_refused() {
    assert!(ConvexHull::from_points(&[]).is_none(), "nothing");
    assert!(
        ConvexHull::from_points(&[Vec3::ZERO, Vec3::X, Vec3::Y]).is_none(),
        "a triangle is not a solid"
    );
    let collinear: Vec<Vec3> = (0..20).map(|i| Vec3::X * i as f32).collect();
    assert!(ConvexHull::from_points(&collinear).is_none(), "a line");

    let coplanar: Vec<Vec3> = (0..40)
        .map(|i| {
            let t = i as f32 * 0.37;
            vec3(t.cos(), 0.0, t.sin())
        })
        .collect();
    assert!(ConvexHull::from_points(&coplanar).is_none(), "a disc");
}

/// A hull of a scaled cube has the inertia of a cube.
///
/// Checked against the analytic box, which is the one solid whose tensor
/// everybody knows: `I = m/12 · (b² + c²)` per axis.
#[test]
fn the_inertia_of_a_box_hull_matches_the_analytic_box() {
    let half = vec3(0.3, 0.5, 0.9);
    let mut points = Vec::new();
    for x in [-half.x, half.x] {
        for y in [-half.y, half.y] {
            for z in [-half.z, half.z] {
                points.push(vec3(x, y, z));
            }
        }
    }
    let hull = ConvexHull::from_points(&points).expect("a box");
    let mass = 7.5;
    let got = hull.inertia_diagonal(mass);

    let size = half * 2.0;
    let expected = vec3(
        mass / 12.0 * (size.y * size.y + size.z * size.z),
        mass / 12.0 * (size.z * size.z + size.x * size.x),
        mass / 12.0 * (size.x * size.x + size.y * size.y),
    );
    for k in 0..3 {
        let error = (got[k] - expected[k]).abs() / expected[k];
        assert!(
            error < 1e-3,
            "axis {k}: hull says {}, the analytic box says {} ({:.2}% off)",
            got[k],
            expected[k],
            error * 100.0
        );
    }
}

/// Random clouds, many of them, all satisfying the definition. The cheapest
/// way to find the case that breaks a hull builder is to try a hundred.
#[test]
fn random_clouds_always_produce_valid_hulls() {
    for seed in 0..100u64 {
        let mut rng = Rng(seed * 2_654_435_761 + 1);
        let count = 4 + (seed as usize % 60);
        let points: Vec<Vec3> = (0..count).map(|_| rng.point_in_ball(1.0)).collect();
        let Some(hull) = ConvexHull::from_points(&points) else {
            continue; // legitimately degenerate
        };
        assert_is_a_hull_of(&points, &hull);
    }
}
