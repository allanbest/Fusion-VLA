use core_math::{vec3, Pose, Quat, Rng, Vec3};
use core_space::{Aabb, Bvh, NeighbourGrid, Shape};

fn shapes() -> Vec<Shape> {
    vec![
        Shape::Sphere { radius: 0.7 },
        Shape::Cuboid {
            half: vec3(0.3, 0.8, 0.5),
        },
        Shape::Capsule {
            radius: 0.25,
            half_length: 0.6,
        },
        Shape::HalfSpace {
            normal: vec3(1.0, 2.0, -0.5).normalize(),
            offset: 0.2,
        },
    ]
}

#[test]
fn distance_gradient_is_the_normal() {
    let mut rng = Rng::new(5);
    for shape in shapes() {
        for _ in 0..500 {
            let p = vec3(
                rng.range(-1.5, 1.5),
                rng.range(-1.5, 1.5),
                rng.range(-1.5, 1.5),
            );
            let (d, n) = shape.distance(p);
            assert!((n.length() - 1.0).abs() < 1e-12);
            // Central differences, away from the medial axis where the
            // gradient jumps.
            let h = 1e-6;
            let grad = Vec3::new(
                shape.distance(p + Vec3::X * h).0 - shape.distance(p - Vec3::X * h).0,
                shape.distance(p + Vec3::Y * h).0 - shape.distance(p - Vec3::Y * h).0,
                shape.distance(p + Vec3::Z * h).0 - shape.distance(p - Vec3::Z * h).0,
            ) / (2.0 * h);
            if (grad.length() - 1.0).abs() < 1e-4 {
                assert!(
                    (grad - n).length() < 1e-4,
                    "{shape:?} at {p:?}: {grad:?} vs {n:?}"
                );
            }
            // Stepping back along the normal by the distance lands on the surface.
            assert!(
                shape.distance(p - n * d).0.abs() < 1e-9,
                "{shape:?} at {p:?}"
            );
        }
    }
}

#[test]
fn mass_properties_match_sampling() {
    let mut rng = Rng::new(6);
    for shape in shapes().into_iter().filter(Shape::is_bounded) {
        let mass = shape.unit_mass().unwrap();
        let bounds = shape.local_bounds();
        let samples = 400_000;
        let (mut inside, mut inertia_x, mut inertia_y) = (0usize, 0.0, 0.0);
        for _ in 0..samples {
            let p = Vec3::new(
                rng.range(bounds.min.x, bounds.max.x),
                rng.range(bounds.min.y, bounds.max.y),
                rng.range(bounds.min.z, bounds.max.z),
            );
            if shape.distance(p).0 <= 0.0 {
                inside += 1;
                inertia_x += p.y * p.y + p.z * p.z;
                inertia_y += p.z * p.z + p.x * p.x;
            }
        }
        let box_volume = bounds.size().x * bounds.size().y * bounds.size().z;
        let cell = box_volume / samples as f64;
        let volume = inside as f64 * cell;
        assert!(
            (volume / mass.volume - 1.0).abs() < 0.01,
            "{shape:?}: volume {volume} vs {}",
            mass.volume
        );
        assert!(
            (inertia_x * cell / mass.inertia[(0, 0)] - 1.0).abs() < 0.02,
            "{shape:?}: Ixx"
        );
        assert!(
            (inertia_y * cell / mass.inertia[(1, 1)] - 1.0).abs() < 0.02,
            "{shape:?}: Iyy"
        );
    }
}

#[test]
fn placed_bounds_contain_the_shape() {
    let mut rng = Rng::new(7);
    for shape in shapes().into_iter().filter(Shape::is_bounded) {
        let pose = Pose::new(
            vec3(1.0, -2.0, 0.5),
            Quat::from_axis_angle(rng.direction(), 1.1),
        );
        let bounds = shape.bounds(&pose).grown(1e-12);
        for _ in 0..2000 {
            let local = rng.direction() * 2.0;
            let (d, n) = shape.distance(local);
            let surface = local - n * d;
            assert!(bounds.contains(pose.transform_point(surface)), "{shape:?}");
        }
    }
}

fn random_boxes(rng: &mut Rng, n: usize) -> Vec<Aabb> {
    (0..n)
        .map(|_| {
            let centre = vec3(
                rng.range(0.0, 10.0),
                rng.range(0.0, 10.0),
                rng.range(0.0, 10.0),
            );
            Aabb::around(
                centre,
                vec3(
                    rng.range(0.05, 0.4),
                    rng.range(0.05, 0.4),
                    rng.range(0.05, 0.4),
                ),
            )
        })
        .collect()
}

fn brute_pairs(boxes: &[Aabb]) -> Vec<(usize, usize)> {
    let mut pairs = Vec::new();
    for a in 0..boxes.len() {
        for b in a + 1..boxes.len() {
            if boxes[a].overlaps(&boxes[b]) {
                pairs.push((a, b));
            }
        }
    }
    pairs
}

#[test]
fn bvh_finds_exactly_the_overlaps() {
    let mut rng = Rng::new(8);
    let mut boxes = random_boxes(&mut rng, 3000);
    let mut bvh = Bvh::build(&boxes);
    for round in 0..2 {
        let mut pairs = Vec::new();
        bvh.self_pairs(|a, b| pairs.push((a, b)));
        pairs.sort_unstable();
        assert_eq!(pairs, brute_pairs(&boxes), "round {round}");
        // Move everything a little and refit.
        for b in boxes.iter_mut() {
            let shift = rng.direction() * 0.2;
            *b = Aabb::new(b.min + shift, b.max + shift);
        }
        bvh.refit(&boxes);
    }
}

#[test]
fn grouped_bvh_excludes_only_the_requested_group_after_refits() {
    let mut rng = Rng::new(732);
    let mut boxes = random_boxes(&mut rng, 300);
    let groups: Vec<u32> = (0..boxes.len()).map(|i| (i % 7) as u32).collect();
    let mut tree = Bvh::build_grouped(&boxes, &groups);
    for _ in 0..2 {
        for i in 0..boxes.len() {
            let query = boxes[i].grown(0.5);
            let mut found = Vec::new();
            tree.query_excluding(&query, groups[i], |j| found.push(j));
            found.sort_unstable();
            let expected: Vec<_> = (0..boxes.len())
                .filter(|&j| groups[j] != groups[i] && boxes[j].overlaps(&query))
                .collect();
            assert_eq!(found, expected);
        }
        let mut found = Vec::new();
        tree.query(&Aabb::EVERYTHING, |j| found.push(j));
        found.sort_unstable();
        assert_eq!(found, (0..boxes.len()).collect::<Vec<_>>());
        for bounds in &mut boxes {
            let shift = rng.direction() * 0.1;
            *bounds = Aabb::new(bounds.min + shift, bounds.max + shift);
        }
        tree.refit(&boxes);
    }
    let empty = Bvh::build_grouped(&[], &[]);
    empty.query_excluding(&Aabb::EVERYTHING, 0, |_| panic!("empty hierarchy"));
}

#[test]
fn affine_bounds_contain_stretched_and_sheared_shapes() {
    use core_math::Mat3;
    let mut rng = Rng::new(197);
    for shape in shapes().into_iter().filter(Shape::is_bounded) {
        for _ in 0..20 {
            let mut matrix = Mat3::IDENTITY;
            for i in 0..3 {
                for j in 0..3 {
                    matrix[(i, j)] += rng.range(-0.8, 0.8);
                }
            }
            let centre = vec3(1.0, -2.0, 0.5);
            let bounds = shape.affine_bounds(centre, matrix).grown(1e-12);
            for _ in 0..300 {
                let p = rng.direction() * 3.0;
                let (d, n) = shape.distance(p);
                assert!(bounds.contains(centre + matrix * (p - n * d)));
            }
        }
    }
    assert_eq!(
        Shape::ground(0.0).affine_bounds(Vec3::ZERO, Mat3::ZERO),
        Aabb::EVERYTHING
    );
}

#[test]
fn neighbours_match_brute_force() {
    let mut rng = Rng::new(9);
    let points: Vec<Vec3> = (0..5000)
        .map(|_| {
            vec3(
                rng.range(-3.0, 3.0),
                rng.range(-3.0, 3.0),
                rng.range(-3.0, 3.0),
            )
        })
        .collect();
    let radius = 0.3;
    let grid = NeighbourGrid::build(&points, radius);
    for &p in points.iter().take(300) {
        let mut found = Vec::new();
        grid.within(&points, p, radius, |j| found.push(j));
        found.sort_unstable();
        let expected: Vec<usize> = (0..points.len())
            .filter(|&j| (points[j] - p).length() <= radius)
            .collect();
        assert_eq!(found, expected);
    }
}

#[test]
fn closest_point_on_triangles() {
    use core_space::closest_on_triangle;
    let mut rng = Rng::new(10);
    for _ in 0..300 {
        let [a, b, c] = [0, 1, 2].map(|_| rng.direction());
        let p = rng.direction() * rng.range(0.0, 2.0);
        let (q, w) = closest_on_triangle(p, a, b, c);
        assert!(w.iter().all(|&x| x >= -1e-12) && (w.iter().sum::<f64>() - 1.0).abs() < 1e-12);
        assert!((a * w[0] + b * w[1] + c * w[2] - q).length() < 1e-12);
        // No sampled point of the triangle is closer.
        let d = (p - q).length();
        for i in 0..=40 {
            for j in 0..=40 - i {
                let (u, v) = (i as f64 / 40.0, j as f64 / 40.0);
                let s = a + (b - a) * u + (c - a) * v;
                assert!((p - s).length() >= d - 1e-12);
            }
        }
    }
    // A degenerate (flat) triangle still gives a point on it.
    let (q, _) = closest_on_triangle(vec3(0.5, 1.0, 0.0), Vec3::ZERO, Vec3::X, Vec3::X * 2.0);
    assert!((q - vec3(0.5, 0.0, 0.0)).length() < 1e-12);
}

#[test]
fn collapsed_triangle_edges_have_finite_consistent_closest_points() {
    use core_space::closest_on_triangle;
    let p = vec3(0.5, 1.0, 0.0);
    for vertices in [
        [Vec3::ZERO, Vec3::ZERO, Vec3::X],
        [Vec3::ZERO, Vec3::X, Vec3::ZERO],
        [Vec3::X, Vec3::ZERO, Vec3::ZERO],
        [Vec3::ZERO, Vec3::ZERO, Vec3::ZERO],
        [Vec3::X, Vec3::X * 2.0, Vec3::ZERO],
    ] {
        let (q, w) = closest_on_triangle(p, vertices[0], vertices[1], vertices[2]);
        assert!(q.is_finite() && w.iter().all(|v| v.is_finite() && *v >= 0.0));
        assert!((w.iter().sum::<f64>() - 1.0).abs() < 1e-15);
        let reconstruction = vertices[0] * w[0] + vertices[1] * w[1] + vertices[2] * w[2];
        assert!((reconstruction - q).length() < 1e-15);
        let expected = if vertices.iter().all(|p| *p == Vec3::ZERO) {
            Vec3::ZERO
        } else {
            Vec3::X * 0.5
        };
        assert!((q - expected).length() < 1e-15, "{vertices:?}: {q:?}");
    }
}
