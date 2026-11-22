use core_math::{vec3, Quat, Rng, Vec3};
use core_space::closest_segments;

#[test]
fn segment_interiors_endpoints_and_collapsed_segments() {
    let cases = [
        // Skew segment interiors.
        (
            [
                vec3(-2.0, 0.0, 0.0),
                vec3(2.0, 0.0, 0.0),
                vec3(0.0, -2.0, 0.3),
                vec3(0.0, 2.0, 0.3),
            ],
            0.3,
        ),
        // Endpoint against interior; endpoint against endpoint.
        (
            [
                Vec3::ZERO,
                Vec3::X,
                vec3(0.5, 1.0, 0.0),
                vec3(0.5, 2.0, 0.0),
            ],
            1.0,
        ),
        ([Vec3::ZERO, Vec3::X, Vec3::X * 2.0, Vec3::X * 3.0], 1.0),
        // Both collapsed, or one collapsed.
        ([Vec3::ZERO, Vec3::ZERO, Vec3::Y, Vec3::Y], 1.0),
        ([Vec3::Y, Vec3::Y, -Vec3::X, Vec3::X], 1.0),
        // Parallel, overlapping projections.
        (
            [
                Vec3::ZERO,
                Vec3::X * 2.0,
                Vec3::X + Vec3::Y,
                Vec3::X * 3.0 + Vec3::Y,
            ],
            1.0,
        ),
        // Almost parallel, interior minimum; a dot-product determinant
        // rounds to zero, but the true cross product is nonzero.
        (
            [
                Vec3::ZERO,
                Vec3::X,
                vec3(0.0, -1e-9, 0.2),
                vec3(1.0, 1e-9, 0.2),
            ],
            0.2,
        ),
    ];
    for (points, expected) in cases {
        for scale in [1e-6, 1.0, 1e6] {
            let [a, b, c, d] = points.map(|p| p * scale);
            let (p, q, [s, t]) = closest_segments(a, b, c, d);
            assert!((0.0..=1.0).contains(&s) && (0.0..=1.0).contains(&t));
            assert!(((p - q).length() / scale - expected).abs() < 1e-12);
            assert!((p - (a + (b - a) * s)).length() < scale * 1e-12);
            assert!((q - (c + (d - c) * t)).length() < scale * 1e-12);
        }
    }
}

#[test]
fn closest_segments_match_sampling_and_transform_with_the_scene() {
    let mut rng = Rng::new(819);
    for _ in 0..300 {
        let [a, b, c, d] = [0; 4].map(|_| rng.direction() * rng.range(0.1, 2.0));
        let (p, q, _) = closest_segments(a, b, c, d);
        let distance = (p - q).length();
        let (p2, q2, _) = closest_segments(d, c, b, a);
        assert!(((p2 - q2).length() - distance).abs() < 1e-12);
        for s in 0..=30 {
            for t in 0..=30 {
                let sampled = a + (b - a) * (s as f64 / 30.0) - c - (d - c) * (t as f64 / 30.0);
                assert!(sampled.length() >= distance - 1e-12);
            }
        }
        let rotation = Quat::from_axis_angle(rng.direction(), 1.1).to_mat3();
        let transform = |v| rotation * v + vec3(3.0, -2.0, 5.0);
        let (p2, q2, _) = closest_segments(transform(a), transform(b), transform(c), transform(d));
        assert!((p2 - transform(p)).length() < 1e-11);
        assert!((q2 - transform(q)).length() < 1e-11);
    }
}
