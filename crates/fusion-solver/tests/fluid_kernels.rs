//! The SPH kernels, against their analytic properties.
//!
//! Every one of these is a closed-form fact about the kernel rather than a
//! property of the simulation, which is what makes them worth asserting: a
//! normalization constant that is wrong by a factor is invisible in a picture
//! and shifts every density in the engine.

use fusion_math::Vec3;
use fusion_solver::fluid::{boundary_occlusion, lattice_density, poly6, spiky_gradient};

/// Numerically integrate a radially symmetric kernel over all space.
fn integrate_poly6(h: f32) -> f32 {
    // 4 pi r^2 W(r) dr, midpoint rule. The integrand is smooth and compactly
    // supported, so a few thousand samples is plenty.
    let steps = 20_000;
    let dr = h / steps as f32;
    (0..steps)
        .map(|i| {
            let r = (i as f32 + 0.5) * dr;
            4.0 * std::f32::consts::PI * r * r * poly6(r * r, h) * dr
        })
        .sum()
}

#[test]
fn poly6_integrates_to_one() {
    for h in [0.05, 0.1, 0.25, 1.0] {
        let total = integrate_poly6(h);
        assert!(
            (total - 1.0).abs() < 1e-3,
            "poly6 at h = {h} integrates to {total}, not 1"
        );
    }
}

#[test]
fn poly6_vanishes_at_and_beyond_its_support() {
    let h = 0.1;
    assert_eq!(poly6(h * h, h), 0.0);
    assert_eq!(poly6(h * h * 4.0, h), 0.0);
    assert!(poly6(0.0, h) > 0.0);
    // Monotone decreasing inside the support.
    let mut previous = poly6(0.0, h);
    for i in 1..50 {
        let r = h * i as f32 / 50.0;
        let w = poly6(r * r, h);
        assert!(w <= previous, "poly6 rose between samples");
        previous = w;
    }
}

#[test]
fn the_spiky_gradient_points_from_the_first_particle_towards_the_second() {
    // The whole reason for using spiky rather than poly6 for gradients: its
    // magnitude does not vanish as particles converge, so two nearly
    // coincident particles still separate.
    let h = 0.1;
    let offset = Vec3::new(0.02, 0.0, 0.0);
    let g = spiky_gradient(offset, offset.length(), h);
    assert!(g.x < 0.0, "gradient should oppose the offset");
    assert_eq!(g.y, 0.0);
    assert_eq!(g.z, 0.0);

    // Steeper as r shrinks, right down to the limit.
    let near = spiky_gradient(Vec3::X * 0.001, 0.001, h).length();
    let far = spiky_gradient(Vec3::X * 0.09, 0.09, h).length();
    assert!(near > far);
    assert!(near.is_finite());
    assert_eq!(spiky_gradient(Vec3::ZERO, 0.0, h), Vec3::ZERO);
    assert_eq!(spiky_gradient(Vec3::X * h, h, h), Vec3::ZERO);
}

#[test]
fn the_lattice_density_is_close_to_the_number_density_it_stands_for() {
    // Sum_j W(r_ij) over a lattice approximates the integral of W times the
    // number density, and the integral is 1. So the sum should come out near
    // 1/spacing^3 — and how near is the quadrature error of the kernel over
    // that lattice, which is the thing being checked.
    for ratio in [2.0f32, 2.5, 3.0] {
        let spacing = 0.05;
        let n = 1.0 / (spacing * spacing * spacing);
        let sum = lattice_density(spacing, ratio * spacing);
        let error = (sum - n).abs() / n;
        assert!(
            error < 0.02,
            "at h = {ratio} spacings the lattice sum is {error:.4} off the number density"
        );
    }
}

#[test]
fn the_lattice_density_does_not_depend_on_the_absolute_scale() {
    // Only the ratio of smoothing radius to spacing matters; the value scales
    // as 1/spacing^3. A calibration that quietly depended on absolute size
    // would make a 1 mm fluid and a 1 m fluid different materials.
    let coarse = lattice_density(0.1, 0.2) * 0.1f32.powi(3);
    let fine = lattice_density(0.01, 0.02) * 0.01f32.powi(3);
    assert!(
        (coarse - fine).abs() / coarse < 1e-3,
        "scale-invariant form differs: {coarse} vs {fine}"
    );
}

#[test]
fn boundary_occlusion_is_half_the_kernel_at_the_surface() {
    // A particle whose centre is exactly on a flat wall is missing precisely
    // half its support. This is the anchor for the closed form; if it is not
    // 0.5, the polynomial is wrong.
    let h = 0.1;
    assert!((boundary_occlusion(0.0, h) - 0.5).abs() < 1e-5);
    assert_eq!(boundary_occlusion(h, h), 0.0);
    assert_eq!(boundary_occlusion(2.0 * h, h), 0.0);
}

#[test]
fn boundary_occlusion_matches_a_numerical_integration_of_the_kernel() {
    // The claim the closed form makes: it equals the integral of poly6 over
    // the half-space beyond a plane at the given distance. Integrating it
    // directly is slow and obvious, which makes it exactly the right thing to
    // check the fast version against.
    let h = 0.1f32;
    let steps = 4000;
    for &d in &[0.0f32, 0.01, 0.025, 0.05, 0.08, 0.099] {
        // Integrate slabs from z = -h to z = -d; each slab of the radially
        // symmetric kernel contributes 315 (h^2 - z^2)^4 / (256 h^9).
        let lo = -h;
        let hi = -d;
        let dz = (hi - lo) / steps as f32;
        let mut total = 0.0f64;
        for i in 0..steps {
            let z = lo + (i as f32 + 0.5) * dz;
            let slab = 315.0 * (h * h - z * z).powi(4) / (256.0 * h.powi(9));
            total += (slab * dz) as f64;
        }
        let closed = boundary_occlusion(d, h);
        assert!(
            (closed as f64 - total).abs() < 1e-4,
            "at d = {d}: closed form {closed}, numerical {total}"
        );
    }
}

#[test]
fn boundary_occlusion_falls_monotonically_with_distance() {
    let h = 0.1;
    let mut previous = boundary_occlusion(0.0, h);
    for i in 1..=100 {
        let d = h * i as f32 / 100.0;
        let value = boundary_occlusion(d, h);
        assert!(value <= previous + 1e-7, "occlusion rose at d = {d}");
        assert!(value >= -1e-7, "occlusion went negative at d = {d}");
        previous = value;
    }
}
