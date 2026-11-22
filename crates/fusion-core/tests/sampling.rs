//! The one authoring verb, against the size it was asked for.
//!
//! PLAN.md §16 makes `fill` the way material gets into a scene: every solid,
//! every fluid and every soft body in Fusion is a lattice this function
//! emitted. It had no test of its own until v1.0, and what that cost is the
//! subject of the first one here.

use fusion_core::{fill, Shape};
use fusion_math::Vec3;

/// A box asked for at `n` points across gets `n` points across.
///
/// ## What this caught
///
/// A 40 cm cube at 10 cm resolution was filled with **eight** material points
/// rather than sixty-four: a 20 cm solid holding an eighth of the mass the
/// scene asked for, in a lattice whose spacing was right and whose extent was
/// half. Nothing reported it. The cause was two floating-point comparisons
/// about numbers a user had chosen to be exact — `(0.4 - 0.1) / 0.1` is
/// `3.0000002`, and the outermost ring of points then sits at exactly one
/// inset from the surface, where "is this inside" is decided by the last bit.
///
/// Which resolutions were wrong depended on which way each division happened
/// to round, so 40 cm at 5 cm was correct and 40 cm at 10 cm was not, which is
/// the shape of bug that survives being looked at.
#[test]
fn a_box_is_filled_to_the_size_it_asked_for() {
    for size in [0.3f32, 0.4, 1.0, 2.5] {
        for n in 1..=12usize {
            let spacing = size / n as f32;
            let mut count = 0usize;
            let mut low = Vec3::splat(f32::MAX);
            let mut high = Vec3::splat(f32::MIN);
            fill(&Shape::cuboid(Vec3::splat(size)), spacing, |p| {
                count += 1;
                low = low.min(p);
                high = high.max(p);
            });
            assert_eq!(
                count,
                n * n * n,
                "a {size} m box at {spacing} m resolution: {count} points, \
                 wanted {n}^3 = {}",
                n * n * n
            );
            // And they span the box: each point stands for a cell of its own
            // spacing, so the outermost centres sit half a spacing in.
            let span = high - low + Vec3::splat(spacing);
            for axis in 0..3 {
                assert!(
                    (span[axis] - size).abs() < 1e-4 * size,
                    "a {size} m box at {spacing} m filled {:.4} m of axis {axis}",
                    span[axis]
                );
            }
        }
    }
}

/// The lattice is centred, so a symmetric shape is filled symmetrically.
#[test]
fn the_lattice_is_centred_on_the_shape() {
    for spacing in [0.07f32, 0.11, 0.13] {
        let mut low = Vec3::splat(f32::MAX);
        let mut high = Vec3::splat(f32::MIN);
        fill(&Shape::cuboid(Vec3::splat(1.0)), spacing, |p| {
            low = low.min(p);
            high = high.max(p);
        });
        let centre = (low + high) * 0.5;
        assert!(
            centre.length() < 1e-5,
            "off-centre by {:.5} m at {spacing} m resolution",
            centre.length()
        );
    }
}

/// A curved shape holds its nominal volume, at any resolution.
///
/// It used to be filled short — only centres half a spacing inside were
/// kept, which takes a shell off anything curved: 85% of a 0.5 m sphere at
/// 5 cm, 69% at 10 cm, and 74% for a ball at the editor's default sampling,
/// which then shrank and lost a quarter of its mass the moment it was played.
/// Keeping every centre inside the shape lets the cells that poke out pay for
/// the corners left uncovered.
#[test]
fn a_sphere_holds_its_volume() {
    for spacing in [0.025f32, 0.05, 0.0833, 0.1] {
        let mut count = 0usize;
        fill(&Shape::sphere(0.5), spacing, |_| count += 1);
        let sampled = count as f32 * spacing * spacing * spacing;
        let nominal = 4.0 / 3.0 * core::f32::consts::PI * 0.125;
        let ratio = sampled / nominal;
        assert!(
            (ratio - 1.0).abs() < 0.06,
            "a 0.5 m sphere at {spacing} m held {:.1}% of its nominal volume",
            ratio * 100.0
        );
    }
}
