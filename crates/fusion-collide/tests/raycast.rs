//! Rays against shapes, checked against distances that are known by hand.

use fusion_collide::{pick, ray_vs_shape, Ray};
use fusion_core::{BodyDesc, Shape, World};
use fusion_math::{vec3, Isometry, Quat, Vec3};

#[test]
fn a_ray_hits_a_sphere_at_the_distance_geometry_says() {
    let shape = Shape::sphere(1.0);
    let pose = Isometry::from_translation(vec3(0.0, 0.0, 5.0));
    let ray = Ray::new(Vec3::ZERO, Vec3::Z);

    let t = ray_vs_shape(&ray, &shape, &pose).expect("a sphere five ahead is a hit");
    assert!((t - 4.0).abs() < 1e-4, "hit at {t}, expected 4");
}

#[test]
fn a_ray_that_misses_reports_nothing() {
    let shape = Shape::sphere(1.0);
    let pose = Isometry::from_translation(vec3(0.0, 5.0, 5.0));
    assert!(ray_vs_shape(&Ray::new(Vec3::ZERO, Vec3::Z), &shape, &pose).is_none());
    // And one pointing away from a shape behind it.
    let behind = Isometry::from_translation(vec3(0.0, 0.0, -5.0));
    assert!(ray_vs_shape(&Ray::new(Vec3::ZERO, Vec3::Z), &shape, &behind).is_none());
}

#[test]
fn a_ray_starting_inside_is_touching_it_now() {
    let shape = Shape::sphere(2.0);
    let pose = Isometry::IDENTITY;
    let t = ray_vs_shape(&Ray::new(Vec3::ZERO, Vec3::X), &shape, &pose).expect("inside is a hit");
    assert_eq!(t, 0.0, "a ray from inside a shape hits it at zero");
}

#[test]
fn a_ray_hits_a_box_face_square_on() {
    let shape = Shape::cuboid(vec3(2.0, 2.0, 2.0)); // half extents of 1
    let pose = Isometry::from_translation(vec3(0.0, 0.0, 10.0));
    let t = ray_vs_shape(&Ray::new(Vec3::ZERO, Vec3::Z), &shape, &pose).expect("hit");
    assert!((t - 9.0).abs() < 1e-4, "hit at {t}, expected 9");
}

/// The axis-aligned rays are the ones an editor actually generates — looking
/// straight down a scene — and they are the ones a naive slab test divides by
/// zero on.
#[test]
fn rays_parallel_to_a_face_do_not_produce_nonsense() {
    let shape = Shape::cuboid(vec3(2.0, 2.0, 2.0));
    let pose = Isometry::IDENTITY;

    // Straight down, offset so it passes beside the box.
    let miss = Ray::new(vec3(5.0, 10.0, 0.0), -Vec3::Y);
    assert!(ray_vs_shape(&miss, &shape, &pose).is_none());

    // Straight down, through it.
    let hit = Ray::new(vec3(0.0, 10.0, 0.0), -Vec3::Y);
    let t = ray_vs_shape(&hit, &shape, &pose).expect("straight down through the box");
    assert!((t - 9.0).abs() < 1e-4, "hit at {t}, expected 9");
    assert!(t.is_finite(), "a parallel ray produced {t}");
}

/// A rotated box is hit in its own frame, not its bounding box.
#[test]
fn a_rotated_box_is_hit_where_it_actually_is() {
    let shape = Shape::cuboid(vec3(2.0, 0.2, 2.0)); // a thin plate
    let flat = Isometry::new(vec3(0.0, 0.0, 0.0), Quat::IDENTITY);
    // Straight down onto the flat plate: a hit.
    let down = Ray::new(vec3(0.0, 5.0, 0.0), -Vec3::Y);
    assert!(ray_vs_shape(&down, &shape, &flat).is_some());

    // Stand the plate on its edge; the same ray now passes the thin side.
    let edge = Isometry::new(
        vec3(0.0, 0.0, 0.0),
        Quat::from_rotation_z(std::f32::consts::FRAC_PI_2),
    );
    let beside = Ray::new(vec3(0.9, 5.0, 0.0), -Vec3::Y);
    assert!(
        ray_vs_shape(&beside, &shape, &edge).is_none(),
        "the ray hit a plate that had been turned out of its way"
    );
}

#[test]
fn picking_returns_the_nearest_body_and_not_merely_a_hit() {
    let mut world = World::new();
    world.add(BodyDesc::sphere(0.5).at(vec3(0.0, 0.0, 10.0))); // far
    world.add(BodyDesc::sphere(0.5).at(vec3(0.0, 0.0, 3.0))); // near
    world.add(BodyDesc::sphere(0.5).at(vec3(9.0, 0.0, 5.0))); // off to one side

    let ray = Ray::new(Vec3::ZERO, Vec3::Z);
    let (index, t) = pick(&ray, &world.bodies).expect("two are in the way");
    assert_eq!(index, 1, "picked the far body over the near one");
    assert!((t - 2.5).abs() < 1e-4, "hit at {t}");

    // A ray into empty space picks nothing.
    assert!(pick(&Ray::new(Vec3::ZERO, -Vec3::Z), &world.bodies).is_none());
}
