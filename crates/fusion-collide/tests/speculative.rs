//! Speculative reach is local, includes driven boundaries, and respects its cap.
use fusion_collide::CollisionPipeline;
use fusion_core::{BodyDesc, World};
use fusion_math::{vec3, Vec3};

#[test]
fn a_fast_driven_boundary_reaches_a_stationary_dynamic_body() {
    let mut world = World::new();
    world.add(
        BodyDesc::sphere(0.05)
            .at(vec3(-0.3, 0.0, 0.0))
            .driven()
            .with_velocity(Vec3::X * 60.0),
    );
    world.add(BodyDesc::sphere(0.05));
    let mut pipeline = CollisionPipeline::default();
    pipeline.sweep_dt = 1.0 / 240.0;
    let mut contacts = Vec::new();
    pipeline.detect(&world.bodies, &mut contacts);
    assert_eq!(contacts.len(), 1, "driven motion must contribute to reach");
    pipeline.max_sweep_distance = 0.1;
    pipeline.detect(&world.bodies, &mut contacts);
    assert!(
        contacts.is_empty(),
        "the configured travel cap must be honored"
    );
    pipeline.max_sweep_distance = 2.0;
    pipeline.sweep_dt = 0.0;
    pipeline.detect(&world.bodies, &mut contacts);
    assert!(
        contacts.is_empty(),
        "disabling prediction must discard its reach"
    );
}

#[test]
fn an_unrelated_fast_body_does_not_inflate_resting_contacts() {
    let mut world = World::new();
    world.add(BodyDesc::sphere(0.05));
    world.add(BodyDesc::sphere(0.05).at(vec3(0.3, 0.0, 0.0)));
    world.add(
        BodyDesc::sphere(0.05)
            .at(vec3(100.0, 0.0, 0.0))
            .with_velocity(Vec3::Y * 1000.0),
    );
    let mut pipeline = CollisionPipeline::new();
    pipeline.sweep_dt = 1.0 / 240.0;
    let mut contacts = Vec::new();
    pipeline.detect(&world.bodies, &mut contacts);
    assert!(contacts.is_empty());
}
