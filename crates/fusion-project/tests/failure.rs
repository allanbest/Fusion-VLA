//! Objects that break, through the file.

use fusion_core::{BodyDesc, Failure, Material, World};
use fusion_math::{vec3, Vec3};
use fusion_project::Project;
use fusion_sim::{state_hash, Simulation, StressSettings};

fn scene() -> Simulation {
    let mut world = World::new();
    world.add(
        BodyDesc::cuboid(vec3(40.0, 1.0, 40.0))
            .at(vec3(0.0, -0.5, 0.0))
            .with_material(Material::CONCRETE)
            .fixed(),
    );
    world.add(
        BodyDesc::cuboid(vec3(1.0, 0.1, 0.1))
            .at(vec3(0.0, 2.0, 0.0))
            .with_angular_velocity(vec3(0.0, 450.0, 0.0))
            .with_material(Material::GLASS)
            .with_failure(Failure::Breaks),
    );
    world.add(BodyDesc::cuboid(Vec3::splat(0.3)).at(vec3(2.0, 0.15, 0.0)));
    let mut sim = Simulation::new(world);
    sim.stress = StressSettings::on();
    sim
}

#[test]
fn a_scene_that_breaks_is_format_7_and_breaks_the_same_way_reloaded() {
    let mut original = scene();
    let text = Project::from_simulation(&original).to_toml().unwrap();
    assert!(text.contains("failure = \"breaks\""), "{text}");
    assert!(text.contains("[solver.stress]"), "{text}");
    assert!(text.contains("[material.strength]"), "{text}");

    let loaded = Project::from_toml(&text).unwrap();
    assert_eq!(loaded.fusion.format, 7);
    let mut rebuilt = loaded.build().unwrap();
    assert_eq!(rebuilt.stress, original.stress);
    for _ in 0..60 {
        original.step();
        rebuilt.step();
    }
    assert!(original.world.bodies.len() > 3, "the spinning glass burst");
    assert_eq!(rebuilt.world.bodies.len(), original.world.bodies.len());
    assert_eq!(state_hash(&original), state_hash(&rebuilt));
}

#[test]
fn a_scene_where_nothing_can_break_writes_no_failure() {
    let mut world = World::new();
    world.add(BodyDesc::cuboid(Vec3::ONE));
    let text = Project::from_simulation(&Simulation::new(world))
        .to_toml()
        .unwrap();
    assert!(!text.contains("failure"), "{text}");
    assert!(!text.contains("[solver.stress]"), "{text}");
}
