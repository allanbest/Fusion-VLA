//! Basic and Advanced, per object.
//!
//! A scene is mostly props that only need to fall and collide; a few objects
//! need stress, breaking and temperature. The tier says which is which, and a
//! Basic object must cost a plain rigid body whatever the scene has on.

use fusion_core::{BodyDesc, Failure, Material, World};
use fusion_math::{vec3, Vec3};
use fusion_sim::{HeatSettings, Simulation, StressSettings};

/// Two identical glass panes struck by identical balls, one Advanced and one
/// Basic: the Advanced one shatters, the Basic one stays whole and is never
/// given a stress or temperature grid.
#[test]
fn a_basic_object_never_breaks_or_heats_whatever_the_scene_has_on() {
    let mut world = World::new();
    world.add(
        BodyDesc::cuboid(vec3(20.0, 1.0, 20.0))
            .at(vec3(0.0, -0.5, 0.0))
            .with_material(Material::CONCRETE)
            .fixed(),
    );
    let mut panes = Vec::new();
    for (x, basic) in [(-2.0f32, false), (2.0, true)] {
        for dx in [-0.35f32, 0.35] {
            world.add(
                BodyDesc::cuboid(vec3(0.1, 0.3, 0.6))
                    .at(vec3(x + dx, 0.15, 0.0))
                    .with_material(Material::CONCRETE)
                    .fixed(),
            );
        }
        let pane = BodyDesc::cuboid(vec3(0.8, 0.02, 0.6))
            .at(vec3(x, 0.311, 0.0))
            .with_material(Material::GLASS)
            .with_failure(Failure::Breaks);
        let pane = if basic { pane.basic() } else { pane };
        panes.push(world.add(pane).index());
        world.add(
            BodyDesc::sphere(0.05)
                .at(vec3(x, 1.5, 0.0))
                .with_material(Material::STEEL),
        );
    }
    let mut sim = Simulation::new(world);
    sim.heat = HeatSettings::on();
    sim.stress = StressSettings::on();
    let mut broke = [false, false];
    for _ in 0..600 {
        sim.step();
        for e in sim.take_failures() {
            let source = sim.world.bodies.source[e.body] as usize;
            for (k, &p) in panes.iter().enumerate() {
                if source == p {
                    broke[k] = true;
                }
            }
        }
    }
    assert!(broke[0], "the Advanced pane shatters");
    assert!(!broke[1], "the Basic pane does not");
    let b = &sim.world.bodies;
    assert!(!b.advanced[panes[1]]);
    assert!(
        !b.stress[panes[1]].is_built(),
        "no stress grid for a Basic body"
    );
    assert!(
        !b.heat[panes[1]].is_built(),
        "no temperature grid for a Basic body"
    );
    assert!(b.heat[panes[0]].is_built());
}

/// A Basic block sliding to rest stays at the temperature it started at; an
/// Advanced one beside it warms from its own friction.
#[test]
fn a_basic_object_keeps_its_temperature() {
    let mut world = World::new();
    world.add(
        BodyDesc::cuboid(vec3(40.0, 1.0, 40.0))
            .at(vec3(0.0, -0.5, 0.0))
            .with_material(Material::CONCRETE)
            .fixed(),
    );
    let advanced = world
        .add(
            BodyDesc::cuboid(Vec3::splat(0.3))
                .at(vec3(-2.0, 0.15, 0.0))
                .with_velocity(vec3(6.0, 0.0, 0.0))
                .with_material(Material::STEEL),
        )
        .index();
    let basic = world
        .add(
            BodyDesc::cuboid(Vec3::splat(0.3))
                .at(vec3(-2.0, 0.15, 2.0))
                .with_velocity(vec3(6.0, 0.0, 0.0))
                .with_material(Material::STEEL)
                .basic(),
        )
        .index();
    let mut sim = Simulation::new(world);
    sim.heat = HeatSettings::on();
    let start = sim.world.bodies.heat[basic].mean();
    for _ in 0..480 {
        sim.step();
    }
    let b = &sim.world.bodies;
    assert!(
        b.heat[advanced].mean() > start + 0.01,
        "the Advanced block warmed"
    );
    assert_eq!(b.heat[basic].mean(), start, "the Basic one did not change");
}
