//! A hull resting under a load holds it: the case a heap of broken pieces is
//! made of. A light shard under a heavy block sank through the floor while
//! its contacts carried anchors from vertices next to the one that was deep.

use fusion_core::{BodyDesc, Material, Shape, World};
use fusion_math::vec3;
use fusion_sim::Simulation;

fn lowest(sim: &Simulation, i: usize) -> f32 {
    let b = &sim.world.bodies;
    let pose = b.pose(i);
    match &b.shape[i] {
        Shape::Hull(h) => h
            .vertices()
            .iter()
            .map(|p| pose.transform_point(*p).y)
            .fold(f32::MAX, f32::min),
        _ => unreachable!(),
    }
}

#[test]
fn a_shard_under_a_heavy_block_stays_on_the_floor() {
    let wedge = [
        vec3(-0.1, -0.02, -0.08),
        vec3(0.1, -0.04, -0.08),
        vec3(0.0, 0.04, -0.05),
        vec3(-0.1, -0.02, 0.09),
        vec3(0.12, -0.04, 0.07),
        vec3(0.02, 0.03, 0.08),
        vec3(0.05, -0.045, 0.0),
    ];
    for load in [20.0f32, 200.0] {
        let mut world = World::new();
        world.add(
            BodyDesc::cuboid(vec3(40.0, 1.0, 40.0))
                .at(vec3(0.0, -0.5, 0.0))
                .with_material(Material::CONCRETE)
                .fixed(),
        );
        let shard = world
            .add(
                BodyDesc::new(Shape::hull(&wedge).unwrap())
                    .at(vec3(0.0, 0.06, 0.0))
                    .with_material(Material::GLASS),
            )
            .index();
        world.add(
            BodyDesc::cuboid(vec3(0.15, 0.1, 0.12))
                .at(vec3(0.0, 0.2, 0.0))
                .with_material(Material::STEEL)
                .with_mass(load),
        );
        let mut sim = Simulation::new(world);
        for _ in 0..1200 {
            sim.step();
        }
        let low = lowest(&sim, shard);
        assert!(
            low > -0.002,
            "{load} kg: the shard is {:.1} mm into the floor",
            -low * 1e3
        );
    }
}
