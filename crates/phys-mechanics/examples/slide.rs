//! A box sliding at 2 m/s on μ = 0.4: deceleration per step against μg.
use core_math::{vec3, Pose, Vec3};
use core_runtime::Pool;
use core_space::Shape;
use core_world::{library, Material, Representation, World};
use phys_mechanics::{Mechanics, Settings};

fn main() {
    for updates in [0, 1] {
        let mut world = World::new();
        let ground = world.add_material(Material {
            friction: 0.4,
            ..library::concrete()
        });
        let wood = world.add_material(Material {
            friction: 0.4,
            ..library::oak()
        });
        world.add_fixed(
            "ground",
            ground,
            Shape::cuboid(vec3(20.0, 1.0, 20.0)),
            Pose::at(vec3(0.0, -0.5, 0.0)),
        );
        let id = world.add_rigid(
            "box",
            wood,
            Shape::cuboid(Vec3::splat(0.2)),
            Pose::at(vec3(0.0, 0.1, 0.0)),
        );
        world.set_velocity(id, vec3(2.0, 0.0, 0.0), Vec3::ZERO);
        let mut m = Mechanics::new(Settings {
            friction_updates: updates,
            ..Default::default()
        });
        let pool = Pool::new(1);
        let mut v0 = 2.0;
        let mut line = String::new();
        for step in 0..12 {
            m.step(&pool, &mut world, 1.0 / 60.0);
            let Representation::Rigid(b) = &world.object(id).representation else {
                unreachable!()
            };
            line += &format!("{:.3} ", (v0 - b.velocity.x) / (0.4 * 9.81 / 60.0));
            v0 = b.velocity.x;
            let _ = step;
        }
        println!("updates {updates}: braking per step as a fraction of μg: {line}");
    }
}
