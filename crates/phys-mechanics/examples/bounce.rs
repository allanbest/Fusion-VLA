//! A rigid ball dropped 1 m: how high it comes back, by restitution and
//! integrator. `cargo run --release -p phys-mechanics --example bounce`
use core_math::{vec3, Pose};
use core_runtime::Pool;
use core_space::Shape;
use core_world::{Material, Representation, World};
use phys_mechanics::{Integrator, Mechanics, Settings};

fn main() {
    for integrator in [Integrator::Newmark, Integrator::TrBdf2] {
        for e in [1.0, 0.5] {
            let mut world = World::new();
            let floor = world.add_material(Material::new("floor", 2400.0, 30e9, 0.2, 0.5, 1.0));
            let wood = world.add_material(Material::new("wood", 600.0, 1e4, 0.3, 0.5, e));
            world.add_fixed(
                "floor",
                floor,
                Shape::cuboid(vec3(20.0, 1.0, 20.0)),
                Pose::at(vec3(0.0, -0.5, 0.0)),
            );
            let r = 0.1;
            let id = world.add_rigid(
                "ball",
                wood,
                Shape::Sphere { radius: r },
                Pose::at(vec3(0.0, 1.0 + r, 0.0)),
            );
            let mut m = Mechanics::new(Settings {
                integrator,
                ..Default::default()
            });
            let pool = Pool::new(1);
            let (mut bounced, mut peak) = (false, 0.0f64);
            let mut log = String::new();
            for step in 0..150 {
                let rep = m.step(&pool, &mut world, 1.0 / 60.0);
                let Representation::Rigid(b) = &world.object(id).representation else {
                    unreachable!()
                };
                let y = b.pose.position.y - r;
                if b.velocity.y > 0.0 {
                    bounced = true;
                }
                if bounced {
                    peak = peak.max(y);
                }
                if (35..=40).contains(&step) {
                    log += &format!(
                        "    {step} y {y:+.5} vy {:+.4} it {} solves {} grips {}\n",
                        b.velocity.y, rep.iterations, rep.solves, rep.grips
                    );
                }
            }
            println!(
                "{integrator:?} e={e}: rebound {peak:.3} m (expect {:.3})\n{log}",
                e * e
            );
        }
    }
}
