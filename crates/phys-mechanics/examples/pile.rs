//! A pile of 27 rigid bodies dropped together: Newton iterations per solve,
//! spheres (contact independent of rotation) against boxes.
use core_math::{vec3, Pose, Quat, Rng, Vec3};
use core_runtime::Pool;
use core_space::Shape;
use core_world::{library, World};
use phys_mechanics::{Mechanics, Settings};

fn main() {
    for (name, shape) in [("boxes", Shape::cuboid(Vec3::splat(0.2)))] {
        let mut world = World::new();
        let friction: f64 = std::env::var("PILE_FRICTION")
            .ok()
            .and_then(|v| v.parse().ok())
            .unwrap_or(0.5);
        let ground = world.add_material(core_world::Material {
            friction,
            ..library::concrete()
        });
        let wood = world.add_material(core_world::Material {
            friction,
            ..library::oak()
        });
        world.add_fixed(
            "floor",
            ground,
            Shape::cuboid(vec3(20.0, 1.0, 20.0)),
            Pose::at(vec3(0.0, -0.5, 0.0)),
        );
        let mut rng = Rng::new(1);
        for i in 0..27 {
            let p = vec3(
                (i % 3) as f64 * 0.4 - 0.4,
                0.3 + (i / 9) as f64 * 0.4,
                ((i / 3) % 3) as f64 * 0.4 - 0.4,
            );
            world.add_rigid(
                "body",
                wood,
                shape,
                Pose::new(
                    p,
                    Quat::from_axis_angle(rng.direction(), rng.range(0.0, 0.5)),
                ),
            );
        }
        let updates: usize = std::env::var("PILE_UPDATES")
            .ok()
            .and_then(|v| v.parse().ok())
            .unwrap_or(1);
        let mut m = Mechanics::new(Settings {
            friction_updates: updates,
            ..Settings::default()
        });
        let threads = std::env::var("PILE_THREADS")
            .ok()
            .and_then(|v| v.parse().ok())
            .unwrap_or(8);
        let steps = std::env::var("PILE_STEPS")
            .ok()
            .and_then(|v| v.parse().ok())
            .unwrap_or(120);
        let pool = Pool::new(threads);
        let (mut solves, mut iterations, mut linear_iterations, mut unconverged, mut splits) =
            (0, 0, 0, 0, 0);
        let start = std::time::Instant::now();
        for _ in 0..steps {
            let r = m.step(&pool, &mut world, 1.0 / 60.0);
            solves += r.solves;
            iterations += r.iterations;
            linear_iterations += r.linear_iterations;
            unconverged += r.failed() as usize;
            splits += r.splits;
        }
        println!("{name}, {threads} threads, {steps} steps: {solves} solves, {iterations} Newton iterations, {linear_iterations} CG iterations, {splits} splits, {unconverged} steps failed, {:.0} ms", start.elapsed().as_secs_f64() * 1e3);
        assert_eq!(unconverged, 0, "benchmark contains rejected steps");
        if std::env::var_os("PILE_PROFILE").is_some() {
            print!("{}", m.profile.report());
        }
    }
}
