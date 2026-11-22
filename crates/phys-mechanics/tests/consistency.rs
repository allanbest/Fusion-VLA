use core_math::{vec3, Pose, Quat, Rng, Vec3};
use core_runtime::Pool;
use core_space::Shape;
use core_world::{library, Material, TetMesh, World};
use phys_mechanics::{Mechanics, Settings};

/// Every kind of term and pair at once: a rigid box and a soft block on the
/// ground, touching, moving, the box turned.
fn scene() -> World {
    let mut world = World::new();
    let ground = world.add_material(Material {
        friction: 0.5,
        ..library::concrete()
    });
    let wood = world.add_material(Material {
        friction: 0.5,
        ..library::oak()
    });
    let gel = world.add_material(Material {
        friction: 0.5,
        ..library::rubber()
    });
    world.add_fixed(
        "ground",
        ground,
        Shape::cuboid(vec3(20.0, 1.0, 20.0)),
        Pose::at(vec3(0.0, -0.5, 0.0)),
    );
    let b = world.add_rigid(
        "box",
        wood,
        Shape::cuboid(Vec3::splat(0.2)),
        Pose::new(vec3(0.0, 0.1001, 0.0), Quat::from_axis_angle(Vec3::Y, 0.3)),
    );
    world.set_velocity(b, vec3(1.0, -0.2, 0.0), vec3(0.0, 1.0, 0.5));
    let s = world.add_solid(
        "gel",
        gel,
        TetMesh::cuboid(Vec3::splat(0.2), 0.1).quadratic(),
        Pose::at(vec3(0.0, 0.3002, 0.0)),
    );
    world.set_velocity(s, vec3(-0.5, -0.5, 0.0), Vec3::ZERO);
    world
}

#[test]
fn the_step_gradient_matches_its_energy() {
    let world = scene();
    let pool = Pool::new(2);
    let mut mechanics = Mechanics::new(Settings::default());
    let mut rng = Rng::new(3);
    let precision = {
        mechanics.gradient_check(&pool, &world, 1.0 / 60.0, |_| Vec3::ZERO);
        mechanics.contact_precision()
    };
    for round in 0..5 {
        let shifts: Vec<Vec3> = (0..200)
            .map(|_| rng.direction() * (0.03 * precision * round as f64))
            .collect();
        let (error, energy) =
            mechanics.gradient_check(&pool, &world, 1.0 / 60.0, |b| shifts[b % shifts.len()]);
        assert!(energy.is_finite(), "round {round}");
        assert!(error < 1e-4, "round {round}: gradient error {error}");
    }
}
