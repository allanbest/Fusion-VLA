use core_math::{vec3, Pose, Quat, Vec3};
use core_runtime::Pool;
use core_space::Shape;
use core_world::{library, Material, Representation, TetMesh, World};
use phys_mechanics::{Integrator, Mechanics, Settings};

/// Neither bar has a corner over the other. Only their edge interiors meet.
#[test]
fn crossing_bars_cannot_pass_through_each_other() {
    let mut world = World::new();
    world.gravity = Vec3::ZERO;
    let material = world.add_material(Material {
        friction: 0.0,
        restitution: 0.0,
        ..library::oak()
    });
    world.add_fixed(
        "support",
        material,
        Shape::cuboid(vec3(2.0, 0.1, 0.1)),
        Pose::at(vec3(0.0, -0.05, 0.0)),
    );
    let bar = world.add_rigid(
        "falling bar",
        material,
        Shape::cuboid(vec3(0.1, 0.1, 2.0)),
        Pose::at(vec3(0.0, 0.2, 0.0)),
    );
    world.set_velocity(bar, vec3(0.0, -5.0, 0.0), Vec3::ZERO);
    let pool = Pool::new(2);
    let mut solver = Mechanics::new(Settings::default());
    for step in 0..4 {
        let report = solver.step(&pool, &mut world, 1.0 / 30.0);
        assert!(!report.failed(), "step {step}: {report:?}");
        let Representation::Rigid(body) = &world.object(bar).representation else {
            unreachable!()
        };
        assert!(
            body.pose.position.y >= 0.048,
            "step {step}: {:?}",
            body.pose
        );
    }
}

#[test]
fn fixed_corners_stop_a_face_with_no_nearby_moving_corners() {
    let mut world = World::new();
    world.gravity = Vec3::ZERO;
    let material = world.add_material(Material {
        friction: 0.0,
        restitution: 0.0,
        ..library::oak()
    });
    world.add_fixed(
        "support",
        material,
        Shape::cuboid(Vec3::splat(0.1)),
        Pose::at(Vec3::ZERO),
    );
    let plate = world.add_rigid(
        "plate",
        material,
        Shape::cuboid(vec3(2.0, 0.1, 2.0)),
        Pose::at(Vec3::Y * 0.2),
    );
    world.set_velocity(plate, -Vec3::Y * 5.0, Vec3::ZERO);
    let pool = Pool::new(2);
    let mut solver = Mechanics::new(Settings::default());
    for step in 0..4 {
        let report = solver.step(&pool, &mut world, 1.0 / 30.0);
        assert!(!report.failed(), "step {step}: {report:?}");
        assert!(world.object(plate).centre_of_mass().unwrap().y >= 0.098);
    }
}

fn moving_bars(restitution: f64) -> World {
    let mut world = World::new();
    world.gravity = Vec3::ZERO;
    let material = world.add_material(Material {
        friction: 0.0,
        restitution,
        ..library::oak()
    });
    for (size, y, velocity) in [
        (vec3(2.0, 0.1, 0.1), -0.1, Vec3::Y),
        (vec3(0.1, 0.1, 2.0), 0.1, -Vec3::Y),
    ] {
        let bar = world.add_rigid("bar", material, Shape::cuboid(size), Pose::at(Vec3::Y * y));
        world.set_velocity(bar, velocity, Vec3::ZERO);
    }
    world
}

#[test]
fn edge_impacts_conserve_momentum_and_follow_restitution() {
    for integrator in [
        Integrator::BackwardEuler,
        Integrator::Bdf2,
        Integrator::Newmark,
        Integrator::TrBdf2,
    ] {
        for restitution in [0.0, 0.5, 1.0] {
            let mut world = moving_bars(restitution);
            let before = world.totals();
            let pool = Pool::new(2);
            let mut solver = Mechanics::new(Settings {
                integrator,
                ..Default::default()
            });
            for step in 0..20 {
                let report = solver.step(&pool, &mut world, 1.0 / 120.0);
                assert!(
                    !report.failed(),
                    "{integrator:?}, e={restitution}, step {step}: {report:?}"
                );
                assert!(
                    world.objects()[1].centre_of_mass().unwrap().y
                        - world.objects()[0].centre_of_mass().unwrap().y
                        > 0.097
                );
            }
            let after = world.totals();
            assert!(
                (after.momentum - before.momentum).length() < 1e-7,
                "{integrator:?}: {:?}",
                after.momentum
            );
            assert!((after.angular_momentum - before.angular_momentum).length() < 1e-7);
            assert!(after.kinetic_energy <= before.kinetic_energy * 1.01);
            let relative = world.objects()[1].momentum().y / world.objects()[1].mass()
                - world.objects()[0].momentum().y / world.objects()[0].mass();
            assert!(
                (relative - 2.0 * restitution).abs() < 0.08,
                "{integrator:?}, e={restitution}: speed {relative}"
            );
        }
    }
}

#[test]
fn a_coarse_elastic_bar_contacts_at_edge_interiors() {
    let mut world = World::new();
    world.gravity = Vec3::ZERO;
    let material = world.add_material(Material::new("elastic", 1000.0, 1e7, 0.3, 0.0, 0.0));
    world.add_fixed(
        "support",
        material,
        Shape::cuboid(vec3(2.0, 0.1, 0.1)),
        Pose::at(-Vec3::Y * 0.05),
    );
    // One cell per axis: no surface node lies over the support.
    let bar = world.add_solid(
        "elastic bar",
        material,
        TetMesh::cuboid(vec3(0.1, 0.1, 2.0), 2.0),
        Pose::at(Vec3::Y * 0.2),
    );
    world.set_velocity(bar, -Vec3::Y * 2.0, Vec3::ZERO);
    let pool = Pool::new(2);
    let mut solver = Mechanics::new(Settings::default());
    for step in 0..16 {
        let report = solver.step(&pool, &mut world, 1.0 / 120.0);
        assert!(!report.failed(), "step {step}: {report:?}");
        assert!(
            world.object(bar).centre_of_mass().unwrap().y > 0.04,
            "step {step}"
        );
    }
}

#[test]
fn edge_contact_gradient_matches_energy() {
    let mut world = moving_bars(0.0);
    let Representation::Rigid(body) = &mut world.objects_mut()[1].representation else {
        unreachable!()
    };
    body.pose.position.y = 0.00005;
    let mut solver = Mechanics::new(Settings::default());
    let pool = Pool::new(2);
    for shift in [Vec3::ZERO, vec3(1e-6, -2e-7, 1e-6)] {
        let (error, energy) = solver.gradient_check(&pool, &world, 1.0 / 120.0, |_| shift);
        assert!(energy.is_finite());
        assert!(error < 1e-4, "gradient error {error}");
    }
}

#[test]
fn capsule_centrelines_collide_between_their_endpoint_spheres() {
    let mut world = World::new();
    world.gravity = Vec3::ZERO;
    let material = world.add_material(Material {
        friction: 0.0,
        restitution: 0.0,
        ..library::oak()
    });
    let capsule = Shape::Capsule {
        radius: 0.05,
        half_length: 1.0,
    };
    world.add_fixed(
        "support",
        material,
        capsule,
        Pose::new(
            -Vec3::Y * 0.05,
            Quat::from_axis_angle(Vec3::Z, std::f64::consts::FRAC_PI_2),
        ),
    );
    let moving = world.add_rigid(
        "moving",
        material,
        capsule,
        Pose::new(
            Vec3::Y * 0.2,
            Quat::from_axis_angle(Vec3::X, std::f64::consts::FRAC_PI_2),
        ),
    );
    world.set_velocity(moving, -Vec3::Y * 5.0, Vec3::ZERO);
    let pool = Pool::new(2);
    let mut solver = Mechanics::new(Settings::default());
    for step in 0..4 {
        let report = solver.step(&pool, &mut world, 1.0 / 30.0);
        assert!(!report.failed(), "step {step}: {report:?}");
        assert!(world.object(moving).centre_of_mass().unwrap().y > 0.049);
    }
}

#[test]
fn thin_objects_set_precision_by_thickness_and_contacts_replay_across_threads() {
    let mut expected = None;
    for threads in [1, 4] {
        let mut world = moving_bars(0.5);
        let pool = Pool::new(threads);
        let mut solver = Mechanics::new(Settings::default());
        solver.prepare(&pool, &world);
        assert!((solver.contact_precision() - 0.0002).abs() < 1e-15);
        for _ in 0..20 {
            assert!(!solver.step(&pool, &mut world, 1.0 / 120.0).failed());
        }
        let bytes = solver.checkpoint(&world).to_bytes();
        if let Some(expected) = &expected {
            assert_eq!(&bytes, expected);
        } else {
            expected = Some(bytes);
        }
        let saved = solver.checkpoint(&world);
        assert!(!solver.step(&pool, &mut world, 1.0 / 120.0).failed());
        let after = solver.checkpoint(&world).to_bytes();
        solver.restore_checkpoint(&mut world, &saved);
        assert!(!solver.step(&pool, &mut world, 1.0 / 120.0).failed());
        assert_eq!(solver.checkpoint(&world).to_bytes(), after);
    }
}

/// Two boxes exactly aligned, corner over corner: every edge of the upper
/// box's base lies parallel over one of the lower box's. Before parallel
/// rigid edges were smoothed, the closest-feature switch of those pairs sat
/// where Newton converges; the gradient alternated between two values and
/// every step failed, which is what stopped stacked and piled boxes.
#[test]
fn boxes_stacked_corner_on_corner_settle_and_land() {
    for drop in [0.0, 0.3] {
        let mut world = World::new();
        let wood = world.add_material(library::oak());
        world.add_fixed(
            "floor",
            wood,
            Shape::cuboid(vec3(4.0, 1.0, 4.0)),
            Pose::at(vec3(0.0, -0.5, 0.0)),
        );
        let lower = world.add_rigid(
            "lower",
            wood,
            Shape::cuboid(Vec3::splat(0.2)),
            Pose::at(vec3(0.0, 0.1001, 0.0)),
        );
        let upper = world.add_rigid(
            "upper",
            wood,
            Shape::cuboid(Vec3::splat(0.2)),
            Pose::at(vec3(0.0, 0.3002 + drop, 0.0)),
        );
        let pool = Pool::new(2);
        let mut solver = Mechanics::new(Settings::default());
        for step in 0..90 {
            let report = solver.step(&pool, &mut world, 1.0 / 60.0);
            assert!(!report.failed(), "drop {drop}, step {step}: {report:?}");
        }
        let height = |id| {
            let Representation::Rigid(body) = &world.object(id).representation else {
                unreachable!()
            };
            body.pose.position.y
        };
        // At rest one on the other, neither through the floor nor each other.
        assert!(
            (height(lower) - 0.1).abs() < 2e-3,
            "lower at {}",
            height(lower)
        );
        assert!(
            (height(upper) - 0.3).abs() < 4e-3,
            "upper at {}",
            height(upper)
        );
    }
}
