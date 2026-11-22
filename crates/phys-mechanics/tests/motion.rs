use core_math::{vec3, Pose, Quat, Vec3};
use core_runtime::Pool;
use core_space::Shape;
use core_world::{library, Material, Representation, TetMesh, World};
use phys_mechanics::{Integrator, Mechanics, Settings};

const H: f64 = 1.0 / 60.0;

fn run(world: &mut World, mechanics: &mut Mechanics, seconds: f64) {
    let pool = Pool::new(4);
    let steps = (seconds / H).round() as usize;
    for _ in 0..steps {
        let report = mechanics.step(&pool, world, H);
        assert!(!report.failed(), "{report:?}");
    }
}

#[test]
fn free_fall() {
    let mut world = World::new();
    let steel = world.add_material(library::structural_steel());
    let rubber = world.add_material(library::rubber());
    let block = world.add_rigid(
        "block",
        steel,
        Shape::cuboid(Vec3::splat(0.2)),
        Pose::at(vec3(0.0, 5.0, 0.0)),
    );
    let jelly = world.add_solid(
        "jelly",
        rubber,
        TetMesh::cuboid(Vec3::splat(0.2), 0.1),
        Pose::at(vec3(3.0, 5.0, 0.0)),
    );
    let mut mechanics = Mechanics::new(Settings::default());
    run(&mut world, &mut mechanics, 1.0);
    let g = -world.gravity.y;
    for id in [block, jelly] {
        let o = world.object(id);
        let fall = 5.0 - o.centre_of_mass().unwrap().y;
        assert!(
            (fall / (0.5 * g) - 1.0).abs() < 1e-3,
            "{}: fell {fall}",
            o.name
        );
        let v = o.momentum().y / o.mass();
        // Newton stops within 10⁻³ ε; TR-BDF2 turns that into a velocity
        // error of a few 10⁻⁴ relative over a second of free fall.
        assert!((v + g).abs() < 5e-4 * g, "{}: v {v}", o.name);
    }
}

#[test]
fn torque_free_spin_keeps_momentum() {
    let mut world = World::new();
    world.gravity = Vec3::ZERO;
    let wood = world.add_material(library::oak());
    // Spinning mostly about the long (minimum) axis with a wobble.
    let id = world.add_rigid(
        "bar",
        wood,
        Shape::cuboid(vec3(0.6, 0.2, 0.1)),
        Pose::at(Vec3::ZERO),
    );
    world.set_velocity(id, Vec3::ZERO, vec3(6.0, 0.4, 0.3));
    let before = world.totals();
    let mut mechanics = Mechanics::new(Settings::default());
    run(&mut world, &mut mechanics, 2.0);
    let after = world.totals();
    assert!(
        (after.angular_momentum - before.angular_momentum).length()
            < 1e-9 * before.angular_momentum.length()
    );
    assert!(
        (after.kinetic_energy / before.kinetic_energy - 1.0).abs() < 1e-3,
        "{} → {}",
        before.kinetic_energy,
        after.kinetic_energy
    );
}

fn resting_world(friction: f64) -> (World, core_world::ObjectId) {
    let mut world = World::new();
    let ground = world.add_material(Material {
        friction,
        ..library::concrete()
    });
    let wood = world.add_material(Material {
        friction,
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
    (world, id)
}

#[test]
fn a_box_rests_on_the_ground() {
    for integrator in [
        Integrator::BackwardEuler,
        Integrator::Bdf2,
        Integrator::TrBdf2,
    ] {
        let (mut world, id) = resting_world(0.5);
        let mut mechanics = Mechanics::new(Settings {
            integrator,
            ..Default::default()
        });
        run(&mut world, &mut mechanics, 2.0);
        let Representation::Rigid(r) = &world.object(id).representation else {
            unreachable!()
        };
        let eps = mechanics.contact_precision();
        assert!(
            (r.pose.position.y - 0.1).abs() < eps,
            "{integrator:?}: height {}",
            r.pose.position.y
        );
        assert!(
            r.velocity.length() < 1e-3 && r.spin.length() < 1e-2,
            "{integrator:?}: {:?} {:?}",
            r.velocity,
            r.spin
        );
        assert!(
            r.pose.rotation.angle_to(Quat::IDENTITY) < 1e-3,
            "{integrator:?}"
        );
    }
}

#[test]
fn sliding_stops_at_coulombs_distance() {
    let mu: f64 = 0.4;
    let v = 2.0;
    let (mut world, id) = resting_world(mu);
    world.set_velocity(id, vec3(v, 0.0, 0.0), Vec3::ZERO);
    let mut mechanics = Mechanics::new(Settings::default());
    run(&mut world, &mut mechanics, 1.5);
    let Representation::Rigid(r) = &world.object(id).representation else {
        unreachable!()
    };
    let expected = v * v / (2.0 * mu * -world.gravity.y);
    assert!(
        (r.pose.position.x / expected - 1.0).abs() < 0.03,
        "slid {} vs {expected}",
        r.pose.position.x
    );
    assert!(r.velocity.length() < 1e-2);
}

/// A rigid ball dropped 1 m comes back to e² of it — no energy from the
/// impact carried into later steps, and the coefficient honoured.
#[test]
fn impacts_follow_newtons_law() {
    for integrator in [Integrator::TrBdf2, Integrator::Newmark, Integrator::Bdf2] {
        for e in [1.0, 0.5, 0.0] {
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
            let mut mechanics = Mechanics::new(Settings {
                integrator,
                ..Default::default()
            });
            let pool = Pool::new(2);
            let (mut rising, mut peak) = (false, 0.0f64);
            for _ in 0..150 {
                let report = mechanics.step(&pool, &mut world, H);
                assert!(!report.failed(), "{integrator:?}, e={e}: {report:?}");
                let Representation::Rigid(b) = &world.object(id).representation else {
                    unreachable!()
                };
                rising |= b.velocity.y > 0.1;
                if rising {
                    peak = peak.max(b.pose.position.y - r);
                }
            }
            assert!(
                (peak - e * e).abs() < 0.02,
                "{integrator:?}, e = {e}: rose {peak:.3} m"
            );
        }
    }
}

/// A small pile of boxes with friction lands and settles without one failed
/// step: friction on rigid bodies must measure slip from where each body's
/// points truly start (a mismatch reads as a phantom slip every stage, and
/// the pile freezes).
#[test]
fn a_pile_of_boxes_settles_without_failing() {
    let mut world = World::new();
    let ground = world.add_material(library::concrete());
    let wood = world.add_material(library::oak());
    world.add_fixed(
        "ground",
        ground,
        Shape::cuboid(vec3(20.0, 1.0, 20.0)),
        Pose::at(vec3(0.0, -0.5, 0.0)),
    );
    let mut rng = core_math::Rng::new(5);
    for i in 0..27 {
        let p = vec3(
            (i % 3) as f64 * 0.4 - 0.4,
            0.3 + (i / 9) as f64 * 0.4,
            ((i / 3) % 3) as f64 * 0.4 - 0.4,
        );
        let turn = Quat::from_axis_angle(rng.direction(), rng.range(0.0, 0.5));
        world.add_rigid(
            "box",
            wood,
            Shape::cuboid(Vec3::splat(0.2)),
            Pose::new(p, turn),
        );
    }
    let mut mechanics = Mechanics::new(Settings::default());
    let pool = Pool::new(4);
    for step in 0..120 {
        let report = mechanics.step(&pool, &mut world, H);
        assert!(!report.failed(), "step {step} failed: {report:?}");
    }
    let resting = world
        .objects()
        .iter()
        .filter_map(|o| match &o.representation {
            Representation::Rigid(r) => Some(r.velocity.length()),
            _ => None,
        });
    assert!(
        resting.fold(0.0, f64::max) < 0.1,
        "the pile is still moving"
    );
}
