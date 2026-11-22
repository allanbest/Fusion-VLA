//! Settled islands sleep: a resting stack costs no solves, a thrown body is
//! never frozen in the air, a sleeping stack wakes when struck, and replay
//! through sleep and wake stays exact.
use core_math::{vec3, Pose, Vec3};
use core_runtime::Pool;
use core_space::Shape;
use core_world::{library, ObjectId, Representation, World};
use phys_mechanics::{Mechanics, Settings};

const H: f64 = 1.0 / 60.0;

fn floor(world: &mut World) {
    let wood = world.add_material(library::oak());
    world.add_fixed(
        "floor",
        wood,
        Shape::cuboid(vec3(4.0, 1.0, 4.0)),
        Pose::at(vec3(0.0, -0.5, 0.0)),
    );
}

fn add_box(world: &mut World, name: &str, at: Vec3) -> ObjectId {
    let wood = world.add_material(library::oak());
    world.add_rigid(name, wood, Shape::cuboid(Vec3::splat(0.2)), Pose::at(at))
}

fn rigid(world: &World, id: ObjectId) -> (Vec3, Vec3, Vec3) {
    let Representation::Rigid(r) = &world.object(id).representation else {
        unreachable!()
    };
    (r.pose.position, r.velocity, r.spin)
}

/// Two boxes stacked on a floor, released from just above contact.
fn stack() -> (World, ObjectId, ObjectId) {
    let mut world = World::new();
    floor(&mut world);
    let lower = add_box(&mut world, "lower", vec3(0.0, 0.1001, 0.0));
    let upper = add_box(&mut world, "upper", vec3(0.0, 0.3002, 0.0));
    (world, lower, upper)
}

#[test]
fn a_resting_stack_settles_and_costs_no_solves() {
    let (mut world, lower, upper) = stack();
    let pool = Pool::new(2);
    let mut solver = Mechanics::new(Settings {
        energy_accounting: true,
        ..Settings::default()
    });
    let mut settled_at = None;
    for step in 0..120 {
        let report = solver.step(&pool, &mut world, H);
        assert!(!report.failed(), "step {step}: {report:?}");
        if report.sleeping_islands > 0 && settled_at.is_none() {
            settled_at = Some(step);
            // Settling removes what little motion was left, and says so.
            let ledger = report.energy.expect("accounting is on");
            assert!(ledger.settling_work <= 0.0, "{ledger:?}");
            assert!(ledger.residual.abs() < 1e-9, "{ledger:?}");
        }
        if settled_at.is_some() {
            assert_eq!(report.island_solves, 0, "step {step}: {report:?}");
            assert_eq!(report.solves, 0, "step {step}: {report:?}");
        }
    }
    let settled_at = settled_at.expect("the stack never settled");
    assert!(settled_at < 90, "settled only at step {settled_at}");
    for id in [lower, upper] {
        let (_, velocity, spin) = rigid(&world, id);
        assert_eq!((velocity, spin), (Vec3::ZERO, Vec3::ZERO));
    }
    assert!((rigid(&world, lower).0.y - 0.1).abs() < 2e-3);
    assert!((rigid(&world, upper).0.y - 0.3).abs() < 4e-3);
}

#[test]
fn a_thrown_box_is_not_frozen_at_the_top_of_its_arc() {
    let mut world = World::new();
    floor(&mut world);
    let thrown = add_box(&mut world, "thrown", vec3(0.0, 1.0, 0.0));
    world.set_velocity(thrown, vec3(0.0, 1.0, 0.0), Vec3::ZERO);
    let pool = Pool::new(1);
    let mut solver = Mechanics::new(Settings::default());
    // The apex is at 0.10 s; follow it to 0.25 s, still well above the floor.
    for step in 0..15 {
        let report = solver.step(&pool, &mut world, H);
        assert!(!report.failed(), "step {step}: {report:?}");
        assert_eq!(report.sleeping_islands, 0, "step {step}: frozen in the air");
    }
    let t = 15.0 * H;
    let expected = 1.0 + 1.0 * t - 0.5 * 9.81 * t * t;
    let (position, velocity, _) = rigid(&world, thrown);
    assert!(
        (position.y - expected).abs() < 1e-3,
        "{} vs {expected}",
        position.y
    );
    assert!(velocity.y < -1.0, "{velocity:?}");
}

#[test]
fn a_settled_box_wakes_when_another_lands_on_it() {
    let mut world = World::new();
    floor(&mut world);
    let lower = add_box(&mut world, "lower", vec3(0.0, 0.1001, 0.0));
    // Lands at about 0.45 s, long after the lower box has settled.
    let upper = add_box(&mut world, "upper", vec3(0.0, 1.2, 0.0));
    let pool = Pool::new(2);
    let mut solver = Mechanics::new(Settings::default());
    let mut asleep_while_falling = false;
    for step in 0..20 {
        let report = solver.step(&pool, &mut world, H);
        assert!(!report.failed(), "step {step}: {report:?}");
        asleep_while_falling |= report.sleeping_islands > 0;
    }
    assert!(asleep_while_falling, "the lower box never settled");
    let mut struck = false;
    for step in 20..90 {
        let report = solver.step(&pool, &mut world, H);
        assert!(!report.failed(), "step {step}: {report:?}");
        struck |= rigid(&world, lower).1 != Vec3::ZERO;
    }
    assert!(struck, "the lower box never woke");
    // Both end up at rest, one on the other.
    assert!((rigid(&world, lower).0.y - 0.1).abs() < 2e-3);
    assert!((rigid(&world, upper).0.y - 0.3).abs() < 4e-3);
}

#[test]
fn sleep_and_wake_replay_exactly_from_checkpoints_and_across_threads() {
    let run = |threads: usize| {
        let mut world = World::new();
        floor(&mut world);
        add_box(&mut world, "lower", vec3(0.0, 0.1001, 0.0));
        add_box(&mut world, "upper", vec3(0.03, 1.2, 0.0));
        let pool = Pool::new(threads);
        let mut solver = Mechanics::new(Settings::default());
        let mut saved = None;
        for step in 0..90 {
            assert!(!solver.step(&pool, &mut world, H).failed());
            if step == 15 {
                // The lower box is asleep and the upper one is falling.
                saved = Some(solver.checkpoint(&world));
            }
        }
        let end = solver.checkpoint(&world).to_bytes();
        solver.restore_checkpoint(&mut world, saved.as_ref().unwrap());
        for _ in 16..90 {
            assert!(!solver.step(&pool, &mut world, H).failed());
        }
        assert_eq!(
            solver.checkpoint(&world).to_bytes(),
            end,
            "{threads} threads"
        );
        end
    };
    assert_eq!(run(1), run(4));
}

#[test]
fn without_a_settle_time_a_resting_stack_keeps_solving() {
    let (mut world, _, _) = stack();
    let pool = Pool::new(1);
    let mut solver = Mechanics::new(Settings {
        settle_time: None,
        ..Settings::default()
    });
    let mut last = None;
    for _ in 0..120 {
        last = Some(solver.step(&pool, &mut world, H));
    }
    assert!(last.unwrap().island_solves > 0);
}

#[test]
fn setting_a_sleeping_body_moving_wakes_it() {
    let (mut world, _, upper) = stack();
    let pool = Pool::new(1);
    let mut solver = Mechanics::new(Settings::default());
    for _ in 0..120 {
        assert!(!solver.step(&pool, &mut world, H).failed());
    }
    assert_eq!(
        rigid(&world, upper).1,
        Vec3::ZERO,
        "the stack never settled"
    );
    let before = rigid(&world, upper).0;
    world.set_velocity(upper, vec3(0.5, 0.0, 0.0), Vec3::ZERO);
    let report = solver.step(&pool, &mut world, H);
    assert!(!report.failed(), "{report:?}");
    assert!(report.island_solves > 0, "{report:?}");
    assert!(
        rigid(&world, upper).0.x - before.x > 1e-3,
        "it did not move"
    );
}

#[test]
fn a_thrown_box_over_a_halfspace_ground_is_not_frozen_at_exact_apex() {
    let mut world = World::new();
    let wood = world.add_material(library::oak());
    world.add_fixed("ground", wood, Shape::ground(0.0), Pose::IDENTITY);
    // Launch well above the ground with an upward velocity that reaches its
    // exact apex between steps 5 and 6 so a step has nearly zero displacement.
    let thrown = add_box(&mut world, "thrown", vec3(0.0, 2.0, 0.0));
    let v0 = 5.5 * core_math::constants::STANDARD_GRAVITY * H;
    world.set_velocity(thrown, vec3(0.0, v0, 0.0), Vec3::ZERO);
    let pool = Pool::new(2);
    let mut solver = Mechanics::new(Settings::default());
    for step in 0..15 {
        let report = solver.step(&pool, &mut world, H);
        assert!(!report.failed(), "step {step}: {report:?}");
        assert_eq!(report.sleeping_islands, 0, "step {step}: frozen in mid-air");
    }
    let t = 15.0 * H;
    let expected = 2.0 + v0 * t - 0.5 * core_math::constants::STANDARD_GRAVITY * t * t;
    let (position, velocity, _) = rigid(&world, thrown);
    assert!(
        (position.y - expected).abs() < 1e-3,
        "{} vs {expected}",
        position.y
    );
    assert!(velocity.y < -0.5, "{velocity:?}");
}

#[test]
fn rotating_a_sleeping_box_wakes_it() {
    let mut world = World::new();
    floor(&mut world);
    let id = add_box(&mut world, "box", vec3(0.0, 0.1001, 0.0));
    let pool = Pool::new(1);
    let mut solver = Mechanics::new(Settings::default());
    let mut settled = false;
    for _ in 0..120 {
        let r = solver.step(&pool, &mut world, H);
        assert!(!r.failed(), "{r:?}");
        if r.sleeping_islands > 0 {
            settled = true;
            break;
        }
    }
    assert!(settled, "box never settled");
    let Representation::Rigid(r) = &mut world.object_mut(id).representation else {
        unreachable!()
    };
    r.pose.rotation = core_math::Quat::from_axis_angle(Vec3::Z, 0.7);
    r.pose.position.y = 0.2;
    let mut woke = false;
    for _ in 0..10 {
        let report = solver.step(&pool, &mut world, H);
        assert!(!report.failed(), "{report:?}");
        woke |= report.island_solves > 0;
    }
    assert!(woke, "rotating a sleeping box did not wake it");
}

#[test]
fn a_resting_box_has_no_phantom_trbdf2_velocity() {
    let mut world = World::new();
    floor(&mut world);
    let id = add_box(&mut world, "box", vec3(0.0, 0.1001, 0.0));
    let pool = Pool::new(1);
    let mut solver = Mechanics::new(Settings {
        settle_time: None,
        ..Settings::default()
    });
    for _ in 0..120 {
        let report = solver.step(&pool, &mut world, H);
        assert!(!report.failed(), "{report:?}");
    }
    let (_, velocity, spin) = rigid(&world, id);
    assert!(
        velocity.length() < 1e-4,
        "phantom velocity persisted: {velocity:?}"
    );
    assert!(spin.length() < 1e-4, "phantom spin persisted: {spin:?}");
}
