//! Independence, conservative coupling, rollback and portable history.
use core_math::{vec3, Pose, Quat, Vec3};
use core_runtime::Pool;
use core_space::Shape;
use core_world::{library, JointDrive, Material, ObjectId, Representation, TetMesh, World};
use phys_mechanics::{Integrator, Mechanics, MechanicsState, Settings};

fn step(s: &mut Mechanics, pool: &Pool, w: &mut World, h: f64) -> phys_mechanics::StepReport {
    let r = s.step(pool, w, h);
    assert!(!r.failed(), "{r:?}");
    r
}

fn oscillator(remote_contact: bool) -> (World, ObjectId) {
    let mut w = World::new();
    let m = w.add_material(Material::new(
        "soft elastic",
        1000.0,
        5000.0,
        0.25,
        0.4,
        0.0,
    ));
    // Every island sees this same fixed support; it must not join them.
    w.add_fixed("floor", m, Shape::ground(0.0), Pose::IDENTITY);
    if remote_contact {
        w.add_rigid(
            "tiny resting sphere",
            m,
            Shape::Sphere { radius: 0.01 },
            Pose::at(vec3(10.0, 0.01, 0.0)),
        );
    }
    let id = w.add_solid(
        "oscillator",
        m,
        TetMesh::cuboid(Vec3::splat(0.2), 0.1),
        Pose::at(vec3(0.0, 2.0, 0.0)),
    );
    let Representation::Solid(s) = &mut w.object_mut(id).representation else {
        unreachable!()
    };
    for p in &mut s.nodes.position {
        p.x *= 1.03;
    }
    (w, id)
}

fn assert_same_solid(a: &World, ai: ObjectId, b: &World, bi: ObjectId) {
    let (Representation::Solid(a), Representation::Solid(b)) =
        (&a.object(ai).representation, &b.object(bi).representation)
    else {
        unreachable!()
    };
    assert_eq!(a.nodes.position, b.nodes.position);
    assert_eq!(a.nodes.velocity, b.nodes.velocity);
}

#[test]
fn distant_contact_and_tiny_geometry_do_not_change_isolated_motion() {
    for integrator in [
        Integrator::BackwardEuler,
        Integrator::Bdf2,
        Integrator::Newmark,
        Integrator::TrBdf2,
    ] {
        for threads in [1, 4] {
            let pool = Pool::new(threads);
            let (mut alone, ai) = oscillator(false);
            let (mut together, bi) = oscillator(true);
            let settings = Settings {
                integrator,
                ..Default::default()
            };
            let (mut a, mut b) = (Mechanics::new(settings), Mechanics::new(settings));
            let mut contacts = 0;
            for _ in 0..24 {
                let ar = step(&mut a, &pool, &mut alone, 1.0 / 120.0);
                let br = step(&mut b, &pool, &mut together, 1.0 / 120.0);
                assert_eq!(ar.islands, 1);
                assert_eq!(br.islands, 2);
                assert_eq!(ar.largest_island_blocks, br.largest_island_blocks);
                contacts += br.grips;
                assert_same_solid(&alone, ai, &together, bi);
            }
            assert!(
                contacts > 0,
                "remote sphere must actually contact the floor"
            );
        }
    }
}

fn approaching(speed: f64) -> World {
    let mut w = World::new();
    w.gravity = Vec3::ZERO;
    let m = w.add_material(Material {
        friction: 0.0,
        restitution: 1.0,
        ..library::rubber()
    });
    for (x, v) in [(-0.5, speed), (0.5, -speed)] {
        let id = w.add_rigid(
            "sphere",
            m,
            Shape::Sphere { radius: 0.1 },
            Pose::at(Vec3::X * x),
        );
        w.set_velocity(id, Vec3::X * v, Vec3::ZERO);
    }
    w
}

#[test]
fn swept_bounds_join_fast_bodies_before_they_can_cross() {
    let pool = Pool::new(2);
    let mut w = approaching(20.0);
    let mut s = Mechanics::new(Settings::default());
    // Each free endpoint lies beyond the other body; current-only and
    // end-only tests would both miss the collision.
    let r = step(&mut s, &pool, &mut w, 0.05);
    assert_eq!(r.islands, 1);
    assert_eq!(r.largest_island_blocks, 8);
    assert!(r.impacts > 0);
    assert!(
        w.objects()[1].centre_of_mass().unwrap().x - w.objects()[0].centre_of_mass().unwrap().x
            > 0.19
    );
    assert!(w.totals().momentum.length() < 1e-9);
}

#[test]
fn components_split_again_and_replay_without_cached_solver_state() {
    let h = 1.0 / 120.0;
    let mut w = approaching(2.0);
    let pool = Pool::new(1);
    let mut s = Mechanics::new(Settings::default());
    s.prepare(&pool, &w);
    let start_world = w.clone();
    let start = s.checkpoint(&w);
    let mut partitions = Vec::new();
    let mut middle = None;
    for i in 0..60 {
        let r = step(&mut s, &pool, &mut w, h);
        partitions.push(r.islands);
        if middle.is_none() && r.impacts > 0 {
            middle = Some((i + 1, w.clone(), s.checkpoint(&w)));
        }
    }
    assert_eq!(partitions[0], 2);
    let joined = partitions.iter().position(|&n| n == 1).unwrap();
    assert!(partitions[joined + 1..].contains(&2));
    let expected = s.checkpoint(&w).to_bytes();
    // Fresh storage, a portable checkpoint and a different worker count.
    let mut replay = start_world;
    let pool = Pool::new(4);
    let mut other = Mechanics::new(Settings::default());
    other.prepare(&pool, &replay);
    let restored =
        MechanicsState::from_bytes(&start.to_bytes(), &other.checkpoint(&replay)).unwrap();
    other.restore_checkpoint(&mut replay, &restored);
    for n in partitions {
        assert_eq!(step(&mut other, &pool, &mut replay, h).islands, n);
    }
    assert_eq!(other.checkpoint(&replay).to_bytes(), expected);
    // Restore during contact, when restart flags and affine history matter.
    let (completed, mut mid_world, mid_state) = middle.expect("fixture must strike");
    let mut mid_solver = Mechanics::new(Settings::default());
    mid_solver.prepare(&pool, &mid_world);
    let mid_state =
        MechanicsState::from_bytes(&mid_state.to_bytes(), &mid_solver.checkpoint(&mid_world))
            .unwrap();
    mid_solver.restore_checkpoint(&mut mid_world, &mid_state);
    for _ in completed..60 {
        step(&mut mid_solver, &pool, &mut mid_world, h);
    }
    assert_eq!(mid_solver.checkpoint(&mid_world).to_bytes(), expected);
    // Restore an old checkpoint into a solver retaining the *final* split caches.
    s.restore_checkpoint(&mut w, &start);
    for _ in 0..60 {
        step(&mut s, &pool, &mut w, h);
    }
    assert_eq!(s.checkpoint(&w).to_bytes(), expected);
}

#[test]
fn failure_in_a_later_island_undoes_already_advanced_islands() {
    let pool = Pool::new(1);
    let mut w = World::new();
    let m = w.add_material(library::rubber());
    w.add_rigid(
        "first, free",
        m,
        Shape::Sphere { radius: 0.1 },
        Pose::at(Vec3::Y),
    );
    w.add_rigid(
        "second, invalid",
        m,
        Shape::Sphere { radius: 0.1 },
        Pose::at(Vec3::X * 10.0),
    );
    w.add_fixed(
        "overlap",
        m,
        Shape::cuboid(Vec3::ONE),
        Pose::at(Vec3::X * 10.0),
    );
    let mut s = Mechanics::new(Settings::default());
    s.prepare(&pool, &w);
    let before = s.checkpoint(&w).to_bytes();
    let r = s.step(&pool, &mut w, 1.0 / 60.0);
    assert!(r.failed() && r.invalid);
    assert_eq!(r.islands, 2);
    assert_eq!(r.island_solves, 2, "first island must have been attempted");
    assert!(
        r.iterations > 0,
        "first island must have advanced before rollback"
    );
    assert_eq!(s.checkpoint(&w).to_bytes(), before);
    // Remove the overlap without changing layout; a failed worker cache is reusable.
    let Representation::Fixed(f) = &mut w.objects_mut()[2].representation else {
        unreachable!()
    };
    f.pose.position.y = -10.0;
    step(&mut s, &pool, &mut w, 1.0 / 60.0);
}

#[test]
fn fixed_only_world_advances_once_without_an_empty_newton_solve() {
    let pool = Pool::new(1);
    let mut w = World::new();
    let m = w.add_material(library::concrete());
    w.add_fixed("floor", m, Shape::ground(0.0), Pose::IDENTITY);
    let mut s = Mechanics::new(Settings::default());
    for _ in 0..4 {
        let before = w.time;
        let r = step(&mut s, &pool, &mut w, 0.1);
        assert_eq!(w.time, before + 0.1);
        assert_eq!(r.islands + r.solves + r.island_solves, 0);
    }
}

/// A zero-velocity compressed solid expands far beyond its initial bounds;
/// ballistic prediction cannot see the contact. Verification must retry with
/// both the remote dynamic object and newly reached fixed support present.
fn expanding(fixed: bool) -> World {
    let mut w = World::new();
    w.gravity = Vec3::ZERO;
    let m = w.add_material(Material::new("elastic", 1000.0, 1e6, 0.25, 0.0, 0.0));
    let id = w.add_solid(
        "compressed",
        m,
        TetMesh::cuboid(Vec3::splat(0.2), 0.1),
        Pose::IDENTITY,
    );
    let Representation::Solid(s) = &mut w.object_mut(id).representation else {
        unreachable!()
    };
    for p in &mut s.nodes.position {
        p.x *= 0.2;
    }
    if fixed {
        w.add_fixed(
            "wall",
            m,
            Shape::cuboid(vec3(0.1, 1.0, 1.0)),
            Pose::at(vec3(0.12, 0.0, 0.0)),
        );
    } else {
        w.add_rigid(
            "remote sphere",
            m,
            Shape::Sphere { radius: 0.02 },
            Pose::at(vec3(0.1, 0.0, 0.0)),
        );
    }
    w
}

#[test]
fn elastic_motion_outside_prediction_adds_missing_contacts_and_retries() {
    let pool = Pool::new(1);
    for fixed in [false, true] {
        let mut w = expanding(fixed);
        let mut reference = w.clone();
        let settings = Settings {
            contact_precision: Some(1e-4),
            ..Default::default()
        };
        let mut s = Mechanics::new(settings);
        let mut coupled = Mechanics::new(Settings {
            islands: false,
            ..settings
        });
        let r = step(&mut s, &pool, &mut w, 0.02);
        let cr = step(&mut coupled, &pool, &mut reference, 0.02);
        assert!(r.island_rebuilds > 0, "fixed={fixed}: {r:?}");
        assert_eq!(r.islands, 1);
        assert!(r.island_solves > 1);
        assert!(
            r.iterations >= cr.iterations,
            "discarded speculative work is counted"
        );
        let calls = s
            .profile
            .entries()
            .iter()
            .find(|e| e.name == "solve (all)")
            .unwrap()
            .calls;
        assert_eq!(
            calls as usize, r.solves,
            "delegated profiles retain every call"
        );
        assert!(r.linear_block_iterations <= r.largest_island_blocks * r.linear_iterations);
        // Once merged, solve and history match the full coupled reference exactly.
        assert_eq!(
            s.checkpoint(&w).to_bytes(),
            coupled.checkpoint(&reference).to_bytes()
        );
    }
}

#[test]
fn bdf_retry_resets_only_its_own_history_and_survives_portable_restore() {
    let pool = Pool::new(1);
    let (mut combined, solid) = oscillator(false);
    let mut solid_only = combined.clone();
    let m = combined.objects()[solid.0 as usize].material;
    let ball = combined.add_rigid(
        "independent",
        m,
        Shape::Sphere { radius: 0.1 },
        Pose::at(vec3(10.0, 2.0, 0.0)),
    );
    let mut alone = World::new();
    let m = alone.add_material(combined.material(m).clone());
    let alone_ball = alone.add_rigid(
        "independent",
        m,
        Shape::Sphere { radius: 0.1 },
        Pose::at(vec3(10.0, 2.0, 0.0)),
    );
    let settings = Settings {
        integrator: Integrator::Bdf2,
        // Keep this fixture focused on the existing BDF2 subdivision fallback.
        force_startup: false,
        max_iterations: 2,
        tolerance: 0.002,
        contact_precision: Some(1e-4),
        ..Default::default()
    };
    let mut s = Mechanics::new(settings);
    let mut a = Mechanics::new(settings);
    let mut solo_solid = Mechanics::new(Settings {
        islands: false,
        ..settings
    });
    let r = step(&mut s, &pool, &mut combined, 0.05);
    assert!(
        r.splits > 0,
        "fixture must exercise successful local subdivision"
    );
    assert_eq!(step(&mut a, &pool, &mut alone, 0.05).splits, 0);
    assert!(step(&mut solo_solid, &pool, &mut solid_only, 0.05).splits > 0);
    assert_same_solid(&combined, solid, &solid_only, solid);
    let saved = s.checkpoint(&combined);
    let mut replay = combined.clone();
    let other_pool = Pool::new(4);
    let mut restored = Mechanics::new(Settings {
        max_iterations: 100,
        ..settings
    });
    restored.prepare(&other_pool, &replay);
    let decoded =
        MechanicsState::from_bytes(&saved.to_bytes(), &restored.checkpoint(&replay)).unwrap();
    restored.restore_checkpoint(&mut replay, &decoded);
    s.settings.max_iterations = 100;
    a.settings.max_iterations = 100;
    solo_solid.settings.max_iterations = 100;
    for _ in 0..4 {
        step(&mut s, &pool, &mut combined, 0.05);
        step(&mut a, &pool, &mut alone, 0.05);
        step(&mut solo_solid, &pool, &mut solid_only, 0.05);
        step(&mut restored, &other_pool, &mut replay, 0.05);
        assert_same_solid(&combined, solid, &solid_only, solid);
        let (Representation::Rigid(b), Representation::Rigid(ab)) = (
            &combined.object(ball).representation,
            &alone.object(alone_ball).representation,
        ) else {
            unreachable!()
        };
        assert_eq!(
            (b.pose, b.velocity, b.spin),
            (ab.pose, ab.velocity, ab.spin)
        );
        assert_eq!(
            s.checkpoint(&combined).to_bytes(),
            restored.checkpoint(&replay).to_bytes()
        );
    }
}

#[test]
fn rotating_long_bodies_use_rotation_safe_bounds() {
    let pool = Pool::new(1);
    let mut w = World::new();
    w.gravity = Vec3::ZERO;
    let m = w.add_material(library::rubber());
    let bar = w.add_rigid(
        "spinning thin bar",
        m,
        Shape::cuboid(vec3(2.0, 0.05, 0.05)),
        Pose::IDENTITY,
    );
    w.set_velocity(bar, Vec3::ZERO, Vec3::Z * 90.0);
    w.add_rigid(
        "near the rotation sweep",
        m,
        Shape::Sphere { radius: 0.1 },
        Pose::at(Vec3::Y * 0.8),
    );
    // Their current shape AABBs do not overlap and their centres do not move.
    // The rotation sphere must still include the nearby body in the same solve.
    let mut s = Mechanics::new(Settings::default());
    let r = step(&mut s, &pool, &mut w, 1.0 / 240.0);
    assert_eq!(r.islands, 1);
    assert_eq!(r.largest_island_blocks, 8);
}

#[test]
fn multirate_subcycling_subcycles_fast_island_and_merges_on_midstep_collision() {
    let build = || {
        let mut w = World::new();
        w.gravity = Vec3::new(0.0, -9.81, 0.0);
        let wood = w.add_material(library::oak());
        w.add_fixed("floor", wood, Shape::ground(0.0), Pose::IDENTITY);

        // Island 1: resting box on the floor at x = -4.0 (slow / will sleep).
        let resting = w.add_rigid(
            "resting",
            wood,
            Shape::cuboid(Vec3::splat(0.2)),
            Pose::at(vec3(-4.0, 0.1001, 0.0)),
        );

        // Island 2: fast spinning projectile in the air at x = 2.0 (will subcycle under CFL).
        let fast = w.add_rigid(
            "fast",
            wood,
            Shape::Sphere { radius: 0.08 },
            Pose::at(vec3(2.0, 1.5, 0.0)),
        );
        w.set_velocity(fast, vec3(8.0, 0.0, 0.0), vec3(0.0, 0.0, 25.0));

        // Island 3: stiffly driven hinge actuator at x = 8.0 (will subcycle under natural frequency).
        let arm = w.add_rigid(
            "driven_arm",
            wood,
            Shape::cuboid(vec3(0.3, 0.06, 0.06)),
            Pose::at(vec3(8.15, 1.0, 0.0)),
        );
        let hinge = w.add_hinge(
            arm,
            None,
            vec3(-0.15, 0.0, 0.0),
            vec3(8.0, 1.0, 0.0),
            Vec3::Z,
            Quat::IDENTITY,
            2e6,
            5e4,
            Some((-1.0, 1.0)),
            JointDrive::pd(0.4, 8000.0, 120.0),
        );
        (w, resting, fast, arm, hinge)
    };

    let run = |threads| {
        let pool = Pool::new(threads);
        let (mut w, _resting, fast, _arm, _hinge) = build();
        let mut s = Mechanics::new(Settings {
            cfl_subcycling: Some(0.25),
            settle_time: Some(0.08),
            energy_accounting: true,
            ..Default::default()
        });
        let h = 1.0 / 60.0;
        let mut saw_subcycling = false;
        let mut saw_sleeping = false;
        for _ in 0..30 {
            let r = step(&mut s, &pool, &mut w, h);
            assert_eq!(r.islands, 3);
            if r.max_island_substeps > 1 {
                saw_subcycling = true;
            }
            if r.sleeping_islands > 0 {
                saw_sleeping = true;
            }
            let ledger = r.energy.expect("energy accounting enabled");
            assert!(ledger.residual.is_finite());
        }
        assert!(
            saw_subcycling,
            "fast and stiffly driven islands must subcycle with cfl_subcycling"
        );
        assert!(
            saw_sleeping,
            "resting island must sleep while fast island subcycles"
        );
        let Representation::Rigid(rf) = &w.object(fast).representation else {
            unreachable!()
        };
        assert!(rf.pose.position.x > 5.0);
        s.checkpoint(&w).to_bytes()
    };

    assert_eq!(run(1), run(4));
}

#[test]
fn world_mounted_joint_island_settles_and_wakes_on_joint_target_change() {
    let pool = Pool::new(2);
    let mut w = World::new();
    let mat = w.add_material(library::aluminium());
    // World-mounted damped hinge (b: None) with NO Fixed support object in the world.
    let link = w.add_rigid(
        "world_hinged_arm",
        mat,
        Shape::cuboid(vec3(0.3, 0.06, 0.06)),
        Pose::at(vec3(0.15, 0.0, 0.0)),
    );
    let hinge = w.add_hinge(
        link,
        None,
        vec3(-0.15, 0.0, 0.0),
        Vec3::ZERO,
        Vec3::Z,
        Quat::IDENTITY,
        2e6,
        5e4,
        Some((-1.0, 1.0)),
        JointDrive::pd(0.0, 4000.0, 180.0),
    );

    let mut s = Mechanics::new(Settings {
        settle_time: Some(0.05),
        ..Default::default()
    });
    let h = 1.0 / 240.0;
    let mut settled = false;
    for _ in 0..120 {
        let r = step(&mut s, &pool, &mut w, h);
        if r.sleeping_islands == 1 && r.solves == 0 {
            settled = true;
            break;
        }
    }
    assert!(
        settled,
        "damped world-mounted hinge should settle without requiring a Fixed support object"
    );

    // Changing the joint target in-place must immediately wake the settled island.
    w.set_joint_target(hinge, 0.35);
    let wake = step(&mut s, &pool, &mut w, h);
    assert_eq!(wake.sleeping_islands, 0);
    assert!(wake.solves > 0);
    for _ in 0..80 {
        step(&mut s, &pool, &mut w, h);
    }
    let q = w.joint_coordinate(hinge).unwrap();
    assert!(
        (q - 0.35).abs() < 2e-2,
        "woken hinge should track new 0.35 rad target, got {q}"
    );
}
