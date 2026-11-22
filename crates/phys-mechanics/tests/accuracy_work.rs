use core_math::{vec3, Pose, Vec3};
use core_runtime::Pool;
use core_space::Shape;
use core_world::{library, Material, Representation, World};
use phys_mechanics::{Integrator, Mechanics, MechanicsState, Settings};

fn ball_world() -> World {
    let mut w = World::new();
    let m = w.add_material(library::rubber());
    w.add_rigid(
        "ball",
        m,
        Shape::Sphere { radius: 0.1 },
        Pose::at(Vec3::Y * 2.0),
    );
    w
}
fn rigid(w: &World, i: usize) -> &core_world::Rigid {
    let Representation::Rigid(r) = &w.objects()[i].representation else {
        unreachable!()
    };
    r
}
fn advance(s: &mut Mechanics, p: &Pool, w: &mut World, h: f64) -> phys_mechanics::StepReport {
    let r = s.step(p, w, h);
    assert!(!r.failed(), "{r:?}");
    r
}

#[test]
fn second_order_startup_matches_exact_gravity_from_the_first_interval() {
    for integrator in [Integrator::Newmark, Integrator::TrBdf2, Integrator::Bdf2] {
        let pool = Pool::new(1);
        let mut w = ball_world();
        let g = w.gravity;
        let mut s = Mechanics::new(Settings {
            integrator,
            energy_accounting: true,
            ..Settings::default()
        });
        let h = 0.05;
        let r = advance(&mut s, &pool, &mut w, h);
        assert!(
            (rigid(&w, 0).pose.position - (Vec3::Y * 2.0 + g * (0.5 * h * h))).length() < 1e-12
        );
        assert!((rigid(&w, 0).velocity - g * h).length() < 1e-11);
        let ledger = r.energy.unwrap();
        assert!(ledger.residual.abs() < 1e-10, "{ledger:?}");
        assert!((ledger.after.total() - ledger.before.total()).abs() < 1e-10);
        assert_eq!(
            ledger.friction_work + ledger.restitution_work + ledger.barrier_work,
            0.0
        );
    }
}

#[test]
fn variable_intervals_and_gravity_changes_keep_bdf_history_consistent_and_replayable() {
    let pool = Pool::new(1);
    let mut w = ball_world();
    let mut s = Mechanics::new(Settings {
        integrator: Integrator::Bdf2,
        ..Settings::default()
    });
    let mut t = 0.0;
    for h in [0.01, 0.025, 0.008, 0.04, 0.015] {
        advance(&mut s, &pool, &mut w, h);
        t += h;
        assert!(
            (rigid(&w, 0).pose.position - (Vec3::Y * 2.0 + w.gravity * (0.5 * t * t))).length()
                < 2e-11
        );
        assert!((rigid(&w, 0).velocity - w.gravity * t).length() < 2e-10);
    }
    w.gravity = vec3(0.2, -0.5, 0.1);
    let saved = s.checkpoint(&w);
    let mut replay = w.clone();
    let mut other = Mechanics::new(s.settings);
    other.prepare(&pool, &replay);
    let decoded =
        MechanicsState::from_bytes(&saved.to_bytes(), &other.checkpoint(&replay)).unwrap();
    other.restore_checkpoint(&mut replay, &decoded);
    let (x, v) = (rigid(&w, 0).pose.position, rigid(&w, 0).velocity);
    let h = 0.02;
    advance(&mut s, &pool, &mut w, h);
    advance(&mut other, &pool, &mut replay, h);
    assert!(
        (rigid(&w, 0).pose.position - (x + v * h + w.gravity * (0.5 * h * h))).length() < 1e-11
    );
    assert!((rigid(&w, 0).velocity - (v + w.gravity * h)).length() < 1e-10);
    assert_eq!(
        s.checkpoint(&w).to_bytes(),
        other.checkpoint(&replay).to_bytes()
    );
}

fn stationary_balls() -> World {
    let mut w = ball_world();
    w.gravity = Vec3::ZERO;
    let m = w.objects()[0].material;
    for i in 1..8 {
        w.add_rigid(
            "rest",
            m,
            Shape::Sphere { radius: 0.1 },
            Pose::at(vec3(i as f64 * 2.0, 2.0, 0.0)),
        );
    }
    w
}

#[test]
fn exact_equilibrium_sleep_preserves_bits_and_wakes_for_gravity_and_step_changes() {
    let pool = Pool::new(4);
    let mut a = stationary_balls();
    let mut b = a.clone();
    let mut sa = Mechanics::new(Settings::default());
    let mut sb = Mechanics::new(Settings {
        sleeping: false,
        parallel_islands: false,
        ..Settings::default()
    });
    let mut slept = 0;
    for _ in 0..8 {
        slept += advance(&mut sa, &pool, &mut a, 0.01).sleeping_islands;
        advance(&mut sb, &pool, &mut b, 0.01);
        assert_eq!(sa.checkpoint(&a).to_bytes(), sb.checkpoint(&b).to_bytes());
    }
    assert!(slept >= 8);
    a.gravity = Vec3::Y * -0.5;
    b.gravity = a.gravity;
    assert_eq!(advance(&mut sa, &pool, &mut a, 0.01).sleeping_islands, 0);
    advance(&mut sb, &pool, &mut b, 0.01);
    assert_eq!(sa.checkpoint(&a).to_bytes(), sb.checkpoint(&b).to_bytes());
    assert!((rigid(&a, 0).velocity - a.gravity * 0.01).length() < 1e-11);
    assert_eq!(advance(&mut sa, &pool, &mut a, 0.02).sleeping_islands, 0);
    advance(&mut sb, &pool, &mut b, 0.02);
    assert_eq!(sa.checkpoint(&a).to_bytes(), sb.checkpoint(&b).to_bytes());
}

#[test]
fn sleeping_body_wakes_before_a_new_collider_reaches_it() {
    let pool = Pool::new(4);
    let mut w = stationary_balls();
    let mut s = Mechanics::new(Settings::default());
    for _ in 0..4 {
        advance(&mut s, &pool, &mut w, 0.01);
    }
    // Change an existing far body's physical input. It will join the sleeping
    // target's swept group; no cache may freeze either side of the collision.
    let Representation::Rigid(r) = &mut w.objects_mut()[1].representation else {
        unreachable!()
    };
    r.pose.position = vec3(0.5, 2.0, 0.0);
    r.velocity = -Vec3::X * 10.0;
    let mut reference = w.clone();
    let state = s.checkpoint(&w);
    let mut other = Mechanics::new(Settings {
        sleeping: false,
        parallel_islands: false,
        ..Settings::default()
    });
    other.prepare(&pool, &reference);
    other.restore_checkpoint(&mut reference, &state);
    for _ in 0..8 {
        let r = advance(&mut s, &pool, &mut w, 0.01);
        advance(&mut other, &pool, &mut reference, 0.01);
        assert!(r.sleeping_islands < 8);
        assert_eq!(
            s.checkpoint(&w).to_bytes(),
            other.checkpoint(&reference).to_bytes()
        );
    }
}

#[test]
fn parallel_islands_and_optional_energy_accounting_preserve_exact_trajectories() {
    let pool = Pool::new(4);
    let mut a = ball_world();
    let m = a.objects()[0].material;
    for i in 1..8 {
        let id = a.add_rigid(
            "moving",
            m,
            Shape::cuboid(Vec3::splat(0.1)),
            Pose::at(vec3(i as f64, 2.0, 0.0)),
        );
        a.set_velocity(id, vec3(0.1, 0.2, -0.3), vec3(0.2, 0.0, 0.1));
    }
    let mut b = a.clone();
    let mut sa = Mechanics::new(Settings {
        energy_accounting: true,
        ..Settings::default()
    });
    let mut sb = Mechanics::new(Settings {
        parallel_islands: false,
        ..Settings::default()
    });
    for _ in 0..12 {
        let r = advance(&mut sa, &pool, &mut a, 1.0 / 120.0);
        advance(&mut sb, &pool, &mut b, 1.0 / 120.0);
        assert!(r.energy.is_some());
        assert_eq!(sa.checkpoint(&a).to_bytes(), sb.checkpoint(&b).to_bytes());
    }
}

#[test]
fn friction_work_is_dissipative_and_matches_the_known_stopping_energy() {
    let pool = Pool::new(1);
    let mut w = World::new();
    let m = w.add_material(Material::new("slider", 1000.0, 1e5, 0.3, 0.4, 0.0));
    w.add_fixed("floor", m, Shape::ground(0.0), Pose::IDENTITY);
    w.add_rigid(
        "box",
        m,
        Shape::cuboid(Vec3::splat(0.2)),
        Pose::at(Vec3::Y * 0.1),
    );
    w.set_velocity(core_world::ObjectId(1), Vec3::X * 2.0, Vec3::ZERO);
    let kinetic = w.totals().kinetic_energy;
    let mut s = Mechanics::new(Settings {
        energy_accounting: true,
        ..Settings::default()
    });
    let mut friction = 0.0;
    for _ in 0..120 {
        let e = advance(&mut s, &pool, &mut w, 1.0 / 120.0).energy.unwrap();
        assert!(e.friction_work <= 1e-14, "{e:?}");
        friction += e.friction_work;
    }
    assert!(w.totals().kinetic_energy < 0.001 * kinetic);
    assert!(
        (-friction / kinetic - 1.0).abs() < 0.04,
        "{friction} vs {kinetic}"
    );
}

#[test]
fn rejected_intervals_publish_no_accepted_energy_and_restore_every_history_field() {
    let pool = Pool::new(1);
    let mut w = ball_world();
    let mut s = Mechanics::new(Settings {
        energy_accounting: true,
        ..Settings::default()
    });
    advance(&mut s, &pool, &mut w, 0.01);
    w.gravity *= 0.5;
    let before = s.checkpoint(&w).to_bytes();
    s.settings.max_iterations = 0;
    let r = s.step(&pool, &mut w, 0.02);
    assert!(r.failed());
    assert!(r.energy.is_none());
    assert_eq!(s.checkpoint(&w).to_bytes(), before);
}

#[test]
fn version_one_checkpoints_remain_readable_and_new_timing_metadata_is_checked() {
    let pool = Pool::new(1);
    let mut w = ball_world();
    let mut s = Mechanics::new(Settings::default());
    s.prepare(&pool, &w);
    let state = s.checkpoint(&w);
    let full = state.to_bytes();
    // Version 4 appends the per-solid plasticity state (8 bytes for 0 solids).
    let v4_tail = 8;
    let mut v3 = full[..full.len() - v4_tail].to_vec();
    v3[..8].copy_from_slice(&3u64.to_le_bytes());
    let loaded = MechanicsState::from_bytes(&v3, &state).unwrap();
    assert_eq!(loaded.to_bytes(), full, "v3 decodes cleanly to v4");
    // Version 3 appends the per-rigid unloading budget: count + one f64.
    let v3_tail = 8 + 8 * s.integration_shape().1;
    let mut v2 = v3[..v3.len() - v3_tail].to_vec();
    v2[..8].copy_from_slice(&2u64.to_le_bytes());
    let loaded = MechanicsState::from_bytes(&v2, &state).unwrap();
    assert_eq!(loaded.to_bytes(), full, "v2 decodes with a zero budget");
    let mut old = v2;
    // A prepared state has gravity present (1+24 bytes), previous h absent (1).
    old.truncate(old.len() - 26);
    old[..8].copy_from_slice(&1u64.to_le_bytes());
    let loaded = MechanicsState::from_bytes(&old, &state).unwrap();
    s.restore_checkpoint(&mut w, &loaded);
    advance(&mut s, &pool, &mut w, 0.01);
    let state = s.checkpoint(&w);
    let mut invalid = state.to_bytes();
    let n = invalid.len() - v3_tail - v4_tail;
    invalid[n - 8..n].copy_from_slice(&0.0_f64.to_le_bytes());
    assert!(MechanicsState::from_bytes(&invalid, &state).is_err());
    let mut invalid = state.to_bytes();
    let n = invalid.len() - v4_tail;
    invalid[n - 8..n].copy_from_slice(&(-1.0_f64).to_le_bytes());
    assert!(MechanicsState::from_bytes(&invalid, &state).is_err());
}

#[allow(dead_code)]
#[path = "../examples/support/efficiency.rs"]
mod efficiency;
#[test]
fn parallel_large_elastic_islands_match_serial_internal_reductions_exactly() {
    let w = efficiency::assembly(2, 0.075);
    let a = efficiency::run(
        w.clone(),
        Settings {
            parallel_islands: false,
            ..Settings::default()
        },
        4,
        3,
    )
    .unwrap();
    let b = efficiency::run(
        w.clone(),
        Settings {
            parallel_islands: true,
            ..Settings::default()
        },
        4,
        3,
    )
    .unwrap();
    assert_eq!(a.bytes, b.bytes);
    assert_eq!((a.newton, a.cg, a.products), (b.newton, b.cg, b.products));
}

#[test]
fn unmet_temporal_accuracy_budget_rolls_back_a_converged_nonlinear_solve() {
    let pool = Pool::new(1);
    let mut w = World::new();
    w.gravity = Vec3::ZERO;
    let m = w.add_material(Material::new(
        "spinning elastic",
        1000.0,
        1e5,
        0.3,
        0.0,
        0.0,
    ));
    let id = w.add_solid(
        "strained",
        m,
        core_world::TetMesh::cuboid(Vec3::splat(0.2), 0.1),
        Pose::IDENTITY,
    );
    let Representation::Solid(solid) = &mut w.object_mut(id).representation else {
        unreachable!()
    };
    for p in &mut solid.nodes.position {
        p.x *= 1.3;
    }
    w.set_velocity(id, Vec3::ZERO, Vec3::Z * 2.0);
    let mut s = Mechanics::new(Settings {
        angular_tolerance: Some(1e-12),
        accuracy_subdivisions: 0,
        tolerance: 1e-5,
        contact_precision: Some(1e-4),
        energy_accounting: true,
        ..Settings::default()
    });
    s.prepare(&pool, &w);
    let before = s.checkpoint(&w).to_bytes();
    let r = s.step(&pool, &mut w, 0.02);
    assert!(r.accuracy_failed && r.newton.converged, "{r:?}");
    assert!(r.energy.is_none());
    assert_eq!(s.checkpoint(&w).to_bytes(), before);
}
