use core_math::{vec3, Pose, Rng, Vec3};
use core_runtime::Pool;
use core_space::Shape;
use core_world::{Material, World};
use phys_mechanics::{Integrator, Mechanics, MutualGravitySettings, Settings};
use std::f64::consts::PI;

fn body_material(mass: f64, radius: f64) -> Material {
    let volume = (4.0 / 3.0) * PI * radius.powi(3);
    Material::new("body", mass / volume, 1e8, 0.3, 0.0, 1.0)
}

#[test]
fn mutual_gravity_gradients_match_central_differences_for_direct_and_barnes_hut() {
    for &theta in &[0.0, 0.5] {
        let mut world = World::new();
        world.gravity = Vec3::ZERO;
        let mut rng = Rng::new(77);
        let mat = world.add_material(body_material(1e10, 0.05));
        for i in 0..32 {
            let pos = vec3(
                -4.0 + 8.0 * rng.uniform(),
                -4.0 + 8.0 * rng.uniform(),
                -4.0 + 8.0 * rng.uniform(),
            );
            world.add_rigid(
                &format!("b{i}"),
                mat,
                Shape::Sphere { radius: 0.05 },
                Pose::at(pos),
            );
        }
        let pool = Pool::new(2);
        let mut solver = Mechanics::new(Settings {
            mutual_gravity: Some(MutualGravitySettings {
                constant: 1.0,
                softening: 0.1,
                theta,
            }),
            contact_precision: Some(1e-3),
            ..Default::default()
        });
        let (error, _) = solver.gradient_check(&pool, &world, 0.01, |b| {
            let k = (b % 5) as f64;
            vec3(1e-3 * (k + 1.0), -8e-4 * k, 5e-4 * (2.0 - k))
        });
        assert!(
            error < 1e-7,
            "mutual gravity FD gradient check failed for theta={theta}: {error}"
        );
    }
}

#[test]
fn two_body_circular_kepler_orbit_keeps_radius_energy_and_angular_momentum() {
    let g = 1.0_f64;
    let m1 = 100.0_f64;
    let m2 = 1.0_f64;
    let sep = 2.0_f64;
    let total_m = m1 + m2;
    let r1 = sep * m2 / total_m;
    let r2 = sep * m1 / total_m;
    let omega = (g * total_m / sep.powi(3)).sqrt();
    let period = 2.0 * PI / omega;

    let mut world = World::new();
    world.gravity = Vec3::ZERO;
    let mat1 = world.add_material(body_material(m1, 0.05));
    let mat2 = world.add_material(body_material(m2, 0.05));
    let b1 = world.add_rigid(
        "star",
        mat1,
        Shape::Sphere { radius: 0.05 },
        Pose::at(vec3(-r1, 0.0, 0.0)),
    );
    let b2 = world.add_rigid(
        "planet",
        mat2,
        Shape::Sphere { radius: 0.05 },
        Pose::at(vec3(r2, 0.0, 0.0)),
    );
    world.set_velocity(b1, vec3(0.0, -omega * r1, 0.0), Vec3::ZERO);
    world.set_velocity(b2, vec3(0.0, omega * r2, 0.0), Vec3::ZERO);

    let pool = Pool::new(2);
    let mut solver = Mechanics::new(Settings {
        integrator: Integrator::TrBdf2,
        energy_accounting: true,
        mutual_gravity: Some(MutualGravitySettings {
            constant: g,
            softening: 0.0,
            theta: 0.0,
        }),
        ..Default::default()
    });
    let steps = 200;
    let h = period / steps as f64;
    let t0 = world.totals();
    let mut e0 = None;
    let mut e_end = 0.0;
    for s in 0..steps {
        let report = solver.step(&pool, &mut world, h);
        assert!(!report.failed());
        let ledger = report.energy.unwrap();
        if s == 0 {
            e0 = Some(ledger.before.total());
        }
        e_end = ledger.after.total();
        let p1 = world.object(b1).centre_of_mass().unwrap();
        let p2 = world.object(b2).centre_of_mass().unwrap();
        assert!(
            ((p2 - p1).length() - sep).abs() < 2e-3,
            "circular orbit radius drifted: {}",
            (p2 - p1).length()
        );
    }
    let e0 = e0.unwrap();
    let t_end = world.totals();
    let p2_end = world.object(b2).centre_of_mass().unwrap();
    assert!((p2_end - vec3(r2, 0.0, 0.0)).length() < 1e-2);
    assert!((t_end.momentum - t0.momentum).length() < 1e-10);
    let ang_rel =
        (t_end.angular_momentum - t0.angular_momentum).length() / t0.angular_momentum.length();
    let eng_rel = (e_end - e0).abs() / e0.abs();
    assert!(
        ang_rel < 2e-3,
        "circular angular momentum rel drift: {ang_rel}"
    );
    assert!(eng_rel < 2e-3, "circular energy rel drift: {eng_rel}");

    let run_circular = |n_steps: usize| -> (f64, f64) {
        let g = 1.0_f64;
        let m1 = 100.0_f64;
        let m2 = 1.0_f64;
        let mu = g * (m1 + m2);
        let sep = 2.0_f64;
        let r1 = sep * m2 / (m1 + m2);
        let r2 = sep * m1 / (m1 + m2);
        let v = (mu / sep).sqrt();
        let period = 2.0 * PI * (sep.powi(3) / mu).sqrt();
        let mut w = World::new();
        w.gravity = Vec3::ZERO;
        let mat1 = w.add_material(body_material(m1, 0.04));
        let mat2 = w.add_material(body_material(m2, 0.02));
        let b1 = w.add_rigid(
            "star",
            mat1,
            Shape::Sphere { radius: 0.04 },
            Pose::at(vec3(-r1, 0.0, 0.0)),
        );
        let b2 = w.add_rigid(
            "planet",
            mat2,
            Shape::Sphere { radius: 0.02 },
            Pose::at(vec3(r2, 0.0, 0.0)),
        );
        w.set_velocity(b1, vec3(0.0, -v * m2 / (m1 + m2), 0.0), Vec3::ZERO);
        w.set_velocity(b2, vec3(0.0, v * m1 / (m1 + m2), 0.0), Vec3::ZERO);
        let mut s = Mechanics::new(Settings {
            integrator: Integrator::TrBdf2,
            energy_accounting: true,
            mutual_gravity: Some(MutualGravitySettings {
                constant: g,
                softening: 0.0,
                theta: 0.0,
            }),
            ..Default::default()
        });
        let h = period / n_steps as f64;
        let t_init = w.totals();
        let mut e_init = None;
        let mut e_last = 0.0;
        for step_idx in 0..n_steps {
            let report = s.step(&pool, &mut w, h);
            assert!(!report.failed());
            let ledger = report.energy.unwrap();
            if step_idx == 0 {
                e_init = Some(ledger.before.total());
            }
            e_last = ledger.after.total();
        }
        let t_last = w.totals();
        let e_init = e_init.unwrap();
        let l_err = (t_last.angular_momentum - t_init.angular_momentum).length()
            / t_init.angular_momentum.length();
        let e_err = (e_last - e_init).abs() / e_init.abs();
        (l_err, e_err)
    };
    let (l_err_200, e_err_200) = run_circular(200);
    let (l_err_400, e_err_400) = run_circular(400);
    assert!(
        l_err_200 / l_err_400 >= 3.0,
        "circular angular momentum error did not shrink >=3x on 200->400 refinement: {l_err_200} vs {l_err_400}"
    );
    assert!(
        e_err_200 / e_err_400 >= 3.0,
        "circular energy error did not shrink >=3x on 200->400 refinement: {e_err_200} vs {e_err_400}"
    );
}

#[test]
fn elliptical_kepler_orbit_matches_apoapsis_periapsis_and_conservation() {
    let g = 1.0_f64;
    let mu = 100.0_f64;
    let ecc = 0.5_f64;
    let a = 2.0_f64;
    let r_peri = a * (1.0 - ecc);
    let r_apo = a * (1.0 + ecc);
    let v_peri = (mu * (1.0 + ecc) / r_peri).sqrt();
    let period = 2.0 * PI * (a.powi(3) / mu).sqrt();

    let m1 = 99.0_f64;
    let m2 = 1.0_f64;
    let mut world = World::new();
    world.gravity = Vec3::ZERO;
    let mat1 = world.add_material(body_material(m1, 0.04));
    let mat2 = world.add_material(body_material(m2, 0.02));
    let b1 = world.add_rigid(
        "star",
        mat1,
        Shape::Sphere { radius: 0.04 },
        Pose::at(vec3(-r_peri * m2 / mu, 0.0, 0.0)),
    );
    let b2 = world.add_rigid(
        "planet",
        mat2,
        Shape::Sphere { radius: 0.02 },
        Pose::at(vec3(r_peri * m1 / mu, 0.0, 0.0)),
    );
    world.set_velocity(b1, vec3(0.0, -v_peri * m2 / mu, 0.0), Vec3::ZERO);
    world.set_velocity(b2, vec3(0.0, v_peri * m1 / mu, 0.0), Vec3::ZERO);

    let pool = Pool::new(2);
    let mut solver = Mechanics::new(Settings {
        integrator: Integrator::TrBdf2,
        energy_accounting: true,
        mutual_gravity: Some(MutualGravitySettings {
            constant: g,
            softening: 0.0,
            theta: 0.0,
        }),
        ..Default::default()
    });
    let steps = 240;
    let h = period / steps as f64;
    let mut min_r = f64::INFINITY;
    let mut max_r = 0.0_f64;
    let mut e0 = None;
    let mut e_end = 0.0;
    let t0 = world.totals();
    for s in 0..steps {
        let report = solver.step(&pool, &mut world, h);
        assert!(!report.failed());
        let ledger = report.energy.unwrap();
        if s == 0 {
            e0 = Some(ledger.before.total());
        }
        e_end = ledger.after.total();
        let d = (world.object(b2).centre_of_mass().unwrap()
            - world.object(b1).centre_of_mass().unwrap())
        .length();
        min_r = min_r.min(d);
        max_r = max_r.max(d);
    }
    assert!(
        (min_r - r_peri).abs() < 5e-3,
        "periapsis {min_r} vs {r_peri}"
    );
    assert!((max_r - r_apo).abs() < 1e-2, "apoapsis {max_r} vs {r_apo}");
    let t_end = world.totals();
    assert!((t_end.momentum - t0.momentum).length() < 1e-10);
    let ang_rel =
        (t_end.angular_momentum - t0.angular_momentum).length() / t0.angular_momentum.length();
    let eng_rel = (e_end - e0.unwrap()).abs() / e0.unwrap().abs();
    assert!(
        ang_rel < 5e-3,
        "elliptical angular momentum rel drift: {ang_rel}"
    );
    assert!(eng_rel < 5e-3, "elliptical energy rel drift: {eng_rel}");
}

#[test]
fn hundred_body_barnes_hut_cluster_conserves_momentum_and_replays_across_threads() {
    let run = |threads| {
        let pool = Pool::new(threads);
        let mut world = World::new();
        world.gravity = Vec3::ZERO;
        let mat = world.add_material(body_material(1.0, 0.03));
        let mut rng = Rng::new(12345);
        for i in 0..100 {
            let pos = vec3(
                -2.0 + 4.0 * rng.uniform(),
                -2.0 + 4.0 * rng.uniform(),
                -2.0 + 4.0 * rng.uniform(),
            );
            let vel = vec3(
                -0.1 + 0.2 * rng.uniform(),
                -0.1 + 0.2 * rng.uniform(),
                -0.1 + 0.2 * rng.uniform(),
            );
            let b = world.add_rigid(
                &format!("p{i}"),
                mat,
                Shape::Sphere { radius: 0.03 },
                Pose::at(pos),
            );
            world.set_velocity(b, vel, Vec3::ZERO);
        }
        let p0 = world.totals().momentum;
        let mut solver = Mechanics::new(Settings {
            mutual_gravity: Some(MutualGravitySettings {
                constant: 1e-2,
                softening: 0.1,
                theta: 0.5,
            }),
            ..Default::default()
        });
        for _ in 0..10 {
            let r = solver.step(&pool, &mut world, 0.05);
            assert!(!r.failed());
        }
        let p_end = world.totals().momentum;
        assert!(
            (p_end - p0).length() < 1e-11,
            "momentum drifted: {:?}",
            p_end - p0
        );
        solver.checkpoint(&world).to_bytes()
    };

    assert_eq!(run(1), run(4));
}
