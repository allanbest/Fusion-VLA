#[allow(dead_code)]
#[path = "../examples/support/rolling.rs"]
mod rolling;
use core_math::Vec3;
use core_math::{Pose, Quat};
use core_runtime::Pool;
use core_space::Shape;
use core_world::{Material, Representation, World};
use phys_mechanics::{Mechanics, MechanicsState, Settings};

#[test]
fn friction_spins_a_sliding_sphere_up_to_the_known_rolling_speed() {
    let r = rolling::level(1.0 / 120.0, 1, Vec3::ZERO, true).unwrap();
    assert!((r.velocity - r.expected_velocity).length() < 1e-3);
    assert!((r.spin - r.expected_spin).length() < 1e-2);
    assert!(r.slip < 1e-3);
    assert!((r.energy_ratio - 5.0 / 7.0).abs() < 1e-3);
    assert!((-r.friction_work / r.initial_energy - 2.0 / 7.0).abs() < 0.02);
}

#[test]
fn already_rolling_spheres_keep_their_speed_without_rolling_resistance() {
    let r = rolling::level(1.0 / 120.0, 1, -Vec3::Z * 10.0, false).unwrap();
    assert!((r.velocity - Vec3::X).length() < 1e-3);
    assert!((r.spin + Vec3::Z * 10.0).length() < 0.01);
    assert!((r.energy_ratio - 1.0).abs() < 1e-3);
    assert!(r.slip < 1e-3);
}

#[test]
fn skew_sliding_and_rolling_approach_the_vector_reference_under_time_refinement() {
    let spin = rolling::launches()[2].1;
    let coarse = rolling::level(1.0 / 60.0, 1, spin, false).unwrap();
    let fine = rolling::level(1.0 / 240.0, 4, spin, false).unwrap();
    assert!((fine.velocity - fine.expected_velocity).length() < 1e-3);
    assert!((fine.spin - fine.expected_spin).length() < 0.1);
    assert!(
        (fine.spin - fine.expected_spin).length()
            < 0.4 * (coarse.spin - coarse.expected_spin).length()
    );
    assert!(fine.slip < 0.2 * coarse.slip);
    assert!((fine.energy_ratio - fine.expected_energy_ratio).abs() < 1e-3);
}

#[test]
fn rolling_contact_is_exactly_replayable_across_threads_and_energy_diagnostics() {
    let a = rolling::level(1.0 / 120.0, 1, Vec3::ZERO, false).unwrap();
    let b = rolling::level(1.0 / 120.0, 4, Vec3::ZERO, true).unwrap();
    assert_eq!(a.bytes, b.bytes);
}

#[test]
fn a_saved_spinning_contact_rebuilds_surface_witnesses_and_replays_exactly() {
    let mut w = World::new();
    let m = w.add_material(Material::new("sphere", 1000.0, 1e6, 0.3, 0.4, 0.0));
    w.add_fixed("floor", m, Shape::ground(0.0), Pose::IDENTITY);
    let id = w.add_rigid(
        "sphere",
        m,
        Shape::Sphere { radius: 0.1 },
        Pose::at(Vec3::Y * 0.1),
    );
    w.set_velocity(id, Vec3::X, Vec3::ZERO);
    let pool = Pool::new(1);
    let settings = Settings {
        contact_precision: Some(1e-4),
        tolerance: 1e-5,
        ..Default::default()
    };
    let mut solver = Mechanics::new(settings);
    for _ in 0..20 {
        assert!(!solver.step(&pool, &mut w, 1.0 / 240.0).failed());
    }
    let bytes = solver.checkpoint(&w).to_bytes();
    let mut resumed = w.clone();
    let other_pool = Pool::new(4);
    let mut other = Mechanics::new(settings);
    other.prepare(&other_pool, &resumed);
    let decoded = MechanicsState::from_bytes(&bytes, &other.checkpoint(&resumed)).unwrap();
    other.restore_checkpoint(&mut resumed, &decoded);
    for _ in 0..80 {
        assert!(!solver.step(&pool, &mut w, 1.0 / 240.0).failed());
        assert!(!other.step(&other_pool, &mut resumed, 1.0 / 240.0).failed());
        assert_eq!(
            solver.checkpoint(&w).to_bytes(),
            other.checkpoint(&resumed).to_bytes()
        );
    }
}

#[test]
fn a_capsule_uses_its_surface_radius_and_axial_inertia_when_rolling() {
    let mut w = World::new();
    let m = w.add_material(Material::new("capsule", 1000.0, 1e6, 0.3, 0.4, 0.0));
    let radius = 0.1;
    w.add_fixed("floor", m, Shape::ground(0.0), Pose::IDENTITY);
    let id = w.add_rigid(
        "capsule",
        m,
        Shape::Capsule {
            radius,
            half_length: 0.15,
        },
        Pose::new(
            Vec3::Y * radius,
            Quat::from_axis_angle(Vec3::X, std::f64::consts::FRAC_PI_2),
        ),
    );
    w.set_velocity(id, Vec3::X, Vec3::ZERO);
    let Representation::Rigid(body) = &w.object(id).representation else {
        unreachable!()
    };
    let c = body.world_inertia()[(2, 2)] / (body.mass * radius * radius);
    let expected = 1.0 / (1.0 + c);
    let pool = Pool::new(2);
    let mut s = Mechanics::new(Settings {
        contact_precision: Some(1e-4),
        tolerance: 1e-5,
        ..Default::default()
    });
    for _ in 0..120 {
        let r = s.step(&pool, &mut w, 1.0 / 240.0);
        assert!(!r.failed(), "{r:?}");
    }
    let Representation::Rigid(body) = &w.object(id).representation else {
        unreachable!()
    };
    assert!((body.velocity.x - expected).abs() < 0.005, "{body:?}");
    assert!((body.spin.z + expected / radius).abs() < 0.05, "{body:?}");
    assert!((body.velocity.x + body.spin.z * radius).abs() < 0.005);
}

#[test]
fn resolved_oblique_impacts_match_the_impulse_reference_and_bound_angular_drift() {
    for e in [0.0, 0.5] {
        for dt in [1.0 / 240.0, 1.0 / 480.0, 1.0 / 960.0] {
            let r = rolling::oblique(dt, 1, e).unwrap();
            // Post-impact solver potential no longer returns as separation,
            // so the inelastic rows are as tight as the restitutive ones.
            assert!(
                r.velocity_error < 2e-3,
                "e={e}, dt={dt}: {}",
                r.velocity_error
            );
            assert!(r.spin_error < 0.02, "e={e}, dt={dt}: {}", r.spin_error);
            assert!(
                r.angular_error < 1e-3,
                "e={e}, dt={dt}: {}",
                r.angular_error
            );
            assert!(r.momentum_error < 1e-8);
            assert!((r.energy_ratio - r.expected_energy_ratio).abs() < 1e-3);
            if dt == 1.0 / 240.0 || dt == 1.0 / 960.0 {
                let other = rolling::oblique(dt, 4, e).unwrap();
                assert_eq!(r.bytes, other.bytes);
            }
        }
    }
}

#[test]
fn separating_oblique_impact_clears_friction_without_discarded_refreshes() {
    use core_math::vec3;
    let (radius, mu, dt) = (0.1, 0.4, 1.0 / 240.0);
    let mut w = World::new();
    w.gravity = Vec3::ZERO;
    let m = w.add_material(Material::new("oblique", 1000.0, 1e6, 0.3, mu, 0.0));
    let common = vec3(0.1, 0.2, -0.3);
    let positions = [-Vec3::X * 0.11, Vec3::X * 0.11];
    let velocities = [common + vec3(1.0, 0.0, 0.25), common - vec3(1.0, 0.0, 0.25)];
    for i in 0..2 {
        let id = w.add_rigid(
            "sphere",
            m,
            Shape::Sphere { radius },
            Pose::at(positions[i]),
        );
        w.set_velocity(id, velocities[i], Vec3::ZERO);
    }
    let pool = Pool::new(1);
    let mut solver = Mechanics::new(Settings {
        contact_precision: Some(1e-5),
        tolerance: 1e-6,
        ..Default::default()
    });
    solver.prepare(&pool, &w);
    let mut post_impact = None;
    for _ in 0..10 {
        let r = solver.step(&pool, &mut w, dt);
        assert!(!r.failed(), "{r:?}");
        assert_eq!(r.refreshes_discarded, 0, "{r:?}");
        let Representation::Rigid(b0) = &w.objects()[0].representation else {
            unreachable!()
        };
        if r.grips > 0 {
            post_impact = Some((b0.velocity, b0.spin));
        } else if let Some((v_impact, spin_impact)) = post_impact {
            // Friction is gone. The pair is still inside the barrier after the
            // post-impact hold, so one more stage may see a residual push
            // through the stage's affine A. It must stay far below the
            // fixture's resolved velocity/spin error and never separate.
            let Representation::Rigid(b1) = &w.objects()[1].representation else {
                unreachable!()
            };
            let normal = (b0.pose.position - b1.pose.position).normalize();
            let change = b0.velocity - v_impact;
            assert!(change.length() < 1e-5, "{change:?}");
            assert!(normal.dot(change) < 1e-6, "{change:?}");
            assert!((b0.spin - spin_impact).length() < 1e-4);
        }
    }
    assert!(post_impact.is_some());
}
