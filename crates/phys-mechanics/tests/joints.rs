use core_math::{vec3, Pose, Quat, Vec3};
use core_runtime::Pool;
use core_space::Shape;
use core_world::{library, JointDrive, LinkPoint, Material, Probe, ProbeReading, TetMesh, World};
use phys_mechanics::{Integrator, Mechanics, Settings};
use std::f64::consts::PI;

fn spring_unit_mass(radius: f64) -> f64 {
    let volume = (4.0 / 3.0) * PI * radius.powi(3);
    1.0 / volume
}

#[test]
fn joint_gradients_match_central_differences() {
    for &with_damping in &[false, true] {
        let mut world = World::new();
        let mat = world.add_material(library::oak());
        let world_id = world.add_fixed(
            "wall",
            mat,
            Shape::cuboid(Vec3::splat(0.2)),
            Pose::new(vec3(-1.5, 0.0, 0.0), Quat::from_axis_angle(Vec3::Y, 0.3)),
        );
        let r0 = world.add_rigid(
            "body0",
            mat,
            Shape::cuboid(vec3(0.4, 0.3, 0.2)),
            Pose::new(vec3(0.0, 0.6, 0.1), Quat::from_axis_angle(Vec3::Z, 0.2)),
        );
        let r1 = world.add_rigid(
            "body1",
            mat,
            Shape::Capsule {
                radius: 0.1,
                half_length: 0.25,
            },
            Pose::new(vec3(0.8, 0.7, -0.2), Quat::from_axis_angle(Vec3::X, -0.35)),
        );
        let s0 = world.add_solid(
            "tet",
            mat,
            TetMesh::cuboid(Vec3::splat(0.3), 0.15),
            Pose::at(vec3(0.2, -0.4, 0.0)),
        );

        let damp = if with_damping { 12.0 } else { 0.0 };
        // Stretched nonzero-rest spring (tension) between rigid0 and node 0 of solid.
        world.add_spring(
            LinkPoint::Body {
                object: r0,
                local: vec3(0.15, -0.1, 0.05),
            },
            LinkPoint::Node {
                object: s0,
                node: 0,
            },
            0.3,
            800.0,
            damp,
        );
        // Compressed nonzero-rest spring between wall (fixed) and rigid1.
        world.add_spring(
            LinkPoint::Body {
                object: world_id,
                local: vec3(0.0, 0.1, 0.0),
            },
            LinkPoint::Body {
                object: r1,
                local: vec3(-0.05, 0.2, 0.0),
            },
            4.0,
            500.0,
            damp,
        );
        // Ball joint to world point.
        world.add_ball_joint(
            LinkPoint::Body {
                object: r0,
                local: vec3(-0.1, 0.1, 0.0),
            },
            LinkPoint::World(vec3(-0.15, 0.65, 0.05)),
            1200.0,
        );
        // Rigid-rigid weld + rigid-fixed weld.
        world.add_weld(
            r0,
            Some(r1),
            vec3(0.2, 0.0, 0.0),
            vec3(-0.1, 0.0, 0.0),
            Quat::from_axis_angle(Vec3::Y, 0.4),
            1500.0,
            300.0,
        );
        world.add_weld(
            r1,
            Some(world_id),
            vec3(0.0, -0.1, 0.0),
            vec3(2.0, 0.5, 0.0),
            Quat::from_axis_angle(Vec3::Z, -0.25),
            600.0,
            200.0,
        );
        // Hinge with active limit, PD drive, and smooth max_force saturation.
        world.add_hinge(
            r0,
            Some(r1),
            vec3(-0.05, 0.05, 0.02),
            vec3(0.05, -0.05, -0.02),
            vec3(0.2, 0.9, 0.1),
            Quat::from_axis_angle(Vec3::Z, 0.15),
            900.0,
            250.0,
            Some((-0.05, 0.05)),
            JointDrive::pd(-0.4, 80.0, damp)
                .with_velocity(0.3)
                .with_max_force(25.0),
        );
        // Slider with active limit, PD drive, and smooth max_force saturation.
        world.add_slider(
            r1,
            Some(world_id),
            vec3(0.02, 0.03, -0.01),
            vec3(1.8, 0.4, 0.0),
            vec3(1.0, 0.2, -0.1),
            Quat::from_axis_angle(Vec3::X, 0.1),
            700.0,
            180.0,
            Some((-0.1, 0.1)),
            JointDrive::pd(0.0, 120.0, damp)
                .with_velocity(-0.2)
                .with_max_force(40.0),
        );

        let pool = Pool::new(2);
        let mut solver = Mechanics::new(Settings {
            contact_precision: Some(1e-3),
            ..Default::default()
        });
        let (error, _) = solver.gradient_check(&pool, &world, 0.01, |b| {
            let k = (b % 7) as f64;
            vec3(0.002 * (k + 1.0), -0.0015 * k, 0.001 * (3.0 - k))
        });
        assert!(
            error < 1e-7,
            "joint FD gradient check (damping={with_damping}) failed: {error}"
        );
    }
}

#[test]
fn harmonic_oscillator_period_and_energy_conservation() {
    let radius = 0.05;
    let mut world = World::new();
    world.gravity = Vec3::ZERO;
    let mat = world.add_material(Material::new(
        "osc",
        spring_unit_mass(radius),
        1e7,
        0.3,
        0.0,
        0.0,
    ));
    let m = 1.0_f64;
    let k = 100.0_f64;
    let x0 = 0.1_f64;
    let body = world.add_rigid(
        "mass",
        mat,
        Shape::Sphere { radius },
        Pose::at(vec3(x0, 0.0, 0.0)),
    );
    assert!((world.object(body).mass() - m).abs() < 1e-12);
    world.add_spring(
        LinkPoint::World(Vec3::ZERO),
        LinkPoint::Body {
            object: body,
            local: Vec3::ZERO,
        },
        0.0,
        k,
        0.0,
    );

    let omega = (k / m).sqrt();
    let period = 2.0 * PI / omega;
    let steps = 250;
    let h = period / steps as f64;
    let pool = Pool::new(1);
    let mut solver = Mechanics::new(Settings {
        integrator: Integrator::TrBdf2,
        energy_accounting: true,
        ..Default::default()
    });
    let e0 = 0.5 * k * x0 * x0;
    for _ in 0..steps {
        let report = solver.step(&pool, &mut world, h);
        assert!(!report.failed());
    }
    let core_world::Representation::Rigid(r) = &world.object(body).representation else {
        unreachable!()
    };
    let e_end = 0.5 * m * r.velocity.length_squared() + 0.5 * k * r.pose.position.length_squared();
    assert!(
        (r.pose.position.x - x0).abs() < 5e-4,
        "after one period, expected x≈{x0}, got {}",
        r.pose.position.x
    );
    assert!(
        r.velocity.x.abs() < 5e-3,
        "after one period, expected v≈0, got {}",
        r.velocity.x
    );
    assert!(
        (e_end - e0).abs() / e0 < 5e-3,
        "energy drift too large: {e_end} vs {e0}"
    );
}

#[test]
fn damped_oscillator_decays_and_accounts_damping_work() {
    let radius = 0.05;
    let mut world = World::new();
    world.gravity = Vec3::ZERO;
    let mat = world.add_material(Material::new(
        "damped",
        spring_unit_mass(radius),
        1e7,
        0.3,
        0.0,
        0.0,
    ));
    let m = 1.0_f64;
    let k = 100.0_f64;
    let c = 2.0_f64;
    let x0 = 0.2_f64;
    let body = world.add_rigid(
        "mass",
        mat,
        Shape::Sphere { radius },
        Pose::at(vec3(x0, 0.0, 0.0)),
    );
    world.add_spring(
        LinkPoint::World(Vec3::ZERO),
        LinkPoint::Body {
            object: body,
            local: Vec3::ZERO,
        },
        0.0,
        k,
        c,
    );

    let pool = Pool::new(2);
    let mut solver = Mechanics::new(Settings {
        integrator: Integrator::TrBdf2,
        energy_accounting: true,
        ..Default::default()
    });
    let h = 1.0 / 600.0;
    let duration = 1.5_f64;
    let steps = (duration / h).round() as usize;
    let mut total_damping_work = 0.0;
    let mut max_residual = 0.0_f64;
    for _ in 0..steps {
        let report = solver.step(&pool, &mut world, h);
        assert!(!report.failed());
        let ledger = report.energy.unwrap();
        assert!(ledger.damping_work <= 0.0);
        total_damping_work += ledger.damping_work;
        max_residual = max_residual.max(ledger.residual.abs());
    }
    let omega_d = (k / m - (c / (2.0 * m)).powi(2)).sqrt();
    let t = world.time;
    let expected_x = x0
        * (-c * t / (2.0 * m)).exp()
        * ((omega_d * t).cos() + (c / (2.0 * m * omega_d)) * (omega_d * t).sin());
    let core_world::Representation::Rigid(r) = &world.object(body).representation else {
        unreachable!()
    };
    assert!(
        (r.pose.position.x - expected_x).abs() < 2e-3,
        "damped response {} differed from analytical {expected_x}",
        r.pose.position.x
    );
    assert!(total_damping_work < -1.0);
    assert!(max_residual < 1e-4, "residual too large: {max_residual}");
}

#[test]
fn rigid_pendulum_on_ball_joint_matches_period_and_conserves_energy() {
    let mut world = World::new();
    let mat = world.add_material(library::structural_steel());
    let rod_half = vec3(0.04, 0.5, 0.04);
    let pivot_local = vec3(0.0, rod_half.y, 0.0);
    let theta0 = 0.1_f64;
    let rot0 = Quat::from_axis_angle(Vec3::Z, theta0);
    let pos0 = rot0.rotate(-pivot_local);
    let rod = world.add_rigid(
        "pendulum",
        mat,
        Shape::cuboid(rod_half * 2.0),
        Pose::new(pos0, rot0),
    );
    world.add_ball_joint(
        LinkPoint::World(Vec3::ZERO),
        LinkPoint::Body {
            object: rod,
            local: pivot_local,
        },
        1e7,
    );

    let core_world::Representation::Rigid(r0) = &world.object(rod).representation else {
        unreachable!()
    };
    let i_cm = r0.inertia[(2, 2)];
    let m = r0.mass;
    let l = rod_half.y;
    let i_pivot = i_cm + m * l * l;
    let period = 2.0 * PI * (i_pivot / (m * world.gravity.length() * l)).sqrt();

    let steps = 240;
    let h = period / steps as f64;
    let pool = Pool::new(2);
    let mut solver = Mechanics::new(Settings {
        integrator: Integrator::TrBdf2,
        energy_accounting: true,
        ..Default::default()
    });
    let mut e0 = None;
    let mut e_end = 0.0;
    for step in 0..steps {
        let report = solver.step(&pool, &mut world, h);
        assert!(!report.failed());
        let ledger = report.energy.unwrap();
        if step == 0 {
            e0 = Some(ledger.before.total());
        }
        e_end = ledger.after.total();
    }
    let e0 = e0.unwrap();
    let core_world::Representation::Rigid(r) = &world.object(rod).representation else {
        unreachable!()
    };
    let pivot_err = r.pose.transform_point(pivot_local).length();
    let axis = r.pose.rotation.rotate(Vec3::Y);
    let final_theta = (-axis.x).atan2(axis.y);
    assert!(pivot_err < 1e-4, "pivot drifted by {pivot_err}");
    assert!(
        (final_theta - theta0).abs() < 5e-3,
        "expected angle≈{theta0} after 1 period, got {final_theta}"
    );
    assert!(
        (e_end - e0).abs() / e0.abs().max(1.0) < 2e-3,
        "pendulum energy drifted: {e_end} vs {e0}"
    );
}

#[test]
fn welded_rigid_bodies_move_together_and_replay_deterministically() {
    let build = || {
        let mut world = World::new();
        let mat = world.add_material(library::aluminium());
        let rel = Quat::from_axis_angle(Vec3::Z, 0.3);
        let a = world.add_rigid(
            "left",
            mat,
            Shape::cuboid(vec3(0.4, 0.2, 0.2)),
            Pose::at(vec3(-0.3, 1.0, 0.0)),
        );
        let local_a = vec3(0.25, 0.0, 0.0);
        let local_b = vec3(-0.25, 0.0, 0.0);
        let inv_rel = rel.conjugate();
        let pivot = vec3(-0.3, 1.0, 0.0) + local_a;
        let pos_b = pivot - inv_rel.rotate(local_b);
        let b = world.add_rigid(
            "right",
            mat,
            Shape::cuboid(vec3(0.4, 0.2, 0.2)),
            Pose::new(pos_b, inv_rel),
        );
        // Weld holds a at b * rel => rel = q_b^-1 * q_a = rel.
        world.add_weld(a, Some(b), local_a, local_b, rel, 1e6, 5e4);
        let spin = vec3(0.0, 0.0, 2.5);
        let v_a = vec3(0.5, 0.2, 0.0);
        world.set_velocity(a, v_a, spin);
        world.set_velocity(b, v_a + spin.cross(pos_b - vec3(-0.3, 1.0, 0.0)), spin);
        (world, a, b, local_a, local_b, rel)
    };

    let run = |threads| {
        let pool = Pool::new(threads);
        let (mut world, a, b, local_a, local_b, rel) = build();
        let mut solver = Mechanics::new(Settings::default());
        for _ in 0..40 {
            let report = solver.step(&pool, &mut world, 1.0 / 120.0);
            assert!(!report.failed(), "step failed: {report:?}");
        }
        let core_world::Representation::Rigid(ra) = &world.object(a).representation else {
            unreachable!()
        };
        let core_world::Representation::Rigid(rb) = &world.object(b).representation else {
            unreachable!()
        };
        let pos_err =
            (ra.pose.transform_point(local_a) - rb.pose.transform_point(local_b)).length();
        let rot_err = (ra.pose.rotation.to_mat3() - rb.pose.rotation.to_mat3() * rel.to_mat3())
            .frobenius_squared()
            .sqrt();
        assert!(pos_err < 5e-4, "weld point separated by {pos_err}");
        assert!(rot_err < 5e-4, "weld rotation drifted by {rot_err}");
        solver.checkpoint(&world).to_bytes()
    };

    assert_eq!(run(1), run(4));
}

#[test]
fn hinge_joint_pendulum_and_angle_limits_hold_axis_and_stop_at_limits() {
    let mut world = World::new();
    let mat = world.add_material(library::aluminium());
    let half = vec3(0.04, 0.3, 0.04);
    let pivot_local = vec3(0.0, half.y, 0.0);
    let theta0 = 0.2_f64;
    let rot0 = Quat::from_axis_angle(Vec3::Z, theta0);
    let arm = world.add_rigid(
        "hinge_arm",
        mat,
        Shape::cuboid(half * 2.0),
        Pose::new(rot0.rotate(-pivot_local), rot0),
    );
    // Give the hinge an out-of-plane spin impulse to verify the hinge locks off-axis tilt.
    world.set_velocity(arm, Vec3::ZERO, vec3(0.4, 0.2, -1.5));
    let limit_lo = -0.25;
    let limit_hi = 0.25;
    let hinge = world.add_hinge(
        arm,
        None,
        pivot_local,
        Vec3::ZERO,
        Vec3::Z,
        Quat::IDENTITY,
        2e6,
        5e4,
        Some((limit_lo, limit_hi)),
        JointDrive::pd(0.0, 0.0, 2.5),
    );

    let pool = Pool::new(2);
    let mut solver = Mechanics::new(Settings::default());
    let mut min_angle = f64::INFINITY;
    let mut max_angle = f64::NEG_INFINITY;
    for _ in 0..120 {
        let report = solver.step(&pool, &mut world, 1.0 / 240.0);
        assert!(!report.failed(), "hinge step failed: {report:?}");
        let angle = world.joint_coordinate(hinge).unwrap();
        min_angle = min_angle.min(angle);
        max_angle = max_angle.max(angle);
    }
    let core_world::Representation::Rigid(r) = &world.object(arm).representation else {
        unreachable!()
    };
    let pivot_err = r.pose.transform_point(pivot_local).length();
    let axis_err = (r.pose.rotation.rotate(Vec3::Z) - Vec3::Z).length();
    assert!(pivot_err < 1e-3, "hinge pivot drifted by {pivot_err}");
    assert!(axis_err < 2e-3, "hinge axis tilted by {axis_err}");
    assert!(
        min_angle > limit_lo - 0.02 && max_angle < limit_hi + 0.02,
        "hinge angle [{min_angle}, {max_angle}] violated limits [{limit_lo}, {limit_hi}]"
    );
}

#[test]
fn slider_joint_guides_translation_and_tracks_pd_drive_with_limits() {
    let mut world = World::new();
    world.gravity = Vec3::new(0.0, -9.81, 0.0);
    let mat = world.add_material(library::aluminium());
    let carriage = world.add_rigid(
        "carriage",
        mat,
        Shape::cuboid(vec3(0.16, 0.1, 0.1)),
        Pose::at(Vec3::ZERO),
    );
    let slider = world.add_slider(
        carriage,
        None,
        Vec3::ZERO,
        Vec3::ZERO,
        Vec3::X,
        Quat::IDENTITY,
        2e6,
        2e4,
        Some((-0.2, 0.35)),
        JointDrive::pd(0.22, 4000.0, 220.0).with_max_force(800.0),
    );

    let pool = Pool::new(2);
    let mut solver = Mechanics::new(Settings::default());
    for _ in 0..90 {
        let report = solver.step(&pool, &mut world, 1.0 / 180.0);
        assert!(!report.failed(), "slider step failed: {report:?}");
    }
    let q1 = world.joint_coordinate(slider).unwrap();
    let v1 = world.joint_velocity(slider).unwrap();
    let core_world::Representation::Rigid(r1) = &world.object(carriage).representation else {
        unreachable!()
    };
    assert!(
        (q1 - 0.22).abs() < 5e-3,
        "slider failed to track 0.22 m target, got {q1}"
    );
    assert!(v1.abs() < 2e-2, "slider did not settle at target, v={v1}");
    assert!(
        r1.pose.position.y.abs() < 2e-3 && r1.pose.position.z.abs() < 1e-4,
        "slider drifted off rail: {:?}",
        r1.pose.position
    );

    // Command a target beyond the upper limit (0.60 > 0.35) in-place; the limit must stop it.
    world.set_joint_target(slider, 0.60);
    for _ in 0..90 {
        let report = solver.step(&pool, &mut world, 1.0 / 180.0);
        assert!(!report.failed());
    }
    let q2 = world.joint_coordinate(slider).unwrap();
    assert!(
        (q2 - 0.35).abs() < 1.5e-2,
        "slider expected to clamp near upper limit 0.35 m, got {q2}"
    );
}

#[test]
fn articulated_robot_arm_and_gripper_reaches_pose_sleeps_wakes_and_grasps_object() {
    let build_scene = || {
        let mut world = World::new();
        let metal = world.add_material(Material::new("robot_alloy", 1200.0, 1e7, 0.3, 0.8, 0.0));
        let grip_pad = world.add_material(Material::new("gripper_pad", 900.0, 5e6, 0.35, 1.4, 0.0));

        // Fixed table & robot pedestal base.
        world.add_fixed("floor", metal, Shape::ground(0.0), Pose::IDENTITY);
        let base = world.add_fixed(
            "pedestal",
            metal,
            Shape::cuboid(vec3(0.24, 0.20, 0.24)),
            Pose::at(vec3(0.0, 0.10, 0.0)),
        );

        // Link 1 (upper arm): hinged to pedestal at (0, 0.20, 0), extends +X to (0.30, 0.20, 0).
        // Overlapping geometry with pedestal is automatically excluded by the Hinge joint.
        let link1 = world.add_rigid(
            "upper_arm",
            metal,
            Shape::cuboid(vec3(0.32, 0.06, 0.06)),
            Pose::at(vec3(0.15, 0.20, 0.0)),
        );
        let shoulder = world.add_hinge(
            link1,
            Some(base),
            vec3(-0.15, 0.0, 0.0),
            vec3(0.0, 0.10, 0.0),
            Vec3::Z,
            Quat::IDENTITY,
            2e6,
            4e4,
            Some((-1.2, 1.2)),
            JointDrive::pd(0.0, 3500.0, 180.0).with_max_force(1500.0),
        );

        // Link 2 (forearm / wrist mount): hinged to link1 at (0.30, 0.20, 0), extends +X to (0.54, 0.20, 0).
        let link2 = world.add_rigid(
            "forearm",
            metal,
            Shape::cuboid(vec3(0.26, 0.05, 0.08)),
            Pose::at(vec3(0.42, 0.20, 0.0)),
        );
        let elbow = world.add_hinge(
            link2,
            Some(link1),
            vec3(-0.12, 0.0, 0.0),
            vec3(0.15, 0.0, 0.0),
            Vec3::Z,
            Quat::IDENTITY,
            2e6,
            4e4,
            Some((-1.4, 1.4)),
            JointDrive::pd(0.0, 2800.0, 140.0).with_max_force(1200.0),
        );

        // Parallel-jaw gripper fingers mounted via Slider joints along local Z of link2 at x = 0.54, y = 0.11.
        let finger_shape = Shape::cuboid(vec3(0.06, 0.14, 0.024));
        let left_finger = world.add_rigid(
            "left_finger",
            grip_pad,
            finger_shape,
            Pose::at(vec3(0.54, 0.11, 0.065)),
        );
        let right_finger = world.add_rigid(
            "right_finger",
            grip_pad,
            finger_shape,
            Pose::at(vec3(0.54, 0.11, -0.065)),
        );
        let left_slider = world.add_slider(
            left_finger,
            Some(link2),
            vec3(0.0, 0.05, 0.0),
            vec3(0.12, -0.04, 0.0),
            Vec3::Z,
            Quat::IDENTITY,
            2e6,
            3e4,
            Some((0.02, 0.08)),
            JointDrive::pd(0.065, 4500.0, 180.0).with_max_force(600.0),
        );
        let right_slider = world.add_slider(
            right_finger,
            Some(link2),
            vec3(0.0, 0.05, 0.0),
            vec3(0.12, -0.04, 0.0),
            Vec3::Z,
            Quat::IDENTITY,
            2e6,
            3e4,
            Some((-0.08, -0.02)),
            JointDrive::pd(-0.065, 4500.0, 180.0).with_max_force(600.0),
        );

        // Target payload block resting on the floor between the open gripper jaws.
        let payload = world.add_rigid(
            "payload",
            grip_pad,
            Shape::cuboid(vec3(0.06, 0.08, 0.06)),
            Pose::at(vec3(0.54, 0.04, 0.0)),
        );

        (world, shoulder, elbow, left_slider, right_slider, payload)
    };

    let run = |threads| {
        let pool = Pool::new(threads);
        let (mut world, shoulder, elbow, left_slider, right_slider, payload) = build_scene();
        let mut solver = Mechanics::new(Settings {
            settle_time: Some(0.06),
            ..Default::default()
        });
        let h = 1.0 / 240.0;

        // Phase 1: hold home pose until both the robot arm island and payload island settle to sleep.
        let mut slept = false;
        for _ in 0..160 {
            let report = solver.step(&pool, &mut world, h);
            assert!(!report.failed(), "phase 1 step failed: {report:?}");
            if report.sleeping_islands == report.islands && report.solves == 0 {
                slept = true;
                break;
            }
        }
        assert!(
            slept,
            "robot arm and payload should settle and sleep at home pose"
        );

        // Phase 2: RL action update — close gripper jaws in-place without invalidating solver layout.
        // The sleeping robot arm island must wake immediately on the next step.
        world.set_joint_target(left_slider, 0.028);
        world.set_joint_target(right_slider, -0.028);
        let wake_report = solver.step(&pool, &mut world, h);
        assert!(!wake_report.failed());
        assert!(
            wake_report.solves > 0,
            "changing joint target must wake the sleeping robot arm island"
        );
        for _ in 0..80 {
            let report = solver.step(&pool, &mut world, h);
            assert!(!report.failed(), "phase 2 grasp step failed: {report:?}");
        }

        // Phase 3: command shoulder and elbow hinges upward to lift the grasped payload off the floor.
        world.set_joint_target(shoulder, 0.28);
        world.set_joint_target(elbow, 0.15);
        for _ in 0..120 {
            let report = solver.step(&pool, &mut world, h);
            assert!(!report.failed(), "phase 3 lift step failed: {report:?}");
        }

        let core_world::Representation::Rigid(p) = &world.object(payload).representation else {
            unreachable!()
        };
        let shoulder_q = world.joint_coordinate(shoulder).unwrap();
        assert!(
            shoulder_q > 0.18,
            "shoulder hinge should rotate upward, got {shoulder_q}"
        );
        assert!(
            p.pose.position.y > 0.11,
            "grasped payload should be lifted well above its 0.04 m resting height, got y={}",
            p.pose.position.y
        );
        solver.checkpoint(&world).to_bytes()
    };

    assert_eq!(run(1), run(4));
}

#[test]
fn extended_joints_and_tendon_gradients_match_central_differences() {
    for &with_damping in &[false, true] {
        let mut world = World::new();
        let mat = world.add_material(library::oak());
        let fixed = world.add_fixed(
            "base",
            mat,
            Shape::cuboid(Vec3::splat(0.2)),
            Pose::new(vec3(-1.2, 0.0, 0.0), Quat::from_axis_angle(Vec3::Y, 0.2)),
        );
        let r0 = world.add_rigid(
            "r0",
            mat,
            Shape::cuboid(vec3(0.4, 0.3, 0.2)),
            Pose::new(vec3(0.0, 0.5, 0.1), Quat::from_axis_angle(Vec3::Z, 0.15)),
        );
        let r1 = world.add_rigid(
            "r1",
            mat,
            Shape::cuboid(vec3(0.3, 0.25, 0.2)),
            Pose::new(vec3(0.55, 0.4, -0.15), Quat::from_axis_angle(Vec3::X, -0.2)),
        );
        let s0 = world.add_solid(
            "solid",
            mat,
            TetMesh::cuboid(Vec3::splat(0.25), 0.125),
            Pose::at(vec3(0.9, -0.1, 0.0)),
        );
        let damp = if with_damping { 8.0 } else { 0.0 };

        // Cylindrical (4 DOF constrained, screw axis along Y/Z mix)
        world.add_cylindrical(
            r0,
            Some(r1),
            vec3(0.15, 0.02, 0.0),
            vec3(-0.1, -0.02, 0.0),
            vec3(0.0, 1.0, 0.2).normalize(),
            Quat::from_axis_angle(Vec3::Y, 0.1),
            700.0,
            240.0,
            Some((-0.1, 0.1)),
            Some((-0.2, 0.2)),
            JointDrive::pd(0.05, 90.0, damp).with_max_force(25.0),
            JointDrive::pd(-0.15, 75.0, damp).with_max_force(20.0),
        );
        // Universal (2-DOF gimbal between fixed and r0)
        world.add_universal(
            r0,
            Some(fixed),
            vec3(-0.1, 0.0, 0.05),
            vec3(0.1, 0.1, 0.0),
            Vec3::Y,
            Vec3::X,
            800.0,
            300.0,
        );
        // Planar SE(2) (confined to XY-like tilted plane on r1 -> world)
        world.add_planar(
            r1,
            None,
            vec3(0.0, 0.05, 0.0),
            vec3(0.5, 0.4, -0.15),
            vec3(0.0, 0.98, 0.2).normalize(),
            Vec3::Y,
            650.0,
            210.0,
        );
        // 3-waypoint routed tendon (r0 -> r1 -> solid node 0) under taut stretch
        world.add_tendon(
            vec![
                LinkPoint::Body {
                    object: r0,
                    local: vec3(0.1, 0.1, 0.0),
                },
                LinkPoint::Body {
                    object: r1,
                    local: vec3(0.0, 0.1, 0.05),
                },
                LinkPoint::Node {
                    object: s0,
                    node: 0,
                },
            ],
            450.0,
            damp,
            true,
            JointDrive::pd(0.2, 180.0, damp).with_max_force(45.0),
        );

        let pool = Pool::new(2);
        let mut solver = Mechanics::new(Settings {
            contact_precision: Some(1e-3),
            ..Default::default()
        });
        let (error, _) = solver.gradient_check(&pool, &world, 0.01, |b| {
            let k = (b % 5) as f64;
            vec3(0.0018 * (k + 1.0), -0.0012 * k, 0.0011 * (2.0 - k))
        });
        assert!(
            error < 1e-7,
            "extended joints/tendon FD gradient check (damping={with_damping}) failed: {error}"
        );
    }
}

#[test]
fn augmented_lagrangian_tightens_bilateral_joint_drift_below_micron() {
    let build = || {
        let mut world = World::new();
        let mat = world.add_material(library::oak());
        let link = world.add_rigid(
            "pendulum",
            mat,
            Shape::cuboid(vec3(0.4, 0.08, 0.08)),
            Pose::at(vec3(0.2, 0.0, 0.0)),
        );
        world.add_ball_joint(
            LinkPoint::Body {
                object: link,
                local: vec3(-0.2, 0.0, 0.0),
            },
            LinkPoint::World(Vec3::ZERO),
            1.5e5,
        );
        (world, link)
    };

    let measure_drift = |al_passes: usize| {
        let pool = Pool::new(2);
        let (mut world, link) = build();
        let mut solver = Mechanics::new(Settings {
            integrator: Integrator::BackwardEuler,
            augmented_lagrangian_passes: al_passes,
            tolerance: 1e-9,
            ..Default::default()
        });
        for _ in 0..10 {
            let r = solver.step(&pool, &mut world, 1.0 / 120.0);
            assert!(!r.failed());
        }
        let core_world::Representation::Rigid(r) = &world.object(link).representation else {
            unreachable!()
        };
        r.pose.transform_point(vec3(-0.2, 0.0, 0.0)).length()
    };

    let drift_penalty = measure_drift(0);
    let drift_al = measure_drift(8);
    assert!(
        drift_penalty > 3e-5,
        "pure penalty should show noticeable sag ({drift_penalty})"
    );
    assert!(
        drift_al < 1e-6,
        "AL passes should drive bilateral drift below 1 um, got {drift_al}"
    );
}

#[test]
fn virtual_probes_report_joint_state_wrench_tactile_and_imu() {
    let mut world = World::new();
    let mat = world.add_material(library::aluminium());
    world.add_fixed(
        "floor",
        mat,
        Shape::HalfSpace {
            normal: Vec3::Y,
            offset: 0.0,
        },
        Pose::IDENTITY,
    );

    // 1. Hinged arm held horizontally under gravity -> wrench balances arm weight.
    let arm = world.add_rigid(
        "arm",
        mat,
        Shape::cuboid(vec3(0.4, 0.08, 0.08)),
        Pose::at(vec3(0.2, 1.0, 0.0)),
    );
    let hinge = world.add_hinge(
        arm,
        None,
        vec3(-0.2, 0.0, 0.0),
        vec3(0.0, 1.0, 0.0),
        Vec3::Z,
        Quat::IDENTITY,
        5e5,
        2e4,
        None,
        JointDrive::pd(0.2, 200.0, 20.0),
    );

    // 2. Heavy block resting flat on floor (tactile + IMU probes).
    let block = world.add_rigid(
        "block",
        mat,
        Shape::cuboid(vec3(0.2, 0.2, 0.2)),
        Pose::at(vec3(1.5, 0.1, 0.0)),
    );

    let p_js = world.add_probe(Probe::JointState { link: hinge });
    let p_wr = world.add_probe(Probe::Wrench { link: hinge });
    let p_tac = world.add_probe(Probe::Tactile {
        object: block,
        local_origin: vec3(0.0, -0.1, 0.0),
        local_normal: -Vec3::Y,
        radius: 0.25,
    });
    let p_imu = world.add_probe(Probe::Imu {
        object: block,
        local: Vec3::ZERO,
    });

    let pool = Pool::new(2);
    let mut solver = Mechanics::new(Settings {
        augmented_lagrangian_passes: 4,
        ..Default::default()
    });
    for _ in 0..40 {
        let r = solver.step(&pool, &mut world, 1.0 / 240.0);
        assert!(!r.failed());
    }

    let readings = solver.sample_probes(&pool, &world);
    let ProbeReading::JointState { coordinate, .. } = readings[p_js.0 as usize] else {
        panic!("expected JointState")
    };
    assert!(coordinate.abs() < 1.0 && coordinate > 0.0);

    let ProbeReading::Wrench { force, .. } = readings[p_wr.0 as usize] else {
        panic!("expected Wrench")
    };
    let arm_weight = world.object(arm).mass() * 9.81;
    assert!(
        force.length() > 0.3 * arm_weight,
        "hinge wrench magnitude {} too small vs arm weight {arm_weight}",
        force.length()
    );

    let ProbeReading::Tactile {
        normal_force,
        center_of_pressure,
        active_contacts,
        ..
    } = readings[p_tac.0 as usize]
    else {
        panic!("expected Tactile")
    };
    let block_weight = world.object(block).mass() * 9.81;
    assert!(active_contacts > 0);
    assert!(center_of_pressure.is_some());
    assert!(
        (normal_force - block_weight).abs() / block_weight < 0.15,
        "tactile normal force {normal_force} vs block weight {block_weight}"
    );

    let ProbeReading::Imu {
        linear_acceleration,
        angular_velocity,
    } = readings[p_imu.0 as usize]
    else {
        panic!("expected Imu")
    };
    assert!(
        (linear_acceleration.y - 9.81).abs() < 0.5,
        "resting IMU proper accel should oppose gravity (~+9.81), got {linear_acceleration:?}"
    );
    assert!(angular_velocity.length() < 0.1);
}
