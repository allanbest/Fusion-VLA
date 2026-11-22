use core_math::{vec3, Pose, Vec3};
use core_runtime::Pool;
use core_space::Shape;
use core_world::{library, JointDrive, LinkPoint, Probe, ProbeReading, Representation, World};
use phys_mechanics::{AdjointLoss, DomainRandomizer, Integrator, Mechanics, Settings, WorldBatch};

const ARM_URDF: &str = r#"<?xml version="1.0"?>
<robot name="two_link_arm">
  <link name="base">
    <inertial>
      <mass value="5.0"/>
      <inertia ixx="0.05" iyy="0.05" izz="0.05"/>
    </inertial>
    <collision>
      <geometry><box size="0.12 0.12 0.12"/></geometry>
    </collision>
  </link>
  <link name="upper">
    <inertial>
      <origin xyz="0.15 0 0"/>
      <mass value="1.5"/>
      <inertia ixx="0.01" iyy="0.02" izz="0.02"/>
    </inertial>
    <collision>
      <origin xyz="0.15 0 0"/>
      <geometry><box size="0.30 0.06 0.06"/></geometry>
    </collision>
  </link>
  <link name="forearm">
    <inertial>
      <origin xyz="0.12 0 0"/>
      <mass value="1.0"/>
      <inertia ixx="0.008" iyy="0.015" izz="0.015"/>
    </inertial>
    <collision>
      <origin xyz="0.12 0 0"/>
      <geometry><cylinder radius="0.03" length="0.24"/></geometry>
    </collision>
  </link>
  <joint name="shoulder" type="revolute">
    <parent link="base"/>
    <child link="upper"/>
    <origin xyz="0 0.06 0"/>
    <axis xyz="0 0 1"/>
    <limit lower="-1.5" upper="1.5"/>
  </joint>
  <joint name="elbow" type="revolute">
    <parent link="upper"/>
    <child link="forearm"/>
    <origin xyz="0.30 0 0"/>
    <axis xyz="0 0 1"/>
    <limit lower="-1.5" upper="1.5"/>
  </joint>
</robot>"#;

#[test]
fn urdf_importer_loads_and_simulates_articulated_arm_with_tendon_and_probes() {
    let mut world = World::new();
    let mat = world.add_material(library::aluminium());
    let model = world
        .load_urdf(ARM_URDF, mat, Pose::at(vec3(0.0, 1.0, 0.0)), true)
        .expect("URDF should parse");
    let shoulder = model.joint("shoulder").unwrap();
    let elbow = model.joint("elbow").unwrap();
    let base = model.link("base").unwrap();
    let forearm = model.link("forearm").unwrap();

    world.set_joint_drive(shoulder, JointDrive::pd(0.45, 250.0, 20.0));
    world.set_joint_drive(elbow, JointDrive::pd(-0.25, 180.0, 15.0));
    let tendon = world.add_tendon(
        vec![
            LinkPoint::Body {
                object: base,
                local: vec3(0.0, 0.12, 0.0),
            },
            LinkPoint::Body {
                object: forearm,
                local: vec3(0.15, 0.03, 0.0),
            },
        ],
        80.0,
        2.0,
        true,
        JointDrive::pd(0.28, 120.0, 5.0),
    );
    let p_js = world.add_probe(Probe::JointState { link: shoulder });
    let p_wr = world.add_probe(Probe::Wrench { link: tendon });

    let pool = Pool::new(2);
    let mut solver = Mechanics::new(Settings {
        augmented_lagrangian_passes: 3,
        ..Default::default()
    });
    for _ in 0..80 {
        let r = solver.step(&pool, &mut world, 1.0 / 240.0);
        assert!(!r.failed());
    }

    let readings = solver.sample_probes(&pool, &world);
    let ProbeReading::JointState { coordinate, .. } = readings[p_js.0 as usize] else {
        panic!("expected JointState");
    };
    assert!(
        (coordinate - 0.45).abs() < 0.2,
        "shoulder did not track target 0.45 rad (got {coordinate})"
    );
    let ProbeReading::Wrench { force, .. } = readings[p_wr.0 as usize] else {
        panic!("expected Wrench");
    };
    assert!(force.length() >= 0.0);
}

#[test]
fn world_batch_steps_domain_randomized_envs_with_thread_determinism() {
    let mut template = World::new();
    let mat = template.add_material(library::aluminium());
    let arm = template.add_rigid(
        "arm",
        mat,
        Shape::cuboid(vec3(0.4, 0.08, 0.08)),
        Pose::at(vec3(0.2, 1.0, 0.0)),
    );
    let hinge = template.add_hinge(
        arm,
        None,
        vec3(-0.2, 0.0, 0.0),
        vec3(0.0, 1.0, 0.0),
        Vec3::Z,
        core_math::Quat::IDENTITY,
        2e5,
        1e4,
        None,
        JointDrive::pd(0.4, 80.0, 6.0),
    );
    template.add_probe(Probe::JointState { link: hinge });

    let rand_cfg = DomainRandomizer {
        seed: 777,
        mass_scale: (0.7, 1.3),
        friction_scale: (0.5, 1.5),
        stiffness_scale: (0.8, 1.25),
        damping_scale: (0.8, 1.2),
        gravity_jitter: vec3(0.0, 0.4, 0.0),
    };

    let run = |threads: usize| {
        let pool = Pool::new(threads);
        let mut batch = WorldBatch::from_template(template.clone(), Settings::default(), 8);
        batch.randomize_all(&rand_cfg);
        for env in 0..batch.len() {
            batch.set_joint_targets(env, &[(hinge, 0.3 + 0.05 * env as f64)]);
        }
        for _ in 0..15 {
            batch.step_all(&pool, 1.0 / 120.0);
            assert!(batch.reports().iter().all(|r| !r.failed()));
        }
        let probes = batch.sample_all_probes(&pool);
        // Different randomized environments should diverge from each other.
        assert_ne!(probes[0][0], probes[1][0]);
        probes
    };

    assert_eq!(run(1), run(4));
}

#[test]
fn ift_adjoint_backward_step_matches_central_finite_differences() {
    let build_scene = |target: f64, stiffness: f64, grav: Vec3, v0: Vec3, p0_shift: Vec3| {
        let mut w = World::new();
        w.gravity = grav;
        let mat = w.add_material(library::aluminium());
        let body = w.add_rigid(
            "link",
            mat,
            Shape::cuboid(vec3(0.4, 0.1, 0.1)),
            Pose::at(vec3(0.2, 0.0, 0.0) + p0_shift),
        );
        w.set_velocity(body, v0, Vec3::ZERO);
        let joint = w.add_hinge(
            body,
            None,
            vec3(-0.2, 0.0, 0.0),
            Vec3::ZERO,
            Vec3::Z,
            core_math::Quat::IDENTITY,
            5e4,
            1e4,
            None,
            JointDrive::pd(target, stiffness, 0.0),
        );
        (w, body, joint)
    };

    let target0 = 0.35;
    let stiff0 = 60.0;
    let grav0 = vec3(0.0, -9.81, 0.0);
    let v0 = vec3(0.1, -0.2, 0.05);
    let h = 0.02;
    let goal_pos = vec3(0.25, 0.12, -0.02);
    let goal_vel = vec3(-0.05, 0.3, 0.0);

    let settings = Settings {
        integrator: Integrator::BackwardEuler,
        islands: false,
        sleeping: false,
        impact_law: false,
        contact_precision: Some(1e-3),
        tolerance: 1e-9,
        max_iterations: 200,
        ..Default::default()
    };
    let pool = Pool::new(2);

    let eval_loss = |w: &mut World, body: core_world::ObjectId| -> f64 {
        let mut s = Mechanics::new(settings);
        let rep = s.step(&pool, w, h);
        assert!(!rep.failed());
        let Representation::Rigid(r) = &w.object(body).representation else {
            unreachable!()
        };
        let dp = r.pose.position - goal_pos;
        let dv = r.velocity - goal_vel;
        0.5 * dp.length_squared() + 0.25 * dv.length_squared()
    };

    // Analytical IFT gradients.
    let (w0, body0, joint0) = build_scene(target0, stiff0, grav0, v0, Vec3::ZERO);
    let mut adj_solver = Mechanics::new(settings);
    let mut preview = w0.clone();
    assert!(!adj_solver.step(&pool, &mut preview, h).failed());
    let Representation::Rigid(r_end) = &preview.object(body0).representation else {
        unreachable!()
    };
    let grads = adj_solver.backward_step(
        &pool,
        &w0,
        h,
        &AdjointLoss {
            object_position_grads: vec![(body0, r_end.pose.position - goal_pos)],
            object_velocity_grads: vec![(body0, (r_end.velocity - goal_vel) * 0.5)],
            ..Default::default()
        },
    );
    assert!(grads.converged);

    let fd_err = |analytical: f64, plus: f64, minus: f64, eps: f64| -> f64 {
        let fd = (plus - minus) / (2.0 * eps);
        let scale = analytical.abs().max(fd.abs()).max(1e-4);
        (analytical - fd).abs() / scale
    };

    // 1. Joint target sensitivity
    let eps = 1e-5;
    let lu_p = eval_loss(
        &mut build_scene(target0 + eps, stiff0, grav0, v0, Vec3::ZERO).0,
        body0,
    );
    let lu_m = eval_loss(
        &mut build_scene(target0 - eps, stiff0, grav0, v0, Vec3::ZERO).0,
        body0,
    );
    let an_u = grads.d_joint_targets[joint0.0 as usize];
    let err_u = fd_err(an_u, lu_p, lu_m, eps);

    // 2. Joint stiffness sensitivity
    let eps_k = 1e-4;
    let lk_p = eval_loss(
        &mut build_scene(target0, stiff0 + eps_k, grav0, v0, Vec3::ZERO).0,
        body0,
    );
    let lk_m = eval_loss(
        &mut build_scene(target0, stiff0 - eps_k, grav0, v0, Vec3::ZERO).0,
        body0,
    );
    let an_k = grads.d_joint_stiffnesses[joint0.0 as usize];
    let err_k = fd_err(an_k, lk_p, lk_m, eps_k);

    // 3. Gravity sensitivity (y component)
    let lg_p = eval_loss(
        &mut build_scene(target0, stiff0, grav0 + vec3(0.0, eps, 0.0), v0, Vec3::ZERO).0,
        body0,
    );
    let lg_m = eval_loss(
        &mut build_scene(target0, stiff0, grav0 - vec3(0.0, eps, 0.0), v0, Vec3::ZERO).0,
        body0,
    );
    let an_gy = grads.d_gravity.y;
    let err_gy = fd_err(an_gy, lg_p, lg_m, eps);

    // 4. Initial linear velocity sensitivity (x component, block 0)
    let lv_p = eval_loss(
        &mut build_scene(target0, stiff0, grav0, v0 + vec3(eps, 0.0, 0.0), Vec3::ZERO).0,
        body0,
    );
    let lv_m = eval_loss(
        &mut build_scene(target0, stiff0, grav0, v0 - vec3(eps, 0.0, 0.0), Vec3::ZERO).0,
        body0,
    );
    let an_vx = grads.d_initial_velocities[0];
    let err_vx = fd_err(an_vx, lv_p, lv_m, eps);

    // 5. Initial COM position sensitivity (y component, block 0 -> index 1)
    let lp_p = eval_loss(
        &mut build_scene(target0, stiff0, grav0, v0, vec3(0.0, eps, 0.0)).0,
        body0,
    );
    let lp_m = eval_loss(
        &mut build_scene(target0, stiff0, grav0, v0, vec3(0.0, -eps, 0.0)).0,
        body0,
    );
    let an_py = grads.d_initial_positions[1];
    let err_py = fd_err(an_py, lp_p, lp_m, eps);

    assert!(
        err_u < 1e-6 && err_k < 1e-6 && err_gy < 1e-6 && err_vx < 1e-6 && err_py < 1e-6,
        "IFT errors: u={err_u:.3e} (an={an_u:.6e}, fd={:.6e}), \
k={err_k:.3e} (an={an_k:.6e}, fd={:.6e}), \
gy={err_gy:.3e} (an={an_gy:.6e}, fd={:.6e}), \
vx={err_vx:.3e} (an={an_vx:.6e}, fd={:.6e}), \
py={err_py:.3e} (an={an_py:.6e}, fd={:.6e})",
        (lu_p - lu_m) / (2.0 * eps),
        (lk_p - lk_m) / (2.0 * eps_k),
        (lg_p - lg_m) / (2.0 * eps),
        (lv_p - lv_m) / (2.0 * eps),
        (lp_p - lp_m) / (2.0 * eps),
    );
}

const MJCF_SCENE: &str = r#"<mujoco model="zup_arm_and_box">
  <compiler angle="degree"/>
  <option gravity="0 0 -9.81"/>
  <default>
    <joint damping="1.0"/>
    <position kp="220" kv="18" forcerange="-80 80"/>
  </default>
  <worldbody>
    <geom name="ground" type="plane" size="4 4 0.1" pos="0 0 0" friction="0.8 0.005 0.0001"/>
    <site name="anchor" pos="0 0.08 0.8"/>
    <body name="pedestal" pos="0 0 0.15">
      <geom name="pedestal_box" type="box" size="0.06 0.06 0.15"/>
      <body name="upper_arm" pos="0 0 0.15">
        <inertial pos="0.15 0 0" mass="1.2" diaginertia="0.01 0.015 0.015"/>
        <joint name="shoulder" type="hinge" pos="0 0 0" axis="0 1 0" range="-90 90"/>
        <geom type="capsule" fromto="0 0 0 0.30 0 0" size="0.03"/>
        <site name="arm_imu_site" pos="0.15 0 0"/>
        <body name="forearm" pos="0.30 0 0">
          <inertial pos="0.12 0 0" mass="0.8" diaginertia="0.006 0.01 0.01"/>
          <joint name="elbow" type="hinge" pos="0 0 0" axis="0 1 0" range="-120 120"/>
          <geom type="capsule" fromto="0 0 0 0.24 0 0" size="0.025"/>
          <site name="arm_tip" pos="0.24 0 0" size="0.02" zaxis="0 0 -1"/>
        </body>
      </body>
    </body>
    <body name="block" pos="0.45 0 0.06">
      <freejoint/>
      <geom name="block_geom" type="box" size="0.05 0.05 0.05"/>
      <site name="block_bottom" pos="0 0 -0.05" size="0.08" zaxis="0 0 -1"/>
    </body>
  </worldbody>
  <contact>
    <exclude body1="pedestal" body2="upper_arm"/>
  </contact>
  <tendon>
    <spatial name="brace" stiffness="40" damping="1.0" springlength="0.72">
      <site site="anchor"/>
      <site site="arm_tip"/>
    </spatial>
  </tendon>
  <actuator>
    <position name="m_shoulder" joint="shoulder"/>
    <position name="m_elbow" joint="elbow"/>
  </actuator>
  <sensor>
    <jointpos name="s_shoulder" joint="shoulder"/>
    <touch name="s_block_touch" site="block_bottom"/>
    <accelerometer name="s_arm_imu" site="arm_imu_site"/>
    <actuatorfrc name="s_shoulder_wrench" actuator="m_shoulder"/>
  </sensor>
</mujoco>"#;

#[test]
fn mjcf_robot_arm_and_tendon_simulate_with_probes_and_determinism() {
    let run = |threads: usize| {
        let mut world = World::new();
        let mat = world.add_material(library::aluminium());
        let model = world
            .load_mjcf(MJCF_SCENE, mat, Pose::IDENTITY)
            .expect("MJCF scene should parse");

        let shoulder = model.actuator("m_shoulder").unwrap();
        let elbow = model.actuator("m_elbow").unwrap();
        world.set_joint_target(shoulder, -0.35);
        world.set_joint_target(elbow, 0.25);

        let pool = Pool::new(threads);
        let mut solver = Mechanics::new(Settings {
            augmented_lagrangian_passes: 2,
            ..Default::default()
        });
        for step in 0..80 {
            let r = solver.step(&pool, &mut world, 1.0 / 240.0);
            assert!(!r.failed(), "MJCF step {step} failed: {r:?}");
        }

        let readings = solver.sample_probes(&pool, &world);
        let p_shoulder = model.probe("s_shoulder").unwrap();
        let p_touch = model.probe("s_block_touch").unwrap();
        let p_imu = model.probe("s_arm_imu").unwrap();
        let p_frc = model.probe("s_shoulder_wrench").unwrap();

        let ProbeReading::JointState { coordinate, .. } = readings[p_shoulder.0 as usize] else {
            panic!("expected JointState");
        };
        assert!(
            (coordinate - (-0.35)).abs() < 0.25,
            "shoulder should move toward -0.35 rad, got {coordinate}"
        );
        let ProbeReading::Tactile { normal_force, .. } = readings[p_touch.0 as usize] else {
            panic!("expected Tactile");
        };
        assert!(
            normal_force > 0.0,
            "block resting on plane should read positive tactile force, got {normal_force}"
        );
        let ProbeReading::Imu {
            linear_acceleration,
            ..
        } = readings[p_imu.0 as usize]
        else {
            panic!("expected Imu");
        };
        assert!(linear_acceleration.length().is_finite());
        let ProbeReading::Wrench { force, torque } = readings[p_frc.0 as usize] else {
            panic!("expected Wrench");
        };
        assert!(force.length().is_finite() && torque.length().is_finite());

        let block_id = model.link("block").unwrap();
        let Representation::Rigid(blk) = &world.object(block_id).representation else {
            unreachable!()
        };
        assert!(blk.pose.position.z > 0.045 && blk.pose.position.z < 0.07);
        readings
    };

    assert_eq!(run(1), run(4));
}

#[test]
fn world_batch_vectorized_rl_helpers_reset_and_randomize_subsets() {
    let mut template = World::new();
    let mat = template.add_material(library::aluminium());
    let arm = template.add_rigid(
        "arm",
        mat,
        Shape::cuboid(vec3(0.4, 0.08, 0.08)),
        Pose::at(vec3(0.2, 1.0, 0.0)),
    );
    let hinge = template.add_hinge(
        arm,
        None,
        vec3(-0.2, 0.0, 0.0),
        vec3(0.0, 1.0, 0.0),
        Vec3::Z,
        core_math::Quat::IDENTITY,
        2e5,
        1e4,
        None,
        JointDrive::pd(0.0, 400.0, 12.0),
    );

    let pool = Pool::new(2);
    let mut batch = WorldBatch::from_template(template, Settings::default(), 4);
    batch.set_batch_joint_targets(&[hinge], &[0.1, 0.2, -0.1, -0.2]);
    for _ in 0..10 {
        batch.step_all(&pool, 1.0 / 120.0);
        assert!(batch.reports().iter().all(|r| !r.failed()));
    }
    let coords = batch.joint_coordinates(&[hinge]);
    let vels = batch.joint_velocities(&[hinge]);
    assert_eq!(coords.len(), 4);
    assert_eq!(vels.len(), 4);
    assert!(coords[0] > 0.0 && coords[2] < 0.0);

    // Reset subset [0, 2] back to zero pose/time without disturbing [1, 3].
    batch.reset_envs(&[0, 2]);
    assert_eq!(batch.world(0).time, 0.0);
    assert_eq!(batch.world(2).time, 0.0);
    assert!(batch.world(1).time > 0.0);

    let cfg = DomainRandomizer {
        seed: 42,
        mass_scale: (0.8, 1.2),
        ..Default::default()
    };
    batch.randomize_envs(&[0, 2], &cfg, 1);
    let m0 = batch.world(0).object(arm).mass();
    batch.randomize_envs(&[0, 2], &cfg, 2);
    let m0_next = batch.world(0).object(arm).mass();
    assert_ne!(m0, m0_next);

    batch.reset_all();
    for i in 0..batch.len() {
        assert_eq!(batch.world(i).time, 0.0);
    }
}

#[test]
fn explicit_exclude_contact_prevents_collision() {
    let mut world = World::new();
    world.gravity = Vec3::ZERO;
    let mat = world.add_material(library::aluminium());
    let a = world.add_rigid(
        "a",
        mat,
        Shape::Sphere { radius: 0.1 },
        Pose::at(vec3(-0.2, 0.0, 0.0)),
    );
    let b = world.add_rigid(
        "b",
        mat,
        Shape::Sphere { radius: 0.1 },
        Pose::at(vec3(0.2, 0.0, 0.0)),
    );
    world.set_velocity(a, vec3(2.0, 0.0, 0.0), Vec3::ZERO);
    world.set_velocity(b, vec3(-2.0, 0.0, 0.0), Vec3::ZERO);
    world.exclude_contact(a, b);

    let pool = Pool::new(2);
    let mut solver = Mechanics::new(Settings::default());
    for _ in 0..25 {
        let r = solver.step(&pool, &mut world, 0.01);
        assert!(!r.failed());
    }
    let Representation::Rigid(ra) = &world.object(a).representation else {
        unreachable!()
    };
    let Representation::Rigid(rb) = &world.object(b).representation else {
        unreachable!()
    };
    assert!(ra.pose.position.x > 0.25 && rb.pose.position.x < -0.25);
}
