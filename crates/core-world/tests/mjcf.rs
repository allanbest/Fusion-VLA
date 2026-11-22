use std::f64::consts::{FRAC_PI_2, PI};

use core_math::{vec3, Pose, Quat, Vec3};
use core_space::Shape;
use core_world::{library, Link, Probe, Representation, World};

#[test]
fn mjcf_parses_defaults_poses_geoms_joints_tendons_actuators_and_sensors() {
    let xml = r#"<!DOCTYPE mujoco>
<mujoco model="test_bot">
  <compiler angle="degree" eulerseq="xyz" autolimits="true"/>
  <option gravity="0 0 -9.81"/>
  <default>
    <joint damping="1.5" limited="true"/>
    <geom friction="0.7 0.005 0.0001"/>
    <default class="arm_joint">
      <joint type="hinge" range="-90 90"/>
      <position kp="220" kv="18" forcerange="-40 40"/>
    </default>
  </default>
  <worldbody>
    <geom name="floor" type="plane" size="5 5 0.1" pos="0 0 0"/>
    <site name="world_anchor" pos="0 0 1.2"/>
    <body name="base" pos="0 0 0.5" childclass="arm_joint">
      <geom name="base_box" type="box" size="0.1 0.1 0.05" pos="0 0 0"/>
      <geom name="base_cap" type="capsule" size="0.03 0.05" pos="0.08 0 0"/>
      <geom name="base_vis" type="sphere" size="0.3" contype="0" conaffinity="0"/>
      <body name="link1" pos="0 0 0.1" quat="0.70710678 0 0 0.70710678">
        <inertial pos="0 0 0.15" quat="1 0 0 0" mass="2.0" diaginertia="0.02 0.02 0.01"/>
        <joint name="j_shoulder" pos="0 0 0" axis="0 1 0"/>
        <geom name="link1_geom" type="capsule" fromto="0 0 0 0 0 0.3" size="0.04"/>
        <site name="link1_mid" pos="0 0 0.15" size="0.02"/>
        <body name="slider_part" pos="0 0 0.3" axisangle="0 0 1 90">
          <inertial pos="0 0 0" mass="0.8" fullinertia="0.005 0.005 0.002 0.0001 0 0"/>
          <joint name="j_slide" type="slide" axis="1 0 0" range="-0.15 0.15" damping="0.4"/>
          <geom name="slider_cyl" type="cylinder" size="0.025 0.08"/>
          <site name="tip_touch" pos="0 0 0.08" size="0.015" zaxis="0 0 1"/>
        </body>
      </body>
    </body>
    <body name="ball_link" pos="0.4 0 0.6" euler="0 90 0">
      <joint name="j_ball" type="ball"/>
      <geom type="ellipsoid" size="0.05 0.08 0.10"/>
    </body>
    <body name="free_prop" pos="-0.4 0 0.4" xyaxes="0 1 0 -1 0 0">
      <freejoint name="prop_free"/>
      <geom name="prop_box" type="box" size="0.05 0.05 0.05"/>
    </body>
  </worldbody>
  <contact>
    <exclude body1="base" body2="free_prop"/>
  </contact>
  <tendon>
    <spatial name="pull_cable" stiffness="150" damping="2.5" springlength="0.45">
      <site site="world_anchor"/>
      <site site="link1_mid"/>
    </spatial>
  </tendon>
  <actuator>
    <position name="act_shoulder" class="arm_joint" joint="j_shoulder"/>
    <velocity name="act_slide" joint="j_slide" kv="12"/>
    <motor name="act_cable" tendon="pull_cable" ctrlrange="-20 20" gear="2"/>
  </actuator>
  <sensor>
    <jointpos name="s_shoulder_pos" joint="j_shoulder"/>
    <touch name="s_tip_touch" site="tip_touch"/>
    <accelerometer name="s_mid_imu" site="link1_mid"/>
    <force name="s_tip_wrench" site="tip_touch"/>
    <actuatorfrc name="s_shoulder_frc" actuator="act_shoulder"/>
  </sensor>
</mujoco>"#;

    let mut world = World::new();
    let steel = world.add_material(library::structural_steel());
    let model = world
        .load_mjcf(xml, steel, Pose::IDENTITY)
        .expect("MJCF should parse cleanly");

    assert_eq!(model.name, "test_bot");
    assert!((world.gravity - vec3(0.0, 0.0, -9.81)).length() < 1e-12);

    // Compound fixed base geoms + excluded visual geom
    assert!(model.geom("floor").is_some());
    assert!(model.geom("base_box").is_some());
    assert!(model.geom("base_cap").is_some());
    assert!(model.geom("base_vis").is_none());

    // Floor is an infinite half-space with normal +Z
    let floor_id = model.geom("floor").unwrap();
    let Representation::Fixed(floor) = &world.object(floor_id).representation else {
        panic!("floor must be Fixed");
    };
    assert!(matches!(floor.shape, Shape::HalfSpace { .. }));

    // Rigid mass & rotated inertia from <inertial>
    let link1_id = model.link("link1").unwrap();
    let Representation::Rigid(r_link1) = &world.object(link1_id).representation else {
        panic!("link1 must be Rigid");
    };
    assert!((r_link1.mass - 2.0).abs() < 1e-12);
    // Capsule fromto along Z at 0.15 local Z => midpoint Z = 0.5 + 0.1 + 0.15 = 0.75
    assert!((r_link1.pose.position - vec3(0.0, 0.0, 0.75)).length() < 1e-9);

    // Contact exclusion between body1="base" and body2="free_prop" covers both base geoms
    let prop_id = model.link("free_prop").unwrap();
    let base_box_id = model.geom("base_box").unwrap();
    let base_cap_id = model.geom("base_cap").unwrap();
    let p1 = if base_box_id < prop_id {
        (base_box_id, prop_id)
    } else {
        (prop_id, base_box_id)
    };
    let p2 = if base_cap_id < prop_id {
        (base_cap_id, prop_id)
    } else {
        (prop_id, base_cap_id)
    };
    assert!(world.excluded_contacts().contains(&p1));
    assert!(world.excluded_contacts().contains(&p2));

    // Shoulder hinge converted limits from degrees to radians & received actuator kp/kv/max_force
    let shoulder = model.joint("j_shoulder").unwrap();
    let Link::Hinge { limits, drive, .. } = world.link(shoulder) else {
        panic!("expected Hinge");
    };
    let (lo, hi) = limits.expect("degree limits should be parsed");
    assert!((lo + FRAC_PI_2).abs() < 1e-12);
    assert!((hi - FRAC_PI_2).abs() < 1e-12);
    assert!((drive.stiffness - 220.0).abs() < 1e-12);
    assert!((drive.damping - (1.5 + 18.0)).abs() < 1e-12);
    assert_eq!(drive.max_force, Some(40.0));

    // Prismatic slider keeps limits in metres & combines passive damping + velocity kv
    let slide = model.joint("j_slide").unwrap();
    let Link::Slider {
        limits: s_limits,
        drive: s_drive,
        ..
    } = world.link(slide)
    else {
        panic!("expected Slider");
    };
    assert_eq!(*s_limits, Some((-0.15, 0.15)));
    assert!((s_drive.damping - (0.4 + 12.0)).abs() < 1e-12);

    // Tendon & motor cap
    let cable = model.tendon("pull_cable").unwrap();
    assert_eq!(model.actuator("act_cable").unwrap(), cable);
    let Link::Tendon {
        stiffness,
        damping,
        drive: c_drive,
        slack_only,
        ..
    } = world.link(cable)
    else {
        panic!("expected Tendon");
    };
    assert!(*slack_only);
    assert!((stiffness - 150.0).abs() < 1e-12);
    assert!((damping - 2.5).abs() < 1e-12);
    assert!((c_drive.target - 0.45).abs() < 1e-12);
    assert_eq!(c_drive.max_force, Some(40.0));

    // Ball joint and sensors
    assert!(matches!(
        world.link(model.joint("j_ball").unwrap()),
        Link::BallJoint { .. }
    ));
    assert_eq!(model.probes.len(), 5);
    let s_touch = world.probe(model.probe("s_tip_touch").unwrap());
    assert!(matches!(s_touch, Probe::Tactile { .. }));
    let s_imu = world.probe(model.probe("s_mid_imu").unwrap());
    assert!(matches!(s_imu, Probe::Imu { .. }));
    let s_wrench = world.probe(model.probe("s_tip_wrench").unwrap());
    assert!(matches!(s_wrench, Probe::Wrench { .. }));
    let s_frc = world.probe(model.probe("s_shoulder_frc").unwrap());
    assert!(matches!(s_frc, Probe::Wrench { .. }));
}

#[test]
fn mjcf_combines_multi_joint_bodies_and_radian_compiler_and_general_actuator() {
    let xml = r#"
<mujoco model="multi_dof">
  <compiler angle="radian" eulerseq="XYZ"/>
  <worldbody>
    <body name="frame" pos="0 0 0">
      <geom type="box" size="0.1 0.1 0.1"/>
      <body name="cyl_body" pos="0.3 0 0">
        <joint name="c_slide" type="slide" axis="0 1 0" range="-0.5 0.5"/>
        <joint name="c_hinge" type="hinge" axis="0 1 0" range="-3.14159 3.14159"/>
        <geom type="capsule" size="0.03 0.1"/>
      </body>
      <body name="univ_body" pos="-0.3 0 0">
        <joint name="u_x" type="hinge" axis="1 0 0"/>
        <joint name="u_y" type="hinge" axis="0 1 0"/>
        <geom type="sphere" size="0.05"/>
      </body>
      <body name="planar_body" pos="0 0.4 0">
        <joint name="p_x" type="slide" axis="1 0 0"/>
        <joint name="p_z" type="slide" axis="0 0 1"/>
        <joint name="p_ry" type="hinge" axis="0 1 0"/>
        <geom type="box" size="0.05 0.05 0.05"/>
      </body>
    </body>
  </worldbody>
  <actuator>
    <general name="gen_cyl" joint="c_hinge" biastype="affine" gainprm="150" biasprm="0 -150 -8" forcerange="-25 25"/>
  </actuator>
</mujoco>"#;

    let mut world = World::new();
    let mat = world.add_material(library::aluminium());
    let model = world.load_mjcf(xml, mat, Pose::IDENTITY).unwrap();

    let cyl_id = model.joint("c_hinge").unwrap();
    assert_eq!(model.joint("c_slide").unwrap(), cyl_id);
    let Link::Cylindrical {
        linear_limits,
        angular_limits,
        angular_drive,
        ..
    } = world.link(cyl_id)
    else {
        panic!("expected Cylindrical link");
    };
    assert_eq!(*linear_limits, Some((-0.5, 0.5)));
    assert!((angular_limits.unwrap().1 - PI).abs() < 1e-4);
    assert!((angular_drive.stiffness - 150.0).abs() < 1e-12);
    assert!((angular_drive.damping - 8.0).abs() < 1e-12);
    assert_eq!(angular_drive.max_force, Some(25.0));

    assert!(matches!(
        world.link(model.joint("u_x").unwrap()),
        Link::Universal { .. }
    ));
    assert!(matches!(
        world.link(model.joint("p_ry").unwrap()),
        Link::Planar { .. }
    ));

    // Pose check for Pose::at offset rotation alignment
    let mut w2 = World::new();
    let m2 = w2.add_material(library::aluminium());
    let rot_y = Quat::from_axis_angle(Vec3::Y, FRAC_PI_2);
    let shifted = w2
        .load_mjcf(xml, m2, Pose::new(vec3(1.0, 2.0, 3.0), rot_y))
        .unwrap();
    let Representation::Fixed(frame) = &w2.object(shifted.root).representation else {
        unreachable!()
    };
    assert!((frame.pose.position - vec3(1.0, 2.0, 3.0)).length() < 1e-12);
}

#[test]
fn mjcf_rejects_malformed_or_unsupported_descriptions() {
    let mut world = World::new();
    let mat = world.add_material(library::aluminium());
    assert!(world
        .load_mjcf("<robot name='x'/>", mat, Pose::IDENTITY)
        .is_err());
    assert!(world
        .load_mjcf(
            "<mujoco><worldbody></worldbody></mujoco>",
            mat,
            Pose::IDENTITY
        )
        .is_err());
    assert!(world
        .load_mjcf(
            "<mujoco><worldbody><body><freejoint/><joint type='hinge'/><geom size='0.1'/></body></worldbody></mujoco>",
            mat,
            Pose::IDENTITY
        )
        .is_err());
}
