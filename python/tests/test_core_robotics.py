"""Contracts for `fusion_vla.core` Python bindings (URDF, MJCF, FEM solids,
vectorized WorldBatch + DomainRandomizer, Probes, Checkpoints, and IFT Adjoints).

    pytest python/tests   (after `pip install .`)
"""

import math
import fusion_vla
import fusion_vla.core as fc
from fusion_vla.core import (
    AdjointLoss,
    DomainRandomizer,
    JointDrive,
    LinkPoint,
    Material,
    Mechanics,
    Settings,
    World,
    WorldBatch,
)


ARM_URDF = """<?xml version="1.0"?>
<robot name="py_arm">
  <link name="base">
    <inertial><mass value="4.0"/><inertia ixx="0.04" iyy="0.04" izz="0.04"/></inertial>
    <collision><geometry><box size="0.12 0.12 0.12"/></geometry></collision>
  </link>
  <link name="link1">
    <inertial><origin xyz="0.15 0 0"/><mass value="1.2"/><inertia ixx="0.01" iyy="0.015" izz="0.015"/></inertial>
    <collision><origin xyz="0.15 0 0"/><geometry><cylinder radius="0.03" length="0.30"/></geometry></collision>
  </link>
  <joint name="j1" type="revolute">
    <parent link="base"/><child link="link1"/>
    <origin xyz="0 0.06 0"/><axis xyz="0 0 1"/>
    <limit lower="-1.5" upper="1.5"/>
  </joint>
</robot>"""

ARM_MJCF = """<mujoco model="py_mjcf_arm">
  <compiler angle="degree"/>
  <option gravity="0 0 -9.81"/>
  <default>
    <joint damping="1.0"/>
    <position kp="240" kv="18" forcerange="-60 60"/>
  </default>
  <worldbody>
    <geom name="ground" type="plane" size="4 4 0.1" pos="0 0 0"/>
    <body name="stand" pos="0 0 0.05">
      <geom type="box" size="0.06 0.06 0.05"/>
      <body name="arm" pos="0 0 0.05">
        <inertial pos="0.15 0 0" mass="1.0" diaginertia="0.01 0.012 0.012"/>
        <joint name="shoulder" type="hinge" pos="0 0 0" axis="0 1 0" range="-90 90"/>
        <geom type="capsule" fromto="0 0 0 0.30 0 0" size="0.025"/>
        <site name="tip" pos="0.30 0 0" size="0.02" zaxis="0 0 -1"/>
      </body>
    </body>
    <body name="cube" pos="0.4 0.15 0.05">
      <freejoint/>
      <geom name="cube_geom" type="box" size="0.04 0.04 0.04"/>
      <site name="cube_pad" pos="0 0 -0.04" size="0.07" zaxis="0 0 -1"/>
    </body>
  </worldbody>
  <contact>
    <exclude body1="stand" body2="arm"/>
  </contact>
  <actuator>
    <position name="act_shoulder" joint="shoulder"/>
  </actuator>
  <sensor>
    <jointpos name="q_shoulder" joint="shoulder"/>
    <touch name="cube_touch" site="cube_pad"/>
    <accelerometer name="tip_imu" site="tip"/>
  </sensor>
</mujoco>"""


def test_urdf_and_checkpoint_replay():
    world = World()
    mat = world.add_material(Material.aluminium())
    model = world.load_urdf(ARM_URDF, mat, pos=(0.0, 1.0, 0.0), fixed_base=True)
    j1 = model.joint("j1")
    world.set_joint_drive(j1, JointDrive(target=0.4, stiffness=220.0, damping=18.0))
    p_j = world.add_joint_probe(j1)
    p_w = world.add_wrench_probe(j1)

    mech = Mechanics(Settings(augmented_lagrangian_passes=2), threads=2)
    snap0 = mech.checkpoint_bytes(world)
    for _ in range(50):
        rep = mech.step(world, 1.0 / 240.0)
        assert not rep.failed
    q_after = world.joint_coordinate(j1)
    assert abs(q_after - 0.4) < 0.2, q_after

    readings = mech.sample_probes(world)
    assert readings[p_j.index].kind == "joint_state"
    assert abs(readings[p_j.index].coordinate - q_after) < 1e-12
    assert readings[p_w.index].kind == "wrench"
    assert len(readings[p_w.index].as_vector()) == 6

    mech.restore_checkpoint_bytes(world, snap0)
    assert world.time == 0.0
    for _ in range(50):
        mech.step(world, 1.0 / 240.0)
    assert abs(world.joint_coordinate(j1) - q_after) < 1e-12


def test_mjcf_arm_and_fem_solid():
    w = World()
    alu = w.add_material(Material.aluminium())
    model = w.load_mjcf(ARM_MJCF, alu)
    act = model.actuator("act_shoulder")
    w.set_joint_target(act, -0.3)

    mech = Mechanics(threads=2)
    for _ in range(60):
        rep = mech.step(w, 1.0 / 240.0)
        assert not rep.failed

    readings = mech.sample_probes(w)
    touch = readings[model.probe("cube_touch").index]
    imu = readings[model.probe("tip_imu").index]
    assert touch.normal_force > 0.0, touch.to_dict()
    assert math.isfinite(imu.linear_acceleration[2])
    assert abs(w.joint_coordinate(model.joint("shoulder")) - (-0.3)) < 0.2

    # Deformable FEM solid with anchored left face bending under gravity
    ws = World()
    rubber = ws.add_material(Material.rubber().with_plasticity(5e6, 1e5))
    solid = ws.add_solid_box(rubber, (0.4, 0.1, 0.1), edge=0.1, pos=(0.2, 1.0, 0.0))
    anchor = ws.anchor_in_box(solid, (-0.01, 0.9, -0.1), (0.01, 1.1, 0.1))
    assert anchor is not None
    ms = Mechanics(threads=2)
    for _ in range(10):
        assert not ms.step(ws, 1.0 / 120.0).failed
    assert ms.strain_energy(ws) > 0.0
    assert len(ws.solid_node_positions(solid)) > 8


def test_vectorized_world_batch_and_determinism():
    tmpl = World()
    mat = tmpl.add_material(Material.aluminium())
    arm = tmpl.add_rigid_box(mat, (0.2, 0.04, 0.04), pos=(0.2, 1.0, 0.0))
    hinge = tmpl.add_hinge(
        arm,
        local_a=(-0.2, 0.0, 0.0),
        local_b=(0.0, 1.0, 0.0),
        axis_a=(0.0, 0.0, 1.0),
        drive=JointDrive(target=0.0, stiffness=90.0, damping=6.0),
    )
    tmpl.add_joint_probe(hinge)

    rand = DomainRandomizer(
        seed=99,
        mass_scale=(0.8, 1.25),
        stiffness_scale=(0.85, 1.15),
        gravity_jitter=(0.0, 0.3, 0.0),
    )

    def rollout(threads):
        batch = WorldBatch(tmpl, 4, threads=threads)
        batch.randomize_all(rand)
        batch.set_batch_joint_targets([hinge], [0.25, -0.25, 0.4, -0.4])
        for _ in range(12):
            batch.step_all(1.0 / 120.0)
            assert all(not r.failed for r in batch.reports)
        return batch.sample_observations(), batch.joint_coordinates([hinge])

    obs1, q1 = rollout(1)
    obs4, q4 = rollout(4)
    assert obs1 == obs4 and q1 == q4
    assert q1[0] > 0.0 and q1[1] < 0.0
    assert len(obs1) == 4 and len(obs1[0]) == 2

    batch = WorldBatch(tmpl, 4, threads=2)
    batch.set_batch_joint_targets([hinge], [0.2, 0.2, 0.2, 0.2])
    batch.step_all(1.0 / 120.0)
    batch.reset_envs([1, 3])
    assert batch.world_snapshot(1).time == 0.0
    assert batch.world_snapshot(0).time > 0.0


def test_ift_adjoint_matches_finite_differences():
    def build(target, stiffness, gy, vx, py):
        w = World(gravity=(0.0, gy, 0.0))
        mat = w.add_material(Material.aluminium())
        body = w.add_rigid_box(mat, (0.2, 0.05, 0.05), pos=(0.2, py, 0.0))
        w.set_velocity(body, (vx, -0.2, 0.05))
        hinge = w.add_hinge(
            body,
            local_a=(-0.2, 0.0, 0.0),
            local_b=(0.0, 0.0, 0.0),
            axis_a=(0.0, 0.0, 1.0),
            stiffness=5e4,
            angular_stiffness=1e4,
            drive=JointDrive(target=target, stiffness=stiffness, damping=0.0),
        )
        return w, body, hinge

    settings = Settings(
        integrator="backward_euler",
        islands=False,
        sleeping=False,
        impact_law=False,
        contact_precision=1e-3,
        tolerance=1e-9,
        max_iterations=200,
    )
    h = 0.02
    goal_p = (0.25, 0.12, -0.02)
    goal_v = (-0.05, 0.3, 0.0)

    def eval_loss(target, stiffness, gy, vx, py):
        w, body, _ = build(target, stiffness, gy, vx, py)
        m = Mechanics(settings, threads=2)
        assert not m.step(w, h).failed
        p = w.object_position(body)
        v = w.object_velocity(body)
        dp2 = sum((a - b) ** 2 for a, b in zip(p, goal_p))
        dv2 = sum((a - b) ** 2 for a, b in zip(v, goal_v))
        return 0.5 * dp2 + 0.25 * dv2

    t0, k0, gy0, vx0, py0 = 0.35, 60.0, -9.81, 0.1, 0.0
    w0, b0, h0 = build(t0, k0, gy0, vx0, py0)
    m_prev = Mechanics(settings, threads=2)
    w_end = w0.clone()
    assert not m_prev.step(w_end, h).failed
    p_end = w_end.object_position(b0)
    v_end = w_end.object_velocity(b0)

    loss = AdjointLoss()
    loss.add_object_position_grad(b0, tuple(a - b for a, b in zip(p_end, goal_p)))
    loss.add_object_velocity_grad(b0, tuple(0.5 * (a - b) for a, b in zip(v_end, goal_v)))
    grads = m_prev.backward_step(w0, h, loss)
    assert grads.converged

    eps = 1e-5
    eps_k = 1e-4

    def rel_err(an, plus, minus, step):
        fd = (plus - minus) / (2.0 * step)
        return abs(an - fd) / max(abs(an), abs(fd), 1e-4)

    assert rel_err(
        grads.joint_target_grad(h0),
        eval_loss(t0 + eps, k0, gy0, vx0, py0),
        eval_loss(t0 - eps, k0, gy0, vx0, py0),
        eps,
    ) < 1e-5
    assert rel_err(
        grads.joint_stiffness_grad(h0),
        eval_loss(t0, k0 + eps_k, gy0, vx0, py0),
        eval_loss(t0, k0 - eps_k, gy0, vx0, py0),
        eps_k,
    ) < 1e-5
    assert rel_err(
        grads.d_gravity[1],
        eval_loss(t0, k0, gy0 + eps, vx0, py0),
        eval_loss(t0, k0, gy0 - eps, vx0, py0),
        eps,
    ) < 1e-5
    assert rel_err(
        grads.d_initial_velocities[0],
        eval_loss(t0, k0, gy0, vx0 + eps, py0),
        eval_loss(t0, k0, gy0, vx0 - eps, py0),
        eps,
    ) < 1e-5
    assert rel_err(
        grads.d_initial_positions[1],
        eval_loss(t0, k0, gy0, vx0, py0 + eps),
        eval_loss(t0, k0, gy0, vx0, py0 - eps),
        eps,
    ) < 1e-5


def test_invalid_arguments_raise_clean_python_errors():
    w = World()
    mat = w.add_material(Material.steel())
    assert fc.__name__ == "fusion_vla.core" and hasattr(fusion_vla, "core")
    for fn in (
        lambda: Settings(integrator="bogus"),
        lambda: WorldBatch(w, 0),
        lambda: Mechanics(threads=0),
        lambda: w.load_mjcf("<not_mjcf/>", mat),
    ):
        try:
            fn()
        except ValueError:
            pass
        else:
            raise AssertionError("expected ValueError")

    box = w.add_rigid_box(mat, (0.1, 0.1, 0.1))
    try:
        w.object_position(fc.ObjectId if hasattr(box, "nope") else box)
    except Exception:
        raise AssertionError("valid box id failed")
    try:
        w.set_joint_target(w.add_ball_joint(LinkPoint.body(box), LinkPoint.world((0, 0, 0)), 1e3), 0.1)
    except ValueError:
        pass
    else:
        raise AssertionError("expected ValueError for BallJoint set_joint_target")


if __name__ == "__main__":
    tests = [
        test_urdf_and_checkpoint_replay,
        test_mjcf_arm_and_fem_solid,
        test_vectorized_world_batch_and_determinism,
        test_ift_adjoint_matches_finite_differences,
        test_invalid_arguments_raise_clean_python_errors,
    ]
    for test in tests:
        test()
        print(f"ok  {test.__name__}")
    print(f"{len(tests)} passed")
