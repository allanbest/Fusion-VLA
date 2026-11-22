"""Smoke-test rollout: a two-joint MJCF arm and a cube, random policy, episodes to ./episodes.

    python examples/reach_arm.py            # physics only
    python examples/reach_arm.py --render   # plus 224x224 GPU camera images in every step

Swap ``RandomPolicy`` for your VLA adapter (see docs/HANDOFF.md).
"""

import sys
import time

import numpy as np

from fusion_vla import EpisodeRecorder, FusionEnv, TaskSpec, core, rollout
from fusion_vla.adapter import RandomPolicy

ARM_MJCF = """<mujoco model="reach_arm">
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
      <body name="upper" pos="0 0 0.05">
        <inertial pos="0.15 0 0" mass="1.0" diaginertia="0.01 0.012 0.012"/>
        <joint name="shoulder" type="hinge" axis="0 1 0" range="-90 90"/>
        <geom type="capsule" fromto="0 0 0 0.30 0 0" size="0.025"/>
        <body name="fore" pos="0.30 0 0">
          <inertial pos="0.12 0 0" mass="0.6" diaginertia="0.004 0.006 0.006"/>
          <joint name="elbow" type="hinge" axis="0 1 0" range="-120 120"/>
          <geom type="capsule" fromto="0 0 0 0.24 0 0" size="0.02"/>
        </body>
      </body>
    </body>
    <body name="cube" pos="0.45 0.15 0.05">
      <freejoint/>
      <geom name="cube_geom" type="box" size="0.04 0.04 0.04"/>
    </body>
  </worldbody>
  <contact><exclude body1="stand" body2="upper"/><exclude body1="upper" body2="fore"/></contact>
  <actuator>
    <position name="act_shoulder" joint="shoulder"/>
    <position name="act_elbow" joint="elbow"/>
  </actuator>
  <sensor>
    <jointpos name="q_shoulder" joint="shoulder"/>
    <jointpos name="q_elbow" joint="elbow"/>
  </sensor>
</mujoco>"""

GOAL = np.array([-0.4, 0.6])


def make_task() -> TaskSpec:
    world = core.World()
    alu = world.add_material(core.Material.aluminium())
    model = world.load_mjcf(ARM_MJCF, alu)
    return TaskSpec(
        world=world,
        actuators=[model.actuator("act_shoulder"), model.actuator("act_elbow")],
        joints=[model.joint("shoulder"), model.joint("elbow")],
        tracked_objects=[model.link("cube")],
        action_low=[-1.5, -2.0],
        action_high=[1.5, 2.0],
        reward_fn=lambda obs, act: -np.linalg.norm(obs["qpos"] - GOAL, axis=1),
        done_fn=lambda obs: np.linalg.norm(obs["qpos"] - GOAL, axis=1) < 0.05,
        instruction="move the arm to the target pose",
        randomizer=core.DomainRandomizer(seed=7, mass_scale=(0.9, 1.1), friction_scale=(0.8, 1.2)),
    )


def make_renderer(task: TaskSpec):
    from fusion_vla.render import Camera, FusionRenderer

    cams = {
        "front": Camera(eye=(1.1, -1.0, 0.7), target=(0.25, 0.0, 0.1), fov_deg=50),
        "wrist_view": Camera(eye=(0.0, -0.05, 0.9), target=(0.4, 0.1, 0.0), fov_deg=70),
    }
    r = FusionRenderer(cams, 224, 224, up_axis="z")  # MJCF scenes are z-up
    r.set_color(task.tracked_objects[0], (0.85, 0.1, 0.1))  # the cube
    return r


if __name__ == "__main__":
    task = make_task()
    renderer = make_renderer(task) if "--render" in sys.argv else None
    env = FusionEnv(task, num_envs=8, control_dt=1 / 30, physics_dt=1 / 240, max_episode_steps=60,
                    renderer=renderer)
    policy = RandomPolicy(env.task.action_low, env.task.action_high)
    rec = EpisodeRecorder("episodes")
    t0 = time.perf_counter()
    stats = rollout(env, policy, steps=120, recorder=rec)
    dt = time.perf_counter() - t0
    print(stats)
    print(f"{stats['transitions'] / dt:.0f} control steps/s ({env.substeps} physics substeps each), "
          f"{rec.written} episodes written to ./episodes")
