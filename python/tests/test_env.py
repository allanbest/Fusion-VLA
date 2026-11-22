"""Contracts for the thin env / adapter / recorder layer."""

import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "examples"))

from reach_arm import make_task  # noqa: E402

from fusion_vla import EpisodeRecorder, FusionEnv, Policy, Renderer, rollout, scene_geometry  # noqa: E402
from fusion_vla.adapter import RandomPolicy  # noqa: E402


def test_shapes_and_autoreset():
    env = FusionEnv(make_task(), num_envs=3, max_episode_steps=4, threads=2)
    obs = env.reset()
    assert obs["qpos"].shape == (3, 2) and obs["qvel"].shape == (3, 2)
    assert obs["object_poses"].shape == (3, 1, 7)
    assert obs["probes"].shape[0] == 3
    q = obs["object_poses"][0, 0, 3:]
    assert abs(np.linalg.norm(q) - 1.0) < 1e-9
    a = np.zeros((3, 2))
    for t in range(4):
        obs, r, term, trunc, info = env.step(a)
        assert r.shape == (3,) and not info["solver_failed"].any()
    assert trunc.all() and set(info["ended"].tolist()) == {0, 1, 2}
    assert info["final_obs"]["qpos"].shape == (3, 2)
    assert (env.episode_step == 0).all()


def test_actions_drive_joints():
    env = FusionEnv(make_task(), num_envs=2, threads=2)
    env.reset()
    target = np.array([[-0.5, 0.5], [-0.3, -0.4]])  # arm raised: clear of the ground
    for _ in range(30):
        obs, *_ = env.step(target)
    assert np.abs(obs["qpos"] - target).max() < 0.15, obs["qpos"]


def test_determinism_across_thread_counts():
    def run(threads):
        env = FusionEnv(make_task(), num_envs=4, threads=threads)
        env.reset()
        pol = RandomPolicy([-1, -1], [1, 1], seed=3)
        for _ in range(5):
            obs, *_ = env.step(pol.act({}, env.instructions))
        return obs["qpos"]

    assert np.array_equal(run(1), run(4))


def test_geometry_and_renderer_hook():
    class FakeRenderer:
        def setup(self, geometry):
            self.n = len(geometry)

        def render(self, poses):
            assert poses.shape[1:] == (self.n, 7)
            return {"front": np.zeros((poses.shape[0], 8, 8, 3), np.uint8)}

    task = make_task()
    geo = scene_geometry(task.world)
    kinds = {g["shape"]["type"] for g in geo if g["shape"]}
    assert {"halfspace", "box", "capsule"} <= kinds
    r = FakeRenderer()
    assert isinstance(r, Renderer)
    env = FusionEnv(task, num_envs=2, renderer=r)
    assert env.reset()["image/front"].shape == (2, 8, 8, 3)


def test_rollout_records_episodes(tmp_path):
    env = FusionEnv(make_task(), num_envs=2, max_episode_steps=5, threads=2)
    pol = RandomPolicy(env.task.action_low, env.task.action_high)
    assert isinstance(pol, Policy)
    rec = EpisodeRecorder(str(tmp_path))
    stats = rollout(env, pol, steps=10, recorder=rec)
    assert stats["episodes"] == 4 and stats["solver_failures"] == 0
    assert rec.written == 4
    ep = np.load(tmp_path / "episode_000000.npz")
    assert ep["action"].shape == (5, 2) and ep["obs/qpos"].shape == (5, 2)
    assert str(ep["instruction"]) == "move the arm to the target pose"
