"""GPU camera rendering. Skipped on machines with no usable GPU adapter."""

import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "examples"))

from reach_arm import make_task  # noqa: E402

from fusion_vla import FusionEnv  # noqa: E402
from fusion_vla.render import HAS_RENDER, Camera, FusionRenderer, gpu_info, save_png  # noqa: E402

pytestmark = pytest.mark.skipif(not HAS_RENDER or gpu_info() is None, reason="no GPU adapter")

CAMS = {
    "front": Camera((1.1, -1.0, 0.7), (0.25, 0.0, 0.1), 50),
    "top": Camera((0.25, 0.0, 1.6), (0.25, 0.01, 0.0), 60),
}


def make_env(n=2):
    task = make_task()
    r = FusionRenderer(CAMS, 96, 64, up_axis="z")
    r.set_color(task.tracked_objects[0], (0.9, 0.1, 0.1))
    return FusionEnv(task, num_envs=n, renderer=r), r


def test_image_obs_shapes_and_content():
    env, _ = make_env(3)
    obs = env.reset()
    for cam in CAMS:
        img = obs[f"image/{cam}"]
        assert img.shape == (3, 64, 96, 3) and img.dtype == np.uint8
        assert img.std() > 5.0  # not a blank frame
    # Every env starts identical, so the frames are identical.
    assert np.array_equal(obs["image/front"][0], obs["image/front"][2])


def test_images_follow_the_physics():
    env, _ = make_env(2)
    before = env.reset()["image/front"].copy()
    for _ in range(30):
        obs, *_ = env.step(np.array([[-0.8, 1.2], [0.6, -0.9]]))
    after = obs["image/front"]
    # The arm moved, so pixels changed, and differently per env.
    assert np.abs(after[0].astype(int) - before[0]).mean() > 0.5
    assert not np.array_equal(after[0], after[1])


def test_red_cube_is_visible_from_above():
    env, _ = make_env(1)
    img = env.reset()["image/top"][0].astype(int)
    red = (img[..., 0] > 120) & (img[..., 1] < 70) & (img[..., 2] < 70)
    assert red.sum() > 10


def test_render_world_and_png(tmp_path):
    env, r = make_env(1)
    env.reset()
    frame = r.render_world(env.world(0), "front")
    assert frame.shape == (64, 96, 3)
    p = tmp_path / "f.png"
    save_png(str(p), frame)
    assert p.read_bytes()[:8] == b"\x89PNG\r\n\x1a\n"


def test_bad_arguments():
    with pytest.raises(ValueError):
        FusionRenderer(CAMS, 4, 4)
    with pytest.raises(ValueError):
        FusionRenderer(CAMS, 64, 64, up_axis="x")
    with pytest.raises(ValueError):
        FusionRenderer({"c": Camera((0, 0, 0), (0, 0, 0))}, 64, 64).render_world(make_task().world, "c")


def test_autoreset_with_images():
    env, _ = make_env(2)
    env.max_episode_steps = 2
    env.reset()
    for _ in range(2):
        obs, _, _, trunc, info = env.step(np.zeros((2, 2)))
    assert trunc.all() and info["final_obs"]["image/front"].shape == (2, 64, 96, 3)
    assert obs["image/front"].flags.writeable


def test_autoreset_rerenders_reset_envs_only_and_correctly():
    env, r = make_env(2)
    first = env.reset()["image/front"].copy()
    env.step(np.array([[-0.8, 1.2], [-0.8, 1.2]]))
    env.episode_step[:] = [0, 10**6]  # force env 1 to truncate on the next step
    env.max_episode_steps = 10**6
    obs, _, _, trunc, info = env.step(np.array([[-0.8, 1.2], [-0.8, 1.2]]))
    assert info["ended"].tolist() == [1]
    assert np.array_equal(obs["image/front"][1], first[1])      # fresh episode = start frame
    assert not np.array_equal(obs["image/front"][0], first[0])  # env 0 kept moving
