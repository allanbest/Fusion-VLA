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


def test_batched_matches_single_frames_and_workers_are_deterministic():
    env, r = make_env(5)
    env.reset()
    env.step(np.array([[-0.5, 0.8]] * 5))
    for i in range(5):
        env.step(np.array([[-0.5 + 0.1 * i, 0.8]] * 5))
    batched = r.render_batch(env.batch)
    for i in range(5):
        single = r.render_world(env.world(i), "front")
        assert np.array_equal(batched["front"][i], single)
    for w in (1, 3):
        r.workers = w
        again = r.render_batch(env.batch)
        assert all(np.array_equal(again[k], batched[k]) for k in batched)
    sub = r.render_batch(env.batch, env_indices=[4, 1])
    assert np.array_equal(sub["top"][0], batched["top"][4]) and np.array_equal(sub["top"][1], batched["top"][1])


def test_tuning_knobs():
    _, r = make_env(1)
    assert r.workers == 2 and r.shadow_resolution == 1024
    r.shadow_resolution = 512
    assert r.shadow_resolution == 512
    with pytest.raises(ValueError):
        r.workers = 0


# -- reliability ---------------------------------------------------------------


def test_render_from_another_thread():
    # Training loops often step envs on a background thread. The renderer used
    # to be bound to the thread that created it and panicked anywhere else.
    import threading

    env, _ = make_env(2)
    env.reset()
    out, errors = [], []

    def worker():
        try:
            out.append(env.step(np.zeros((2, 2)))[0]["image/front"])
        except BaseException as e:  # noqa: BLE001 - a PanicException is a BaseException
            errors.append(e)

    t = threading.Thread(target=worker)
    t.start()
    t.join()
    assert not errors, errors
    assert out[0].shape == (2, 64, 96, 3)


def test_big_batches_split_into_several_readbacks():
    # One readback buffer per batch used to overflow the GPU's buffer limit at
    # around 1,200 envs x 2 cameras and panic. A tiny limit forces one frame
    # per readback; the images must not change.
    env, r = make_env(5)
    whole = env.reset()
    r.native.readback_limit = 1
    assert r.native.readback_limit == 1
    pieces = env.observe()
    for cam in CAMS:
        assert np.array_equal(whole[f"image/{cam}"], pieces[f"image/{cam}"])


def _use_inherited_renderer(renderer, world, conn):
    try:
        renderer.render_world(world, "front")
        conn.send("rendered")
    except RuntimeError as e:
        conn.send(f"RuntimeError: {e}")
    except BaseException as e:  # noqa: BLE001
        conn.send(f"{type(e).__name__}: {e}")


def _new_renderer_in_child(conn):
    try:
        FusionRenderer(CAMS, 32, 32)
        conn.send("rendered")
    except RuntimeError as e:
        conn.send(f"RuntimeError: {e}")


@pytest.mark.skipif(not hasattr(os, "fork"), reason="no fork()")
def test_fork_raises_instead_of_crashing():
    # GPU devices don't survive fork(). A forked child used to segfault; now
    # it gets a RuntimeError that says to use 'spawn', both when it reuses the
    # parent's renderer and when it makes its own (the parent had already
    # started the GPU, and on macOS Metal can't start after fork() at all).
    import multiprocessing as mp

    ctx = mp.get_context("fork")
    env, r = make_env(1)
    env.reset()
    for target, args in ((_use_inherited_renderer, (r, env.world(0))), (_new_renderer_in_child, ())):
        parent, child = ctx.Pipe()
        p = ctx.Process(target=target, args=(*args, child))
        p.start()
        msg = parent.recv() if parent.poll(60) else "no answer"
        p.join(60)
        assert p.exitcode == 0, f"child crashed with exit code {p.exitcode}"
        assert msg.startswith("RuntimeError") and "spawn" in msg, msg
