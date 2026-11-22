"""The seam between Fusion and a VLA model.

Fusion-VLA does not ship a model. To plug one in, implement :class:`Policy`
(and, for image observations, :class:`Renderer`) and call :func:`rollout`, or
drive :class:`~fusion_vla.env.FusionEnv` from your own training loop.

Both are structural ``Protocol`` types: any object with these methods works,
no subclassing required.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Protocol, Sequence, runtime_checkable

import numpy as np

from .env import FusionEnv


@runtime_checkable
class Policy(Protocol):
    """Maps a batch of observations + instructions to joint-target actions."""

    def reset(self, env_ids: np.ndarray) -> None:
        """Called when the listed envs start a new episode (clear history/caches)."""

    def act(self, obs: Dict[str, np.ndarray], instructions: Sequence[str]) -> np.ndarray:
        """Return actions of shape ``(num_envs, env.action_dim)``.

        ``obs`` keys: ``qpos``, ``qvel`` ``(N, J)``; ``object_poses`` ``(N, K, 7)``;
        ``probes`` ``(N, P)``; ``image/<camera>`` ``(N, H, W, C)`` when a renderer is set.
        A VLA that predicts action chunks can buffer them here and return one per call.
        """
        ...


@runtime_checkable
class Renderer(Protocol):
    """Produces camera images from Fusion scene state. :class:`fusion_vla.render.FusionRenderer` is the built-in one."""

    def setup(self, geometry: List[Dict[str, Any]]) -> None:
        """Called once with :func:`fusion_vla.scene.scene_geometry` of the template world."""

    def render(self, poses: np.ndarray) -> Dict[str, np.ndarray]:
        """``poses``: ``(N, num_objects, 7)`` ``[px,py,pz,qx,qy,qz,qw]``, object order as in ``setup``.

        Return ``{camera_name: (N, H, W, C) uint8}``.
        """
        ...


class RandomPolicy:
    """Uniform random joint targets — a smoke-test stand-in for a real model."""

    def __init__(self, low: Sequence[float], high: Sequence[float], seed: int = 0) -> None:
        self.low = np.asarray(low, np.float64)
        self.high = np.asarray(high, np.float64)
        self.rng = np.random.default_rng(seed)

    def reset(self, env_ids: np.ndarray) -> None:
        pass

    def act(self, obs: Dict[str, np.ndarray], instructions: Sequence[str]) -> np.ndarray:
        n = len(instructions)
        return self.rng.uniform(self.low, self.high, size=(n, self.low.size))


def rollout(
    env: FusionEnv,
    policy: Policy,
    steps: int,
    *,
    recorder: Optional[Any] = None,
) -> Dict[str, Any]:
    """Run ``policy`` in ``env`` for ``steps`` control ticks (all envs in lockstep).

    With a :class:`~fusion_vla.recorder.EpisodeRecorder`, every finished episode
    is written to disk. Returns simple stats.
    """
    obs = env.reset()
    policy.reset(np.arange(env.num_envs))
    instructions = env.instructions
    episodes = 0
    returns = np.zeros(env.num_envs)
    finished_returns: List[float] = []
    solver_failures = 0
    for _ in range(steps):
        action = np.asarray(policy.act(obs, instructions), np.float64)
        next_obs, reward, terminated, truncated, info = env.step(action)
        returns += reward
        solver_failures += int(info["solver_failed"].sum())
        if recorder is not None:
            recorder.add(obs, action, reward, terminated, truncated, info, instructions)
        ended = info["ended"]
        if ended.size:
            episodes += ended.size
            finished_returns.extend(returns[ended].tolist())
            returns[ended] = 0.0
            policy.reset(ended)
        obs = next_obs
    if recorder is not None:
        recorder.flush()
    return {
        "steps": steps,
        "transitions": steps * env.num_envs,
        "episodes": episodes,
        "mean_return": float(np.mean(finished_returns)) if finished_returns else float("nan"),
        "solver_failures": solver_failures,
    }
