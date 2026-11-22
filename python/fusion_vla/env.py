"""A batched, numpy-in / numpy-out environment over Fusion Core's ``WorldBatch``.

Deliberately thin: no task library, no model code. A task is a template
``World`` plus which joints are actions, which are observed, and optional
reward / termination callbacks (:class:`TaskSpec`). Everything else — the
policy, the VLA model, tokenisation, cameras — lives on the adapter side
(see :mod:`fusion_vla.adapter`).

Conventions
-----------
* All arrays are float64, leading axis = env.
* Actions are **joint position targets** for ``TaskSpec.actuators`` (the drive
  stiffness/damping come from the URDF/MJCF or ``World.set_joint_drive``).
* One ``step`` = one control tick = ``substeps`` physics steps of ``physics_dt``.
* Vector-env autoreset: envs that end are reset inside ``step``; their last
  observation is in ``info["final_obs"]`` (dict of arrays, rows for ended envs
  only, aligned with ``info["ended"]``).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np

from . import core
from .scene import scene_geometry, scene_state

Obs = Dict[str, np.ndarray]
RewardFn = Callable[[Obs, np.ndarray], np.ndarray]
DoneFn = Callable[[Obs], np.ndarray]


@dataclass
class TaskSpec:
    """Everything the env needs to know about a task."""

    #: Template world. Every env starts as a copy of it on each reset.
    world: "core.World"
    #: Joints/actuators driven by the action vector (position targets).
    actuators: Sequence["core.LinkId"]
    #: Joints reported in ``obs["qpos"]`` / ``obs["qvel"]``. Defaults to ``actuators``.
    joints: Optional[Sequence["core.LinkId"]] = None
    #: Objects whose pose goes in ``obs["object_poses"]`` (e.g. the thing to grasp).
    tracked_objects: Sequence["core.ObjectId"] = ()
    #: Optional per-actuator clip bounds for actions.
    action_low: Optional[Sequence[float]] = None
    action_high: Optional[Sequence[float]] = None
    #: ``reward_fn(obs, action) -> (N,)``. Omitted -> zeros.
    reward_fn: Optional[RewardFn] = None
    #: ``done_fn(obs) -> (N,) bool`` for task success/failure. Omitted -> never.
    done_fn: Optional[DoneFn] = None
    #: Language instruction handed to the policy. One string, or one per env.
    instruction: Any = ""
    #: Applied on every reset if given (mass / friction / stiffness / gravity jitter).
    randomizer: Optional["core.DomainRandomizer"] = None
    #: Solver settings; ``None`` uses Fusion Core defaults.
    settings: Optional["core.Settings"] = None
    #: Free-form metadata recorded alongside episodes.
    metadata: Dict[str, Any] = field(default_factory=dict)


class FusionEnv:
    """``num_envs`` copies of a task, stepped in parallel by Fusion Core."""

    def __init__(
        self,
        task: TaskSpec,
        num_envs: int = 1,
        *,
        control_dt: float = 1.0 / 30.0,
        physics_dt: float = 1.0 / 240.0,
        max_episode_steps: Optional[int] = None,
        threads: Optional[int] = None,
        renderer: Optional[Any] = None,
    ) -> None:
        if num_envs < 1:
            raise ValueError("num_envs must be >= 1")
        substeps = control_dt / physics_dt
        if abs(substeps - round(substeps)) > 1e-9 or round(substeps) < 1:
            raise ValueError("control_dt must be a whole multiple of physics_dt")
        self.task = task
        self.num_envs = num_envs
        self.control_dt = control_dt
        self.physics_dt = physics_dt
        self.substeps = int(round(substeps))
        self.max_episode_steps = max_episode_steps
        self.actuators: List[core.LinkId] = list(task.actuators)
        self.joints: List[core.LinkId] = list(task.joints if task.joints is not None else task.actuators)
        self.tracked: List[core.ObjectId] = list(task.tracked_objects)
        self.action_dim = len(self.actuators)
        self._low = None if task.action_low is None else np.asarray(task.action_low, np.float64)
        self._high = None if task.action_high is None else np.asarray(task.action_high, np.float64)

        self.batch = core.WorldBatch(task.world, num_envs, settings=task.settings, threads=threads)
        self.has_probes = task.world.probe_count > 0
        self.all_objects: List[core.ObjectId] = task.world.object_ids()
        self.geometry = scene_geometry(task.world)
        self.renderer = renderer
        if renderer is not None:
            renderer.setup(self.geometry)

        self.episode_step = np.zeros(num_envs, dtype=np.int64)
        self.episode_count = np.zeros(num_envs, dtype=np.int64)
        self._resets = 0

    # ------------------------------------------------------------------ API

    @property
    def instructions(self) -> List[str]:
        ins = self.task.instruction
        if isinstance(ins, str):
            return [ins] * self.num_envs
        ins = list(ins)
        if len(ins) != self.num_envs:
            raise ValueError("TaskSpec.instruction must be a str or one per env")
        return ins

    def reset(self, env_ids: Optional[Sequence[int]] = None) -> Obs:
        """Reset all envs (or ``env_ids``) to the template, re-randomised. Returns obs for **all** envs."""
        ids = list(range(self.num_envs)) if env_ids is None else [int(i) for i in env_ids]
        self._reset_ids(ids)
        return self.observe()

    def step(self, action: np.ndarray) -> Tuple[Obs, np.ndarray, np.ndarray, np.ndarray, Dict[str, Any]]:
        """Apply joint targets ``(num_envs, action_dim)`` and advance one control tick.

        Returns ``obs, reward, terminated, truncated, info`` (gymnasium vector-env order).
        ``info["solver_failed"]`` flags envs whose implicit solve failed; those are
        terminated and reset, since their state is not trustworthy.
        """
        a = np.asarray(action, dtype=np.float64)
        if a.shape != (self.num_envs, self.action_dim):
            raise ValueError(f"action must have shape {(self.num_envs, self.action_dim)}, got {a.shape}")
        if not np.all(np.isfinite(a)):
            raise ValueError("action contains non-finite values")
        if self._low is not None or self._high is not None:
            a = np.clip(a, self._low, self._high)

        if self.action_dim:
            self.batch.set_batch_joint_targets(self.actuators, a.ravel().tolist())
        failed = np.zeros(self.num_envs, dtype=bool)
        unconverged = np.zeros(self.num_envs, dtype=bool)
        for _ in range(self.substeps):
            self.batch.step_all(self.physics_dt)
            reps = self.batch.reports
            failed |= np.fromiter((r.failed for r in reps), bool, self.num_envs)
            unconverged |= np.fromiter((r.unconverged for r in reps), bool, self.num_envs)
        self.episode_step += 1

        obs = self.observe()
        reward = (
            np.zeros(self.num_envs)
            if self.task.reward_fn is None
            else np.asarray(self.task.reward_fn(obs, a), dtype=np.float64).reshape(self.num_envs)
        )
        terminated = failed.copy()
        if self.task.done_fn is not None:
            terminated |= np.asarray(self.task.done_fn(obs), dtype=bool).reshape(self.num_envs)
        truncated = np.zeros(self.num_envs, dtype=bool)
        if self.max_episode_steps is not None:
            truncated = (self.episode_step >= self.max_episode_steps) & ~terminated

        info: Dict[str, Any] = {
            "solver_failed": failed,
            "solver_unconverged": unconverged,
            "episode_step": self.episode_step.copy(),
        }
        ended = np.flatnonzero(terminated | truncated)
        info["ended"] = ended
        if ended.size:
            info["final_obs"] = {k: v[ended] for k, v in obs.items()}
            self._reset_ids(ended.tolist())
            fresh = self.observe()
            for k in obs:
                obs[k][ended] = fresh[k][ended]
        return obs, reward, terminated, truncated, info

    def observe(self) -> Obs:
        """Current observation for all envs (no stepping)."""
        n = self.num_envs
        obs: Obs = {}
        if self.joints:
            obs["qpos"] = np.asarray(self.batch.joint_coordinates(self.joints), np.float64).reshape(n, -1)
            obs["qvel"] = np.asarray(self.batch.joint_velocities(self.joints), np.float64).reshape(n, -1)
        if self.tracked:
            obs["object_poses"] = scene_state(self.batch, self.tracked)
        if self.has_probes:
            obs["probes"] = np.asarray(self.batch.sample_observations(), np.float64).reshape(n, -1)
        if self.renderer is not None:
            images = self.renderer.render(scene_state(self.batch, self.all_objects))
            for cam, img in images.items():
                obs[f"image/{cam}"] = np.asarray(img)
        return obs

    def world(self, env_index: int) -> "core.World":
        """A copy of one env's world (for debugging, custom queries, solid meshes)."""
        return self.batch.world_snapshot(env_index)

    # ------------------------------------------------------------ internals

    def _reset_ids(self, ids: List[int]) -> None:
        if not ids:
            return
        self.batch.reset_envs(ids)
        if self.task.randomizer is not None:
            # A fresh offset each reset so every episode draws new parameters,
            # while the whole run stays reproducible from the randomizer's seed.
            self._resets += 1
            self.batch.randomize_envs(ids, self.task.randomizer, seed_offset=self._resets * self.num_envs)
        self.episode_step[ids] = 0
        self.episode_count[ids] += 1
