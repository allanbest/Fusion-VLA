"""Episode recording: one ``.npz`` per finished episode.

Format (all arrays have leading axis T = episode length):

* ``obs/<key>``   — observation *before* the action (e.g. ``obs/qpos``, ``obs/image/front``)
* ``action``      — ``(T, action_dim)``
* ``reward``      — ``(T,)``
* ``terminated``, ``truncated``, ``solver_failed`` — ``(T,)`` bool
* ``instruction`` — 0-d string
* ``env_index``, ``episode_index`` — ints

Simple on purpose; convert to LeRobot / RLDS / your own format on the model side.
"""

from __future__ import annotations

import os
from collections import defaultdict
from typing import Any, Dict, List, Sequence

import numpy as np


class EpisodeRecorder:
    def __init__(self, out_dir: str, *, keep_partial: bool = False) -> None:
        self.out_dir = out_dir
        self.keep_partial = keep_partial
        os.makedirs(out_dir, exist_ok=True)
        self._buf: Dict[int, Dict[str, List[Any]]] = defaultdict(lambda: defaultdict(list))
        self._instr: Dict[int, str] = {}
        self.written = 0

    def add(
        self,
        obs: Dict[str, np.ndarray],
        action: np.ndarray,
        reward: np.ndarray,
        terminated: np.ndarray,
        truncated: np.ndarray,
        info: Dict[str, Any],
        instructions: Sequence[str],
    ) -> None:
        n = action.shape[0]
        for i in range(n):
            b = self._buf[i]
            for k, v in obs.items():
                b["obs/" + k].append(np.array(v[i]))
            b["action"].append(action[i].copy())
            b["reward"].append(float(reward[i]))
            b["terminated"].append(bool(terminated[i]))
            b["truncated"].append(bool(truncated[i]))
            b["solver_failed"].append(bool(info["solver_failed"][i]))
            self._instr[i] = instructions[i]
        for i in info["ended"]:
            self._write(int(i))

    def flush(self) -> None:
        """Write unfinished episodes too if ``keep_partial``; otherwise drop them."""
        for i in list(self._buf):
            if self.keep_partial and self._buf[i]["action"]:
                self._write(i)
        self._buf.clear()

    def _write(self, i: int) -> None:
        b = self._buf.pop(i, None)
        if not b or not b["action"]:
            return
        arrays = {k: np.stack(v) if k.startswith("obs/") or k == "action" else np.asarray(v) for k, v in b.items()}
        arrays["instruction"] = np.array(self._instr.get(i, ""))
        arrays["env_index"] = np.array(i)
        arrays["episode_index"] = np.array(self.written)
        path = os.path.join(self.out_dir, f"episode_{self.written:06d}.npz")
        np.savez_compressed(path, **arrays)
        self.written += 1
