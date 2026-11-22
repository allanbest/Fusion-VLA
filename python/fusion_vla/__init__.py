"""Fusion-VLA: the Fusion Core solver packaged for VLA / robot-learning training.

Layers, lowest first:

* :mod:`fusion_vla.core`     — the raw Rust solver (World, Mechanics, WorldBatch, ...)
* :mod:`fusion_vla.env`      — :class:`FusionEnv`, a batched, numpy-in/numpy-out env
* :mod:`fusion_vla.scene`    — geometry + poses for an external renderer (cameras)
* :mod:`fusion_vla.adapter`  — the ``Policy`` / ``Renderer`` protocols a VLA plugs into,
  and :func:`rollout`
* :mod:`fusion_vla.recorder` — episode recording to ``.npz``

Fusion Core has no renderer. Image observations come from whatever renderer the
adapter supplies, fed by :func:`fusion_vla.scene.scene_state`.
"""

from . import core
from .adapter import Policy, Renderer, rollout
from .env import FusionEnv, TaskSpec
from .recorder import EpisodeRecorder
from .scene import scene_geometry, scene_state

__version__ = core.__version__

__all__ = [
    "core",
    "FusionEnv",
    "TaskSpec",
    "Policy",
    "Renderer",
    "rollout",
    "EpisodeRecorder",
    "scene_geometry",
    "scene_state",
]
