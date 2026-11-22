"""Fusion-VLA: the Fusion Core solver packaged for VLA / robot-learning training.

Layers, lowest first:

* :mod:`fusion_vla.core`     — the raw Rust solver (World, Mechanics, WorldBatch, ...)
* :mod:`fusion_vla.env`      — :class:`FusionEnv`, a batched, numpy-in/numpy-out env
* :mod:`fusion_vla.render`   — GPU camera images (:class:`FusionRenderer`, :class:`Camera`)
* :mod:`fusion_vla.scene`    — geometry + poses, for plugging in a different renderer
* :mod:`fusion_vla.adapter`  — the ``Policy`` / ``Renderer`` protocols a VLA plugs into,
  and :func:`rollout`
* :mod:`fusion_vla.recorder` — episode recording to ``.npz``

Image observations come from :class:`fusion_vla.render.FusionRenderer` (the Fusion
viewport's wgpu pipeline, headless), or any object implementing
:class:`fusion_vla.adapter.Renderer`.
"""

from . import core
from .adapter import Policy, Renderer, rollout
from .env import FusionEnv, TaskSpec
from .recorder import EpisodeRecorder
from .render import Camera, FusionRenderer, gpu_info
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
    "Camera",
    "FusionRenderer",
    "gpu_info",
]
