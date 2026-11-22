"""GPU camera images for Fusion envs.

:class:`FusionRenderer` draws Fusion Core worlds through the Fusion viewport's
offscreen wgpu pipeline, which has shadows, ambient occlusion and tone
mapping. It runs headless (Metal on macOS, Vulkan or GL on Linux, DX12 or
Vulkan on Windows), so no display or window is needed.

    from fusion_vla import FusionEnv
    from fusion_vla.render import Camera, FusionRenderer

    cams = {"front": Camera(eye=(1.0, -1.2, 0.8), target=(0.2, 0.0, 0.1), fov_deg=55)}
    env = FusionEnv(task, num_envs=8, renderer=FusionRenderer(cams, 224, 224, up_axis="z"))
    obs = env.reset()
    obs["image/front"]          # (8, 224, 224, 3) uint8

The renderer draws collision geometry, not visual meshes. Use
:meth:`FusionRenderer.set_color` to give task-relevant objects distinct
colours (for example "the red cube").
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Mapping, Optional, Sequence, Tuple

import numpy as np

from . import _core

Vec3 = Tuple[float, float, float]

HAS_RENDER: bool = bool(getattr(_core, "HAS_RENDER", False))


def gpu_info() -> Optional[dict]:
    """The adapter wgpu would render on, or ``None`` when there is no usable GPU."""
    if not HAS_RENDER:
        return None
    return _core.gpu_info()


@dataclass(frozen=True)
class Camera:
    """A pinhole camera, in the scene's own frame and units."""

    eye: Vec3
    target: Vec3
    fov_deg: float = 60.0


class FusionRenderer:
    """Renders every env of a :class:`~fusion_vla.env.FusionEnv` from named cameras.

    It implements the :class:`~fusion_vla.adapter.Renderer` protocol
    (``setup``/``render``) and also ``render_batch``, which ``FusionEnv``
    prefers. ``render_batch`` reads the worlds directly, so FEM solids are
    drawn too.

    Each ``render_batch`` call renders every camera of every env as one batch:

    * scene geometry for all envs is built on all CPU cores in parallel;
    * frames are split across ``workers`` renderer instances (each its own GPU
      device and queue, on its own thread), so frame encoding uses several
      cores and the GPU always has work queued;
    * each worker queues its frames back to back and the CPU waits once.

    Tuning: ``workers`` (default 2) — try 2–8 on a discrete GPU;
    ``shadow_resolution`` (default 1024 texels) — 512 is faster, 2048 crisper.
    Output is bit-identical for any ``workers`` value.
    """

    def __init__(
        self,
        cameras: Mapping[str, Camera],
        width: int = 224,
        height: int = 224,
        *,
        up_axis: str = "y",
        workers: int = 2,
        shadow_resolution: int = 1024,
    ) -> None:
        if not HAS_RENDER:
            raise RuntimeError("fusion_vla was built without the 'render' feature")
        if not cameras:
            raise ValueError("need at least one camera")
        self.cameras: Dict[str, Camera] = dict(cameras)
        self.width = width
        self.height = height
        self.native = _core.Renderer(width, height, up_axis)
        self.native.shadow_resolution = shadow_resolution
        self.native.workers = workers

    # -- configuration -------------------------------------------------------

    def set_color(self, object_id, rgb: Vec3) -> None:
        """Colour one object (linear RGB in 0..1) in every env."""
        self.native.set_color(object_id, tuple(float(c) for c in rgb))

    def set_floor(self, extent: float = 4.0, tile: float = 0.25, color_a: Vec3 = (0.42, 0.43, 0.45),
                  color_b: Vec3 = (0.30, 0.31, 0.33)) -> None:
        self.native.set_floor(extent, tile, color_a, color_b)

    def set_lighting(self, **kwargs) -> None:
        """``exposure``, ``ambient``, ``shadow_extent``, ``ao_strength``, ``studio``."""
        self.native.set_lighting(**kwargs)

    # -- rendering -----------------------------------------------------------

    def render_world(self, world, camera: str) -> np.ndarray:
        """One ``(H, W, 3)`` uint8 frame of a single ``core.World``."""
        c = self.cameras[camera]
        raw = self.native.render_world(world, c.eye, c.target, c.fov_deg)
        return np.frombuffer(raw, np.uint8).reshape(self.height, self.width, 3).copy()

    def render_batch(self, batch, env_indices: Optional[Sequence[int]] = None) -> Dict[str, np.ndarray]:
        """``{camera: (N, H, W, 3) uint8}`` for every env (or ``env_indices``) of a ``WorldBatch``.

        Every camera of every env is rendered in one GPU batch: scene geometry is
        built on all CPU cores in parallel, all frames are queued back to back on
        the GPU, and the CPU waits once.
        """
        ids = None if env_indices is None else [int(i) for i in env_indices]
        n = len(batch) if ids is None else len(ids)
        names = list(self.cameras)
        cams = [(c.eye, c.target, c.fov_deg) for c in self.cameras.values()]
        raw = self.native.render_cameras(batch, cams, ids)
        # Copied: frombuffer views are read-only, and callers (autoreset) write into obs.
        frames = np.frombuffer(raw, np.uint8).reshape(len(names), n, self.height, self.width, 3).copy()
        return {name: frames[i] for i, name in enumerate(names)}

    @property
    def workers(self) -> int:
        """Renderer instances a batch is split across (each with its own GPU queue)."""
        return self.native.workers

    @workers.setter
    def workers(self, n: int) -> None:
        self.native.workers = n

    @property
    def shadow_resolution(self) -> int:
        return self.native.shadow_resolution

    @shadow_resolution.setter
    def shadow_resolution(self, texels: int) -> None:
        self.native.shadow_resolution = texels

    @property
    def threads(self) -> int:
        """CPU threads used to build scene geometry (default: all cores)."""
        return self.native.threads

    @threads.setter
    def threads(self, n: int) -> None:
        self.native.threads = n

    # -- adapter.Renderer protocol (unused by FusionEnv, which calls render_batch) --

    def setup(self, geometry) -> None:
        pass

    def render(self, poses: np.ndarray) -> Dict[str, np.ndarray]:
        raise NotImplementedError("FusionRenderer renders from worlds: use render_batch(env.batch)")


def save_png(path: str, image: np.ndarray) -> None:
    """Write an ``(H, W, 3)`` uint8 image as PNG, with no extra dependencies."""
    import struct
    import zlib

    img = np.ascontiguousarray(image, dtype=np.uint8)
    h, w, c = img.shape
    if c != 3:
        raise ValueError("expected (H, W, 3)")
    raw = b"".join(b"\x00" + img[y].tobytes() for y in range(h))

    def chunk(tag: bytes, data: bytes) -> bytes:
        return struct.pack(">I", len(data)) + tag + data + struct.pack(">I", zlib.crc32(tag + data) & 0xFFFFFFFF)

    png = b"\x89PNG\r\n\x1a\n" + chunk(b"IHDR", struct.pack(">IIBBBBB", w, h, 8, 2, 0, 0, 0))
    png += chunk(b"IDAT", zlib.compress(raw, 6)) + chunk(b"IEND", b"")
    with open(path, "wb") as f:
        f.write(png)
