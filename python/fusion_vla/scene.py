"""What an external renderer needs to draw a Fusion world.

Fusion Core simulates; it does not render. A camera pipeline (MuJoCo, pyrender,
Isaac, Blender, a custom rasteriser...) needs two things:

* :func:`scene_geometry` — static, once per scene: each object's name, kind and
  collision shape in its own frame.
* :func:`scene_state`    — per step: every object's pose in every env, as an
  ``(envs, objects, 7)`` array of ``[px, py, pz, qx, qy, qz, qw]``.

Quaternions are ``(x, y, z, w)``. Units are SI. The up axis is whatever the
scene uses: ``World()`` defaults to gravity ``(0, -9.81, 0)`` (y-up); an MJCF
file sets its own (usually z-up). Capsules lie along their local +y axis.
"""

from __future__ import annotations

from typing import Any, Dict, List, Sequence

import numpy as np

from . import core


def scene_geometry(world: "core.World") -> List[Dict[str, Any]]:
    """One dict per object: ``{"id", "name", "kind", "shape"}``.

    ``shape`` is ``None`` for deformable FEM solids; draw those from
    ``world.solid_node_positions(id)``.
    """
    out = []
    for oid in world.object_ids():
        out.append(
            {
                "id": oid,
                "name": world.object_name(oid),
                "kind": world.object_kind(oid),
                "shape": world.object_shape(oid),
            }
        )
    return out


def scene_state(batch: "core.WorldBatch", objects: Sequence["core.ObjectId"]) -> np.ndarray:
    """Poses of ``objects`` in every env: float64 ``(envs, len(objects), 7)``."""
    flat = np.asarray(batch.object_poses(list(objects)), dtype=np.float64)
    return flat.reshape(len(batch), len(objects), 7)

