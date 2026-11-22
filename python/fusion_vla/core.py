"""Fusion Core: the native solver bindings.

Everything here is implemented in Rust (``crates/fusion-core-py/src/core.rs``)
and re-exported unchanged. See ``docs/HANDOFF.md`` for the API contract.
"""

from ._core import (  # noqa: F401
    AdjointGradients,
    AdjointLoss,
    DomainRandomizer,
    JointDrive,
    LinkId,
    LinkPoint,
    Material,
    MaterialId,
    Mechanics,
    ObjectId,
    ProbeId,
    ProbeReading,
    RobotModel,
    Settings,
    StepReport,
    World,
    WorldBatch,
    __version__,
)

__all__ = [
    "AdjointGradients",
    "AdjointLoss",
    "DomainRandomizer",
    "JointDrive",
    "LinkId",
    "LinkPoint",
    "Material",
    "MaterialId",
    "Mechanics",
    "ObjectId",
    "ProbeId",
    "ProbeReading",
    "RobotModel",
    "Settings",
    "StepReport",
    "World",
    "WorldBatch",
]
