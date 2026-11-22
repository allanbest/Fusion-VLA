# Fusion-VLA

This repo packages the **Fusion Core** physics solver on its own, with a thin
Python interface for training vision-language-action (VLA) and other
robot-learning models.

It contains no model, tokenizer, or training loop. You plug those in through a
small adapter; see [docs/HANDOFF.md](docs/HANDOFF.md).

```
crates/                Rust. Fusion Core: std-only, zero third-party crates
  core-math            L0  f64 vectors, rotations, deterministic RNG
  core-runtime         L1  deterministic worker pool
  core-numerics        L2  sparse blocks, CG, Newton, sparse Cholesky, AD
  core-space           L3  shapes, SDFs, BVHs
  core-world           L4  materials, objects, joints, probes, URDF + MJCF import
  phys-mechanics       L5  implicit rigid + FEM mechanics, IPC contact, WorldBatch, adjoints
  phys-thermal / phys-fluids
  fusion-core-py       pyo3 bindings -> fusion_vla._core
python/fusion_vla/
  core.py              the raw solver API (World, Mechanics, WorldBatch, ...)
  env.py               FusionEnv: batched numpy env, joint-target actions, autoreset
  scene.py             geometry + poses for an external renderer
  adapter.py           Policy / Renderer protocols, rollout(), RandomPolicy
  recorder.py          one .npz per episode
examples/reach_arm.py  MJCF arm, random policy, records episodes
docs/                  HANDOFF.md (the API contract) + engine design docs
```

## Install

You need a Rust toolchain (1.87 or newer) and Python 3.9 or newer. The build
produces one abi3 wheel that works on every Python from 3.9 up.

```bash
pip install .            # builds the Rust extension with maturin (release profile)
pip install -e .[test]   # editable install plus pytest; rerun after changing Rust code
```

Verified so far on macOS arm64 only. The crates are std-only and pyo3 builds
on Linux, but no one has run it on a Linux training box yet. To build a wheel
you can copy elsewhere, use `pip wheel . -w dist/` or `maturin build --release`.

## Quick start

```python
import numpy as np
from fusion_vla import FusionEnv, TaskSpec, core

world = core.World()                                   # y-up by default; MJCF sets its own gravity
alu = world.add_material(core.Material.aluminium())
robot = world.load_mjcf_file("my_arm.xml", alu)        # or load_urdf_file(...)

task = TaskSpec(
    world=world,
    actuators=[robot.actuator(n) for n in robot.actuator_names()],
    tracked_objects=[robot.link("cube")],
    instruction="pick up the red cube",
)
env = FusionEnv(task, num_envs=64, control_dt=1/30, physics_dt=1/240, max_episode_steps=300)

obs = env.reset()                                      # {"qpos","qvel","object_poses","probes"}
obs, reward, terminated, truncated, info = env.step(np.zeros((64, env.action_dim)))
```

## Tests

```bash
cargo test --release --workspace   # about 200 Rust solver tests
pytest python/tests                # binding and env contracts (after pip install)
python examples/reach_arm.py       # smoke rollout; writes ./episodes/*.npz
```

## What it is not (yet)

- **No renderer.** Fusion Core simulates; it does not draw. For image
  observations, pass a `Renderer` to `FusionEnv`. The renderer gets each
  object's shape once (`scene_geometry`) and every pose on every step
  (`scene_state`). Any rasterizer works.
- **CPU only.** Envs are parallelized across a deterministic thread pool.
  Results are bit-identical whatever the thread count. There is no GPU batching.
- **Primitive collision shapes only:** spheres, boxes, capsules and
  half-spaces, plus FEM tetrahedral solids. Importers approximate the rest:
  cylinders become capsules, MJCF ellipsoids and meshes become spheres, and
  URDF `<mesh>` is not supported. Robots described with mesh collision
  geometry need primitive collision geoms.

## Provenance

The solver was extracted from the main Fusion repo at commit `1a03b12`
(branch `core-step-speed`, committed state only). Uncommitted solver work at
that time (the AMG / step-speed pass) is **not** included. The bindings are
`crates/app/fusion-py/src/core.rs` from that commit, with these additions:
`World.object_ids/object_name/object_kind/object_shape` and
`WorldBatch.object_poses`.

License: MIT OR Apache-2.0, as declared by the source crates.
