# Fusion-VLA

Fusion without the editor and hub, packaged for training vision-language-action
(VLA) and other robot-learning models. It includes the **Fusion Core** physics
solver, a **GPU renderer** for camera observations, the Standard engine, the
GPU compute layer and the platform crates, with a thin Python interface on top.

It contains no model, tokenizer, or training loop. You plug those in through a
small adapter; see [docs/HANDOFF.md](docs/HANDOFF.md).

```
crates/
  Fusion Core          f64 variational mechanics; std-only, zero third-party crates
    core-math            L0  f64 vectors, rotations, deterministic RNG
    core-runtime         L1  deterministic worker pool
    core-numerics        L2  sparse blocks, CG, Newton, sparse Cholesky, AD
    core-space           L3  shapes, SDFs, BVHs
    core-world           L4  materials, objects, joints, probes, URDF + MJCF import
    phys-mechanics       L5  implicit rigid + FEM mechanics, IPC contact, WorldBatch, adjoints
    phys-thermal / phys-fluids
  GPU (wgpu)
    fusion-viewport      renderer: offscreen camera images, shadows, SSAO, tone mapping
    fusion-compute       general GPU compute layer: buffers, kernels, command graph
    fusion-gpu           GPU fluid neighbour grid for the Standard engine
  Standard engine      f32 XPBD rigid bodies, PBF fluids, cloth, soft bodies, MPM
    fusion-math, fusion-core, fusion-collide, fusion-solver, fusion-mpm
  Platform
    fusion-sim           Simulation: one step loop over either engine
    fusion-project       the .fusion project format
    fusion-research      sweeps, run comparison, convergence, results store
    fusion-samples       headless training-data rollouts (NumPy shards)
    fusion-script        behaviour scripts in a deterministic subset of Python
  fusion-core-py       pyo3 bindings -> fusion_vla._core (Fusion Core + renderer)
python/fusion_vla/
  core.py              the raw solver API (World, Mechanics, WorldBatch, Renderer, ...)
  env.py               FusionEnv: batched numpy env, joint-target actions, autoreset
  render.py            FusionRenderer + Camera: GPU images as obs["image/<camera>"]
  scene.py             geometry + poses, for plugging in a different renderer
  adapter.py           Policy / Renderer protocols, rollout(), RandomPolicy
  recorder.py          one .npz per episode
examples/reach_arm.py  MJCF arm, random policy, optional cameras, records episodes
docs/                  HANDOFF.md (the API contract) + engine design docs
```

The Python package uses Fusion Core for physics. The Standard engine and
platform crates are included as Rust crates, but they are not exposed to
Python yet.

## Install

You need a Rust toolchain (1.87 or newer) and Python 3.9 or newer. The build
produces one abi3 wheel that works on every Python from 3.9 up.

```bash
pip install .            # builds the Rust extension with maturin (release profile)
pip install -e .[test]   # editable install plus pytest; rerun after changing Rust code
```

For a physics-only wheel without the GPU stack (wgpu), install with:

```bash
pip install . --config-settings=build-args="--no-default-features"
```

Rendering needs a GPU driver that wgpu can use:

- macOS: Metal, built in.
- Linux: Vulkan, from the NVIDIA driver or `mesa-vulkan-drivers`.
- Windows: DX12 or Vulkan.

On a headless Linux server, Vulkan works without a display. Check what was
found with:

```python
import fusion_vla
fusion_vla.gpu_info()   # {'name': 'Apple M4 Pro', 'backend': 'Metal', ...} or None
```

Verified so far on macOS arm64 (Apple M4 Pro, Metal) only. A Linux training
box hasn't been tried yet.

## Quick start

```python
import numpy as np
from fusion_vla import Camera, FusionEnv, FusionRenderer, TaskSpec, core

world = core.World()                                   # y-up by default; MJCF sets its own gravity
alu = world.add_material(core.Material.aluminium())
robot = world.load_mjcf_file("my_arm.xml", alu)        # or load_urdf_file(...)
cube = robot.link("cube")

task = TaskSpec(
    world=world,
    actuators=[robot.actuator(n) for n in robot.actuator_names()],
    tracked_objects=[cube],
    instruction="pick up the red cube",
)
renderer = FusionRenderer(
    {"front": Camera(eye=(1.1, -1.0, 0.7), target=(0.25, 0.0, 0.1), fov_deg=50)},
    224, 224, up_axis="z",                             # MJCF files are z-up
)
renderer.set_color(cube, (0.85, 0.1, 0.1))
env = FusionEnv(task, num_envs=64, control_dt=1/30, physics_dt=1/240,
                max_episode_steps=300, renderer=renderer)

obs = env.reset()       # {"qpos","qvel","object_poses","probes","image/front": (64,224,224,3) uint8}
obs, reward, terminated, truncated, info = env.step(np.zeros((64, env.action_dim)))
```

## Tests

```bash
cargo test --release --workspace         # every Rust crate's tests
pytest python/tests                      # bindings, env, renderer (GPU tests skip without a GPU)
python examples/reach_arm.py --render    # smoke rollout with two cameras; writes ./episodes/*.npz
```

## Performance (Apple M4 Pro, reach-arm scene, 224×224 RGB)

| | |
|---|---|
| physics, 8 envs, one control step (8 substeps) | about 1.2 ms |
| one camera frame, rendered alone (the old path) | about 1.6–2.0 ms |
| one camera frame, batched (`render_batch`, the default) | about 0.26–0.45 ms, or 2,500–3,900 frames/s |
| `env.step`, 64 envs × 2 cameras, physics + 128 images | about 43 ms, or about 1,500 env-steps/s |

The ranges come from a busy machine (best of several runs). Batched
rendering works like this:

- Scene geometry is built on every CPU core.
- Frames are split across `workers` renderer instances, each with its own GPU
  queue on its own thread.
- Every frame is queued back to back, and the CPU waits for the GPU once per
  step.

`FusionRenderer(..., workers=2, shadow_resolution=1024)` are the defaults.
On a discrete GPU, try more workers. A lower shadow resolution is faster.

## What runs on the GPU

- **Rendering** runs on the GPU through `fusion-viewport` (wgpu): every camera
  image, batched across envs and cameras, and spread over several GPU queues.
- **Fusion Core physics runs on the CPU.** Envs are parallelized across a
  deterministic thread pool, and results are bit-identical whatever the thread
  count. No GPU solver for Fusion Core exists yet.
- **`fusion-compute` and `fusion-gpu`** are the GPU compute layer and a GPU
  neighbour-grid kernel for the Standard engine's fluids. They're included and
  built, but nothing in the training path uses them.

## Threads, processes and errors

- Envs and renderers work from any Python thread. Physics and rendering
  release the GIL.
- For worker processes, use `spawn` or `forkserver`, not `fork`. GPU drivers
  don't survive `fork()`, and on macOS Metal can't start in a forked process.
  A renderer in a forked child raises a `RuntimeError` instead of crashing.
- Renderer failures raise `RuntimeError`, never a Rust panic. A failed batch
  rebuilds the GPU devices and retries once first.

## Limits

- **Primitive collision shapes only:** spheres, boxes, capsules and
  half-spaces, plus FEM tetrahedral solids. Importers approximate the rest:
  cylinders become capsules, MJCF ellipsoids and meshes become spheres, and
  URDF `<mesh>` is not supported. The renderer draws this collision geometry,
  not visual meshes.
- **The renderer is a lit-geometry renderer.** There are no textures,
  materials from MJCF/URDF, or depth and segmentation outputs yet. Per-object
  colours come from `FusionRenderer.set_color`.

## Provenance

The code was extracted from the main Fusion repo at commit `1a03b12` (branch
`core-step-speed`, committed state only). Uncommitted solver work at that time
(the AMG / step-speed pass) is **not** included.

Left out are the editor (`fusion-app`), the hub (`fusion-hub`), the demos
(`fusion-demos`) and the Standard-engine Python bindings (`fusion-py/src/lib.rs`).

The Python bindings are `crates/app/fusion-py/src/core.rs` from that commit.
Added since:

- `World.object_ids`, `object_name`, `object_kind` and `object_shape`
- `WorldBatch.object_poses`
- `render.rs`: `Renderer` and `gpu_info`
- In `fusion-viewport`: `Renderer::read_back_many` (many frames, one sync,
  split to fit the device's buffer limit, GPU errors returned as `Err`),
  `Renderer::set_shadow_resolution` and `Renderer::set_readback_limit`

License: MIT OR Apache-2.0, as declared by the source crates.
