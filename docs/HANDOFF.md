# Handoff: wiring a VLA model to Fusion-VLA

This is the contract for whoever writes the model-side adapter. It covers what
the package gives you, what you have to supply, and the details that are easy
to get wrong.

## 1. What you implement

Two structural protocols live in `fusion_vla/adapter.py`. You don't subclass
anything; any object with these methods works.

```python
class Policy:
    def reset(self, env_ids: np.ndarray) -> None: ...           # new episodes in these envs
    def act(self, obs: dict[str, np.ndarray], instructions: list[str]) -> np.ndarray:
        ...                                                     # (num_envs, action_dim) float

class Renderer:                                                 # only if the model needs images
    def setup(self, geometry: list[dict]) -> None: ...          # once
    def render(self, poses: np.ndarray) -> dict[str, np.ndarray]:
        ...                                                     # {cam: (N,H,W,C) uint8}
```

Then call one of:

- `fusion_vla.rollout(env, policy, steps, recorder=EpisodeRecorder(dir))` to
  collect data or evaluate a policy.
- `env.reset()` and `env.step(action)` from your own loop (RL, DAgger, online
  fine-tuning).

## 2. The env

```python
FusionEnv(task: TaskSpec, num_envs, *, control_dt=1/30, physics_dt=1/240,
          max_episode_steps=None, threads=None, renderer=None)
reset(env_ids=None) -> obs
step(action) -> obs, reward, terminated, truncated, info      # gymnasium vector-env order
observe() -> obs
world(i) -> core.World                                        # a copy of env i, for custom queries
env.action_dim, env.num_envs, env.instructions, env.geometry, env.batch (raw WorldBatch)
```

`TaskSpec` fields: `world` (the template), `actuators`, `joints`,
`tracked_objects`, `action_low`/`action_high`, `reward_fn(obs, action)`,
`done_fn(obs)`, `instruction` (one string or one per env), `randomizer`,
`settings` and `metadata`.

### Observations

Everything is float64 with the env as the leading axis:

| key | shape | meaning |
|---|---|---|
| `qpos`, `qvel` | `(N, J)` | joint coordinate (rad or m) and rate, for `TaskSpec.joints` |
| `object_poses` | `(N, K, 7)` | `[px, py, pz, qx, qy, qz, qw]` for `tracked_objects` |
| `probes` | `(N, P)` | every probe flattened in probe-id order (MJCF `<sensor>`s become probes) |
| `image/<cam>` | `(N, H, W, C)` | only present when a `Renderer` is set |

Each probe contributes a fixed number of values, so `P` is constant for a
scene. The order within one probe is given by `ProbeReading.as_vector()` in
`crates/fusion-core-py/src/core.rs` (`reading_as_vec`).

### Actions

An action is a **joint position target** for each entry in
`TaskSpec.actuators`. A PD drive tracks it inside the implicit solve; its
stiffness, damping and force limit come from the MJCF `<position kp kv
forcerange>` or URDF, or from `World.set_joint_drive`. Torque control is not
exposed. For velocity-style control, integrate on the policy side and send
targets.

Actions are clipped to `action_low`/`action_high` if set. Non-finite actions
raise an error.

### Timing

One `step` is one control tick, which runs `control_dt / physics_dt` solver
steps. Every solver step is a full implicit Newton solve, so contact and
joints stay stable at `physics_dt = 1/240`, and often at 1/120.

### Episodes and autoreset

When an env ends (`terminated` or `truncated`), `step` resets it before
returning. The returned `obs` row for that env is already the first
observation of the new episode. The final observation is in
`info["final_obs"]`, with rows matching `info["ended"]`. Call
`policy.reset(info["ended"])`; `rollout` does this for you.

### Solver failure

`info["solver_failed"]` marks envs whose nonlinear solve failed even after
the solver's own retries. Their state is not trustworthy, so the env
terminates and resets them. Count these failures; outside extreme scenes they
should be zero. `info["solver_unconverged"]` is softer: the step finished but
did not reach tolerance.

## 3. Rendering

Fusion Core has no renderer. Build one from:

- `env.geometry`, one dict per object, in object-id order:
  `{"id", "name", "kind": "fixed"|"rigid"|"solid", "shape"}`, where `shape` is
  one of:
  - `{"type": "sphere", "radius"}`
  - `{"type": "box", "half_extents": (hx, hy, hz)}`
  - `{"type": "capsule", "radius", "half_length"}`, with its axis along local **+y**
  - `{"type": "halfspace", "normal", "offset"}`, the ground or a wall: points with `normal·p ≤ offset`
  - `None` for FEM solids. Draw those from `env.world(i).solid_node_positions(id)`.
- `render(poses)`, where `poses[n, k]` is object `k`'s pose in env `n`, in the
  same order as `geometry`.

Rules for the renderer:

- Quaternions are **(x, y, z, w)**. Units are SI (m, kg, s, rad).
- The up axis depends on the scene. A bare `core.World()` is **y-up** (gravity
  `(0, -9.81, 0)`). An MJCF file brings its own `<option gravity>`, usually
  z-up. Place cameras to suit.
- Shapes come from collision geometry, not visual meshes. For realistic images,
  map `name` (the MJCF/URDF link or geom name) to your own visual assets and
  use Fusion only for the poses.

## 4. The raw solver (`fusion_vla.core`)

`FusionEnv` is roughly 200 lines over these classes. Use them directly when
the env doesn't fit:

- `World`: `add_material`, `add_ground`, `add_rigid_box/sphere/capsule`,
  `add_fixed_*`, `add_solid_box/ball` (FEM), `load_urdf[_file]`,
  `load_mjcf[_file]`, joints (`add_hinge`, `add_slider`, `add_ball_joint`,
  `add_weld`, `add_universal`, `add_cylindrical`, `add_planar`,
  `add_tendon`, `add_spring`), probes (`add_joint_probe`, `add_wrench_probe`,
  `add_tactile_probe`, `add_imu_probe`), state (`set_pose`, `set_velocity`,
  `set_joint_target`, `set_joint_drive`, `joint_coordinate`,
  `joint_velocity`, `object_position`, `object_orientation`,
  `object_velocity`, `object_angular_velocity`), and introspection
  (`object_ids`, `object_name`, `object_kind`, `object_shape`).
- `RobotModel` (returned by the loaders): name lookups via `link`, `joint`,
  `actuator`, `site`, `probe`, `geom`, `tendon`, plus `*_names()` and dict
  forms.
- `Mechanics(settings, threads)`: single-world `step(world, dt) ->
  StepReport`, `sample_probes`, `checkpoint_bytes`/`restore_checkpoint_bytes`
  (exact replay), `backward_step` (implicit-function-theorem adjoint, which
  gives gradients for joint targets, stiffnesses, initial state and gravity),
  and `strain_energy`.
- `WorldBatch(template, count, settings, threads)`: `step_all`,
  `reset_envs`, `randomize_envs`, `set_batch_joint_targets`,
  `joint_coordinates`, `joint_velocities`, `object_positions`,
  `object_poses`, `sample_observations`, `world_snapshot`, `reports`.
- `DomainRandomizer(seed, mass_scale, friction_scale, stiffness_scale,
  damping_scale, gravity_jitter)`.
- `Settings(...)`: integrator (`"tr_bdf2"` by default), tolerance, islands,
  sleeping, substepping and more. The defaults are reasonable.

`step`, `step_all`, `sample_probes`, `sample_observations` and `backward_step` release the GIL.

## 5. Guarantees and limits

- **Deterministic.** The same seed and actions give bit-identical results for
  any `threads` value. `test_determinism_across_thread_counts` checks this.
- **CPU only.** Throughput scales with cores through `threads`. The
  reach-arm example runs about 4.5k control steps per second (8 envs, 8
  substeps, on an M-series laptop). Expect much less with rich contact.
- **Primitive collision shapes.** URDF/MJCF meshes are approximated; see the
  README.
- **Not exposed yet** (in Rust, but not in the bindings): thermal and fluid
  modules, the energy ledger, and torque actuation. Adding a binding means
  editing `crates/fusion-core-py/src/core.rs` and rebuilding with
  `pip install .`.

## 6. Data format (`EpisodeRecorder`)

The recorder writes one `episode_XXXXXX.npz` per finished episode, with:

- `obs/<key>` of shape `(T, ...)`: the observation **before** each action.
- `action` of shape `(T, A)` and `reward` of shape `(T,)`.
- `terminated`, `truncated` and `solver_failed`, each of shape `(T,)`.
- `instruction`, `env_index` and `episode_index`.

Partial episodes are dropped at the end of a rollout unless
`keep_partial=True`. Convert to LeRobot, RLDS or your own format on the model
side.
