# Core Fusion Engine: specification and implementation

Updated 2026-11-20 against the [Core vision](CORE_ENGINE_VISION.md) and
[October 14 architecture](CORE_ENGINE_ARCHITECTURE.md). This concerns
`crates/core/` and the main app's **Fusion Core** backend. The older
[Fusion Solver core document](FUSION_SOLVER_CORE.md) describes the MPM-based
solver. Physics implemented only in that backend does not count here.

**Current state (V1 + Robotics/VLA Python Stack): an eight-crate,
zero-third-party-dependency Core Fusion Engine spanning rigid/deformable IPC
contact mechanics, extended robotics articulations (`Cylindrical`, `Universal`,
`Planar`, routed 1D `Tendon`s, augmented-Lagrangian joints), URDF and MJCF
(MuJoCo XML) importers & typed telemetry `Probe`s, sparse direct Cholesky
fallback, vectorized `WorldBatch` rollouts with `DomainRandomizer`,
implicit-function-theorem `backward_step` adjoints, Python bindings
(`fusion.core` in `fusion-py`), $J_2$ plasticity & cohesive damage
(`Column::DAMAGE` + v5 checkpoints), FEM heat conduction with dissipative
heating & latent-heat phase change, N-body mutual gravity, and Wendland $C^2$
WCSPH fluid/two-way FSI.**
“Implemented” below means code and focused regression coverage exist, not that
arbitrary scenes have a certified error bound. Remaining post-V1 work comprises
USD importers, AMG/GPU batched kernels, monolithic transport-mechanics
Jacobians, topological crack splitting, Eulerian/MPM fluids, electromagnetics,
combustion, atomistics, and exposing the new V1 capabilities in the desktop
editor UI.

## Architecture checklist

| Specification area | In the new engine today | Remaining work |
|---|---|---|
| Foundation | `f64` vectors/matrices, closed-form/Jacobi 3×3 eigensolver & SVD, rotations, deterministic RNG, SI parameters | Dimension-checked units, island-local origins, validation across the full scale ladder |
| Runtime | Persistent CPU worker pool, deterministic parallel islands/reductions, exact stationary-state reuse, profiling | General arenas, GPU backend, broader activity-based scheduling |
| Numerics | Sparse block matrices (reusable `Triplets` scratch), `SparseCholesky` (RCM ordering + skyline $LDL^\top$ + diagonal equilibration), `NewtonSettings::direct_blocks` with verified CG fallback (zero-start matvec skip, preconditioned steepest-descent fallback), nodal/rigid/joint-coupled/optional FEM-patch Cholesky preconditioning, dense Cholesky PD fast path, free translation coarse solve, Newton/line search, small SVD/eigen, forward AD, tetrahedral quadrature | Algebraic multigrid (AMG) preconditioning, SIMD/GPU batched linear solvers, general error-controlled time integration |
| Space | Analytic shapes/SDFs, refitted/grouped BVHs, symmetric dual-tree Barnes–Hut octree (`BarnesHut`), particle neighbor search, point/triangle and edge/edge geometry | Adaptive sparse grids, higher-order multipoles/FMM, adaptive geometric resolution |
| World | Materials (`Plasticity`, `Fracture`, phase-change `Thermal` with `effective_specific_heat`, SI presets across `library::*()`); fixed, rigid and FEM objects; explicit pair exclusions (`World::exclude_contact` propagated through `World::subset`); `Column::{TEMPERATURE, DAMAGE, PLASTIC_STRAIN}` propagation; `LinkPoint` (`World`/`Body`/`Node`), `JointDrive`, `Link::{Anchor, Spring, BallJoint, Weld, Hinge, Slider, Cylindrical, Universal, Planar, Tendon}`, `Probe` / `ProbeReading` (`JointState`, `Wrench`, `Tactile`, `Imu`), zero-dependency `World::load_urdf` and `World::load_mjcf` (MuJoCo XML `<compiler>`, `<default>`, `<inertial>`, geoms/`fromto`, multi-joint combinations, `<contact><exclude>`, `<site>`, `<tendon><spatial>`, `<actuator>`, `<sensor>`) -> `RobotModel`, & live RL actuation/proprioception APIs (`set_joint_target`, `joint_coordinate`, `joint_velocity`, `material_mut`) | USD importer, topological fracture remeshing, extensible third-party module state |
| Physics interfaces | Mechanics energy/gradient/Hessian/step-limit `Term` interface (`RigidBodies`, `Elastic`, `Joints`, `MutualGravity`, `Contact`, `Gravity`), thermal `Process` trait (`HeatConduction`), `FluidSolver` (`phys-fluids` Wendland $C^2$ WCSPH + `step_coupled` two-way FSI), differentiable `Mechanics::backward_step` (`AdjointLoss`/`AdjointGradients`), vectorized `WorldBatch`/`DomainRandomizer` RL batching (`reset_all`, `reset_envs`, `randomize_envs`, `set_batch_joint_targets`, `joint_coordinates`, `joint_velocities`), and `fusion.core` Python bindings (`crates/app/fusion-py/src/core.rs`) | General multi-module discovery/coverage/time-scale scheduler, monolithic transport-mechanics coupling |
| Orchestration | Verified independent parallel contact islands; exact fixed-point sleep/wake; settled-island sleep on fixed supports and damped world-mounted joints with instant wake on in-place joint target updates; proactive multi-rate per-island subcycling (`max_substep`, `cfl_subcycling`); force-consistent smooth startup, variable-step BDF2, outer augmented-Lagrangian passes (`augmented_lagrangian_passes`), consistent Euler contact/release, optional angular-defect subdivision and whole-interval rollback | General temporal error estimates, adaptive spatial refinement/coarsening and representation switching |
| Output/API | Main-app/headless integration, `fusion.core` Python extension submodule (GIL-released `step`, `sample_probes`, `backward_step`, `step_all`, `sample_observations`), reset, rewind, v1–v5 checkpoint reading/writing (v4 adds plastic strain, conditional v5 adds cohesive damage while preserving v1–v4 bytes) and saved-run continuation; optional accepted-interval energy/work ledger (`gravity`, `barrier`, `friction`, `damping`, `plastic`, `fracture`, `restitution`, `settling` work), `dissipative_thermal_power`, `sample_probes`, and solver-work counters | Complete bake/recording service, compressed field recordings, general achieved-error reporting, independently audited boundary work and separated numerical dissipation |

The eight core foundation and physics crates (`core-math`, `core-runtime`,
`core-numerics`, `core-space`, `core-world`, `phys-mechanics`,
`phys-thermal`, `phys-fluids`) are owned by the main workspace and have zero
third-party dependencies (`std`-only). Evidence:
[engine crates](../crates/),
[term interface](../crates/phys-mechanics/src/term.rs),
[thermal process](../crates/phys-thermal/src/lib.rs),
[fluids solver](../crates/phys-fluids/src/lib.rs),
[world model](../crates/core-world/src/world.rs),
[numerics](../crates/core-numerics/src/),
[space](../crates/core-space/src/),
[app adapter](../crates/platform/fusion-sim/src/core_engine.rs).

Retrying a failed nonlinear solve at half the interval is **convergence
recovery**, not an estimate that the time discretization is accurate. The
existing rewind/checkpoint system is useful infrastructure, not completion
of the specified resumable/refinable bake service.

## Physics checklist

| Physics goal | Status in the new core | What is missing |
|---|---|---|
| Rigid mechanics | Implemented: translation, rotation, mass/inertia, uniform gravity; affine unknowns coupled with solids; full joint zoo (`Link::{Spring, BallJoint, Weld, Hinge, Slider, Cylindrical, Universal, Planar, Tendon}`) with limits, PD `JointDrive` with smooth $C^\infty$ pseudo-Huber `max_force` saturation, outer augmented-Lagrangian multiplier updates (`augmented_lagrangian_passes`) achieving `< 1e-6 m` joint drift, live RL actuation/proprioception, URDF (`World::load_urdf`) and MJCF (`World::load_mjcf`) importers, explicit pair contact exclusion (`World::exclude_contact`), typed `Probe` sensors (`JointState`, `Wrench`, `Tactile`, `Imu`), fixed-pattern Hessian binding, joint-coupled Cholesky preconditioning, sparse direct Cholesky fallback, `WorldBatch`/`DomainRandomizer`, `Mechanics::backward_step` IFT adjoint gradients, `fusion.core` Python API, and connected-pair contact exclusion | USD importer and broader quantitative validation |
| Contact/friction/impact | Implemented: point/SDF, point/triangle, edge/edge, swept step limits (with separating-plane jump and deadlock-free minimum floor), smoothed Coulomb friction, impact law (closed-form PSD rigid curvature Hessian, multi-hold unloading split); self-contact, consistent contact stages, surface moment arms/rolling, exact affine SDF gradients, corrected rigid effective masses, discrete work accounting, tactile patch extraction (`Contact::tactile_patch`) and frictional thermal power export | Broad oblique-impact convergence, difficult mesh/scale tests, audited boundary work, exact angular conservation and rolling resistance |
| Elastic solids | Implemented: linear/quadratic tetrahedra, isotropic compressible neo-Hookean law (sign-gated negative spectral mode projection), real stiffness, deformation and strain energy | Additional constitutive laws, anisotropy/grain/layers, near-incompressibility validation, general stress probes, broader 3D/mesh-quality convergence beyond the new beam/rod checks |
| Material behavior | Implemented ($J_2$ plasticity + cohesive damage): density, elastic property constants/temperature tables (`Property::Table`), $J_2$ associative Hencky elasto-plasticity (`Column::PLASTIC_STRAIN`, `EnergyLedger::plastic_work`), cohesive tensile damage (`Material::with_fracture`, `Column::DAMAGE`, `EnergyLedger::fracture_work`, quadratic stiffness degradation, conditional v5 checkpoints), friction/restitution and SI thermal/mechanical presets | Topological crack splitting/remeshing, kinematic hardening and viscoplasticity |
| Heat | Implemented (conduction + staggered dissipative thermo-mechanics + phase change): `phys-thermal` FEM conduction on linear/quadratic tets (Backward Euler / Crank–Nicolson, Dirichlet, volumetric source, `NodalPower`, convection, linearized radiation, latent-heat `effective_specific_heat` phase change, parallel Jacobi-PCG) updating `Column::TEMPERATURE`; isotropic thermal stretch, temperature-dependent Lamé moduli, and automatic mechanical frictional/plastic/damage power wiring via `dissipative_thermal_power` -> `HeatConduction::set_nodal_power` | Monolithic thermo-mechanical Jacobian, convective fluid thermal transport, and contact thermal conductance |
| Fluids and aerodynamics | Implemented (V1 WCSPH + two-way FSI): `phys-fluids::FluidSolver` deterministic `f64` Wendland $C^2$ weakly-compressible SPH on `Pool` with Tait pressure state, XSPH smoothing, artificial viscosity, hydrostatic pressure gradient, and `FluidSolver::step_coupled` two-way buoyancy/contact/drag impulse coupling with `World` rigid bodies | Eulerian/APIC/MPM grid fluids, incompressible projection, surface tension and compressible aerodynamics |
| Combustion | Not implemented | Species/reactions, fuel/oxygen/heat coupling, structural weakening |
| Long-range gravity | Implemented (N-body rigid/deformable-node mutual gravity): `core-space::BarnesHut` symmetric dual-tree octree + `phys-mechanics::MutualGravity` ($O(N \log N)$ COM + top-down adjoint push, exact `theta = 0` mode, Plummer softening, gravitational energy ledger) | Symplectic orbital integrators, higher-order multipoles/FMM, volume-integrated continuum self-gravity |
| Electromagnetics | Not implemented | Electric/magnetic fields, charges/currents, induction, forces and heating |
| Classical atoms | Not implemented | MD representations, force fields, bonds/reactive chemistry and atomistic-to-continuum coupling |
| Representation ladder | Partial: fixed/rigid/FEM and N-body rigid bodies | Reduced solids, bonded particles, new-core MPM, field grids and switching/coupling between representations |

Evidence: [mechanics step](../crates/phys-mechanics/src/step.rs),
[contact](../crates/phys-mechanics/src/contact.rs),
[joints](../crates/phys-mechanics/src/joints.rs),
[mutual gravity](../crates/phys-mechanics/src/gravity.rs),
[elastic law](../crates/phys-mechanics/src/law.rs),
[elastic discretization](../crates/phys-mechanics/src/elastic.rs),
[heat conduction](../crates/phys-thermal/src/conduction.rs),
[material data](../crates/core-world/src/material.rs).
Quantum mechanics and relativity remain outside the specification.

## What the main app can author

The editor runs rigid boxes/spheres and quadratic FEM cuboids/voxelized
spheres, with fixed supports. The lower-level core additionally has analytic
capsules/half-spaces, linear elements, nodal anchors, mechanical
joints/springs/welds, $J_2$ plasticity, finite-element heat conduction
(`phys-thermal`) and N-body mutual gravity. Those are not all exposed as
main-app authoring capabilities. Arbitrary imported FEM meshes, cloth,
fluids, joints, fracture, heat, gravity clusters and air are not yet
supported by the Core adapter. Unsupported scene physics is rejected
explicitly. Runtime numerical overrides are not serialized as arbitrary
project settings. See the [integration guide](FUSION_CORE_ENGINE.md) for exact
scene limits.

## Core fixes, speedups, and classical-physics expansion, November 19

### Bug fixes

- **Step-scaled rigid orthogonality and sleep/wake integrity:** Scaled rigid
  orthogonality constraint stiffness by interval length (`ORTHOGONALITY_DEFAULT
  = 3e2` in [rigid bodies](../crates/phys-mechanics/src/rigid.rs)), added
  exact `Shape::HalfSpace` AABB culling so arcs over ground planes do not falsely
  intersect the infinite half-space, added rotation-edit wake checks in sleep
  matching, synchronized stored accelerations, `earlier` histories and
  `unloading` budgets across settled sleep/wake transitions, and divided the
  `unloading` restitution budget across simultaneous hold contacts instead of
  applying full unloading per contact row
  ([islands](../crates/phys-mechanics/src/islands.rs),
  [settling tests](../crates/phys-mechanics/tests/settling.rs)).
- **Closed-form rigid rotational barrier Hessian and CG fallback:** Added the
  closed-form PSD rotational curvature Hessian block for `Pair::PointBody` on
  `Attach::Rigid` in [contact](../crates/phys-mechanics/src/contact.rs),
  and wired a preconditioned steepest-descent fallback (`fallback = true`) when
  verified CG fails to converge in
  [Newton](../crates/core-numerics/src/newton.rs).
- **Conservative advancement zero-step deadlock fix:** Fixed `Contact::step_limit`
  (`advance_edges`, `advance_signed`, `advance`) returning `t = 0.0` before any
  positive motion when the first advancement fraction fell below `1e-6 * limit`,
  and added a 4-corner separating-plane jump and `stop_margin` in
  `advance_edges`.

### Compute improvements

- **Closed-form 3×3 spectral path and sign-gated negative modes:** Added
  `Mat3::symmetric_eigen_fast`, `symmetric_eigen_fast_or`, and `svd_fast` in
  [Mat3](../crates/core-math/src/mat3.rs) plus `NeoHookean::stress_and_factors`
  and `for_each_negative_mode` in
  [elastic law](../crates/phys-mechanics/src/law.rs), skipping positive
  eigenmode outer products during PSD Hessian assembly while preserving bit-exact
  element summation order in
  [elastic](../crates/phys-mechanics/src/elastic.rs).
- **Linear-algebra fast paths:** Added a Cholesky positive-definite fast path in
  `dense::project_positive`
  ([dense](../crates/core-numerics/src/dense.rs)), skipped the initial
  `A·x` product when CG starts from zero
  ([cg](../crates/core-numerics/src/cg.rs)), and kept sparse triplet
  permutation scratch buffers reusable across Newton solves
  ([block](../crates/core-numerics/src/block.rs)).
- **Measured `--release -j 2` gate (offline):**
  - `law_cost`: `modes` `622.6 ns` → `for_each_negative_mode` `172.0 ns`
    (3.62×); full 55-entry element Hessian block build `1066.3 ns` →
    negative-mode-only `442.4 ns` (2.41×); `symmetric eigen (FᵀF)` `259.2 ns` vs
    `svd` `345.8 ns`.
  - `pile` (8 threads, 120 steps): `1428` solves, `12257` Newton iterations,
    `137232` CG iterations, `46` splits, `0` failed steps, `1109 ms`.
  - `accuracy_efficiency`: 8-copy `parallel` run `0.2917 s` (vs `0.5318 s`
    single-batch); 128-copy `sleep` `0.0163 s` (vs `0.0341 s`); `60 Hz`
    vibration energy retention `96.29%`.
  - `mechanics_accuracy`: fine cantilever sag error `0.1237%`, fine
    longitudinal frequency error `0.3353%`, `0` failed steps.
  - `rolling_accuracy`: 120 Hz slide velocity error `5.736e-5 m/s` (energy ratio
    `0.71421` vs `5/7`); resolved oblique impacts (`1/240`–`1/960 s`) within
    `1.026e-3 m/s` velocity error and `8.66e-3 rad/s` spin error.

### Classical-physics expansion

- **Mechanical joints, springs, welds, hinges, sliders, limits, and RL drives:**
  Added `LinkPoint::{World, Body, Node}`, `JointDrive` (PD actuator with
  `target`, `target_velocity`, `stiffness`, `damping`, and smooth $C^\infty$
  pseudo-Huber `max_force` saturation), `Link::{Spring, BallJoint, Weld, Hinge,
  Slider}` (`World::add_spring`, `add_ball_joint`, `add_weld`, `add_hinge`,
  `add_slider`), live RL action/proprioception APIs (`World::set_joint_target`,
  `set_joint_drive`, `set_spring_rest_length`, `set_weld_rotation`,
  `joint_coordinate`, `joint_velocity`, `material_mut`), extended `Link::{Cylindrical,
  Universal, Planar, Tendon}`, `Probe`/`ProbeReading` (`JointState`, `Wrench`,
  `Tactile`, `Imu`), zero-dependency `World::load_urdf` -> `RobotModel`, and
  [joints term](../crates/phys-mechanics/src/joints.rs) with pre-bound
  fixed-pattern `BlockMatrix` Hessian assembly, live parameter synchronization
  (`Joints::sync_world`), outer augmented-Lagrangian multiplier updates
  (`Settings::augmented_lagrangian_passes`), connected-pair contact exclusion in
  `Contact`, joint-coupled dense Cholesky preconditioning in `Mechanics::ensure_terms`,
  world-mounted joint settling + instant wake on target updates, and proactive
  multi-rate island subcycling (`Settings::max_substep`,
  `Settings::cfl_subcycling`, `StepReport::max_island_substeps`)
  ([joint tests](../crates/phys-mechanics/tests/joints.rs),
  [island tests](../crates/phys-mechanics/tests/islands.rs): FD gradient
  `< 1e-7` across all joint/limit/drive/tendon types, AL joint drift `< 1e-6 m`,
  virtual `Probe` suite, multi-DOF robot arm + parallel-jaw gripper grasping with
  sleep/wake and 1/4 thread determinism, and multi-rate CFL subcycling).
- **Sparse direct Cholesky & differentiable VLA/RL batching:** Added zero-dependency
  [`SparseCholesky`](../crates/core-numerics/src/direct.rs) (RCM fill-reducing
  ordering, skyline $LDL^\top$, diagonal equilibration) integrated into `Newton` via
  `NewtonSettings::direct_blocks` with automatic verified-CG fallback, plus
  implicit-function-theorem [`Mechanics::backward_step`](../crates/phys-mechanics/src/adjoint.rs)
  (`AdjointLoss`, `AdjointGradients` with exact unclipped rigid Hessians) and
  [`WorldBatch` / `DomainRandomizer`](../crates/phys-mechanics/src/batch.rs)
  ([numerics tests](../crates/core-numerics/tests/numerics.rs),
  [robotics VLA tests](../crates/phys-mechanics/tests/robotics_vla.rs):
  URDF arm + tendon + probe rollout, thread-deterministic domain-randomized
  `WorldBatch`, and `< 1e-6` IFT adjoint agreement vs central finite differences).
- **$J_2$ Hencky elasto-plasticity & cohesive fracture:** Added `Material::with_plasticity`,
  `Material::with_fracture` (`Fracture`), `Column::{PLASTIC_STRAIN, DAMAGE}`,
  `NeoHookean::hencky_j2_return`, element quadrature damage softening
  $g(d) = (1-d)^2(1-k_{\min}) + k_{\min}$, `EnergyLedger::{plastic_work, fracture_work}`
  and **version 4/5 checkpoints** preserving v1–v4 byte parity when damage is inactive
  ([plasticity tests](../crates/phys-mechanics/tests/plasticity.rs):
  linear & quadratic tets, post-yield `< 1e-7` FD gradient, cohesive tensile damage
  softening + v5 replay across 1/4 threads). Topological crack splitting and
  kinematic hardening remain open.
- **Heat conduction, dissipative heating, phase change & thermo-mechanics:** Added
  [`phys-thermal`](../crates/phys-thermal/src/lib.rs) (`Process` trait
  and `HeatConduction` FEM solver over linear & quadratic tets with Backward Euler
  and Crank–Nicolson, Dirichlet, volumetric source, `NodalPower`, convection,
  linearized Stefan–Boltzmann radiation, `Thermal::with_phase_change` /
  `effective_specific_heat` across melting windows, parallel Jacobi-PCG, and SI
  thermal properties on all nine `library::*()` materials) plus coupled thermal
  stretch, temperature-dependent `Property::Table` Lamé moduli, and automatic
  mechanical frictional/plastic/damage heating via `Mechanics::dissipative_thermal_power`
  -> `HeatConduction::set_nodal_power`
  ([thermal tests](../crates/phys-thermal/tests/thermal.rs),
  [thermo-mechanics tests](../crates/phys-mechanics/tests/thermo_mechanics.rs)).
  Monolithic Jacobian coupling remains open.
- **WCSPH fluids & two-way FSI (`phys-fluids`):** Added the `std`-only
  [`phys-fluids`](../crates/phys-fluids/src/lib.rs) crate (`FluidSolver`)
  implementing deterministic `f64` Wendland $C^2$ weakly-compressible SPH on `Pool`
  with Tait pressure equation, artificial viscosity, XSPH smoothing, domain box
  boundaries, and `FluidSolver::step_coupled` two-way rigid buoyancy/contact/drag
  transfer ([fluid tests](../crates/phys-fluids/tests/fluids.rs): hydrostatic
  pressure settling + 1/4 thread determinism and Archimedes two-way buoyancy).
- **Long-range N-body gravity:** Added symmetric dual-tree octree
  [`BarnesHut`](../crates/core-space/src/tree.rs) and
  [`MutualGravity`](../crates/phys-mechanics/src/gravity.rs) (`Settings::mutual_gravity`,
  `MutualGravitySettings` with `constant`, `softening`, `theta`) using live COM
  evaluation + top-down adjoint push (`theta > 0`) and exact all-pairs (`theta = 0`),
  island merging, gravitational energy accounting and FD gradient `< 1e-7`
  ([tree tests](../crates/core-space/tests/tree.rs),
  [orbital tests](../crates/phys-mechanics/tests/orbital.rs): circular and
  eccentric 0.5 Kepler orbits, 200→400 TR-BDF2 refinement convergence, 100-body
  momentum conservation and deterministic 1/4-thread replay). Symplectic orbital
  integrators and volume-integrated continuum self-gravity remain open.

## Stacked and piled boxes fixed November 10

Boxes stacked corner on corner, and the lab's 64-box pile, failed every step:
not a crash, but Newton stalling until the step's retry budget ran out, so
time stopped advancing. The cause was edge-edge contact between rigid bodies.
IPC's parallel-edge mollifier was applied only to a solid's own mesh edges,
so two rigid edges lying exactly parallel (every edge of one box's base over
one of the box below) kept the closest-feature switch of `closest_segments`
where Newton converges. The gradient alternated between two values, the line
search accepted only 2⁻¹¹ of each step, and the 0.2 µm step tolerance was
never reached. The mollifier now covers every edge-edge pair: mesh, rigid and
fixed. Where edges are parallel the pair fades out and corner/point barriers
carry the contact, as in the IPC formulation.

Lab bench, before → after: the 5-box stack failed at t = 0 → drifts 0.6 mm in
3 s (the Standard engine drifts 2.5 mm); the 64-box pile failed at 0.3 s →
runs 0–8 s with no failed steps or retries, at 2011 ms per simulated second
falling and 1887 ms at rest. Every other lab row is unchanged within timing
noise. A resting pile then still cost nearly as much as a falling one; the
next section fixes that.
Regression: `boxes_stacked_corner_on_corner_settle_and_land` in
[edge contact tests](../crates/phys-mechanics/tests/edge_contact.rs).

## Settled islands sleep, November 11

Sleep used to reuse only bitwise-stationary equilibria, and a pile under
friction never becomes one, so a resting pile cost as much as a falling one.
`Settings::settle_time` (default 0.1 s; `None` turns it off) now puts an
island to sleep when it rests on a fixed support and every point of it moved
less than the contact precision per settle time over the last step (and less
than the precision, however long the step). Its velocities and acceleration
become exactly zero, its integrator restarts with no memory, and it costs no
solve until a moving body's swept bounds reach it, gravity changes or its
velocity is edited. The decision reads only checkpointed state, so replay is
bitwise exact through sleep and wake, on any thread count, with no change to
the checkpoint format. With energy accounting on, the kinetic energy removed
is reported as `EnergyLedger::settling_work`.

The test is **displacement over the last step, not the stored velocity**.
Resting boxes in contact stay on the trapezoidal stage of TR-BDF2 (no impact
is detected, so no Euler restart), and there they carried a steady velocity
of about 4 mm/s upward and 1.1 m/s² of stored acceleration while their
positions moved 0.0007 mm/s. That phantom velocity is an open integrator
defect: it shows as kinetic energy in a resting scene until the island
settles, and it makes Newton work each step. Settling clears it.

Lab bench, before → after (ms of computing per simulated second): 64-box pile
falling 2011 → 632, at rest 1887 → 1 (Standard engine: 24); four jelly blocks
at rest 815 → 1 (Standard: 394). Every physics check is unchanged. Floating
islands never settle, which keeps the free-vibration accuracy gates intact.
Regressions: [settling tests](../crates/phys-mechanics/tests/settling.rs).

## The step solve costs a quarter of what it did, November 12

Seven changes to how a step is solved, none to what it solves. Every one
is bit-identical: the checkpoint bytes of every step of every lab physics
check and cost scene match the commit before (on eight threads; five of
the scenes were also compared on one and on four), and the lab's physics
table gives the same verdict and value on every row.

Lab bench, ms of computing per simulated second, every commit's binary
timed in one interleaved session (best of 5 to 7):

| change | four jelly blocks falling | 64-box pile falling | porcelain block landing |
|---|---|---|---|
| before | 4430 | 671 | 276 |
| contacts of a state found once, not once per question | 3355 | 621 | 227 |
| pairs near contact listed between searches | 2609 | 617 | 204 |
| the step limit looks at listed pairs only | 2277 | 604 | 192 |
| contact matrix summed without sorting or copying | 2268 | 317 | 191 |
| the pool lets a task hand out work of its own | 1751 | 315 | 192 |
| a solid's derivative passes cut 8 elements to a chunk | 1033 | 318 | 133 |
| matrix products cut 32 rows to a chunk | 945 | 313 | 94 |

The Standard engine's figures for the same scenes are 424–576, 390 and
522–1015: the pile and the porcelain block are now cheaper on the core,
the jelly blocks still about twice the price.

**Contact.** A solve asks for the pairs in contact at one state several
times (energy, derivatives, impact check, friction), and at states a small
move apart (line-search trials). `Contact::active` keeps what it found at
the unknowns it was last asked about, and [`contact/near.rs`](../crates/phys-mechanics/src/contact/near.rs)
lists, at a reference state, every pair within a margin of contact with the
room it has. A gap is a distance, so it shrinks by no more than its sides
move towards each other; that is measured per object as a shared
translation and a spread about it. While nothing has moved further than
the margin allows, only listed pairs with too little room are evaluated,
with the same box test and in the same order as the full search, which is
what makes the result identical. The step limit uses the same lists when
the step itself is within what the unlisted pairs' room allows, and sweeps
all of space as before when it is not (about a quarter of the jelly
scene's step limits, typically a solve's first). The margin adapts
between 4 and 256 precisions; it changes cost, never results, so it is
not checkpointed. A jelly block's contact fell from 56% of its solve to
about 20%.

**Contact matrix.** A settled pile writes 112,000 block entries per Newton
iteration for a matrix of 7,000 blocks. `Triplets` now takes each chunk's
list whole (`append`), sums duplicates in one pass in insertion order
through a table (`assemble_into`), and keeps all its memory; `Contact`
keeps its shares. The stage's end asks contact for its gradient alone.

**Threads.** The pool held one open batch, so a pass handed out inside an
island ran on that island's thread: four jelly blocks on eight threads
used two. It now holds up to 64 open batches; idle threads help with any,
and the outermost caller helps while it waits. Passes whose chunks write
only their own places (element blocks, matrix rows, matrix-vector
products) are cut fine enough for a hundred-node body to share: none of
them is a sum, so their chunk size is not part of any result. Energies and
dot products keep their chunks.

**Tried, not kept.** Making only the negative modes' eigenmatrices of the
elastic Hessian (18% off the eigensystem in isolation, nothing measurable
in a scene). Returning early from an edge pair's conservative-advancement
march when a separating direction proves it will arrive (exact, but the
marches that cost are the ones that do limit the step). Narrower contact
margins (every cap below 64 precisions was slower).

**What is left.** On the jelly scene the solve is now about 40% elastic
derivatives, 35% conjugate gradients and vector work, 20% contact. The
elastic derivatives are, per quadrature point, about a third each the
dense tangent, the eigensystem that finds negative modes and the
corrections that remove them: a closed-form 3×3 eigensolver and a folded
tangent would roughly halve them, but both change the Hessian's rounding,
so results would match to solver tolerance, no longer to the bit. Conjugate gradients take ten iterations per Newton iteration under
block Jacobi; a better preconditioner is the same trade. On the pile, 55%
of the solve is the step limit, where each resting edge pair's march
takes some 160 distance evaluations to find a limit that is genuinely
small: the lever there is the advancement scheme itself. Beyond those,
the count of solves (730 for 60 steps of four blocks) is set by the
integrator and the contact restarts, which is accuracy, not waste.

Regressions added: batches within batches, more open batches than slots,
nested reductions on any thread count, several callers at once and a panic
in a nested batch ([pool tests](../crates/core-runtime/tests/pool.rs));
every block of a triplet-built matrix is its entries' sum in insertion
order by either route ([numerics tests](../crates/core-numerics/tests/numerics.rs)).

## Independent islands added October 30

The core now solves independent contact groups separately. Each has its own
Newton/CG convergence, automatic contact precision, integrator history and
smaller-step recovery. Contact in one group no longer changes the integration
rule of a distant body. Sharing a fixed floor does not connect groups. Meshes
stay in the shared world; compact layouts and cached local solve storage avoid
copying geometry or allocating a second global derivative system.

Planning uses swept, rotation-safe bounds. Bounds from all evaluated solver
configurations verify the plan before acceptance. Unexpected elastic motion
that reaches another group or a previously omitted support rolls back the
entire interval and repeats with the missing coupling. Groups can split again
on later intervals. That milestone kept the checkpoint encoding unchanged,
with exact replay through merging, splitting, contact and local BDF2 subdivision.

Eight new checks establish these behaviors, including bitwise identical
isolated motion beside a tiny contact object for all four integrators and
one/four threads. The [island guide](CORE_ISLANDS.md) describes the algorithm,
work counters, cost measurements and limits. In the 24-body benchmark, three
paired runs measured median stepping time 38.40 → 16.40 s (2.34×), with 15.2%
fewer size-weighted CG block-iterations and zero tracked isolated-motion error.
These are fixture-specific measurements, with timing ranges in the record. This is conservative object-level
partitioning. Groups initially advanced serially; November 1 added parallel
dispatch and exact fixed-point sleeping. Multirate integration remains open.

## Mechanics accuracy baseline added October 30

The [new engine-only validation gate](CORE_MECHANICS_VALIDATION.md) compares
static beam deflection, bending/longitudinal vibration, moving self-contact
and known gravity impulse against independent physical references. It tracks
accuracy, total physical energy, attempted solve work and elapsed time at
multiple mesh/time resolutions. Eight checks were added, including tiny shear
energy and near-parallel contact derivatives/CCD. The stricter existing
self-contact conservation/replay check now passes too.

Contact/release stages now use one Euler rule across each coupled island,
replacing per-node rule changes and post-solve velocity overwrites. Smooth
contact-free motion retains the selected integrator. A repeated contact solve
is exposed through `StepReport::contact_restarts` and counted in work totals.
Parallel mesh self-edges use a differentiable mollifier, with matching force
and friction scaling; their geometric CCD remains active. Near-rest elastic
energy uses a cancellation-resistant expression of the same physical law.

In three paired runs, the moving-contact fixture at dt=1/120 s reduced max
momentum error from 0.007295 to 0.000237 kg m/s. At dt=1/240 s the old engine
ended with 240% of initial physical energy; the new result is 43%, within the
non-growth gate. The latter is not claimed as an exact rebound-energy target.
The fine beam's deflection differs by 0.124% from its shear-corrected reference,
and rod frequency by 0.335% at the finest tested step. The accuracy milestone
passed 97 core-only tests plus 20 integration checks before the eight island
checks were added; affected targets passed Clippy with warnings denied. The [record](benchmarks/core-mechanics-20261030.txt) has raw accuracy,
work counts, timing, source hashes and the before/after source patch.

This milestone is not a complete accuracy certificate. Euler contact stages
are dissipative within each contact island; angular
momentum is bounded rather than exact. November 1 replaces the Euler smooth
startup and adds discrete work accounting and optional angular refinement. The
[validation guide](CORE_MECHANICS_VALIDATION.md) states these limits.

## Accuracy and compute improvements added November 1

The [accuracy/compute guide](CORE_ACCURACY_20261101.md) describes force-consistent
second-order startup, stable variable-step BDF2 history, true-residual CG checks,
a free-island translation coarse solve, finite rigid angular impulses and
corrected restitution effective masses. Version 2 checkpoints preserve force
and interval metadata while retaining a version 1 reader.

Optional accepted-interval energy accounting separates gravity, barrier,
friction and restitution work from a signed unresolved balance. Optional
angular-defect subdivision rejects excessive free-island drift transactionally;
it is not a general temporal error estimate or exact angular conservation.
Parallel workers share mesh/material storage, commit only after swept
verification, and reuse only exact stationary discrete equilibria. FEM dense
patches are tunable; the default remains one nodal block pending workload-specific
benefit. See the [paired record](benchmarks/core-accuracy-20261101.txt) for
accuracy and compute tradeoffs rather than treating iteration count as speed.
At 60 Hz, the rod retains 96.29% of initial energy versus 75.70% with the old
startup. Optional angular refinement cuts the tested impact's drift 3.53× at
7.90× elapsed time. Eight independent FEM bodies run 2.18× faster with parallel
dispatch and identical checkpoint bytes. These are measured fixtures, not
general accuracy or performance guarantees.

The [surface-contact follow-up](CORE_ROLLING_CONTACT.md) fixes centre/centreline
friction on rounded primitives and normalization of affine SDF force gradients.
Spheres now transition from sliding into rolling, with known velocity/spin and
energy references; capsules use their own axial inertia. At 120 Hz, launch
velocity error falls from 0.7143 to 0.0000574 m/s. The 2026-11-04 follow-up
fixes spurious mid-air friction on separating stages, post-impact acceleration
extrapolation, newly-sliding radial secant damping, post-impact solver-potential
unloading and separating-plane bisection in conservative advancement: all three
oblique-impact steps (`1/240`, `1/480` and `1/960 s` for `e=0` and `e=0.5`) now
pass the impulse and angular-drift gate (`<2e-3 m/s` velocity error, `<1e-2
rad/s` spin error) without discarded friction refreshes and with zero Newton/CG
iteration regression on sliding/rolling contacts, though fixed-step collision-phase
alignment between `1/480` and `1/960 s` remains open.

## October 28 mechanics additions

- **Deformable self-contact.** Non-incident surface vertex/triangle and
  edge/edge pairs now interact inside a single solid. Only shared-vertex
  incidences are excluded; an entire object is no longer excluded from its
  own collision search. Spatial searches reuse BVHs and preserve sorted,
  deterministic pair order. Rigid self pairs remain excluded.
- **A true-distance self barrier and swept edge limits.** A self gap uses
  `s = distance`, rather than the inter-object tolerance offset. Initially
  separated self surfaces therefore cannot pass through the barrier's
  tolerance band. Adjacent triangles are not false contacts. Self-contact
  defaults on in `Settings`, including the main app's default Core solver.
- **Degenerate triangle geometry.** Exactly collapsed triangles reduce to
  their segments, returning finite closest points and barycentric weights
  instead of dividing by zero when an edge collapses.
- **Cheaper impact sweeps.** Each contact now reads only its attached
  velocities. Previously each contact copied all scene velocities on each
  impulse sweep (up to 20 sweeps). The frozen pre-impact snapshot is still
  taken once per stage. The equations and pair order are unchanged. The
  [paired comparison](benchmarks/core-self-impact-20261028.txt) has identical
  final checkpoint bytes and solve work; the small timing change is within
  run variation, so no overall speedup is claimed.

Regression coverage includes a connected U whose arms close, disconnected
parts of one mesh, linear/quadratic elements, a self-contact-disabled control
that actually crosses, energy/gradient agreement, swept interior edge
crossings, common-motion invariance, checkpoint replay across 1/4 threads,
bounded energy/momentum checks and coincident-surface rejection without
advancing a prepared solver. Source:
[self-contact tests](../crates/phys-mechanics/tests/self_contact.rs),
[CCD tests](../crates/phys-mechanics/src/contact.rs),
[geometry tests](../crates/core-space/tests/space.rs).

Self-contact requires initially valid, separated non-incident surfaces.
Coincident non-incident self surfaces are singular; they are not repaired
by stepping. This is not a general initial-mesh intersection repair/detection
service. Quadratic FEM contact uses piecewise triangular boundary facets,
not exact curved faces. The non-incident primitive approach follows the
[IPC formulation](https://github.com/ipc-sim/IPC/wiki).

## Accuracy work that remains

The historical lab architecture's 12/14 lab checks are not the main-workspace
test count or an accuracy certificate. The main workspace now has analytic
beam/rod gates with mesh/time refinement, bounded energy and momentum checks,
collision/derivative tests, and exact replay. This is an initial quantitative
baseline, not validation of every elastic material, mesh or impact regime.

The October 28 soft-impact test exposed ~0.0074 kg m/s linear and ~0.126 kg m²/s
angular drift from approximately zero. October 30 contact handling addresses
the mixed-rule mechanism; the strict regression now requires linear drift
<1e-5 kg m/s and angular drift <0.003 kg m²/s. More general moving-contact
checks use explicit scale-based tolerances. Force-consistent startup and a
discrete work ledger now exist. Exact angular conservation, independently
audited boundary work, separated integration loss, full 3D refinement and
independent impact-time/contact-precision studies remain unfinished. See
[the validation baseline](CORE_MECHANICS_VALIDATION.md) for actual criteria.

## Post-V1 implementation order

1. **Broaden scene import & GPU/AMG scaling:** USD importer alongside existing
   `World::load_urdf` and `World::load_mjcf`, algebraic multigrid (AMG)
   preconditioning for very large dense contact piles, and SIMD/GPU kernel
   acceleration for massive parallel `WorldBatch` RL throughput. Keep
   accuracy/cost gates mandatory.
2. **Deeper continuum & fracture topology:** Topological element splitting on
   cohesive cracks, kinematic hardening / viscoplastic flow, broad oblique-impact
   and 3D mesh validation, audited boundary work, exact angular conservation /
   symplectic options, and general temporal error estimation.
3. **Monolithic multiphysics & Eulerian/MPM transport:** Monolithic thermo-mechanical
   Jacobians, convective thermal transfer, Eulerian/MPM incompressible fluids and
   aerodynamic wakes, followed by combustion, electromagnetics, and classical atoms.
4. **App authoring exposure:** Expose the validated V1 joints/tendons, URDF/MJCF
   import, virtual probes, plasticity/damage, thermal coupling, and
   `phys-fluids` FSI inside `fusion-sim` / project serialization and editor UI.

This follows the architecture's dependency order. Training-data work is not a
substitute for these engine capabilities and was not expanded in this pass.

## Verification record

**188 Rust tests + 11 Python contract tests pass cleanly after the MJCF, vectorized `WorldBatch`, and `fusion.core` additions** (zero failures, zero ignored):
- **Tested core crates (`188` passed):** `core-math` (11), `core-numerics` (20),
  `core-runtime` (15), `core-space` (13), `core-world` (12, including `mjcf` 3),
  `phys-mechanics` (117, including `joints` 11, `plasticity` 4, `robotics_vla` 6,
  `thermo_mechanics` 4, `orbital` 4). Earlier V1 checks (`phys-thermal` 6,
  `phys-fluids` 2, `fusion-sim`, `fusion-project`, `fusion-app`, `fusion-samples`)
  were untouched in this pass.
- **Python bindings (`11` passed via Python 3.13.3):** `test_bindings.py` (6) and
  `test_core_robotics.py` (5: URDF + checkpoints, MJCF arm + FEM solid,
  vectorized `WorldBatch` + `DomainRandomizer` 1 vs 4 thread determinism,
  `< 1e-5` IFT adjoint gradients vs central finite differences, error validation).

Clippy passes with `-D warnings` across all targets of all 7 updated/tested packages
(`core-math`, `core-runtime`, `core-numerics`, `core-space`, `core-world`,
`phys-mechanics`, `fusion-py`), and `cargo fmt --all -- --check` passes across
the entire workspace. The pre-existing third-party `block 0.1.6`
future-compatibility advisory is unchanged.

Reproduce the focused gate:

```sh
cargo fmt --all -- --check
cargo clippy --offline -p core-math -p core-runtime -p core-numerics -p core-space \
  -p core-world -p phys-mechanics -p fusion-py --all-targets -- -D warnings
cargo test --offline -p core-math -p core-runtime -p core-numerics -p core-space \
  -p core-world -p phys-mechanics
cargo build --offline -p fusion-py && mkdir -p target/py-test && cp target/debug/libfusion.dylib target/py-test/fusion.so
PYTHONPATH=target/py-test python3 crates/app/fusion-py/tests/test_bindings.py
PYTHONPATH=target/py-test python3 crates/app/fusion-py/tests/test_core_robotics.py
```
