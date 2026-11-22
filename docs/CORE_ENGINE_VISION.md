# Fusion Core — Vision and Goals (V1 Release)

> **Updated 2026-11-19 (V1 Release).** **Fusion Core** (`crates/core/`) delivers a deterministic, differentiable `f64` physics engine for **Robotics, Embodied AI, VLA & RL Training, and High-Accuracy Contact/Deformable Manipulation**, alongside classical multiphysics across layers L0–L5c.
> See [the implementation audit](CORE_ENGINE_STATUS.md) for verified test evidence, and [FUSION_ARCHITECTURE.latex](FUSION_ARCHITECTURE.latex) for the mathematical reference.

*The physics engine itself: the code and the mathematics, not the UI app. The app, RL trainers, data generators, and robotics pipelines are clients of it.*

## In One Sentence

**One deterministic, differentiable `f64` physics core for robotics, VLA/RL training, and classical multiphysics — simulating articulated robots, contact-rich dexterous manipulation, deformable/plastic/fracturing solids, heat, fluids, and long-range fields on one variational substrate without sacrificing physical truth.**

## Priorities, in Order

1. **Physical Truth & Manipulation Fidelity.** Real SI units, true material stiffnesses (steel and soft elastomers in one solve), barrier non-interpenetration, exact surface rolling friction, and analytical contact/joint derivatives. Never fake contact compliance or drop constraints to hit a frame budget.
2. **RL Throughput & Differentiability by Design.** Cost follows activity via verified independent contact islands, settled support/joint sleep with instant wake on policy target edits, multi-rate subcycling, fixed-pattern sparse reuse, vectorized batched rollouts (`WorldBatch`), and implicit-function adjoint gradients (`Mechanics::backward_step`).

## Capability Pillars & Physical Layers (Fusion Core V1)

All eight `crates/core/` crates (`L0–L5c`) and the four V1 capability pillars are now **Implemented**:

| Layer / Pillar | Physics & Robotics Scope | Status |
|---|---|---|
| **Core Mechanics** | 12-DOF affine rigid bodies, P1/P2 neo-Hookean solids, IPC barrier contact & universal edge mollifier, Coulomb friction, restitution impacts, uniform gravity | **Implemented** (`phys-mechanics`) |
| **Articulated Joints & RL Control** | Complete joint zoo (`Anchor`, `Spring`, `BallJoint`, `Weld`, `Hinge`, `Slider`, `Cylindrical`, `Universal`, `Planar`), routed 1D `Tendon`, limits, PD `JointDrive` with $C^\infty$ pseudo-Huber saturation, live `set_joint_target` / `joint_coordinate` / `joint_velocity`, joint-coupled Cholesky, connected-pair contact exclusion | **Implemented** (`core-world`, `phys-mechanics`) |
| **Scheduling & Sleep** | Verified parallel islands, settled support & damped world-joint sleep, instant wake on link/target edits, proactive multi-rate `max_substep` / `cfl_subcycling` | **Implemented** (`phys-mechanics`) |
| **Materials & Plasticity** | SI material presets (`library::*()`), temperature tables (`Property::Table`), Hencky $J_2$ associative elasto-plasticity (`Column::PLASTIC_STRAIN`), v1–v5 checkpoints | **Implemented** (`core-world`, `phys-mechanics`) |
| **Heat & Thermo-Mechanics** | P1/P2 FEM heat conduction (`phys-thermal`), Backward Euler / Crank–Nicolson, Dirichlet/source/convection/radiation/`NodalPower` BCs, staggered thermal stretch & softening | **Implemented** (`phys-thermal`, `phys-mechanics`) |
| **Long-Range Gravity** | $O(N \log N)$ symmetric dual-tree `BarnesHut` octree + Plummer-softened `MutualGravity` | **Implemented** (`core-space`, `phys-mechanics`) |
| **Pillar 1: Robotics & RL Stack** | Pure-Rust URDF importer (`World::load_urdf` / `RobotModel`), full joint/tendon zoo, virtual `Probe::{JointState, Wrench, Tactile, Imu}` sensors (`Mechanics::sample_probes`), vectorized `WorldBatch` parallel envs, deterministic `DomainRandomizer` | **Implemented** (`core-world`, `phys-mechanics`) |
| **Pillar 2: Hardened Solvers** | Outer Augmented-Lagrangian passes (`Settings::augmented_lagrangian_passes`) over bilateral joints, zero-dependency `SparseCholesky` $LDL^\top$ solver (`NewtonSettings::direct_blocks` + automatic CG fallback for `<= 256` blocks) | **Implemented** (`core-numerics`, `phys-mechanics`) |
| **Pillar 4: Differentiable Physics** | Implicit Function Theorem (IFT) backward adjoint pass (`Mechanics::backward_step`, `AdjointLoss`, `AdjointGradients`) reusing exact converged stage Hessians for initial positions/velocities, gravity, joint targets, and joint stiffnesses | **Implemented** (`phys-mechanics`) |
| **Pillar 3: Continuum Multiphysics** | Cohesive fracture (`Material::with_fracture`, `Column::DAMAGE`, `EnergyLedger::fracture_work`), automatic frictional/plastic/damage heating (`dissipative_thermal_power` -> `HeatConduction::set_nodal_power`), latent-heat phase change (`Thermal::with_phase_change`), deterministic `f64` WCSPH fluids & two-way FSI (`phys-fluids`) | **Implemented** (`core-world`, `phys-mechanics`, `phys-thermal`, `phys-fluids`) |
| **Post-V1 / Excluded Scope** | MJCF/USD importers, SIMD/GPU batched backend, AMG preconditioner, combustion, electromagnetics, classical atomistics (post-V1); quantum/relativity (out of scope) | Planned / Excluded |

## Scale: One Engine, Appropriate Representations

No single discretization fits nanometre crystals and warehouse manipulation alike. Matter selects its representation by scale and couples across boundaries:

| Scale / Role | Representation | Examples in Robotics & Classical Physics |
|---|---|---|
| Whole links & grippers | 12-DOF affine rigid bodies | Robot links, parallel jaws, rigid tools, payloads |
| Deformable manipulators & objects | P1/P2 tetrahedral FEM | Soft fingers, cables, fruit, foam, compliant fixtures |
| Joints, cables & actuators | `Link` constraints + 1D `Tendon` | Revolute/prismatic/cylindrical/universal/planar actuators, tendon-driven hands |
| Fracture & cohesive damage | `Material::with_fracture` + `Column::DAMAGE` | Cracking shells, brittle failure, energy-accounted softening |
| Liquids & FSI | WCSPH particles + SDF two-way coupling (`phys-fluids`) | Pouring liquids, buoyancy, submerged manipulation |
| Macroscopic fields | Symmetric multipole octrees (`BarnesHut`) | Orbital mechanics, N-body gravity clusters |

## Architectural Pillars

1. **Truthful & Error-Accounted.** SI units and real stiffnesses throughout. Optional `EnergyLedger` separates conservative stored energies from gravity, barrier, friction, joint damping, plastic, fracture, restitution, and settling work.
2. **One Coupled World.** Articulated joints, rigid/soft tactile contact, thermal conduction/expansion/phase change, dissipative heating, cohesive fracture, and two-way WCSPH fluids couple cleanly on one `World`.
3. **Zero-Allocation RL Loops.** Joint targets (`set_joint_target`), drives, and domain-randomized physical parameters update cleanly without invalidating fixed `BlockMatrix` sparsity patterns.
4. **Bitwise Determinism Across Threads.** Fixed-chunk parallel work and ordered reductions guarantee 1 thread and $N$ threads produce identical checkpoints and batch probe readings—critical for reproducible RL evals and VLA data curation.
5. **Dual-Mode Execution.** Fast headless throughput for multi-million-step RL/VLA data generation, or tight-tolerance / angular-error-audited / AL-tightened trajectories for engineering validation and system identification.
6. **Zero Third-Party Dependencies.** Pure Rust `crates/core/` with zero external crate dependencies (`std` only) for fast compilation and portable deployment.
