# Fusion Core — Architecture (V1 Release)

> **Updated 2026-11-19 (V1 Release).** Reflects the complete V1 `crates/core/` implementation across Layers L0–L5c for robotics, VLA/RL training, differentiable physics, and classical multiphysics ([CORE_ENGINE_VISION.md](CORE_ENGINE_VISION.md), [FUSION_ARCHITECTURE.latex](FUSION_ARCHITECTURE.latex)).
> See [CORE_ENGINE_STATUS.md](CORE_ENGINE_STATUS.md) for verified test evidence and exact implementation boundaries.

## Two Mathematical Core Abstractions (+ Adjoint Sensitivity)

Classical mechanics, robotics actuation, and continuum multiphysics in Fusion Core V1 organize around two forward abstractions and one backward sensitivity pass:

### 1. Variational Mechanics as Incremental Potential Minimization

Rigid bodies, FEM solids, contact barriers, lagged Coulomb friction, gravity, and articulated robot joints/drives minimize one scalar incremental potential per integrator stage. Outer augmented-Lagrangian multiplier updates (**Pillar 2**, `Settings::augmented_lagrangian_passes`) tighten bilateral constraints $\mathbf{c}(\mathbf{x}) = \mathbf{0}$ cleanly without extreme penalty ill-conditioning:

```text
    E(x, λ) = ½ (x − x̃)ᵀ (M / β²) (x − x̃) + Σ E_term(x) + λᵀ c(x) + (ρ / 2) ‖c(x)‖²
```

- **Composability (`Term` trait):** Each mechanical law supplies energy, gradient, PSD-projected Hessian blocks, and a barrier/inversion step limit $\alpha_{\max}$.
- **Fixed vs. Changing Sparsity:** Static-topology terms (`Inertia`, `Elastic`, `RigidBodies`, and all `Joints` constraints/drives including `Cylindrical`, `Universal`, `Planar`, and `Tendon`) pre-bind directly into fixed CSR `BlockMatrix` slots (`hessian.0`), while dynamic terms (`Contact`, `MutualGravity`) stream into thread-local `Triplets` (`hessian.1`). Live RL joint updates (`set_joint_target`, `set_joint_drive`) incur zero symbolic matrix rebuilds.
- **Exact Contact & Exclusion for Robots:** Structural joint pairs (`BallJoint`, `Weld`, `Hinge`, `Slider`, `Cylindrical`, `Universal`, `Planar`) are excluded in `Contact::new` so overlapping link housings at joint pivots do not self-collide, while gripper jaws, non-adjacent links, and payloads collide through IPC barriers and surface rolling arms.

### 2. Conservation-Law Transport (`Process` trait) & Two-Way Fluids (`phys-fluids`)

Spatial fields evolve via `∂q/∂t + ∇·F(q) = s` over shared SoA `Columns` on `core-world::World`. `phys-thermal::HeatConduction` writes `Column::TEMPERATURE` using temperature-dependent apparent capacitance `Thermal::effective_specific_heat` (latent-heat phase change) and nodal dissipative heating (`HeatConduction::set_nodal_power` fed by `Mechanics::dissipative_thermal_power`), driving thermal expansion and temperature-tabulated moduli `μ(T), λ(T)` in `phys-mechanics`. `phys-fluids::FluidSolver` simulates deterministic `f64` weakly-compressible Wendland $C^2$ SPH over `NeighbourGrid` with two-way FSI momentum/buoyancy coupling (`FluidSolver::step_coupled`).

### 3. Implicit Function Adjoint Differentiation (Pillar 4)

Because each backward-Euler stage converges to stationarity $\nabla_{\mathbf{x}} E(\mathbf{x}^*, \boldsymbol{\theta}) = \mathbf{0}$, `Mechanics::backward_step` solves the exact unclipped adjoint system (`RigidBodies::exact_hessian = true` during adjoint assembly):

```text
    (∇²_xx E) w = (∂L / ∂x*)ᵀ      ⇒      dL / dθ = ∂L / dθ − wᵀ ∇²_xθ E
```

computing exact gradients (`AdjointGradients`) wrt initial positions, initial velocities, uniform gravity, joint drive targets/rest lengths, and joint stiffnesses via `SparseCholesky` / CG (`adjoint.rs` + `Joints::adjoint_link_gradients`).

## Layered Crate Graph (L0–L5c)

| Layer | Crate | Status | Responsibility |
|---|---|---|---|
| **L0** | `core-math` | Implemented | `f64` vectors/matrices/quaternions/poses, CODATA SI constants, deterministic `Rng`, fast trigonometric & Jacobi 3×3 eigen/SVD |
| **L1** | `core-runtime` | Implemented | Persistent lock-free worker `Pool` (64 nested batch slots), deterministic fixed-chunk `map_reduce`/`split_reduce`, `Profile` |
| **L2** | `core-numerics` | Implemented | `BlockMatrix`, reusable `Triplets`, verified CG (zero-start skip, steepest-descent fallback), equilibrated `BlockJacobi`, zero-dependency RCM + skyline $LDL^\top$ `SparseCholesky` (with automatic CG fallback `<= 256` blocks and `NewtonSettings::direct_blocks`), Newton, `Dual` AD |
| **L3** | `core-space` | Implemented | Analytic SDF `Shape::{Sphere, Cuboid, Capsule, HalfSpace}`, `Aabb`, refitted/grouped `Bvh`, `NeighbourGrid`, triangle/segment queries, symmetric dual-tree `BarnesHut` |
| **L4** | `core-world` | Implemented | `Material` (`Plasticity`, `Fracture`, `Thermal` latent heat), P1/P2 `TetMesh`, sparse `Columns::{TEMPERATURE, DAMAGE, PLASTIC_STRAIN}`, complete `Link` zoo (`Anchor`, `Spring`, `BallJoint`, `Weld`, `Hinge`, `Slider`, `Cylindrical`, `Universal`, `Planar`, `Tendon`), virtual `Probe::{JointState, Wrench, Tactile, Imu}`, pure-Rust `urdf` (`World::load_urdf` / `RobotModel`), `World::subset` |
| **L5a** | `phys-mechanics` | Implemented | Incremental-potential mechanics (`Inertia`, `Gravity`, `MutualGravity`, `Elastic`, `RigidBodies`, `Joints`, `Contact`), $J_2$ Hencky plasticity, cohesive fracture (`damage` degradation + `EnergyLedger::fracture_work`), AL constraint passes, `sample_probes`, `dissipative_thermal_power`, `backward_step` IFT adjoints, `WorldBatch` + `DomainRandomizer`, multi-rate islands, support/joint sleep, v1–v5 checkpoints |
| **L5b** | `phys-thermal` | Implemented | P1/P2 FEM `HeatConduction` (`BackwardEuler`, `CrankNicolson`, Dirichlet/`VolumetricSource`/`Convection`/`Radiation`/`NodalPower`, enthalpy-capacitance phase change) |
| **L5c** | `phys-fluids` | Implemented | Deterministic `f64` Wendland $C^2$ WCSPH `FluidSolver` (`FluidParticles`, `FluidSettings`, `step` and `step_coupled` two-way FSI) |

## World Model & Robotics Interfaces

```text
World
 ├─ Materials     SI density, elastic tables μ(T)/λ(T), friction/restitution, Plasticity, Fracture, Thermal (with_phase_change)
 ├─ Objects       Fixed, 12-DOF affine Rigid, P1/P2 Solid (+ SoA Columns: TEMPERATURE, PLASTIC_STRAIN, DAMAGE)
 ├─ Links         Anchor, Spring, BallJoint, Weld, Hinge, Slider, Cylindrical, Universal, Planar, Tendon (+ JointDrive PD + smooth pseudo-Huber max_force)
 ├─ Probes        JointState, Wrench, Tactile, Imu -> ProbeReading (via Mechanics::sample_probes)
 └─ Importers     World::load_urdf(...) -> RobotModel
```

### Smooth Pseudo-Huber Actuator Saturation & Hinge Kinematics (`phys-mechanics::joints`)
- **Actuator Saturation:** `JointDrive` adds a smooth $C^\infty$ potential $\psi(\delta) = F_{\max}\bigl(\sqrt{\delta^2 + (F_{\max}/k)^2} - F_{\max}/k\bigr)$ with positive-definite scalar second derivative $\psi''(\delta) > 0$, avoiding torque-clamp corners in Newton-CG.
- **Large-Angle Hinges:** Each step initializes $\theta_0$ from exact full rotations at `begin_step` and evaluates a stage-linearized hinge coordinate $\theta(\mathbf{x})$ inside Newton so arbitrary $360^\circ$ rotations remain exact across steps while keeping the stage Hessian rank-1 PSD.

## Orchestration: Islands, Sleep, Wake & Multi-Rate Subcycling

1. **Swept + Linked Island Partitioning (`Plan`):** Swept velocity/gravity AABBs (`HalfSpace` exact-plane culled), mutual gravity, and mechanical links (including multi-body `Tendon` routing) partition objects into independent islands without spuriously linking objects resting on a common `Fixed` floor.
2. **Settling & Instant RL Wake (`Worker::at_rest`):** Islands resting on `Fixed` supports or damped world-mounted joints (`Weld`, `Planar`, `Hinge`, `Slider`, `Cylindrical` with `damping > 0`) settle via `settle_time` once full-step translational and rotational displacements fall below $\varepsilon / \texttt{settle\_time}$. Mutating any joint target (`set_joint_target`), drive, or body state in-place wakes the sleeping island on the very next step (`World::subset_links_match`).
3. **Proactive Multi-Rate Subcycling (`Settings::{max_substep, cfl_subcycling}`):** Active islands choose $N_{\text{sub}} \in [1, 2^{\text{MAX\_SPLITS}}]$ from local velocity-to-size ratios and maximum spring/drive/tendon natural frequencies $\sqrt{\omega_{\max}^2} / 2\pi$ (`StepReport::max_island_substeps`), recording cumulative swept bounds across all substeps and rolling back atomically if newly coupled islands meet.
4. **Atomic Verification:** Cumulative swept bounds across all substeps of every island are re-checked at outer interval end; any newly reached island triggers a whole-interval rollback and merged re-solve.
5. **Thread-Count Bitwise Determinism:** Fixed work-splitting chunks and sorted merge keys guarantee identical binary checkpoints (`v1–v5`) and `WorldBatch` probe readings on 1 or $N$ threads.
