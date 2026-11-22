# Fusion Core in the main app

2026-10-17. First main-app integration of the October 14 variational-engine
architecture. This is a mechanics preview, not completion of the multiphysics
roadmap. See the [current specification audit](CORE_ENGINE_STATUS.md),
[Core vision](CORE_ENGINE_VISION.md) and [architecture](CORE_ENGINE_ARCHITECTURE.md).

## Use it

In the Solver inspector, choose **Physics engine → Fusion Core** before playing.
The Standard engine still offers the Classic and Unified MPM continua. Existing
projects keep their original engine. A new-core project writes
`[solver] kind = "core"` in format 11.

Open `examples/core-mechanics.fusion` for a rigid rubber ball and an elastic
jelly block above a floor. Generate and run that scene headlessly with:

```sh
cargo run --release -p fusion-project --example core_scene
cargo run --release -p fusion-app --bin fusion -- examples/core-mechanics.fusion
```

The app and headless `Project::build` use the same `Simulation` backend.

`examples/core-edge-contact.fusion` demonstrates crossed bars landing at their
interiors. Regenerate it and verify the landing with
`cargo run --release -p fusion-project --example core_edge_contact`.

## What runs

- Double-precision rigid bodies and quadratic tetrahedral elastic solids in
  coupled implicit solves within independent contact islands, with gravity,
  barrier contact, friction and
  the core's impact model.
- TR-BDF2 for smooth motion, with force-consistent second-order startup. Active contact and
  its release use a consistent backward-Euler rule within the affected island.
  This avoids mixed nodal rules and stale contact-force history. Unrelated
  islands retain their selected integrator. Difficult steps
  retry at smaller intervals; an exhausted retry rejects the entire requested
  interval and restores its physical state and integration history.
- Rigid boxes and spheres, including static supports and explicit mass/inertia.
- Edge contact between rigid boxes and finite-element boundaries, including
  fixed-body corners and edges. The lower-level core also checks capsule
  centrelines. Swept edge bounds and conservative advancement catch interior
  crossings even when neither endpoint finishes near the other object.
- Deformable self-contact between non-incident surface vertices/triangles and
  edges, enabled by default in the same coupled solve. Neighboring primitives
  that share a vertex are excluded. Swept queries also check same-solid edges.
- Contact precision retains the compact-object default and caps the length
  scale at twice an object's thickness for slender objects, separately within
  each island. Edge contacts use
  the same friction and impact laws as point contacts.
- Isotropic neo-Hookean solids without fracture or thermal evolution, including
  the jelly, rubber and foam presets. Material stiffness is not reduced.
- Surface friction on rounded primitives, including sphere sliding-to-rolling
  motion. The lower-level core also supports capsule rolling. See
  [the contact accuracy guide](CORE_ROLLING_CONTACT.md) for physical checks;
  rolling resistance and uniformly converged oblique impacts remain open.
- The solid's existing **cell size** controls tetrahedral meshing. Quadratic
  elements are used; requests beyond the preview's 6000-cell per-solid budget
  fail explicitly. The sphere mesher is a voxel approximation of a sphere.
- The editor draws actual finite-element boundary triangles. Point mass and
  volume views use the mesh's lumped nodal masses. Deformation shown in the
  inspector is volume-averaged at shared nodes; the energy readout uses the
  solver's quadrature, not that smoothed display field.
- Reset, in-memory rewind, project capture and saved-run continuation. State
  format 7 includes core integration history and checks backend/layout/time on
  decode. Older state formats remain readable on the Standard engine.

## Limits

Unsupported scene physics is rejected when building: fluids, cloth, soft-body
constraints, joints, scripts/controllers, kinematic bodies, imported hulls,
rigid stress, fracture/plasticity, fibres, heat and air. These continue to run
on the Standard engine. The new core has verified independent parallel islands
and exact fixed-point sleeping; general error-controlled multirate stepping
remains open. See [the island guide](CORE_ISLANDS.md).
Self-contact requires initially valid, separated non-incident surfaces;
coincident self surfaces reject the step. It does not repair intersecting meshes.
Contact-rich scenes can be expensive. Self-contact uses a true-distance
barrier; inter-object contact retains the existing surface tolerance. Initially
separated mesh edges retain positive geometric separation during a Newton
sweep. Exactly coincident zero-thickness edges on different objects retain
the existing contact tolerance at initialization and have no unique side.
Neither the solver's convergence nor a passing software test is a certificate
of physical accuracy for arbitrary scenes.

Core owns authoritative f64 state; the existing body/solid arrays are output
views. Author scene changes through the project and rebuild/reset. Runtime
editing of those views is not an input API for the core; changed poses,
velocities, shapes, masses or topology reject the next step with a reset message. Scene gravity and
rigid damping are read each step. Damping is an explicit operator split.
Low-level `core_settings_mut` overrides are for numerical experiments; the
project UI currently selects the default settings rather than serializing
arbitrary runtime overrides.

## Source and ownership

`crates/core/` contains a source snapshot of the lab's `core-math`,
`core-runtime`, `core-numerics`, `core-space`, `core-world` and
`phys-mechanics` crates, brought into the main workspace. The main app does
not depend on the ignored, separately versioned `lab/` checkout. That checkout
was left untouched. Future lab changes need explicit review and migration.

`fusion-sim/src/core_engine.rs` adapts app scene definitions and output views.
Mechanics and numerical methods remain in the engine crates, independent of
windowing and rendering. This migration does not yet replace the app's scene
schema with the complete world/API described in the October 14 plan.

The imported retry path was hardened: a failed first half cannot be hidden by
success in the second half; a failed last retry restores state; rejected work
is counted; invalid intervals do not advance the clock. Versioned checkpoints
preserve acceleration, earlier-stage state, impact-restart flags and affine
rigid state as well as positions and velocities.
The November 1 version 2 core payload also preserves previous gravity and
interval length; version 1 remains readable. Smooth startup and variable-step
BDF2 use this history consistently. See [the accuracy/compute guide](CORE_ACCURACY_20261101.md)
for optional work accounting, angular-error subdivision and exact rest reuse.

## Verification

The focused integration checks cover rigid contact, finite-element free fall
and mass, real boundary meshes, unsupported scenes, failure atomicity and
retained requested time, reset/seek, and exact in-process and saved-run replay.
The imported core tests cover math, deterministic runtime, numerics, world
state, derivative consistency and mechanics. Run:

```sh
cargo test --release -p core-math -p core-runtime -p core-numerics -p core-space -p core-world -p phys-mechanics -p fusion-sim -p fusion-project
cargo test --release -p fusion-app --test core_engine
cargo check -p fusion-app
```

No completed scientific validation or production readiness is claimed by this
integration. Contact searches now refit cached analytic-body hierarchies,
reuse triangle/edge bounds, and prune entire subtrees of the same object.
Analytic affine bounds avoid repeated polar decompositions while bounding the
actual intermediate rigid matrices. Moving-target CCD uses conservative
advancement with a fixed-separating-plane certificate for safe sliding paths;
it no longer assumes a changing body's distance is convex.

Regression checks cover crossed bars, fixed corners against a moving face,
capsule interiors, a coarse deformable bar, momentum and restitution under all
four integrators, high-speed sweeps, gradient consistency, common motion,
affine bounds, grouped searches, and exact replay across thread counts. The
new bar-interior failure is also checked through the main app's simulation
adapter. Conservative edge advancement follows
[Li et al.'s additive CCD](https://ipc-sim.github.io/C-IPC/).

The [mechanics accuracy baseline](CORE_MECHANICS_VALIDATION.md) adds analytic
beam/vibration, energy and impact-conservation gates independent of the app.

The remaining work is tracked in the [specification audit](CORE_ENGINE_STATUS.md):
broader mechanics validation and impact conservation, scheduler/bake basics, plasticity
and fracture, then transport and the other classical physics layers.

## Inspect solver cost

The inner linear solve uses diagonally equilibrated Cholesky factors for
3x3 nodal blocks and coupled rigid-body blocks. This follows the
[Cholesky factor/inverse route documented by LAPACK](https://www.netlib.org/lapack/explore-html/d4/d12/group__potri_gaced925926290482532c433f73b30b9f1.html).
Equilibration prevents differences in mass units alone from making a sound
block appear singular. A failed coupled factorization falls back to its
local blocks; an unsafe local block uses a positive finite diagonal inverse.
This changes the preconditioner, leaving the assembled physical equations,
materials and convergence tolerances unchanged.

Factor buffers are reused across Newton iterations, with one shared scratch
buffer per fixed batch of eight coupled bodies. Independent batches factor
in parallel. Sorted body groups let each CG vector chunk visit only its own
groups. A group crossing a chunk boundary uses local inverses consistently
on both sides, preserving a symmetric preconditioner and deterministic
results across thread counts.

The fixed 27-box benchmark reports total Newton and CG iterations, retry
splits and rejected steps as well as elapsed time. It fails if any requested
step is rejected. For example:

```sh
PILE_THREADS=1 PILE_PROFILE=1 cargo run --release -p phys-mechanics --example pile
PILE_THREADS=4 PILE_PROFILE=1 cargo run --release -p phys-mechanics --example pile
```

`PILE_STEPS` defaults to 120. Compare identical workloads without concurrent
builds or tests; adding collision coverage changes the workload, so reduced
search overhead alone is not a claim of an overall solver speedup.
The [recorded contact check](benchmarks/core-contact-20261023.txt) includes the
27-box workload, work counters and single-run timings. Profiling sections
overlap (`solve (all)` includes nested sections), so their times must not be
added together.

The [October 25 linear-solver comparison](benchmarks/core-linear-solve-20261025.txt)
records three paired runs of the unchanged 27-box workload: median elapsed
time fell from 5.603 s to 4.573 s (18.4% less time), CG iterations from 216,548
to 170,271 (21.4% fewer), and retry splits from 32 to 28. No requested step was
rejected. These are measurements on that workload and host. The record also
includes independently checked linear residuals, microbenchmarks, build flags,
source hashes and a source patch for reproducing the baseline separately.

The impulse sweep now reads only the velocities attached to each contact;
it no longer copies all scene velocities for every contact on every sweep.
The [self-contact cost example](../crates/phys-mechanics/examples/self_contact_cost.rs)
reports elapsed time, Newton/CG work, splits, impacts and rejected steps:

```sh
SELF_THREADS=2 cargo run --release -p phys-mechanics --example self_contact_cost
```

The [October 28 comparison](benchmarks/core-self-impact-20261028.txt) produced
identical old/new checkpoint bytes and work counts. Its small timing difference
was within ordinary run variation; it establishes the copying reduction,
not a meaningful overall speedup on that workload.

The [November 1 measurements](benchmarks/core-accuracy-20261101.txt) compare
force-consistent startup, optional angular-error refinement, tunable FEM
patches, parallel islands and exact sleeping. CG now verifies the true equation
residual, and reports actual matrix-vector products in addition to iterations.
Larger patches can reduce iterations while increasing elapsed time; the
default remains one nodal block. The [guide](CORE_ACCURACY_20261101.md) states
which options are on by default and their accuracy/cost limits.

## Training throughput

The Core also has a headless runner and streaming dataset generator. See
[training samples](TRAINING_SAMPLES.md) for commands, the array schema,
parallel settings and reference-quality checks.
