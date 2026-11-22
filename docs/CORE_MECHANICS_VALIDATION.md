# Core mechanics: accuracy baseline

Updated 2026-11-01; baseline established 2026-10-30. This validates
`phys-mechanics` and its foundation crates, independently
of the editor, project format, rendering or training-data generation. The full
classical-physics roadmap remains in [Core Engine Status](CORE_ENGINE_STATUS.md).

## Run the checks and measurements

```sh
CARGO_PROFILE_RELEASE_LTO=false CARGO_PROFILE_RELEASE_CODEGEN_UNITS=16 \
  cargo test --release -j 2 -p phys-mechanics --offline
ACCURACY_THREADS=1 CARGO_PROFILE_RELEASE_LTO=false CARGO_PROFILE_RELEASE_CODEGEN_UNITS=16 \
  cargo run --release -j 2 -p phys-mechanics --example mechanics_accuracy --offline
```

The example emits CSV, with the same physical fixtures used by
[the acceptance tests](../crates/phys-mechanics/tests/accuracy.rs).
Each row records geometry resolution, time step, node count, measured/reference
value, energy and momentum diagnostics, attempted Newton/CG work, time splits,
contact restarts, impacts and elapsed solve time. A failed physical step aborts
the example; passing it is not a substitute for running the acceptance tests.

Timing covers stepping, including rejected/repeated work. Mesh generation,
preparation, measurement and file output are outside the timed region. Newton
and CG iterations are work indicators, not FLOP counts. Compare runs without
concurrent builds or other tests. Relative error is blank for a zero reference;
for cases without energy/modal/angular measurements those fields are zero and
mean **not measured**, not a claim of zero physical energy or error.

## Physical references and acceptance criteria

All inputs use SI units. The beam and rod use quadratic tetrahedra and Poisson's
ratio zero to isolate the intended deformation from lateral Poisson effects.

| Check | Setup and independent reference | Gate |
|---|---|---|
| Static cantilever | L=1 m, square side=0.1 m, E=1 MPa, rho=1000 kg/m³, g=0.01 m/s². Clamp the end plane. Tip deflection `q L^4/(8EI) + q L^2/(2 k G A)`, `q=rho g A`, `k=5/6`, `G=E/2`: **1.512 mm** | Coarse/fine longitudinal mesh sizes 0.2/0.1 m: <2%/<1% error and fine improves. Require four consecutive settled steps, max nodal speed <1e-7 m/s |
| Free rod vibration | L=1 m, side=0.1 m, E=0.1 MPa, rho=1000 kg/m³. Small sinusoidal axial velocity. `f=sqrt(E/rho)/(2L)` = **5 Hz** | At dt=1/60, 1/120, 1/240 s, frequency error decreases and finest <0.5%; energy retention improves and finest >97%. Modal amplitude per period >99%; no energy growth above 0.1%. Also check mesh refinement at fixed dt |
| Cantilever vibration | Same beam, zero gravity, small bending displacement with section rotation. `f1=beta1²/(2 pi)*sqrt(EI/(rho A L^4))`, `beta1=1.8751040687`: **0.510835 Hz** | Both longitudinal mesh sizes within 2% of Euler–Bernoulli theory; frequency change under refinement <0.5%; energy and amplitude bounded |
| Moving self-contact | Two quadratic pieces of one mesh, total mass 38.4 kg, approaching at ±1 m/s plus a common nonzero velocity. No supports | dt=1/60, 1/120, 1/240 s: max linear momentum error <1e-3 kg m/s, angular error <0.1% of characteristic `m v L`, total physical energy does not grow by >1% |
| Known external impulse | Same contact fixture under uniform g=(0,-0.5,0) m/s² | All four integrators: compare total momentum at every step to `P0 + M g t`; error <1e-3 kg m/s |
| Strict self-contact replay | Existing isolated two-part fixture, absolute nonlinear position convergence 1e-9 m | Linear momentum error <1e-5 kg m/s, bounded angular drift and energy; exact checkpoint replay with 1/4 threads |
| Tiny elastic shear | Simple shear at strains 1e-2 through 1e-10, E=0.1 MPa and 200 GPa, multiple rotations | Nonnegative energy matching exact `mu gamma²/2`, allowing the expected floating-point error relative to strain |
| Force-consistent startup | Same 5 Hz free rod, dt=1/60 s, new startup versus old startup within the same engine | Frequency error <2%; new retained energy >93% and >15 percentage points above the old startup |
| Angular-defect refinement | Same unsupported impact, dt=1/60 s, nonlinear tolerance 1e-7 and optional angular budget 1e-4 | Refined drift <75% of unrefined drift, linear drift <1e-5 kg m/s, bounded energy; report refinement work |
| FEM patches | Same fine-step rod, four-node block patches | Frequency error <0.5%, retained energy >99%, linear momentum error <1e-9 kg m/s |
| Work/history/scheduling | Analytic free fall from the first interval, changing intervals/gravity, Coulomb stopping; serial/parallel rigid and large FEM islands | Exact gravity reference and ledger balance; frictional work within 4% of known stopping energy; exact checkpoint replay and byte-identical diagnostic/parallel/sleep trajectories; whole-interval rollback if accuracy budget fails |
| Surface friction/rolling | Homogeneous sphere sliding into rolling, an already rolling sphere, skew initial spin and a horizontal capsule | Known contact velocity, spin and physical-energy targets; capsule's actual inertia; saved spinning-contact replay. Exact criteria in [the follow-up guide](CORE_ROLLING_CONTACT.md) |
| Resolved oblique rigid impact | Two free spheres with nonzero common velocity; analytic first contact and normal/sticking tangential impulse, e=0/0.5 | dt=1/480 and 1/960 s: velocity error <0.015 m/s, spin error <0.02 rad/s, angular drift <1e-3 kg m²/s, linear drift <1e-8 kg m/s, energy-ratio error <1e-3. Coarser diagnostic rows do not meet these criteria |

A 10:1 beam is a 3D solid: shear, rotational inertia, clamp effects and finite
strain mean Euler–Bernoulli theory is an approximation, particularly for bending
frequency. The static reference includes Timoshenko shear. The mesh checks here
refine the beam/rod's length; they are not a complete 3D refinement certificate
for arbitrary shapes or near-incompressible materials.

References: [University of Alberta beam equations](https://engcourses-uofa.ca/beam-structures/plane-beam-approximations/),
[Jönköping University's beam theory and rectangular shear correction](https://mechanics.ju.se/SolidMechanics/BeamTheory.html),
[TU Graz cantilever modes](https://lampz.tugraz.at/~hadley/memm/mechanics/cbeam.php).
The rod reference follows the one-dimensional wave equation with free ends;
it does not use another implementation of Fusion as its reference.

## What changed in the solver

Previously struck nodes alone switched integration rules and had their
velocities overwritten after the position solve. That mixed effective masses
and history across elastically/contact-coupled unknowns and disturbed the
balance of internal impulses.

Now a stage with active contact uses **one backward-Euler rule across the
coupled island**. If a smooth stage discovers contact, it is repeated before
writing physical state. A stalled smooth contact solve can also take this path.
The next stage, including the first stage after release, starts without old
contact-force memory. Resting/slower contacts need this too: the restitution
speed threshold alone did not prevent energy growth in a finer-step test.
Restitution then applies pairwise impulses, without overwriting individual
nodal velocities. Smooth, contact-free motion retains the selected integrator.
`StepReport::contact_restarts` exposes repeated work, including failed attempts.

Parallel mesh self-edges also need a differentiable contact potential. Their
barrier now uses the [IPC parallel-edge mollifier](https://ipc-sim.github.io/file/IPC-paper-fullRes.pdf):
`z=|ea cross eb|²/(1e-3 |ea_rest|² |eb_rest|²)`, `m=z(2-z)` below z=1,
otherwise m=1. The gradient includes the product rule; the Hessian uses a
positive approximation for angular smoothing. Lagged friction uses the
smoothed normal force. Vertex/triangle barriers remain, and edge CCD still
runs even at exact parallel alignment. This change is currently restricted
to self-edges of a deformable mesh, not every analytic rigid-edge pair.
Derivative checks cover parallel, nearly parallel and ordinary configurations.

Near a rotation, the neo-Hookean energy is evaluated without subtracting large
first-order terms to obtain tiny second-order strain energy. The rewritten
invariants and `log1p`/series expression are algebraically the same material
law; stiffness, stress and the physical Hessian are not weakened.

These changes do not constitute a complete Moreau–Jean or energy–momentum
integrator. The distinction between smooth integration and nonsmooth contact
is motivated by the issues discussed in
[Acary's contact integration analysis](https://arxiv.org/abs/1410.2499).

## Limits and next work

- Backward Euler dissipates energy and does not exactly conserve angular
  momentum. The tests bound drift; they do not establish monotone angular-error
  convergence for arbitrary impacts or a restitution-independent continuum limit.
- Contact/release damping is now local to each [independent island](CORE_ISLANDS.md).
  Eight checks cover isolation, swept grouping, verified replanning, rollback
  and portable history through contact and successful BDF2 retries. Conservative
  bounds can still group objects that never physically contact; one large FEM
  object stays together. November 1 adds parallel dispatch and exact fixed-point
  sleep/wake, with separate trajectory-equality and approaching-collider tests.
- Smooth motion now uses force-consistent higher-order startup. The optional
  old-startup control exposes how much energy its Euler first step removed;
  per-period retention alone would hide that loss. Contact and release remain
  Euler and dissipative. Variable-step BDF2 and gravity history now have their
  own exact-reference and checkpoint gates.
- Collision refinement should consider both dt and contact precision. These
  fixtures keep contact precision fixed; they do not prove every impact law
  converges to the same physical solution as dt shrinks.
- An optional accepted-interval ledger now records physical energy and discrete
  gravity, barrier, Coulomb and restitution work. Its signed residual combines
  integration/quadrature, solve tolerance, projection and unaccounted boundary
  work; it is not measured heat or isolated numerical damping. The contact
  barrier is not counted as physical material strain energy.
- Optional free-island angular-defect refinement reduces drift in the tested
  impact at a substantial compute cost. It is not exact angular conservation
  or a general trajectory-error bound; boundary contact is excluded. The
  default remains disabled. Details and settings are in
  [the November 1 guide](CORE_ACCURACY_20261101.md).
- Additional materials, arbitrary mesh quality, anisotropy, incompressibility,
  high-speed impacts and much larger scale ranges need their own gates.

The [measurement record](benchmarks/core-mechanics-20261030.txt) retains source
hashes, before/after cases and accuracy-versus-cost observations. Heat and other
multiphysics modules are not implemented by this milestone.
The [November 1 paired record](benchmarks/core-accuracy-20261101.txt) measures
the subsequent startup, refinement and scheduling changes separately.
The [surface-contact follow-up](CORE_ROLLING_CONTACT.md) adds physical rolling
references and exact affine SDF force gradients. Its oblique-impact record
retains coarse failures and nonmonotone errors under refinement; the resolved
gates are not a general impact-time convergence certificate.
