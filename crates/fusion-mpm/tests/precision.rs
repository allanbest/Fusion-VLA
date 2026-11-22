//! Where single precision changes the answer, and where it does not.
//!
//! PLAN.md §9 asks for precision to be "a per-solver choice, never a global
//! switch", and names the case: "an FP64 thermal solve coupled to FP32
//! material points". This file is that decision, measured rather than assumed.
//! One field in the continuum cannot be carried in `f32` at all; the rest can,
//! and the tests below are the evidence for both halves.

use fusion_core::{
    build_solid, solid::AMBIENT, BodySet, SolidDesc, SolidMaterial, SolidParams, SolidSet, Thermal,
};
use fusion_math::{vec3, DVec3, Vec3};
use fusion_mpm::MpmSolver;

/// Momentum summed in double, so that the instrument is not what is being
/// measured.
///
/// [`SolidSet::momentum`] sums in `f32` because it is a readout. A drift test
/// that used it would be measuring its own summation error — which is how the
/// first version of this test reported a drift half again as large as the
/// real one.
fn momentum(solid: &SolidSet) -> DVec3 {
    let mass = solid.params.point_mass() as f64;
    (0..solid.len()).fold(DVec3::ZERO, |sum, i| {
        let v = solid.particles.velocity[i];
        sum + DVec3::new(v.x as f64, v.y as f64, v.z as f64) * mass
    })
}

/// A weak heat source does not exist in single precision.
///
/// This is arithmetic, not a solver test, and it is the whole argument for
/// [`SolidSet::enthalpy`] being `f64`. Specific enthalpy near room temperature
/// is a quarter of a million J/kg; the spacing between representable `f32`
/// values up there is 0.031 J/kg. Two watts into half a kilogram delivers
/// 0.0028 J/kg per substep — under a *tenth* of that gap — so every increment
/// rounds away and the material never warms at all.
///
/// The twenty-watt case is worse in the way that matters more. It does not
/// vanish; it comes out 12% high, every time, in the same direction, because
/// repeatedly rounding a constant increment to the same lattice is a bias and
/// not noise. A wrong answer that looks like a plausible one is the failure
/// mode this project spends its tests on.
#[test]
fn a_weak_heat_source_vanishes_in_single_precision() {
    let alloy = SolidMaterial::ALLOY;
    let base = alloy.thermal.enthalpy_at(AMBIENT);
    let c = alloy.thermal.specific_heat as f64;
    let ulp = (f32::from_bits((base as f32).to_bits() + 1) - base as f32) as f64;
    assert!(
        ulp > 0.03,
        "an f32 near {base:.0} J/kg is quantised to {ulp} J/kg"
    );

    // Ten seconds of a torch, at the substep rate the solver runs at.
    let substeps = 10 * 60 * 24;
    let per_substep = |watts: f64| watts * (1.0 / 60.0 / 24.0) / 0.5;
    let warmed = |watts: f64| {
        let step = per_substep(watts);
        let mut wide = base;
        let mut narrow = base as f32;
        for _ in 0..substeps {
            wide += step;
            narrow += step as f32;
        }
        ((wide - base) / c, (narrow - base as f32) as f64 / c)
    };

    let (wide, narrow) = warmed(2.0);
    assert!(
        (wide - 0.0444).abs() < 1e-3,
        "two watts for ten seconds is +0.044 K, got {wide}"
    );
    assert_eq!(
        narrow, 0.0,
        "in f32 the same energy warms the material by {narrow} K, which should be nothing at all"
    );

    let (wide, narrow) = warmed(20.0);
    assert!((wide - 0.4444).abs() < 1e-3, "twenty watts gives {wide} K");
    assert!(
        narrow > wide * 1.1,
        "the f32 sum should be biased high, not merely noisy: {narrow} against {wide}"
    );
}

/// And the solver carries it: two watts, and the block warms by what the
/// energy says it should.
///
/// The same source as above, poured into a real solid through the same field
/// the diffusion pass integrates. Nothing here is a tolerance chosen to pass —
/// the answer is `Q / (m c)` and the assertion is a percent of it.
#[test]
fn the_thermal_solve_carries_two_watts() {
    let spacing = 0.02;
    let mut solid = build_solid(
        &SolidDesc::block(vec3(0.12, 0.12, 0.12))
            .with_params(SolidParams::at(spacing).of(SolidMaterial::ALLOY)),
    );
    let mut bodies = BodySet::new();
    let mut mpm = MpmSolver::new();

    let mass = solid.mass() as f64;
    let watts = 2.0;
    let seconds = 10.0;
    let frames = 600;
    let dt = 1.0 / 60.0;
    let start = solid.heat();
    for _ in 0..frames {
        let per_kilo = watts * dt as f64 / mass;
        for i in 0..solid.len() {
            solid.enthalpy[i] += per_kilo;
        }
        mpm.step(&mut solid, &mut bodies, Vec3::ZERO, dt);
    }

    // The energy went in and stayed in: diffusion moves heat about and may
    // neither make nor lose it.
    let expected = watts * seconds;
    let gained = solid.heat() - start;
    assert!(
        (gained - expected).abs() < 0.01 * expected,
        "poured in {expected} J and the solid holds {gained} J more"
    );
    // And it is visible as a temperature, which is the part f32 could not do.
    let warmed = solid.peak_temperature() - AMBIENT;
    let analytic = expected / (mass * SolidMaterial::ALLOY.thermal.specific_heat as f64);
    assert!(
        (warmed as f64 - analytic).abs() < 0.05 * analytic,
        "the block warmed {warmed} K where the energy says {analytic} K"
    );
}

/// A solid at one temperature stays at one temperature.
///
/// The transfer that carries heat onto the grid is a weighted mean — a sum of
/// `w m h` over a sum of `w m` — and this is the test that the two sums are
/// the same sum. They were not: the numerator was accumulated in `f64` and the
/// denominator was the momentum solve's `f32` nodal mass, and the quotient
/// carried the difference between them as **6 x 10^-5 K of temperature that
/// was not there**, different at every node, on a body at perfect thermal
/// equilibrium. The diffusion pass then did what it is built to do with a
/// temperature difference.
///
/// It went unnoticed because 6 x 10^-5 K is invisible next to any temperature
/// anybody had looked at. It was found by a test of a *microkelvin* gradient
/// failing in a way that made no sense — the gradient grew — which is the
/// argument for testing a solver at scales nobody intends to run it at.
#[test]
fn a_uniform_temperature_survives_the_transfer() {
    let material = SolidMaterial {
        thermal: Thermal::ALLOY,
        ..SolidMaterial::JELLY
    };
    let mut solid = build_solid(
        &SolidDesc::block(vec3(0.16, 0.08, 0.08)).with_params(SolidParams::at(0.02).of(material)),
    );
    let uniform = material.thermal.enthalpy_at(AMBIENT);
    for i in 0..solid.len() {
        solid.enthalpy[i] = uniform;
    }

    let mut bodies = BodySet::new();
    let mut mpm = MpmSolver::new();
    for _ in 0..10 {
        mpm.step(&mut solid, &mut bodies, Vec3::ZERO, 1.0 / 60.0);
    }

    let c = material.thermal.specific_heat as f64;
    let worst = (0..solid.len())
        .map(|i| (solid.enthalpy[i] - uniform).abs() / c)
        .fold(0.0f64, f64::max);
    assert!(
        worst < 1e-9,
        "a body in thermal equilibrium invented {worst} K of temperature"
    );
}

/// A temperature gradient finer than an `f32` can express still conducts.
///
/// The diffusion pass works on *differences* between neighbouring
/// temperatures. Near 300 K the gap between representable `f32` values is
/// 3 x 10^-5 K, so a microkelvin gradient has no digits left in single
/// precision — it is not small, it is absent, and the heat equation stops
/// running rather than running slowly.
///
/// The material is an elastic one with a metal's conductivity, and that is not
/// incidental: the first version of this test used `ALLOY`, whose model
/// yields, and a settling block's plastic self-heating buried a signal a
/// millionth of a kelvin tall. A test of one field has to be run on a material
/// where nothing else is writing to it.
#[test]
fn a_gradient_below_the_single_precision_gap_still_diffuses() {
    // Elastic, and a conductor far better than any of the library's — heat in
    // a real alloy crosses one cell in seven milliseconds and the whole bar in
    // twenty minutes, and a test that took twenty minutes to see a gradient
    // move would not be run. The conductivity is the only thing exaggerated;
    // the arithmetic under test is untouched by it.
    let material = SolidMaterial {
        thermal: Thermal {
            conductivity: 2000.0,
            ..Thermal::ALLOY
        },
        ..SolidMaterial::JELLY
    };
    let mut solid = build_solid(
        &SolidDesc::block(vec3(0.24, 0.06, 0.06)).with_params(SolidParams::at(0.02).of(material)),
    );
    let thermal = material.thermal;
    let c = thermal.specific_heat as f64;

    // A millionth of a kelvin, hot half against cold half. As an enthalpy that
    // is a difference of 9 x 10^-4 J/kg on a value of 2.6 x 10^5 — a
    // thirty-fifth of the gap between neighbouring `f32`s up there.
    let amplitude = 1e-6;
    let cold = thermal.enthalpy_at(AMBIENT);
    let hot = cold + amplitude * c;
    for i in 0..solid.len() {
        solid.enthalpy[i] = if solid.particles.translation(i).x > 0.0 {
            hot
        } else {
            cold
        };
    }
    let temperature = move |s: &SolidSet, i: usize| thermal.state_of(s.enthalpy[i]).0;
    let spread = |s: &SolidSet| {
        let (mut lo, mut hi) = (f64::INFINITY, f64::NEG_INFINITY);
        for i in 0..s.len() {
            lo = lo.min(temperature(s, i));
            hi = hi.max(temperature(s, i));
        }
        hi - lo
    };
    let before = spread(&solid);
    let heat_before = solid.heat();
    assert!(
        (before - amplitude).abs() < 1e-9,
        "the step should start at {amplitude} K, it is {before} K"
    );
    // Two values and nothing between them, which is what a step function is.
    let between = |s: &SolidSet| {
        let (lo, hi) = (thermal.state_of(cold).0, thermal.state_of(hot).0);
        let margin = 0.05 * (hi - lo);
        (0..s.len())
            .filter(|i| {
                let t = temperature(s, *i);
                t > lo + margin && t < hi - margin
            })
            .count()
    };
    assert_eq!(between(&solid), 0, "the initial profile is a step");

    let mut bodies = BodySet::new();
    let mut mpm = MpmSolver::new();
    for _ in 0..60 {
        mpm.step(&mut solid, &mut bodies, Vec3::ZERO, 1.0 / 60.0);
    }

    // The step has smoothed into a profile — most of the material now sits at
    // a temperature that is neither of the two it started at. In `f32` every
    // one of these increments is below half an ULP and the profile would still
    // be a step, unchanged, with the solver having run a thousand sub-cycles
    // to produce exactly nothing.
    let smoothed = between(&solid);
    assert!(
        smoothed > solid.len() / 3,
        "only {smoothed} of {} points found an intermediate temperature",
        solid.len()
    );
    // Diffusion has a maximum principle: it can flatten a profile and cannot
    // steepen one. This is also the check that catches a wide field being
    // quietly narrowed somewhere in the transfer, which would show up as
    // noise rather than as smoothing.
    let after = spread(&solid);
    assert!(
        after <= before,
        "the gradient grew, from {before} K to {after} K"
    );
    assert!(
        after > 0.0,
        "and decay is not the same as being rounded off"
    );
    // Conducting is moving heat, not making it.
    let leaked = (solid.heat() - heat_before).abs() / heat_before.abs();
    assert!(
        leaked < 1e-12,
        "the solid gained {leaked} of its heat by conducting"
    );
}

/// The mechanical solve does not need double precision, and here is the
/// measurement that says so.
///
/// A block sliding through empty space: no gravity, no bodies, no forces. Every
/// answer is known exactly — momentum is what it started with and `det F` is
/// one — so anything else is round-off, and this is the arrangement where it
/// has nowhere to hide.
///
/// Over 1200 steps the deformation error stays under `10^-6` and the momentum
/// drift under `2 x 10^-6` relative, and neither *grows*: sampled every 300
/// steps the drift wandered between 1.0 and 1.3 x 10^-6 and ended where it
/// started. Round-off in a stable solve is not a random walk — the elastic
/// response pulls the state back faster than rounding pushes it out.
///
/// That is why the deep grid runs FP32 mechanics coupled to an FP64 thermal
/// field, exactly as §9 predicts. It was worth building the alternative to be
/// sure: a version of this solver that accumulated nodal mass and momentum in
/// `f64` while keeping the particle state narrow — §9's "store narrow,
/// accumulate wide" — measured **no improvement at all** (8 x 10^-7 against
/// 5 x 10^-7, drift unchanged at 1.1 x 10^-6). The residue does not enter
/// through the sum; it enters through the narrow state either side of it, and
/// widening the sum alone cannot reach it. A real FP64 mechanical path means
/// widening the particle state — velocity, affine, deformation — and §9's own
/// list says where that earns its keep: **stiff implicit solves**. Fusion's
/// integrator is explicit, so that work belongs with the implicit one.
#[test]
fn the_mechanical_solve_holds_its_error_bound_in_single_precision() {
    let mut solid = build_solid(
        &SolidDesc::block(vec3(0.16, 0.16, 0.16))
            .with_velocity(vec3(1.0, 0.0, 0.0))
            .with_params(SolidParams::at(0.02).of(SolidMaterial::JELLY)),
    );
    let mut bodies = BodySet::new();
    let mut mpm = MpmSolver::new();
    let start = momentum(&solid);

    for _ in 0..1200 {
        mpm.step(&mut solid, &mut bodies, Vec3::ZERO, 1.0 / 60.0);
    }

    let strain = solid.peak_strain();
    assert!(
        strain < 1e-5,
        "a block that has done nothing shows |det F - 1| = {strain}"
    );
    let drift = (momentum(&solid).x - start.x).abs() / start.x;
    assert!(
        drift < 1e-5,
        "momentum drifted {drift} relative in a scene with no forces in it"
    );
}

/// Enthalpy is a state variable, so the checkpoint and the hash have to carry
/// all of it.
///
/// Widening a field is exactly the change that leaves a hash reading the old
/// width. This is the same lesson v0.8 learned from the deformation gradient,
/// asserted at the width the field is actually stored in: two runs that differ
/// by one part in 10^10 of one point's enthalpy — invisible to an `f32` hash —
/// must not report as identical.
#[test]
fn the_hash_sees_the_whole_width_of_the_enthalpy() {
    let build = || {
        build_solid(
            &SolidDesc::block(vec3(0.08, 0.08, 0.08))
                .with_params(SolidParams::at(0.02).of(SolidMaterial::ALLOY)),
        )
    };
    let a = build();
    let mut b = build();
    // A difference an f32 could not hold: below its ULP at this magnitude.
    b.enthalpy[0] += 1e-5;
    assert_ne!(
        a.enthalpy[0].to_bits(),
        b.enthalpy[0].to_bits(),
        "the difference has to survive being stored before a hash can find it"
    );
    assert_eq!(
        (a.enthalpy[0] as f32).to_bits(),
        (b.enthalpy[0] as f32).to_bits(),
        "and it is precisely the difference single precision cannot see"
    );
}

/// The library's thermal constants survive the widening unchanged.
///
/// [`Thermal::state_of`] and [`Thermal::enthalpy_at`] now work in double, and
/// the round trip through them is the property that matters: a temperature
/// turned into an enthalpy and back is the same temperature.
#[test]
fn temperature_and_enthalpy_still_invert_each_other() {
    for thermal in [Thermal::INERT, Thermal::ICE, Thermal::ALLOY] {
        for temperature in [100.0, 273.0, 293.15, 400.0, 455.0] {
            if thermal.melts() && (temperature - thermal.melt_point).abs() < 1.0 {
                // At the melting point itself the map is many-to-one by
                // construction — that is what a phase change is.
                continue;
            }
            let back = thermal.state_of(thermal.enthalpy_at(temperature)).0;
            assert!(
                (back - temperature as f64).abs() < 1e-6,
                "{temperature} K round-tripped to {back} K"
            );
        }
    }
}

/// The strain energy is the potential the stress is the gradient of.
///
/// `p_i = dPsi/dsigma_i`, checked by central differences on random
/// deformations. This is the test that makes [`fusion_mpm::energy_density`]
/// and [`fusion_mpm::principal_stress`] the same model rather than two
/// plausible functions that happen to sit in the same file — and it is the
/// only kind of check that catches a factor of two in one of them, which is
/// the classic error in an energy density and is invisible by inspection.
#[test]
fn the_strain_energy_differentiates_into_the_stress() {
    use fusion_core::solid::Constitutive;
    use fusion_mpm::{energy_density, principal_stress};

    let (mu, lambda) = (3.2e5, 7.1e5);
    let models = [
        Constitutive::FixedCorotated,
        Constitutive::NeoHookean,
        Constitutive::Snow {
            critical_compression: 0.02,
            critical_stretch: 0.01,
            hardening: 5.0,
        },
    ];
    // The rest state first, where both functions have an exact answer: no
    // deformation, no stress, no stored energy. Checked as an equality rather
    // than through the difference below, because a numerical derivative at a
    // stationary point is all error and no signal.
    for model in models {
        let sigma = Vec3::new(1.0, 1.0, 1.0);
        assert_eq!(
            energy_density(model, sigma, mu, lambda),
            0.0,
            "{model:?} stores energy in an undeformed material"
        );
        let p = principal_stress(model, sigma, mu, lambda);
        assert_eq!((p.x, p.y, p.z), (0.0, 0.0, 0.0), "{model:?} at rest");
    }

    // And a deterministic spread of real deformations: compressed, stretched,
    // and mixtures of the two.
    let cases = [
        Vec3::new(1.1, 0.95, 1.02),
        Vec3::new(0.8, 0.9, 1.3),
        Vec3::new(1.4, 1.4, 0.6),
        Vec3::new(0.7, 1.05, 0.98),
    ];
    for model in models {
        for sigma in cases {
            let analytic = principal_stress(model, sigma, mu, lambda);
            for axis in 0..3 {
                let step = 1e-3;
                let mut up = sigma;
                let mut down = sigma;
                up[axis] += step;
                down[axis] -= step;
                let numeric = (energy_density(model, up, mu, lambda)
                    - energy_density(model, down, mu, lambda))
                    / (2.0 * step);
                // The floor is not slack, it is the arithmetic: differencing
                // two `f32` energies costs about `mu * eps / 2h`, which is
                // twenty joules per cubic metre at these constants, against
                // stresses of order 10^4. A factor of two in either function —
                // the classic error in an energy density — is 10^4 out and
                // fails this by three orders.
                let tolerance = 0.01 * analytic[axis].abs().max(2.0e3);
                assert!(
                    (numeric - analytic[axis]).abs() < tolerance,
                    "{model:?} at {sigma:?} axis {axis}: dPsi/dsigma is {numeric} \
                     and the stress says {}",
                    analytic[axis]
                );
            }
        }
    }
}

/// An undeformed solid stores no strain energy, and a squashed one does.
#[test]
fn the_energy_budget_knows_a_squashed_solid_from_a_resting_one() {
    use fusion_mpm::ContinuumEnergy;

    let solid = build_solid(
        &SolidDesc::block(vec3(0.1, 0.1, 0.1))
            .with_params(SolidParams::at(0.02).of(SolidMaterial::JELLY)),
    );
    let resting = ContinuumEnergy::of(&solid, Vec3::ZERO);
    assert_eq!(
        resting.strain, 0.0,
        "a solid at F = I stores no strain energy"
    );
    assert_eq!(resting.kinetic, 0.0);
    assert!(resting.heat > 0.0, "and it holds heat, being at 293 K");

    let mut squashed = solid.clone();
    for f in squashed.deformation.iter_mut() {
        *f = fusion_math::Mat3::from_diagonal(vec3(1.0, 0.9, 1.0));
    }
    let loaded = ContinuumEnergy::of(&squashed, Vec3::ZERO);
    assert!(
        loaded.strain > 0.0,
        "a solid squashed a tenth stores {} J",
        loaded.strain
    );
    // Hooke, roughly: the energy of a small strain goes as its square, so
    // halving the strain should quarter the energy. Not exact — these models
    // are not linear — but a factor near four is the check that the density
    // is quadratic in strain rather than linear in it.
    let mut half = solid.clone();
    for f in half.deformation.iter_mut() {
        *f = fusion_math::Mat3::from_diagonal(vec3(1.0, 0.95, 1.0));
    }
    let ratio = loaded.strain / ContinuumEnergy::of(&half, Vec3::ZERO).strain;
    assert!(
        (3.0..5.0).contains(&ratio),
        "halving the strain changed the energy by a factor of {ratio}, not about four"
    );
}
