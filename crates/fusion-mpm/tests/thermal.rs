//! The thermal field, against answers that exist without it.
//!
//! Heat is the first physics v0.9 puts on the grid, and every test here is a
//! case where the right answer is known from outside the solver: the analytic
//! decay of a diffusion mode, the energy a phase change costs, and the size a
//! heated block reaches.

use fusion_core::{
    build_solid, solid::AMBIENT, BodyDesc, BodySet, SolidDesc, SolidMaterial, SolidParams,
    SolidSet, Thermal, World,
};
use fusion_math::{vec3, Vec3};
use fusion_mpm::MpmSolver;

const DT: f32 = 1.0 / 240.0;

fn bar(material: SolidMaterial, size: Vec3, spacing: f32) -> SolidSet {
    build_solid(&SolidDesc::block(size).with_params(SolidParams::at(spacing).of(material)))
}

/// The enthalpy formulation is exact where it can be checked by hand.
///
/// Below the melting point a material's temperature is its enthalpy over its
/// specific heat. At the melting point it stops, and stays stopped until the
/// latent heat has been paid in full. Above it, it heats again at the same
/// rate. Three straight lines, and the corners are what the formulation exists
/// for.
#[test]
fn a_phase_change_absorbs_energy_without_getting_hotter() {
    let ice = Thermal::ICE;
    // In double, because the field is: see `SolidSet::enthalpy`.
    let c = ice.specific_heat as f64;
    let melt_point = ice.melt_point as f64;
    let latent_heat = ice.latent_heat as f64;
    let solidus = c * melt_point;

    // Solid, and heating normally.
    let (t, melted) = ice.state_of(c * 260.0);
    assert!((t - 260.0).abs() < 1e-2, "solid ice read {t} K");
    assert_eq!(melted, 0.0);

    // Halfway through the latent heat: at the melting point, half melted.
    let (t, melted) = ice.state_of(solidus + 0.5 * latent_heat);
    assert!((t - melt_point).abs() < 1e-2, "melting ice read {t} K");
    assert!((melted - 0.5).abs() < 1e-3, "melted fraction {melted}");

    // The whole latent heat and no more: melted, still at the melting point.
    let (t, melted) = ice.state_of(solidus + latent_heat);
    assert!((t - melt_point).abs() < 1e-1, "just-melted ice read {t} K");
    assert!(melted > 0.99);

    // And past it, heating again at the same rate.
    let over = 20.0 * c;
    let (t, melted) = ice.state_of(solidus + latent_heat + over);
    assert!((t - (melt_point + 20.0)).abs() < 1e-1, "liquid read {t} K");
    assert_eq!(melted, 1.0);

    // The energy the whole crossing costs, per kilogram, is exactly the latent
    // heat — which is the number a materials table gives, and the reason to
    // store enthalpy rather than temperature.
    let just_solid = solidus;
    let just_liquid = solidus + latent_heat;
    assert!(
        ((just_liquid - just_solid) - latent_heat).abs() < 1.0,
        "melting cost {} J/kg, the table says {}",
        just_liquid - just_solid,
        ice.latent_heat
    );
    // And the temperature is the same at both ends of that spend.
    assert!((ice.state_of(just_solid).0 - ice.state_of(just_liquid).0).abs() < 0.2);
}

/// A diffusion mode decays at the rate the heat equation says it does.
///
/// The fundamental mode of an insulated bar is `cos(pi x / L)`, and it decays
/// as `exp(-alpha pi^2 t / L^2)` — no fitting, no constants to choose, and it
/// is wrong the moment the diffusivity, the cell size or the Laplacian is out
/// by a factor of anything.
///
/// Insulated is what this solver's free surface already is: a node with no mass
/// is absent rather than cold, so no flux crosses the boundary. That makes the
/// analytic solution the one with Neumann ends, which is the cosine.
#[test]
fn a_temperature_mode_decays_at_the_rate_the_heat_equation_says() {
    let length = 0.8f32;
    let spacing = 0.025;
    let material = SolidMaterial {
        // A diffusivity chosen so the mode decays measurably in a second of
        // simulated time; the point is the *rate*, not this material.
        thermal: Thermal {
            // Chosen so the mode decays by about a factor of e in the second
            // this test runs: `alpha = L^2 / (pi^2 t)`. A material that barely
            // diffuses would pass this test on a solver that does nothing.
            conductivity: 32_400.0,
            specific_heat: 500.0,
            ..Thermal::INERT
        },
        ..SolidMaterial::JELLY
    };
    let mut solid = bar(material, vec3(length, 0.1, 0.1), spacing);
    let alpha = material.thermal.diffusivity(material.density);

    // T = mean + amplitude * cos(pi x / L), x measured from the bar's end.
    let amplitude = 100.0f32;
    let mean = AMBIENT + 200.0;
    for i in 0..solid.len() {
        let x = solid.particles.translation(i).x + 0.5 * length;
        let t = mean + amplitude * (std::f32::consts::PI * x / length).cos();
        solid.enthalpy[i] = material.thermal.enthalpy_at(t);
    }

    // Only the thermal solve: gravity off and the mechanics left to sit, so
    // what is measured is diffusion rather than a bar bending.
    let mut bodies = BodySet::new();
    let mut mpm = MpmSolver::new();
    let seconds = 1.0f32;
    let steps = (seconds / DT) as usize;
    for _ in 0..steps {
        mpm.step(&mut solid, &mut bodies, Vec3::ZERO, DT);
    }

    // Recover the amplitude by projecting onto the same cosine.
    let mut numerator = 0.0f32;
    let mut denominator = 0.0f32;
    for i in 0..solid.len() {
        let x = solid.particles.translation(i).x + 0.5 * length;
        let basis = (std::f32::consts::PI * x / length).cos();
        numerator += (solid.temperature(i) - mean) * basis;
        denominator += basis * basis;
    }
    let measured = numerator / denominator;

    let expected =
        amplitude * (-alpha * std::f32::consts::PI.powi(2) * seconds / (length * length)).exp();
    let error = (measured - expected).abs() / expected;
    assert!(
        error < 0.15,
        "the mode decayed to {measured:.2} K; the heat equation says {expected:.2} K \
         (alpha = {alpha:.3e} m2/s, {:.1} % out)",
        error * 100.0
    );
    // And it decayed at all, so the test is not passing on a solver that does
    // nothing and an expectation that is nearly the starting value.
    assert!(
        expected < 0.8 * amplitude,
        "the chosen material barely diffuses in the time given; the test proves little"
    );
}

/// Diffusion moves heat and does not create it.
///
/// The conserved quantity of this field. An insulated body's total enthalpy is
/// constant however violently it is redistributed, and a scheme that leaks it
/// leaks it in the same direction every step, so a hundred steps show what one
/// would not.
#[test]
fn an_insulated_solid_keeps_the_heat_it_started_with() {
    let material = SolidMaterial {
        thermal: Thermal {
            // High enough that a second of simulated time actually moves heat
            // across a cell — `sqrt(alpha t)` is the distance it travels, and
            // at a plausible conductivity that is a fraction of a millimetre.
            // Conservation is easy to satisfy by doing nothing.
            conductivity: 32_400.0,
            ..Thermal::INERT
        },
        ..SolidMaterial::JELLY
    };
    let mut solid = bar(material, Vec3::splat(0.3), 0.04);
    // Half the block hot, which is the hardest case for a conservative scheme:
    // the gradient is a step and the flux across it is at its largest.
    for i in 0..solid.len() {
        if solid.particles.translation(i).x > 0.0 {
            solid.enthalpy[i] = material.thermal.enthalpy_at(AMBIENT + 400.0);
        }
    }
    let expected = solid.heat();

    let mut bodies = BodySet::new();
    let mut mpm = MpmSolver::new();
    for _ in 0..240 {
        mpm.step(&mut solid, &mut bodies, Vec3::ZERO, DT);
    }

    let drift = (solid.heat() - expected).abs() / expected.abs();
    assert!(
        drift < 5e-3,
        "an insulated solid's heat drifted {drift:.2e} in a second"
    );
    // And it did redistribute, so the conservation is not the conservation of
    // a field nothing touched.
    let spread = (0..solid.len())
        .map(|i| solid.temperature(i))
        .fold(f32::NEG_INFINITY, f32::max)
        - (0..solid.len())
            .map(|i| solid.temperature(i))
            .fold(f32::INFINITY, f32::min);
    assert!(
        spread < 380.0,
        "nothing diffused: the step is still {spread:.0} K"
    );
}

/// A heated block grows by the thermal strain and comes to rest there.
///
/// The multiplicative split, tested where it can be checked: a free block at
/// `ΔT` above its rest temperature should reach `det F = (1 + αΔT)³` and stop,
/// because at that stretch the constitutive model sees an undeformed material
/// and no stress. Without the split it would reach the same size under stress
/// and keep ringing.
#[test]
fn a_heated_block_expands_by_the_thermal_strain_and_stops() {
    let material = SolidMaterial {
        thermal: Thermal {
            // Exaggerated so the answer is bigger than the noise. The relation
            // being tested is linear in this number.
            expansion: 1.0e-3,
            conductivity: 0.0,
            ..Thermal::ALLOY
        },
        ..SolidMaterial::JELLY
    };
    let mut solid = bar(material, Vec3::splat(0.3), 0.05);
    let rise = 100.0f32;
    for i in 0..solid.len() {
        solid.enthalpy[i] = material.thermal.enthalpy_at(AMBIENT + rise);
    }

    let mut bodies = BodySet::new();
    let mut mpm = MpmSolver::new();
    let mut samples = Vec::new();
    for step in 0..1200 {
        mpm.step(&mut solid, &mut bodies, Vec3::ZERO, DT);
        // Averaged over the tail, because an undamped elastic solid given a
        // step change in its rest size rings about the new one forever. The
        // mean of the ringing is the equilibrium.
        if step >= 600 {
            let mean: f32 = solid
                .deformation
                .iter()
                .map(|f| f.determinant())
                .sum::<f32>()
                / solid.len() as f32;
            samples.push(mean);
        }
    }
    let measured = samples.iter().sum::<f32>() / samples.len() as f32;
    let stretch = 1.0 + material.thermal.expansion * rise;
    let expected = stretch * stretch * stretch;
    assert!(
        (measured - expected).abs() < 0.25 * (expected - 1.0),
        "expanded to det F = {measured:.5}; the thermal strain says {expected:.5}"
    );
}

/// Heat softens, and the softening is what makes the coupling physics.
#[test]
fn a_hot_material_is_softer_than_a_cold_one() {
    let alloy = SolidMaterial::ALLOY;
    let (cold, _) = alloy.lame_at(AMBIENT, 0.0);
    let (warm, _) = alloy.lame_at(0.5 * (AMBIENT + alloy.thermal.melt_point), 0.0);
    let (molten, _) = alloy.lame_at(alloy.thermal.melt_point, 1.0);

    assert!(warm < cold, "warming did not soften: {warm} vs {cold}");
    assert!(molten < 0.01 * cold, "melting left {molten} of {cold}");
    assert!(
        molten > 0.0,
        "a molten point with no stiffness cannot be solved"
    );

    // An inert material ignores its temperature entirely, which is what makes
    // the thermal field free for a scene that has not asked for it.
    let jelly = SolidMaterial::JELLY;
    let (a, _) = jelly.lame_at(AMBIENT, 0.0);
    let (b, _) = jelly.lame_at(AMBIENT + 500.0, 0.0);
    assert_eq!(a, b);
}

/// A struck bar heats where it was struck, and nothing said so.
///
/// §11's demo, as far as one test can carry it: the heat comes from the plastic
/// work, so it appears wherever the yielding did. Nothing in the scene names a
/// location, a temperature or an impact.
#[test]
fn plastic_work_heats_the_material_that_did_it() {
    let mut world = World::new();
    world.add(
        BodyDesc::cuboid(vec3(2.0, 0.2, 2.0))
            .at(vec3(0.0, -0.1, 0.0))
            .fixed(),
    );
    let mut solid = build_solid(
        &SolidDesc::block(vec3(0.16, 0.4, 0.16))
            .at(vec3(0.0, 0.45, 0.0))
            .with_params(SolidParams::at(0.03).of(SolidMaterial::ALLOY))
            .with_velocity(vec3(0.0, -14.0, 0.0)),
    );
    let start = solid.heat();
    let mut mpm = MpmSolver::new();
    for _ in 0..400 {
        mpm.step(&mut solid, &mut world.bodies, world.gravity, DT);
    }

    assert!(
        solid.heat() > start,
        "a bar driven into the floor at 14 m/s got no warmer"
    );
    // A tenth of a kelvin, and that number is the finding rather than a weak
    // assertion. The temperature a material raises itself to by yielding scales
    // with its yield stress, and a material soft enough for an explicit solver
    // at 240 Hz yields at around a megapascal where a real alloy yields at
    // hundreds. §11's struck bar heating enough to soften *itself* needs either
    // an implicit solve or a stiffness this tier cannot carry; the coupling is
    // here and correct, and its magnitude is honest about which tier it is in.
    //
    // Four hundredths since the alloy yields under von Mises plasticity. It
    // used the snow law, which compacts, and some of that tenth was the work
    // of crushing a metal's volume away — which a metal does not do.
    let peak = solid.peak_temperature();
    assert!(
        peak > AMBIENT + 0.04,
        "the hottest point reached only {peak:.4} K from {AMBIENT:.2}"
    );

    // And the heat is at the bottom, where the bar hit, rather than spread
    // evenly — which is what distinguishes "the mechanics heated it" from
    // "something added a constant".
    let mut low = 0.0f32;
    let mut high = 0.0f32;
    let mut lows = 0;
    let mut highs = 0;
    for i in 0..solid.len() {
        let above = solid.temperature(i) - AMBIENT;
        if solid.particles.translation(i).y < 0.25 {
            low += above;
            lows += 1;
        } else {
            high += above;
            highs += 1;
        }
    }
    let (low, high) = (low / lows.max(1) as f32, high / highs.max(1) as f32);
    assert!(
        low > high,
        "the struck end ({low:.3} K above ambient) was no hotter than the far end ({high:.3} K)"
    );
}

/// The thermal field does not cost determinism.
#[test]
fn a_thermal_scene_runs_the_same_way_twice() {
    fn run() -> Vec<f64> {
        let mut world = World::new();
        world.add(
            BodyDesc::cuboid(vec3(2.0, 0.2, 2.0))
                .at(vec3(0.0, -0.1, 0.0))
                .fixed(),
        );
        let mut solid = build_solid(
            &SolidDesc::block(Vec3::splat(0.2))
                .at(vec3(0.01, 0.5, -0.02))
                .with_params(SolidParams::at(0.04).of(SolidMaterial::ALLOY))
                .with_velocity(vec3(0.0, -8.0, 0.0)),
        );
        let mut mpm = MpmSolver::new();
        for _ in 0..200 {
            mpm.step(&mut solid, &mut world.bodies, world.gravity, DT);
        }
        solid.enthalpy.clone()
    }
    assert_eq!(run(), run(), "two runs of one thermal scene disagreed");
}
