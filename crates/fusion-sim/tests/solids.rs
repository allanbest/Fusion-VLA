//! v0.8: the continuum, driven through the engine rather than the solver.
use fusion_core::{BodyDesc, Material, SolidDesc, SolidMaterial, SolidParams, World};
use fusion_math::{vec3, Vec3};
use fusion_sim::Simulation;

fn scene(material: SolidMaterial) -> Simulation {
    let mut world = World::new();
    world.add(
        BodyDesc::cuboid(vec3(2.0, 0.2, 2.0))
            .at(vec3(0.0, -0.1, 0.0))
            .with_material(Material::CONCRETE)
            .fixed(),
    );
    world.add_solid(
        &SolidDesc::block(Vec3::splat(0.24))
            .at(vec3(0.0, 0.5, 0.0))
            .with_params(SolidParams::at(0.04).of(material)),
    );
    Simulation::new(world)
}

/// A solid added to a world is stepped by the simulation, lands, and stays.
#[test]
fn the_engine_carries_a_solid_from_end_to_end() {
    let mut sim = scene(SolidMaterial::JELLY);
    assert!(
        sim.world.solid_point_count() > 100,
        "the block was not sampled"
    );
    let mass = sim.world.total_mass();

    sim.run_frames(150, |_, _| {});

    assert!(sim.is_stable(), "the scene went non-finite");
    assert!(
        (sim.world.total_mass() - mass).abs() < 1e-4 * mass,
        "mass changed under a method that creates none"
    );
    let report = sim.mpm_report();
    assert_eq!(report.points, sim.world.solid_point_count());
    assert!(!report.refused && !report.unstable, "{report:?}");

    let solid = &sim.world.solids[0];
    let lowest = (0..solid.len())
        .map(|i| solid.particles.translation(i).y)
        .fold(f32::INFINITY, f32::min);
    assert!(lowest > -0.1, "the solid sank to {lowest:.3} m");
}

/// Reset rebuilds a solid's history, not merely its position.
///
/// The failure this guards is specific to a continuum: putting the points back
/// where they started while leaving `F` as the squash left it gives a block at
/// rest under the stress of a landing that has been undone, which detonates on
/// the first step.
#[test]
fn reset_forgets_the_deformation_as_well_as_the_position() {
    let mut sim = scene(SolidMaterial::SNOW);
    sim.run_frames(120, |_, _| {});
    assert!(sim.world.solids[0].peak_strain() > 1e-4, "nothing deformed");

    sim.reset();
    assert!(
        sim.world.solids[0].peak_strain() < 1e-6,
        "reset kept {:.5} of strain",
        sim.world.solids[0].peak_strain()
    );
    assert!(
        sim.world.solids[0].plastic.iter().all(|p| *p == 1.0),
        "reset kept the plastic history"
    );

    sim.run_frames(60, |_, _| {});
    assert!(sim.is_stable(), "the scene detonated after a reset");
}

/// And the whole thing replays bit-for-bit from a checkpoint (§17, §22).
#[test]
fn a_solid_survives_a_checkpoint() {
    let mut sim = scene(SolidMaterial::JELLY);
    for _ in 0..80 {
        sim.step();
    }
    let checkpoint = sim.checkpoint();
    let mut expected = Vec::new();
    for _ in 0..60 {
        sim.step();
        expected.push(sim.world.solids[0].particles.position.clone());
    }

    sim.restore(&checkpoint);
    for (step, want) in expected.iter().enumerate() {
        sim.step();
        assert_eq!(
            &sim.world.solids[0].particles.position,
            want,
            "replay diverged {} steps after the checkpoint",
            step + 1
        );
    }
}

/// The state hash covers a solid's deformation, not merely where its points
/// are.
///
/// Everything §22 promises is checked through this hash: `study determinism`
/// compares it, `Project::verify` compares it, and the CI determinism diff
/// compares it. A hash that stopped at position and velocity would call two
/// visibly different continua identical — and every one of those tools would
/// agree, confidently, in writing.
///
/// Found by looking for what v0.8 had not touched rather than by a failure,
/// which is the only way this kind of hole is ever found: nothing goes red
/// when a hash ignores a field. It goes green.
#[test]
fn the_state_hash_can_tell_two_deformations_apart() {
    use fusion_sim::state_hash;

    let mut sim = scene(SolidMaterial::JELLY);
    sim.run_frames(30, |_, _| {});
    let before = state_hash(&sim);

    // Positions and velocities left exactly as they are; only the history
    // changes. A hash that ignores it cannot see this.
    for f in &mut sim.world.solids[0].deformation {
        *f = *f * 1.05;
    }
    assert_ne!(
        before,
        state_hash(&sim),
        "the hash could not tell a squashed solid from an unsquashed one"
    );

    // And the plastic history, which is the other half of the state.
    let mut sim = scene(SolidMaterial::SNOW);
    sim.run_frames(30, |_, _| {});
    let before = state_hash(&sim);
    sim.world.solids[0].plastic[0] *= 0.5;
    assert_ne!(before, state_hash(&sim), "the hash ignored plastic history");
}

/// And it can tell two temperatures apart (v0.9).
///
/// Written at the same time as the field it hashes, which is the only defence
/// against the failure the deformation gradient demonstrated one version ago:
/// state that the hash cannot see is state every reproducibility tool in the
/// project agrees to ignore.
#[test]
fn the_state_hash_can_tell_two_temperatures_apart() {
    use fusion_sim::state_hash;

    let mut sim = scene(SolidMaterial::ALLOY);
    sim.run_frames(20, |_, _| {});
    let before = state_hash(&sim);
    sim.world.solids[0].enthalpy[0] *= 1.01;
    assert_ne!(
        before,
        state_hash(&sim),
        "the hash ignored the thermal field"
    );
}

/// Heat poured into a solid melts it, and the phase change shows as a plateau.
///
/// The v0.9 headline through the engine rather than the solver: energy in, a
/// temperature that stops at the melting point while the latent heat is paid,
/// and a material that flows afterwards. It is the `heat` demo's scene, run
/// headless, so the demo cannot quietly stop working.
#[test]
fn a_solid_heated_past_its_melting_point_melts_and_flows() {
    let mut world = World::new();
    world.add(
        BodyDesc::cuboid(vec3(0.7, 0.24, 0.4))
            .at(vec3(0.0, 0.12, 0.0))
            .with_material(Material::CONCRETE)
            .fixed(),
    );
    world.add_solid(
        &SolidDesc::block(vec3(0.24, 0.24, 0.16))
            .at(vec3(0.0, 0.36, 0.0))
            .with_params(SolidParams::at(0.02).of(SolidMaterial::ALLOY)),
    );
    let mut sim = Simulation::new(world);
    let melt = SolidMaterial::ALLOY.thermal.melt_point;

    let height = |sim: &Simulation| {
        let s = &sim.world.solids[0];
        (0..s.len())
            .map(|i| s.particles.translation(i).y)
            .sum::<f32>()
            / s.len() as f32
    };
    let start_height = height(&sim);

    let mut plateau = 0;
    let mut previous = 0.0f32;
    for _ in 0..900 {
        {
            // 100 kW into the base, as energy rather than as a temperature: a
            // thermostat would hold the number and hide the phase change,
            // because the plateau *is* the temperature refusing to move while
            // energy goes in.
            let solid = &mut sim.world.solids[0];
            let heated: Vec<usize> = (0..solid.len())
                .filter(|i| solid.particles.translation(*i).y < 0.30)
                .collect();
            let mass = solid.params.point_mass() * heated.len() as f32;
            let per_kilo = 100_000.0 / 60.0 / mass as f64;
            for i in heated {
                solid.enthalpy[i] += per_kilo;
            }
        }
        sim.advance_frame();
        let peak = sim.mpm_report().peak_temperature;
        // Frames spent within a kelvin of the melting point while energy is
        // still going in. A material with no latent heat crosses in one.
        if (peak - melt).abs() < 1.0 && (peak - previous).abs() < 0.5 {
            plateau += 1;
        }
        previous = peak;
    }

    assert!(sim.is_stable(), "the scene went non-finite");
    let report = sim.mpm_report();
    assert!(
        report.peak_temperature > melt,
        "the base only reached {:.0} K of {melt:.0} K",
        report.peak_temperature
    );
    assert!(
        report.melted > 0.1,
        "only {:.1}% melted after 15 s at 100 kW",
        report.melted * 100.0
    );
    assert!(
        plateau > 30,
        "the temperature crossed the melting point in {plateau} frames; \
         the latent heat should hold it there for a second or more"
    );
    // The top never gets there, because conduction takes time and this is a
    // field rather than a switch.
    let solid = &sim.world.solids[0];
    let coldest = (0..solid.len())
        .map(|i| solid.temperature(i))
        .fold(f32::INFINITY, f32::min);
    assert!(
        coldest < melt - 100.0,
        "the whole block heated evenly: {coldest:.0} K"
    );
    // And it slumped, which is the softening doing something mechanical.
    assert!(
        height(&sim) < start_height - 0.02,
        "a block whose base melted did not settle: {:.3} m to {:.3} m",
        start_height,
        height(&sim)
    );
}

/// Damage reaches the state hash, like every other kind of state.
///
/// Third time: the deformation gradient was missed in v0.8 and found a version
/// later, the thermal field was covered in the change that added it, and this
/// is damage. The check is cheap and the failure it prevents is silent, which
/// is the whole argument for doing it every time.
#[test]
fn the_state_hash_can_tell_a_broken_solid_from_a_whole_one() {
    use fusion_sim::state_hash;

    let mut sim = scene(SolidMaterial::PORCELAIN);
    sim.run_frames(30, |_, _| {});
    let before = state_hash(&sim);
    sim.world.solids[0].damage[0] = 0.5;
    assert_ne!(before, state_hash(&sim), "the hash ignored damage");
}

/// The continuum's energy budget balances, and the term that makes it balance
/// is the one nothing could see before v0.9.
///
/// A jelly block dropped on a slab. At the bottom of the bounce **4.6 of its
/// 5.9 joules are elastic strain energy** — not kinetic, not potential, stored
/// in the deformation gradient. An energy diagnostic without that term watches
/// three quarters of a scene's energy vanish on impact and reappear a
/// twentieth of a second later, every bounce, and reports it as a conservation
/// failure.
///
/// What the test asserts is the direction: energy may leave (this solve is
/// dissipative, and measurably so — see below) and may not arrive. A total
/// that *climbs* is the failure mode PLAN.md §9 warns about and the one an
/// energy diagnostic exists to catch.
#[test]
fn a_bouncing_solid_never_gains_energy() {
    let mut world = World::new();
    world.add(
        BodyDesc::cuboid(vec3(1.0, 0.2, 1.0))
            .at(vec3(0.0, -0.1, 0.0))
            .with_material(Material::CONCRETE)
            .fixed(),
    );
    world.add_solid(
        &SolidDesc::block(vec3(0.12, 0.12, 0.12))
            .at(vec3(0.0, 0.35, 0.0))
            .with_params(SolidParams::at(0.02).of(SolidMaterial::JELLY)),
    );
    let mut sim = Simulation::new(world);

    let start = sim.mechanical_energy();
    let mut peak_strain = 0.0f64;
    let mut highest = start;
    for _ in 0..180 {
        sim.advance_frame();
        highest = highest.max(sim.mechanical_energy());
        peak_strain = peak_strain.max(sim.continuum_energy().strain);
    }

    assert!(
        highest <= start * 1.01,
        "the scene reached {highest} J having started with {start} J"
    );
    assert!(
        peak_strain > 0.2 * start,
        "the block stored only {peak_strain} J of strain energy out of {start} J, \
         which would mean the impact never happened"
    );
    // And it ends with less than it began: three seconds of bouncing on an
    // explicit solve with a friction boundary costs about three quarters of
    // the budget. That number is documented in docs/STATUS.md rather than
    // asserted tightly here — what matters to the test is the sign.
    assert!(sim.mechanical_energy() < start);
}

/// Heat is a separate ledger from the mechanical one, and stays whole.
///
/// The continuum's thermal energy is conserved by its own solve: diffusion
/// moves it and may not make or lose it. Reported alongside the mechanical
/// budget rather than added into it, because plastic work moves joules from
/// one to the other in one direction only, and a single total would hide the
/// transfer.
#[test]
fn the_energy_report_keeps_heat_apart_from_mechanics() {
    let mut world = World::new();
    world.add_solid(
        &SolidDesc::block(vec3(0.1, 0.1, 0.1))
            .with_params(SolidParams::at(0.025).of(SolidMaterial::ALLOY)),
    );
    world.gravity = Vec3::ZERO;
    let mut sim = Simulation::new(world);

    let before = sim.continuum_energy().heat;
    assert!(before > 0.0, "a solid at 293 K holds heat");
    let report = sim.advance_frame().expect("a frame runs at least one step");
    assert!(
        (report.heat as f64 - before).abs() < 1e-3 * before,
        "the step report says {} J of heat where the solid holds {before} J",
        report.heat
    );
    // The mechanical total is not the thermal one — 10^6 joules of enthalpy
    // would swamp every mechanical number in the scene if the two were added.
    assert!(
        sim.mechanical_energy().abs() < 0.01 * before,
        "the mechanical budget has picked up the heat"
    );
}
