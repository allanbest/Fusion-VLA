//! The energy ledger (PLAN.md §23).
//!
//! Two things have to be true for a breakdown like this to be worth having.
//! It must **add up** — every joule the step gained or lost attributed to
//! exactly one stage, so the rows can be trusted against the total. And it
//! must **accuse the right stage** — a diagnostic that points at `integrate`
//! every time is worse than none, because it costs a day before you stop
//! believing it.

use fusion_core::{BodyDesc, ClothDesc, FluidDesc, FluidParams, Material, World};
use fusion_math::{vec3, Vec3};
use fusion_sim::Simulation;
use fusion_solver::ledger::mechanical_energy;
use fusion_solver::Stage;

fn energy_of(sim: &Simulation) -> f64 {
    mechanical_energy(
        &sim.world.bodies,
        &sim.world.particles,
        &sim.world.fluids,
        sim.world.gravity,
    )
}

fn stack(boxes: usize) -> World {
    let mut world = World::new();
    world.add(
        BodyDesc::cuboid(vec3(20.0, 1.0, 20.0))
            .at(vec3(0.0, -0.5, 0.0))
            .with_material(Material::CONCRETE)
            .fixed(),
    );
    for k in 0..boxes {
        world.add(
            BodyDesc::cuboid(Vec3::ONE)
                .at(vec3(0.0, 0.5 + k as f32, 0.0))
                .with_material(Material::WOOD),
        );
    }
    world
}

#[test]
fn the_ledger_is_off_until_asked_for() {
    let mut sim = Simulation::new(stack(4));
    let report = sim.step();
    assert!(report.energy_stages.is_none());
    assert!(!sim.ledger().is_enabled());
    assert_eq!(sim.ledger().summary(), "energy ledger: disabled");

    sim.energy_ledger = true;
    let report = sim.step();
    assert!(report.energy_stages.is_some());
    assert!(sim.ledger().is_enabled());
}

#[test]
fn the_stages_add_up_to_the_step() {
    // The property that makes every other row believable. If the parts do not
    // sum to the whole, some energy is going somewhere the ledger cannot see,
    // and no individual attribution can be trusted.
    let mut sim = Simulation::new(stack(6));
    sim.energy_ledger = true;
    // Awake: a body that falls asleep leaves the ledger's sum, and a settled
    // stack does fall asleep. That is a transfer out of the books, not a
    // change of energy, and this test is about the books adding up.
    sim.sleep.enabled = false;
    // Let the stack land, so the interesting stages are actually doing work.
    sim.run_frames(30, |_, _| {});

    for step in 0..60 {
        let before = energy_of(&sim);
        let report = sim.step();
        let after = energy_of(&sim);

        let stages: f32 = report.energy_stages.unwrap().iter().sum();
        let measured = (after - before) as f32;
        let scale = after.abs().max(1.0) as f32;

        assert!(
            (stages - measured).abs() <= 1e-4 * scale,
            "step {step}: stages sum to {stages} J, the step actually changed by {measured} J"
        );
        assert!(
            (sim.ledger().total() - measured).abs() <= 1e-4 * scale,
            "step {step}: ledger total disagrees with the measured change"
        );
    }
}

#[test]
fn free_fall_matches_the_integrator_s_analytic_energy_loss() {
    // Semi-implicit Euler does not conserve energy in free fall, and the
    // amount it loses is derivable rather than mysterious. Per substep:
    //
    //     v' = v + g h,  x' = x + v' h
    //     dKE = m v g h + 1/2 m g^2 h^2
    //     dPE = -m g (v + g h) h
    //     dE  = -1/2 m g^2 h^2
    //
    // Independent of velocity, so it is the same every substep of the fall.
    // This is the sharpest possible test of the ledger: not "the number is
    // small" but "the number is *this* number", to five figures.
    let radius = 0.5f32;
    let mut world = World::new();
    world.add(BodyDesc::sphere(radius).at(vec3(0.0, 50.0, 0.0)));
    let mut sim = Simulation::new(world);
    sim.energy_ledger = true;

    let mass = 1000.0 * (4.0 / 3.0) * std::f32::consts::PI * radius.powi(3);
    let h = sim.solver_dt / sim.solver.substeps as f32;
    let expected = -0.5 * mass * 9.81f32.powi(2) * h * h * sim.solver.substeps as f32;

    // The tolerances are the *simulation's* precision, not the ledger's. The
    // ledger sums in `f64`; the velocities it reads are `f32`, and energy is
    // quadratic in velocity — so the per-step agreement with the closed form
    // loosens as the body speeds up, while the mean stays put because the
    // rounding is unbiased. Asserting both is the honest pair: the mean says
    // the ledger has the right answer, the band says how much of the run's
    // own noise it is faithfully reporting.
    // The per-step band is derived, not tuned. Velocities are `f32`, energy
    // is quadratic in them, so one ulp of velocity is worth `m v * v eps` of
    // energy — that is `eps` times the kinetic energy — and a substep can lose
    // one. Eight substeps gives `8 eps KE`, which at the bottom of this fall
    // is a few thousandths of a joule against a signal of 0.055. The mean is
    // held far tighter because the rounding is unbiased and averages out.
    let mut sum = 0.0f64;
    let steps = 120;
    for step in 0..steps {
        let report = sim.step();
        let total: f32 = report.energy_stages.unwrap().iter().sum();
        sum += total as f64;
        let band = 1e-3 * expected.abs() + 8.0 * f32::EPSILON * report.kinetic_energy.abs();
        assert!(
            (total - expected).abs() < band,
            "step {step}: ledger says {total} J, analytic is {expected} J, band is {band} J"
        );
    }
    // Two percent on the mean, and the residual is real rather than noise:
    // the closed form assumes exact arithmetic, while the velocity it is
    // predicted from is an `f32` accumulating `g h` a hundred times, which
    // drifts systematically from `n g h`. The ledger is measuring the
    // simulation that exists, not the one in the derivation — and the check
    // that it measures *that* one exactly is `the_stages_add_up_to_the_step`,
    // which compares it against an independent double-precision sum of the
    // same step and holds to one part in ten thousand.
    let mean = (sum / steps as f64) as f32;
    assert!(
        (mean - expected).abs() < 2e-2 * expected.abs(),
        "mean per-step loss {mean} J against an analytic {expected} J"
    );
}

#[test]
fn a_settled_stack_never_gains_energy_overall() {
    // The ledger's actual job. A settled stack is doing nothing, so the step
    // has no business ending with more energy than it started with — and a
    // step that does is the mechanism by which a tall tower walks itself
    // apart while nobody is touching it.
    //
    // The *total* is the assertion, not any individual row. A resting stack
    // pushes tens of kilojoules a second through `recover` and takes slightly
    // more back out at the velocity pass; what matters is the remainder.
    let mut sim = Simulation::new(stack(4));
    sim.energy_ledger = true;
    sim.run_frames(120, |_, _| {}); // let it settle

    let mut gains = Vec::new();
    for step in 0..120 {
        sim.step();
        if let Some(gain) = sim.ledger().net_gain() {
            gains.push((step, gain));
        }
    }
    assert!(
        gains.is_empty(),
        "a settled stack gained energy on {} of 120 steps: {:?}",
        gains.len(),
        &gains[..gains.len().min(5)]
    );
}

#[test]
fn only_the_velocity_stages_can_be_accused() {
    // `suspects` is deliberately narrow. An earlier version listed every
    // constraint solve as unable to gain energy and duly accused `contacts` of
    // gaining 18.8 kJ in a ten-box tower — a constraint holding a box up
    // against gravity raises its potential energy, which is what holding
    // something up means. This pins the narrowed rule so it cannot drift back.
    let mut sim = Simulation::new(stack(10));
    sim.energy_ledger = true;
    // Awake, so the last step still has a stage holding the tower up.
    sim.sleep.enabled = false;
    for _ in 0..480 {
        sim.step();
        let accused = sim.ledger().suspects(1e-6);
        assert!(
            accused.is_empty(),
            "{} was accused of gaining energy",
            accused[0].0.name()
        );
    }
    // And the stage that legitimately gains is visibly doing so, so the
    // narrowing has not made the ledger blind.
    assert!(sim.ledger().stage(Stage::Contacts) > 0.0);
}

#[test]
fn vorticity_confinement_is_the_stage_allowed_to_gain() {
    // It adds energy by construction — it exists to put back rotational
    // detail the density solve removed. The point of this test is that the
    // ledger *attributes it to the right stage*: folded into `recover` it
    // would be invisible, and the one place energy legitimately appears would
    // look like a bug in the position solve.
    let mut world = World::new().with_gravity(Vec3::ZERO);
    let mut params = FluidParams::water_at(0.06);
    params.vorticity = 4.0; // far above the default, so the effect is loud
    params.viscosity = 0.0;
    world.add_fluid(
        &FluidDesc::block(Vec3::splat(0.5))
            .with_params(params)
            .with_velocity(vec3(0.4, 0.0, 0.0)),
    );
    // Something to stir it against, so there is shear for vorticity to find.
    world.add(
        BodyDesc::sphere(0.12)
            .at(vec3(0.0, 0.0, 0.0))
            .with_material(Material::STEEL)
            .fixed(),
    );

    let mut sim = Simulation::new(world);
    sim.energy_ledger = true;

    let mut fluid_post = 0.0f32;
    let mut clamped = 0.0f32;
    for _ in 0..60 {
        sim.step();
        fluid_post += sim.ledger().stage(Stage::FluidPost);
        clamped += sim.ledger().stage(Stage::SpeedClamp);
    }
    assert!(
        fluid_post > 0.0,
        "vorticity confinement did not show up as a gain: {fluid_post} J"
    );
    // The speed clamp only ever removes, and at this gain it fires hard. It
    // is a separate stage precisely so it cannot mask the row above: folded
    // together the pair nets negative and vorticity looks harmless.
    assert!(clamped <= 0.0, "the speed clamp added {clamped} J");
    // And none of it is blamed on the position solve.
    assert!(sim.ledger().suspects(1e-2).is_empty());
}

#[test]
fn the_ledger_resolves_a_small_delta_against_a_large_total() {
    // Why the sums are `f64`. This scene holds megajoules; the per-step
    // changes worth seeing are hundredths of a joule. In single precision the
    // spacing between representable values at 10^7 is about 1 J, so an `f32`
    // ledger would report its own rounding and nothing else.
    let mut world = World::new();
    world.add(
        BodyDesc::cuboid(vec3(200.0, 2.0, 200.0))
            .at(vec3(0.0, -1.0, 0.0))
            .fixed(),
    );
    // Heavy, and high up: 500 t at 300 m is about 1.5 GJ of potential energy.
    for k in 0..8 {
        world.add(
            BodyDesc::cuboid(Vec3::splat(2.0))
                .at(vec3(0.0, 300.0 + k as f32 * 2.1, 0.0))
                .with_material(Material::STEEL),
        );
    }
    let mut sim = Simulation::new(world);
    sim.energy_ledger = true;

    let total_energy = energy_of(&sim).abs();
    assert!(
        total_energy > 1e8,
        "the scene is not large enough to make the point: {total_energy} J"
    );

    // Free fall again, so the change per step is the integrator's analytic
    // -1/2 m g^2 h^2 and nothing else: about 52 J against a total of 1.5 GJ,
    // a ratio of three parts in a hundred million. An `f32` ledger cannot
    // represent a difference that small at this magnitude at all.
    let mass = 8.0 * 8.0 * 7850.0f32;
    let h = sim.solver_dt / sim.solver.substeps as f32;
    let expected = -0.5 * mass * 9.81f32.powi(2) * h * h * sim.solver.substeps as f32;

    // A percent, and the reason it is not tighter is worth naming: half a
    // million kilograms moving at a metre a second means `dKE/dv` is about
    // 6 x 10^5 J per m/s, so one `f32` ulp of velocity is already a tenth of a
    // joule per body. The ledger is not the limit here — the state is.
    for step in 0..30 {
        let before = energy_of(&sim);
        sim.step();
        let after = energy_of(&sim);
        let measured = (after - before) as f32;
        assert!(
            (measured - expected).abs() < 1e-2 * expected.abs(),
            "step {step}: measured {measured} J against an analytic {expected} J"
        );
        // The ledger and an independent double-precision measurement of the
        // same step must agree far more tightly than either agrees with
        // theory: they are measuring the same arithmetic.
        assert!(
            (sim.ledger().total() - measured).abs() < 1e-4 * expected.abs(),
            "step {step}: ledger says {} J, measurement says {measured} J",
            sim.ledger().total()
        );
    }
}

#[test]
fn the_reported_energy_includes_particles_and_fluid() {
    // A defect through v0.4: `StepReport` summed rigid bodies only, so
    // `energy_delta` was meaningless for exactly the scenes where a solver bug
    // is hardest to see by eye.
    let mut world = World::new();
    world.add(
        BodyDesc::cuboid(vec3(10.0, 0.5, 10.0))
            .at(vec3(0.0, -0.25, 0.0))
            .fixed(),
    );
    world.add_cloth(
        &ClothDesc::new(10, 10, 0.1)
            .at(vec3(0.0, 2.0, 0.0))
            .with_mass(1.0),
    );
    world.add_fluid(
        &FluidDesc::block(Vec3::splat(0.3))
            .at(vec3(1.0, 1.0, 0.0))
            .with_params(FluidParams::water_at(0.06)),
    );

    let mut sim = Simulation::new(world);
    let report = sim.step();

    // Bodies alone would be zero: the only rigid body is static.
    assert!(
        report.kinetic_energy != 0.0 || report.potential_energy != 0.0,
        "a scene of cloth and water reported no energy at all"
    );
    let independent = mechanical_energy(
        &sim.world.bodies,
        &sim.world.particles,
        &sim.world.fluids,
        sim.world.gravity,
    ) as f32;
    let scale = independent.abs().max(1.0);
    assert!(
        (report.total_energy() - independent).abs() < 1e-3 * scale,
        "step report says {} J, an independent sum says {independent} J",
        report.total_energy()
    );
}

#[test]
fn the_summary_names_the_stage_that_gained() {
    let mut sim = Simulation::new(stack(3));
    sim.energy_ledger = true;
    sim.run_frames(20, |_, _| {});
    let text = sim.ledger().summary();
    assert!(text.contains("total"));
    // Every row that appears is one of the named stages.
    for line in text.lines() {
        let name = line.split_whitespace().next().unwrap_or("");
        if name == "total" || name.is_empty() {
            continue;
        }
        assert!(
            Stage::ALL.iter().any(|s| s.name().starts_with(name)),
            "unrecognised stage row: {line:?}"
        );
    }
}

#[test]
fn worst_stage_agrees_with_the_ledger() {
    let mut sim = Simulation::new(stack(6));
    sim.energy_ledger = true;
    sim.run_frames(20, |_, _| {});
    for _ in 0..40 {
        let report = sim.step();
        assert_eq!(report.worst_stage(), sim.ledger().worst_gain());
    }
}

#[test]
fn enabling_the_ledger_does_not_change_the_simulation() {
    // A diagnostic that perturbs what it measures is not a diagnostic. The
    // ledger only reads, and this is the assertion that keeps it that way.
    let mut plain = Simulation::new(stack(6));
    let mut measured = Simulation::new(stack(6));
    measured.energy_ledger = true;

    for step in 0..200 {
        plain.step();
        measured.step();
        assert_eq!(
            fusion_sim::state_hash(&plain),
            fusion_sim::state_hash(&measured),
            "the ledger changed the answer at step {step}"
        );
    }
}
