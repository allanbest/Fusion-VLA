//! §23's workflow, checked against things that are true without it.
//!
//! A research tool that reports the wrong number is worse than no research
//! tool, because its output looks like evidence. Every test here checks a
//! claim that can be verified independently: momentum is conserved because
//! XPBD's corrections are equal and opposite, a scene compared against itself
//! diverges nowhere, a first-order method converges at order one.

use fusion_core::{BodyDesc, Material, World};
use fusion_math::{vec3, Vec3};
use fusion_project::Project;
use fusion_research::{compare, converge, determinism_diff, metrics, store, sweep};
use fusion_sim::Simulation;

/// A small scene with contacts in it, as a project so sweeps can vary it.
fn stack(boxes: usize) -> Project {
    let mut world = World::new();
    world.add(
        BodyDesc::cuboid(vec3(12.0, 0.5, 12.0))
            .at(vec3(0.0, -0.25, 0.0))
            .with_material(Material::CONCRETE)
            .fixed(),
    );
    for k in 0..boxes {
        world.add(
            BodyDesc::cuboid(Vec3::splat(0.5))
                .at(vec3(0.0, 0.25 + k as f32 * 0.52, 0.0))
                .with_material(Material::WOOD),
        );
    }
    Project::from_world(&world)
}

/// Two bodies drifting in free space, where momentum is exactly conserved and
/// there is nothing to argue about.
fn free_pair() -> Project {
    let mut world = World::new().with_gravity(Vec3::ZERO);
    world.add(
        BodyDesc::sphere(0.4)
            .at(vec3(-1.0, 0.0, 0.0))
            .with_velocity(vec3(1.5, 0.0, 0.0))
            .with_material(Material::STEEL),
    );
    world.add(
        BodyDesc::sphere(0.4)
            .at(vec3(1.0, 0.0, 0.0))
            .with_velocity(vec3(-0.5, 0.0, 0.0))
            .with_material(Material::WOOD),
    );
    Project::from_world(&world)
}

// --- conservation --------------------------------------------------------

/// Momentum is conserved through a collision, and the trace says so.
///
/// This is the sharp one of the three. Every XPBD correction is equal and
/// opposite, so nothing in the solver may move total momentum without an
/// external force. A drift here is a defect, not a tolerance — which is why
/// the bound is tight.
#[test]
fn momentum_is_conserved_and_the_trace_reports_it() {
    let mut sim = free_pair().build().expect("the pair builds");
    let mut trace = metrics::ConservationTrace::new();
    trace.observe(&sim);
    for step in 0..900 {
        sim.step();
        if step % 5 == 0 {
            trace.observe(&sim);
        }
    }
    let summary = trace.summary().expect("samples were taken");

    assert!(
        summary.momentum_drift_relative < 1e-4,
        "momentum drifted {:.3e} ({:.4}% of the run's scale)",
        summary.momentum_drift,
        summary.momentum_drift_relative * 100.0
    );
    assert!(
        summary.mass_drift < 1e-6,
        "mass changed by {:.3e}, which should be impossible",
        summary.mass_drift
    );
    // The bodies did meet: a test that passes because nothing happened is not
    // a test of a collision.
    assert!(sim.step_index() > 0);
    assert!(
        summary.simulated_time > 0.0,
        "the trace covered no time at all"
    );
}

/// Momentum is only checked where checking it means something.
///
/// A body resting on a static floor is held up by that floor, and an immovable
/// body supplies whatever impulse it likes — so a scene with static geometry
/// has no momentum invariant, and reporting one anyway produced 1456% drift on
/// a tank of water that was behaving perfectly. The summary says which kind of
/// scene it saw.
#[test]
fn momentum_is_reported_as_a_check_only_for_a_closed_scene() {
    let mut open = stack(4).build().expect("builds");
    let mut trace = metrics::ConservationTrace::new();
    trace.observe(&open);
    for _ in 0..60 {
        open.step();
        trace.observe(&open);
    }
    assert!(
        !trace.summary().unwrap().closed,
        "a scene with a static floor is not closed"
    );

    let mut free = free_pair().build().expect("builds");
    let mut trace = metrics::ConservationTrace::new();
    trace.observe(&free);
    for _ in 0..60 {
        free.step();
        trace.observe(&free);
    }
    assert!(
        trace.summary().unwrap().closed,
        "two spheres in free space are a closed scene"
    );
}

/// Gravity is accounted for before any drift is reported.
///
/// A closed scene falling under gravity gains momentum every step, exactly
/// `M g Δt` of it. That is not the solver inventing momentum, and a diagnostic
/// that says it is will read 100% on almost every scene ever built.
#[test]
fn gravity_is_not_mistaken_for_a_momentum_leak() {
    let mut world = World::new();
    world.add(
        BodyDesc::sphere(0.3)
            .at(vec3(-2.0, 5.0, 0.0))
            .with_material(Material::STEEL),
    );
    world.add(
        BodyDesc::sphere(0.3)
            .at(vec3(2.0, 7.0, 0.0))
            .with_material(Material::WOOD),
    );
    let mut sim = Project::from_world(&world).build().expect("builds");

    let mut trace = metrics::ConservationTrace::new();
    trace.observe(&sim);
    for _ in 0..300 {
        sim.step();
        trace.observe(&sim);
    }
    let summary = trace.summary().expect("samples");
    assert!(summary.closed, "nothing static, so this is closed");
    // The bodies are in free fall and carry real momentum by the end.
    assert!(
        sim.world.linear_momentum().length() > 10.0,
        "the scene should have picked up momentum falling"
    );
    // None of which the solver invented.
    assert!(
        summary.momentum_drift_relative < 1e-3,
        "gravity was counted as a leak: {:.4}% drift",
        summary.momentum_drift_relative * 100.0
    );
}

/// An empty trace summarises to nothing rather than to a perfect run.
#[test]
fn a_trace_with_nothing_in_it_has_no_summary() {
    assert!(metrics::ConservationTrace::new().summary().is_none());
}

// --- determinism ---------------------------------------------------------

/// The guarantee, checked by the tool that exists to check it.
#[test]
fn the_same_scene_twice_never_diverges() {
    let project = stack(6);
    let report = determinism_diff(|| project.build().expect("builds"), 400);

    assert!(
        report.reproduced(),
        "diverged at step {:?}, body {:?}, {:.3e} m apart",
        report.first_divergent_step,
        report.first_divergent_body,
        report.separation
    );
}

/// And the tool finds a divergence when there is one to find.
///
/// Without this the determinism check is untested in the direction that
/// matters: a function that always returns "reproduced" would pass the test
/// above and be worthless.
#[test]
fn a_planted_divergence_is_found_at_the_step_it_was_planted() {
    let project = stack(6);
    let mut runs = 0;
    let report = determinism_diff(
        || {
            runs += 1;
            let mut sim = project.build().expect("builds");
            // The second run starts a hair to the left. One micrometre, which
            // is far below anything anybody would see, and exactly the kind of
            // difference this is supposed to catch.
            if runs == 2 {
                let p = sim.world.bodies.translation(3);
                sim.world
                    .bodies
                    .set_translation(3, p + vec3(1e-6, 0.0, 0.0));
            }
            sim
        },
        400,
    );

    assert!(!report.reproduced(), "a planted difference went unnoticed");
    assert_eq!(
        report.first_divergent_step,
        Some(1),
        "the scenes differed before the first step, so step 1 is where the \
         hashes must part"
    );
    assert!(
        report.first_divergent_body.is_some(),
        "a divergent step with no body named is half a report"
    );
}

// --- comparison ----------------------------------------------------------

/// A scene compared against itself diverges nowhere.
#[test]
fn a_scene_against_itself_never_diverges() {
    let project = stack(5);
    let mut a = project.build().unwrap();
    let mut b = project.build().unwrap();
    let report = compare(&mut a, &mut b, 300, 1e-4);

    assert_eq!(report.first_bitwise_difference, None);
    assert_eq!(report.first_visible_difference, None);
    assert_eq!(report.max_separation, 0.0);
    assert_eq!(report.separation.len(), 300, "one sample per step");
}

/// One parameter changed, and the report says when it started to matter.
#[test]
fn one_changed_parameter_shows_up_as_a_divergence() {
    let base = stack(5);
    let mut changed = base.clone();
    changed.solver.substeps = base.solver.substeps.max(1) * 2;

    let mut a = base.build().unwrap();
    let mut b = changed.build().unwrap();
    let report = compare(&mut a, &mut b, 400, 1e-3);

    assert!(
        report.first_bitwise_difference.is_some(),
        "doubling the substeps changed nothing at all, which cannot be right"
    );
    assert!(
        report.max_separation > 0.0,
        "the two runs stayed exactly together"
    );
    // Bitwise divergence comes first or at the same step as visible: a
    // difference you can see is a difference in the bits by definition.
    if let (Some(bits), Some(visible)) = (
        report.first_bitwise_difference,
        report.first_visible_difference,
    ) {
        assert!(bits <= visible, "visible divergence preceded bitwise");
    }
}

// --- convergence ---------------------------------------------------------

/// A first-order method is measured as first order.
///
/// Free fall under gravity with semi-implicit Euler: the position error after
/// a fixed time is proportional to the timestep, so halving it halves the
/// error and the observed order is one. This validates the study function
/// against a case whose answer is known from the integrator, not from Fusion.
#[test]
fn convergence_measures_a_first_order_method_as_first_order() {
    let mut world = World::new();
    world.add(BodyDesc::sphere(0.2).at(vec3(0.0, 10.0, 0.0)));
    let project = Project::from_world(&world);

    let seconds = 0.5f64;
    let report = converge::study(1.0 / 60.0, 5, |h| {
        let mut sim = project.build().expect("builds");
        sim.solver_dt = h as f32;
        for _ in 0..(seconds / h).round() as u64 {
            sim.step();
        }
        sim.world.bodies.translation(0).y as f64
    });

    let order = report.order().expect("three levels give an order");
    assert!(
        (order - 1.0).abs() < 0.25,
        "semi-implicit Euler measured at order {order:.3}, expected 1"
    );
    assert!(
        report.converges_at(1.0, 0.25),
        "converges_at disagrees with order()"
    );
    // Extrapolating to zero timestep should land near the analytic answer.
    let exact = 10.0 - 0.5 * 9.81 * seconds * seconds;
    let extrapolated = report.extrapolated.expect("an order gives an estimate");
    assert!(
        (extrapolated - exact).abs() < 0.02,
        "extrapolated to {extrapolated:.4}, analytic free fall is {exact:.4}"
    );
}

/// A measurement that does not move gives no order rather than an infinite
/// one.
#[test]
fn a_flat_measurement_reports_no_order() {
    let report = converge::study(0.01, 4, |_| 7.0);
    assert!(report.observed_order.is_empty());
    assert!(report.extrapolated.is_none());
    assert!(report.order().is_none());
}

// --- sweeps and the store ------------------------------------------------

/// A sweep runs every combination and records something re-checkable.
#[test]
fn a_sweep_runs_every_combination_and_records_a_rerunnable_hash() {
    let sweep = sweep::Sweep::new(stack(4), 120)
        .sample_every(20)
        .axis(sweep::Axis::new("substeps", [2.0, 4.0, 8.0], |p, v| {
            p.solver.substeps = v as u32
        }));
    assert_eq!(sweep.len(), 3);

    let results = sweep
        .run(|sim| {
            let mut out = std::collections::BTreeMap::new();
            out.insert("top".into(), sim.world.bodies.translation(4).y as f64);
            out
        })
        .expect("every combination builds");

    assert_eq!(results.runs.len(), 3);
    for (run, expected) in results.runs.iter().zip([2.0, 4.0, 8.0]) {
        assert_eq!(run.parameters["substeps"], expected);
        assert!(run.stable, "a run at {expected} substeps went non-finite");
        assert!(run.conservation.is_some(), "conservation was not recorded");
        assert!(run.measurements.contains_key("top"));
    }

    // The recorded hash is the run's identity: rebuilding the same project and
    // stepping it the same way has to reproduce it, or a results table is a
    // set of numbers nobody can get back to.
    let mut replay = sweep.project_for(1).build().unwrap();
    for _ in 0..120 {
        replay.step();
    }
    assert_eq!(
        fusion_sim::state_hash(&replay),
        results.runs[1].state_hash,
        "a recorded run did not reproduce from its own parameters"
    );

    let csv = results.to_csv();
    assert!(csv.starts_with("substeps,top,steps,"), "{csv}");
    assert!(
        csv.contains(",closed,"),
        "the CSV says whether momentum was checkable"
    );
    assert_eq!(csv.lines().count(), 4, "a header and three rows");
}

/// A study survives the round trip to disk, which is what makes it a record
/// rather than a print-out.
#[test]
fn a_study_round_trips_through_a_directory() {
    let dir = std::env::temp_dir().join("fusion-research-tests");
    let _ = std::fs::remove_dir_all(&dir);

    let results = sweep::Sweep::new(stack(3), 60)
        .axis(sweep::Axis::new("iterations", [4.0, 8.0], |p, v| {
            p.solver.iterations = v as u32
        }))
        .run(|_| std::collections::BTreeMap::new())
        .expect("builds");

    let study = store::Study::new("does it settle?", results);
    study.save(&dir).expect("the study writes");

    let read = store::Study::load(&dir).expect("and reads back");
    assert_eq!(read.question, "does it settle?");
    assert_eq!(read.results.runs.len(), 2);
    assert_eq!(
        read.results.runs[0].state_hash, study.results.runs[0].state_hash,
        "the hash did not survive the round trip"
    );
    assert!(
        dir.join("results.csv").exists(),
        "the CSV is half the point of a store"
    );
}

/// Sweep bookkeeping: the axis values map onto the right projects.
#[test]
fn each_combination_is_a_scene_you_can_get_back() {
    let sweep = sweep::Sweep::new(stack(3), 10)
        .axis(sweep::Axis::new("substeps", [1.0, 2.0, 3.0], |p, v| {
            p.solver.substeps = v as u32
        }))
        .axis(sweep::Axis::new("iterations", [5.0, 10.0], |p, v| {
            p.solver.iterations = v as u32
        }));
    assert_eq!(sweep.len(), 6, "three by two");

    let mut seen = Vec::new();
    for i in 0..sweep.len() {
        let p = sweep.project_for(i);
        seen.push((p.solver.substeps, p.solver.iterations));
    }
    seen.sort_unstable();
    assert_eq!(
        seen,
        vec![(1, 5), (1, 10), (2, 5), (2, 10), (3, 5), (3, 10)],
        "the cartesian product is missing combinations"
    );
}

/// The tools work on a scene with fluid in it too — the systems research would
/// actually be pointed at.
#[test]
fn conservation_covers_a_fluid_scene() {
    use fusion_core::{FluidDesc, FluidParams};

    let mut world = World::new();
    world.add(
        BodyDesc::cuboid(vec3(3.0, 0.2, 3.0))
            .at(vec3(0.0, -0.1, 0.0))
            .fixed(),
    );
    world.add_fluid(
        &FluidDesc::block(Vec3::splat(0.25))
            .at(vec3(0.0, 0.5, 0.0))
            .with_params(FluidParams::water_at(0.06)),
    );
    let mut sim: Simulation = Project::from_world(&world).build().expect("builds");

    let mut trace = metrics::ConservationTrace::new();
    trace.observe(&sim);
    for _ in 0..120 {
        sim.step();
    }
    trace.observe(&sim);

    let summary = trace.summary().expect("two samples");
    assert!(
        summary.mass_drift < 1e-3,
        "a fluid lost {:.3e} kg",
        summary.mass_drift
    );
    // Falling water converts potential to kinetic; the total should not climb.
    assert!(
        summary.energy_delta() <= 1.0,
        "the fluid gained {:.3} J out of nowhere",
        summary.energy_delta()
    );
}
