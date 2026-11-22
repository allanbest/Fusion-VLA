//! Determinism, checkpoints, and replay.
//!
//! PLAN.md §34 tier (a) — same machine, same build, bit-identical — is not a
//! property the code has by accident. It is a property every summation in the
//! engine has to be written to preserve, and the only thing standing between
//! it and a slow silent regression is this file.
//!
//! The scenes here are chosen for having *cross-step memory*: a stack, whose
//! contacts inherit anchors from the previous step and whose broad phase keeps
//! a sorted order across frames, and a fluid, whose neighbour lists are built
//! from a hash grid. Both are places where an implementation that reproduces
//! positions but forgets its own bookkeeping diverges a few seconds later.
//!
//! The continuum is the third, and it went untested until v1.0 — which is
//! worth admitting rather than quietly fixing, because it was the mode with
//! the most state per point and the one where an accidental order dependence
//! would have been hardest to see. A material point carries a deformation
//! gradient, an affine velocity, a plastic ratio, a damage scalar and an
//! enthalpy, all of which persist across steps. The solver also chooses its
//! substep count *from the state it is about to advance*, so a difference too
//! small to see in a position can become a different number of substeps and
//! then a different simulation — though not in the scene below, where the
//! count is pinned by the stiffest material's wave speed and measured to stay
//! there.

use fusion_core::{
    BodyDesc, FluidDesc, FluidParams, Material, SolidDesc, SolidMaterial, SolidParams, World,
};
use fusion_math::{vec3, Quat, Vec3};
use fusion_sim::{state_hash, Determinism, DeterminismTier, Simulation, Timeline};

/// A scene with everything that remembers something across a step: a stack
/// (contact anchors, sleeping, broad-phase order), a joint, and a fluid
/// (neighbour lists, boundary occlusion).
fn awkward_scene() -> World {
    let mut world = World::new();
    world.add(
        BodyDesc::cuboid(vec3(4.0, 0.4, 4.0))
            .at(vec3(0.0, -0.2, 0.0))
            .fixed(),
    );
    // A tower, slightly rotated so nothing is axis-aligned by luck.
    for i in 0..6 {
        world.add(
            BodyDesc::cuboid(Vec3::splat(0.3))
                .at(vec3(0.8, 0.15 + i as f32 * 0.31, 0.0))
                .rotated(Quat::from_rotation_y(0.13 * i as f32))
                .with_material(Material::WOOD),
        );
    }
    // A pendulum, so the joint solver contributes too.
    let anchor = world.add(BodyDesc::sphere(0.1).at(vec3(-1.2, 1.4, 0.0)).fixed());
    let bob = world.add(BodyDesc::sphere(0.15).at(vec3(-1.2, 0.9, 0.0)));
    world.connect(
        anchor,
        bob,
        vec3(-1.2, 1.4, 0.0),
        fusion_core::JointKind::Ball,
    );
    // Walls and water.
    for (dx, dz) in [(1.0f32, 0.0f32), (-1.0, 0.0), (0.0, 1.0), (0.0, -1.0)] {
        let size = if dx != 0.0 {
            vec3(0.2, 1.2, 1.4)
        } else {
            vec3(1.4, 1.2, 0.2)
        };
        world.add(
            BodyDesc::cuboid(size)
                .at(vec3(dx * -0.6 + -0.1, 0.6, dz * 0.7))
                .fixed(),
        );
    }
    world.add_fluid(
        &FluidDesc::block(vec3(0.5, 0.4, 0.5))
            .at(vec3(-0.35, 0.5, 0.0))
            .with_params(FluidParams::water_at(0.06)),
    );
    world
}

#[test]
fn the_guarantee_is_stated_in_the_api() {
    // PLAN.md §9's principle applied to §34: a promise that lives only in
    // documentation is a promise nobody reads when it matters.
    let g = Determinism::GUARANTEE;
    assert_eq!(g.guaranteed, DeterminismTier::SameBuild);
    assert!(g.claims(DeterminismTier::SameBuild));
    assert!(!g.claims(DeterminismTier::CrossPlatformBounded));
    assert!(!g.claims(DeterminismTier::CrossPlatformExact));
    assert_eq!(g.declined, DeterminismTier::CrossPlatformExact);
}

#[test]
fn two_runs_of_the_same_scene_hash_the_same() {
    let mut a = Simulation::new(awkward_scene());
    let mut b = Simulation::new(awkward_scene());
    assert_eq!(state_hash(&a), state_hash(&b));
    for step in 0..240 {
        a.step();
        b.step();
        assert_eq!(
            state_hash(&a),
            state_hash(&b),
            "diverged at step {step}, t = {:.4}",
            a.time()
        );
    }
}

#[test]
fn the_hash_notices_a_change_that_small() {
    // A fingerprint that could not see one micron would be no use as a
    // divergence detector, which is the only job it has.
    let mut a = Simulation::new(awkward_scene());
    let mut b = Simulation::new(awkward_scene());
    a.run_frames(20, |_, _| {});
    b.run_frames(20, |_, _| {});
    assert_eq!(state_hash(&a), state_hash(&b));

    b.world.bodies.position[1].x += 1e-6;
    assert_ne!(state_hash(&a), state_hash(&b));
}

#[test]
fn replay_from_a_checkpoint_is_bit_identical() {
    // The claim rewind rests on. Run once and record the hash at every step;
    // then go back to a checkpoint and run forward again, comparing.
    //
    // This is the test that proves a checkpoint captures the contact anchors
    // and the broad-phase order and not merely the positions. Capturing only
    // the world passes the first few steps and drifts after that, which is
    // exactly the failure mode a weaker test would miss.
    let mut sim = Simulation::new(awkward_scene());
    let mut hashes = Vec::new();
    let mut checkpoint = None;
    for step in 0..300 {
        sim.step();
        hashes.push(state_hash(&sim));
        if step == 99 {
            checkpoint = Some(sim.checkpoint());
        }
    }
    let checkpoint = checkpoint.unwrap();

    sim.restore(&checkpoint);
    assert_eq!(sim.step_index(), 100);
    assert_eq!(
        state_hash(&sim),
        hashes[99],
        "restore did not land on the state"
    );

    for (step, expected) in hashes.iter().enumerate().skip(100) {
        sim.step();
        assert_eq!(
            state_hash(&sim),
            *expected,
            "replay diverged {} steps after the checkpoint",
            step - 99
        );
    }
}

#[test]
fn a_world_only_checkpoint_diverges() {
    // The claim `checkpoint.rs` makes, demonstrated rather than asserted: a
    // checkpoint has to capture the solver's cross-step bookkeeping — contact
    // anchors, broad-phase order, island membership — and not merely where
    // everything is and how fast it is going.
    //
    // The checkpoint is taken at step 40, while the stack is still settling
    // and contacts are forming. That matters: a world-only restore is exactly
    // right for a scene at rest and wrong on the very next step for a scene in
    // contact, so a test that checkpointed a quiet moment would pass and prove
    // nothing. The reference run is stepped to a completely different time
    // first, so the pipeline it holds is genuinely stale rather than
    // coincidentally correct.
    const AT: usize = 40;
    const TOTAL: usize = 400;

    let mut reference = Simulation::new(awkward_scene());
    let mut hashes = Vec::new();
    let mut checkpoint = None;
    for step in 0..TOTAL {
        reference.step();
        hashes.push(state_hash(&reference));
        if step == AT {
            checkpoint = Some(reference.checkpoint());
        }
    }
    let checkpoint = checkpoint.unwrap();

    // A full restore into a simulation whose pipeline is 360 steps out of date
    // still reproduces the original run exactly, to the last step.
    let mut sim = Simulation::new(awkward_scene());
    for _ in 0..(TOTAL - 1) {
        sim.step();
    }
    sim.restore(&checkpoint);
    for (step, expected) in hashes.iter().enumerate().skip(AT + 1) {
        sim.step();
        assert_eq!(
            state_hash(&sim),
            *expected,
            "a full restore diverged {} steps in",
            step - AT
        );
    }

    // The world-only restore starts from identical positions and velocities
    // and does not survive a single step.
    let mut sim = Simulation::new(awkward_scene());
    for _ in 0..(TOTAL - 1) {
        sim.step();
    }
    sim.restore_physical_state_only(&checkpoint);
    assert_eq!(
        state_hash(&sim),
        hashes[AT],
        "the two restores should agree about the state itself"
    );

    let mut diverged_at = None;
    for (step, expected) in hashes.iter().enumerate().skip(AT + 1) {
        sim.step();
        if state_hash(&sim) != *expected {
            diverged_at = Some(step - AT);
            break;
        }
    }
    let diverged_at = diverged_at.expect(
        "a world-only restore reproduced the run — if the solver stopped \
         carrying state across steps, the pipeline in Checkpoint can go",
    );
    assert_eq!(
        diverged_at, 1,
        "divergence took {diverged_at} steps rather than appearing immediately"
    );
}

#[test]
fn restoring_twice_gives_the_same_answer_twice() {
    // Rewind is not a one-shot: a user drags the scrubber back and forth.
    let mut sim = Simulation::new(awkward_scene());
    sim.run_frames(30, |_, _| {});
    let checkpoint = sim.checkpoint();

    let mut results = Vec::new();
    for _ in 0..3 {
        sim.restore(&checkpoint);
        for _ in 0..50 {
            sim.step();
        }
        results.push(state_hash(&sim));
    }
    assert_eq!(results[0], results[1]);
    assert_eq!(results[1], results[2]);
}

#[test]
fn a_checkpoint_restores_the_clock_as_well_as_the_state() {
    let mut sim = Simulation::new(awkward_scene());
    sim.run_frames(45, |_, _| {});
    let checkpoint = sim.checkpoint();
    let (time, step, frame) = (sim.time(), sim.step_index(), sim.frame_index());

    sim.run_frames(20, |_, _| {});
    assert_ne!(sim.step_index(), step);

    sim.restore(&checkpoint);
    assert_eq!(sim.time(), time);
    assert_eq!(sim.step_index(), step);
    assert_eq!(sim.frame_index(), frame);
    assert_eq!(checkpoint.time(), time);
    assert_eq!(checkpoint.step_index(), step);
}

#[test]
fn a_timeline_seeks_backwards_and_lands_on_the_state_it_recorded() {
    let mut sim = Simulation::new(awkward_scene());
    let mut timeline = Timeline::new(0.25);

    let mut at_second = Vec::new();
    timeline.observe(&sim);
    for _ in 0..(240 * 2) {
        sim.step();
        timeline.observe(&sim);
        at_second.push((sim.time(), state_hash(&sim)));
    }
    assert!(
        timeline.len() >= 8,
        "recorded only {} checkpoints",
        timeline.len()
    );

    // Seek to times the run passed through, and check we land where the
    // original run was at that time. Backwards and forwards, interleaved,
    // because a scrubber is dragged both ways.
    for target in [1.5f64, 0.75, 1.9, 0.3] {
        let backwards = target < sim.time();
        let result = timeline.seek(&mut sim, target);
        assert!(result.exact, "could not reach {target}");
        assert_eq!(
            result.restored_from.is_some(),
            backwards,
            "seeking {} to {target} {} restore",
            if backwards { "back" } else { "forward" },
            if backwards { "should" } else { "should not" }
        );
        let (time, hash) = at_second
            .iter()
            .rev()
            .find(|(t, _)| *t <= sim.time() + 1e-9)
            .copied()
            .unwrap();
        assert!((time - sim.time()).abs() < 1e-9);
        assert_eq!(
            hash,
            state_hash(&sim),
            "seek to {target} landed on a different state than the original run"
        );
    }
}

#[test]
fn seeking_forward_does_not_restore_anything() {
    // Going forward is just stepping. Restoring first would be correct and
    // would throw away work the simulation has already done.
    let mut sim = Simulation::new(awkward_scene());
    let mut timeline = Timeline::new(0.25);
    timeline.observe(&sim);
    for _ in 0..120 {
        sim.step();
        timeline.observe(&sim);
    }
    let before = sim.step_index();

    let target = sim.time() + 0.5;
    let result = timeline.seek(&mut sim, target);
    assert_eq!(result.restored_from, None);
    assert_eq!(result.steps_replayed, sim.step_index() - before);
    assert!(result.steps_replayed > 0);
}

#[test]
fn seeking_backwards_costs_at_most_one_interval_of_replay() {
    // The whole trade of §17: memory for recompute, with the recompute bounded
    // by the interval. A backward seek that could cost the whole run would make
    // the scrubber unusable exactly when the run is long.
    //
    // Forward seeks are not bounded and cannot be. A simulation is causal, so
    // moving forward one second *is* simulating one second — there is no
    // shortcut, and a checkpoint ahead of where you are does not exist yet.
    let interval = 0.25;
    let mut sim = Simulation::new(awkward_scene());
    let mut timeline = Timeline::new(interval);
    timeline.observe(&sim);
    for _ in 0..(240 * 2) {
        sim.step();
        timeline.observe(&sim);
    }

    let max_steps = (interval / sim.solver_dt as f64).ceil() as u64 + 1;
    // Descending, so every one of these is a seek backwards.
    for target in [1.9f64, 1.74, 1.1, 0.4, 0.05] {
        assert!(target < sim.time(), "target {target} is not behind us");
        let result = timeline.seek(&mut sim, target);
        assert!(result.restored_from.is_some());
        assert!(
            result.steps_replayed <= max_steps,
            "seek back to {target} replayed {} steps; one interval is {max_steps}",
            result.steps_replayed
        );
    }
}

#[test]
fn seeking_before_the_first_checkpoint_fails_rather_than_guessing() {
    let mut sim = Simulation::new(awkward_scene());
    let mut timeline = Timeline::new(0.25);
    // Start recording late, so there is no checkpoint covering the beginning.
    for _ in 0..240 {
        sim.step();
    }
    timeline.observe(&sim);
    for _ in 0..240 {
        sim.step();
        timeline.observe(&sim);
    }

    let result = timeline.seek(&mut sim, 0.1);
    assert!(!result.exact);
    assert_eq!(result.restored_from, None);
    assert_eq!(result.steps_replayed, 0);
    assert!(timeline.earliest().unwrap() >= 0.9);
}

#[test]
fn a_timeline_over_budget_thins_instead_of_forgetting_the_beginning() {
    // Dropping the oldest is the obvious policy and the wrong one: the start
    // of a run is where a user scrubs to when they are looking for when
    // something went wrong.
    let mut sim = Simulation::new(awkward_scene());
    let mut timeline = Timeline::new(0.1).with_budget(8);
    timeline.observe(&sim);
    let first = timeline.earliest().unwrap();

    for _ in 0..(240 * 3) {
        sim.step();
        timeline.observe(&sim);
    }

    assert!(timeline.len() <= 8);
    assert_eq!(
        timeline.earliest().unwrap(),
        first,
        "thinning lost the beginning of the run"
    );
    assert!(
        timeline.interval() > 0.1,
        "interval should have grown as the timeline thinned"
    );
    // And it still seeks.
    let result = timeline.seek(&mut sim, 0.5);
    assert!(result.exact);
}

#[test]
fn a_memory_budget_turns_into_a_checkpoint_count() {
    let mut sim = Simulation::new(awkward_scene());
    let mut timeline = Timeline::new(0.05).with_memory_budget(2 * 1024 * 1024);
    timeline.observe(&sim);
    for _ in 0..(240 * 2) {
        sim.step();
        timeline.observe(&sim);
    }
    assert!(timeline.len() >= 2);
    assert!(
        timeline.total_bytes() <= 4 * 1024 * 1024,
        "held {} bytes against a 2 MiB budget",
        timeline.total_bytes()
    );
}

#[test]
fn truncating_drops_the_future_that_is_no_longer_going_to_happen() {
    let mut sim = Simulation::new(awkward_scene());
    let mut timeline = Timeline::new(0.2);
    timeline.observe(&sim);
    for _ in 0..(240 * 2) {
        sim.step();
        timeline.observe(&sim);
    }
    assert!(timeline.latest().unwrap() > 1.0);

    timeline.truncate_after(0.5);
    assert!(timeline.latest().unwrap() <= 0.5);
    assert!(timeline.earliest().unwrap() < 0.2);
}

#[test]
#[should_panic(expected = "a checkpoint belongs to the simulation it was taken from")]
fn restoring_into_the_wrong_world_is_refused() {
    let mut small = Simulation::new(World::new());
    let big = Simulation::new(awkward_scene());
    small.restore(&big.checkpoint());
}

#[test]
fn reset_and_a_checkpoint_of_the_start_agree() {
    // Two routes back to t = 0 that had better not disagree: rebuilding the
    // state from the scene, and restoring a checkpoint taken before the first
    // step. If they differ, one of them is not doing what it says.
    let mut sim = Simulation::new(awkward_scene());
    let start = sim.checkpoint();
    let hash_at_start = state_hash(&sim);

    sim.run_frames(30, |_, _| {});
    sim.reset();
    let after_reset = state_hash(&sim);

    sim.run_frames(30, |_, _| {});
    sim.restore(&start);
    let after_restore = state_hash(&sim);

    assert_eq!(hash_at_start, after_reset);
    assert_eq!(hash_at_start, after_restore);
}

/// The deep grid's version of `awkward_scene`: everything in the continuum
/// that remembers something, plus a rigid body for the coupling to carry.
///
/// Three materials rather than one, because the state that persists differs by
/// material. Jelly carries a deformation gradient and nothing else; snow adds
/// a plastic ratio through the return map and an enthalpy through the thermal
/// solve, whose diffusion sub-cycle count is itself chosen from the state;
/// porcelain adds a damage scalar that is monotone, so a single step's
/// disagreement can never be undone by a later one.
fn continuum_scene() -> World {
    let mut world = World::new();
    world.add(
        BodyDesc::cuboid(vec3(4.0, 0.4, 4.0))
            .at(vec3(0.0, -0.2, 0.0))
            .with_material(Material::CONCRETE)
            .fixed(),
    );
    // A rigid box in the middle of it, so the two-way coupling contributes:
    // the impulse the boundary condition takes off the grid is handed to a
    // body, and a body that moves changes the boundary next step.
    world.add(
        BodyDesc::cuboid(Vec3::splat(0.2))
            .at(vec3(0.0, 0.12, 0.0))
            .with_material(Material::WOOD),
    );
    for (index, material) in [
        SolidMaterial::JELLY,
        SolidMaterial::SNOW,
        SolidMaterial::PORCELAIN,
    ]
    .into_iter()
    .enumerate()
    {
        world.add_solid(
            &SolidDesc::block(Vec3::splat(0.24))
                .at(vec3(index as f32 * 0.5 - 0.5, 0.6, 0.0))
                // Sideways, so the blocks meet the floor and the box off
                // square and nothing is symmetric by luck.
                .with_velocity(vec3(0.3, 0.0, 0.15))
                .with_params(SolidParams::at(0.04).of(material)),
        );
    }
    world
}

/// Two runs of a continuum scene agree bit for bit, every step.
#[test]
fn two_runs_of_the_same_continuum_scene_hash_the_same() {
    let mut a = Simulation::new(continuum_scene());
    let mut b = Simulation::new(continuum_scene());
    assert_eq!(state_hash(&a), state_hash(&b));
    let mut substeps = std::collections::BTreeSet::new();
    for step in 0..240 {
        a.step();
        b.step();
        substeps.insert(a.mpm_report().substeps);
        assert_eq!(
            state_hash(&a),
            state_hash(&b),
            "diverged at step {step}, t = {:.4}",
            a.time()
        );
    }
    // And the run was worth hashing. A scene that never deformed would pass
    // every assertion above while proving nothing about the deep grid, so what
    // it actually exercised is pinned here: the blocks deform, the brittle one
    // breaks, and the substep count moves — which is the path dependence that
    // makes this test worth having, because a difference too small to see in a
    // position becomes a different number of substeps and then a different
    // simulation.
    let report = a.mpm_report();
    assert!(
        report.peak_strain > 0.05,
        "the material barely deformed: peak strain {:.4}",
        report.peak_strain
    );
    assert!(
        report.broken > 0.1,
        "nothing fractured, so the damage model contributed nothing: {:.3}",
        report.broken
    );
    // The substep count is *not* asserted to vary, because measuring it says
    // it does not: this scene's count is pinned at 7 by the stiffest material's
    // wave speed the whole way through, and the advection bound — the half a
    // cell per substep that would make it move — never binds at these speeds.
    // Worth writing down, because it means the sharpest edge of the path
    // dependence is untested here and would need a scene thrown much harder.
    assert_eq!(
        substeps,
        std::collections::BTreeSet::from([7]),
        "the substep count moved, which is now worth a test of its own"
    );
}

/// And the fingerprint can see a deformation gradient, not just a position.
///
/// The claim `fusion_sim::determinism` makes in prose — two material points in
/// the same place moving the same way are different states if one has been
/// stretched — checked rather than read. A hash that stopped at the particle
/// set would call a divergent continuum reproducible, and every tool resting
/// on it would agree.
#[test]
fn the_hash_notices_a_deformation_that_small() {
    let mut a = Simulation::new(continuum_scene());
    let mut b = Simulation::new(continuum_scene());
    a.run_frames(20, |_, _| {});
    b.run_frames(20, |_, _| {});
    assert_eq!(state_hash(&a), state_hash(&b));

    // Nothing moves, nothing speeds up: one point is a millionth more
    // stretched than it was.
    b.world.solids[0].deformation[0].x_axis.x += 1e-6;
    assert_ne!(
        state_hash(&a),
        state_hash(&b),
        "the fingerprint cannot see the deformation gradient"
    );

    // Likewise the fields v0.9 added, each of which is state a step reads.
    let mut c = Simulation::new(continuum_scene());
    c.run_frames(20, |_, _| {});
    c.world.solids[1].enthalpy[0] += 1e-3;
    assert_ne!(state_hash(&a), state_hash(&c), "enthalpy is not hashed");

    let mut d = Simulation::new(continuum_scene());
    d.run_frames(20, |_, _| {});
    d.world.solids[2].damage[0] += 1e-4;
    assert_ne!(state_hash(&a), state_hash(&d), "damage is not hashed");
}

/// A continuum replays from a checkpoint exactly, which is what the scrubber
/// rests on for a solid the same way it does for a stack.
#[test]
fn a_continuum_replays_from_a_checkpoint() {
    let mut original = Simulation::new(continuum_scene());
    let mut hashes = Vec::new();
    let mut checkpoint = None;
    for step in 0..200 {
        original.step();
        hashes.push(state_hash(&original));
        if step == 80 {
            checkpoint = Some(original.checkpoint());
        }
    }

    let mut replay = Simulation::new(continuum_scene());
    replay.restore(&checkpoint.expect("a checkpoint at step 80"));
    assert_eq!(state_hash(&replay), hashes[80], "restored to a wrong state");
    for (step, expected) in hashes.iter().enumerate().skip(81) {
        replay.step();
        assert_eq!(
            state_hash(&replay),
            *expected,
            "replay diverged at step {step}"
        );
    }
}
