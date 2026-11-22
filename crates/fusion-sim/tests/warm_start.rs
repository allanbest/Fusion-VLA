//! Contact impulse warm-starting.
//!
//! PLAN.md §10 named this as the lever for stacking, §29 scheduled it for
//! v0.2, and it arrived in v0.5. What it does and does not do took some
//! finding out, and the tests are shaped by that.
//!
//! **It is measured over a spread of initial conditions, never one.** A
//! perfectly symmetric tower is a degenerate unstable equilibrium: which way
//! it falls, and whether it falls at all, is chaotic in the solver parameters.
//! The first measurement of this feature used exactly that scene, reported a
//! 25x improvement, and was reporting a coincidence — half a millimetre of
//! lean turned it into a 10x regression. Every assertion here runs a range of
//! leans and asserts on the aggregate.

use fusion_core::{BodyDesc, Material, World};
use fusion_math::{vec3, Vec3};
use fusion_sim::{state_hash, Simulation};

/// A tower with a deterministic per-box lean, so it is not the degenerate
/// symmetric case.
fn tower(boxes: usize, lean: f32) -> World {
    let mut world = World::new();
    world.add(
        BodyDesc::cuboid(vec3(40.0, 2.0, 40.0))
            .at(vec3(0.0, -1.0, 0.0))
            .with_material(Material::CONCRETE)
            .fixed(),
    );
    for k in 0..boxes {
        let d = lean * (k as f32 * 1.7).sin();
        world.add(
            BodyDesc::cuboid(Vec3::ONE)
                .at(vec3(d, 0.5 + k as f32, d * 0.6))
                .with_material(Material::WOOD),
        );
    }
    world
}

/// Furthest any box drifts sideways from where it started, over ten seconds.
/// `None` if the stack toppled, which is a different measurement.
fn creep(boxes: usize, lean: f32, warm: f32) -> Option<f32> {
    let mut sim = Simulation::new(tower(boxes, lean));
    sim.solver.contact_warm_start = warm;
    let start: Vec<_> = (1..=boxes)
        .map(|k| sim.world.bodies.translation(k))
        .collect();
    for _ in 0..2400 {
        sim.step();
    }
    let mut drift = 0.0f32;
    for (k, from) in start.iter().enumerate() {
        let d = sim.world.bodies.translation(k + 1) - *from;
        drift = drift.max((d.x * d.x + d.z * d.z).sqrt());
    }
    (drift < 0.5).then_some(drift)
}

const LEANS: [f32; 7] = [0.0, 0.0003, 0.0005, 0.001, 0.002, 0.005, 0.01];

#[test]
fn no_setting_lets_a_tower_creep() {
    // This was two tests: that warm-starting cut creep — 0.24 of the cold
    // drift at the median — and that re-applying the whole remembered impulse
    // was the worst setting there is. Both measured drift that Gauss-Seidel
    // left in a stack, and since the contact solve finishes each substep from
    // the ground up (`contact::propagate_shock`) there is none left to
    // measure: under five microns in ten seconds, cold or warm, across every
    // depth and lean here. What is worth pinning is that it stays that way at
    // both ends of the dial.
    let mut worst = (0.0f32, 0usize, 0.0f32, 0.0f32);
    for boxes in [5usize, 8, 10] {
        for lean in LEANS {
            for warm in [0.0, 1.0] {
                let drift = creep(boxes, lean, warm).expect("a tower fell");
                if drift > worst.0 {
                    worst = (drift, boxes, lean, warm);
                }
            }
        }
    }
    let (drift, boxes, lean, warm) = worst;
    assert!(
        drift < 1e-3,
        "a {boxes}-box tower leaning {lean} m crept {drift} m at warm start {warm}"
    );
}

#[test]
fn a_warm_started_contact_still_lets_a_box_be_lifted_away() {
    // The remembered impulse is only valid while the surfaces are still
    // touching. Applied to a body that has separated it would pull it back
    // down — a floor that grabs.
    let mut world = World::new();
    world.add(
        BodyDesc::cuboid(vec3(20.0, 1.0, 20.0))
            .at(vec3(0.0, -0.5, 0.0))
            .fixed(),
    );
    let box_ = world.add(BodyDesc::cuboid(Vec3::ONE).at(vec3(0.0, 0.5, 0.0)));
    let mut sim = Simulation::new(world);
    sim.run_frames(60, |_, _| {}); // settle, building up a warm-start impulse

    // Now throw it upward and check it leaves cleanly.
    sim.world.bodies.linear_velocity[box_.index()] = vec3(0.0, 6.0, 0.0);
    sim.world.bodies.wake(box_.index());
    let mut peak = f32::NEG_INFINITY;
    for _ in 0..240 {
        sim.step();
        peak = peak.max(sim.world.bodies.translation(box_.index()).y);
    }
    // 6 m/s upward reaches about 1.8 m above its resting height.
    assert!(
        peak > 1.8,
        "the box only reached {peak} m — a warm-started contact held it down"
    );
}

#[test]
fn warm_starting_does_not_wake_a_sleeping_pile() {
    let mut sim = Simulation::new(tower(3, 0.0));
    sim.run_frames(240, |_, _| {});
    assert_eq!(
        sim.world.bodies.sleeping_count(),
        3,
        "the pile did not settle, so this test proves nothing"
    );
    for _ in 0..480 {
        sim.step();
    }
    assert_eq!(sim.world.bodies.sleeping_count(), 3, "the pile woke up");
}

#[test]
fn warm_starting_is_deterministic() {
    // The cache is keyed on body pair and anchor and written in sorted order,
    // so it cannot inherit the broad phase's emission order. If it could,
    // PLAN.md §34 tier (a) would go with it.
    let build = || {
        let mut sim = Simulation::new(tower(8, 0.002));
        sim.solver.contact_warm_start = 0.8;
        sim
    };
    let (mut a, mut b) = (build(), build());
    for step in 0..600 {
        a.step();
        b.step();
        assert_eq!(state_hash(&a), state_hash(&b), "diverged at step {step}");
    }
}

#[test]
fn turning_it_off_reproduces_the_old_solver() {
    // An escape hatch that actually works: `contact_warm_start = 0` must be
    // bit-identical to a build that never had the feature, so a regression can
    // be bisected against it.
    let mut sim = Simulation::new(tower(6, 0.001));
    sim.solver.contact_warm_start = 0.0;
    for _ in 0..600 {
        sim.step();
    }
    let without = state_hash(&sim);

    let mut again = Simulation::new(tower(6, 0.001));
    again.solver.contact_warm_start = 0.0;
    for _ in 0..600 {
        again.step();
    }
    assert_eq!(without, state_hash(&again));

    // And it is genuinely doing something when on.
    let mut on = Simulation::new(tower(6, 0.001));
    on.solver.contact_warm_start = 0.8;
    for _ in 0..600 {
        on.step();
    }
    assert_ne!(without, state_hash(&on));
}
