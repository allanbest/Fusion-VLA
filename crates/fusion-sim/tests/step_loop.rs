//! End-to-end simulation tests.
//!
//! PLAN.md §30 asks v0.1 for three things: GJK/EPA against analytic cases
//! (covered in `fusion-collide`), an energy-drift regression on a bouncing
//! scene, and a stacking smoke test asserting N boxes do not explode. All
//! three are here plus the checks that catch the silent failures — a body
//! sinking through the floor, momentum leaking, a run not reproducing.

use fusion_core::{BodyDesc, Material, World};
use fusion_math::{vec3, Vec3};
use fusion_sim::{Mode, Simulation};

/// Ground slab whose top surface sits at y = 0.
fn ground(world: &mut World) {
    world.add(
        BodyDesc::cuboid(vec3(80.0, 2.0, 80.0))
            .at(vec3(0.0, -1.0, 0.0))
            .with_material(Material::CONCRETE)
            .fixed(),
    );
}

fn run_seconds(sim: &mut Simulation, seconds: f64) {
    let steps = (seconds / sim.solver_dt as f64).round() as u64;
    for _ in 0..steps {
        sim.step();
    }
}

#[test]
fn free_fall_matches_the_closed_form() {
    let mut world = World::new();
    world.angular_damping = 0.0;
    let ball = world.add(BodyDesc::sphere(0.5).at(vec3(0.0, 100.0, 0.0)));
    let mut sim = Simulation::new(world);

    run_seconds(&mut sim, 1.0);

    // Semi-implicit Euler lands one half-step below the analytic parabola:
    // y = y0 - g*t^2/2 - g*h*t/2. With h = 1/240 that is about 2 cm.
    let expected = 100.0 - 0.5 * 9.81 * 1.0;
    let y = sim.world.bodies.translation(ball.index()).y;
    assert!(
        (y - expected).abs() < 0.03,
        "expected about {expected}, got {y}"
    );
    let vy = sim.world.bodies.linear_velocity[ball.index()].y;
    assert!(
        (vy + 9.81).abs() < 0.05,
        "velocity after 1 s should be -9.81, got {vy}"
    );
}

#[test]
fn a_sphere_comes_to_rest_on_the_ground_without_sinking() {
    let mut world = World::new();
    ground(&mut world);
    let ball = world.add(BodyDesc::sphere(0.5).at(vec3(0.0, 4.0, 0.0)));
    let mut sim = Simulation::new(world);

    run_seconds(&mut sim, 4.0);

    let p = sim.world.bodies.translation(ball.index());
    let v = sim.world.bodies.linear_velocity[ball.index()];
    assert!(sim.is_stable());
    assert!(
        (p.y - 0.5).abs() < 0.01,
        "should rest with its centre one radius up, got y={}",
        p.y
    );
    assert!(
        v.length() < 0.05,
        "should be at rest, speed is {}",
        v.length()
    );
}

#[test]
fn a_box_lands_flat_and_stays_put() {
    let mut world = World::new();
    ground(&mut world);
    let b = world.add(
        BodyDesc::cuboid(Vec3::ONE)
            .at(vec3(0.0, 3.0, 0.0))
            .with_material(Material::WOOD),
    );
    let mut sim = Simulation::new(world);

    run_seconds(&mut sim, 4.0);

    let p = sim.world.bodies.translation(b.index());
    assert!(sim.is_stable());
    assert!(
        (p.y - 0.5).abs() < 0.02,
        "resting height should be half a side, got {}",
        p.y
    );
    // Two properties, and the second is the strict one.
    //
    // It settles near where it was dropped — a few centimetres, accumulated
    // while the impact is still resolving and the contact is sliding faster
    // than static friction's threshold. Before persistent contact anchors this
    // was 0.4 m and still growing.
    assert!(
        p.x.abs() < 0.1 && p.z.abs() < 0.1,
        "a box dropped flat wandered too far: {p:?}"
    );
    // And then it genuinely stops, rather than creeping.
    let settled = sim.world.bodies.translation(b.index());
    run_seconds(&mut sim, 2.0);
    let later = sim.world.bodies.translation(b.index());
    assert!(
        (settled - later).length() < 1e-4,
        "the box crept {:.6} m after settling",
        (settled - later).length()
    );
    let tilt = sim.world.bodies.orientation[b.index()];
    assert!(
        tilt.approx_eq(fusion_math::Quat::IDENTITY, 1e-3),
        "it should not have tipped: {tilt:?}"
    );
}

#[test]
fn a_stack_of_boxes_does_not_explode() {
    // PLAN.md §30's stacking smoke test: ten boxes, resting on each other.
    //
    // Resting rather than gapped, on purpose. Dropping ten boxes onto each
    // other simultaneously is a chain impact, and how tidily the pile survives
    // one is a *quality* number that moves with solver settings — it belongs
    // in the benchmark suite, which reports it, not in a test, which would
    // have to assert some particular value of it. What a test can assert is
    // the invariant: a stack that is already at rest stays where it is, stays
    // finite, and does not gain energy.
    let mut world = World::new();
    ground(&mut world);
    // Five boxes. Deeper towers creep laterally without contact persistence,
    // and how much they creep is a quality number that moves with solver
    // settings — the benchmark suite reports it for 10, 20 and 50. What a test
    // can assert is the invariant: a stack that is already at rest stays put,
    // stays finite, and does not gain energy.
    const N: usize = 10;
    let mut handles = Vec::new();
    for k in 0..N {
        handles.push(
            world.add(
                BodyDesc::cuboid(Vec3::ONE)
                    .at(vec3(0.0, 0.5 + k as f32 * 1.02, 0.0))
                    .with_material(Material::WOOD),
            ),
        );
    }
    let mut sim = Simulation::new(world);
    sim.solver.substeps = 8;

    run_seconds(&mut sim, 5.0);

    assert!(sim.is_stable(), "the state went non-finite");
    // The distinction that matters: a stack that shuffles while shedding
    // energy is a stack settling. One that gains energy is a solver failing.
    assert!(
        sim.diagnostics.relative_drift() <= 0.0,
        "the stack gained {:.2}% energy",
        sim.diagnostics.relative_drift() * 100.0
    );
    for (k, h) in handles.iter().enumerate() {
        let p = sim.world.bodies.translation(h.index());
        let expected = 0.5 + k as f32;
        assert!(
            (p.y - expected).abs() < 0.03,
            "box {k} settled at y={} instead of about {expected}",
            p.y
        );
        assert!(
            // It drifted about 6 cm over five seconds, from the Gauss-Seidel
            // ordering of contact points accumulating a small asymmetry.
            // Since the contact solve finishes each substep from the ground
            // up (`propagate_shock`), it drifts 0.04 mm. A millimetre is the
            // headroom over that.
            p.x.abs() < 1e-3 && p.z.abs() < 1e-3,
            "box {k} slid out of the stack to {p:?}"
        );
    }
}

#[test]
fn a_bouncing_scene_never_gains_energy() {
    // The drift regression. XPBD is not energy-conserving (its `SolverInfo`
    // says so), but it must not *gain* energy: that is the signature of a
    // contact solver fighting the integrator, and it is what turns a stack
    // into an explosion.
    let mut world = World::new();
    ground(&mut world);
    for k in 0..12 {
        let a = k as f32 * 0.7;
        world.add(
            BodyDesc::sphere(0.35)
                .at(vec3(a.cos() * 1.5, 1.0 + k as f32 * 0.8, a.sin() * 1.5))
                .with_velocity(vec3(a.sin(), 0.0, a.cos()) * 2.0),
        );
    }
    let mut sim = Simulation::new(world);
    let start_energy = sim.world.mechanical_energy();

    run_seconds(&mut sim, 6.0);

    assert!(sim.is_stable());
    let end_energy = sim.world.mechanical_energy();
    assert!(
        end_energy <= start_energy + 1e-3 * start_energy.abs(),
        "energy rose from {start_energy} to {end_energy}"
    );
    // No single step may inject a meaningful amount either — a run can end
    // lower overall while still having spiked violently in the middle.
    let worst = sim.diagnostics.worst_energy_gain;
    assert!(
        worst < 0.05 * start_energy.abs().max(1.0),
        "largest single-step energy gain was {worst} J"
    );
}

#[test]
fn momentum_is_conserved_when_nothing_external_acts() {
    // Two spheres approaching head-on in free space. Every XPBD correction is
    // an equal and opposite impulse pair, so linear momentum is conserved by
    // construction — this test is what proves the implementation actually is.
    let mut world = World::new().with_gravity(Vec3::ZERO);
    world.angular_damping = 0.0;
    world.add(
        BodyDesc::sphere(0.5)
            .at(vec3(-2.0, 0.0, 0.0))
            .with_velocity(vec3(3.0, 0.0, 0.0)),
    );
    world.add(
        BodyDesc::sphere(0.5)
            .at(vec3(2.0, 0.3, 0.0))
            .with_velocity(vec3(-1.0, 0.0, 0.0))
            .with_mass(2.0),
    );
    let mut sim = Simulation::new(world);
    let p0 = sim.world.bodies.linear_momentum();

    run_seconds(&mut sim, 3.0);

    let p1 = sim.world.bodies.linear_momentum();
    assert!(
        (p1 - p0).length() < 1e-3 * p0.length().max(1.0),
        "momentum drifted from {p0:?} to {p1:?}"
    );
}

#[test]
fn nothing_tunnels_through_the_floor() {
    // Continuous collision detection is explicitly out of v0.1 (§30), so this
    // asserts the discrete case holds at ordinary speeds, and documents where
    // the limit is rather than pretending there is none.
    let mut world = World::new();
    ground(&mut world);
    let mut handles = Vec::new();
    for k in 0..8 {
        handles.push(
            world.add(
                BodyDesc::sphere(0.4)
                    .at(vec3(k as f32 * 1.5 - 5.0, 8.0, 0.0))
                    .with_velocity(vec3(0.0, -12.0, 0.0)),
            ),
        );
    }
    let mut sim = Simulation::new(world);

    run_seconds(&mut sim, 3.0);

    for (k, h) in handles.iter().enumerate() {
        let y = sim.world.bodies.translation(h.index()).y;
        assert!(y > 0.3, "sphere {k} ended below the floor at y={y}");
    }
}

#[test]
fn the_same_scene_run_twice_gives_bit_identical_results() {
    // PLAN.md §34 defers the determinism *tier* to v0.4, but tier (a) —
    // same machine, same build, bit-identical — is the one that rewind and
    // replay depend on, and it is free if nothing in the step depends on
    // iteration order over a hash map or on wall-clock time. Asserting it now
    // is how it stays true.
    fn build() -> Simulation {
        let mut world = World::new();
        ground(&mut world);
        for k in 0..24 {
            let a = k as f32 * 1.1;
            world.add(
                BodyDesc::cuboid(Vec3::splat(0.6))
                    .at(vec3(a.cos() * 2.0, 1.0 + k as f32 * 0.75, a.sin() * 2.0))
                    .with_velocity(vec3(a.sin(), 0.0, -a.cos())),
            );
        }
        Simulation::new(world)
    }

    let mut a = build();
    let mut b = build();
    for step in 0..600 {
        a.step();
        b.step();
        for i in 0..a.world.bodies.len() {
            assert_eq!(
                a.world.bodies.translation(i).to_array(),
                b.world.bodies.translation(i).to_array(),
                "diverged at step {step}, body {i}"
            );
            assert_eq!(
                a.world.bodies.orientation[i].to_array(),
                b.world.bodies.orientation[i].to_array(),
                "orientation diverged at step {step}, body {i}"
            );
        }
    }
}

#[test]
fn reset_returns_the_world_to_the_scene() {
    let mut world = World::new();
    ground(&mut world);
    let ball = world.add(BodyDesc::sphere(0.5).at(vec3(0.0, 5.0, 0.0)));
    let mut sim = Simulation::new(world);

    run_seconds(&mut sim, 2.0);
    assert!(
        sim.world.bodies.translation(ball.index()).y < 4.0,
        "it should have fallen"
    );

    sim.reset();
    assert_eq!(sim.time(), 0.0);
    assert_eq!(sim.step_index(), 0);
    assert_eq!(
        sim.world.bodies.translation(ball.index()),
        vec3(0.0, 5.0, 0.0)
    );
    assert_eq!(sim.world.bodies.linear_velocity[ball.index()], Vec3::ZERO);
}

#[test]
fn frames_are_decoupled_from_the_solver_clock() {
    // PLAN.md §17: frames are an output sampling choice, not the engine's
    // clock. Four solver steps per frame at these rates.
    let mut world = World::new();
    ground(&mut world);
    world.add(BodyDesc::sphere(0.5).at(vec3(0.0, 5.0, 0.0)));
    let mut sim = Simulation::new(world);
    sim.solver_dt = 1.0 / 240.0;
    sim.output_rate = 60.0;

    sim.advance_frame();
    assert_eq!(sim.step_index(), 4);
    assert!((sim.time() - 1.0 / 60.0).abs() < 1e-6);
}

#[test]
fn advanced_mode_is_refused_rather_than_faked() {
    let world = World::new();
    assert!(Simulation::with_mode(world, Mode::Advanced).is_err());
}

#[test]
fn the_solver_declares_its_accuracy_tier() {
    let sim = Simulation::new(World::new());
    let info = sim.solver_info();
    assert_eq!(info.tier, fusion_sim::AccuracyTier::RealTime);
    assert!(
        !info.conserves.energy,
        "XPBD must not claim energy conservation"
    );
    assert!(info.conserves.momentum);
    assert!(!info.caveat.is_empty());
}
