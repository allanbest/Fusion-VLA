//! Physical invariants.
//!
//! These check properties that must hold no matter what the solver does
//! internally: conservation laws, and independence from the arbitrary choices
//! of where the origin is and which way the axes point. A solver can pass
//! every "does it look right" test and still fail these, and when it does, it
//! is wrong in a way that will eventually be visible.

use fusion_core::{BodyDesc, BodySet, Material, World};
use fusion_math::{vec3, Mat3, Quat, Vec3};
use fusion_sim::Simulation;

fn run(sim: &mut Simulation, seconds: f64) {
    let steps = (seconds / sim.solver_dt as f64).round() as u64;
    for _ in 0..steps {
        sim.step();
    }
}

/// World-space angular momentum, `I_world * omega`, summed over dynamic bodies.
#[allow(dead_code)]
fn angular_momentum(bodies: &BodySet) -> Vec3 {
    let mut total = Vec3::ZERO;
    for i in 0..bodies.len() {
        if !bodies.is_dynamic(i) {
            continue;
        }
        let inv = bodies.inv_inertia_local[i];
        let inertia_local = vec3(
            if inv.x > 0.0 { 1.0 / inv.x } else { 0.0 },
            if inv.y > 0.0 { 1.0 / inv.y } else { 0.0 },
            if inv.z > 0.0 { 1.0 / inv.z } else { 0.0 },
        );
        let r = Mat3::from_quat(bodies.orientation[i]);
        let inertia_world = r.mul_diagonal_mul_transpose(inertia_local);
        total += inertia_world * bodies.angular_velocity[i];
        // Orbital term about the origin.
        let m = 1.0 / bodies.inv_mass[i];
        total += bodies.translation(i).cross(bodies.linear_velocity[i] * m);
    }
    total
}

#[test]
fn a_freely_spinning_body_conserves_angular_momentum() {
    // Euler's equations: with no torque, *angular momentum* is constant, not
    // angular velocity. For a body whose principal moments differ, the
    // world-space inertia turns with the body, so omega must turn too.
    //
    // Sampled throughout the run rather than only at the end. Checking one
    // final value is not enough: L rotates about the momentum axis at a rate
    // set by the spin, so a run that happens to finish after a whole number of
    // revolutions looks perfect no matter how wrong the integrator is.
    let mut world = World::new().with_gravity(Vec3::ZERO);
    world.angular_damping = 0.0;
    let brick = world.add(
        BodyDesc::cuboid(vec3(1.0, 2.0, 3.0))
            .with_angular_velocity(vec3(3.0, 5.0, 2.0))
            .with_material(Material::WOOD),
    );
    let mut sim = Simulation::new(world);
    let l0 = sim.world.bodies.angular_momentum(brick.index());

    let mut worst = 0.0f32;
    for _ in 0..1200 {
        sim.step();
        let l = sim.world.bodies.angular_momentum(brick.index());
        worst = worst.max((l - l0).length() / l0.length());
    }
    assert!(
        worst < 0.05,
        "angular momentum drifted by {:.1}% at worst",
        worst * 100.0
    );
}

#[test]
fn an_asymmetric_body_tumbles_like_a_thrown_book() {
    // The tennis-racket theorem, or the Dzhanibekov effect: rotation about the
    // intermediate principal axis is unstable, and a real body periodically
    // flips end over end. An integrator that carries omega forward unchanged
    // spins placidly forever instead.
    //
    // Extents (1, 2, 3) give principal moments in roughly the ratio 13:10:5,
    // so y is the intermediate axis.
    let mut world = World::new().with_gravity(Vec3::ZERO);
    world.angular_damping = 0.0;
    let plate = world.add(
        BodyDesc::cuboid(vec3(1.0, 2.0, 3.0))
            .with_angular_velocity(vec3(0.1, 10.0, 0.0))
            .with_material(Material::WOOD),
    );
    let mut sim = Simulation::new(world);

    let start_axis = sim.world.bodies.orientation[plate.index()].rotate(Vec3::Y);
    let mut flips = 0;
    let mut sign = 1.0f32;
    for _ in 0..2400 {
        sim.step();
        let axis = sim.world.bodies.orientation[plate.index()].rotate(Vec3::Y);
        let now = axis.dot(start_axis).signum();
        if now != sign {
            flips += 1;
            sign = now;
        }
    }
    assert!(
        flips >= 2,
        "expected the body to flip repeatedly, saw {flips} flips in 10 s"
    );
}

#[test]
fn a_symmetric_body_does_not_tumble() {
    // The other half of the same property, and the guard against overcorrecting:
    // a sphere or a cube has equal principal moments, no gyroscopic torque, and
    // must spin about a fixed axis forever.
    for shape in [BodyDesc::sphere(0.7), BodyDesc::cuboid(Vec3::ONE)] {
        let mut world = World::new().with_gravity(Vec3::ZERO);
        world.angular_damping = 0.0;
        let b = world.add(shape.with_angular_velocity(vec3(2.0, 7.0, -3.0)));
        let mut sim = Simulation::new(world);
        let w0 = sim.world.bodies.angular_velocity[b.index()];
        run(&mut sim, 5.0);
        let w1 = sim.world.bodies.angular_velocity[b.index()];
        assert!(
            (w1 - w0).length() < 1e-3,
            "isotropic inertia must leave omega alone: {w0:?} -> {w1:?}"
        );
    }
}

#[test]
fn the_result_does_not_depend_on_where_the_origin_is() {
    // Physics is translation invariant. f32 arithmetic is not, so the scene is
    // deliberately non-chaotic: a box and a sphere that fall, land and settle.
    // A pile of colliding boxes would diverge from a one-ULP difference no
    // matter how correct the engine is, and would be testing chaos rather than
    // the engine.
    // Restitution zero and no spheres, on purpose. A bouncing box and a
    // rolling ball are both bifurcation-prone: which way a sphere rolls off a
    // single contact point is decided by noise, and amplifying noise is what
    // chaos does. That would be a test of chaos, not of the engine.
    let material = Material::new(1500.0, 0.5, 0.0);

    fn build(offset: Vec3, material: Material) -> Simulation {
        let mut world = World::new();
        world.add(
            BodyDesc::cuboid(vec3(40.0, 2.0, 40.0))
                .at(offset + vec3(0.0, -1.0, 0.0))
                .with_material(material)
                .fixed(),
        );
        // Dropped flat.
        world.add(
            BodyDesc::cuboid(Vec3::ONE)
                .at(offset + vec3(0.0, 3.0, 0.0))
                .with_material(material),
        );
        // And sliding, so friction is exercised too.
        world.add(
            BodyDesc::cuboid(Vec3::ONE)
                .at(offset + vec3(5.0, 0.5, 0.0))
                .with_velocity(vec3(-4.0, 0.0, 0.0))
                .with_material(material),
        );
        // Sleeping off: it is a deliberate approximation with a threshold in
        // it, and a body that crosses that threshold one step earlier in one
        // run than the other freezes in a slightly different place. That is
        // the heuristic being sensitive, not the physics being wrong, and it
        // has its own tests.
        let mut sim = Simulation::new(world);
        sim.sleep.enabled = false;
        sim
    }

    let mut origin = build(Vec3::ZERO, material);
    run(&mut origin, 4.0);

    for offset in [
        vec3(50.0, 0.0, -30.0),
        vec3(500.0, 0.0, 500.0),
        vec3(0.0, 200.0, 0.0),
        vec3(20_000.0, 0.0, -20_000.0),
    ] {
        let mut shifted = build(offset, material);
        run(&mut shifted, 4.0);
        // Five millimetres, flat, at any distance — including 28 km out.
        //
        // This used to need a tolerance that *grew* with the offset, because
        // f32 positions could not resolve a contact correction far from the
        // origin: the same slide travelled 1.7 m at the origin and stopped
        // after 0.85 m two hundred metres up. Holding the centre of mass in
        // double precision removed the dependence on distance entirely; what
        // is left is the residual from the quantities that are still f32
        // (velocities, orientations, bounds), and that does not grow.
        let tolerance = 0.005;
        for i in 1..origin.world.bodies.len() {
            let a = origin.world.bodies.translation(i);
            let b = shifted.world.bodies.translation(i) - offset;
            assert!(
                (a - b).length() < tolerance,
                "body {i} ended {:.5} m away when the scene moved to {offset:?}",
                (a - b).length()
            );
        }
    }
}

#[test]
fn the_result_does_not_depend_on_which_way_is_down() {
    // Rotate the whole scene, gravity included, and the answer must rotate
    // with it. This catches anything that quietly assumes +Y is up — an easy
    // assumption to bake into contact code without noticing.
    let turn = Quat::from_axis_angle(vec3(0.3, 0.2, 1.0), 0.9);

    let material = Material::new(1500.0, 0.5, 0.0);

    fn build(turn: Quat, material: Material) -> Simulation {
        let mut world = World::new().with_gravity(turn.rotate(vec3(0.0, -9.81, 0.0)));
        world.add(
            BodyDesc::cuboid(vec3(40.0, 2.0, 40.0))
                .at(turn.rotate(vec3(0.0, -1.0, 0.0)))
                .rotated(turn)
                .with_material(material)
                .fixed(),
        );
        world.add(
            BodyDesc::cuboid(Vec3::ONE)
                .at(turn.rotate(vec3(0.0, 3.0, 0.0)))
                .rotated(turn)
                .with_material(material),
        );
        world.add(
            BodyDesc::cuboid(Vec3::ONE)
                .at(turn.rotate(vec3(5.0, 0.5, 0.0)))
                .rotated(turn)
                .with_velocity(turn.rotate(vec3(-4.0, 0.0, 0.0)))
                .with_material(material),
        );
        let mut sim = Simulation::new(world);
        sim.sleep.enabled = false;
        sim
    }

    let mut upright = build(Quat::IDENTITY, material);
    let mut tilted = build(turn, material);
    run(&mut upright, 4.0);
    run(&mut tilted, 4.0);

    for i in 1..upright.world.bodies.len() {
        let expected = turn.rotate(upright.world.bodies.translation(i));
        let got = tilted.world.bodies.translation(i);
        // Looser than the translation case, and legitimately so: rotating a
        // scene changes which axis the broad phase sweeps, and the arithmetic
        // of a rotated scene is not the exact arithmetic of an axis-aligned
        // one. The answer must be the same physics, not the same arithmetic.
        //
        // One centimetre on a slide of about 1.7 m; measured, 1 mm. It was
        // eight, and measured 28, while the velocity pass solved contact
        // points one at a time: a rotated scene visits them in a different
        // order, and a friction-arrested slide accumulates that over
        // thousands of visits. Each manifold's normal impulses are now solved
        // as a converged block, which does not care about order — so this is
        // tight enough to catch it coming back. An engine that assumed +Y is
        // up would miss by metres.
        assert!(
            (expected - got).length() < 0.01,
            "body {i} landed {:.5} m from where the rotated scene should put it",
            (expected - got).length()
        );
    }
}

#[test]
fn a_light_body_does_not_push_a_heavy_one_around() {
    // A 1000:1 mass ratio is where an under-converged solver shows itself: the
    // heavy body gets shoved by the light one, or the light one sinks in.
    let mut world = World::new();
    world.add(
        BodyDesc::cuboid(vec3(40.0, 2.0, 40.0))
            .at(vec3(0.0, -1.0, 0.0))
            .with_material(Material::CONCRETE)
            .fixed(),
    );
    let heavy = world.add(
        BodyDesc::cuboid(Vec3::ONE)
            .at(vec3(0.0, 0.5, 0.0))
            .with_mass(1000.0),
    );
    let light = world.add(
        BodyDesc::cuboid(Vec3::ONE)
            .at(vec3(0.0, 1.5, 0.0))
            .with_mass(1.0),
    );
    let mut sim = Simulation::new(world);

    run(&mut sim, 4.0);

    let h = sim.world.bodies.translation(heavy.index());
    let l = sim.world.bodies.translation(light.index());
    assert!(
        (h.y - 0.5).abs() < 0.02,
        "the heavy box sank or rose to {}",
        h.y
    );
    assert!((l.y - 1.5).abs() < 0.05, "the light box ended at {}", l.y);
    assert!(
        h.x.abs() < 0.02 && h.z.abs() < 0.02,
        "the heavy box was shoved sideways: {h:?}"
    );
}

#[test]
fn bodies_spawned_inside_each_other_separate_calmly() {
    // Scene-authoring mistakes must degrade, not detonate. PLAN.md's
    // depenetration limit exists precisely so that this resolves over several
    // frames instead of firing everything apart in one.
    let mut world = World::new().with_gravity(Vec3::ZERO);
    for k in 0..12 {
        world.add(
            BodyDesc::sphere(0.5)
                .at(vec3(k as f32 * 0.02, 0.0, 0.0))
                .with_material(Material::WOOD),
        );
    }
    let mut sim = Simulation::new(world);

    let mut peak_speed = 0.0f32;
    for _ in 0..6000 {
        sim.step();
        for i in 0..sim.world.bodies.len() {
            peak_speed = peak_speed.max(sim.world.bodies.linear_velocity[i].length());
        }
    }
    assert!(sim.is_stable());
    // The depenetration budget is per contact point, so a body caught in
    // eleven deep overlaps at once can leave faster than any single one of
    // them allows. What must not happen is a detonation.
    assert!(
        peak_speed < 15.0,
        "twelve coincident spheres reached {peak_speed} m/s"
    );

    // They must end up genuinely apart.
    for i in 0..sim.world.bodies.len() {
        for j in (i + 1)..sim.world.bodies.len() {
            let gap = sim
                .world
                .bodies
                .translation(i)
                .distance(sim.world.bodies.translation(j));
            // Depenetration is deliberately rate-limited, and a jammed cluster
            // unpacks from its ends inward, so this is a slow process by
            // design. What matters is that it finishes rather than detonating.
            assert!(
                gap > 0.99,
                "spheres {i} and {j} are still overlapping at {gap:.3} m"
            );
        }
    }

    // And once apart, coasting: no gravity, no damping, no contacts, so
    // nothing should still be changing their velocity. "Comes to rest" would
    // be the wrong assertion — it would be asking for a force that is not
    // there.
    let before: Vec<Vec3> = sim.world.bodies.linear_velocity.clone();
    run(&mut sim, 1.0);
    for (i, v0) in before.iter().enumerate() {
        let v1 = sim.world.bodies.linear_velocity[i];
        assert!(
            (v1 - *v0).length() < 1e-4,
            "sphere {i} is still being accelerated after separating: {v0:?} -> {v1:?}"
        );
    }
}

#[test]
fn an_empty_or_single_body_world_steps_without_panicking() {
    let mut empty = Simulation::new(World::new());
    empty.step();
    assert_eq!(empty.step_index(), 1);

    let mut world = World::new();
    world.add(BodyDesc::sphere(1.0));
    let mut one = Simulation::new(world);
    for _ in 0..10 {
        one.step();
    }
    assert!(one.is_stable());
}

#[test]
fn extreme_velocities_stay_finite() {
    let mut world = World::new();
    world.add(
        BodyDesc::cuboid(vec3(200.0, 2.0, 200.0))
            .at(vec3(0.0, -1.0, 0.0))
            .with_material(Material::CONCRETE)
            .fixed(),
    );
    for k in 0..6 {
        world.add(
            BodyDesc::sphere(0.4)
                .at(vec3(k as f32 * 2.0 - 5.0, 20.0, 0.0))
                .with_velocity(vec3(0.0, -1000.0, 0.0)),
        );
    }
    let mut sim = Simulation::new(world);
    run(&mut sim, 2.0);
    // Tunnelling is expected at 1000 m/s without CCD (v0.3). Producing a NaN
    // is not: that would poison every body it later touches.
    assert!(sim.is_stable(), "the state went non-finite at high speed");
}

#[test]
fn resting_contact_reaches_a_true_fixed_point() {
    // A resting body must stop *changing*, not merely stay in place on
    // average. Jitter at rest is what island sleeping (v0.2) is there to
    // exploit, and it cannot exploit what does not settle.
    let mut world = World::new();
    world.add(
        BodyDesc::cuboid(vec3(40.0, 2.0, 40.0))
            .at(vec3(0.0, -1.0, 0.0))
            .with_material(Material::CONCRETE)
            .fixed(),
    );
    let b = world.add(BodyDesc::cuboid(Vec3::ONE).at(vec3(0.0, 0.5, 0.0)));
    let mut sim = Simulation::new(world);
    run(&mut sim, 2.0);

    let settled = sim.world.bodies.translation(b.index());
    run(&mut sim, 2.0);
    let later = sim.world.bodies.translation(b.index());
    assert!(
        (settled - later).length() < 1e-4,
        "a resting box drifted {:.6} m over two seconds",
        (settled - later).length()
    );
    assert!(sim.world.bodies.linear_velocity[b.index()].length() < 1e-3);
    assert!(sim.world.bodies.angular_velocity[b.index()].length() < 1e-3);
}
