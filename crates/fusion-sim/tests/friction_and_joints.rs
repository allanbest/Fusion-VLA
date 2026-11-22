//! v0.2 features: Coulomb friction, restitution, and joints.
//!
//! Friction is checked against the textbook result rather than against
//! "it looks like it grips": a block on an incline stays put while
//! `tan(theta) < mu` and slides once it does not. That is a sharp, falsifiable
//! prediction, and a solver that only damps tangential velocity fails it —
//! such a solver can make a block creep slowly, never hold it still.

use fusion_core::{BodyDesc, Joint, JointKind, Material, World};
use fusion_math::{vec3, Quat, Vec3};
use fusion_sim::Simulation;

fn run(sim: &mut Simulation, seconds: f64) {
    let steps = (seconds / sim.solver_dt as f64).round() as u64;
    for _ in 0..steps {
        sim.step();
    }
}

/// A block resting on an incline of `angle`, both sharing `material`.
fn incline(angle: f32, material: Material) -> (Simulation, usize) {
    let mut world = World::new();
    let tilt = Quat::from_rotation_z(angle);
    world.add(
        BodyDesc::cuboid(vec3(40.0, 2.0, 40.0))
            .at(tilt.rotate(vec3(0.0, -1.0, 0.0)))
            .rotated(tilt)
            .with_material(material)
            .fixed(),
    );
    let block = world.add(
        BodyDesc::cuboid(Vec3::ONE)
            .at(tilt.rotate(vec3(0.0, 0.5, 0.0)))
            .rotated(tilt)
            .with_material(material),
    );
    (Simulation::new(world), block.index())
}

#[test]
fn a_block_holds_on_a_shallow_slope_and_slides_on_a_steep_one() {
    // Steel on steel: mu = 0.6, so the friction angle is atan(0.6) = 31
    // degrees. The pair rule is the geometric mean, which for a self-pair is
    // just mu again.
    let material = Material::STEEL;
    let friction_angle = material.friction.atan();

    let (mut shallow, block) = incline(friction_angle * 0.6, material);
    let start = shallow.world.bodies.translation(block);
    run(&mut shallow, 4.0);
    let travelled = (shallow.world.bodies.translation(block) - start).length();
    assert!(
        travelled < 0.02,
        "the block crept {travelled:.4} m on a slope it should hold"
    );

    let (mut steep, block) = incline(friction_angle * 1.6, material);
    let start = steep.world.bodies.translation(block);
    run(&mut steep, 4.0);
    let travelled = (steep.world.bodies.translation(block) - start).length();
    assert!(
        travelled > 1.0,
        "the block only moved {travelled:.4} m on a slope it should slide down"
    );
}

#[test]
fn ice_slides_where_rubber_grips() {
    let angle = 0.35; // about 20 degrees: above ice's friction angle, below rubber's
    let (mut slippery, block) = incline(angle, Material::ICE);
    let start = slippery.world.bodies.translation(block);
    run(&mut slippery, 3.0);
    let ice_travel = (slippery.world.bodies.translation(block) - start).length();

    let (mut grippy, block) = incline(angle, Material::RUBBER);
    let start = grippy.world.bodies.translation(block);
    run(&mut grippy, 3.0);
    let rubber_travel = (grippy.world.bodies.translation(block) - start).length();

    assert!(
        ice_travel > 1.0,
        "ice should slide, moved {ice_travel:.3} m"
    );
    assert!(
        rubber_travel < 0.05,
        "rubber should grip, moved {rubber_travel:.3} m"
    );
}

#[test]
fn a_sliding_block_decelerates_at_the_coulomb_rate() {
    // On the flat, kinetic friction gives a = -mu * g regardless of mass.
    let material = Material::new(2000.0, 0.4, 0.0);
    let mut world = World::new();
    world.add(
        BodyDesc::cuboid(vec3(200.0, 2.0, 200.0))
            .at(vec3(0.0, -1.0, 0.0))
            .with_material(material)
            .fixed(),
    );
    let block = world.add(
        BodyDesc::cuboid(Vec3::ONE)
            .at(vec3(0.0, 0.5, 0.0))
            .with_velocity(vec3(8.0, 0.0, 0.0))
            .with_material(material),
    );
    let mut sim = Simulation::new(world);

    // Let it settle onto the surface for a moment, then measure.
    run(&mut sim, 0.2);
    let v0 = sim.world.bodies.linear_velocity[block.index()].x;
    run(&mut sim, 0.5);
    let v1 = sim.world.bodies.linear_velocity[block.index()].x;

    let measured = (v0 - v1) / 0.5;
    let expected = material.friction * 9.81;
    assert!(
        (measured - expected).abs() < 0.25 * expected,
        "deceleration was {measured:.3} m/s^2, expected about {expected:.3}"
    );
}

#[test]
fn restitution_controls_the_bounce_height() {
    // A ball dropped from h returns to about e^2 * h. XPBD restitution is
    // approximate, so the tolerance is generous — but the *ordering* and the
    // rough magnitude must be right, and zero must mean zero.
    fn bounce_height(restitution: f32) -> f32 {
        let mut world = World::new();
        let material = Material::new(1000.0, 0.5, restitution);
        world.add(
            BodyDesc::cuboid(vec3(40.0, 2.0, 40.0))
                .at(vec3(0.0, -1.0, 0.0))
                .with_material(material)
                .fixed(),
        );
        let ball = world.add(
            BodyDesc::sphere(0.5)
                .at(vec3(0.0, 2.5, 0.0))
                .with_material(material),
        );
        let mut sim = Simulation::new(world);

        // The rebound is measured from the moment the ball is unambiguously in
        // contact — within 5 mm of its resting height, which it can only be
        // while touching. Anything looser re-measures the tail of the descent,
        // and anything that also demands "and rising" can be stepped straight
        // over: at 5 m/s the ball covers 2 cm per step.
        let mut touched = false;
        let mut peak = 0.0f32;
        for _ in 0..1200 {
            sim.step();
            let y = sim.world.bodies.translation(ball.index()).y;
            if y <= 0.505 {
                touched = true;
            }
            if touched {
                peak = peak.max(y - 0.5);
            }
        }
        peak
    }

    let dead = bounce_height(0.0);
    assert!(
        dead < 0.02,
        "restitution 0 must not bounce, reached {dead:.4} m"
    );

    let lively = bounce_height(0.8);
    // Drop height above the resting position is 2.0 m; e^2 * h = 1.28 m.
    assert!(
        (0.6..2.0).contains(&lively),
        "restitution 0.8 bounced to {lively:.3} m, expected roughly 1.3 m"
    );
    assert!(
        lively > bounce_height(0.4),
        "more restitution must bounce higher"
    );
}

#[test]
fn a_ball_joint_holds_a_pendulum_at_a_fixed_radius() {
    let mut world = World::new();
    let anchor = world.add(BodyDesc::sphere(0.2).at(vec3(0.0, 5.0, 0.0)).fixed());
    let bob = world.add(
        BodyDesc::sphere(0.3)
            .at(vec3(2.0, 5.0, 0.0))
            .with_material(Material::STEEL),
    );
    world.connect(anchor, bob, vec3(0.0, 5.0, 0.0), JointKind::Ball);
    let mut sim = Simulation::new(world);

    let pivot = vec3(0.0, 5.0, 0.0);
    let start = vec3(2.0, 5.0, 0.0);
    let mut worst_stretch = 0.0f32;
    let mut furthest = 0.0f32;
    let mut lowest = f32::INFINITY;
    for _ in 0..2400 {
        sim.step();
        let p = sim.world.bodies.translation(bob.index());
        worst_stretch = worst_stretch.max(((p - pivot).length() - 2.0).abs());
        furthest = furthest.max(p.distance(start));
        lowest = lowest.min(p.y);
    }
    assert!(
        worst_stretch < 0.02,
        "the pendulum arm stretched by {worst_stretch:.4} m"
    );
    // Measured over the whole run, not at the end: a pendulum sampled once
    // after a whole number of periods is back where it started, and would look
    // identical to one that never moved.
    assert!(
        furthest > 3.0,
        "the pendulum only reached {furthest:.3} m from its start"
    );
    assert!(
        lowest < 3.2,
        "the pendulum never swung down, lowest y was {lowest:.3}"
    );
}

#[test]
fn a_distance_joint_is_slack_until_it_is_taut() {
    // The inequality form: a rope does nothing until it runs out.
    let mut world = World::new();
    let anchor = world.add(BodyDesc::sphere(0.2).at(vec3(0.0, 8.0, 0.0)).fixed());
    let weight = world.add(BodyDesc::sphere(0.3).at(vec3(0.0, 7.5, 0.0)));
    let mut joint = Joint::new(anchor, weight, JointKind::Distance { min: 0.0, max: 3.0 });
    joint.local_anchor_a = Vec3::ZERO;
    joint.local_anchor_b = Vec3::ZERO;
    world.add_joint(joint);
    let mut sim = Simulation::new(world);

    // Early on the weight is well inside the rope's length and must fall freely.
    run(&mut sim, 0.3);
    let y = sim.world.bodies.translation(weight.index()).y;
    let free_fall = 7.5 - 0.5 * 9.81 * 0.3 * 0.3;
    assert!(
        (y - free_fall).abs() < 0.05,
        "a slack rope must not restrain: {y} vs {free_fall}"
    );

    run(&mut sim, 4.0);
    let drop = 8.0 - sim.world.bodies.translation(weight.index()).y;
    assert!(
        (drop - 3.0).abs() < 0.05,
        "the rope stopped it at {drop:.4} m instead of 3 m"
    );
}

#[test]
fn a_hinge_permits_one_axis_of_rotation_and_no_other() {
    let mut world = World::new();
    let frame = world.add(
        BodyDesc::cuboid(Vec3::splat(0.4))
            .at(vec3(0.0, 5.0, 0.0))
            .fixed(),
    );
    // A door hanging off a vertical hinge, pushed sideways.
    let door = world.add(
        BodyDesc::cuboid(vec3(2.0, 3.0, 0.2))
            .at(vec3(1.0, 5.0, 0.0))
            .with_velocity(vec3(0.0, 0.0, 3.0))
            .with_material(Material::WOOD),
    );
    world.connect_axis(frame, door, vec3(0.0, 5.0, 0.0), Vec3::Y, JointKind::Hinge);
    let mut sim = Simulation::new(world);

    let mut worst_tilt = 0.0f32;
    for _ in 0..1200 {
        sim.step();
        // The door's own +Y must stay parallel to the hinge axis.
        let up = sim.world.bodies.orientation[door.index()].rotate(Vec3::Y);
        worst_tilt = worst_tilt.max(up.reject_from_unit(Vec3::Y).length());
    }
    assert!(
        worst_tilt < 0.05,
        "the door tilted off its hinge axis by {worst_tilt:.4}"
    );

    // It must have swung about the hinge.
    let swing = sim.world.bodies.translation(door.index()).z.abs();
    assert!(swing > 0.2, "the door never swung, z = {swing:.4}");
}

#[test]
fn a_fixed_joint_welds_two_bodies_together() {
    let mut world = World::new();
    let base = world.add(BodyDesc::cuboid(Vec3::ONE).at(vec3(0.0, 5.0, 0.0)).fixed());
    let welded = world.add(
        BodyDesc::cuboid(Vec3::ONE)
            .at(vec3(1.5, 5.0, 0.0))
            .with_material(Material::STEEL),
    );
    world.connect(base, welded, vec3(0.75, 5.0, 0.0), JointKind::Fixed);
    let mut sim = Simulation::new(world);

    run(&mut sim, 4.0);
    let p = sim.world.bodies.translation(welded.index());
    assert!(
        (p - vec3(1.5, 5.0, 0.0)).length() < 0.05,
        "a welded body drifted to {p:?}"
    );
    let q = sim.world.bodies.orientation[welded.index()];
    assert!(
        q.approx_eq(Quat::IDENTITY, 1e-2),
        "a welded body rotated to {q:?}"
    );
}

#[test]
fn a_slider_travels_along_its_axis_and_stops_at_the_limit() {
    let mut world = World::new();
    let rail = world.add(
        BodyDesc::cuboid(Vec3::splat(0.3))
            .at(vec3(0.0, 5.0, 0.0))
            .fixed(),
    );
    let carriage = world.add(
        BodyDesc::cuboid(Vec3::splat(0.5))
            .at(vec3(0.0, 5.0, 0.0))
            .with_material(Material::STEEL),
    );
    // Vertical rail, so gravity drives the carriage down to its stop.
    world.connect_axis(
        rail,
        carriage,
        vec3(0.0, 5.0, 0.0),
        -Vec3::Y,
        JointKind::Slider { min: 0.0, max: 2.0 },
    );
    let mut sim = Simulation::new(world);

    run(&mut sim, 4.0);
    let p = sim.world.bodies.translation(carriage.index());
    assert!(
        p.x.abs() < 0.02 && p.z.abs() < 0.02,
        "the carriage left its rail sideways: {p:?}"
    );
    assert!(
        (p.y - 3.0).abs() < 0.05,
        "the carriage stopped at y={:.4}, expected 3.0",
        p.y
    );
}

#[test]
fn a_chain_of_joints_stays_connected() {
    // Ten links, each joined to the last. Chains are where a joint solver's
    // convergence shows: error propagates from the anchor to the free end.
    let mut world = World::new();
    let mut previous = world.add(BodyDesc::sphere(0.2).at(vec3(0.0, 8.0, 0.0)).fixed());
    let mut links = Vec::new();
    for k in 1..=10 {
        let link = world.add(
            BodyDesc::sphere(0.2)
                .at(vec3(k as f32 * 0.5, 8.0, 0.0))
                .with_material(Material::STEEL),
        );
        world.connect(
            previous,
            link,
            vec3((k as f32 - 0.5) * 0.5, 8.0, 0.0),
            JointKind::Ball,
        );
        links.push(link);
        previous = link;
    }
    let mut sim = Simulation::new(world);
    sim.solver.iterations = 8;

    run(&mut sim, 5.0);
    assert!(sim.is_stable());

    // The constraint is that each joint's two anchor points coincide — not
    // that the link centres stay a fixed distance apart. Ball joints leave
    // rotation free, so a chain of spheres is entitled to fold; asserting on
    // centre spacing would be asserting that the links cannot turn.
    let _ = &links;
    let mut worst = 0.0f32;
    for joint in sim.world.joints() {
        let pose_a = sim.world.bodies.pose(joint.body_a.index());
        let pose_b = sim.world.bodies.pose(joint.body_b.index());
        let (anchor_a, anchor_b) = joint.world_anchors(&pose_a, &pose_b);
        worst = worst.max(anchor_a.distance(anchor_b));
    }
    assert!(
        worst < 0.02,
        "a joint in the chain came apart by {worst:.4} m"
    );

    // And the chain must hang, not float.
    let tip = sim.world.bodies.translation(links[9].index());
    assert!(
        tip.y < 6.0,
        "the free end of the chain never fell, y = {:.3}",
        tip.y
    );
}

#[test]
fn a_settled_pile_falls_asleep_and_wakes_when_disturbed() {
    // PLAN.md §26 puts sleeping first among the adaptive-resolution levers.
    // The test is that it does both halves: a pile that has stopped moving
    // stops being simulated, and it starts again the moment something arrives.
    let mut world = World::new();
    world.add(
        BodyDesc::cuboid(vec3(40.0, 2.0, 40.0))
            .at(vec3(0.0, -1.0, 0.0))
            .with_material(Material::CONCRETE)
            .fixed(),
    );
    let mut boxes = Vec::new();
    // Three boxes. Sleeping is clean to this depth and, since the contact
    // solve finishes from the ground up, deeper.
    for k in 0..3 {
        boxes.push(
            world.add(
                BodyDesc::cuboid(Vec3::ONE)
                    .at(vec3(0.0, 0.5 + k as f32, 0.0))
                    .with_material(Material::WOOD),
            ),
        );
    }
    let mut sim = Simulation::new(world);

    run(&mut sim, 4.0);
    let asleep = sim.world.bodies.sleeping_count();
    assert_eq!(
        asleep,
        boxes.len(),
        "the whole stack should be asleep, {asleep} are"
    );

    // A sleeping body must hold its position exactly — that is the point.
    let frozen: Vec<_> = boxes
        .iter()
        .map(|b| sim.world.bodies.translation(b.index()))
        .collect();
    run(&mut sim, 2.0);
    for (k, before) in frozen.iter().enumerate() {
        let now = sim.world.bodies.translation(boxes[k].index());
        assert_eq!(before.to_array(), now.to_array(), "sleeping box {k} moved");
    }

    // Now drop something on it.
    let projectile = sim.world.add(
        BodyDesc::sphere(0.6)
            .at(vec3(0.0, 12.0, 0.0))
            .with_velocity(vec3(0.0, -20.0, 0.0))
            .with_material(Material::STEEL),
    );
    // Woken by the hit, at some point in the second after it. Not at the end
    // of it: a pile that has taken the hit and settled again goes back to
    // sleep, which it does inside the second.
    let mut fewest = boxes.len();
    for _ in 0..240 {
        sim.step();
        fewest = fewest.min(sim.world.bodies.sleeping_count());
    }
    assert!(fewest < boxes.len(), "the stack slept through being hit");
    assert!(sim.is_stable());
    let _ = projectile;
}

#[test]
fn sleeping_does_not_change_where_things_come_to_rest() {
    // Sleeping is an optimization, so it has to be invisible in the answer.
    fn build(sleeping: bool) -> Simulation {
        let mut world = World::new();
        world.add(
            BodyDesc::cuboid(vec3(40.0, 2.0, 40.0))
                .at(vec3(0.0, -1.0, 0.0))
                .with_material(Material::CONCRETE)
                .fixed(),
        );
        // Three boxes. Sleeping is clean to this depth and, since the contact
        // solve finishes from the ground up, deeper.
        for k in 0..3 {
            let a = k as f32 * 0.9;
            world.add(
                BodyDesc::cuboid(Vec3::ONE)
                    .at(vec3(a.cos() * 2.0, 1.0 + k as f32 * 1.3, a.sin() * 2.0))
                    .with_material(Material::WOOD),
            );
        }
        let mut sim = Simulation::new(world);
        sim.sleep.enabled = sleeping;
        sim
    }

    let mut with = build(true);
    let mut without = build(false);
    run(&mut with, 6.0);
    run(&mut without, 6.0);

    for i in 1..with.world.bodies.len() {
        let a = with.world.bodies.translation(i);
        let b = without.world.bodies.translation(i);
        // Sleeping is an approximation with a threshold in it, so the two
        // runs are not identical — but a pile must not settle into a visibly
        // different shape because of it.
        assert!(
            (a - b).length() < 0.15,
            "body {i} rests {:.4} m apart depending on whether sleeping is on",
            (a - b).length()
        );
    }
    assert!(with.world.bodies.sleeping_count() > 0, "nothing ever slept");
}
