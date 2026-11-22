//! Heat in rigid bodies, against answers that need no simulation.
//!
//! Energy is the check that matters: with the air shut off, every joule a
//! contact takes out of the motion has to turn up as heat, and every joule of
//! heat that crosses a contact has to leave one body and arrive in the other.
//! The rest are textbook: a small hot body in moving air cools exponentially
//! with Newton's time constant, and heat runs from hot to cold.

use fusion_core::{BodyDesc, Material, World};
use fusion_math::{vec3, Vec3};
use fusion_sim::{HeatSettings, Simulation};

fn floor(world: &mut World, material: Material) {
    world.add(
        BodyDesc::cuboid(vec3(40.0, 1.0, 40.0))
            .at(vec3(0.0, -0.5, 0.0))
            .with_material(material)
            .fixed(),
    );
}

/// Kinetic plus potential energy of every dynamic body, in double.
///
/// Counted here rather than read from `Simulation::mechanical_energy`, which
/// leaves a sleeping body's potential energy out — so a block falling asleep
/// at the end of its slide reads as losing `m g y` the step it does.
fn mechanical(sim: &Simulation) -> f64 {
    let bodies = &sim.world.bodies;
    let g = fusion_math::to_f64(sim.world.gravity);
    (0..bodies.len())
        .filter(|&i| bodies.inv_mass[i] > 0.0)
        .map(|i| {
            let m = 1.0 / bodies.inv_mass[i] as f64;
            let v = fusion_math::to_f64(bodies.linear_velocity[i]);
            let w = bodies.angular_velocity[i];
            let inertia = bodies.inertia_local[i];
            let local = bodies.orientation[i].rotate_inverse(w);
            let spin = 0.5
                * (inertia.x as f64 * (local.x as f64).powi(2)
                    + inertia.y as f64 * (local.y as f64).powi(2)
                    + inertia.z as f64 * (local.z as f64).powi(2));
            0.5 * m * v.length_squared() + spin - m * g.dot(bodies.position[i])
        })
        .sum()
}

/// A block sliding to a stop turns its kinetic energy into heat — all of it,
/// to the accuracy the solver's own energy books are kept.
#[test]
fn friction_turns_the_lost_motion_into_heat() {
    let mut world = World::new();
    floor(&mut world, Material::STEEL);
    let block = world.add(
        BodyDesc::cuboid(Vec3::splat(0.5))
            .at(vec3(0.0, 0.25, 0.0))
            .with_velocity(vec3(5.0, 0.0, 0.0))
            .with_material(Material::STEEL),
    );
    let mut sim = Simulation::new(world);
    sim.heat = HeatSettings::insulated();
    // One step first, so the grids exist and the clock of the books starts
    // with the block already sliding.
    sim.step();
    let before = (mechanical(&sim), sim.rigid_thermal_energy());
    for _ in 0..720 {
        sim.step();
    }
    let speed = sim.world.bodies.linear_velocity[block.index()].length();
    assert!(speed < 0.01, "still sliding at {speed} m/s");

    let lost = before.0 - mechanical(&sim);
    let gained = sim.rigid_thermal_energy() - before.1;
    assert!(lost > 1000.0, "only {lost} J of motion lost");
    // Measured: 0.01%.
    let error = (gained - lost).abs() / lost;
    assert!(
        error < 0.01,
        "{lost:.1} J of motion became {gained:.1} J of heat ({:.1}% off)",
        100.0 * error
    );

    // And it is in the face that slid, not spread evenly through the block.
    let middle = sim.world.bodies.translation(block.index());
    let bottom = sim.body_temperature_at(block, middle - vec3(0.0, 0.24, 0.0));
    let top = sim.body_temperature_at(block, middle + vec3(0.0, 0.24, 0.0));
    assert!(bottom > top + 0.05, "bottom {bottom} K, top {top} K");
}

/// A dead landing turns the fall into heat; a bounce keeps `e^2` of it as
/// motion and heats by the rest.
#[test]
fn a_landing_heats_by_what_it_did_not_give_back() {
    let heat_of = |restitution: f32| {
        let mut world = World::new();
        floor(
            &mut world,
            Material {
                restitution,
                ..Material::STEEL
            },
        );
        world.add(
            BodyDesc::sphere(0.1)
                .at(vec3(0.0, 1.0, 0.0))
                .with_material(Material {
                    restitution,
                    ..Material::STEEL
                }),
        );
        let mut sim = Simulation::new(world);
        sim.heat = HeatSettings::insulated();
        sim.step();
        let before = (mechanical(&sim), sim.rigid_thermal_energy());
        // Long enough to land, short enough that a bouncing ball is still in
        // the air after its first bounce.
        for _ in 0..130 {
            sim.step();
        }
        let lost = before.0 - mechanical(&sim);
        let gained = sim.rigid_thermal_energy() - before.1;
        (lost, gained)
    };
    let (lost, gained) = heat_of(0.0);
    let mass = Material::STEEL.density as f64 * (4.0 / 3.0) * std::f64::consts::PI * 1e-3;
    let fall = mass * 9.81 * 0.9;
    assert!(
        (lost - fall).abs() / fall < 0.1,
        "lost {lost} J of a {fall} J fall"
    );
    // Measured: 0.05%.
    assert!(
        (gained - lost).abs() / lost < 0.01,
        "a dead landing lost {lost:.2} J and heated by {gained:.2} J"
    );

    // Measured: 186.48 J against the 186.0 that `1 - e^2` of the dead
    // landing's heat predicts.
    let (bounce_lost, bounce_gained) = heat_of(0.6);
    let predicted = (1.0 - 0.36) * gained;
    assert!(
        (bounce_gained - predicted).abs() / predicted < 0.02,
        "a bounce heated by {bounce_gained:.2} J where 1 - e^2 says {predicted:.2}"
    );
    assert!(
        (bounce_gained - bounce_lost).abs() / bounce_lost < 0.01,
        "a bounce lost {bounce_lost:.2} J and heated by {bounce_gained:.2} J"
    );
}

/// Newton's law of cooling: a small, well-conducting body in moving air
/// relaxes to the air's temperature as `exp(-h A t / (m c))`.
#[test]
fn a_hot_sphere_cools_with_newtons_time_constant() {
    let mut world = World::new();
    world.gravity = Vec3::ZERO;
    let ball = world.add(
        BodyDesc::sphere(0.05)
            .with_material(Material::ALUMINIUM)
            .with_temperature(400.0)
            .fixed(),
    );
    let mut sim = Simulation::new(world);
    sim.solver_dt = 1.0 / 60.0;
    let h = 1000.0;
    sim.heat = HeatSettings {
        convection: h,
        radiation: false,
        ..HeatSettings::on()
    };
    let air = sim.heat.air_temperature as f64;
    let r = 0.05f64;
    let mass = Material::ALUMINIUM.density as f64 * (4.0 / 3.0) * std::f64::consts::PI * r.powi(3);
    let area = 4.0 * std::f64::consts::PI * r * r;
    let tau = mass * Material::ALUMINIUM.thermal.specific_heat as f64 / (h as f64 * area);
    let steps = (tau / sim.solver_dt as f64).round() as usize;
    for _ in 0..steps {
        sim.step();
    }
    let expected = air + (400.0 - air) * (-1.0f64).exp();
    let got = sim.body_temperature(ball) as f64;
    // Biot number 0.08, so the lumped answer is good to a few percent of the
    // drop; backward Euler at a sixtieth of a second adds well under one.
    // Measured: 1.2% warm, which is the inside lagging the surface.
    assert!(
        (got - expected).abs() < 0.03 * (400.0 - air),
        "after one time constant: {got:.2} K, Newton says {expected:.2} K"
    );
}

/// Heat crosses a contact from the hot body to the cold one, and none is made
/// or lost on the way.
#[test]
fn a_hot_block_warms_the_cold_one_it_sits_on() {
    let mut world = World::new();
    let base = world.add(
        BodyDesc::cuboid(Vec3::splat(0.1))
            .at(vec3(0.0, 0.05, 0.0))
            .with_material(Material::ALUMINIUM)
            .with_temperature(300.0)
            .fixed(),
    );
    let top = world.add(
        BodyDesc::cuboid(Vec3::splat(0.1))
            .at(vec3(0.0, 0.15, 0.0))
            .with_material(Material::ALUMINIUM)
            .with_temperature(500.0),
    );
    let mut sim = Simulation::new(world);
    sim.heat = HeatSettings::insulated();
    sim.step();
    let start = sim.rigid_thermal_energy();
    for _ in 0..2400 {
        sim.step();
    }
    // Measured: 481 K and 319 K after ten seconds, the top block asleep for
    // most of them.
    let (hot, cold) = (sim.body_temperature(top), sim.body_temperature(base));
    assert!(hot < 495.0, "the hot block is still at {hot} K");
    assert!(cold > 305.0, "the cold block is still at {cold} K");
    assert!(hot > cold, "{hot} K above {cold} K: heat ran the wrong way");
    // The cold block is warmer where the hot one touches it.
    let p = sim.world.bodies.translation(base.index());
    let near = sim.body_temperature_at(base, p + vec3(0.0, 0.045, 0.0));
    let far = sim.body_temperature_at(base, p - vec3(0.0, 0.045, 0.0));
    assert!(near > far, "top of the base {near} K, bottom {far} K");

    let drift = (sim.rigid_thermal_energy() - start).abs() / start;
    assert!(
        drift < 1e-6,
        "insulated, and the heat changed by {drift:e} of itself"
    );
}

/// With heat off, nothing is built and nothing changes.
#[test]
fn heat_off_costs_nothing_and_changes_nothing() {
    let mut world = World::new();
    floor(&mut world, Material::STEEL);
    let block = world.add(
        BodyDesc::cuboid(Vec3::splat(0.5))
            .at(vec3(0.0, 0.25, 0.0))
            .with_velocity(vec3(5.0, 0.0, 0.0))
            .with_material(Material::STEEL)
            .with_temperature(350.0),
    );
    let mut sim = Simulation::new(world);
    for _ in 0..240 {
        sim.step();
    }
    assert!(sim.world.bodies.heat.iter().all(|h| !h.is_built()));
    assert_eq!(sim.body_temperature(block), 350.0);
}

/// Two runs of the same hot, sliding scene agree to the bit, and a
/// checkpoint written to bytes and read back resumes the same temperatures.
#[test]
fn heat_is_deterministic_and_survives_a_checkpoint() {
    let build = || {
        let mut world = World::new();
        floor(&mut world, Material::CONCRETE);
        world.add(
            BodyDesc::cuboid(Vec3::splat(0.4))
                .at(vec3(0.0, 0.2, 0.0))
                .with_velocity(vec3(4.0, 0.0, 1.0))
                .with_material(Material::STEEL)
                .with_temperature(600.0),
        );
        let mut sim = Simulation::new(world);
        sim.heat = HeatSettings::on();
        sim
    };
    let mut a = build();
    let mut b = build();
    for _ in 0..120 {
        a.step();
        b.step();
    }
    assert_eq!(fusion_sim::state_hash(&a), fusion_sim::state_hash(&b));

    let saved = a.checkpoint();
    let bytes = saved.to_bytes();
    for _ in 0..120 {
        a.step();
    }
    let template = b.checkpoint();
    let restored = fusion_sim::Checkpoint::from_bytes(&bytes, &template).expect("reads back");
    b.restore(&restored);
    for _ in 0..120 {
        b.step();
    }
    assert_eq!(fusion_sim::state_hash(&a), fusion_sim::state_hash(&b));
}

/// A block sitting still on a floor stays at the temperature it started at.
///
/// Not a given. Gravity is integrated before the contact holds a resting body
/// up, so every substep the position solve stops a `g h` approach — and
/// counted as dissipation, that heated a half-metre steel block by 24 W for as
/// long as it sat there.
#[test]
fn a_resting_block_does_not_warm_itself() {
    let mut world = World::new();
    floor(&mut world, Material::STEEL);
    let block = world.add(
        BodyDesc::cuboid(Vec3::splat(0.5))
            .at(vec3(0.0, 0.25, 0.0))
            .with_material(Material::STEEL),
    );
    let mut sim = Simulation::new(world);
    sim.heat = HeatSettings::insulated();
    // Kept awake, so the contact is solved every step rather than skipped.
    sim.sleep.enabled = false;
    // Settling into the contact slop first: half a millimetre at `m g`, about
    // 5 J, which is a real (if tiny) landing and is correctly heat.
    for _ in 0..10 {
        sim.step();
    }
    let start = sim.rigid_thermal_energy();
    for _ in 0..1200 {
        sim.step();
    }
    let gained = sim.rigid_thermal_energy() - start;
    assert!(
        gained.abs() < 0.01,
        "five seconds at rest made {gained:.3} J of heat"
    );
    let (_, hottest) = sim.body_temperature_range(block);
    assert!(
        hottest - sim.heat.air_temperature < 1e-3,
        "hottest point {hottest} K"
    );
}

/// Something glowing loses most of its heat by radiation, and something
/// warm hardly any — `T^4` is the whole story.
#[test]
fn radiation_matters_only_once_something_is_hot() {
    let cooled = |start: f32, radiation: bool| {
        let mut world = World::new();
        world.gravity = Vec3::ZERO;
        let ball = world.add(
            BodyDesc::sphere(0.05)
                .with_material(Material::STEEL)
                .with_temperature(start)
                .fixed(),
        );
        let mut sim = Simulation::new(world);
        sim.solver_dt = 1.0 / 60.0;
        sim.heat = HeatSettings {
            radiation,
            ..HeatSettings::on()
        };
        for _ in 0..600 {
            sim.step();
        }
        start - sim.body_temperature(ball)
    };
    let glowing = (cooled(1200.0, true), cooled(1200.0, false));
    assert!(
        glowing.0 > 3.0 * glowing.1,
        "at 1200 K: {:.1} K lost with radiation, {:.1} K without",
        glowing.0,
        glowing.1
    );
    let warm = (cooled(320.0, true), cooled(320.0, false));
    assert!(
        warm.0 < 2.0 * warm.1,
        "at 320 K: {:.3} K lost with radiation, {:.3} K without",
        warm.0,
        warm.1
    );
}
