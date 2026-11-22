//! v0.3: deformables and aerodynamics.

use fusion_core::{BodyDesc, ClothDesc, Material, World};
use fusion_math::{vec3, Vec3};
use fusion_sim::Simulation;
use fusion_solver::AirModel;

fn run(sim: &mut Simulation, seconds: f64) {
    let steps = (seconds / sim.solver_dt as f64).round() as u64;
    for _ in 0..steps {
        sim.step();
    }
}

/// Longest edge as a fraction of its rest length — how far the sheet has been
/// pulled out of shape.
fn worst_stretch(sim: &Simulation) -> f32 {
    sim.world
        .distance_constraints
        .iter()
        .map(|c| {
            let d = sim
                .world
                .particles
                .relative(c.a as usize, c.b as usize)
                .length();
            (d / c.rest - 1.0).abs()
        })
        .fold(0.0f32, f32::max)
}

#[test]
fn a_hanging_sheet_settles_without_stretching() {
    // Cloth is near-inextensible along the weave. A sheet hanging under its
    // own weight should reach a stable shape whose edges are still their rest
    // length — that is the property XPBD's zero-compliance constraints buy,
    // and a stiffness-based solver would need a tiny timestep to match it.
    let mut world = World::new();
    let cloth = world.add_cloth(
        &ClothDesc::new(16, 16, 0.1)
            .at(vec3(0.0, 3.0, 0.0))
            .with_mass(2.0),
    );
    for handle in cloth.top_edge() {
        world.particles.pin(handle.index());
    }
    let mut sim = Simulation::new(world);

    // A sheet released horizontally from a pinned edge swings down like a
    // hinged flap and takes a while to give that energy up.
    run(&mut sim, 10.0);

    assert!(sim.world.particles.is_finite());
    let stretch = worst_stretch(&sim);
    assert!(
        stretch < 0.05,
        "the sheet stretched by {:.1}%",
        stretch * 100.0
    );

    // It must actually hang — the free edge falls below where it started.
    let bottom = cloth.at(8, 15);
    let y = sim.world.particles.translation(bottom.index()).y;
    assert!(y < 2.5, "the free edge never fell, y = {y:.3}");

    // And it must come to rest.
    let speed = (0..sim.world.particles.len())
        .map(|i| sim.world.particles.velocity[i].length())
        .fold(0.0f32, f32::max);
    // Not zero: nothing in this scene damps a swinging sheet except the
    // solver's own cloth damping, because air is off. Real fabric is quieted
    // mostly by the air around it, which is what `AirModel` is for.
    assert!(speed < 1.0, "the sheet is still moving at {speed:.3} m/s");
}

#[test]
fn a_pinned_sheet_holds_its_pins_exactly() {
    let mut world = World::new();
    let cloth = world.add_cloth(&ClothDesc::new(12, 12, 0.1).at(vec3(0.0, 2.0, 0.0)));
    let pinned = cloth.at(0, 0);
    world.particles.pin(pinned.index());
    let mut sim = Simulation::new(world);
    let start = sim.world.particles.translation(pinned.index());

    run(&mut sim, 3.0);

    let now = sim.world.particles.translation(pinned.index());
    assert_eq!(start.to_array(), now.to_array(), "a pinned particle moved");
}

#[test]
fn cloth_drapes_over_a_sphere_instead_of_through_it() {
    let mut world = World::new();
    let radius = 0.6;
    world.add(BodyDesc::sphere(radius).at(vec3(0.0, 1.0, 0.0)).fixed());
    world.add(
        BodyDesc::cuboid(vec3(20.0, 1.0, 20.0))
            .at(vec3(0.0, -0.5, 0.0))
            .with_material(Material::CONCRETE)
            .fixed(),
    );
    let cloth = world.add_cloth(&ClothDesc::new(20, 20, 0.09).at(vec3(0.0, 2.2, 0.0)));
    let mut sim = Simulation::new(world);

    run(&mut sim, 4.0);

    assert!(sim.world.particles.is_finite());
    // Nothing may end up inside the sphere.
    let centre = vec3(0.0, 1.0, 0.0);
    let mut deepest = 0.0f32;
    for i in 0..sim.world.particles.len() {
        let d = sim.world.particles.translation(i).distance(centre);
        deepest = deepest.max(radius - d);
    }
    assert!(deepest < 0.02, "cloth sank {deepest:.4} m into the sphere");

    // And the middle of the sheet should be resting on top of it.
    let middle = sim.world.particles.translation(cloth.at(10, 10).index()).y;
    assert!(
        middle > 1.3,
        "the sheet did not drape over the sphere, middle y = {middle:.3}"
    );
    let _ = cloth;
}

#[test]
fn stiffer_cloth_holds_itself_out_further() {
    // Bending compliance is the dial between silk and cardboard, and it has to
    // point the right way. Isolating it took three attempts, and the two that
    // failed are worth recording because both looked obviously right:
    //
    // - *Hang it and measure the droop.* Gravity pulls a hanging sheet taut
    //   whatever its stiffness; both ends of the dial gave 1.29 m for a 1.30 m
    //   sheet.
    // - *Drape it over a bar and measure the fold.* A near-rigid sheet
    //   balanced on a narrow bar tips and slides off, which is correct
    //   behaviour and useless as a measurement.
    //
    // Pinning the centre row leaves bending as the only thing resisting the
    // two halves folding down, and horizontal reach then reads straight off
    // it: a stiff sheet stays out near its full half-length, a floppy one
    // hangs down and reaches nowhere.
    fn reach(bend_compliance: f32) -> f32 {
        let mut world = World::new();
        let desc = ClothDesc::new(9, 16, 0.08)
            .at(vec3(0.0, 3.0, 0.0))
            .with_bend_compliance(bend_compliance);
        let cloth = world.add_cloth(&desc);
        for col in 0..cloth.columns {
            world.particles.pin(cloth.at(col, 8).index());
        }
        let mut sim = Simulation::new(world);
        run(&mut sim, 5.0);
        let pin = sim.world.particles.translation(cloth.at(4, 8).index());
        let tip = sim.world.particles.translation(cloth.at(4, 15).index());
        // Horizontal only: how far out it holds itself, ignoring how far down.
        ((tip.x - pin.x).powi(2) + (tip.z - pin.z).powi(2)).sqrt()
    }

    let floppy = reach(1e-1);
    let stiff = reach(1e-9);
    assert!(
        stiff > floppy + 0.05,
        "stiff cloth reached {stiff:.3} m, floppy {floppy:.3} m — \
         expected the stiff one to hold itself out further"
    );
}

#[test]
fn drag_produces_a_terminal_velocity() {
    // The textbook consequence of `F = 1/2 rho v^2 Cd A`: a falling body stops
    // accelerating when drag balances weight, at `sqrt(2mg / (rho Cd A))`.
    let radius = 0.3f32;
    let density = 40.0f32; // light, so terminal velocity is reachable quickly
    let mut world = World::new();
    let ball = world.add(
        BodyDesc::sphere(radius)
            .at(vec3(0.0, 500.0, 0.0))
            .with_material(Material::new(density, 0.5, 0.0)),
    );
    let mut sim = Simulation::new(world);
    sim.world.angular_damping = 0.0;
    sim.world.linear_damping = 0.0;
    sim.air = AirModel::still();

    run(&mut sim, 12.0);

    let speed = sim.world.bodies.linear_velocity[ball.index()].length();
    let mass = (4.0 / 3.0) * std::f32::consts::PI * radius.powi(3) * density;
    let area = std::f32::consts::PI * radius * radius;
    let expected = (2.0 * mass * 9.81 / (1.225 * 0.47 * area)).sqrt();
    assert!(
        (speed - expected).abs() < 0.1 * expected,
        "terminal velocity was {speed:.2} m/s, theory says {expected:.2}"
    );
}

#[test]
fn wind_pushes_things_downwind() {
    let mut world = World::new();
    let ball = world.add(
        BodyDesc::sphere(0.4)
            .at(vec3(0.0, 50.0, 0.0))
            .with_material(Material::new(30.0, 0.5, 0.0)),
    );
    let mut sim = Simulation::new(world);
    sim.air = AirModel::with_wind(vec3(12.0, 0.0, 0.0));

    run(&mut sim, 3.0);

    let p = sim.world.bodies.translation(ball.index());
    assert!(p.x > 3.0, "the wind moved it only {:.3} m downwind", p.x);
    assert!(p.z.abs() < 0.05, "it drifted sideways to z = {:.3}", p.z);
}

#[test]
fn the_magnus_effect_curves_a_spinning_ball() {
    // Spin about +Y while travelling along +X puts the force along -Z or +Z
    // depending on the sign, and reversing the spin must reverse the curve.
    fn sideways(spin: f32) -> f32 {
        let mut world = World::new().with_gravity(Vec3::ZERO);
        world.angular_damping = 0.0;
        let ball = world.add(
            BodyDesc::sphere(0.35)
                .at(Vec3::ZERO)
                .with_velocity(vec3(25.0, 0.0, 0.0))
                .with_angular_velocity(vec3(0.0, spin, 0.0))
                .with_material(Material::new(120.0, 0.5, 0.0)),
        );
        let mut sim = Simulation::new(world);
        sim.air = AirModel {
            magnus: 0.35,
            ..AirModel::still()
        };
        run(&mut sim, 1.5);
        sim.world.bodies.translation(ball.index()).z
    }

    let clockwise = sideways(60.0);
    let anticlockwise = sideways(-60.0);
    assert!(
        clockwise.abs() > 0.2,
        "a spinning ball barely curved: {clockwise:.3} m"
    );
    assert!(
        clockwise * anticlockwise < 0.0,
        "reversing the spin must reverse the curve: {clockwise:.3} vs {anticlockwise:.3}"
    );
}

#[test]
fn a_flag_in_wind_lifts_and_stays_stable() {
    // The §13 payoff: per-triangle aero on cloth, which is what gives flags,
    // sails and parachutes.
    let mut world = World::new();
    let cloth = world.add_cloth(
        &ClothDesc::new(16, 10, 0.08)
            .at(vec3(0.0, 3.0, 0.0))
            .with_mass(0.3),
    );
    // Pin the hoist edge.
    for row in 0..cloth.rows {
        world.particles.pin(cloth.at(0, row).index());
    }
    let mut sim = Simulation::new(world);
    sim.air = AirModel::with_wind(vec3(9.0, 0.0, 0.0));

    let still_y = sim.world.particles.translation(cloth.at(15, 5).index()).y;
    run(&mut sim, 4.0);

    assert!(sim.world.particles.is_finite(), "the flag went non-finite");
    let tip = sim.world.particles.translation(cloth.at(15, 5).index());
    assert!(
        tip.x > 0.3,
        "the flag did not stream downwind, x = {:.3}",
        tip.x
    );
    // It should be held up by the airflow rather than hanging straight down.
    assert!(
        tip.y > still_y - 0.9,
        "the flag hung limp, y = {:.3}",
        tip.y
    );
    assert!(worst_stretch(&sim) < 0.1, "the flag stretched under load");
}

#[test]
fn a_fast_body_is_stopped_by_a_thin_wall() {
    // Speculative contact, from PLAN.md §29's v0.3 list. A projectile crossing
    // more than the wall's thickness in a single step would, with discrete
    // detection alone, simply be found on the far side — there is never a step
    // at which the two overlap.
    fn final_x(speed: f32, speculative: bool) -> f32 {
        let mut world = World::new().with_gravity(Vec3::ZERO);
        // A wall 4 cm thick.
        world.add(
            BodyDesc::cuboid(vec3(0.04, 6.0, 6.0))
                .at(vec3(2.0, 0.0, 0.0))
                .with_material(Material::CONCRETE)
                .fixed(),
        );
        let bullet = world.add(
            BodyDesc::sphere(0.05)
                .at(Vec3::ZERO)
                .with_velocity(vec3(speed, 0.0, 0.0))
                .with_material(Material::STEEL),
        );
        let mut sim = Simulation::new(world);
        sim.solver.speculative = speculative;
        run(&mut sim, 0.5);
        sim.world.bodies.translation(bullet.index()).x
    }

    // The wall spans x in [1.98, 2.02]. Which side the bullet ends up on is
    // the whole question, so that is what the test asks — not how far it got.
    //
    // At 40 m/s the bullet covers 17 cm per step, four times the wall's
    // thickness: there is no step at which the two overlap.
    let stopped = final_x(40.0, true);
    assert!(
        stopped < 1.98,
        "the bullet should stop on the near face, ended at {stopped:.3}"
    );

    // Without speculation it passes through and is discovered on the far side,
    // where the contact solver dutifully pushes it out. That is the failure
    // mode the feature exists to prevent, and it is worth pinning down so the
    // test cannot pass for the wrong reason.
    let tunnelled = final_x(40.0, false);
    assert!(
        tunnelled > 2.02,
        "expected the bullet through the wall without speculation, got {tunnelled:.3}"
    );
}

#[test]
fn speculative_contact_does_not_disturb_ordinary_resting() {
    // A feature that stops bullets and also makes a resting box hover would
    // not be worth having.
    let mut world = World::new();
    world.add(
        BodyDesc::cuboid(vec3(40.0, 2.0, 40.0))
            .at(vec3(0.0, -1.0, 0.0))
            .with_material(Material::CONCRETE)
            .fixed(),
    );
    let b = world.add(
        BodyDesc::cuboid(Vec3::ONE)
            .at(vec3(0.0, 0.5, 0.0))
            .with_material(Material::new(700.0, 0.45, 0.0)),
    );
    let mut sim = Simulation::new(world);
    run(&mut sim, 3.0);
    let y = sim.world.bodies.translation(b.index()).y;
    assert!(
        (y - 0.5).abs() < 0.01,
        "the box rests at {y:.4}, expected 0.5"
    );
}

#[test]
fn a_soft_body_squashes_on_impact_and_recovers() {
    use fusion_core::SoftBodyDesc;

    let mut world = World::new();
    world.add(
        BodyDesc::cuboid(vec3(20.0, 1.0, 20.0))
            .at(vec3(0.0, -0.5, 0.0))
            .with_material(Material::CONCRETE)
            .fixed(),
    );
    let block = world.add_soft_body(
        &SoftBodyDesc::new(Vec3::ONE, 4)
            .at(vec3(0.0, 3.0, 0.0))
            .with_compliance(3e-4),
    );
    let mut sim = Simulation::new(world);

    let height = |sim: &Simulation| {
        let ys: Vec<f32> = block
            .iter()
            .map(|h| sim.world.particles.translation(h.index()).y)
            .collect();
        ys.iter().cloned().fold(f32::MIN, f32::max) - ys.iter().cloned().fold(f32::MAX, f32::min)
    };
    assert!(
        (height(&sim) - 1.0).abs() < 1e-4,
        "the lattice should start a metre tall"
    );

    // Squash: at some point during the impact it is measurably shorter.
    let mut shortest = f32::MAX;
    for _ in 0..480 {
        sim.step();
        shortest = shortest.min(height(&sim));
    }
    assert!(
        shortest < 0.95,
        "the block never deformed, minimum height {shortest:.3}"
    );

    // Recover: once settled it is close to its original shape again.
    run(&mut sim, 4.0);
    assert!(sim.world.particles.is_finite());
    let settled = height(&sim);
    assert!(
        (settled - 1.0).abs() < 0.12,
        "the block did not recover its shape, height {settled:.3}"
    );

    // And it sits on the floor rather than in it.
    let lowest = block
        .iter()
        .map(|h| sim.world.particles.translation(h.index()).y)
        .fold(f32::MAX, f32::min);
    assert!(
        lowest > -0.05,
        "the block sank into the ground to {lowest:.3}"
    );
}
