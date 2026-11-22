//! What the step report and the stability check say about a scene, and
//! whether they say it about all five systems or only the rigid bodies.
//!
//! These exist because the diagnostics are what everything else is judged by.
//! A solver bug shows up as a number in a `StepReport`; if that number covers
//! bodies and quietly ignores cloth, soft bodies and fluid, then the whole
//! instrument reads zero on four fifths of the engine, and reads it
//! confidently. Three of the defects listed in `docs/STATUS.md` through v0.6
//! were exactly that, and this file is what stops them coming back.

use fusion_core::{BodyDesc, ClothDesc, FluidDesc, FluidParams, Material, World};
use fusion_math::{vec3, Vec3};
use fusion_sim::Simulation;

const SPACING: f32 = 0.05;

fn floor(world: &mut World) {
    world.add(
        BodyDesc::cuboid(vec3(4.0, 0.2, 4.0))
            .at(vec3(0.0, -0.1, 0.0))
            .fixed()
            .with_material(Material::CONCRETE),
    );
}

fn run(sim: &mut Simulation, seconds: f64) {
    let steps = (seconds / sim.solver_dt as f64).round() as u64;
    for _ in 0..steps {
        sim.step();
    }
}

// --- is_stable covers everything that can hold a NaN ---------------------

/// A NaN in a fluid particle is not a stable scene.
///
/// `is_stable` tested the rigid arrays alone until v0.5, which is precisely
/// backwards: cloth and fluid are the two systems most able to produce a NaN,
/// and they were the two it did not look at. Fixed, and unpinned until now —
/// every other test in the suite asserts `is_stable()` is *true*, so nothing
/// would have noticed it going back to answering that unconditionally.
#[test]
fn a_nan_in_a_fluid_particle_is_not_a_stable_scene() {
    let mut world = World::new();
    floor(&mut world);
    let fluid = world.add_fluid(
        &FluidDesc::block(Vec3::splat(0.2))
            .at(vec3(0.0, 1.0, 0.0))
            .with_params(FluidParams::water_at(SPACING)),
    );
    let mut sim = Simulation::new(world);
    sim.step();
    assert!(sim.is_stable(), "the scene is fine before the injection");

    sim.world.fluids[fluid.index()].particles.position[0].x = f64::NAN;
    assert!(
        !sim.is_stable(),
        "a NaN in a fluid particle has to be reported"
    );
}

/// And the same for a cloth particle.
#[test]
fn a_nan_in_a_cloth_particle_is_not_a_stable_scene() {
    let mut world = World::new();
    floor(&mut world);
    world.add_cloth(&ClothDesc::new(8, 8, 0.1).at(vec3(0.0, 1.0, 0.0)));
    let mut sim = Simulation::new(world);
    sim.step();
    assert!(sim.is_stable());

    sim.world.particles.velocity[0] = Vec3::splat(f32::NAN);
    assert!(
        !sim.is_stable(),
        "a NaN in a cloth particle's velocity has to be reported"
    );
}

// --- the contact count covers particles ----------------------------------

/// Water sitting on a floor is in contact with it, and the report says so.
///
/// A defect through v0.6: `contact_points` counts rigid-rigid manifold points,
/// so a dam break against a wall of boxes reported zero contacts — the one
/// number a person would check to find out whether the fluid had noticed the
/// wall at all.
#[test]
fn a_fluid_resting_on_a_body_reports_particle_contacts() {
    let mut world = World::new();
    floor(&mut world);
    world.add_fluid(
        &FluidDesc::block(Vec3::splat(0.2))
            .at(vec3(0.0, 0.4, 0.0))
            .with_params(FluidParams::water_at(SPACING)),
    );
    let mut sim = Simulation::new(world);
    run(&mut sim, 0.6);
    let report = sim.step();

    assert_eq!(
        report.contact_points, 0,
        "there is only one body, so there is nothing rigid to touch"
    );
    assert!(
        report.particle_contacts > 0,
        "the water is standing on the floor and reported {} contacts",
        report.particle_contacts
    );
    // A particle can be in contact with the floor once. More contacts than
    // particles would mean the number is counting solver passes.
    let particles = sim.world.fluid_particle_count();
    assert!(
        report.particle_contacts <= particles,
        "{} contacts for {particles} particles is a count of solves, not contacts",
        report.particle_contacts
    );
}

/// Cloth draped over a box is in contact with it too.
#[test]
fn cloth_landing_on_a_body_reports_particle_contacts() {
    let mut world = World::new();
    floor(&mut world);
    world.add_cloth(&ClothDesc::new(12, 12, 0.08).at(vec3(0.0, 0.5, 0.0)));
    let mut sim = Simulation::new(world);
    let before = sim.step();
    assert_eq!(
        before.particle_contacts, 0,
        "the sheet has not fallen yet at t = 0"
    );

    run(&mut sim, 1.2);
    let report = sim.step();
    assert!(
        report.particle_contacts > 0,
        "the sheet is lying on the floor and reported {} contacts",
        report.particle_contacts
    );
}

/// A scene made only of rigid bodies reports no particle contacts, which is
/// the other half of the claim: the new number is measuring particles, not
/// double-counting the rigid ones.
#[test]
fn a_rigid_only_scene_reports_no_particle_contacts() {
    let mut world = World::new();
    floor(&mut world);
    for k in 0..3 {
        world.add(
            BodyDesc::cuboid(Vec3::splat(0.4))
                .at(vec3(0.0, 0.2 + k as f32 * 0.42, 0.0))
                .with_material(Material::WOOD),
        );
    }
    let mut sim = Simulation::new(world);
    run(&mut sim, 1.0);
    let report = sim.step();

    assert!(
        report.contact_points > 0,
        "three boxes on a floor are in contact with something"
    );
    assert_eq!(report.particle_contacts, 0, "there are no particles");
}

/// The count is a state of the scene, not an artefact of how hard the solver
/// worked on it. Doubling the iteration count must not double the contacts.
#[test]
fn the_particle_contact_count_does_not_scale_with_iterations() {
    fn settled(iterations: u32) -> usize {
        let mut world = World::new();
        floor(&mut world);
        world.add_fluid(
            &FluidDesc::block(Vec3::splat(0.2))
                .at(vec3(0.0, 0.4, 0.0))
                .with_params(FluidParams::water_at(SPACING)),
        );
        let mut sim = Simulation::new(world);
        sim.solver.fluid_iterations = iterations;
        run(&mut sim, 0.6);
        sim.step().particle_contacts
    }

    let once = settled(2);
    let twice = settled(4);
    assert!(once > 0 && twice > 0);
    // The two runs are different simulations — more iterations is a stiffer
    // fluid, so the pool settles slightly differently — but a count that
    // accumulated over iterations would come back about twice as large, and
    // this is nowhere near that.
    let ratio = twice as f32 / once as f32;
    assert!(
        (0.5..1.6).contains(&ratio),
        "{once} contacts at 2 iterations and {twice} at 4 is a per-iteration sum"
    );
}
