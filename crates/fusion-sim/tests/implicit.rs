//! Stiff solids solved implicitly instead of with substeps.
//!
//! What it has to get right: a stiff block takes a step or two a frame where
//! it took a dozen, and is still a solid — it lands without making energy,
//! a plastic one absorbs its landing as it does with substeps, a spinning one
//! keeps spinning; in water a light block still floats and a heavy one still
//! sinks, with the water no longer taking a stiff block's substeps; and where
//! it would not halve the substeps it is not used at all.

use fusion_core::{BodyDesc, FluidDesc, FluidParams, Material, SolidDesc, SolidMaterial, World};
use fusion_math::{to_f32, vec3, Vec3};
use fusion_sim::{ContinuumSolver, FluidModel, Simulation};

fn floor(world: &mut World) {
    world.add(
        BodyDesc::cuboid(vec3(6.0, 1.0, 6.0))
            .at(vec3(0.0, -0.5, 0.0))
            .with_material(Material::CONCRETE)
            .fixed(),
    );
}

fn run(world: World, implicit: bool) -> Simulation {
    let mut sim = Simulation::new(world);
    sim.continuum = ContinuumSolver::Unified;
    sim.unified.implicit_solids = implicit;
    sim
}

fn dropped(material: SolidMaterial, implicit: bool) -> Simulation {
    let mut world = World::new();
    floor(&mut world);
    world.add_solid(
        &SolidDesc::block(Vec3::splat(0.2))
            .at(vec3(0.0, 1.0, 0.0))
            .of(material),
    );
    run(world, implicit)
}

/// Kinetic and gravitational energy over `m g`: the height the block could
/// reach with all of it.
fn height_of_energy(sim: &Simulation) -> f32 {
    let solid = &sim.world.solids[0];
    let mass = solid.params.point_mass();
    let mut energy = 0.0;
    for i in 0..solid.len() {
        energy += 0.5 * mass * solid.particles.velocity[i].length_squared()
            + mass * 9.81 * to_f32(solid.particles.position[i]).y;
    }
    energy / (solid.len() as f32 * mass * 9.81)
}

/// A porcelain block from a metre: a dozen substeps a frame explicitly, one
/// or two implicitly — and never more energy than it fell with. The first
/// implicit solve left the stress at the start of the step and the block
/// came off the floor with 40% more.
#[test]
fn a_stiff_block_lands_on_a_step_or_two_without_making_energy() {
    let mut explicit = dropped(SolidMaterial::PORCELAIN, false);
    explicit.step();
    let substeps_explicit = explicit.unified_report().substeps;

    let mut sim = dropped(SolidMaterial::PORCELAIN, true);
    let start = height_of_energy(&sim);
    let mut substeps = 0;
    let mut most = 0.0f32;
    for _ in 0..360 {
        sim.step();
        let report = sim.unified_report();
        assert!(!report.refused, "{report:?}");
        substeps += report.substeps;
        most = most.max(height_of_energy(&sim));
    }
    let mean = substeps as f32 / 360.0;
    assert!(substeps_explicit >= 10, "explicitly {substeps_explicit}");
    assert!(mean < 2.0, "implicitly {mean} substeps a step");
    assert!(
        most <= start * 1.01,
        "never more energy than it fell with: {most} against {start}"
    );
    for p in &sim.world.solids[0].particles.position {
        assert!(to_f32(*p).is_finite() && to_f32(*p).y > -0.05);
    }
}

/// An alloy block yields when it lands and stays down, as it does with
/// substeps: the implicit stiffness knows where a point is yielding. Without
/// that it sprang back up from an impact it should have absorbed.
#[test]
fn a_plastic_block_absorbs_its_landing() {
    let mut sim = dropped(SolidMaterial::ALLOY, true);
    for _ in 0..360 {
        sim.step();
    }
    let after = height_of_energy(&sim);
    assert!(
        after < 0.2,
        "it fell a metre and kept the energy of {after} m"
    );
}

/// The warped stiffness takes rotation out before it acts, so a spinning
/// stiff bar keeps spinning. Not quite as well as with substeps: a step
/// integrates a rotation as a slight stretch too, and backward Euler damps
/// what that stretch stores — measured, 95% of the spin left after half a
/// second against 98%.
#[test]
fn a_spinning_stiff_bar_keeps_spinning() {
    let spin = |implicit: bool| {
        let mut world = World::new();
        world.gravity = Vec3::ZERO;
        world.add_solid(
            &SolidDesc::block(vec3(0.6, 0.2, 0.2))
                .at(vec3(0.0, 1.0, 0.0))
                .of(SolidMaterial::RUBBER),
        );
        let mut sim = run(world, implicit);
        let centre = vec3(0.0, 1.0, 0.0);
        let solid = &mut sim.world.solids[0];
        for i in 0..solid.len() {
            let r = to_f32(solid.particles.position[i]) - centre;
            solid.particles.velocity[i] = vec3(0.0, 4.0, 0.0).cross(r);
        }
        let angular = |sim: &Simulation| {
            let solid = &sim.world.solids[0];
            (0..solid.len())
                .map(|i| {
                    let r = to_f32(solid.particles.position[i]) - centre;
                    r.cross(solid.particles.velocity[i]).y
                })
                .sum::<f32>()
        };
        let before = angular(&sim);
        for _ in 0..120 {
            sim.step();
        }
        angular(&sim) / before
    };
    let explicit = spin(false);
    let implicit = spin(true);
    assert!(
        implicit > 0.92 && explicit > 0.95,
        "spin kept: {implicit} implicitly, {explicit} with substeps"
    );
}

/// Stiff blocks in projected water: the water no longer takes their
/// substeps, and a light block still floats and a heavy one still sinks — it
/// sank, the light one, while its stiffness was solved apart from the water
/// it was in contact with.
#[test]
fn stiff_blocks_in_projected_water_float_and_sink_on_a_step_or_two() {
    let build = |implicit: bool| {
        let mut world = World::new();
        floor(&mut world);
        let size = Vec3::splat(0.8);
        let t = 0.2;
        for (centre, extent) in [
            (vec3(-0.5, 0.4, 0.0), vec3(t, size.y, size.z + 2.0 * t)),
            (vec3(0.5, 0.4, 0.0), vec3(t, size.y, size.z + 2.0 * t)),
            (vec3(0.0, 0.4, -0.5), vec3(size.x, size.y, t)),
            (vec3(0.0, 0.4, 0.5), vec3(size.x, size.y, t)),
        ] {
            world.add(
                BodyDesc::cuboid(extent)
                    .at(centre)
                    .with_material(Material::CONCRETE)
                    .fixed(),
            );
        }
        world.add_fluid(
            &FluidDesc::block(vec3(0.8, 0.4, 0.8))
                .at(vec3(0.0, 0.21, 0.0))
                .with_params(FluidParams {
                    spacing: 0.04,
                    smoothing_radius: 0.08,
                    ..FluidParams::WATER
                }),
        );
        for (x, density) in [(-0.2, 300.0), (0.2, 3000.0)] {
            world.add_solid(
                &SolidDesc::block(Vec3::splat(0.2))
                    .at(vec3(x, 0.62, 0.0))
                    .of(SolidMaterial {
                        density,
                        youngs_modulus: 2e5,
                        ..SolidMaterial::JELLY
                    }),
            );
        }
        let mut sim = run(world, implicit);
        sim.unified.fluid_model = FluidModel::Incompressible;
        sim
    };
    let mut explicit = build(false);
    explicit.step();
    let substeps_explicit = explicit.unified_report().substeps;

    let mut sim = build(true);
    let mut substeps = 0;
    for _ in 0..360 {
        sim.step();
        substeps += sim.unified_report().substeps;
    }
    let centre = |s: usize| {
        let solid = &sim.world.solids[s];
        (0..solid.len())
            .map(|i| to_f32(solid.particles.position[i]).y)
            .sum::<f32>()
            / solid.len() as f32
    };
    let (light, heavy) = (centre(0), centre(1));
    let mean = substeps as f32 / 360.0;
    assert!(substeps_explicit >= 6, "explicitly {substeps_explicit}");
    assert!(mean < 2.0, "implicitly {mean} substeps a step");
    assert!(light > 0.38, "the light block floats: centre at {light}");
    assert!(heavy < 0.2, "the heavy block sinks: centre at {heavy}");
}

/// Where water already asks for most of the substeps a stiff solid would,
/// the implicit solve would cost more than it saves, and it is not used: the
/// run is the explicit one, bit for bit.
#[test]
fn where_it_would_not_halve_the_substeps_nothing_changes() {
    let build = |implicit: bool| {
        let mut world = World::new();
        floor(&mut world);
        world.add_fluid(
            &FluidDesc::block(vec3(0.4, 0.2, 0.4))
                .at(vec3(0.0, 0.11, 0.0))
                .with_params(FluidParams {
                    spacing: 0.04,
                    smoothing_radius: 0.08,
                    ..FluidParams::WATER
                }),
        );
        world.add_solid(
            &SolidDesc::block(Vec3::splat(0.2))
                .at(vec3(0.0, 0.5, 0.0))
                .of(SolidMaterial::JELLY),
        );
        run(world, implicit)
    };
    let mut a = build(false);
    let mut b = build(true);
    for _ in 0..60 {
        a.step();
        b.step();
    }
    assert_eq!(b.unified_report().implicit.stiff_points, 0);
    assert_eq!(fusion_sim::state_hash(&a), fusion_sim::state_hash(&b));
}
