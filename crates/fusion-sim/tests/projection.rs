//! The incompressible fluid model: a pressure projection on the unified grid.
//!
//! What it has to get right: still water stands level and quiet with one
//! substep a frame, where the weakly compressible model needs several; a dam
//! break keeps its volume — no point lost into the tank, none crowded into a
//! corner — and comes to rest; and a light solid still floats and a heavy one
//! still sinks, which is the pressure reaching the solid through the grid.

use fusion_core::{BodyDesc, FluidDesc, FluidParams, Material, SolidDesc, SolidMaterial, World};
use fusion_math::{to_f32, vec3, Vec3};
use fusion_mpm::unified::FluidModel;
use fusion_sim::{ContinuumSolver, Simulation};

/// A tank: a floor and four walls, `size` inside, open at the top. Built of
/// boxes that touch, as one is built in the editor, so its seams count.
fn tank(world: &mut World, size: Vec3) {
    world.add(
        BodyDesc::cuboid(vec3(6.0, 1.0, 6.0))
            .at(vec3(0.0, -0.5, 0.0))
            .with_material(Material::CONCRETE)
            .fixed(),
    );
    let t = 0.2;
    for (centre, extent) in [
        (
            vec3(-size.x * 0.5 - t * 0.5, size.y * 0.5, 0.0),
            vec3(t, size.y, size.z + 2.0 * t),
        ),
        (
            vec3(size.x * 0.5 + t * 0.5, size.y * 0.5, 0.0),
            vec3(t, size.y, size.z + 2.0 * t),
        ),
        (
            vec3(0.0, size.y * 0.5, -size.z * 0.5 - t * 0.5),
            vec3(size.x, size.y, t),
        ),
        (
            vec3(0.0, size.y * 0.5, size.z * 0.5 + t * 0.5),
            vec3(size.x, size.y, t),
        ),
    ] {
        world.add(
            BodyDesc::cuboid(extent)
                .at(centre)
                .with_material(Material::CONCRETE)
                .fixed(),
        );
    }
}

fn water() -> FluidParams {
    FluidParams {
        spacing: 0.04,
        smoothing_radius: 0.08,
        ..FluidParams::WATER
    }
}

fn run(world: World, model: FluidModel) -> Simulation {
    let mut sim = Simulation::new(world);
    sim.continuum = ContinuumSolver::Unified;
    sim.unified.fluid_model = model;
    sim
}

fn still_water(model: FluidModel) -> Simulation {
    let mut world = World::new();
    tank(&mut world, Vec3::splat(0.8));
    world.add_fluid(
        &FluidDesc::block(vec3(0.8, 0.4, 0.8))
            .at(vec3(0.0, 0.21, 0.0))
            .with_params(water()),
    );
    run(world, model)
}

fn fastest(sim: &Simulation) -> f32 {
    sim.world.fluids[0]
        .particles
        .velocity
        .iter()
        .map(|v| v.length())
        .fold(0.0, f32::max)
}

/// The point of projecting: no sound to resolve, so no substeps for it.
#[test]
fn still_water_stands_level_and_quiet_on_one_substep_a_frame() {
    let mut weak = still_water(FluidModel::WeaklyCompressible);
    let mut weak_substeps = 0;
    for _ in 0..30 {
        weak.step();
        weak_substeps = weak_substeps.max(weak.unified_report().substeps);
    }
    let mut sim = still_water(FluidModel::Incompressible);
    let top = |sim: &Simulation| {
        sim.world.fluids[0]
            .particles
            .position
            .iter()
            .map(|p| p.y as f32)
            .fold(f32::MIN, f32::max)
    };
    let before = top(&sim);
    for _ in 0..480 {
        sim.step();
        let report = sim.unified_report();
        assert_eq!(report.substeps, 1, "one substep a frame");
        assert!(report.projection.converged, "{:?}", report.projection);
    }
    assert!(weak_substeps > 1, "the weak model needs {weak_substeps}");
    let after = top(&sim);
    assert!(
        (after - before).abs() < 0.015,
        "the surface stays where it was: {before} -> {after}"
    );
    let fastest = fastest(&sim);
    assert!(fastest < 0.03, "and still: fastest point {fastest} m/s");
}

/// The column of water from `unified.rs`, projected. It once settled at half
/// its depth: points slid into the seams where the tank's walls meet its
/// floor, and crowded into its corners where no cell could see them.
#[test]
fn a_projected_dam_break_keeps_its_volume_and_comes_to_rest() {
    let inside = vec3(1.2, 1.0, 0.4);
    let mut world = World::new();
    tank(&mut world, inside);
    world.add_fluid(
        &FluidDesc::block(vec3(0.4, 0.6, 0.4))
            .at(vec3(-0.4, 0.31, 0.0))
            .with_params(water()),
    );
    let mut sim = run(world, FluidModel::Incompressible);
    for _ in 0..720 {
        sim.step();
    }
    let fluid = &sim.world.fluids[0];
    let n = fluid.len();
    let margin = 0.03;
    let escaped = fluid
        .particles
        .position
        .iter()
        .map(|p| to_f32(*p))
        .filter(|p| {
            p.x.abs() > inside.x * 0.5 + margin
                || p.z.abs() > inside.z * 0.5 + margin
                || p.y < -margin
        })
        .count();
    assert!(escaped * 200 <= n, "{escaped} of {n} points left the tank");
    // 0.096 m³ over a 1.2 × 0.4 floor stands 0.2 deep, so its points' median
    // height is near 0.1 — less a point's half-spacing at the floor, which a
    // point resting on it gives up. It was 0.04.
    let mut heights: Vec<f32> = fluid
        .particles
        .position
        .iter()
        .map(|p| p.y as f32)
        .collect();
    heights.sort_by(f32::total_cmp);
    let median = heights[n / 2];
    assert!(
        median > 0.08,
        "median height {median}: the volume went somewhere"
    );
    let kinetic: f32 = fluid
        .particles
        .velocity
        .iter()
        .map(|v| 0.5 * fluid.params.particle_mass() * v.length_squared())
        .sum();
    assert!(kinetic < 0.05, "at rest after twelve seconds: {kinetic} J");
    assert!(
        fluid.volume_ratio.iter().all(|&j| j == 1.0),
        "a projected point's volume ratio is held at one"
    );
}

/// Buoyancy, which is fluid pressure on a solid through the shared grid, has
/// to survive the pressure becoming a solve. Both blocks in one tank, so one
/// run answers for both.
#[test]
fn a_light_solid_floats_on_projected_water_and_a_heavy_one_sinks() {
    let mut world = World::new();
    tank(&mut world, Vec3::splat(0.8));
    world.add_fluid(
        &FluidDesc::block(vec3(0.8, 0.4, 0.8))
            .at(vec3(0.0, 0.21, 0.0))
            .with_params(water()),
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
    let mut sim = run(world, FluidModel::Incompressible);
    for _ in 0..720 {
        sim.step();
    }
    let centre = |s: usize| {
        let solid = &sim.world.solids[s];
        (0..solid.len())
            .map(|i| to_f32(solid.particles.position[i]).y)
            .sum::<f32>()
            / solid.len() as f32
    };
    let (light, heavy) = (centre(0), centre(1));
    // The pool is 0.4 m deep. Foam at 0.3 g/cc rides with most of itself
    // above the surface; a block three times water's density is on the
    // bottom.
    assert!(light > 0.38, "the light block floats: centre at {light}");
    assert!(heavy < 0.2, "the heavy block sinks: centre at {heavy}");
    // And the water the heavy block drove onto the floor came apart again.
    // It once stayed a packed sheet with a layer a third short above it,
    // which the grid's density, smoothed over three cells, could not see.
    let mut layers = [0usize; 10];
    for p in &sim.world.fluids[0].particles.position {
        let layer = (p.y as f32 / 0.04).floor();
        if (0.0..10.0).contains(&layer) {
            layers[layer as usize] += 1;
        }
    }
    let body = layers[2..7].iter().sum::<usize>() as f32 / 5.0;
    let above_floor = layers[1] as f32 / body;
    assert!(
        above_floor > 0.78,
        "the layer above the floor holds {above_floor:.2} of the water's density: {layers:?}"
    );
}
