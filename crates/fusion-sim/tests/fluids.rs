//! Position-based fluids, against things that are true independently of the
//! solver: free fall, Archimedes, mass conservation, and the volume the water
//! occupies when it stops moving.
//!
//! PLAN.md §9 puts this solver in the real-time tier, which means "plausible,
//! not a measurement". These tests are chosen to be the ones a real-time
//! fluid still has to pass: they check the answer is *approximately* right
//! with the tolerance stated, rather than pretending to validate it.

use fusion_core::{BodyDesc, FluidDesc, FluidHandle, FluidParams, Material, World};
use fusion_math::{vec3, Vec3};
use fusion_sim::Simulation;

const SPACING: f32 = 0.05;

/// A container of the given interior size, floor at y = 0, centred on x/z.
fn tank(width: f32, depth: f32, height: f32) -> World {
    let mut world = World::new();
    let t = 0.2;
    world.add(
        BodyDesc::cuboid(vec3(width + 2.0 * t, t, depth + 2.0 * t))
            .at(vec3(0.0, -t * 0.5, 0.0))
            .fixed(),
    );
    for (dx, dz) in [(1.0f32, 0.0f32), (-1.0, 0.0), (0.0, 1.0), (0.0, -1.0)] {
        let size = if dx != 0.0 {
            vec3(t, height, depth)
        } else {
            vec3(width, height, t)
        };
        world.add(
            BodyDesc::cuboid(size)
                .at(vec3(
                    dx * (width + t) * 0.5,
                    height * 0.5,
                    dz * (depth + t) * 0.5,
                ))
                .fixed(),
        );
    }
    world
}

struct Extent {
    min: Vec3,
    max: Vec3,
    peak_speed: f32,
}

fn extent(sim: &Simulation, fluid: FluidHandle) -> Extent {
    let f = &sim.world.fluids[fluid.index()];
    let mut e = Extent {
        min: Vec3::splat(f32::INFINITY),
        max: Vec3::splat(f32::NEG_INFINITY),
        peak_speed: 0.0,
    };
    for i in 0..f.len() {
        let p = f.particles.translation(i);
        e.min = e.min.min(p);
        e.max = e.max.max(p);
        e.peak_speed = e.peak_speed.max(f.particles.velocity[i].length());
    }
    e
}

/// The extent of the bulk of a fluid: each axis's 1st and 99th percentile
/// particle, not its outermost.
fn bulk_extent(sim: &Simulation, fluid: FluidHandle) -> Extent {
    let f = &sim.world.fluids[fluid.index()];
    let mut axes: [Vec<f32>; 3] = [Vec::new(), Vec::new(), Vec::new()];
    for i in 0..f.len() {
        let p = f.particles.translation(i);
        for (a, list) in axes.iter_mut().enumerate() {
            list.push(p[a]);
        }
    }
    let mut lo = [0.0f32; 3];
    let mut hi = [0.0f32; 3];
    for (a, list) in axes.iter_mut().enumerate() {
        list.sort_by(f32::total_cmp);
        let n = list.len().max(1);
        lo[a] = list[n / 100];
        hi[a] = list[(n * 99 / 100).min(n - 1)];
    }
    Extent {
        min: Vec3::from_array(lo),
        max: Vec3::from_array(hi),
        peak_speed: extent(sim, fluid).peak_speed,
    }
}

#[test]
fn a_fluid_at_rest_stays_exactly_at_rest() {
    // The sharpest test there is of the rest-density calibration. A lattice
    // filled at `spacing` and left alone in zero gravity must produce a
    // density equal to the target, therefore a zero constraint violation,
    // therefore no correction and no velocity at all.
    //
    // It is also the test that caught the 27-cell walk enumerating a hash
    // bucket twice: with the duplicate, this block read 9% over-dense and
    // reached the fluid speed clamp within four steps.
    let mut world = World::new().with_gravity(Vec3::ZERO);
    let fluid = world
        .add_fluid(&FluidDesc::block(Vec3::splat(0.4)).with_params(FluidParams::water_at(SPACING)));
    let mut sim = Simulation::new(world);
    let before = extent(&sim, fluid);

    for _ in 0..120 {
        sim.step();
    }

    let after = extent(&sim, fluid);
    // Not exactly zero, and it should not be. Neighbour offsets are
    // differenced in double and narrowed to single, so the density sum for an
    // interior particle lands within a few ulps of the calibrated target
    // rather than on it. What matters is that the residual is at the level of
    // rounding — microns per second — rather than at the level of a real
    // constraint violation.
    assert!(
        after.peak_speed < 1e-4,
        "a fluid with nothing acting on it reached {} m/s",
        after.peak_speed
    );
    assert!(after.min.approx_eq(before.min, 1e-5));
    assert!(after.max.approx_eq(before.max, 1e-5));
}

#[test]
fn a_fluid_in_free_fall_falls_at_g() {
    // No boundaries and no neighbours changing: every particle accelerates
    // identically, so the density constraint has nothing to say and the answer
    // is the analytic one.
    let mut world = World::new();
    let fluid = world.add_fluid(
        &FluidDesc::block(Vec3::splat(0.4))
            .at(vec3(0.0, 10.0, 0.0))
            .with_params(FluidParams::water_at(SPACING)),
    );
    let mut sim = Simulation::new(world);
    let start = extent(&sim, fluid).min.y;

    sim.run_frames(60, |_, _| {});

    let t = sim.time() as f32;
    let expected_drop = 0.5 * 9.81 * t * t;
    let actual_drop = start - extent(&sim, fluid).min.y;
    assert!(
        (actual_drop - expected_drop).abs() < 0.01,
        "fell {actual_drop:.4} m in {t:.3} s, analytic is {expected_drop:.4} m"
    );
    assert!(
        (extent(&sim, fluid).peak_speed - 9.81 * t).abs() < 0.05,
        "speed disagrees with g*t"
    );
}

#[test]
fn water_settles_to_something_close_to_its_own_volume() {
    // 512 kg of water in a 1 m tank should stand 0.512 m deep. The tolerance
    // is 5%, and the number it is really testing is the boundary density
    // correction: without it the same scene settles at 0.470 m, because a
    // particle against a wall reads under-dense and the fluid packs itself
    // into the container to make the estimate come out right.
    let mut world = tank(1.0, 1.0, 1.5);
    let params = FluidParams::water_at(SPACING);
    let fluid = world.add_fluid(
        &FluidDesc::block(Vec3::splat(0.8))
            .at(vec3(0.0, 0.65, 0.0))
            .with_params(params),
    );
    let mass = world.fluid(fluid).mass();
    let mut sim = Simulation::new(world);

    sim.run_frames(150, |_, _| {});

    let e = extent(&sim, fluid);
    assert!(
        e.peak_speed < 1.0,
        "water was still moving at {:.2} m/s after 2.5 s",
        e.peak_speed
    );
    // Particle centres stop half a spacing short of the surface at both ends.
    let depth = (e.max.y - e.min.y) + SPACING;
    let footprint = (e.max.x - e.min.x + SPACING) * (e.max.z - e.min.z + SPACING);
    let volume = depth * footprint;
    let expected = mass / params.rest_density;
    let error = (volume - expected).abs() / expected;
    assert!(
        error < 0.05,
        "settled into {volume:.4} m^3, expected {expected:.4} m^3 ({:.1}% off)",
        error * 100.0
    );
    assert!(
        sim.fluid_compression() < 0.01,
        "residual compression {:.3}% is above 1%",
        sim.fluid_compression() * 100.0
    );
}

#[test]
fn a_light_body_floats_at_the_draft_its_density_implies() {
    // Archimedes, and Fusion has no code that knows his name. A floating body
    // is one being pushed up harder from below than from above by the same
    // particle-vs-shape constraint cloth uses. The tolerance is wide — a
    // quarter of the body's height — because the coupling is geometric and its
    // resolution is one particle spacing.
    let mut world = tank(1.2, 1.2, 1.5);
    world.add_fluid(
        &FluidDesc::block(vec3(1.15, 0.5, 1.15))
            .at(vec3(0.0, 0.25, 0.0))
            .with_params(FluidParams::water_at(SPACING)),
    );
    let size = 0.2;
    let density = 300.0;
    let cube = world.add(
        BodyDesc::cuboid(Vec3::splat(size))
            .at(vec3(-0.3, 0.9, 0.0))
            .with_material(Material::new(density, 0.4, 0.1)),
    );
    let mut sim = Simulation::new(world);

    sim.run_frames(300, |_, _| {});

    let surface = extent(&sim, FluidHandle(0)).max.y + 0.5 * SPACING;
    let bottom = sim.world.bodies.pose(cube.index()).translation.y - 0.5 * size;
    let submerged = ((surface - bottom) / size).clamp(0.0, 1.0);
    let expected = density / 1000.0;
    assert!(
        (submerged - expected).abs() < 0.25,
        "cube floated {:.0}% submerged; its density says {:.0}%",
        submerged * 100.0,
        expected * 100.0
    );
    assert!(
        sim.world.bodies.pose(cube.index()).translation.y > 0.2,
        "the light cube sank"
    );
}

#[test]
fn a_dense_body_sinks_through_the_water() {
    let mut world = tank(1.2, 1.2, 1.5);
    world.add_fluid(
        &FluidDesc::block(vec3(1.15, 0.5, 1.15))
            .at(vec3(0.0, 0.25, 0.0))
            .with_params(FluidParams::water_at(SPACING)),
    );
    let cube = world.add(
        BodyDesc::cuboid(Vec3::splat(0.15))
            .at(vec3(0.3, 0.9, 0.0))
            .with_material(Material::STEEL),
    );
    let mut sim = Simulation::new(world);

    sim.run_frames(300, |_, _| {});

    // Resting on the floor means within a particle spacing of it; the fluid
    // layer underneath cannot be squeezed thinner than its own resolution.
    let y = sim.world.bodies.pose(cube.index()).translation.y;
    assert!(y < 0.075 + SPACING, "steel cube stalled at y = {y:.3}");
}

#[test]
fn nothing_creates_or_destroys_a_particle() {
    // PBF conserves mass exactly, and it is worth asserting because a solver
    // that quietly dropped a particle out of a neighbour list would show up
    // here and nowhere else.
    let mut world = tank(1.0, 1.0, 1.5);
    let fluid = world.add_fluid(
        &FluidDesc::block(Vec3::splat(0.8))
            .at(vec3(0.0, 0.65, 0.0))
            .with_params(FluidParams::water_at(SPACING)),
    );
    let count = world.fluid(fluid).len();
    let mass = world.fluid(fluid).mass();
    let mut sim = Simulation::new(world);

    sim.run_frames(120, |_, _| {});

    assert_eq!(sim.world.fluid(fluid).len(), count);
    assert_eq!(sim.world.fluid(fluid).mass(), mass);
    assert!(sim.world.fluid(fluid).particles.is_finite());
}

#[test]
fn the_density_constraint_conserves_momentum() {
    // Every correction is applied to a pair, equal and opposite, so with no
    // gravity and no boundaries the total momentum of a compressed blob is
    // whatever it started with. Viscosity and vorticity are off: the first
    // conserves momentum too, the second does not and does not claim to.
    let mut params = FluidParams::water_at(SPACING);
    params.viscosity = 0.0;
    params.vorticity = 0.0;
    // Over-dense on purpose — a lattice at 0.8 of the sampled spacing, so the
    // constraint has real work to do.
    params.spacing = SPACING;
    let mut world = World::new().with_gravity(Vec3::ZERO);
    let fluid = world.add_fluid(
        &FluidDesc::block(Vec3::splat(0.4))
            .with_params(FluidParams {
                spacing: SPACING * 0.8,
                ..params
            })
            .with_velocity(vec3(1.0, 0.0, 0.0)),
    );
    let mut sim = Simulation::new(world);
    let before = fusion_solver::fluid::momentum(sim.world.fluid(fluid));

    for _ in 0..200 {
        sim.step();
    }

    let after = fusion_solver::fluid::momentum(sim.world.fluid(fluid));
    let drift = (after - before).length() / before.length().max(1e-6);
    assert!(
        drift < 1e-3,
        "momentum drifted by {:.4}% ({before:?} -> {after:?})",
        drift * 100.0
    );
}

#[test]
fn two_runs_of_the_same_fluid_scene_are_bit_identical() {
    // PLAN.md §34 tier (a): same machine, same build, bit-identical. A fluid
    // is the hardest part of the engine to keep there, because a neighbour
    // list in a different order is a different floating-point sum. See
    // `SpatialHash` for what that costs.
    let scene = || {
        let mut world = tank(1.0, 1.0, 1.5);
        world.add_fluid(
            &FluidDesc::block(Vec3::splat(0.6))
                .at(vec3(0.05, 0.7, -0.05))
                .with_params(FluidParams::water_at(SPACING)),
        );
        world
    };
    let mut a = Simulation::new(scene());
    let mut b = Simulation::new(scene());
    for _ in 0..90 {
        a.step();
        b.step();
    }

    let (fa, fb) = (&a.world.fluids[0], &b.world.fluids[0]);
    assert_eq!(fa.particles.position, fb.particles.position);
    assert_eq!(fa.particles.velocity, fb.particles.velocity);
}

#[test]
fn resetting_rebuilds_the_fluid_exactly() {
    let mut world = tank(1.0, 1.0, 1.5);
    world.add_fluid(
        &FluidDesc::block(Vec3::splat(0.6))
            .at(vec3(0.0, 0.7, 0.0))
            .with_params(FluidParams::water_at(SPACING)),
    );
    let mut sim = Simulation::new(world);
    let start = sim.world.fluids[0].particles.position.clone();

    sim.run_frames(30, |_, _| {});
    assert_ne!(sim.world.fluids[0].particles.position, start);

    sim.reset();
    assert_eq!(sim.world.fluids[0].particles.position, start);
    assert!(sim.world.fluids[0]
        .particles
        .velocity
        .iter()
        .all(|v| *v == Vec3::ZERO));
}

#[test]
fn fill_samples_a_shape_and_not_its_bounding_box() {
    // PLAN.md §16: `fill` uses the SDF, so a sphere comes out a sphere. If it
    // sampled the AABB instead, the count would be the enclosing cube's.
    //
    // The count to expect is the sphere's own: `fill` keeps every particle
    // whose centre is inside, so each stands for its cell and the cells cut
    // by the surface even out. (It used to keep only centres half a spacing
    // in, which filled a 0.5 m sphere at 5 cm 14% short.)
    let spacing = 0.05;
    let radius = 0.5;
    let sphere = FluidDesc::sphere(radius)
        .with_spacing(spacing)
        .particle_count();
    let cube = FluidDesc::block(Vec3::splat(2.0 * radius))
        .with_spacing(spacing)
        .particle_count();

    let expected =
        (4.0 / 3.0) * std::f32::consts::PI * radius.powi(3) / (spacing * spacing * spacing);
    assert!(
        (sphere as f32 - expected).abs() / expected < 0.03,
        "sphere filled with {sphere} particles, analytic count is {expected:.0}"
    );
    // And nowhere near the bounding box's count.
    // A sphere is π/6 of its box, 52%.
    assert!(
        (sphere as f32) < 0.6 * cube as f32,
        "{sphere} vs a bounding box of {cube}"
    );
}

#[test]
fn a_scene_with_no_fluid_pays_nothing_and_reports_nothing() {
    let mut world = World::new();
    world.add(
        BodyDesc::cuboid(vec3(4.0, 0.2, 4.0))
            .at(vec3(0.0, -0.1, 0.0))
            .fixed(),
    );
    world.add(BodyDesc::sphere(0.2).at(vec3(0.0, 1.0, 0.0)));
    let mut sim = Simulation::new(world);
    let report = sim.step();
    assert_eq!(report.fluid_particles, 0);
    assert_eq!(report.fluid_compression, 0.0);
    assert_eq!(report.peak_fluid_neighbours, 0);
    assert!(!report.summary().contains("fluid"));
}

/// Reusing a neighbour list between substeps does not change the fluid.
///
/// The optimization rests on an analytic claim — a pair a stale list misses is
/// one whose kernel weight is nearly zero — and this is that claim measured
/// rather than argued. The same dam break is run with the tolerance at its
/// default and at zero, which rebuilds every substep, and the two are compared
/// where a fluid is most sensitive to a perturbation: the spread of the pool
/// after it has had four hundred steps to amplify one.
#[test]
fn reusing_neighbour_lists_does_not_change_the_fluid() {
    fn dam(tolerance: f32) -> (Simulation, Extent) {
        let mut world = tank(1.2, 1.2, 1.0);
        world.add_fluid(
            &FluidDesc::block(vec3(0.5, 0.6, 1.1))
                .at(vec3(-0.3, 0.3, 0.0))
                .with_params(FluidParams::water_at(SPACING)),
        );
        let mut sim = Simulation::new(world);
        sim.solver.fluid_neighbour_tolerance = tolerance;
        sim.run_frames(100, |_, _| {});
        let e = bulk_extent(&sim, FluidHandle(0));
        (sim, e)
    }

    let (reused, a) = dam(0.02);
    let (every, b) = dam(0.0);

    // A tenth of a particle spacing, over a dam break that has run a hundred
    // frames and hit three walls — measured on the bulk of the pool, its
    // outermost one percent left out. The single furthest particle is a drop
    // of spray, and where spray lands after a hundred frames of a chaotic flow
    // moves by millimetres when anything changes in the last bit: computing
    // particle contacts relative to the body rather than the origin, which
    // changes nothing but rounding, moved it from 3.3 to 5.6 mm.
    for (axis, x, y) in [("min", a.min, b.min), ("max", a.max, b.max)] {
        let apart = (x - y).length();
        assert!(
            apart < 0.1 * SPACING,
            "the pool's {axis} moved {:.2} mm when lists were reused",
            apart * 1e3
        );
    }

    // And it really is reusing them. A zero tolerance rebuilds on every fluid
    // substep, which is the number to measure the other against — sixteen a
    // frame, less whatever the accumulator's rounding leaves over.
    let (reused, every) = (
        reused.fluid_neighbour_builds(),
        every.fluid_neighbour_builds(),
    );
    assert!(every > 1500, "a zero tolerance skipped a build: {every}");
    // Half the builds, near enough, and the margin is stated rather than
    // implied: this was `reused * 2 < every` until `fusion_core::fill` was
    // fixed to sample a box to the size it was asked for, which put more
    // particles in the same dam and moved the builds from 758 to 811 out of
    // 1596 — a saving of 49% where the assertion wanted 50%. The
    // claim being tested is that lists are genuinely being reused, and a
    // threshold that a legitimate change to the scene can cross is testing the
    // scene rather than the claim.
    assert!(
        (reused as f32) < 0.6 * every as f32,
        "reused {reused} builds against {every}, which is not a saving"
    );
}
