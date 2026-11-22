//! The unified continuum solver: solids and fluids on one grid.
//!
//! What it has to get right, in the order it is checked: it loses no mass; a
//! column of water collapses into a pool that keeps its volume and stays out
//! of the floor; one core and many give the same bits; a light solid floats
//! on water and a dense one sinks, which only happens when the two are in
//! one solve; fibres stiffen a material the way they run and not across; and
//! the classic path is untouched by any of it.

use fusion_core::{BodyDesc, FluidDesc, FluidParams, Material, SolidDesc, SolidMaterial, World};
use fusion_math::{to_f32, vec3, Vec3};
use fusion_sim::{state_hash, ContinuumSolver, Simulation};

fn floor(world: &mut World) {
    world.add(
        BodyDesc::cuboid(vec3(6.0, 1.0, 6.0))
            .at(vec3(0.0, -0.5, 0.0))
            .with_material(Material::CONCRETE)
            .fixed(),
    );
}

/// A tank: a floor and four walls, `size` inside, open at the top.
fn tank(world: &mut World, size: Vec3) {
    floor(world);
    let t = 0.2;
    for (centre, half) in [
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
            BodyDesc::cuboid(half)
                .at(centre)
                .with_material(Material::CONCRETE)
                .fixed(),
        );
    }
}

fn water(spacing: f32) -> FluidParams {
    FluidParams {
        spacing,
        smoothing_radius: 2.0 * spacing,
        ..FluidParams::WATER
    }
}

fn unified(world: World) -> Simulation {
    let mut sim = Simulation::new(world);
    sim.continuum = ContinuumSolver::Unified;
    sim
}

fn fluid_mass(sim: &Simulation) -> f64 {
    sim.world
        .fluids
        .iter()
        .map(|f| f.len() as f64 * f.params.particle_mass() as f64)
        .sum()
}

/// A dam break: a column of water released in a tank spreads into a pool.
#[test]
fn a_column_of_water_collapses_into_a_pool_that_keeps_its_volume() {
    let mut world = World::new();
    let inside = vec3(1.2, 1.0, 0.4);
    tank(&mut world, inside);
    let spacing = 0.04;
    world.add_fluid(
        &FluidDesc::block(vec3(0.4, 0.6, 0.4))
            .at(vec3(-0.4, 0.3 + 0.01, 0.0))
            .with_params(water(spacing)),
    );
    let mut sim = unified(world);
    let mass = fluid_mass(&sim);
    let volume = 0.4 * 0.6 * 0.4;
    for _ in 0..720 {
        sim.step();
    }
    assert!(
        (fluid_mass(&sim) - mass).abs() < 1e-9,
        "mass is points, and points stay"
    );
    let fluid = &sim.world.fluids[0];
    let mut highest = f32::MIN;
    let mut lowest = f32::MAX;
    let mut leftmost = f32::MAX;
    let mut rightmost = f32::MIN;
    for i in 0..fluid.len() {
        let p = to_f32(fluid.particles.position[i]);
        assert!(p.is_finite());
        highest = highest.max(p.y);
        lowest = lowest.min(p.y);
        leftmost = leftmost.min(p.x);
        rightmost = rightmost.max(p.x);
    }
    // Out of the floor, and spread across the tank.
    assert!(lowest > -0.5 * spacing, "lowest point {lowest}");
    // And out of the walls. The tank's boxes touch: a point resting on the
    // floor that moved into a wall was nearer the wall's bottom face than
    // its side, and was once put back there, on the floor, inside the wall.
    let escaped = (0..fluid.len())
        .map(|i| to_f32(fluid.particles.position[i]))
        .filter(|p| p.x.abs() > inside.x * 0.5 + spacing || p.z.abs() > inside.z * 0.5 + spacing)
        .count();
    assert_eq!(escaped, 0, "points inside the tank's walls");
    assert!(
        rightmost - leftmost > 1.0,
        "it spread: {leftmost}..{rightmost}"
    );
    // A pool: the column was 0.6 m tall, the tank's floor 1.2 x 0.4, so the
    // water stands about a quarter that high, waves aside.
    let depth = volume / (inside.x * inside.z);
    assert!(
        highest < 0.6 * 0.6,
        "it fell: highest {highest}, pool depth {depth}"
    );
    // And it kept its volume: the points' own volume ratios, J, are one to
    // within a couple of percent in the body of the pool. The median rather
    // than the mean, because spray in the air has nothing to push it back
    // together and is meant to be free.
    let mut ratios = fluid.volume_ratio.clone();
    ratios.sort_by(f32::total_cmp);
    let median = ratios[ratios.len() / 2];
    assert!((median - 1.0).abs() < 0.02, "median volume ratio {median}");
}

/// The transfer in is cut into fixed runs merged in a fixed order, so the
/// number of cores does not change the answer.
#[test]
fn one_core_and_many_give_the_same_bits() {
    let build = || {
        let mut world = World::new();
        tank(&mut world, vec3(1.0, 0.8, 0.4));
        world.add_fluid(
            &FluidDesc::block(vec3(0.4, 0.4, 0.3))
                .at(vec3(-0.25, 0.21, 0.0))
                .with_params(water(0.03)),
        );
        world.add_solid(
            &SolidDesc::block(Vec3::splat(0.16))
                .at(vec3(0.25, 0.4, 0.0))
                .of(SolidMaterial::JELLY),
        );
        unified(world)
    };
    let mut one = build();
    one.unified.threads = 1;
    one.unified.serial_below = 0;
    let mut many = build();
    many.unified.threads = 8;
    many.unified.serial_below = 0;
    for _ in 0..120 {
        one.step();
        many.step();
    }
    assert_eq!(state_hash(&one), state_hash(&many));
    assert!(many.unified_report().points > 1500);
}

/// A foam block dropped into water floats, and a dense one sinks. Neither
/// needs any coupling code: the two share grid nodes, and so a velocity.
#[test]
fn a_light_solid_floats_on_water_and_a_heavy_one_sinks() {
    let run = |density: f32| {
        let mut world = World::new();
        tank(&mut world, vec3(0.8, 0.8, 0.8));
        world.add_fluid(
            &FluidDesc::block(vec3(0.8, 0.4, 0.8))
                .at(vec3(0.0, 0.2 + 0.01, 0.0))
                .with_params(water(0.04)),
        );
        let material = SolidMaterial {
            density,
            youngs_modulus: 2e5,
            ..SolidMaterial::JELLY
        };
        world.add_solid(
            &SolidDesc::block(Vec3::splat(0.2))
                .at(vec3(0.0, 0.62, 0.0))
                .of(material),
        );
        let mut sim = unified(world);
        for _ in 0..960 {
            sim.step();
        }
        let solid = &sim.world.solids[0];
        let mean: f32 = (0..solid.len())
            .map(|i| to_f32(solid.particles.position[i]).y)
            .sum::<f32>()
            / solid.len() as f32;
        mean
    };
    let light = run(300.0);
    let heavy = run(3000.0);
    // The pool is 0.4 m deep. Foam at 0.3 g/cc rides with most of itself
    // above the surface; a block three times water's density is on the
    // bottom.
    assert!(light > 0.38, "the light block floats: centre at {light}");
    assert!(heavy < 0.2, "the heavy block sinks: centre at {heavy}");
}

/// Fibres make a material stiffer pulled along them and change nothing
/// pulled across them.
#[test]
fn fibres_stiffen_a_material_along_them_and_not_across() {
    // A soft bar hung from nothing but its own inertia: dropped on its end,
    // it squashes; it is the stretch on rebound that the fibres resist. Read
    // more directly: a bar stretched by a step of velocity on its two ends.
    let stretch = |fibre: Option<Vec3>| {
        let mut world = World::new();
        world.gravity = Vec3::ZERO;
        let mut desc = SolidDesc::block(vec3(0.6, 0.12, 0.12))
            .at(Vec3::ZERO)
            .of(SolidMaterial {
                youngs_modulus: 5e4,
                ..SolidMaterial::JELLY
            });
        desc.params.fibre = fibre.map(|direction| fusion_core::Fibre {
            direction,
            stiffness: 20.0,
        });
        world.add_solid(&desc);
        // Pull: the two halves fly apart at 1 m/s each.
        let solid = &mut world.solids[0];
        for i in 0..solid.len() {
            let x = to_f32(solid.particles.position[i]).x;
            solid.particles.velocity[i] = Vec3::X * x.signum();
        }
        let extent = |sim: &Simulation| {
            let solid = &sim.world.solids[0];
            let (lo, hi) = (0..solid.len()).fold((f32::MAX, f32::MIN), |(lo, hi), i| {
                let x = to_f32(solid.particles.position[i]).x;
                (lo.min(x), hi.max(x))
            });
            hi - lo
        };
        let mut sim = unified(world);
        let rest = extent(&sim);
        let mut longest = 0.0f32;
        for _ in 0..240 {
            sim.step();
            let solid = &sim.world.solids[0];
            let (lo, hi) = (0..solid.len()).fold((f32::MAX, f32::MIN), |(lo, hi), i| {
                let x = to_f32(solid.particles.position[i]).x;
                (lo.min(x), hi.max(x))
            });
            longest = longest.max(hi - lo);
        }
        // How far past its own length it went.
        longest - rest
    };
    let plain = stretch(None);
    let along = stretch(Some(Vec3::X));
    let across = stretch(Some(Vec3::Y));
    assert!(plain > 0.02, "the pull stretches a plain bar: {plain}");
    assert!(
        along < plain * 0.5,
        "fibres along the pull hold it: {along} vs {plain}"
    );
    assert!(
        (across - plain).abs() < plain * 0.1,
        "fibres across it do nothing: {across} vs {plain}"
    );
}

/// The switch: the same scene on either solver, and Classic exactly as it
/// was.
#[test]
fn the_two_solvers_run_the_same_scene_and_classic_is_unchanged() {
    let build = || {
        let mut world = World::new();
        tank(&mut world, vec3(1.0, 0.8, 0.4));
        world.add_fluid(
            &FluidDesc::block(vec3(0.4, 0.4, 0.3))
                .at(vec3(-0.25, 0.21, 0.0))
                .with_params(water(0.05)),
        );
        world.add_solid(
            &SolidDesc::block(Vec3::splat(0.16))
                .at(vec3(0.25, 0.4, 0.0))
                .of(SolidMaterial::JELLY),
        );
        world
    };
    let mut classic = Simulation::new(build());
    let mut again = Simulation::new(build());
    let mut modern = unified(build());
    for _ in 0..240 {
        classic.step();
        again.step();
        modern.step();
    }
    assert_eq!(state_hash(&classic), state_hash(&again));
    assert_eq!(
        classic.unified_report().points,
        0,
        "classic never ran the unified solver"
    );
    assert!(modern.unified_report().points > 0);
    // Both kept everything above the floor.
    for sim in [&classic, &modern] {
        for fluid in &sim.world.fluids {
            for p in &fluid.particles.position {
                assert!(to_f32(*p).y > -0.1);
            }
        }
    }
    // And a run can switch part-way: the fluid picks up the unified state
    // at rest and carries on.
    classic.continuum = ContinuumSolver::Unified;
    for _ in 0..60 {
        classic.step();
    }
    assert!(classic.world.fluids[0].volume_ratio.len() == classic.world.fluids[0].len());
}

/// A unified run saved part-way, as bytes, comes back and carries on to the
/// same bits: the fluids' volume ratios and affine fields, and the solids'
/// fibres, travel in the state.
#[test]
fn a_unified_run_saved_and_restored_carries_on_the_same() {
    let build = || {
        let mut world = World::new();
        tank(&mut world, vec3(1.0, 0.8, 0.4));
        world.add_fluid(
            &FluidDesc::block(vec3(0.4, 0.4, 0.3))
                .at(vec3(-0.25, 0.21, 0.0))
                .with_params(water(0.05)),
        );
        let mut solid = SolidDesc::block(Vec3::splat(0.16))
            .at(vec3(0.25, 0.4, 0.0))
            .of(SolidMaterial::JELLY);
        solid.params.fibre = Some(fusion_core::Fibre {
            direction: Vec3::Y,
            stiffness: 5.0,
        });
        world.add_solid(&solid);
        unified(world)
    };
    let mut first = build();
    for _ in 0..120 {
        first.step();
    }
    let bytes = first.checkpoint().to_bytes();
    let mut second = build();
    let read =
        fusion_sim::Checkpoint::from_bytes(&bytes, &second.checkpoint()).expect("reads back");
    second.restore(&read);
    assert_eq!(
        second.world.solids[0].params.fibre,
        first.world.solids[0].params.fibre
    );
    for _ in 0..60 {
        first.step();
        second.step();
    }
    assert_eq!(state_hash(&first), state_hash(&second));
}

#[test]
fn a_rejected_fusion_step_restores_the_coupled_world_and_can_retry() {
    let mut world = World::new();
    world.add(BodyDesc::cuboid(Vec3::splat(0.2)).at(Vec3::Y));
    world.add_fluid(&FluidDesc::block(Vec3::splat(0.2)).at(Vec3::Y * 2.0));
    let mut sim = unified(world);
    let before = state_hash(&sim);
    let checkpoint = sim.checkpoint();
    sim.unified.max_blocks = 1;
    let report = sim.step();
    assert_eq!(report.failure, Some(fusion_sim::StepFailure::GridBudget));
    assert_eq!(report.dt, 0.0);
    assert_eq!(sim.time(), 0.0);
    assert_eq!(sim.step_index(), 0);
    assert_eq!(state_hash(&sim), before);
    sim.unified.max_blocks = fusion_mpm::unified::MAX_BLOCKS;
    assert!(sim.step().failure.is_none());
    let retried = state_hash(&sim);
    sim.restore(&checkpoint);
    assert!(sim.step().failure.is_none());
    assert_eq!(state_hash(&sim), retried);
}

#[test]
fn advance_retains_rejected_time_for_retry() {
    let mut world = World::new();
    world.add_fluid(&FluidDesc::block(Vec3::splat(0.2)).at(Vec3::Y));
    let mut sim = unified(world);
    sim.unified.max_blocks = 1;
    assert!(sim.advance(sim.solver_dt as f64).unwrap().failure.is_some());
    assert_eq!(sim.step_index(), 0);
    sim.unified.max_blocks = fusion_mpm::unified::MAX_BLOCKS;
    assert!(sim.advance(0.0).unwrap().failure.is_none());
    assert_eq!(sim.step_index(), 1);
}

#[test]
fn seeking_stops_at_a_rejected_interval_instead_of_looping_forever() {
    let mut world = World::new();
    world.add_fluid(&FluidDesc::block(Vec3::splat(0.2)).at(Vec3::Y));
    let mut sim = unified(world);
    sim.unified.max_blocks = 1;
    let timeline = fusion_sim::Timeline::new(0.1);
    let result = timeline.seek(&mut sim, 1.0);
    assert!(!result.exact);
    assert_eq!(result.steps_replayed, 0);
    assert_eq!(result.reached, 0.0);
}

#[test]
fn rigid_only_fusion_scenes_do_not_pay_or_fail_for_unused_continuum_settings() {
    let mut world = World::new();
    world.add(BodyDesc::cuboid(Vec3::splat(0.2)).at(Vec3::Y));
    let mut sim = unified(world);
    sim.unified.max_substeps = 0;
    sim.unified.max_blocks = 0;
    assert!(sim.step().failure.is_none());
    assert_eq!(sim.step_index(), 1);
    assert!(sim.world.bodies.linear_velocity[0].y < 0.0);
    assert_eq!(sim.unified_report().attempted_substeps, 0);
}

/// Two solids that meet keep their own velocities at the nodes they share:
/// blocks sliding past each other keep sliding instead of fusing into one
/// that moves at the average, and the pair's momentum is what it was.
#[test]
fn touching_solids_slide_past_each_other_instead_of_fusing() {
    let run = |separate: bool| {
        let mut world = World::new();
        world.gravity = Vec3::ZERO;
        for (x, v) in [(-0.1f32, 1.0f32), (0.1, -1.0)] {
            world.add_solid(
                &SolidDesc::block(vec3(0.2, 0.2, 0.2))
                    .at(vec3(x, 0.0, 0.0))
                    .of(SolidMaterial::JELLY)
                    .with_velocity(vec3(0.0, 0.0, v)),
            );
        }
        let mut sim = unified(world);
        sim.unified.separate_fields = separate;
        sim.unified.contact_friction = 0.0;
        let momentum = |sim: &Simulation| {
            sim.world.solids.iter().fold(Vec3::ZERO, |m, s| {
                m + s.particles.velocity.iter().fold(Vec3::ZERO, |a, v| a + *v)
                    * s.params.point_mass()
            })
        };
        let before = momentum(&sim);
        for _ in 0..60 {
            sim.step();
        }
        let mean_z = |s: &fusion_core::SolidSet| {
            s.particles.velocity.iter().map(|v| v.z).sum::<f32>() / s.len() as f32
        };
        let relative = mean_z(&sim.world.solids[0]) - mean_z(&sim.world.solids[1]);
        (relative, (momentum(&sim) - before).length())
    };
    let (sliding, drift) = run(true);
    let (fused, _) = run(false);
    assert!(
        sliding > 1.8,
        "they keep sliding at nearly 2 m/s relative: {sliding}"
    );
    assert!(
        fused < sliding * 0.9,
        "one shared field drags them together: {fused}"
    );
    assert!(drift < 1e-3, "contact conserves momentum: drift {drift}");
}

/// A block lifted off another comes away clean: the one underneath is not
/// pulled up with it, which a single shared field does.
#[test]
fn a_solid_lifted_off_another_comes_away_cleanly() {
    let run = |separate: bool| {
        let mut world = World::new();
        world.gravity = Vec3::ZERO;
        world.add_solid(
            &SolidDesc::block(Vec3::splat(0.2))
                .at(vec3(0.0, 0.0, 0.0))
                .of(SolidMaterial::JELLY),
        );
        world.add_solid(
            &SolidDesc::block(Vec3::splat(0.2))
                .at(vec3(0.0, 0.2, 0.0))
                .of(SolidMaterial::JELLY)
                .with_velocity(vec3(0.0, 1.0, 0.0)),
        );
        let mut sim = unified(world);
        sim.unified.separate_fields = separate;
        for _ in 0..60 {
            sim.step();
        }
        let bottom = &sim.world.solids[0];
        bottom.particles.velocity.iter().map(|v| v.y).sum::<f32>() / bottom.len() as f32
    };
    let clean = run(true);
    let dragged = run(false);
    assert!(clean.abs() < 0.05, "the lower block stays put: {clean}");
    assert!(
        dragged > 0.1,
        "with one shared field it is dragged up: {dragged}"
    );
}

/// A real viscosity: from the same column, honey at 10 Pa·s spreads less in
/// a second than water at a millipascal-second, and a paste at a thousand
/// barely slumps.
#[test]
fn a_viscous_fluid_spreads_slower_than_water() {
    let spread = |pascal_seconds: f32| {
        let mut world = World::new();
        tank(&mut world, vec3(1.2, 0.8, 0.4));
        world.add_fluid(
            &FluidDesc::block(vec3(0.3, 0.4, 0.4))
                .at(vec3(-0.45, 0.21, 0.0))
                .with_params(water(0.04).with_dynamic_viscosity(pascal_seconds)),
        );
        let mut sim = unified(world);
        for _ in 0..240 {
            sim.step();
        }
        let fluid = &sim.world.fluids[0];
        (0..fluid.len())
            .map(|i| to_f32(fluid.particles.position[i]).x)
            .fold(f32::MIN, f32::max)
    };
    let water_front = spread(1.0e-3);
    let honey_front = spread(10.0);
    let paste_front = spread(1000.0);
    assert!(
        honey_front < water_front - 0.05,
        "honey's front {honey_front:.2} m against water's {water_front:.2} m"
    );
    // A thousand pascal-seconds — a paste — barely slumps.
    assert!(
        paste_front < honey_front - 0.2,
        "paste's front {paste_front:.2} m against honey's {honey_front:.2} m"
    );
}
