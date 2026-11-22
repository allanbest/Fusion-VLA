//! Reproductions from the collision/material audit.
use fusion_core::{BodyDesc, Material, SolidDesc, SolidMaterial, World};
use fusion_math::{vec3, Vec3};
use fusion_sim::Simulation;

#[test]
fn approaching_fast_bodies_do_not_cross_each_other_between_detections() {
    let mut world = World::new().with_gravity(Vec3::ZERO);
    world.linear_damping = 0.0;
    let surface = Material::new(1000.0, 0.0, 0.0);
    let a = world.add(
        BodyDesc::sphere(0.05)
            .at(vec3(-0.2, 0.0, 0.0))
            .with_velocity(Vec3::X * 60.0)
            .with_material(surface),
    );
    let b = world.add(
        BodyDesc::sphere(0.05)
            .at(vec3(0.2, 0.0, 0.0))
            .with_velocity(-Vec3::X * 60.0)
            .with_material(surface),
    );
    let mut sim = Simulation::new(world);
    sim.step();
    assert!(
        sim.world.bodies.position[a.index()].x < sim.world.bodies.position[b.index()].x,
        "the two spheres crossed without a contact"
    );
}

#[test]
fn stability_diagnostics_include_continuum_material_history() {
    let mut world = World::new();
    world.add_solid(&SolidDesc::block(Vec3::splat(0.1)).of(SolidMaterial::JELLY));
    let mut sim = Simulation::new(world);
    assert!(sim.is_stable());
    sim.world.solids[0].deformation[0].x_axis.x = f32::NAN;
    assert!(
        !sim.is_stable(),
        "invalid material deformation was reported stable"
    );
}

/// Every corner of a shape, in world space; the lowest point of a sphere.
fn corners(b: &fusion_core::BodySet, i: usize) -> Vec<Vec3> {
    use fusion_core::Shape;
    let pose = b.pose(i);
    match &b.shape[i] {
        Shape::Hull(h) => h
            .vertices()
            .iter()
            .map(|p| pose.transform_point(*p))
            .collect(),
        Shape::Cuboid { half_extents: h } => (0..8)
            .map(|k| {
                pose.transform_point(Vec3::new(
                    if k & 1 == 0 { -h.x } else { h.x },
                    if k & 2 == 0 { -h.y } else { h.y },
                    if k & 4 == 0 { -h.z } else { h.z },
                ))
            })
            .collect(),
        Shape::Sphere { radius } => vec![pose.translation - Vec3::Y * *radius],
    }
}

/// A body that lands fast is caught before it is in the floor, not found in
/// it and pushed out. Contacts used to be made only within 5 mm, so anything
/// faster than about 1.2 m/s was a centimetre or more into the floor before
/// it had one: the clip, and then the pop. Now each body's contacts reach as
/// far as it can travel in the step. A tumbling hull — the shape broken
/// pieces are — once swung a corner 11 cm in, because a contact's four kept
/// points were chosen by present depth; they are chosen by depth at the end
/// of the step.
///
/// Up to 25 m/s, a fall of about thirty metres. Past that a box landing on an
/// edge comes off spinning near 100 rad/s — a quarter turn a step, which no
/// linear prediction over the step follows — and part of that spin is a twist
/// about the vertical the bounce pass puts in by resolving the edge's two
/// corners unequally. That is a known limit, not a guarantee this pins.
#[test]
fn a_body_landing_fast_does_not_clip_into_the_floor() {
    use fusion_core::Shape;
    let half = Vec3::splat(0.15);
    let hull = Shape::hull(
        &(0..8)
            .map(|q| {
                Vec3::new(
                    if q & 1 == 0 { -half.x } else { half.x },
                    if q & 2 == 0 { -half.y } else { half.y },
                    if q & 4 == 0 { -half.z } else { half.z },
                )
            })
            .collect::<Vec<_>>(),
    )
    .unwrap();
    for shape in [Shape::cuboid(half * 2.0), Shape::sphere(0.15), hull] {
        for speed in [10.0f32, 25.0] {
            let mut world = World::new();
            world.add(
                BodyDesc::cuboid(vec3(40.0, 1.0, 40.0))
                    .at(vec3(0.0, -0.5, 0.0))
                    .with_material(Material::CONCRETE)
                    .fixed(),
            );
            let k = world
                .add(
                    BodyDesc::new(shape.clone())
                        .at(vec3(0.0, 0.6, 0.0))
                        .rotated(fusion_math::Quat::from_rotation_z(0.3))
                        .with_velocity(vec3(0.0, -speed, 0.0))
                        .with_material(Material::STEEL),
                )
                .index();
            let mut sim = Simulation::new(world);
            let mut deepest = 0.0f32;
            for _ in 0..240 {
                sim.step();
                for c in corners(&sim.world.bodies, k) {
                    deepest = deepest.max(-c.y);
                }
            }
            assert!(
                deepest < 0.002,
                "{shape:?} at {speed} m/s went {:.1} mm into the floor",
                deepest * 1e3
            );
        }
    }
}

/// A bullet does not pass through a thin pane, at any speed a scene throws.
#[test]
fn a_bullet_does_not_pass_through_a_thin_pane() {
    for (thickness, speed) in [(0.01f32, 100.0f32), (0.02, 50.0), (0.05, 100.0)] {
        let mut world = World::new().with_gravity(Vec3::ZERO);
        world.add(
            BodyDesc::cuboid(vec3(thickness, 1.0, 1.0))
                .with_material(Material::GLASS)
                .fixed(),
        );
        let k = world
            .add(
                BodyDesc::sphere(0.01)
                    .at(vec3(-0.5, 0.0, 0.0))
                    .with_velocity(vec3(speed, 0.0, 0.0))
                    .with_material(Material::STEEL),
            )
            .index();
        let mut sim = Simulation::new(world);
        for _ in 0..120 {
            sim.step();
        }
        let x = sim.world.bodies.translation(k).x;
        assert!(
            x < 0.0,
            "{speed} m/s through a {thickness} m pane: ended at x = {x}"
        );
    }
}

#[test]
fn invalid_step_intervals_do_not_corrupt_either_solver() {
    use fusion_sim::{state_hash, ContinuumSolver};
    for mode in [ContinuumSolver::Classic, ContinuumSolver::Unified] {
        for dt in [0.0, -0.1, f32::NAN, f32::INFINITY] {
            let mut world = World::new();
            world.add(BodyDesc::sphere(0.1).with_velocity(Vec3::X));
            let mut sim = Simulation::new(world);
            sim.continuum = mode;
            sim.solver_dt = dt;
            let before = state_hash(&sim);
            let report = sim.step();
            assert!(report.failure.is_some());
            assert_eq!(report.dt, 0.0);
            assert_eq!(sim.time(), 0.0);
            assert_eq!(state_hash(&sim), before);
        }
    }
}

/// Deepest overlap between any two bodies, metres.
fn deepest_overlap(b: &fusion_core::BodySet) -> f32 {
    use fusion_math::Isometry;
    let mut deepest = 0.0f32;
    let mut contacts = Vec::new();
    for i in 0..b.len() {
        for j in i + 1..b.len() {
            if !b.is_dynamic(i) && !b.is_dynamic(j) {
                continue;
            }
            let pa = Isometry::new(Vec3::ZERO, b.pose(i).rotation);
            let pb = Isometry::new(b.relative_translation(i, j), b.pose(j).rotation);
            contacts.clear();
            fusion_collide::narrow::primitives::collide(
                &b.shape[i],
                &pa,
                &b.shape[j],
                &pb,
                0.0,
                &mut contacts,
            );
            for c in &contacts {
                deepest = deepest.max(-c.separation);
            }
        }
    }
    deepest
}

fn concrete_floor(world: &mut World) {
    world.add(
        BodyDesc::cuboid(vec3(40.0, 1.0, 40.0))
            .at(vec3(0.0, -0.5, 0.0))
            .with_material(Material::CONCRETE)
            .fixed(),
    );
}

/// A light board under a heavy block stays out of the floor.
///
/// The iterations trade a contact's error to its neighbours at a rate set by
/// the mass ratio: the floor pushed a 1 kg board up and a 1000 kg block
/// pushed it straight back down, and the board went its whole 5 cm into the
/// floor. A 10 kg block landing at 5 m/s sank one 13 mm. Shock propagation
/// finishes each substep from the ground up, the body underneath held.
#[test]
fn a_light_board_under_a_heavy_block_stays_out_of_the_floor() {
    for (block, speed) in [(1000.0f32, 0.0f32), (1000.0, 5.0), (10.0, 5.0)] {
        let mut world = World::new();
        concrete_floor(&mut world);
        let board = world
            .add(
                BodyDesc::cuboid(vec3(0.5, 0.05, 0.5))
                    .at(vec3(0.0, 0.025, 0.0))
                    .with_mass(1.0),
            )
            .index();
        let height = if speed == 0.0 { 0.25 } else { 1.0 };
        world.add(
            BodyDesc::cuboid(Vec3::splat(0.4))
                .at(vec3(0.05, height, 0.0))
                .with_velocity(vec3(0.0, -speed, 0.0))
                .with_mass(block),
        );
        let mut sim = Simulation::new(world);
        let mut lowest = f32::INFINITY;
        for _ in 0..480 {
            sim.step();
            lowest = lowest.min(sim.world.bodies.translation(board).y);
        }
        assert!(
            lowest > 0.025 - 0.002,
            "a {block} kg block at {speed} m/s put the board's middle at {lowest} m"
        );
    }
}

/// Twenty boxes stand. They used to sink into each other and come down.
#[test]
fn a_twenty_box_tower_stands() {
    let mut world = World::new();
    concrete_floor(&mut world);
    for k in 0..20 {
        world.add(
            BodyDesc::cuboid(Vec3::ONE)
                .at(vec3(0.0, 0.5 + k as f32 * 1.02, 0.0))
                .with_material(Material::WOOD),
        );
    }
    let mut sim = Simulation::new(world);
    sim.sleep.enabled = false;
    for _ in 0..1200 {
        sim.step();
    }
    for k in 0..20 {
        let p = sim.world.bodies.translation(k + 1);
        assert!(
            (p.y - (0.5 + k as f32)).abs() < 0.015 && p.x.abs().max(p.z.abs()) < 0.01,
            "box {k} stands at {p:?}"
        );
    }
    assert!(sim.world.bodies.kinetic_energy() < 0.1);
}

/// Boxes meeting corner first at a combined 60 m/s do not go into each
/// other. They come apart spinning, as an off-centre hit makes them, and
/// with the contacts found at the top of the step they went 75–89 mm in:
/// the corners a spin turns in had no contact, and a face swinging in had
/// a different normal from the corner that hit. Pairs that turn are found
/// again within the step; corners on their way in come with their own.
#[test]
fn boxes_meeting_corner_first_do_not_pass_into_each_other() {
    use fusion_math::Quat;
    for (ax, ay, bx, by) in [
        (0.6f32, 0.4f32, -0.5f32, 0.7f32),
        (0.785, 0.615, 0.3, -0.2),
        (0.2, 0.9, 1.1, 0.4),
    ] {
        let mut world = World::new().with_gravity(Vec3::ZERO);
        world.add(
            BodyDesc::cuboid(Vec3::splat(0.3))
                .at(vec3(-1.0, 0.0, 0.0))
                .rotated(Quat::from_rotation_x(ax) * Quat::from_rotation_y(ay))
                .with_velocity(Vec3::X * 30.0),
        );
        world.add(
            BodyDesc::cuboid(Vec3::splat(0.3))
                .at(vec3(1.0, 0.05, 0.0))
                .rotated(Quat::from_rotation_x(bx) * Quat::from_rotation_y(by))
                .with_velocity(-Vec3::X * 30.0),
        );
        let mut sim = Simulation::new(world);
        let mut deepest = 0.0f32;
        for _ in 0..60 {
            sim.step();
            deepest = deepest.max(deepest_overlap(&sim.world.bodies));
        }
        assert!(
            deepest < 0.015,
            "corner hit {ax},{ay} / {bx},{by} went {:.1} mm in",
            deepest * 1e3
        );
    }
}

/// A rolling ball stays on the floor. Its contact anchor is a point of the
/// ball, carried from step to step as it turns and rising off the floor as
/// it goes round, and the ball sank three millimetres and popped back, over
/// and over.
#[test]
fn a_rolling_ball_stays_on_the_floor() {
    let mut world = World::new();
    concrete_floor(&mut world);
    let r = 0.15;
    let ball = world
        .add(
            BodyDesc::sphere(r)
                .at(vec3(0.0, r, 0.0))
                .with_velocity(vec3(3.0, 0.0, 1.0))
                .with_angular_velocity(vec3(1.0, 0.0, -3.0) / r)
                .with_material(Material::STEEL),
        )
        .index();
    let mut sim = Simulation::new(world);
    let mut deepest = 0.0f32;
    for _ in 0..480 {
        sim.step();
        deepest = deepest.max(r - sim.world.bodies.translation(ball).y);
    }
    assert!(
        deepest < 1e-3,
        "the ball sank {:.2} mm into the floor",
        deepest * 1e3
    );
}

/// Water and cloth thrown at a 1 cm pane stay on their side of it. A
/// particle that crossed it in a substep was found clear on the far side,
/// and one more than halfway in was pushed out through the far face; both
/// now go back out the way they came.
#[test]
fn water_and_cloth_do_not_pass_through_a_thin_pane() {
    use fusion_core::{ClothDesc, FluidDesc, FluidParams};
    let pane = |world: &mut World| {
        world.add(BodyDesc::cuboid(vec3(0.01, 3.0, 3.0)).fixed());
    };
    let mut world = World::new().with_gravity(Vec3::ZERO);
    pane(&mut world);
    world.add_fluid(
        &FluidDesc::block(vec3(0.3, 0.3, 0.3))
            .at(vec3(-0.5, 0.0, 0.0))
            .with_params(FluidParams::water_at(0.05)),
    );
    for v in &mut world.fluids[0].particles.velocity {
        *v = Vec3::X * 30.0;
    }
    let mut sim = Simulation::new(world);
    for _ in 0..120 {
        sim.step();
    }
    let through = sim.world.fluids[0]
        .particles
        .position
        .iter()
        .filter(|p| p.x > 0.0)
        .count();
    assert_eq!(through, 0, "{through} water particles went through");

    let mut world = World::new().with_gravity(Vec3::ZERO);
    pane(&mut world);
    world.add_cloth(
        &ClothDesc::new(10, 10, 0.05)
            .at(vec3(-0.5, 0.0, 0.0))
            .with_mass(0.2),
    );
    for v in &mut world.particles.velocity {
        *v = Vec3::X * 60.0;
    }
    let mut sim = Simulation::new(world);
    for _ in 0..120 {
        sim.step();
    }
    let through = sim
        .world
        .particles
        .position
        .iter()
        .filter(|p| p.x > 0.0)
        .count();
    assert_eq!(through, 0, "{through} cloth particles went through");
}

/// A pile of mixed pieces dropped from up to thirty metres stays out of
/// itself and gains nothing. A pile like this is chaotic — which piece lands
/// on which edge changes with the last bit of anything — so the bound is on
/// the average overlap over the run, which is not: 7.7 mm before shock
/// propagation and mid-step re-detection, about 1 mm after. The worst single
/// moment is bounded loosely; it is a 25 m/s impact between thin pieces.
#[test]
fn a_dropped_pile_stays_out_of_itself_and_gains_nothing() {
    use fusion_core::{BodySet, Shape};
    let shard = Shape::hull(&[
        vec3(0.1, -0.1, 0.05),
        vec3(-0.12, 0.08, -0.1),
        vec3(0.05, 0.13, 0.11),
        vec3(-0.07, -0.11, 0.12),
        vec3(0.14, 0.04, -0.09),
        vec3(-0.02, -0.06, -0.14),
    ])
    .unwrap();
    let sliver = Shape::hull(&[
        vec3(0.0, 0.0, 0.0),
        vec3(0.3, 0.0, 0.0),
        vec3(0.0, 0.02, 0.0),
        vec3(0.0, 0.0, 0.2),
        vec3(0.3, 0.02, 0.2),
    ])
    .unwrap();
    let shapes = [
        Shape::cuboid(Vec3::splat(0.3)),
        Shape::cuboid(vec3(0.6, 0.06, 0.4)),
        Shape::cuboid(vec3(0.04, 0.04, 1.0)),
        Shape::sphere(0.15),
        shard,
        sliver,
    ];
    let energy = |b: &BodySet| -> f32 {
        (0..b.len())
            .filter(|&i| b.is_dynamic(i))
            .map(|i| {
                let m = 1.0 / b.inv_mass[i];
                let w = b.angular_velocity[i];
                0.5 * m * b.linear_velocity[i].length_squared()
                    + 0.5 * w.dot(b.inv_inertia_world(i).inverse() * w)
                    + m * 9.81 * b.translation(i).y
            })
            .sum()
    };
    let mut world = World::new();
    concrete_floor(&mut world);
    for k in 0..24 {
        let a = k as f32;
        world.add(
            BodyDesc::new(shapes[k % shapes.len()].clone())
                .at(vec3(
                    (a * 1.3).sin() * 0.5,
                    0.6 + a * 1.05,
                    (a * 0.7).cos() * 0.5,
                ))
                .rotated(
                    fusion_math::Quat::from_rotation_x(a)
                        * fusion_math::Quat::from_rotation_z(a * 0.5),
                )
                .with_material(if k % 3 == 0 {
                    Material::STEEL
                } else {
                    Material::WOOD
                }),
        );
    }
    let mut sim = Simulation::new(world);
    let (mut deepest, mut total) = (0.0f32, 0.0f32);
    let mut before = energy(&sim.world.bodies);
    let steps = 960;
    for step in 0..steps {
        sim.step();
        let d = deepest_overlap(&sim.world.bodies);
        deepest = deepest.max(d);
        total += d;
        let now = energy(&sim.world.bodies);
        assert!(now - before < 1.0, "step {step} gained {} J", now - before);
        before = now;
    }
    let mean = total / steps as f32;
    assert!(
        mean < 3e-3,
        "the pile sat {:.1} mm into itself on average",
        mean * 1e3
    );
    assert!(
        deepest < 0.15,
        "the pile went {:.1} mm into itself",
        deepest * 1e3
    );
}

/// A soft solid landing on a box stays out of it, on either continuum
/// solver. The grid kept material out only at nodes within half a cell of a
/// surface, and at the editor's default sampling a cell is 17 cm: a jelly
/// ball landing at 12 m/s put a sixth of its points inside the box.
#[test]
fn a_deformable_landing_on_a_box_stays_out_of_it() {
    use fusion_core::Shape;
    use fusion_sim::ContinuumSolver;
    for solver in [ContinuumSolver::Classic, ContinuumSolver::Unified] {
        let mut world = World::new();
        concrete_floor(&mut world);
        world.add(
            BodyDesc::cuboid(Vec3::ONE)
                .at(vec3(0.0, 0.5, 0.0))
                .with_material(Material::WOOD)
                .fixed(),
        );
        let r = 0.5f32;
        let spacing = r / 6.0;
        world.add_solid(
            &SolidDesc::new(Shape::sphere(r))
                .at(vec3(0.0, 9.5, 0.0))
                .with_params(fusion_core::SolidParams {
                    spacing,
                    cell_size: 2.0 * spacing,
                    material: SolidMaterial::JELLY,
                    fibre: None,
                }),
        );
        let mut sim = Simulation::new(world);
        sim.continuum = solver;
        let mut most = 0;
        for _ in 0..480 {
            sim.step();
            let inside = sim.world.solids[0]
                .particles
                .position
                .iter()
                .filter(|p| p.x.abs() < 0.49 && p.z.abs() < 0.49 && p.y > 0.01 && p.y < 0.99)
                .count();
            most = most.max(inside);
        }
        assert_eq!(most, 0, "{solver:?}: {most} material points inside the box");
    }
}

/// What a heavy landing takes out of the motion is what the contacts hand
/// the heat solve. A resting contact counts nothing it removes, by design,
/// and that rule counted nothing at the floor under a cube a 25 t ball
/// landed on: 40–60% of the energy went nowhere.
#[test]
fn a_heavy_landing_heats_by_everything_it_lost() {
    let mut world = World::new();
    concrete_floor(&mut world);
    world.add(
        BodyDesc::cuboid(Vec3::ONE)
            .at(vec3(0.0, 0.5, 0.0))
            .with_material(Material::WOOD),
    );
    world.add(
        BodyDesc::sphere(0.5)
            .at(vec3(0.0, 6.0, 0.0))
            .with_material(Material::STEEL)
            .with_mass(25_000.0),
    );
    let mut sim = Simulation::new(world);
    sim.heat.enabled = true;
    let mechanical = |s: &Simulation| -> f64 {
        let b = &s.world.bodies;
        (0..b.len())
            .filter(|&i| b.is_dynamic(i))
            .map(|i| {
                let m = b.mass(i) as f64;
                0.5 * m * b.linear_velocity[i].length_squared() as f64
                    + m * 9.81 * b.translation(i).y as f64
            })
            .sum()
    };
    let before = mechanical(&sim);
    let mut dissipated = 0.0f64;
    for _ in 0..720 {
        sim.step();
        dissipated += sim
            .contact_reports()
            .iter()
            .map(|r| r.dissipated as f64)
            .sum::<f64>();
    }
    let lost = before - mechanical(&sim);
    assert!(lost > 1e6, "the ball lost only {lost:.0} J");
    assert!(
        (dissipated - lost).abs() < 0.05 * lost,
        "{lost:.0} J was lost and {dissipated:.0} J of it reached the heat solve"
    );
}

/// A steel ball that yields on landing is dented where it landed. It was
/// made a continuum after the rigid solve had already bounced it, so it
/// came off whole and went back up; and the law it yielded under compacted,
/// so a harder landing crushed it to a sheet no wider than itself.
#[test]
fn a_ball_that_yields_is_dented_where_it_landed() {
    let mut world = World::new();
    concrete_floor(&mut world);
    world.add(
        BodyDesc::cuboid(Vec3::ONE)
            .at(vec3(0.0, 0.5, 0.0))
            .with_material(Material::CONCRETE)
            .fixed(),
    );
    world.add(
        BodyDesc::sphere(0.5)
            .at(vec3(0.0, 2.2, 0.0))
            .with_material(Material::STEEL)
            .with_velocity(vec3(0.0, -60.0, 0.0))
            .with_failure(fusion_core::Failure::Deforms),
    );
    let mut sim = Simulation::new(world);
    for _ in 0..240 {
        sim.step();
    }
    let solid = sim.world.solids.first().expect("the ball yielded");
    let points: Vec<Vec3> = solid
        .particles
        .position
        .iter()
        .map(|p| fusion_math::to_f32(*p))
        .collect();
    let (low, high) = points.iter().fold(
        (Vec3::splat(f32::INFINITY), Vec3::splat(f32::NEG_INFINITY)),
        |(l, h), p| (l.min(*p), h.max(*p)),
    );
    let centre = points.iter().fold(Vec3::ZERO, |a, p| a + *p) / points.len() as f32;
    let (top, bottom) = (high.y - centre.y, centre.y - low.y);
    let size = high - low;
    assert!(
        bottom < top - 0.03,
        "flattened at the bottom {bottom:.3} vs top {top:.3}"
    );
    assert!(size.x > size.y, "squashed and spread: {size:?}");
    assert!(size.x < 1.6, "spread like a liquid: {size:?}");
}

/// A blow far past a body's strength breaks it into more than a wedge would.
#[test]
fn a_crushing_blow_breaks_wood_into_more_than_two() {
    let mut world = World::new();
    concrete_floor(&mut world);
    world.add(
        BodyDesc::cuboid(Vec3::ONE)
            .at(vec3(0.0, 0.5, 0.0))
            .with_material(Material::WOOD)
            .with_failure(fusion_core::Failure::Breaks),
    );
    world.add(
        BodyDesc::sphere(0.5)
            .at(vec3(0.0, 7.5, 0.0))
            .with_material(Material::STEEL)
            .with_mass(25_000.0),
    );
    let mut sim = Simulation::new(world);
    for _ in 0..480 {
        sim.step();
    }
    let b = &sim.world.bodies;
    let pieces = (0..b.len())
        .filter(|&k| b.reshaped[k] && !b.is_removed(k))
        .count();
    assert!(pieces >= 5, "a 25 t ball broke the cube into {pieces}");
}
