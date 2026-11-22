//! Rigid bodies that break, deform and melt.
//!
//! What a break has to keep is the bookkeeping: the pieces weigh what the body
//! did and carry its momentum, the run stays deterministic, and a checkpoint or
//! a saved state taken after the break puts the pieces back. What it has to
//! decide is when: a beam set down on its supports stays whole, and the same
//! beam with a tonne of steel dropped on it does not.

use fusion_core::{BodyDesc, Failure, Material, Parting, Pattern, World};
use fusion_math::{vec3, Vec3};
use fusion_sim::{state_hash, Checkpoint, HeatSettings, Outcome, Simulation};

fn floor(world: &mut World) {
    world.add(
        BodyDesc::cuboid(vec3(40.0, 1.0, 40.0))
            .at(vec3(0.0, -0.5, 0.0))
            .with_material(Material::CONCRETE)
            .fixed(),
    );
}

/// A concrete beam on two supports, that breaks. With a tonne of steel above
/// it, or not.
fn beam(weight: bool) -> Simulation {
    let mut world = World::new();
    floor(&mut world);
    for x in [-1.8f32, 1.8] {
        world.add(
            BodyDesc::cuboid(vec3(0.4, 1.0, 0.6))
                .at(vec3(x, 0.5, 0.0))
                .with_material(Material::CONCRETE)
                .fixed(),
        );
    }
    world.add(
        BodyDesc::cuboid(vec3(4.0, 0.4, 0.4))
            .at(vec3(0.0, 1.2, 0.0))
            .with_material(Material::CONCRETE)
            .with_failure(Failure::Breaks),
    );
    if weight {
        world.add(
            BodyDesc::cuboid(Vec3::splat(0.5))
                .at(vec3(0.0, 3.0, 0.0))
                .with_material(Material::STEEL),
        );
    }
    Simulation::new(world)
}

const BEAM: usize = 3;

fn dynamic_mass(sim: &Simulation) -> f64 {
    let b = &sim.world.bodies;
    (0..b.len())
        .filter(|&i| b.inv_mass[i] > 0.0 && !b.is_removed(i))
        .map(|i| 1.0 / b.inv_mass[i] as f64)
        .sum()
}

#[test]
fn a_beam_set_down_on_its_supports_stays_whole() {
    let mut sim = beam(false);
    for _ in 0..480 {
        sim.step();
    }
    assert!(sim.take_failures().is_empty());
    assert_eq!(sim.world.bodies.len(), 4);
    // And it is loaded, just not past its strength.
    let u = sim.body_utilisation(fusion_core::BodyHandle(BEAM as u32));
    assert!(u > 0.05 && u < 0.5, "{u}");
}

#[test]
fn a_tonne_of_steel_snaps_it_and_the_pieces_weigh_what_it_did() {
    let mut sim = beam(true);
    let mass = dynamic_mass(&sim);
    let mut broke = None;
    for step in 0..480 {
        sim.step();
        for event in sim.take_failures() {
            if event.body == BEAM && broke.is_none() {
                broke = Some((step, event));
            }
        }
        assert!(
            (dynamic_mass(&sim) - mass).abs() < 1e-3 * mass,
            "step {step}: {} against {mass}",
            dynamic_mass(&sim)
        );
    }
    let (step, event) = broke.expect("the beam broke");
    assert!(event.utilisation >= 1.0);
    let Outcome::Broke { pieces } = &event.outcome else {
        panic!("{event:?}")
    };
    assert!(!pieces.is_empty());
    // When the weight landed, not before.
    assert!(step > 100, "{step}");
    // Every piece remembers the beam it came from.
    for &p in pieces {
        assert_eq!(sim.world.bodies.source[p] as usize, BEAM);
    }
    assert!(sim.is_stable());
}

/// The break itself, in isolation: momentum, angular momentum and kinetic
/// energy are what they were.
#[test]
fn breaking_keeps_the_motion() {
    let mut world = World::new().with_gravity(Vec3::ZERO);
    // A glass bar, spinning and moving, and spun fast enough that its own
    // centripetal load cracks it: ρ ω² L² / 8 is 63 MPa at its middle.
    let b = world.add(
        BodyDesc::cuboid(vec3(1.0, 0.1, 0.1))
            .at(vec3(0.0, 2.0, 0.0))
            .with_velocity(vec3(1.0, 0.5, -0.25))
            .with_angular_velocity(vec3(0.0, 450.0, 0.0))
            .with_material(Material::GLASS)
            .with_failure(Failure::Breaks),
    );
    let mut sim = Simulation::new(world);
    let measure = |sim: &Simulation| {
        let bodies = &sim.world.bodies;
        let mut m = 0.0f64;
        let mut p = fusion_math::DVec3::ZERO;
        let mut moment = fusion_math::DVec3::ZERO;
        for i in 0..bodies.len() {
            m += bodies.mass(i) as f64;
        }
        let mut centre = fusion_math::DVec3::ZERO;
        for i in 0..bodies.len() {
            centre += bodies.position[i] * bodies.mass(i) as f64;
        }
        centre *= 1.0 / m;
        let mut energy = 0.0f64;
        for i in 0..bodies.len() {
            let mi = bodies.mass(i) as f64;
            let v = fusion_math::to_f64(bodies.linear_velocity[i]);
            p += v * mi;
            let w = bodies.angular_velocity[i];
            let local = bodies.orientation[i].rotate_inverse(w);
            let spin = bodies.inertia_local[i].mul_element_wise(local);
            let spin_world = fusion_math::to_f64(bodies.orientation[i].rotate(spin));
            let r = bodies.position[i] - centre;
            moment += r.cross(v * mi) + spin_world;
            energy += 0.5 * mi * v.length_squared() + 0.5 * spin.dot(local) as f64;
        }
        (m, p, moment, energy)
    };
    let before = measure(&sim);
    sim.step();
    let events = sim.take_failures();
    assert_eq!(events.len(), 1, "{events:?}");
    assert_eq!(events[0].body, b.index());
    let after = measure(&sim);
    assert!(
        (after.0 - before.0).abs() < 1e-4 * before.0,
        "{before:?} {after:?}"
    );
    assert!(
        (after.1 - before.1).length() < 1e-4 * before.1.length(),
        "{before:?} {after:?}"
    );
    assert!(
        (after.2 - before.2).length() < 1e-3 * before.2.length(),
        "{before:?} {after:?}"
    );
    assert!(
        (after.3 / before.3 - 1.0).abs() < 1e-3,
        "{before:?} {after:?}"
    );
}

#[test]
fn a_break_is_deterministic_and_comes_back_from_a_checkpoint() {
    let run = || {
        let mut sim = beam(true);
        let mut checkpoint: Option<Checkpoint> = None;
        for step in 0..360 {
            if step == 100 {
                checkpoint = Some(sim.checkpoint());
            }
            sim.step();
        }
        (sim, checkpoint.unwrap())
    };
    let (mut first, before_the_break) = run();
    let (second, _) = run();
    assert!(first.world.bodies.len() > 5, "it broke");
    assert_eq!(state_hash(&first), state_hash(&second));

    // Back before the break, with fewer bodies than there are now, and
    // forward again through it.
    let after = state_hash(&first);
    let pieces = first.world.bodies.len();
    first.restore(&before_the_break);
    assert_eq!(first.world.bodies.len(), 5);
    for _ in 100..360 {
        first.step();
    }
    assert_eq!(first.world.bodies.len(), pieces);
    assert_eq!(state_hash(&first), after);
}

#[test]
fn a_saved_state_carries_the_pieces() {
    let mut sim = beam(true);
    for _ in 0..360 {
        sim.step();
    }
    assert!(sim.world.bodies.len() > 5, "it broke");
    let bytes = sim.checkpoint().to_bytes();

    // Into a fresh copy of the scene, which has never broken.
    let mut fresh = beam(true);
    let restored = Checkpoint::from_bytes(&bytes, &fresh.checkpoint()).unwrap();
    fresh.restore(&restored);
    assert_eq!(fresh.world.bodies.len(), sim.world.bodies.len());
    for i in 0..sim.world.bodies.len() {
        let (a, b) = (&sim.world.bodies, &fresh.world.bodies);
        assert_eq!(a.position[i], b.position[i]);
        assert_eq!(a.inv_mass[i], b.inv_mass[i]);
        assert_eq!(a.shape[i].volume(), b.shape[i].volume());
        assert_eq!(a.source[i], b.source[i]);
        assert_eq!(a.grain[i], b.grain[i]);
    }
    assert!(fresh.is_stable());
    for _ in 0..60 {
        fresh.step();
    }
    assert!(fresh.is_stable());
}

/// An aluminium block hotter than its melting point stops being rigid, and
/// the continuum that replaces it weighs what it did and is exactly as far
/// through melting as its heat pays for.
#[test]
fn a_block_past_its_melting_point_becomes_a_continuum() {
    let mut world = World::new();
    floor(&mut world);
    let b = world.add(
        BodyDesc::cuboid(Vec3::splat(0.3))
            .at(vec3(0.0, 0.15, 0.0))
            .with_material(Material::ALUMINIUM)
            .with_temperature(940.0)
            .with_failure(Failure::Deforms),
    );
    let mut sim = Simulation::new(world);
    sim.heat = HeatSettings::insulated();
    let mass = sim.world.bodies.mass(b.index());
    sim.step();
    let events = sim.take_failures();
    assert_eq!(events.len(), 1, "{events:?}");
    assert!(matches!(
        events[0].outcome,
        Outcome::Deformed { melted: true, .. }
    ));
    assert!(sim.world.bodies.is_removed(b.index()));
    assert_eq!(sim.world.solids.len(), 1);
    let solid = &sim.world.solids[0];
    assert!((solid.mass() - mass).abs() < 1e-3 * mass);
    // Sensible heat only: (940 - 933) K of it, at 900 J/kg K, into 397 kJ/kg.
    let expected = (940.0 - 933.0) * 900.0 / 397_000.0;
    assert!(
        (solid.melted_fraction() - expected).abs() < 0.1 * expected,
        "{} against {expected}",
        solid.melted_fraction()
    );
    for _ in 0..120 {
        sim.step();
    }
    assert!(sim.is_stable());
    assert!(!sim.mpm_report().unstable);
}

/// Nothing that cannot fail costs anything: no stress grid is built.
#[test]
fn a_scene_where_nothing_can_fail_does_no_failure_work() {
    let mut world = World::new();
    floor(&mut world);
    let b = world.add(
        BodyDesc::cuboid(Vec3::splat(0.5))
            .at(vec3(0.0, 2.0, 0.0))
            .with_material(Material::GLASS),
    );
    let mut sim = Simulation::new(world);
    for _ in 0..240 {
        sim.step();
    }
    assert!(!sim.world.bodies.stress[b.index()].is_built());
    assert_eq!(sim.stress_effort(), (0, 0));
}

/// Every piece of what scene body `source` became, with how far it runs along
/// its own grain.
fn pieces_along_grain(sim: &Simulation, source: usize) -> Vec<f32> {
    let b = &sim.world.bodies;
    (0..b.len())
        .filter(|&i| b.source[i] as usize == source && !b.is_removed(i))
        .map(|i| {
            let g = b.grain[i];
            let points: Vec<Vec3> = match &b.shape[i] {
                fusion_core::Shape::Hull(h) => h.vertices().to_vec(),
                shape => {
                    let a = shape.local_aabb();
                    vec![a.min, a.max]
                }
            };
            let (lo, hi) = points.iter().fold((f32::MAX, f32::MIN), |(lo, hi), p| {
                (lo.min(p.dot(g)), hi.max(p.dot(g)))
            });
            hi - lo
        })
        .collect()
}

/// A plank on two supports with a steel ball dropped on its middle snaps
/// across its grain — two long halves and a few splinters — rather than
/// shattering into chunks the way glass would, or splitting into slats.
#[test]
fn a_plank_struck_in_the_middle_snaps_across_its_grain() {
    let mut world = World::new();
    floor(&mut world);
    for x in [-0.8f32, 0.8] {
        world.add(
            BodyDesc::cuboid(vec3(0.2, 0.5, 0.3))
                .at(vec3(x, 0.25, 0.0))
                .with_material(Material::CONCRETE)
                .fixed(),
        );
    }
    let plank = world
        .add(
            BodyDesc::cuboid(vec3(1.9, 0.04, 0.15))
                .at(vec3(0.0, 0.52, 0.0))
                .with_material(Material::WOOD)
                .with_failure(Failure::Breaks),
        )
        .index();
    world.add(
        BodyDesc::sphere(0.15)
            .at(vec3(0.0, 2.5, 0.0))
            .with_material(Material::STEEL),
    );
    let mut sim = Simulation::new(world);
    let mut first = None;
    for _ in 0..240 {
        sim.step();
        if let Some(e) = sim.take_failures().into_iter().find(|e| e.body == plank) {
            first.get_or_insert(e);
        }
    }
    let first = first.expect("it broke");
    assert_eq!(first.parting, Parting::Snap, "{first:?}");
    let lengths = pieces_along_grain(&sim, plank);
    let halves = lengths.iter().filter(|&&l| l > 0.7).count();
    assert!(halves >= 2, "two long halves: {lengths:?}");
    // Halves: broken across. A half may split a slat off afterwards, the way
    // a snapped plank's ends often do, but nothing is the whole length.
    assert!(
        lengths.iter().all(|&l| l < 1.2),
        "broken across: {lengths:?}"
    );
}

/// A log lying on the ground with a steel blade dropped on it splits along
/// its grain, into pieces every one of which is the log's full length.
#[test]
fn a_log_struck_along_its_length_splits_along_its_grain() {
    let mut world = World::new();
    floor(&mut world);
    let log = world
        .add(
            BodyDesc::cuboid(vec3(0.5, 0.2, 0.2))
                .at(vec3(0.0, 0.1, 0.0))
                .with_material(Material::WOOD)
                .with_failure(Failure::Breaks),
        )
        .index();
    world.add(
        BodyDesc::cuboid(vec3(0.6, 0.3, 0.04))
            .at(vec3(0.0, 1.5, 0.0))
            .with_velocity(vec3(0.0, -8.0, 0.0))
            .with_material(Material::STEEL)
            .with_mass(40.0),
    );
    let mut sim = Simulation::new(world);
    let mut first = None;
    for _ in 0..120 {
        sim.step();
        if let Some(e) = sim.take_failures().into_iter().find(|e| e.body == log) {
            first.get_or_insert(e);
        }
    }
    let first = first.expect("it split");
    assert_eq!(first.parting, Parting::Split, "{first:?}");
    let lengths = pieces_along_grain(&sim, log);
    assert!(lengths.len() >= 2);
    for l in &lengths {
        assert!(
            (l - 0.5).abs() < 0.01,
            "every piece the log's length: {lengths:?}"
        );
    }
}

/// Volumes of what scene body `source` became, as fractions of it, largest
/// first.
fn piece_fractions(sim: &Simulation, source: usize) -> Vec<f32> {
    let b = &sim.world.bodies;
    let whole = sim.world.scene()[source].shape.volume();
    let mut f: Vec<f32> = (0..b.len())
        .filter(|&i| b.source[i] as usize == source && !b.is_removed(i))
        .map(|i| b.shape[i].volume() / whole)
        .collect();
    f.sort_by(|a, b| b.total_cmp(a));
    f
}

/// A glass pane on two supports, struck in the middle by a steel ball,
/// shatters into shards — not two halves.
#[test]
fn a_struck_pane_of_glass_shatters() {
    let mut world = World::new();
    floor(&mut world);
    for x in [-0.35f32, 0.35] {
        world.add(
            BodyDesc::cuboid(vec3(0.1, 0.3, 0.6))
                .at(vec3(x, 0.15, 0.0))
                .with_material(Material::CONCRETE)
                .fixed(),
        );
    }
    let pane = world
        .add(
            BodyDesc::cuboid(vec3(0.8, 0.02, 0.6))
                .at(vec3(0.0, 0.311, 0.0))
                .with_material(Material::GLASS)
                .with_failure(Failure::Breaks),
        )
        .index();
    world.add(
        BodyDesc::sphere(0.05)
            .at(vec3(0.0, 1.5, 0.0))
            .with_material(Material::STEEL),
    );
    let mut sim = Simulation::new(world);
    let mut first = None;
    for _ in 0..600 {
        sim.step();
        if let Some(e) = sim.take_failures().into_iter().find(|e| e.body == pane) {
            first.get_or_insert(e);
        }
    }
    let first = first.expect("it broke");
    assert_eq!(first.pattern, Pattern::Shatter);
    let pieces = piece_fractions(&sim, pane);
    assert!(pieces.len() >= 8, "shards, not halves: {pieces:?}");
    assert!(pieces[0] < 0.5, "no piece is most of it: {pieces:?}");
    // Every cut went through the pane: no shard is thinner than it.
    let b = &sim.world.bodies;
    for i in (0..b.len()).filter(|&i| b.source[i] as usize == pane) {
        let a = b.shape[i].local_aabb();
        assert!(
            (a.max - a.min).min_element() > 0.015,
            "a layer split off the pane"
        );
    }
}

/// A stone block dropped a metre onto a corner loses a chip off it and stays
/// most of a block.
#[test]
fn stone_dropped_on_a_corner_chips() {
    let mut world = World::new();
    floor(&mut world);
    let block = world
        .add(
            BodyDesc::cuboid(Vec3::splat(0.3))
                .at(vec3(0.0, 1.3, 0.0))
                .rotated(
                    fusion_math::Quat::from_rotation_x(0.6)
                        * fusion_math::Quat::from_rotation_z(0.6),
                )
                .with_material(Material::STONE)
                .with_failure(Failure::Breaks),
        )
        .index();
    let mut sim = Simulation::new(world);
    let mut first = None;
    for _ in 0..900 {
        sim.step();
        if let Some(e) = sim.take_failures().into_iter().find(|e| e.body == block) {
            first.get_or_insert(e);
        }
    }
    let first = first.expect("it chipped");
    assert_eq!(first.pattern, Pattern::Chip);
    let pieces = piece_fractions(&sim, block);
    assert!(pieces[0] > 0.75, "the block survives: {pieces:?}");
    assert!(pieces.len() <= 5, "chips, not rubble: {pieces:?}");
    assert!(
        pieces[1..].iter().all(|&f| f < 0.2),
        "small chips: {pieces:?}"
    );
}
