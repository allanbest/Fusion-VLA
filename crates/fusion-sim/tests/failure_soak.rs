//! Breaking, at random: every library material, every failure option, shapes
//! of every kind dropped and thrown from heights that break most of them. What
//! must hold however it goes — nothing becomes NaN, nothing is lost but what
//! turns into a continuum, the world stays bounded, and a checkpoint taken in
//! the middle of it all replays to the same bits.

use fusion_core::{BodyDesc, Failure, Material, World};
use fusion_math::{vec3, Quat};
use fusion_sim::{state_hash, Checkpoint, HeatSettings, Simulation, StressSettings};

const MATERIALS: [Material; 9] = [
    Material::GENERIC,
    Material::STEEL,
    Material::ALUMINIUM,
    Material::CONCRETE,
    Material::WOOD,
    Material::RUBBER,
    Material::ICE,
    Material::GLASS,
    Material::STONE,
];

fn scene(seed: u64) -> Simulation {
    let mut world = World::new();
    world.add(
        BodyDesc::cuboid(vec3(30.0, 1.0, 30.0))
            .at(vec3(0.0, -0.5, 0.0))
            .with_material(Material::CONCRETE)
            .fixed(),
    );
    let mut rng = fusion_core::Rng::with_stream(seed, 7);
    for k in 0..7 {
        let material =
            MATERIALS[(rng.range(0.0, MATERIALS.len() as f32) as usize).min(MATERIALS.len() - 1)];
        let failure = [
            Failure::Breaks,
            Failure::Breaks,
            Failure::Deforms,
            Failure::Never,
        ][(rng.range(0.0, 4.0) as usize).min(3)];
        let size = vec3(
            rng.range(0.05, 0.8),
            rng.range(0.03, 0.5),
            rng.range(0.05, 0.6),
        );
        let desc = if k % 3 == 2 {
            BodyDesc::sphere(size.y.max(0.05))
        } else {
            BodyDesc::cuboid(size)
        };
        world.add(
            desc.at(vec3(
                rng.range(-1.5, 1.5),
                rng.range(0.5, 6.0),
                rng.range(-1.5, 1.5),
            ))
            .rotated(
                Quat::from_rotation_x(rng.range(-1.0, 1.0))
                    * Quat::from_rotation_z(rng.range(-1.0, 1.0)),
            )
            .with_velocity(vec3(
                rng.range(-3.0, 3.0),
                rng.range(-12.0, 0.0),
                rng.range(-3.0, 3.0),
            ))
            .with_material(material)
            .with_failure(failure),
        );
    }
    let mut sim = Simulation::new(world);
    sim.heat = HeatSettings::on();
    sim.stress = StressSettings::on();
    sim
}

/// Mass of everything that moves: rigid bodies and continuum solids.
fn moving_mass(sim: &Simulation) -> f64 {
    let b = &sim.world.bodies;
    let rigid: f64 = (0..b.len())
        .filter(|&i| b.inv_mass[i] > 0.0 && !b.is_removed(i))
        .map(|i| 1.0 / b.inv_mass[i] as f64)
        .sum();
    let solid: f64 = sim.world.solids.iter().map(|s| s.mass() as f64).sum();
    rigid + solid
}

fn check(sim: &Simulation, what: &str) {
    let b = &sim.world.bodies;
    for i in 0..b.len() {
        if b.is_removed(i) {
            continue;
        }
        let p = fusion_math::to_f32(b.position[i]);
        assert!(
            p.is_finite() && b.linear_velocity[i].is_finite() && b.angular_velocity[i].is_finite(),
            "{what}: body {i} went non-finite"
        );
        let q = b.orientation[i];
        assert!(
            q.x.is_finite() && q.y.is_finite() && q.z.is_finite() && q.w.is_finite(),
            "{what}: body {i} orientation"
        );
        assert!(
            p.y > -2.0,
            "{what}: body {i} fell through the floor to {p:?}"
        );
        let s = &b.stress[i];
        assert!(
            s.utilisation.is_finite() && s.peak_von_mises.is_finite(),
            "{what}: body {i} stress"
        );
    }
    assert!(b.len() < 4096);
}

#[test]
fn breaking_at_random_keeps_its_books() {
    let mut total = 0;
    for seed in 1..=4u64 {
        let mut sim = scene(seed);
        let mass = moving_mass(&sim);
        let mut events = 0;
        let mut middle = None;
        for step in 0..300 {
            sim.step();
            events += sim.take_failures().len();
            check(&sim, &format!("seed {seed} step {step}"));
            if step == 150 {
                middle = Some((sim.checkpoint(), sim.world.bodies.len()));
            }
        }
        let now = moving_mass(&sim);
        assert!(
            ((now - mass) / mass).abs() < 1e-3,
            "seed {seed}: mass {mass} became {now} after {events} failures"
        );
        total += events;
        // From the middle of it, the same run again, to the bit.
        let (checkpoint, _) = middle.unwrap();
        let mut again = scene(seed);
        again.restore(&checkpoint);
        let mut first = scene(seed);
        first.restore(&checkpoint);
        for _ in 0..60 {
            first.step();
            again.step();
        }
        assert_eq!(
            state_hash(&first),
            state_hash(&again),
            "seed {seed} replays"
        );
        // And through bytes, as a saved state.
        let bytes = checkpoint.to_bytes();
        let mut fresh = scene(seed);
        let read = Checkpoint::from_bytes(&bytes, &fresh.checkpoint()).expect("reads back");
        fresh.restore(&read);
        for _ in 0..30 {
            fresh.step();
            check(&fresh, &format!("seed {seed} after restoring bytes"));
        }
    }
    assert!(total > 4, "the scenes broke things: {total} failures");
}
