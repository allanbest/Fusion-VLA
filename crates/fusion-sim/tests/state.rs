//! A checkpoint written to bytes and read back.
//!
//! PLAN.md §23's checkpoint/resume, and the half of §22 that says large data
//! belongs in a blob rather than in the manifest. What is tested here is the
//! claim `fusion_sim::state` makes and the limit it admits to, in that order.

use fusion_core::{BodyDesc, Material, SolidDesc, SolidMaterial, SolidParams, World};
use fusion_math::{vec3, Vec3};
use fusion_sim::{state_hash, Checkpoint, Simulation, StateError};

/// Something of every kind, so the encoding is exercised rather than
/// described: eight boxes that stack and settle, a jelly block that deforms,
/// and a cloth that drapes.
fn scene() -> World {
    let mut world = World::new();
    world.add(
        BodyDesc::cuboid(vec3(8.0, 1.0, 8.0))
            .at(vec3(0.0, -0.5, 0.0))
            .with_material(Material::CONCRETE)
            .fixed(),
    );
    for i in 0..8 {
        world.add(
            BodyDesc::cuboid(Vec3::splat(0.3))
                .at(vec3(
                    (i % 3) as f32 * 0.31 - 0.31,
                    0.2 + i as f32 * 0.32,
                    (i % 2) as f32 * 0.05,
                ))
                .with_material(Material::WOOD),
        );
    }
    world.add_solid(
        &SolidDesc::block(Vec3::splat(0.3))
            .at(vec3(1.2, 0.5, 0.0))
            .with_params(SolidParams::at(0.05).of(SolidMaterial::JELLY)),
    );
    world.add_cloth(
        &fusion_core::ClothDesc::new(8, 8, 0.1)
            .at(Vec3::new(-1.2, 1.0, 0.0))
            .with_mass(0.4),
    );
    world
}

fn run(steps: usize) -> Simulation {
    let mut sim = Simulation::new(scene());
    for _ in 0..steps {
        sim.step();
    }
    sim
}

/// Every number the file carries comes back exactly.
///
/// Bit-exact rather than close: the encoding is fixed-width little-endian and
/// there is no rounding anywhere in it, so "approximately" would be hiding
/// something. A deformation gradient that came back to seven digits would
/// restore a solid that is *nearly* the one that was saved, and a continuum
/// diverges from nearly.
#[test]
fn the_physical_state_survives_the_round_trip_bit_for_bit() {
    let saved = run(400);
    let bytes = saved.checkpoint().to_bytes();

    let mut fresh = Simulation::new(scene());
    let restored = Checkpoint::from_bytes(&bytes, &fresh.checkpoint()).expect("decodes");
    fresh.restore(&restored);

    assert_eq!(
        fresh.step_index(),
        saved.step_index(),
        "the clock did not come back"
    );
    for i in 0..saved.world.bodies.len() {
        assert_eq!(
            fresh.world.bodies.position[i], saved.world.bodies.position[i],
            "body {i} moved through the file"
        );
        assert_eq!(
            fresh.world.bodies.orientation[i].to_array(),
            saved.world.bodies.orientation[i].to_array()
        );
        assert_eq!(
            fresh.world.bodies.linear_velocity[i],
            saved.world.bodies.linear_velocity[i]
        );
        assert_eq!(
            fresh.world.bodies.sleeping[i], saved.world.bodies.sleeping[i],
            "body {i} woke up in the file"
        );
    }
    let (a, b) = (&fresh.world.solids[0], &saved.world.solids[0]);
    for i in 0..b.len() {
        assert_eq!(a.particles.position[i], b.particles.position[i]);
        assert_eq!(
            a.deformation[i], b.deformation[i],
            "material point {i} came back with a different deformation"
        );
        assert_eq!(a.enthalpy[i], b.enthalpy[i]);
        assert_eq!(a.damage[i], b.damage[i]);
    }
    for i in 0..saved.world.particles.len() {
        assert_eq!(
            fresh.world.particles.position[i],
            saved.world.particles.position[i]
        );
        assert_eq!(
            fresh.world.particles.velocity[i],
            saved.world.particles.velocity[i]
        );
    }
}

/// And a run continued from the file is the run it was cut from.
///
/// It was not, before the file carried the contact history: the anchors the
/// collision pipeline keeps between steps and the multipliers the solver
/// warm-starts from. Without them a resumed pile started every contact from
/// nothing, and how far it wandered from the original depended on how
/// marginal the pile was — 31 mm on this scene once, and 39 cm when a box
/// balanced most of the way over an edge lost what friction was holding it
/// by and slid off. With them it is the same simulation, to the bit.
#[test]
fn a_resumed_run_is_the_run_it_was_cut_from() {
    let mut straight = run(400);
    let bytes = straight.checkpoint().to_bytes();

    let mut resumed = Simulation::new(scene());
    let restored = Checkpoint::from_bytes(&bytes, &resumed.checkpoint()).expect("decodes");
    resumed.restore(&restored);
    assert_eq!(
        state_hash(&resumed),
        state_hash(&straight),
        "the two are the same state at the cut"
    );

    for step in 0..600 {
        straight.step();
        resumed.step();
        assert_eq!(
            state_hash(&resumed),
            state_hash(&straight),
            "a resumed run parted from the run it was cut from {step} steps on"
        );
    }
}

/// A state file from another scene is refused, and the message says what it
/// found rather than where it panicked.
#[test]
fn a_state_file_belongs_to_its_scene() {
    let bytes = run(30).checkpoint().to_bytes();

    let mut other = World::new();
    other.add(BodyDesc::sphere(0.5).at(vec3(0.0, 2.0, 0.0)));
    let sim = Simulation::new(other);
    match Checkpoint::from_bytes(&bytes, &sim.checkpoint()) {
        Err(StateError::Shape { what, found, .. }) => {
            assert_eq!(what, "bodies");
            assert_eq!(found, 9);
        }
        other => panic!("expected a shape mismatch, got {other:?}"),
    }
}

/// And so is a file that is not one.
#[test]
fn nonsense_is_refused_rather_than_read() {
    let sim = Simulation::new(scene());
    let template = sim.checkpoint();
    assert!(matches!(
        Checkpoint::from_bytes(b"not a state file at all", &template),
        Err(StateError::NotState)
    ));

    let mut truncated = sim.checkpoint().to_bytes();
    truncated.truncate(40);
    assert!(matches!(
        Checkpoint::from_bytes(&truncated, &template),
        Err(StateError::Truncated { .. })
    ));

    let mut future = sim.checkpoint().to_bytes();
    future[8..12].copy_from_slice(&(fusion_sim::STATE_VERSION + 1).to_le_bytes());
    assert!(matches!(
        Checkpoint::from_bytes(&future, &template),
        Err(StateError::Version { .. })
    ));
}

/// Contact history that names a body the scene does not have is refused,
/// and history that names real ones is read.
#[test]
fn contact_history_naming_a_missing_body_is_refused() {
    let sim = Simulation::new(scene());
    let template = sim.checkpoint();
    let mut bytes = template.to_bytes();
    // Before the first step there is no history: the file ends in its two
    // counts, both zero, and then the Fusion Core flag (zero: no Core state,
    // since version 7). Replace the counts with one remembered pair.
    assert_eq!(&bytes[bytes.len() - 12..], &[0u8; 12]);
    bytes.truncate(bytes.len() - 12);
    let with_pair = |b: u32| {
        let mut out = bytes.clone();
        out.extend_from_slice(&1u32.to_le_bytes());
        out.extend_from_slice(&0u32.to_le_bytes());
        out.extend_from_slice(&b.to_le_bytes());
        for v in [0.0f32, 1.0, 0.0, 0.0, 1.0, 0.0, 0.5, 0.0] {
            out.extend_from_slice(&v.to_le_bytes());
        }
        out.push(0);
        out.extend_from_slice(&0u32.to_le_bytes());
        out.extend_from_slice(&0u32.to_le_bytes()); // no Fusion Core state
        out
    };
    assert!(Checkpoint::from_bytes(&with_pair(1), &template).is_ok());
    assert!(matches!(
        Checkpoint::from_bytes(&with_pair(9_999), &template),
        Err(StateError::Invalid(_))
    ));
}
