//! Behaviour components must survive reset, rewind and durable resume.
use fusion_core::{BodyDesc, World};
use fusion_math::{vec3, Vec3};
use fusion_script::Program;
use fusion_sim::{state_hash, Checkpoint, Simulation, StateError};

const MEMORY: &str = "if step == 0:\n    impulse = 0\n    history = vec3(0, 0, 0)\n    enabled = True\n    unused = None\nimpulse = impulse + 1\nhistory = history + vec3(1, 2, 3)\nif enabled:\n    self.force.x = impulse\n";

fn scene(script: &str) -> Simulation {
    let mut world = World::new();
    world.gravity = Vec3::ZERO;
    world.add(BodyDesc::sphere(0.5).with_mass(2.0));
    let mut sim = Simulation::new(world);
    sim.attach_script(0, Program::compile(script).unwrap());
    sim
}

fn steps(sim: &mut Simulation, count: usize) -> Vec<u64> {
    (0..count)
        .map(|_| {
            sim.step();
            state_hash(sim)
        })
        .collect()
}

#[test]
fn stateful_components_replay_exactly_after_an_in_process_checkpoint() {
    let mut sim = scene(MEMORY);
    steps(&mut sim, 37);
    let checkpoint = sim.checkpoint();
    let expected = steps(&mut sim, 41);
    sim.restore(&checkpoint);
    assert_eq!(steps(&mut sim, 41), expected);
}

#[test]
fn durable_resume_restores_every_script_value_type() {
    let mut sim = scene(MEMORY);
    steps(&mut sim, 37);
    let bytes = sim.checkpoint().to_bytes();
    let mut fresh = scene(MEMORY);
    let decoded = Checkpoint::from_bytes(&bytes, &fresh.checkpoint()).unwrap();
    fresh.restore(&decoded);
    assert_eq!(sim.scripts()[0].locals, fresh.scripts()[0].locals);
    assert_eq!(state_hash(&sim), state_hash(&fresh));
    // No contacts in this fixture: discarded disk contact caches cannot mask
    // an error in the restored behaviour component.
    assert_eq!(steps(&mut fresh, 41), steps(&mut sim, 41));
}

#[test]
fn reset_clears_script_memory_and_failure_state() {
    let mut sim = scene("if step == 3:\n    self.force.x = unknown\ncounter = step\n");
    let first = steps(&mut sim, 8);
    assert!(sim.scripts()[0].failed.is_some());
    assert!(!sim.scripts()[0].locals.is_empty());
    sim.reset();
    assert!(sim.scripts()[0].failed.is_none());
    assert!(sim.scripts()[0].locals.is_empty());
    assert!(sim.take_script_errors().is_empty());
    assert_eq!(steps(&mut sim, 8), first);
    assert_eq!(sim.take_script_errors().len(), 1);
}

#[test]
fn failures_and_pending_reports_survive_resume_without_reexecuting() {
    let source = "if step == 3:\n    self.force.x = unknown\ncounter = step\n";
    let mut sim = scene(source);
    steps(&mut sim, 8);
    let checkpoint = sim.checkpoint();
    let mut fresh = scene(source);
    fresh.restore(&Checkpoint::from_bytes(&checkpoint.to_bytes(), &fresh.checkpoint()).unwrap());
    assert_eq!(fresh.scripts()[0].failed, sim.scripts()[0].failed);
    assert_eq!(fresh.take_script_errors(), sim.take_script_errors());
    steps(&mut fresh, 5);
    assert!(fresh.take_script_errors().is_empty());
    let locals = fresh.scripts()[0].locals.clone();
    fresh.restore(&checkpoint);
    assert_eq!(fresh.scripts()[0].locals, locals);
    assert_eq!(fresh.take_script_errors().len(), 1);
}

#[test]
fn fingerprints_detect_hidden_memory_before_it_moves_an_object() {
    let make = |gravity| {
        let mut world = World::new();
        world.gravity = vec3(gravity, 0.0, 0.0);
        world.add(BodyDesc::sphere(0.5).fixed());
        let mut sim = Simulation::new(world);
        sim.attach_script(0, Program::compile("sample = world.gravity.x").unwrap());
        sim.step();
        sim
    };
    let a = make(1.0);
    let b = make(2.0);
    assert_eq!(a.world.bodies.position, b.world.bodies.position);
    assert_ne!(state_hash(&a), state_hash(&b));
}

#[test]
fn legacy_files_load_without_scripts_and_refuse_missing_script_memory() {
    let empty = scene("");
    let mut sim = scene("");
    sim.detach_script(0);
    let mut legacy = sim.checkpoint().to_bytes();
    assert!(sim.world.solids.is_empty());
    // Version 1 had no script or error counts, and no temperatures: a body
    // without a grid writes its temperature (8 bytes) and a zero cell count (4).
    let temperatures = 12 * sim.world.bodies.len();
    // Nor contact history (two zero counts since version 6) or the core
    // engine presence flag (4 bytes since version 7).
    let history = 12;
    legacy.truncate(legacy.len() - 8 - temperatures - history);
    // Nor what failure changes: each body's record ends in a flags byte and a
    // failure byte since version 4. Records start after the 44-byte header and
    // the body count, and are 69 bytes before those two.
    for k in (0..sim.world.bodies.len()).rev() {
        let end = 48 + k * 71 + 69;
        legacy.drain(end..end + 2);
    }
    legacy[8..12].copy_from_slice(&1_u32.to_le_bytes());
    assert!(Checkpoint::from_bytes(&legacy, &sim.checkpoint()).is_ok());
    assert!(matches!(
        Checkpoint::from_bytes(&legacy, &empty.checkpoint()),
        Err(StateError::Invalid(_))
    ));
}

#[test]
fn mismatched_scripts_corrupt_tags_and_truncated_states_are_refused() {
    let mut sim = scene(MEMORY);
    steps(&mut sim, 4);
    let template = sim.checkpoint();
    let bytes = template.to_bytes();
    let different = scene("self.force.x = 1").checkpoint();
    assert!(matches!(
        Checkpoint::from_bytes(&bytes, &different),
        Err(StateError::Invalid(_))
    ));
    for end in 0..bytes.len() {
        assert!(
            Checkpoint::from_bytes(&bytes[..end], &template).is_err(),
            "accepted {end} of {} bytes",
            bytes.len()
        );
    }
    let mut malformed = bytes.clone();
    let key = b"impulse";
    let value_tag = malformed
        .windows(key.len())
        .rposition(|w| w == key)
        .unwrap()
        + key.len();
    malformed[value_tag] = 255;
    assert!(matches!(
        Checkpoint::from_bytes(&malformed, &template),
        Err(StateError::Invalid(_))
    ));
    let mut trailing = bytes;
    trailing.push(0);
    assert!(Checkpoint::from_bytes(&trailing, &template).is_err());
    assert_eq!(template.to_bytes(), sim.checkpoint().to_bytes());
}

#[test]
fn script_checkpoint_bytes_do_not_depend_on_hashmap_insertion_order() {
    let mut a = scene(MEMORY);
    let mut b = scene(MEMORY);
    steps(&mut a, 10);
    steps(&mut b, 10);
    assert_eq!(a.checkpoint().to_bytes(), b.checkpoint().to_bytes());
}

#[test]
fn removing_or_replacing_components_keeps_error_reports_attached_correctly() {
    let mut sim = scene("self.force.x = unknown");
    sim.world.add(BodyDesc::sphere(0.5).at(vec3(5.0, 0.0, 0.0)));
    sim.attach_script(1, Program::compile("self.force.y = missing").unwrap());
    sim.step();
    sim.detach_script(0);
    let cp = sim.checkpoint();
    assert!(Checkpoint::from_bytes(&cp.to_bytes(), &cp).is_ok());
    let errors = sim.take_script_errors();
    assert_eq!(errors.len(), 1);
    assert_eq!(errors[0].0, 0);
    assert_eq!(sim.scripts()[errors[0].0].body, 1);
    sim.reset();
    sim.step();
    sim.attach_script(1, Program::compile("self.force.y = 1").unwrap());
    assert!(sim.take_script_errors().is_empty());
    assert!(sim.scripts()[0].failed.is_none());
}

/// A script that writes a non-finite number into its body gets an error,
/// and the body keeps the value it had.
///
/// Script numbers are `f32`, so `0/0` is NaN and `1e39` is infinite. Written
/// into a body, either one made it vanish without an error to say why. The
/// engine contained it — the box underneath stayed finite — but the scripted
/// body was gone, and the script reported success.
#[test]
fn a_script_cannot_write_a_non_finite_number_into_its_body() {
    for line in [
        "self.spin = vec(0, 0, 0/0)",
        "self.velocity = vec(1e39, 0, 0)",
        "self.force = vec(0, 1e30 * 1e30, 0)",
        "self.torque = vec(0/0, 0, 0)",
    ] {
        let mut world = World::new();
        world.add(BodyDesc::cuboid(Vec3::ONE).at(vec3(0.0, 0.5, 0.0)));
        let mut sim = Simulation::new(world);
        sim.attach_script(0, Program::compile(line).expect("compiles"));
        sim.step();
        let errors = sim.take_script_errors();
        assert_eq!(errors.len(), 1, "{line}: no error was reported");
        assert!(
            errors[0].1.message.contains("must be finite"),
            "{line}: {}",
            errors[0].1.message
        );
        assert!(sim.is_stable(), "{line}: the body went non-finite");
    }
}
