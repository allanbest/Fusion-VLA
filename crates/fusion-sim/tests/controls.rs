//! Numerical and replay contracts for reusable behaviours.
use fusion_core::{BodyDesc, World};
use fusion_math::{vec3, Quat, Vec3};
use fusion_sim::{control::*, state_hash, Checkpoint, Simulation};

fn scene() -> Simulation {
    let mut world = World::new();
    world.gravity = Vec3::ZERO;
    world.linear_damping = 0.0;
    world.angular_damping = 0.0;
    world.add(BodyDesc::sphere(0.5).with_mass(2.0));
    let mut sim = Simulation::new(world);
    sim.inputs = movement_actions();
    sim.components.push(BodyComponents {
        body: 0,
        components: Components {
            controller: Some(ForceController::default()),
            probe: true,
            ..Default::default()
        },
    });
    sim
}
fn steps(sim: &mut Simulation, n: usize) -> Vec<u64> {
    (0..n)
        .map(|_| {
            sim.step();
            state_hash(sim)
        })
        .collect()
}

#[test]
fn controller_obeys_force_mass_and_diagonal_magnitude() {
    let mut sim = scene();
    sim.inputs[0].record(0, 1.0);
    sim.inputs[1].record(0, 1.0);
    sim.step();
    let v = sim.world.bodies.linear_velocity[0];
    let expected = 10.0 / 2.0 * sim.solver_dt;
    assert!((v.length() - expected).abs() < 1e-5, "{v:?}");
    assert!((v.x - v.y).abs() < 1e-6);
    assert_eq!(sim.probe_samples().len(), 1);
    let r = sim.probe_samples().back().unwrap();
    assert_eq!(r.step, 1);
    assert_eq!(r.time, sim.solver_dt as f64);
    assert!((r.kinetic - v.length_squared()).abs() < 1e-6);
}

#[test]
fn controller_constant_force_and_script_add_instead_of_overwriting() {
    let mut sim = scene();
    sim.inputs[0].record(0, 1.0);
    sim.components[0].components.force = Some(AppliedForce {
        force: [2.0, 0.0, 0.0],
        ..Default::default()
    });
    sim.attach_script(
        0,
        fusion_script::Program::compile("self.force.x = input.move_x * 4").unwrap(),
    );
    sim.step();
    assert!(sim.take_script_errors().is_empty());
    assert!((sim.world.bodies.linear_velocity[0].x - 8.0 * sim.solver_dt).abs() < 1e-5);
}

#[test]
fn local_forces_and_torques_rotate_with_the_body() {
    let mut sim = scene();
    sim.components[0].components.controller = None;
    sim.world.bodies.orientation[0] = Quat::from_euler(vec3(0.0, 0.0, std::f32::consts::FRAC_PI_2));
    let inverse = sim.world.bodies.inv_inertia_world(0);
    sim.components[0].components.force = Some(AppliedForce {
        force: [4.0, 0.0, 0.0],
        torque: [2.0, 0.0, 0.0],
        local: true,
        ..Default::default()
    });
    sim.step();
    let v = sim.world.bodies.linear_velocity[0];
    assert!(
        v.x.abs() < 1e-5 && (v.y - 2.0 * sim.solver_dt).abs() < 1e-5,
        "{v:?}"
    );
    let expected = inverse * vec3(0.0, 2.0 * sim.solver_dt, 0.0);
    assert!((sim.world.bodies.angular_velocity[0] - expected).length() < 1e-5);
    assert!(sim.read_body(0).unwrap().kinetic > 0.0);
}

#[test]
fn take_replays_through_reset_memory_and_disk_checkpoints() {
    let mut sim = scene();
    sim.inputs[0].record(0, 1.0);
    sim.inputs[0].record(20, -0.5);
    sim.inputs[0].record(45, 0.0);
    sim.attach_script(
        0,
        fusion_script::Program::compile("self.force.y = input.move_x * 3").unwrap(),
    );
    let first = steps(&mut sim, 17);
    let cp = sim.checkpoint();
    let rest = steps(&mut sim, 55);
    sim.restore(&cp);
    assert!(sim.probe_samples().is_empty());
    assert_eq!(steps(&mut sim, 55), rest);
    let mut other = scene();
    other.inputs = sim.inputs.clone();
    other.attach_script(
        0,
        fusion_script::Program::compile("self.force.y = input.move_x * 3").unwrap(),
    );
    let decoded = Checkpoint::from_bytes(&cp.to_bytes(), &other.checkpoint()).unwrap();
    other.restore(&decoded);
    assert_eq!(steps(&mut other, 55), rest);
    sim.reset();
    assert!(sim.probe_samples().is_empty());
    assert_eq!(steps(&mut sim, 17), first);
    assert_eq!(steps(&mut sim, 55), rest);
}

#[test]
fn events_are_ordered_finite_and_replace_the_future_explicitly() {
    let mut a = InputAction::axis("throttle", "", "space");
    assert_eq!(a.value(0), 0.0);
    a.record(5, 1.0);
    a.record(15, 0.0);
    a.record(10, -1.0);
    assert_eq!(a.value(4), 0.0);
    assert_eq!(a.value(5), 1.0);
    assert_eq!(a.value(100), -1.0);
    assert_eq!(a.events.len(), 2);
    assert!(!a.record(11, f32::NAN));
    assert!(!a.record(11, 2.0));
    a.record(10, 1.0);
    assert_eq!(a.events.len(), 1);
    assert!(validate_actions(&[a.clone()]).is_ok());
    assert!(validate_actions(&[a.clone(), a.clone()]).is_err());
    a.events.push(InputEvent {
        step: 5,
        value: 0.0,
    });
    assert!(validate_actions(&[a]).is_err());
}

#[test]
fn scripts_cannot_mutate_controls_and_unknown_actions_raise() {
    for source in ["input.move_x = 1", "self.force.x = input.typo"] {
        let mut sim = scene();
        sim.attach_script(0, fusion_script::Program::compile(source).unwrap());
        sim.step();
        assert_eq!(sim.take_script_errors().len(), 1, "{source}");
        assert_eq!(sim.inputs[0].value(0), 0.0);
    }
}

#[test]
fn a_live_driven_session_is_its_own_recording() {
    // Drive the scene from outside, the way a game loop would: decide each
    // step's input, then step.
    let mut live = scene();
    let mut hashes = Vec::new();
    for step in 0..40u64 {
        let value = if step < 20 { 1.0 } else { -1.0 };
        assert!(live.set_action("move_x", value) || step % 20 != 0);
        live.step();
        hashes.push(state_hash(&live));
    }

    // Nothing was saved deliberately, yet the take is complete: replaying the
    // scene it left behind reproduces the session step for step.
    let mut replay = scene();
    replay.inputs = live.inputs.clone();
    assert_eq!(steps(&mut replay, 40), hashes);
}

#[test]
fn driving_an_action_reads_back_and_refuses_what_it_should() {
    let mut sim = scene();
    assert_eq!(sim.action_value("move_x"), Some(0.0));
    assert!(sim.set_action("move_x", 1.0));
    assert_eq!(sim.action_value("move_x"), Some(1.0));
    // Setting the same value twice is not a second event.
    assert!(!sim.set_action("move_x", 1.0));

    assert!(!sim.set_action("move_x", 2.0));
    assert!(!sim.set_action("move_x", f32::NAN));
    assert_eq!(sim.action_value("move_x"), Some(1.0));
    assert!(!sim.set_action("no_such_action", 1.0));
    assert_eq!(sim.action_value("no_such_action"), None);
    assert_eq!(
        sim.action_names().collect::<Vec<_>>(),
        ["move_x", "move_y", "move_z"]
    );
}

#[test]
fn driving_after_a_rewind_replaces_the_future_it_rewound_past() {
    let mut sim = scene();
    sim.set_action("move_x", 1.0);
    let early = sim.checkpoint();
    steps(&mut sim, 10);
    sim.set_action("move_x", -1.0);
    steps(&mut sim, 10);
    assert_eq!(sim.inputs[0].value(15), -1.0);

    // Rewind to before the reversal and drive somewhere else: the take that
    // survives is the one that was actually played, not a blend of both.
    sim.restore(&early);
    assert!(sim.set_action("move_x", 0.5));
    assert_eq!(sim.inputs[0].value(15), 0.5);
    assert_eq!(sim.inputs[0].events.len(), 1);
}
