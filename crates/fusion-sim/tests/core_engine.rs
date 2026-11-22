use fusion_core::{BodyDesc, FluidDesc, FluidParams, SolidDesc, SolidMaterial, SolidParams, World};
use fusion_math::{vec3, Vec3};
use fusion_sim::{state_hash, Checkpoint, EngineKind, Simulation, Timeline};

fn scene() -> Simulation {
    let mut world = World::new();
    world.angular_damping = 0.0;
    world.add(
        BodyDesc::cuboid(vec3(4.0, 0.4, 4.0))
            .at(vec3(0.0, -0.2, 0.0))
            .fixed(),
    );
    world.add(
        BodyDesc::sphere(0.1)
            .at(vec3(0.0, 0.45, 0.0))
            .with_mass(2.0),
    );
    let mut sim = Simulation::new(world);
    sim.solver_dt = 1.0 / 120.0;
    sim.use_core_engine().unwrap();
    sim
}
fn steps(sim: &mut Simulation, n: usize) {
    for _ in 0..n {
        let r = sim.step();
        assert!(r.failure.is_none(), "{r:?}");
    }
}
#[test]
fn new_core_drives_the_shared_runtime_and_contacts() {
    let mut sim = scene();
    assert_eq!(sim.engine_kind(), EngineKind::Core);
    assert_eq!(sim.solver_info().name, "Fusion Core V1");
    let mass = sim.world.bodies.mass(1);
    steps(&mut sim, 120);
    assert!((sim.time() - 1.0).abs() < 1e-6);
    assert_eq!(sim.world.bodies.mass(1), mass);
    assert!((sim.world.bodies.position[1].y - 0.1).abs() < 0.002);
    assert!(sim.world.bodies.linear_velocity[1].length() < 0.02);
    assert_eq!(sim.diagnostics.steps, 120);
}

#[test]
fn main_app_backend_stops_crossing_bar_interiors() {
    let mut world = World::new();
    world.gravity = Vec3::ZERO;
    world.angular_damping = 0.0;
    world.add(
        BodyDesc::cuboid(vec3(2.0, 0.1, 0.1))
            .at(vec3(0.0, -0.05, 0.0))
            .fixed(),
    );
    world.add(
        BodyDesc::cuboid(vec3(0.1, 0.1, 2.0))
            .at(vec3(0.0, 0.2, 0.0))
            .with_velocity(vec3(0.0, -5.0, 0.0)),
    );
    let mut sim = Simulation::new(world);
    sim.solver_dt = 1.0 / 30.0;
    sim.use_core_engine().unwrap();
    for _ in 0..4 {
        steps(&mut sim, 1);
        assert!(sim.world.bodies.position[1].y > 0.048);
    }
    let saved = sim.checkpoint();
    steps(&mut sim, 1);
    let expected = state_hash(&sim);
    sim.restore(&saved);
    steps(&mut sim, 1);
    assert_eq!(state_hash(&sim), expected);
}
#[test]
fn exact_checkpoint_and_disk_replay_through_contact() {
    let mut sim = scene();
    steps(&mut sim, 25);
    let checkpoint = sim.checkpoint();
    let bytes = checkpoint.to_bytes();
    let mut hashes = Vec::new();
    for _ in 0..50 {
        steps(&mut sim, 1);
        hashes.push(state_hash(&sim));
    }
    sim.restore(&checkpoint);
    for &hash in &hashes {
        steps(&mut sim, 1);
        assert_eq!(state_hash(&sim), hash);
    }
    let mut resumed = scene();
    let loaded = Checkpoint::from_bytes(&bytes, &resumed.checkpoint()).unwrap();
    resumed.restore(&loaded);
    for &hash in &hashes {
        steps(&mut resumed, 1);
        assert_eq!(state_hash(&resumed), hash);
    }
    assert!(Checkpoint::from_bytes(&bytes[..bytes.len() - 1], &resumed.checkpoint()).is_err());
    assert!(Checkpoint::from_bytes(&bytes, &Simulation::new(World::new()).checkpoint()).is_err());
}
#[test]
fn failed_solve_rolls_back_and_retains_requested_time() {
    let mut sim = scene();
    steps(&mut sim, 20);
    let before = state_hash(&sim);
    let time = sim.time();
    sim.core_settings_mut().unwrap().max_iterations = 0;
    let report = sim.advance(sim.solver_dt as f64).unwrap();
    assert!(report.failure.is_some());
    assert_eq!(report.dt, 0.0);
    assert_eq!(sim.time(), time);
    assert_eq!(state_hash(&sim), before);
    sim.core_settings_mut().unwrap().max_iterations = 30;
    assert!(sim.advance(0.0).unwrap().failure.is_none());
    assert!(sim.time() > time);
}
#[test]
fn reset_and_seek_preserve_the_new_engine() {
    let mut sim = scene();
    let initial = state_hash(&sim);
    let mut timeline = Timeline::new(0.05);
    timeline.observe(&sim);
    let mut hashes = Vec::new();
    for _ in 0..40 {
        steps(&mut sim, 1);
        timeline.observe(&sim);
        hashes.push(state_hash(&sim));
    }
    let target = 15.0 * sim.solver_dt as f64;
    assert!(timeline.seek(&mut sim, target).exact);
    assert_eq!(state_hash(&sim), hashes[14]);
    sim.reset();
    assert_eq!(sim.engine_kind(), EngineKind::Core);
    assert_eq!(state_hash(&sim), initial);
}
#[test]
fn quadratic_solid_has_a_real_boundary_and_conserves_mass_in_free_fall() {
    let mut world = World::new();
    world.angular_damping = 0.0;
    let desc = SolidDesc::block(Vec3::splat(0.2))
        .at(vec3(0.0, 5.0, 0.0))
        .with_params(SolidParams::at(0.05).of(SolidMaterial::JELLY));
    world.add_solid(&desc);
    let mut sim = Simulation::new(world);
    sim.solver_dt = 1.0 / 60.0;
    sim.use_core_engine().unwrap();
    assert!(sim.core_settings_mut().unwrap().self_contact);
    assert!(sim.core_surface().unwrap().is_closed());
    let mass = sim.world.solids[0].mass();
    assert!((mass - 8.0).abs() < 1e-4, "{mass}");
    steps(&mut sim, 30);
    let solid = &sim.world.solids[0];
    let center = solid
        .particles
        .position
        .iter()
        .zip(&solid.particles.inv_mass)
        .map(|(p, &im)| p.y / im as f64)
        .sum::<f64>()
        / mass as f64;
    let expected = 5.0 - 0.5 * 9.81 * sim.time().powi(2);
    assert!((center - expected).abs() < 0.005, "{center} vs {expected}");
    assert_eq!(solid.mass(), mass);
    let saved = sim.checkpoint();
    steps(&mut sim, 1);
    let hash = state_hash(&sim);
    sim.restore(&saved);
    steps(&mut sim, 1);
    assert_eq!(state_hash(&sim), hash);
}
#[test]
fn unsupported_physics_is_rejected_before_selection() {
    let mut world = World::new();
    world.add_fluid(&FluidDesc::block(Vec3::splat(0.2)).with_params(FluidParams::water_at(0.05)));
    let mut sim = Simulation::new(world);
    assert!(sim.use_core_engine().unwrap_err().contains("Fluids"));
    assert_eq!(sim.engine_kind(), EngineKind::Standard);
}

#[test]
fn edits_to_core_output_require_a_reset_instead_of_being_lost() {
    let mut sim = scene();
    sim.world.bodies.position[1].y += 1.0;
    let edited = sim.world.bodies.position[1];
    assert_eq!(
        sim.step().failure,
        Some(fusion_sim::StepFailure::SceneChanged)
    );
    assert_eq!(sim.world.bodies.position[1], edited);
    assert_eq!(sim.time(), 0.0);
    sim.reset();
    assert!(sim.step().failure.is_none());
}

#[test]
fn switching_back_at_reset_rebuilds_the_original_mpm_samples() {
    let mut world = World::new();
    world.add_solid(
        &SolidDesc::block(Vec3::splat(0.2))
            .with_params(SolidParams::at(0.05).of(SolidMaterial::JELLY)),
    );
    let count = world.solids[0].len();
    let mut sim = Simulation::new(world);
    sim.use_core_engine().unwrap();
    assert_ne!(sim.world.solids[0].len(), count);
    sim.use_standard_engine().unwrap();
    assert_eq!(sim.world.solids[0].len(), count);
    assert_eq!(sim.engine_kind(), EngineKind::Standard);
    assert!(sim.step().failure.is_none());
}

#[test]
fn throughput_runner_matches_app_physics_and_full_history_exactly() {
    let mut world = World::new();
    world.add(
        BodyDesc::cuboid(vec3(4.0, 0.4, 4.0))
            .at(vec3(0.0, -0.2, 0.0))
            .fixed(),
    );
    world.add(
        BodyDesc::sphere(0.1)
            .at(vec3(0.4, 0.4, 0.0))
            .with_velocity(vec3(0.1, 0.0, 0.0))
            .with_angular_velocity(vec3(0.1, 0.2, 0.3)),
    );
    world.add_solid(
        &SolidDesc::block(Vec3::splat(0.2))
            .at(vec3(-0.4, 0.16, 0.0))
            .with_params(SolidParams::at(0.05).of(SolidMaterial::JELLY)),
    );
    let mut app = Simulation::new(world.clone());
    app.solver_dt = 1.0 / 120.0;
    app.use_core_engine().unwrap();
    let mut fast = Simulation::new(world);
    fast.solver_dt = app.solver_dt;
    fast.use_core_engine().unwrap();
    let mut runner = fast.into_core_runner(1).unwrap();
    let mut a = Vec::new();
    let mut b = Vec::new();
    for _ in 0..100 {
        assert!(app.step().failure.is_none());
        assert!(!runner.step().failed());
        assert!(app.core_state_into(&mut a));
        runner.state_into(&mut b);
        assert_eq!(
            a.iter().map(|x| x.to_bits()).collect::<Vec<_>>(),
            b.iter().map(|x| x.to_bits()).collect::<Vec<_>>()
        );
    }
    let checkpoint = a;
    runner.reset(123, 0.0, 0.0).unwrap();
    for _ in 0..100 {
        assert!(!runner.step().failed());
    }
    runner.state_into(&mut b);
    assert_eq!(checkpoint, b);
}

#[test]
fn core_supports_joints_springs_and_urdf_import() {
    use fusion_core::{Joint, JointKind};
    let mut world = World::new();
    let anchor = world.add(
        BodyDesc::cuboid(vec3(0.4, 0.4, 0.4))
            .at(vec3(0.0, 4.0, 0.0))
            .fixed(),
    );
    let bob = world.add(BodyDesc::sphere(0.2).at(vec3(1.5, 4.0, 0.0)).with_mass(1.0));
    let mut hinge = Joint::new(anchor, bob, JointKind::Hinge);
    hinge.local_anchor_b = vec3(-1.5, 0.0, 0.0);
    world.add_joint(hinge);

    let mut sim = Simulation::new(world);
    sim.solver_dt = 1.0 / 120.0;
    sim.core_settings.islands = true;
    sim.use_core_engine().unwrap();
    steps(&mut sim, 40);
    // Hinging from x=1.5 around x=0 swings the bob downward without snapping away.
    assert!(sim.world.bodies.position[1].y < 4.0);
    assert!(sim.world.bodies.position[1].y > 2.0);
    assert!(sim.core_report().unwrap().islands >= 1);

    let urdf = r#"
<robot name="two_link">
  <link name="base">
    <inertial><mass value="0"/><inertia ixx="1" ixy="0" ixz="0" iyy="1" iyz="0" izz="1"/></inertial>
    <collision><geometry><box size="0.4 0.4 0.4"/></geometry></collision>
  </link>
  <link name="arm">
    <inertial><mass value="2.0"/><inertia ixx="0.1" ixy="0" ixz="0" iyy="0.1" iyz="0" izz="0.1"/></inertial>
    <collision><geometry><box size="1.0 0.2 0.2"/></geometry></collision>
  </link>
  <joint name="elbow" type="revolute">
    <parent link="base"/>
    <child link="arm"/>
    <origin xyz="0.5 0 0" rpy="0 0 0"/>
    <axis xyz="0 0 1"/>
    <limit lower="-1.5" upper="1.5" effort="50" velocity="5"/>
  </joint>
</robot>
"#;
    let imported = fusion_sim::import_urdf_xml(urdf).unwrap();
    assert_eq!(imported.name, "two_link");
    assert_eq!(imported.links.len(), 2);
    assert_eq!(imported.joints.len(), 1);
    assert_eq!(imported.joints[0].kind, JointKind::Hinge);
}

#[test]
fn runtime_impulse_works_on_core_and_wakes_sleeping_body() {
    let mut world = World::new();
    world.add(
        BodyDesc::cuboid(vec3(6.0, 0.5, 6.0))
            .at(vec3(0.0, -0.25, 0.0))
            .fixed(),
    );
    let box_idx = world
        .add(
            BodyDesc::cuboid(vec3(0.5, 0.5, 0.5))
                .at(vec3(0.0, 0.25, 0.0))
                .with_mass(2.0),
        )
        .index();

    let mut sim = Simulation::new(world);
    sim.solver_dt = 1.0 / 120.0;
    sim.core_settings.sleeping = true;
    sim.core_settings.settle_time = Some(0.05);
    sim.use_core_engine().unwrap();
    steps(&mut sim, 30);

    let y_rest = sim.world.bodies.position[box_idx].y;
    let grab_point = fusion_math::to_f32(sim.world.bodies.position[box_idx]);
    assert!(sim.apply_impulse_at(box_idx, vec3(0.0, 12.0, 0.0), grab_point));
    let report = sim.step();
    assert!(report.failure.is_none(), "{report:?}");
    steps(&mut sim, 10);
    assert!(
        sim.world.bodies.position[box_idx].y > y_rest + 0.05,
        "runtime impulse should wake and lift sleeping Core body (was {y_rest}, now {})",
        sim.world.bodies.position[box_idx].y
    );
}
