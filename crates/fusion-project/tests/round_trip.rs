//! The project format, against the promise PLAN.md §22 makes for it:
//! "a `.fusion` file plus a step number reproduces anything exactly."
//!
//! Every test here is a version of that sentence. A format that round-trips
//! its own fields and produces a different simulation has failed at the only
//! thing it was for.

use fusion_core::{
    BodyDesc, ClothDesc, FluidDesc, FluidParams, JointKind, Material, SoftBodyDesc, SolidDesc,
    SolidMaterial, SolidParams, World,
};
use fusion_math::{vec3, Quat, Vec3};
use fusion_project::{Project, ProjectError};
use fusion_sim::{state_hash, Simulation};

/// A scene using every part of the format: bodies of both kinds, a named
/// material and an unnamed one, a joint, cloth, a soft body, and a fluid.
fn everything() -> World {
    let mut world = World::new()
        .with_seed(0x5eed_1234)
        .with_gravity(vec3(0.0, -9.81, 0.2));
    world.linear_damping = 0.01;
    world.angular_damping = 0.07;

    world.add(
        BodyDesc::cuboid(vec3(6.0, 0.4, 6.0))
            .at(vec3(0.0, -0.2, 0.0))
            .with_material(Material::CONCRETE)
            .fixed(),
    );
    for i in 0..4 {
        world.add(
            BodyDesc::cuboid(Vec3::splat(0.3))
                .at(vec3(1.1, 0.16 + i as f32 * 0.31, 0.0))
                .rotated(Quat::from_rotation_y(0.21 * i as f32))
                .with_material(Material::WOOD),
        );
    }
    // A material that is not in the library, so the fallback naming is used.
    world.add(
        BodyDesc::sphere(0.2)
            .at(vec3(-1.6, 1.0, 0.3))
            .with_velocity(vec3(0.4, 0.0, -0.2))
            .with_angular_velocity(vec3(0.0, 3.0, 0.0))
            .with_material(Material::new(432.0, 0.31, 0.22))
            .with_mass(7.5),
    );

    let anchor = world.add(BodyDesc::sphere(0.08).at(vec3(-0.8, 1.8, 0.0)).fixed());
    let bob = world.add(BodyDesc::sphere(0.14).at(vec3(-0.8, 1.2, 0.0)));
    world.connect(anchor, bob, vec3(-0.8, 1.8, 0.0), JointKind::Ball);

    world.add_cloth(
        &ClothDesc::new(8, 8, 0.09)
            .at(vec3(0.0, 1.6, 1.4))
            .with_mass(0.6),
    );
    world.add_soft_body(&SoftBodyDesc::new(Vec3::splat(0.4), 3).at(vec3(2.0, 1.2, -1.0)));

    // Walls, then water in them.
    for (dx, dz) in [(1.0f32, 0.0f32), (-1.0, 0.0), (0.0, 1.0), (0.0, -1.0)] {
        let size = if dx != 0.0 {
            vec3(0.2, 1.0, 1.2)
        } else {
            vec3(1.2, 1.0, 0.2)
        };
        world.add(
            BodyDesc::cuboid(size)
                .at(vec3(dx * 0.7 - 1.2, 0.5, dz * 0.7 - 1.6))
                .fixed(),
        );
    }
    world.add_fluid(
        &FluidDesc::block(vec3(0.5, 0.4, 0.5))
            .at(vec3(-1.2, 0.5, -1.6))
            .with_params(FluidParams::water_at(0.07))
            .with_velocity(vec3(0.1, 0.0, 0.0)),
    );
    // And a body of continuum, which is the v0.8 half of the format: an
    // elastoplastic one, so the round trip has to carry the yield parameters
    // and not merely the stiffness.
    world.add_solid(
        &SolidDesc::block(Vec3::splat(0.3))
            .at(vec3(1.4, 0.9, 1.1))
            .with_params(SolidParams::at(0.06).of(SolidMaterial::SNOW))
            .with_velocity(vec3(0.0, 0.0, -0.2)),
    );
    world
}

fn tuned_simulation() -> Simulation {
    let mut sim = Simulation::new(everything());
    // Move every knob off its default, so a field the writer forgot shows up
    // as a difference rather than coincidentally matching.
    sim.solver_dt = 1.0 / 180.0;
    sim.output_rate = 48.0;
    sim.max_speculative_distance = 1.25;
    sim.solver.substeps = 6;
    sim.solver.iterations = 5;
    sim.solver.velocity_iterations = 3;
    sim.solver.contact_compliance = 1e-7;
    sim.solver.penetration_slop = 7e-4;
    sim.solver.correction_rate = 0.7;
    sim.solver.max_depenetration_speed = 1.5;
    sim.solver.restitution_threshold = 0.4;
    sim.solver.sliding_threshold = 0.06;
    sim.solver.speculative = true;
    sim.solver.static_friction = true;
    sim.solver.budget_depenetration = true;
    sim.solver.gyroscopic = true;
    sim.solver.cloth_damping = 0.35;
    sim.solver.cloth_friction = 0.42;
    sim.solver.fluid_substeps = 3;
    sim.solver.fluid_iterations = 3;
    sim.solver.fluid_relaxation = 0.2;
    sim.solver.fluid_friction = 0.08;
    sim.solver.max_fluid_speed = 22.0;
    sim.sleep.linear_threshold = 0.03;
    sim.sleep.time_to_sleep = 0.4;
    sim.air = fusion_solver::AirModel::with_wind(vec3(2.0, 0.0, -1.0));
    sim
}

#[test]
fn a_project_reproduces_the_run_it_describes() {
    // The whole promise of §22, as one assertion.
    let mut original = tuned_simulation();
    let project = Project::from_simulation(&original);

    let text = project.to_toml().unwrap();
    let reloaded = Project::from_toml(&text).unwrap();
    let mut rebuilt = reloaded.build().unwrap();

    assert_eq!(state_hash(&original), state_hash(&rebuilt), "t = 0 differs");
    for step in 0..300 {
        original.step();
        rebuilt.step();
        assert_eq!(
            state_hash(&original),
            state_hash(&rebuilt),
            "diverged at step {step}"
        );
    }
}

#[test]
fn verify_says_so_itself() {
    let project = Project::from_simulation(&tuned_simulation());
    let report = project.verify(200).unwrap();
    assert!(report.reproduces(), "{report}");
    assert!(report.identical_text);
    assert_eq!(report.original, report.round_tripped);
    assert!(report.bytes > 0);
    assert!(format!("{report}").contains("reproduces"));
}

#[test]
fn a_file_plus_a_step_number_identifies_one_state() {
    // §22 verbatim. Two independent builds of the same file, stepped to the
    // same number, are the same state — and a different step number is not.
    let project = Project::from_simulation(&tuned_simulation());
    let at_120 = project.fingerprint_at(120).unwrap();
    let again = project.fingerprint_at(120).unwrap();
    let at_121 = project.fingerprint_at(121).unwrap();
    assert_eq!(at_120, again);
    assert_ne!(at_120, at_121);
}

#[test]
fn every_configured_field_survives_the_file() {
    let sim = tuned_simulation();
    let project = Project::from_simulation(&sim);
    let reloaded = Project::from_toml(&project.to_toml().unwrap()).unwrap();
    assert_eq!(project, reloaded);

    let rebuilt = reloaded.build().unwrap();
    assert_eq!(rebuilt.solver_dt, sim.solver_dt);
    assert_eq!(rebuilt.output_rate, sim.output_rate);
    assert_eq!(
        rebuilt.max_speculative_distance,
        sim.max_speculative_distance
    );
    assert_eq!(rebuilt.solver.substeps, sim.solver.substeps);
    assert_eq!(rebuilt.solver.iterations, sim.solver.iterations);
    assert_eq!(rebuilt.solver.fluid_substeps, sim.solver.fluid_substeps);
    assert_eq!(rebuilt.solver.fluid_relaxation, sim.solver.fluid_relaxation);
    assert_eq!(rebuilt.sleep.time_to_sleep, sim.sleep.time_to_sleep);
    assert_eq!(rebuilt.air.wind, sim.air.wind);
    assert_eq!(rebuilt.world.seed, sim.world.seed);
    assert_eq!(rebuilt.world.gravity, sim.world.gravity);
    assert_eq!(rebuilt.world.linear_damping, sim.world.linear_damping);
}

#[test]
fn the_scene_is_stored_and_not_the_state() {
    // PLAN.md §16's split, as a size claim: a fluid of thousands of particles
    // is a handful of numbers, because `fill` regenerates the lattice.
    let mut world = World::new();
    world.add_fluid(
        &FluidDesc::block(Vec3::splat(1.2))
            .with_params(FluidParams::water_at(0.04))
            .at(vec3(0.0, 2.0, 0.0)),
    );
    let particles = world.fluid_particle_count();
    assert!(particles > 20_000, "only {particles} particles");

    let project = Project::from_world(&world);
    let text = project.to_toml().unwrap();
    assert!(
        text.len() < 4_000,
        "{particles} particles took {} bytes to describe",
        text.len()
    );
    assert_eq!(
        project.build_world().unwrap().fluid_particle_count(),
        particles
    );
}

#[test]
fn saving_and_loading_a_file_works() {
    let dir = std::env::temp_dir().join(format!("fusion-project-test-{}", std::process::id()));
    std::fs::create_dir_all(&dir).unwrap();
    let path = dir.join("scene.fusion");

    let project = Project::from_simulation(&tuned_simulation())
        .with_note("a dam break that loses a particle at step 4000")
        .with_step_of_interest(4000);
    project.save(&path).unwrap();

    let loaded = Project::load(&path).unwrap();
    assert_eq!(loaded, project);
    assert_eq!(loaded.metadata.step_of_interest, Some(4000));
    assert!(loaded.metadata.note.unwrap().contains("dam break"));

    let text = std::fs::read_to_string(&path).unwrap();
    assert!(text.starts_with("# A Fusion project"));
    std::fs::remove_dir_all(&dir).ok();
}

#[test]
fn materials_are_named_and_shared() {
    // Four wooden boxes should produce one `[[material]]` entry called "wood",
    // not four copies of three numbers.
    let project = Project::from_world(&everything());
    let wood = project
        .materials
        .iter()
        .filter(|m| m.name == "wood")
        .count();
    assert_eq!(wood, 1);
    assert_eq!(
        project
            .objects
            .iter()
            .filter(|o| o.material == "wood")
            .count(),
        4
    );
    assert!(project.materials.iter().any(|m| m.name == "concrete"));
    // The one that is not in the library still round-trips, under a made-up
    // name rather than being silently snapped to the nearest library entry.
    let odd = project
        .materials
        .iter()
        .find(|m| m.density == 432.0)
        .expect("the unnamed material was lost");
    assert!(odd.name.starts_with("material_"));
    assert_eq!(odd.friction, 0.31);
}

#[test]
fn a_body_referring_to_a_missing_material_is_rejected() {
    let mut project = Project::from_world(&everything());
    project.objects[0].material = "unobtanium".into();
    let error = project.build_world().unwrap_err();
    assert!(matches!(error, ProjectError::Unresolved(_)));
    assert!(format!("{error}").contains("unobtanium"));
}

#[test]
fn a_joint_referring_to_a_missing_body_is_rejected() {
    let mut project = Project::from_world(&everything());
    project.joints[0].body_b = 9999;
    let error = project.build_world().unwrap_err();
    assert!(matches!(error, ProjectError::Unresolved(_)));
}

#[test]
fn a_newer_format_is_refused_rather_than_misread() {
    // The failure mode this prevents is the quiet one: reading a file whose
    // fields mean something else now, running it, and reporting the result as
    // a reproduction.
    let project = Project::from_simulation(&tuned_simulation());
    let text = project.to_toml().unwrap().replace(
        &format!("format = {}", fusion_project::FORMAT_VERSION),
        &format!("format = {}", fusion_project::FORMAT_VERSION + 1),
    );
    let error = Project::from_toml(&text).unwrap_err();
    assert!(matches!(error, ProjectError::UnsupportedFormat { .. }));
    assert!(format!("{error}").contains("format version"));
}

#[test]
fn an_older_file_missing_newer_fields_still_loads() {
    // Forwards compatibility in the other direction: every optional field
    // defaults, so a minimal hand-written file is a valid project.
    let text = r#"
[fusion]
format = 1
version = "0.1.0"

[[material]]
name = "wood"
density = 700.0
friction = 0.45
restitution = 0.3

[[body]]
shape = { kind = "cuboid", size = [10.0, 0.4, 10.0] }
position = [0.0, -0.2, 0.0]
body_type = "static"
material = "wood"

[[body]]
shape = { kind = "sphere", radius = 0.25 }
position = [0.0, 3.0, 0.0]
material = "wood"
"#;
    let project = Project::from_toml(text).unwrap();
    // Read as format 1, folded into objects on the way in (§35).
    assert!(
        project.bodies.is_empty(),
        "the old list was not folded away"
    );
    assert_eq!(project.objects.len(), 2);
    let mut sim = project.build().unwrap();
    // Defaults were filled in, and the thing runs.
    assert_eq!(sim.solver_dt, 1.0 / 240.0);
    sim.run_frames(60, |_, _| {});
    assert!(sim.is_stable());
    let ball = sim.world.bodies.pose(1).translation.y;
    assert!(ball < 3.0 && ball > 0.0, "the ball ended at y = {ball}");
}

#[test]
fn a_hand_written_file_is_readable_and_editable() {
    // The property that makes a project file useful as a bug report: a person
    // can open it, see what the scene is, and change one number.
    let project = Project::from_world(&everything());
    let text = project.to_toml().unwrap();
    assert!(text.contains("[[object]]"));
    assert!(text.contains("[[material]]"));
    assert!(text.contains("[[fluid]]"));
    assert!(text.contains("name = \"wood\""));
    // Full size rather than half-extents, because that is what someone typing
    // "a one metre cube" expects to write.
    assert!(text.contains("size = [6.0, 0.4, 6.0]"));

    let edited = text.replace("size = [6.0, 0.4, 6.0]", "size = [8.0, 0.4, 8.0]");
    let reloaded = Project::from_toml(&edited).unwrap();
    let world = reloaded.build_world().unwrap();
    match &world.scene()[0].shape {
        fusion_core::Shape::Cuboid { half_extents } => assert_eq!(half_extents.x, 4.0),
        other => panic!("expected a cuboid, got {other:?}"),
    }
}

#[test]
fn resetting_a_built_world_returns_it_to_the_file() {
    // `reset` and "build from the project" are two routes to t = 0 and must
    // agree — including for cloth and soft bodies, which until v0.4 were built
    // once and never rebuilt.
    let project = Project::from_simulation(&tuned_simulation());
    let mut sim = project.build().unwrap();
    let start = state_hash(&sim);

    sim.run_frames(60, |_, _| {});
    assert_ne!(state_hash(&sim), start);

    sim.reset();
    assert_eq!(state_hash(&sim), start, "reset did not return to t = 0");

    // And a freshly built one agrees with the reset one.
    let fresh = project.build().unwrap();
    assert_eq!(state_hash(&fresh), start);
}

#[test]
fn the_provenance_records_what_wrote_the_file() {
    let project = Project::from_world(&World::new());
    assert_eq!(project.fusion.format, fusion_project::FORMAT_VERSION);
    assert!(!project.fusion.version.is_empty());
    // No solver hash in an ordinary build, and the field is simply absent
    // rather than present and meaningless.
    let text = project.to_toml().unwrap();
    if project.fusion.solver_hash.is_none() {
        assert!(!text.contains("solver_hash"));
    }
}

#[test]
fn an_empty_world_round_trips() {
    let project = Project::from_world(&World::new());
    let reloaded = Project::from_toml(&project.to_toml().unwrap()).unwrap();
    assert_eq!(project, reloaded);
    assert!(reloaded.build().is_ok());
}

#[test]
fn nonsense_is_an_error_and_not_a_panic() {
    assert!(Project::from_toml("this is not toml {{{").is_err());
    assert!(Project::from_toml("").is_err());
    assert!(matches!(
        Project::load("/nonexistent/path/scene.fusion"),
        Err(ProjectError::Io(_))
    ));
}

/// A solid's material survives the file, including the law it is solved under.
///
/// The failure this guards is quiet: a constitutive model that round-trips as
/// its default turns snow into jelly, and the scene still loads, still builds
/// and still runs. Nothing is reported. The only symptom is that the snowball
/// bounces.
#[test]
fn a_solids_constitutive_law_survives_the_file() {
    let mut world = World::new();
    for material in SolidMaterial::LIBRARY.map(|(_, m)| m) {
        world.add_solid(
            &SolidDesc::block(Vec3::splat(0.2))
                .at(vec3(0.0, 1.0, 0.0))
                .with_params(SolidParams::at(0.05).of(material)),
        );
    }
    let project = Project::from_world(&world);
    let text = project.to_toml().unwrap();
    let rebuilt = Project::from_toml(&text).unwrap().build_world().unwrap();

    assert_eq!(rebuilt.solids.len(), SolidMaterial::LIBRARY.len());
    for (index, (name, expected)) in SolidMaterial::LIBRARY.into_iter().enumerate() {
        let got = rebuilt.solids[index].params.material;
        assert_eq!(
            got.model, expected.model,
            "{name} came back a different law"
        );
        assert!(
            (got.youngs_modulus - expected.youngs_modulus).abs() < 1.0,
            "{name} came back at {} Pa, not {}",
            got.youngs_modulus,
            expected.youngs_modulus
        );
        assert!((got.density - expected.density).abs() < 1e-3);
        assert!((got.poissons_ratio - expected.poissons_ratio).abs() < 1e-6);
    }
}

/// A material's failure strains survive the file, so a scene that breaks still
/// breaks after being saved.
///
/// The quiet failure this guards is the same shape as the constitutive one
/// above: a fracture law that round-trips as its default turns porcelain into
/// something unbreakable, and nothing reports it. The scene loads, builds and
/// runs; the plate simply stops shattering.
#[test]
fn a_solids_failure_strains_survive_the_file() {
    let mut world = World::new();
    world.add_solid(
        &SolidDesc::block(Vec3::splat(0.2))
            .with_params(SolidParams::at(0.05).of(SolidMaterial::PORCELAIN)),
    );
    world.add_solid(
        &SolidDesc::block(Vec3::splat(0.2))
            .at(vec3(0.5, 0.0, 0.0))
            .with_params(SolidParams::at(0.05).of(SolidMaterial::JELLY)),
    );

    let text = Project::from_world(&world).to_toml().unwrap();
    let rebuilt = Project::from_toml(&text).unwrap().build_world().unwrap();

    let brittle = rebuilt.solids[0].params.material.fracture;
    assert!(brittle.breaks(), "porcelain came back unbreakable");
    assert!(
        (brittle.onset - SolidMaterial::PORCELAIN.fracture.onset).abs() < 1e-6,
        "onset came back as {}",
        brittle.onset
    );
    assert!(
        (brittle.failure - SolidMaterial::PORCELAIN.fracture.failure).abs() < 1e-6,
        "failure came back as {}",
        brittle.failure
    );

    // And a tough material comes back tough rather than with a zero onset,
    // which would be a material that breaks the instant it exists.
    assert!(!rebuilt.solids[1].params.material.fracture.breaks());

    // The thermal half, on the same trip.
    let alloy = rebuilt.solids[0].params.material.thermal;
    assert_eq!(alloy.melts(), SolidMaterial::PORCELAIN.thermal.melts());
}

#[test]
fn unified_continuum_reads_both_spellings_and_writes_fusion() {
    let mut project = Project::from_world(&World::new());
    project.solver.continuum = fusion_project::ContinuumEntry::Unified;
    let text = project.to_toml().unwrap();
    assert!(text.contains("continuum = \"fusion\""));
    let legacy = text.replace("continuum = \"fusion\"", "continuum = \"unified\"");
    let loaded = Project::from_toml(&legacy).unwrap();
    assert_eq!(
        loaded.solver.continuum,
        fusion_project::ContinuumEntry::Unified
    );
    assert!(loaded.to_toml().unwrap().contains("continuum = \"fusion\""));
}

/// The fluid model survives a save, reaches the simulation, and is a format
/// bump only when it is the new one: a scene that never chose writes nothing.
#[test]
fn the_fluid_model_round_trips_and_only_the_new_one_bumps_the_format() {
    use fusion_project::FluidModelEntry;
    let mut project = Project::from_world(&World::new());
    project.solver.continuum = fusion_project::ContinuumEntry::Unified;
    let plain = project.to_toml().unwrap();
    assert!(!plain.contains("fluid_model"), "the default is not written");
    assert!(
        plain.contains("format = 8"),
        "and stays readable by format 8"
    );

    project.solver.fluid_model = FluidModelEntry::Incompressible;
    let text = project.to_toml().unwrap();
    assert!(text.contains("fluid_model = \"incompressible\""));
    assert!(text.contains("format = 10"));
    let loaded = Project::from_toml(&text).unwrap();
    assert_eq!(loaded.solver.fluid_model, FluidModelEntry::Incompressible);
    let sim = loaded.build().unwrap();
    assert_eq!(
        sim.unified.fluid_model,
        fusion_sim::FluidModel::Incompressible
    );
    let captured = Project::from_simulation(&sim);
    assert_eq!(captured.solver.fluid_model, FluidModelEntry::Incompressible);

    // Implicit stiff solids ride the same format.
    let mut solids = Project::from_world(&World::new());
    solids.solver.continuum = fusion_project::ContinuumEntry::Unified;
    solids.solver.implicit_solids = true;
    let text = solids.to_toml().unwrap();
    assert!(text.contains("implicit_solids = true") && text.contains("format = 10"));
    let sim = Project::from_toml(&text).unwrap().build().unwrap();
    assert!(sim.unified.implicit_solids);
    assert!(Project::from_simulation(&sim).solver.implicit_solids);
}

#[test]
fn legacy_body_entries_keep_the_full_named_material() {
    use fusion_project::{body_desc, BodyEntry, BodyKindEntry, MaterialEntry, ShapeEntry};
    let material = MaterialEntry::rigid("steel", &Material::STEEL);
    let body = BodyEntry {
        shape: ShapeEntry::Sphere { radius: 0.1 },
        position: [0.0; 3],
        orientation: [0.0, 0.0, 0.0, 1.0],
        linear_velocity: [0.0; 3],
        angular_velocity: [0.0; 3],
        body_type: BodyKindEntry::Dynamic,
        material: "steel".into(),
        mass: None,
        script: None,
    };
    let decoded = body_desc(&body, &[material]).unwrap();
    assert_eq!(
        decoded.material,
        Material::STEEL,
        "legacy bodies must retain strength and thermal properties"
    );
}

/// Choosing the Unified continuum for a scene turns on what it runs best with;
/// a file that names it without them is left as written.
#[test]
fn choosing_the_fusion_solver_turns_on_its_fluid_and_implicit_solids() {
    use fusion_project::{ContinuumEntry, FluidModelEntry};
    let mut project = Project::from_world(&World::new());
    project.solver.choose_unified();
    assert_eq!(project.solver.continuum, ContinuumEntry::Unified);
    assert_eq!(project.solver.fluid_model, FluidModelEntry::Incompressible);
    assert!(project.solver.implicit_solids);

    // Already on it: a choice the person made since is kept.
    project.solver.implicit_solids = false;
    project.solver.choose_unified();
    assert!(!project.solver.implicit_solids);

    // And a file written before these existed reads as it ran.
    let mut old = Project::from_world(&World::new());
    old.solver.continuum = ContinuumEntry::Unified;
    let read = Project::from_toml(&old.to_toml().unwrap()).unwrap();
    assert_eq!(read.solver.fluid_model, FluidModelEntry::WeaklyCompressible);
    assert!(!read.solver.implicit_solids);
}
