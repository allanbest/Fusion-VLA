use fusion_core::{BodyDesc, SolidDesc, SolidMaterial, SolidParams, World};
use fusion_math::{vec3, Vec3};
use fusion_project::{Project, SolverKind};
use fusion_sim::{state_hash, EngineKind};

#[test]
fn core_project_and_saved_run_roundtrip() {
    let mut world = World::new();
    world.add(BodyDesc::sphere(0.1).at(vec3(1.0, 2.0, 0.0)).with_mass(2.0));
    world.add_solid(
        &SolidDesc::block(Vec3::splat(0.2))
            .at(vec3(0.0, 2.0, 0.0))
            .with_params(SolidParams::at(0.05).of(SolidMaterial::JELLY)),
    );
    let mut project = Project::from_world(&world);
    project.solver.kind = SolverKind::Core;
    let mut sim = project.build().unwrap();
    for _ in 0..5 {
        assert!(sim.step().failure.is_none());
    }
    assert_eq!(Project::from_simulation(&sim).solver.kind, SolverKind::Core);
    let dir = std::env::temp_dir().join(format!("fusion-core-roundtrip-{}", std::process::id()));
    std::fs::create_dir_all(&dir).unwrap();
    let path = dir.join("core.fusion");
    project.save_run(&sim, &path).unwrap();
    let (loaded, mut resumed) = Project::load_run(&path).unwrap();
    assert_eq!(loaded.solver.kind, SolverKind::Core);
    assert_eq!(resumed.engine_kind(), EngineKind::Core);
    assert_eq!(state_hash(&sim), state_hash(&resumed));
    for _ in 0..10 {
        assert!(sim.step().failure.is_none());
        assert!(resumed.step().failure.is_none());
        assert_eq!(state_hash(&sim), state_hash(&resumed));
    }
    std::fs::remove_dir_all(dir).unwrap();
}

#[test]
fn engine_choice_keeps_its_file_spelling_and_core_needs_format_11() {
    let mut world = World::new();
    world.add(BodyDesc::sphere(0.1).at(vec3(0.0, 1.0, 0.0)));
    let mut project = Project::from_world(&world);

    // The Standard engine is still written the way every earlier file spells it.
    project.fusion.format = 3;
    let text = project.to_toml().unwrap();
    assert!(text.contains(r#"kind = "xpbd""#), "{text}");
    assert_eq!(
        Project::from_toml(&text).unwrap().solver.kind,
        SolverKind::Standard
    );

    // Fusion Core is format 11 even beside a feature that alone would write 10,
    // so a format-10 reader refuses it by version instead of by parse error.
    project.solver.kind = SolverKind::Core;
    project.solver.implicit_solids = true;
    let text = project.to_toml().unwrap();
    assert!(text.contains(r#"kind = "core""#), "{text}");
    let loaded = Project::from_toml(&text).unwrap();
    assert_eq!(loaded.fusion.format, 11);
    assert_eq!(loaded.solver.kind, SolverKind::Core);
}
