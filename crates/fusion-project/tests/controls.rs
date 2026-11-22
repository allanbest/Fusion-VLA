use fusion_core::{BodyDesc, World};
use fusion_math::Vec3;
use fusion_project::{BodyKindEntry, Project};
use fusion_sim::{control::*, state_hash};
fn project() -> Project {
    let mut w = World::new();
    w.gravity = Vec3::ZERO;
    w.add(BodyDesc::sphere(0.5).with_mass(2.0));
    let mut p = Project::from_world(&w);
    p.inputs = movement_actions();
    p.inputs[0].record(0, 1.0);
    p.inputs[0].record(20, 0.0);
    p.objects[0].components = Components {
        controller: Some(Default::default()),
        probe: true,
        ..Default::default()
    };
    p
}
#[test]
fn project_and_simulation_round_trip_controls_and_scripts() {
    let mut p = project();
    p.objects[0].script = Some("self.force.y = input.move_x".into());
    let sim = p.build().unwrap();
    let captured = Project::from_simulation(&sim);
    assert_eq!(captured.inputs, p.inputs);
    assert_eq!(captured.objects[0].components, p.objects[0].components);
    assert_eq!(captured.objects[0].script, p.objects[0].script);
    let mut first = p.build().unwrap();
    let mut next = Project::from_toml(&captured.to_toml().unwrap())
        .unwrap()
        .build()
        .unwrap();
    for _ in 0..60 {
        first.step();
        next.step();
        assert_eq!(state_hash(&first), state_hash(&next));
    }
}
#[test]
fn invalid_component_references_and_static_force_are_rejected() {
    let mut p = project();
    p.inputs.clear();
    assert!(p.build().is_err());
    p.inputs = movement_actions();
    p.objects[0].body_type = BodyKindEntry::Static;
    assert!(p.build().is_err());
    p.objects[0].components.controller = None;
    assert!(p.build().is_ok(), "a static-body probe is valid");
}
#[test]
fn version_three_saved_runs_keep_their_manifest_fingerprint() {
    let mut p = project();
    p.inputs.clear();
    p.objects[0].components = Default::default();
    p.fusion.format = 3;
    let mut sim = p.build().unwrap();
    for _ in 0..12 {
        sim.step();
    }
    let dir = std::env::temp_dir().join(format!("fusion-controls-resume-{}", std::process::id()));
    std::fs::create_dir_all(&dir).unwrap();
    let path = dir.join("v3.fusion");
    p.save_run(&sim, &path).unwrap();
    let (loaded, restored) = Project::load_run(&path).unwrap();
    assert_eq!(loaded.fusion.format, 3);
    assert_eq!(state_hash(&sim), state_hash(&restored));
    // Adding behaviours upgrades the format even when the in-memory provenance is old.
    p.inputs = movement_actions();
    assert!(p.to_toml().unwrap().contains("format = 4"));
    std::fs::remove_dir_all(dir).unwrap();
}
#[test]
fn controlled_saved_run_continues_on_the_recorded_tape() {
    let p = project();
    let mut sim = p.build().unwrap();
    for _ in 0..10 {
        sim.step();
    }
    let dir = std::env::temp_dir().join(format!("fusion-control-state-{}", std::process::id()));
    std::fs::create_dir_all(&dir).unwrap();
    let path = dir.join("v4.fusion");
    p.save_run(&sim, &path).unwrap();
    let (_, mut resumed) = Project::load_run(&path).unwrap();
    for _ in 0..40 {
        sim.step();
        resumed.step();
        assert_eq!(state_hash(&sim), state_hash(&resumed));
    }
    std::fs::remove_dir_all(dir).unwrap();
}

#[test]
fn shipped_input_lab_builds_and_measures_inverse_mass_acceleration() {
    let p = Project::from_toml(include_str!("data/controls.fusion")).unwrap();
    let mut sim = p.build().unwrap();
    sim.step();
    let light = sim.world.bodies.linear_velocity[0].x;
    let heavy = sim.world.bodies.linear_velocity[1].x;
    assert!((light - 2.0 * heavy).abs() < 1e-5);
    assert_eq!(sim.probe_samples().len(), 2);
}

#[test]
fn a_format_4_scene_gains_the_take_it_never_had() {
    let p = project();
    let text = p.to_toml().unwrap();
    // Written as 4: one take is still readable by an editor that has none.
    assert!(text.contains("format = 4"));
    let loaded = Project::from_toml(&text).unwrap();
    assert_eq!(loaded.takes.len(), 1);
    assert_eq!(loaded.takes[0].name, "Take 1");
    assert_eq!(loaded.active_take, Some(0));
    // The archive agrees with the live take rather than merely existing.
    assert_eq!(loaded.takes[0].streams[0].events, loaded.inputs[0].events);
    assert_eq!(loaded.inputs, p.inputs);
}

#[test]
fn the_archive_is_outside_the_scene_fingerprint() {
    let p = project();
    let mut sim = p.build().unwrap();
    for _ in 0..8 {
        sim.step();
    }
    let dir = std::env::temp_dir().join(format!("fusion-take-hash-{}", std::process::id()));
    std::fs::create_dir_all(&dir).unwrap();
    let path = dir.join("run.fusion");
    p.save_run(&sim, &path).unwrap();

    // Renaming and adding recordings is authoring, not physics: a checkpoint
    // saved before them still belongs to this scene.
    let mut edited = Project::from_toml(&std::fs::read_to_string(&path).unwrap()).unwrap();
    edited.takes[0].name = "first attempt".into();
    edited.takes.push(Take::capture("second", &edited.inputs));
    let text = edited.to_toml().unwrap();
    assert!(text.contains("format = 5"));
    std::fs::write(&path, &text).unwrap();

    let (_, resumed) = Project::load_run(&path).unwrap();
    assert_eq!(state_hash(&sim), state_hash(&resumed));
    std::fs::remove_dir_all(dir).unwrap();
}

#[test]
fn a_take_naming_an_action_the_scene_lost_is_not_an_error() {
    let mut p = project();
    let mut take = Take::capture("recorded", &p.inputs);
    take.streams.push(TakeStream {
        action: "removed_since".into(),
        events: vec![InputEvent {
            step: 0,
            value: 1.0,
        }],
    });
    p.takes = vec![take.clone()];
    p.active_take = Some(0);
    p.build().unwrap();
    // Applying it drives what still exists and silently drops what does not.
    let mut actions = movement_actions();
    take.apply(&mut actions);
    assert_eq!(actions[0].value(0), 1.0);
    assert!(actions[1].events.is_empty());
}

#[test]
fn takes_need_unique_names_and_ordered_events() {
    let p = project();
    let good = Take::capture("a", &p.inputs);
    assert!(validate_takes(std::slice::from_ref(&good)).is_ok());
    let mut same_name = good.clone();
    same_name.name = "a".into();
    assert!(validate_takes(&[good.clone(), same_name]).is_err());
    let mut blank = good.clone();
    blank.name = "  ".into();
    assert!(validate_takes(&[blank]).is_err());
    let mut backwards = good.clone();
    backwards.streams[0].events = vec![
        InputEvent {
            step: 5,
            value: 0.0,
        },
        InputEvent {
            step: 5,
            value: 1.0,
        },
    ];
    assert!(validate_takes(&[backwards]).is_err());
    let mut twice = good;
    twice.streams.push(twice.streams[0].clone());
    assert!(validate_takes(&[twice]).is_err());
}

#[test]
fn take_names_are_numbered_predictably() {
    let p = project();
    let first = Take::capture(unused_take_name("Take", &[]), &p.inputs);
    assert_eq!(first.name, "Take 1");
    let second = unused_take_name("Take", std::slice::from_ref(&first));
    assert_eq!(second, "Take 2");
}

/// A one-take scene is written as format 4 *with* a `[[take]]` block, on the
/// grounds that a format-4 editor ignores what it does not know. That is only
/// true while the schema does not deny unknown fields — which nothing enforces
/// except this test. Adding `deny_unknown_fields` anywhere on the way down
/// must come with a format bump instead.
#[test]
fn an_older_reader_ignores_what_it_does_not_know() {
    let text = project().to_toml().unwrap();
    let from_the_future = format!("{text}\n[[not_invented_yet]]\nanything = 1\n");
    let loaded = Project::from_toml(&from_the_future).unwrap();
    assert_eq!(loaded.inputs, project().inputs);
}
