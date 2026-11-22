//! Format 2: a manifest that describes a run in progress, not a scene.
//!
//! PLAN.md §23 asks for checkpoint/resume — "basic hygiene for anything
//! expensive enough to crash mid-run" — and §22 says where the bytes go: "a
//! human-readable manifest ... referencing binary blobs for meshes, point data
//! and large initial fields". These tests are the pair working, and the
//! failure modes that make the pair worth having at all.

use fusion_core::{BodyDesc, ClothDesc, Material, SolidDesc, SolidMaterial, SolidParams, World};
use fusion_math::{vec3, Vec3};
use fusion_project::{Project, ProjectError};
use fusion_sim::state_hash;

fn scene() -> World {
    let mut world = World::new();
    world.add(
        BodyDesc::cuboid(vec3(8.0, 1.0, 8.0))
            .at(vec3(0.0, -0.5, 0.0))
            .with_material(Material::CONCRETE)
            .fixed(),
    );
    for i in 0..6 {
        world.add(
            BodyDesc::cuboid(Vec3::splat(0.3))
                .at(vec3(0.0, 0.2 + i as f32 * 0.32, 0.0))
                .with_material(Material::WOOD),
        );
    }
    world.add_solid(
        &SolidDesc::block(Vec3::splat(0.3))
            .at(vec3(1.2, 0.5, 0.0))
            .with_params(SolidParams::at(0.05).of(SolidMaterial::JELLY)),
    );
    world.add_cloth(
        &ClothDesc::new(8, 8, 0.1)
            .at(Vec3::new(-1.2, 1.2, 0.0))
            .with_mass(0.4),
    );
    world
}

fn dir(name: &str) -> std::path::PathBuf {
    let dir = std::env::temp_dir().join(format!("fusion-resume-{}-{name}", std::process::id()));
    std::fs::create_dir_all(&dir).expect("temp dir");
    dir
}

/// A run saved mid-flight comes back as the same state.
///
/// The whole point of format 2 in one assertion: the fingerprint of the
/// resumed simulation equals the fingerprint of the one that was saved, which
/// covers every body pose, every velocity, every material point's deformation
/// gradient and enthalpy, and the step index.
#[test]
fn a_run_saved_at_a_step_comes_back_at_that_step() {
    let dir = dir("midflight");
    let path = dir.join("run.fusion");

    let project = Project::from_world(&scene());
    let mut sim = project.build().expect("builds");
    for _ in 0..400 {
        sim.step();
    }
    let saved = state_hash(&sim);
    project.save_run(&sim, &path).expect("saves");

    // The blob really is beside the manifest, and named after it.
    assert!(
        dir.join("run.fusionstate").exists(),
        "no state file beside the manifest"
    );

    let (reloaded, resumed) = Project::load_run(&path).expect("loads");
    assert_eq!(reloaded.fusion.format, fusion_project::FORMAT_VERSION);
    let reference = reloaded.state.as_ref().expect("a state reference");
    assert_eq!(reference.step, 400);
    assert_eq!(resumed.step_index(), 400, "resumed at the wrong step");
    assert_eq!(
        state_hash(&resumed),
        saved,
        "resumed into a different state"
    );

    // And the manifest is still a manifest: the state is in the blob, not in
    // the text. A hundred and forty material points and sixty-four cloth
    // particles would be tens of kilobytes of TOML if they were inline.
    let text = std::fs::read_to_string(&path).expect("readable");
    assert!(
        text.len() < 4096,
        "the manifest grew to {} bytes — state has leaked into the text",
        text.len()
    );
    let blob = std::fs::metadata(dir.join("run.fusionstate"))
        .expect("blob")
        .len();
    assert!(
        blob > 4096,
        "the state blob is suspiciously small: {blob} bytes"
    );

    std::fs::remove_dir_all(&dir).ok();
}

/// A project with no state still builds at t = 0, so `resume` is safe to use
/// everywhere `build` is.
#[test]
fn a_scene_without_state_starts_at_the_beginning() {
    let dir = dir("plain");
    let path = dir.join("scene.fusion");
    let project = Project::from_world(&scene());
    project.save(&path).expect("saves");

    let (loaded, sim) = Project::load_run(&path).expect("loads");
    assert!(loaded.state.is_none());
    assert_eq!(sim.step_index(), 0);
    std::fs::remove_dir_all(&dir).ok();
}

/// A state file that has come apart from its manifest is refused, loudly.
///
/// This is the failure the hashes in [`fusion_project::StateRef`] exist for,
/// and it is worth a test because its symptom without them is not an error —
/// it is a run that starts, looks entirely plausible, and continues from a
/// state that never existed. Here the scene gains a body after the state was
/// written, which is exactly what someone tuning a scene does.
///
/// The error is about the *scene* rather than about the blob, even though the
/// blob is also now the wrong shape, because the scene check runs first: "you
/// edited this file after the checkpoint" is the sentence that says what to do
/// about it, and "the file holds 7 bodies and this simulation has 8" is the
/// same fact with the cause removed.
#[test]
fn a_state_file_that_no_longer_matches_its_scene_is_refused() {
    let dir = dir("stale");
    let path = dir.join("edited.fusion");

    let project = Project::from_world(&scene());
    let mut sim = project.build().expect("builds");
    for _ in 0..200 {
        sim.step();
    }
    project.save_run(&sim, &path).expect("saves");

    let mut edited = Project::load(&path).expect("loads");
    let extra = edited.objects[1].clone();
    edited.objects.push(extra);
    edited.save(&path).expect("saves");

    match Project::load_run(&path) {
        Err(ProjectError::StaleState { what, .. }) => assert_eq!(what, "scene"),
        Err(other) => panic!("expected a stale-state error, got {other}"),
        Ok(_) => panic!("a state file from a different scene was accepted"),
    }
    std::fs::remove_dir_all(&dir).ok();
}

/// And one that is the right shape but the wrong run is refused too.
///
/// The subtler half of the same failure: a scene edited in a way that does not
/// change any *count* — a body moved, a stiffness changed — decodes cleanly
/// and produces a state that is not the one that was saved. The size and the
/// shape both check out; only the fingerprint catches it.
#[test]
fn a_state_file_from_a_different_run_is_refused() {
    let dir = dir("wrong-run");
    let path = dir.join("moved.fusion");

    let project = Project::from_world(&scene());
    let mut sim = project.build().expect("builds");
    for _ in 0..200 {
        sim.step();
    }
    project.save_run(&sim, &path).expect("saves");

    let mut edited = Project::load(&path).expect("loads");
    edited.world.gravity[1] = -3.0;
    edited.save(&path).expect("saves");

    // Gravity is not in the state blob — it is scene — so the blob decodes
    // perfectly, fingerprints correctly, and the run it produces is a
    // different one. Only the hash of the manifest text catches this.
    match Project::load_run(&path) {
        Err(ProjectError::StaleState { what, .. }) => assert_eq!(what, "scene"),
        Err(other) => panic!("expected a stale-state error, got {other}"),
        Ok(_) => panic!("a state file from a different run was accepted"),
    }
    std::fs::remove_dir_all(&dir).ok();
}
