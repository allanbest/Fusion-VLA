//! Format 3: one object, and the material that decides how it is solved.
//!
//! PLAN.md §35. The claim being tested is not that the file is prettier — it
//! is that a cube and a jelly cube are the same object with one field
//! different, and that saying so changed no answer.

use fusion_core::{BodyDesc, Material, SolidDesc, SolidMaterial, SolidParams, World};
use fusion_math::{vec3, Vec3};
use fusion_project::{Project, ProjectError, Representation};
use fusion_sim::state_hash;

/// Rigid and deformable in one scene, interleaved so that object index and
/// body index cannot accidentally agree.
fn mixed() -> World {
    let mut world = World::new();
    world.add(
        BodyDesc::cuboid(vec3(8.0, 1.0, 8.0))
            .at(vec3(0.0, -0.5, 0.0))
            .with_material(Material::CONCRETE)
            .fixed(),
    );
    world.add_solid(
        &SolidDesc::block(Vec3::splat(0.3))
            .at(vec3(0.6, 0.5, 0.0))
            .with_params(SolidParams::at(0.05).of(SolidMaterial::JELLY)),
    );
    world.add(
        BodyDesc::cuboid(Vec3::splat(0.3))
            .at(vec3(0.0, 0.5, 0.0))
            .with_material(Material::WOOD),
    );
    world.add_solid(
        &SolidDesc::block(Vec3::splat(0.25))
            .at(vec3(-0.6, 0.5, 0.0))
            .with_params(SolidParams::at(0.05).of(SolidMaterial::SNOW)),
    );
    world
}

/// A box and a jelly block are the same entry, differing in one field.
#[test]
fn the_difference_between_a_box_and_a_jelly_block_is_one_line() {
    let project = Project::from_world(&mixed());
    assert_eq!(project.objects.len(), 4, "four objects, one list");
    assert!(project.bodies.is_empty() && project.solids.is_empty());

    let text = project.to_toml().unwrap();
    assert!(text.contains("[[object]]"));
    assert!(!text.contains("[[body]]"), "the old lists are not written");
    assert!(!text.contains("[[solid]]"));

    // The wooden box and the jelly block have the same shape and the same
    // size. Everything that makes one of them a continuum is in its material.
    let wood = project
        .objects
        .iter()
        .find(|o| o.material == "wood")
        .expect("a wooden object");
    let jelly = project
        .objects
        .iter()
        .find(|o| o.material == "jelly")
        .expect("a jelly object");
    assert_eq!(wood.shape, jelly.shape);

    let representation = |name: &str| {
        project
            .materials
            .iter()
            .find(|m| m.name == name)
            .unwrap()
            .representation
    };
    assert_eq!(representation("wood"), Representation::Rigid);
    assert_eq!(representation("jelly"), Representation::Deformable);
}

/// Switching an object's material is the whole of turning a box into jelly.
///
/// The behavioural version of the test above: the same scene, one string
/// changed, and what comes out is a continuum where there was a rigid body —
/// with no other edit anywhere, and nothing destroyed and rebuilt.
#[test]
fn changing_the_material_changes_the_representation() {
    let mut project = Project::from_world(&mixed());
    let index = project
        .objects
        .iter()
        .position(|o| o.material == "wood")
        .expect("a wooden object");

    let before = project.build_world().unwrap();
    assert_eq!(before.len(), 2, "two rigid bodies to start");
    assert_eq!(before.solids.len(), 2);

    project.objects[index].material = "jelly".into();
    let after = project.build_world().unwrap();
    assert_eq!(after.len(), 1, "the box should have stopped being a body");
    assert_eq!(after.solids.len(), 3, "and started being a continuum");

    // And the resolution it gets is derived from its own size rather than
    // inherited from whatever the file happened to say, because a rigid object
    // never had one.
    let points = after.solids[2].len();
    assert!(
        points > 100 && points < 2000,
        "a 30 cm block resolved to {points} material points"
    );
}

/// A format-2 file loads, and produces the **same run** it always did.
///
/// The migration test, and the only one that could have caught a migration
/// that produced the same picture in a different order: `state_hash` after two
/// hundred steps, against the same scene built directly. Body order decides
/// the sequence the Gauss-Seidel contact solve visits manifolds in, so a fold
/// that put the solids first would still look right and would not be the same
/// simulation.
#[test]
fn an_old_file_produces_the_same_run() {
    // Build the format-2 shape of the file by hand, since nothing writes it
    // any more: separate `[[body]]` and `[[solid]]` lists.
    let world = mixed();
    let mut legacy = Project::from_world(&world);
    legacy.fusion.format = 2;
    // Unfold what `from_world` folded: objects back into the two lists a
    // format-2 writer produced, keeping their order.
    for object in std::mem::take(&mut legacy.objects) {
        let deformable = legacy
            .materials
            .iter()
            .find(|m| m.name == object.material)
            .unwrap()
            .representation
            == Representation::Deformable;
        if deformable {
            let m = legacy
                .materials
                .iter()
                .find(|m| m.name == object.material)
                .unwrap();
            let solid = m.solid_part();
            legacy
                .solids
                .push(fusion_project::SolidEntry::from(&fusion_core::SolidDesc {
                    shape: fusion_core::Shape::from(object.shape.clone()),
                    pose: fusion_math::Isometry::new(
                        Vec3::from_array(object.position),
                        fusion_math::Quat::from_xyzw(
                            object.orientation[0],
                            object.orientation[1],
                            object.orientation[2],
                            object.orientation[3],
                        ),
                    ),
                    velocity: Vec3::from_array(object.linear_velocity),
                    temperature: object.temperature,
                    params: object.solid_params(&solid),
                }));
        } else {
            legacy.bodies.push(fusion_project::BodyEntry {
                shape: object.shape,
                position: object.position,
                orientation: object.orientation,
                linear_velocity: object.linear_velocity,
                angular_velocity: object.angular_velocity,
                body_type: object.body_type,
                material: object.material,
                mass: object.mass,
                script: object.script,
            });
        }
    }

    let text = legacy.to_toml().unwrap();
    assert!(text.contains("[[body]]"), "this is a format-2 file");

    let loaded = Project::from_toml(&text).unwrap();
    assert_eq!(loaded.fusion.format, 3, "it should arrive as format 3");
    assert!(loaded.bodies.is_empty() && loaded.solids.is_empty());
    assert_eq!(loaded.objects.len(), 4);

    let mut migrated = loaded.build().unwrap();
    let mut direct = fusion_sim::Simulation::new(mixed());
    assert_eq!(
        state_hash(&migrated),
        state_hash(&direct),
        "the migrated scene is not the scene"
    );
    for step in 0..200 {
        migrated.step();
        direct.step();
        assert_eq!(
            state_hash(&migrated),
            state_hash(&direct),
            "the migrated run diverged at step {step}"
        );
    }
}

/// Joints refer to objects, and a joint to a continuum is refused.
#[test]
fn a_joint_to_a_continuum_says_why() {
    let mut project = Project::from_world(&mixed());
    let jelly = project
        .objects
        .iter()
        .position(|o| o.material == "jelly")
        .expect("a jelly object");
    project.joints.push(fusion_project::JointEntry {
        body_a: 0,
        body_b: jelly as u32,
        ..project_joint()
    });
    let error = project.build_world().unwrap_err();
    assert!(matches!(error, ProjectError::Unresolved(_)));
    let text = format!("{error}");
    assert!(text.contains("deformable"), "unhelpful message: {text}");
}

fn project_joint() -> fusion_project::JointEntry {
    let mut world = World::new();
    let a = world.add(BodyDesc::sphere(0.1).at(vec3(0.0, 1.0, 0.0)).fixed());
    let b = world.add(BodyDesc::sphere(0.1).at(vec3(0.0, 0.5, 0.0)));
    world.connect(a, b, vec3(0.0, 0.75, 0.0), fusion_core::JointKind::Ball);
    Project::from_world(&world).joints.remove(0)
}

/// A scene the editor would produce with nothing but defaults still resolves
/// its sampling, so a deformable object never arrives under-sampled.
#[test]
fn a_deformable_object_without_a_resolution_gets_a_usable_one() {
    let mut project = Project::from_world(&World::new());
    project
        .materials
        .push(fusion_project::MaterialEntry::rigid("x", &Material::WOOD));
    let mut jelly = fusion_project::MaterialEntry::rigid("jelly", &Material::WOOD);
    jelly.representation = Representation::Deformable;
    jelly.density = 1000.0;
    jelly.deformable = Some(Default::default());
    project.materials.push(jelly);

    for size in [0.1f32, 0.3, 1.0, 3.0] {
        let mut object = fusion_project::ObjectEntry::new(
            fusion_project::ShapeEntry::Cuboid {
                size: [size, size, size],
            },
            "jelly",
        );
        object.position = [0.0, size, 0.0];
        project.objects = vec![object];
        let world = project.build_world().unwrap();
        let points = world.solids[0].len();
        // Scale-free: the sampling follows the shape, so a 3 m block is not
        // resolved for a 30 cm one and a 10 cm one is not a million points.
        // Six across, whatever the size.
        assert_eq!(
            points, 216,
            "a {size} m block sampled to {points} material points"
        );
    }
}
