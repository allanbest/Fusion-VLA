use fusion_project::{MaterialEntry, MaterialPack, Project, Representation};

fn pack() -> MaterialPack {
    let mut project = Project::from_world(&fusion_core::World::new());
    project.materials.push(MaterialEntry::solid());
    project.materials.push(MaterialEntry::rigid(
        "Custom steel",
        &fusion_core::Material::STEEL,
    ));
    MaterialPack::from_project(&project, "Workshop")
}

#[test]
fn pack_round_trip_preserves_physics_and_appearance() {
    let mut pack = pack();
    pack.materials[1].look.color = [0.2, 0.3, 0.4];
    pack.materials[1].look.metallic = 0.9;
    assert_eq!(
        MaterialPack::from_toml(&pack.to_toml().unwrap()).unwrap(),
        pack
    );
    let mut project = Project::from_world(&fusion_core::World::new());
    assert_eq!(pack.install_into(&mut project).unwrap(), 2);
    assert_eq!(pack.install_into(&mut project).unwrap(), 0);
    assert_eq!(project.materials.len(), 2);
    assert_eq!(
        project.metadata.tags,
        vec![("material-pack:Workshop".into(), "1.0.0".into())]
    );
    assert_eq!(
        Project::from_toml(&project.to_toml().unwrap())
            .unwrap()
            .materials,
        pack.materials
    );
}

#[test]
fn conflicts_fail_before_changing_anything() {
    let pack = pack();
    let mut project = Project::from_world(&fusion_core::World::new());
    project.materials.push(MaterialEntry::rigid(
        "Custom steel",
        &fusion_core::Material::RUBBER,
    ));
    let before = project.clone();
    assert!(pack
        .install_into(&mut project)
        .unwrap_err()
        .contains("Custom steel"));
    assert_eq!(
        project, before,
        "Solid must not be partially imported before the collision"
    );
}

#[test]
fn invalid_and_nonfinite_materials_are_refused() {
    let mut invalid = pack();
    invalid.materials[1].density = f32::NAN;
    assert!(invalid.validate().is_err());
    invalid = pack();
    invalid.materials[1].name = "Solid".into();
    assert!(invalid.validate().is_err());
    invalid = pack();
    invalid.materials[1].look.roughness = -1.0;
    assert!(invalid.validate().is_err());
    invalid = pack();
    invalid.format = 2;
    assert!(invalid.validate().is_err());
}

#[test]
fn advanced_library_parameters_survive_pack_installation() {
    let mut pack = pack();
    for (name, material) in fusion_core::SolidMaterial::LIBRARY {
        let mut entry = MaterialEntry::rigid(name, &fusion_core::Material::GENERIC);
        entry.representation = Representation::Deformable;
        entry.density = material.density;
        entry.deformable = Some(fusion_project::DeformableEntry::from(&material));
        pack.materials.push(entry);
    }
    assert!(pack.requires_advanced());
    assert_eq!(
        MaterialPack::from_toml(&pack.to_toml().unwrap()).unwrap(),
        pack
    );
}
