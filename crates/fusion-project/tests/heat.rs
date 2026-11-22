//! Temperature in rigid bodies, through the file.
//!
//! The same promise as every other part of the format: a scene with heat in
//! it saves, loads, and runs to the same temperatures to the bit. And the
//! other half: a scene without heat writes no heat at all, so it reads and
//! hashes as it did before rigid bodies had a temperature.

use fusion_core::{BodyDesc, Material, World};
use fusion_math::{vec3, Vec3};
use fusion_project::Project;
use fusion_sim::{state_hash, HeatSettings, Simulation};

fn hot_slide(heat: bool) -> Simulation {
    let mut world = World::new();
    world.add(
        BodyDesc::cuboid(vec3(20.0, 1.0, 20.0))
            .at(vec3(0.0, -0.5, 0.0))
            .with_material(Material::CONCRETE)
            .fixed(),
    );
    world.add(
        BodyDesc::cuboid(Vec3::splat(0.4))
            .at(vec3(0.0, 0.2, 0.0))
            .with_velocity(vec3(4.0, 0.0, 0.0))
            .with_material(Material::STEEL)
            .with_temperature(700.0),
    );
    let mut sim = Simulation::new(world);
    if heat {
        sim.heat = HeatSettings {
            convection: 25.0,
            ..HeatSettings::on()
        };
    }
    sim
}

#[test]
fn a_hot_scene_reproduces_through_the_file() {
    let mut original = hot_slide(true);
    let text = Project::from_simulation(&original).to_toml().unwrap();
    assert!(text.contains("[solver.heat]"), "{text}");
    assert!(text.contains("temperature = 700"), "{text}");

    let loaded = Project::from_toml(&text).unwrap();
    // Heat alone is format 6: nothing in it needs the stress and failure of 7.
    assert_eq!(loaded.fusion.format, 6);
    assert_eq!(loaded.solver.heat.convection, 25.0);
    let mut rebuilt = loaded.build().unwrap();
    assert_eq!(rebuilt.heat, original.heat);
    for _ in 0..240 {
        original.step();
        rebuilt.step();
    }
    assert_eq!(state_hash(&original), state_hash(&rebuilt));
    let block = fusion_core::BodyHandle(1);
    assert!(rebuilt.body_temperature(block) < 700.0);
}

#[test]
fn steel_keeps_its_thermal_numbers_and_generic_writes_none() {
    let text = Project::from_simulation(&hot_slide(false))
        .to_toml()
        .unwrap();
    let project = Project::from_toml(&text).unwrap();
    let steel = project
        .materials
        .iter()
        .find(|m| m.name == "steel")
        .expect("steel is in the file");
    assert_eq!(steel.rigid_part(), Material::STEEL);

    // A neutral material has nothing to say about heat, and says nothing.
    let mut world = World::new();
    world.add(BodyDesc::sphere(0.5));
    let text = Project::from_simulation(&Simulation::new(world))
        .to_toml()
        .unwrap();
    assert!(!text.contains("thermal"), "{text}");
    assert!(!text.contains("heat"), "{text}");
}

/// Format 5 had no heat, so a format-5 file with a rigid material in it must
/// read back as the neutral thermal material — and a material that does say
/// it is deformable still conducts like what it says it is when solved rigid.
#[test]
fn older_materials_take_the_neutral_thermal_numbers() {
    let text = Project::from_simulation(&hot_slide(false))
        .to_toml()
        .unwrap();
    // Every `[material.thermal]` table, header and keys, up to the blank line
    // that ends it — which is what a format-5 file is.
    let mut skipping = false;
    let stripped: String = text
        .lines()
        .filter(|l| {
            if l.trim() == "[material.thermal]" {
                skipping = true;
            } else if l.trim().is_empty() {
                skipping = false;
            }
            !skipping
        })
        .collect::<Vec<_>>()
        .join("\n");
    assert!(!stripped.contains("emissivity"), "{stripped}");
    let project = Project::from_toml(&stripped).unwrap();
    for m in &project.materials {
        assert_eq!(
            m.rigid_part().thermal,
            fusion_core::Thermal::GENERIC,
            "{}",
            m.name
        );
    }
}

/// Every library material's strength survives a save, infinite strengths
/// included.
#[test]
fn strengths_come_back_as_they_went() {
    for material in [
        Material::GENERIC,
        Material::STEEL,
        Material::CONCRETE,
        Material::GLASS,
        Material::RUBBER,
        Material::WATER,
        Material::WOOD,
        Material::STONE,
        Material::ICE,
    ] {
        let entry = fusion_project::MaterialEntry::rigid("m", &material);
        let text = toml::to_string(&entry).unwrap();
        let back: fusion_project::MaterialEntry = toml::from_str(&text).unwrap();
        assert_eq!(back.rigid_part().strength, material.strength, "{text}");
    }
}

/// A strength written before break patterns existed reads with the pattern
/// its other numbers imply: wood's for a grain, a metal's for one that yields.
#[test]
fn a_strength_with_no_pattern_gets_the_one_it_implies() {
    let read = |text: &str| {
        let entry: fusion_project::StrengthEntry = toml::from_str(text).unwrap();
        fusion_core::Strength::from(&entry).pattern
    };
    use fusion_core::Pattern;
    assert_eq!(
        read("youngs_modulus = 2e11\npoissons_ratio = 0.3\nyield_stress = 2.5e8\ntensile = 4e8\n"),
        Pattern::Tear
    );
    assert_eq!(
        read("youngs_modulus = 7e10\npoissons_ratio = 0.2\ntensile = 4e7\n"),
        Pattern::Crack
    );
    assert_eq!(
        read("youngs_modulus = 1.1e10\npoissons_ratio = 0.3\ntensile = 4e7\n[grain]\ntensile_across = 2.5e6\ncompressive_across = 5e6\nshear = 7e6\n"),
        Pattern::Splinter
    );
    assert_eq!(
        read("youngs_modulus = 3e10\npoissons_ratio = 0.2\ntensile = 3e6\npattern = \"chip\"\n"),
        Pattern::Chip
    );
}
