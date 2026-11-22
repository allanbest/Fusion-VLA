//! Invalid authored physics must be refused before a solver runs.
use fusion_core::{BodyDesc, Material, World};
use fusion_project::{MaterialEntry, MaterialPack, Project, StrengthEntry, ThermalEntry};

fn project() -> Project {
    let mut world = World::new();
    world.add(BodyDesc::sphere(0.1).with_material(Material::STEEL));
    Project::from_world(&world)
}

#[test]
fn invalid_density_is_refused_before_world_construction() {
    for density in [0.0, -1.0, f32::NAN, f32::INFINITY] {
        let mut p = project();
        p.materials[0].density = density;
        assert!(p.build_world().is_err(), "accepted density {density}");
    }
}

#[test]
fn invalid_clocks_are_refused_before_starting_a_run() {
    for value in [0.0, -1.0, f32::NAN, f32::INFINITY] {
        let mut p = project();
        p.run.solver_dt = value;
        assert!(p.build().is_err(), "accepted timestep {value}");
        p = project();
        p.run.output_rate = value as f64;
        assert!(p.build().is_err(), "accepted output rate {value}");
    }
}

#[test]
fn both_projects_and_packs_reject_invalid_thermal_and_strength_tables() {
    let mut bad_heat = MaterialEntry::solid();
    bad_heat.thermal = Some(ThermalEntry {
        specific_heat: 0.0,
        ..Default::default()
    });
    let mut bad_strength = MaterialEntry::solid();
    bad_strength.strength = Some(StrengthEntry {
        poissons_ratio: 0.5,
        ..Default::default()
    });
    for mut invalid in [bad_heat, bad_strength] {
        let mut p = project();
        invalid.name = p.materials[0].name.clone();
        p.materials[0] = invalid;
        assert!(MaterialPack::from_project(&p, "invalid")
            .validate()
            .is_err());
        assert!(p.build_world().is_err());
    }
}
