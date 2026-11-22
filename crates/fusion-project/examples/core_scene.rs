//! Generate a small scene to open in the main Fusion editor.
use fusion_core::{BodyDesc, Material, SolidDesc, SolidMaterial, SolidParams, World};
use fusion_math::{vec3, Vec3};
use fusion_project::{Project, SolverKind};
fn main() -> Result<(), Box<dyn std::error::Error>> {
    let mut world = World::new();
    world.angular_damping = 0.0;
    world.add(
        BodyDesc::cuboid(vec3(4.0, 0.4, 4.0))
            .at(vec3(0.0, -0.2, 0.0))
            .fixed(),
    );
    world.add(
        BodyDesc::sphere(0.12)
            .at(vec3(0.5, 0.65, 0.0))
            .with_material(Material::RUBBER),
    );
    world.add_solid(
        &SolidDesc::block(Vec3::splat(0.24))
            .at(vec3(-0.4, 0.5, 0.0))
            .with_params(SolidParams::at(0.06).of(SolidMaterial::JELLY)),
    );
    let mut project = Project::from_world(&world).with_note("Fusion Core preview: a rigid rubber ball and a quadratic finite-element jelly block. Gravity, elasticity, contact and friction are solved in the main app's new mechanics backend.");
    project.solver.kind = SolverKind::Core;
    project.run.solver_dt = 1.0 / 120.0;
    let path = std::env::args()
        .nth(1)
        .unwrap_or_else(|| "examples/core-mechanics.fusion".into());
    project.save(&path)?;
    let mut sim = project.build()?;
    for _ in 0..120 {
        let report = sim.step();
        if let Some(failure) = report.failure {
            return Err(failure.into());
        }
    }
    println!(
        "Saved {path}; ran {} steps to {:.3} s, energy {:.6} J",
        sim.step_index(),
        sim.time(),
        sim.mechanical_energy()
    );
    Ok(())
}
