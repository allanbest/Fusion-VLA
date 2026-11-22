//! Generate and verify crossed bars for the main Fusion editor.
use fusion_core::{BodyDesc, World};
use fusion_math::vec3;
use fusion_project::{Project, SolverKind};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let mut world = World::new();
    world.angular_damping = 0.0;
    world.add(
        BodyDesc::cuboid(vec3(2.0, 0.1, 0.1))
            .at(vec3(0.0, -0.05, 0.0))
            .fixed(),
    );
    world.add(
        BodyDesc::cuboid(vec3(0.1, 0.1, 2.0))
            .at(vec3(0.0, 0.3, 0.0))
            .with_velocity(vec3(0.0, -2.0, 0.0)),
    );
    let mut project = Project::from_world(&world).with_note("Fusion Core edge contact: two bars meet at their interiors, with no corner over the other bar. The moving bar lands on the fixed support.");
    project.solver.kind = SolverKind::Core;
    project.run.solver_dt = 1.0 / 120.0;
    let path = std::env::args()
        .nth(1)
        .unwrap_or_else(|| "examples/core-edge-contact.fusion".into());
    project.save(&path)?;
    let mut sim = project.build()?;
    for _ in 0..120 {
        if let Some(failure) = sim.step().failure {
            return Err(failure.into());
        }
        if sim.world.bodies.position[1].y < 0.048 {
            return Err("bar passed through its support".into());
        }
    }
    println!(
        "Saved {path}; bar height {:.6} m after {:.3} s",
        sim.world.bodies.position[1].y,
        sim.time()
    );
    Ok(())
}
