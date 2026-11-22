//! Repeated complete intervals; warm-up uses a separate simulation.
use fusion_core::{BodyDesc, SolidDesc, SolidMaterial, SolidParams, World};
use fusion_math::{vec3, Vec3};
use fusion_project::{Project, SolverKind};
use std::time::Instant;
fn project(contact: bool, across: f32) -> Project {
    let mut w = World::new();
    w.angular_damping = 0.0;
    w.add(
        BodyDesc::cuboid(vec3(4.0, 0.4, 4.0))
            .at(vec3(0.0, -0.2, 0.0))
            .fixed(),
    );
    w.add_solid(
        &SolidDesc::block(Vec3::splat(0.3))
            .at(vec3(0.0, if contact { 0.16 } else { 5.0 }, 0.0))
            .with_params(SolidParams::at(0.3 / (2.0 * across)).of(SolidMaterial::JELLY)),
    );
    let mut p = Project::from_world(&w);
    p.solver.kind = SolverKind::Core;
    p.run.solver_dt = 1.0 / 120.0;
    p
}
fn main() {
    let steps = std::env::args()
        .nth(1)
        .and_then(|s| s.parse::<usize>().ok())
        .unwrap_or(120);
    for (label, contact, across) in [
        ("free_small", false, 2.0),
        ("rest_small", true, 2.0),
        ("rest_medium", true, 4.0),
    ] {
        let p = project(contact, across);
        let mut warm = p.build().unwrap();
        for _ in 0..8 {
            assert!(warm.step().failure.is_none());
        }
        drop(warm);
        let mut times = Vec::new();
        for _ in 0..3 {
            let mut sim = p.build().unwrap();
            let start = Instant::now();
            let mut work = 0;
            for _ in 0..steps {
                let r = sim.step();
                assert!(r.failure.is_none(), "{r:?}");
                work += sim.core_report().unwrap().iterations;
            }
            times.push(start.elapsed().as_secs_f64());
            std::hint::black_box((sim.mechanical_energy(), work));
        }
        times.sort_by(f64::total_cmp);
        println!(
            "{label}: {:.3} ms/step, {:.1} steps/s ({} steps, 3 repeats)",
            times[1] * 1000.0 / steps as f64,
            steps as f64 / times[1],
            steps
        );
        for threads in [1, 4] {
            let mut times = Vec::new();
            let mut work = [0; 3];
            for _ in 0..3 {
                let mut runner = p.build().unwrap().into_core_runner(threads).unwrap();
                runner.set_profiling(std::env::args().any(|a| a == "--profile"));
                let mut sample = Vec::new();
                let start = Instant::now();
                work = [0; 3];
                for _ in 0..steps {
                    let report = runner.step();
                    assert!(!report.failed(), "{report:?}");
                    work[0] += report.iterations;
                    work[1] += report.linear_iterations;
                    work[2] += report.splits;
                    runner.state_into(&mut sample);
                    std::hint::black_box(&sample);
                }
                times.push(start.elapsed().as_secs_f64());
                if std::env::args().any(|a| a == "--profile") {
                    println!("{}", runner.profile());
                }
            }
            times.sort_by(f64::total_cmp);
            println!("{label} runner/{threads}: {:.3} ms/step, {:.1} samples/s; Newton {}, CG {}, splits {}", times[1]*1000.0/steps as f64, steps as f64/times[1],work[0],work[1],work[2]);
        }
    }
}
