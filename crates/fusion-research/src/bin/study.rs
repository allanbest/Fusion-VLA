//! `study` — run §23's workflow against a scene.
//!
//! ```text
//! study determinism scene.fusion --steps 600
//! study sweep       scene.fusion --axis substeps=4,8,16,32 --steps 900
//! study converge    scene.fusion --levels 5
//! study compare     a.fusion b.fusion --steps 900
//! study conserve    scene.fusion --steps 1200
//! ```
//!
//! A research tool is only a research tool if it can be pointed at something
//! without writing Rust. Everything here is the library in `fusion-research`
//! with an argument parser in front of it, and every command prints numbers a
//! person can act on rather than a verdict they have to trust.

use std::collections::BTreeMap;

use fusion_project::Project;
use fusion_research::{compare, converge, determinism_diff, metrics, store, sweep};

fn main() {
    let args: Vec<String> = std::env::args().skip(1).collect();
    let command = args.first().map(String::as_str).unwrap_or("--help");
    let result = match command {
        "determinism" => determinism(&args[1..]),
        "sweep" => run_sweep(&args[1..]),
        "converge" => convergence(&args[1..]),
        "compare" => comparison(&args[1..]),
        "conserve" => conservation(&args[1..]),
        "--help" | "-h" | "help" => {
            usage();
            return;
        }
        other => Err(format!("unknown command {other:?}; try --help")),
    };
    if let Err(e) = result {
        eprintln!("{e}");
        std::process::exit(1);
    }
}

fn usage() {
    println!(
        "study — Fusion's research workflow (PLAN.md §23)\n\n\
         USAGE\n  \
         study determinism <scene.fusion> [--steps N]\n      \
         Run the same scene twice and report the first step and body that\n      \
         disagree. A clean run is the reproducibility guarantee, checked.\n\n  \
         study conserve <scene.fusion> [--steps N] [--every N]\n      \
         Track momentum, angular momentum, energy and mass over a run.\n\n  \
         study compare <a.fusion> <b.fusion> [--steps N] [--tolerance M]\n      \
         Two scenes side by side: where and when they diverge.\n\n  \
         study converge <scene.fusion> [--levels N] [--seconds S]\n      \
         Halve the timestep repeatedly and report the observed order.\n\n  \
         study sweep <scene.fusion> --axis NAME=v1,v2,.. [--steps N] [--out DIR]\n      \
         Vary a parameter across runs and collect the results. Known axes:\n      \
         substeps, iterations, velocity_iterations, gravity, solver_dt.\n"
    );
}

// --- argument helpers ---------------------------------------------------

fn flag(args: &[String], name: &str) -> Option<String> {
    let position = args.iter().position(|a| a == name)?;
    args.get(position + 1).cloned()
}

fn number<T: std::str::FromStr>(args: &[String], name: &str, fallback: T) -> T {
    flag(args, name)
        .and_then(|v| v.parse().ok())
        .unwrap_or(fallback)
}

fn positional(args: &[String], index: usize) -> Option<&String> {
    args.iter().filter(|a| !a.starts_with("--")).nth(index)
}

fn load(args: &[String], index: usize) -> Result<Project, String> {
    let path = positional(args, index).ok_or("a .fusion file is needed")?;
    Project::load(path).map_err(|e| format!("could not open {path}: {e}"))
}

// --- commands -----------------------------------------------------------

fn determinism(args: &[String]) -> Result<(), String> {
    let project = load(args, 0)?;
    let steps: u64 = number(args, "--steps", 600);

    println!("determinism — the same scene, twice, {steps} steps\n");
    let report = determinism_diff(
        || project.build().expect("the scene built once already"),
        steps,
    );

    match report.first_divergent_step {
        None => {
            println!("  reproduced: identical for all {steps} steps");
            println!("\n  This is the guarantee in Determinism::GUARANTEE, checked rather");
            println!("  than assumed. Same build, same machine, bit for bit.");
        }
        Some(step) => {
            println!("  DIVERGED at step {step} of {steps}");
            match report.first_divergent_body {
                Some(body) => println!("  worst body: {body}, {:.3e} m apart", report.separation),
                None => println!("  no body could be identified"),
            }
            println!("\n  This is a defect, not a tolerance. Fusion guarantees bit-identical");
            println!("  replay for one build on one machine (§34), so something is being");
            println!("  iterated or summed in an order the scene does not determine.");
            std::process::exit(2);
        }
    }
    Ok(())
}

fn conservation(args: &[String]) -> Result<(), String> {
    let project = load(args, 0)?;
    let steps: u64 = number(args, "--steps", 1200);
    let every: u64 = number(args, "--every", 10);
    let mut sim = project.build().map_err(|e| e.to_string())?;

    let mut trace = metrics::ConservationTrace::new();
    trace.observe(&sim);
    for step in 0..steps {
        sim.step();
        if (step + 1) % every.max(1) == 0 {
            trace.observe(&sim);
        }
    }
    trace.observe(&sim);

    let summary = trace.summary().ok_or("nothing was recorded")?;
    println!("conservation — {steps} steps, sampled every {every}\n");
    println!("  {:<22} {:>14}  {:>12}", "quantity", "drift", "relative");
    println!("  {}", "-".repeat(52));
    if summary.closed {
        println!(
            "  {:<22} {:>14.3e}  {:>11.4}%",
            "linear momentum",
            summary.momentum_drift,
            summary.momentum_drift_relative * 100.0
        );
        println!(
            "  {:<22} {:>14.3e}  {:>11.4}%",
            "angular momentum",
            summary.angular_drift,
            summary.angular_drift_relative * 100.0
        );
    } else {
        println!(
            "  {:<22} {:>28}",
            "linear momentum", "n/a — scene not closed"
        );
        println!(
            "  {:<22} {:>28}",
            "angular momentum", "n/a — scene not closed"
        );
    }
    println!("  {:<22} {:>14.3e}", "mass", summary.mass_drift);
    println!(
        "  {:<22} {:>14.3} J   ({:.3} -> {:.3})",
        "energy",
        summary.energy_delta(),
        summary.energy_start,
        summary.energy_end
    );
    println!(
        "\n  worst single-step energy gain: {:.4} J at step {}",
        summary.worst_energy_gain, summary.worst_energy_gain_step
    );
    println!("  A slow bleed downward is damping. A spike upward is a solver");
    println!("  putting energy in — point the energy ledger at that step.");
    if !summary.closed {
        println!("\n  This scene has static geometry in it, so there is no momentum");
        println!("  invariant to check: a floor holds things up, and an immovable body");
        println!("  supplies whatever impulse it needs to. Momentum conservation is a");
        println!("  check on a closed scene — free space, everything dynamic — and on");
        println!("  one of those these two rows are sharp.");
    }
    Ok(())
}

fn comparison(args: &[String]) -> Result<(), String> {
    let a_project = load(args, 0)?;
    let b_project = load(args, 1)?;
    let steps: u64 = number(args, "--steps", 900);
    let tolerance: f32 = number(args, "--tolerance", 1e-3);

    let mut a = a_project.build().map_err(|e| e.to_string())?;
    let mut b = b_project.build().map_err(|e| e.to_string())?;
    let report = compare(&mut a, &mut b, steps, tolerance);

    println!("compare — two scenes, {steps} steps, tolerance {tolerance} m\n");
    match report.first_bitwise_difference {
        None => println!("  bit-identical throughout: these are the same run"),
        Some(step) => println!("  first bitwise difference : step {step}"),
    }
    match report.first_visible_difference {
        None => println!("  never diverged beyond {tolerance} m"),
        Some(step) => println!("  first visible difference : step {step}"),
    }
    println!(
        "  furthest apart           : {:.4e} m at step {}{}",
        report.max_separation,
        report.max_separation_step,
        report
            .max_separation_body
            .map(|b| format!(", body {b}"))
            .unwrap_or_default()
    );

    // The divergence curve, as a sparkline over ten buckets.
    if !report.separation.is_empty() {
        let buckets = 10.min(report.separation.len());
        let per = report.separation.len() / buckets;
        println!("\n  divergence over the run:");
        for b in 0..buckets {
            let slice = &report.separation[b * per..((b + 1) * per).min(report.separation.len())];
            let peak = slice.iter().copied().fold(0.0f32, f32::max);
            let bar = if report.max_separation > 0.0 {
                (peak / report.max_separation * 40.0) as usize
            } else {
                0
            };
            println!(
                "    step {:>6}  {:<40} {:.2e}",
                (b + 1) * per,
                "#".repeat(bar),
                peak
            );
        }
    }
    Ok(())
}

fn convergence(args: &[String]) -> Result<(), String> {
    let project = load(args, 0)?;
    let levels: usize = number(args, "--levels", 5);
    let seconds: f64 = number(args, "--seconds", 1.0);

    println!("converge — halving the timestep {levels} times over {seconds} s\n");
    let report = converge::study(1.0 / 60.0, levels, |h| {
        let mut sim = project.build().expect("the scene built once already");
        sim.solver_dt = h as f32;
        let steps = (seconds / h).round() as u64;
        for _ in 0..steps {
            sim.step();
        }
        // The measurement: total mechanical energy at the end. A scalar that
        // every scene has, so this works on any file rather than only on ones
        // with a body somebody nominated.
        sim.mechanical_energy()
    });

    println!("  {:>12}  {:>16}", "timestep", "energy (J)");
    println!("  {}", "-".repeat(32));
    for level in &report.levels {
        println!("  {:>12.6}  {:>16.6}", level.h, level.value);
    }
    if report.observed_order.is_empty() {
        println!("\n  no order could be observed: the measurement did not move.");
        return Ok(());
    }
    println!("\n  observed order, coarsest triple first:");
    for (i, p) in report.observed_order.iter().enumerate() {
        println!("    {i}: {p:.3}");
    }
    if let Some(value) = report.extrapolated {
        println!("\n  extrapolated to zero timestep: {value:.6} J");
    }
    // The interpretation, which is the part a number alone does not give.
    match report.order() {
        Some(p) if p <= 0.05 => {
            println!("\n  NOT CONVERGING. An order at or below zero means the answer moved");
            println!("  further apart as the timestep shrank, so there is no Δt-independent");
            println!("  result here to converge to. For a contact- or fluid-heavy scene under");
            println!("  XPBD that is the expected outcome and the reason §9 puts this solver");
            println!("  in the real-time tier: the iteration count sets the answer, not the");
            println!("  timestep. It means the number this scene produces is a property of");
            println!("  the settings as much as of the physics — fine for a picture, not");
            println!("  something to quote.");
        }
        Some(p) if p < 0.75 => {
            println!("\n  Converging, but below first order ({p:.2}). Something in the step is");
            println!("  not refining with Δt — usually a constraint solved to an iteration");
            println!("  count rather than to a tolerance.");
        }
        Some(p) => {
            println!("\n  Converging at order {p:.2}. Semi-implicit Euler is first order, so");
            println!("  around 1 is the integrator behaving and the extrapolation above is");
            println!("  worth more than any single row of the table.");
        }
        None => {}
    }
    Ok(())
}

fn run_sweep(args: &[String]) -> Result<(), String> {
    let project = load(args, 0)?;
    let steps: u64 = number(args, "--steps", 900);
    let spec = flag(args, "--axis").ok_or("--axis NAME=v1,v2,... is needed")?;
    let (name, values) = spec.split_once('=').ok_or("--axis takes NAME=v1,v2,...")?;
    let values: Vec<f64> = values
        .split(',')
        .map(|v| {
            v.trim()
                .parse::<f64>()
                .map_err(|_| format!("{v:?} is not a number"))
        })
        .collect::<Result<_, _>>()?;

    let axis = axis_named(name, values)?;
    let sweep = sweep::Sweep::new(project, steps).axis(axis);

    println!("sweep — {} runs of {steps} steps\n", sweep.len());
    let results = sweep
        .run(|sim| {
            let mut out = BTreeMap::new();
            out.insert("energy".into(), sim.mechanical_energy());
            out.insert("bodies".into(), sim.world.len() as f64);
            out.insert(
                "highest_body".into(),
                (0..sim.world.len())
                    .map(|i| sim.world.bodies.translation(i).y)
                    .fold(f32::NEG_INFINITY, f32::max) as f64,
            );
            out
        })
        .map_err(|e| e.to_string())?;

    print!("{}", results.to_csv());

    if let Some(dir) = flag(args, "--out") {
        let study = store::Study::new(format!("sweep over {name}"), results);
        let path = study.save(&dir).map_err(|e| e.to_string())?;
        println!("\nwrote {}", path.display());
        println!("and {dir}/results.csv");
    }
    Ok(())
}

/// The axes a scene can be swept over from the command line.
///
/// A closed list rather than a path expression into the project, because a
/// research tool that lets you write `--axis world.gravity[1]` needs a query
/// language, and the five parameters anybody actually sweeps are these.
fn axis_named(name: &str, values: Vec<f64>) -> Result<sweep::Axis, String> {
    Ok(match name {
        "substeps" => sweep::Axis::new(name, values, |p, v| p.solver.substeps = v as u32),
        "iterations" => sweep::Axis::new(name, values, |p, v| p.solver.iterations = v as u32),
        "velocity_iterations" => {
            sweep::Axis::new(name, values, |p, v| p.solver.velocity_iterations = v as u32)
        }
        "gravity" => sweep::Axis::new(name, values, |p, v| p.world.gravity[1] = v as f32),
        "solver_dt" => sweep::Axis::new(name, values, |p, v| p.run.solver_dt = v as f32),
        other => {
            return Err(format!(
                "unknown axis {other:?}. Known: substeps, iterations, \
                 velocity_iterations, gravity, solver_dt"
            ))
        }
    })
}
