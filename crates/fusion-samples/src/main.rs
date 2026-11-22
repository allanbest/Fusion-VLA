use fusion_samples::{generate, Config};
use std::path::PathBuf;
fn main() {
    if let Err(e) = run() {
        eprintln!("fusion-samples: {e}");
        std::process::exit(1);
    }
}
fn run() -> Result<(), Box<dyn std::error::Error>> {
    let mut c = Config::default();
    let mut scene = None;
    let mut output = None;
    let mut args = std::env::args().skip(1);
    while let Some(flag) = args.next() {
        if flag == "--help" || flag == "-h" {
            println!("fusion-samples --scene FILE --output NEW_DIRECTORY [--trajectories 100] [--steps 120] [--workers 1] [--threads 1] [--seed 0] [--velocity 0.5] [--spin 0.5] [--build-id REVISION]\nWrites streaming f64 NumPy rollout shards and manifest.toml. Existing directories are refused. Split train/validation by trajectory ID, not adjacent frames. Each worker owns --threads solver threads; choose their product to fit your CPU.");
            return Ok(());
        }
        let value = args
            .next()
            .ok_or_else(|| format!("missing value for {flag}"))?;
        match flag.as_str() {
            "--scene" => scene = Some(PathBuf::from(value)),
            "--output" => output = Some(PathBuf::from(value)),
            "--trajectories" => c.trajectories = value.parse()?,
            "--steps" => c.steps = value.parse()?,
            "--workers" => c.workers = value.parse()?,
            "--threads" => c.threads = value.parse()?,
            "--seed" => c.seed = value.parse()?,
            "--velocity" => c.velocity = value.parse()?,
            "--spin" => c.spin = value.parse()?,
            "--build-id" => c.build_id = value,
            _ => return Err(format!("unknown option {flag}; use --help").into()),
        }
    }
    let project = fusion_project::Project::load(scene.ok_or("--scene is required")?)?;
    let output = output.ok_or("--output is required")?;
    let r = generate(&project, &output, &c)?;
    println!("{} transitions in {:.3}s: {:.1} samples/s; {} Newton / {} CG iterations, {} retries; {:.2} MiB payload written to {}",r.transitions,r.elapsed_seconds,r.transitions_per_second,r.newton_iterations,r.linear_iterations,r.retry_splits,r.payload_bytes as f64/(1024.0*1024.0),output.display());
    Ok(())
}
