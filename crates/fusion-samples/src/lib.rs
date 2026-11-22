//! Streaming NumPy shards from the same mechanics used in the Fusion app.
use core_world::Representation;
use fusion_project::{Project, SolverKind};
use serde::Serialize;
use std::{
    fs::{self, File, OpenOptions},
    io::{self, BufWriter, Write},
    path::{Path, PathBuf},
    sync::atomic::{AtomicBool, Ordering},
    time::Instant,
};

/// One dataset-generation job. Trajectory IDs determine seeds independently
/// of worker count; workers each own `threads` engine threads.
#[derive(Clone, Debug, Serialize)]
pub struct Config {
    pub trajectories: usize,
    pub steps: usize,
    pub workers: usize,
    pub threads: usize,
    pub seed: u64,
    /// Independent uniform perturbation per linear velocity component, m/s.
    pub velocity: f64,
    /// Independent uniform perturbation per angular velocity component, rad/s.
    pub spin: f64,
    /// User-supplied revision/build identifier for dataset provenance.
    pub build_id: String,
}
impl Default for Config {
    fn default() -> Self {
        Self {
            trajectories: 100,
            steps: 120,
            workers: 1,
            threads: 1,
            seed: 0,
            velocity: 0.5,
            spin: 0.5,
            build_id: "unspecified".into(),
        }
    }
}
#[derive(Debug, Serialize)]
pub struct Summary {
    pub transitions: usize,
    pub elapsed_seconds: f64,
    pub transitions_per_second: f64,
    pub newton_iterations: usize,
    pub linear_iterations: usize,
    pub retry_splits: usize,
    pub payload_bytes: u64,
}
#[derive(Serialize)]
struct Field {
    name: String,
    offset: usize,
    width: usize,
}
#[derive(Serialize)]
struct Shard {
    states: String,
    work: String,
    trajectory_ids: Vec<usize>,
    trajectory_seeds: Vec<u64>,
}
#[derive(Serialize)]
struct Manifest<'a> {
    format: u32,
    generator_version: &'static str,
    engine: &'static str,
    scalar_type: &'static str,
    timestep: f64,
    state_width: usize,
    history_encoding: &'static str,
    work_columns: [&'static str; 5],
    config: &'a Config,
    summary: &'a Summary,
    fields: Vec<Field>,
    shards: Vec<Shard>,
}
fn error(message: impl Into<String>) -> io::Error {
    io::Error::new(io::ErrorKind::InvalidInput, message.into())
}

/// Generate new shards, refusing an existing directory. Only a successfully
/// completed job has `manifest.toml`; interrupted/failed jobs retain INCOMPLETE.
/// State shape is [trajectories_in_shard, steps+1, state_width]. Adjacent frames
/// form input/target pairs without storing every interior frame twice.
pub fn generate(project: &Project, output: &Path, config: &Config) -> io::Result<Summary> {
    let start = Instant::now();
    if project.state.is_some() {
        return Err(error("use a scene without a saved-run state reference; generation starts from the declared initial scene"));
    }
    if project.solver.kind != SolverKind::Core {
        return Err(error(
            "select kind = core in the project; generation never switches physics implicitly",
        ));
    }
    if config.trajectories == 0
        || config.steps == 0
        || config.workers == 0
        || config.threads == 0
        || config.workers > config.trajectories
        || !config.velocity.is_finite()
        || config.velocity < 0.0
        || !config.spin.is_finite()
        || config.spin < 0.0
    {
        return Err(error("positive counts, workers <= trajectories, and finite nonnegative perturbations are required"));
    }
    let transitions = config
        .trajectories
        .checked_mul(config.steps)
        .ok_or_else(|| error("transition count overflow"))?;
    let frame_count = config
        .steps
        .checked_add(1)
        .ok_or_else(|| error("frame count overflow"))?;
    config
        .workers
        .checked_mul(config.threads)
        .ok_or_else(|| error("worker/thread count overflow"))?;
    // Validate physics before creating any output.
    let runner = project
        .build()
        .map_err(|e| error(e.to_string()))?
        .into_core_runner(config.threads)
        .map_err(error)?;
    if !runner.world().objects().iter().any(|o| o.is_dynamic()) {
        return Err(error("scene has no moving objects"));
    }
    let timestep = runner.timestep();
    let fields: Vec<_> = runner
        .state_layout()
        .into_iter()
        .map(|f| Field {
            name: f.name,
            offset: f.offset,
            width: f.width,
        })
        .collect();
    let width = fields.last().map_or(0, |f| f.offset + f.width);
    if width == 0 {
        return Err(error("scene has no state"));
    }
    transitions
        .checked_mul(width)
        .and_then(|n| n.checked_mul(8))
        .ok_or_else(|| error("dataset size overflow"))?;
    fs::create_dir(output)?;
    fs::write(
        output.join("INCOMPLETE"),
        "Do not train from this directory until manifest.toml exists and this marker is removed.\n",
    )?;
    project
        .save(output.join("scene.fusion"))
        .map_err(|e| error(e.to_string()))?;
    save_geometry(runner.world(), output)?;
    drop(runner);
    let cancelled = AtomicBool::new(false);
    let results = std::thread::scope(|scope| {
        let handles: Vec<_> = (0..config.workers)
            .map(|worker| {
                let cancelled = &cancelled;
                scope.spawn(move || {
                    let result = run_worker(
                        project,
                        output,
                        config,
                        worker,
                        frame_count,
                        width,
                        cancelled,
                    );
                    if result.is_err() {
                        cancelled.store(true, Ordering::Relaxed);
                    }
                    result
                })
            })
            .collect();
        handles
            .into_iter()
            .map(|h| {
                h.join()
                    .unwrap_or_else(|_| Err(error("dataset worker panicked")))
            })
            .collect::<Vec<_>>()
    });
    let mut shards = Vec::new();
    let mut newton = 0;
    let mut linear = 0;
    let mut splits = 0;
    if let Some(reason) = results
        .iter()
        .filter_map(|r| r.as_ref().err())
        .find(|e| !e.to_string().contains("job cancelled"))
    {
        return Err(error(reason.to_string()));
    }
    for result in results {
        let (shard, work) = result?;
        shards.push(shard);
        newton += work[0];
        linear += work[1];
        splits += work[2];
    }
    let elapsed = start.elapsed().as_secs_f64();
    let payload_bytes = fs::read_dir(output)?
        .filter(|e| e.as_ref().map_or(true, |e| e.file_name() != "INCOMPLETE"))
        .map(|e| e.and_then(|e| e.metadata()).map(|m| m.len()))
        .collect::<io::Result<Vec<_>>>()?
        .into_iter()
        .sum();
    let summary = Summary {
        transitions,
        elapsed_seconds: elapsed,
        transitions_per_second: transitions as f64 / elapsed,
        newton_iterations: newton,
        linear_iterations: linear,
        retry_splits: splits,
        payload_bytes,
    };
    let manifest = Manifest {
        format: 1,
        generator_version: env!("CARGO_PKG_VERSION"),
        engine: "Fusion Core / TR-BDF2",
        scalar_type: "little-endian f64",
        timestep,
        state_width: width,
        // Matches `CoreRunner::state_layout`, which also names every field.
        history_encoding: "acceleration_present, acceleration[3*blocks], \
            earlier_present, earlier_position[3*blocks], earlier_velocity[3*blocks], \
            restart[blocks], per-rigid(affine_present, quaternion_xyzw[4], \
            matrix_column_major[9]), gravity_present, gravity_xyz[3], \
            previous_step_present, previous_step_seconds, \
            per-rigid(contact_unloading_joules); absent values zero",
        work_columns: [
            "solves",
            "newton_iterations",
            "linear_iterations",
            "retry_splits",
            "wall_seconds",
        ],
        config,
        summary: &summary,
        fields,
        shards,
    };
    let text = toml::to_string_pretty(&manifest).map_err(|e| error(e.to_string()))?;
    fs::write(output.join("manifest.toml.partial"), text)?;
    fs::rename(
        output.join("manifest.toml.partial"),
        output.join("manifest.toml"),
    )?;
    fs::remove_file(output.join("INCOMPLETE"))?;
    Ok(summary)
}
fn run_worker(
    project: &Project,
    output: &Path,
    c: &Config,
    worker: usize,
    frames: usize,
    width: usize,
    cancelled: &AtomicBool,
) -> io::Result<(Shard, [usize; 3])> {
    let mut runner = project
        .build()
        .map_err(|e| error(e.to_string()))?
        .into_core_runner(c.threads)
        .map_err(error)?;
    let ids: Vec<_> = (worker..c.trajectories).step_by(c.workers).collect();
    let seeds: Vec<_> = ids
        .iter()
        .map(|&id| c.seed.wrapping_add(id as u64))
        .collect();
    let states_name = format!("states-{worker:03}.npy");
    let work_name = format!("work-{worker:03}.npy");
    let mut states = Npy::new(
        &output.join(&states_name),
        "<f8",
        &[ids.len(), frames, width],
    )?;
    let mut work = Npy::new(&output.join(&work_name), "<f8", &[ids.len(), c.steps, 5])?;
    let mut state = Vec::with_capacity(width);
    let mut counts = [0; 3];
    for (&id, &seed) in ids.iter().zip(&seeds) {
        runner.reset(seed, c.velocity, c.spin).map_err(error)?;
        runner.state_into(&mut state);
        states.f64s(&state)?;
        for step in 0..c.steps {
            if cancelled.load(Ordering::Relaxed) {
                return Err(error("another worker failed; job cancelled"));
            }
            let start = Instant::now();
            let report = runner.step();
            let seconds = start.elapsed().as_secs_f64();
            if report.failed() {
                return Err(error(format!("trajectory {id}, seed {seed}, step {step}: solve rejected ({report:?}); no target emitted")));
            }
            runner.state_into(&mut state);
            if state.len() != width || state.iter().any(|v| !v.is_finite()) {
                return Err(error(format!(
                    "trajectory {id}, step {step}: invalid state"
                )));
            }
            states.f64s(&state)?;
            work.f64s(&[
                report.solves as f64,
                report.iterations as f64,
                report.linear_iterations as f64,
                report.splits as f64,
                seconds,
            ])?;
            counts[0] += report.iterations;
            counts[1] += report.linear_iterations;
            counts[2] += report.splits;
        }
    }
    states.finish()?;
    work.finish()?;
    Ok((
        Shard {
            states: states_name,
            work: work_name,
            trajectory_ids: ids,
            trajectory_seeds: seeds,
        },
        counts,
    ))
}
fn save_geometry(world: &core_world::World, out: &Path) -> io::Result<()> {
    for (i, o) in world.objects().iter().enumerate() {
        if let Representation::Solid(s) = &o.representation {
            let mut rest = Npy::new(
                &out.join(format!("object-{i}-rest.npy")),
                "<f8",
                &[s.mesh.rest.len(), 3],
            )?;
            for p in &s.mesh.rest {
                rest.f64s(&[p.x, p.y, p.z])?;
            }
            rest.finish()?;
            let mut masses = Npy::new(
                &out.join(format!("object-{i}-mass.npy")),
                "<f8",
                &[s.nodes.mass.len()],
            )?;
            masses.f64s(&s.nodes.mass)?;
            masses.finish()?;
            let mut tets = Npy::new(
                &out.join(format!("object-{i}-tets.npy")),
                "<u4",
                &[s.mesh.tets.len(), 4],
            )?;
            for t in &s.mesh.tets {
                tets.u32s(t)?;
            }
            tets.finish()?;
            let mut mids = Npy::new(
                &out.join(format!("object-{i}-midpoints.npy")),
                "<u4",
                &[s.mesh.midpoints.len(), 6],
            )?;
            for t in &s.mesh.midpoints {
                mids.u32s(t)?;
            }
            mids.finish()?;
        }
    }
    Ok(())
}
struct Npy {
    writer: BufWriter<File>,
    remaining: usize,
    element_bytes: usize,
    path: PathBuf,
    scratch: Vec<u8>,
}
impl Npy {
    fn new(path: &Path, dtype: &str, shape: &[usize]) -> io::Result<Self> {
        let total = shape
            .iter()
            .try_fold(1usize, |n, &s| n.checked_mul(s))
            .ok_or_else(|| error("array size overflow"))?;
        let shape_text = shape
            .iter()
            .map(|s| s.to_string())
            .collect::<Vec<_>>()
            .join(", ");
        let mut header =
            format!("{{'descr': '{dtype}', 'fortran_order': False, 'shape': ({shape_text},), }}");
        let padding = (64 - (10 + header.len() + 1) % 64) % 64;
        header.push_str(&" ".repeat(padding));
        header.push('\n');
        let length = u16::try_from(header.len()).map_err(|_| error("NumPy header too large"))?;
        let file = OpenOptions::new().write(true).create_new(true).open(path)?;
        let mut writer = BufWriter::with_capacity(1024 * 1024, file);
        writer.write_all(b"\x93NUMPY\x01\x00")?;
        writer.write_all(&length.to_le_bytes())?;
        writer.write_all(header.as_bytes())?;
        Ok(Self {
            writer,
            remaining: total,
            element_bytes: if dtype == "<f8" { 8 } else { 4 },
            path: path.into(),
            scratch: Vec::with_capacity(65536),
        })
    }
    fn f64s(&mut self, values: &[f64]) -> io::Result<()> {
        if self.element_bytes != 8 || values.len() > self.remaining {
            return Err(error("NumPy shape mismatch"));
        }
        for chunk in values.chunks(8192) {
            self.scratch.clear();
            for v in chunk {
                self.scratch.extend_from_slice(&v.to_le_bytes());
            }
            self.writer.write_all(&self.scratch)?;
        }
        self.remaining -= values.len();
        Ok(())
    }
    fn u32s(&mut self, values: &[u32]) -> io::Result<()> {
        if self.element_bytes != 4 || values.len() > self.remaining {
            return Err(error("NumPy shape mismatch"));
        }
        for v in values {
            self.writer.write_all(&v.to_le_bytes())?;
        }
        self.remaining -= values.len();
        Ok(())
    }
    fn finish(mut self) -> io::Result<()> {
        if self.remaining != 0 {
            return Err(error(format!(
                "incomplete NumPy array: {}",
                self.path.display()
            )));
        }
        self.writer.flush()
    }
}
