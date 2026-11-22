//! The neighbour build, CPU against GPU.
//!
//! `cargo run --release -p fusion-gpu --example grid_bench`
//!
//! `bench profile` says the neighbour build is the largest single cost in a
//! fluid substep — 42% of one, after v0.6 part 1 made the whole substep 1.4x
//! cheaper. This is the same work on the other device, at sizes from the tank
//! Fusion actually simulates today up to the 10⁵–10⁷ range PLAN.md §24 says the
//! GPU is for.
//!
//! Three columns, because there are three honest answers depending on what the
//! rest of the step does:
//!
//! - **kernel** — steady-state throughput: many builds submitted back to back,
//!   waited on once. What a step pays when everything else is on the GPU too.
//! - **+ upload** — with the positions pushed across from the host each time.
//!   What a step pays when the particles live on the CPU.
//! - **+ readback** — with the lists pulled back. What a step pays when the
//!   density solve stayed behind. This column is the argument against a
//!   half-migrated pipeline and it is not close.
//!
//! Two things to read carefully. The readback column is *three* stalls, not
//! one: reconstructing per-particle lists on the host needs the bucket order,
//! the counts and the lists themselves, and each is its own round trip. One
//! would still cost over a millisecond — `gpu_bench` prices a bare stall at
//! 1.4 ms — so the conclusion does not turn on it. And at the first two sizes
//! the kernel and upload columns are within noise of each other, because both
//! are the fixed cost of asking rather than the cost of the work.

use std::time::Instant;

use fusion_compute::{Device, Graph};
use fusion_core::{FluidDesc, FluidParams, World};
use fusion_gpu::GpuGrid;
use fusion_math::vec3;
use fusion_solver::fluid;
use fusion_solver::FluidScratch;

const REPEATS: usize = 5;

fn best(mut f: impl FnMut()) -> f64 {
    let mut best = f64::INFINITY;
    for _ in 0..REPEATS {
        let start = Instant::now();
        f();
        best = best.min(start.elapsed().as_secs_f64());
    }
    best
}

/// A block of water at a given lattice spacing.
fn fluid_of(spacing: f32, side: f32) -> World {
    let mut world = World::new();
    world.add_fluid(
        &FluidDesc::block(vec3(side, side, side))
            .at(vec3(0.0, side * 0.5, 0.0))
            .with_params(FluidParams::water_at(spacing)),
    );
    world
}

fn main() {
    let Some(device) = Device::headless() else {
        println!("no GPU adapter on this machine — nothing to measure");
        return;
    };
    println!("Fusion 0.6 — neighbour build, CPU against GPU");
    println!("adapter: {} via {:?}\n", device.adapter(), device.backend());
    println!(
        "{:>10} {:>10} {:>10} {:>10} {:>10} {:>10} {:>9}",
        "particles", "cpu ms", "kernel ms", "speedup", "+upload", "+readback", "pairs"
    );
    println!("{}", "-".repeat(76));

    // Spacings chosen to sweep three orders of magnitude of particle count.
    for (spacing, side) in [
        (0.05f32, 0.75f32),
        (0.05, 1.2),
        (0.04, 1.6),
        (0.025, 1.6),
        (0.02, 1.9),
        (0.015, 2.0),
    ] {
        let world = fluid_of(spacing, side);
        let f = &world.fluids[0];
        let n = f.len();
        let h = f.params.smoothing_radius;

        // ---- CPU ----
        let mut scratch = FluidScratch::new();
        scratch.prepare(f);
        fluid::build_neighbours(f, &mut scratch);
        let pairs: usize = (0..n).map(|i| scratch.neighbours(i).len()).sum();
        let cpu = best(|| fluid::build_neighbours(f, &mut scratch));

        // ---- GPU ----
        //
        // The graph is rebuilt inside each timed block rather than hoisted,
        // which is both what a step does and what the borrow checker wants:
        // recording borrows the grid, and `upload` needs it back.
        const RUN: usize = 20;
        let mut grid = GpuGrid::new(&device, n);
        grid.upload(&device, &f.particles.position, h);
        {
            let mut warm = Graph::new("spatial hash");
            grid.record(&mut warm);
            warm.submit(&device);
            device.wait();
        }

        // Steady state: a run of builds, waited on once. Timing a single
        // submission would measure the latency of asking, which is not what a
        // step that keeps going pays.
        let kernel = best(|| {
            let mut graph = Graph::new("spatial hash");
            grid.record(&mut graph);
            for _ in 0..RUN {
                graph.submit(&device);
            }
            device.wait();
        }) / RUN as f64;

        let uploaded = best(|| {
            for _ in 0..RUN {
                grid.upload(&device, &f.particles.position, h);
                let mut graph = Graph::new("spatial hash");
                grid.record(&mut graph);
                graph.submit(&device);
            }
            device.wait();
        }) / RUN as f64;

        // One build, lists pulled back, per iteration. No run: a readback is a
        // stall by construction and cannot be amortized.
        let round_trip = best(|| {
            let mut graph = Graph::new("spatial hash");
            grid.record(&mut graph);
            graph.submit(&device);
            let _ = grid.read_lists(&device);
        });

        println!(
            "{n:>10} {:>10.3} {:>10.3} {:>9.1}x {:>10.3} {:>10.3} {:>9}",
            cpu * 1e3,
            kernel * 1e3,
            cpu / kernel,
            uploaded * 1e3,
            round_trip * 1e3,
            pairs
        );
    }

    println!(
        "\nThe readback column is the whole argument for finishing the migration.\n\
         A step that builds neighbours here and solves density on the host pays\n\
         it every substep, and it is larger than the CPU build it replaced at\n\
         every size in this table — including the smallest, where it is five\n\
         times worse than never having used the GPU at all. It is three stalls\n\
         rather than one (order, counts, lists); one would still cost 1.4 ms."
    );

    // What the sequential chain costs: the build is five kernels plus the
    // scan's, and nothing in it can start before the one before it finished.
    let world = fluid_of(0.05, 1.2);
    let f = &world.fluids[0];
    let mut grid = GpuGrid::new(&device, f.len());
    grid.upload(&device, &f.particles.position, f.params.smoothing_radius);
    let mut graph = Graph::new("spatial hash");
    grid.record(&mut graph);
    println!(
        "\nOne build is {} dispatches with a dependency depth of {} — every stage\n\
         needs the one before it, which is why the small sizes above are latency\n\
         and not throughput.",
        graph.len(),
        graph.depth()
    );
}
