//! What the GPU costs, and what batching saves.
//!
//! `cargo run --release -p fusion-compute --example gpu_bench`
//!
//! PLAN.md §24 makes one claim above all the others — "Minimizing CPU↔GPU sync
//! stalls is the single biggest real-world GPU performance factor here" — and
//! v0.6 built [`Graph`] on the strength of it. This is the measurement. It is
//! a separate binary from `fusion-demos`'s `bench` on purpose: §25 requires the
//! simulation core to run with no display and no GPU, and the headless
//! benchmark proves that by not linking any of this.
//!
//! Three numbers, in the order they matter:
//!
//! 1. **Batched against unbatched.** The same dispatches, one submission or
//!    one submission each.
//! 2. **What a readback costs.** The same step, with and without asking the
//!    CPU for the answer at the end of it.
//! 3. **Where the GPU starts winning.** Throughput against element count, so
//!    the crossover §24 puts "around 10⁵–10⁷ particles" is a measurement
//!    rather than a guess.

use std::time::Instant;

use fusion_compute::{Device, Field, Graph, Kernel, Params, Residency};

/// Two kernels with the shape a particle step actually has: read one array,
/// write another, a few flops per element.
const SHADER: &str = r#"
struct Params {
    count: u32,
    u0: u32, u1: u32, u2: u32,
    f: vec4<f32>,
};

@group(0) @binding(0) var<storage, read_write> velocity: array<vec4<f32>>;
@group(0) @binding(1) var<uniform> p1: Params;

@compute @workgroup_size(64)
fn accelerate(@builtin(global_invocation_id) gid: vec3<u32>) {
    let i = gid.x;
    if (i >= p1.count) { return; }
    velocity[i] = velocity[i] + vec4<f32>(0.0, -9.81 * p1.f.x, 0.0, 0.0);
}
"#;

const PREDICT: &str = r#"
struct Params {
    count: u32,
    u0: u32, u1: u32, u2: u32,
    f: vec4<f32>,
};

@group(0) @binding(0) var<storage, read_write> velocity: array<vec4<f32>>;
@group(0) @binding(1) var<storage, read_write> position: array<vec4<f32>>;
@group(0) @binding(2) var<uniform> p2: Params;

@compute @workgroup_size(64)
fn predict(@builtin(global_invocation_id) gid: vec3<u32>) {
    let i = gid.x;
    if (i >= p2.count) { return; }
    position[i] = position[i] + velocity[i] * p2.f.x;
}
"#;

/// Best of this many, for the reason `bench` gives: the minimum is the closest
/// thing to the cost with nothing else in the way.
const REPEATS: usize = 5;

fn best(repeats: usize, mut f: impl FnMut()) -> f64 {
    let mut best = f64::INFINITY;
    for _ in 0..repeats {
        let start = Instant::now();
        f();
        best = best.min(start.elapsed().as_secs_f64());
    }
    best
}

fn main() {
    let Some(device) = Device::headless() else {
        println!("no GPU adapter on this machine — nothing to measure");
        return;
    };
    println!("Fusion 0.6 — GPU compute benchmark");
    println!("adapter: {} via {:?}\n", device.adapter(), device.backend());

    let accelerate = Kernel::new(&device, "accelerate", SHADER, "accelerate", 1);
    let predict = Kernel::new(&device, "predict", PREDICT, "predict", 2);

    // ---- 1. batching ----
    //
    // A step of twelve dispatches: six substeps of accelerate-then-predict,
    // which is the shape of the fluid's substep loop and roughly its length.
    const N: usize = 100_000;
    const FRAMES: usize = 200;
    let velocity = Field::<[f32; 4]>::new(&device, "velocity", N, Residency::Gpu);
    let position = Field::<[f32; 4]>::new(&device, "position", N, Residency::Gpu);

    let mut graph = Graph::new("step");
    for _ in 0..6 {
        graph
            .step(&accelerate)
            .write(&velocity)
            .params(Params::count(N).with_f([1.0 / 240.0, 0.0, 0.0, 0.0]))
            .items(N);
        graph
            .step(&predict)
            .read(&velocity)
            .write(&position)
            .params(Params::count(N).with_f([1.0 / 240.0, 0.0, 0.0, 0.0]))
            .items(N);
    }
    println!(
        "one step: {} dispatches over {N} elements, dependency depth {}",
        graph.len(),
        graph.depth()
    );

    let batched = best(REPEATS, || {
        for _ in 0..FRAMES {
            graph.submit(&device);
        }
        device.wait();
    }) / FRAMES as f64;
    let unbatched = best(REPEATS, || {
        for _ in 0..FRAMES {
            graph.submit_each(&device);
        }
        device.wait();
    }) / FRAMES as f64;

    println!("\nsubmission strategy, {FRAMES} steps, best of {REPEATS}");
    println!("  {:<34} {:>10} {:>10}", "", "ms/step", "steps/s");
    println!(
        "  {:<34} {:>10.4} {:>10.0}",
        "one submission per step",
        batched * 1e3,
        1.0 / batched
    );
    println!(
        "  {:<34} {:>10.4} {:>10.0}",
        "one submission per dispatch",
        unbatched * 1e3,
        1.0 / unbatched
    );
    println!(
        "  {:<34} {:>10.2}x",
        "batching is worth",
        unbatched / batched
    );

    // ---- 2. the cost of asking ----
    let readback = best(REPEATS, || {
        for _ in 0..FRAMES {
            graph.submit(&device);
            let _ = position.read(&device);
        }
    }) / FRAMES as f64;
    println!("\nreading the answer back every step");
    println!(
        "  {:<34} {:>10.4} ms/step  ({:.1}x the step)",
        "step + full readback",
        readback * 1e3,
        readback / batched
    );
    println!(
        "  {:<34} {:>10.4} ms",
        "the stall alone",
        (readback - batched) * 1e3
    );
    // Is that the bytes or the round trip? One element, same machinery.
    let one = Field::<[f32; 4]>::new(&device, "one", 1, Residency::Gpu);
    let tiny = best(REPEATS, || {
        for _ in 0..FRAMES {
            graph.submit(&device);
            let _ = one.read(&device);
        }
    }) / FRAMES as f64;
    println!(
        "  {:<34} {:>10.4} ms/step  ({:.0} MB -> 16 B)",
        "step + one-element readback",
        tiny * 1e3,
        (N * 16) as f64 / 1e6
    );
    println!(
        "  Reading 16 bytes costs {:.0}% of reading {:.0} MB, so the price is the\n\
         \x20 round trip and not the transfer. Worth stating on a machine with\n\
         \x20 unified memory: §25 notes that Apple Silicon has no copy to pay for,\n\
         \x20 and this is the half of the stall that survives that.",
        (tiny - batched) / (readback - batched) * 100.0,
        (N * 16) as f64 / 1e6
    );

    // ---- 3. where it starts winning ----
    println!("\nthroughput against size, one step of 12 dispatches");
    println!(
        "  {:<12} {:>12} {:>12} {:>14}",
        "elements", "ms/step", "steps/s", "ns/element"
    );
    for n in [1_000usize, 10_000, 100_000, 1_000_000, 4_000_000] {
        let velocity = Field::<[f32; 4]>::new(&device, "velocity", n, Residency::Gpu);
        let position = Field::<[f32; 4]>::new(&device, "position", n, Residency::Gpu);
        let mut graph = Graph::new("step");
        for _ in 0..6 {
            graph
                .step(&accelerate)
                .write(&velocity)
                .params(Params::count(n).with_f([1.0 / 240.0, 0.0, 0.0, 0.0]))
                .items(n);
            graph
                .step(&predict)
                .read(&velocity)
                .write(&position)
                .params(Params::count(n).with_f([1.0 / 240.0, 0.0, 0.0, 0.0]))
                .items(n);
        }
        let frames = 100;
        let seconds = best(REPEATS, || {
            for _ in 0..frames {
                graph.submit(&device);
            }
            device.wait();
        }) / frames as f64;
        println!(
            "  {n:<12} {:>12.4} {:>12.0} {:>14.2}",
            seconds * 1e3,
            1.0 / seconds,
            seconds * 1e9 / n as f64
        );
    }
    println!(
        "  A flat ns/element column means the dispatch is not the cost yet — the\n\
         \x20 submission is. Where it starts falling is where the hardware is\n\
         \x20 actually being used, and that is the crossover §24 is talking about."
    );
}
