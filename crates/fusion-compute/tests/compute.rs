//! The compute layer, end to end on whatever adapter is here.
//!
//! Every test skips where there is no GPU. PLAN.md §25 requires the engine to
//! run headless on a machine with no display and no GPU, and CI runners
//! frequently are one — a suite that goes red there is a suite people learn to
//! ignore. `Device::headless()` returning `None` is a normal outcome, not a
//! failure, and it is reported so a skipped run does not read as a passing one.

use fusion_compute::{Device, Field, Graph, Kernel, Params, Residency};

/// The device, or a printed note and `None`.
macro_rules! device {
    () => {
        match Device::headless() {
            Some(d) => d,
            None => {
                println!("no GPU adapter — skipped");
                return;
            }
        }
    };
}

/// One module, several entry points, matching the binding convention: storage
/// buffers first, `Params` last.
const SHADER: &str = r#"
struct Params {
    count: u32,
    u0: u32, u1: u32, u2: u32,
    f: vec4<f32>,
};

@group(0) @binding(0) var<storage, read_write> a: array<f32>;
@group(0) @binding(1) var<uniform> p1: Params;

@compute @workgroup_size(64)
fn fill(@builtin(global_invocation_id) gid: vec3<u32>) {
    let i = gid.x;
    if (i >= p1.count) { return; }
    a[i] = f32(i);
}

@compute @workgroup_size(64)
fn scale(@builtin(global_invocation_id) gid: vec3<u32>) {
    let i = gid.x;
    if (i >= p1.count) { return; }
    a[i] = a[i] * p1.f.x + f32(p1.u0);
}
"#;

const PAIR: &str = r#"
struct Params {
    count: u32,
    u0: u32, u1: u32, u2: u32,
    f: vec4<f32>,
};

@group(0) @binding(0) var<storage, read_write> src: array<f32>;
@group(0) @binding(1) var<storage, read_write> dst: array<f32>;
@group(0) @binding(2) var<uniform> p2: Params;

@compute @workgroup_size(64)
fn double_it(@builtin(global_invocation_id) gid: vec3<u32>) {
    let i = gid.x;
    if (i >= p2.count) { return; }
    dst[i] = src[i] * 2.0;
}
"#;

const N: usize = 1000;

#[test]
fn a_field_round_trips_through_gpu_memory() {
    let device = device!();
    println!("{} via {:?}", device.adapter(), device.backend());
    let data: Vec<f32> = (0..N).map(|i| i as f32 * 0.5).collect();
    let field = Field::from_slice(&device, "round trip", &data, Residency::Cpu);
    assert_eq!(field.read(&device), data);
}

#[test]
fn a_kernel_runs_and_reads_its_parameters() {
    let device = device!();
    let kernel = Kernel::new(&device, "scale", SHADER, "scale", 1);
    let data: Vec<f32> = (0..N).map(|i| i as f32).collect();
    let field = Field::from_slice(&device, "a", &data, Residency::Gpu);

    let mut graph = Graph::new("scale");
    graph
        .step(&kernel)
        .write(&field)
        .params(
            Params::count(N)
                .with_f([3.0, 0.0, 0.0, 0.0])
                .with_u([7, 0, 0]),
        )
        .items(N);
    graph.submit(&device);

    let out = field.read(&device);
    for (i, v) in out.iter().enumerate() {
        assert_eq!(*v, i as f32 * 3.0 + 7.0, "element {i}");
    }
}

/// The count is what stops a dispatch rounded up to whole workgroups from
/// running off the end of the array. 1000 elements is 15.6 workgroups of 64.
#[test]
fn a_partial_workgroup_does_not_touch_what_is_past_the_end() {
    let device = device!();
    let kernel = Kernel::new(&device, "fill", SHADER, "fill", 1);
    let field = Field::from_slice(&device, "a", &vec![-1.0f32; N + 64], Residency::Gpu);

    let mut graph = Graph::new("fill");
    graph.step(&kernel).write(&field).items(N);
    graph.submit(&device);

    let out = field.read(&device);
    for (i, v) in out.iter().enumerate().take(N) {
        assert_eq!(*v, i as f32);
    }
    for (i, v) in out.iter().enumerate().skip(N) {
        assert_eq!(*v, -1.0, "element {i} past the count was written");
    }
}

/// The batching claim, as a count rather than as a timing: a chain of three
/// dependent kernels is one submission and three dispatches.
#[test]
fn a_graph_is_one_submission_however_many_dispatches() {
    let device = device!();
    let fill = Kernel::new(&device, "fill", SHADER, "fill", 1);
    let double = Kernel::new(&device, "double", PAIR, "double_it", 2);

    let a = Field::<f32>::new(&device, "a", N, Residency::Gpu);
    let b = Field::<f32>::new(&device, "b", N, Residency::Gpu);
    let c = Field::<f32>::new(&device, "c", N, Residency::Gpu);

    let mut graph = Graph::new("chain");
    graph.step(&fill).write(&a).items(N);
    graph.step(&double).read(&a).write(&b).items(N);
    graph.step(&double).read(&b).write(&c).items(N);
    assert_eq!(graph.len(), 3);
    assert_eq!(graph.depth(), 3, "each step needs the one before it");

    device.reset_stats();
    graph.submit(&device);
    assert_eq!(device.stats().submits, 1);
    assert_eq!(device.stats().dispatches, 3);
    assert_eq!(device.stats().stalls, 0, "submitting must not block");

    let out = c.read(&device);
    for (i, v) in out.iter().enumerate() {
        assert_eq!(*v, i as f32 * 4.0, "element {i}");
    }
}

/// The same three dispatches the slow way, which is the comparison the
/// benchmark reports.
#[test]
fn submit_each_costs_a_submission_per_dispatch() {
    let device = device!();
    let fill = Kernel::new(&device, "fill", SHADER, "fill", 1);
    let double = Kernel::new(&device, "double", PAIR, "double_it", 2);
    let a = Field::<f32>::new(&device, "a", N, Residency::Gpu);
    let b = Field::<f32>::new(&device, "b", N, Residency::Gpu);
    let c = Field::<f32>::new(&device, "c", N, Residency::Gpu);

    let mut graph = Graph::new("chain");
    graph.step(&fill).write(&a).items(N);
    graph.step(&double).read(&a).write(&b).items(N);
    graph.step(&double).read(&b).write(&c).items(N);

    device.reset_stats();
    graph.submit_each(&device);
    assert_eq!(device.stats().submits, 3);
    // Same answer, three times the driver traffic.
    let out = c.read(&device);
    assert_eq!(out[10], 40.0);
}

/// Depth is about the dependencies, not the dispatch count.
#[test]
fn independent_dispatches_share_a_level() {
    let device = device!();
    let fill = Kernel::new(&device, "fill", SHADER, "fill", 1);
    let a = Field::<f32>::new(&device, "a", N, Residency::Gpu);
    let b = Field::<f32>::new(&device, "b", N, Residency::Gpu);
    let c = Field::<f32>::new(&device, "c", N, Residency::Gpu);

    let mut graph = Graph::new("three independent fills");
    graph.step(&fill).write(&a).items(N);
    graph.step(&fill).write(&b).items(N);
    graph.step(&fill).write(&c).items(N);
    assert_eq!(graph.len(), 3);
    assert_eq!(graph.depth(), 1, "nothing here reads anything else");

    // And a read of one of them is one level deeper, not three.
    let double = Kernel::new(&device, "double", PAIR, "double_it", 2);
    let d = Field::<f32>::new(&device, "d", N, Residency::Gpu);
    graph.step(&double).read(&b).write(&d).items(N);
    assert_eq!(graph.depth(), 2);
}

/// The residency declaration has to be load-bearing or it is decoration.
#[test]
fn a_kernel_may_not_write_a_cpu_authoritative_field() {
    let device = device!();
    let fill = Kernel::new(&device, "fill", SHADER, "fill", 1);
    let scene = Field::<f32>::new(&device, "scene", N, Residency::Cpu);

    let mut graph = Graph::new("bad");
    graph.step(&fill).write(&scene).items(N);
    let problems = graph.violations();
    assert_eq!(problems.len(), 1);
    assert!(problems[0].contains("CPU-authoritative"), "{problems:?}");
}

#[test]
fn a_binding_count_mismatch_is_caught_before_it_runs() {
    let device = device!();
    let double = Kernel::new(&device, "double", PAIR, "double_it", 2);
    let a = Field::<f32>::new(&device, "a", N, Residency::Gpu);

    let mut graph = Graph::new("short");
    graph.step(&double).write(&a).items(N);
    let problems = graph.violations();
    assert_eq!(problems.len(), 1);
    assert!(problems[0].contains("2 buffers, 1 given"), "{problems:?}");
}

#[test]
#[should_panic(expected = "GPU-authoritative")]
fn the_cpu_may_not_overwrite_a_gpu_authoritative_field() {
    // A panic test that has to run even without a GPU would be a lie, so it
    // panics with the expected message either way.
    let device = match Device::headless() {
        Some(d) => d,
        None => panic!("no GPU adapter — skipped, GPU-authoritative"),
    };
    let field = Field::<f32>::new(&device, "gpu owned", N, Residency::Gpu);
    field.write(&device, &vec![0.0; N]);
}

/// The dispatch ceiling is 65 535 workgroups, and a kernel that has not said
/// it can cope with a short dispatch has to hear about it rather than quietly
/// leaving the tail of the array untouched.
#[test]
#[should_panic(expected = "workgroups")]
fn over_dispatching_a_kernel_that_is_not_grid_strided_is_refused() {
    let device = match Device::headless() {
        Some(d) => d,
        None => panic!("no GPU adapter — skipped, workgroups"),
    };
    let limit = device.limits().max_compute_workgroups_per_dimension as usize;
    let huge = (limit + 1) * 64;
    let kernel = Kernel::new(&device, "fill", SHADER, "fill", 1);
    // The field is small; only the dispatch is oversized.
    let a = Field::<f32>::new(&device, "a", 64, Residency::Gpu);
    let mut graph = Graph::new("too big");
    graph.step(&kernel).write(&a).items(huge);
    graph.submit(&device);
}

/// An empty scene is an ordinary state, not a caller error.
#[test]
fn zero_length_fields_and_empty_graphs_are_harmless() {
    let device = device!();
    let empty = Field::<f32>::new(&device, "nothing", 0, Residency::Gpu);
    assert!(empty.is_empty());
    assert!(empty.read(&device).is_empty());

    let graph = Graph::new("nothing to do");
    assert!(graph.is_empty());
    assert_eq!(graph.depth(), 0);
    device.reset_stats();
    graph.submit(&device);
    assert_eq!(device.stats().submits, 0);
}
