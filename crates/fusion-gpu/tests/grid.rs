//! The GPU grid against the CPU one it has to agree with.
//!
//! Skips where there is no adapter, like everything else that needs a GPU.

use fusion_compute::Device;
use fusion_core::{FluidDesc, FluidParams, World};
use fusion_gpu::{GpuGrid, Scan, MAX_NEIGHBOURS};
use fusion_math::{vec3, DVec3};
use fusion_solver::fluid;
use fusion_solver::FluidScratch;

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

/// A deterministic cloud, so a failure is reproducible.
fn cloud(n: usize, extent: f64) -> Vec<DVec3> {
    let mut seed = 0x9e37_79b9_u32;
    let mut next = move || {
        seed ^= seed << 13;
        seed ^= seed >> 17;
        seed ^= seed << 5;
        (seed >> 8) as f64 / 16_777_216.0
    };
    (0..n)
        .map(|_| DVec3::new(next() * extent, next() * extent, next() * extent))
        .collect()
}

#[test]
fn the_scan_matches_an_exclusive_prefix_sum() {
    let device = device!();
    // Lengths either side of the block size, and a few blocks deep, so the
    // recursion and the partial tail both run.
    for len in [1usize, 7, 255, 256, 257, 1000, 65_536, 300_000] {
        let scan = Scan::new(&device, len);
        let data: Vec<u32> = (0..len).map(|i| (i % 7) as u32).collect();
        let field = fusion_compute::Field::from_slice(
            &device,
            "data",
            &data,
            fusion_compute::Residency::Gpu,
        );

        let mut graph = fusion_compute::Graph::new("scan");
        scan.record(&mut graph, &field, len);
        graph.submit(&device);

        let got = field.read(&device);
        let mut running = 0u32;
        for (i, v) in data.iter().enumerate() {
            assert_eq!(got[i], running, "len {len}, element {i}");
            running += v;
        }
    }
}

/// The counting sort has to produce the same permutation the CPU's does:
/// bucket order, ascending particle index within each bucket.
#[test]
fn the_sorted_order_matches_the_cpu_counting_sort() {
    let device = device!();
    let points = cloud(20_000, 3.0);
    let cell = 0.2f32;

    let mut gpu = GpuGrid::new(&device, points.len());
    gpu.upload(&device, &points, cell);
    gpu.build(&device);

    let mut cpu = fusion_collide::SpatialHash::new();
    cpu.build(&points, cell);

    assert_eq!(gpu.read_sorted(&device), cpu.sorted_indices());
}

/// Twice through, same answer. The atomic scatter is order-dependent by
/// nature; `sort_buckets` is what makes this true, and without it this test is
/// the one that fails.
#[test]
fn two_builds_of_the_same_points_agree() {
    let device = device!();
    let points = cloud(50_000, 4.0);
    let mut gpu = GpuGrid::new(&device, points.len());
    gpu.upload(&device, &points, 0.25);

    gpu.build(&device);
    let first = gpu.read_sorted(&device);
    for _ in 0..8 {
        gpu.build(&device);
        assert_eq!(gpu.read_sorted(&device), first);
    }
}

/// The claim that matters: the same neighbour lists the fluid solver builds,
/// particle for particle, in the same order.
#[test]
fn the_neighbour_lists_are_identical_to_the_solvers() {
    let device = device!();

    // A real fluid rather than a cloud, settled a little so the distribution
    // is the one a solve actually sees.
    let mut world = World::new();
    world.add_fluid(
        &FluidDesc::block(vec3(1.0, 0.6, 1.0))
            .at(vec3(0.4, 0.9, -0.3))
            .with_params(FluidParams::water_at(0.05)),
    );
    let fluid = &world.fluids[0];
    let n = fluid.len();
    assert!(n > 3_000, "want a scene worth testing, got {n}");

    let mut scratch = FluidScratch::new();
    scratch.prepare(fluid);
    fluid::build_neighbours(fluid, &mut scratch);

    let mut gpu = GpuGrid::new(&device, n);
    gpu.upload(
        &device,
        &fluid.particles.position,
        fluid.params.smoothing_radius,
    );
    gpu.build(&device);
    let lists = gpu.read_lists(&device);

    assert_eq!(lists.len(), n);
    let mut pairs = 0usize;
    for (i, got) in lists.iter().enumerate() {
        let expected = scratch.neighbours(i);
        assert_eq!(
            got,
            expected,
            "particle {i}: {} neighbours on the GPU, {} on the CPU",
            got.len(),
            expected.len()
        );
        pairs += expected.len();
    }
    assert!(
        pairs > n * 20,
        "want a real neighbourhood, got {pairs} pairs"
    );
    println!("{n} particles, {pairs} pairs, identical");
}

/// The cap is a guard against a collapsed scene, and both paths have to hit it
/// at the same place.
#[test]
fn a_collapsed_cloud_truncates_at_the_same_cap() {
    let device = device!();
    // Everything inside one smoothing radius: every particle sees every other.
    let points = cloud(400, 0.01);
    let cell = 1.0f32;

    let mut gpu = GpuGrid::new(&device, points.len());
    gpu.upload(&device, &points, cell);
    gpu.build(&device);
    let lists = gpu.read_lists(&device);

    for (i, list) in lists.iter().enumerate() {
        assert_eq!(list.len(), MAX_NEIGHBOURS, "particle {i} was not capped");
    }
}

#[test]
fn an_empty_grid_is_harmless() {
    let device = device!();
    let mut gpu = GpuGrid::new(&device, 1000);
    gpu.upload(&device, &[], 0.1);
    assert!(gpu.is_empty());
    gpu.build(&device);
    assert!(gpu.read_sorted(&device).is_empty());
}
