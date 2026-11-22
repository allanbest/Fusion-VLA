//! The fluid's spatial hash and neighbour sweep, on the GPU.

use fusion_compute::{Device, Field, Graph, Kernel, Params, Residency};
use fusion_math::DVec3;

use crate::Scan;

/// Neighbours kept per particle, and the stride of the neighbour array.
///
/// The same cap the CPU uses, so the two produce the same lists. It is a fixed
/// stride here rather than a packed list with offsets, and that is the one
/// place the GPU version costs more memory than the CPU one: 512 bytes a
/// particle whether it has thirty-three neighbours or none. Packing would need
/// a count pass, a scan and a fill pass — two sweeps instead of one — and the
/// sweep is the expensive part.
pub const MAX_NEIGHBOURS: usize = 128;

/// A uniform spatial hash over particle positions, built and queried on the
/// GPU.
///
/// ## What it agrees with
///
/// The neighbour lists this produces are **identical** to
/// `fusion_solver::fluid::build_neighbours` — same particles, same order, for
/// the same input — and a test asserts exactly that over a randomized cloud.
/// Not "equivalent": PLAN.md §34 guarantees bit-identical replay on the same
/// build, and a fluid whose neighbour order depends on which device solved it
/// would be two fluids wearing one name.
///
/// Getting there costs one kernel that would otherwise not exist. A GPU
/// scatter allocates slots with an atomic, so bucket contents come out in
/// whatever order the hardware happened to run — different between runs of the
/// same binary. The `sort_buckets` kernel puts each bucket back into ascending
/// particle order afterwards, which is what the CPU's counting sort produces
/// for free by inserting in index order.
///
/// ## What it costs to use from the CPU
///
/// Everything here stays on the device. Reading the lists back is a stall, and
/// `gpu_bench` prices one at fifteen times a step — so a solver that builds
/// neighbours here and solves density on the host has spent more on the
/// crossing than the sweep ever cost. The lists are meant to be consumed by
/// the next kernel, and [`GpuGrid::read_lists`] exists for tests and
/// diagnostics rather than for a step.
pub struct GpuGrid {
    histogram: Kernel,
    scatter: Kernel,
    sort_buckets: Kernel,
    sweep: Kernel,
    scan: Scan,

    /// Local `f32` positions, uploaded from the host. CPU-authoritative: the
    /// scene is authored there.
    positions: Field<[f32; 4]>,
    bucket_of: Field<u32>,
    /// Histogram, then bucket offsets after the scan. `buckets + 1` entries.
    starts: Field<u32>,
    cursor: Field<u32>,
    sorted: Field<u32>,
    sorted_pos: Field<[f32; 4]>,
    counts: Field<u32>,
    lists: Field<u32>,

    capacity: usize,
    buckets: usize,
    len: usize,
    origin: DVec3,
    cell_size: f32,
}

impl GpuGrid {
    /// A grid sized for up to `capacity` particles.
    ///
    /// The bucket table is fixed at construction from the capacity, unlike the
    /// CPU's which is resized per build. Reallocating a GPU buffer every
    /// substep is the kind of thing that does not show up in a profile as
    /// itself; it shows up as the driver being slow.
    pub fn new(device: &Device, capacity: usize) -> Self {
        let source = include_str!("shaders/grid.wgsl");
        let buckets = (2 * capacity.max(1)).next_power_of_two().max(64);
        Self {
            histogram: Kernel::new(device, "histogram", source, "histogram", 3).strided(),
            scatter: Kernel::new(device, "scatter", source, "scatter", 6).strided(),
            sort_buckets: Kernel::new(device, "sort_buckets", source, "sort_buckets", 3).strided(),
            sweep: Kernel::new(device, "sweep", source, "sweep", 5).strided(),
            scan: Scan::new(device, buckets + 1),

            positions: Field::new(device, "positions", capacity, Residency::Cpu),
            bucket_of: Field::new(device, "bucket of", capacity, Residency::Gpu),
            starts: Field::new(device, "bucket starts", buckets + 1, Residency::Gpu),
            cursor: Field::new(device, "bucket cursor", buckets, Residency::Gpu),
            sorted: Field::new(device, "sorted", capacity, Residency::Gpu),
            sorted_pos: Field::new(device, "sorted positions", capacity, Residency::Gpu),
            counts: Field::new(device, "neighbour counts", capacity, Residency::Gpu),
            lists: Field::new(
                device,
                "neighbours",
                capacity * MAX_NEIGHBOURS,
                Residency::Gpu,
            ),

            capacity,
            buckets,
            len: 0,
            origin: DVec3::ZERO,
            cell_size: 0.0,
        }
    }

    pub fn capacity(&self) -> usize {
        self.capacity
    }

    pub fn buckets(&self) -> usize {
        self.buckets
    }

    pub fn len(&self) -> usize {
        self.len
    }

    pub fn is_empty(&self) -> bool {
        self.len == 0
    }

    /// The local frame the uploaded positions are in.
    pub fn origin(&self) -> DVec3 {
        self.origin
    }

    /// Snap the origin and upload the points, in the grid's local `f32` frame.
    ///
    /// The origin is chosen and the narrowing is done on the host, for the
    /// reason `SpatialHash` gives: the store is `f64` so a scene can sit
    /// kilometres from the world origin, and neighbour geometry has never
    /// needed world coordinates. Doing it here rather than in a kernel also
    /// keeps the answer identical to the CPU's, which a parallel min-reduction
    /// over `f64` would not be worth risking.
    pub fn upload(&mut self, device: &Device, points: &[DVec3], cell_size: f32) {
        assert!(
            points.len() <= self.capacity,
            "grid built for {} particles, given {}",
            self.capacity,
            points.len()
        );
        self.len = points.len();
        self.cell_size = cell_size.max(1e-6);
        if points.is_empty() {
            return;
        }

        let mut min = points[0];
        for p in points {
            min = min.min(*p);
        }
        let cell = self.cell_size as f64;
        self.origin = DVec3::new(
            (min.x / cell).floor() * cell,
            (min.y / cell).floor() * cell,
            (min.z / cell).floor() * cell,
        );

        let local: Vec<[f32; 4]> = points
            .iter()
            .map(|p| {
                let d = *p - self.origin;
                [d.x as f32, d.y as f32, d.z as f32, 0.0]
            })
            .collect();
        self.positions.write(device, &local);
    }

    /// Record the whole build — histogram, scan, scatter, order, sweep.
    ///
    /// Five kernels plus the scan's, in one graph, so the caller submits once.
    /// The dependency chain is genuinely sequential: nothing can be scattered
    /// before the offsets are known and nothing can be swept before the
    /// buckets are ordered. `Graph::depth` reports it, and it is the reason
    /// this is latency-bound at small particle counts.
    pub fn record<'a>(&'a self, graph: &mut Graph<'a>) {
        if self.len == 0 {
            return;
        }
        let common = Params::count(self.len)
            .with_u([
                (self.buckets - 1) as u32,
                self.buckets as u32,
                MAX_NEIGHBOURS as u32,
            ])
            .with_f([
                1.0 / self.cell_size,
                self.cell_size * self.cell_size,
                0.0,
                0.0,
            ]);

        graph.clear(&self.starts);
        graph.clear(&self.cursor);
        graph
            .step(&self.histogram)
            .read(&self.positions)
            .write(&self.bucket_of)
            .write(&self.starts)
            .params(common)
            .items(self.len);

        // The scan turns counts into offsets. It runs over `buckets + 1`
        // entries: the histogram writes bucket `b` at index `b`, so an
        // exclusive scan puts the number of particles in buckets below `b` at
        // index `b`, and the total at index `buckets`.
        self.scan.record(graph, &self.starts, self.buckets + 1);

        graph
            .step(&self.scatter)
            .read(&self.positions)
            .read(&self.bucket_of)
            .read(&self.starts)
            .write(&self.cursor)
            .write(&self.sorted)
            .write(&self.sorted_pos)
            .params(common)
            .items(self.len);
        graph
            .step(&self.sort_buckets)
            .read(&self.starts)
            .write(&self.sorted)
            .write(&self.sorted_pos)
            .params(Params {
                count: self.buckets as u32,
                ..common
            })
            .items(self.buckets);
        graph
            .step(&self.sweep)
            .read(&self.starts)
            .read(&self.sorted)
            .read(&self.sorted_pos)
            .write(&self.counts)
            .write(&self.lists)
            .params(common)
            .items(self.len);
    }

    /// Build the grid and submit it, as one submission.
    pub fn build(&self, device: &Device) {
        let mut graph = Graph::new("spatial hash");
        self.record(&mut graph);
        graph.submit(device);
    }

    /// Point indices in bucket order — the GPU's `sorted` array.
    ///
    /// A stall. For tests and diagnostics.
    pub fn read_sorted(&self, device: &Device) -> Vec<u32> {
        self.sorted.read(device)[..self.len].to_vec()
    }

    /// Neighbour lists, keyed by *particle* index rather than by slot.
    ///
    /// A stall, and a large one — the lists are 512 bytes a particle. For
    /// tests and diagnostics; see the type's note on why a step must not do
    /// this.
    pub fn read_lists(&self, device: &Device) -> Vec<Vec<u32>> {
        let sorted = self.read_sorted(device);
        let counts = self.counts.read(device);
        let flat = self.lists.read(device);
        let mut out = vec![Vec::new(); self.len];
        for (slot, &particle) in sorted.iter().enumerate() {
            let from = slot * MAX_NEIGHBOURS;
            let n = counts[slot] as usize;
            out[particle as usize] = flat[from..from + n].to_vec();
        }
        out
    }
}
