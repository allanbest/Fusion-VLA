//! Exclusive prefix sum, of any length.

use fusion_compute::{Device, Field, Graph, Kernel, Params, Residency};

/// Elements one workgroup scans. Matches `@workgroup_size(256)` in the shader
/// and the `array<u32, 256>` of workgroup memory it uses.
pub const BLOCK: usize = 256;

/// An exclusive prefix sum over a `u32` field, in place.
///
/// Sized at construction, because the intermediate block-sum buffers are
/// allocated once and reused. A scan over 262 144 buckets needs 1024 block
/// sums, and a scan over *those* needs 4, and a scan over those is one
/// workgroup and the recursion stops — three levels, three buffers, allocated
/// once instead of once a substep.
pub struct Scan {
    block: Kernel,
    add: Kernel,
    /// Block sums per level. `levels[0]` has one entry per block of the input.
    levels: Vec<Field<u32>>,
}

impl Scan {
    /// A scan able to handle inputs up to `capacity` elements.
    pub fn new(device: &Device, capacity: usize) -> Self {
        let source = include_str!("shaders/scan.wgsl");
        let mut levels = Vec::new();
        let mut len = capacity.max(1).div_ceil(BLOCK);
        loop {
            levels.push(Field::new(
                device,
                "scan block sums",
                len.max(1),
                Residency::Gpu,
            ));
            if len <= 1 {
                break;
            }
            len = len.div_ceil(BLOCK);
        }
        Self {
            block: Kernel::sized(device, "scan_block", source, "scan_block", 2, BLOCK as u32),
            add: Kernel::sized(
                device,
                "add_offsets",
                source,
                "add_offsets",
                2,
                BLOCK as u32,
            ),
            levels,
        }
    }

    /// Largest input this scan was built for.
    pub fn capacity(&self) -> usize {
        self.levels[0].len() * BLOCK
    }

    /// Record the dispatches that scan `data[..len]` in place.
    ///
    /// Nothing is submitted — the caller's graph decides when, which is the
    /// point of recording rather than running: a grid build is a scan
    /// sandwiched between a histogram and a scatter, and all of it should be
    /// one submission.
    pub fn record<'a>(&'a self, graph: &mut Graph<'a>, data: &'a Field<u32>, len: usize) {
        assert!(
            len <= self.capacity(),
            "scan built for {} elements, asked for {len}",
            self.capacity()
        );
        self.record_level(graph, data, len, 0);
    }

    fn record_level<'a>(
        &'a self,
        graph: &mut Graph<'a>,
        data: &'a Field<u32>,
        len: usize,
        level: usize,
    ) {
        if len == 0 {
            return;
        }
        let blocks = len.div_ceil(BLOCK);
        let sums = &self.levels[level];
        graph
            .step(&self.block)
            .write(data)
            .write(sums)
            .params(Params::count(len))
            .items(len);
        if blocks > 1 {
            // The block totals are themselves an array wanting a running total.
            self.record_level(graph, sums, blocks, level + 1);
            graph
                .step(&self.add)
                .write(data)
                .read(sums)
                .params(Params::count(len))
                .items(len);
        }
    }
}
