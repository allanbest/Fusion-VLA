//! Broad phase: a uniform grid, for when there are a great many bodies of
//! about the same size.
//!
//! [`crate::broad::SweepAndPrune`] is the right structure for the scenes
//! PLAN.md §10 had in mind — a few thousand bodies of wildly different sizes,
//! moving coherently. It is the wrong one for a hundred thousand, and the
//! reason is not the sort but the sweep. A one-axis SAP compares each body
//! against every interval still open at its own minimum, and for a fixed
//! density that active set grows as n^(2/3): ten times the bodies is about
//! four and a half times the active list, so the sweep costs forty-odd times
//! as much. Measured on 100 000 spheres it was 542 ms of a 1 049 ms step —
//! more than half the frame, in the cheapest phase of the pipeline.
//!
//! This is the same argument [`crate::grid`] makes for fluid particles, one
//! layer up. Bodies of similar size have one cell size that fits them all, and
//! then a pair query is a walk over the cells a body actually occupies rather
//! than a scan of everything open along an axis.
//!
//! ## The floor problem
//!
//! Real scenes are not uniform: the 100 000-sphere benchmark is 100 000
//! spheres *and a 420-metre floor*. A uniform grid handles that body as badly
//! as it is possible to handle anything — it spans about 180 000 cells, and
//! inserting it into each of them would cost more than the sweep it replaced.
//!
//! So bodies are split by how many cells they cover. The small ones go in the
//! grid. The oversized ones — usually a floor, a wall, a container, and there
//! are rarely more than a handful — are tested against everything, brute
//! force, which is one AABB test per body and cheap enough to disappear. Where
//! that assumption fails, and a scene turns out to be mostly oversized bodies,
//! the pipeline says so and uses the sweep instead: see
//! [`Grid::suits`].
//!
//! ## The same pairs, in the same order
//!
//! Every broad phase here emits a **canonically sorted, deduplicated** list.
//! Two bodies can share more than one cell, so the grid finds some pairs
//! several times; and the order cells are visited in has nothing to do with
//! the order a sweep would visit them. Both of those would otherwise reach the
//! solver, which is Gauss-Seidel and answers differently when constraints
//! arrive in a different order.
//!
//! Sorting makes the pair list a property of the *scene* rather than of the
//! structure that found it, so switching structures is invisible to physics —
//! a scene does not change behaviour because it crossed a body count. It is
//! also what lets one test assert the grid and the sweep produce *identical*
//! lists rather than equivalent ones, which is the standard v0.6 part 3 set
//! for the GPU neighbour build.

use fusion_math::{Aabb, Vec3};

use crate::broad::BroadPair;

/// A body covering more cells than this is handled outside the grid.
///
/// Eight cells is a body up to twice the cell size on every axis, which is
/// what "about the same size" means here. The bound is on the *product*, so a
/// long thin plank is allowed to be long as long as it is thin.
const MAX_CELLS_PER_BODY: usize = 8;

/// Uniform-grid broad phase over body bounds.
#[derive(Clone, Debug, Default)]
pub struct Grid {
    cell: f32,
    inv_cell: f32,
    /// Hash table of bucket start offsets, `buckets.len()` is a power of two
    /// plus one.
    bucket_start: Vec<u32>,
    /// Body indices grouped by bucket, ascending within each bucket.
    sorted: Vec<u32>,
    /// Scratch: bodies too big for the grid.
    oversized: Vec<u32>,
    /// Scratch: `(bucket, body)` before the counting sort.
    entries: Vec<(u32, u32)>,
}

impl Grid {
    /// An empty grid. Its tables are sized on the first build and reused.
    pub fn new() -> Self {
        Self::default()
    }

    /// Cell size chosen for the last build.
    pub fn cell_size(&self) -> f32 {
        self.cell
    }

    /// Bodies that were too large for the grid on the last build.
    pub fn oversized_count(&self) -> usize {
        self.oversized.len()
    }

    /// Whether a grid is the right structure for this set of bounds.
    ///
    /// Two ways it is not. Below a few hundred bodies the sweep is already
    /// fast, and the grid's table build is a fixed cost it cannot earn back.
    /// And a scene that is mostly oversized bodies is a scene with no uniform
    /// cell size to find, which is exactly the heterogeneous case §10 chose
    /// sweep-and-prune for.
    pub fn suits(aabbs: &[Aabb]) -> bool {
        // Measured, not guessed. On a settled sphere cloud the grid overtakes
        // the sweep between 256 and 512 bodies — 0.75x at 256, 1.29x at 512,
        // 2.4x at 1024 — and the losing side of that crossover costs
        // microseconds while the winning side goes on to save half a second.
        // The table is in `docs/STATUS.md` and comes from `bench broad`.
        const MIN_BODIES: usize = 512;
        if aabbs.len() < MIN_BODIES {
            return false;
        }
        let cell = cell_size(aabbs);
        if !(cell.is_finite() && cell > 0.0) {
            return false;
        }
        let inv = 1.0 / cell;
        let oversized = aabbs.iter().filter(|b| cells_covered(b, inv) == 0).count();
        oversized * 16 <= aabbs.len()
    }

    /// Collect every pair of overlapping bounds, sorted and deduplicated.
    ///
    /// Produces exactly what [`crate::broad::find_pairs_brute_force`] does, in
    /// the same order.
    pub fn find_pairs(&mut self, aabbs: &[Aabb], dynamic: &[bool], out: &mut Vec<BroadPair>) {
        out.clear();
        let n = aabbs.len();
        if n < 2 {
            return;
        }

        self.cell = cell_size(aabbs);
        if !(self.cell.is_finite() && self.cell > 0.0) {
            crate::broad::find_pairs_brute_force(aabbs, dynamic, out);
            out.sort_unstable();
            return;
        }
        self.inv_cell = 1.0 / self.cell;

        // --- classify, and lay out the (bucket, body) entries ---
        //
        // A body is entered once per cell it covers. Two bodies whose bounds
        // overlap must share at least one cell, so no pair can be missed;
        // several shared cells mean a pair found several times, which the
        // final dedup removes.
        self.oversized.clear();
        self.entries.clear();
        let buckets = (n * 2).next_power_of_two();
        let mask = (buckets - 1) as u32;
        for (i, bounds) in aabbs.iter().enumerate() {
            let covered = cells_covered(bounds, self.inv_cell);
            if covered == 0 {
                self.oversized.push(i as u32);
                continue;
            }
            let lo = cell_of(bounds.min, self.inv_cell);
            let hi = cell_of(bounds.max, self.inv_cell);
            for z in lo[2]..=hi[2] {
                for y in lo[1]..=hi[1] {
                    for x in lo[0]..=hi[0] {
                        self.entries.push((hash(x, y, z) & mask, i as u32));
                    }
                }
            }
        }

        // --- counting sort into buckets ---
        //
        // The same structure `grid::SpatialHash` uses and for the same two
        // reasons: no allocation per bucket, and ascending body order within
        // every bucket on every run. A `HashMap` here would be seeded randomly
        // and would take deterministic replay down with it.
        self.bucket_start.clear();
        self.bucket_start.resize(buckets + 1, 0);
        for &(bucket, _) in &self.entries {
            self.bucket_start[bucket as usize + 1] += 1;
        }
        for b in 0..buckets {
            self.bucket_start[b + 1] += self.bucket_start[b];
        }
        self.sorted.clear();
        self.sorted.resize(self.entries.len(), 0);
        let mut cursor: Vec<u32> = self.bucket_start[..buckets].to_vec();
        for &(bucket, body) in &self.entries {
            let slot = &mut cursor[bucket as usize];
            self.sorted[*slot as usize] = body;
            *slot += 1;
        }

        // --- pairs within each bucket ---
        //
        // Bucket, not cell: two distant cells can hash to the same bucket, and
        // the pair they invent is rejected by the overlap test every candidate
        // has to pass anyway. Collisions cost work, never correctness.
        for b in 0..buckets {
            let (start, end) = (
                self.bucket_start[b] as usize,
                self.bucket_start[b + 1] as usize,
            );
            let bucket = &self.sorted[start..end];
            for (offset, &i) in bucket.iter().enumerate() {
                for &j in &bucket[offset + 1..] {
                    if i == j {
                        continue;
                    }
                    let (a, b) = if i < j { (i, j) } else { (j, i) };
                    if !dynamic[a as usize] && !dynamic[b as usize] {
                        continue;
                    }
                    if aabbs[a as usize].overlaps(&aabbs[b as usize]) {
                        out.push((a, b));
                    }
                }
            }
        }

        // --- oversized bodies against everything ---
        for (index, &big) in self.oversized.iter().enumerate() {
            for other in 0..n as u32 {
                if other == big {
                    continue;
                }
                // Two oversized bodies would be found twice; take each pair
                // from the earlier one only.
                if self.oversized[..index].contains(&other) {
                    continue;
                }
                let (a, b) = if big < other {
                    (big, other)
                } else {
                    (other, big)
                };
                if !dynamic[a as usize] && !dynamic[b as usize] {
                    continue;
                }
                if aabbs[a as usize].overlaps(&aabbs[b as usize]) {
                    out.push((a, b));
                }
            }
        }

        out.sort_unstable();
        out.dedup();
    }
}

/// How many cells a body covers, or zero if that is more than the grid will
/// take. Zero is unambiguous: every body covers at least one cell.
fn cells_covered(bounds: &Aabb, inv_cell: f32) -> usize {
    let lo = cell_of(bounds.min, inv_cell);
    let hi = cell_of(bounds.max, inv_cell);
    let mut total = 1usize;
    for k in 0..3 {
        let span = (hi[k] as i64 - lo[k] as i64 + 1).max(1);
        if span > MAX_CELLS_PER_BODY as i64 {
            return 0;
        }
        total = total.saturating_mul(span as usize);
        if total > MAX_CELLS_PER_BODY {
            return 0;
        }
    }
    total
}

fn cell_of(p: Vec3, inv_cell: f32) -> [i32; 3] {
    // Saturating rather than wrapping: a body at 1e30 would otherwise land in
    // a cell next to one at the origin.
    let mut out = [0i32; 3];
    for k in 0..3 {
        let v = (p[k] * inv_cell).floor();
        out[k] = if v.is_nan() {
            0
        } else {
            v.clamp(i32::MIN as f32 / 2.0, i32::MAX as f32 / 2.0) as i32
        };
    }
    out
}

/// Hash three cell coordinates. The multipliers are the large primes used by
/// Teschner et al., which is the same set [`crate::grid`] uses — one hash
/// function in the crate rather than two that have to be argued about
/// separately.
fn hash(x: i32, y: i32, z: i32) -> u32 {
    let h = (x as u32).wrapping_mul(73_856_093)
        ^ (y as u32).wrapping_mul(19_349_663)
        ^ (z as u32).wrapping_mul(83_492_791);
    // The low bits of that xor are not well mixed on their own, and the bucket
    // index is exactly the low bits.
    h ^ (h >> 16)
}

/// A cell size that fits the bodies rather than the outliers.
///
/// The mean over everything is not it: one 420-metre floor among 100 000
/// one-metre spheres barely moves the mean, but one *large* scene object among
/// a few hundred small ones drags it far enough that nothing fits any more. So
/// the mean is taken twice, the second time over the bodies that were not
/// wildly bigger than the first estimate. Two linear passes, no sort, and the
/// answer is stable under a handful of outliers of any size.
fn cell_size(aabbs: &[Aabb]) -> f32 {
    let extent = |b: &Aabb| {
        let e = b.max - b.min;
        e[0].max(e[1]).max(e[2])
    };
    let mut sum = 0.0f64;
    for b in aabbs {
        sum += extent(b) as f64;
    }
    let rough = (sum / aabbs.len() as f64) as f32;
    if !(rough.is_finite() && rough > 0.0) {
        return rough;
    }

    let ceiling = rough * 4.0;
    let mut sum = 0.0f64;
    let mut count = 0usize;
    for b in aabbs {
        let e = extent(b);
        if e <= ceiling {
            sum += e as f64;
            count += 1;
        }
    }
    if count == 0 {
        return rough;
    }
    // Slightly larger than the bodies themselves, so a typical body covers one
    // or two cells per axis rather than sitting exactly on a boundary and
    // covering two every time.
    (sum / count as f64) as f32 * 1.25
}
