//! Broad phase: sweep and prune.
//!
//! PLAN.md §10 picks SAP over a dynamic BVH for v0.1 and says why: it is
//! simpler and cache-friendlier for coherent motion, and a BVH only wins once
//! object sizes are wildly heterogeneous. The structure here keeps its sorted
//! order between frames, so the insertion sort that maintains it is close to
//! linear on the coherent motion physics scenes actually have.

use fusion_math::Aabb;

/// A candidate pair of body indices, ordered so `a < b`.
pub type BroadPair = (u32, u32);

/// Sorted-endpoint broad phase over a single axis.
#[derive(Clone, Debug, Default)]
pub struct SweepAndPrune {
    /// Body indices sorted by AABB minimum along the sweep axis. Persisted
    /// across frames — this is where the coherence saving comes from.
    order: Vec<u32>,
    /// Sweep axis, re-chosen each frame from the variance of the centroids.
    axis: usize,
}

impl SweepAndPrune {
    /// An empty sweep. The sorted order it keeps between frames is built on
    /// the first call and maintained after that.
    pub fn new() -> Self {
        Self {
            order: Vec::new(),
            axis: 0,
        }
    }

    /// Collect every pair of overlapping AABBs.
    ///
    /// `dynamic` marks which bodies can move; static-static pairs are skipped
    /// because nothing will ever come of solving them.
    pub fn find_pairs(&mut self, aabbs: &[Aabb], dynamic: &[bool], out: &mut Vec<BroadPair>) {
        out.clear();
        let n = aabbs.len();
        if n < 2 {
            return;
        }

        // Rebuild the order if the body count changed. v0.1 only ever appends,
        // so this is a startup cost, not a per-frame one.
        if self.order.len() != n {
            self.order = (0..n as u32).collect();
        }
        self.axis = best_axis(aabbs);
        let axis = self.axis;

        // Insertion sort: O(n) when the order barely changed, which is the
        // common case, and correct when it changed a lot.
        let key = |i: u32| aabbs[i as usize].min[axis];
        for i in 1..self.order.len() {
            let item = self.order[i];
            let k = key(item);
            let mut j = i;
            while j > 0 && key(self.order[j - 1]) > k {
                self.order[j] = self.order[j - 1];
                j -= 1;
            }
            self.order[j] = item;
        }

        // Sweep: an interval can only overlap those still "active" — the ones
        // whose maximum has not yet been passed.
        let mut active: Vec<u32> = Vec::new();
        for &i in &self.order {
            let a = &aabbs[i as usize];
            active.retain(|&j| aabbs[j as usize].max[axis] >= a.min[axis]);
            for &j in &active {
                if !dynamic[i as usize] && !dynamic[j as usize] {
                    continue;
                }
                if a.overlaps(&aabbs[j as usize]) {
                    out.push(if i < j { (i, j) } else { (j, i) });
                }
            }
            active.push(i);
        }
    }

    /// Axis the last sweep ran along, chosen from the spread of the bodies.
    pub fn axis(&self) -> usize {
        self.axis
    }
}

/// Sweep along whichever axis spreads the bodies out most — it produces the
/// shortest active list, which is the whole cost of the sweep.
fn best_axis(aabbs: &[Aabb]) -> usize {
    // Variance about the first centre rather than about zero.
    //
    // `E[x^2] - E[x]^2` is the textbook formula and a well-known way to lose
    // every significant digit: for a scene sitting 500 m from the origin it
    // subtracts two numbers near 250000 to find a spread of a few metres.
    // Shifting first costs one subtraction and makes the result depend on how
    // spread out the bodies are rather than on where they happen to be.
    let n = aabbs.len() as f32;
    let shift = aabbs[0].center();
    let mut sum = [0.0f32; 3];
    let mut sum_sq = [0.0f32; 3];
    for b in aabbs {
        let c = b.center() - shift;
        for k in 0..3 {
            sum[k] += c[k];
            sum_sq[k] += c[k] * c[k];
        }
    }
    let mut best = 0;
    let mut best_variance = f32::NEG_INFINITY;
    for k in 0..3 {
        let mean = sum[k] / n;
        let variance = sum_sq[k] / n - mean * mean;
        if variance > best_variance {
            best_variance = variance;
            best = k;
        }
    }
    best
}

/// Brute-force pair finding. Kept as the reference the SAP is tested against —
/// a broad phase that misses pairs fails silently and looks like a solver bug.
pub fn find_pairs_brute_force(aabbs: &[Aabb], dynamic: &[bool], out: &mut Vec<BroadPair>) {
    out.clear();
    for i in 0..aabbs.len() {
        for j in (i + 1)..aabbs.len() {
            if !dynamic[i] && !dynamic[j] {
                continue;
            }
            if aabbs[i].overlaps(&aabbs[j]) {
                out.push((i as u32, j as u32));
            }
        }
    }
}
