//! A uniform spatial hash over points.
//!
//! Sweep-and-prune (see [`crate::broad`]) is the right broad phase for a few
//! thousand bodies of wildly different sizes. It is the wrong one for a
//! hundred thousand fluid particles that are all exactly the same size and all
//! within a smoothing radius of thirty neighbours: SAP's active list becomes
//! enormous along the sweep axis, and the sort it maintains stops being nearly
//! sorted the moment the fluid churns.
//!
//! A uniform grid is the standard answer, and it is the right one here for the
//! reason SAP was right there — the data suits it. Every particle has the same
//! radius, so one cell size fits all of them, and a query is a fixed 27-cell
//! walk with no sorting at all.
//!
//! ## Why hashing, and why counting sort
//!
//! A dense 3D array of cells costs memory proportional to the *volume* of the
//! domain, which a splash that travels ten metres would make ruinous. Hashing
//! the cell coordinate into a fixed table costs memory proportional to the
//! particle count instead. The price is collisions — two distant cells landing
//! in the same bucket — and the query pays it by distance-testing every
//! candidate, which it must do anyway.
//!
//! The bucket contents are built with a counting sort rather than a
//! `Vec<Vec<u32>>` or a `HashMap`, for two reasons that both matter:
//!
//! - **Allocation.** One flat index array and one offset array, reused across
//!   steps. A vector per bucket allocates once per bucket per rebuild.
//! - **Determinism** (PLAN.md §34). A counting sort that inserts in ascending
//!   particle order leaves every bucket in ascending particle order, every
//!   time. `HashMap` iteration order is not specified and is seeded randomly,
//!   so a solver that summed forces in hash order would give a different
//!   answer on every run of the same binary — which would take rewind and
//!   replay with it.

use fusion_math::{DVec3, Vec3};

/// Uniform hashed grid over a set of points.
#[derive(Clone, Debug, Default)]
pub struct SpatialHash {
    cell_size: f32,
    inv_cell_size: f32,
    /// Grid origin. Cell coordinates are taken relative to this rather than to
    /// the world origin, so a scene sitting 20 km out gets the same cell
    /// indices — and the same neighbour sets — as one at the origin.
    origin: DVec3,
    /// `bucket_start[b] .. bucket_start[b + 1]` indexes `sorted`.
    bucket_start: Vec<u32>,
    /// Particle indices, grouped by bucket, ascending within each bucket.
    sorted: Vec<u32>,
    /// The same points as `sorted`, in the same order, as three parallel axis
    /// arrays of `f32` **relative to `origin`**.
    ///
    /// This is the array the neighbour sweeps actually read, and both halves
    /// of its description are load-bearing.
    ///
    /// *In bucket order*, so a query walks contiguous memory instead of
    /// chasing `points[j]` around the caller's array for every one of the two
    /// hundred-odd candidates a 27-cell walk turns up. That is what makes the
    /// sweep vectorizable at all: four consecutive candidates are one load per
    /// axis.
    ///
    /// *Relative to the origin, in `f32`*, which looks like it throws away the
    /// precision PLAN.md §7 bought by storing positions in `f64` — and does
    /// not. The `f64` store exists so a scene can sit 20 km from the world
    /// origin without the lattice dissolving. Neighbour geometry never needs
    /// world coordinates: `origin` is snapped to the cell lattice near the
    /// points themselves, so these locals span the *set*, not the world. A
    /// fluid 20 m across resolves to 1.5 µm here whether it sits at the origin
    /// or a kilometre away, and the difference against `f64` shows up only for
    /// a point within 10⁻⁷ of the cutoff — where every kernel in the solver is
    /// already zero.
    sorted_x: Vec<f32>,
    sorted_y: Vec<f32>,
    sorted_z: Vec<f32>,
    /// Per-particle bucket, kept so the second counting pass does not have to
    /// recompute the hash.
    bucket_of: Vec<u32>,
    mask: u32,
}

impl SpatialHash {
    /// An empty hash. Its tables are sized on the first build and reused
    /// across steps, which is most of why it is a struct and not a function.
    pub fn new() -> Self {
        Self::default()
    }

    /// Side length of one cell. Set to the query radius: a 27-cell walk then
    /// covers every point within that radius and no more than it has to.
    pub fn cell_size(&self) -> f32 {
        self.cell_size
    }

    /// Rebuild the grid over `points`, for queries of radius `cell_size`.
    pub fn build(&mut self, points: &[DVec3], cell_size: f32) {
        let n = points.len();
        self.cell_size = cell_size.max(1e-6);
        self.inv_cell_size = 1.0 / self.cell_size;

        // Power-of-two table, at least twice the particle count, so the
        // modulo is a mask and the average bucket holds well under one cell.
        let buckets = (2 * n.max(1)).next_power_of_two().max(64);
        self.mask = (buckets - 1) as u32;

        self.bucket_start.clear();
        self.bucket_start.resize(buckets + 1, 0);
        self.bucket_of.clear();
        self.bucket_of.resize(n, 0);
        self.sorted.clear();
        self.sorted.resize(n, 0);
        for axis in [&mut self.sorted_x, &mut self.sorted_y, &mut self.sorted_z] {
            axis.clear();
            axis.resize(n, 0.0);
        }
        if n == 0 {
            return;
        }

        // The origin is snapped to the cell lattice rather than taken from the
        // raw minimum. An unsnapped origin drifts with whichever particle
        // happens to be lowest, and every cell boundary in the scene moves
        // when one particle at the edge does — so a particle far away changes
        // which bucket its neighbours land in.
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

        for (i, p) in points.iter().enumerate() {
            let bucket = self.bucket_at(self.cell_coords(*p));
            self.bucket_of[i] = bucket;
            self.bucket_start[bucket as usize + 1] += 1;
        }
        for b in 0..buckets {
            self.bucket_start[b + 1] += self.bucket_start[b];
        }
        // Second pass in ascending particle order, so each bucket ends up
        // ascending too. `cursor` walks each bucket's span as it fills.
        let mut cursor = self.bucket_start.clone();
        for (i, p) in points.iter().enumerate() {
            let bucket = self.bucket_of[i] as usize;
            let slot = cursor[bucket] as usize;
            self.sorted[slot] = i as u32;
            cursor[bucket] += 1;
            // The axis arrays are permuted with the index array rather than in
            // a pass of their own, so the scattered write happens once instead
            // of twice.
            let local = self.to_local(*p);
            self.sorted_x[slot] = local.x;
            self.sorted_y[slot] = local.y;
            self.sorted_z[slot] = local.z;
        }
    }

    /// Origin of the local frame the axis arrays are expressed in.
    pub fn origin(&self) -> DVec3 {
        self.origin
    }

    /// Point indices in bucket order. `sorted_axes()[k]` is the position of
    /// `sorted_indices()[k]`.
    pub fn sorted_indices(&self) -> &[u32] {
        &self.sorted
    }

    /// The three axis arrays, parallel to [`Self::sorted_indices`].
    pub fn sorted_axes(&self) -> (&[f32], &[f32], &[f32]) {
        (&self.sorted_x, &self.sorted_y, &self.sorted_z)
    }

    /// Integer cell coordinate of a point already in the local frame.
    ///
    /// **Every cell coordinate in this file comes through here**, including
    /// the ones the build computes from `f64` world positions — which reach it
    /// by way of `to_local` first. That is deliberate and it is not
    /// free: narrowing to `f32` before flooring loses the last bits of a
    /// coordinate, so a point a hair under a cell boundary in `f64` can land a
    /// hair over it here.
    ///
    /// Rounding the same way everywhere is worth more than rounding exactly.
    /// A grid that hashed a point into one cell during the build and queried
    /// around a different one would walk 27 cells centred one over, missing a
    /// whole face of true neighbours — and it would do it for roughly one
    /// point in ten million, which is often enough to happen and rare enough
    /// to never be reproduced. Sharing the function makes the disagreement
    /// impossible rather than unlikely.
    ///
    /// The precision this rests on is the local frame's, not the world's: the
    /// origin sits at the point set's own snapped minimum, so what has to stay
    /// resolvable is the *extent* of the set against the cell size. `f32`
    /// gives 24 bits of that, so the grid degrades only once a set spans some
    /// millions of cells in one axis — far past the point where the index
    /// arrays themselves would not fit.
    #[inline]
    pub fn cell_of_local(&self, local: Vec3) -> [i32; 3] {
        [
            (local.x * self.inv_cell_size).floor() as i32,
            (local.y * self.inv_cell_size).floor() as i32,
            (local.z * self.inv_cell_size).floor() as i32,
        ]
    }

    /// The spans of `sorted_indices` / `sorted_axes` holding every candidate
    /// for a query anywhere in cell `c` — the 27 neighbouring buckets,
    /// deduplicated, in the fixed cell order [`Self::for_each_candidate`] uses.
    ///
    /// This exists so that the 27 hashes and the duplicate check are paid once
    /// per *cell* rather than once per point. At the default fluid resolution a
    /// cell holds about eight particles, all of which see exactly these
    /// candidates, and the sweep across them is the part worth doing four at a
    /// time.
    ///
    /// Empty spans are dropped: a hash table sized at twice the point count is
    /// mostly empty, so about two thirds of the 27 lookups have nothing in
    /// them and would otherwise cost the caller a loop entry each.
    pub fn candidate_spans(&self, c: [i32; 3], out: &mut Vec<(u32, u32)>) {
        out.clear();
        if self.sorted.is_empty() {
            return;
        }
        let mut seen = [u32::MAX; 27];
        let mut count = 0usize;
        for dz in -1..=1 {
            for dy in -1..=1 {
                for dx in -1..=1 {
                    let bucket = self.bucket_at([c[0] + dx, c[1] + dy, c[2] + dz]);
                    if seen[..count].contains(&bucket) {
                        continue;
                    }
                    seen[count] = bucket;
                    count += 1;
                    let (start, end) = (
                        self.bucket_start[bucket as usize],
                        self.bucket_start[bucket as usize + 1],
                    );
                    if start != end {
                        out.push((start, end));
                    }
                }
            }
        }
    }

    /// A world point in the grid's local frame.
    #[inline]
    fn to_local(&self, p: DVec3) -> Vec3 {
        let d = p - self.origin;
        Vec3::new(d.x as f32, d.y as f32, d.z as f32)
    }

    /// Integer cell coordinate of a point.
    #[inline]
    fn cell_coords(&self, p: DVec3) -> [i32; 3] {
        self.cell_of_local(self.to_local(p))
    }

    /// Hash a cell coordinate into a bucket.
    ///
    /// The three large primes are the usual Teschner et al. choice. XOR rather
    /// than a sum, so that permuting the axes does not collide, and a
    /// multiply-shift finish so that the low bits — which the mask keeps — are
    /// mixed rather than being the raw low bits of `x`.
    #[inline]
    fn bucket_at(&self, c: [i32; 3]) -> u32 {
        let h = (c[0] as u32).wrapping_mul(0x8da6_b343)
            ^ (c[1] as u32).wrapping_mul(0xd837_2179)
            ^ (c[2] as u32).wrapping_mul(0xcb1a_b31f);
        let h = h ^ (h >> 16);
        h.wrapping_mul(0x7feb_352d) & self.mask
    }

    /// Call `f` with every point index in the 27 cells around `point`, each
    /// index exactly once.
    ///
    /// Candidates, not neighbours: a bucket may hold points from an unrelated
    /// cell that hashed to the same place, and even a true neighbouring cell
    /// holds points up to `sqrt(3) * cell_size` away. The caller distance-tests,
    /// which it was going to do regardless.
    ///
    /// **Each of the 27 buckets is visited at most once**, and that is not an
    /// optimization. Two of the 27 cell coordinates can hash to the same
    /// bucket — with 27 draws from a 1024-entry table it happens about a third
    /// of the time — and enumerating that bucket twice hands the caller the
    /// same point twice. A distance test does not catch it, because the point
    /// really is a neighbour; it is just being counted again. An SPH density
    /// sum built on that reads 9% high on an undisturbed lattice, tells a
    /// fluid at rest that it is compressed, and blows it apart in the first
    /// substep.
    #[inline]
    pub fn for_each_candidate(&self, point: DVec3, mut f: impl FnMut(u32)) {
        if self.sorted.is_empty() {
            return;
        }
        let c = self.cell_coords(point);
        let mut seen = [u32::MAX; 27];
        let mut count = 0usize;
        for dz in -1..=1 {
            for dy in -1..=1 {
                for dx in -1..=1 {
                    let bucket = self.bucket_at([c[0] + dx, c[1] + dy, c[2] + dz]);
                    if seen[..count].contains(&bucket) {
                        continue;
                    }
                    seen[count] = bucket;
                    count += 1;
                    let bucket = bucket as usize;
                    let (start, end) = (
                        self.bucket_start[bucket] as usize,
                        self.bucket_start[bucket + 1] as usize,
                    );
                    for &index in &self.sorted[start..end] {
                        f(index);
                    }
                }
            }
        }
    }

    /// Collect the indices within `radius` of `point` into `out`, ascending.
    ///
    /// Ascending order is not incidental. A Gauss-Seidel-flavoured pass that
    /// visits neighbours in a different order produces a different answer in
    /// floating point, and PLAN.md §34 commits Fusion to bit-identical replay
    /// on the same machine and build.
    pub fn neighbours(&self, points: &[DVec3], point: DVec3, radius: f32, out: &mut Vec<u32>) {
        out.clear();
        let r_sq = (radius as f64) * (radius as f64);
        self.for_each_candidate(point, |i| {
            if (points[i as usize] - point).length_squared() <= r_sq {
                out.push(i);
            }
        });
        // A 27-cell walk visits 27 buckets in cell order, and each bucket is
        // ascending on its own — but the concatenation of them is not. No
        // dedup: `for_each_candidate` already guarantees each index once.
        out.sort_unstable();
    }

    /// Number of points currently indexed.
    pub fn len(&self) -> usize {
        self.sorted.len()
    }

    /// True where nothing has been inserted.
    pub fn is_empty(&self) -> bool {
        self.sorted.is_empty()
    }

    /// Largest number of points in any one bucket. A diagnostic: a grid whose
    /// peak bucket holds hundreds is either badly sized or looking at a scene
    /// that has collapsed to a point.
    pub fn peak_bucket(&self) -> usize {
        self.bucket_start
            .windows(2)
            .map(|w| (w[1] - w[0]) as usize)
            .max()
            .unwrap_or(0)
    }
}

/// Convenience for callers holding single-precision points.
pub fn to_double(points: &[Vec3], out: &mut Vec<DVec3>) {
    out.clear();
    out.extend(points.iter().map(|p| fusion_math::to_f64(*p)));
}
