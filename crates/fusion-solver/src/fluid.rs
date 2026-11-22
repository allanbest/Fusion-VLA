//! Position-based fluids.
//!
//! PLAN.md §29 puts "PBF/SPH" in v0.4, and §9 puts the resulting solver in the
//! **real-time** tier alongside XPBD contacts: "stable and plausible, not
//! conservative, never present as a measurement." That is not a hedge, it is
//! the defining property of the method — see [`CAVEAT`] at the bottom of this
//! file for what specifically is untrue about it.
//!
//! ## Why PBF and not weakly-compressible SPH
//!
//! Both are Lagrangian; the difference is how incompressibility is enforced.
//! WCSPH computes a pressure from a stiff equation of state and integrates the
//! resulting force, which means the timestep is bounded by the speed of sound
//! in the fluid — for water that is 1500 m/s, and the CFL condition it implies
//! puts the timestep at around 10⁻⁵ s. PBF instead treats constant density as
//! a *constraint* and projects positions onto it, which is unconditionally
//! stable at any timestep for the same reason XPBD's contacts are.
//!
//! That is also why it belongs here rather than in a new solver: the density
//! constraint is `C(x) = ρ/ρ₀ - 1 = 0`, solved by the same predict-project-
//! recover loop that already covers contacts, joints, cloth and soft bodies.
//! PLAN.md §10 promised "one mechanism, fewer moving parts to own", and a
//! fluid is the fifth system to arrive without a second numerical stack.
//!
//! ## The method
//!
//! Macklin & Müller, *Position Based Fluids*, SIGGRAPH 2013. Per substep:
//!
//! ```text
//! predict      x* = x + v h            (gravity already applied)
//! neighbours   uniform grid, radius h_kernel
//! iterate      density -> multiplier -> position correction
//! recover      v = (x* - x) / h
//! post         vorticity confinement, then XSPH viscosity
//! ```
//!
//! ## Densities in kernel units
//!
//! The published formulation folds particle mass into the rest density and
//! works with `rho_hat = sum_j W(r_ij)`, which has units of 1/m³. This module
//! does the same, and calibrates the target against the *actual* lattice the
//! fluid was filled on rather than against the textbook `rho_0 / m`.
//!
//! The difference is not cosmetic. A finite kernel over a finite lattice does
//! not sum to exactly `rho_0 / m` — with `h = 2 * spacing` it comes out about
//! 4% low — and using the textbook value tells a block of water that has not
//! moved yet that it is 4% under-dense. It then contracts, overshoots, and
//! rings for the first half second of every simulation. Calibrating makes
//! `C = 0` exactly true for an undisturbed interior particle, which is what
//! "at rest" ought to mean.

use fusion_collide::SpatialHash;
use fusion_core::{BodySet, FluidSet};
use fusion_math::simd::{F32x4, Vec3x4};
use fusion_math::Vec3;

/// Poly6 smoothing kernel, unnormalized argument in squared distance.
///
/// Used for densities and for the artificial-pressure ratio, never for
/// gradients — its gradient vanishes at the origin, which is exactly where
/// particles that have clumped together need to be pushed apart.
#[inline]
pub fn poly6(r_sq: f32, h: f32) -> f32 {
    let h_sq = h * h;
    if r_sq >= h_sq {
        return 0.0;
    }
    let d = h_sq - r_sq;
    // 315 / (64 pi h^9)
    let norm = 315.0 / (64.0 * std::f32::consts::PI * h.powi(9));
    norm * d * d * d
}

/// Gradient of the spiky kernel, with respect to the first position.
///
/// Spiky rather than Poly6 for gradients, for the reason above: this one has a
/// non-vanishing gradient as `r -> 0`, so two coincident particles separate
/// instead of sitting on top of each other forever.
#[inline]
pub fn spiky_gradient(offset: Vec3, r: f32, h: f32) -> Vec3 {
    if r >= h || r < 1e-9 {
        return Vec3::ZERO;
    }
    let d = h - r;
    // -45 / (pi h^6), then along the unit offset.
    let norm = -45.0 / (std::f32::consts::PI * h.powi(6));
    offset * (norm * d * d / r)
}

/// Density a perfect lattice of spacing `spacing` produces under the Poly6
/// kernel at radius `h`, in kernel units.
///
/// Closed form is not available; this sums the lattice directly, which is
/// exact for the configuration that matters and costs a few dozen kernel
/// evaluations once per prepare.
pub fn lattice_density(spacing: f32, h: f32) -> f32 {
    let reach = (h / spacing).ceil() as i32;
    let mut sum = 0.0;
    for iz in -reach..=reach {
        for iy in -reach..=reach {
            for ix in -reach..=reach {
                let offset = Vec3::new(ix as f32, iy as f32, iz as f32) * spacing;
                sum += poly6(offset.length_squared(), h);
            }
        }
    }
    sum
}

/// Fraction of a particle's kernel support that is cut off by a solid surface
/// `distance` away, assuming the surface is locally flat.
///
/// ## Why this exists
///
/// An SPH density is a sum over neighbours, and a particle against a wall has
/// none on the far side of it. It reads under-dense — by 22% for a particle
/// resting on the floor at the default smoothing radius — the density
/// constraint sees a deficit it is not allowed to fix, and the fluid packs
/// itself tighter against every surface until the estimate comes out right.
/// Measured on a 1 m tank: 512 kg of water settled to a depth of 0.47 m
/// instead of 0.51, an 8% volume error, and it got *worse* with a larger
/// smoothing radius because a wider kernel means a thicker affected layer.
///
/// The textbook fix is boundary particles — sample the container surface,
/// give the samples a pseudo-mass, include them in the sum. It works, and it
/// costs a particle set per static body, a second neighbour structure, and a
/// sampling pass on every shape.
///
/// Fusion does not need it, because PLAN.md §8 already requires every shape to
/// carry an SDF. The distance to the surface is one query, and for a flat wall
/// the missing part of the kernel integrates in closed form:
///
/// ```text
/// I(d) = (315/256) * [ 128/315 - F(d/h) ],
///     F(t) = t - 4t^3/3 + 6t^5/5 - 4t^7/7 + t^9/9
/// ```
///
/// which is the Poly6 kernel integrated over the half-space beyond the wall.
/// `I(0) = 0.5` — half the support, as it must be for a particle exactly on
/// the surface. Multiplying by the rest density gives the density the missing
/// fluid *would* have contributed, which is the assumption being made: that
/// the material on the far side of the wall is at rest density. For a
/// container wall that is very nearly true.
///
/// The approximation this makes is that the surface is planar over the kernel
/// radius. Wrong for a sphere smaller than the smoothing radius, and wrong in
/// a sharp corner where two walls each claim their own occlusion and the
/// overlap is counted twice. Both err towards *over*-estimating the missing
/// density, which pushes fluid away from corners rather than sucking it in —
/// the benign direction.
pub fn boundary_occlusion(distance: f32, h: f32) -> f32 {
    if distance >= h || h <= 0.0 {
        return 0.0;
    }
    let t = (distance.max(0.0) / h).min(1.0);
    let (t2, t3) = (t * t, t * t * t);
    let f =
        t - 4.0 * t3 / 3.0 + 6.0 * t3 * t2 / 5.0 - 4.0 * t3 * t2 * t2 / 7.0 + t3 * t3 * t3 / 9.0;
    // 128/315 is F(1), the value of the integral over the full half-space.
    (315.0 / 256.0) * (128.0 / 315.0 - f)
}

/// Ceiling on neighbours kept per particle.
///
/// Not a physical limit — at the default `h = 2 * spacing` a particle has
/// about 33 neighbours, and 128 is four times that. It is a guard against the
/// one failure that turns a slow frame into a hang: if a scene collapses to a
/// point, an uncapped neighbour list is quadratic in the particle count and
/// allocates until the process dies. Dropping neighbours past the cap makes
/// the density estimate wrong in a way the diagnostics can report, which is
/// strictly better than not returning.
pub const MAX_NEIGHBOURS: usize = 128;

/// Per-fluid solver scratch: neighbour lists, densities, multipliers.
///
/// Kept in the solver's working set and not in [`FluidSet`], on the same
/// boundary the contact solver draws — the fluid is where the particles are,
/// not what a particular method needs to remember about them.
#[derive(Clone, Debug, Default)]
pub struct FluidScratch {
    grid: SpatialHash,
    /// Predicted positions again, in `f32` and relative to the grid's origin —
    /// three parallel axis arrays, indexed by particle.
    ///
    /// The grid holds the same points in *bucket* order, which is what the
    /// neighbour build sweeps; these are in *particle* order, which is what
    /// everything downstream gathers by neighbour index. Both exist because
    /// neither order serves the other loop, and 12 bytes a particle is a
    /// cheaper duplicate than a permutation on every access.
    ///
    /// See `SpatialHash::sorted_axes` for why `f32` here does not undo the
    /// `f64` position store.
    lx: Vec<f32>,
    ly: Vec<f32>,
    lz: Vec<f32>,
    /// Flat neighbour lists. Particle `i` owns
    /// `neighbours[start[i] .. start[i] + count[i]]`.
    ///
    /// A start *and* a count rather than a prefix-sum boundary array, because
    /// the build no longer visits particles in index order — it visits them in
    /// cell order, so that the 27-bucket lookup is paid once per cell rather
    /// than once per particle. Each particle's own run is still contiguous;
    /// the runs simply are not in index order within the flat array.
    neighbours: Vec<u32>,
    start: Vec<u32>,
    count: Vec<u32>,
    /// Candidate spans of the current cell, reused across the particles in it.
    spans: Vec<(u32, u32)>,
    /// The current cell's candidates, gathered out of those spans into one
    /// contiguous run: three axis arrays and the particle index.
    ///
    /// The spans are 27 unrelated slices of the sorted arrays, and scanning
    /// them one at a time made the distance test almost entirely prologue. At
    /// the default `h = 2 * spacing` a bucket holds about five particles, so
    /// each span is one full lane-quad and a remainder of one — twenty-seven
    /// loop set-ups and twenty-seven scalar tails to reject about a hundred
    /// and forty candidates.
    ///
    /// It showed up as the neighbour build costing three times as much per
    /// particle in a settled pool as in a free-falling lattice, which packs
    /// eight to a cell rather than five. More candidates, less time: the work
    /// was not in the candidates.
    ///
    /// Flattening once per cell — a memcpy per span, amortized over the
    /// particles in that cell — turns the per-particle scan into a single
    /// contiguous SIMD loop with one tail. The order is unchanged, span by
    /// span and ascending within each, so every neighbour list comes out
    /// exactly as it did before.
    cand_x: Vec<f32>,
    cand_y: Vec<f32>,
    cand_z: Vec<f32>,
    cand_id: Vec<u32>,
    /// The local positions the current neighbour lists were built from.
    ///
    /// See [`FluidScratch::neighbours_are_stale`]. Same frame as `lx/ly/lz`,
    /// which stays valid precisely because a skipped rebuild is also a skipped
    /// *grid* rebuild, so the origin they are relative to does not move.
    built_x: Vec<f32>,
    built_y: Vec<f32>,
    built_z: Vec<f32>,
    /// Both kernels evaluated once per neighbour *pair*, parallel to
    /// `neighbours`: the Poly6 weight and the spiky gradient, as four axis
    /// arrays.
    ///
    /// ## Why the cache exists
    ///
    /// Six loops in this file walk the neighbour lists — three in the density
    /// solve, three in the post-pass — and every one of them was evaluating
    /// the same `W(r_ij)` and `∇W(r_ij)` from scratch. That is six square
    /// roots and six divisions per pair per substep to produce six copies of
    /// two numbers.
    ///
    /// Worse, each of those loops reached its neighbours by *gathering*:
    /// `lx[j]` for scattered `j`. A gather is four bounds-checked loads that
    /// LLVM will not vectorize across, so the loops stayed scalar however
    /// carefully they were written four lanes wide — verified by reading the
    /// emitted assembly, which contained no vector arithmetic at all.
    ///
    /// Evaluating both kernels once into arrays *parallel to the neighbour
    /// list* fixes both problems with one change. The gather happens once,
    /// here; every consumer afterwards reads its own particle's run as a
    /// contiguous slice, which is one bounds check per four lanes and does
    /// vectorize.
    ///
    /// ## What it costs
    ///
    /// Sixteen bytes per neighbour pair — about 530 bytes a particle at the
    /// default `h = 2 * spacing`, so 1.9 MB for the 3500-particle tank and
    /// 53 MB at a hundred thousand particles. That is the largest single
    /// allocation in the solver and it is the price of the arrangement, not an
    /// oversight. PLAN.md §24 puts the GPU crossover at about 10⁵ particles;
    /// this is one more reason the crossover is there.
    ///
    /// ## Staleness, which is deliberate
    ///
    /// Rebuilt at the top of every density iteration, from the positions that
    /// iteration will use. The post-pass does *not* rebuild it and reads
    /// whatever the last density iteration left — which is exactly what the
    /// post-pass did before the cache existed, since it never refreshed the
    /// positions either. Preserved rather than fixed: changing it changes the
    /// physics, and that belongs in its own change with its own measurement.
    kernel_w: Vec<f32>,
    kernel_gx: Vec<f32>,
    kernel_gy: Vec<f32>,
    kernel_gz: Vec<f32>,
    /// Density in kernel units, 1/m³.
    density: Vec<f32>,
    /// Per-particle kernel fraction occluded by solid geometry, in [0, 1).
    /// See [`boundary_occlusion`].
    boundary: Vec<f32>,
    lambda: Vec<f32>,
    delta: Vec<Vec3>,
    /// Every correction applied to each particle during the current substep —
    /// density iterations and boundary contacts alike.
    ///
    /// Velocity is recovered from this sum rather than by differencing the
    /// position before and after. Same argument the rigid path makes: the
    /// positions are `f64` and large, the corrections are `f32` and tiny, and
    /// subtracting the former to recover the latter throws away most of it.
    accumulated: Vec<Vec3>,
    /// Velocity before the post-pass, so viscosity and vorticity both read the
    /// same field instead of one seeing the other's output.
    velocity_in: Vec<Vec3>,
    curl: Vec<Vec3>,
    /// Target density in kernel units, calibrated from the fill lattice.
    rest: f32,
    /// Neighbour lists actually built, since the scratch was created.
    ///
    /// The claim [`FluidScratch::neighbours_are_stale`] makes is that most of
    /// these are avoidable, and a claim about how often something runs needs a
    /// count attached or it is a hope.
    builds: u64,
    /// Peak neighbour count seen in the last build, and whether the cap bit.
    peak_neighbours: usize,
    truncated: usize,
}

impl FluidScratch {
    /// Empty scratch for one body of fluid. Reused across steps: the
    /// neighbour lists it holds are megabytes at scale.
    pub fn new() -> Self {
        Self::default()
    }

    /// Size the scratch for a fluid and calibrate its rest density. Called
    /// once per step from `prepare`, so a resolution change takes effect
    /// without the caller having to say so.
    pub fn prepare(&mut self, fluid: &FluidSet) {
        let n = fluid.len();
        self.lx.resize(n, 0.0);
        self.ly.resize(n, 0.0);
        self.lz.resize(n, 0.0);
        self.count.resize(n, 0);
        self.density.resize(n, 0.0);
        self.boundary.resize(n, 0.0);
        self.lambda.resize(n, 0.0);
        self.delta.resize(n, Vec3::ZERO);
        self.accumulated.resize(n, Vec3::ZERO);
        self.velocity_in.resize(n, Vec3::ZERO);
        self.curl.resize(n, Vec3::ZERO);
        self.start.resize(n, 0);
        self.rest = lattice_density(fluid.params.spacing, fluid.params.smoothing_radius);
        // The neighbour lists may be reused between the substeps of a step and
        // never across one. Dropping them here is what keeps that true.
        //
        // Not a performance detail — it is what keeps the scratch *derivable*
        // from the world at every step boundary, which is the property §17's
        // rewind rests on. A list carried across a step is cross-step solver
        // state, and cross-step solver state has to be in the checkpoint or
        // the replay diverges. Two tests in `fusion-sim` say so directly, and
        // they said so the first time this was written without this line.
        self.built_x.clear();
        self.built_y.clear();
        self.built_z.clear();
    }

    /// Re-read the particle positions into the local `f32` frame the solve
    /// works in.
    ///
    /// Once per neighbour build and once per density iteration — the geometry
    /// has to be current even though the neighbour *lists* are not rebuilt;
    /// see the note in [`solve_density`] about what happens when it is not.
    fn refresh_local(&mut self, fluid: &FluidSet) {
        let origin = self.grid.origin();
        for (i, p) in fluid.particles.position.iter().enumerate() {
            let d = *p - origin;
            self.lx[i] = d.x as f32;
            self.ly[i] = d.y as f32;
            self.lz[i] = d.z as f32;
        }
    }

    /// Neighbour lists built since this scratch was created.
    pub fn builds(&self) -> u64 {
        self.builds
    }

    /// Whether the neighbour lists have to be rebuilt before the next solve.
    ///
    /// `tolerance` is a distance: no particle may have moved further than this
    /// since the lists were built. Zero rebuilds every time.
    ///
    /// ## Why a list may be reused at all
    ///
    /// The lists are already reused across the density iterations of a substep
    /// — see the note in [`solve_density`] — on the grounds that a particle
    /// does not move far enough between iterations to change who its
    /// neighbours are. This is the same argument over a longer horizon, and it
    /// is worth being precise about what it costs rather than waving at it.
    ///
    /// A pair missed by a stale list is one that was further apart than `h`
    /// when the list was built and is closer now. If neither particle has
    /// moved more than `t`, such a pair is still no closer than `h - 2t`, and
    /// Poly6 there is `(1 - (1 - 2t/h)²)³` of its peak — at the default
    /// `t = h/50` that is five parts in ten thousand of one neighbour's
    /// contribution to a sum over about forty. The pairs a stale list gets
    /// wrong are the ones whose weight is nearly zero, which is not a
    /// coincidence: the kernel goes to zero at exactly the radius the list is
    /// cut at.
    ///
    /// Pairs that have *left* need no handling at all — both kernels clamp to
    /// zero beyond `h`, so a neighbour that has moved out contributes nothing
    /// while it remains on the list.
    ///
    /// ## The horizon is one step, not one run
    ///
    /// [`FluidScratch::prepare`] drops the record at the top of every step, so
    /// the longest a list can live is the four fluid substeps of one step.
    /// That bound is not about accuracy — it is what keeps the scratch
    /// derivable from the world at a step boundary, which rewind needs.
    ///
    /// ## Why it is adaptive rather than a stride
    ///
    /// A splash rebuilds every substep and a settled pool rebuilds rarely,
    /// which is the behaviour wanted and is not what "rebuild every fourth
    /// substep" gives: that spends the same time on both and is wrong exactly
    /// when the fluid is moving fastest.
    pub fn neighbours_are_stale(&mut self, fluid: &FluidSet, tolerance: f32) -> bool {
        let n = fluid.len();
        if tolerance <= 0.0 || self.built_x.len() != n || self.start.len() != n {
            return true;
        }
        self.refresh_local(fluid);
        let limit = tolerance * tolerance;
        for i in 0..n {
            let dx = self.lx[i] - self.built_x[i];
            let dy = self.ly[i] - self.built_y[i];
            let dz = self.lz[i] - self.built_z[i];
            if dx * dx + dy * dy + dz * dz > limit {
                return true;
            }
        }
        false
    }

    /// Clear the per-substep correction sum. Called once per substep, before
    /// the first density iteration.
    pub fn begin_substep(&mut self) {
        self.accumulated.iter_mut().for_each(|d| *d = Vec3::ZERO);
    }

    /// The corrections applied so far this substep.
    pub fn accumulated(&self) -> &[Vec3] {
        &self.accumulated
    }

    /// Mutable, so the boundary pass can add its own corrections to the same
    /// sum the density solve writes into.
    pub fn accumulated_mut(&mut self) -> &mut [Vec3] {
        &mut self.accumulated
    }

    /// Density of each particle, kg/m³. The physical reading of the kernel
    /// estimate, for diagnostics and for colouring a debug view.
    pub fn density(&self) -> &[f32] {
        &self.density
    }

    /// Mean density error as a fraction of rest, over particles that have a
    /// full neighbourhood.
    ///
    /// The qualifier is the whole point. A surface particle is genuinely
    /// under-dense — half its neighbourhood is air — and averaging it in makes
    /// the compression error look several times worse than it is, which makes
    /// the number useless for telling a converged solve from a diverged one.
    pub fn compression_error(&self) -> f32 {
        if self.rest <= 0.0 {
            return 0.0;
        }
        let mut sum = 0.0;
        let mut count = 0usize;
        for &d in &self.density {
            // Interior, by the only definition available locally: at or above
            // rest. Under-dense particles are surface or free-fall.
            if d >= self.rest {
                sum += d / self.rest - 1.0;
                count += 1;
            }
        }
        if count == 0 {
            0.0
        } else {
            sum / count as f32
        }
    }

    /// Kernel fraction each particle has occluded by solid geometry.
    pub fn boundary_occlusion(&self) -> &[f32] {
        &self.boundary
    }

    /// Particle `i`'s neighbour list, exactly as the density solve reads it.
    ///
    /// Public for diagnostics and for the GPU path in `fusion-gpu`, which
    /// asserts that its lists are identical to these — same particles, same
    /// order. A weaker claim than identity would make the two paths two
    /// different fluids, and PLAN.md §34 promises one.
    pub fn neighbours(&self, i: usize) -> &[u32] {
        let (from, len) = run_of(self, i);
        &self.neighbours[from..from + len]
    }

    /// Largest neighbourhood any particle had in the last build. It rises
    /// well before a fluid solve gets into trouble.
    pub fn peak_neighbours(&self) -> usize {
        self.peak_neighbours
    }

    /// Particles whose neighbour list hit [`MAX_NEIGHBOURS`]. Non-zero means
    /// the density estimate is wrong somewhere, and says where to look.
    pub fn truncated_neighbourhoods(&self) -> usize {
        self.truncated
    }

    /// Rest density in kernel units, as calibrated.
    pub fn kernel_rest_density(&self) -> f32 {
        self.rest
    }
}

/// Rebuild the neighbour lists from the current (predicted) positions.
///
/// Once per substep rather than once per step. Reusing a step-old list is the
/// usual optimization and it is wrong here for a specific reason: with eight
/// substeps a fast splash moves a particle several spacings between rebuilds,
/// and a particle that has left its neighbours' lists but still has them in
/// its own gets a one-sided density correction — it pushes and is not pushed
/// back. Momentum stops being conserved and the fluid gains energy from
/// nothing. Rebuilding costs about 20% of the fluid step and buys symmetry.
pub fn build_neighbours(fluid: &FluidSet, scratch: &mut FluidScratch) {
    let n = fluid.len();
    let h = fluid.params.smoothing_radius;
    // Straight off the particle store: the grid reads `f64` world positions
    // and produces the local `f32` frame everything downstream works in, so
    // there is no second copy of the positions to keep.
    scratch.grid.build(&fluid.particles.position, h);
    scratch.refresh_local(fluid);

    scratch.builds += 1;
    scratch.neighbours.clear();
    scratch.peak_neighbours = 0;
    scratch.truncated = 0;
    if n == 0 {
        return;
    }

    // Split the borrow: the grid is read while the lists are written.
    let FluidScratch {
        grid,
        neighbours,
        start,
        count,
        spans,
        cand_x,
        cand_y,
        cand_z,
        cand_id,
        peak_neighbours,
        truncated,
        ..
    } = scratch;

    let (xs, ys, zs) = grid.sorted_axes();
    let ids = grid.sorted_indices();
    let h_sq = F32x4::splat(h * h);

    // Cell-major, not particle-major.
    //
    // `ids` is in bucket order, so consecutive entries are almost always in
    // the same cell — and every particle in one cell has the *same* 27
    // candidate buckets. Hashing them and rejecting the duplicates costs about
    // 350 comparisons; doing it once per cell instead of once per particle
    // divides that by the cell occupancy, which at the default `h = 2 *
    // spacing` is about eight.
    //
    // The cell is compared rather than tracked, because a bucket can hold two
    // cells that collided in the hash. Comparing catches that; assuming one
    // bucket is one cell would hand a particle its neighbour's candidate list.
    let mut last_cell = [i32::MIN; 3];
    for k in 0..n {
        let i = ids[k] as usize;
        let centre = Vec3::new(xs[k], ys[k], zs[k]);
        let cell = grid.cell_of_local(centre);
        if cell != last_cell {
            grid.candidate_spans(cell, spans);
            cand_x.clear();
            cand_y.clear();
            cand_z.clear();
            cand_id.clear();
            for &(from, to) in spans.iter() {
                let (from, to) = (from as usize, to as usize);
                cand_x.extend_from_slice(&xs[from..to]);
                cand_y.extend_from_slice(&ys[from..to]);
                cand_z.extend_from_slice(&zs[from..to]);
                cand_id.extend_from_slice(&ids[from..to]);
            }
            last_cell = cell;
        }

        start[i] = neighbours.len() as u32;
        let base = neighbours.len();
        let wide_centre = Vec3x4::splat(centre);

        // Four candidates at a time, straight off contiguous memory. Driven by
        // `chunks_exact` rather than by an index, because that is the form the
        // loads vectorize in — see [`F32x4::from_chunk`].
        let mut chunks = Vec3x4::chunks(cand_x, cand_y, cand_z);
        let mut at = 0usize;
        'full: {
            for candidates in &mut chunks {
                let mut hits = (candidates - wide_centre).length_squared().le_mask(h_sq);
                while hits != 0 {
                    let lane = hits.trailing_zeros() as usize;
                    hits &= hits - 1;
                    if neighbours.len() - base >= MAX_NEIGHBOURS {
                        break 'full;
                    }
                    neighbours.push(cand_id[at + lane]);
                }
                at += 4;
            }
            let (rx, ry, rz) = chunks.remainder();
            for lane in 0..rx.len() {
                let d = Vec3::new(rx[lane], ry[lane], rz[lane]) - centre;
                if d.length_squared() <= h * h {
                    if neighbours.len() - base >= MAX_NEIGHBOURS {
                        break 'full;
                    }
                    neighbours.push(cand_id[at + lane]);
                }
            }
        }

        // Deliberately *not* sorted.
        //
        // Determinism (§34) needs the summation order to be fixed, and it
        // already is: the spans are the 27 cells in a fixed sequence, each
        // bucket is ascending from the counting sort, and the bucket dedup is
        // order-independent — so the list is a pure function of the particle
        // positions. Sorting would additionally make it *ascending*, which
        // nothing needs.
        //
        // The distinction is worth 20% of a fluid step. Sorting ~35 indices
        // per particle per substep is 45% of the neighbour build, which is
        // itself the largest single cost in the solver — larger than the
        // density solve it feeds. It was there because "fixed order" and
        // "sorted order" got conflated while writing the determinism rules.
        let found = neighbours.len() - base;
        count[i] = found as u32;
        if found >= MAX_NEIGHBOURS {
            *truncated += 1;
        }
        *peak_neighbours = (*peak_neighbours).max(found);
    }

    // What the lists were built from, for `neighbours_are_stale`.
    scratch.built_x.clear();
    scratch.built_x.extend_from_slice(&scratch.lx);
    scratch.built_y.clear();
    scratch.built_y.extend_from_slice(&scratch.ly);
    scratch.built_z.clear();
    scratch.built_z.extend_from_slice(&scratch.lz);
}

/// Measure how much of each particle's kernel support is inside solid
/// geometry, for the density correction above.
///
/// Once per substep, alongside the neighbour build, and not once per density
/// iteration: the particles move by well under a millimetre between
/// iterations and the occlusion is a smooth function of distance.
///
/// Occlusions from several surfaces are summed rather than maximised. Two
/// perpendicular walls each remove close to half the support and the wedge
/// they share is small, so summing is the better estimate; the total is capped
/// below one so that a particle wedged in a corner cannot be told it is
/// surrounded by more fluid than exists.
pub fn build_boundary_density(fluid: &FluidSet, bodies: &BodySet, scratch: &mut FluidScratch) {
    let h = fluid.params.smoothing_radius;
    scratch.boundary.iter_mut().for_each(|b| *b = 0.0);
    if bodies.is_empty() {
        return;
    }
    for body in 0..bodies.len() {
        if bodies.is_removed(body) {
            continue;
        }
        let pose = bodies.pose(body);
        let bounds = bodies.aabb(body).expanded(h);
        let shape = &bodies.shape[body];
        for i in 0..fluid.len() {
            let point = fluid.particles.translation(i);
            if !bounds.contains_point(point) {
                continue;
            }
            // Distance from the particle *centre*, because that is where the
            // kernel is centred and the occlusion is a statement about the
            // kernel. Measuring from the particle surface instead — subtracting
            // its radius, on the reasoning that a particle can never touch the
            // wall — over-states the occlusion by a factor of more than two at
            // the floor (0.5 of the support rather than 0.22), and the fluid
            // is then told it is 50% over-dense the instant it lands.
            let distance = shape
                .signed_distance_local(pose.inverse_transform_point(point))
                .max(0.0);
            if distance >= h {
                continue;
            }
            scratch.boundary[i] += boundary_occlusion(distance, h);
        }
    }
    // A particle cannot be missing more fluid than a full kernel holds.
    for b in &mut scratch.boundary {
        *b = b.min(0.9);
    }
}

/// The two SPH kernels, four lanes wide, with their normalizations computed
/// once.
///
/// Two things are folded in here that the scalar [`poly6`] and
/// [`spiky_gradient`] do per call: the normalization — `315/(64 π h⁹)` is a
/// ninth power and a division — and the support test. Both matter on a loop
/// that runs thirty-three times per particle, three times per iteration, twice
/// per substep, four substeps a step.
///
/// ## The support test is arithmetic, not a branch
///
/// `poly6` returns zero outside `h`, and the vector form gets that from
/// `max(h² - r², 0)`: beyond the radius the difference clamps to zero and the
/// cube of zero is zero. Identical to the scalar result, with no comparison
/// and no mask — which matters because building a lane mask is the one
/// operation NEON has no instruction for (see [`fusion_math::simd`]).
///
/// ## The reciprocal is guarded the same way
///
/// The spiky gradient carries a `1/r` and has to survive `r = 0`, which
/// happens on every particle: its own index is in its neighbour list. The
/// scalar version branches on `r < 1e-9` and returns zero. The vector version
/// uses `r / (r² + ε)` instead, which *is* `1/r` to well inside `f32` for any
/// separation a fluid can have, and goes smoothly to zero as `r` does rather
/// than to infinity.
///
/// That is what removes the self-term skip from the loops below: a particle's
/// own entry produces a zero offset, a zero gradient and a zero contribution,
/// so it can be summed like any other neighbour instead of tested for.
struct Kernel {
    h: F32x4,
    h_sq: F32x4,
    poly6_norm: F32x4,
    spiky_norm: F32x4,
    epsilon: F32x4,
}

impl Kernel {
    fn new(h: f32) -> Self {
        Self {
            h: F32x4::splat(h),
            h_sq: F32x4::splat(h * h),
            poly6_norm: F32x4::splat(315.0 / (64.0 * std::f32::consts::PI * h.powi(9))),
            spiky_norm: F32x4::splat(-45.0 / (std::f32::consts::PI * h.powi(6))),
            // Small enough to be invisible at any separation a fluid actually
            // has — at a tenth of a millimetre it shifts `1/r` by one part in
            // 10¹⁰ — and large enough to take the singularity out. At `1e-9`,
            // where the scalar guard gives up and returns zero, this returns
            // half of `1/r`; below that it falls away to zero rather than
            // running to infinity.
            epsilon: F32x4::splat(1e-18),
        }
    }

    #[inline(always)]
    fn poly6(&self, r_sq: F32x4) -> F32x4 {
        let d = (self.h_sq - r_sq).max(F32x4::ZERO);
        self.poly6_norm * d * d * d
    }

    /// Gradient of the spiky kernel for four offsets at once.
    #[inline(always)]
    fn spiky(&self, offset: Vec3x4, r_sq: F32x4) -> Vec3x4 {
        let r = r_sq.sqrt();
        let d = (self.h - r).max(F32x4::ZERO);
        let inv_r = r / (r_sq + self.epsilon);
        offset.scale_lanes(self.spiky_norm * d * d * inv_r)
    }
}

/// Four neighbour indices, padding a short chunk with `pad`.
#[inline(always)]
fn chunk_indices(chunk: &[u32], pad: usize) -> [usize; 4] {
    let at = |k: usize| chunk.get(k).map_or(pad, |&j| j as usize);
    [at(0), at(1), at(2), at(3)]
}

/// Evaluate both kernels for every neighbour pair, into the parallel arrays
/// described on [`FluidScratch::kernel_w`].
///
/// This is the one gather-bound loop left in the density solve, and it is
/// written four wide because it is the only place the gather is paid. The pad
/// lanes of a short chunk repeat the centre particle's index, whose gradient
/// is zero and whose weight is overwritten by the scalar tail below — so the
/// tail is written after the wide loop rather than being fused into it.
fn build_kernel_cache(scratch: &mut FluidScratch, h: f32) {
    let pairs = scratch.neighbours.len();
    // Resized, not cleared and refilled. Every slot is written below, so
    // zeroing first is half a megabyte of memset per iteration that nothing
    // reads — and `resize` on a vector that is already the right length is
    // free.
    for axis in [
        &mut scratch.kernel_w,
        &mut scratch.kernel_gx,
        &mut scratch.kernel_gy,
        &mut scratch.kernel_gz,
    ] {
        axis.resize(pairs, 0.0);
    }

    let kernel = Kernel::new(h);
    let FluidScratch {
        neighbours,
        start,
        count,
        lx,
        ly,
        lz,
        kernel_w,
        kernel_gx,
        kernel_gy,
        kernel_gz,
        ..
    } = scratch;

    for i in 0..start.len() {
        let (a, c) = (start[i] as usize, count[i] as usize);
        let centre = Vec3x4::splat(Vec3::new(lx[i], ly[i], lz[i]));
        let mut t = 0;
        while t < c {
            let chunk = &neighbours[a + t..(a + t + 4).min(a + c)];
            let offsets = centre - Vec3x4::gather(lx, ly, lz, chunk_indices(chunk, i));
            let r_sq = offsets.length_squared();
            let w = kernel.poly6(r_sq);
            let g = kernel.spiky(offsets, r_sq);
            for lane in 0..chunk.len() {
                kernel_w[a + t + lane] = w.lane(lane);
                kernel_gx[a + t + lane] = g.x.lane(lane);
                kernel_gy[a + t + lane] = g.y.lane(lane);
                kernel_gz[a + t + lane] = g.z.lane(lane);
            }
            t += 4;
        }
    }
}

/// The cached spiky gradient of one neighbour pair.
#[inline(always)]
fn cached_gradient(scratch: &FluidScratch, at: usize) -> Vec3 {
    Vec3::new(
        scratch.kernel_gx[at],
        scratch.kernel_gy[at],
        scratch.kernel_gz[at],
    )
}

/// One particle's run of cached gradients, as three equal-length slices ready
/// for [`Vec3x4::chunks`].
#[inline(always)]
fn cached_gradients(scratch: &FluidScratch, run: usize, len: usize) -> (&[f32], &[f32], &[f32]) {
    (
        &scratch.kernel_gx[run..run + len],
        &scratch.kernel_gy[run..run + len],
        &scratch.kernel_gz[run..run + len],
    )
}

/// Macklin & Müller's artificial pressure term, from a cached Poly6 weight.
///
/// An extra repulsion that grows as particles approach, cancelling the tensile
/// instability that otherwise pulls them into strings and sheets. It also
/// reads as surface tension, which is a happy accident rather than a model of
/// one.
///
/// The only term in the solver with no vector instruction behind it — `powf`
/// is four library calls — and it is off by default, so its caller keeps the
/// branch outside the chunk arithmetic rather than inside it.
#[inline(always)]
fn artificial_pressure(w: F32x4, reference: f32, exponent: f32, k: f32) -> F32x4 {
    (w / F32x4::splat(reference)).powf(exponent).scale(-k)
}

/// Where particle `i`'s run starts in the neighbour list and the kernel cache,
/// and how long it is.
#[inline(always)]
fn run_of(scratch: &FluidScratch, i: usize) -> (usize, usize) {
    (scratch.start[i] as usize, scratch.count[i] as usize)
}

/// One density-constraint iteration: densities, multipliers, corrections.
///
/// Jacobi, not Gauss-Seidel, and deliberately the opposite choice from the one
/// the contact solver makes. Contacts are few and sequential dependence is
/// what stops a four-point manifold applying four times the impulse it needs.
/// A fluid particle has thirty neighbours and the constraint is symmetric;
/// solving in place would make the answer depend on particle index — the
/// bottom-left corner of a dam break would be stiffer than the top-right — and
/// the published method is Jacobi for exactly that reason.
pub fn solve_density(fluid: &mut FluidSet, scratch: &mut FluidScratch, relaxation: f32) {
    let n = fluid.len();
    if n == 0 || scratch.rest <= 0.0 {
        return;
    }
    let params = fluid.params;
    let h = params.smoothing_radius;
    let inv_rest = 1.0 / scratch.rest;

    // Re-read the positions. The neighbour *lists* are from the top of the
    // substep and stay that way — rebuilding them per iteration is the
    // expensive half and particles do not move far enough between iterations
    // to change who is a neighbour. The *geometry* is another matter: without
    // this refresh every iteration evaluates the density at the positions the
    // substep started from, so a second iteration recomputes an identical
    // correction and applies it again rather than converging. Two iterations
    // at half strength then really were one iteration at full strength, and
    // the under-relaxation had to absorb it.
    scratch.refresh_local(fluid);

    build_kernel_cache(scratch, h);

    // ---- densities ----
    //
    // A sum of cached weights, four at a time off contiguous memory, with a
    // scalar tail for the last one to three. Every loop below has this shape;
    // it is what the cache was built to allow.
    for i in 0..n {
        let (run, len) = run_of(scratch, i);
        let mut wide = F32x4::ZERO;
        let mut chunks = scratch.kernel_w[run..run + len].chunks_exact(4);
        for chunk in &mut chunks {
            wide = wide + F32x4::from_chunk(chunk);
        }
        let mut density = wide.sum();
        for &w in chunks.remainder() {
            density += w;
        }
        // Plus the fluid that a wall is standing in for. Without this term a
        // particle against a surface reads under-dense and the fluid packs
        // itself into the container; see `boundary_occlusion`.
        scratch.density[i] = density + scratch.boundary[i] * scratch.rest;
    }

    // ---- multipliers ----
    //
    // lambda_i = -C_i / (sum_k |grad_k C_i|^2 + eps)
    //
    // The sum runs over every particle whose position appears in C_i: the
    // neighbours, each contributing -grad W, and particle i itself,
    // contributing the sum of all of them. Dropping the self term is a common
    // shortcut and it under-estimates the denominator by roughly a factor of
    // two, which shows up as a fluid that over-corrects and jitters.
    let epsilon = params.relaxation;
    for i in 0..n {
        let c = scratch.density[i] * inv_rest - 1.0;
        // Only compression is resisted, which is a deliberate departure from
        // the published method. A particle at the surface is genuinely short
        // of neighbours; solving its deficit pulls it inward, and since its
        // inward neighbours are being pulled too, the standard result is
        // "tensile instability" — particles collapse into strings and sheets.
        // Macklin & Müller fix that with an artificial pressure term.
        //
        // Clamping instead removes the cause rather than cancelling it, at a
        // cost that is worth stating: with no attractive term at all, a free
        // surface has no cohesion, so a thin splash disperses into individual
        // particles where real water would hold together. XSPH viscosity
        // supplies some apparent cohesion; it is not the same thing.
        // See `FluidParams::surface_tension` for the other half of this pair.
        if c <= 0.0 {
            scratch.lambda[i] = 0.0;
            continue;
        }
        // No self-term skip: the cached gradient is exactly zero at zero
        // separation, so a particle's own entry contributes nothing to either
        // sum. See [`Kernel`].
        let (run, len) = run_of(scratch, i);
        let (gx, gy, gz) = cached_gradients(scratch, run, len);
        let wide_rest = F32x4::splat(inv_rest);
        let mut wide_sum_sq = F32x4::ZERO;
        let mut wide_self = Vec3x4::ZERO;
        let mut chunks = Vec3x4::chunks(gx, gy, gz);
        for gradient in &mut chunks {
            let g = gradient.scale_lanes(wide_rest);
            wide_self = wide_self + g;
            wide_sum_sq = wide_sum_sq + g.length_squared();
        }
        let mut sum_sq = wide_sum_sq.sum();
        let mut grad_self = wide_self.sum();
        let (rx, ry, rz) = chunks.remainder();
        for t in 0..rx.len() {
            let g = Vec3::new(rx[t], ry[t], rz[t]) * inv_rest;
            grad_self += g;
            sum_sq += g.length_squared();
        }
        sum_sq += grad_self.length_squared();
        scratch.lambda[i] = -c / (sum_sq + epsilon);
    }

    // ---- position corrections ----
    let k = params.surface_tension;
    let exponent = params.tension_exponent;
    let w_reference = poly6(
        (params.tension_reference * h) * (params.tension_reference * h),
        h,
    );
    for i in 0..n {
        if !fluid.particles.is_movable(i) {
            scratch.delta[i] = Vec3::ZERO;
            continue;
        }
        let (run, len) = run_of(scratch, i);
        let (gx, gy, gz) = cached_gradients(scratch, run, len);
        let list = &scratch.neighbours[run..run + len];
        let weights = &scratch.kernel_w[run..run + len];
        let lambda_i = F32x4::splat(scratch.lambda[i]);
        let tension = k > 0.0 && w_reference > 0.0;

        let mut wide = Vec3x4::ZERO;
        let mut chunks = Vec3x4::chunks(gx, gy, gz);
        let mut pairs = list.chunks_exact(4).zip(weights.chunks_exact(4));
        for (gradient, (js, w)) in (&mut chunks).zip(&mut pairs) {
            // The one gather left: a neighbour's multiplier, which is indexed
            // by particle and cannot be laid out along the pair list because
            // it is not a property of the pair.
            let index = chunk_indices(js, i);
            let mut weight = lambda_i + F32x4::gather(&scratch.lambda, index);
            if tension {
                weight =
                    weight + artificial_pressure(F32x4::from_chunk(w), w_reference, exponent, k);
            }
            wide = wide + gradient.scale_lanes(weight);
        }
        let mut correction = wide.sum();
        let (rx, ry, rz) = chunks.remainder();
        // Where the chunked part stopped: the three iterators walk in step, so
        // one remainder length places all of them.
        let tail = len - rx.len();
        for t in 0..rx.len() {
            let j = list[tail + t] as usize;
            let mut weight = scratch.lambda[i] + scratch.lambda[j];
            if tension {
                weight +=
                    artificial_pressure(F32x4::splat(weights[tail + t]), w_reference, exponent, k)
                        .lane(0);
            }
            correction += Vec3::new(rx[t], ry[t], rz[t]) * weight;
        }
        scratch.delta[i] = correction * (inv_rest * relaxation);
    }

    for i in 0..n {
        if !fluid.particles.is_movable(i) {
            continue;
        }
        fluid.particles.displace(i, scratch.delta[i]);
        scratch.accumulated[i] += scratch.delta[i];
    }
}

/// Vorticity confinement, then XSPH viscosity. Velocity-level, after recovery.
///
/// Order matters and this is the published one. Vorticity confinement puts
/// rotational energy *back*; XSPH takes high-frequency noise *out*. Running
/// viscosity first and confinement second amplifies whatever noise the
/// viscosity pass failed to remove, and a still pool starts to boil.
pub fn apply_viscosity_and_vorticity(fluid: &mut FluidSet, scratch: &mut FluidScratch, h_dt: f32) {
    let n = fluid.len();
    if n == 0 {
        return;
    }
    let params = fluid.params;
    let h = params.smoothing_radius;
    let inv_rest = 1.0 / scratch.rest;

    // Normally the cache is whatever the last density iteration left, which is
    // the geometry this pass has always used. With the density solve disabled
    // — `fluid_iterations = 0`, or a fluid whose rest density has not
    // calibrated — there is nothing to inherit, so build it here.
    if scratch.kernel_w.len() != scratch.neighbours.len() {
        build_kernel_cache(scratch, h);
    }

    scratch.velocity_in.clear();
    scratch
        .velocity_in
        .extend_from_slice(&fluid.particles.velocity);

    // ---- vorticity ----
    //
    // The curl estimate carries a `1/rho` that is easy to drop and expensive
    // to drop. An SPH curl is
    //
    //     omega_i = (1 / rho_i) sum_j m_j (v_j - v_i) x grad W_ij
    //
    // and without the density division the result is not a vorticity at all —
    // it has units of 1/(s m^3) and its magnitude scales with the *kernel*,
    // so at a 10 cm smoothing radius `grad W` peaks near 3.6e4 and a
    // millimetre-per-second velocity difference across thirty neighbours
    // "confines" at a thousand metres per second squared. Vorticity
    // confinement is a positive feedback by construction — it exists to
    // amplify rotation the solve damped out — so an over-scaled one does not
    // merely look wrong, it diverges. A still block of water left alone in
    // vacuum reached the speed clamp within four steps.
    //
    // With the division, `omega` is a real vorticity in 1/s, the gain below is
    // a number of order one, and the same block does not move at all.
    if params.vorticity > 0.0 {
        for i in 0..n {
            let inv_density = if scratch.density[i] > 1e-6 {
                1.0 / scratch.density[i]
            } else {
                0.0
            };
            let (run, len) = run_of(scratch, i);
            let mut curl = Vec3::ZERO;
            for t in 0..len {
                // No self-term skip anywhere in this function: the cached
                // gradient is zero at zero separation and the velocity
                // difference against oneself is zero, so `j == i` contributes
                // nothing to any of the three sums.
                let j = scratch.neighbours[run + t] as usize;
                let gradient = cached_gradient(scratch, run + t);
                curl += (scratch.velocity_in[j] - scratch.velocity_in[i]).cross(gradient);
            }
            scratch.curl[i] = curl * inv_density;
        }
        for i in 0..n {
            if !fluid.particles.is_movable(i) {
                continue;
            }
            // eta = grad |omega|, same kernel and same normalization.
            let (run, len) = run_of(scratch, i);
            let mut eta = Vec3::ZERO;
            for t in 0..len {
                let j = scratch.neighbours[run + t] as usize;
                eta += cached_gradient(scratch, run + t) * scratch.curl[j].length();
            }
            let (direction, magnitude) = (eta * inv_rest).normalize_and_length();
            if magnitude < 1e-9 {
                continue;
            }
            let force = direction.cross(scratch.curl[i]) * params.vorticity;
            fluid.particles.velocity[i] += force * h_dt;
        }
    }

    // ---- XSPH viscosity ----
    //
    // Weighted by the *rest* density rather than each particle's own. The
    // textbook form divides by `rho_j`, which is smaller at a free surface and
    // therefore weights surface neighbours more heavily — precisely the
    // particles whose density estimate is least trustworthy. Using rest makes
    // the blend a fixed fraction everywhere and costs nothing that matters:
    // this is a numerical smoother, not a transport model.
    if params.viscosity > 0.0 {
        for i in 0..n {
            if !fluid.particles.is_movable(i) {
                continue;
            }
            let (run, len) = run_of(scratch, i);
            let mut blend = Vec3::ZERO;
            for t in 0..len {
                let j = scratch.neighbours[run + t] as usize;
                blend +=
                    (scratch.velocity_in[j] - scratch.velocity_in[i]) * scratch.kernel_w[run + t];
            }
            fluid.particles.velocity[i] += blend * (params.viscosity * inv_rest);
        }
    }
}

/// Keep fluid particles inside rigid geometry, and push the geometry back.
///
/// Thin wrapper over the same routine cloth uses — a fluid particle meeting a
/// box is the same problem as a cloth particle meeting a box, and reusing it
/// is what makes a beach ball float without a line of buoyancy code anywhere
/// in the engine. Buoyancy is not modelled; it is what a column of fluid
/// particles pushing upward on a light body *does*.
#[allow(clippy::too_many_arguments)]
pub fn solve_boundaries(
    fluid: &mut FluidSet,
    bodies: &mut BodySet,
    delta: &mut [Vec3],
    body_delta_position: &mut [Vec3],
    body_delta_rotation: &mut [Vec3],
    tolerance: f32,
    friction: f32,
    h: f32,
) -> usize {
    crate::soft::solve_particle_body_contacts(
        &mut fluid.particles,
        bodies,
        delta,
        body_delta_position,
        body_delta_rotation,
        tolerance,
        friction,
        h,
    )
}

/// Clamp fluid speed.
///
/// A fluid has no equivalent of a rigid body's contact budget: a particle that
/// ends up deep inside another after a bad step is corrected by the full
/// density gradient in one go, and `delta / h` at an 8th of a 240 Hz step is a
/// large number. The clamp is not physics and is not pretending to be — it is
/// the difference between a splash that recovers and one that ejects a
/// particle to infinity and takes the neighbour grid's bounds with it.
pub fn clamp_speed(fluid: &mut FluidSet, max_speed: f32) -> usize {
    if max_speed <= 0.0 {
        return 0;
    }
    let mut clamped = 0;
    for v in &mut fluid.particles.velocity {
        let speed = v.length();
        if speed > max_speed {
            *v *= max_speed / speed;
            clamped += 1;
        }
    }
    clamped
}

/// Total momentum of a fluid, kg m/s. A conservation check with teeth: the
/// density constraint applies equal and opposite corrections, so in the
/// absence of gravity and boundaries this must not drift.
pub fn momentum(fluid: &FluidSet) -> Vec3 {
    let m = fluid.params.particle_mass();
    let mut total = Vec3::ZERO;
    for v in &fluid.particles.velocity {
        total += *v * m;
    }
    total
}

/// What this solver is not.
///
/// Surfaced through [`crate::SolverInfo::caveat`] rather than left in the
/// documentation, per PLAN.md §9: the accuracy tier has to appear where the
/// result is used.
///
/// - **Density is enforced approximately.** A finite number of Jacobi
///   iterations leaves a residual compression, typically a percent or two, and
///   it grows with the timestep and with how violent the flow is. Reported by
///   [`FluidScratch::compression_error`].
/// - **Viscosity is not a viscosity.** XSPH blending is a numerical smoother.
///   The effective kinematic viscosity depends on resolution and timestep, so
///   the coefficient is a look-right dial and not a material property.
/// - **Vorticity confinement adds energy.** By construction: it exists to put
///   back detail the solve removed, and it does not ask where the energy comes
///   from.
/// - **No surface tension model.** The artificial-pressure term looks like
///   surface tension and is not derived from one; it will not give a correct
///   contact angle or capillary rise.
/// - **Free-slip walls.** Boundaries are handled by pushing particles out of
///   rigid shapes, so there is no boundary layer and no no-slip condition.
pub const CAVEAT: &str = "Position-based fluids enforce incompressibility \
                          approximately and add energy through vorticity \
                          confinement; densities and velocities are plausible, \
                          not measurements.";

/// Apply an external acceleration field to a fluid — wind, a stirring force,
/// a moving frame. Separated from the solve so a caller can add one without
/// reaching into the velocity array.
pub fn accelerate(fluid: &mut FluidSet, acceleration: Vec3, dt: f32) {
    for v in &mut fluid.particles.velocity {
        *v += acceleration * dt;
    }
}

/// Push a fluid particle out of a shape without a body behind it — used for
/// domain walls, which have no rigid body and should not need one.
pub fn confine_to_bounds(fluid: &mut FluidSet, bounds: &fusion_math::Aabb, restitution: f32) {
    let r = fluid.params.particle_radius();
    let lo = bounds.min + Vec3::splat(r);
    let hi = bounds.max - Vec3::splat(r);
    for i in 0..fluid.len() {
        let p = fluid.particles.translation(i);
        let clamped = p.clamp(lo, hi);
        if clamped == p {
            continue;
        }
        fluid.particles.displace(i, clamped - p);
        // Reflect only the component that hit the wall.
        for k in 0..3 {
            if clamped[k] != p[k] {
                let v = fluid.particles.velocity[i][k];
                if (clamped[k] - p[k]) * v < 0.0 {
                    fluid.particles.velocity[i][k] = -v * restitution;
                }
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The solver stopped calling [`poly6`] and [`spiky_gradient`] and started
    /// calling [`Kernel`], which evaluates the same two functions four lanes at
    /// a time with the branches replaced by arithmetic. The scalar pair is
    /// pinned against analytic values in `tests/fluid_kernels.rs`; this pins
    /// the pair that actually runs against the pair that is checked.
    ///
    /// Without it the kernels have no test at all — which is the failure mode
    /// where a fluid keeps working, keeps looking like water, and is 4% denser
    /// than it says it is.
    #[test]
    fn the_vector_kernels_agree_with_the_scalar_ones() {
        let h = 0.2;
        let kernel = Kernel::new(h);
        for step in 0..=200 {
            let r = step as f32 * h / 100.0; // out to twice the radius
            let r_sq = r * r;
            let offset = Vec3::new(r * 0.6, r * -0.8, 0.0); // length exactly r

            // Poly6 is bit-identical: `max(h² - r², 0)` reproduces the early
            // return exactly, and the three multiplies are in the same order.
            let wide = kernel.poly6(F32x4::splat(r_sq)).lane(0);
            assert_eq!(wide, poly6(r_sq, h), "poly6 at r = {r}");

            // The gradient is not, because the branch on `r < 1e-9` became a
            // guarded reciprocal. Everywhere a fluid actually has particles,
            // the difference is rounding.
            let wide = kernel
                .spiky(Vec3x4::splat(offset), F32x4::splat(r_sq))
                .lane(0);
            let scalar = spiky_gradient(offset, r, h);
            let scale = scalar.length().max(1e-6);
            assert!(
                (wide - scalar).length() / scale < 1e-5,
                "spiky at r = {r}: {wide:?} vs {scalar:?}"
            );
        }
    }

    /// Zero separation is not a corner case here: every particle's own index is
    /// in its own neighbour list, so this path runs tens of times per particle
    /// per iteration. It is what replaced the `j == i` skip in six loops.
    #[test]
    fn the_gradient_vanishes_at_zero_separation_instead_of_diverging() {
        let kernel = Kernel::new(0.2);
        let at = |r: f32| {
            let offset = Vec3::new(r, 0.0, 0.0);
            kernel
                .spiky(Vec3x4::splat(offset), F32x4::splat(r * r))
                .lane(0)
                .length()
        };
        assert_eq!(at(0.0), 0.0);
        // Approaching zero the guarded reciprocal turns over and comes back
        // down, rather than running to infinity the way a bare `1 / r` would.
        assert!(at(1e-12) < at(1e-6), "{} vs {}", at(1e-12), at(1e-6));
        assert!(at(1e-30).is_finite());
    }

    /// Past the smoothing radius both kernels are exactly zero — which is what
    /// lets the neighbour lists be built from a cheaper cutoff test than the
    /// solve strictly needs. A candidate accepted a hair beyond `h` costs a
    /// list entry and contributes nothing.
    #[test]
    fn both_kernels_are_exactly_zero_outside_the_support() {
        let h = 0.15;
        let kernel = Kernel::new(h);
        for r in [h, h * 1.0001, h * 2.0, h * 50.0] {
            assert_eq!(kernel.poly6(F32x4::splat(r * r)).lane(0), 0.0, "w at {r}");
            let g = kernel
                .spiky(Vec3x4::splat(Vec3::new(r, 0.0, 0.0)), F32x4::splat(r * r))
                .lane(0);
            assert_eq!(g, Vec3::ZERO, "grad at {r}");
        }
    }
}
