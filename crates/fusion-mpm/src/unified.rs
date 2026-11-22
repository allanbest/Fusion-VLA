//! The unified continuum solver: every solid and every fluid on one grid.
//!
//! The classic path solves each continuum on its own — every deformable solid
//! gets a dense MLS-MPM grid sized to its own bounding box, and fluids are a
//! separate position-based (PBF) solve. That is what lets a jelly cube and a
//! pool of water pass straight through each other: they never meet in one
//! solve. This module puts them all on one background grid, so they meet the
//! way the method was made for — two materials that share a grid node share
//! a velocity there — and water pushes a soft block, a block displaces water,
//! and a light one floats, with no coupling code at all.
//!
//! ```text
//!     sort        material points by the grid block they fall in (once a step)
//!     P2G         mass, momentum and stress onto the grid, in parallel
//!     grid        one explicit update; rigid bodies as moving walls, two-way
//!     G2P         velocity, the affine field and the deformation back, in parallel
//! ```
//!
//! ## The grid is sparse
//!
//! Nodes live in blocks of 4×4×4, allocated only where material is. A dense
//! grid over a pool and a block thrown ten metres from it is a grid the size
//! of the gap; this one is the size of the material. The classic solver's
//! cost grows with the bounding box, and this one's with the material.
//!
//! ## In parallel, and the same answer on any machine
//!
//! The two transfers and the grid update run on every core. The transfer in
//! is the part that is hard to parallelise *deterministically*: many points
//! add into the same node, and floating-point addition depends on the order.
//! So the points are cut into [`CHUNKS`] fixed runs, each run accumulates into
//! its own blocks, and the runs are merged in a fixed order. The cut does not
//! depend on how many cores there are, so one core and sixteen give the same
//! bits — measured, in `tests/unified.rs`.
//!
//! ## Fluids
//!
//! Weakly compressible: a fluid point carries its volume ratio `J`, and its
//! pressure is Tait's equation of state, `p = ρ₀c²/γ (J^-γ - 1)` with `γ = 7`,
//! clamped at zero so a free surface pulls nothing with it. The speed of sound
//! `c` is a numerical one — [`UnifiedSolver::sound_speed`], 20 m/s by default
//! rather than water's 1 480 — which keeps compression to about one percent
//! at the speeds a scene reaches while letting the step be fifty times longer.
//! That is the trade every weakly compressible method makes, said out loud.
//! Viscosity is a real Newtonian term, `2μ ε̇`.
//!
//! ## Material structure
//!
//! A solid can carry fibres ([`fusion_core::Fibre`]): a direction in its rest
//! frame along which it is stiffer in tension, added to its constitutive law
//! as the standard fibre-reinforced term. Wood's grain, a muscle, a rope.
//!
//! ## What it does not do yet
//!
//! Heat: a solid's temperature is carried but not diffused, and nothing melts
//! on this path. Rigid bodies are not material points — they stay rigid, on
//! the rigid solver, and meet the grid as moving walls that are pushed back.

/// Stability estimates and transactional-step failure contracts.
pub mod control;
mod implicit;
mod measure;
mod projection;
mod separation;
pub use control::{LimitKind, RepairPolicy, Repairs, StepFailure, StepLimit};
pub use implicit::ImplicitReport;
pub use measure::ContinuumTotals;
pub use projection::ProjectionReport;

/// How a fluid on this grid gets its pressure.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum FluidModel {
    /// From how far each point is squeezed, through Tait's stiff equation of
    /// state, at [`UnifiedSolver::sound_speed`]. Compresses about a percent,
    /// and the speed of sound sets the step.
    #[default]
    WeaklyCompressible,
    /// From a pressure projection on the grid each substep: the velocity is
    /// made divergence-free, as the incompressible Navier–Stokes equations
    /// ask, so nothing compresses and no sound speed limits the step, and
    /// crowded points are moved apart. Each point's volume ratio is held at
    /// one: the grid keeps the volume, and a ratio integrated from the
    /// velocity gradient only drifted. See `projection`.
    Incompressible,
}

use std::collections::HashMap;
use std::sync::atomic::{AtomicUsize, Ordering};

use fusion_core::{BodySet, FluidSet, SolidSet};
use fusion_math::{outer_product, to_f64, DVec3, Mat3, Vec3};

use crate::grid::weights;
use crate::model::{self, hardened, return_map, stress, stress_with_damage};

/// Nodes along one edge of a grid block.
const EDGE: i64 = 4;
/// Nodes in one block.
const BLOCK: usize = 64;

/// Points per thread when the solver picks how many cores to use, measured:
/// one per five hundred was best or level on every scene tried, from a
/// thousand-point porcelain block to twenty-four thousand points of water.
const POINTS_PER_THREAD: usize = 500;

/// Fixed runs the material points are cut into for the transfer in. Fixed, so
/// that the answer does not depend on how many cores ran it.
pub const CHUNKS: usize = 32;

/// Blocks past which a step is refused rather than allocated: eight million
/// nodes, the classic solver's ceiling.
pub const MAX_BLOCKS: usize = 125_000;

/// Tait's exponent for water.
const GAMMA: f32 = 7.0;

/// A fast hash for block coordinates. Deterministic, like every hash here:
/// nothing iterates the map, so its order never reaches the answer — the
/// blocks are numbered in the order the points first touched them.
#[derive(Clone, Copy, Debug, Default)]
struct BlockHash(u64);

impl std::hash::Hasher for BlockHash {
    fn finish(&self) -> u64 {
        self.0
    }
    fn write(&mut self, bytes: &[u8]) {
        for b in bytes {
            self.0 = (self.0 ^ *b as u64).wrapping_mul(0x100_0000_01b3);
        }
    }
    fn write_i64(&mut self, v: i64) {
        self.0 = (self.0 ^ v as u64)
            .wrapping_mul(0x9e37_79b9_7f4a_7c15)
            .rotate_left(23);
    }
    fn write_usize(&mut self, _: usize) {}
}

type BlockMap<V> = HashMap<[i64; 3], V, std::hash::BuildHasherDefault<BlockHash>>;

/// The blocks a point's 3×3×3 stencil can reach: two along each axis, the
/// base node's and the next. Looked up once per point rather than once per
/// node — twenty-seven hash lookups a point became eight.
#[derive(Clone, Copy)]
struct Reach {
    base: [i64; 3],
    first: [i64; 3],
    blocks: [[[u32; 2]; 2]; 2],
}

impl Reach {
    fn new(base: [i64; 3], index: &BlockMap<u32>) -> Self {
        let first = block_of(base);
        let mut blocks = [[[0u32; 2]; 2]; 2];
        for (z, plane) in blocks.iter_mut().enumerate() {
            for (y, row) in plane.iter_mut().enumerate() {
                for (x, slot) in row.iter_mut().enumerate() {
                    let key = [
                        first[0] + x as i64,
                        first[1] + y as i64,
                        first[2] + z as i64,
                    ];
                    // A neighbour block the stencil never reaches may not
                    // exist; it is never read either.
                    *slot = index.get(&key).copied().unwrap_or(u32::MAX);
                }
            }
        }
        Self {
            base,
            first,
            blocks,
        }
    }

    /// Block and slot within it of stencil node `(i, j, k)`.
    #[inline]
    fn node(&self, i: usize, j: usize, k: usize) -> (u32, usize) {
        let node = [
            self.base[0] + i as i64,
            self.base[1] + j as i64,
            self.base[2] + k as i64,
        ];
        let b = block_of(node);
        let block = self.blocks[(b[2] - self.first[2]) as usize][(b[1] - self.first[1]) as usize]
            [(b[0] - self.first[0]) as usize];
        (block, local_index(node))
    }
}

/// No continuum: an empty field slot.
const NONE: u32 = u32::MAX;

/// One block of grid nodes, each with two field slots.
///
/// A node carries the material of up to two continua *separately* — two
/// velocity fields, after Bardenhagen's multi-velocity MPM — so that where a
/// jelly block meets another, or water meets a solid, the two can slide past
/// each other and come apart instead of sharing one velocity and fusing. Away
/// from interfaces only the first slot is used and the second costs nothing
/// but memory. A third continuum at the same node is folded into the second
/// slot: rare, and the old shared-velocity behaviour for those two.
#[derive(Clone)]
struct Block {
    /// Which continuum each slot holds, [`NONE`] where it holds none.
    field: [[u32; 2]; BLOCK],
    mass: [[f32; 2]; BLOCK],
    /// Momentum during the transfer in, velocity after the grid update.
    momentum: [[Vec3; 2]; BLOCK],
    /// The gradient of each slot's mass, whose difference is the interface
    /// normal.
    gradient: [[Vec3; 2]; BLOCK],
}

impl Default for Block {
    fn default() -> Self {
        Self {
            field: [[NONE; 2]; BLOCK],
            mass: [[0.0; 2]; BLOCK],
            momentum: [[Vec3::ZERO; 2]; BLOCK],
            gradient: [[Vec3::ZERO; 2]; BLOCK],
        }
    }
}

impl Block {
    /// The slot continuum `field` goes in at node `n`, claiming a free one.
    #[inline]
    fn slot(&mut self, n: usize, field: u32) -> usize {
        let [a, b] = self.field[n];
        if a == field || a == NONE {
            self.field[n][0] = field;
            0
        } else {
            if b == NONE {
                self.field[n][1] = field;
            }
            1
        }
    }

    /// The slot continuum `field` reads at node `n`.
    #[inline]
    fn read(&self, n: usize, field: u32) -> usize {
        usize::from(self.field[n][0] != field && self.field[n][1] != NONE)
    }

    #[inline]
    fn add(&mut self, n: usize, field: u32, mass: f32, momentum: Vec3, gradient: Vec3) {
        let s = self.slot(n, field);
        self.mass[n][s] += mass;
        self.momentum[n][s] += momentum;
        self.gradient[n][s] += gradient;
    }

    fn total_mass(&self, n: usize) -> f32 {
        self.mass[n][0] + self.mass[n][1]
    }
}

/// Which velocity field a point belongs to: one per continuum.
fn field_of(owner: Owner) -> u32 {
    match owner {
        Owner::Solid(s, _) => s,
        Owner::Fluid(f, _) => 0x8000_0000 | f,
    }
}

/// The quadratic B-spline's derivatives at offset `fx` from the base node, in
/// cells, for the three nodes the weights are for.
#[inline]
fn slopes(fx: f32) -> [f32; 3] {
    [fx - 1.5, -2.0 * (fx - 1.0), fx - 0.5]
}

/// Which continuum a material point belongs to, and its index in it.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord)]
enum Owner {
    Solid(u32, u32),
    Fluid(u32, u32),
}

/// What one unified step did.
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct UnifiedReport {
    /// Material points stepped, solids and fluids together.
    pub points: usize,
    /// Grid nodes that carried mass at the end of the step.
    pub active_nodes: usize,
    /// Grid blocks allocated.
    pub blocks: usize,
    /// Substeps taken.
    pub substeps: u32,
    /// Whether the substep budget was exceeded; the interval was rolled back.
    pub unstable: bool,
    /// Whether the interval was rejected; inspect `failure` for its cause.
    pub refused: bool,
    /// Explicit cause of rejection, if any. No physical state was committed.
    pub failure: Option<StepFailure>,
    /// Physical time committed, seconds; zero on rejection.
    pub advanced_dt: f32,
    /// Work attempted, including work rolled back after a later failure.
    pub attempted_substeps: u32,
    /// Smallest estimated stable timestep encountered, seconds.
    pub stable_dt: f32,
    /// Mechanism that imposed the smallest timestep.
    pub limiting: LimitKind,
    /// Numerical interventions, including those in rejected attempts.
    pub repairs: Repairs,
    /// Cores the step ran on.
    pub threads: usize,
    /// Cell size of the shared grid, metres.
    pub cell: f32,
    /// The pressure projections this step, for an incompressible fluid:
    /// iterations summed, divergences the worst of any substep, and
    /// `converged` only if every one was.
    pub projection: ProjectionReport,
    /// The implicit solids' solves this step, summed.
    pub implicit: ImplicitReport,
}

/// The unified continuum solver. Keep one and reuse it: its buffers are what
/// retain sparse-grid capacity. Transaction snapshots and transfer jobs still
/// allocate; measure those costs before claiming allocation-free execution.
#[derive(Clone, Debug)]
pub struct UnifiedSolver {
    /// Cores to run on. Zero is all of them.
    pub threads: usize,
    /// Coulomb friction against rigid bodies.
    pub friction: f32,
    /// Whether the material pushes back on the bodies it meets.
    pub two_way: bool,
    /// Most substeps a step may take.
    pub max_substeps: u32,
    /// The fluids' numerical speed of sound, m/s. See the module docs.
    pub sound_speed: f32,
    /// Whether two continua meeting at a node keep separate velocities there
    /// and meet by contact — sliding, separating — rather than sharing one.
    /// On by default; off is the preview's single shared field.
    pub separate_fields: bool,
    /// Coulomb friction where two continua slide on each other.
    pub contact_friction: f32,
    /// Maximum sparse blocks, checked during insertion before grid allocation.
    pub max_blocks: usize,
    /// Whether numerical repairs are recorded or cause transactional rejection.
    pub repair_policy: RepairPolicy,
    /// Below this many material points a step runs on one core. Zero by
    /// default: the number of cores already follows the number of points.
    pub serial_below: usize,
    /// How fluids get their pressure. See [`FluidModel`].
    pub fluid_model: FluidModel,
    /// The pressure solve's budget and tolerance, for
    /// [`FluidModel::Incompressible`].
    pub pressure: crate::linear::SolveSettings,
    projection: projection::Projection,
    projected: ProjectionReport,
    /// Whether solids stiffer than the step can resolve explicitly are
    /// solved implicitly instead of asking for substeps. See `implicit`.
    pub implicit_solids: bool,
    /// The implicit solids' solve budget and tolerance.
    pub stiffness: crate::linear::SolveSettings,
    implicit: implicit::Implicit,
    implicated: ImplicitReport,
    blocks: Vec<Block>,
    keys: Vec<[i64; 3]>,
    index: BlockMap<u32>,
    order: Vec<Owner>,
}

impl std::fmt::Debug for Block {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str("Block")
    }
}

impl Default for UnifiedSolver {
    fn default() -> Self {
        Self::new()
    }
}

/// A material point's mass, read once per transfer.
struct Kind {
    mass: f32,
}

/// What the transfer out computes for one point, written back afterwards.
#[derive(Clone, Copy)]
struct Update {
    velocity: Vec3,
    affine: Mat3,
    deformation: Mat3,
    plastic: f32,
    damage: f32,
    ratio: f32,
    uninverted: bool,
    volume_clamped: bool,
    plastic_clamped: bool,
    finite: bool,
}

impl UnifiedSolver {
    /// A solver with the defaults: every core, friction 0.5, two-way.
    pub fn new() -> Self {
        Self {
            threads: 0,
            friction: 0.5,
            two_way: true,
            max_substeps: 64,
            sound_speed: 20.0,
            separate_fields: true,
            contact_friction: 0.3,
            max_blocks: MAX_BLOCKS,
            repair_policy: RepairPolicy::Record,
            serial_below: 0,
            fluid_model: FluidModel::default(),
            pressure: crate::linear::SolveSettings {
                relative_tolerance: 1e-3,
                absolute_tolerance: 1e-9,
                max_iterations: 200,
            },
            projection: projection::Projection::default(),
            projected: ProjectionReport::default(),
            implicit_solids: false,
            stiffness: crate::linear::SolveSettings {
                relative_tolerance: 1e-2,
                absolute_tolerance: 1e-9,
                max_iterations: 100,
            },
            implicit: implicit::Implicit::default(),
            implicated: ImplicitReport::default(),
            blocks: Vec::new(),
            keys: Vec::new(),
            index: BlockMap::default(),
            order: Vec::new(),
        }
    }

    /// The cell size the shared grid uses for these continua: the finest any
    /// of them asks for.
    pub fn cell_for(solids: &[SolidSet], fluids: &[FluidSet]) -> f32 {
        let solid = solids
            .iter()
            .filter(|s| !s.is_empty())
            .map(|s| s.params.cell_size);
        let fluid = fluids
            .iter()
            .filter(|f| !f.is_empty())
            .map(|f| 2.0 * f.params.spacing);
        solid.chain(fluid).fold(f32::INFINITY, f32::min)
    }

    /// The sound speed the fluids' pressure is carried at: none where it is
    /// projected rather than carried.
    fn acoustic_speed(&self) -> f32 {
        match self.fluid_model {
            FluidModel::WeaklyCompressible => self.sound_speed,
            FluidModel::Incompressible => 0.0,
        }
    }

    fn cores(&self, points: usize) -> usize {
        if points < self.serial_below {
            return 1;
        }
        let available = std::thread::available_parallelism()
            .map(|n| n.get())
            .unwrap_or(1);
        if self.threads == 0 {
            // As many cores as the work keeps busy: threads are started per
            // pass, and twelve of them on four thousand points cost more than
            // they saved — four jelly blocks ran fastest on four.
            available.min(points.div_ceil(POINTS_PER_THREAD)).max(1)
        } else {
            self.threads.min(available.max(self.threads))
        }
    }

    /// Advance the whole requested interval or leave all physical state unchanged.
    ///
    /// Stability is re-estimated after every substep. Work-budget exhaustion,
    /// invalid arithmetic and strict repair failures reject the entire interval.
    /// `Record` preserves the preview's clamps but makes every one observable.
    pub fn step(
        &mut self,
        solids: &mut [SolidSet],
        fluids: &mut [FluidSet],
        bodies: &mut BodySet,
        gravity: Vec3,
        dt: f32,
    ) -> UnifiedReport {
        let points = solids.iter().map(|s| s.len()).sum::<usize>()
            + fluids.iter().map(|f| f.len()).sum::<usize>();
        let cell = Self::cell_for(solids, fluids);
        let mut report = UnifiedReport {
            points,
            cell,
            ..Default::default()
        };
        let failure = if !dt.is_finite()
            || dt < 0.0
            || !gravity.is_finite()
            || !self.sound_speed.is_finite()
            || self.sound_speed <= 0.0
            || !self.friction.is_finite()
            || self.friction < 0.0
            || self.max_substeps == 0
            || self.max_blocks == 0
        {
            Some(StepFailure::InvalidConfiguration)
        } else if points > 0 {
            control::validate(solids, fluids, cell).err()
        } else {
            None
        };
        if let Some(failure) = failure {
            report.failure = Some(failure);
            report.refused = true;
            return report;
        }
        if points == 0 || dt == 0.0 {
            report.advanced_dt = dt;
            return report;
        }
        if !bodies.is_finite() {
            report.failure = Some(StepFailure::InvalidState);
            report.refused = true;
            return report;
        }
        let initial = control::estimate(
            solids,
            fluids,
            cell,
            self.acoustic_speed(),
            gravity,
            self.implicit_solids,
        );
        report.stable_dt = initial.dt;
        report.limiting = initial.kind;
        if dt as f64 > initial.dt as f64 * self.max_substeps as f64 {
            report.failure = Some(StepFailure::SubstepBudget);
            report.unstable = true;
            report.refused = true;
            return report;
        }

        // Copy physical state, never grid scratch. Only body velocities/sleep
        // can change in this engine, so collision geometry is not duplicated.
        let saved_solids = solids.to_vec();
        let saved_fluids = fluids.to_vec();
        let linear = bodies.linear_velocity.clone();
        let angular = bodies.angular_velocity.clone();
        let sleeping = bodies.sleeping.clone();
        let timers = bodies.sleep_timer.clone();
        for fluid in fluids.iter_mut() {
            if fluid.affine.is_empty() && fluid.volume_ratio.is_empty() {
                fluid.affine.resize(fluid.len(), Mat3::ZERO);
                fluid.volume_ratio.resize(fluid.len(), 1.0);
            }
        }
        report.threads = self.cores(points);
        self.sort(solids, fluids, cell);
        let mut remaining = dt as f64;
        while remaining > 0.0 {
            let limit = control::estimate(
                solids,
                fluids,
                cell,
                self.acoustic_speed(),
                gravity,
                self.implicit_solids,
            );
            if limit.dt < report.stable_dt {
                report.stable_dt = limit.dt;
                report.limiting = limit.kind;
            }
            let available = self.max_substeps - report.attempted_substeps;
            if available == 0
                || limit.dt.is_nan()
                || limit.dt <= 0.0
                || remaining > limit.dt as f64 * available as f64
            {
                report.failure = Some(StepFailure::SubstepBudget);
                break;
            }
            // Equal subdivisions of the remaining interval avoid a tiny tail
            // while never increasing the current stable-step estimate.
            let count = (remaining / limit.dt as f64).ceil().max(1.0);
            let h = (remaining / count) as f32;
            if h <= 0.0 || remaining - h as f64 == remaining {
                report.failure = Some(StepFailure::SubstepBudget);
                break;
            }
            report.attempted_substeps += 1;
            self.projected = ProjectionReport::default();
            if let Err(failure) = self.substep(
                solids,
                fluids,
                bodies,
                gravity,
                cell,
                h,
                report.threads,
                &mut report.repairs,
                count == 1.0,
            ) {
                report.failure = Some(failure);
                break;
            }
            let i = self.implicated;
            report.implicit.unknowns = report.implicit.unknowns.max(i.unknowns);
            report.implicit.stiff_points = report.implicit.stiff_points.max(i.stiff_points);
            report.implicit.iterations += i.iterations;
            report.implicit.converged = if report.attempted_substeps == 1 {
                i.converged || i.stiff_points == 0
            } else {
                report.implicit.converged && (i.converged || i.stiff_points == 0)
            };
            let p = self.projected;
            if p.cells > 0 {
                report.projection.converged = if report.projection.cells == 0 {
                    p.converged
                } else {
                    report.projection.converged && p.converged
                };
                report.projection.cells = report.projection.cells.max(p.cells);
                report.projection.iterations += p.iterations;
                report.projection.divergence_before =
                    report.projection.divergence_before.max(p.divergence_before);
                report.projection.divergence_after =
                    report.projection.divergence_after.max(p.divergence_after);
            }
            // The last substep represents the remaining interval to f32
            // accuracy. Never run an extra substep solely for rounding residue.
            remaining = if count == 1.0 {
                0.0
            } else {
                (remaining - h as f64).max(0.0)
            };
        }
        report.blocks = self.blocks.len();
        if let Some(failure) = report.failure {
            solids.clone_from_slice(&saved_solids);
            fluids.clone_from_slice(&saved_fluids);
            bodies.linear_velocity = linear;
            bodies.angular_velocity = angular;
            bodies.sleeping = sleeping;
            bodies.sleep_timer = timers;
            report.refused = true;
            report.unstable = failure == StepFailure::SubstepBudget;
        } else {
            report.substeps = report.attempted_substeps;
            report.advanced_dt = dt;
            let floor = self.mass_floor(solids, fluids);
            report.active_nodes = self
                .blocks
                .iter()
                .map(|b| (0..BLOCK).filter(|&n| b.total_mass(n) > floor).count())
                .sum();
        }
        report
    }

    /// Estimate initial substep demand without changing state. A true second
    /// result means the interval cannot fit the budget (or inputs are invalid).
    /// The live engine also re-evaluates the estimate during integration.
    pub fn substeps_for(
        &self,
        solids: &[SolidSet],
        fluids: &[FluidSet],
        cell: f32,
        dt: f32,
    ) -> (u32, bool) {
        if !dt.is_finite()
            || dt < 0.0
            || !self.sound_speed.is_finite()
            || self.sound_speed <= 0.0
            || self.max_substeps == 0
            || control::validate(solids, fluids, cell).is_err()
        {
            return (0, true);
        }
        let limit = control::estimate(
            solids,
            fluids,
            cell,
            self.acoustic_speed(),
            Vec3::ZERO,
            self.implicit_solids,
        );
        let wanted = (dt as f64 / limit.dt as f64).ceil().max(1.0);
        (
            wanted.min(self.max_substeps as f64) as u32,
            wanted > self.max_substeps as f64,
        )
    }

    fn mass_floor(&self, solids: &[SolidSet], fluids: &[FluidSet]) -> f32 {
        let smallest = solids
            .iter()
            .map(|s| s.params.point_mass())
            .chain(fluids.iter().map(|f| f.params.particle_mass()))
            .fold(f32::INFINITY, f32::min);
        (smallest * 1e-9).max(f32::MIN_POSITIVE)
    }

    /// Order the points by the block they start the step in. Every transfer
    /// then walks the grid in memory order, and the fixed chunks of the
    /// transfer in each touch a compact patch of it.
    fn sort(&mut self, solids: &[SolidSet], fluids: &[FluidSet], cell: f32) {
        let mut keyed: Vec<(u64, Owner)> = Vec::new();
        let key = |p: DVec3| -> u64 {
            let b = |v: f64| ((v / cell as f64).floor() as i64).div_euclid(EDGE);
            // Morton order on the block coordinates, offset into the positive
            // range, so neighbouring blocks are near each other in the list.
            let spread = |v: i64| -> u64 {
                let mut x = (v + (1 << 20)) as u64 & 0x1f_ffff;
                x = (x | (x << 32)) & 0x1f00000000ffff;
                x = (x | (x << 16)) & 0x1f0000ff0000ff;
                x = (x | (x << 8)) & 0x100f00f00f00f00f;
                x = (x | (x << 4)) & 0x10c30c30c30c30c3;
                (x | (x << 2)) & 0x1249249249249249
            };
            spread(b(p.x)) | (spread(b(p.y)) << 1) | (spread(b(p.z)) << 2)
        };
        for (s, solid) in solids.iter().enumerate() {
            for i in 0..solid.len() {
                keyed.push((
                    key(solid.particles.position[i]),
                    Owner::Solid(s as u32, i as u32),
                ));
            }
        }
        for (f, fluid) in fluids.iter().enumerate() {
            for i in 0..fluid.len() {
                keyed.push((
                    key(fluid.particles.position[i]),
                    Owner::Fluid(f as u32, i as u32),
                ));
            }
        }
        // Stable on (key, owner), so equal keys keep a fixed order.
        keyed.sort_unstable();
        self.order.clear();
        self.order.extend(keyed.into_iter().map(|(_, o)| o));
    }

    #[allow(clippy::too_many_arguments)]
    fn substep(
        &mut self,
        solids: &mut [SolidSet],
        fluids: &mut [FluidSet],
        bodies: &mut BodySet,
        gravity: Vec3,
        cell: f32,
        h: f32,
        threads: usize,
        repairs: &mut Repairs,
        closing: bool,
    ) -> Result<(), StepFailure> {
        // ---- the blocks this substep needs, allocated in point order ----
        self.index.clear();
        self.keys.clear();
        // Points are in block order, so most of them reach the blocks the one
        // before reached: those are checked once, not once a point. Eight hash
        // lookups a point was a third of a step.
        let mut last: Option<([i64; 3], [bool; 3])> = None;
        for &owner in &self.order {
            let p = position(solids, fluids, owner);
            let (base, _) = stencil(p, cell);
            let first = block_of(base);
            let crosses = [0, 1, 2].map(|a| block_of([base[a] + 2; 3])[0] != first[a]);
            if last == Some((first, crosses)) {
                continue;
            }
            last = Some((first, crosses));
            for dz in [0, 2] {
                for dy in [0, 2] {
                    for dx in [0, 2] {
                        let k = block_of([base[0] + dx, base[1] + dy, base[2] + dz]);
                        if !self.index.contains_key(&k) {
                            if self.keys.len() >= self.max_blocks.min(MAX_BLOCKS) {
                                return Err(StepFailure::GridBudget);
                            }
                            self.index.insert(k, self.keys.len() as u32);
                            self.keys.push(k);
                        }
                    }
                }
            }
        }
        self.blocks.resize_with(self.keys.len(), Block::default);
        for b in &mut self.blocks {
            *b = Block::default();
        }

        // ---- particles to grid: fixed chunks, each into its own blocks ----
        let d_inv = 4.0 / (cell * cell);
        let order = &self.order;
        let index = &self.index;
        let sound = self.acoustic_speed();
        // Points whose stiffness the implicit solve takes put none in here.
        let implicit_cell = self.implicit_solids.then_some(cell);
        let ranges = chunk_ranges(order.len(), CHUNKS);
        let solids_ro: &[SolidSet] = solids;
        let fluids_ro: &[FluidSet] = fluids;
        let partials = parallel(threads, ranges.len(), |c| {
            let mut local: Vec<(u32, Block)> = Vec::new();
            let mut slot: HashMap<u32, usize, std::hash::BuildHasherDefault<BlockHash>> =
                HashMap::default();
            let mut reach: Option<Reach> = None;
            let mut mine = [[[0usize; 2]; 2]; 2];
            for &owner in &order[ranges[c].clone()] {
                let p = position(solids_ro, fluids_ro, owner);
                let (base, fx) = stencil(p, cell);
                // The same blocks as the point before, most of the time: their
                // lookups, and this chunk's slots for them, are kept.
                let same = reach.is_some_and(|r| r.first == block_of(base));
                let reach = match (same, reach.as_mut()) {
                    (true, Some(r)) => {
                        r.base = base;
                        *r
                    }
                    _ => {
                        let fresh = Reach::new(base, index);
                        reach = Some(fresh);
                        fresh
                    }
                };
                // The point's blocks, in this chunk's own storage.
                if !same {
                    for (z, plane) in mine.iter_mut().enumerate() {
                        for (y, row) in plane.iter_mut().enumerate() {
                            for (x, at) in row.iter_mut().enumerate() {
                                let global = reach.blocks[z][y][x];
                                if global == u32::MAX {
                                    continue;
                                }
                                *at = *slot.entry(global).or_insert_with(|| {
                                    local.push((global, Block::default()));
                                    local.len() - 1
                                });
                            }
                        }
                    }
                }
                let (velocity, q, kind) =
                    transfer_in(solids_ro, fluids_ro, owner, h, d_inv, sound, implicit_cell);
                let field = field_of(owner);
                let (wx, wy, wz) = (weights(fx.x), weights(fx.y), weights(fx.z));
                let (sx, sy, sz) = (slopes(fx.x), slopes(fx.y), slopes(fx.z));
                for (k, wk) in wz.iter().enumerate() {
                    for (j, wj) in wy.iter().enumerate() {
                        for (i, wi) in wx.iter().enumerate() {
                            let w = wi * wj * wk;
                            // The gradient of this weight with respect to the
                            // node: towards the point, so a field's mass
                            // gradient points into it.
                            let gradient =
                                Vec3::new(sx[i] * wj * wk, wi * sy[j] * wk, wi * wj * sz[k])
                                    * (-kind.mass / cell);
                            let node = [base[0] + i as i64, base[1] + j as i64, base[2] + k as i64];
                            let dpos = (Vec3::new(i as f32, j as f32, k as f32) - fx) * cell;
                            let b = block_of(node);
                            let at = mine[(b[2] - reach.first[2]) as usize]
                                [(b[1] - reach.first[1]) as usize]
                                [(b[0] - reach.first[0]) as usize];
                            let n = local_index(node);
                            local[at].1.add(
                                n,
                                field,
                                w * kind.mass,
                                (velocity * kind.mass + q * dpos) * w,
                                gradient,
                            );
                        }
                    }
                }
            }
            local
        });
        // Merged in chunk order, slot by slot, so the slots a node's fields
        // land in — like every sum here — do not depend on the core count.
        for partial in partials {
            for (global, block) in partial {
                let into = &mut self.blocks[global as usize];
                for n in 0..BLOCK {
                    for s in 0..2 {
                        let field = block.field[n][s];
                        if field != NONE {
                            into.add(
                                n,
                                field,
                                block.mass[n][s],
                                block.momentum[n][s],
                                block.gradient[n][s],
                            );
                        }
                    }
                }
            }
        }

        // ---- the grid: momentum to velocity, gravity, the bodies ----
        let floor = self.mass_floor(solids, fluids);
        let friction = self.friction;
        let contact_friction = self.contact_friction;
        let separate = self.separate_fields;
        let two_way = self.two_way;
        // Which bodies come near each block, found once rather than per node.
        let near: Vec<Vec<u32>> = self
            .keys
            .iter()
            .map(|k| {
                let lo = Vec3::new(
                    (k[0] * EDGE) as f32 * cell,
                    (k[1] * EDGE) as f32 * cell,
                    (k[2] * EDGE) as f32 * cell,
                );
                let hi = lo + Vec3::splat(EDGE as f32 * cell);
                let block_box = fusion_math::Aabb {
                    min: lo - Vec3::splat(cell),
                    max: hi + Vec3::splat(cell),
                };
                (0..bodies.len())
                    .filter(|&b| !bodies.is_removed(b) && bodies.aabb(b).overlaps(&block_box))
                    .map(|b| b as u32)
                    .collect()
            })
            .collect();
        let bodies_ro: &BodySet = bodies;
        let keys = &self.keys;
        let mut blocks = std::mem::take(&mut self.blocks);
        let block_ranges = chunk_ranges(blocks.len(), CHUNKS);
        let impulses = {
            let blocks_ref = &blocks;
            parallel(threads, block_ranges.len(), |c| {
                let mut updated: Vec<(usize, [[Vec3; 2]; BLOCK])> = Vec::new();
                let mut pushed: Vec<(u32, Vec3, Vec3)> = Vec::new();
                for b in block_ranges[c].clone() {
                    let data = &blocks_ref[b];
                    let mut velocities = [[Vec3::ZERO; 2]; BLOCK];
                    for (n, slot) in velocities.iter_mut().enumerate() {
                        let total = data.total_mass(n);
                        if total <= floor {
                            continue;
                        }
                        let [m0, m1] = data.mass[n];
                        let shared =
                            (data.momentum[n][0] + data.momentum[n][1]) / total + gravity * h;
                        let mut v = [shared; 2];
                        let interface = separate && m0 > floor && m1 > floor;
                        if interface {
                            v = interface_velocities(data, n, gravity * h, contact_friction);
                        }
                        let node = node_of(keys[b], n);
                        let x = Vec3::new(node[0] as f32, node[1] as f32, node[2] as f32) * cell;
                        let masses = if interface { [m0, m1] } else { [total, 0.0] };
                        for (s, vs) in v.iter_mut().enumerate().take(if interface { 2 } else { 1 })
                        {
                            for &body in &near[b] {
                                let before = *vs;
                                *vs = wall(bodies_ro, body as usize, x, *vs, cell, friction);
                                // Each boundary receives exactly its own reaction,
                                // including when several bodies meet the same node.
                                if two_way && *vs != before && bodies_ro.is_dynamic(body as usize) {
                                    pushed.push((body, (before - *vs) * masses[s], x));
                                }
                            }
                        }
                        if !interface {
                            v[1] = v[0];
                        }
                        *slot = v;
                    }
                    updated.push((b, velocities));
                }
                (updated, pushed)
            })
        };
        for (updated, pushed) in impulses {
            for (b, velocities) in updated {
                blocks[b].momentum = velocities;
            }
            for (body, impulse, at) in pushed {
                bodies.apply_impulse_at(body as usize, impulse, at);
            }
        }
        self.blocks = blocks;

        // ---- the pressure, for an incompressible fluid ----
        if self.fluid_model == FluidModel::Incompressible && fluids.iter().any(|f| !f.is_empty()) {
            // Where a body bounds the fluid. Beside one, the fluid's velocity
            // through the wall is the wall's — in, and out: the wall pass
            // above only stops approach, and a node inside a wall left moving
            // out of it was, to the solve, a source pumping fluid out of the
            // wall a little every substep; still water gained energy until it
            // sloshed. Then the pressure may move the node along the wall and
            // not through it. Up to three walls at a corner, each normal taken
            // out of what the pressure may do.
            //
            // Every block on every core: each node's distance to each body
            // near it, read once, says all three things asked of it.
            let threads = self.cores(self.order.len());
            let (keys, blocks_ro, near_ro): (&[[i64; 3]], &[Block], &[Vec<u32>]) =
                (&self.keys, &self.blocks, &near);
            let bodies_ro: &BodySet = bodies;
            let ranges = chunk_ranges(keys.len(), CHUNKS);
            type Bounded = ([Mat3; BLOCK], [f32; BLOCK], Option<[[Vec3; 2]; BLOCK]>);
            let runs = parallel(threads, ranges.len(), |run| {
                let mut out: Vec<Bounded> = Vec::with_capacity(ranges[run].len());
                let mut matched: Vec<(u32, Vec3, Vec3)> = Vec::new();
                for b in ranges[run].clone() {
                    let mut free = [Mat3::IDENTITY; BLOCK];
                    // And how much of each node's reach is open to fluid at
                    // all, which is how much mass fluid at rest would give
                    // it there.
                    let mut room = [1.0f32; BLOCK];
                    if near_ro[b].is_empty() {
                        out.push((free, room, None));
                        continue;
                    }
                    let block = &blocks_ro[b];
                    let mut momentum = block.momentum;
                    for (n, slot) in free.iter_mut().enumerate() {
                        let total = block.total_mass(n);
                        if total <= floor {
                            continue;
                        }
                        let node = node_of(keys[b], n);
                        let x = Vec3::new(node[0] as f32, node[1] as f32, node[2] as f32) * cell;
                        let mut seen: [(u32, f32, Vec3); 8] = [(0, f32::INFINITY, Vec3::ZERO); 8];
                        let mut count = 0;
                        for &body in &near_ro[b] {
                            if count == seen.len() {
                                break;
                            }
                            if let Some((distance, local)) =
                                distance_to(bodies_ro, body as usize, x, cell)
                            {
                                room[n] *= open_fraction(distance / cell);
                                seen[count] = (body, distance, local);
                                count += 1;
                            }
                        }
                        let seen = &seen[..count];
                        // A node inside a body is the body's: it moves as the
                        // body does, every way, and the pressure has no say.
                        // Left free along the wall, as a node at the surface
                        // is, the nodes inside a tank's walls — nearly
                        // massless, so nearly free to the solve — became a
                        // channel the pressure sent fluid down, and points in
                        // the corners rode it, stacked on one spot, round and
                        // round for as long as it ran.
                        if let Some(&(body, _, _)) = seen.iter().find(|s| s.1 < -0.1 * cell) {
                            let surface = bodies_ro.velocity_at_point(body as usize, x);
                            let [m0, m1] = block.mass[n];
                            for (s, m) in [(0usize, m0), (1, m1)] {
                                let fix = momentum[n][s] - surface;
                                momentum[n][s] = surface;
                                if fix != Vec3::ZERO
                                    && m > 0.0
                                    && two_way
                                    && bodies_ro.is_dynamic(body as usize)
                                {
                                    matched.push((body, fix * m, x));
                                }
                            }
                            *slot = Mat3::ZERO;
                            continue;
                        }
                        let mut normals: [Vec3; 3] = [Vec3::ZERO; 3];
                        let mut walls = 0;
                        for &(body, distance, local) in seen {
                            if walls == 3 {
                                break;
                            }
                            if distance > 0.5 * cell {
                                continue;
                            }
                            let pose = bodies_ro.pose(body as usize);
                            let normal = pose
                                .rotation
                                .rotate(crate::sdf_normal(&bodies_ro.shape[body as usize], local));
                            let mut d = normal;
                            for q in &normals[..walls] {
                                d -= *q * d.dot(*q);
                            }
                            if d.length_squared() <= 1e-4 {
                                continue;
                            }
                            let d = d.normalize();
                            normals[walls] = d;
                            walls += 1;
                            let surface = bodies_ro.velocity_at_point(body as usize, x);
                            let [m0, m1] = block.mass[n];
                            for (s, m) in [(0usize, m0), (1, m1)] {
                                let v = momentum[n][s];
                                let fix = d * d.dot(v - surface);
                                if fix != Vec3::ZERO && m > 0.0 {
                                    momentum[n][s] = v - fix;
                                    if two_way && bodies_ro.is_dynamic(body as usize) {
                                        matched.push((body, fix * m, x));
                                    }
                                }
                            }
                        }
                        let mut p = Mat3::IDENTITY;
                        for q in &normals[..walls] {
                            p = p - outer_product(*q, *q);
                        }
                        *slot = p;
                    }
                    out.push((free, room, Some(momentum)));
                }
                (out, matched)
            });
            let mut walled: Vec<[Mat3; BLOCK]> = Vec::with_capacity(self.keys.len());
            let mut open: Vec<[f32; BLOCK]> = Vec::with_capacity(self.keys.len());
            let mut b = 0;
            for (out, matched) in runs {
                for (free, room, momentum) in out {
                    if let Some(momentum) = momentum {
                        self.blocks[b].momentum = momentum;
                    }
                    walled.push(free);
                    open.push(room);
                    b += 1;
                }
                for (body, impulse, at) in matched {
                    bodies.apply_impulse_at(body as usize, impulse, at);
                }
            }
            self.projected = self.projection.project(
                &mut self.blocks,
                &self.index,
                &self.keys,
                fluids,
                &self.order,
                cell,
                h,
                floor,
                &walled,
                &open,
                self.pressure,
                threads,
            );
            // The bodies again: the projection knows nothing of them, and
            // what it pushed into one the wall takes back — and hands the
            // body, which is the fluid's pressure on it.
            walls_again(
                &mut self.blocks,
                &self.keys,
                &near,
                bodies,
                cell,
                floor,
                separate,
                friction,
                two_way,
            );
        }

        // ---- the solids' stiffness, implicitly, past the explicit limit ----
        //
        // After the pressure, so a block the water pushes on answers the push
        // in the same step: solved before it, a floating block took a frame's
        // buoyancy with nothing resisting it and stood a third taller.
        self.implicated = ImplicitReport::default();
        if self.implicit_solids && solids.iter().any(|s| !s.is_empty()) {
            let threads = self.cores(self.order.len());
            self.implicated = self.implicit.solve(
                &mut self.blocks,
                &self.keys,
                &self.index,
                solids,
                &self.order,
                cell,
                h,
                floor,
                separate,
                threads,
                self.stiffness,
            );
            if self.implicated.stiff_points > 0 {
                contact_again(&mut self.blocks, floor, separate, contact_friction);
                walls_again(
                    &mut self.blocks,
                    &self.keys,
                    &near,
                    bodies,
                    cell,
                    floor,
                    separate,
                    friction,
                    two_way,
                );
            }
        }

        // ---- grid to particles, then written back ----
        let blocks = &self.blocks;
        let order = &self.order;
        let index = &self.index;
        let point_ranges = chunk_ranges(order.len(), CHUNKS);
        let solids_ro: &[SolidSet] = solids;
        let fluids_ro: &[FluidSet] = fluids;
        let held = self.fluid_model == FluidModel::Incompressible;
        let updates = parallel(threads, point_ranges.len(), |c| {
            let mut cached: Option<Reach> = None;
            order[point_ranges[c].clone()]
                .iter()
                .map(|&owner| {
                    let p = position(solids_ro, fluids_ro, owner);
                    let field = field_of(owner);
                    let (base, fx) = stencil(p, cell);
                    let reach = match cached {
                        Some(mut r) if r.first == block_of(base) => {
                            r.base = base;
                            r
                        }
                        _ => Reach::new(base, index),
                    };
                    cached = Some(reach);
                    let (wx, wy, wz) = (weights(fx.x), weights(fx.y), weights(fx.z));
                    let mut velocity = Vec3::ZERO;
                    let mut affine = Mat3::ZERO;
                    for (k, wk) in wz.iter().enumerate() {
                        for (j, wj) in wy.iter().enumerate() {
                            for (i, wi) in wx.iter().enumerate() {
                                let w = wi * wj * wk;
                                let dpos = (Vec3::new(i as f32, j as f32, k as f32) - fx) * cell;
                                let (block, n) = reach.node(i, j, k);
                                let data = &blocks[block as usize];
                                let v = data.momentum[n][data.read(n, field)];
                                velocity += v * w;
                                affine += outer_product(v * w, dpos) * d_inv;
                            }
                        }
                    }
                    transfer_out(solids_ro, fluids_ro, owner, velocity, affine, h, held)
                })
                .collect::<Vec<Update>>()
        });
        // Validate the complete staged point update before committing it.
        for (k, u) in updates.iter().flatten().enumerate() {
            if !u.finite
                || !u.velocity.is_finite()
                || !control::matrix_finite(u.affine)
                || !control::matrix_finite(u.deformation)
                || !u.plastic.is_finite()
                || !u.damage.is_finite()
                || !u.ratio.is_finite()
                || !(position(solids, fluids, self.order[k]) + to_f64(u.velocity * h)).is_finite()
            {
                return Err(StepFailure::NonFiniteUpdate);
            }
            repairs.deformation += usize::from(u.uninverted);
            repairs.volume += usize::from(u.volume_clamped);
            repairs.plastic += usize::from(u.plastic_clamped);
        }
        if !bodies.is_finite() {
            return Err(StepFailure::NonFiniteUpdate);
        }
        if self.repair_policy == RepairPolicy::Reject && repairs.total() > 0 {
            return Err(StepFailure::RepairRequired);
        }
        let mut travel: Vec<Vec3> = Vec::with_capacity(self.order.len());
        for (k, u) in updates.into_iter().flatten().enumerate() {
            travel.push(u.velocity * h);
            write_back(solids, fluids, self.order[k], u, h);
        }
        // Projected fluid points that packed too tight are moved apart, at
        // the end of each step: see `separation`. Before the bodies are
        // checked, so that none is moved into one.
        if self.fluid_model == FluidModel::Incompressible && closing {
            let mut moves: Vec<Vec<Vec3>> = Vec::with_capacity(fluids.len());
            for fluid in fluids.iter() {
                let mut m = Vec::new();
                separation::separation(fluid, &mut m);
                moves.push(m);
            }
            for (k, &owner) in self.order.iter().enumerate() {
                if let Owner::Fluid(f, i) = owner {
                    let d = moves[f as usize][i as usize];
                    if d != Vec3::ZERO {
                        fluids[f as usize].particles.position[i as usize] += to_f64(d);
                        travel[k] += d;
                    }
                }
            }
        }
        // The point-level half of the boundary: see `crate::keep_out`. In
        // the points' own order, so the result does not depend on how the
        // grid was split across cores.
        let mut reach = fusion_math::Aabb {
            min: Vec3::splat(f32::INFINITY),
            max: Vec3::splat(f32::NEG_INFINITY),
        };
        for &owner in &self.order {
            let p = fusion_math::to_f32(position(solids, fluids, owner));
            reach.min = reach.min.min(p);
            reach.max = reach.max.max(p);
        }
        let reach = reach.expanded(cell);
        let near: Vec<usize> = (0..bodies.len())
            .filter(|&b| !bodies.is_removed(b) && bodies.aabb(b).overlaps(&reach))
            .collect();
        if !near.is_empty() {
            for (k, &owner) in self.order.iter().enumerate() {
                let (position, velocity, mass) = match owner {
                    Owner::Solid(s, i) => {
                        let solid = &mut solids[s as usize];
                        let mass = solid.params.point_mass();
                        let particles = &mut solid.particles;
                        let i = i as usize;
                        (&mut particles.position[i], &mut particles.velocity[i], mass)
                    }
                    Owner::Fluid(f, i) => {
                        let fluid = &mut fluids[f as usize];
                        let mass = fluid.params.particle_mass();
                        let particles = &mut fluid.particles;
                        let i = i as usize;
                        (&mut particles.position[i], &mut particles.velocity[i], mass)
                    }
                };
                crate::keep_out(
                    position, velocity, mass, bodies, &near, friction, two_way, travel[k],
                );
            }
        }
        Ok(())
    }
}

/// The two velocities at a node where two continua meet.
///
/// Each field's own velocity, and the centre-of-mass velocity of the pair.
/// The interface normal is the difference of the two mass gradients, from the
/// first field towards the second. Moving apart, each keeps its own velocity
/// and they separate; closing, the normal part of their relative velocity is
/// removed and Coulomb friction takes what it can of the tangential part —
/// scaled, never reversed. Both are changed by equal and opposite amounts of
/// momentum, so the pair's momentum is exactly what it was.
fn interface_velocities(data: &Block, n: usize, gravity: Vec3, friction: f32) -> [Vec3; 2] {
    let [m0, m1] = data.mass[n];
    let v0 = data.momentum[n][0] / m0 + gravity;
    let v1 = data.momentum[n][1] / m1 + gravity;
    let total = m0 + m1;
    let centre = (data.momentum[n][0] + data.momentum[n][1]) / total + gravity;
    let normal = (data.gradient[n][1] - data.gradient[n][0]).normalize_or_zero();
    if normal == Vec3::ZERO || (v0 - v1).dot(normal) <= 0.0 {
        return [v0, v1];
    }
    let relative = v0 - centre;
    let closing = relative.dot(normal);
    let tangent = relative - normal * closing;
    let speed = tangent.length();
    let keep = if speed > 1e-9 {
        (1.0 - friction * closing / speed).max(0.0)
    } else {
        0.0
    };
    let slide = tangent * keep;
    [centre + slide, centre - slide * (m0 / m1)]
}

/// Two continua at a node kept from closing on each other again, as
/// [`interface_velocities`] does, on velocities something has since changed:
/// the implicit solids move their own slot, and a light block whose stiffness
/// was solved after the contact sank through the water it should have
/// floated on.
fn contact_again(blocks: &mut [Block], floor: f32, separate: bool, friction: f32) {
    if !separate {
        return;
    }
    for block in blocks.iter_mut() {
        for n in 0..BLOCK {
            let [m0, m1] = block.mass[n];
            if m0 <= floor || m1 <= floor {
                continue;
            }
            let [v0, v1] = block.momentum[n];
            let normal = (block.gradient[n][1] - block.gradient[n][0]).normalize_or_zero();
            if normal == Vec3::ZERO || (v0 - v1).dot(normal) <= 0.0 {
                continue;
            }
            let centre = (v0 * m0 + v1 * m1) / (m0 + m1);
            let relative = v0 - centre;
            let closing = relative.dot(normal);
            let tangent = relative - normal * closing;
            let speed = tangent.length();
            let keep = if speed > 1e-9 {
                (1.0 - friction * closing / speed).max(0.0)
            } else {
                0.0
            };
            let slide = tangent * keep;
            block.momentum[n] = [centre + slide, centre - slide * (m0 / m1)];
        }
    }
}

/// Positions of a point, whichever continuum it is in.
fn position(solids: &[SolidSet], fluids: &[FluidSet], owner: Owner) -> DVec3 {
    match owner {
        Owner::Solid(s, i) => solids[s as usize].particles.position[i as usize],
        Owner::Fluid(f, i) => fluids[f as usize].particles.position[i as usize],
    }
}

/// The base node of a point's 3×3×3 stencil, and its offset from it, cells.
fn stencil(p: DVec3, cell: f32) -> ([i64; 3], Vec3) {
    let c = cell as f64;
    let local = [p.x / c - 0.5, p.y / c - 0.5, p.z / c - 0.5];
    let base = local.map(|v| v.floor() as i64);
    let fx = Vec3::new(
        (p.x / c - base[0] as f64) as f32,
        (p.y / c - base[1] as f64) as f32,
        (p.z / c - base[2] as f64) as f32,
    );
    (base, fx)
}

fn block_of(node: [i64; 3]) -> [i64; 3] {
    node.map(|v| v.div_euclid(EDGE))
}

fn local_index(node: [i64; 3]) -> usize {
    let [x, y, z] = node.map(|v| v.rem_euclid(EDGE) as usize);
    x + 4 * (y + 4 * z)
}

fn node_of(block: [i64; 3], n: usize) -> [i64; 3] {
    let (x, y, z) = (n % 4, (n / 4) % 4, n / 16);
    [
        block[0] * EDGE + x as i64,
        block[1] * EDGE + y as i64,
        block[2] * EDGE + z as i64,
    ]
}

/// A point's velocity, the affine-plus-stress matrix it puts onto the grid,
/// and its mass and volume.
fn transfer_in(
    solids: &[SolidSet],
    fluids: &[FluidSet],
    owner: Owner,
    h: f32,
    d_inv: f32,
    sound: f32,
    implicit_cell: Option<f32>,
) -> (Vec3, Mat3, Kind) {
    match owner {
        Owner::Solid(s, i) => {
            let solid = &solids[s as usize];
            let i = i as usize;
            let material = solid.params.material;
            let (mu, lambda) = material.lame();
            let (mu, lambda) = hardened(material.model, mu, lambda, solid.plastic[i]);
            let f = solid.deformation[i];
            let piola = if material.fracture.breaks() {
                stress_with_damage(material.model, f, mu, lambda, solid.damage[i])
            } else {
                stress(material.model, f, mu, lambda)
            };
            // Kirchhoff stress, τ = P Fᵀ, and the fibres' share of it.
            let mut kirchhoff = piola * f.transpose();
            if let Some(fibre) = solid.params.fibre {
                let now = f * fibre.direction;
                let stretch = now.length_squared();
                if stretch > 1.0 {
                    let k = fibre.stiffness * material.youngs_modulus;
                    kirchhoff += outer_product(now, now) * (k * (stretch - 1.0));
                }
            }
            let mass = solid.params.point_mass();
            let volume = solid.volume[i];
            if implicit_cell.is_some_and(|cell| implicit::stiff(solid, i, cell, h)) {
                return (
                    solid.particles.velocity[i],
                    solid.affine[i] * mass,
                    Kind { mass },
                );
            }
            let q = kirchhoff * (-h * volume * d_inv) + solid.affine[i] * mass;
            (solid.particles.velocity[i], q, Kind { mass })
        }
        Owner::Fluid(f, i) => {
            let fluid = &fluids[f as usize];
            let i = i as usize;
            let params = fluid.params;
            let mass = params.particle_mass();
            let rest = params.spacing.powi(3);
            let j = fluid.volume_ratio[i].clamp(0.2, 5.0);
            let c = fluid.affine[i];
            // Tait, clamped: a free surface does not pull.
            let bulk = params.rest_density * sound * sound / GAMMA;
            let pressure = (bulk * (j.powf(-GAMMA) - 1.0)).max(0.0);
            let mu = viscosity_of(&params);
            let strain_rate = (c + c.transpose()) * 0.5;
            // Kirchhoff: J σ, with σ = -p I + 2 μ ε̇.
            let kirchhoff = (Mat3::IDENTITY * -pressure + strain_rate * (2.0 * mu)) * j;
            let q = kirchhoff * (-h * rest * d_inv) + c * mass;
            (fluid.particles.velocity[i], q, Kind { mass })
        }
    }
}

/// A fluid's dynamic viscosity, Pa·s: its own where it has one, else its
/// legacy dial through [`dynamic_viscosity`].
pub fn viscosity_of(params: &fusion_core::FluidParams) -> f32 {
    params
        .dynamic_viscosity
        .filter(|v| v.is_finite() && *v >= 0.0)
        .unwrap_or_else(|| dynamic_viscosity(params.viscosity))
}

/// The fluid's XSPH-style viscosity dial as a dynamic viscosity, Pa·s: water's
/// 0.1 is a millipascal-second and change, honey's 0.5 about half a
/// pascal-second. A curve, not a measurement — it maps the dial the classic
/// fluid already has onto a real viscosity so one scene means one thing on
/// both paths.
pub fn dynamic_viscosity(dial: f32) -> f32 {
    let d = dial.clamp(0.0, 2.0);
    1e-3 + 4.0 * d * d * d
}

/// What the grid hands back to a point.
fn transfer_out(
    solids: &[SolidSet],
    fluids: &[FluidSet],
    owner: Owner,
    velocity: Vec3,
    affine: Mat3,
    h: f32,
    held: bool,
) -> Update {
    match owner {
        Owner::Solid(s, i) => {
            let solid = &solids[s as usize];
            let i = i as usize;
            let material = solid.params.material;
            let mut f = (Mat3::IDENTITY + affine * h) * solid.deformation[i];
            let finite = control::matrix_finite(f);
            // Never send overflow into the constitutive SVD. The staged
            // update is marked invalid and rejected before commit.
            if !finite {
                f = Mat3::IDENTITY;
            }
            let mut uninverted = false;
            if let Some(pulled) = model::uninvert(f) {
                f = pulled;
                uninverted = true;
            }
            let (elastic, plastic) = return_map(material.model, f);
            let mut damage = solid.damage[i];
            if material.fracture.breaks() {
                let (_, sigma, _) = elastic.svd();
                let stretch = sigma.x.abs().max(sigma.y.abs()).max(sigma.z.abs());
                damage = damage.max(material.fracture.damage_at(stretch));
            }
            Update {
                velocity,
                affine,
                deformation: elastic,
                plastic: (solid.plastic[i] * plastic).clamp(0.05, 20.0),
                damage,
                ratio: 1.0,
                uninverted,
                volume_clamped: false,
                finite: finite && (solid.plastic[i] * plastic).is_finite(),
                plastic_clamped: !(0.05..=20.0).contains(&(solid.plastic[i] * plastic)),
            }
        }
        Owner::Fluid(f, i) => {
            let fluid = &fluids[f as usize];
            let i = i as usize;
            let trace = if held {
                0.0
            } else {
                affine.x_axis.x + affine.y_axis.y + affine.z_axis.z
            };
            let before = if held { 1.0 } else { fluid.volume_ratio[i] };
            Update {
                velocity,
                affine,
                deformation: Mat3::IDENTITY,
                plastic: 1.0,
                damage: 0.0,
                ratio: (before * (1.0 + h * trace)).clamp(0.2, 5.0),
                uninverted: false,
                volume_clamped: !(0.2..=5.0).contains(&(before * (1.0 + h * trace)))
                    || !(0.2..=5.0).contains(&before),
                plastic_clamped: false,
                finite: (before * (1.0 + h * trace)).is_finite(),
            }
        }
    }
}

fn write_back(solids: &mut [SolidSet], fluids: &mut [FluidSet], owner: Owner, u: Update, h: f32) {
    match owner {
        Owner::Solid(s, i) => {
            let solid = &mut solids[s as usize];
            let i = i as usize;
            solid.deformation[i] = u.deformation;
            solid.plastic[i] = u.plastic;
            solid.damage[i] = u.damage;
            solid.affine[i] = u.affine;
            solid.particles.velocity[i] = u.velocity;
            solid.particles.position[i] += to_f64(u.velocity * h);
            let _ = u.uninverted;
        }
        Owner::Fluid(f, i) => {
            let fluid = &mut fluids[f as usize];
            let i = i as usize;
            fluid.affine[i] = u.affine;
            fluid.volume_ratio[i] = u.ratio;
            fluid.particles.velocity[i] = u.velocity;
            fluid.particles.position[i] += to_f64(u.velocity * h);
        }
    }
}

/// Every node near a body through its walls once more, after something —
/// the pressure, the implicit solids — changed velocities the grid update had
/// already kept out of them. Each body is handed what its wall took.
#[allow(clippy::too_many_arguments)]
fn walls_again(
    blocks: &mut [Block],
    keys: &[[i64; 3]],
    near: &[Vec<u32>],
    bodies: &mut BodySet,
    cell: f32,
    floor: f32,
    separate: bool,
    friction: f32,
    two_way: bool,
) {
    let mut pushed: Vec<(u32, Vec3, Vec3)> = Vec::new();
    for (b, block) in blocks.iter_mut().enumerate() {
        if near[b].is_empty() {
            continue;
        }
        for n in 0..BLOCK {
            let total = block.total_mass(n);
            if total <= floor {
                continue;
            }
            let [m0, m1] = block.mass[n];
            let interface = separate && m0 > floor && m1 > floor;
            let masses = if interface { [m0, m1] } else { [total, 0.0] };
            let node = node_of(keys[b], n);
            let x = Vec3::new(node[0] as f32, node[1] as f32, node[2] as f32) * cell;
            for (s, &mass) in masses
                .iter()
                .enumerate()
                .take(if interface { 2 } else { 1 })
            {
                for &body in &near[b] {
                    let before = block.momentum[n][s];
                    let after = wall(bodies, body as usize, x, before, cell, friction);
                    block.momentum[n][s] = after;
                    if two_way && after != before && bodies.is_dynamic(body as usize) {
                        pushed.push((body, (before - after) * mass, x));
                    }
                }
            }
            if !interface {
                block.momentum[n][1] = block.momentum[n][0];
            }
        }
    }
    for (body, impulse, at) in pushed {
        bodies.apply_impulse_at(body as usize, impulse, at);
    }
}

/// A node's signed distance to a body, and where it is in the body's frame,
/// for a node near enough — a cell and a half — that the body bounds it.
fn distance_to(bodies: &BodySet, body: usize, x: Vec3, cell: f32) -> Option<(f32, Vec3)> {
    if !bodies.aabb(body).expanded(1.5 * cell).contains_point(x) {
        return None;
    }
    let local = bodies.pose(body).inverse_transform_point(x);
    Some((bodies.shape[body].signed_distance_local(local), local))
}

/// How much of a node's spline reach lies outside a body `s` cells from it:
/// the spline's own cumulative weight across the body's surface, taken as a
/// plane — all of it a cell and a half clear, half on the surface, none
/// deep in.
fn open_fraction(s: f32) -> f32 {
    if s >= 1.5 {
        1.0
    } else if s <= -1.5 {
        0.0
    } else if s <= -0.5 {
        (s + 1.5).powi(3) / 6.0
    } else if s <= 0.5 {
        1.0 / 6.0 + 0.75 * (s + 0.5) - (s * s * s + 0.125) / 3.0
    } else {
        1.0 - (1.5 - s).powi(3) / 6.0
    }
}

/// A body as a moving wall: a node within half a cell of it loses its
/// approach and keeps what friction allows of its slide. The classic solver's
/// rule, the same SDF, the same Coulomb clamp.
fn wall(bodies: &BodySet, body: usize, x: Vec3, v: Vec3, cell: f32, friction: f32) -> Vec3 {
    if !bodies.aabb(body).expanded(cell).contains_point(x) {
        return v;
    }
    let pose = bodies.pose(body);
    let shape = &bodies.shape[body];
    let local = pose.inverse_transform_point(x);
    if shape.signed_distance_local(local) > 0.5 * cell {
        return v;
    }
    let normal = pose.rotation.rotate(crate::sdf_normal(shape, local));
    let surface = bodies.velocity_at_point(body, x);
    let relative = v - surface;
    let approach = relative.dot(normal);
    if approach >= 0.0 {
        return v;
    }
    let tangent = relative - normal * approach;
    let speed = tangent.length();
    let scale = if speed > 1e-6 {
        (1.0 + friction * approach / speed).max(0.0)
    } else {
        0.0
    };
    surface + tangent * scale
}

/// `n` items in `chunks` contiguous runs, the remainder spread over the first.
fn chunk_ranges(n: usize, chunks: usize) -> Vec<std::ops::Range<usize>> {
    let chunks = chunks.max(1).min(n.max(1));
    let size = n / chunks;
    let extra = n % chunks;
    let mut out = Vec::with_capacity(chunks);
    let mut start = 0;
    for c in 0..chunks {
        let len = size + usize::from(c < extra);
        out.push(start..start + len);
        start += len;
    }
    out
}

/// Run `job(0..jobs)` on `threads` cores, results in job order whatever order
/// they finished in.
fn parallel<T: Send>(threads: usize, jobs: usize, job: impl Fn(usize) -> T + Sync) -> Vec<T> {
    if threads <= 1 || jobs <= 1 {
        return (0..jobs).map(&job).collect();
    }
    let next = AtomicUsize::new(0);
    let mut done: Vec<(usize, T)> = std::thread::scope(|scope| {
        let workers: Vec<_> = (0..threads.min(jobs))
            .map(|_| {
                scope.spawn(|| {
                    let mut mine = Vec::new();
                    loop {
                        let k = next.fetch_add(1, Ordering::Relaxed);
                        if k >= jobs {
                            break;
                        }
                        mine.push((k, job(k)));
                    }
                    mine
                })
            })
            .collect();
        workers
            .into_iter()
            .flat_map(|w| w.join().expect("a solver thread panicked"))
            .collect()
    });
    done.sort_by_key(|(k, _)| *k);
    done.into_iter().map(|(_, t)| t).collect()
}
