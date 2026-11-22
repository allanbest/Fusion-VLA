//! Incompressible pressure projection on the shared grid.
//!
//! The weakly compressible fluid gets its pressure from how far each point
//! has been squeezed, through a stiff equation of state. That works, and the
//! stiffness is the whole price of it: a pressure that answers compression
//! has to be carried by waves, waves at the speed of sound have to be
//! resolved by the timestep, and even a 20 m/s numerical sound speed — not
//! water's 1 480 — is what sets most fluid steps. And the fluid still
//! compresses by a percent and rings while it does.
//!
//! This is the other standard answer, the one the Navier–Stokes equations
//! for an incompressible fluid ask for directly: after the explicit update
//! (gravity, viscosity, contact), find the pressure whose gradient makes the
//! velocity field divergence-free, and subtract it — Chorin's projection,
//! solved implicitly, on the grid the material is already on. The material
//! points only carry the fluid and its free surface around, which is what
//! they are good at and a grid is not.
//!
//! ## The discretisation
//!
//! Pressure lives at the centres of the cells that hold fluid, and a cell's
//! divergence is read from its eight corner nodes, `(D u)_c = Σ_k g_k · u_k`
//! with `g_k = (±1, ±1, ±1) / 4h`. That is not a grid-only shortcut: at a
//! cell's centre it is exactly the divergence of the quadratic B-spline
//! field the points move with. The correction is `u += W P Dᵀ p`, `W` the
//! substep over each node's lumped mass and `P` what the walls leave free,
//! so `D W P Dᵀ p = target − D u*` is symmetric and positive semidefinite and
//! is applied without a matrix.
//!
//! - **Solve:** conjugate gradients on the assembled matrix, cells in grid
//!   order, preconditioned by its incomplete Cholesky factor — a third to a
//!   sixth of Jacobi's iterations, which in a scene whose solids set eight
//!   substeps a frame made the solve, not the particles, the cost — and
//!   warm-started from the last substep's pressure, inside a real-time
//!   iteration budget.
//! - **Free surface:** cells without points have no pressure, so zero. The
//!   nodes just outside the fluid, which the surface points still read, get
//!   the projected velocity of their neighbours.
//! - **Other continua:** a node's mass is everything on it, and the
//!   correction is applied to every field there alike — so fluid pressure
//!   pushes a soft solid exactly as it pushes fluid. Buoyancy is that.
//! - **Rigid bodies:** a node inside a body moves with it and has no say; a
//!   node at its surface matches the body's normal velocity, in and out, and
//!   the pressure may move it along the wall only. Walls are applied again
//!   afterwards, and that reaction is the pressure on the body.
//! - **Volume:** a divergence-free grid still lets points crowd where its
//!   cells cannot see — a dam break lost half its depth that way, into
//!   a tank's corners. The grid's own masses, against what fluid at rest
//!   would give each node beside its walls, say where; those cells are asked
//!   to open up a little each substep.
//!
//! ## What did not work, and why
//!
//! - Pressure at the nodes, read by each point through its own spline
//!   weights: equal-order velocity and pressure has spurious modes, and the
//!   water settled in layers of alternating density.
//! - Pressure per cell from the points' own gradients: held volume, but to a
//!   point a cell's pressure is a step, and beside a wall it never added up
//!   to the floor's support — still water sank into the floor at 4 cm/s.
//! - Each node's mass floored at the rest density: halved the support walls
//!   gave. Walls that only stopped approach, not matched it: a node inside a
//!   wall left moving out was a source, and still water gained energy.
//! - Volume recovery from each point's volume ratio: the ratio, integrated
//!   from the velocity gradient, drifted up while the points crowded, and
//!   the correction compressed the water further. Under this model the
//!   ratio is held at one.

use std::cell::RefCell;

use fusion_core::FluidSet;
use fusion_math::{Mat3, Vec3};

use super::Block;
use super::{block_of, local_index, BlockMap, Owner, BLOCK};
use crate::linear::{IncompleteCholesky, Pcg, SolveSettings, SparseSymmetric};

/// Fraction of a cell's crowding undone per substep.
const SPREAD: f64 = 0.2;

/// Most substeps one incomplete factor serves the same cells.
const REFACTOR: u32 = 4;

/// Shift added to the diagonal, as a fraction of it, so that a pressure
/// pattern with no gradient anywhere cannot stall the solve.
const SHIFT: f64 = 1e-6;

/// What one projection did.
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct ProjectionReport {
    /// Cells that held fluid and had a pressure.
    pub cells: usize,
    /// Conjugate-gradient iterations.
    pub iterations: usize,
    /// Whether the solve met its tolerance inside its budget.
    pub converged: bool,
    /// Largest cell divergence before the projection, 1/s.
    pub divergence_before: f32,
    /// Largest departure from the target after it, 1/s.
    pub divergence_after: f32,
}

/// Reusable workspace. Keep one: its capacity and its pressures are what
/// make the next substep's solve cheap.
#[derive(Clone, Debug, Default)]
pub(super) struct Projection {
    /// Each fluid point's cell, keyed z-first so that sorting puts cells
    /// in grid order, and its fluid's rest density. Scratch.
    keyed: Vec<([i64; 3], f32)>,
    cells: Vec<[i64; 3]>,
    /// Each cell's eight corners, as (block, node in block), x fastest.
    corners: Vec<[(u32, u8); 8]>,
    count: Vec<u32>,
    /// Sum of each cell's points' rest densities, kg/m³.
    density: Vec<f64>,
    rhs: Vec<f64>,
    diagonal: Vec<f64>,
    pressure: Vec<f64>,
    /// Last substep's cells and pressures, to start the next solve from.
    previous_cells: Vec<[i64; 3]>,
    previous: Vec<f64>,
    /// The cells the factor was computed for, and how many substeps ago; it
    /// is kept while the cells are the same ones, for a few substeps — an
    /// approximate inverse is all a preconditioner has to be.
    factored: Vec<[i64; 3]>,
    factor_age: u32,

    /// Per node: `h V / m`, the velocity a unit pressure gradient gives it.
    weight: Vec<[f32; BLOCK]>,
    /// Per node: the directions the pressure may move it in — identity in
    /// open fluid, the tangent plane of a wall beside one.
    free: Vec<[Mat3; BLOCK]>,
    /// Per node: the mass-weighted velocity of everything on it.
    velocity: Vec<[Vec3; BLOCK]>,
    gradient: RefCell<Vec<[Vec3; BLOCK]>>,
    /// Per node: whether it is a corner of a fluid cell. Scratch.
    corner_mask: Vec<[bool; BLOCK]>,
    /// The matrix, assembled, and its incomplete factor.
    matrix: SparseSymmetric,
    /// Cell numbers over the fluid's bounds, for finding neighbours.
    table: Vec<u32>,
    factor: IncompleteCholesky,
    pcg: Pcg,
}

/// Each corner's signs, x fastest.
const SIGNS: [Vec3; 8] = [
    Vec3::new(-1.0, -1.0, -1.0),
    Vec3::new(1.0, -1.0, -1.0),
    Vec3::new(-1.0, 1.0, -1.0),
    Vec3::new(1.0, 1.0, -1.0),
    Vec3::new(-1.0, -1.0, 1.0),
    Vec3::new(1.0, -1.0, 1.0),
    Vec3::new(-1.0, 1.0, 1.0),
    Vec3::new(1.0, 1.0, 1.0),
];

/// The gradient weight of corner `k` (x fastest) at a cell's centre.
#[inline]
fn corner_weight(k: usize, cell: f32) -> Vec3 {
    let s = |bit: usize| if k & bit == 0 { -1.0 } else { 1.0 };
    Vec3::new(s(1), s(2), s(4)) / (4.0 * cell)
}

impl Projection {
    /// `D W P Dᵀ`, assembled: two cells couple through each corner they
    /// share, which a cell does with each of its 26 neighbours.
    fn assemble(&mut self, cell: f32, threads: usize) {
        let n = self.cells.len();
        // The neighbours, from a dense table over the fluid's bounds when
        // that is not mostly empty — twenty-six hash lookups a cell were
        // most of the cost of assembling — and from the cell map otherwise.
        let (mut lo, mut hi) = ([i64::MAX; 3], [i64::MIN; 3]);
        for key in &self.cells {
            for a in 0..3 {
                lo[a] = lo[a].min(key[a]);
                hi[a] = hi[a].max(key[a]);
            }
        }
        let dims = [hi[0] - lo[0] + 3, hi[1] - lo[1] + 3, hi[2] - lo[2] + 3];
        let span = dims[0] * dims[1] * dims[2];
        let dense = span <= 64 * n as i64 + 4096;
        let slot = |k: [i64; 3]| {
            (((k[2] - lo[2] + 1) * dims[1] + (k[1] - lo[1] + 1)) * dims[0] + (k[0] - lo[0] + 1))
                as usize
        };
        self.table.clear();
        if dense {
            self.table.resize(span as usize, u32::MAX);
            for (at, key) in self.cells.iter().enumerate() {
                self.table[slot(*key)] = at as u32;
            }
        }
        let (table, cells) = (&self.table, &self.cells);
        let find = |k: [i64; 3]| -> Option<u32> {
            if dense {
                Some(table[slot(k)]).filter(|&c| c != u32::MAX)
            } else {
                cells
                    .binary_search_by_key(&(k[2], k[1], k[0]), |c| (c[2], c[1], c[0]))
                    .ok()
                    .map(|at| at as u32)
            }
        };
        // `g_k · W P g_j` is `W / 16h² · s_k · P s_j`, `s` the corners' signs.
        let scale = 1.0 / (16.0 * cell * cell);
        let (cells, corners, weight, free, diagonal) = (
            &self.cells,
            &self.corners,
            &self.weight,
            &self.free,
            &self.diagonal,
        );
        // Rows are independent: in fixed runs, on every core, joined in order.
        let ranges = super::chunk_ranges(n, super::CHUNKS);
        let runs = super::parallel(threads, ranges.len(), |run| {
            let mut columns: Vec<u32> = Vec::with_capacity(ranges[run].len() * 27);
            let mut values: Vec<f64> = Vec::with_capacity(ranges[run].len() * 27);
            let mut ends: Vec<usize> = Vec::with_capacity(ranges[run].len());
            for at in ranges[run].clone() {
                let key = cells[at];
                // Each pair of this cell's corner `k` and a neighbour's
                // corner `j` naming the same node couples the two cells; the
                // neighbour is the one `bits(k) − bits(j)` away. Accumulated
                // by that offset, 0..27 in grid order, which is also column
                // order.
                let mut row = [0.0f32; 27];
                for (k, &(b, node)) in corners[at].iter().enumerate() {
                    if b == u32::MAX {
                        continue;
                    }
                    let (b, node) = (b as usize, node as usize);
                    let w = weight[b][node];
                    if w == 0.0 {
                        continue;
                    }
                    let p = free[b][node];
                    let mine = SIGNS[k];
                    for (j, theirs) in SIGNS.iter().enumerate() {
                        let offset = ((k & 1) as i64 - (j & 1) as i64 + 1)
                            + 3 * (((k >> 1) & 1) as i64 - ((j >> 1) & 1) as i64 + 1)
                            + 9 * (((k >> 2) & 1) as i64 - ((j >> 2) & 1) as i64 + 1);
                        row[offset as usize] += w * mine.dot(p * *theirs);
                    }
                }
                for (offset, &value) in row.iter().enumerate() {
                    let o = offset as i64;
                    let (dx, dy, dz) = (o % 3 - 1, (o / 3) % 3 - 1, o / 9 - 1);
                    if offset == 13 {
                        columns.push(at as u32);
                        values.push((value * scale) as f64 + SHIFT * diagonal[at]);
                    } else if value != 0.0 {
                        if let Some(other) = find([key[0] + dx, key[1] + dy, key[2] + dz]) {
                            columns.push(other);
                            values.push((value * scale) as f64);
                        }
                    }
                }
                ends.push(columns.len());
            }
            (columns, values, ends)
        });
        let m = &mut self.matrix;
        m.starts.clear();
        m.columns.clear();
        m.values.clear();
        m.starts.push(0);
        for (columns, values, ends) in runs {
            let base = m.columns.len();
            m.starts.extend(ends.iter().map(|e| base + e));
            m.columns.extend_from_slice(&columns);
            m.values.extend_from_slice(&values);
        }
    }

    /// Make the grid's velocities divergence-free where fluid is.
    #[allow(clippy::too_many_arguments)]
    pub(super) fn project(
        &mut self,
        blocks: &mut [Block],
        block_index: &BlockMap<u32>,
        keys: &[[i64; 3]],
        fluids: &[FluidSet],
        order: &[Owner],
        cell: f32,
        h: f32,
        floor: f32,
        walled: &[[Mat3; BLOCK]],
        open: &[[f32; BLOCK]],
        settings: SolveSettings,
        threads: usize,
    ) -> ProjectionReport {
        // ---- the fluid cells, in grid order ----
        //
        // z slowest: the order the incomplete factor sweeps in, and the one
        // it is good in. Sorted rather than hashed, a point at a time.
        self.keyed.clear();
        let c = cell as f64;
        for &owner in order {
            let Owner::Fluid(f, i) = owner else {
                continue;
            };
            let fluid = &fluids[f as usize];
            let p = fluid.particles.position[i as usize];
            self.keyed.push((
                [
                    (p.z / c).floor() as i64,
                    (p.y / c).floor() as i64,
                    (p.x / c).floor() as i64,
                ],
                fluid.params.rest_density,
            ));
        }
        self.keyed.sort_unstable_by_key(|&(k, _)| k);
        self.cells.clear();
        self.corners.clear();
        self.count.clear();
        self.density.clear();
        for &(k, density) in &self.keyed {
            let key = [k[2], k[1], k[0]];
            if self.cells.last() != Some(&key) {
                self.cells.push(key);
                self.count.push(0);
                self.density.push(0.0);
            }
            *self.count.last_mut().unwrap() += 1;
            *self.density.last_mut().unwrap() += density as f64;
        }
        let n = self.cells.len();
        if n == 0 {
            self.previous_cells.clear();
            self.previous.clear();
            return ProjectionReport::default();
        }
        // A cell's corners lie in its base block and the next one along
        // each axis: eight blocks at most, looked up once for every cell
        // that shares the base block — four in a row, in grid order.
        let mut base_key = [i64::MAX; 3];
        let mut near = [u32::MAX; 8];
        let mut known = [false; 8];
        for key in &self.cells {
            let base = block_of(*key);
            if base != base_key {
                base_key = base;
                known = [false; 8];
            }
            let mut corners = [(u32::MAX, 0u8); 8];
            for (k, corner) in corners.iter_mut().enumerate() {
                let node = [
                    key[0] + (k & 1) as i64,
                    key[1] + ((k >> 1) & 1) as i64,
                    key[2] + ((k >> 2) & 1) as i64,
                ];
                let block = block_of(node);
                let which = ((block[0] - base[0])
                    + 2 * (block[1] - base[1])
                    + 4 * (block[2] - base[2])) as usize;
                if !known[which] {
                    near[which] = block_index.get(&block).copied().unwrap_or(u32::MAX);
                    known[which] = true;
                }
                if near[which] != u32::MAX {
                    *corner = (near[which], local_index(node) as u8);
                }
            }
            self.corners.push(corners);
        }

        // ---- each node's lumped mass and mixture velocity ----
        let volume = cell * cell * cell;
        self.weight.resize(blocks.len(), [0.0; BLOCK]);
        self.velocity.resize(blocks.len(), [Vec3::ZERO; BLOCK]);
        self.free.clear();
        self.free.extend_from_slice(walled);
        for (b, block) in blocks.iter().enumerate() {
            for node in 0..BLOCK {
                let total = block.total_mass(node);
                if total > floor {
                    self.weight[b][node] = h * volume / total;
                    self.velocity[b][node] = (block.momentum[node][0] * block.mass[node][0]
                        + block.momentum[node][1] * block.mass[node][1])
                        / total;
                } else {
                    self.weight[b][node] = 0.0;
                    self.velocity[b][node] = Vec3::ZERO;
                }
            }
        }

        // ---- right-hand side and diagonal ----
        self.rhs.clear();
        self.diagonal.clear();
        let mut before = 0.0f64;
        let mut targets = Vec::with_capacity(n);
        for (at, corners) in self.corners.iter().enumerate() {
            let mut divergence = 0.0f64;
            let mut diagonal = 0.0f64;
            let mut unwalled = 0.0f64;
            let mut fluid_mass = 0.0f64;
            let mut room = 0.0f64;
            for (k, &(b, node)) in corners.iter().enumerate() {
                if b == u32::MAX {
                    continue;
                }
                let (b, node) = (b as usize, node as usize);
                for s in 0..2 {
                    if blocks[b].field[node][s] & 0x8000_0000 != 0 {
                        fluid_mass += blocks[b].mass[node][s] as f64;
                    }
                }
                room += open[b][node] as f64;
                let g = corner_weight(k, cell);
                divergence += g.dot(self.velocity[b][node]) as f64;
                diagonal += (self.weight[b][node] * g.dot(self.free[b][node] * g)) as f64;
                unwalled += (self.weight[b][node] * g.dot(g)) as f64;
            }
            before = before.max(divergence.abs());
            // How full the cell is, from the grid: a node's mass is its
            // points' spline weights, which in fluid at rest sum to exactly
            // the rest density — smoothly, where counting points per cell
            // was noise, and since only crowding counts (a cell at the
            // surface is part empty by right) the noise inflated still water
            // by 5%.
            let rest = self.density[at] / self.count[at].max(1) as f64;
            // Against what fluid at rest would put there: beside a wall a
            // node's reach is part wall, and a tank's corner packed forty
            // deep read as under-full.
            let fill = if room > 0.5 {
                fluid_mass / (room * rest * c * c * c)
            } else {
                0.0
            };
            // A cell the walls hold shut asks only what its free corners
            // can give, or the solve answers it with an enormous pressure
            // next door.
            let freedom = if unwalled > 0.0 {
                diagonal / unwalled
            } else {
                0.0
            };
            let target = SPREAD * (fill - 1.0).max(0.0) * freedom / h as f64;
            targets.push(target);
            self.rhs.push(target - divergence);
            self.diagonal
                .push(if diagonal > 0.0 { diagonal } else { 1.0 });
        }

        // ---- solve, from where the last substep left the pressure ----
        // Both in grid order: a merge.
        self.pressure.clear();
        let mut q = 0;
        let order_of = |k: &[i64; 3]| (k[2], k[1], k[0]);
        for key in &self.cells {
            while q < self.previous_cells.len() && order_of(&self.previous_cells[q]) < order_of(key)
            {
                q += 1;
            }
            self.pressure
                .push(if self.previous_cells.get(q) == Some(key) {
                    self.previous[q]
                } else {
                    0.0
                });
        }
        self.gradient
            .borrow_mut()
            .resize(blocks.len(), [Vec3::ZERO; BLOCK]);
        let solve = {
            self.assemble(cell, threads);
            if self.cells != self.factored || self.factor_age >= REFACTOR {
                self.factor.factor(&self.matrix);
                self.factored.clone_from(&self.cells);
                self.factor_age = 0;
            }
            self.factor_age += 1;
            {
                let (matrix, factor) = (&self.matrix, &self.factor);
                let precondition = |r: &[f64], z: &mut [f64]| factor.apply(r, z);
                match self.pcg.solve_preconditioned_within_budget(
                    &self.rhs,
                    &mut self.pressure,
                    settings,
                    |x, y| matrix.apply(x, y),
                    precondition,
                ) {
                    Ok(report) => report,
                    Err(_) => {
                        self.previous_cells.clear();
                        self.previous.clear();
                        return ProjectionReport {
                            cells: n,
                            divergence_before: before as f32,
                            divergence_after: before as f32,
                            ..Default::default()
                        };
                    }
                }
            }
        };

        // ---- the correction: u += W Dᵀ p, on every field at the node ----
        let mut t = self.gradient.borrow_mut();
        for cs in &self.corners {
            for &(b, node) in cs {
                if b != u32::MAX {
                    t[b as usize][node as usize] = Vec3::ZERO;
                }
            }
        }
        for (at, cs) in self.corners.iter().enumerate() {
            let p = self.pressure[at] as f32;
            for (k, &(b, node)) in cs.iter().enumerate() {
                if b != u32::MAX {
                    t[b as usize][node as usize] += corner_weight(k, cell) * p;
                }
            }
        }
        for cs in &self.corners {
            for &(b, node) in cs {
                if b == u32::MAX {
                    continue;
                }
                let (b, node) = (b as usize, node as usize);
                let push = t[b][node];
                // Each node once: spent as it is applied.
                t[b][node] = Vec3::ZERO;
                let change = self.free[b][node] * push * self.weight[b][node];
                if change == Vec3::ZERO {
                    continue;
                }
                self.velocity[b][node] += change;
                blocks[b].momentum[node][0] += change;
                blocks[b].momentum[node][1] += change;
            }
        }
        drop(t);

        // ---- the projected velocity, one layer out into the air ----
        //
        // A point at the surface reads the nodes above it too, and those are
        // corners of no fluid cell: nothing projected them, they carry only
        // gravity, and the top layer read a velocity the rest of the water
        // did not have — still water's surface rose 5 cm in two seconds and
        // its volume spread. Given the mean of their projected neighbours, as
        // a grid fluid extends its velocity past its surface, they move with
        // the water under them. Only nodes that hold nothing but fluid: a
        // solid's node keeps its own.
        let mut corner = std::mem::take(&mut self.corner_mask);
        corner.clear();
        corner.resize(blocks.len(), [false; BLOCK]);
        for cs in &self.corners {
            for &(b, node) in cs {
                if b != u32::MAX {
                    corner[b as usize][node as usize] = true;
                }
            }
        }
        let mut extended: Vec<(usize, usize, Vec3)> = Vec::new();
        for (b, key) in keys.iter().enumerate() {
            for node in 0..BLOCK {
                // Nor a node a body holds.
                if corner[b][node]
                    || blocks[b].total_mass(node) <= floor
                    || self.free[b][node] == Mat3::ZERO
                {
                    continue;
                }
                let only_fluid = (0..2).all(|s| {
                    blocks[b].mass[node][s] <= 0.0 || blocks[b].field[node][s] & 0x8000_0000 != 0
                });
                if !only_fluid {
                    continue;
                }
                let at = super::node_of(*key, node);
                let mut sum = Vec3::ZERO;
                let mut count = 0;
                for d in [
                    [1, 0, 0],
                    [-1, 0, 0],
                    [0, 1, 0],
                    [0, -1, 0],
                    [0, 0, 1],
                    [0, 0, -1],
                ] {
                    let neighbour = [at[0] + d[0], at[1] + d[1], at[2] + d[2]];
                    let Some(&nb) = block_index.get(&block_of(neighbour)) else {
                        continue;
                    };
                    let nn = local_index(neighbour);
                    if corner[nb as usize][nn] {
                        sum += self.velocity[nb as usize][nn];
                        count += 1;
                    }
                }
                if count > 0 {
                    extended.push((b, node, sum / count as f32));
                }
            }
        }
        for (b, node, v) in extended {
            self.velocity[b][node] = v;
            blocks[b].momentum[node] = [v, v];
        }
        self.corner_mask = corner;

        let mut after = 0.0f64;
        for (at, cs) in self.corners.iter().enumerate().take(targets.len()) {
            let mut divergence = 0.0f64;
            for (k, &(b, node)) in cs.iter().enumerate() {
                if b != u32::MAX {
                    divergence +=
                        corner_weight(k, cell).dot(self.velocity[b as usize][node as usize]) as f64;
                }
            }
            after = after.max((divergence - targets[at]).abs());
        }
        std::mem::swap(&mut self.previous_cells, &mut self.cells);
        std::mem::swap(&mut self.previous, &mut self.pressure);
        ProjectionReport {
            cells: n,
            iterations: solve.iterations,
            converged: solve.iterations < settings.max_iterations,
            divergence_before: before as f32,
            divergence_after: after as f32,
        }
    }
}
