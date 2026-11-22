//! Stress inside rigid bodies.
//!
//! A rigid body's motion is solved as if it could not deform, and that is a
//! very good approximation for steel and concrete: the strains that carry the
//! loads are a thousandth or less. But the loads are real, and so is the
//! stress they cause — which is what decides whether a beam snaps, a glass
//! shatters on a tiled floor, or a flywheel bursts.
//!
//! ## Inertia relief
//!
//! Each step, every loaded body is treated as a free elastic solid in
//! equilibrium with everything acting on it:
//!
//! - the **contact forces** the solver applied this step, spread over each
//!   contact's patch of cells;
//! - the **inertial load** of its own acceleration — `-ρ (a + α × r +
//!   ω × (ω × r))` per unit volume, with `a` and `α` the accelerations those
//!   same forces produce — which is what balances them. Gravity drops out:
//!   it accelerates every particle of a body equally and stresses none of it,
//!   which is why a falling body is unstressed until it lands;
//! - and, with heat on, **thermal strain**: a hot cell wants to be bigger
//!   than its cold neighbours let it be. A temperature that varies linearly
//!   across a free body causes no stress at all; anything less regular does.
//!
//! The loads balance by construction, so the body needs no supports: the six
//! rigid motions are projected out of the solve, and what is left is the
//! deformation, and from it the stress. This is the "inertia relief" analysis
//! structural codes use for aircraft and spacecraft in flight, at the coarse
//! resolution of the body's [`Voxels`].
//!
//! ## The discretization
//!
//! Each inside cell is a trilinear hexahedral element; the nodes are the cell
//! corners. The element stiffness of a cube scales as `E c`, so it is
//! computed once for a unit cube and scaled — and the solve is written for
//! `E u` rather than `u`, so Young's modulus drops out of it entirely except
//! through thermal strain. Real 200 GPa steel costs the same as rubber. The
//! system is solved by conjugate gradients with a Jacobi preconditioner,
//! started from last step's answer: a steady load converges in a handful of
//! iterations and only a new impact pays for a full solve. Stress is read at
//! each element's centre, the point a trilinear element is most accurate at.
//!
//! ## How long an impact lasts
//!
//! A rigid solver stops a landing within one step, so the impulse it reports
//! is spread over `dt`. A real impact is over much sooner, and how much sooner
//! decides whether anything breaks. Two things set it:
//!
//! - **The wave.** The contact lasts at least as long as a stress wave takes
//!   to cross the body and come back, `2 L / c` with `c = sqrt(E / ρ)`: about
//!   0.1 ms for a 25 cm steel block, against 4 ms for a step. For a bar struck
//!   end-on this reproduces the textbook impact stress `ρ c Δv / 2`.
//! - **The body.** Treating the body as decelerated all at once is only right
//!   for a load slower than its own vibration. A long beam dropped onto two
//!   supports flexes over milliseconds, however quickly its ends were
//!   stopped, and feels the impulse as a load lasting about `1 / ω` — its
//!   lowest natural frequency, estimated each impact by Rayleigh's quotient
//!   from the very shape the impact bends it into.
//!
//! So the impact part of a contact's impulse — what stopped motion, as the
//! solver reports it, not what held a weight up — is solved on its own and
//! divided by the longer of the two. Spread over the step instead, a glass
//! landing on its corner would feel a fortieth of its real load; divided by
//! the wave time alone, a concrete beam set down at walking pace snapped
//! into eight.
//!
//! ## What it is not
//!
//! Linear, small-strain and isotropic. Joints do not load the bodies they
//! connect. A point or edge contact is a load on the nearest cell, so the
//! stress under a sharp corner is the cell's average, not the singularity
//! elasticity predicts — and the answer anywhere is the average over a cell
//! an eighth of the body across.

use fusion_core::{BodySet, BodyStress, Voxels};
use fusion_math::Vec3;
use fusion_solver::ContactReport;

use crate::heat::{normals, patch_cells};

/// How rigid bodies carry load.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct StressSettings {
    /// Measure the stress in every loaded rigid body. Off by default; a body
    /// that can break or deform is measured whether or not this is on, because
    /// it has to be.
    pub enabled: bool,
    /// Count thermal strain. Only does anything with heat on.
    pub thermal: bool,
    /// Shorten an impact's load to the time it really lasts. See the module
    /// documentation for why this matters.
    pub impact_duration: bool,
}

impl Default for StressSettings {
    fn default() -> Self {
        Self {
            enabled: false,
            thermal: true,
            impact_duration: true,
        }
    }
}

impl StressSettings {
    /// Stress measured in every body.
    pub fn on() -> Self {
        Self {
            enabled: true,
            ..Self::default()
        }
    }
}

/// Scratch, reused across steps.
#[derive(Clone, Debug, Default)]
pub struct StressWorking {
    /// Element stiffness of a unit cube at unit modulus, keyed by Poisson's
    /// ratio.
    stiffness: Vec<([u32; 4], Box<[f64; 576]>)>,
    wanted: Vec<bool>,
    loads: Vec<Load>,
    patch: Vec<usize>,
    points: Vec<Vec3>,
    surface: Vec<Load>,
    f: Vec<f64>,
    g: Vec<f64>,
    u: Vec<f64>,
    v: Vec<f64>,
    r: Vec<f64>,
    z: Vec<f64>,
    p: Vec<f64>,
    ap: Vec<f64>,
    diag: Vec<f64>,
    lumped: Vec<f64>,
    modes: Vec<Vec<f64>>,
    nodes: Vec<u32>,
    elements: Vec<usize>,
    /// Factored stiffness matrices, by body, most recently used last.
    factors: Vec<Factor>,
    /// Conjugate-gradient iterations the hardest body needed last step.
    pub iterations: u32,
    /// Bodies whose stress was solved last step.
    pub solved: usize,
}

/// A contact's load on one cell of one body, in the body's frame: a steady
/// force, and the impulse of an impact. Where in the cell it acts is in
/// unit-cell coordinates, so a load on a face goes to that face's nodes.
#[derive(Clone, Copy, Debug)]
struct Load {
    body: u32,
    cell: u32,
    force: Vec3,
    impulse: Vec3,
    at: Vec3,
    /// The wave's round trip across the body along this contact's normal.
    wave: f32,
    /// This cell's share of the kinetic energy the impact gave this body to
    /// store, J.
    energy: f32,
}

/// A body's stiffness matrix, factored once.
///
/// A rigid body's stiffness never changes — its shape and material are what
/// they are — so the matrix is factored the first time the body is loaded and
/// every solve after that is two triangular sweeps: exact, and about a
/// millisecond for the largest grid. Conjugate gradients on the same system
/// needed three hundred iterations and did not converge for a long plank,
/// which made each impact on one a tenth of a second and an underestimate.
///
/// Banded: the degrees of freedom are numbered with the grid's longest axis
/// slowest, so the matrix's band is a cross-section of the body wide. The
/// six rigid motions are removed by pinning three nodes the 3-2-1 way, which
/// leaves a positive definite system with the same stresses: the loads are in
/// equilibrium, so the pins carry nothing.
#[derive(Clone, Debug)]
struct Factor {
    body: usize,
    key: [u32; 8],
    /// Band width, below the diagonal.
    band: usize,
    /// Natural degree of freedom to banded.
    order: Vec<u32>,
    /// The factor, row by row: `band + 1` entries ending on the diagonal.
    lower: Vec<f64>,
    /// Pinned degrees of freedom, natural numbering.
    pinned: Vec<u32>,
}

/// Most factors kept at once.
const FACTORS: usize = 48;

impl Factor {
    fn at(&self, i: usize, j: usize) -> usize {
        // Row i, column j <= i, within the band.
        i * (self.band + 1) + (j + self.band - i)
    }

    /// Solve `K x = f` into `x`, both in natural numbering.
    fn solve(&self, f: &[f64], x: &mut [f64]) {
        let n = self.order.len();
        let mut y = vec![0.0; n];
        for (natural, &banded) in self.order.iter().enumerate() {
            y[banded as usize] = f[natural];
        }
        for &p in &self.pinned {
            y[self.order[p as usize] as usize] = 0.0;
        }
        // Forward: L y' = y, a row of the factor at a time.
        for i in 0..n {
            let start = i.saturating_sub(self.band);
            let row = &self.lower[self.at(i, start)..self.at(i, i)];
            y[i] = (y[i] - lanes_dot(row, &y[start..i])) / self.lower[self.at(i, i)];
        }
        // Back: L^T x' = y', down a column of it.
        for i in (0..n).rev() {
            let end = (i + self.band).min(n - 1);
            let below: f64 = (i + 1..=end)
                .map(|j| self.lower[self.at(j, i)] * y[j])
                .sum();
            y[i] = (y[i] - below) / self.lower[self.at(i, i)];
        }
        for (natural, &banded) in self.order.iter().enumerate() {
            x[natural] = y[banded as usize];
        }
    }
}

/// Factor the stiffness of a body's grid. `None` if it is not positive
/// definite once pinned, which a body that hangs together always is.
#[allow(clippy::too_many_arguments)]
fn factorize(
    body: usize,
    key: [u32; 8],
    dims: [usize; 3],
    ke: &[f64; 576],
    nodes: &[u32],
    elements: usize,
    active: &[bool],
    position: impl Fn(usize) -> Vec3,
) -> Option<Factor> {
    let mx = dims[0] + 1;
    let my = dims[1] + 1;
    let mz = dims[2] + 1;
    let node_count = mx * my * mz;
    let n = 3 * node_count;
    // Axes from fastest to slowest: the longest last.
    let mut axes = [0usize, 1, 2];
    axes.sort_by_key(|&k| [mx, my, mz][k]);
    let counts = [mx, my, mz];
    let mut order = vec![0u32; n];
    for node in 0..node_count {
        let c = [node % mx, (node / mx) % my, node / (mx * my)];
        let banded_node =
            c[axes[0]] + counts[axes[0]] * (c[axes[1]] + counts[axes[1]] * c[axes[2]]);
        for d in 0..3 {
            order[3 * node + d] = (3 * banded_node + d) as u32;
        }
    }
    let band = 3 * (counts[axes[0]] * counts[axes[1]] + counts[axes[0]] + 1) + 2;
    let width = band + 1;
    let mut lower = vec![0.0; n * width];
    let at = |i: usize, j: usize| i * width + (j + band - i);
    for e in 0..elements {
        let en = &nodes[8 * e..8 * e + 8];
        for a in 0..8 {
            for da in 0..3 {
                let r = order[3 * en[a] as usize + da] as usize;
                for b in 0..8 {
                    for db in 0..3 {
                        let c = order[3 * en[b] as usize + db] as usize;
                        if c <= r {
                            lower[at(r, c)] += ke[(3 * a + da) * 24 + 3 * b + db];
                        }
                    }
                }
            }
        }
    }
    // Unused degrees of freedom stand alone.
    for node in 0..node_count {
        if !active[node] {
            for d in 0..3 {
                let r = order[3 * node + d] as usize;
                lower[at(r, r)] = 1.0;
            }
        }
    }
    // The 3-2-1 pins: one node fixed, a far one held in two directions, and
    // a third off their line held in one.
    let live: Vec<usize> = (0..node_count).filter(|&k| active[k]).collect();
    let first = *live.first()?;
    let p0 = position(first);
    let second = *live.iter().max_by(|&&a, &&b| {
        (position(a) - p0)
            .length_squared()
            .total_cmp(&(position(b) - p0).length_squared())
    })?;
    let axis = (position(second) - p0).normalize_or_zero();
    let third = *live.iter().max_by(|&&a, &&b| {
        let da = (position(a) - p0).cross(axis).length_squared();
        let db = (position(b) - p0).cross(axis).length_squared();
        da.total_cmp(&db)
    })?;
    let dominant = |v: Vec3| {
        let a = v.abs();
        if a.x >= a.y && a.x >= a.z {
            0
        } else if a.y >= a.z {
            1
        } else {
            2
        }
    };
    let along = dominant(axis);
    let normal = dominant((position(second) - p0).cross(position(third) - p0));
    let mut pinned = vec![3 * first as u32, 3 * first as u32 + 1, 3 * first as u32 + 2];
    for d in 0..3 {
        if d != along {
            pinned.push((3 * second + d) as u32);
        }
    }
    pinned.push((3 * third + normal) as u32);
    for &p in &pinned {
        let r = order[p as usize] as usize;
        for j in r.saturating_sub(band)..r {
            lower[at(r, j)] = 0.0;
        }
        for i in r + 1..(r + band + 1).min(n) {
            lower[at(i, r)] = 0.0;
        }
        lower[at(r, r)] = 1.0;
    }
    // Cholesky, in the band.
    for i in 0..n {
        let start = i.saturating_sub(band);
        for j in start..=i {
            // Both rows are contiguous over the columns they share.
            let k0 = start.max(j.saturating_sub(band));
            let ri = &lower[at(i, k0)..at(i, j)];
            let rj = &lower[at(j, k0)..at(j, j)];
            let sum = lower[at(i, j)] - lanes_dot(ri, rj);
            if i == j {
                if sum.is_nan() || sum <= 0.0 {
                    return None;
                }
                lower[at(i, i)] = sum.sqrt();
            } else {
                lower[at(i, j)] = sum / lower[at(j, j)];
            }
        }
    }
    Some(Factor {
        body,
        key,
        band,
        order,
        lower,
        pinned,
    })
}

/// `a · b` in four lanes, which the compiler can keep in vector registers.
fn lanes_dot(a: &[f64], b: &[f64]) -> f64 {
    let mut lanes = [0.0f64; 4];
    let (a4, a1) = a.split_at(a.len() / 4 * 4);
    let (b4, b1) = b.split_at(a4.len());
    for (x, y) in a4.chunks_exact(4).zip(b4.chunks_exact(4)) {
        for k in 0..4 {
            lanes[k] += x[k] * y[k];
        }
    }
    let tail: f64 = a1.iter().zip(b1).map(|(x, y)| x * y).sum();
    lanes[0] + lanes[1] + lanes[2] + lanes[3] + tail
}

/// Relative residual a solve stops at under a steady load.
///
/// Loose, because a steady load is solved again next step from this answer,
/// so the error that matters is the one the warm start carries rather than
/// one step's: a resting beam's stress settles to well under a percent of its
/// converged value. What is left of the residual from step to step is the
/// solver's own contact noise, about a percent, and chasing it costs a hundred
/// iterations a step for nothing.
const TOLERANCE: f64 = 1e-2;

/// Relative residual for an impact, which is solved once and acted on: the
/// step a glass lands is the step it does or does not break. A tenth of a
/// percent of the load, which moves the stress a great deal less than the
/// coarseness of the grid does.
const IMPACT_TOLERANCE: f64 = 1e-3;

/// Ceiling on iterations. A cold solve of an 8x8x8 body takes about forty,
/// and a 40x4x4 beam about a hundred and fifty.
const MAX_ITERATIONS: u32 = 300;

/// Measure the stress in every rigid body that needs it, from the contacts of
/// the step that just ran.
pub fn step(
    bodies: &mut BodySet,
    reports: &[ContactReport],
    settings: &StressSettings,
    w: &mut StressWorking,
    dt: f32,
) {
    let n = bodies.len();
    w.iterations = 0;
    w.solved = 0;
    w.wanted.clear();
    w.wanted.extend((0..n).map(|i| {
        !bodies.is_removed(i)
            && bodies.advanced[i]
            && bodies.is_movable(i)
            && bodies.material[i].strength.carries_load()
            && (settings.enabled || bodies.failure[i].can_fail())
    }));
    if !w.wanted.iter().any(|&b| b) {
        return;
    }
    for i in 0..n {
        if w.wanted[i] && !bodies.stress[i].is_built() {
            build(bodies, i);
        }
    }

    // ---- contact loads, onto cells ----
    w.loads.clear();
    let mut start = 0;
    while start < reports.len() {
        let (a, b) = (reports[start].body_a, reports[start].body_b);
        let mut end = start + 1;
        while end < reports.len() && reports[end].body_a == a && reports[end].body_b == b {
            end += 1;
        }
        let group = &reports[start..end];
        start = end;
        let (a, b) = (a as usize, b as usize);
        if !(w.wanted[a] || w.wanted[b]) {
            continue;
        }
        let (na, nb) = normals(bodies, a, b, &group[0]);
        let normal_world = bodies.orientation[a].rotate(group[0].local_normal_a);
        for (body, outward, on_a) in [(a, na, true), (b, nb, false)] {
            if !w.wanted[body] {
                continue;
            }
            // The manifold's total, spread evenly over its patch. How the
            // solver splits a resting load between a face's corners is not
            // determined by anything — four corners can carry a block in
            // infinitely many ways — and it shuffles from step to step while
            // the total holds still. Spread per corner, the stress under a
            // resting beam flickered by a third every step; spread as a
            // pressure, it is the even bearing a flat face on a flat face is.
            let sign = if on_a { -1.0 } else { 1.0 };
            let mut total = Vec3::ZERO;
            let mut impact = 0.0;
            let mut speed = 0.0f32;
            for r in group {
                total += r.impulse * sign;
                impact += r.impact_impulse;
                speed = speed.max(r.impact_speed);
            }
            // The kinetic energy of the approach, which at the moment of
            // greatest compression is all elastic: `m_eff v² / 2`, written
            // from the impulse that stopped it, `m_eff (1 + e) v`. Shared
            // between the two bodies as contact compliance is, by `1 / E`,
            // so rubber hitting steel stores nearly all of it.
            let restitution = group[0].restitution.clamp(0.0, 1.0);
            let approach = 0.5 * impact.max(0.0) * speed / (1.0 + restitution);
            let compliance = |k: usize| {
                let e = bodies.material[k].strength.youngs_modulus;
                if e > 0.0 && e.is_finite() {
                    1.0 / e
                } else {
                    0.0
                }
            };
            let (ca, cb) = (compliance(a), compliance(b));
            let mine = if on_a { ca } else { cb };
            let energy = if ca + cb > 0.0 {
                approach * mine / (ca + cb)
            } else {
                0.0
            };
            let impulse = if settings.impact_duration {
                normal_world * (impact * sign)
            } else {
                Vec3::ZERO
            };
            let force = (total - impulse) / dt;
            if force == Vec3::ZERO && impulse == Vec3::ZERO {
                continue;
            }
            w.points.clear();
            w.points.extend(
                group
                    .iter()
                    .map(|r| if on_a { r.local_a } else { r.local_b }),
            );
            let grid = &bodies.stress[body].grid;
            patch_cells(grid, &w.points, outward, &mut w.patch);
            if w.patch.is_empty() {
                continue;
            }
            let rotation = bodies.orientation[body];
            let share = 1.0 / w.patch.len() as f32;
            let force = rotation.rotate_inverse(force) * share;
            let impulse = rotation.rotate_inverse(impulse) * share;
            // On the face of each cell that the contact is on.
            let at =
                (Vec3::splat(0.5) + outward.normalize_or_zero() * 0.5).clamp(Vec3::ZERO, Vec3::ONE);
            let wave = wave_time(bodies, body, outward);
            let energy = energy * share;
            for &c in &w.patch {
                w.loads.push(Load {
                    body: body as u32,
                    cell: c as u32,
                    force,
                    impulse,
                    at,
                    wave,
                    energy,
                });
            }
        }
    }
    w.loads.sort_by_key(|load| (load.body, load.cell));

    // ---- one solve per body ----
    let mut next = 0;
    for i in 0..n {
        if !w.wanted[i] {
            continue;
        }
        w.surface.clear();
        while next < w.loads.len() && (w.loads[next].body as usize) < i {
            next += 1;
        }
        while next < w.loads.len() && w.loads[next].body as usize == i {
            w.surface.push(w.loads[next]);
            next += 1;
        }
        solve_body(bodies, i, settings, w, dt);
    }
}

/// Lay the stress grid over a body. Finer than the heat grid across a thin
/// body — see [`Voxels::for_stress`] — so the two are separate, and thermal
/// strain reads the heat grid at each stress cell's centre.
fn build(bodies: &mut BodySet, i: usize) {
    let budget = if bodies.reshaped[i] {
        fusion_core::voxels::STRESS_PIECE_CELLS
    } else {
        fusion_core::voxels::STRESS_MAX_CELLS
    };
    let Some(grid) = Voxels::for_stress_within(&bodies.shape[i], budget) else {
        return;
    };
    let [nx, ny, nz] = grid.dims;
    let nodes = (nx + 1) * (ny + 1) * (nz + 1);
    bodies.stress[i] = BodyStress {
        stress: vec![[0.0; 6]; grid.len()],
        displacement: vec![0.0; 3 * nodes],
        grid,
        settling: bodies.stress[i].settling,
        ..BodyStress::default()
    };
}

/// The elastic energy that would load all of a body of `volume` to its
/// weakest strength, `s² V / 2E`, J — the scale an impact's energy is judged
/// against. A material with no finite strength is judged against 100 MPa.
fn breaking_energy(material: &fusion_core::Material, volume: f32) -> f32 {
    let s = &material.strength;
    let weakest = [s.tensile, s.compressive, s.yield_stress]
        .into_iter()
        .filter(|v| v.is_finite() && *v > 0.0)
        .fold(f32::INFINITY, f32::min);
    let weakest = if weakest.is_finite() { weakest } else { 1e8 };
    weakest * weakest * volume / (2.0 * s.youngs_modulus.max(1.0))
}

/// A stress wave's round trip across `body` along `outward`, seconds.
fn wave_time(bodies: &BodySet, body: usize, outward: Vec3) -> f32 {
    let shape = &bodies.shape[body];
    let n = outward.normalize_or_zero();
    if n == Vec3::ZERO {
        return 0.0;
    }
    let width = (shape.support_local(n) - shape.support_local(-n)).dot(n);
    let material = &bodies.material[body];
    let c = material.strength.bar_wave_speed(material.density);
    if !(c > 0.0 && width > 0.0) {
        return 0.0;
    }
    2.0 * width / c
}

/// The eight corners of a unit cube, as `(x, y, z)` in `{0, 1}`, in the order
/// element matrices are written.
const CORNERS: [[usize; 3]; 8] = [
    [0, 0, 0],
    [1, 0, 0],
    [0, 1, 0],
    [1, 1, 0],
    [0, 0, 1],
    [1, 0, 1],
    [0, 1, 1],
    [1, 1, 1],
];

/// Gradient of corner `a`'s shape function at `xi` in a unit cube.
fn gradient(a: usize, xi: [f64; 3]) -> [f64; 3] {
    let [cx, cy, cz] = CORNERS[a];
    let f = |c: usize, t: f64| if c == 1 { t } else { 1.0 - t };
    let s = |c: usize| if c == 1 { 1.0 } else { -1.0 };
    [
        s(cx) * f(cy, xi[1]) * f(cz, xi[2]),
        f(cx, xi[0]) * s(cy) * f(cz, xi[2]),
        f(cx, xi[0]) * f(cy, xi[1]) * s(cz),
    ]
}

/// Strain from the 24 corner displacements, `[xx, yy, zz, γxy, γyz, γzx]`,
/// for gradients `g`.
fn strain(g: &[[f64; 3]; 8], u: &[f64; 24]) -> [f64; 6] {
    let mut e = [0.0; 6];
    for a in 0..8 {
        let [gx, gy, gz] = g[a];
        let (ux, uy, uz) = (u[3 * a], u[3 * a + 1], u[3 * a + 2]);
        e[0] += gx * ux;
        e[1] += gy * uy;
        e[2] += gz * uz;
        e[3] += gy * ux + gx * uy;
        e[4] += gz * uy + gy * uz;
        e[5] += gx * uz + gz * ux;
    }
    e
}

/// Stress from strain at unit modulus, Voigt order with engineering shear.
fn hooke(nu: f64, e: [f64; 6]) -> [f64; 6] {
    let lambda = nu / ((1.0 + nu) * (1.0 - 2.0 * nu));
    let mu = 1.0 / (2.0 * (1.0 + nu));
    let trace = e[0] + e[1] + e[2];
    [
        lambda * trace + 2.0 * mu * e[0],
        lambda * trace + 2.0 * mu * e[1],
        lambda * trace + 2.0 * mu * e[2],
        mu * e[3],
        mu * e[4],
        mu * e[5],
    ]
}

/// A shape-function gradient in a unit cube, carried into a box of edges
/// `size`.
fn scaled(g: [f64; 3], size: [f64; 3]) -> [f64; 3] {
    [g[0] / size[0], g[1] / size[1], g[2] / size[2]]
}

/// Element stiffness of a box of edges `size` at unit modulus, by 2x2x2 Gauss
/// quadrature, which integrates a trilinear element's stiffness exactly.
fn element_stiffness(nu: f64, size: [f64; 3]) -> Box<[f64; 576]> {
    let mut k = Box::new([0.0; 576]);
    let g = 0.5 / 3f64.sqrt();
    let weight = 0.125 * size[0] * size[1] * size[2];
    for qz in [0.5 - g, 0.5 + g] {
        for qy in [0.5 - g, 0.5 + g] {
            for qx in [0.5 - g, 0.5 + g] {
                let grads: [[f64; 3]; 8] =
                    std::array::from_fn(|a| scaled(gradient(a, [qx, qy, qz]), size));
                // Column j of K is B^T D B e_j: the forces a unit displacement
                // of DOF j produces.
                for j in 0..24 {
                    let mut unit = [0.0; 24];
                    unit[j] = 1.0;
                    let s = hooke(nu, strain(&grads, &unit));
                    for a in 0..8 {
                        let [gx, gy, gz] = grads[a];
                        // B^T s for corner a.
                        let fx = gx * s[0] + gy * s[3] + gz * s[5];
                        let fy = gy * s[1] + gx * s[3] + gz * s[4];
                        let fz = gz * s[2] + gy * s[4] + gx * s[5];
                        k[(3 * a) * 24 + j] += weight * fx;
                        k[(3 * a + 1) * 24 + j] += weight * fy;
                        k[(3 * a + 2) * 24 + j] += weight * fz;
                    }
                }
            }
        }
    }
    k
}

fn dot(a: &[f64], b: &[f64]) -> f64 {
    a.iter().zip(b).map(|(x, y)| x * y).sum()
}

/// Remove every rigid motion from `v`.
fn project(modes: &[Vec<f64>], v: &mut [f64]) {
    for q in modes {
        let c = dot(q, v);
        for (x, y) in v.iter_mut().zip(q) {
            *x -= c * y;
        }
    }
}

/// `out = K u` for the body whose elements are `elements`, with element
/// stiffness `cell * ke`.
fn apply(
    ke: &[f64; 576],
    cell: f64,
    nodes: &[u32],
    elements: usize,
    u: &[f64],
    out: &mut Vec<f64>,
) {
    out.clear();
    out.resize(u.len(), 0.0);
    let mut ue = [0.0; 24];
    for e in 0..elements {
        let en = &nodes[8 * e..8 * e + 8];
        for (a, &node) in en.iter().enumerate() {
            let base = 3 * node as usize;
            ue[3 * a..3 * a + 3].copy_from_slice(&u[base..base + 3]);
        }
        for (a, &node) in en.iter().enumerate() {
            let base = 3 * node as usize;
            for d in 0..3 {
                let row = &ke[(3 * a + d) * 24..(3 * a + d) * 24 + 24];
                out[base + d] += cell * row.iter().zip(&ue).map(|(k, x)| k * x).sum::<f64>();
            }
        }
    }
}

/// Preconditioned conjugate gradients on `K u = f`, in the space orthogonal
/// to the rigid motions, from whatever `u` holds. Returns the iterations.
#[allow(clippy::too_many_arguments)]
fn conjugate_gradients(
    ke: &[f64; 576],
    cell: f64,
    nodes: &[u32],
    elements: usize,
    diag: &[f64],
    modes: &[Vec<f64>],
    f: &[f64],
    u: &mut [f64],
    scratch: [&mut Vec<f64>; 4],
    tolerance: f64,
) -> u32 {
    let [r, z, p, ap] = scratch;
    project(modes, u);
    let f_norm = dot(f, f).sqrt();
    if f_norm == 0.0 {
        u.fill(0.0);
        return 0;
    }
    apply(ke, cell, nodes, elements, u, ap);
    r.clear();
    r.extend(f.iter().zip(ap.iter()).map(|(f, a)| f - a));
    project(modes, r);
    let precondition = |r: &[f64], z: &mut Vec<f64>| {
        z.clear();
        z.extend(
            r.iter()
                .zip(diag)
                .map(|(r, d)| if *d > 0.0 { r / d } else { 0.0 }),
        );
        project(modes, z);
    };
    precondition(r, z);
    p.clear();
    p.extend_from_slice(z);
    let mut rz = dot(r, z);
    let mut iterations = 0;
    while iterations < MAX_ITERATIONS && dot(r, r).sqrt() > tolerance * f_norm {
        apply(ke, cell, nodes, elements, p, ap);
        let pap = dot(p, ap);
        if pap <= 0.0 {
            break;
        }
        let step = rz / pap;
        for k in 0..u.len() {
            u[k] += step * p[k];
            r[k] -= step * ap[k];
        }
        precondition(r, z);
        let rz_next = dot(r, z);
        let beta = rz_next / rz;
        rz = rz_next;
        for k in 0..u.len() {
            p[k] = z[k] + beta * p[k];
        }
        iterations += 1;
    }
    iterations
}

/// Add surface loads, and the inertial loads that balance them, to the nodal
/// vector `f`. `value` picks the steady force or the impulse out of a load.
#[allow(clippy::too_many_arguments)]
fn load_vector(
    grid: &Voxels,
    surface: &[Load],
    value: impl Fn(&Load) -> Vec3,
    elements: &[usize],
    element_of: &[u32],
    nodes: &[u32],
    mass: f32,
    inertia: Vec3,
    omega: Option<Vec3>,
    f: &mut [f64],
) {
    let mut total = Vec3::ZERO;
    let mut moment = Vec3::ZERO;
    for load in surface {
        let v = value(load);
        if v == Vec3::ZERO {
            continue;
        }
        total += v;
        let point = grid.cell_centre(load.cell as usize)
            + (load.at - Vec3::splat(0.5)).mul_element_wise(grid.size);
        moment += point.cross(v);
        let e = element_of[load.cell as usize];
        if e == u32::MAX {
            continue;
        }
        let en = &nodes[8 * e as usize..8 * e as usize + 8];
        for (a, &node) in en.iter().enumerate() {
            let weight = (0..3)
                .map(|k| {
                    if CORNERS[a][k] == 1 {
                        load.at[k]
                    } else {
                        1.0 - load.at[k]
                    }
                })
                .product::<f32>();
            if weight == 0.0 {
                continue;
            }
            let base = 3 * node as usize;
            for d in 0..3 {
                f[base + d] += (v[d] * weight) as f64;
            }
        }
    }
    // Inertia relief: the body accelerates under these loads, and every cell's
    // share of mass resists that with its own inertial force. Spin's
    // centripetal and gyroscopic terms are steady, so they go with the force.
    let omega_v = omega.unwrap_or(Vec3::ZERO);
    let a = total / mass;
    let gyroscopic = omega_v.cross(inertia.mul_element_wise(omega_v));
    let alpha = (moment - gyroscopic).div_element_wise(inertia.max(Vec3::splat(1e-12)));
    let cell_mass = mass / elements.len().max(1) as f32;
    for (e, &c) in elements.iter().enumerate() {
        let r = grid.cell_centre(c);
        let accel = a + alpha.cross(r) + omega_v.cross(omega_v.cross(r));
        let share = accel * (-cell_mass / 8.0);
        for &node in &nodes[8 * e..8 * e + 8] {
            let base = 3 * node as usize;
            for d in 0..3 {
                f[base + d] += share[d] as f64;
            }
        }
    }
}

// Cells, nodes and degrees of freedom are three index spaces that refer to
// one another; the loops say which one they walk.
#[allow(clippy::needless_range_loop)]
fn solve_body(
    bodies: &mut BodySet,
    i: usize,
    settings: &StressSettings,
    w: &mut StressWorking,
    dt: f32,
) {
    let material = bodies.material[i];
    let strength = material.strength;
    let nu = (strength.poissons_ratio as f64).clamp(0.0, 0.45);
    let e_modulus = strength.youngs_modulus as f64;

    let heat = &bodies.heat[i];
    let grid = bodies.stress[i].grid.clone();
    let mean_temperature = heat.mean();
    let thermal = settings.thermal && heat.grid.is_built() && material.thermal.expansion > 0.0 && {
        let (lo, hi) = heat.range();
        hi - lo > 1e-3
    };

    let mass = bodies.mass(i);
    let omega = bodies.orientation[i].rotate_inverse(bodies.angular_velocity[i]);
    let radius = bodies.shape[i].bounding_radius();
    let spinning =
        (omega.length_squared() * radius * radius) as f64 * material.density as f64 > 1.0;
    let loaded = !w.surface.is_empty();
    // An impact is solved as one only if it could matter: if the energy it
    // brings is at least a hundredth of what it would take to load the whole
    // body to its weakest strength. Below that it is a body settling — a pile
    // of broken glass jostles at a few centimetres a second, every step, and
    // solving each nudge cold cost three hundred iterations a step for
    // stresses a thousandth of the glass's strength. Its impulse is then
    // spread over the step like any other load.
    let impact_energy: f32 = w.surface.iter().map(|l| l.energy).sum();
    let impact = w.surface.iter().any(|l| l.impulse != Vec3::ZERO)
        && impact_energy >= 0.01 * breaking_energy(&material, bodies.shape[i].volume());
    if !impact {
        for load in &mut w.surface {
            load.force += load.impulse / dt;
            load.impulse = Vec3::ZERO;
        }
    }
    if !(loaded || thermal || spinning) || !mass.is_finite() {
        bodies.stress[i].clear();
        return;
    }
    // Each stress cell's temperature, from the heat cell it sits in.
    let temperature: Vec<f64> = (0..grid.len())
        .map(|c| {
            if grid.inside[c] {
                heat.at_local(grid.cell_centre(c))
            } else {
                mean_temperature
            }
        })
        .collect();

    // ---- the elements, and the stiffness ----
    let [nx, ny, nz] = grid.dims;
    let (mx, my) = (nx + 1, ny + 1);
    let node_count = mx * my * (nz + 1);
    let dofs = 3 * node_count;
    let size = [grid.size.x as f64, grid.size.y as f64, grid.size.z as f64];
    let key = [
        (nu as f32).to_bits(),
        grid.size.x.to_bits(),
        grid.size.y.to_bits(),
        grid.size.z.to_bits(),
    ];
    let k_index = match w.stiffness.iter().position(|(k, _)| *k == key) {
        Some(index) => index,
        None => {
            // Bounded: one entry per distinct cell box, and a scene has few.
            if w.stiffness.len() > 64 {
                w.stiffness.clear();
            }
            w.stiffness.push((key, element_stiffness(nu, size)));
            w.stiffness.len() - 1
        }
    };
    // The element matrix is already for this box; nothing scales it.
    let cell = 1.0f64;
    w.nodes.clear();
    w.elements.clear();
    let mut element_of = vec![u32::MAX; grid.len()];
    for c in 0..grid.len() {
        if !grid.inside[c] {
            continue;
        }
        let [x, y, z] = grid.coordinates(c);
        element_of[c] = w.elements.len() as u32;
        w.elements.push(c);
        for [cx, cy, cz] in CORNERS {
            w.nodes
                .push(((x + cx) + mx * ((y + cy) + my * (z + cz))) as u32);
        }
    }
    let elements = w.elements.len();
    let ke = &w.stiffness[k_index].1;
    w.diag.clear();
    w.diag.resize(dofs, 0.0);
    w.lumped.clear();
    w.lumped.resize(dofs, 0.0);
    let cell_mass = (mass / elements.max(1) as f32) as f64;
    for e in 0..elements {
        for (a, &node) in w.nodes[8 * e..8 * e + 8].iter().enumerate() {
            let base = 3 * node as usize;
            for d in 0..3 {
                w.diag[base + d] += cell * ke[(3 * a + d) * 25];
                w.lumped[base + d] += cell_mass / 8.0;
            }
        }
    }

    // ---- the rigid motions, orthonormal ----
    let node_position = |n: usize| {
        let x = n % mx;
        let y = (n / mx) % my;
        let z = n / (mx * my);
        grid.origin
            + (Vec3::new(x as f32, y as f32, z as f32) - Vec3::splat(0.5))
                .mul_element_wise(grid.size)
    };
    let mut active = vec![false; node_count];
    for &n in &w.nodes {
        active[n as usize] = true;
    }
    let mut centre = Vec3::ZERO;
    let mut count = 0.0;
    for (n, &on) in active.iter().enumerate() {
        if on {
            centre += node_position(n);
            count += 1.0;
        }
    }
    centre /= count;
    const AXES: [Vec3; 3] = [Vec3::X, Vec3::Y, Vec3::Z];
    w.modes.resize(6, Vec::new());
    for (m, mode) in w.modes.iter_mut().enumerate() {
        mode.clear();
        mode.resize(dofs, 0.0);
        for (n, &on) in active.iter().enumerate() {
            if !on {
                continue;
            }
            let v = if m < 3 {
                AXES[m]
            } else {
                AXES[m - 3].cross(node_position(n) - centre)
            };
            for d in 0..3 {
                mode[3 * n + d] = v[d] as f64;
            }
        }
    }
    for m in 0..6 {
        let (done, rest) = w.modes.split_at_mut(m);
        let mode = &mut rest[0];
        for q in done.iter() {
            let c = dot(q, mode);
            for (x, y) in mode.iter_mut().zip(q) {
                *x -= c * y;
            }
        }
        let norm = dot(mode, mode).sqrt();
        if norm > 0.0 {
            for x in mode.iter_mut() {
                *x /= norm;
            }
        }
    }

    // ---- the factored stiffness, built the first time it is needed ----
    let factor_key = [
        key[0],
        key[1],
        key[2],
        key[3],
        grid.dims[0] as u32,
        grid.dims[1] as u32,
        grid.dims[2] as u32,
        // Which cells are inside: a body broken in two can keep its grid's
        // dimensions and lose half its cells.
        grid.inside.iter().fold(0x811c_9dc5u32, |h, &on| {
            (h ^ on as u32).wrapping_mul(0x0100_0193)
        }),
    ];
    let factor_index = match w
        .factors
        .iter()
        .position(|f| f.body == i && f.key == factor_key)
    {
        Some(k) => Some(k),
        None => {
            // A stale factor for this body — its shape changed — goes first.
            w.factors.retain(|f| f.body != i);
            let built = factorize(
                i,
                factor_key,
                grid.dims,
                ke,
                &w.nodes,
                elements,
                &active,
                node_position,
            );
            built.map(|f| {
                if w.factors.len() >= FACTORS {
                    w.factors.remove(0);
                }
                w.factors.push(f);
                w.factors.len() - 1
            })
        }
    };

    // ---- the steady part: resting loads, spin and heat ----
    w.f.clear();
    w.f.resize(dofs, 0.0);
    let inertia = bodies.inertia_local[i];
    load_vector(
        &grid,
        &w.surface,
        |l| l.force,
        &w.elements,
        &element_of,
        &w.nodes,
        mass,
        inertia,
        Some(omega),
        &mut w.f,
    );
    let expansion = material.thermal.expansion as f64;
    if thermal {
        // ∫∇N dV over a box is its volume over four times the edge, per
        // axis, with the corner's sign.
        let volume = size[0] * size[1] * size[2];
        let scale: [f64; 3] =
            std::array::from_fn(|k| e_modulus * volume / (4.0 * size[k] * (1.0 - 2.0 * nu)));
        for (e, &c) in w.elements.iter().enumerate() {
            let strain = expansion * (temperature[c] - mean_temperature);
            for (a, &node) in w.nodes[8 * e..8 * e + 8].iter().enumerate() {
                let base = 3 * node as usize;
                for d in 0..3 {
                    let s = if CORNERS[a][d] == 1 { 1.0 } else { -1.0 };
                    w.f[base + d] += scale[d] * strain * s;
                }
            }
        }
    }
    project(&w.modes, &mut w.f);

    // Nothing has moved since the last solve: a stack at rest, whose loads
    // shuffle by a fraction of a percent from step to step. The answer it
    // has is the answer to within the tolerance a solve would stop at, so it
    // keeps it — the solve, and the stress read at every sample point, cost
    // a resting body ten times what its motion does.
    if !impact {
        let previous = &bodies.stress[i].load;
        if previous.len() == dofs {
            let change: f64 =
                w.f.iter()
                    .zip(previous)
                    .map(|(a, b)| (a - b) * (a - b))
                    .sum::<f64>()
                    .sqrt();
            if change <= TOLERANCE * dot(&w.f, &w.f).sqrt() {
                return;
            }
        }
    }

    w.u.clear();
    let stored = &bodies.stress[i].displacement;
    if stored.len() == dofs {
        w.u.extend_from_slice(stored);
    } else {
        w.u.resize(dofs, 0.0);
    }
    let tolerance = if impact { IMPACT_TOLERANCE } else { TOLERANCE };
    let mut iterations = match factor_index {
        Some(k) => {
            w.factors[k].solve(&w.f, &mut w.u);
            project(&w.modes, &mut w.u);
            0
        }
        None => conjugate_gradients(
            ke,
            cell,
            &w.nodes,
            elements,
            &w.diag,
            &w.modes,
            &w.f,
            &mut w.u,
            [&mut w.r, &mut w.z, &mut w.p, &mut w.ap],
            tolerance,
        ),
    };
    bodies.stress[i].displacement.clone_from(&w.u);
    // An impact's stress is this step's alone, so the next step must solve
    // again rather than keep it.
    if impact {
        bodies.stress[i].load.clear();
    } else {
        bodies.stress[i].load.clone_from(&w.f);
    }

    // ---- the impact part, over the time it really lasts ----
    if impact {
        w.g.clear();
        w.g.resize(dofs, 0.0);
        load_vector(
            &grid,
            &w.surface,
            |l| l.impulse,
            &w.elements,
            &element_of,
            &w.nodes,
            mass,
            inertia,
            None,
            &mut w.g,
        );
        project(&w.modes, &mut w.g);
        w.v.clear();
        let previous = &bodies.stress[i].impact_displacement;
        if previous.len() == dofs {
            w.v.extend_from_slice(previous);
        } else {
            w.v.resize(dofs, 0.0);
        }
        iterations += match factor_index {
            Some(k) => {
                w.factors[k].solve(&w.g, &mut w.v);
                project(&w.modes, &mut w.v);
                0
            }
            None => conjugate_gradients(
                ke,
                cell,
                &w.nodes,
                elements,
                &w.diag,
                &w.modes,
                &w.g,
                &mut w.v,
                [&mut w.r, &mut w.z, &mut w.p, &mut w.ap],
                IMPACT_TOLERANCE,
            ),
        };
        bodies.stress[i].impact_displacement.clone_from(&w.v);
        // How hard the impact loads the body. The shape is `v`; its size is
        // set by the energy it has to store, `E W` for strain energy
        // `k² (v · g) / 2E` — then held between the two limits a contact
        // cannot pass: no gentler than the solver's own impulse spread over
        // the step, and no harder than that impulse over a stress wave's
        // round trip, which is what decides it for a compact body struck
        // end-on.
        let energy: f64 = w.surface.iter().map(|l| l.energy as f64).sum();
        let work = dot(&w.v, &w.g);
        let wave = w
            .surface
            .iter()
            .filter(|l| l.impulse != Vec3::ZERO)
            .map(|l| l.wave)
            .fold(0.0f32, f32::max)
            .clamp(1e-7, dt);
        let stored = if work > 0.0 && energy > 0.0 {
            (2.0 * energy * e_modulus / work).sqrt()
        } else {
            0.0
        };
        let k = stored.clamp(1.0 / dt as f64, 1.0 / wave as f64);

        for (u, v) in w.u.iter_mut().zip(&w.v) {
            *u += k * v;
        }
    }
    w.iterations = w.iterations.max(iterations);
    w.solved += 1;

    // ---- stress at each element's centre, and its worst sample point ----
    //
    // The centre is the most accurate point of a trilinear element and is
    // what gets drawn. Failure is judged at the eight quadrature points as
    // well, which sit a fifth of a cell from each corner: in a bent beam the
    // stress peaks at the surface, and the centre of the outermost cell
    // understates it.
    let g = 0.5 / 3f64.sqrt();
    let mut samples = [[0.0; 3]; 9];
    samples[0] = [0.5; 3];
    for (k, sample) in samples[1..].iter_mut().enumerate() {
        *sample = CORNERS[k].map(|c| if c == 1 { 0.5 + g } else { 0.5 - g });
    }
    let sample_grads: [[[f64; 3]; 8]; 9] =
        std::array::from_fn(|q| std::array::from_fn(|a| scaled(gradient(a, samples[q]), size)));
    let normal_thermal = e_modulus / (1.0 - 2.0 * nu);
    let mut peak = 0.0f32;
    let mut most = 0.0f32;
    let mut worst = Worst::default();
    let grain = bodies.grain[i];
    let grained = strength.grain.is_some();
    // Cells a contact loads directly are drawn but do not decide failure.
    // The stress under a contact at this resolution is the load over one
    // cell's face, and a real contact spreads it — a steel corner dents, a
    // concrete one chips — over an area the grid cannot see: put a 63 kg
    // steel cube's 30 cm corner landing on one 2.5 cm cell and it reads
    // 800 MPa and snaps. Saint-Venant's principle is the engineering answer
    // and this is it: judge the body away from where loads go in, where its
    // stress is the body's and not the contact's.
    let mut contact = vec![false; grid.len()];
    for load in &w.surface {
        contact[load.cell as usize] = true;
    }
    let mut cell_worst = vec![0.0f32; elements];
    let mut allowed_of = Vec::with_capacity(elements);
    {
        let body_stress = &mut bodies.stress[i];
        for (e, &c) in w.elements.iter().enumerate() {
            let en = &w.nodes[8 * e..8 * e + 8];
            let mut ue = [0.0; 24];
            for (a, &node) in en.iter().enumerate() {
                let base = 3 * node as usize;
                ue[3 * a..3 * a + 3].copy_from_slice(&w.u[base..base + 3]);
            }
            let t = if thermal {
                normal_thermal * expansion * (temperature[c] - mean_temperature)
            } else {
                0.0
            };
            let allowed = strength.at_temperature(&material.thermal, temperature[c] as f32);
            allowed_of.push(allowed);
            for (q, grads) in sample_grads.iter().enumerate() {
                let mut s = hooke(nu, strain(grads, &ue));
                for v in &mut s[..3] {
                    *v -= t;
                }
                let s = s.map(|v| v as f32);
                let vm = fusion_core::stress::von_mises(s);
                if q == 0 {
                    body_stress.stress[c] = s;
                    peak = peak.max(vm);
                }
                if contact[c] {
                    continue;
                }
                let at = Vec3::new(
                    samples[q][0] as f32,
                    samples[q][1] as f32,
                    samples[q][2] as f32,
                );
                let point =
                    grid.cell_centre(c) + (at - Vec3::splat(0.5)).mul_element_wise(grid.size);
                if grained {
                    // Tension along the grain only, here; the rest below.
                    let a = allowed.assess_parts(s, [0.0; 6], grain);
                    most = most.max(a.utilisation);
                    cell_worst[e] = cell_worst[e].max(a.fracture);
                    worst.consider(a, c, point, s);
                } else {
                    let principal = fusion_core::stress::principal_values(s);
                    let fracture = allowed.fracture_ratio(principal, vm);
                    most = most.max(allowed.utilisation(principal, vm));
                    cell_worst[e] = cell_worst[e].max(fracture);
                    // Where it is nearest parting is where it will break,
                    // which for a ductile metal is not always where it is
                    // worked hardest.
                    if fracture > worst.fracture {
                        worst.consider(allowed.assess(s, grain), c, point, s);
                    }
                }
            }
        }
    }
    // Everything but tension along the grain, judged over a neighbourhood.
    // Wood is fifteen times weaker across its grain than along it, and next
    // to a contact — a ball landing on a plank — the stress across it at this
    // resolution is the contact's, not the plank's, for a cell or two further
    // than the cells it touches. Crushing along it is the same: a steel ball
    // on a block's end grain crushes the fibres under it, which is a dent. Judged cell by cell, a plank struck in the middle split
    // into twenty slats when the bending that actually breaks it was
    // fourteen times nearer its strength. Averaged over the cells around
    // each, the way nonlocal damage models judge a crack by the stress over
    // a length of material rather than at a point, a split has to be one
    // the whole section is carrying.
    if grained {
        let body_stress = &bodies.stress[i];
        for (e, &c) in w.elements.iter().enumerate() {
            if contact[c] {
                continue;
            }
            let [x, y, z] = grid.coordinates(c);
            let mut sum = [0.0f32; 6];
            let mut count = 0.0f32;
            for dz in -1i64..=1 {
                for dy in -1i64..=1 {
                    for dx in -1i64..=1 {
                        let (nx, ny, nz) = (x as i64 + dx, y as i64 + dy, z as i64 + dz);
                        if nx < 0 || ny < 0 || nz < 0 {
                            continue;
                        }
                        let (nx, ny, nz) = (nx as usize, ny as usize, nz as usize);
                        if nx >= grid.dims[0] || ny >= grid.dims[1] || nz >= grid.dims[2] {
                            continue;
                        }
                        let n = nx + grid.dims[0] * (ny + grid.dims[1] * nz);
                        if !grid.inside[n] || contact[n] {
                            continue;
                        }
                        for k in 0..6 {
                            sum[k] += body_stress.stress[n][k];
                        }
                        count += 1.0;
                    }
                }
            }
            let smoothed = sum.map(|v| v / count.max(1.0));
            let a = allowed_of[e].assess_parts([0.0; 6], smoothed, grain);
            most = most.max(a.utilisation);
            cell_worst[e] = cell_worst[e].max(a.fracture);
            worst.consider(a, c, grid.cell_centre(c), smoothed);
        }
    }
    let overloaded = cell_worst.iter().filter(|&&u| u >= 1.0).count();
    let body_stress = &mut bodies.stress[i];
    body_stress.overloaded = overloaded as f32 / elements.max(1) as f32;
    body_stress.peak_von_mises = peak;
    body_stress.utilisation = most;
    body_stress.worst = worst.cell;
    body_stress.worst_point = worst.point;
    body_stress.worst_stress = worst.stress;
    body_stress.fracture = worst.fracture;
    body_stress.worst_normal = worst.normal;
    body_stress.worst_parting = worst.parting;
}

/// The point nearest parting, while a body's cells are judged.
#[derive(Default)]
struct Worst {
    fracture: f32,
    cell: usize,
    point: Vec3,
    stress: [f32; 6],
    normal: Vec3,
    parting: fusion_core::Parting,
}

impl Worst {
    fn consider(&mut self, a: fusion_core::Assessment, cell: usize, point: Vec3, stress: [f32; 6]) {
        if a.fracture > self.fracture {
            *self = Self {
                fracture: a.fracture,
                cell,
                point,
                stress,
                normal: a.normal,
                parting: a.mode,
            };
        }
    }
}
