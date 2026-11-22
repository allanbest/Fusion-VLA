//! Heat in rigid bodies: where it comes from, how it moves, where it goes.
//!
//! The continuum has had a temperature since v0.9, and it was the only thing
//! that did — a steel block dragged across a floor stayed at room temperature
//! however far it slid, because a rigid body had nowhere to keep heat. This is
//! that place, and the three things that change it:
//!
//! 1. **Sources.** Every joule of kinetic energy a contact removes — sliding
//!    friction, an impact that does not bounce all the way back — is deposited
//!    as heat at the contact, split between the two bodies by thermal
//!    effusivity `sqrt(k rho c)`, which is how the heat of a sliding contact
//!    divides between two semi-infinite solids. Copper against wood takes
//!    almost all of it into the copper, which is why a wooden match lights and
//!    a copper one would not.
//! 2. **Conduction**, inside a body on its grid and between bodies across
//!    every contact that is actually pressed together.
//! 3. **Losses** to the air from every exposed face: convection, and
//!    radiation once something is hot enough for it to matter.
//!
//! Off by default. A scene that has not asked for heat pays one `f64` per
//! body and one branch per step.
//!
//! **What it is not.** The mechanics stays rigid: nothing expands, softens or
//! melts yet, and a body's temperature does not change how it moves. Energy
//! removed by joints, air drag and the solver's own damping is not counted as
//! heat — only contacts are — so the books balance for a scene whose losses
//! are all contacts, and are short otherwise.

use fusion_core::{BodyHeat, BodySet, Voxels};
use fusion_math::Vec3;
use fusion_solver::ContactReport;

/// Stefan–Boltzmann constant, W/(m^2 K^4).
pub const STEFAN_BOLTZMANN: f64 = 5.670_374_419e-8;

/// How the heat solve treats a scene.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct HeatSettings {
    /// Whether rigid bodies carry a temperature field at all.
    pub enabled: bool,
    /// Temperature of the surrounding air, Kelvin.
    pub air_temperature: f32,
    /// Convective heat-transfer coefficient to the air, W/(m^2 K).
    ///
    /// Ten is still air around something warm; a breeze is twenty-five to
    /// fifty, a fan a hundred. Zero isolates the scene from the air, which is
    /// what an energy-balance test wants.
    pub convection: f32,
    /// Whether surfaces radiate, by their material's emissivity.
    pub radiation: bool,
    /// Conductance of the interface between two bodies in contact,
    /// W/(m^2 K).
    ///
    /// Two real surfaces touch at their asperities, not over their whole
    /// face, and the gap between those is what limits heat flow across a
    /// contact. A few thousand is typical of machined metal on metal under
    /// light load; polished and clamped is tens of thousands.
    pub contact_conductance: f32,
}

impl Default for HeatSettings {
    fn default() -> Self {
        Self {
            enabled: false,
            air_temperature: fusion_core::solid::AMBIENT,
            convection: 10.0,
            radiation: true,
            contact_conductance: 5000.0,
        }
    }
}

impl HeatSettings {
    /// Heat on, with the defaults: still room-temperature air, radiation.
    pub fn on() -> Self {
        Self {
            enabled: true,
            ..Self::default()
        }
    }

    /// Heat on and nothing lost to the air — the setting under which every
    /// joule of heat in the scene is accounted for.
    pub fn insulated() -> Self {
        Self {
            enabled: true,
            convection: 0.0,
            radiation: false,
            ..Self::default()
        }
    }
}

/// Scratch for the heat solve. Holds nothing a checkpoint needs.
#[derive(Clone, Debug, Default)]
pub struct HeatWorking {
    /// Per-body: whether anything was deposited this step.
    touched: Vec<bool>,
    patch: Vec<usize>,
    compact: Vec<u32>,
    cells: Vec<u32>,
    exposure: Vec<f64>,
    loss: Vec<f64>,
    rhs: Vec<f64>,
    x: Vec<f64>,
    r: Vec<f64>,
    z: Vec<f64>,
    p: Vec<f64>,
    ap: Vec<f64>,
    diag: Vec<f64>,
    /// Conjugate-gradient iterations the last step took, summed over bodies.
    pub iterations: u32,
}

/// The mass a body's heat capacity is built from.
///
/// The real mass for anything the solver moves, and density times volume for
/// the rest — a static floor has no inverse mass to invert, but it is still
/// two tonnes of concrete to heat.
pub fn thermal_mass(bodies: &BodySet, i: usize) -> f64 {
    let inv = bodies.inv_mass[i];
    if inv > 0.0 {
        1.0 / inv as f64
    } else {
        bodies.material[i].density as f64 * bodies.shape[i].volume() as f64
    }
}

/// Heat capacity of one inside cell of body `i`, J/K.
pub fn cell_capacity(bodies: &BodySet, i: usize) -> f64 {
    let heat = &bodies.heat[i];
    let n = heat.inside_count().max(1) as f64;
    thermal_mass(bodies, i) * bodies.material[i].thermal.specific_heat as f64 / n
}

/// Heat held by every rigid body, relative to absolute zero, joules.
///
/// Sensible heat only, with constant specific heat. The number to difference:
/// its change over a run is what the run turned into heat.
pub fn thermal_energy(bodies: &BodySet) -> f64 {
    (0..bodies.len())
        .map(|i| {
            let heat = &bodies.heat[i];
            if heat.is_built() {
                let c = cell_capacity(bodies, i);
                heat.temperature
                    .iter()
                    .zip(&heat.grid.inside)
                    .filter(|(_, &inside)| inside)
                    .map(|(&t, _)| t * c)
                    .sum::<f64>()
            } else {
                heat.uniform
                    * thermal_mass(bodies, i)
                    * bodies.material[i].thermal.specific_heat as f64
            }
        })
        .sum()
}

fn effusivity(bodies: &BodySet, i: usize) -> f64 {
    let t = &bodies.material[i].thermal;
    (t.conductivity as f64 * bodies.material[i].density as f64 * t.specific_heat as f64)
        .max(0.0)
        .sqrt()
}

/// Area of the patch a manifold's points span, m^2: the convex hull of up to
/// four coplanar points, which for four is the largest of the three ways to
/// pair their diagonals.
fn patch_area(points: &[Vec3]) -> f32 {
    match points.len() {
        3 => {
            0.5 * (points[1] - points[0])
                .cross(points[2] - points[0])
                .length()
        }
        4 => {
            let quad = |a: usize, b: usize, c: usize, d: usize| {
                0.5 * (points[c] - points[a])
                    .cross(points[d] - points[b])
                    .length()
            };
            quad(0, 1, 2, 3).max(quad(0, 1, 3, 2)).max(quad(0, 2, 1, 3))
        }
        _ => 0.0,
    }
}

/// Advance every rigid body's temperature by `dt`.
///
/// `reports` are the contacts of the step that just ran. Bodies are given a
/// grid the first time this runs; see [`BodyHeat::build`].
pub fn step(
    bodies: &mut BodySet,
    reports: &[ContactReport],
    settings: &HeatSettings,
    working: &mut HeatWorking,
    dt: f32,
) {
    if !settings.enabled || dt <= 0.0 {
        return;
    }
    let n = bodies.len();
    // Only Advanced bodies carry a temperature grid. A Basic body never gets
    // one, so contacts with it deposit and conduct nothing into it, and it
    // stays at the temperature it started at — the whole cost of heat for it
    // is this test.
    for i in 0..n {
        if !bodies.advanced[i] {
            continue;
        }
        let shape = bodies.shape[i].clone();
        bodies.heat[i].build(&shape);
    }
    working.touched.clear();
    working.touched.resize(n, false);
    let dt = dt as f64;

    // ---- sources and conduction, one manifold at a time ----
    let conductance = settings.contact_conductance.max(0.0) as f64;
    let mut start = 0;
    while start < reports.len() {
        let pair = (reports[start].body_a, reports[start].body_b);
        let mut end = start;
        while end < reports.len() && (reports[end].body_a, reports[end].body_b) == pair {
            end += 1;
        }
        let group = &reports[start..end];
        start = end;
        let (a, b) = (pair.0 as usize, pair.1 as usize);
        deposit(bodies, a, b, group, working);
        if conductance > 0.0 {
            conduct(bodies, a, b, group, conductance, working, dt);
        }
    }

    // ---- inside each body, and out to the air ----
    working.iterations = 0;
    for i in 0..n {
        solve_body(bodies, i, settings, working, dt);
    }
}

/// The two bodies' outward normals at a contact, each in its own frame.
pub(crate) fn normals(
    bodies: &BodySet,
    a: usize,
    b: usize,
    report: &ContactReport,
) -> (Vec3, Vec3) {
    let world = bodies.orientation[a].rotate(report.local_normal_a);
    (
        report.local_normal_a,
        -bodies.orientation[b].rotate_inverse(world),
    )
}

/// Whether `p` lies in the triangle `abc`, all in one plane.
fn in_triangle(p: Vec3, a: Vec3, b: Vec3, c: Vec3) -> bool {
    let t = (b - a).cross(c - a);
    if t.length_squared() < 1e-12 {
        return false;
    }
    (b - a).cross(p - a).dot(t) >= 0.0
        && (c - b).cross(p - b).dot(t) >= 0.0
        && (a - c).cross(p - c).dot(t) >= 0.0
}

/// The cells of a body that lie under a contact patch.
///
/// A manifold reports the *corners* of the region two bodies share — four
/// points for a box face on a floor — and putting all of a slide's heat into
/// four corner cells paints four hot spots on a face that is, in fact,
/// uniformly rubbed. So where the points span an area, every surface cell
/// under that area takes its share: the cells whose centres are within one
/// cell beneath the contact plane and inside the points' convex hull (pulled
/// half a cell towards its middle, so the edge cells whose centres sit just
/// inside a corner are not lost to rounding). Fewer than three points, or a
/// patch smaller than a cell, falls back to the cell under each point.
pub(crate) fn patch_cells(grid: &Voxels, points: &[Vec3], outward: Vec3, out: &mut Vec<usize>) {
    out.clear();
    if !grid.is_built() || points.is_empty() {
        return;
    }
    // The cell's depth along the contact normal, and its breadth across it:
    // the same for a grid of cubes, and not for a stress grid of thin layers.
    let depth_cell = grid.depth_along(outward);
    let cell = grid.cell;
    if points.len() >= 3 && patch_area(points) >= cell * cell {
        let plane = points.iter().map(|p| p.dot(outward)).sum::<f32>() / points.len() as f32;
        let middle = points.iter().copied().fold(Vec3::ZERO, |s, p| s + p) / points.len() as f32;
        for (i, &inside) in grid.inside.iter().enumerate() {
            if !inside {
                continue;
            }
            let c = grid.cell_centre(i);
            let depth = plane - c.dot(outward);
            if !(-0.5 * depth_cell..=depth_cell).contains(&depth) {
                continue;
            }
            let on_plane = c + outward * depth;
            let towards = middle - on_plane;
            let reach = towards.length();
            let q = if reach > 0.0 {
                on_plane + towards * ((0.5 * cell).min(reach) / reach)
            } else {
                on_plane
            };
            let n = points.len();
            let hit = (0..n).any(|x| {
                (x + 1..n)
                    .any(|y| (y + 1..n).any(|z| in_triangle(q, points[x], points[y], points[z])))
            });
            if hit {
                out.push(i);
            }
        }
    }
    if out.is_empty() {
        for p in points {
            if let Some(i) = grid.nearest_inside(*p) {
                if !out.contains(&i) {
                    out.push(i);
                }
            }
        }
    }
}

/// Put what a manifold dissipated into the two bodies, divided by effusivity
/// and spread over each one's side of the patch.
fn deposit(bodies: &mut BodySet, a: usize, b: usize, group: &[ContactReport], w: &mut HeatWorking) {
    let q: f64 = group
        .iter()
        .map(|r| r.dissipated as f64)
        .filter(|q| q.is_finite() && *q > 0.0)
        .sum();
    if q <= 0.0 {
        return;
    }
    let (ea, eb) = (effusivity(bodies, a), effusivity(bodies, b));
    let share_a = if ea + eb > 0.0 { ea / (ea + eb) } else { 0.5 };
    let (na, nb) = normals(bodies, a, b, &group[0]);
    for (body, share, outward, on_a) in [(a, share_a, na, true), (b, 1.0 - share_a, nb, false)] {
        if share <= 0.0 {
            continue;
        }
        let points: Vec<Vec3> = group
            .iter()
            .map(|r| if on_a { r.local_a } else { r.local_b })
            .collect();
        let mut cells = std::mem::take(&mut w.patch);
        patch_cells(&bodies.heat[body].grid, &points, outward, &mut cells);
        if !cells.is_empty() {
            let per = q * share / (cell_capacity(bodies, body) * cells.len() as f64);
            let heat = &mut bodies.heat[body];
            for &i in &cells {
                heat.temperature[i] += per;
            }
            w.touched[body] = true;
        }
        w.patch = cells;
    }
}

/// Heat across a contact, cell to facing cell.
///
/// Explicit, pair by pair, as an exact two-cell relaxation: each exchange
/// moves the two cells towards their common temperature by
/// `1 - exp(-G dt / C_eff)` of the gap, which cannot overshoot however stiff
/// the contact or long the step. The interior solve is implicit; the contacts
/// are not, because they would couple every body in a pile into one system.
///
/// The conductance of each pair is half a cell of each body in series with
/// the interface, over the patch's area shared out among the cells under it.
/// A single point or an edge spans no area — the real patch under it is set
/// by elastic flattening the rigid model does not have — and a quarter-cell
/// square stands in for it.
fn conduct(
    bodies: &mut BodySet,
    a: usize,
    b: usize,
    group: &[ContactReport],
    conductance: f64,
    w: &mut HeatWorking,
    dt: f64,
) {
    let touching: Vec<&ContactReport> = group.iter().filter(|r| r.touching).collect();
    if touching.is_empty() || !bodies.heat[a].is_built() || !bodies.heat[b].is_built() {
        return;
    }
    // Walk the finer grid's side, so every cell of it gets a partner.
    let (from, to) = if bodies.heat[a].grid.cell <= bodies.heat[b].grid.cell {
        (a, b)
    } else {
        (b, a)
    };
    let (na, nb) = normals(bodies, a, b, &group[0]);
    let outward = if from == a { na } else { nb };
    let points: Vec<Vec3> = touching
        .iter()
        .map(|r| if from == a { r.local_a } else { r.local_b })
        .collect();
    let mut cells = std::mem::take(&mut w.patch);
    patch_cells(&bodies.heat[from].grid, &points, outward, &mut cells);
    if cells.is_empty() {
        w.patch = cells;
        return;
    }
    let cell = bodies.heat[from].grid.cell.min(bodies.heat[to].grid.cell);
    let floor = (0.25 * cell).powi(2) * points.len() as f32;
    let area = patch_area(&points).max(floor) as f64 / cells.len() as f64;
    let kf = bodies.material[from].thermal.conductivity.max(1e-6) as f64;
    let kt = bodies.material[to].thermal.conductivity.max(1e-6) as f64;
    let resistance = 0.5 * bodies.heat[from].grid.cell as f64 / kf
        + 0.5 * bodies.heat[to].grid.cell as f64 / kt
        + 1.0 / conductance;
    let g = area / resistance;
    let (cf, ct) = (cell_capacity(bodies, from), cell_capacity(bodies, to));
    let c_eff = cf * ct / (cf + ct);
    let fraction = 1.0 - (-g * dt / c_eff).exp();
    let plane = points.iter().map(|p| p.dot(outward)).sum::<f32>() / points.len() as f32;
    let (pose_from, pose_to) = (bodies.pose(from), bodies.pose(to));
    for &i in &cells {
        let c = bodies.heat[from].cell_centre(i);
        let surface = c + outward * (plane - c.dot(outward));
        let there = pose_to.inverse_transform_point(pose_from.transform_point(surface));
        let Some(j) = bodies.heat[to].nearest_inside(there) else {
            continue;
        };
        let (tf, tt) = (
            bodies.heat[from].temperature[i],
            bodies.heat[to].temperature[j],
        );
        if tf == tt {
            continue;
        }
        let q = (tt - tf) * c_eff * fraction;
        bodies.heat[from].temperature[i] += q / cf;
        bodies.heat[to].temperature[j] -= q / ct;
        w.touched[from] = true;
        w.touched[to] = true;
    }
    w.patch = cells;
}

/// One body's interior conduction and surface losses, implicit (backward
/// Euler) and solved by Jacobi-preconditioned conjugate gradients in double.
///
/// Implicit because the explicit limit is brutal for a good conductor on a
/// coarse grid: aluminium in 2.5 cm cells would want a quarter-second step,
/// and a copper heat sink in 1 cm cells several hundred steps per second.
/// Backward Euler has no limit and errs towards smoothing, which is the
/// physically safe direction.
fn solve_body(
    bodies: &mut BodySet,
    i: usize,
    settings: &HeatSettings,
    w: &mut HeatWorking,
    dt: f64,
) {
    let air = settings.air_temperature as f64;
    let h_conv = settings.convection.max(0.0) as f64;
    let thermal = bodies.material[i].thermal;
    let emissivity = if settings.radiation {
        thermal.emissivity.clamp(0.0, 1.0) as f64
    } else {
        0.0
    };
    let loses = h_conv > 0.0 || emissivity > 0.0;
    let c = cell_capacity(bodies, i);
    let true_area = bodies.shape[i].surface_area() as f64;
    let heat: &mut BodyHeat = &mut bodies.heat[i];
    if !heat.is_built() {
        return;
    }
    // Nothing to do for a body that is uniform and either at the air's
    // temperature or not losing heat to it: every term below would be zero.
    let (lo, hi) = heat.range();
    if !w.touched[i] && hi - lo <= 1e-12 * hi.abs().max(1.0) && (!loses || (hi - air).abs() <= 1e-9)
    {
        return;
    }

    let [nx, ny, nz] = heat.grid.dims;
    w.compact.clear();
    w.compact.resize(heat.len(), u32::MAX);
    w.cells.clear();
    for (cell, &inside) in heat.grid.inside.iter().enumerate() {
        if inside {
            w.compact[cell] = w.cells.len() as u32;
            w.cells.push(cell as u32);
        }
    }
    let m = w.cells.len();
    let face = (heat.grid.cell as f64).powi(2);
    let g = thermal.conductivity.max(0.0) as f64 * heat.grid.cell as f64;

    // Exposed faces per cell, scaled so they add up to the shape's true
    // surface area: a voxelized sphere has half as much face again as the
    // sphere it stands for.
    let neighbours = |cell: usize| -> [Option<usize>; 6] {
        let x = cell % nx;
        let y = (cell / nx) % ny;
        let z = cell / (nx * ny);
        [
            (x > 0).then(|| cell - 1),
            (x + 1 < nx).then(|| cell + 1),
            (y > 0).then(|| cell - nx),
            (y + 1 < ny).then(|| cell + nx),
            (z > 0).then(|| cell - nx * ny),
            (z + 1 < nz).then(|| cell + nx * ny),
        ]
    };
    w.exposure.clear();
    w.exposure.resize(m, 0.0);
    let mut voxel_area = 0.0;
    for (k, &cell) in w.cells.iter().enumerate() {
        let open = neighbours(cell as usize)
            .iter()
            .filter(|n| n.is_none_or(|j| !heat.grid.inside[j]))
            .count();
        w.exposure[k] = open as f64 * face;
        voxel_area += w.exposure[k];
    }
    let scale = if voxel_area > 0.0 {
        true_area / voxel_area
    } else {
        0.0
    };

    // Loss coefficient per cell, W/K: convection plus radiation linearized
    // about the current temperature, `eps sigma (T^2 + Ta^2)(T + Ta)`. Lagged
    // one step, which is the usual price of keeping the system linear.
    w.loss.clear();
    w.diag.clear();
    w.rhs.clear();
    w.x.clear();
    for (k, &cell) in w.cells.iter().enumerate() {
        let t = heat.temperature[cell as usize];
        let h_rad = emissivity * STEFAN_BOLTZMANN * (t * t + air * air) * (t + air);
        let loss = (h_conv + h_rad) * w.exposure[k] * scale;
        let links = neighbours(cell as usize)
            .iter()
            .filter(|n| n.is_some_and(|j| heat.grid.inside[j]))
            .count() as f64;
        w.loss.push(loss);
        w.diag.push(c / dt + loss + g * links);
        w.rhs.push(c / dt * t + loss * air);
        w.x.push(t);
    }

    let apply = |x: &[f64], out: &mut Vec<f64>, cells: &[u32], compact: &[u32], diag: &[f64]| {
        out.clear();
        for (k, &cell) in cells.iter().enumerate() {
            let mut v = diag[k] * x[k];
            for j in neighbours(cell as usize).into_iter().flatten() {
                let jj = compact[j];
                if jj != u32::MAX {
                    v -= g * x[jj as usize];
                }
            }
            out.push(v);
        }
    };

    // r = b - A x
    apply(&w.x, &mut w.ap, &w.cells, &w.compact, &w.diag);
    w.r.clear();
    w.r.extend(w.rhs.iter().zip(&w.ap).map(|(b, ax)| b - ax));
    w.z.clear();
    w.z.extend(w.r.iter().zip(&w.diag).map(|(r, d)| r / d));
    w.p.clone_from(&w.z);
    let norm_b = w.rhs.iter().map(|b| b * b).sum::<f64>().sqrt().max(1e-300);
    let mut rz: f64 = w.r.iter().zip(&w.z).map(|(r, z)| r * z).sum();
    let limit = (4 * m).clamp(16, 500);
    for _ in 0..limit {
        let residual = w.r.iter().map(|r| r * r).sum::<f64>().sqrt();
        if residual <= 1e-13 * norm_b {
            break;
        }
        apply(&w.p, &mut w.ap, &w.cells, &w.compact, &w.diag);
        let pap: f64 = w.p.iter().zip(&w.ap).map(|(p, ap)| p * ap).sum();
        if pap <= 0.0 {
            break;
        }
        let alpha = rz / pap;
        for k in 0..m {
            w.x[k] += alpha * w.p[k];
            w.r[k] -= alpha * w.ap[k];
            w.z[k] = w.r[k] / w.diag[k];
        }
        let rz_next: f64 = w.r.iter().zip(&w.z).map(|(r, z)| r * z).sum();
        let beta = rz_next / rz;
        rz = rz_next;
        for k in 0..m {
            w.p[k] = w.z[k] + beta * w.p[k];
        }
        w.iterations += 1;
    }
    for (k, &cell) in w.cells.iter().enumerate() {
        heat.temperature[cell as usize] = w.x[k];
    }
    heat.uniform = heat.mean();
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_square_patch_has_its_area_in_any_order() {
        let square = [
            Vec3::new(0.0, 0.0, 0.0),
            Vec3::new(1.0, 0.0, 1.0),
            Vec3::new(1.0, 0.0, 0.0),
            Vec3::new(0.0, 0.0, 1.0),
        ];
        assert!((patch_area(&square) - 1.0).abs() < 1e-6);
    }
}
