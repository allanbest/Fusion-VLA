//! Finite-element heat conduction on tetrahedral solids with deterministic
//! parallel Jacobi-preconditioned conjugate gradients.

use core_math::constants::STEFAN_BOLTZMANN;
use core_math::{Mat3, Vec3};
use core_runtime::Pool;
use core_world::{Column, ObjectId, Representation, TetMesh, Thermal, World, EDGES};

use crate::Process;

const CHUNK: usize = 1024;
const ROWS: usize = 64;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ThermalIntegrator {
    BackwardEuler,
    CrankNicolson,
}

#[derive(Clone, Debug, PartialEq)]
pub enum ThermalBoundary {
    Dirichlet {
        object: ObjectId,
        nodes: Vec<u32>,
        temperature: f64,
    },
    VolumetricSource {
        object: ObjectId,
        /// W/m³.
        rate: f64,
    },
    Convection {
        object: ObjectId,
        /// W/(m²·K).
        coefficient: f64,
        /// Ambient temperature, K.
        ambient: f64,
    },
    Radiation {
        object: ObjectId,
        /// Ambient radiative temperature, K.
        ambient: f64,
    },
    /// Explicit nodal heat power source (W = J/s per mesh node), e.g. from
    /// mechanical frictional/plastic/damage dissipation.
    NodalPower { object: ObjectId, power: Vec<f64> },
}

#[derive(Clone, Debug, PartialEq)]
pub struct ThermalSettings {
    pub integrator: ThermalIntegrator,
    pub tolerance: f64,
    pub max_iterations: usize,
    pub boundaries: Vec<ThermalBoundary>,
}

impl Default for ThermalSettings {
    fn default() -> Self {
        Self {
            integrator: ThermalIntegrator::BackwardEuler,
            tolerance: 1e-10,
            max_iterations: 1000,
            boundaries: Vec::new(),
        }
    }
}

#[derive(Clone, Copy, Debug, Default)]
pub struct ThermalReport {
    pub iterations: usize,
    pub converged: bool,
    pub min_temperature: f64,
    pub max_temperature: f64,
    pub thermal_energy: f64,
}

#[derive(Clone, Debug, Default)]
struct ScalarCsr {
    n: usize,
    row_start: Vec<usize>,
    cols: Vec<usize>,
    values: Vec<f64>,
    diag_slot: Vec<usize>,
}

impl ScalarCsr {
    fn from_mesh(mesh: &TetMesh) -> Self {
        let n = mesh.rest.len();
        let mut adj: Vec<Vec<usize>> = (0..n).map(|i| vec![i]).collect();
        for e in 0..mesh.tets.len() {
            let elem = mesh.element(e);
            for &a in &elem {
                for &b in &elem {
                    adj[a as usize].push(b as usize);
                }
            }
        }
        let mut row_start = Vec::with_capacity(n + 1);
        let mut cols = Vec::new();
        let mut diag_slot = Vec::with_capacity(n);
        row_start.push(0);
        for row in &mut adj {
            row.sort_unstable();
            row.dedup();
            let first = cols.len();
            let pos = row.binary_search(&row_start.len().saturating_sub(1));
            cols.extend_from_slice(row);
            diag_slot.push(first + pos.expect("diagonal always present"));
            row_start.push(cols.len());
        }
        let values = vec![0.0; cols.len()];
        Self {
            n,
            row_start,
            cols,
            values,
            diag_slot,
        }
    }

    #[inline]
    fn slot(&self, r: usize, c: usize) -> usize {
        let start = self.row_start[r];
        let end = self.row_start[r + 1];
        let rel = self.cols[start..end]
            .binary_search(&c)
            .expect("edge present in sparsity pattern");
        start + rel
    }

    fn spmv(&self, pool: &Pool, x: &[f64], y: &mut [f64]) {
        pool.for_chunks_mut(y, ROWS, |offset, part| {
            for (k, out) in part.iter_mut().enumerate() {
                let row = offset + k;
                let mut sum = 0.0;
                for idx in self.row_start[row]..self.row_start[row + 1] {
                    sum += self.values[idx] * x[self.cols[idx]];
                }
                *out = sum;
            }
        });
    }
}

#[derive(Clone, Debug)]
struct SolidThermal {
    object: usize,
    thermal: Thermal,
    density: f64,
    capacity: Vec<f64>,
    matrix_k: ScalarCsr,
    matrix_m: ScalarCsr,
    elem_slots: Vec<Vec<usize>>,
}

#[derive(Clone, Debug, Default)]
pub struct HeatConduction {
    pub settings: ThermalSettings,
    solids: Vec<SolidThermal>,
    built_objects: usize,
}

impl HeatConduction {
    pub fn new(settings: ThermalSettings) -> Self {
        Self {
            settings,
            solids: Vec::new(),
            built_objects: usize::MAX,
        }
    }

    pub fn invalidate(&mut self) {
        self.built_objects = usize::MAX;
    }

    /// Replaces any existing nodal power source for `object` with `power` (W per node).
    pub fn set_nodal_power(&mut self, object: ObjectId, power: Vec<f64>) {
        if let Some(existing) =
            self.settings.boundaries.iter_mut().find(
                |b| matches!(b, ThermalBoundary::NodalPower { object: o, .. } if *o == object),
            )
        {
            *existing = ThermalBoundary::NodalPower { object, power };
        } else {
            self.settings
                .boundaries
                .push(ThermalBoundary::NodalPower { object, power });
        }
    }

    fn ensure_prepared(&mut self, pool: &Pool, world: &World) {
        if self.built_objects != world.objects().len() {
            self.prepare(pool, world);
        }
    }
}

fn dot_det(pool: &Pool, a: &[f64], b: &[f64]) -> f64 {
    pool.map_reduce(
        a.len(),
        CHUNK,
        0.0,
        |r| {
            a[r.clone()]
                .iter()
                .zip(&b[r])
                .map(|(&x, &y)| x * y)
                .sum::<f64>()
        },
        |x, y| x + y,
    )
}

fn linear_grads(rest: &[Vec3; 4]) -> (f64, [Vec3; 4]) {
    let ref_g = [Vec3::splat(-1.0), Vec3::X, Vec3::Y, Vec3::Z];
    let mut jac = Mat3::ZERO;
    for a in 0..4 {
        jac += Mat3::outer(rest[a], ref_g[a]);
    }
    let det = jac.determinant();
    let inv_t = jac.inverse().transpose();
    (det / 6.0, ref_g.map(|g| inv_t * g))
}

fn quadratic_grads(l: [f64; 4]) -> [Vec3; 10] {
    let dl = [Vec3::splat(-1.0), Vec3::X, Vec3::Y, Vec3::Z];
    let mut g = [Vec3::ZERO; 10];
    for i in 0..4 {
        g[i] = dl[i] * (4.0 * l[i] - 1.0);
    }
    for (k, (i, j)) in EDGES.into_iter().enumerate() {
        g[4 + k] = (dl[j] * l[i] + dl[i] * l[j]) * 4.0;
    }
    g
}

fn assemble_conductivity(st: &mut SolidThermal, pos: &[Vec3], rest: &[Vec3], mesh: &TetMesh) {
    st.matrix_k.values.fill(0.0);
    let cond = st.thermal.conductivity;
    if cond <= 0.0 || mesh.tets.is_empty() {
        return;
    }
    let k_nodes = if mesh.is_quadratic() { 10 } else { 4 };
    if mesh.is_quadratic() {
        let (a, b) = (0.585_410_196_624_968_5, 0.138_196_601_125_010_5);
        let rule = [[a, b, b, b], [b, a, b, b], [b, b, a, b], [b, b, b, a]];
        for (e, slots) in st.elem_slots.iter().enumerate() {
            let elem = mesh.element(e);
            let mut p10: [Vec3; 10] = std::array::from_fn(|k| pos[elem[k] as usize]);
            let j0 = Mat3::from_cols(p10[1] - p10[0], p10[2] - p10[0], p10[3] - p10[0]);
            if j0.determinant() <= 0.0 {
                p10 = std::array::from_fn(|k| rest[elem[k] as usize]);
            }
            for &l in &rule {
                let g_ref = quadratic_grads(l);
                let mut jac = Mat3::ZERO;
                for k in 0..10 {
                    jac += Mat3::outer(p10[k], g_ref[k]);
                }
                let det = jac.determinant();
                if det <= 0.0 {
                    continue;
                }
                let inv_t = jac.inverse().transpose();
                let gw = g_ref.map(|g| inv_t * g);
                let w = cond * det / 24.0;
                for row in 0..k_nodes {
                    for col in row..k_nodes {
                        let v = w * gw[row].dot(gw[col]);
                        let s_rc = slots[row * k_nodes + col];
                        st.matrix_k.values[s_rc] += v;
                        if row != col {
                            let s_cr = slots[col * k_nodes + row];
                            st.matrix_k.values[s_cr] += v;
                        }
                    }
                }
            }
        }
    } else {
        for (e, slots) in st.elem_slots.iter().enumerate() {
            let t = mesh.tets[e];
            let mut p4 = t.map(|k| pos[k as usize]);
            let (mut w, mut gw) = linear_grads(&p4);
            if w <= 0.0 {
                p4 = t.map(|k| rest[k as usize]);
                (w, gw) = linear_grads(&p4);
            }
            if w <= 0.0 {
                continue;
            }
            let kw = cond * w;
            for row in 0..4 {
                for col in row..4 {
                    let v = kw * gw[row].dot(gw[col]);
                    st.matrix_k.values[slots[row * 4 + col]] += v;
                    if row != col {
                        st.matrix_k.values[slots[col * 4 + row]] += v;
                    }
                }
            }
        }
    }
}

fn solve_cg(
    pool: &Pool,
    a: &ScalarCsr,
    b: &[f64],
    x: &mut [f64],
    tol: f64,
    max_iters: usize,
) -> (usize, bool) {
    let n = a.n;
    if n == 0 {
        return (0, true);
    }
    let inv_diag: Vec<f64> = a
        .diag_slot
        .iter()
        .map(|&s| {
            let d = a.values[s];
            if d > 0.0 && d.is_finite() {
                1.0 / d
            } else {
                1.0
            }
        })
        .collect();

    let b_norm = dot_det(pool, b, b).sqrt();
    if b_norm == 0.0 {
        x.fill(0.0);
        return (0, true);
    }

    let mut r = vec![0.0; n];
    let mut z = vec![0.0; n];
    let mut p = vec![0.0; n];
    let mut ap = vec![0.0; n];

    a.spmv(pool, x, &mut r);
    for i in 0..n {
        r[i] = b[i] - r[i];
        z[i] = inv_diag[i] * r[i];
        p[i] = z[i];
    }
    let mut rz = dot_det(pool, &r, &z);
    let mut rr = dot_det(pool, &r, &r);
    let target = tol * b_norm;
    let mut iterations = 0;

    while iterations < max_iters {
        if !rr.is_finite() || !rz.is_finite() {
            return (iterations, false);
        }
        if rr.sqrt() <= target {
            // Recompute true residual before declaring convergence.
            a.spmv(pool, x, &mut ap);
            for i in 0..n {
                r[i] = b[i] - ap[i];
            }
            rr = dot_det(pool, &r, &r);
            if rr.sqrt() <= target {
                return (iterations, true);
            }
            for i in 0..n {
                z[i] = inv_diag[i] * r[i];
                p[i] = z[i];
            }
            rz = dot_det(pool, &r, &z);
            continue;
        }
        a.spmv(pool, &p, &mut ap);
        let p_ap = dot_det(pool, &p, &ap);
        if p_ap <= 0.0 || !p_ap.is_finite() {
            return (iterations, false);
        }
        let alpha = rz / p_ap;
        for i in 0..n {
            x[i] += alpha * p[i];
            r[i] -= alpha * ap[i];
            z[i] = inv_diag[i] * r[i];
        }
        let rz_new = dot_det(pool, &r, &z);
        rr = dot_det(pool, &r, &r);
        let beta = if rz != 0.0 { rz_new / rz } else { 0.0 };
        for i in 0..n {
            p[i] = z[i] + beta * p[i];
        }
        rz = rz_new;
        iterations += 1;
    }
    (iterations, rr.sqrt() <= target)
}

impl Process for HeatConduction {
    type Report = ThermalReport;

    fn prepare(&mut self, _pool: &Pool, world: &World) {
        self.solids.clear();
        for (index, object) in world.objects().iter().enumerate() {
            let Representation::Solid(solid) = &object.representation else {
                continue;
            };
            let mat = world.material(object.material);
            let Some(thermal) = mat.thermal else {
                continue;
            };
            let matrix_k = ScalarCsr::from_mesh(&solid.mesh);
            let k_nodes = if solid.mesh.is_quadratic() { 10 } else { 4 };
            let mut elem_slots = Vec::with_capacity(solid.mesh.tets.len());
            for e in 0..solid.mesh.tets.len() {
                let elem = solid.mesh.element(e);
                let mut slots = Vec::with_capacity(k_nodes * k_nodes);
                for &ra in &elem {
                    for &cb in &elem {
                        slots.push(matrix_k.slot(ra as usize, cb as usize));
                    }
                }
                elem_slots.push(slots);
            }
            let capacity = solid
                .nodes
                .mass
                .iter()
                .map(|&m| m * thermal.specific_heat)
                .collect();
            self.solids.push(SolidThermal {
                object: index,
                thermal,
                density: mat.density,
                capacity,
                matrix_k: matrix_k.clone(),
                matrix_m: matrix_k,
                elem_slots,
            });
        }
        self.built_objects = world.objects().len();
    }

    fn step(&mut self, pool: &Pool, world: &mut World, h: f64) -> Self::Report {
        if !h.is_finite() || h <= 0.0 {
            return ThermalReport::default();
        }
        self.ensure_prepared(pool, world);

        let theta = match self.settings.integrator {
            ThermalIntegrator::BackwardEuler => 1.0,
            ThermalIntegrator::CrankNicolson => 0.5,
        };
        let mut report = ThermalReport {
            converged: true,
            min_temperature: f64::INFINITY,
            max_temperature: f64::NEG_INFINITY,
            ..ThermalReport::default()
        };

        for st in &mut self.solids {
            let Representation::Solid(solid) = &mut world.objects_mut()[st.object].representation
            else {
                unreachable!()
            };
            let n = solid.nodes.len();
            if n == 0 {
                continue;
            }
            let t_old = {
                let col = solid
                    .nodes
                    .columns
                    .ensure(Column::TEMPERATURE, st.thermal.reference_temperature);
                col.to_vec()
            };

            assemble_conductivity(st, &solid.nodes.position, &solid.mesh.rest, &solid.mesh);

            let mut f_eff = vec![0.0; n];
            let mut dirichlet: Vec<Option<f64>> = vec![None; n];
            let mut nodal_area: Option<Vec<f64>> = None;
            let area_of = |slot: &mut Option<Vec<f64>>| -> Vec<f64> {
                slot.get_or_insert_with(|| {
                    let mut a = vec![0.0; n];
                    for &[ia, ib, ic] in &solid.surface.faces {
                        let pa = solid.nodes.position[ia as usize];
                        let pb = solid.nodes.position[ib as usize];
                        let pc = solid.nodes.position[ic as usize];
                        let share = 0.5 * (pb - pa).cross(pc - pa).length() / 3.0;
                        a[ia as usize] += share;
                        a[ib as usize] += share;
                        a[ic as usize] += share;
                    }
                    a
                })
                .clone()
            };

            for boundary in &self.settings.boundaries {
                match boundary {
                    ThermalBoundary::Dirichlet {
                        object,
                        nodes,
                        temperature,
                    } if object.0 as usize == st.object => {
                        for &i in nodes {
                            if (i as usize) < n {
                                dirichlet[i as usize] = Some(*temperature);
                            }
                        }
                    }
                    ThermalBoundary::VolumetricSource { object, rate }
                        if object.0 as usize == st.object =>
                    {
                        for (i, f) in f_eff.iter_mut().enumerate() {
                            *f += rate * (solid.nodes.mass[i] / st.density);
                        }
                    }
                    ThermalBoundary::Convection {
                        object,
                        coefficient,
                        ambient,
                    } if object.0 as usize == st.object => {
                        let areas = area_of(&mut nodal_area);
                        for (i, &a) in areas.iter().enumerate() {
                            let h_a = coefficient * a;
                            st.matrix_k.values[st.matrix_k.diag_slot[i]] += h_a;
                            f_eff[i] += h_a * ambient;
                        }
                    }
                    ThermalBoundary::Radiation { object, ambient }
                        if object.0 as usize == st.object =>
                    {
                        let areas = area_of(&mut nodal_area);
                        let ta = ambient.max(0.0);
                        let eps_sigma = st.thermal.emissivity * STEFAN_BOLTZMANN;
                        for (i, &a) in areas.iter().enumerate() {
                            let t = t_old[i].max(0.0);
                            let h_rad = eps_sigma * (t * t + ta * ta) * (t + ta) * a;
                            st.matrix_k.values[st.matrix_k.diag_slot[i]] += h_rad;
                            f_eff[i] += h_rad * ta;
                        }
                    }
                    ThermalBoundary::NodalPower { object, power }
                        if object.0 as usize == st.object =>
                    {
                        for (i, f) in f_eff.iter_mut().enumerate().take(power.len()) {
                            *f += power[i];
                        }
                    }
                    _ => {}
                }
            }

            for (i, cap) in st.capacity.iter_mut().enumerate() {
                *cap = solid.nodes.mass[i] * st.thermal.effective_specific_heat(t_old[i]);
            }

            let mut k_t = vec![0.0; n];
            st.matrix_k.spmv(pool, &t_old, &mut k_t);
            let mut rhs = vec![0.0; n];
            for i in 0..n {
                rhs[i] = st.capacity[i] * t_old[i] - (1.0 - theta) * h * k_t[i] + h * f_eff[i];
            }

            // Form M = C + theta * h * K_eff.
            for (dst, &kval) in st.matrix_m.values.iter_mut().zip(&st.matrix_k.values) {
                *dst = theta * h * kval;
            }
            for i in 0..n {
                st.matrix_m.values[st.matrix_m.diag_slot[i]] += st.capacity[i];
            }

            // Symmetric Dirichlet elimination so CG solves an SPD system.
            let mut t_next = t_old;
            for r in 0..n {
                let start = st.matrix_m.row_start[r];
                let end = st.matrix_m.row_start[r + 1];
                if let Some(t_d) = dirichlet[r] {
                    for idx in start..end {
                        st.matrix_m.values[idx] = 0.0;
                    }
                    st.matrix_m.values[st.matrix_m.diag_slot[r]] = 1.0;
                    rhs[r] = t_d;
                    t_next[r] = t_d;
                } else {
                    for idx in start..end {
                        let c = st.matrix_m.cols[idx];
                        if let Some(t_d) = dirichlet[c] {
                            rhs[r] -= st.matrix_m.values[idx] * t_d;
                            st.matrix_m.values[idx] = 0.0;
                        }
                    }
                }
            }

            let (iters, ok) = solve_cg(
                pool,
                &st.matrix_m,
                &rhs,
                &mut t_next,
                self.settings.tolerance,
                self.settings.max_iterations,
            );
            report.iterations += iters;
            report.converged &= ok;

            let col = solid.nodes.columns.get_mut(Column::TEMPERATURE).unwrap();
            col.copy_from_slice(&t_next);
            for (i, &t) in t_next.iter().enumerate() {
                report.min_temperature = report.min_temperature.min(t);
                report.max_temperature = report.max_temperature.max(t);
                report.thermal_energy += st.capacity[i] * t;
            }
        }

        if self.solids.is_empty() {
            report.min_temperature = 0.0;
            report.max_temperature = 0.0;
        }
        report
    }
}
