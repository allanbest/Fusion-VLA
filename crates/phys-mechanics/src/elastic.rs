//! Elasticity of finite-element solids: tetrahedra, linear (4 nodes) or
//! quadratic (10 nodes), a hyperelastic law per material.
//!
//! An element's energy is integrated at quadrature points: one for a
//! linear element (its strain is constant), four for a quadratic one (its
//! strain varies linearly — what bending needs; linear elements lock).
//!
//! Assembly is in two parallel passes, so no two threads ever write the
//! same place and the result does not depend on the thread count:
//! 1. each element computes its nodal forces and Hessian blocks into its
//!    own cache entry;
//! 2. each row of the matrix (each node) gathers the blocks of the elements
//!    around it, in a fixed order, straight into its pattern slots.

use core_math::constants::ROOM_TEMPERATURE;
use core_math::{Mat3, Vec3};
use core_numerics::{BlockMatrix, Triplets};
use core_runtime::Pool;
use core_world::{Column, Representation, TetMesh, World};

use crate::dofs::{Layout, HELD};
use crate::law::{NeoHookean, Warm};
use crate::term::{Context, Term};

/// Elements per parallel chunk of an energy: the chunks' sums are added in
/// order, so this is part of the energy's value to the last bit.
const ELEMENTS_PER_CHUNK: usize = 128;

/// Elements per parallel chunk of the forces and Hessian blocks, which each
/// element writes to a place of its own: any size gives the same result.
/// Small, so that a body of a hundred elements still spreads over the
/// threads free to help — an element's blocks are some ten microseconds of
/// work, a chunk's hand-off a fraction of one.
const ELEMENTS_PER_PASS: usize = 8;

/// Rows (nodes) per parallel chunk of gathering element blocks into the
/// matrix and forces into the gradient; again any size gives the same
/// result.
const ROWS_PER_PASS: usize = 32;

/// The shape-function gradients and weight of one quadrature point:
/// `F = Σ xₐ gₐᵀ`, energy `w Ψ(F)`.
#[derive(Clone, Copy, Debug)]
struct Point<const K: usize> {
    weight: f64,
    gradients: [Vec3; K],
}

/// Elements of one kind: `K` nodes, `KK = K²` Hessian blocks.
#[derive(Clone, Debug, Default)]
struct Family<const K: usize, const KK: usize> {
    /// Per element: its solid (index in the layout), local element index
    /// within that solid, and nodes.
    solid: Vec<u32>,
    elem_in_solid: Vec<u32>,
    nodes: Vec<[u32; K]>,
    /// Per element, reference quadrature points (`points` each), and live
    /// quadrature points incorporating `F_p⁻¹` / thermal stretch.
    rest: Vec<Point<K>>,
    quadrature: Vec<Point<K>>,
    /// Empty unless at least one solid has [`Column::TEMPERATURE`].
    point_laws: Vec<NeoHookean>,
    points: usize,
    /// Per element: the slot of each Hessian block, or `u32::MAX` where a
    /// node is held.
    slots: Vec<[u32; KK]>,
    /// Per element: forces on its nodes and Hessian blocks, weighted.
    forces: Vec<[Vec3; K]>,
    hessians: Vec<[Mat3; KK]>,
    /// Per element, each quadrature point's warm start for its eigensystem.
    warm: Vec<[Warm; 4]>,
}

/// Per-solid plastic and cohesive-damage internal state, indexed by
/// `element * points + qp`. Empty vectors when inactive.
#[derive(Clone, Debug, Default, PartialEq)]
pub(crate) struct SolidPlastic {
    pub fp_inv: Vec<Mat3>,
    pub eps_p: Vec<f64>,
    pub damage: Vec<f64>,
}

const KEAST_A: f64 = 0.585_410_196_624_968_5;
const KEAST_B: f64 = 0.138_196_601_125_010_5;
const KEAST_RULE: [[f64; 4]; 4] = [
    [KEAST_A, KEAST_B, KEAST_B, KEAST_B],
    [KEAST_B, KEAST_A, KEAST_B, KEAST_B],
    [KEAST_B, KEAST_B, KEAST_A, KEAST_B],
    [KEAST_B, KEAST_B, KEAST_B, KEAST_A],
];

fn eval_temperature<const K: usize>(temps: &[f64], nodes: &[u32; K], q: usize) -> f64 {
    if K == 4 {
        0.25 * (temps[nodes[0] as usize]
            + temps[nodes[1] as usize]
            + temps[nodes[2] as usize]
            + temps[nodes[3] as usize])
    } else {
        let l = KEAST_RULE[q];
        let mut t = 0.0;
        for i in 0..4 {
            t += l[i] * (2.0 * l[i] - 1.0) * temps[nodes[i] as usize];
        }
        for (k, (i, j)) in core_world::EDGES.into_iter().enumerate() {
            t += 4.0 * l[i] * l[j] * temps[nodes[4 + k] as usize];
        }
        t
    }
}

impl<const K: usize, const KK: usize> Family<K, KK> {
    #[inline]
    fn point_law<'a>(&'a self, laws: &'a [NeoHookean], e: usize, q: usize) -> &'a NeoHookean {
        if self.point_laws.is_empty() {
            &laws[self.solid[e] as usize]
        } else {
            &self.point_laws[e * self.points + q]
        }
    }

    fn len(&self) -> usize {
        self.nodes.len()
    }

    fn push(
        &mut self,
        solid: usize,
        elem_in_solid: usize,
        nodes: [u32; K],
        quadrature: impl IntoIterator<Item = Point<K>>,
    ) {
        self.solid.push(solid as u32);
        self.elem_in_solid.push(elem_in_solid as u32);
        self.nodes.push(nodes);
        for p in quadrature {
            self.rest.push(p);
            self.quadrature.push(p);
        }
    }

    fn finish(&mut self, derivatives: bool) {
        self.points = if self.nodes.is_empty() {
            1
        } else {
            self.quadrature.len() / self.nodes.len()
        };
        if !derivatives {
            return;
        }
        self.forces = vec![[Vec3::ZERO; K]; self.len()];
        self.warm = vec![[Warm::default(); 4]; self.len()];
        self.hessians = vec![[Mat3::ZERO; KK]; self.len()];
    }

    fn apply_state(
        &mut self,
        world: &World,
        layout: &Layout,
        plastic: &[SolidPlastic],
        laws: &[NeoHookean],
    ) {
        let any_temp = layout.solids.iter().any(|d| {
            matches!(
                &world.objects()[d.object].representation,
                Representation::Solid(s) if s.nodes.columns.has(Column::TEMPERATURE)
            )
        });
        let any_damage = plastic.iter().any(|p| !p.damage.is_empty());
        let any_local = any_temp || any_damage;
        if !any_local && self.point_laws.is_empty() && plastic.iter().all(|p| p.fp_inv.is_empty()) {
            self.quadrature.copy_from_slice(&self.rest);
            return;
        }
        if any_local {
            self.point_laws.resize(
                self.len() * self.points,
                NeoHookean {
                    mu: 0.0,
                    lambda: 0.0,
                },
            );
        } else {
            self.point_laws.clear();
        }
        for e in 0..self.len() {
            let s = self.solid[e] as usize;
            let sp = &plastic[s];
            let local_e = self.elem_in_solid[e] as usize;
            let q_start = e * self.points;
            let p_start = local_e * self.points;
            let obj = &world.objects()[layout.solids[s].object];
            let Representation::Solid(solid) = &obj.representation else {
                unreachable!()
            };
            let mat = world.material(obj.material);
            let temps = if any_temp {
                solid.nodes.columns.get(Column::TEMPERATURE)
            } else {
                None
            };
            let k_min = mat
                .fracture
                .map_or(1.0, |fr| fr.residual_stiffness.clamp(1e-6, 1.0));
            for q in 0..self.points {
                let fp_inv_t = if sp.fp_inv.is_empty() {
                    Mat3::IDENTITY
                } else {
                    sp.fp_inv[p_start + q].transpose()
                };
                let (mut mu, mut lambda, gamma) = if let Some(t_col) = temps {
                    let t_q = eval_temperature(t_col, &self.nodes[e], q);
                    let (mu, lambda) = mat.lame(t_q);
                    let gamma = mat.thermal.map_or(1.0, |th| {
                        (1.0 + th.expansion * (t_q - th.reference_temperature)).max(0.1)
                    });
                    (mu, lambda, gamma)
                } else {
                    (laws[s].mu, laws[s].lambda, 1.0)
                };
                if any_local {
                    if !sp.damage.is_empty() {
                        let d = sp.damage[p_start + q].clamp(0.0, 1.0);
                        let one_m = 1.0 - d;
                        let g_d = one_m * one_m * (1.0 - k_min) + k_min;
                        mu *= g_d;
                        lambda *= g_d;
                    }
                    self.point_laws[q_start + q] = NeoHookean { mu, lambda };
                }
                let rest = &self.rest[q_start + q];
                let inv_gamma = 1.0 / gamma;
                let gamma3 = gamma * gamma * gamma;
                self.quadrature[q_start + q] = Point {
                    weight: rest.weight * gamma3,
                    gradients: rest.gradients.map(|g| (fp_inv_t * g) * inv_gamma),
                };
            }
        }
    }

    fn update_plasticity(
        &mut self,
        ctx: &Context,
        laws: &[NeoHookean],
        plastic: &mut [SolidPlastic],
        x: &[f64],
        mut heat: Option<&mut [Vec<f64>]>,
    ) -> f64 {
        let has_plastic = plastic.iter().any(|p| !p.fp_inv.is_empty());
        if !has_plastic || self.len() == 0 {
            return 0.0;
        }
        let params: Vec<_> = ctx
            .layout
            .solids
            .iter()
            .map(|d| {
                ctx.world
                    .material(ctx.world.objects()[d.object].material)
                    .plasticity
            })
            .collect();
        type ElemPlastic = ([Mat3; 4], [f64; 4], f64);
        let mut elem_results: Vec<Option<ElemPlastic>> = vec![None; self.len()];
        let mut warm = std::mem::take(&mut self.warm);
        if warm.len() < self.len() {
            warm.resize(self.len(), [Warm::default(); 4]);
        }
        let this = &*self;
        ctx.pool.split_reduce(
            (&mut elem_results[..], &mut warm[..]),
            ELEMENTS_PER_PASS,
            (),
            |offset, (out, wpart)| {
                for (i, (slot, wslice)) in out.iter_mut().zip(wpart.iter_mut()).enumerate() {
                    let e = offset + i;
                    let s = this.solid[e] as usize;
                    let Some(pl) = params[s] else { continue };
                    let sp = &plastic[s];
                    if sp.fp_inv.is_empty() {
                        continue;
                    }
                    let p_start = (this.elem_in_solid[e] as usize) * this.points;
                    let q_start = e * this.points;
                    let mut new_fp = [Mat3::IDENTITY; 4];
                    let mut new_ep = [0.0; 4];
                    let mut changed = false;
                    let mut drop = 0.0;
                    for q in 0..this.points {
                        let law = this.point_law(laws, e, q);
                        let pt = &this.quadrature[q_start + q];
                        let fe_trial = this.deformation(ctx, e, pt, x);
                        let fp_old = sp.fp_inv[p_start + q];
                        let ep_old = sp.eps_p[p_start + q];
                        if let Some((fe_new, delta_fp_inv, ep_new)) = law.hencky_j2_return(
                            &fe_trial,
                            ep_old,
                            pl.yield_stress,
                            pl.hardening,
                            &mut wslice[q],
                        ) {
                            let updated = fp_old * delta_fp_inv;
                            let det = updated.determinant();
                            new_fp[q] = if det > 0.0 {
                                updated * det.cbrt().recip()
                            } else {
                                updated
                            };
                            new_ep[q] = ep_new;
                            drop += pt.weight * (law.energy(&fe_new) - law.energy(&fe_trial));
                            changed = true;
                        } else {
                            new_fp[q] = fp_old;
                            new_ep[q] = ep_old;
                        }
                    }
                    if changed {
                        *slot = Some((new_fp, new_ep, drop));
                    }
                }
            },
            |_, _| (),
        );
        self.warm = warm;
        let pts = self.points;
        let mut total_drop = 0.0;
        for (e, res) in elem_results.into_iter().enumerate() {
            if let Some((new_fp, new_ep, drop)) = res {
                let s = self.solid[e] as usize;
                let sp = &mut plastic[s];
                let p_start = (self.elem_in_solid[e] as usize) * pts;
                sp.fp_inv[p_start..p_start + pts].copy_from_slice(&new_fp[..pts]);
                sp.eps_p[p_start..p_start + pts].copy_from_slice(&new_ep[..pts]);
                total_drop += drop;
                if drop < 0.0 {
                    if let Some(heat_solids) = heat.as_deref_mut() {
                        let per_node = -drop / (K as f64);
                        for &node in &self.nodes[e] {
                            heat_solids[s][node as usize] += per_node;
                        }
                    }
                }
            }
        }
        total_drop
    }

    fn update_damage(
        &self,
        ctx: &Context,
        laws: &[NeoHookean],
        plastic: &mut [SolidPlastic],
        x: &[f64],
        mut heat: Option<&mut [Vec<f64>]>,
    ) -> f64 {
        let has_damage = plastic.iter().any(|p| !p.damage.is_empty());
        if !has_damage || self.len() == 0 {
            return 0.0;
        }
        let params: Vec<_> = ctx
            .layout
            .solids
            .iter()
            .map(|d| {
                let obj = &ctx.world.objects()[d.object];
                let temps = match &obj.representation {
                    Representation::Solid(s) => s.nodes.columns.get(Column::TEMPERATURE),
                    _ => None,
                };
                let mat = ctx.world.material(obj.material);
                (mat, mat.fracture, temps)
            })
            .collect();
        let mut elem_results: Vec<Option<([f64; 4], f64)>> = vec![None; self.len()];
        let this = self;
        ctx.pool
            .for_chunks_mut(&mut elem_results, ELEMENTS_PER_PASS, |offset, out| {
                for (i, slot) in out.iter_mut().enumerate() {
                    let e = offset + i;
                    let s = this.solid[e] as usize;
                    let (mat, Some(fr), temps) = params[s] else {
                        continue;
                    };
                    let sp = &plastic[s];
                    if sp.damage.is_empty() {
                        continue;
                    }
                    let p_start = (this.elem_in_solid[e] as usize) * this.points;
                    let q_start = e * this.points;
                    let k_min = fr.residual_stiffness.clamp(1e-6, 1.0);
                    let mut new_d = [0.0; 4];
                    let mut changed = false;
                    let mut drop = 0.0;
                    for (q, new_d_slot) in new_d.iter_mut().enumerate().take(this.points) {
                        let d_old = sp.damage[p_start + q].clamp(0.0, 1.0);
                        *new_d_slot = d_old;
                        let (mu0, lambda0) = if let Some(t_col) = temps {
                            mat.lame(eval_temperature(t_col, &this.nodes[e], q))
                        } else {
                            (laws[s].mu, laws[s].lambda)
                        };
                        let pt = &this.quadrature[q_start + q];
                        let fe = this.deformation(ctx, e, pt, x);
                        let j = fe.determinant();
                        if j <= 1e-12 {
                            continue;
                        }
                        let log_j = j.ln();
                        let psi_iso =
                            (0.5 * mu0 * ((fe.frobenius_squared() - 3.0) - 2.0 * log_j)).max(0.0);
                        let psi_vol_tens = if j > 1.0 {
                            0.5 * lambda0 * log_j * log_j
                        } else {
                            0.0
                        };
                        let psi_plus = psi_iso + psi_vol_tens;
                        if psi_plus > fr.toughness {
                            let excess = psi_plus - fr.toughness;
                            let d_cand =
                                (1.0 - (-(excess / fr.softening.max(1e-12))).exp()).clamp(0.0, 1.0);
                            if d_cand > d_old {
                                let g_old = (1.0 - d_old) * (1.0 - d_old) * (1.0 - k_min) + k_min;
                                let g_new = (1.0 - d_cand) * (1.0 - d_cand) * (1.0 - k_min) + k_min;
                                let psi_total = NeoHookean {
                                    mu: mu0,
                                    lambda: lambda0,
                                }
                                .energy(&fe);
                                drop += pt.weight * (g_new - g_old) * psi_total;
                                *new_d_slot = d_cand;
                                changed = true;
                            }
                        }
                    }
                    if changed {
                        *slot = Some((new_d, drop));
                    }
                }
            });
        let pts = self.points;
        let mut total_drop = 0.0;
        for (e, res) in elem_results.into_iter().enumerate() {
            if let Some((new_d, drop)) = res {
                let s = self.solid[e] as usize;
                let sp = &mut plastic[s];
                let p_start = (self.elem_in_solid[e] as usize) * pts;
                sp.damage[p_start..p_start + pts].copy_from_slice(&new_d[..pts]);
                total_drop += drop;
                if drop < 0.0 {
                    if let Some(heat_solids) = heat.as_deref_mut() {
                        let per_node = -drop / (K as f64);
                        for &node in &self.nodes[e] {
                            heat_solids[s][node as usize] += per_node;
                        }
                    }
                }
            }
        }
        total_drop
    }

    fn blocks(&self, layout: &Layout, e: usize) -> [u32; K] {
        let dofs = &layout.solids[self.solid[e] as usize];
        self.nodes[e].map(|i| dofs.block[i as usize])
    }

    fn deformation(&self, ctx: &Context, e: usize, point: &Point<K>, x: &[f64]) -> Mat3 {
        let s = self.solid[e] as usize;
        let mut f = Mat3::ZERO;
        for (a, &node) in self.nodes[e].iter().enumerate() {
            f += Mat3::outer(
                ctx.frame.node(ctx.layout, s, node as usize, x),
                point.gradients[a],
            );
        }
        f
    }

    fn pattern(&self, layout: &Layout, entries: &mut Vec<(usize, usize)>) {
        for e in 0..self.len() {
            let blocks = self.blocks(layout, e);
            for &a in &blocks {
                for &b in &blocks {
                    if a != HELD && b != HELD {
                        entries.push((a as usize, b as usize));
                    }
                }
            }
        }
    }

    fn bind(&mut self, layout: &Layout, matrix: &BlockMatrix) {
        self.slots = (0..self.len())
            .map(|e| {
                let blocks = self.blocks(layout, e);
                std::array::from_fn(|ab| {
                    let (a, b) = (blocks[ab / K], blocks[ab % K]);
                    if a == HELD || b == HELD {
                        u32::MAX
                    } else {
                        matrix.slot(a as usize, b as usize).unwrap() as u32
                    }
                })
            })
            .collect();
    }

    fn energy(&self, ctx: &Context, laws: &[NeoHookean], x: &[f64]) -> f64 {
        ctx.pool.map_reduce(
            self.len(),
            ELEMENTS_PER_CHUNK,
            0.0,
            |range| {
                let mut sum = 0.0;
                for e in range {
                    let q_start = e * self.points;
                    for q in 0..self.points {
                        let point = &self.quadrature[q_start + q];
                        let law = self.point_law(laws, e, q);
                        sum += point.weight * law.energy(&self.deformation(ctx, e, point, x));
                    }
                }
                sum
            },
            |a, b| a + b,
        )
    }

    /// Physical force gradient only: startup must not compute eigensystems
    /// or allocate/assemble another tangent matrix.
    fn forces_only(&mut self, ctx: &Context, laws: &[NeoHookean], x: &[f64]) {
        let mut forces = std::mem::take(&mut self.forces);
        let this = &*self;
        ctx.pool
            .for_chunks_mut(&mut forces, ELEMENTS_PER_PASS, |offset, part| {
                for (i, force) in part.iter_mut().enumerate() {
                    let e = offset + i;
                    *force = [Vec3::ZERO; K];
                    let q_start = e * this.points;
                    for q in 0..this.points {
                        let point = &this.quadrature[q_start + q];
                        let law = this.point_law(laws, e, q);
                        let p = law.stress(&this.deformation(ctx, e, point, x)) * point.weight;
                        for (out, g) in force.iter_mut().zip(&point.gradients) {
                            *out += p * *g;
                        }
                    }
                }
            });
        self.forces = forces;
    }

    /// Pass 1: every element's forces and Hessian blocks, scaled by `scale`.
    fn compute(&mut self, ctx: &Context, laws: &[NeoHookean], x: &[f64], scale: f64) {
        let mut forces = std::mem::take(&mut self.forces);
        let mut hessians = std::mem::take(&mut self.hessians);
        let mut warm = std::mem::take(&mut self.warm);
        let this = &*self;
        ctx.pool.split_reduce(
            (&mut forces[..], &mut hessians[..], &mut warm[..]),
            ELEMENTS_PER_PASS,
            (),
            |offset, (forces, hessians, warm)| {
                for (i, ((force, hessian), warm)) in forces
                    .iter_mut()
                    .zip(hessians.iter_mut())
                    .zip(warm.iter_mut())
                    .enumerate()
                {
                    let e = offset + i;
                    *force = [Vec3::ZERO; K];
                    *hessian = [Mat3::ZERO; KK];
                    let q_start = e * this.points;
                    for (q, warm) in warm.iter_mut().take(this.points).enumerate() {
                        let point = &this.quadrature[q_start + q];
                        let law = this.point_law(laws, e, q);
                        let f = this.deformation(ctx, e, point, x);
                        let w = point.weight * scale;
                        let (p, _, log_j, inverse_t) = law.stress_and_factors(&f);
                        let pw = p * w;
                        for (f, g) in force.iter_mut().zip(&point.gradients) {
                            *f += pw * *g;
                        }
                        let transformed = point.gradients.map(|g| inverse_t * g);
                        let a_coefficient = law.mu - law.lambda * log_j;
                        for a in 0..K {
                            let ga = point.gradients[a];
                            for b in a..K {
                                let block = Mat3::scalar(law.mu * ga.dot(point.gradients[b]))
                                    + Mat3::outer(transformed[b], transformed[a]) * a_coefficient
                                    + Mat3::outer(transformed[a], transformed[b]) * law.lambda;
                                hessian[K * a + b] += block * w;
                            }
                        }
                        law.for_each_negative_mode(&f, warm, |value, q| {
                            let v = point.gradients.map(|g| q * g);
                            for a in 0..K {
                                let va = v[a] * (-value * w);
                                for b in a..K {
                                    hessian[K * a + b] += Mat3::outer(va, v[b]);
                                }
                            }
                        });
                    }
                    for a in 0..K {
                        for b in 0..a {
                            hessian[K * a + b] = hessian[K * b + a].transpose();
                        }
                    }
                }
            },
            |_, _| (),
        );
        self.forces = forces;
        self.hessians = hessians;
        self.warm = warm;
    }
}

/// Which element, of which family, and its local node: one entry of a
/// node's incidence list.
#[derive(Clone, Copy, Debug)]
struct Incident {
    quadratic: bool,
    element: u32,
    local: u8,
}

#[derive(Clone, Debug, Default)]
pub struct Elastic {
    linear: Family<4, 16>,
    quadratic: Family<10, 100>,
    /// Per solid in the layout.
    laws: Vec<NeoHookean>,
    /// Per block: the elements that touch it, as ranges into `incident`.
    incident_start: Vec<u32>,
    incident: Vec<Incident>,
}

/// The ten quadratic shape functions' gradients in reference coordinates
/// at barycentric point `l`: corners `Lᵢ(2Lᵢ − 1)`, edges `4 LᵢLⱼ`.
fn quadratic_gradients(l: [f64; 4]) -> [Vec3; 10] {
    let dl = [Vec3::splat(-1.0), Vec3::X, Vec3::Y, Vec3::Z];
    let mut g = [Vec3::ZERO; 10];
    for i in 0..4 {
        g[i] = dl[i] * (4.0 * l[i] - 1.0);
    }
    for (k, (i, j)) in core_world::EDGES.into_iter().enumerate() {
        g[4 + k] = (dl[j] * l[i] + dl[i] * l[j]) * 4.0;
    }
    g
}

/// Maps reference gradients to material ones for an element at rest
/// positions `rest`: the point, with weight `w · det J`.
fn material_point<const K: usize>(rest: &[Vec3; K], reference: [Vec3; K], weight: f64) -> Point<K> {
    let mut jacobian = Mat3::ZERO;
    for a in 0..K {
        jacobian += Mat3::outer(rest[a], reference[a]);
    }
    let inverse_t = jacobian.inverse().transpose();
    Point {
        weight: weight * jacobian.determinant(),
        gradients: reference.map(|g| inverse_t * g),
    }
}

type Elements = (Vec<[u32; 4]>, Vec<Point<4>>, Vec<[u32; 10]>, Vec<Point<10>>);

/// The quadrature points of every element of a mesh.
fn element_points(mesh: &TetMesh) -> Elements {
    let (mut linear, mut linear_points, mut quadratic, mut quadratic_points) =
        (Vec::new(), Vec::new(), Vec::new(), Vec::new());
    if mesh.is_quadratic() {
        // Four-point rule, exact for quadratics, weights 1/24 on the unit
        // tetrahedron.
        let (a, b) = (0.585_410_196_624_968_5, 0.138_196_601_125_010_5);
        let rule = [[a, b, b, b], [b, a, b, b], [b, b, a, b], [b, b, b, a]];
        for (t, m) in mesh.tets.iter().zip(&mesh.midpoints) {
            let nodes: [u32; 10] = std::array::from_fn(|k| if k < 4 { t[k] } else { m[k - 4] });
            let rest = nodes.map(|i| mesh.rest[i as usize]);
            quadratic.push(nodes);
            quadratic_points.extend(
                rule.iter()
                    .map(|&l| material_point(&rest, quadratic_gradients(l), 1.0 / 24.0)),
            );
        }
    } else {
        let reference = [Vec3::splat(-1.0), Vec3::X, Vec3::Y, Vec3::Z];
        for t in &mesh.tets {
            let rest = t.map(|i| mesh.rest[i as usize]);
            linear.push(*t);
            linear_points.push(material_point(&rest, reference, 1.0 / 6.0));
        }
    }
    (linear, linear_points, quadratic, quadratic_points)
}

impl Elastic {
    /// The elements of every solid in the world, at rest as meshed.
    pub fn new(ctx: &Context) -> Self {
        Self::build(ctx, true)
    }

    /// Quadrature only, without force, Hessian or eigensystem scratch.
    pub(crate) fn for_energy(ctx: &Context) -> Self {
        Self::build(ctx, false)
    }

    fn build(ctx: &Context, derivatives: bool) -> Self {
        let mut e = Self::default();
        for (s, dofs) in ctx.layout.solids.iter().enumerate() {
            let object = &ctx.world.objects()[dofs.object];
            let Representation::Solid(solid) = &object.representation else {
                unreachable!()
            };
            let (mu, lambda) = ctx.world.material(object.material).lame(ROOM_TEMPERATURE);
            e.laws.push(NeoHookean { mu, lambda });
            let (linear, linear_points, quadratic, quadratic_points) = element_points(&solid.mesh);
            for (k, nodes) in linear.into_iter().enumerate() {
                e.linear.push(s, k, nodes, [linear_points[k]]);
            }
            for (k, nodes) in quadratic.into_iter().enumerate() {
                e.quadratic.push(
                    s,
                    k,
                    nodes,
                    quadratic_points[4 * k..4 * k + 4].iter().copied(),
                );
            }
        }
        e.linear.finish(derivatives);
        e.quadratic.finish(derivatives);
        if !derivatives {
            return e;
        }
        // Which elements touch which block.
        let mut lists: Vec<Vec<Incident>> = vec![Vec::new(); ctx.layout.blocks];
        for k in 0..e.linear.len() {
            for (a, &b) in e.linear.blocks(ctx.layout, k).iter().enumerate() {
                if b != HELD {
                    lists[b as usize].push(Incident {
                        quadratic: false,
                        element: k as u32,
                        local: a as u8,
                    });
                }
            }
        }
        for k in 0..e.quadratic.len() {
            for (a, &b) in e.quadratic.blocks(ctx.layout, k).iter().enumerate() {
                if b != HELD {
                    lists[b as usize].push(Incident {
                        quadratic: true,
                        element: k as u32,
                        local: a as u8,
                    });
                }
            }
        }
        e.incident_start.push(0);
        for list in lists {
            e.incident.extend(list);
            e.incident_start.push(e.incident.len() as u32);
        }
        e
    }

    fn incident(&self, block: usize) -> &[Incident] {
        &self.incident[self.incident_start[block] as usize..self.incident_start[block + 1] as usize]
    }

    pub(crate) fn sync_state(&mut self, world: &World, layout: &Layout, plastic: &[SolidPlastic]) {
        self.linear.apply_state(world, layout, plastic, &self.laws);
        self.quadratic
            .apply_state(world, layout, plastic, &self.laws);
    }

    pub(crate) fn update_plasticity(
        &mut self,
        ctx: &Context,
        plastic: &mut [SolidPlastic],
        x: &[f64],
        mut heat: Option<&mut [Vec<f64>]>,
    ) -> f64 {
        if plastic.iter().all(|p| p.fp_inv.is_empty()) {
            return 0.0;
        }
        let work = self
            .linear
            .update_plasticity(ctx, &self.laws, plastic, x, heat.as_deref_mut())
            + self
                .quadratic
                .update_plasticity(ctx, &self.laws, plastic, x, heat);
        self.sync_state(ctx.world, ctx.layout, plastic);
        work
    }

    pub(crate) fn update_damage(
        &mut self,
        ctx: &Context,
        plastic: &mut [SolidPlastic],
        x: &[f64],
        mut heat: Option<&mut [Vec<f64>]>,
    ) -> f64 {
        if plastic.iter().all(|p| p.damage.is_empty()) {
            return 0.0;
        }
        let work = self
            .linear
            .update_damage(ctx, &self.laws, plastic, x, heat.as_deref_mut())
            + self
                .quadratic
                .update_damage(ctx, &self.laws, plastic, x, heat);
        self.sync_state(ctx.world, ctx.layout, plastic);
        work
    }

    pub(crate) fn sync_plastic_columns(
        layout: &Layout,
        plastic: &[SolidPlastic],
        world: &mut World,
    ) {
        for (s, d) in layout.solids.iter().enumerate() {
            let sp = &plastic[s];
            if sp.fp_inv.is_empty() && sp.damage.is_empty() {
                continue;
            }
            let Representation::Solid(solid) = &mut world.objects_mut()[d.object].representation
            else {
                unreachable!()
            };
            if solid.mesh.tets.is_empty() {
                continue;
            }
            let pts = if solid.mesh.is_quadratic() { 4 } else { 1 };
            if !sp.fp_inv.is_empty()
                && solid.mesh.tets.len() * pts == sp.eps_p.len()
                && (sp.eps_p.iter().any(|&v| v > 0.0) || solid.elements.has(Column::PLASTIC_STRAIN))
            {
                let col = solid.elements.ensure(Column::PLASTIC_STRAIN, 0.0);
                for (e, dst) in col.iter_mut().enumerate() {
                    let slice = &sp.eps_p[e * pts..(e + 1) * pts];
                    *dst = slice.iter().sum::<f64>() / pts as f64;
                }
            }
            if !sp.damage.is_empty()
                && solid.mesh.tets.len() * pts == sp.damage.len()
                && (sp.damage.iter().any(|&v| v > 0.0) || solid.elements.has(Column::DAMAGE))
            {
                let col = solid.elements.ensure(Column::DAMAGE, 0.0);
                for (e, dst) in col.iter_mut().enumerate() {
                    let slice = &sp.damage[e * pts..(e + 1) * pts];
                    *dst = slice.iter().sum::<f64>() / pts as f64;
                }
            }
        }
    }

    /// Evaluate physical energy with the prepared quadrature and material data.
    pub(crate) fn physical_energy(&self, ctx: &Context, x: &[f64]) -> f64 {
        self.linear.energy(ctx, &self.laws, x) + self.quadratic.energy(ctx, &self.laws, x)
    }

    pub(crate) fn physical_gradient(&mut self, ctx: &Context, x: &[f64], out: &mut [f64]) {
        self.linear.forces_only(ctx, &self.laws, x);
        self.quadratic.forces_only(ctx, &self.laws, x);
        for b in 0..ctx.layout.blocks {
            self.gather_force(b).write(out, b);
        }
    }

    /// Pass 2: node `block`'s force, gathered from the element caches.
    fn gather_force(&self, block: usize) -> Vec3 {
        self.incident(block)
            .iter()
            .map(|i| {
                let (e, a) = (i.element as usize, i.local as usize);
                if i.quadratic {
                    self.quadratic.forces[e][a]
                } else {
                    self.linear.forces[e][a]
                }
            })
            .sum()
    }

    /// Pass 2: node `row`'s Hessian row, gathered into its slots.
    fn gather_row(&self, row: usize, first: usize, blocks: &mut [Mat3]) {
        for i in self.incident(row) {
            let (e, a) = (i.element as usize, i.local as usize);
            if i.quadratic {
                gather(&self.quadratic, e, a, first, blocks);
            } else {
                gather(&self.linear, e, a, first, blocks);
            }
        }
    }
}

fn gather<const K: usize, const KK: usize>(
    family: &Family<K, KK>,
    e: usize,
    a: usize,
    first: usize,
    blocks: &mut [Mat3],
) {
    for b in 0..K {
        let slot = family.slots[e][K * a + b];
        if slot != u32::MAX {
            blocks[slot as usize - first] += family.hessians[e][K * a + b];
        }
    }
}

impl Term for Elastic {
    fn name(&self) -> &'static str {
        "elasticity"
    }

    fn begin_step(&mut self, _ctx: &Context) {
        // Warm starts carry between one solve's iterations only.
        self.linear.warm.fill([Warm::default(); 4]);
        self.quadratic.warm.fill([Warm::default(); 4]);
    }

    fn pattern(&self, ctx: &Context, entries: &mut Vec<(usize, usize)>) {
        self.linear.pattern(ctx.layout, entries);
        self.quadratic.pattern(ctx.layout, entries);
    }

    fn bind(&mut self, ctx: &Context, matrix: &BlockMatrix) {
        self.linear.bind(ctx.layout, matrix);
        self.quadratic.bind(ctx.layout, matrix);
    }

    fn energy(&mut self, ctx: &Context, x: &[f64]) -> f64 {
        (self.linear.energy(ctx, &self.laws, x) + self.quadratic.energy(ctx, &self.laws, x))
            * ctx.frame.weight
    }

    fn derivatives(
        &mut self,
        ctx: &Context,
        x: &[f64],
        gradient: &mut [f64],
        fixed: &mut BlockMatrix,
        _changing: &mut Triplets,
    ) {
        let weight = ctx.frame.weight;
        self.linear.compute(ctx, &self.laws, x, weight);
        self.quadratic.compute(ctx, &self.laws, x, weight);
        let this = &*self;
        ctx.pool
            .for_chunks_mut(gradient, 3 * ROWS_PER_PASS, |offset, part| {
                for (k, out) in part.chunks_exact_mut(3).enumerate() {
                    let sum = this.gather_force(offset / 3 + k);
                    out[0] += sum.x;
                    out[1] += sum.y;
                    out[2] += sum.z;
                }
            });
        fixed.fill_rows(ctx.pool, ROWS_PER_PASS, |row, first, blocks| {
            this.gather_row(row, first, blocks)
        });
    }
}

/// The elastic (strain) energy stored in every solid of `world`, J — for
/// the energy ledger.
pub fn strain_energy(world: &World) -> f64 {
    let pool = Pool::new(1);
    let layout = Layout::new(world);
    let frame = crate::dofs::Frame::default();
    let ctx = Context {
        pool: &pool,
        world,
        layout: &layout,
        frame: &frame,
    };
    let elastic = Elastic::new(&ctx);
    // At the world's node positions (held nodes read where they are held).
    let mut x = vec![0.0; 3 * layout.blocks];
    for dofs in &layout.solids {
        let Representation::Solid(s) = &world.objects()[dofs.object].representation else {
            unreachable!()
        };
        for (node, &b) in dofs.block.iter().enumerate() {
            if b != HELD {
                s.nodes.position[node].write(&mut x, b as usize);
            }
        }
    }
    elastic.linear.energy(&ctx, &elastic.laws, &x)
        + elastic.quadratic.energy(&ctx, &elastic.laws, &x)
}

/// Deformation at mesh nodes, volume-averaged over incident elements.
/// An output field for visualisation, not a replacement for quadrature energies.
pub fn nodal_deformation(solid: &core_world::Solid) -> Vec<Mat3> {
    let mesh = &solid.mesh;
    let mut result = vec![Mat3::ZERO; mesh.rest.len()];
    let mut weights = vec![0.0; mesh.rest.len()];
    let volumes = mesh.volumes();
    for (e, &volume) in volumes.iter().enumerate() {
        let nodes = mesh.element(e);
        if mesh.is_quadratic() {
            let rest: [Vec3; 10] = std::array::from_fn(|a| mesh.rest[nodes[a] as usize]);
            for (local, &node) in nodes.iter().enumerate() {
                let mut bary = [0.0; 4];
                if local < 4 {
                    bary[local] = 1.0;
                } else {
                    let (a, b) = core_world::EDGES[local - 4];
                    bary[a] = 0.5;
                    bary[b] = 0.5;
                }
                let point = material_point(&rest, quadratic_gradients(bary), 1.0);
                let mut f = Mat3::ZERO;
                for (a, &n) in nodes.iter().enumerate() {
                    f += Mat3::outer(solid.nodes.position[n as usize], point.gradients[a]);
                }
                result[node as usize] += f * volume;
                weights[node as usize] += volume;
            }
        } else {
            let rest: [Vec3; 4] = std::array::from_fn(|a| mesh.rest[nodes[a] as usize]);
            let point = material_point(&rest, [Vec3::splat(-1.0), Vec3::X, Vec3::Y, Vec3::Z], 1.0);
            let mut f = Mat3::ZERO;
            for (a, &n) in nodes.iter().enumerate() {
                f += Mat3::outer(solid.nodes.position[n as usize], point.gradients[a]);
            }
            for &node in &nodes {
                result[node as usize] += f * volume;
                weights[node as usize] += volume;
            }
        }
    }
    for (f, w) in result.iter_mut().zip(weights) {
        if w > 0.0 {
            *f *= 1.0 / w;
        }
    }
    result
}
