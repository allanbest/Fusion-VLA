//! Long-range mutual Newtonian gravity using [`core_space::BarnesHut`].

use core_math::constants::GRAVITATIONAL;
use core_math::{Mat3, Vec3};
use core_numerics::{BlockMatrix, Triplets};
use core_space::{BarnesHut, Interaction};
use core_world::{Representation, World};

use crate::dofs::Layout;
use crate::term::{Context, Term};

/// Configuration for N-body mutual Newtonian gravitation.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct MutualGravitySettings {
    /// Gravitational constant `G`, m³/(kg·s²). Defaults to [`GRAVITATIONAL`].
    pub constant: f64,
    /// Plummer softening radius `ε`, m.
    pub softening: f64,
    /// Barnes–Hut dual-tree opening angle `θ` (`0.0` evaluates all O(N²) pairs).
    pub theta: f64,
}

impl Default for MutualGravitySettings {
    fn default() -> Self {
        Self {
            constant: GRAVITATIONAL,
            softening: 1e-3,
            theta: 0.5,
        }
    }
}

#[derive(Clone, Debug, Default)]
pub struct MutualGravity {
    pub settings: Option<MutualGravitySettings>,
    /// `(block, mass)` matching `Gravity::weights`.
    pub weights: Vec<(usize, f64)>,
    positions: Vec<Vec3>,
    masses: Vec<f64>,
    tree: BarnesHut,
    direct: Vec<(u32, u32)>,
    cells: Vec<(u32, u32)>,
}

#[inline]
fn pair_terms(d: Vec3, ma: f64, mb: f64, g_scale: f64, eps2: f64) -> (Vec3, Mat3) {
    let r2 = d.length_squared();
    let s2 = r2 + eps2;
    let s = s2.sqrt();
    if s <= 0.0 {
        return (Vec3::ZERO, Mat3::ZERO);
    }
    let s3 = s * s2;
    let factor = g_scale * ma * mb / s3;
    let grad = d * factor;
    let hess = if r2 > 0.0 {
        let u_outer = Mat3::outer(d, d) * (1.0 / r2);
        let radial = (1.0 - 3.0 * r2 / s2).max(0.0);
        ((Mat3::IDENTITY - u_outer) + u_outer * radial) * factor
    } else {
        Mat3::scalar(factor)
    };
    (grad, hess)
}

impl MutualGravity {
    /// Exact pairwise physical mutual potential energy of all dynamic masses
    /// in `world` (`-Σ G m_i m_j / sqrt(r² + ε²)`), J.
    pub(crate) fn world_energy(
        settings: Option<MutualGravitySettings>,
        world: &World,
        layout: &Layout,
    ) -> f64 {
        let Some(settings) = settings else {
            return 0.0;
        };
        if settings.constant == 0.0 {
            return 0.0;
        }
        let mut points: Vec<(Vec3, f64)> = Vec::new();
        for d in &layout.rigids {
            let Representation::Rigid(r) = &world.objects()[d.object].representation else {
                unreachable!()
            };
            if r.mass > 0.0 {
                points.push((r.pose.position, r.mass));
            }
        }
        for d in &layout.solids {
            let Representation::Solid(s) = &world.objects()[d.object].representation else {
                unreachable!()
            };
            for (&p, &m) in s.nodes.position.iter().zip(&s.nodes.mass) {
                if m > 0.0 {
                    points.push((p, m));
                }
            }
        }
        let eps2 = settings.softening * settings.softening;
        let g = settings.constant;
        let mut energy = 0.0;
        for i in 0..points.len() {
            let (pi, mi) = points[i];
            for &(pj, mj) in &points[i + 1..] {
                let s = ((pi - pj).length_squared() + eps2).sqrt();
                if s > 0.0 {
                    energy -= g * mi * mj / s;
                }
            }
        }
        energy
    }

    fn live_coms(&self, x: &[f64]) -> Vec<Vec3> {
        let count = self.tree.cell_count();
        if count == 0 || self.cells.is_empty() {
            return Vec::new();
        }
        let mut coms = vec![Vec3::ZERO; count];
        for idx in (0..count).rev() {
            let c = idx as u32;
            let m = self.tree.cell_mass(c);
            if m <= 0.0 {
                continue;
            }
            let sum = if self.tree.cell_is_leaf(c) {
                self.tree
                    .cell_members(c)
                    .iter()
                    .map(|&i| Vec3::read(x, self.weights[i].0) * self.masses[i])
                    .sum()
            } else {
                let mut acc = Vec3::ZERO;
                for &ch in &self.tree.cell_children(c) {
                    if ch != u32::MAX {
                        acc += coms[ch as usize] * self.tree.cell_mass(ch);
                    }
                }
                acc
            };
            coms[idx] = sum * (1.0 / m);
        }
        coms
    }

    /// Physical gradient in N (unit weight) for second-order startup.
    pub(crate) fn physical_gradient(&self, ctx: &Context, x: &[f64], out: &mut [f64]) {
        let Some(settings) = self.settings else {
            return;
        };
        if settings.constant == 0.0 || self.weights.len() < 2 {
            return;
        }
        let eps2 = settings.softening * settings.softening;
        let g = settings.constant;
        for &(ia, ib) in &self.direct {
            let (ia, ib) = (ia as usize, ib as usize);
            let (ba, ma) = self.weights[ia];
            let (bb, mb) = self.weights[ib];
            let d = Vec3::read(x, ba) - Vec3::read(x, bb);
            let s2 = d.length_squared() + eps2;
            let s = s2.sqrt();
            if s > 0.0 {
                let grad = d * (g * ma * mb / (s * s2));
                grad.add_to(out, ba);
                (-grad).add_to(out, bb);
            }
        }
        if !self.cells.is_empty() {
            let coms = self.live_coms(x);
            let cell_count = self.tree.cell_count();
            let mut cell_grad = vec![Vec3::ZERO; cell_count];
            for &(ca, cb) in &self.cells {
                let (ma, mb) = (self.tree.cell_mass(ca), self.tree.cell_mass(cb));
                let d = coms[ca as usize] - coms[cb as usize];
                let s2 = d.length_squared() + eps2;
                let s = s2.sqrt();
                if s > 0.0 {
                    let grad = d * (g * ma * mb / (s * s2));
                    cell_grad[ca as usize] += grad;
                    cell_grad[cb as usize] -= grad;
                }
            }
            let mut inv_m_grad = vec![Vec3::ZERO; cell_count];
            for idx in 0..cell_count {
                let c = idx as u32;
                let m = self.tree.cell_mass(c);
                if m > 0.0 {
                    inv_m_grad[idx] += cell_grad[idx] * (1.0 / m);
                }
                let parent_g = inv_m_grad[idx];
                if self.tree.cell_is_leaf(c) {
                    for &i in self.tree.cell_members(c) {
                        (parent_g * self.masses[i]).add_to(out, self.weights[i].0);
                    }
                } else {
                    for &ch in &self.tree.cell_children(c) {
                        if ch != u32::MAX {
                            inv_m_grad[ch as usize] += parent_g;
                        }
                    }
                }
            }
        }
        let _ = ctx;
    }
}

impl Term for MutualGravity {
    fn name(&self) -> &'static str {
        "mutual gravity"
    }

    fn begin_step(&mut self, ctx: &Context) {
        self.direct.clear();
        self.cells.clear();
        let Some(settings) = self.settings else {
            return;
        };
        if settings.constant == 0.0 || self.weights.len() < 2 {
            return;
        }
        self.positions.clear();
        self.masses.clear();
        for &(b, m) in &self.weights {
            self.positions.push(Vec3::read(&ctx.frame.start, b));
            self.masses.push(m);
        }
        self.tree = BarnesHut::build(&self.positions, &self.masses);
        let direct = &mut self.direct;
        let cells = &mut self.cells;
        self.tree.for_each_pair(settings.theta, |pair| match pair {
            Interaction::Direct(i, j) => direct.push((i as u32, j as u32)),
            Interaction::Cells { a, b } => cells.push((a, b)),
        });
    }

    fn energy(&mut self, ctx: &Context, x: &[f64]) -> f64 {
        let Some(settings) = self.settings else {
            return 0.0;
        };
        if settings.constant == 0.0 || self.weights.len() < 2 {
            return 0.0;
        }
        let g_scale = settings.constant * ctx.frame.weight;
        let eps2 = settings.softening * settings.softening;
        let direct_e = ctx.pool.map_reduce(
            self.direct.len(),
            256,
            0.0,
            |range| {
                range
                    .map(|k| {
                        let (ia, ib) = self.direct[k];
                        let (ba, ma) = self.weights[ia as usize];
                        let (bb, mb) = self.weights[ib as usize];
                        let s = ((Vec3::read(x, ba) - Vec3::read(x, bb)).length_squared() + eps2)
                            .sqrt();
                        if s > 0.0 {
                            -g_scale * ma * mb / s
                        } else {
                            0.0
                        }
                    })
                    .sum::<f64>()
            },
            |a, b| a + b,
        );
        if self.cells.is_empty() {
            return direct_e;
        }
        let coms = self.live_coms(x);
        let cell_e: f64 = self
            .cells
            .iter()
            .map(|&(ca, cb)| {
                let (ma, mb) = (self.tree.cell_mass(ca), self.tree.cell_mass(cb));
                let s = ((coms[ca as usize] - coms[cb as usize]).length_squared() + eps2).sqrt();
                if s > 0.0 {
                    -g_scale * ma * mb / s
                } else {
                    0.0
                }
            })
            .sum();
        direct_e + cell_e
    }

    fn derivatives(
        &mut self,
        ctx: &Context,
        x: &[f64],
        gradient: &mut [f64],
        _fixed: &mut BlockMatrix,
        changing: &mut Triplets,
    ) {
        let Some(settings) = self.settings else {
            return;
        };
        if settings.constant == 0.0 || self.weights.len() < 2 {
            return;
        }
        let g_scale = settings.constant * ctx.frame.weight;
        let eps2 = settings.softening * settings.softening;

        let pair_parts = ctx.pool.map_reduce(
            self.direct.len(),
            256,
            Vec::new(),
            |range| {
                let mut out = Vec::with_capacity(range.len());
                for k in range {
                    let (ia, ib) = self.direct[k];
                    let (ba, ma) = self.weights[ia as usize];
                    let (bb, mb) = self.weights[ib as usize];
                    let d = Vec3::read(x, ba) - Vec3::read(x, bb);
                    let (grad, hess) = pair_terms(d, ma, mb, g_scale, eps2);
                    out.push((ba, bb, grad, hess));
                }
                out
            },
            |mut a, b| {
                a.extend(b);
                a
            },
        );
        for (ba, bb, grad, hess) in pair_parts {
            grad.add_to(gradient, ba);
            (-grad).add_to(gradient, bb);
            changing.add(ba, ba, hess);
            changing.add(bb, bb, hess);
            changing.add(ba, bb, -hess);
            changing.add(bb, ba, -hess);
        }

        if !self.cells.is_empty() {
            let coms = self.live_coms(x);
            let cell_count = self.tree.cell_count();
            let mut cell_grad = vec![Vec3::ZERO; cell_count];
            let mut cell_hess = vec![Mat3::ZERO; cell_count];
            for &(ca, cb) in &self.cells {
                let (ma, mb) = (self.tree.cell_mass(ca), self.tree.cell_mass(cb));
                let d = coms[ca as usize] - coms[cb as usize];
                let (grad, hess) = pair_terms(d, ma, mb, g_scale, eps2);
                cell_grad[ca as usize] += grad;
                cell_grad[cb as usize] -= grad;
                cell_hess[ca as usize] += hess * (1.0 / (ma * ma));
                cell_hess[cb as usize] += hess * (1.0 / (mb * mb));
            }
            let mut inv_m_grad = vec![Vec3::ZERO; cell_count];
            let mut diag_per_m2 = vec![Mat3::ZERO; cell_count];
            for idx in 0..cell_count {
                let c = idx as u32;
                let m = self.tree.cell_mass(c);
                if m > 0.0 {
                    inv_m_grad[idx] += cell_grad[idx] * (1.0 / m);
                }
                diag_per_m2[idx] += cell_hess[idx];
                let parent_g = inv_m_grad[idx];
                let parent_h = diag_per_m2[idx];
                if self.tree.cell_is_leaf(c) {
                    for &i in self.tree.cell_members(c) {
                        let (bi, mi) = self.weights[i];
                        (parent_g * mi).add_to(gradient, bi);
                        changing.add(bi, bi, parent_h * (mi * mi));
                    }
                } else {
                    for &ch in &self.tree.cell_children(c) {
                        if ch != u32::MAX {
                            inv_m_grad[ch as usize] += parent_g;
                            diag_per_m2[ch as usize] += parent_h;
                        }
                    }
                }
            }
        }
    }
}
