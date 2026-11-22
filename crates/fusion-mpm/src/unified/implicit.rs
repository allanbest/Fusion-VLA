//! Soft solids that keep up with the frame instead of setting it.
//!
//! An explicit step has to resolve the fastest wave a material can carry,
//! and a solid's elastic waves are fast: a half-metre rubber block asked for
//! ten substeps a frame, porcelain fourteen, a ten-centimetre one sixty — and
//! every substep paid for every point in the scene, water and all. A point
//! the step outruns has its stress solved implicitly instead — backward
//! Euler, the stress at the end of the step:
//!
//! ```text
//!     m_i (v_i − v*_i) + h Σ_p V_p P(F_p(v)) F_pⁿᵀ ∇w_ip = 0,
//!     F_p(v) = (I + h Σ_i v_i ⊗ ∇w_ip) F_pⁿ
//! ```
//!
//! `v*` the grid's velocity without those points' stress, `P` the material's
//! own law — plastic return map, damage and all — so a yielding point yields
//! inside the solve. Newton's method, each step a conjugate-gradient solve of
//! `(M + h² K) δ = −r` with `K` the *warped* elastic stiffness, `δF ↦ R (2μ
//! dev ε + κ tr ε I)`, `ε = sym(Rᵀ δF)`: symmetric and positive semidefinite,
//! blind to rotation — a spinning solid feels nothing from it — and softened
//! where the point is yielding or damaged, as its real stiffness is there.
//!
//! The first version solved only that linear stiffness implicitly and left the
//! material's stress at the start of the step: stable for a spring, and not
//! for a block of porcelain landing on a floor, which came off it with 40%
//! more energy than it fell with — the stress lagged the compression on the
//! way in and pushed on it on the way out. At the end of the step it cannot.
//!
//! Backward Euler takes energy out of the modes the step cannot follow, which
//! is what makes it stable; points the step does follow stay explicit and
//! keep every bit of theirs.

use std::cell::RefCell;

use fusion_core::SolidSet;
use fusion_math::{outer_product, Mat3, Vec3};

use super::{
    chunk_ranges, field_of, parallel, stencil, weights, Block, BlockMap, Owner, Reach, BLOCK,
    CHUNKS,
};
use crate::linear::{Pcg, SolveSettings};
use crate::model::{hardened, return_map, stress, stress_with_damage};

/// No unknown.
const NO: u32 = u32::MAX;

/// Stiffness a yielding point keeps in what it yields in.
const YIELDED: f32 = 0.02;

/// Most Newton steps a substep takes.
const NEWTON: usize = 2;

/// Residual small enough to stop at, as a share of the momentum gravity gives
/// the heaviest node in a step.
const TOLERANCE: f64 = 1e-2;

/// Fewer stiff points than this are solved on one core.
const SERIAL_BELOW: usize = 2_000;

/// What one implicit solve did.
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct ImplicitReport {
    /// Grid velocities solved for.
    pub unknowns: usize,
    /// Solid points with any implicit stiffness.
    pub stiff_points: usize,
    /// Conjugate-gradient iterations.
    pub iterations: usize,
    /// Newton steps taken.
    pub newton: usize,
    /// Whether the residual met its tolerance.
    pub converged: bool,
}

/// A solid point's share of the operator.
#[derive(Clone, Debug)]
struct Point {
    unknowns: [u32; 27],
    gradients: [Vec3; 27],
    f: Mat3,
    rotation: Mat3,
    /// The principal axes of the point's stretch, and the square root of
    /// how much stiffness each keeps.
    frame: Mat3,
    keep: Vec3,
    /// Shear and bulk stiffness: `2μ dev ε + κ tr ε I`.
    shear: f32,
    bulk: f32,
    /// `V_p`.
    weight: f32,
    /// What the material's own stress needs.
    solid: u32,
    particle: u32,
}

/// Whether point `i` of `solid` is one a step of `h` outruns: its stress is
/// then the implicit solve's, and the transfer in leaves it out. Fibres keep
/// their explicit limit — see `control::estimate`.
pub(super) fn stiff(solid: &SolidSet, i: usize, cell: f32, h: f32) -> bool {
    if solid.params.fibre.is_some() {
        return false;
    }
    let material = solid.params.material;
    let (mu, lambda) = material.lame();
    let (mu, lambda) = hardened(material.model, mu, lambda, solid.plastic[i]);
    let wave = ((lambda + 2.0 * mu) / material.density.max(1e-6)).sqrt();
    let rate = super::control::norm(solid.affine[i]);
    let explicit = 0.4 * cell / (wave + solid.particles.velocity[i].length() + cell * rate);
    explicit < h
}

/// The material's own stress at deformation `f`, as `P`: return map,
/// hardening and damage, as the transfer in computes it.
fn stress_at(solid: &SolidSet, i: usize, f: Mat3) -> Mat3 {
    let material = solid.params.material;
    let (elastic, _) = return_map(material.model, f);
    let (mu, lambda) = material.lame();
    let (mu, lambda) = hardened(material.model, mu, lambda, solid.plastic[i]);
    if material.fracture.breaks() {
        stress_with_damage(material.model, elastic, mu, lambda, solid.damage[i])
    } else {
        stress(material.model, elastic, mu, lambda)
    }
}

/// Reusable workspace.
#[derive(Clone, Debug, Default)]
pub(super) struct Implicit {
    /// Per block, node and slot: its unknown.
    index: Vec<[[u32; 2]; BLOCK]>,
    /// Per unknown: block, node, slot, and whether both slots share it.
    owner: Vec<(u32, u8, u8, bool)>,
    mass: Vec<f64>,
    points: Vec<Point>,
    rhs: Vec<f64>,
    x: Vec<f64>,
    diagonal: Vec<f64>,
    stress: RefCell<Vec<Mat3>>,
    /// Per unknown, the (point, stencil node) pairs that reach it, so that a
    /// force can be gathered on every core rather than scattered on one.
    reach_start: Vec<u32>,
    reached_by: Vec<(u32, u8)>,
    /// What the last solve did to each node's velocity, by block key, and
    /// the substep it did it in: the next solve starts from it.
    previous: BlockMap<[[Vec3; 2]; BLOCK]>,
    previous_h: f32,
    pcg: Pcg,
}

impl Implicit {
    /// Solve the solids' stiffness implicitly on the grid's velocities, which
    /// `blocks` holds after the explicit update.
    #[allow(clippy::too_many_arguments)]
    pub(super) fn solve(
        &mut self,
        blocks: &mut [Block],
        keys: &[[i64; 3]],
        index: &BlockMap<u32>,
        solids: &[SolidSet],
        order: &[Owner],
        cell: f32,
        h: f32,
        floor: f32,
        separate: bool,
        threads: usize,
        settings: SolveSettings,
    ) -> ImplicitReport {
        let d_inv = 4.0 / (cell * cell);
        self.index.clear();
        self.index.resize(blocks.len(), [[NO; 2]; BLOCK]);
        self.owner.clear();
        self.mass.clear();
        self.points.clear();
        let mut cached: Option<Reach> = None;
        for &owner in order {
            let Owner::Solid(s, i) = owner else {
                continue;
            };
            let solid = &solids[s as usize];
            let i = i as usize;
            let material = solid.params.material;
            if !stiff(solid, i, cell, h) {
                continue;
            }
            let (mu, lambda) = material.lame();
            let (mu, lambda) = hardened(material.model, mu, lambda, solid.plastic[i]);
            let f = solid.deformation[i];
            let (u, sigma, v) = f.svd();
            let rotation = u * v.transpose();
            let intact = if material.fracture.breaks() {
                (1.0 - solid.damage[i].clamp(0.0, 1.0)).max(0.0)
            } else {
                1.0
            };
            // Where the material is yielding it has no stiffness left in
            // what it yields in: a plastic point stiffened elastically sprang
            // back from an impact it should have absorbed, and an alloy block
            // that comes to rest explicitly bounced on implicitly.
            let mut shear = mu;
            let mut yielded = Vec3::ONE;
            match material.model {
                fusion_core::solid::Constitutive::VonMises { yield_strain } => {
                    let m = sigma.abs().max(Vec3::splat(1e-3));
                    let strain = Vec3::new(m.x.ln(), m.y.ln(), m.z.ln());
                    let mean = (strain.x + strain.y + strain.z) / 3.0;
                    if (strain - Vec3::splat(mean)).length() >= 0.95 * yield_strain.max(1e-5) {
                        shear *= YIELDED;
                    }
                }
                fusion_core::solid::Constitutive::Snow {
                    critical_compression,
                    critical_stretch,
                    ..
                } => {
                    let low = (1.0 - critical_compression).max(1e-3) * 1.01;
                    let high = (1.0 + critical_stretch) * 0.99;
                    let clamp = |s: f32| {
                        if s <= low || s >= high {
                            YIELDED.sqrt()
                        } else {
                            1.0
                        }
                    };
                    yielded = Vec3::new(clamp(sigma.x), clamp(sigma.y), clamp(sigma.z));
                }
                _ => {}
            }
            let soften = |s: f32| if s > 1.0 { intact.sqrt() } else { 1.0 };
            let keep = Vec3::new(
                soften(sigma.x) * yielded.x,
                soften(sigma.y) * yielded.y,
                soften(sigma.z) * yielded.z,
            );

            let p = solid.particles.position[i];
            let field = field_of(owner);
            let (base, fx) = stencil(p, cell);
            let reach = match cached {
                Some(mut r) if r.first == super::block_of(base) => {
                    r.base = base;
                    r
                }
                _ => Reach::new(base, index),
            };
            cached = Some(reach);
            let (wx, wy, wz) = (weights(fx.x), weights(fx.y), weights(fx.z));
            let mut point = Point {
                unknowns: [NO; 27],
                gradients: [Vec3::ZERO; 27],
                f,
                rotation,
                frame: v,
                keep,
                shear,
                bulk: lambda + 2.0 * mu / 3.0,
                weight: solid.volume[i],
                solid: s,
                particle: i as u32,
            };
            let mut at = 0;
            for (k, wk) in wz.iter().enumerate() {
                for (j, wj) in wy.iter().enumerate() {
                    for (ii, wi) in wx.iter().enumerate() {
                        let w = wi * wj * wk;
                        let dpos = (Vec3::new(ii as f32, j as f32, k as f32) - fx) * cell;
                        let (b, n) = reach.node(ii, j, k);
                        if b == u32::MAX {
                            at += 1;
                            continue;
                        }
                        let data = &blocks[b as usize];
                        let [m0, m1] = data.mass[n];
                        if m0 + m1 <= floor {
                            at += 1;
                            continue;
                        }
                        // Two continua the grid update has just put in contact
                        // move together through the solve: solved apart, the
                        // solid's stiffness moved it into the water it rested
                        // on, and a floating block sank; merged again after,
                        // the water dragged its edges and stretched it.
                        let interface = separate && m0 > floor && m1 > floor && !touching(data, n);
                        let slot = if interface { data.read(n, field) } else { 0 };
                        let mut u = self.index[b as usize][n][slot];
                        if u == NO {
                            u = self.owner.len() as u32;
                            self.owner.push((b, n as u8, slot as u8, !interface));
                            self.mass.push(if interface {
                                data.mass[n][slot] as f64
                            } else {
                                (m0 + m1) as f64
                            });
                            self.index[b as usize][n][slot] = u;
                            if !interface {
                                self.index[b as usize][n][1 - slot] = u;
                            }
                        }
                        point.unknowns[at] = u;
                        point.gradients[at] = dpos * (w * d_inv);
                        at += 1;
                    }
                }
            }
            self.points.push(point);
        }
        let unknowns = self.owner.len();
        if unknowns == 0 || self.points.is_empty() {
            return ImplicitReport {
                converged: true,
                ..Default::default()
            };
        }

        // ---- which points reach each unknown ----
        self.reach_start.clear();
        self.reach_start.resize(unknowns + 1, 0);
        for point in &self.points {
            for &u in &point.unknowns {
                if u != NO {
                    self.reach_start[u as usize + 1] += 1;
                }
            }
        }
        for u in 0..unknowns {
            self.reach_start[u + 1] += self.reach_start[u];
        }
        self.reached_by.clear();
        self.reached_by
            .resize(self.reach_start[unknowns] as usize, (0, 0));
        let mut fill = self.reach_start.clone();
        for (p, point) in self.points.iter().enumerate() {
            for (k, &u) in point.unknowns.iter().enumerate() {
                if u != NO {
                    self.reached_by[fill[u as usize] as usize] = (p as u32, k as u8);
                    fill[u as usize] += 1;
                }
            }
        }

        // ---- the explicit velocities, and a Jacobi diagonal ----
        let n3 = 3 * unknowns;
        self.rhs.clear();
        self.rhs.resize(n3, 0.0);
        self.x.clear();
        self.x.resize(n3, 0.0);
        self.diagonal.clear();
        self.diagonal.resize(n3, 0.0);
        let mut start = vec![0.0f64; n3];
        // Newton starts where the last solve ended up, in what it did to
        // each node, for a substep this long: a block resting on a floor
        // took three dozen iterations a step to find the same balance
        // against gravity it had found the step before.
        let carry = if self.previous_h > 0.0 {
            h / self.previous_h
        } else {
            0.0
        };
        for (u, &(b, n, slot, _)) in self.owner.iter().enumerate() {
            let v = blocks[b as usize].momentum[n as usize][slot as usize];
            let was = self
                .previous
                .get(&keys[b as usize])
                .map_or(Vec3::ZERO, |d| d[n as usize][slot as usize] * carry);
            let m = self.mass[u];
            for a in 0..3 {
                start[3 * u + a] = v[a] as f64;
                self.x[3 * u + a] = (v[a] + was[a]) as f64;
                self.diagonal[3 * u + a] = m;
            }
        }
        let h2 = (h * h) as f64;
        for point in &self.points {
            let stiff = point.weight * (2.0 * point.shear + point.bulk);
            for (k, &u) in point.unknowns.iter().enumerate() {
                if u == NO {
                    continue;
                }
                let g = point.f.transpose() * point.gradients[k];
                let d = h2 * (stiff * g.length_squared()) as f64;
                for a in 0..3 {
                    self.diagonal[3 * u as usize + a] += d;
                }
            }
        }
        self.stress
            .borrow_mut()
            .resize(self.points.len(), Mat3::ZERO);
        let ranges = chunk_ranges(self.points.len(), CHUNKS);
        let unknown_ranges = chunk_ranges(unknowns, CHUNKS);
        // Close enough: a hundredth of the momentum gravity gives the
        // heaviest node in a step, or of what it has, if more. Absolute, so
        // that a good start is a short solve: relative to where it started,
        // a solve takes as many iterations however close it starts.
        let tolerance = TOLERANCE
            * start
                .chunks(3)
                .zip(&self.mass)
                .map(|(v, m)| m * (v.iter().fold(0.0f64, |a, x| a.max(x.abs())) + 9.81 * h as f64))
                .fold(0.0f64, f64::max)
                .max(1e-12);
        let settings = SolveSettings {
            absolute_tolerance: settings.absolute_tolerance.max(0.5 * tolerance),
            ..settings
        };

        // ---- Newton ----
        let mut out = ImplicitReport {
            unknowns,
            stiff_points: self.points.len(),
            converged: false,
            ..Default::default()
        };
        let mut delta = vec![0.0f64; n3];
        // Starting threads costs more than a few thousand points take on
        // one core, and an iterative solve starts them every iteration.
        let threads = if self.points.len() < SERIAL_BELOW {
            1
        } else {
            threads
        };
        for _ in 0..NEWTON {
            let mut residual = self.residual(
                &self.x,
                &start,
                solids,
                h,
                &ranges,
                &unknown_ranges,
                threads,
            );
            if out.newton == 0 && carry > 0.0 {
                // The last solve's answer is a guess, and after an impact a
                // bad one — it holds the impact's one-off part, and two
                // Newton steps did not take it back out. Kept only where it
                // is nearer the answer than starting from nothing.
                let cold =
                    self.residual(&start, &start, solids, h, &ranges, &unknown_ranges, threads);
                let size = |r: &[f64]| r.iter().fold(0.0f64, |a, v| a.max(v.abs()));
                if size(&cold) < size(&residual) {
                    self.x.copy_from_slice(&start);
                    residual = cold;
                }
            }
            let size = residual.iter().fold(0.0f64, |a, r| a.max(r.abs()));
            if size <= tolerance {
                out.converged = true;
                break;
            }
            // The step: (M + h² K) δ = −r.
            for (r, d) in self.rhs.iter_mut().zip(&residual) {
                *r = -d;
            }
            delta.iter_mut().for_each(|d| *d = 0.0);
            let (points, mass, stress, reach_start, reached_by) = (
                &self.points,
                &self.mass,
                &self.stress,
                &self.reach_start,
                &self.reached_by,
            );
            let apply = |x: &[f64], y: &mut [f64]| {
                let runs = parallel(threads, ranges.len(), |run| {
                    ranges[run]
                        .clone()
                        .map(|p| {
                            let point = &points[p];
                            let g = gradient(point, x);
                            warped(point, g * point.f) * point.f.transpose() * point.weight
                        })
                        .collect::<Vec<Mat3>>()
                });
                let mut s = stress.borrow_mut();
                for (run, out) in runs.into_iter().enumerate() {
                    s[ranges[run].clone()].copy_from_slice(&out);
                }
                drop(s);
                for (u, m) in mass.iter().enumerate() {
                    for a in 0..3 {
                        y[3 * u + a] = m * x[3 * u + a];
                    }
                }
                gather(
                    points,
                    &stress.borrow(),
                    reach_start,
                    reached_by,
                    &unknown_ranges,
                    threads,
                    h2,
                    y,
                );
            };
            let Ok(solved) = self.pcg.solve_within_budget(
                &self.rhs,
                &self.diagonal,
                &mut delta,
                settings,
                apply,
            ) else {
                break;
            };
            out.iterations += solved.iterations;
            for (x, d) in self.x.iter_mut().zip(&delta) {
                *x += d;
            }
            out.newton += 1;
        }
        if !self.x.iter().all(|v| v.is_finite()) {
            out.converged = false;
            return out;
        }
        self.previous.clear();
        self.previous_h = h;
        for (u, &(b, n, slot, shared)) in self.owner.iter().enumerate() {
            let v = Vec3::new(
                self.x[3 * u] as f32,
                self.x[3 * u + 1] as f32,
                self.x[3 * u + 2] as f32,
            );
            let was = Vec3::new(
                start[3 * u] as f32,
                start[3 * u + 1] as f32,
                start[3 * u + 2] as f32,
            );
            let entry = self
                .previous
                .entry(keys[b as usize])
                .or_insert([[Vec3::ZERO; 2]; BLOCK]);
            if shared {
                entry[n as usize] = [v - was; 2];
            } else {
                entry[n as usize][slot as usize] = v - was;
            }
            let block = &mut blocks[b as usize];
            if shared {
                block.momentum[n as usize] = [v, v];
            } else {
                block.momentum[n as usize][slot as usize] = v;
            }
        }
        out
    }
}

/// Whether a node's two continua were just put in contact: the grid update
/// leaves the velocities of two that were closing equal along the interface
/// normal.
fn touching(data: &Block, n: usize) -> bool {
    let normal = (data.gradient[n][1] - data.gradient[n][0]).normalize_or_zero();
    if normal == Vec3::ZERO {
        return false;
    }
    let [v0, v1] = data.momentum[n];
    (v0 - v1).dot(normal).abs() <= 1e-5 * (1.0 + v0.length() + v1.length())
}

impl Implicit {
    /// The residual at velocities `x`: inertia against the explicit
    /// velocities `start`, and the material's stress at the end of the step,
    /// each point on its own core.
    #[allow(clippy::too_many_arguments)]
    fn residual(
        &self,
        x: &[f64],
        start: &[f64],
        solids: &[SolidSet],
        h: f32,
        ranges: &[std::ops::Range<usize>],
        unknown_ranges: &[std::ops::Range<usize>],
        threads: usize,
    ) -> Vec<f64> {
        let points = &self.points;
        let runs = parallel(threads, ranges.len(), |run| {
            ranges[run]
                .clone()
                .map(|p| {
                    let point = &points[p];
                    let g = gradient(point, x);
                    let f = (Mat3::IDENTITY + g * h) * point.f;
                    let solid = &solids[point.solid as usize];
                    stress_at(solid, point.particle as usize, f)
                        * point.f.transpose()
                        * (h * point.weight)
                })
                .collect::<Vec<Mat3>>()
        });
        let mut residual: Vec<f64> = x
            .iter()
            .zip(start)
            .enumerate()
            .map(|(i, (x, s))| self.mass[i / 3] * (x - s))
            .collect();
        {
            let mut st = self.stress.borrow_mut();
            for (run, out) in runs.into_iter().enumerate() {
                st[ranges[run].clone()].copy_from_slice(&out);
            }
        }
        gather(
            points,
            &self.stress.borrow(),
            &self.reach_start,
            &self.reached_by,
            unknown_ranges,
            threads,
            1.0,
            &mut residual,
        );
        residual
    }
}

/// `y_u += scale · Σ A_p ∇w_pk` over the points that reach each unknown, in a
/// fixed order, unknowns split across cores.
#[allow(clippy::too_many_arguments)]
fn gather(
    points: &[Point],
    stress: &[Mat3],
    reach_start: &[u32],
    reached_by: &[(u32, u8)],
    ranges: &[std::ops::Range<usize>],
    threads: usize,
    scale: f64,
    y: &mut [f64],
) {
    let runs = parallel(threads, ranges.len(), |run| {
        ranges[run]
            .clone()
            .map(|u| {
                let mut sum = Vec3::ZERO;
                for &(p, k) in &reached_by[reach_start[u] as usize..reach_start[u + 1] as usize] {
                    sum += stress[p as usize] * points[p as usize].gradients[k as usize];
                }
                sum
            })
            .collect::<Vec<Vec3>>()
    });
    for (run, sums) in runs.into_iter().enumerate() {
        for (u, f) in ranges[run].clone().zip(sums) {
            y[3 * u] += scale * f.x as f64;
            y[3 * u + 1] += scale * f.y as f64;
            y[3 * u + 2] += scale * f.z as f64;
        }
    }
}

/// The velocity gradient a point reads from the unknowns `x`.
fn gradient(point: &Point, x: &[f64]) -> Mat3 {
    let mut g = Mat3::ZERO;
    for (k, &u) in point.unknowns.iter().enumerate() {
        if u == NO {
            continue;
        }
        let u = 3 * u as usize;
        let v = Vec3::new(x[u] as f32, x[u + 1] as f32, x[u + 2] as f32);
        g += outer_product(v, point.gradients[k]);
    }
    g
}

/// The warped stiffness applied to a change in the deformation gradient:
/// `R (2μ ε + λ tr ε I)`, `ε = sym(Rᵀ δF)`, with each stretched axis's share
/// softened by the point's damage.
fn warped(point: &Point, delta: Mat3) -> Mat3 {
    let local = point.rotation.transpose() * delta;
    let strain = (local + local.transpose()) * 0.5;
    // In the principal frame, scaled on both sides so the map stays
    // symmetric.
    let v = point.frame;
    let s = point.keep;
    let scale = Mat3::from_diagonal(s);
    let principal = scale * (v.transpose() * strain * v) * scale;
    let trace = principal.x_axis.x + principal.y_axis.y + principal.z_axis.z;
    let deviatoric = principal - Mat3::IDENTITY * (trace / 3.0);
    let response = deviatoric * (2.0 * point.shear) + Mat3::IDENTITY * (point.bulk * trace);
    let back = v * (scale * response * scale) * v.transpose();
    point.rotation * back
}
