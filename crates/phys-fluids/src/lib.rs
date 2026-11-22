//! Layer 5c of the Fusion core: deterministic weakly-compressible Smoothed
//! Particle Hydrodynamics (WCSPH) with two-way fluid-structure coupling.
//!
//! Uses Wendland $C^2$ kernels over [`core_space::NeighbourGrid`] for bitwise
//! thread-count invariant parallel execution on [`core_runtime::Pool`].

use core_math::{Pose, Vec3};
use core_runtime::Pool;
use core_space::{Aabb, NeighbourGrid, Shape};
use core_world::{Representation, World};
use std::f64::consts::PI;

const PARTICLES_PER_CHUNK: usize = 64;

/// State arrays for a cloud of Lagrangian SPH fluid particles.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct FluidParticles {
    /// Particle world positions, m.
    pub positions: Vec<Vec3>,
    /// Particle linear velocities, m/s.
    pub velocities: Vec<Vec3>,
    /// Per-particle mass, kg.
    pub masses: Vec<f64>,
    /// Last evaluated SPH mass density, kg/m³.
    pub densities: Vec<f64>,
    /// Last evaluated Tait EOS pressure, Pa.
    pub pressures: Vec<f64>,
}

impl FluidParticles {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn len(&self) -> usize {
        self.positions.len()
    }

    pub fn is_empty(&self) -> bool {
        self.positions.is_empty()
    }

    /// Creates a regular axis-aligned particle lattice filling `bounds` at
    /// particle spacing `spacing` and nominal density `rest_density`.
    pub fn block(bounds: Aabb, spacing: f64, rest_density: f64) -> Self {
        assert!(spacing > 0.0 && rest_density > 0.0);
        let size = bounds.size();
        let nx = ((size.x / spacing).floor() as usize).max(1);
        let ny = ((size.y / spacing).floor() as usize).max(1);
        let nz = ((size.z / spacing).floor() as usize).max(1);
        let mass = rest_density * spacing * spacing * spacing;
        let n = nx * ny * nz;
        let mut positions = Vec::with_capacity(n);
        for k in 0..nz {
            let z = bounds.min.z + (k as f64 + 0.5) * spacing;
            for j in 0..ny {
                let y = bounds.min.y + (j as f64 + 0.5) * spacing;
                for i in 0..nx {
                    let x = bounds.min.x + (i as f64 + 0.5) * spacing;
                    positions.push(Vec3::new(x, y, z));
                }
            }
        }
        Self {
            positions,
            velocities: vec![Vec3::ZERO; n],
            masses: vec![mass; n],
            densities: vec![rest_density; n],
            pressures: vec![0.0; n],
        }
    }

    /// Total fluid kinetic energy, J.
    pub fn kinetic_energy(&self) -> f64 {
        self.velocities
            .iter()
            .zip(&self.masses)
            .map(|(v, &m)| 0.5 * m * v.length_squared())
            .sum()
    }
}

/// Physical and numerical parameters for [`FluidSolver`].
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct FluidSettings {
    /// Reference rest density `ρ₀`, kg/m³.
    pub rest_density: f64,
    /// Compact Wendland $C^2$ support radius `h`, m (typically ~2.2 × spacing).
    pub smoothing_radius: f64,
    /// Numerical speed of sound `c₀` in Tait's weakly-compressible EOS, m/s.
    pub sound_speed: f64,
    /// Tait EOS polytropic exponent `γ` (typically 7.0 for water).
    pub gamma: f64,
    /// Artificial dynamic shear viscosity coefficient `μ`, Pa·s.
    pub viscosity: f64,
    /// XSPH velocity smoothing factor `[0, 0.5]` for coherent free surfaces.
    pub xsph: f64,
    /// Optional rigid axis-aligned tank domain confining the fluid.
    pub domain: Option<Aabb>,
    /// Normal velocity restitution `[0, 1]` against `domain` boundaries.
    pub domain_restitution: f64,
    /// Two-way FSI contact stiffness multiplier relative to fluid acoustic stiffness `(c₀/h)²`.
    pub fsi_stiffness: f64,
    /// CFL safety factor controlling acoustic/convective substep count.
    pub cfl: f64,
}

impl Default for FluidSettings {
    fn default() -> Self {
        Self {
            rest_density: 1000.0,
            smoothing_radius: 0.08,
            sound_speed: 20.0,
            gamma: 7.0,
            viscosity: 0.02,
            xsph: 0.05,
            domain: None,
            domain_restitution: 0.1,
            fsi_stiffness: 2.0,
            cfl: 0.25,
        }
    }
}

/// Summary statistics from a fluid simulation step.
#[derive(Clone, Copy, Debug, Default)]
pub struct FluidReport {
    pub substeps: usize,
    pub max_density: f64,
    pub max_pressure: f64,
    pub max_speed: f64,
    pub kinetic_energy: f64,
}

/// Wendland $C^2$ 3D kernel value and radial derivative factor `(W, (dW/dr) / r)`.
#[inline]
fn wendland_c2(r: f64, h: f64) -> (f64, f64) {
    let q = r / h;
    if q >= 1.0 {
        return (0.0, 0.0);
    }
    let sigma = 21.0 / (2.0 * PI * h * h * h);
    let one_q = 1.0 - q;
    let one_q2 = one_q * one_q;
    let one_q3 = one_q2 * one_q;
    let w = sigma * one_q2 * one_q2 * (1.0 + 4.0 * q);
    let grad_factor = sigma * (-20.0 * one_q3) / (h * h);
    (w, grad_factor)
}

#[derive(Clone, Copy, Debug)]
struct Obstacle {
    object: usize,
    dynamic: bool,
    mass: f64,
    shape: Shape,
    pose: Pose,
    velocity: Vec3,
    spin: Vec3,
}

impl Obstacle {
    fn distance(&self, p_world: Vec3) -> (f64, Vec3) {
        let local = self.pose.inverse_transform_point(p_world);
        let (sdf, local_n) = self.shape.distance(local);
        let world_n = self.pose.rotation.rotate(local_n);
        (sdf, world_n)
    }

    fn velocity_at(&self, p_world: Vec3) -> Vec3 {
        if self.dynamic {
            self.velocity + self.spin.cross(p_world - self.pose.position)
        } else {
            Vec3::ZERO
        }
    }
}

/// Deterministic `f64` weakly-compressible SPH fluid solver.
#[derive(Clone, Debug)]
pub struct FluidSolver {
    pub settings: FluidSettings,
    cell: f64,
    grid: NeighbourGrid,
    accel: Vec<Vec3>,
    xsph_delta: Vec<Vec3>,
    body_impulses: Vec<(Vec3, Vec3)>,
    per_particle_fsi: Vec<(usize, Vec3, Vec3)>,
}

impl Default for FluidSolver {
    fn default() -> Self {
        Self::new(FluidSettings::default())
    }
}

impl FluidSolver {
    pub fn new(settings: FluidSettings) -> Self {
        let cell = settings.smoothing_radius.max(1e-4);
        Self {
            settings,
            cell,
            grid: NeighbourGrid::build(&[], cell),
            accel: Vec::new(),
            xsph_delta: Vec::new(),
            body_impulses: Vec::new(),
            per_particle_fsi: Vec::new(),
        }
    }

    /// Advances free `fluid` particles under uniform `gravity` for `h` seconds.
    pub fn step(
        &mut self,
        pool: &Pool,
        fluid: &mut FluidParticles,
        gravity: Vec3,
        h: f64,
    ) -> FluidReport {
        self.step_impl(pool, fluid, None, gravity, h)
    }

    /// Advances `fluid` particles coupled two-way against shapes and dynamic
    /// rigid bodies in `world`.
    ///
    /// Imparts hydrodynamic and buoyancy impulses onto dynamic
    /// [`Representation::Rigid`] objects in `world` while leaving body gravity
    /// and rigid-rigid contacts to `phys_mechanics::Mechanics`.
    pub fn step_coupled(
        &mut self,
        pool: &Pool,
        fluid: &mut FluidParticles,
        world: &mut World,
        h: f64,
    ) -> FluidReport {
        let gravity = world.gravity;
        self.step_impl(pool, fluid, Some(world), gravity, h)
    }

    fn step_impl(
        &mut self,
        pool: &Pool,
        fluid: &mut FluidParticles,
        mut world: Option<&mut World>,
        gravity: Vec3,
        h: f64,
    ) -> FluidReport {
        let n = fluid.len();
        if n == 0 || !h.is_finite() || h <= 0.0 {
            return FluidReport::default();
        }
        let h_s = self.settings.smoothing_radius.max(1e-6);
        let vmax = fluid
            .velocities
            .iter()
            .map(|v| v.length())
            .fold(0.0_f64, f64::max);
        let dt_max = (self.settings.cfl * h_s / (self.settings.sound_speed + vmax)).max(1e-6);
        let substeps = ((h / dt_max).ceil() as usize).clamp(1, 64);
        let dt = h / (substeps as f64);

        if fluid.densities.len() != n {
            fluid.densities.resize(n, self.settings.rest_density);
        }
        if fluid.pressures.len() != n {
            fluid.pressures.resize(n, 0.0);
        }
        self.accel.resize(n, Vec3::ZERO);
        self.xsph_delta.resize(n, Vec3::ZERO);

        if (self.cell - h_s).abs() > 1e-12 {
            self.cell = h_s;
            self.grid = NeighbourGrid::build(&fluid.positions, h_s);
        }

        let b_tait =
            self.settings.rest_density * self.settings.sound_speed * self.settings.sound_speed
                / self.settings.gamma.max(1.0);
        let rho0 = self.settings.rest_density.max(1e-6);
        let visc_coef = self.settings.viscosity.max(0.0);
        let xsph_eps = self.settings.xsph.clamp(0.0, 0.5);
        let fsi_shell = 0.45 * h_s;
        let omega_fsi = self.settings.sound_speed / h_s;
        let k_fsi = 0.08 * self.settings.fsi_stiffness.max(0.0) * omega_fsi * omega_fsi;
        let c_fsi = 0.8 * self.settings.fsi_stiffness.max(0.0).sqrt() * omega_fsi;

        for _sub in 0..substeps {
            self.grid.rebuild(&fluid.positions);

            let obstacles: Vec<Obstacle> = world
                .as_deref()
                .map(|w| {
                    w.objects()
                        .iter()
                        .enumerate()
                        .filter_map(|(idx, obj)| match &obj.representation {
                            Representation::Fixed(f) => Some(Obstacle {
                                object: idx,
                                dynamic: false,
                                mass: f64::INFINITY,
                                shape: f.shape,
                                pose: f.pose,
                                velocity: Vec3::ZERO,
                                spin: Vec3::ZERO,
                            }),
                            Representation::Rigid(r) => Some(Obstacle {
                                object: idx,
                                dynamic: true,
                                mass: r.mass.max(1e-9),
                                shape: r.shape,
                                pose: r.pose,
                                velocity: r.velocity,
                                spin: r.spin,
                            }),
                            Representation::Solid(_) => None,
                        })
                        .collect()
                })
                .unwrap_or_default();

            let positions = &fluid.positions;
            let velocities = &fluid.velocities;
            let masses = &fluid.masses;
            let grid = &self.grid;
            let domain = self.settings.domain;
            let gamma = self.settings.gamma;

            // 1. Density & Tait pressure (with half-space boundary kernel compensation).
            pool.split_reduce(
                (&mut fluid.densities[..], &mut fluid.pressures[..]),
                PARTICLES_PER_CHUNK,
                (),
                |offset, (rho_out, p_out)| {
                    let w_self = wendland_c2(0.0, h_s).0;
                    for (local, (rho_dst, p_dst)) in
                        rho_out.iter_mut().zip(p_out.iter_mut()).enumerate()
                    {
                        let i = offset + local;
                        let pi = positions[i];
                        let mut rho = masses[i] * w_self;
                        let mut neighbors = 0usize;
                        grid.within(positions, pi, h_s, |j| {
                            if j != i {
                                let r = (pi - positions[j]).length();
                                let (w, _) = wendland_c2(r, h_s);
                                if w > 0.0 {
                                    rho += masses[j] * w;
                                    neighbors += 1;
                                }
                            }
                        });
                        if neighbors >= 2 {
                            let mut min_wall_dist = f64::INFINITY;
                            if let Some(dom) = domain {
                                for axis in 0..3 {
                                    min_wall_dist = min_wall_dist
                                        .min((pi[axis] - dom.min[axis]).max(0.0))
                                        .min((dom.max[axis] - pi[axis]).max(0.0));
                                }
                            }
                            for obs in &obstacles {
                                let (sdf, _) = obs.distance(pi);
                                min_wall_dist = min_wall_dist.min(sdf.max(0.0));
                            }
                            if min_wall_dist < 0.85 * h_s {
                                let t = (1.0 - min_wall_dist / (0.85 * h_s)).clamp(0.0, 1.0);
                                rho += 0.45 * rho0 * t * t;
                            }
                        }
                        let ratio = (rho / rho0).max(1.0);
                        *rho_dst = rho;
                        *p_dst = (b_tait * (ratio.powf(gamma) - 1.0)).max(0.0);
                    }
                },
                |_, _| (),
            );

            // 2. Pressure, viscosity, XSPH & particle-body FSI forces.
            let densities = &fluid.densities;
            let pressures = &fluid.pressures;
            let num_objs = world.as_deref().map_or(0, |w| w.objects().len());
            self.body_impulses.clear();
            self.body_impulses
                .resize(num_objs, (Vec3::ZERO, Vec3::ZERO));
            let mut per_particle_fsi = std::mem::take(&mut self.per_particle_fsi);
            per_particle_fsi.resize(n, (usize::MAX, Vec3::ZERO, Vec3::ZERO));
            let has_obstacles = !obstacles.is_empty();

            pool.split_reduce(
                (
                    &mut self.accel[..],
                    &mut self.xsph_delta[..],
                    &mut per_particle_fsi[..],
                ),
                PARTICLES_PER_CHUNK,
                (),
                |offset, (acc_out, xsph_out, fsi_out)| {
                    for (local, ((acc_dst, xsph_dst), fsi_slot)) in acc_out
                        .iter_mut()
                        .zip(xsph_out.iter_mut())
                        .zip(fsi_out.iter_mut())
                        .enumerate()
                    {
                        let i = offset + local;
                        let pi = positions[i];
                        let vi = velocities[i];
                        let rho_i = densities[i].max(1e-6);
                        let p_i = pressures[i];
                        let p_over_rho2_i = p_i / (rho_i * rho_i);

                        let mut a = gravity;
                        let mut dv_xsph = Vec3::ZERO;
                        grid.within(positions, pi, h_s, |j| {
                            if j == i {
                                return;
                            }
                            let diff = pi - positions[j];
                            let r2 = diff.length_squared();
                            if r2 <= 1e-24 || r2 >= h_s * h_s {
                                return;
                            }
                            let r = r2.sqrt();
                            let (w, grad_f) = wendland_c2(r, h_s);
                            let grad_w = diff * grad_f;
                            let rho_j = densities[j].max(1e-6);
                            let p_j = pressures[j];
                            let mj = masses[j];
                            let p_term = p_over_rho2_i + p_j / (rho_j * rho_j);
                            a -= grad_w * (mj * p_term);

                            if visc_coef > 0.0 {
                                let vij = vi - velocities[j];
                                let mu_factor = 2.0 * visc_coef * mj / (rho_i * rho_j);
                                let lap = -grad_f;
                                a -= vij * (mu_factor * lap * 0.5);
                            }

                            if xsph_eps > 0.0 && w > 0.0 {
                                let rho_avg = 0.5 * (rho_i + rho_j);
                                dv_xsph += (velocities[j] - vi) * (xsph_eps * mj / rho_avg * w);
                            }
                        });

                        *fsi_slot = (usize::MAX, Vec3::ZERO, Vec3::ZERO);
                        if has_obstacles {
                            let mut best_pen = 0.0;
                            let mut best_hit: Option<&Obstacle> = None;
                            for obs in &obstacles {
                                let (sdf, _) = obs.distance(pi);
                                let pen = fsi_shell - sdf;
                                if pen > best_pen {
                                    best_pen = pen;
                                    best_hit = Some(obs);
                                }
                            }
                            if let Some(obs) = best_hit {
                                let (_, n_out) = obs.distance(pi);
                                let v_body = obs.velocity_at(pi);
                                let v_rel = vi - v_body;
                                let vn = v_rel.dot(n_out);
                                let mass_scale = if obs.mass.is_finite() {
                                    obs.mass / (obs.mass + 8.0 * masses[i])
                                } else {
                                    1.0
                                };
                                let fn_acc = ((k_fsi * best_pen - c_fsi * vn.min(0.0))
                                    * mass_scale)
                                    .max(0.0);
                                let vt = v_rel - n_out * vn;
                                let drag_rate = (0.05 * omega_fsi).min(0.2 / dt);
                                let f_acc = n_out * fn_acc - vt * (drag_rate * mass_scale);
                                a += f_acc;
                                if obs.dynamic {
                                    let impulse = (-f_acc) * (masses[i] * dt);
                                    let ang_impulse = (pi - obs.pose.position).cross(impulse);
                                    *fsi_slot = (obs.object, impulse, ang_impulse);
                                }
                            }
                        }

                        *acc_dst = a;
                        *xsph_dst = dv_xsph;
                    }
                },
                |_, _| (),
            );

            // Deterministic integration & optional domain box walls.
            for i in 0..n {
                let mut v = fluid.velocities[i] + self.accel[i] * dt + self.xsph_delta[i];
                let mut p = fluid.positions[i] + v * dt;
                if let Some(dom) = self.settings.domain {
                    let e = self.settings.domain_restitution.clamp(0.0, 1.0);
                    for axis in 0..3 {
                        if p[axis] < dom.min[axis] {
                            p[axis] = dom.min[axis];
                            if v[axis] < 0.0 {
                                v[axis] *= -e;
                            }
                        } else if p[axis] > dom.max[axis] {
                            p[axis] = dom.max[axis];
                            if v[axis] > 0.0 {
                                v[axis] *= -e;
                            }
                        }
                    }
                }
                fluid.positions[i] = p;
                fluid.velocities[i] = v;
            }

            // Deterministically reduce FSI impulses onto World rigid bodies in index order.
            if let Some(w) = world.as_deref_mut() {
                for &(obj_idx, lin_imp, ang_imp) in &per_particle_fsi {
                    if obj_idx < self.body_impulses.len() {
                        self.body_impulses[obj_idx].0 += lin_imp;
                        self.body_impulses[obj_idx].1 += ang_imp;
                    }
                }
                for (obj_idx, &(lin_imp, ang_imp)) in self.body_impulses.iter().enumerate() {
                    if lin_imp == Vec3::ZERO && ang_imp == Vec3::ZERO {
                        continue;
                    }
                    if let Representation::Rigid(r) = &mut w.objects_mut()[obj_idx].representation {
                        if r.mass > 0.0 {
                            r.velocity += lin_imp / r.mass;
                            r.spin += r.world_inertia().inverse() * ang_imp;
                        }
                    }
                }
            }
            self.per_particle_fsi = per_particle_fsi;
        }

        FluidReport {
            substeps,
            max_density: fluid.densities.iter().copied().fold(0.0_f64, f64::max),
            max_pressure: fluid.pressures.iter().copied().fold(0.0_f64, f64::max),
            max_speed: fluid
                .velocities
                .iter()
                .map(|v| v.length())
                .fold(0.0_f64, f64::max),
            kinetic_energy: fluid.kinetic_energy(),
        }
    }
}
