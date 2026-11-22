//! Batched parallel physics environments and deterministic domain
//! randomization for robotics RL and VLA policy data collection.

use core_math::Vec3;
use core_runtime::Pool;
use core_world::{Link, LinkId, ProbeReading, Representation, World};

use crate::{Mechanics, Settings, StepReport};

/// Deterministic in-place domain randomization specification applied on episode reset.
#[derive(Clone, Debug, PartialEq)]
pub struct DomainRandomizer {
    /// Master PRNG seed (combined deterministically with environment index).
    pub seed: u64,
    /// Multiplicative range `(min, max)` applied to body/node masses and inertia.
    pub mass_scale: (f64, f64),
    /// Multiplicative range `(min, max)` applied to surface friction coefficients.
    pub friction_scale: (f64, f64),
    /// Multiplicative range `(min, max)` applied to link/drive stiffnesses.
    pub stiffness_scale: (f64, f64),
    /// Multiplicative range `(min, max)` applied to link/drive damping gains.
    pub damping_scale: (f64, f64),
    /// Per-axis additive jitter amplitude `±gravity_jitter` applied to `world.gravity` (m/s²).
    pub gravity_jitter: Vec3,
}

impl Default for DomainRandomizer {
    fn default() -> Self {
        Self {
            seed: 1,
            mass_scale: (1.0, 1.0),
            friction_scale: (1.0, 1.0),
            stiffness_scale: (1.0, 1.0),
            damping_scale: (1.0, 1.0),
            gravity_jitter: Vec3::ZERO,
        }
    }
}

struct Rng(u64);

impl Rng {
    fn new(seed: u64, index: usize) -> Self {
        let s = seed
            ^ ((index as u64)
                .wrapping_add(1)
                .wrapping_mul(0x9e37_79b9_7f4a_7c15));
        Self(s.max(1))
    }

    fn next_u64(&mut self) -> u64 {
        self.0 = self.0.wrapping_add(0x9e37_79b9_7f4a_7c15);
        let mut z = self.0;
        z = (z ^ (z >> 30)).wrapping_mul(0xbf58_476d_1ce4_e5b9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94d0_49bb_1331_11eb);
        z ^ (z >> 31)
    }

    fn unit(&mut self) -> f64 {
        (self.next_u64() >> 11) as f64 * (1.0 / ((1u64 << 53) as f64))
    }

    fn range(&mut self, lo: f64, hi: f64) -> f64 {
        let (a, b) = if lo <= hi { (lo, hi) } else { (hi, lo) };
        a + (b - a) * self.unit()
    }
}

/// Vectorized batch of independent `(World, Mechanics)` environments stepped
/// in parallel across [`Pool`] worker threads with thread-invariant determinism.
pub struct WorldBatch {
    template: World,
    worlds: Vec<World>,
    solvers: Vec<Mechanics>,
    reports: Vec<StepReport>,
}

impl WorldBatch {
    /// Creates `count` identical environments initialized from `template` and `settings`.
    pub fn from_template(template: World, settings: Settings, count: usize) -> Self {
        let worlds = (0..count).map(|_| template.clone()).collect();
        let solvers = (0..count).map(|_| Mechanics::new(settings)).collect();
        let reports = vec![StepReport::default(); count];
        Self {
            template,
            worlds,
            solvers,
            reports,
        }
    }

    pub fn len(&self) -> usize {
        self.worlds.len()
    }

    pub fn is_empty(&self) -> bool {
        self.worlds.is_empty()
    }

    pub fn world(&self, index: usize) -> &World {
        &self.worlds[index]
    }

    pub fn world_mut(&mut self, index: usize) -> &mut World {
        &mut self.worlds[index]
    }

    pub fn solver(&self, index: usize) -> &Mechanics {
        &self.solvers[index]
    }

    pub fn reports(&self) -> &[StepReport] {
        &self.reports
    }

    /// Resets environment `index` cleanly back to the unmodified template state.
    pub fn reset_env(&mut self, index: usize) {
        self.worlds[index] = self.template.clone();
        self.solvers[index].invalidate();
        self.reports[index] = StepReport::default();
    }

    /// Resets all environments in the batch back to the unmodified template state.
    pub fn reset_all(&mut self) {
        for i in 0..self.worlds.len() {
            self.reset_env(i);
        }
    }

    /// Resets a subset of environment indices back to the template state.
    pub fn reset_envs(&mut self, indices: &[usize]) {
        for &i in indices {
            self.reset_env(i);
        }
    }

    /// Resets environment `index` to `template` and applies deterministic
    /// domain randomization keyed by `(config.seed, index)`.
    pub fn randomize_env(&mut self, index: usize, config: &DomainRandomizer) {
        self.randomize_env_with_seed(index, config, config.seed);
    }

    /// Resets and randomizes a subset of environment indices using
    /// `config.seed.wrapping_add(seed_offset)`.
    pub fn randomize_envs(
        &mut self,
        indices: &[usize],
        config: &DomainRandomizer,
        seed_offset: u64,
    ) {
        let seed = config.seed.wrapping_add(seed_offset);
        for &i in indices {
            self.randomize_env_with_seed(i, config, seed);
        }
    }

    fn randomize_env_with_seed(&mut self, index: usize, config: &DomainRandomizer, seed: u64) {
        let mut world = self.template.clone();
        let mut rng = Rng::new(seed, index);

        let m_scale = rng
            .range(config.mass_scale.0, config.mass_scale.1)
            .max(1e-6);
        let f_scale = rng
            .range(config.friction_scale.0, config.friction_scale.1)
            .max(0.0);
        let k_scale = rng
            .range(config.stiffness_scale.0, config.stiffness_scale.1)
            .max(0.0);
        let d_scale = rng
            .range(config.damping_scale.0, config.damping_scale.1)
            .max(0.0);

        if m_scale != 1.0 {
            for obj in world.objects_mut() {
                match &mut obj.representation {
                    Representation::Rigid(r) => {
                        r.mass *= m_scale;
                        r.inertia *= m_scale;
                    }
                    Representation::Solid(s) => {
                        for m in &mut s.nodes.mass {
                            *m *= m_scale;
                        }
                    }
                    Representation::Fixed(_) => {}
                }
            }
        }
        if f_scale != 1.0 {
            for mat in world.materials_mut() {
                mat.friction = (mat.friction * f_scale).max(0.0);
            }
        }
        if k_scale != 1.0 || d_scale != 1.0 {
            let link_count = world.links().len();
            for i in 0..link_count {
                match world.link_mut(LinkId(i as u32)) {
                    Link::Anchor { .. } => {}
                    Link::Spring {
                        stiffness, damping, ..
                    } => {
                        *stiffness *= k_scale;
                        *damping *= d_scale;
                    }
                    Link::BallJoint { stiffness, .. } | Link::Weld { stiffness, .. } => {
                        *stiffness *= k_scale;
                    }
                    Link::Hinge {
                        stiffness,
                        angular_stiffness,
                        drive,
                        ..
                    } => {
                        *stiffness *= k_scale;
                        *angular_stiffness *= k_scale;
                        drive.stiffness *= k_scale;
                        drive.damping *= d_scale;
                    }
                    Link::Slider {
                        stiffness,
                        rotational_stiffness,
                        drive,
                        ..
                    } => {
                        *stiffness *= k_scale;
                        *rotational_stiffness *= k_scale;
                        drive.stiffness *= k_scale;
                        drive.damping *= d_scale;
                    }
                    Link::Cylindrical {
                        stiffness,
                        angular_stiffness,
                        linear_drive,
                        angular_drive,
                        ..
                    } => {
                        *stiffness *= k_scale;
                        *angular_stiffness *= k_scale;
                        linear_drive.stiffness *= k_scale;
                        linear_drive.damping *= d_scale;
                        angular_drive.stiffness *= k_scale;
                        angular_drive.damping *= d_scale;
                    }
                    Link::Universal {
                        stiffness,
                        angular_stiffness,
                        ..
                    }
                    | Link::Planar {
                        stiffness,
                        angular_stiffness,
                        ..
                    } => {
                        *stiffness *= k_scale;
                        *angular_stiffness *= k_scale;
                    }
                    Link::Tendon {
                        stiffness,
                        damping,
                        drive,
                        ..
                    } => {
                        *stiffness *= k_scale;
                        *damping *= d_scale;
                        drive.stiffness *= k_scale;
                        drive.damping *= d_scale;
                    }
                }
            }
        }
        if config.gravity_jitter != Vec3::ZERO {
            world.gravity += Vec3::new(
                rng.range(-config.gravity_jitter.x, config.gravity_jitter.x),
                rng.range(-config.gravity_jitter.y, config.gravity_jitter.y),
                rng.range(-config.gravity_jitter.z, config.gravity_jitter.z),
            );
        }

        self.worlds[index] = world;
        self.solvers[index].invalidate();
        self.reports[index] = StepReport::default();
    }

    /// Resets and applies deterministic domain randomization to all environments.
    pub fn randomize_all(&mut self, config: &DomainRandomizer) {
        for i in 0..self.worlds.len() {
            self.randomize_env(i, config);
        }
    }

    /// Sets joint drive targets `(link, target)` on environment `env_index`.
    pub fn set_joint_targets(&mut self, env_index: usize, targets: &[(LinkId, f64)]) {
        let w = &mut self.worlds[env_index];
        for &(link, target) in targets {
            w.set_joint_target(link, target);
        }
    }

    /// Applies row-major `[env_count * links.len()]` joint targets across the
    /// entire batch without Python-to-Rust per-env loops.
    pub fn set_batch_joint_targets(&mut self, links: &[LinkId], targets_row_major: &[f64]) {
        assert_eq!(
            targets_row_major.len(),
            self.worlds.len() * links.len(),
            "expected {} target values, got {}",
            self.worlds.len() * links.len(),
            targets_row_major.len()
        );
        if links.is_empty() {
            return;
        }
        for (env_idx, chunk) in targets_row_major.chunks_exact(links.len()).enumerate() {
            let w = &mut self.worlds[env_idx];
            for (&link, &target) in links.iter().zip(chunk) {
                w.set_joint_target(link, target);
            }
        }
    }

    /// Reads current scalar coordinates for `links` across all environments as
    /// a flat row-major vector `[env_count * links.len()]` (`NaN` if unsupported).
    pub fn joint_coordinates(&self, links: &[LinkId]) -> Vec<f64> {
        let mut out = Vec::with_capacity(self.worlds.len() * links.len());
        for w in &self.worlds {
            for &link in links {
                out.push(w.joint_coordinate(link).unwrap_or(f64::NAN));
            }
        }
        out
    }

    /// Reads current scalar velocities for `links` across all environments as a
    /// flat row-major vector `[env_count * links.len()]` (`NaN` if unsupported).
    pub fn joint_velocities(&self, links: &[LinkId]) -> Vec<f64> {
        let mut out = Vec::with_capacity(self.worlds.len() * links.len());
        for w in &self.worlds {
            for &link in links {
                out.push(w.joint_velocity(link).unwrap_or(f64::NAN));
            }
        }
        out
    }

    /// Advances every environment in the batch by `h` seconds in parallel.
    pub fn step_all(&mut self, pool: &Pool, h: f64) {
        pool.split_reduce(
            (
                &mut self.worlds[..],
                &mut self.solvers[..],
                &mut self.reports[..],
            ),
            1,
            (),
            |_, (ws, ss, rs)| {
                rs[0] = ss[0].step(pool, &mut ws[0], h);
            },
            |_, _| (),
        );
    }

    /// Samples all world [`core_world::Probe`] sensors across every environment.
    pub fn sample_all_probes(&mut self, pool: &Pool) -> Vec<Vec<ProbeReading>> {
        let mut out = vec![Vec::new(); self.worlds.len()];
        pool.split_reduce(
            (&self.worlds[..], &mut self.solvers[..], &mut out[..]),
            1,
            (),
            |_, (ws, ss, os)| {
                os[0] = ss[0].sample_probes(pool, &ws[0]);
            },
            |_, _| (),
        );
        out
    }
}
