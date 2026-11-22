//! What Fusion promises about reproducing a run.
//!
//! PLAN.md §34 lists the determinism decision as open, offers three tiers, and
//! recommends one. This module is that decision, made:
//!
//! | tier | promise | verdict |
//! |---|---|---|
//! | (a) same machine, same build | bit-identical | **guaranteed**, and asserted by tests |
//! | (b) across platforms | agrees within a stated tolerance | **documented**, not enforced |
//! | (c) across platforms | bit-identical | **declined** |
//!
//! §34's own reasoning for declining (c) is that it requires forbidding FMA
//! contraction and hand-rolling every transcendental, and that it is "very
//! hard and almost certainly not worth it". Nothing found while building v0.4
//! argues otherwise. What (a) buys — and it is the whole reason the decision
//! is made here rather than later — is **rewind**. PLAN.md §17: a simulation
//! is causal and has no seek, so a time slider is checkpoints plus replay, and
//! replay is only a seek if it lands on the same state. Without (a), Fusion
//! would have play and reset and nothing else, forever.
//!
//! ## What (a) costs, in practice
//!
//! Bit-identical replay is not free and is not automatic. It is a constraint
//! on how every summation in the engine is written, and v0.4 broke it twice
//! before the tests caught it. The rules that follow are the ones that were
//! actually load-bearing:
//!
//! - **No `HashMap` iteration in a numerical path.** Rust's default hasher is
//!   randomly seeded per process, so the same binary on the same machine
//!   iterates in a different order on every run. This is why
//!   [`fusion_collide::SpatialHash`] is a counting sort over a flat array
//!   rather than the obvious map of cells to particle lists.
//! - **Fixed summation order.** Floating-point addition is not associative.
//!   A neighbour list gathered from a 27-cell walk arrives in cell order,
//!   which is not particle order, so it is sorted before any density is summed
//!   over it. The sort is not tidiness; without it the density differs in the
//!   last bits, the difference amplifies through the solve, and two runs of
//!   the same scene visibly diverge inside a few seconds.
//! - **No parallel reduction.** §34 names this: a work-stealing sum gives a
//!   different answer per run because the association changes. Fusion is
//!   single-threaded today, and when it stops being, any reduction has to be
//!   deterministic — fixed chunking with a fixed combine order — or the
//!   guarantee goes.
//! - **Iteration order over discovered structures is part of the state.**
//!   The broad phase keeps its sorted order across frames, and contacts
//!   inherit anchors from the previous step. Both change the answer, and both
//!   are therefore in [`crate::Checkpoint`].
//!
//! ## What (b) means and does not mean
//!
//! Across platforms, the same scene diverges: FMA contraction is a
//! compiler-and-target decision, and `sin`, `cos` and `sqrt` are allowed to
//! differ in the last bit between libm implementations. Fusion does not chase
//! this. What it claims instead is that the *physics* agrees — a stack that
//! stands on one machine stands on the other, a fluid settles to the same
//! depth — while individual coordinates do not. That is enough for research
//! reproducibility and is not enough to compare two runs by hashing them.
//!
//! Which is why [`state_hash`] exists and why it is honest about its scope:
//! it is a fingerprint for tier (a) only.

use crate::Simulation;

/// The three tiers of PLAN.md §34.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord)]
pub enum DeterminismTier {
    /// Same machine, same build: bit-identical.
    SameBuild,
    /// Any platform, agreeing within a tolerance rather than exactly.
    CrossPlatformBounded,
    /// Any platform, bit-identical.
    CrossPlatformExact,
}

/// Fusion's determinism guarantee, stated where it is used.
///
/// The same argument [`fusion_solver::AccuracyTier`] makes: a promise that
/// lives only in documentation is a promise nobody reads at the moment it
/// matters.
#[derive(Clone, Copy, Debug)]
pub struct Determinism {
    /// The strongest tier Fusion will stand behind.
    pub guaranteed: DeterminismTier,
    /// The strongest tier it declines to attempt.
    pub declined: DeterminismTier,
    /// Why, in the words the API is willing to stand behind.
    pub note: &'static str,
}

impl Determinism {
    /// What Fusion promises about reproducibility (§34), as a value rather
    /// than as a paragraph in a document nobody reads at the right moment.
    pub const GUARANTEE: Self = Self {
        guaranteed: DeterminismTier::SameBuild,
        declined: DeterminismTier::CrossPlatformExact,
        note: "Bit-identical for the same build on the same machine, which is \
               what checkpoint-and-replay rewind requires. Across platforms, \
               results agree physically but not bit-for-bit: FMA contraction \
               and libm differences are not fought.",
    };

    /// Whether a tier is claimed.
    pub fn claims(&self, tier: DeterminismTier) -> bool {
        tier <= self.guaranteed
    }
}

/// A fingerprint of everything a step reads and writes.
///
/// PLAN.md §22 wants a project file plus a step number to reproduce anything
/// exactly, and the missing half of that is a way to say whether it did. This
/// is that: one number that changes if any bit of the simulation state
/// changes, cheap enough to record every step in a bug report.
///
/// **Tier (a) only.** Two runs on different platforms will produce different
/// hashes for the same scene while both being correct, so a mismatch across
/// machines means nothing. Within one build on one machine, a mismatch means a
/// real divergence and nothing else.
///
/// FNV-1a over the raw bits, in a fixed order. Not cryptographic and not
/// trying to be — the adversary here is a solver that changed, not a person.
pub fn state_hash(sim: &Simulation) -> u64 {
    let mut h = Hasher::new();
    let world = &sim.world;
    if let Some(core) = &sim.core {
        h.u64(0x465553494f4e434f);
        for byte in core.checkpoint().to_bytes() {
            h.u64(byte as u64);
        }
    }

    let bodies = &world.bodies;
    for i in 0..bodies.len() {
        let p = bodies.position[i];
        h.f64(p.x);
        h.f64(p.y);
        h.f64(p.z);
        let q = bodies.orientation[i];
        for v in q.to_array() {
            h.f32(v);
        }
        for v in bodies.linear_velocity[i].to_array() {
            h.f32(v);
        }
        for v in bodies.angular_velocity[i].to_array() {
            h.f32(v);
        }
        h.u64(bodies.sleeping[i] as u64);
        h.f32(bodies.sleep_timer[i]);
    }
    // Temperature, once the heat solve has given a body a grid. Only then: a
    // run with heat off has the same hash it had before rigid bodies had a
    // temperature at all, so every recorded fingerprint still matches.
    for (i, heat) in bodies.heat.iter().enumerate() {
        if heat.is_built() {
            h.u64(i as u64);
            for t in &heat.temperature {
                h.f64(*t);
            }
        }
    }

    // What failure changed, only where it changed something, for the same
    // reason: a run where nothing broke keeps every fingerprint it had.
    for i in 0..bodies.len() {
        if bodies.removed[i] || bodies.reshaped[i] {
            h.u64(i as u64);
            h.u64(bodies.removed[i] as u64 | (bodies.reshaped[i] as u64) << 1);
            h.f32(bodies.inv_mass[i]);
        }
    }

    hash_particles(&mut h, &world.particles);
    for fluid in &world.fluids {
        hash_particles(&mut h, &fluid.particles);
    }
    // A material point's position and velocity are not its state. Two points
    // in the same place moving the same way are different states if one has
    // been stretched, so a hash that stopped at the particle set would report
    // two visibly different runs as identical — and every tool that rests on
    // this hash would agree with it. `study determinism` would call a
    // divergent continuum reproducible, and `Project::verify` would pass a
    // file that rebuilt a different solid.
    for solid in &world.solids {
        hash_particles(&mut h, &solid.particles);
        for f in &solid.deformation {
            for column in [f.x_axis, f.y_axis, f.z_axis] {
                for value in column.to_array() {
                    h.f32(value);
                }
            }
        }
        for p in &solid.plastic {
            h.f32(*p);
        }
        // v0.9's field, added here in the same change that added it to the
        // solid rather than in the one that noticed it was missing. The
        // deformation gradient taught this lesson one version ago: a hash that
        // ignores a kind of state reports two different runs as identical, and
        // nothing goes red to say so.
        for e in &solid.enthalpy {
            h.f64(*e);
        }
        for d in &solid.damage {
            h.f32(*d);
        }
    }

    // Locals and failures influence future steps even while every body is still.
    // Keep the no-script fingerprint compatible with older state files.
    if !sim.scripts().is_empty() {
        for byte in crate::state::script_state_bytes(sim.scripts()) {
            h.u64(byte as u64);
        }
    }
    h.u64(sim.step_index());
    h.f64(sim.time());
    h.finish()
}

fn hash_particles(h: &mut Hasher, particles: &fusion_core::ParticleSet) {
    for i in 0..particles.len() {
        let p = particles.position[i];
        h.f64(p.x);
        h.f64(p.y);
        h.f64(p.z);
        for v in particles.velocity[i].to_array() {
            h.f32(v);
        }
    }
}

/// FNV-1a, 64-bit.
struct Hasher(u64);

impl Hasher {
    fn new() -> Self {
        Self(0xcbf2_9ce4_8422_2325)
    }

    #[inline]
    fn u64(&mut self, value: u64) {
        for byte in value.to_le_bytes() {
            self.0 ^= byte as u64;
            self.0 = self.0.wrapping_mul(0x1000_0000_01b3);
        }
    }

    /// Hashed as bits, not as a number.
    ///
    /// `to_bits` rather than a cast, so that the two zeros and the NaNs are
    /// distinguished: a simulation that has just produced its first `-0.0`
    /// where it used to produce `0.0` has diverged, and a hash that could not
    /// see the difference would be the wrong tool for finding out.
    #[inline]
    fn f32(&mut self, value: f32) {
        self.u64(value.to_bits() as u64);
    }

    #[inline]
    fn f64(&mut self, value: f64) {
        self.u64(value.to_bits());
    }

    fn finish(self) -> u64 {
        self.0
    }
}
