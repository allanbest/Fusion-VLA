//! Checkpoints, and the rewind they make possible.
//!
//! PLAN.md §17 states the problem exactly: "a video is random-access — frame
//! 500 exists independently. A simulation is causal — the state at t = 10 s
//! exists only as the consequence of every step before it. There is no seek."
//!
//! So a time slider is not a seek. It is: keep the full state every so often,
//! and when the user asks for a time, restore the nearest one before it and
//! re-simulate forward. Seeking back costs up to one checkpoint interval of
//! recompute; seeking forward from where you are costs nothing extra.
//!
//! That only works if re-simulating forward lands on the same state the
//! original run did. It is the payoff §17 promises for determinism, and it is
//! why [`crate::determinism`] guarantees tier (a) rather than leaving the
//! question open.
//!
//! ## A checkpoint is not the physical state
//!
//! This is the part that is easy to get wrong, and the way it fails is worse
//! than the failure itself.
//!
//! The step function reads more than positions and velocities. It reads the
//! contact anchors inherited from the previous step — which decide whether
//! static friction has a reference to pull back to — and it reads the broad
//! phase's sorted order, which decides the sequence the Gauss-Seidel contact
//! solve visits manifolds in. Neither is physics. Both change the answer.
//!
//! Measured, on a scene holding a stack, a joint and a fluid: restoring only
//! the world and the clock, and stepping forward from a checkpoint taken at
//! step *n*, against the same run's recorded hashes.
//!
//! | checkpoint at step | full restore | world-only restore |
//! |---|---|---|
//! | 5 | identical throughout | identical throughout |
//! | 20 | identical throughout | **diverges after 1 step** |
//! | 40 | identical throughout | **diverges after 1 step** |
//! | 60 | identical throughout | **diverges after 1 step** |
//! | 100 | identical throughout | **diverges after 1 step** |
//! | 150 | identical throughout | identical throughout |
//! | 250 | identical throughout | identical throughout |
//!
//! Note the shape of that column. The world-only restore is not slightly
//! wrong and it does not drift: where it is wrong it is wrong on the very
//! next step, and where the scene is quiet — nothing yet touching at step 5,
//! everything settled by step 150 — it is exactly right. So a checkpoint
//! missing this state passes every test written against a scene at rest, and
//! fails the moment anything is in contact. It is the kind of defect that
//! ships.
//!
//! Rather than enumerate what to keep, a checkpoint **clones the entire
//! pipeline**, transient scratch buffers included. That wastes some memory on
//! arrays that are about to be overwritten anyway. It is also the only version
//! of this that cannot be quietly wrong the next time somebody adds a field:
//! there is no list to forget to update. The guard on the whole arrangement is
//! `replay_from_a_checkpoint_is_bit_identical`, which runs a scene with
//! stacking *and* fluid in it, because those are the two systems with
//! cross-step memory.
//!
//! ```
//! use fusion_core::{BodyDesc, World};
//! use fusion_math::{vec3, Vec3};
//! use fusion_sim::{state_hash, Simulation, Timeline};
//!
//! let mut world = World::new();
//! world.add(BodyDesc::cuboid(vec3(20.0, 1.0, 20.0)).at(vec3(0.0, -0.5, 0.0)).fixed());
//! world.add(BodyDesc::sphere(0.4).at(vec3(0.0, 4.0, 0.0)));
//! let mut sim = Simulation::new(world);
//!
//! let mut timeline = Timeline::new(0.25);
//! timeline.observe(&sim);
//! let mut history = Vec::new();
//! for _ in 0..240 {
//!     sim.step();
//!     timeline.observe(&sim);
//!     history.push((sim.time(), state_hash(&sim)));
//! }
//!
//! // Scrub back. A backward seek costs at most one interval of replay, and
//! // lands on the state the original run was in — not merely a similar one.
//! let result = timeline.seek(&mut sim, 0.4);
//! assert!(result.exact);
//! let expected = history.iter().rev().find(|(t, _)| *t <= sim.time() + 1e-9).unwrap().1;
//! assert_eq!(expected, state_hash(&sim));
//! ```

use fusion_collide::{CollisionPipeline, ContactManifold};
use fusion_core::{BodySet, FluidSet, Joint, ParticleSet, SolidSet};
use fusion_math::Vec3;

use crate::diagnostics::RunDiagnostics;
use crate::islands::IslandSet;
use crate::Simulation;

/// Everything needed to put a simulation back exactly as it was.
///
/// Belongs to the simulation it came from: it holds state, not scene, so
/// restoring it into a world built from a different description is a
/// programming error and is rejected rather than producing nonsense.
#[derive(Clone, Debug)]
pub struct Checkpoint {
    pub(crate) core: Option<phys_mechanics::MechanicsState>,
    bodies: BodySet,
    particles: ParticleSet,
    fluids: Vec<FluidSet>,
    /// Every material point's position, velocity, deformation and plastic
    /// history.
    ///
    /// The deformation gradient is the reason a solid cannot be checkpointed
    /// by position alone. Two material points at the same place moving at the
    /// same speed are different states if one of them has been squashed, so a
    /// "physical state" that means position and velocity is not a state at all
    /// here — it is a photograph of a solid with its history removed.
    solids: Vec<SolidSet>,
    joints: Vec<Joint>,
    gravity: Vec3,
    linear_damping: f32,
    angular_damping: f32,

    pipeline: CollisionPipeline,
    manifolds: Vec<ContactManifold>,
    islands: IslandSet,

    /// What the solver remembers between steps — the converged contact
    /// impulses it will warm-start from. Small, and load-bearing: without it
    /// a restored run solves its contacts from zero for one step while the
    /// original resumed them, and the two diverge immediately.
    solver_memory: fusion_solver::SolverMemory,

    time: f64,
    step_index: u64,
    frame_index: u64,
    accumulator: f64,
    last_energy: Option<f32>,
    diagnostics: RunDiagnostics,
    scripts: Vec<crate::script::BodyScript>,
    components: Vec<crate::control::BodyComponents>,
    script_errors: Vec<(usize, fusion_script::ScriptError)>,
}

impl Checkpoint {
    /// Simulated time this checkpoint was taken at.
    pub fn time(&self) -> f64 {
        self.time
    }

    /// Solver step this checkpoint was taken after.
    pub fn step_index(&self) -> u64 {
        self.step_index
    }

    /// Rough memory cost, bytes. Approximate on purpose — it exists so a
    /// [`Timeline`] can be given a budget in megabytes rather than in a
    /// checkpoint count nobody can reason about.
    pub fn size_bytes(&self) -> usize {
        let body =
            self.bodies.len() * (std::mem::size_of::<f64>() * 3 + std::mem::size_of::<f32>() * 20);
        let particle = |p: &ParticleSet| p.len() * (std::mem::size_of::<f64>() * 3 + 20);
        let fluid: usize = self.fluids.iter().map(|f| particle(&f.particles)).sum();
        // A material point costs its point plus two matrices and two scalars,
        // which is roughly five times a fluid particle. Worth knowing before
        // giving a timeline a budget in megabytes.
        let solid: usize = self
            .solids
            .iter()
            .map(|s| s.len() * (std::mem::size_of::<f64>() * 3 + 20 + 72 + 8))
            .sum();
        let manifolds = self.manifolds.len() * std::mem::size_of::<ContactManifold>();
        let scripts: usize = self
            .scripts
            .iter()
            .map(|script| {
                script.program.source().len()
                    + script
                        .locals
                        .keys()
                        .map(|key| key.len() + 48)
                        .sum::<usize>()
                    + script
                        .failed
                        .as_ref()
                        .map_or(0, |error| error.message.len() + 8)
            })
            .sum();
        let errors: usize = self
            .script_errors
            .iter()
            .map(|(_, error)| error.message.len() + 16)
            .sum();
        body + particle(&self.particles)
            + fluid
            + solid
            + manifolds
            + scripts
            + errors
            + self.core.as_ref().map_or(0, |c| c.encoded_len())
    }

    // The accessors `crate::state` encodes through. Deliberately narrow and
    // deliberately here rather than `pub` fields: a checkpoint is a snapshot
    // to be restored, not a scene to be edited, and the one caller that needs
    // to reach inside is the one that writes it to a file.

    /// Body poses, velocities and sleep state.
    pub(crate) fn bodies_of(&self) -> &BodySet {
        &self.bodies
    }
    /// Cloth and soft-body particles.
    pub(crate) fn particles_of(&self) -> &ParticleSet {
        &self.particles
    }
    /// Every fluid.
    pub(crate) fn fluids_of(&self) -> &[FluidSet] {
        &self.fluids
    }
    /// Every solid, with its deformation history.
    pub(crate) fn solids_of(&self) -> &[SolidSet] {
        &self.solids
    }
    /// Output frame this checkpoint was taken on.
    pub(crate) fn frame_index_of(&self) -> u64 {
        self.frame_index
    }
    /// Left-over simulated time from the last `advance`.
    pub(crate) fn accumulator_of(&self) -> f64 {
        self.accumulator
    }

    pub(crate) fn bodies_mut(&mut self) -> &mut BodySet {
        &mut self.bodies
    }
    pub(crate) fn particles_mut(&mut self) -> &mut ParticleSet {
        &mut self.particles
    }
    pub(crate) fn fluids_mut(&mut self) -> &mut Vec<FluidSet> {
        &mut self.fluids
    }
    pub(crate) fn solids_mut(&mut self) -> &mut Vec<SolidSet> {
        &mut self.solids
    }
    pub(crate) fn scripts_of(&self) -> &[crate::script::BodyScript] {
        &self.scripts
    }
    pub(crate) fn scripts_mut(&mut self) -> &mut Vec<crate::script::BodyScript> {
        &mut self.scripts
    }
    pub(crate) fn script_errors_of(&self) -> &[(usize, fusion_script::ScriptError)] {
        &self.script_errors
    }
    pub(crate) fn script_errors_mut(&mut self) -> &mut Vec<(usize, fusion_script::ScriptError)> {
        &mut self.script_errors
    }
    pub(crate) fn pipeline_of(&self) -> &CollisionPipeline {
        &self.pipeline
    }
    pub(crate) fn pipeline_mut(&mut self) -> &mut CollisionPipeline {
        &mut self.pipeline
    }
    pub(crate) fn solver_memory_of(&self) -> &fusion_solver::SolverMemory {
        &self.solver_memory
    }
    pub(crate) fn solver_memory_mut(&mut self) -> &mut fusion_solver::SolverMemory {
        &mut self.solver_memory
    }
    pub(crate) fn set_clock(&mut self, time: f64, step: u64, frame: u64, accumulator: f64) {
        self.time = time;
        self.step_index = step;
        self.frame_index = frame;
        self.accumulator = accumulator;
    }
}

impl Simulation {
    /// Capture the current state.
    pub fn checkpoint(&self) -> Checkpoint {
        Checkpoint {
            core: self.core.as_ref().map(|c| c.checkpoint()),
            bodies: self.world.bodies.clone(),
            particles: self.world.particles.clone(),
            fluids: self.world.fluids.clone(),
            solids: self.world.solids.clone(),
            joints: self.world.joints().to_vec(),
            gravity: self.world.gravity,
            linear_damping: self.world.linear_damping,
            angular_damping: self.world.angular_damping,
            pipeline: self.pipeline.clone(),
            manifolds: self.manifolds.clone(),
            islands: self.islands.clone(),
            solver_memory: self.working.memory(),
            time: self.time,
            step_index: self.step_index,
            frame_index: self.frame_index,
            accumulator: self.accumulator,
            last_energy: self.last_energy,
            diagnostics: self.diagnostics,
            scripts: self.scripts.clone(),
            components: self.components.clone(),
            script_errors: self.script_errors.clone(),
        }
    }

    /// A checkpoint of everything but the continua — the solids and fluids —
    /// for the step's own rollback. The Unified continuum snapshots those itself,
    /// and restores them when it rejects an interval; copying them a second
    /// time here was most of what a step of four jelly blocks spent on safety.
    pub(crate) fn rigid_checkpoint(&mut self) -> Checkpoint {
        let solids = std::mem::take(&mut self.world.solids);
        let fluids = std::mem::take(&mut self.world.fluids);
        let checkpoint = self.checkpoint();
        self.world.solids = solids;
        self.world.fluids = fluids;
        checkpoint
    }

    /// Put back a [`Simulation::rigid_checkpoint`], keeping the continua as
    /// they are — which, after a rejected interval, is as they were.
    pub(crate) fn restore_rigid(&mut self, checkpoint: &Checkpoint) {
        // Copied, not moved: `restore` checks the checkpoint against the
        // world as it stands. A rejected step is the rare case, so the copy
        // is paid only then.
        let mut full = checkpoint.clone();
        full.solids = self.world.solids.clone();
        full.fluids = self.world.fluids.clone();
        self.restore(&full);
    }

    /// Put the state back.
    ///
    /// Panics if the checkpoint came from a differently-shaped world. Stepping
    /// on with mismatched arrays would index out of bounds somewhere deep in a
    /// solver, and the message there would be about a slice rather than about
    /// the mistake that was actually made.
    ///
    /// "Shaped" means the scene: a checkpoint may hold more bodies or solids
    /// than the world has right now, or fewer, where something broke or
    /// melted on one side of it and not the other. Both carry everything a
    /// body is, so putting one back puts back the pieces too.
    pub fn restore(&mut self, checkpoint: &Checkpoint) {
        assert_eq!(
            self.core.is_some(),
            checkpoint.core.is_some(),
            "checkpoint belongs to a different mechanics engine"
        );
        let scene = self.world.scene().len();
        // Every body past the scene's has to be a piece of one that broke.
        let fits = checkpoint.bodies.len() >= scene
            && (scene..checkpoint.bodies.len()).all(|i| checkpoint.bodies.reshaped[i]);
        assert!(
            fits,
            "checkpoint holds {} bodies, this scene has {} — a checkpoint \
             belongs to the simulation it was taken from",
            checkpoint.bodies.len(),
            scene
        );
        assert_eq!(
            checkpoint.fluids.len(),
            self.world.fluids.len(),
            "checkpoint holds {} fluids, this world has {}",
            checkpoint.fluids.len(),
            self.world.fluids.len()
        );
        assert!(
            checkpoint.solids.len() >= self.world.solid_scene().len(),
            "checkpoint holds {} solids, this scene has {}",
            checkpoint.solids.len(),
            self.world.solid_scene().len()
        );

        assert!(
            self.scripts
                .iter()
                .map(|script| (script.body, script.program.source()))
                .eq(checkpoint
                    .scripts
                    .iter()
                    .map(|script| (script.body, script.program.source()))),
            "checkpoint belongs to a different set of behaviour scripts"
        );
        assert_eq!(
            self.components, checkpoint.components,
            "checkpoint belongs to different components"
        );
        self.probe_samples.clear();
        self.scripts.clone_from(&checkpoint.scripts);
        self.script_errors.clone_from(&checkpoint.script_errors);

        self.world.bodies.clone_from(&checkpoint.bodies);
        self.world.particles.clone_from(&checkpoint.particles);
        self.world.fluids.clone_from(&checkpoint.fluids);
        self.world.solids.clone_from(&checkpoint.solids);
        self.world
            .joints_storage_mut()
            .clone_from(&checkpoint.joints);
        self.world.gravity = checkpoint.gravity;
        self.world.linear_damping = checkpoint.linear_damping;
        self.world.angular_damping = checkpoint.angular_damping;

        self.pipeline.clone_from(&checkpoint.pipeline);
        self.manifolds.clone_from(&checkpoint.manifolds);
        self.islands.clone_from(&checkpoint.islands);

        self.time = checkpoint.time;
        self.step_index = checkpoint.step_index;
        self.frame_index = checkpoint.frame_index;
        self.accumulator = checkpoint.accumulator;
        self.last_energy = checkpoint.last_energy;
        self.diagnostics = checkpoint.diagnostics;

        // Solver scratch is thrown away rather than restored: `prepare`
        // rebuilds all of it from the state at the top of every step.
        //
        // All of it except the contact memory, which is the whole point of
        // `SolverMemory` existing. Contact warm-starting made the solver carry
        // something across a step boundary for the first time, and the moment
        // it did, three replay tests failed — correctly. The rule is not
        // "scratch is disposable", it is "whatever `prepare` cannot rebuild is
        // state", and the tests are what tell the difference.
        self.working = fusion_solver::XpbdWorkingSet::new();
        self.working.restore_memory(&checkpoint.solver_memory);
        if let (Some(core), Some(state)) = (&mut self.core, &checkpoint.core) {
            core.restore(state, &mut self.world);
        }
    }

    /// Restore positions, velocities, the clock and the solver's contact
    /// memory — but *not* the collision pipeline, the manifolds, or the
    /// islands.
    ///
    /// This is the tempting version of [`Self::restore`], the one that treats
    /// a checkpoint as "the physical state". It exists so that
    /// `a_world_only_checkpoint_diverges` can demonstrate how wrong that is,
    /// rather than leaving the module documentation to assert it. Nothing else
    /// should call it.
    ///
    /// It deliberately *does* carry the contact memory, which makes the
    /// demonstration sharper rather than weaker: with the warm-start impulses
    /// held constant, the only thing left to explain the divergence is the
    /// pipeline state, which is exactly the claim being made.
    #[doc(hidden)]
    pub fn restore_physical_state_only(&mut self, checkpoint: &Checkpoint) {
        self.world.bodies.clone_from(&checkpoint.bodies);
        self.world.particles.clone_from(&checkpoint.particles);
        self.world.fluids.clone_from(&checkpoint.fluids);
        self.world.solids.clone_from(&checkpoint.solids);
        self.time = checkpoint.time;
        self.step_index = checkpoint.step_index;
        self.frame_index = checkpoint.frame_index;
        self.accumulator = checkpoint.accumulator;
        self.working = fusion_solver::XpbdWorkingSet::new();
        self.working.restore_memory(&checkpoint.solver_memory);
    }
}

/// A recording of past states, and the ability to go back to them.
///
/// Drives PLAN.md §17's **live** mode: play, pause, single-step, and a
/// scrubber that costs at most one interval of recompute. The **baked** mode
/// — write every frame to a cache and scrub it like video — is a different
/// artefact and is not this.
///
/// Not owned by [`Simulation`], and deliberately: a headless sweep across a
/// hundred parameter values wants none of this and should not pay for it, and
/// a caller who does want it should be the one choosing how much memory it
/// gets.
#[derive(Clone, Debug)]
pub struct Timeline {
    interval: f64,
    checkpoints: Vec<Checkpoint>,
    budget: usize,
    /// Set by `with_memory_budget`, resolved into a checkpoint count once the
    /// first checkpoint reveals how big one is.
    memory_budget: Option<usize>,
    /// How many times the interval has been doubled to stay inside the budget.
    thinned: u32,
}

impl Timeline {
    /// Record a checkpoint every `interval` seconds of simulated time.
    ///
    /// One second is the figure §17 uses, and the trade behind it is:
    /// seeking backwards costs up to one interval of re-simulation, and
    /// keeping the checkpoints costs one full state per interval. Scenes that
    /// step cheaply want a longer interval; scenes that step expensively want
    /// a shorter one, which is the opposite of what memory would prefer.
    pub fn new(interval: f64) -> Self {
        Self {
            interval: interval.max(1e-6),
            checkpoints: Vec::new(),
            budget: 128,
            memory_budget: None,
            thinned: 0,
        }
    }

    /// Cap on how many checkpoints to keep.
    pub fn with_budget(mut self, checkpoints: usize) -> Self {
        self.budget = checkpoints.max(2);
        self
    }

    /// Cap expressed in bytes instead, converted once the first checkpoint
    /// exists and its size is known.
    pub fn with_memory_budget(mut self, bytes: usize) -> Self {
        self.budget = 0; // resolved on the first `observe`
        self.memory_budget = Some(bytes);
        self
    }

    /// Offer the current state to the recorder, which keeps it if one is due.
    ///
    /// Call it after every step. Cheap when nothing is due — one comparison —
    /// which is what lets the caller not think about it.
    pub fn observe(&mut self, sim: &Simulation) {
        let due = match self.checkpoints.last() {
            None => true,
            Some(last) => sim.time() >= last.time + self.effective_interval(),
        };
        if !due {
            return;
        }
        let checkpoint = sim.checkpoint();
        if let (Some(bytes), true) = (self.memory_budget, self.budget == 0) {
            self.budget = (bytes / checkpoint.size_bytes().max(1)).clamp(2, 4096);
        }
        self.checkpoints.push(checkpoint);
        if self.checkpoints.len() > self.budget {
            self.thin();
        }
    }

    /// Halve the number of checkpoints by dropping every second one.
    ///
    /// Not "drop the oldest", which is the obvious policy and the wrong one
    /// here. Dropping the oldest makes the beginning of a run unreachable —
    /// and the beginning is where a user scrubs to when something went wrong,
    /// because they are looking for when it started. Thinning keeps the whole
    /// run seekable and doubles the worst-case replay cost instead, which is
    /// the resource that degrades gracefully.
    fn thin(&mut self) {
        let mut keep = 0;
        self.checkpoints.retain(|_| {
            keep += 1;
            keep % 2 == 1
        });
        self.thinned += 1;
    }

    fn effective_interval(&self) -> f64 {
        self.interval * (1u64 << self.thinned.min(32)) as f64
    }

    /// Simulated seconds between checkpoints as things currently stand, which
    /// grows past the requested interval once thinning has started.
    pub fn interval(&self) -> f64 {
        self.effective_interval()
    }

    /// Checkpoints currently held.
    pub fn len(&self) -> usize {
        self.checkpoints.len()
    }

    /// True where nothing has been recorded yet.
    pub fn is_empty(&self) -> bool {
        self.checkpoints.is_empty()
    }

    /// Earliest time that can be returned to.
    pub fn earliest(&self) -> Option<f64> {
        self.checkpoints.first().map(|c| c.time)
    }

    /// Latest time a seek can land on without replaying forward.
    pub fn latest(&self) -> Option<f64> {
        self.checkpoints.last().map(|c| c.time)
    }

    /// Rough memory held by every checkpoint. See [`Checkpoint::size_bytes`]
    /// for why it is approximate.
    pub fn total_bytes(&self) -> usize {
        self.checkpoints.iter().map(|c| c.size_bytes()).sum()
    }

    /// Move a simulation to `time`, restoring and replaying as needed.
    ///
    /// Returns how many steps had to be re-simulated — the honest cost of the
    /// seek, and the number a UI should be showing when the scrubber stalls.
    ///
    /// Seeking *forward* from the current time never restores anything: the
    /// simulation is already the checkpoint it would have used. Seeking
    /// backwards restores the latest checkpoint at or before the target.
    pub fn seek(&self, sim: &mut Simulation, time: f64) -> SeekResult {
        let target = time.max(0.0);
        let restart = self
            .checkpoints
            .iter()
            .rev()
            .find(|c| c.time <= target + 1e-12);

        let mut restored_from = None;
        if sim.time() > target + 1e-12 {
            let Some(checkpoint) = restart else {
                return SeekResult {
                    reached: sim.time(),
                    steps_replayed: 0,
                    restored_from: None,
                    exact: false,
                };
            };
            sim.restore(checkpoint);
            restored_from = Some(checkpoint.time);
        }

        // Step to the target, stopping at the last step that does not overshoot
        // it. A simulation advances in whole solver steps; landing between two
        // of them is not a state the simulation has, and interpolating to it
        // would be inventing one.
        let dt = sim.solver_dt as f64;
        let mut steps = 0u64;
        let mut rejected = !dt.is_finite() || dt <= 0.0;
        while !rejected && sim.time() + dt <= target + dt * 1e-6 {
            rejected = sim.step().failure.is_some();
            if !rejected {
                steps += 1;
            }
        }
        SeekResult {
            reached: sim.time(),
            steps_replayed: steps,
            restored_from,
            exact: !rejected && (sim.time() - target).abs() <= dt,
        }
    }

    /// Discard everything recorded after `time` — what a caller does when the
    /// user scrubs back and then changes something, making the future that was
    /// recorded no longer the future that will happen.
    /// Drop every checkpoint after `time`.
    ///
    /// What a caller does when it rewinds and then simulates something
    /// different: the recorded future is no longer the future this run has.
    pub fn truncate_after(&mut self, time: f64) {
        self.checkpoints.retain(|c| c.time <= time + 1e-12);
    }

    /// Forget everything recorded.
    pub fn clear(&mut self) {
        self.checkpoints.clear();
        self.thinned = 0;
    }
}

/// What a [`Timeline::seek`] actually did.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct SeekResult {
    /// Simulated time arrived at. On a step boundary, so at most one solver
    /// step before the requested time.
    pub reached: f64,
    /// Steps re-simulated to get there. The cost of the seek.
    pub steps_replayed: u64,
    /// Time of the checkpoint that was restored, if one was.
    pub restored_from: Option<f64>,
    /// False when the target was before the earliest checkpoint, so the seek
    /// could not be honoured.
    pub exact: bool,
}
