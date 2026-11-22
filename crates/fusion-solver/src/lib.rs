//! Fusion L3 — numerics.
//!
//! PLAN.md §5 warns about the failure that ossifies engines of this kind:
//! system logic swallowing solver logic, until the rigid body system owns a
//! private linear solve that the FEM system cannot reuse and the engine has N
//! incompatible numerical stacks. So a solver here is a first-class citizen
//! with a declared interface, addressed generically from above, even though
//! v0.1 ships exactly one of them.
//!
//! The other half of that interface is the **accuracy tier** (§9). A result
//! that looks right and a result that *is* right are different products, and
//! the difference has to be visible where it is used rather than buried in
//! documentation. Every solver states its tier, and XPBD states `RealTime`.

#![deny(unsafe_code)]
// The public API freeze (PLAN.md §29, `docs/API.md`): an undocumented public
// item is one nobody agreed to freeze.
#![deny(missing_docs)]

pub mod aero;
pub mod constraint;
pub mod contact;
pub mod fluid;
pub mod integrate;
pub mod joint;
pub mod ledger;
pub mod soft;
pub mod xpbd;

use fusion_math::Vec3;

pub use aero::AirModel;
pub use contact::{ContactParameters, ContactScratch};
pub use fluid::FluidScratch;
pub use integrate::{integrate_velocities, predict_positions, recover_velocities};
pub use joint::JointScratch;
pub use ledger::{EnergyLedger, Stage};
pub use xpbd::{ContactReport, RememberedContact, SolverMemory, XpbdSolver, XpbdWorkingSet};

/// How a solver behaves numerically, as declared by the solver itself.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Stability {
    /// Conditionally stable: the timestep is bounded by the stiffest term.
    Explicit,
    /// Unconditionally stable, at the price of solving a system per step.
    Implicit,
    /// Explicit, but preserving phase-space volume — energy wanders and does
    /// not drift away, which is what makes long orbital runs behave.
    Symplectic,
    /// Constraints solved on positions, velocities derived afterwards. Stable
    /// at any stiffness and not energy-conserving; see [`AccuracyTier`].
    PositionBased,
}

/// What a solver conserves, as opposed to what it approximately preserves.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct Conserves {
    /// Linear and angular momentum, exactly, up to rounding.
    pub momentum: bool,
    /// Total mechanical energy. Position-based methods do not.
    pub energy: bool,
    /// Mass. False only for methods that add or remove material.
    pub mass: bool,
}

/// PLAN.md §9. The whole point of this enum is that it appears in the API and
/// the UI, not only in the docs.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord)]
pub enum AccuracyTier {
    /// Stable and plausible. Not conservative. Never present as a measurement.
    RealTime,
    /// Conserves what it claims to, and demonstrably converges under
    /// refinement. The realistic v1.0 ceiling.
    Consistent,
    /// Agrees with published reference results for a specific problem class.
    /// Claimed per benchmark, never per engine.
    Validated,
}

/// A solver's declaration about itself, surfaced in the API and the UI
/// rather than left in a document (PLAN.md §9).
#[derive(Clone, Copy, Debug)]
#[non_exhaustive]
pub struct SolverInfo {
    /// Human-readable name, e.g. `"XPBD"`.
    pub name: &'static str,
    /// Order of accuracy in the timestep.
    pub order: u8,
    /// How it behaves numerically.
    pub stability: Stability,
    /// What it conserves exactly, as opposed to approximately.
    pub conserves: Conserves,
    /// What its results may be used for.
    pub tier: AccuracyTier,
    /// One sentence on where this solver stops being trustworthy. Surfaced to
    /// the user alongside the tier.
    pub caveat: &'static str,
}

/// Everything a solver needs about the current step that is not state.
#[derive(Clone, Copy, Debug)]
pub struct StepContext {
    /// Full step duration. The solver subdivides it as it sees fit.
    pub dt: f32,
    /// Uniform acceleration applied to everything with mass.
    pub gravity: Vec3,
    /// Per-second velocity decay factors, applied as `v /= 1 + damping * h`,
    /// which is stable for any timestep.
    pub linear_damping: f32,
    /// Angular decay, same form.
    pub angular_damping: f32,
    /// The air the scene is moving through.
    pub air: AirModel,
    /// Attribute this step's energy change to the stage that caused it
    /// (PLAN.md §23). Off by default: it costs a pass over every body and
    /// particle ten times per substep.
    pub energy_ledger: bool,
    /// The narrow phase's base tolerance, metres: what a solver re-detecting
    /// a pair part-way through the step uses, as the pipeline did at the top.
    pub contact_tolerance: f32,
    /// The farthest a body's contacts reach ahead of it, metres. The same cap
    /// the pipeline applied.
    pub max_sweep_distance: f32,
    /// Make the energy the contacts report dissipating add up to what they
    /// took out of the bodies. Set while something reads it — the heat
    /// solve — because it costs two passes over the bodies a substep. See
    /// `XpbdWorkingSet::close_dissipation`.
    pub close_dissipation: bool,
}

impl Default for StepContext {
    fn default() -> Self {
        Self {
            dt: 1.0 / 60.0,
            gravity: fusion_math::GRAVITY,
            linear_damping: 0.0,
            angular_damping: 0.05,
            air: AirModel::default(),
            energy_ledger: false,
            contact_tolerance: fusion_collide::narrow::primitives::DEFAULT_CONTACT_TOLERANCE,
            max_sweep_distance: 2.0,
            close_dissipation: false,
        }
    }
}

/// Everything a solver is allowed to move, bundled.
///
/// Passing these as separate arguments worked while there were two of them.
/// With deformables there are five, and every future system — fluids, grids —
/// adds another. A bundle keeps the trait stable while the contents grow,
/// which is the point of having a solver interface at all (PLAN.md §5).
pub struct SolverScene<'a> {
    /// The rigid bodies.
    pub bodies: &'a mut fusion_core::BodySet,
    /// Cloth and soft-body particles, in one array shared by both.
    pub particles: &'a mut fusion_core::ParticleSet,
    /// Joints between bodies.
    pub joints: &'a [fusion_core::Joint],
    /// Distance constraints between particles — the weave of a cloth and the
    /// edges of a soft body.
    pub distance: &'a [fusion_core::DistanceConstraint],
    /// Bending constraints, which resist a fold rather than a stretch.
    pub bending: &'a [fusion_core::BendingConstraint],
    /// Triangles of the particle surfaces, for aerodynamics.
    pub triangles: &'a [fusion_core::SurfaceTriangle],
    /// Bodies of fluid. Each is solved against its own density constraint and
    /// against the rigid bodies; fluid-fluid interaction across two sets is
    /// not modelled, so oil genuinely does not sit on water yet.
    pub fluids: &'a mut [fusion_core::FluidSet],
}

/// A rigid body solver.
///
/// `prepare` builds whatever per-step scratch the method needs — for XPBD, the
/// previous poses and the Lagrange multipliers. Keeping it out of the body
/// state is deliberate: `BodySet` holds the physical state of the world, not
/// the bookkeeping of whichever solver happens to be running.
pub trait RigidSolver {
    /// Per-step scratch this solver needs, owned by the caller so that it
    /// survives between steps and can be checkpointed.
    type WorkingSet;

    /// Build the scratch for one step from the scene and its contacts.
    fn prepare(
        &self,
        scene: &SolverScene<'_>,
        manifolds: &[fusion_collide::ContactManifold],
        working: &mut Self::WorkingSet,
    );

    /// Advance the scene by `ctx.dt`, subdividing it however the method sees
    /// fit.
    fn step(&self, scene: &mut SolverScene<'_>, working: &mut Self::WorkingSet, ctx: &StepContext);

    /// What this solver declares about itself.
    fn info(&self) -> SolverInfo;
}
