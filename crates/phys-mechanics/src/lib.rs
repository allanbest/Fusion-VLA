//! Layer 5 of the Fusion core: mechanics.
//!
//! Each independent contact island minimises one incremental potential over
//! its finite-element solids and rigid bodies, so that they push, carry
//! and rub on each other in the same solve:
//!
//! | term | what it is |
//! |---|---|
//! | [`Inertia`] | each unknown's mass pulling it along its course |
//! | [`Gravity`] | a uniform field |
//! | [`MutualGravity`] | long-range N-body gravitation via Barnes–Hut |
//! | [`Elastic`] | hyperelastic finite elements ([`law`]) |
//! | [`Joints`] | springs, ball joints and welds |
//! | [`Contact`] | barrier contact and smoothed Coulomb friction |
//!
//! [`Mechanics`] steps a [`core_world::World`] with them.

pub mod batch;
mod checkpoint;
mod contact;
pub mod dofs;
mod elastic;
mod gravity;
mod inertia;
mod joints;
pub mod law;
pub mod rigid;
mod step;
mod term;

pub use batch::{DomainRandomizer, WorldBatch};
pub use contact::{Contact, Impact};
pub use elastic::{nodal_deformation, strain_energy, Elastic};
pub use gravity::{MutualGravity, MutualGravitySettings};
pub use inertia::{Gravity, Inertia};
pub use joints::Joints;
pub use rigid::RigidBodies;
pub use step::{
    AdjointGradients, AdjointLoss, EnergyLedger, Integrator, MechanicalEnergy, Mechanics,
    MechanicsState, Settings, StepReport,
};
pub use term::{Context, Term};
