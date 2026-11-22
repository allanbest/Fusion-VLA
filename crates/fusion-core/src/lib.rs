//! Fusion L2 — simulation primitives.
//!
//! PLAN.md §5 lists seven primitives that everything in either mode composes
//! from, and warns against a universal simulation object. v0.1 needs three of
//! them — `State` (here, as [`BodySet`]), `Shape`, and `Material` — plus
//! [`Domain`]. `ParticleSet` arrived with cloth in v0.3 and is reused
//! unchanged by fluids in v0.4, which is the composition §5 asks for working
//! as advertised. `Field` and the rest join them as the systems that need
//! them are built.
//!
//! Nothing in this crate knows about collision detection, solvers, rendering,
//! or windows.

#![deny(unsafe_code)]
// The public API freeze (PLAN.md §29, `docs/API.md`): an undocumented public
// item is one nobody agreed to freeze.
#![deny(missing_docs)]

pub mod body;
pub mod cloth;
pub mod convex;
pub mod domain;
pub mod fluid;
pub mod heat;
pub mod joint;
pub mod material;
pub mod particle;
pub mod rng;
pub mod shape;
pub mod solid;
pub mod stress;
pub mod voxels;
pub mod world;

pub use body::{BodyHandle, BodySet, BodyType, BodyView};
pub use cloth::{build_cloth, build_soft_body, ClothDesc, ClothHandles, SoftBodyDesc};
pub use convex::{ConvexHull, HullFace, SharedHull};
pub use domain::Domain;
pub use fluid::{build_fluid, fill, FluidDesc, FluidHandle, FluidParams, FluidSet};
pub use heat::BodyHeat;
pub use joint::{Joint, JointHandle, JointKind};
pub use material::{Assessment, Grain, Material, Parting, Pattern, Strength};
pub use particle::{
    BendingConstraint, DistanceConstraint, ParticleHandle, ParticleSet, SurfaceTriangle,
};
pub use rng::Rng;
pub use shape::{MassProperties, Shape};
pub use solid::{
    build_solid, Constitutive, Fibre, Fracture, SolidDesc, SolidHandle, SolidMaterial, SolidParams,
    SolidSet, Thermal,
};
pub use stress::{BodyStress, Failure};
pub use voxels::Voxels;
pub use world::{BodyDesc, World};

pub use fusion_math as math;
