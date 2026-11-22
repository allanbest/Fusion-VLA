//! Layer 3 of the Fusion core: space — what is where, and what is near.
//!
//! - [`Aabb`]: axis-aligned bounds.
//! - [`Shape`]: analytic shapes with signed distance, bounds and mass.
//! - [`Bvh`]: which boxes overlap, for contact.
//! - [`NeighbourGrid`]: which points are near, for particles.
//! - [`closest_on_triangle`]: point–triangle distance, for mesh contact.
//!
//! Sparse adaptive grids (for fields) and multipole trees (for long-range
//! forces) join with the modules that need them.

mod aabb;
mod bvh;
mod neighbours;
mod segment;
mod shape;
mod tree;
mod triangle;

pub use aabb::Aabb;
pub use bvh::Bvh;
pub use neighbours::NeighbourGrid;
pub use segment::closest_segments;
pub use shape::{Shape, UnitMass};
pub use tree::{BarnesHut, Interaction};
pub use triangle::closest_on_triangle;
