//! Layer 4 of the Fusion core: the world as data.
//!
//! ```text
//! World
//!  ├─ materials   what things are made of            Material
//!  ├─ objects     a material + a representation      Object, Representation
//!  ├─ links       anchors, joints, springs and welds Link
//!  └─ gravity     the uniform field
//! ```
//!
//! No physics happens here: modules read and write this state. Fields
//! (temperature, flow, electromagnetism) and probes join with the first
//! module that needs them.

mod columns;
mod material;
mod mesh;
mod mjcf;
mod object;
mod urdf;
mod world;
mod xml;

pub use columns::{Column, Columns};
pub use material::{library, Fracture, Material, Plasticity, Property, Thermal};
pub use mesh::{Surface, TetMesh, EDGES};
pub use object::{Fixed, Nodes, Object, Representation, Rigid, Solid};
pub use urdf::RobotModel;
pub use world::{
    ortho_basis, JointDrive, Link, LinkId, LinkPoint, MaterialId, ObjectId, Probe, ProbeId,
    ProbeReading, Totals, World,
};
