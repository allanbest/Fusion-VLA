//! Layer 5 of the Fusion core: heat transfer.
//!
//! Solves implicit finite-element heat conduction over linear and quadratic
//! tetrahedral solids with lumped thermal capacitance, volumetric heat
//! generation, convective cooling, linearized radiation and Dirichlet
//! boundary conditions, writing [`core_world::Column::TEMPERATURE`].

use core_runtime::Pool;
use core_world::World;

mod conduction;

pub use conduction::{
    HeatConduction, ThermalBoundary, ThermalIntegrator, ThermalReport, ThermalSettings,
};

/// Interface implemented by physical field processes integrated over [`World`].
pub trait Process {
    type Report;

    fn prepare(&mut self, pool: &Pool, world: &World);
    fn step(&mut self, pool: &Pool, world: &mut World, h: f64) -> Self::Report;
}
