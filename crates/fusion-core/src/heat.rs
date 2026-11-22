//! Temperature inside a rigid body.
//!
//! A rigid body never deforms, but it still gets hot: a brake disc, a block
//! dragged across concrete, a steel ball dropped on an anvil, a hot ingot set
//! down on a cold plate. None of that needs the continuum — the shape is fixed,
//! so the heat equation can be solved on a grid that is fixed *in the body* and
//! moves with it, and the mechanics stays rigid.
//!
//! This module is the state: one coarse voxel grid per body, in the body's own
//! frame. The solve that moves heat through it lives in `fusion-sim`, because
//! it needs the contacts. A body whose grid has not been built carries one
//! uniform temperature and costs nothing, so a scene with heat switched off
//! pays for one `f64` per body.
//!
//! The grid is a [`Voxels`], shared with stress: every inside cell carries an
//! equal share of the body's *true* mass, so the total heat capacity is right
//! even where the voxelization of a sphere is blocky.

use fusion_math::Vec3;

use crate::shape::Shape;
use crate::solid::AMBIENT;
use crate::voxels::Voxels;
pub use crate::voxels::CELLS_ALONG_LONGEST;

/// The temperature field of one rigid body, in its own frame.
#[derive(Clone, Debug, PartialEq)]
pub struct BodyHeat {
    /// The cells, empty until [`BodyHeat::build`]. Outside cells hold no heat
    /// and are skipped by every loop.
    pub grid: Voxels,
    /// Temperature per cell, Kelvin. Double precision for the same reason
    /// body positions are: heat is an accumulation, and a slow source on a
    /// large capacity — a sliding block on a massive floor — adds increments
    /// a single-precision kelvin would round away.
    pub temperature: Vec<f64>,
    /// The temperature of the whole body while it has no grid.
    pub uniform: f64,
}

impl Default for BodyHeat {
    fn default() -> Self {
        Self::uniform(AMBIENT)
    }
}

impl BodyHeat {
    /// A body at one temperature everywhere, with no grid yet.
    pub fn uniform(temperature: f32) -> Self {
        Self {
            grid: Voxels::default(),
            temperature: Vec::new(),
            uniform: temperature as f64,
        }
    }

    /// Whether the grid exists.
    #[inline]
    pub fn is_built(&self) -> bool {
        !self.temperature.is_empty()
    }

    /// Lay a grid over `shape` with [`CELLS_ALONG_LONGEST`] cells along its
    /// longest side, every inside cell at the current uniform temperature.
    ///
    /// Building twice is a no-op, so the solve can call it on every body it
    /// meets without asking.
    pub fn build(&mut self, shape: &Shape) {
        if self.is_built() {
            return;
        }
        let Some(grid) = Voxels::of(shape) else {
            return;
        };
        self.temperature = vec![self.uniform; grid.len()];
        self.grid = grid;
    }

    /// Number of cells, inside or not.
    #[inline]
    pub fn len(&self) -> usize {
        self.temperature.len()
    }

    /// True where there is no grid.
    #[inline]
    pub fn is_empty(&self) -> bool {
        self.temperature.is_empty()
    }

    /// How many cells hold heat.
    pub fn inside_count(&self) -> usize {
        self.grid.inside_count()
    }

    /// Centre of a cell, body-local.
    #[inline]
    pub fn cell_centre(&self, i: usize) -> Vec3 {
        self.grid.cell_centre(i)
    }

    /// The inside cell nearest a body-local point, or `None` without a grid.
    pub fn nearest_inside(&self, p: Vec3) -> Option<usize> {
        self.grid.nearest_inside(p)
    }

    /// Temperature at a body-local point: the nearest inside cell's, or the
    /// uniform temperature without a grid.
    pub fn at_local(&self, p: Vec3) -> f64 {
        match self.nearest_inside(p) {
            Some(i) => self.temperature[i],
            None => self.uniform,
        }
    }

    /// Mean temperature over the material. Every inside cell carries the same
    /// mass, so the plain mean is the mass-weighted one.
    pub fn mean(&self) -> f64 {
        if !self.is_built() {
            return self.uniform;
        }
        let (sum, n) = self
            .temperature
            .iter()
            .zip(&self.grid.inside)
            .filter(|(_, &inside)| inside)
            .fold((0.0, 0usize), |(s, n), (&t, _)| (s + t, n + 1));
        sum / n.max(1) as f64
    }

    /// Hottest and coldest cell, Kelvin.
    pub fn range(&self) -> (f64, f64) {
        if !self.is_built() {
            return (self.uniform, self.uniform);
        }
        self.temperature
            .iter()
            .zip(&self.grid.inside)
            .filter(|(_, &inside)| inside)
            .fold((f64::INFINITY, f64::NEG_INFINITY), |(lo, hi), (&t, _)| {
                (lo.min(t), hi.max(t))
            })
    }

    /// Set the whole body to one temperature, keeping its grid if it has one.
    pub fn set_uniform(&mut self, temperature: f64) {
        self.uniform = temperature;
        for t in &mut self.temperature {
            *t = temperature;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_built_grid_starts_at_the_uniform_temperature() {
        let mut heat = BodyHeat::uniform(300.0);
        heat.build(&Shape::cuboid(Vec3::splat(1.0)));
        assert_eq!(heat.grid.dims, [8, 8, 8]);
        assert_eq!(heat.inside_count(), 512);
        assert_eq!(heat.mean(), 300.0);
    }

    #[test]
    fn building_twice_keeps_the_field() {
        let mut heat = BodyHeat::uniform(300.0);
        heat.build(&Shape::sphere(0.5));
        heat.temperature[0] = 500.0;
        heat.build(&Shape::sphere(0.5));
        assert_eq!(heat.temperature[0], 500.0);
    }
}
