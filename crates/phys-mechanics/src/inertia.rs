//! Inertia and uniform gravity.

use core_math::{Mat3, Vec3};
use core_numerics::{BlockMatrix, Triplets};

use crate::term::{Context, Term};

/// `Σ ½ (x − x̃)ᵀ (M/β²) (x − x̃)`: what keeps each unknown on its course.
/// The integrator sets `x̃` and `M/β²` per block each step.
#[derive(Clone, Debug, Default)]
pub struct Inertia {
    /// Per block, `x̃`.
    pub target: Vec<f64>,
    /// Per block, `M / β²` (a scalar mass for points, a 3×3 inertia for
    /// rotations).
    pub mass: Vec<Mat3>,
    slots: Vec<usize>,
}

impl Term for Inertia {
    fn name(&self) -> &'static str {
        "inertia"
    }

    fn pattern(&self, ctx: &Context, entries: &mut Vec<(usize, usize)>) {
        entries.extend((0..ctx.layout.blocks).map(|b| (b, b)));
    }

    fn bind(&mut self, ctx: &Context, matrix: &BlockMatrix) {
        self.slots = (0..ctx.layout.blocks)
            .map(|b| matrix.slot(b, b).unwrap())
            .collect();
    }

    fn energy(&mut self, ctx: &Context, x: &[f64]) -> f64 {
        ctx.pool.map_reduce(
            self.mass.len(),
            1024,
            0.0,
            |range| {
                range
                    .map(|b| {
                        let d = Vec3::read(x, b) - Vec3::read(&self.target, b);
                        0.5 * d.dot(self.mass[b] * d)
                    })
                    .sum::<f64>()
            },
            |a, b| a + b,
        )
    }

    fn derivatives(
        &mut self,
        _ctx: &Context,
        x: &[f64],
        gradient: &mut [f64],
        fixed: &mut BlockMatrix,
        _changing: &mut Triplets,
    ) {
        let values = fixed.values_mut();
        for (b, m) in self.mass.iter().enumerate() {
            let d = Vec3::read(x, b) - Vec3::read(&self.target, b);
            (*m * d).add_to(gradient, b);
            values[self.slots[b]] += *m;
        }
    }
}

/// `−h² Σ m g·x` over everything that has weight.
#[derive(Clone, Debug, Default)]
pub struct Gravity {
    /// `(block, mass)`.
    pub weights: Vec<(usize, f64)>,
}

impl Term for Gravity {
    fn name(&self) -> &'static str {
        "gravity"
    }

    fn energy(&mut self, ctx: &Context, x: &[f64]) -> f64 {
        let g = ctx.world.gravity * ctx.frame.weight;
        -self
            .weights
            .iter()
            .map(|&(b, m)| m * g.dot(Vec3::read(x, b)))
            .sum::<f64>()
    }

    fn derivatives(
        &mut self,
        ctx: &Context,
        _x: &[f64],
        gradient: &mut [f64],
        _fixed: &mut BlockMatrix,
        _changing: &mut Triplets,
    ) {
        let g = ctx.world.gravity * ctx.frame.weight;
        for &(b, m) in &self.weights {
            (g * -m).add_to(gradient, b);
        }
    }
}
