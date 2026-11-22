//! The background grid: the "deep grid" of PLAN.md §11, one step at a time.
//!
//! MPM's grid holds nothing between steps. It exists to do the one thing
//! particles are bad at — letting neighbours talk to each other without
//! anybody having to work out who the neighbours are — and is thrown away
//! before the step ends. That is why there is no grid on [`SolidSet`] and no
//! grid in the project file: it is scratch, in the same sense the contact
//! manifolds are scratch.

use fusion_math::{to_f32, DVec3, Vec3};

/// Nodes past which a step is refused rather than allocated.
///
/// The grid is dense over the material's bounding box, so a solid that has
/// exploded asks for a box the size of its worst particle and an allocation
/// that will not come back. Refusing is recoverable and reportable; the
/// alternative is the process dying with no message, which is the one
/// behaviour worse than a wrong answer.
pub const MAX_NODES: usize = 8_000_000;

/// A uniform grid over one solid, rebuilt every step.
#[derive(Clone, Debug, Default)]
pub struct Grid {
    origin: DVec3,
    cell: f32,
    dims: [usize; 3],
    floor: f32,
    /// Nodal mass, kg.
    pub mass: Vec<f32>,
    /// Nodal momentum during the transfer in, nodal velocity after.
    ///
    /// One array rather than two because the second is the first divided by
    /// the mass, and keeping both would mean a pass to fill the one nothing
    /// has read yet.
    pub velocity: Vec<Vec3>,
    /// Nodal specific enthalpy, J/kg — mass-weighted from the points (v0.9).
    ///
    /// In double, like [`SolidSet::enthalpy`](fusion_core::SolidSet::enthalpy)
    /// and for the measured reason given there: the value is of order 10^5 and
    /// a sub-cycle's flux is of order 10^-3, so single precision cannot hold
    /// the sum of the two.
    pub enthalpy: Vec<f64>,
    /// What the diffusion pass wants to add to it, J/kg.
    ///
    /// Separate from `enthalpy` because the Laplacian has to read the state
    /// the whole grid started the pass in. Written in place, a node updated
    /// early would feed its new value to its neighbour's stencil, which is
    /// Gauss–Seidel rather than the explicit scheme this is, and diffuses at a
    /// rate that depends on the order the nodes happen to be visited in.
    pub enthalpy_delta: Vec<f64>,
    /// The nodal enthalpy before the diffusion pass, so the *total* change
    /// across every sub-cycle can be differenced out at the end.
    pub enthalpy_start: Vec<f64>,
    /// The nodal mass the *thermal* transfer divides by, kg.
    ///
    /// The same quantity as [`Grid::mass`] and deliberately not the same
    /// number. A node's specific enthalpy is a weighted mean — a sum of
    /// `w m h` over a sum of `w m` — and if the two sums are not accumulated
    /// the same way the quotient is not the mean.
    ///
    /// Dividing the `f64` numerator by the `f32` momentum mass gives each node
    /// a relative error of about `1e-7`, which on an enthalpy of `2.6e5 J/kg`
    /// is **6 x 10^-5 K of temperature that is not there** — measured, on a
    /// solid at a perfectly uniform temperature. The diffusion pass then does
    /// what it is supposed to do with a temperature difference, and a scene
    /// with a gradient finer than that never settles.
    ///
    /// Accumulated from the identical summand as the numerator, so a uniform
    /// field stays uniform to the last bit.
    pub enthalpy_mass: Vec<f64>,
}

impl Grid {
    /// An empty grid. Its arrays are sized on the first build and reused.
    pub fn new() -> Self {
        Self::default()
    }

    /// Size and clear the grid to cover `points` with two cells of margin,
    /// returning false where that would need more than [`MAX_NODES`].
    ///
    /// The margin is two cells rather than one: a quadratic B-spline reaches
    /// one cell either side of the node it is centred on, and a particle in
    /// the outermost cell must still find nodes beyond it to give its momentum
    /// to. One cell of margin loses that momentum silently, and what it looks
    /// like is a solid that sheds mass from its own surface.
    pub fn rebuild(&mut self, points: &[DVec3], cell: f32, point_mass: f32) -> bool {
        self.cell = cell.max(1e-6);
        // A billionth of one material point, which is the "this node holds no
        // material" threshold. See [`Grid::has_mass`] for why zero is not it.
        self.floor = (point_mass * 1e-9).max(f32::MIN_POSITIVE);
        self.dims = [0; 3];
        self.mass.clear();
        self.velocity.clear();
        self.enthalpy.clear();
        self.enthalpy_delta.clear();
        self.enthalpy_start.clear();
        self.enthalpy_mass.clear();
        if points.is_empty() {
            return true;
        }

        let (mut min, mut max) = (points[0], points[0]);
        for p in points {
            min = min.min(*p);
            max = max.max(*p);
        }
        // Snapped to the cell lattice for the reason the fluid's spatial hash
        // snaps its own: an origin taken from the raw minimum moves whenever
        // the lowest particle does, and every node in the grid then moves with
        // it. A solid at rest would shimmer because one particle at its edge
        // twitched.
        let cell64 = self.cell as f64;
        self.origin = DVec3::new(
            (min.x / cell64).floor() * cell64 - 2.0 * cell64,
            (min.y / cell64).floor() * cell64 - 2.0 * cell64,
            (min.z / cell64).floor() * cell64 - 2.0 * cell64,
        );
        let span = max - self.origin;
        let dims = [
            (span.x / cell64).ceil() as usize + 4,
            (span.y / cell64).ceil() as usize + 4,
            (span.z / cell64).ceil() as usize + 4,
        ];
        let count = dims[0].saturating_mul(dims[1]).saturating_mul(dims[2]);
        if count > MAX_NODES {
            return false;
        }
        self.dims = dims;
        self.mass.resize(count, 0.0);
        self.velocity.resize(count, Vec3::ZERO);
        self.enthalpy.resize(count, 0.0);
        self.enthalpy_delta.resize(count, 0.0);
        self.enthalpy_start.resize(count, 0.0);
        self.enthalpy_mass.resize(count, 0.0);
        true
    }

    /// True where a node holds enough material to divide by.
    ///
    /// **Not `mass > 0.0`**, and the difference is a NaN rather than a nicety.
    /// A quadratic B-spline weight goes smoothly to zero at the edge of its
    /// support, so a material point sitting exactly on a half-cell boundary
    /// gives its outermost node a weight around `1e-14` — cube that for three
    /// axes and the node's mass is `1e-42`, which is a denormal.
    ///
    /// `Vec3 / f32` divides by multiplying by the reciprocal, and the
    /// reciprocal of `1e-42` overflows to infinity. Zero momentum times
    /// infinity is NaN, and one NaN node poisons every particle whose stencil
    /// touches it — eight of them by the next step, the whole solid within a
    /// second.
    ///
    /// It depends on where the sampling lattice falls against the grid, which
    /// is why it survived v0.8: every scene there happened to be aligned so
    /// that no weight landed that close to zero. It appeared the first time a
    /// spacing was chosen that put a block's corner exactly on a boundary.
    #[inline]
    pub fn has_mass(&self, node: usize) -> bool {
        self.mass[node] > self.floor
    }

    /// The mass below which a node is treated as empty.
    #[inline]
    pub fn mass_floor(&self) -> f32 {
        self.floor
    }

    /// Cell size, metres.
    #[inline]
    pub fn cell(&self) -> f32 {
        self.cell
    }

    /// Node counts along each axis.
    #[inline]
    pub fn dims(&self) -> [usize; 3] {
        self.dims
    }

    /// How many nodes there are.
    #[inline]
    pub fn len(&self) -> usize {
        self.mass.len()
    }

    /// True where the grid holds no nodes.
    #[inline]
    pub fn is_empty(&self) -> bool {
        self.mass.is_empty()
    }

    /// A world point in the grid's own frame, in cells.
    #[inline]
    pub fn to_local(&self, p: DVec3) -> Vec3 {
        to_f32(p - self.origin) / self.cell
    }

    /// Where node `(i, j, k)` is, in world space.
    #[inline]
    pub fn node_position(&self, i: usize, j: usize, k: usize) -> Vec3 {
        to_f32(self.origin) + Vec3::new(i as f32, j as f32, k as f32) * self.cell
    }

    /// Flat index of a node, unchecked against the dimensions.
    #[inline]
    pub fn index(&self, i: usize, j: usize, k: usize) -> usize {
        i + self.dims[0] * (j + self.dims[1] * k)
    }

    /// The base node of the 3x3x3 stencil around a point in local (cell)
    /// coordinates, and the offset of the point from it.
    ///
    /// Quadratic B-splines are centred on nodes rather than on cells, so the
    /// half-cell shift is not a fudge: it puts the point in the middle of the
    /// three nodes whose weights are about to be evaluated, which is where the
    /// weights sum to one.
    #[inline]
    pub fn stencil(&self, local: Vec3) -> ([usize; 3], Vec3) {
        let base = [
            (local.x - 0.5).floor(),
            (local.y - 0.5).floor(),
            (local.z - 0.5).floor(),
        ];
        let fx = Vec3::new(local.x - base[0], local.y - base[1], local.z - base[2]);
        (
            [
                base[0].max(0.0) as usize,
                base[1].max(0.0) as usize,
                base[2].max(0.0) as usize,
            ],
            fx,
        )
    }

    /// True where the whole 3x3x3 stencil at `base` is inside the grid.
    #[inline]
    pub fn holds(&self, base: [usize; 3]) -> bool {
        base[0] + 2 < self.dims[0] && base[1] + 2 < self.dims[1] && base[2] + 2 < self.dims[2]
    }
}

/// Quadratic B-spline weights for the three nodes along one axis.
///
/// `fx` is the offset from the base node in cells and lies in `[0.5, 1.5]`, so
/// each of the three is non-negative and the three sum to one — a property
/// worth stating because it is what makes the transfer conserve mass exactly
/// rather than approximately.
///
/// Quadratic rather than linear because the gradient of a linear spline is
/// piecewise constant, so a particle crossing a cell boundary sees the force
/// on it jump. That shows up as a solid that buzzes while sitting still, and
/// no amount of damping removes it because it is being re-injected every step.
#[inline]
pub fn weights(fx: f32) -> [f32; 3] {
    let a = 1.5 - fx;
    let b = fx - 1.0;
    let c = fx - 0.5;
    [0.5 * a * a, 0.75 - b * b, 0.5 * c * c]
}
