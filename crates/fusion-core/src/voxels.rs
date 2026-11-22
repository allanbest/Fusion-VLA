//! A coarse voxel grid laid over a rigid body, in the body's own frame.
//!
//! The one lattice every interior field of a rigid body lives on: temperature
//! ([`crate::BodyHeat`]) and stress ([`crate::BodyStress`]) both index the same
//! cells, so a hot cell and a stressed cell are the same piece of material and
//! thermal stress reads one grid with the other's index.
//!
//! **Resolution is deliberately coarse.** About eight cells along the longest
//! side: enough to show that the face being dragged is hotter than the face on
//! top, or that a column carries more load at its foot than at its head, and
//! not enough to resolve a thin skin or a notch. A cell is either inside or
//! outside by its centre.

use fusion_math::Vec3;

use crate::shape::Shape;

/// Cells along a body's longest side.
pub const CELLS_ALONG_LONGEST: usize = 8;

/// Fewest cells across a body's thinnest side when it is gridded for stress.
///
/// Bending is what snaps a beam, and a beam one cell thick cannot show it: the
/// stress varies from tension on one face to compression on the other, and a
/// cell averages that to nothing. Four is the fewest that resolve it.
pub const STRESS_CELLS_ACROSS: usize = 4;

/// Cells along a body's longest side when it is gridded for stress, before
/// [`STRESS_CELLS_ACROSS`] asks for more across a thin one.
///
/// Fewer than heat's eight, because a stress solve is a sparse linear system
/// where heat's is a diffusion: a cube at six is 216 cells and at eight is
/// 512, and every step pays for the difference. Six resolve a block's load
/// paths; a beam's bending is what needs cells, and gets them across.
pub const STRESS_CELLS_ALONG_LONGEST: usize = 6;

/// Most cells a stress grid may have. Bounds what a long thin body costs,
/// at the price of resolving its bending less well.
pub const STRESS_MAX_CELLS: usize = 1200;

/// Most cells a broken piece's stress grid may have. A piece is smaller than
/// what it broke from and there can be many at once, each needing its grid
/// factored the moment it appears.
pub const STRESS_PIECE_CELLS: usize = 400;

/// Longest a stress cell may be against its shortest edge.
///
/// Stress cells are boxes rather than cubes, so a pane or a plank can have
/// several thin layers through its thickness without a cube's worth of cells
/// along its length. A trilinear element locks — grows stiffer than the
/// material in bending — as it gets more elongated, and by four to one that
/// is a few percent; past it the plank would be too strong.
pub const STRESS_ASPECT: f32 = 4.0;

/// Which cells of a box around a body hold material.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct Voxels {
    /// Cells along x, y and z. All zero for a grid that has not been built.
    pub dims: [usize; 3],
    /// Longest cell edge, metres: the cell size for a grid of cubes.
    pub cell: f32,
    /// Cell edges along x, y and z, metres. Equal for a grid of cubes; the
    /// stress grid's are not.
    pub size: Vec3,
    /// Centre of cell `(0, 0, 0)`, in the body's local frame.
    pub origin: Vec3,
    /// Whether each cell's centre is inside the shape. Outside cells exist so
    /// that indexing is a plain `x + nx * (y + ny * z)`.
    pub inside: Vec<bool>,
}

impl Voxels {
    /// Lay a grid over `shape` with [`CELLS_ALONG_LONGEST`] cells along its
    /// longest side. `None` for a shape with no finite extent.
    ///
    /// A shape thinner than a cell in every direction that matters still gets
    /// one inside cell, the one nearest its centre, so it has material to hold
    /// a field at all.
    pub fn of(shape: &Shape) -> Option<Self> {
        let aabb = shape.local_aabb();
        let longest = (aabb.max - aabb.min).max_element();
        Self::with_cell(shape, longest / CELLS_ALONG_LONGEST as f32)
    }

    /// The grid stress is solved on: box cells, at least
    /// [`STRESS_CELLS_ACROSS`] through the thinnest side, each no more than
    /// [`STRESS_ASPECT`] times longer than it is thick, and no more than
    /// [`STRESS_MAX_CELLS`] in all. Where the budget cannot hold that, the
    /// thinnest side drops to two layers and then everything coarsens
    /// together.
    pub fn for_stress(shape: &Shape) -> Option<Self> {
        Self::for_stress_within(shape, STRESS_MAX_CELLS)
    }

    /// [`Voxels::for_stress`] with at most `budget` cells.
    pub fn for_stress_within(shape: &Shape, budget: usize) -> Option<Self> {
        let aabb = shape.local_aabb();
        let size = aabb.max - aabb.min;
        let longest = size.max_element();
        if !(longest.is_finite() && longest > 0.0) {
            return None;
        }
        let extent = [size.x, size.y, size.z];
        let thinnest = extent
            .iter()
            .copied()
            .fold(f32::INFINITY, f32::min)
            .max(1e-6);
        let base = (longest / STRESS_CELLS_ALONG_LONGEST as f32)
            .min(thinnest / STRESS_CELLS_ACROSS as f32);
        // The thin direction at its layer edge, the others stretched by
        // `aspect` — as little as the budget allows.
        let count = |edge: f32, aspect: f32| -> [usize; 3] {
            std::array::from_fn(|k| {
                let e = extent[k];
                let along = if e <= thinnest * 1.0001 {
                    edge
                } else {
                    edge * aspect
                };
                ((e / along).round() as usize).max(1)
            })
        };
        let fits = |dims: [usize; 3]| dims.iter().product::<usize>() <= budget;
        let mut edge = base;
        let mut dims = count(edge, 1.0);
        for aspect in [1.5, 2.0, 3.0, STRESS_ASPECT] {
            if fits(dims) {
                break;
            }
            dims = count(edge, aspect);
        }
        // Then two layers through the thinnest side, then coarser everywhere.
        if !fits(dims) {
            edge = thinnest / 2.0;
            dims = count(edge, STRESS_ASPECT);
        }
        while !fits(dims) {
            edge *= 1.15;
            dims = count(edge, STRESS_ASPECT);
        }
        let mut grid = Self::with_dims(shape, dims)?;
        grid.keep_largest_piece();
        Some(grid)
    }

    /// A grid of `dims` box cells exactly filling the shape's bounds.
    pub fn with_dims(shape: &Shape, dims: [usize; 3]) -> Option<Self> {
        let aabb = shape.local_aabb();
        let extent = aabb.max - aabb.min;
        let size = Vec3::new(
            extent.x / dims[0].max(1) as f32,
            extent.y / dims[1].max(1) as f32,
            extent.z / dims[2].max(1) as f32,
        );
        Self::build(shape, dims, size)
    }

    /// Drop every inside cell not joined face-to-face to the largest group.
    ///
    /// An elastic solve needs its cells to hold together: two groups that
    /// touch only along an edge, or not at all, can move against each other
    /// for free, and a solver asked to find their stress never converges. A
    /// thin sliver broken off a sphere voxelizes into exactly that. The cells
    /// dropped are the ones too thin to carry anything at this resolution.
    pub fn keep_largest_piece(&mut self) {
        let mut label = vec![u32::MAX; self.len()];
        let mut best = (0usize, u32::MAX);
        let mut stack = Vec::new();
        let mut next = 0u32;
        for start in 0..self.len() {
            if !self.inside[start] || label[start] != u32::MAX {
                continue;
            }
            let mut count = 0;
            label[start] = next;
            stack.push(start);
            while let Some(i) = stack.pop() {
                count += 1;
                let [x, y, z] = self.coordinates(i);
                let mut visit = |j: usize, stack: &mut Vec<usize>| {
                    if self.inside[j] && label[j] == u32::MAX {
                        label[j] = next;
                        stack.push(j);
                    }
                };
                if x > 0 {
                    visit(self.index(x - 1, y, z), &mut stack);
                }
                if x + 1 < self.dims[0] {
                    visit(self.index(x + 1, y, z), &mut stack);
                }
                if y > 0 {
                    visit(self.index(x, y - 1, z), &mut stack);
                }
                if y + 1 < self.dims[1] {
                    visit(self.index(x, y + 1, z), &mut stack);
                }
                if z > 0 {
                    visit(self.index(x, y, z - 1), &mut stack);
                }
                if z + 1 < self.dims[2] {
                    visit(self.index(x, y, z + 1), &mut stack);
                }
            }
            if count > best.0 {
                best = (count, next);
            }
            next += 1;
        }
        if next > 1 {
            for (inside, l) in self.inside.iter_mut().zip(&label) {
                *inside = *l == best.1;
            }
        }
    }

    /// A grid of cubic cells of edge `cell`, centred on the shape.
    pub fn with_cell(shape: &Shape, cell: f32) -> Option<Self> {
        let aabb = shape.local_aabb();
        let size = aabb.max - aabb.min;
        if !(cell.is_finite() && cell > 0.0 && size.max_element().is_finite()) {
            return None;
        }
        let along = |extent: f32| ((extent / cell).round() as usize).max(1);
        let dims = [along(size.x), along(size.y), along(size.z)];
        Self::build(shape, dims, Vec3::splat(cell))
    }

    fn build(shape: &Shape, dims: [usize; 3], size: Vec3) -> Option<Self> {
        let aabb = shape.local_aabb();
        if !(size.min_element() > 0.0 && size.max_element().is_finite()) {
            return None;
        }
        let centre = (aabb.min + aabb.max) * 0.5;
        let origin = centre
            - Vec3::new(
                (dims[0] - 1) as f32,
                (dims[1] - 1) as f32,
                (dims[2] - 1) as f32,
            )
            .mul_element_wise(size * 0.5);
        let count = dims[0] * dims[1] * dims[2];
        let mut inside = vec![false; count];
        let mut any = false;
        for z in 0..dims[2] {
            for y in 0..dims[1] {
                for x in 0..dims[0] {
                    let p = origin + Vec3::new(x as f32, y as f32, z as f32).mul_element_wise(size);
                    let i = x + dims[0] * (y + dims[1] * z);
                    inside[i] = shape.signed_distance_local(p) <= 0.0;
                    any |= inside[i];
                }
            }
        }
        if !any {
            let mid = dims[0] / 2 + dims[0] * (dims[1] / 2 + dims[1] * (dims[2] / 2));
            inside[mid] = true;
        }
        Some(Self {
            dims,
            cell: size.max_element(),
            size,
            origin,
            inside,
        })
    }

    /// Whether the grid exists.
    #[inline]
    pub fn is_built(&self) -> bool {
        !self.inside.is_empty()
    }

    /// Number of cells, inside or not.
    #[inline]
    pub fn len(&self) -> usize {
        self.inside.len()
    }

    /// True where there is no grid.
    #[inline]
    pub fn is_empty(&self) -> bool {
        self.inside.is_empty()
    }

    /// How many cells hold material.
    pub fn inside_count(&self) -> usize {
        self.inside.iter().filter(|&&b| b).count()
    }

    /// Flat index of a cell.
    #[inline]
    pub fn index(&self, x: usize, y: usize, z: usize) -> usize {
        x + self.dims[0] * (y + self.dims[1] * z)
    }

    /// The three coordinates of a flat index.
    #[inline]
    pub fn coordinates(&self, i: usize) -> [usize; 3] {
        [
            i % self.dims[0],
            (i / self.dims[0]) % self.dims[1],
            i / (self.dims[0] * self.dims[1]),
        ]
    }

    /// Centre of a cell, body-local.
    #[inline]
    pub fn cell_centre(&self, i: usize) -> Vec3 {
        let [x, y, z] = self.coordinates(i);
        self.origin + Vec3::new(x as f32, y as f32, z as f32).mul_element_wise(self.size)
    }

    /// A value per cell, read at a body-local point: the trilinear blend of
    /// the eight cell centres around it, over the ones that hold material.
    /// What a field drawn on a body's surface reads, so it shades smoothly
    /// across the cells instead of in blocks. `None` off the grid, or where
    /// `values` is not one per cell.
    pub fn sample(&self, values: &[f32], p: Vec3) -> Option<f32> {
        if !self.is_built() || values.len() != self.len() {
            return None;
        }
        let local = (p - self.origin).div_element_wise(self.size);
        let base = |v: f32, n: usize| (v.floor().max(0.0) as usize).min(n.saturating_sub(2));
        let (bx, by, bz) = (
            base(local.x, self.dims[0]),
            base(local.y, self.dims[1]),
            base(local.z, self.dims[2]),
        );
        let frac = |v: f32, b: usize, n: usize| {
            if n < 2 {
                0.0
            } else {
                (v - b as f32).clamp(0.0, 1.0)
            }
        };
        let f = [
            frac(local.x, bx, self.dims[0]),
            frac(local.y, by, self.dims[1]),
            frac(local.z, bz, self.dims[2]),
        ];
        let (mut sum, mut weight) = (0.0f32, 0.0f32);
        for corner in 0..8 {
            let d = [corner & 1, (corner >> 1) & 1, (corner >> 2) & 1];
            let (x, y, z) = (bx + d[0], by + d[1], bz + d[2]);
            if x >= self.dims[0] || y >= self.dims[1] || z >= self.dims[2] {
                continue;
            }
            let i = self.index(x, y, z);
            if !self.inside[i] {
                continue;
            }
            let w = (0..3)
                .map(|k| if d[k] == 1 { f[k] } else { 1.0 - f[k] })
                .product::<f32>();
            sum += w * values[i];
            weight += w;
        }
        if weight > 1e-6 {
            Some(sum / weight)
        } else {
            self.nearest_inside(p).map(|i| values[i])
        }
    }

    /// How deep a cell is along a direction: its edge along that axis, for a
    /// direction along one.
    pub fn depth_along(&self, direction: Vec3) -> f32 {
        let d = direction.abs();
        let s = d.x + d.y + d.z;
        if s <= 0.0 {
            return self.cell;
        }
        (d.x * self.size.x + d.y * self.size.y + d.z * self.size.z) / s
    }

    /// The inside cell nearest a body-local point, or `None` without a grid.
    ///
    /// A contact point sits on the surface, which is usually half a cell from
    /// the nearest centre and sometimes in an outside cell's corner; clamping
    /// to the grid and then searching the immediate neighbourhood finds the
    /// cell that actually holds the material under it.
    pub fn nearest_inside(&self, p: Vec3) -> Option<usize> {
        if !self.is_built() {
            return None;
        }
        let local = (p - self.origin).div_element_wise(self.size);
        let clamp = |v: f32, n: usize| (v.round().max(0.0) as usize).min(n - 1);
        let (cx, cy, cz) = (
            clamp(local.x, self.dims[0]),
            clamp(local.y, self.dims[1]),
            clamp(local.z, self.dims[2]),
        );
        let i = self.index(cx, cy, cz);
        if self.inside[i] {
            return Some(i);
        }
        // Widen until something is found. Bounded by the grid, and a
        // convex shape's inside cells are never far from its surface.
        let reach = self.dims.iter().copied().max().unwrap_or(1);
        for r in 1..=reach as isize {
            let mut best: Option<(f32, usize)> = None;
            for dz in -r..=r {
                for dy in -r..=r {
                    for dx in -r..=r {
                        let (x, y, z) = (cx as isize + dx, cy as isize + dy, cz as isize + dz);
                        if x < 0
                            || y < 0
                            || z < 0
                            || x >= self.dims[0] as isize
                            || y >= self.dims[1] as isize
                            || z >= self.dims[2] as isize
                        {
                            continue;
                        }
                        let j = self.index(x as usize, y as usize, z as usize);
                        if !self.inside[j] {
                            continue;
                        }
                        let d = (self.cell_centre(j) - p).length_squared();
                        if best.is_none_or(|(b, _)| d < b) {
                            best = Some((d, j));
                        }
                    }
                }
            }
            if let Some((_, j)) = best {
                return Some(j);
            }
        }
        None
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_cube_fills_its_grid() {
        let grid = Voxels::of(&Shape::cuboid(Vec3::splat(1.0))).unwrap();
        assert_eq!(grid.dims, [8, 8, 8]);
        assert_eq!(grid.inside_count(), 512);
    }

    #[test]
    fn a_sphere_leaves_its_corners_out() {
        let grid = Voxels::of(&Shape::sphere(0.5)).unwrap();
        let fraction = grid.inside_count() as f32 / grid.len() as f32;
        // pi/6 of the cube, give or take the blockiness of eight cells.
        assert!(
            (fraction - std::f32::consts::FRAC_PI_6).abs() < 0.08,
            "{fraction}"
        );
    }

    #[test]
    fn a_thin_plate_still_has_material() {
        let grid = Voxels::of(&Shape::cuboid(Vec3::new(4.0, 0.01, 4.0))).unwrap();
        assert_eq!(grid.dims[1], 1);
        assert!(grid.inside_count() > 0);
    }

    #[test]
    fn a_surface_point_finds_the_cell_under_it() {
        let grid = Voxels::of(&Shape::sphere(0.5)).unwrap();
        let i = grid.nearest_inside(Vec3::new(0.0, -0.5, 0.0)).unwrap();
        assert!(grid.inside[i]);
        assert!(grid.cell_centre(i).y < -0.3, "{:?}", grid.cell_centre(i));
    }

    #[test]
    fn a_sample_blends_the_cells_around_it() {
        let grid = Voxels::with_dims(&Shape::cuboid(Vec3::new(4.0, 1.0, 1.0)), [4, 1, 1]).unwrap();
        let values = [0.0, 1.0, 2.0, 3.0];
        // On a centre: that cell's value. Halfway between two: their mean.
        assert_eq!(grid.sample(&values, grid.cell_centre(1)), Some(1.0));
        let between = (grid.cell_centre(1) + grid.cell_centre(2)) * 0.5;
        assert!((grid.sample(&values, between).unwrap() - 1.5).abs() < 1e-5);
        // Past the last centre, out at the surface: the last value, not more.
        let end = grid.cell_centre(3) + Vec3::X * 0.5;
        assert!((grid.sample(&values, end).unwrap() - 3.0).abs() < 1e-5);
    }

    #[test]
    fn a_beam_is_four_cells_thick_for_stress() {
        let grid = Voxels::for_stress(&Shape::cuboid(Vec3::new(4.0, 0.4, 0.4))).unwrap();
        assert_eq!(&grid.dims[1..], &[4, 4]);
        assert!(
            grid.size.x / grid.size.y <= STRESS_ASPECT + 1e-3,
            "{:?}",
            grid.size
        );
        // A pane: at least two layers through it, within budget.
        let pane = Voxels::for_stress(&Shape::cuboid(Vec3::new(0.8, 0.02, 0.6))).unwrap();
        assert!(pane.dims[1] >= 2, "{:?}", pane.dims);
        assert!(pane.len() <= STRESS_MAX_CELLS, "{:?}", pane.dims);
    }

    #[test]
    fn only_the_largest_connected_piece_is_kept() {
        let mut grid = Voxels::of(&Shape::cuboid(Vec3::splat(1.0))).unwrap();
        // Cut the cube into a 3-cell slab and a 5-cell slab with no face
        // between them.
        for i in 0..grid.len() {
            if grid.coordinates(i)[0] == 3 {
                grid.inside[i] = false;
            }
        }
        grid.keep_largest_piece();
        assert_eq!(grid.inside_count(), 4 * 64);
        assert!((0..grid.len()).all(|i| !grid.inside[i] || grid.coordinates(i)[0] > 3));
    }

    #[test]
    fn coordinates_round_trip() {
        let grid = Voxels::of(&Shape::cuboid(Vec3::new(1.0, 0.5, 0.25))).unwrap();
        for i in 0..grid.len() {
            let [x, y, z] = grid.coordinates(i);
            assert_eq!(grid.index(x, y, z), i);
        }
    }
}
