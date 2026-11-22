//! Finding points near a point: a uniform grid of cells, stored sorted so
//! that it needs no hashing and visits neighbours in a fixed order.

use core_math::Vec3;

type Cell = [i64; 3];

/// Points binned into cubic cells.
#[derive(Clone, Debug, Default)]
pub struct NeighbourGrid {
    cell: f64,
    /// `(cell, point)`, sorted.
    entries: Vec<(Cell, u32)>,
}

impl NeighbourGrid {
    /// Bins `points` into cells of side `cell` — best about the search
    /// radius.
    pub fn build(points: &[Vec3], cell: f64) -> Self {
        let mut grid = Self {
            cell,
            entries: Vec::new(),
        };
        grid.rebuild(points);
        grid
    }

    /// Bins again after the points moved, keeping the memory.
    pub fn rebuild(&mut self, points: &[Vec3]) {
        let cell = self.cell;
        self.entries.clear();
        self.entries.extend(
            points
                .iter()
                .enumerate()
                .map(|(i, &p)| (cell_of(cell, p), i as u32)),
        );
        self.entries.sort_unstable();
    }

    /// Calls `found` with every point within `radius` of `p` — of the
    /// points the grid was built with — in a fixed order.
    pub fn within(&self, points: &[Vec3], p: Vec3, radius: f64, mut found: impl FnMut(usize)) {
        let lo = cell_of(self.cell, p - Vec3::splat(radius));
        let hi = cell_of(self.cell, p + Vec3::splat(radius));
        let r2 = radius * radius;
        for z in lo[2]..=hi[2] {
            for y in lo[1]..=hi[1] {
                for x in lo[0]..=hi[0] {
                    let cell = [x, y, z];
                    let start = self.entries.partition_point(|e| e.0 < cell);
                    for &(c, i) in &self.entries[start..] {
                        if c != cell {
                            break;
                        }
                        if (points[i as usize] - p).length_squared() <= r2 {
                            found(i as usize);
                        }
                    }
                }
            }
        }
    }
}

fn cell_of(cell: f64, p: Vec3) -> Cell {
    [
        (p.x / cell).floor() as i64,
        (p.y / cell).floor() as i64,
        (p.z / cell).floor() as i64,
    ]
}
