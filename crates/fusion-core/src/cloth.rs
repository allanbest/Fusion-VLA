//! Cloth: a grid of particles wired together with constraints.
//!
//! PLAN.md §16 is emphatic that nobody places a million particles by hand, so
//! cloth is authored as a description — size, resolution, where it hangs from
//! — and built into particles and constraints. This is the same
//! scene-then-state split the rigid bodies use.
//!
//! Three families of constraint, which between them are what makes fabric feel
//! like fabric:
//!
//! - **Structural**, along the weave. Resists stretching.
//! - **Shear**, across the diagonals. Without it the grid folds flat like a
//!   trellis, because a square with four fixed edges is not rigid.
//! - **Bending**, between particles two apart. Resists creasing, and is what
//!   separates silk from cardboard.

use fusion_math::{Isometry, Quat, Vec3};

use crate::particle::{
    BendingConstraint, DistanceConstraint, ParticleHandle, ParticleSet, SurfaceTriangle,
};

/// A rectangular sheet of cloth.
#[derive(Clone, Copy, Debug)]
pub struct ClothDesc {
    /// Particles along each edge. Cost is the product, and constraint count
    /// grows with it — this is the main dial on how expensive a cloth is.
    pub columns: usize,
    /// Particles down the sheet.
    pub rows: usize,
    /// Spacing between adjacent particles, metres.
    pub spacing: f32,
    /// Total mass of the sheet, kg.
    pub mass: f32,
    /// Where the sheet sits. Built in the local XZ plane, so an identity
    /// rotation gives a horizontal sheet.
    pub pose: Isometry,
    /// Inverse stiffness along the weave, m/N. Zero is inextensible.
    pub stretch_compliance: f32,
    /// Inverse stiffness along the diagonals, which is what resists shear.
    pub shear_compliance: f32,
    /// Inverse stiffness against folding.
    pub bend_compliance: f32,
    /// Collision stand-off. Cloth is a surface with no thickness of its own.
    pub thickness: f32,
}

impl Default for ClothDesc {
    fn default() -> Self {
        Self {
            columns: 24,
            rows: 24,
            spacing: 0.1,
            mass: 1.0,
            pose: Isometry::IDENTITY,
            // Near-inextensible along the weave, floppy in bending. Fabric
            // stretches very little and folds very easily, and getting that
            // ratio right matters more than either absolute value.
            stretch_compliance: 0.0,
            shear_compliance: 1e-6,
            bend_compliance: 5e-4,
            thickness: 0.01,
        }
    }
}

impl ClothDesc {
    /// A sheet of `columns` by `rows` particles, `spacing` metres apart.
    pub fn new(columns: usize, rows: usize, spacing: f32) -> Self {
        Self {
            columns,
            rows,
            spacing,
            ..Default::default()
        }
    }

    /// Builder: place its centre here.
    pub fn at(mut self, position: Vec3) -> Self {
        self.pose.translation = position;
        self
    }

    /// Builder: orient it. A sheet is created in the xz plane.
    pub fn rotated(mut self, rotation: Quat) -> Self {
        self.pose.rotation = rotation;
        self
    }

    /// Builder: total mass of the sheet, kg, divided evenly.
    pub fn with_mass(mut self, mass: f32) -> Self {
        self.mass = mass;
        self
    }

    /// Builder: how readily it folds. See [`ClothDesc::bend_compliance`].
    pub fn with_bend_compliance(mut self, compliance: f32) -> Self {
        self.bend_compliance = compliance;
        self
    }

    /// How many particles this description would create.
    pub fn particle_count(&self) -> usize {
        self.columns * self.rows
    }
}

/// Everything one cloth contributed to a world, so a caller can address it
/// afterwards — pin a corner, read its shape, apply a wind gust.
#[derive(Clone, Debug)]
pub struct ClothHandles {
    /// Particle handles in row-major order: `index(column, row)`.
    pub particles: Vec<ParticleHandle>,
    /// Particles across.
    pub columns: usize,
    /// Particles down.
    pub rows: usize,
}

impl ClothHandles {
    /// The particle at a grid position, which panics outside the sheet.
    #[inline]
    pub fn at(&self, column: usize, row: usize) -> ParticleHandle {
        self.particles[row * self.columns + column]
    }

    /// The four corner particles, in the order (0,0), (max,0), (0,max),
    /// (max,max).
    pub fn corners(&self) -> [ParticleHandle; 4] {
        [
            self.at(0, 0),
            self.at(self.columns - 1, 0),
            self.at(0, self.rows - 1),
            self.at(self.columns - 1, self.rows - 1),
        ]
    }

    /// The row of particles along the sheet's leading edge, for hanging it up.
    pub fn top_edge(&self) -> Vec<ParticleHandle> {
        (0..self.columns).map(|c| self.at(c, 0)).collect()
    }
}

/// Build a cloth into the given particle set and constraint lists.
pub fn build_cloth(
    desc: &ClothDesc,
    particles: &mut ParticleSet,
    distance: &mut Vec<DistanceConstraint>,
    bending: &mut Vec<BendingConstraint>,
    triangles: &mut Vec<SurfaceTriangle>,
) -> ClothHandles {
    let (cols, rows) = (desc.columns.max(2), desc.rows.max(2));
    let per_particle = desc.mass / (cols * rows) as f32;
    let mut handles = Vec::with_capacity(cols * rows);

    for row in 0..rows {
        for col in 0..cols {
            // Centred on the pose, laid out in the local XZ plane.
            let local = Vec3::new(
                (col as f32 - (cols - 1) as f32 * 0.5) * desc.spacing,
                0.0,
                (row as f32 - (rows - 1) as f32 * 0.5) * desc.spacing,
            );
            handles.push(particles.push(
                desc.pose.transform_point(local),
                per_particle,
                desc.thickness,
            ));
        }
    }

    let index = |col: usize, row: usize| handles[row * cols + col];
    let diagonal = desc.spacing * std::f32::consts::SQRT_2;

    for row in 0..rows {
        for col in 0..cols {
            // Structural, along the weave.
            if col + 1 < cols {
                distance.push(
                    DistanceConstraint::new(index(col, row), index(col + 1, row), desc.spacing)
                        .with_compliance(desc.stretch_compliance)
                        // Fabric resists stretching hard and buckles freely.
                        .with_compression_scale(0.25),
                );
            }
            if row + 1 < rows {
                distance.push(
                    DistanceConstraint::new(index(col, row), index(col, row + 1), desc.spacing)
                        .with_compliance(desc.stretch_compliance)
                        .with_compression_scale(0.25),
                );
            }
            // Shear, across the diagonals. A grid of four-sided cells with
            // only edge constraints is a trellis, not a sheet.
            if col + 1 < cols && row + 1 < rows {
                distance.push(
                    DistanceConstraint::new(index(col, row), index(col + 1, row + 1), diagonal)
                        .with_compliance(desc.shear_compliance),
                );
                distance.push(
                    DistanceConstraint::new(index(col + 1, row), index(col, row + 1), diagonal)
                        .with_compliance(desc.shear_compliance),
                );
            }
            // Bending, between particles two apart.
            if col + 2 < cols {
                bending.push(BendingConstraint::new(
                    index(col, row),
                    index(col + 2, row),
                    desc.spacing * 2.0,
                    desc.bend_compliance,
                ));
            }
            if row + 2 < rows {
                bending.push(BendingConstraint::new(
                    index(col, row),
                    index(col, row + 2),
                    desc.spacing * 2.0,
                    desc.bend_compliance,
                ));
            }
            // Two triangles per cell, for rendering and per-triangle aero.
            if col + 1 < cols && row + 1 < rows {
                triangles.push(SurfaceTriangle {
                    a: index(col, row).0,
                    b: index(col, row + 1).0,
                    c: index(col + 1, row + 1).0,
                });
                triangles.push(SurfaceTriangle {
                    a: index(col, row).0,
                    b: index(col + 1, row + 1).0,
                    c: index(col + 1, row).0,
                });
            }
        }
    }

    ClothHandles {
        particles: handles,
        columns: cols,
        rows,
    }
}

/// A deformable solid: a lattice of particles wired to its neighbours.
///
/// PLAN.md §5 lists a soft body next to cloth for a reason — it is the same
/// three ingredients, with the lattice in three dimensions instead of two. No
/// new solver, no new state, and it inherits contact and aerodynamics for
/// free.
///
/// This is a lattice rather than a tetrahedral mesh with a real strain energy.
/// A lattice needs no meshing, which PLAN.md §11 names as "the single largest
/// practical burden in FEM" — the thing MPM is chosen to avoid at the
/// continuum scale, avoided here for the same reason at the object scale. What
/// it gives up is a defensible stress tensor, which is Advanced's job, not
/// Basic's.
#[derive(Clone, Copy, Debug)]
pub struct SoftBodyDesc {
    /// Full extents of the block, metres.
    pub size: Vec3,
    /// Particles along each axis. Two is a hollow shell of eight corners;
    /// three or four is usually enough to look solid.
    pub resolution: (usize, usize, usize),
    /// Total mass of the block, kg.
    pub mass: f32,
    /// Where it is and how it is oriented.
    pub pose: Isometry,
    /// Inverse stiffness of the lattice, m/N. Zero is a rigid jelly, which is
    /// to say a rigid body with extra steps; 1e-5 wobbles convincingly.
    pub compliance: f32,
    /// Radius given to each particle for contact, metres. Too small and the
    /// block's surface sinks into the floor before anything pushes back.
    pub thickness: f32,
}

impl Default for SoftBodyDesc {
    fn default() -> Self {
        Self {
            size: Vec3::ONE,
            resolution: (4, 4, 4),
            mass: 5.0,
            pose: Isometry::IDENTITY,
            compliance: 1e-5,
            thickness: 0.03,
        }
    }
}

impl SoftBodyDesc {
    /// A block of the given size at the given lattice resolution on every
    /// axis.
    pub fn new(size: Vec3, resolution: usize) -> Self {
        Self {
            size,
            resolution: (resolution, resolution, resolution),
            ..Default::default()
        }
    }

    /// Builder: place its centre here.
    pub fn at(mut self, position: Vec3) -> Self {
        self.pose.translation = position;
        self
    }

    /// Builder: total mass, kg.
    pub fn with_mass(mut self, mass: f32) -> Self {
        self.mass = mass;
        self
    }

    /// Builder: how soft the lattice is. See [`SoftBodyDesc::compliance`].
    pub fn with_compliance(mut self, compliance: f32) -> Self {
        self.compliance = compliance;
        self
    }
}

/// Build a soft body. Returns its particle handles in x-major order.
pub fn build_soft_body(
    desc: &SoftBodyDesc,
    particles: &mut ParticleSet,
    distance: &mut Vec<DistanceConstraint>,
) -> Vec<ParticleHandle> {
    let (nx, ny, nz) = (
        desc.resolution.0.max(2),
        desc.resolution.1.max(2),
        desc.resolution.2.max(2),
    );
    let count = nx * ny * nz;
    let per_particle = desc.mass / count as f32;
    let step = Vec3::new(
        desc.size.x / (nx - 1) as f32,
        desc.size.y / (ny - 1) as f32,
        desc.size.z / (nz - 1) as f32,
    );

    let mut handles = Vec::with_capacity(count);
    for z in 0..nz {
        for y in 0..ny {
            for x in 0..nx {
                let local = Vec3::new(
                    (x as f32 - (nx - 1) as f32 * 0.5) * step.x,
                    (y as f32 - (ny - 1) as f32 * 0.5) * step.y,
                    (z as f32 - (nz - 1) as f32 * 0.5) * step.z,
                );
                handles.push(particles.push(
                    desc.pose.transform_point(local),
                    per_particle,
                    desc.thickness,
                ));
            }
        }
    }

    // Connect every particle to the 26 cells around it: axis neighbours hold
    // the edges, face diagonals stop each face shearing into a rhombus, and
    // body diagonals stop the cell collapsing along its longest direction. A
    // lattice with only axis neighbours folds flat, for the same reason a
    // square with four fixed edges is not rigid.
    let index = |x: usize, y: usize, z: usize| handles[z * ny * nx + y * nx + x];
    for z in 0..nz {
        for y in 0..ny {
            for x in 0..nx {
                for dz in 0..2usize {
                    for dy in 0..2usize {
                        for dx in 0..2usize {
                            if dx == 0 && dy == 0 && dz == 0 {
                                continue;
                            }
                            let (nx2, ny2, nz2) = (x + dx, y + dy, z + dz);
                            if nx2 >= nx || ny2 >= ny || nz2 >= nz {
                                continue;
                            }
                            let a = index(x, y, z);
                            let b = index(nx2, ny2, nz2);
                            let rest = Vec3::new(
                                dx as f32 * step.x,
                                dy as f32 * step.y,
                                dz as f32 * step.z,
                            )
                            .length();
                            distance.push(
                                DistanceConstraint::new(a, b, rest)
                                    .with_compliance(desc.compliance),
                            );
                        }
                    }
                }
            }
        }
    }
    handles
}
