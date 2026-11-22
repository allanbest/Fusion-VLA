//! Surfacing: turning a cloud of particles into a triangle mesh.
//!
//! PLAN.md §29 lists "marching-cubes surfacing" beside the fluid solver in
//! v0.4, and the pairing is not decorative. A fluid solved as particles has no
//! surface — it has ten thousand spheres — and a rendering of ten thousand
//! spheres reads as gravel, not water. The surface is a separate artefact
//! reconstructed from the particles, and reconstructing it is where most of
//! the *look* of a fluid comes from.
//!
//! Nothing here knows about the renderer, which is the point: this crate is
//! headless by construction (see the crate docs), so surfacing produces a mesh
//! and stops. The viewport consumes it; so could a file writer, a collision
//! query, or a volume measurement.
//!
//! ## The field
//!
//! Marching cubes needs a scalar field, and which field is chosen matters more
//! than the polygonizer does.
//!
//! The obvious one is the SPH density with an iso-level partway down — it is
//! already computed, and it works. It also produces the characteristic
//! "blobby" look, because a flat sheet of particles on a lattice gives a field
//! whose iso-surface bulges over every particle and dips between them. The
//! bumps are the particle lattice showing through, and they do not go away
//! with more particles; they get smaller and stay just as visible.
//!
//! [`zhu_bridson`] is used instead: the field is the distance to a
//! *kernel-weighted average particle position*, minus a radius. On the inside
//! the average sits where the sample is, so the field is flat; near a surface
//! the average is dragged inward by the particles that are there, and the
//! zero-crossing follows the shape of the cloud rather than the shape of
//! individual particles. Flat things come out flat. It costs one extra vector
//! accumulator per grid vertex and is the standard choice for exactly this
//! reason (Zhu & Bridson, *Animating Sand as a Fluid*, 2005).
//!
//! ```
//! use fusion_core::{FluidDesc, FluidParams, World};
//! use fusion_math::Vec3;
//! use fusion_sim::surface::{SurfaceMesh, SurfaceSettings, Surfacer};
//!
//! let mut world = World::new();
//! let fluid = world.add_fluid(
//!     &FluidDesc::block(Vec3::splat(0.4)).with_params(FluidParams::water_at(0.05)),
//! );
//!
//! let set = world.fluid(fluid);
//! let mut mesh = SurfaceMesh::default();
//! Surfacer::new().extract_fluid(set, &SurfaceSettings::for_fluid(set), &mut mesh);
//!
//! assert!(mesh.triangle_count() > 0);
//! assert!(mesh.is_closed());
//! ```
//!
//! ## The known hole in marching cubes
//!
//! The classic Lorensen-Cline table is not face-consistent. Some corner
//! configurations cut a shared face two different ways depending on which of
//! the two cells is asked, and where two such cells meet the mesh has a gap.
//! This is the well-known "ambiguous face" problem, and Fusion has it.
//!
//! Measured over a 32³ lattice, as a fraction of edges that are not shared by
//! exactly two triangles:
//!
//! | field | non-manifold edges |
//! |---|---|
//! | uniform random signs at every vertex | 3.97% |
//! | a sphere plus 2% noise | 0.00% |
//!
//! The first row is the adversarial case and does not occur: it requires the
//! field to change sign between *every* adjacent pair of samples, which means
//! the lattice is far too coarse for whatever is being surfaced. The second is
//! what a fluid actually produces, and it is exact. A field sampled finely
//! enough to be worth polygonizing does not generate ambiguous faces often
//! enough to see.
//!
//! The fix, if it is ever needed, is the asymptotic decider — evaluate the
//! bilinear interpolant at the saddle point of the ambiguous face and let it
//! choose. It is not implemented because nothing has yet produced a hole, and
//! a correctness mechanism with no failing case to point at is a guess.

use fusion_core::FluidSet;
use fusion_math::{Aabb, Vec3};

/// An indexed triangle mesh.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct SurfaceMesh {
    /// Vertex positions, in world space.
    pub positions: Vec<Vec3>,
    /// One per position, area-weighted from the faces that share it.
    pub normals: Vec<Vec3>,
    /// Three indices per triangle, counter-clockwise seen from outside.
    pub indices: Vec<u32>,
}

impl SurfaceMesh {
    /// Triangles in the mesh.
    pub fn triangle_count(&self) -> usize {
        self.indices.len() / 3
    }

    /// True where nothing was above the iso-level — an empty mesh rather
    /// than a failure.
    pub fn is_empty(&self) -> bool {
        self.indices.is_empty()
    }

    /// Total surface area, m².
    pub fn area(&self) -> f32 {
        self.indices
            .chunks_exact(3)
            .map(|t| {
                let (a, b, c) = (
                    self.positions[t[0] as usize],
                    self.positions[t[1] as usize],
                    self.positions[t[2] as usize],
                );
                0.5 * (b - a).cross(c - a).length()
            })
            .sum()
    }

    /// Volume enclosed, m³, by the divergence theorem.
    ///
    /// Meaningful only for a closed mesh, and *diagnostic* rather than
    /// decorative: a marching-cubes implementation with a wrong table entry
    /// produces a mesh that looks plausible and encloses the wrong volume, and
    /// a mesh with a hole in it encloses a number that depends on where the
    /// origin is. Both are caught here and nowhere else.
    pub fn volume(&self) -> f32 {
        let sum: f64 = self
            .indices
            .chunks_exact(3)
            .map(|t| {
                let (a, b, c) = (
                    self.positions[t[0] as usize],
                    self.positions[t[1] as usize],
                    self.positions[t[2] as usize],
                );
                a.dot(b.cross(c)) as f64 / 6.0
            })
            .sum();
        sum as f32
    }

    /// Bounding box of the mesh.
    pub fn bounds(&self) -> Aabb {
        self.positions
            .iter()
            .fold(Aabb::EMPTY, |b, p| b.extended(*p))
    }

    /// True when every edge is shared by exactly two triangles.
    ///
    /// The one structural property marching cubes is supposed to guarantee and
    /// the one a mistranscribed table breaks. Slower than the rest of the file
    /// put together, so it lives here for tests to call rather than being
    /// checked on every extraction.
    pub fn is_closed(&self) -> bool {
        let mut edges: Vec<(u32, u32)> = Vec::with_capacity(self.indices.len());
        for t in self.indices.chunks_exact(3) {
            for (a, b) in [(t[0], t[1]), (t[1], t[2]), (t[2], t[0])] {
                edges.push(if a < b { (a, b) } else { (b, a) });
            }
        }
        edges.sort_unstable();
        // Every edge exactly twice: once from each of the two triangles that
        // share it. Four copies is a different defect from zero copies and
        // both mean the mesh is not a surface, so count runs rather than
        // checking parity.
        let mut i = 0;
        while i < edges.len() {
            let mut j = i;
            while j < edges.len() && edges[j] == edges[i] {
                j += 1;
            }
            if j - i != 2 {
                return false;
            }
            i = j;
        }
        !edges.is_empty()
    }

    /// Wavefront OBJ, for looking at the result in something that is not
    /// Fusion.
    ///
    /// The debug viewport can draw this mesh, and being able to draw it is not
    /// the same as being able to check it. A surface that is subtly wrong —
    /// inverted normals, a seam, a hole a few cells across — is far easier to
    /// see in a tool built for looking at meshes, and OBJ is the format every
    /// one of them reads.
    pub fn to_obj(&self) -> String {
        let mut out = String::with_capacity(self.positions.len() * 48);
        out.push_str("# Fusion surface\n");
        for p in &self.positions {
            out.push_str(&format!("v {} {} {}\n", p.x, p.y, p.z));
        }
        for n in &self.normals {
            out.push_str(&format!("vn {} {} {}\n", n.x, n.y, n.z));
        }
        for t in self.indices.chunks_exact(3) {
            // OBJ indices are one-based, and reference position//normal.
            let (a, b, c) = (t[0] + 1, t[1] + 1, t[2] + 1);
            out.push_str(&format!("f {a}//{a} {b}//{b} {c}//{c}\n"));
        }
        out
    }

    fn recompute_normals(&mut self) {
        self.normals.clear();
        self.normals.resize(self.positions.len(), Vec3::ZERO);
        for t in self.indices.chunks_exact(3) {
            let (ia, ib, ic) = (t[0] as usize, t[1] as usize, t[2] as usize);
            let (a, b, c) = (self.positions[ia], self.positions[ib], self.positions[ic]);
            // Un-normalized, so the accumulation is area-weighted: a large
            // triangle should have more say in the normal of a vertex it
            // shares than a sliver does.
            let face = (b - a).cross(c - a);
            self.normals[ia] += face;
            self.normals[ib] += face;
            self.normals[ic] += face;
        }
        for n in &mut self.normals {
            *n = n.normalize_or_zero();
        }
    }
}

/// How finely, and over what, to reconstruct a surface.
#[derive(Clone, Copy, Debug)]
pub struct SurfaceSettings {
    /// Marching-cubes lattice spacing, metres. The dominant cost: halving it
    /// multiplies both time and memory by eight.
    pub cell_size: f32,
    /// Radius of the weighting kernel that averages particle positions.
    pub influence: f32,
    /// Radius subtracted from the distance-to-average. Sets how far outside
    /// the outermost particles the surface sits.
    pub radius: f32,
    /// Extra space around the particle bounds, so the surface has room to
    /// close over the outside of the cloud instead of being clipped flat
    /// against the edge of the grid.
    pub padding: f32,
    /// Ceiling on lattice vertices.
    ///
    /// A surfacing call is one of the few places in the engine where a
    /// plausible-looking argument allocates unboundedly: a 10 m splash at a
    /// 5 mm cell size is 8 x 10^9 vertices. `cell_size` is coarsened until the
    /// grid fits rather than the process dying, and [`Surfacer::last_cell_size`]
    /// reports what was actually used.
    pub max_vertices: usize,
}

impl SurfaceSettings {
    /// Settings matched to a fluid's own resolution.
    ///
    /// Two cells per particle spacing, which is the point where the lattice
    /// stops being visible in the result and before which more cells only cost
    /// more. The influence radius is the fluid's own smoothing radius, so the
    /// surface is reconstructed from the same neighbourhood the solver used.
    pub fn for_fluid(fluid: &FluidSet) -> Self {
        let spacing = fluid.params.spacing;
        Self {
            cell_size: 0.5 * spacing,
            influence: fluid.params.smoothing_radius,
            // Half a spacing puts the surface where the fluid's own volume
            // says it is: particle centres stop half a spacing short of the
            // free surface, exactly as they do half a spacing short of a wall.
            radius: 0.5 * spacing,
            padding: fluid.params.smoothing_radius,
            max_vertices: 24_000_000,
        }
    }

    /// Builder form: sample the field on a lattice this fine, in metres.
    /// Halving it multiplies both the cost and the vertex count by eight.
    pub fn with_cell_size(mut self, cell_size: f32) -> Self {
        self.cell_size = cell_size;
        self
    }
}

/// Reusable buffers for surface extraction.
///
/// Held across frames for the same reason [`fusion_collide::CollisionPipeline`]
/// is: a fluid surfaced every frame would otherwise allocate several megabytes
/// per frame and free them again.
#[derive(Clone, Debug, Default)]
pub struct Surfacer {
    field: Vec<f32>,
    weight: Vec<f32>,
    centroid: Vec<Vec3>,
    /// Vertex index per grid edge, or `u32::MAX`. Three edges per lattice
    /// vertex — the ones running in +x, +y and +z from it — which covers every
    /// edge in the lattice exactly once.
    edge_vertex: Vec<u32>,
    dims: [usize; 3],
    origin: Vec3,
    cell_size: f32,
}

impl Surfacer {
    /// An empty surfacer. Keep one and reuse it: the buffers it holds are
    /// the point.
    pub fn new() -> Self {
        Self::default()
    }

    /// Cell size actually used by the last extraction, which may be coarser
    /// than requested if the grid would have exceeded `max_vertices`.
    pub fn last_cell_size(&self) -> f32 {
        self.cell_size
    }

    /// Lattice dimensions used by the last extraction.
    pub fn last_dimensions(&self) -> [usize; 3] {
        self.dims
    }

    /// Reconstruct the surface of a fluid.
    pub fn extract_fluid(
        &mut self,
        fluid: &FluidSet,
        settings: &SurfaceSettings,
        out: &mut SurfaceMesh,
    ) {
        let points: Vec<Vec3> = (0..fluid.len())
            .map(|i| fluid.particles.translation(i))
            .collect();
        self.extract_points(&points, settings, out);
    }

    /// Reconstruct the zero level set of an arbitrary scalar field.
    ///
    /// Negative is inside, matching [`fusion_core::Shape::signed_distance_local`].
    /// Particles are one source of such a field and not the only one — an SDF
    /// shape, a thermal field, or a stress magnitude are all polygonizable by
    /// the same routine, which is why this is the general entry point and
    /// [`Self::extract_points`] is the convenience.
    pub fn extract_field(
        &mut self,
        bounds: &Aabb,
        cell_size: f32,
        max_vertices: usize,
        field: impl Fn(Vec3) -> f32,
        out: &mut SurfaceMesh,
    ) {
        out.positions.clear();
        out.normals.clear();
        out.indices.clear();
        let settings = SurfaceSettings {
            cell_size,
            influence: cell_size,
            radius: 0.0,
            padding: 0.0,
            max_vertices,
        };
        self.size_grid(bounds, &settings);
        let [nx, ny, nz] = self.dims;
        for z in 0..nz {
            for y in 0..ny {
                for x in 0..nx {
                    let index = self.index(x, y, z);
                    self.field[index] = field(self.vertex_position(x, y, z));
                }
            }
        }
        self.polygonize(out);
        out.recompute_normals();
    }

    /// Polygonize whatever is currently in the field buffer. Exposed for tests
    /// that need to drive every corner configuration directly rather than hope
    /// a smooth field happens to produce them.
    #[doc(hidden)]
    pub fn polygonize_field(
        &mut self,
        dims: [usize; 3],
        cell_size: f32,
        origin: Vec3,
        values: &[f32],
        out: &mut SurfaceMesh,
    ) {
        assert_eq!(values.len(), dims[0] * dims[1] * dims[2]);
        out.positions.clear();
        out.normals.clear();
        out.indices.clear();
        self.dims = dims;
        self.cell_size = cell_size;
        self.origin = origin;
        self.field.clear();
        self.field.extend_from_slice(values);
        self.edge_vertex.clear();
        self.edge_vertex.resize(3 * values.len(), u32::MAX);
        self.polygonize(out);
        out.recompute_normals();
    }

    /// Reconstruct the surface of an arbitrary point cloud.
    pub fn extract_points(
        &mut self,
        points: &[Vec3],
        settings: &SurfaceSettings,
        out: &mut SurfaceMesh,
    ) {
        out.positions.clear();
        out.normals.clear();
        out.indices.clear();
        if points.is_empty() {
            self.dims = [0; 3];
            return;
        }

        let bounds = points
            .iter()
            .fold(Aabb::EMPTY, |b, p| b.extended(*p))
            .expanded(settings.padding + settings.radius);
        self.size_grid(&bounds, settings);
        self.splat(points, settings);
        self.polygonize(out);
        out.recompute_normals();
    }

    /// Choose lattice dimensions, coarsening the cell size if the requested
    /// one would not fit in `max_vertices`.
    fn size_grid(&mut self, bounds: &Aabb, settings: &SurfaceSettings) {
        let extents = bounds.extents();
        let mut cell = settings.cell_size.max(1e-6);
        loop {
            let dims = [0usize, 1, 2].map(|k| (extents[k] / cell).ceil() as usize + 2);
            let total = dims[0].saturating_mul(dims[1]).saturating_mul(dims[2]);
            if total <= settings.max_vertices.max(8) {
                self.dims = dims;
                break;
            }
            // Grow geometrically rather than solving for the exact size: the
            // cube root of the overshoot is the right factor, and one extra
            // iteration is cheaper than getting the rounding wrong.
            cell *= 1.26;
        }
        self.cell_size = cell;
        self.origin = bounds.min - Vec3::splat(0.5 * cell);

        let total = self.dims[0] * self.dims[1] * self.dims[2];
        self.weight.clear();
        self.weight.resize(total, 0.0);
        self.centroid.clear();
        self.centroid.resize(total, Vec3::ZERO);
        self.field.clear();
        self.field.resize(total, 0.0);
        self.edge_vertex.clear();
        self.edge_vertex.resize(3 * total, u32::MAX);
    }

    #[inline]
    fn index(&self, x: usize, y: usize, z: usize) -> usize {
        (z * self.dims[1] + y) * self.dims[0] + x
    }

    #[inline]
    fn vertex_position(&self, x: usize, y: usize, z: usize) -> Vec3 {
        self.origin + Vec3::new(x as f32, y as f32, z as f32) * self.cell_size
    }

    /// Accumulate the weighted particle average at every lattice vertex, then
    /// turn it into the Zhu-Bridson field.
    ///
    /// Scatter, not gather. Each particle touches the vertices within its
    /// influence radius, which for the default settings is about five cells
    /// across — a few hundred writes per particle. Gathering instead would
    /// mean a neighbour query per *vertex*, and there are more vertices than
    /// particles by construction.
    fn splat(&mut self, points: &[Vec3], settings: &SurfaceSettings) {
        let influence = settings.influence.max(1e-6);
        let inv_cell = 1.0 / self.cell_size;
        let reach = (influence * inv_cell).ceil() as i64;
        let inv_influence_sq = 1.0 / (influence * influence);

        for &p in points {
            let local = (p - self.origin) * inv_cell;
            let base = [
                local.x.floor() as i64,
                local.y.floor() as i64,
                local.z.floor() as i64,
            ];
            for dz in -reach..=reach {
                let z = base[2] + dz;
                if z < 0 || z >= self.dims[2] as i64 {
                    continue;
                }
                for dy in -reach..=reach {
                    let y = base[1] + dy;
                    if y < 0 || y >= self.dims[1] as i64 {
                        continue;
                    }
                    for dx in -reach..=reach {
                        let x = base[0] + dx;
                        if x < 0 || x >= self.dims[0] as i64 {
                            continue;
                        }
                        let (x, y, z) = (x as usize, y as usize, z as usize);
                        let offset = self.vertex_position(x, y, z) - p;
                        let s_sq = offset.length_squared() * inv_influence_sq;
                        if s_sq >= 1.0 {
                            continue;
                        }
                        // k(s) = (1 - s^2)^3, the weighting Zhu & Bridson use:
                        // smooth, compactly supported, and zero-derivative at
                        // the support boundary so the field has no crease
                        // where a particle drops out.
                        let t = 1.0 - s_sq;
                        let w = t * t * t;
                        let index = self.index(x, y, z);
                        self.weight[index] += w;
                        self.centroid[index] += p * w;
                    }
                }
            }
        }

        // Far outside the cloud the weight is zero and the average is
        // meaningless. Those vertices are given a positive field value one
        // cell deep rather than something large: marching cubes interpolates
        // linearly along an edge between an inside and an outside sample, and
        // a huge outside value would drag every boundary vertex almost onto
        // the inside sample, leaving a mesh visibly inflated by half a cell.
        let outside = self.cell_size;
        for i in 0..self.field.len() {
            self.field[i] = if self.weight[i] > 1e-12 {
                let average = self.centroid[i] / self.weight[i];
                let vertex = {
                    let d = self.dims;
                    let x = i % d[0];
                    let y = (i / d[0]) % d[1];
                    let z = i / (d[0] * d[1]);
                    self.vertex_position(x, y, z)
                };
                (vertex - average).length() - settings.radius
            } else {
                outside
            };
        }
    }

    /// Walk every cell of the lattice and emit the triangles the field cuts.
    fn polygonize(&mut self, out: &mut SurfaceMesh) {
        self.edge_vertex.iter_mut().for_each(|e| *e = u32::MAX);
        let [nx, ny, nz] = self.dims;
        if nx < 2 || ny < 2 || nz < 2 {
            return;
        }

        for z in 0..nz - 1 {
            for y in 0..ny - 1 {
                for x in 0..nx - 1 {
                    let mut corner_index = [0usize; 8];
                    let mut value = [0f32; 8];
                    let mut cube = 0u8;
                    for (c, offset) in CORNER_OFFSETS.iter().enumerate() {
                        let index = self.index(
                            x + offset[0] as usize,
                            y + offset[1] as usize,
                            z + offset[2] as usize,
                        );
                        corner_index[c] = index;
                        value[c] = self.field[index];
                        // Bit set means the corner is *inside*, which for this
                        // field means negative. Matching the sign convention of
                        // `Shape::signed_distance_local` rather than inventing
                        // a second one is what makes the triangle winding come
                        // out with normals pointing away from the fluid.
                        if value[c] < 0.0 {
                            cube |= 1 << c;
                        }
                    }
                    let triangles = &TRI_TABLE[cube as usize];
                    if triangles[0] < 0 {
                        continue;
                    }
                    let mut slot = 0;
                    while triangles[slot] >= 0 {
                        // Reversed within each triple: the table is written for
                        // the convention where a set bit means *outside*, so
                        // reading it with the opposite convention mirrors every
                        // triangle. Swapping two of the three indices puts the
                        // winding back rather than maintaining a second table.
                        let tri = [
                            triangles[slot] as usize,
                            triangles[slot + 2] as usize,
                            triangles[slot + 1] as usize,
                        ];
                        for edge in tri {
                            let vertex =
                                self.edge_vertex_for(edge, x, y, z, &corner_index, &value, out);
                            out.indices.push(vertex);
                        }
                        slot += 3;
                    }
                }
            }
        }
    }

    /// Index of the vertex on one edge of one cell, creating it if this is the
    /// first cell to ask.
    ///
    /// Sharing is what makes the mesh closed rather than a soup of unconnected
    /// triangles, and it also makes vertex normals meaningful — a normal
    /// averaged over the faces around a vertex needs those faces to agree that
    /// it is the same vertex.
    #[allow(clippy::too_many_arguments)]
    fn edge_vertex_for(
        &mut self,
        edge: usize,
        x: usize,
        y: usize,
        z: usize,
        corner_index: &[usize; 8],
        value: &[f32; 8],
        out: &mut SurfaceMesh,
    ) -> u32 {
        let (low_corner, axis) = EDGE_KEY[edge];
        let key = 3 * corner_index[low_corner as usize] + axis as usize;
        if self.edge_vertex[key] != u32::MAX {
            return self.edge_vertex[key];
        }

        // Always interpolate from the lower corner along the axis to the upper
        // one, never in the order the cube happens to list them. Two cells
        // sharing an edge must place the vertex at bit-identical coordinates
        // or the mesh cracks, and `a + t (b - a)` is not `b + (1-t) (a - b)`.
        let (a, b) = EDGE_ENDPOINTS[edge];
        let (a, b) = if CORNER_OFFSETS[a as usize][axis as usize]
            < CORNER_OFFSETS[b as usize][axis as usize]
        {
            (a as usize, b as usize)
        } else {
            (b as usize, a as usize)
        };

        let (va, vb) = (value[a], value[b]);
        let denominator = vb - va;
        let t = if denominator.abs() > 1e-12 {
            (-va / denominator).clamp(0.0, 1.0)
        } else {
            0.5
        };

        let offset_a = CORNER_OFFSETS[a];
        let start = self.vertex_position(
            x + offset_a[0] as usize,
            y + offset_a[1] as usize,
            z + offset_a[2] as usize,
        );
        let mut position = start;
        position[axis as usize] += t * self.cell_size;

        let index = out.positions.len() as u32;
        out.positions.push(position);
        self.edge_vertex[key] = index;
        index
    }
}

/// The Zhu-Bridson field at one point, for callers that want to sample it
/// directly rather than take a mesh.
///
/// `phi(x) = |x - xbar| - r`, where `xbar` is the kernel-weighted average of
/// the nearby particle positions. Negative inside.
pub fn zhu_bridson(points: &[Vec3], at: Vec3, influence: f32, radius: f32) -> f32 {
    let inv_sq = 1.0 / (influence * influence);
    let mut weight = 0.0;
    let mut centroid = Vec3::ZERO;
    for &p in points {
        let s_sq = (at - p).length_squared() * inv_sq;
        if s_sq >= 1.0 {
            continue;
        }
        let t = 1.0 - s_sq;
        let w = t * t * t;
        weight += w;
        centroid += p * w;
    }
    if weight <= 1e-12 {
        return influence;
    }
    (at - centroid / weight).length() - radius
}

/// Cell corner offsets, in the numbering the triangle table is written for.
///
/// ```text
///    4--------5
///   /|       /|
///  7--------6 |
///  | 0------|-1
///  |/       |/
///  3--------2
/// ```
const CORNER_OFFSETS: [[u8; 3]; 8] = [
    [0, 0, 0],
    [1, 0, 0],
    [1, 1, 0],
    [0, 1, 0],
    [0, 0, 1],
    [1, 0, 1],
    [1, 1, 1],
    [0, 1, 1],
];

/// The two corners each of the twelve edges connects.
const EDGE_ENDPOINTS: [(u8, u8); 12] = [
    (0, 1),
    (1, 2),
    (2, 3),
    (3, 0),
    (4, 5),
    (5, 6),
    (6, 7),
    (7, 4),
    (0, 4),
    (1, 5),
    (2, 6),
    (3, 7),
];

/// `(corner the edge starts from, axis it runs along)`, used to give the edge
/// a name that two adjacent cells agree on.
const EDGE_KEY: [(u8, u8); 12] = [
    (0, 0),
    (1, 1),
    (3, 0),
    (0, 1),
    (4, 0),
    (5, 1),
    (7, 0),
    (4, 1),
    (0, 2),
    (1, 2),
    (2, 2),
    (3, 2),
];

/// The marching-cubes triangle table (Lorensen & Cline; this transcription
/// follows Bourke's).
///
/// Row `i` lists the edges to join, three at a time, for the corner
/// configuration `i`, terminated by `-1`. The convention it is written for is
/// "bit set means the corner is outside"; [`Surfacer::polygonize`] sets the bit
/// for *inside* and reverses each triple to compensate, which is one line
/// instead of a second 256-row table.
///
/// This table is data, and data transcribed by hand is data with a typo in it
/// until something proves otherwise. What proves otherwise is
/// [`SurfaceMesh::is_closed`] over a field that exercises every case, plus the
/// analytic volume and area of a sphere — a wrong entry leaves a hole, and a
/// hole fails all three.
#[rustfmt::skip]
const TRI_TABLE: [[i8; 16]; 256] = [
    [-1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1],
    [0, 8, 3, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1],
    [0, 1, 9, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1],
    [1, 8, 3, 9, 8, 1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1],
    [1, 2, 10, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1],
    [0, 8, 3, 1, 2, 10, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1],
    [9, 2, 10, 0, 2, 9, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1],
    [2, 8, 3, 2, 10, 8, 10, 9, 8, -1, -1, -1, -1, -1, -1, -1],
    [3, 11, 2, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1],
    [0, 11, 2, 8, 11, 0, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1],
    [1, 9, 0, 2, 3, 11, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1],
    [1, 11, 2, 1, 9, 11, 9, 8, 11, -1, -1, -1, -1, -1, -1, -1],
    [3, 10, 1, 11, 10, 3, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1],
    [0, 10, 1, 0, 8, 10, 8, 11, 10, -1, -1, -1, -1, -1, -1, -1],
    [3, 9, 0, 3, 11, 9, 11, 10, 9, -1, -1, -1, -1, -1, -1, -1],
    [9, 8, 10, 10, 8, 11, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1],
    [4, 7, 8, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1],
    [4, 3, 0, 7, 3, 4, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1],
    [0, 1, 9, 8, 4, 7, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1],
    [4, 1, 9, 4, 7, 1, 7, 3, 1, -1, -1, -1, -1, -1, -1, -1],
    [1, 2, 10, 8, 4, 7, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1],
    [3, 4, 7, 3, 0, 4, 1, 2, 10, -1, -1, -1, -1, -1, -1, -1],
    [9, 2, 10, 9, 0, 2, 8, 4, 7, -1, -1, -1, -1, -1, -1, -1],
    [2, 10, 9, 2, 9, 7, 2, 7, 3, 7, 9, 4, -1, -1, -1, -1],
    [8, 4, 7, 3, 11, 2, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1],
    [11, 4, 7, 11, 2, 4, 2, 0, 4, -1, -1, -1, -1, -1, -1, -1],
    [9, 0, 1, 8, 4, 7, 2, 3, 11, -1, -1, -1, -1, -1, -1, -1],
    [4, 7, 11, 9, 4, 11, 9, 11, 2, 9, 2, 1, -1, -1, -1, -1],
    [3, 10, 1, 3, 11, 10, 7, 8, 4, -1, -1, -1, -1, -1, -1, -1],
    [1, 11, 10, 1, 4, 11, 1, 0, 4, 7, 11, 4, -1, -1, -1, -1],
    [4, 7, 8, 9, 0, 11, 9, 11, 10, 11, 0, 3, -1, -1, -1, -1],
    [4, 7, 11, 4, 11, 9, 9, 11, 10, -1, -1, -1, -1, -1, -1, -1],
    [9, 5, 4, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1],
    [9, 5, 4, 0, 8, 3, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1],
    [0, 5, 4, 1, 5, 0, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1],
    [8, 5, 4, 8, 3, 5, 3, 1, 5, -1, -1, -1, -1, -1, -1, -1],
    [1, 2, 10, 9, 5, 4, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1],
    [3, 0, 8, 1, 2, 10, 4, 9, 5, -1, -1, -1, -1, -1, -1, -1],
    [5, 2, 10, 5, 4, 2, 4, 0, 2, -1, -1, -1, -1, -1, -1, -1],
    [2, 10, 5, 3, 2, 5, 3, 5, 4, 3, 4, 8, -1, -1, -1, -1],
    [9, 5, 4, 2, 3, 11, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1],
    [0, 11, 2, 0, 8, 11, 4, 9, 5, -1, -1, -1, -1, -1, -1, -1],
    [0, 5, 4, 0, 1, 5, 2, 3, 11, -1, -1, -1, -1, -1, -1, -1],
    [2, 1, 5, 2, 5, 8, 2, 8, 11, 4, 8, 5, -1, -1, -1, -1],
    [10, 3, 11, 10, 1, 3, 9, 5, 4, -1, -1, -1, -1, -1, -1, -1],
    [4, 9, 5, 0, 8, 1, 8, 10, 1, 8, 11, 10, -1, -1, -1, -1],
    [5, 4, 0, 5, 0, 11, 5, 11, 10, 11, 0, 3, -1, -1, -1, -1],
    [5, 4, 8, 5, 8, 10, 10, 8, 11, -1, -1, -1, -1, -1, -1, -1],
    [9, 7, 8, 5, 7, 9, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1],
    [9, 3, 0, 9, 5, 3, 5, 7, 3, -1, -1, -1, -1, -1, -1, -1],
    [0, 7, 8, 0, 1, 7, 1, 5, 7, -1, -1, -1, -1, -1, -1, -1],
    [1, 5, 3, 3, 5, 7, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1],
    [9, 7, 8, 9, 5, 7, 10, 1, 2, -1, -1, -1, -1, -1, -1, -1],
    [10, 1, 2, 9, 5, 0, 5, 3, 0, 5, 7, 3, -1, -1, -1, -1],
    [8, 0, 2, 8, 2, 5, 8, 5, 7, 10, 5, 2, -1, -1, -1, -1],
    [2, 10, 5, 2, 5, 3, 3, 5, 7, -1, -1, -1, -1, -1, -1, -1],
    [7, 9, 5, 7, 8, 9, 3, 11, 2, -1, -1, -1, -1, -1, -1, -1],
    [9, 5, 7, 9, 7, 2, 9, 2, 0, 2, 7, 11, -1, -1, -1, -1],
    [2, 3, 11, 0, 1, 8, 1, 7, 8, 1, 5, 7, -1, -1, -1, -1],
    [11, 2, 1, 11, 1, 7, 7, 1, 5, -1, -1, -1, -1, -1, -1, -1],
    [9, 5, 8, 8, 5, 7, 10, 1, 3, 10, 3, 11, -1, -1, -1, -1],
    [5, 7, 0, 5, 0, 9, 7, 11, 0, 1, 0, 10, 11, 10, 0, -1],
    [11, 10, 0, 11, 0, 3, 10, 5, 0, 8, 0, 7, 5, 7, 0, -1],
    [11, 10, 5, 7, 11, 5, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1],
    [10, 6, 5, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1],
    [0, 8, 3, 5, 10, 6, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1],
    [9, 0, 1, 5, 10, 6, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1],
    [1, 8, 3, 1, 9, 8, 5, 10, 6, -1, -1, -1, -1, -1, -1, -1],
    [1, 6, 5, 2, 6, 1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1],
    [1, 6, 5, 1, 2, 6, 3, 0, 8, -1, -1, -1, -1, -1, -1, -1],
    [9, 6, 5, 9, 0, 6, 0, 2, 6, -1, -1, -1, -1, -1, -1, -1],
    [5, 9, 8, 5, 8, 2, 5, 2, 6, 3, 2, 8, -1, -1, -1, -1],
    [2, 3, 11, 10, 6, 5, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1],
    [11, 0, 8, 11, 2, 0, 10, 6, 5, -1, -1, -1, -1, -1, -1, -1],
    [0, 1, 9, 2, 3, 11, 5, 10, 6, -1, -1, -1, -1, -1, -1, -1],
    [5, 10, 6, 1, 9, 2, 9, 11, 2, 9, 8, 11, -1, -1, -1, -1],
    [6, 3, 11, 6, 5, 3, 5, 1, 3, -1, -1, -1, -1, -1, -1, -1],
    [0, 8, 11, 0, 11, 5, 0, 5, 1, 5, 11, 6, -1, -1, -1, -1],
    [3, 11, 6, 0, 3, 6, 0, 6, 5, 0, 5, 9, -1, -1, -1, -1],
    [6, 5, 9, 6, 9, 11, 11, 9, 8, -1, -1, -1, -1, -1, -1, -1],
    [5, 10, 6, 4, 7, 8, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1],
    [4, 3, 0, 4, 7, 3, 6, 5, 10, -1, -1, -1, -1, -1, -1, -1],
    [1, 9, 0, 5, 10, 6, 8, 4, 7, -1, -1, -1, -1, -1, -1, -1],
    [10, 6, 5, 1, 9, 7, 1, 7, 3, 7, 9, 4, -1, -1, -1, -1],
    [6, 1, 2, 6, 5, 1, 4, 7, 8, -1, -1, -1, -1, -1, -1, -1],
    [1, 2, 5, 5, 2, 6, 3, 0, 4, 3, 4, 7, -1, -1, -1, -1],
    [8, 4, 7, 9, 0, 5, 0, 6, 5, 0, 2, 6, -1, -1, -1, -1],
    [7, 3, 9, 7, 9, 4, 3, 2, 9, 5, 9, 6, 2, 6, 9, -1],
    [3, 11, 2, 7, 8, 4, 10, 6, 5, -1, -1, -1, -1, -1, -1, -1],
    [5, 10, 6, 4, 7, 2, 4, 2, 0, 2, 7, 11, -1, -1, -1, -1],
    [0, 1, 9, 4, 7, 8, 2, 3, 11, 5, 10, 6, -1, -1, -1, -1],
    [9, 2, 1, 9, 11, 2, 9, 4, 11, 7, 11, 4, 5, 10, 6, -1],
    [8, 4, 7, 3, 11, 5, 3, 5, 1, 5, 11, 6, -1, -1, -1, -1],
    [5, 1, 11, 5, 11, 6, 1, 0, 11, 7, 11, 4, 0, 4, 11, -1],
    [0, 5, 9, 0, 6, 5, 0, 3, 6, 11, 6, 3, 8, 4, 7, -1],
    [6, 5, 9, 6, 9, 11, 4, 7, 9, 7, 11, 9, -1, -1, -1, -1],
    [10, 4, 9, 6, 4, 10, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1],
    [4, 10, 6, 4, 9, 10, 0, 8, 3, -1, -1, -1, -1, -1, -1, -1],
    [10, 0, 1, 10, 6, 0, 6, 4, 0, -1, -1, -1, -1, -1, -1, -1],
    [8, 3, 1, 8, 1, 6, 8, 6, 4, 6, 1, 10, -1, -1, -1, -1],
    [1, 4, 9, 1, 2, 4, 2, 6, 4, -1, -1, -1, -1, -1, -1, -1],
    [3, 0, 8, 1, 2, 9, 2, 4, 9, 2, 6, 4, -1, -1, -1, -1],
    [0, 2, 4, 4, 2, 6, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1],
    [8, 3, 2, 8, 2, 4, 4, 2, 6, -1, -1, -1, -1, -1, -1, -1],
    [10, 4, 9, 10, 6, 4, 11, 2, 3, -1, -1, -1, -1, -1, -1, -1],
    [0, 8, 2, 2, 8, 11, 4, 9, 10, 4, 10, 6, -1, -1, -1, -1],
    [3, 11, 2, 0, 1, 6, 0, 6, 4, 6, 1, 10, -1, -1, -1, -1],
    [6, 4, 1, 6, 1, 10, 4, 8, 1, 2, 1, 11, 8, 11, 1, -1],
    [9, 6, 4, 9, 3, 6, 9, 1, 3, 11, 6, 3, -1, -1, -1, -1],
    [8, 11, 1, 8, 1, 0, 11, 6, 1, 9, 1, 4, 6, 4, 1, -1],
    [3, 11, 6, 3, 6, 0, 0, 6, 4, -1, -1, -1, -1, -1, -1, -1],
    [6, 4, 8, 11, 6, 8, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1],
    [7, 10, 6, 7, 8, 10, 8, 9, 10, -1, -1, -1, -1, -1, -1, -1],
    [0, 7, 3, 0, 10, 7, 0, 9, 10, 6, 7, 10, -1, -1, -1, -1],
    [10, 6, 7, 1, 10, 7, 1, 7, 8, 1, 8, 0, -1, -1, -1, -1],
    [10, 6, 7, 10, 7, 1, 1, 7, 3, -1, -1, -1, -1, -1, -1, -1],
    [1, 2, 6, 1, 6, 8, 1, 8, 9, 8, 6, 7, -1, -1, -1, -1],
    [2, 6, 9, 2, 9, 1, 6, 7, 9, 0, 9, 3, 7, 3, 9, -1],
    [7, 8, 0, 7, 0, 6, 6, 0, 2, -1, -1, -1, -1, -1, -1, -1],
    [7, 3, 2, 6, 7, 2, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1],
    [2, 3, 11, 10, 6, 8, 10, 8, 9, 8, 6, 7, -1, -1, -1, -1],
    [2, 0, 7, 2, 7, 11, 0, 9, 7, 6, 7, 10, 9, 10, 7, -1],
    [1, 8, 0, 1, 7, 8, 1, 10, 7, 6, 7, 10, 2, 3, 11, -1],
    [11, 2, 1, 11, 1, 7, 10, 6, 1, 6, 7, 1, -1, -1, -1, -1],
    [8, 9, 6, 8, 6, 7, 9, 1, 6, 11, 6, 3, 1, 3, 6, -1],
    [0, 9, 1, 11, 6, 7, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1],
    [7, 8, 0, 7, 0, 6, 3, 11, 0, 11, 6, 0, -1, -1, -1, -1],
    [7, 11, 6, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1],
    [7, 6, 11, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1],
    [3, 0, 8, 11, 7, 6, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1],
    [0, 1, 9, 11, 7, 6, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1],
    [8, 1, 9, 8, 3, 1, 11, 7, 6, -1, -1, -1, -1, -1, -1, -1],
    [10, 1, 2, 6, 11, 7, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1],
    [1, 2, 10, 3, 0, 8, 6, 11, 7, -1, -1, -1, -1, -1, -1, -1],
    [2, 9, 0, 2, 10, 9, 6, 11, 7, -1, -1, -1, -1, -1, -1, -1],
    [6, 11, 7, 2, 10, 3, 10, 8, 3, 10, 9, 8, -1, -1, -1, -1],
    [7, 2, 3, 6, 2, 7, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1],
    [7, 0, 8, 7, 6, 0, 6, 2, 0, -1, -1, -1, -1, -1, -1, -1],
    [2, 7, 6, 2, 3, 7, 0, 1, 9, -1, -1, -1, -1, -1, -1, -1],
    [1, 6, 2, 1, 8, 6, 1, 9, 8, 8, 7, 6, -1, -1, -1, -1],
    [10, 7, 6, 10, 1, 7, 1, 3, 7, -1, -1, -1, -1, -1, -1, -1],
    [10, 7, 6, 1, 7, 10, 1, 8, 7, 1, 0, 8, -1, -1, -1, -1],
    [0, 3, 7, 0, 7, 10, 0, 10, 9, 6, 10, 7, -1, -1, -1, -1],
    [7, 6, 10, 7, 10, 8, 8, 10, 9, -1, -1, -1, -1, -1, -1, -1],
    [6, 8, 4, 11, 8, 6, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1],
    [3, 6, 11, 3, 0, 6, 0, 4, 6, -1, -1, -1, -1, -1, -1, -1],
    [8, 6, 11, 8, 4, 6, 9, 0, 1, -1, -1, -1, -1, -1, -1, -1],
    [9, 4, 6, 9, 6, 3, 9, 3, 1, 11, 3, 6, -1, -1, -1, -1],
    [6, 8, 4, 6, 11, 8, 2, 10, 1, -1, -1, -1, -1, -1, -1, -1],
    [1, 2, 10, 3, 0, 11, 0, 6, 11, 0, 4, 6, -1, -1, -1, -1],
    [4, 11, 8, 4, 6, 11, 0, 2, 9, 2, 10, 9, -1, -1, -1, -1],
    [10, 9, 3, 10, 3, 2, 9, 4, 3, 11, 3, 6, 4, 6, 3, -1],
    [8, 2, 3, 8, 4, 2, 4, 6, 2, -1, -1, -1, -1, -1, -1, -1],
    [0, 4, 2, 4, 6, 2, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1],
    [1, 9, 0, 2, 3, 4, 2, 4, 6, 4, 3, 8, -1, -1, -1, -1],
    [1, 9, 4, 1, 4, 2, 2, 4, 6, -1, -1, -1, -1, -1, -1, -1],
    [8, 1, 3, 8, 6, 1, 8, 4, 6, 6, 10, 1, -1, -1, -1, -1],
    [10, 1, 0, 10, 0, 6, 6, 0, 4, -1, -1, -1, -1, -1, -1, -1],
    [4, 6, 3, 4, 3, 8, 6, 10, 3, 0, 3, 9, 10, 9, 3, -1],
    [10, 9, 4, 6, 10, 4, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1],
    [4, 9, 5, 7, 6, 11, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1],
    [0, 8, 3, 4, 9, 5, 11, 7, 6, -1, -1, -1, -1, -1, -1, -1],
    [5, 0, 1, 5, 4, 0, 7, 6, 11, -1, -1, -1, -1, -1, -1, -1],
    [11, 7, 6, 8, 3, 4, 3, 5, 4, 3, 1, 5, -1, -1, -1, -1],
    [9, 5, 4, 10, 1, 2, 7, 6, 11, -1, -1, -1, -1, -1, -1, -1],
    [6, 11, 7, 1, 2, 10, 0, 8, 3, 4, 9, 5, -1, -1, -1, -1],
    [7, 6, 11, 5, 4, 10, 4, 2, 10, 4, 0, 2, -1, -1, -1, -1],
    [3, 4, 8, 3, 5, 4, 3, 2, 5, 10, 5, 2, 11, 7, 6, -1],
    [7, 2, 3, 7, 6, 2, 5, 4, 9, -1, -1, -1, -1, -1, -1, -1],
    [9, 5, 4, 0, 8, 6, 0, 6, 2, 6, 8, 7, -1, -1, -1, -1],
    [3, 6, 2, 3, 7, 6, 1, 5, 0, 5, 4, 0, -1, -1, -1, -1],
    [6, 2, 8, 6, 8, 7, 2, 1, 8, 4, 8, 5, 1, 5, 8, -1],
    [9, 5, 4, 10, 1, 6, 1, 7, 6, 1, 3, 7, -1, -1, -1, -1],
    [1, 6, 10, 1, 7, 6, 1, 0, 7, 8, 7, 0, 9, 5, 4, -1],
    [4, 0, 10, 4, 10, 5, 0, 3, 10, 6, 10, 7, 3, 7, 10, -1],
    [7, 6, 10, 7, 10, 8, 5, 4, 10, 4, 8, 10, -1, -1, -1, -1],
    [6, 9, 5, 6, 11, 9, 11, 8, 9, -1, -1, -1, -1, -1, -1, -1],
    [3, 6, 11, 0, 6, 3, 0, 5, 6, 0, 9, 5, -1, -1, -1, -1],
    [0, 11, 8, 0, 5, 11, 0, 1, 5, 5, 6, 11, -1, -1, -1, -1],
    [6, 11, 3, 6, 3, 5, 5, 3, 1, -1, -1, -1, -1, -1, -1, -1],
    [1, 2, 10, 9, 5, 11, 9, 11, 8, 11, 5, 6, -1, -1, -1, -1],
    [0, 11, 3, 0, 6, 11, 0, 9, 6, 5, 6, 9, 1, 2, 10, -1],
    [11, 8, 5, 11, 5, 6, 8, 0, 5, 10, 5, 2, 0, 2, 5, -1],
    [6, 11, 3, 6, 3, 5, 2, 10, 3, 10, 5, 3, -1, -1, -1, -1],
    [5, 8, 9, 5, 2, 8, 5, 6, 2, 3, 8, 2, -1, -1, -1, -1],
    [9, 5, 6, 9, 6, 0, 0, 6, 2, -1, -1, -1, -1, -1, -1, -1],
    [1, 5, 8, 1, 8, 0, 5, 6, 8, 3, 8, 2, 6, 2, 8, -1],
    [1, 5, 6, 2, 1, 6, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1],
    [1, 3, 6, 1, 6, 10, 3, 8, 6, 5, 6, 9, 8, 9, 6, -1],
    [10, 1, 0, 10, 0, 6, 9, 5, 0, 5, 6, 0, -1, -1, -1, -1],
    [0, 3, 8, 5, 6, 10, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1],
    [10, 5, 6, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1],
    [11, 5, 10, 7, 5, 11, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1],
    [11, 5, 10, 11, 7, 5, 8, 3, 0, -1, -1, -1, -1, -1, -1, -1],
    [5, 11, 7, 5, 10, 11, 1, 9, 0, -1, -1, -1, -1, -1, -1, -1],
    [10, 7, 5, 10, 11, 7, 9, 8, 1, 8, 3, 1, -1, -1, -1, -1],
    [11, 1, 2, 11, 7, 1, 7, 5, 1, -1, -1, -1, -1, -1, -1, -1],
    [0, 8, 3, 1, 2, 7, 1, 7, 5, 7, 2, 11, -1, -1, -1, -1],
    [9, 7, 5, 9, 2, 7, 9, 0, 2, 2, 11, 7, -1, -1, -1, -1],
    [7, 5, 2, 7, 2, 11, 5, 9, 2, 3, 2, 8, 9, 8, 2, -1],
    [2, 5, 10, 2, 3, 5, 3, 7, 5, -1, -1, -1, -1, -1, -1, -1],
    [8, 2, 0, 8, 5, 2, 8, 7, 5, 10, 2, 5, -1, -1, -1, -1],
    [9, 0, 1, 5, 10, 3, 5, 3, 7, 3, 10, 2, -1, -1, -1, -1],
    [9, 8, 2, 9, 2, 1, 8, 7, 2, 10, 2, 5, 7, 5, 2, -1],
    [1, 3, 5, 3, 7, 5, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1],
    [0, 8, 7, 0, 7, 1, 1, 7, 5, -1, -1, -1, -1, -1, -1, -1],
    [9, 0, 3, 9, 3, 5, 5, 3, 7, -1, -1, -1, -1, -1, -1, -1],
    [9, 8, 7, 5, 9, 7, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1],
    [5, 8, 4, 5, 10, 8, 10, 11, 8, -1, -1, -1, -1, -1, -1, -1],
    [5, 0, 4, 5, 11, 0, 5, 10, 11, 11, 3, 0, -1, -1, -1, -1],
    [0, 1, 9, 8, 4, 10, 8, 10, 11, 10, 4, 5, -1, -1, -1, -1],
    [10, 11, 4, 10, 4, 5, 11, 3, 4, 9, 4, 1, 3, 1, 4, -1],
    [2, 5, 1, 2, 8, 5, 2, 11, 8, 4, 5, 8, -1, -1, -1, -1],
    [0, 4, 11, 0, 11, 3, 4, 5, 11, 2, 11, 1, 5, 1, 11, -1],
    [0, 2, 5, 0, 5, 9, 2, 11, 5, 4, 5, 8, 11, 8, 5, -1],
    [9, 4, 5, 2, 11, 3, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1],
    [2, 5, 10, 3, 5, 2, 3, 4, 5, 3, 8, 4, -1, -1, -1, -1],
    [5, 10, 2, 5, 2, 4, 4, 2, 0, -1, -1, -1, -1, -1, -1, -1],
    [3, 10, 2, 3, 5, 10, 3, 8, 5, 4, 5, 8, 0, 1, 9, -1],
    [5, 10, 2, 5, 2, 4, 1, 9, 2, 9, 4, 2, -1, -1, -1, -1],
    [8, 4, 5, 8, 5, 3, 3, 5, 1, -1, -1, -1, -1, -1, -1, -1],
    [0, 4, 5, 1, 0, 5, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1],
    [8, 4, 5, 8, 5, 3, 9, 0, 5, 0, 3, 5, -1, -1, -1, -1],
    [9, 4, 5, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1],
    [4, 11, 7, 4, 9, 11, 9, 10, 11, -1, -1, -1, -1, -1, -1, -1],
    [0, 8, 3, 4, 9, 7, 9, 11, 7, 9, 10, 11, -1, -1, -1, -1],
    [1, 10, 11, 1, 11, 4, 1, 4, 0, 7, 4, 11, -1, -1, -1, -1],
    [3, 1, 4, 3, 4, 8, 1, 10, 4, 7, 4, 11, 10, 11, 4, -1],
    [4, 11, 7, 9, 11, 4, 9, 2, 11, 9, 1, 2, -1, -1, -1, -1],
    [9, 7, 4, 9, 11, 7, 9, 1, 11, 2, 11, 1, 0, 8, 3, -1],
    [11, 7, 4, 11, 4, 2, 2, 4, 0, -1, -1, -1, -1, -1, -1, -1],
    [11, 7, 4, 11, 4, 2, 8, 3, 4, 3, 2, 4, -1, -1, -1, -1],
    [2, 9, 10, 2, 7, 9, 2, 3, 7, 7, 4, 9, -1, -1, -1, -1],
    [9, 10, 7, 9, 7, 4, 10, 2, 7, 8, 7, 0, 2, 0, 7, -1],
    [3, 7, 10, 3, 10, 2, 7, 4, 10, 1, 10, 0, 4, 0, 10, -1],
    [1, 10, 2, 8, 7, 4, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1],
    [4, 9, 1, 4, 1, 7, 7, 1, 3, -1, -1, -1, -1, -1, -1, -1],
    [4, 9, 1, 4, 1, 7, 0, 8, 1, 8, 7, 1, -1, -1, -1, -1],
    [4, 0, 3, 7, 4, 3, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1],
    [4, 8, 7, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1],
    [9, 10, 8, 10, 11, 8, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1],
    [3, 0, 9, 3, 9, 11, 11, 9, 10, -1, -1, -1, -1, -1, -1, -1],
    [0, 1, 10, 0, 10, 8, 8, 10, 11, -1, -1, -1, -1, -1, -1, -1],
    [3, 1, 10, 11, 3, 10, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1],
    [1, 2, 11, 1, 11, 9, 9, 11, 8, -1, -1, -1, -1, -1, -1, -1],
    [3, 0, 9, 3, 9, 11, 1, 2, 9, 2, 11, 9, -1, -1, -1, -1],
    [0, 2, 11, 8, 0, 11, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1],
    [3, 2, 11, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1],
    [2, 3, 8, 2, 8, 10, 10, 8, 9, -1, -1, -1, -1, -1, -1, -1],
    [9, 10, 2, 0, 9, 2, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1],
    [2, 3, 8, 2, 8, 10, 0, 1, 8, 1, 10, 8, -1, -1, -1, -1],
    [1, 10, 2, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1],
    [1, 3, 8, 9, 1, 8, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1],
    [0, 9, 1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1],
    [0, 3, 8, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1],
    [-1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1],
];
