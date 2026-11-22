//! A scalar field on a lattice, and the ray marcher that looks inside it.
//!
//! PLAN.md §29's v0.9 row ends with "volume ray marching", and the reason it
//! is on the deep grid's row rather than the renderer's is §2: the picture is
//! a diagnostic. Every other way of drawing a continuum shows you its
//! *surface* — the marching-cubes mesh, the coloured points, the outline —
//! and the fields v0.9 added are interior. Where a bar is hottest is inside
//! it. Where it is about to break is inside it. A surface render of a
//! fracturing solid shows a shape with red edges and hides the crack.
//!
//! ## What is on the lattice
//!
//! Two numbers per voxel, and both are needed. The **value** is whatever field
//! is being looked at, normalized against a [`ScalarField`]'s fixed range so
//! that the colours mean the same thing from frame to frame (§19). The
//! **fill** is how much material is in the voxel at all, which is what makes
//! the empty space around a solid transparent rather than cold. A volume with
//! only a value channel renders a box the size of the bounding box, coloured
//! at the bottom of the scale, which is a picture of the sampling grid and not
//! of the material.
//!
//! ## Why the splat is the same one MPM uses
//!
//! Trilinear, mass-weighted: each material point deposits into the eight
//! voxels around it, weighted by how close it is to each. That is a coarser
//! kernel than the solver's quadratic B-spline and it is deliberately the
//! *same idea* — a value and a weight accumulated together, divided at the
//! end. A different reconstruction here would draw a field the solver does not
//! have, and the point of a diagnostic is that it shows the state rather than
//! an interpretation of it.

use fusion_core::SolidSet;
use fusion_math::Vec3;

use crate::colormap::ScalarField;

/// A scalar field sampled on a uniform lattice, ready to be marched.
///
/// Cheap to hand to the renderer and expensive to build, so it is built when
/// the simulation steps and handed over behind an `Arc` — see
/// [`Geometry::volume`](crate::Geometry::volume).
#[derive(Clone, Debug)]
pub struct Volume {
    /// World position of voxel (0, 0, 0).
    pub origin: Vec3,
    /// Edge length of one voxel, metres.
    pub cell: f32,
    /// Voxel counts along each axis.
    pub dims: [usize; 3],
    /// Value in `[0, 1]` — normalized against [`Volume::field`].
    pub value: Vec<f32>,
    /// How full of material the voxel is, in `[0, 1]`.
    pub fill: Vec<f32>,
    /// The scale the values were normalized against, and the colours to draw
    /// them in. Carried with the data so that the legend and the volume
    /// cannot drift apart.
    pub field: ScalarField,
    /// How opaque a full voxel is, per metre of ray travelled through it.
    ///
    /// Not per sample: an opacity per sample makes the picture depend on the
    /// step size, so halving the step to see more detail doubles the density
    /// of the material.
    pub density: f32,
    /// How opaque material at the *bottom* of the scale is, as a fraction of
    /// [`Volume::density`].
    ///
    /// The transfer function, and the difference between a diagnostic and a
    /// silhouette. Drawn at a uniform opacity, a block whose interior is hot
    /// renders as a solid dark shell with the interesting part behind it —
    /// which is what a surface render already showed, at more cost. Making
    /// opacity climb with the value lets the cold material be a faint mist the
    /// hot core shines through, and that is the picture volume rendering
    /// exists to produce.
    ///
    /// Not zero, because "no material" and "material at the bottom of the
    /// scale" have to stay distinguishable: at zero, an undamaged solid in the
    /// damage view is invisible, and the absence of a crack looks the same as
    /// the absence of a solid.
    pub base: f32,
}

/// The largest lattice a volume will be built on, per axis.
///
/// A hundred and twenty-eight cubed is two million voxels and 16 MB of upload
/// at `Rgba16Float`. Past that the ray march is not the cost — the transfer
/// is, every frame the field changes.
pub const MAX_DIM: usize = 128;

/// How opaque the bottom of the scale is by default. See [`Volume::base`].
pub const DEFAULT_BASE: f32 = 0.12;

impl Volume {
    /// Splat one scalar per material point onto a lattice.
    ///
    /// `values` is read per point index, which keeps this function ignorant of
    /// *which* field is being drawn: temperature, damage, speed and strain all
    /// arrive the same way, and the editor's view-mode switch is a different
    /// closure rather than a different code path.
    pub fn from_solid(
        solid: &SolidSet,
        field: ScalarField,
        density: f32,
        values: impl Fn(usize) -> f32,
    ) -> Option<Self> {
        Self::build(
            std::slice::from_ref(solid),
            field,
            density,
            DEFAULT_BASE,
            |_, i| values(i),
        )
    }

    /// One lattice over every solid in the scene.
    ///
    /// A scene can hold several continua and the renderer marches one volume,
    /// so they share a lattice rather than taking turns. That is also the
    /// physically honest arrangement: two solids that touch are one field
    /// across the join, and drawing them as two volumes would put a seam
    /// through material that does not have one.
    pub fn from_solids(
        solids: &[SolidSet],
        field: ScalarField,
        density: f32,
        values: impl Fn(&SolidSet, usize) -> f32,
    ) -> Option<Self> {
        Self::build(solids, field, density, DEFAULT_BASE, values)
    }

    /// The general form, with the transfer function's floor spelled out.
    pub fn build(
        solids: &[SolidSet],
        field: ScalarField,
        density: f32,
        base: f32,
        values: impl Fn(&SolidSet, usize) -> f32,
    ) -> Option<Self> {
        let live: Vec<&SolidSet> = solids.iter().filter(|s| !s.is_empty()).collect();
        if live.is_empty() {
            return None;
        }
        let (mut low, mut high) = (Vec3::splat(f32::MAX), Vec3::splat(f32::MIN));
        for solid in &live {
            for i in 0..solid.len() {
                let p = solid.particles.translation(i);
                low = low.min(p);
                high = high.max(p);
            }
        }
        // The finest solid's own cell, coarsened until the lattice fits. A
        // volume of a well-resolved solid is then the grid the solver ran on,
        // and a volume of an enormous one is as fine as the upload allows.
        let mut cell = live
            .iter()
            .map(|s| s.params.cell_size)
            .fold(f32::MAX, f32::min)
            .max(1e-4);
        // Two cells of margin either side, so the material's own surface is
        // inside the lattice and the fill can fall to zero across it. Without
        // it the outermost material meets the edge of the texture, where the
        // sampler clamps, and the solid renders with a hard skin.
        let span = high - low + Vec3::splat(4.0 * cell);
        let longest = span.x.max(span.y).max(span.z);
        if longest / cell > MAX_DIM as f32 {
            cell = longest / MAX_DIM as f32;
        }
        let origin = low - Vec3::splat(2.0 * cell);
        let dims = [
            ((span.x / cell).ceil() as usize + 1).clamp(2, MAX_DIM),
            ((span.y / cell).ceil() as usize + 1).clamp(2, MAX_DIM),
            ((span.z / cell).ceil() as usize + 1).clamp(2, MAX_DIM),
        ];

        let count = dims[0] * dims[1] * dims[2];
        let mut weighted = vec![0.0f32; count];
        let mut weight = vec![0.0f32; count];
        let index = |i: usize, j: usize, k: usize| i + dims[0] * (j + dims[1] * k);
        let voxel = cell * cell * cell;

        for solid in &live {
            // Weighted by how much material a point stands for rather than by
            // point count, so that two solids sampled at different densities
            // contribute in proportion to their matter and not to their
            // bookkeeping.
            let share = solid.params.point_volume() / voxel;
            for p in 0..solid.len() {
                let local = (solid.particles.translation(p) - origin) / cell;
                let base_node = [
                    local.x.floor() as isize,
                    local.y.floor() as isize,
                    local.z.floor() as isize,
                ];
                let frac = Vec3::new(
                    local.x - base_node[0] as f32,
                    local.y - base_node[1] as f32,
                    local.z - base_node[2] as f32,
                );
                let value = values(solid, p);
                for (dk, wk) in [(0usize, 1.0 - frac.z), (1, frac.z)] {
                    for (dj, wj) in [(0usize, 1.0 - frac.y), (1, frac.y)] {
                        for (di, wi) in [(0usize, 1.0 - frac.x), (1, frac.x)] {
                            let (i, j, k) = (
                                base_node[0] + di as isize,
                                base_node[1] + dj as isize,
                                base_node[2] + dk as isize,
                            );
                            if i < 0 || j < 0 || k < 0 {
                                continue;
                            }
                            let (i, j, k) = (i as usize, j as usize, k as usize);
                            if i >= dims[0] || j >= dims[1] || k >= dims[2] {
                                continue;
                            }
                            let w = wi * wj * wk * share;
                            let n = index(i, j, k);
                            weighted[n] += w * value;
                            weight[n] += w;
                        }
                    }
                }
            }
        }

        let mut value = vec![0.0f32; count];
        let mut fill = vec![0.0f32; count];
        for n in 0..count {
            if weight[n] > 1e-9 {
                value[n] = field.normalize(weighted[n] / weight[n]);
                fill[n] = weight[n].min(1.0);
            }
        }

        Some(Self {
            origin,
            cell,
            dims,
            value,
            fill,
            field,
            density,
            base: base.clamp(0.0, 1.0),
        })
    }

    /// Voxel count.
    pub fn len(&self) -> usize {
        self.value.len()
    }

    /// True where the lattice holds nothing.
    pub fn is_empty(&self) -> bool {
        self.value.is_empty()
    }

    /// The lattice as `Rgba16Float` texels: value, fill, and two spare
    /// channels the shader ignores.
    ///
    /// Half precision, because the value is already normalized to `[0, 1]` and
    /// eleven bits of mantissa is four decimal digits on a scale a colour map
    /// resolves to two hundred and fifty six levels. It is also the format
    /// that filters everywhere: `R32Float` needs a device feature that not
    /// every backend has, and an unfiltered volume renders as visible cubes.
    pub fn texels(&self) -> Vec<[u16; 4]> {
        (0..self.len())
            .map(|i| [half(self.value[i]), half(self.fill[i]), 0, 0])
            .collect()
    }

    /// 256 colours across the field's fixed range, as the shader's lookup.
    ///
    /// Sampled from the same [`ScalarField`] the points and the legend use, so
    /// a material point drawn at 400 K and the volume at 400 K are the same
    /// colour — which is the whole of §19's argument for fixed scales, applied
    /// across two renderers of the same data.
    pub fn lookup(&self) -> Vec<[u8; 4]> {
        (0..256)
            .map(|i| {
                let t = i as f32 / 255.0;
                let value = self.field.low + t * (self.field.high - self.field.low);
                let [r, g, b] = self.field.color(value);
                [srgb(r), srgb(g), srgb(b), 255]
            })
            .collect()
    }
}

/// `f32` to IEEE half, by hand.
///
/// Twenty lines against a dependency at L6 that would be the only one in the
/// crate outside wgpu and winit. Rounds toward zero, which for a field in
/// `[0, 1]` costs half an ULP nobody can see in a colour.
fn half(value: f32) -> u16 {
    let bits = value.to_bits();
    let sign = ((bits >> 16) & 0x8000) as u16;
    let exponent = ((bits >> 23) & 0xff) as i32 - 127 + 15;
    let mantissa = (bits & 0x7f_ffff) >> 13;
    if exponent <= 0 {
        // Subnormal or zero. The field is normalized, so this is the bottom of
        // the scale rather than a case worth being clever about.
        sign
    } else if exponent >= 0x1f {
        sign | 0x7c00
    } else {
        sign | ((exponent as u16) << 10) | mantissa as u16
    }
}

/// Linear to sRGB, for the lookup texture.
fn srgb(value: f32) -> u8 {
    let v = value.clamp(0.0, 1.0);
    let encoded = if v <= 0.003_130_8 {
        v * 12.92
    } else {
        1.055 * v.powf(1.0 / 2.4) - 0.055
    };
    (encoded * 255.0 + 0.5) as u8
}
