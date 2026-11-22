//! Geometry buffers for the debug viewport.
//!
//! Everything the viewport draws is one of two primitives: a shaded triangle
//! or a coloured line. PLAN.md §19 makes the point that visualization modes
//! are one renderer plus a colour-mapping stage rather than N renderers; this
//! is that idea at its smallest, with the colour chosen on the CPU.

use fusion_core::Shape;
use fusion_math::{vec3, Isometry, Vec3};

#[repr(C)]
#[derive(Clone, Copy, Debug, bytemuck::Pod, bytemuck::Zeroable)]
pub struct Vertex {
    pub position: [f32; 3],
    /// Zero-length normals mark unlit geometry: lines and overlays.
    pub normal: [f32; 3],
    pub color: [f32; 3],
    /// `[roughness, metallic]`.
    ///
    /// Per-vertex rather than per-draw because the viewport rebuilds one
    /// vertex stream a frame and issues two draws for the whole scene; a
    /// per-draw material would mean sorting geometry by material and issuing
    /// a draw per bucket, which is the right design for a renderer with a
    /// scene graph and the wrong one for a debug view with neither.
    pub material: [f32; 2],
}

impl Vertex {
    pub const LAYOUT: wgpu::VertexBufferLayout<'static> = wgpu::VertexBufferLayout {
        array_stride: std::mem::size_of::<Vertex>() as wgpu::BufferAddress,
        step_mode: wgpu::VertexStepMode::Vertex,
        attributes: &wgpu::vertex_attr_array![
            0 => Float32x3, 1 => Float32x3, 2 => Float32x3, 3 => Float32x2
        ],
    };
}

/// How a surface responds to light: the two parameters a metallic-roughness
/// workflow needs, and no more.
///
/// PLAN.md §18 asks for PBR shading in the v0.5 viewport, not a material
/// system. Roughness and metalness are what separate steel from rubber
/// visually; everything else a full model carries — clearcoat, anisotropy,
/// transmission — is for a renderer Fusion has said it will not become.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Surface {
    pub roughness: f32,
    pub metallic: f32,
}

impl Default for Surface {
    fn default() -> Self {
        Self::MATTE
    }
}

impl Surface {
    /// A neutral dielectric. What everything is until told otherwise.
    pub const MATTE: Self = Self {
        roughness: 0.75,
        metallic: 0.0,
    };
    pub const PLASTIC: Self = Self {
        roughness: 0.35,
        metallic: 0.0,
    };
    pub const METAL: Self = Self {
        roughness: 0.28,
        metallic: 1.0,
    };
    /// Water and other liquids: smooth, and a dielectric.
    pub const LIQUID: Self = Self {
        roughness: 0.08,
        metallic: 0.0,
    };

    fn to_array(self) -> [f32; 2] {
        [
            self.roughness.clamp(0.02, 1.0),
            self.metallic.clamp(0.0, 1.0),
        ]
    }
}

/// One frame's worth of debug geometry, rebuilt from scratch each frame.
///
/// Rebuilding rather than caching is the right call at v0.1: the CPU cost is
/// trivial next to the solve, and a cache would be one more thing that can
/// disagree with the simulation.
pub struct Geometry {
    pub triangles: Vec<Vertex>,
    pub lines: Vec<Vertex>,
    /// Screen-space triangles in clip coordinates, drawn last with no depth
    /// test and no lighting. The legend lives here.
    pub overlay: Vec<Vertex>,
    /// A scalar field to look *into*, marched after the scene is drawn.
    ///
    /// Behind an `Arc` because it is megabytes and it is handed over every
    /// frame while changing only when the simulation steps — the renderer
    /// compares the pointer and skips the upload when it is the same lattice
    /// it already has.
    pub volume: Option<std::sync::Arc<crate::Volume>>,
    unit_sphere: Vec<([f32; 3], [f32; 3])>,
    /// Material applied to triangles emitted from now on. Set-and-draw rather
    /// than an argument on every call, so that adding materials did not have
    /// to touch every existing call site.
    surface: Surface,
}

/// Written out rather than derived, and that is the whole point of it.
///
/// A derived `Default` leaves `unit_sphere` empty, and an empty unit sphere is
/// not an empty *geometry* — it is a geometry that silently draws nothing
/// where a sphere should be. The editor built its `Geometry` with
/// `Geometry::default()` and the demo binaries built theirs with
/// `Geometry::new()`, so **every sphere in the editor was invisible** and
/// every sphere in a demo was fine, which is why it survived: the two paths
/// disagreed and only one of them was ever looked at.
///
/// The lesson is narrower than "avoid derived Default". It is that a derived
/// `Default` is only correct when *every* field's zero value is a valid state,
/// and a lookup table's is not.
impl Default for Geometry {
    fn default() -> Self {
        Self {
            triangles: Vec::new(),
            lines: Vec::new(),
            overlay: Vec::new(),
            volume: None,
            unit_sphere: unit_sphere_triangles(16, 10),
            surface: Surface::default(),
        }
    }
}

/// Most pieces a face is cut into along an edge when it carries a field.
const MAX_FIELD_DIVISIONS: usize = 40;

impl Geometry {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn clear(&mut self) {
        self.triangles.clear();
        self.lines.clear();
        self.overlay.clear();
        self.volume = None;
        self.surface = Surface::default();
    }

    /// Material for subsequent triangles.
    pub fn set_surface(&mut self, surface: Surface) {
        self.surface = surface;
    }

    pub fn surface(&self) -> Surface {
        self.surface
    }

    /// Run `body` with a different material, then put the old one back.
    pub fn with_surface(&mut self, surface: Surface, body: impl FnOnce(&mut Self)) {
        let previous = self.surface;
        self.surface = surface;
        body(self);
        self.surface = previous;
    }

    pub fn is_empty(&self) -> bool {
        self.triangles.is_empty() && self.lines.is_empty() && self.overlay.is_empty()
    }

    pub fn line(&mut self, a: Vec3, b: Vec3, color: [f32; 3]) {
        self.lines.push(Vertex {
            position: a.to_array(),
            normal: [0.0; 3],
            color,
            material: [1.0, 0.0],
        });
        self.lines.push(Vertex {
            position: b.to_array(),
            normal: [0.0; 3],
            color,
            material: [1.0, 0.0],
        });
    }

    /// A small three-axis cross. The debug viewport's stand-in for a point,
    /// since point primitives are one pixel and effectively invisible.
    pub fn point(&mut self, p: Vec3, size: f32, color: [f32; 3]) {
        let s = Vec3::splat(size);
        self.line(p - Vec3::X * s.x, p + Vec3::X * s.x, color);
        self.line(p - Vec3::Y * s.y, p + Vec3::Y * s.y, color);
        self.line(p - Vec3::Z * s.z, p + Vec3::Z * s.z, color);
    }

    pub fn arrow(&mut self, from: Vec3, to: Vec3, color: [f32; 3]) {
        self.line(from, to, color);
        let dir = (to - from).normalize_or_zero();
        if dir == Vec3::ZERO {
            return;
        }
        let side = dir.any_orthonormal();
        let head = (to - from).length() * 0.18;
        self.line(to, to - dir * head + side * head * 0.5, color);
        self.line(to, to - dir * head - side * head * 0.5, color);
    }

    /// A single flat-shaded triangle. Public because cloth is a triangle soup
    /// rather than a shape.
    pub fn triangle_shaded(&mut self, a: Vec3, b: Vec3, c: Vec3, color: [f32; 3]) {
        self.triangle(a, b, c, color);
    }

    /// An indexed, smoothly shaded surface.
    ///
    /// The viewport deliberately knows nothing about where a mesh came from:
    /// it can be a fluid reconstruction, an SDF, or a future imported asset.
    /// Invalid triangle indices are ignored so a diagnostic view cannot turn a
    /// partially-built mesh into a panic in the render loop.
    pub fn indexed_surface(
        &mut self,
        positions: &[Vec3],
        normals: &[Vec3],
        indices: &[u32],
        color: [f32; 3],
    ) {
        for triangle in indices.chunks_exact(3) {
            let [a, b, c] = [
                triangle[0] as usize,
                triangle[1] as usize,
                triangle[2] as usize,
            ];
            let (Some(a), Some(b), Some(c)) =
                (positions.get(a), positions.get(b), positions.get(c))
            else {
                continue;
            };
            let face_normal = (*b - *a).cross(*c - *a).normalize_or_zero();
            for (index, position) in [
                (triangle[0] as usize, a),
                (triangle[1] as usize, b),
                (triangle[2] as usize, c),
            ] {
                let normal = normals
                    .get(index)
                    .copied()
                    .filter(|normal| normal.length_squared() > 1e-8)
                    .unwrap_or(face_normal);
                self.triangles.push(Vertex {
                    position: position.to_array(),
                    normal: normal.to_array(),
                    color,
                    material: self.surface.to_array(),
                });
            }
        }
    }

    /// [`Geometry::indexed_surface`] with a colour per vertex, blended across
    /// each triangle. Vertices without a colour are drawn grey.
    pub fn indexed_surface_coloured(
        &mut self,
        positions: &[Vec3],
        normals: &[Vec3],
        indices: &[u32],
        colours: &[[f32; 3]],
    ) {
        let material = self.surface.to_array();
        for triangle in indices.chunks_exact(3) {
            let corners = [
                triangle[0] as usize,
                triangle[1] as usize,
                triangle[2] as usize,
            ];
            let (Some(a), Some(b), Some(c)) = (
                positions.get(corners[0]),
                positions.get(corners[1]),
                positions.get(corners[2]),
            ) else {
                continue;
            };
            let face_normal = (*b - *a).cross(*c - *a).normalize_or_zero();
            for (index, position) in corners.into_iter().zip([a, b, c]) {
                let normal = normals
                    .get(index)
                    .copied()
                    .filter(|normal| normal.length_squared() > 1e-8)
                    .unwrap_or(face_normal);
                self.triangles.push(Vertex {
                    position: position.to_array(),
                    normal: normal.to_array(),
                    color: colours.get(index).copied().unwrap_or([0.6; 3]),
                    material,
                });
            }
        }
    }

    fn triangle(&mut self, a: Vec3, b: Vec3, c: Vec3, color: [f32; 3]) {
        let normal = (b - a).cross(c - a).normalize_or_zero();
        self.smooth_triangle([(a, normal), (b, normal), (c, normal)], color);
    }

    /// One triangle with a normal per vertex, so a curved surface is shaded as
    /// one rather than as its facets.
    ///
    /// The distinction is not subtle on the one shape that needs it. A sphere
    /// is drawn from a 16 x 10 lattice, and handing those 320 triangles a
    /// *face* normal each is a faceted ball — which is what "spheres don't
    /// have renders" turned out to mean. The normals were already being
    /// computed, in `unit_sphere_triangles`, and thrown away one line later.
    fn smooth_triangle(&mut self, vertices: [(Vec3, Vec3); 3], color: [f32; 3]) {
        let material = self.surface.to_array();
        for (position, normal) in vertices {
            self.triangles.push(Vertex {
                position: position.to_array(),
                normal: normal.to_array(),
                color,
                material,
            });
        }
    }

    /// Solid, flat-shaded geometry for a shape.
    pub fn solid_shape(&mut self, shape: &Shape, pose: &Isometry, color: [f32; 3]) {
        match shape {
            // The hull's own faces, which are triangles already — the shape
            // the solver sees is the shape on screen, which for an imported
            // mesh is the point: a hull drawn as its original triangles would
            // hide the convexification that collision actually uses.
            Shape::Hull(hull) => {
                for face in hull.faces() {
                    let p =
                        |k: usize| pose.transform_point(hull.vertices()[face.indices[k] as usize]);
                    self.triangle(p(0), p(1), p(2), color);
                }
            }
            Shape::Cuboid { half_extents } => {
                let half_extents = *half_extents;
                for [a, b, c] in cuboid_triangles(half_extents) {
                    self.triangle(
                        pose.transform_point(a),
                        pose.transform_point(b),
                        pose.transform_point(c),
                        color,
                    );
                }
            }
            Shape::Sphere { radius } => {
                let radius = *radius;
                // Moved out and back so the shared unit sphere can be read
                // while `self.triangle` writes — cheaper than cloning it for
                // every sphere in the scene, every frame.
                let sphere = std::mem::take(&mut self.unit_sphere);
                for chunk in sphere.chunks(3) {
                    // The stored normal, rotated by the pose rather than
                    // recomputed from the face: on a sphere the two differ by
                    // exactly the faceting.
                    let v = |k: usize| {
                        (
                            pose.transform_point(Vec3::from_array(chunk[k].0) * radius),
                            pose.transform_direction(Vec3::from_array(chunk[k].1)),
                        )
                    };
                    self.smooth_triangle([v(0), v(1), v(2)], color);
                }
                self.unit_sphere = sphere;
            }
        }
    }

    /// Solid geometry for a shape, coloured point by point: `colour_at` is
    /// asked for the colour at body-local points over the surface, and the
    /// renderer blends between them. Flat faces are cut into triangles no
    /// longer than `detail` along an edge (up to a limit), so a field that
    /// changes across a face — the band of load along a beam's underside, the
    /// heat climbing a block — shades smoothly instead of as one flat colour
    /// or a cloud of dots.
    pub fn solid_shape_field(
        &mut self,
        shape: &Shape,
        pose: &Isometry,
        detail: f32,
        mut colour_at: impl FnMut(Vec3) -> [f32; 3],
    ) {
        let material = self.surface.to_array();
        let mut flat = |g: &mut Self, [a, b, c]: [Vec3; 3]| {
            let longest = (b - a).length().max((c - b).length()).max((a - c).length());
            let n = ((longest / detail.max(1e-6)).ceil() as usize).clamp(1, MAX_FIELD_DIVISIONS);
            let normal = pose
                .transform_direction((b - a).cross(c - a))
                .normalize_or_zero()
                .to_array();
            // The lattice of points on this triangle, row by row.
            let at = |i: usize, j: usize| {
                a + (b - a) * (i as f32 / n as f32) + (c - a) * (j as f32 / n as f32)
            };
            let mut colours = Vec::with_capacity((n + 1) * (n + 2) / 2);
            let mut row_start = Vec::with_capacity(n + 1);
            for i in 0..=n {
                row_start.push(colours.len());
                for j in 0..=(n - i) {
                    colours.push(colour_at(at(i, j)));
                }
            }
            let colour = |i: usize, j: usize| colours[row_start[i] + j];
            let vertex = |g: &mut Self, i: usize, j: usize| {
                g.triangles.push(Vertex {
                    position: pose.transform_point(at(i, j)).to_array(),
                    normal,
                    color: colour(i, j),
                    material,
                });
            };
            for i in 0..n {
                for j in 0..(n - i) {
                    vertex(g, i, j);
                    vertex(g, i + 1, j);
                    vertex(g, i, j + 1);
                    if j + 1 < n - i {
                        vertex(g, i + 1, j);
                        vertex(g, i + 1, j + 1);
                        vertex(g, i, j + 1);
                    }
                }
            }
        };
        match shape {
            Shape::Hull(hull) => {
                for face in hull.faces() {
                    let p = |k: usize| hull.vertices()[face.indices[k] as usize];
                    flat(self, [p(0), p(1), p(2)]);
                }
            }
            Shape::Cuboid { half_extents } => {
                for triangle in cuboid_triangles(*half_extents) {
                    flat(self, triangle);
                }
            }
            Shape::Sphere { radius } => {
                // The sphere's own lattice is fine enough to carry a field.
                let radius = *radius;
                let sphere = std::mem::take(&mut self.unit_sphere);
                for chunk in sphere.chunks(3) {
                    for &(p, n) in chunk {
                        let local = Vec3::from_array(p) * radius;
                        self.triangles.push(Vertex {
                            position: pose.transform_point(local).to_array(),
                            normal: pose.transform_direction(Vec3::from_array(n)).to_array(),
                            color: colour_at(local),
                            material,
                        });
                    }
                }
                self.unit_sphere = sphere;
            }
        }
    }

    /// Wireframe outline for a shape.
    pub fn wire_shape(&mut self, shape: &Shape, pose: &Isometry, color: [f32; 3]) {
        match shape {
            // Every face edge. Drawn twice where two faces share one, which is
            // cheaper than building the unique-edge set for a shape that is
            // usually a few dozen triangles.
            Shape::Hull(hull) => {
                for face in hull.faces() {
                    let p =
                        |k: usize| pose.transform_point(hull.vertices()[face.indices[k] as usize]);
                    self.line(p(0), p(1), color);
                    self.line(p(1), p(2), color);
                    self.line(p(2), p(0), color);
                }
            }
            Shape::Cuboid { half_extents } => {
                let c = corners(*half_extents, pose);
                for (i, j) in CUBE_EDGES {
                    self.line(c[i], c[j], color);
                }
            }
            Shape::Sphere { radius } => {
                let radius = *radius;
                // Three great circles read as a sphere from any angle.
                for axis in 0..3 {
                    let segments = 24;
                    let mut prev = None;
                    for k in 0..=segments {
                        let t = k as f32 / segments as f32 * std::f32::consts::TAU;
                        let (s, c) = t.sin_cos();
                        let local = match axis {
                            0 => vec3(0.0, s, c),
                            1 => vec3(c, 0.0, s),
                            _ => vec3(c, s, 0.0),
                        } * radius;
                        let p = pose.transform_point(local);
                        if let Some(q) = prev {
                            self.line(q, p, color);
                        }
                        prev = Some(p);
                    }
                }
            }
        }
    }

    /// The twelve edges of an axis-aligned box, for AABB overlays.
    pub fn wire_aabb(&mut self, min: Vec3, max: Vec3, color: [f32; 3]) {
        let half = (max - min) * 0.5;
        let pose = Isometry::from_translation((max + min) * 0.5);
        let c = corners(half, &pose);
        for (i, j) in CUBE_EDGES {
            self.line(c[i], c[j], color);
        }
    }

    /// Ground grid, drawn on the y = `height` plane.
    pub fn grid(&mut self, extent: f32, spacing: f32, height: f32, color: [f32; 3]) {
        let n = (extent / spacing) as i32;
        for k in -n..=n {
            let t = k as f32 * spacing;
            let strong = k % 5 == 0;
            let c = if strong {
                color
            } else {
                [color[0] * 0.55, color[1] * 0.55, color[2] * 0.55]
            };
            self.line(vec3(-extent, height, t), vec3(extent, height, t), c);
            self.line(vec3(t, height, -extent), vec3(t, height, extent), c);
        }
    }
}

const CUBE_EDGES: [(usize, usize); 12] = [
    (0, 1),
    (1, 3),
    (3, 2),
    (2, 0),
    (4, 5),
    (5, 7),
    (7, 6),
    (6, 4),
    (0, 4),
    (1, 5),
    (2, 6),
    (3, 7),
];

fn corners(half: Vec3, pose: &Isometry) -> [Vec3; 8] {
    let mut out = [Vec3::ZERO; 8];
    for (i, slot) in out.iter_mut().enumerate() {
        let s = vec3(
            if i & 1 == 0 { -1.0 } else { 1.0 },
            if i & 2 == 0 { -1.0 } else { 1.0 },
            if i & 4 == 0 { -1.0 } else { 1.0 },
        );
        *slot = pose.transform_point(half.mul_element_wise(s));
    }
    out
}

fn cuboid_triangles(half: Vec3) -> Vec<[Vec3; 3]> {
    let c = corners(half, &Isometry::IDENTITY);
    // Wound counter-clockwise when seen from outside.
    let faces = [
        [0, 2, 3, 1], // -z
        [4, 5, 7, 6], // +z
        [0, 1, 5, 4], // -y
        [2, 6, 7, 3], // +y
        [0, 4, 6, 2], // -x
        [1, 3, 7, 5], // +x
    ];
    let mut out = Vec::with_capacity(12);
    for f in faces {
        out.push([c[f[0]], c[f[1]], c[f[2]]]);
        out.push([c[f[0]], c[f[2]], c[f[3]]]);
    }
    out
}

/// UV sphere as a flat triangle list of (position, normal) on the unit sphere.
fn unit_sphere_triangles(segments: usize, rings: usize) -> Vec<([f32; 3], [f32; 3])> {
    let point = |ring: usize, seg: usize| {
        let phi = ring as f32 / rings as f32 * std::f32::consts::PI;
        let theta = seg as f32 / segments as f32 * std::f32::consts::TAU;
        let v = vec3(phi.sin() * theta.cos(), phi.cos(), phi.sin() * theta.sin());
        (v.to_array(), v.to_array())
    };
    let mut out = Vec::with_capacity(segments * rings * 6);
    for ring in 0..rings {
        for seg in 0..segments {
            let (a, b) = (point(ring, seg), point(ring, seg + 1));
            let (c, d) = (point(ring + 1, seg + 1), point(ring + 1, seg));
            out.extend_from_slice(&[a, b, c]);
            out.extend_from_slice(&[a, c, d]);
        }
    }
    out
}

// ------------------------------------------------------------------ overlay

/// A 3x5 bitmap font, as one `u16` per glyph column-major-ish: bit `r*3 + c`
/// is the pixel at row `r` from the top, column `c`.
///
/// Three pixels wide is the smallest a digit can be and still be a digit. The
/// alternative to carrying this was a legend with a colour bar and no numbers,
/// which PLAN.md §19 is explicit is not a legend — "a legend showing units
/// always visible". Fifteen bits per glyph is a cheap price for the section's
/// third rule.
///
/// Only the glyphs a scale actually needs: digits, a sign, a decimal point,
/// exponent notation, and the letters that appear in SI units.
fn glyph(c: char) -> u16 {
    match c.to_ascii_lowercase() {
        '0' => 0b111_101_101_101_111,
        '1' => 0b010_110_010_010_111,
        '2' => 0b111_001_111_100_111,
        '3' => 0b111_001_111_001_111,
        '4' => 0b101_101_111_001_001,
        '5' => 0b111_100_111_001_111,
        '6' => 0b111_100_111_101_111,
        '7' => 0b111_001_001_001_001,
        '8' => 0b111_101_111_101_111,
        '9' => 0b111_101_111_001_111,
        '.' => 0b000_000_000_000_010,
        '-' => 0b000_000_111_000_000,
        '+' => 0b000_010_111_010_000,
        'e' => 0b111_100_110_100_111,
        '^' => 0b010_101_000_000_000,
        '/' => 0b001_001_010_100_100,
        ' ' => 0,
        'a' => 0b111_101_111_101_101,
        'c' => 0b111_100_100_100_111,
        'd' => 0b110_101_101_101_110,
        'g' => 0b111_100_101_101_111,
        'h' => 0b101_101_111_101_101,
        'i' => 0b111_010_010_010_111,
        'k' => 0b101_101_110_101_101,
        'l' => 0b100_100_100_100_111,
        'm' => 0b101_111_111_101_101,
        'n' => 0b110_101_101_101_101,
        'o' => 0b111_101_101_101_111,
        'p' => 0b111_101_111_100_100,
        'r' => 0b111_101_110_101_101,
        's' => 0b111_100_111_001_111,
        't' => 0b111_010_010_010_010,
        'u' => 0b101_101_101_101_111,
        'y' => 0b101_101_111_001_111,
        _ => 0b111_101_101_101_111,
    }
}

impl Geometry {
    /// A filled rectangle in normalized device coordinates: `-1..1` on both
    /// axes, `-1` at the bottom left.
    ///
    /// Overlay geometry skips the camera entirely — it is emitted in clip
    /// space and drawn last with no depth test. That is what makes a legend
    /// "always visible" in the sense §19 means: not merely on top, but
    /// unaffected by where the camera happens to be pointing.
    pub fn overlay_quad(&mut self, min: [f32; 2], max: [f32; 2], color: [f32; 3]) {
        let v = |x: f32, y: f32| Vertex {
            position: [x, y, 0.0],
            normal: [0.0; 3],
            color,
            material: [1.0, 0.0],
        };
        for p in [
            (min[0], min[1]),
            (max[0], min[1]),
            (max[0], max[1]),
            (min[0], min[1]),
            (max[0], max[1]),
            (min[0], max[1]),
        ] {
            self.overlay.push(v(p.0, p.1));
        }
    }

    /// A line in normalized device coordinates, drawn as a thin quad so it is
    /// the same primitive as everything else in the overlay.
    pub fn overlay_line(&mut self, from: [f32; 2], to: [f32; 2], color: [f32; 3]) {
        let (dx, dy) = (to[0] - from[0], to[1] - from[1]);
        let len = (dx * dx + dy * dy).sqrt().max(1e-6);
        let (nx, ny) = (-dy / len * 0.0015, dx / len * 0.0015);
        let v = |x: f32, y: f32| Vertex {
            position: [x, y, 0.0],
            normal: [0.0; 3],
            color,
            material: [1.0, 0.0],
        };
        let (a, b) = ([from[0] + nx, from[1] + ny], [from[0] - nx, from[1] - ny]);
        let (c, d) = ([to[0] + nx, to[1] + ny], [to[0] - nx, to[1] - ny]);
        for p in [a, b, c, b, d, c] {
            self.overlay.push(v(p[0], p[1]));
        }
    }

    /// Text in normalized device coordinates, at `size` NDC units per glyph
    /// height.
    pub fn overlay_text(&mut self, text: &str, at: [f32; 2], size: f32, color: [f32; 3]) {
        // Square pixels regardless of the window's aspect would need the
        // aspect ratio here; the legend is small and short enough that the
        // stretch is not worth another uniform.
        let px = size / 5.0;
        let mut x = at[0];
        for ch in text.chars() {
            let bits = glyph(ch);
            for row in 0..5 {
                for col in 0..3 {
                    if bits >> (14 - (row * 3 + col)) & 1 == 1 {
                        let x0 = x + col as f32 * px;
                        let y0 = at[1] + (4 - row) as f32 * px;
                        self.overlay_quad([x0, y0], [x0 + px, y0 + px], color);
                    }
                }
            }
            x += px * 4.0;
        }
    }
}
