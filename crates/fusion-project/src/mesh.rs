//! Wavefront `.obj` import — the second of PLAN.md §16's five creation verbs.
//!
//! §16: "mesh import → rigid body + collision proxy". The proxy is a convex
//! hull, and [`fusion_core::ConvexHull`] has been able to build one from a
//! point cloud since v0.5; what was missing was a way to get points *in*.
//!
//! ## Why only `v` lines matter
//!
//! The support function of a point set is already the support function of its
//! convex hull — the faces of the source mesh cannot change what the hull is.
//! So the parser reads vertices and ignores everything else: faces, normals,
//! texture coordinates, materials, groups, smoothing. That is not laziness
//! about the format, it is what the geometry says. A loader that parsed faces
//! would spend the effort to produce the identical hull.
//!
//! The consequence is worth being plain about, and [`fusion_core::ConvexHull`] is: **a
//! concave mesh is simulated as its convex hull**. A teacup holds no water and
//! a chair has no gap under the seat. Convex decomposition is §16's
//! `convex_decompose=True` and is not built.
//!
//! ## What this refuses
//!
//! A file with fewer than four vertices, or whose vertices are all on one line
//! or one plane, encloses no volume — and a body with no volume has no mass.
//! That is reported rather than approximated, because the alternative is a
//! body with an infinite acceleration and no clue why.

use std::path::Path;

use fusion_core::Shape;
use fusion_math::Vec3;

/// What went wrong reading a mesh.
#[derive(Debug)]
pub enum MeshError {
    /// The file could not be read.
    Io(std::io::Error),
    /// A `v` line that is not three numbers.
    Malformed {
        /// 1-based line number, as an editor would show it.
        line: usize,
    },
    /// Fewer than four vertices, or no volume between them.
    NoVolume {
        /// How many vertices were read.
        vertices: usize,
    },
}

impl std::fmt::Display for MeshError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            MeshError::Io(e) => write!(f, "{e}"),
            MeshError::Malformed { line } => {
                write!(f, "line {line} is not three numbers after `v`")
            }
            MeshError::NoVolume { vertices } => write!(
                f,
                "{vertices} vertices enclose no volume — a flat or empty mesh has no mass"
            ),
        }
    }
}

impl std::error::Error for MeshError {}

/// Read the vertices of a `.obj` file.
pub fn read_obj_points(path: impl AsRef<Path>) -> Result<Vec<Vec3>, MeshError> {
    let text = std::fs::read_to_string(path).map_err(MeshError::Io)?;
    parse_obj_points(&text)
}

/// The same, from text already in memory.
pub fn parse_obj_points(text: &str) -> Result<Vec<Vec3>, MeshError> {
    let mut points = Vec::new();
    for (index, line) in text.lines().enumerate() {
        let line = line.trim();
        // `v` exactly: `vn` and `vt` are normals and texture coordinates, and
        // a prefix test would read them as geometry.
        let Some(rest) = line.strip_prefix("v ").or_else(|| line.strip_prefix("v\t")) else {
            continue;
        };
        let mut numbers = rest
            .split_whitespace()
            .filter_map(|f| f.parse::<f32>().ok());
        let (Some(x), Some(y), Some(z)) = (numbers.next(), numbers.next(), numbers.next()) else {
            return Err(MeshError::Malformed { line: index + 1 });
        };
        points.push(Vec3::new(x, y, z));
    }
    Ok(points)
}

/// Read a `.obj` and return the convex hull of its vertices, centred on the
/// hull's own centroid.
///
/// Centred because a body's frame is its centre of mass: a mesh exported with
/// its origin at the artist's floor would otherwise spin about a point outside
/// itself, which looks like a broken inertia tensor and is a broken *origin*.
/// The offset is returned so the caller can place the body where the mesh was.
pub fn read_obj_hull(path: impl AsRef<Path>) -> Result<(Shape, Vec3), MeshError> {
    let points = read_obj_points(path)?;
    hull_of(&points)
}

/// The hull of a point cloud, centred, with the offset that was removed.
pub fn hull_of(points: &[Vec3]) -> Result<(Shape, Vec3), MeshError> {
    let vertices = points.len();
    let shape = Shape::hull(points).ok_or(MeshError::NoVolume { vertices })?;
    let Shape::Hull(hull) = &shape else {
        unreachable!("Shape::hull returns a hull");
    };
    let centre = hull.centroid();
    if centre.length() < 1e-6 {
        return Ok((shape, Vec3::ZERO));
    }
    let moved: Vec<Vec3> = hull.vertices().iter().map(|v| *v - centre).collect();
    let centred = Shape::hull(&moved).ok_or(MeshError::NoVolume { vertices })?;
    Ok((centred, centre))
}
