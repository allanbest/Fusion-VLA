//! Importing a mesh, and what it becomes.

use fusion_core::Shape;
use fusion_math::{vec3, Vec3};
use fusion_project::{hull_of, parse_obj_points, read_obj_hull, MeshError};

/// A unit cube, as an exporter would write it: vertices, normals, texture
/// coordinates and faces, in the order Blender emits them.
const CUBE: &str = "\
# Blender v3.6 OBJ File
mtllib cube.mtl
o Cube
v -0.500000 -0.500000 0.500000
v -0.500000 0.500000 0.500000
v -0.500000 -0.500000 -0.500000
v -0.500000 0.500000 -0.500000
v 0.500000 -0.500000 0.500000
v 0.500000 0.500000 0.500000
v 0.500000 -0.500000 -0.500000
v 0.500000 0.500000 -0.500000
vn -1.0000 0.0000 0.0000
vn 0.0000 0.0000 -1.0000
vt 0.375000 0.000000
usemtl Material
s off
f 1/1/1 2/2/1 4/3/1
f 1/1/1 4/3/1 3/4/1
";

/// Vertex lines are read; `vn` and `vt` are not.
///
/// The one parsing mistake worth a test: `line.starts_with("v")` reads a
/// normal as a position, and a mesh imported that way is a hull with points
/// scattered a unit away from it in every direction — bigger than the model,
/// the wrong shape, and no error anywhere.
#[test]
fn only_vertex_lines_are_geometry() {
    let points = parse_obj_points(CUBE).expect("the cube parses");
    assert_eq!(points.len(), 8, "eight vertices, not eleven");
    for p in &points {
        assert!(
            (p.length() - 0.75f32.sqrt()).abs() < 1e-5,
            "{p:?} is not a corner of a unit cube"
        );
    }
}

/// A cube imports as a cube: eight vertices, unit volume, and a box's inertia.
#[test]
fn a_cube_imports_as_a_cube() {
    let (shape, offset) = hull_of(&parse_obj_points(CUBE).unwrap()).expect("it has volume");
    assert!(offset.length() < 1e-5, "a centred cube needs no offset");

    let Shape::Hull(hull) = &shape else {
        panic!("expected a hull");
    };
    assert_eq!(hull.vertices().len(), 8);
    assert!(
        (shape.volume() - 1.0).abs() < 1e-4,
        "volume {} for a unit cube",
        shape.volume()
    );

    // The inertia of a solid cube is `m a^2 / 6` on every axis, and the hull
    // integral has to agree with the closed form — this is the check that the
    // polyhedron inertia is right rather than merely plausible.
    let properties = shape.mass_properties(1000.0);
    let expected = 1000.0 / 6.0;
    for axis in 0..3 {
        assert!(
            (properties.inertia_diagonal[axis] - expected).abs() < 0.02 * expected,
            "inertia {:?} against a cube's {expected}",
            properties.inertia_diagonal
        );
    }
}

/// A mesh whose origin is not its centre is re-centred, and says by how much.
///
/// An exporter often puts the origin at the model's feet. A body's frame is
/// its centre of mass, so a hull left off-centre spins about a point outside
/// itself — which reads as a broken inertia tensor and is a broken origin.
#[test]
fn an_off_centre_mesh_is_recentred() {
    let shifted: Vec<Vec3> = parse_obj_points(CUBE)
        .unwrap()
        .iter()
        .map(|p| *p + vec3(3.0, 10.0, -2.0))
        .collect();
    let (shape, offset) = hull_of(&shifted).expect("it has volume");
    assert!(
        (offset - vec3(3.0, 10.0, -2.0)).length() < 1e-3,
        "the offset should be where the mesh was: {offset:?}"
    );
    let Shape::Hull(hull) = &shape else {
        panic!("expected a hull")
    };
    assert!(
        hull.centroid().length() < 1e-3,
        "the hull is not centred: {:?}",
        hull.centroid()
    );
}

/// A flat mesh is refused rather than approximated.
#[test]
fn a_mesh_with_no_volume_is_refused() {
    // A square in the xy plane: four vertices, no thickness.
    let flat = "v 0 0 0\nv 1 0 0\nv 1 1 0\nv 0 1 0\n";
    let points = parse_obj_points(flat).unwrap();
    assert_eq!(points.len(), 4);
    match hull_of(&points) {
        Err(MeshError::NoVolume { vertices }) => assert_eq!(vertices, 4),
        other => panic!("a flat mesh should have no volume: {other:?}"),
    }
}

/// And a malformed vertex line is an error with a line number on it.
#[test]
fn a_broken_vertex_line_says_which_one() {
    let broken = "v 0 0 0\nv 1 0\nv 1 1 0\n";
    match parse_obj_points(broken) {
        Err(MeshError::Malformed { line }) => assert_eq!(line, 2),
        other => panic!("expected a line number: {other:?}"),
    }
}

/// The round trip through a file, which is the path the editor uses.
#[test]
fn a_file_on_disk_becomes_a_shape() {
    let path = std::env::temp_dir().join("fusion-import-test.obj");
    std::fs::write(&path, CUBE).unwrap();
    let (shape, _) = read_obj_hull(&path).expect("it imports");
    assert!((shape.volume() - 1.0).abs() < 1e-4);
    let _ = std::fs::remove_file(&path);
}
