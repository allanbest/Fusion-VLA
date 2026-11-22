//! What the viewport draws, checked by counting rather than by looking.

use fusion_core::Shape;
use fusion_math::{Isometry, Vec3};
use fusion_viewport::mesh::Geometry;

/// Every shape the engine has draws something, from a `Geometry` made either
/// way.
///
/// ## The bug this is here for
///
/// `Geometry` derived `Default`, which left its unit-sphere lookup table
/// empty, and an empty table is not an empty geometry — it is one that draws
/// **nothing at all** where a sphere should be. The editor built its geometry
/// with `Geometry::default()` and the demo binaries built theirs with
/// `Geometry::new()`, so every sphere in the editor was invisible and every
/// sphere in a demo was fine. It was reported as "the spheres don't have
/// renders", which is exactly what it was.
///
/// Both constructors are checked, because the bug was the difference between
/// them.
#[test]
fn every_shape_draws_something() {
    for (label, mut geometry) in [
        ("Geometry::default()", Geometry::default()),
        ("Geometry::new()", Geometry::new()),
    ] {
        for shape in [
            Shape::cuboid(Vec3::ONE),
            Shape::sphere(0.5),
            Shape::hull(&[
                Vec3::ZERO,
                Vec3::X,
                Vec3::Y,
                Vec3::Z,
                Vec3::new(1.0, 1.0, 1.0),
            ])
            .expect("a tetrahedron has volume"),
        ] {
            let before = geometry.triangles.len();
            geometry.solid_shape(&shape, &Isometry::IDENTITY, [1.0; 3]);
            assert!(
                geometry.triangles.len() > before,
                "{label} drew no triangles for {shape:?}"
            );

            let before = geometry.lines.len();
            geometry.wire_shape(&shape, &Isometry::IDENTITY, [1.0; 3]);
            assert!(
                geometry.lines.len() > before,
                "{label} drew no wireframe for {shape:?}"
            );
        }
    }
}

/// A sphere is shaded as a sphere, not as its facets.
///
/// The normals are on the unit-sphere table already; they were being thrown
/// away one line after they were read, and the face normal computed from the
/// winding put in their place. On a 16 x 10 lattice the difference is 320
/// visible facets.
#[test]
fn a_sphere_is_smooth_shaded() {
    let mut geometry = Geometry::default();
    geometry.solid_shape(&Shape::sphere(1.0), &Isometry::IDENTITY, [1.0; 3]);
    let mut worst: f32 = 0.0;
    for vertex in &geometry.triangles {
        let position = Vec3::from_array(vertex.position);
        let normal = Vec3::from_array(vertex.normal);
        if position.length() < 1e-6 {
            continue;
        }
        // On a unit sphere the outward normal *is* the position.
        worst = worst.max((normal - position.normalize()).length());
    }
    assert!(
        worst < 1e-3,
        "a vertex normal is {worst:.4} away from the surface normal, so the \
         sphere is being shaded by its facets"
    );
}

#[test]
fn zoom_keeps_the_focus_inside_the_default_clip_planes() {
    use fusion_viewport::camera::OrbitCamera;
    let mut camera = OrbitCamera::default();
    camera.zoom(1e10);
    assert!(camera.distance < camera.far);
    let distance = camera.distance;
    camera.zoom(f32::NAN);
    camera.zoom(-1.0);
    assert_eq!(camera.distance, distance);
    camera.zoom(1e-10);
    assert!(camera.distance > camera.near);
}
