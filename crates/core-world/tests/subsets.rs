use core_math::{Pose, Vec3};
use core_world::{library, Representation, TetMesh, World};
use std::sync::Arc;
#[test]
fn compact_views_share_geometry_and_commit_only_their_own_dynamic_state() {
    let mut world = World::new();
    let m = world.add_material(library::rubber());
    let a = world.add_solid("a", m, TetMesh::cuboid(Vec3::ONE, 0.5), Pose::IDENTITY);
    let b = world.add_solid(
        "b",
        m,
        TetMesh::cuboid(Vec3::ONE, 0.5),
        Pose::at(Vec3::X * 3.0),
    );
    let mut view = world.subset(&[b.0 as usize]);
    let (Representation::Solid(original), Representation::Solid(local)) = (
        &world.object(b).representation,
        &view.objects()[0].representation,
    ) else {
        unreachable!()
    };
    assert!(Arc::ptr_eq(&original.mesh, &local.mesh));
    assert!(Arc::ptr_eq(&original.surface, &local.surface));
    let untouched = world.object(a).centre_of_mass().unwrap();
    let Representation::Solid(local) = &mut view.objects_mut()[0].representation else {
        unreachable!()
    };
    for p in &mut local.nodes.position {
        *p += Vec3::Y;
    }
    world.commit_subset(&view, &[b.0 as usize]);
    assert_eq!(world.object(a).centre_of_mass().unwrap(), untouched);
    assert!((world.object(b).centre_of_mass().unwrap().y - 1.0).abs() < 1e-12);
}
