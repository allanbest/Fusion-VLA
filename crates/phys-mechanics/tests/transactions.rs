use core_math::{vec3, Pose, Vec3};
use core_runtime::Pool;
use core_space::Shape;
use core_world::{library, World};
use phys_mechanics::{Mechanics, MechanicsState, Settings};
fn scene() -> (World, Mechanics, Pool) {
    let pool = Pool::new(1);
    let mut w = World::new();
    let m = w.add_material(library::rubber());
    w.add_rigid(
        "ball",
        m,
        Shape::Sphere { radius: 0.1 },
        Pose::at(vec3(0.0, 2.0, 0.0)),
    );
    let mut s = Mechanics::new(Settings::default());
    s.prepare(&pool, &w);
    (w, s, pool)
}
#[test]
fn exhausted_retries_preserve_state_and_history() {
    let (mut w, mut s, pool) = scene();
    assert!(!s.step(&pool, &mut w, 1.0 / 60.0).failed());
    let before = s.checkpoint(&w).to_bytes();
    s.settings.max_iterations = 0;
    let r = s.step(&pool, &mut w, 1.0 / 60.0);
    assert!(r.failed());
    assert!(r.splits > 0);
    assert!(r.solves >= 5);
    assert_eq!(s.checkpoint(&w).to_bytes(), before);
    s.settings.max_iterations = 30;
    assert!(!s.step(&pool, &mut w, 1.0 / 60.0).failed());
}
#[test]
fn invalid_intervals_are_atomic() {
    let (mut w, mut s, pool) = scene();
    let saved = s.checkpoint(&w).to_bytes();
    for h in [0.0, -1.0, f64::NAN, f64::INFINITY] {
        assert!(s.step(&pool, &mut w, h).failed());
        assert_eq!(s.checkpoint(&w).to_bytes(), saved);
    }
}
#[test]
fn portable_history_replays_exactly() {
    let (mut w, mut s, pool) = scene();
    for _ in 0..5 {
        assert!(!s.step(&pool, &mut w, 1.0 / 60.0).failed());
    }
    let saved = s.checkpoint(&w);
    let bytes = saved.to_bytes();
    assert_eq!(saved.encoded_len(), bytes.len());
    assert!(!s.step(&pool, &mut w, 1.0 / 60.0).failed());
    let expected = s.checkpoint(&w).to_bytes();
    let state = MechanicsState::from_bytes(&bytes, &saved).unwrap();
    s.restore_checkpoint(&mut w, &state);
    assert!(!s.step(&pool, &mut w, 1.0 / 60.0).failed());
    assert_eq!(s.checkpoint(&w).to_bytes(), expected);
    assert!(MechanicsState::from_bytes(&bytes[..8], &saved).is_err());
}
#[test]
fn overlapping_initial_state_does_not_advance() {
    let (mut w, mut s, pool) = scene();
    let m = w.add_material(library::concrete());
    w.add_fixed(
        "overlap",
        m,
        Shape::cuboid(Vec3::ONE),
        Pose::at(vec3(0.0, 2.0, 0.0)),
    );
    s.prepare(&pool, &w);
    let before = s.checkpoint(&w).to_bytes();
    assert!(s.step(&pool, &mut w, 1.0 / 60.0).failed());
    assert_eq!(s.checkpoint(&w).to_bytes(), before);
}

#[test]
fn displayed_deformation_reproduces_an_affine_field_on_quadratic_elements() {
    use core_math::Mat3;
    use core_world::{Solid, TetMesh};
    let mut s = Solid::new(
        TetMesh::cuboid(Vec3::ONE, 0.5).quadratic(),
        &Pose::default(),
        1000.0,
    );
    let f = Mat3::from_cols(vec3(1.1, 0.1, 0.0), vec3(0.0, 0.9, 0.0), Vec3::Z);
    for p in &mut s.nodes.position {
        *p = f * *p + vec3(3.0, -2.0, 1.0);
    }
    for actual in phys_mechanics::nodal_deformation(&s) {
        for i in 0..3 {
            assert!((actual.cols[i] - f.cols[i]).length() < 1e-12);
        }
    }
}

/// A deformable solid's step replays exactly from a checkpoint: what its
/// elements cache to speed one solve must not leak into the next.
#[test]
fn deformable_history_replays_exactly() {
    let pool = Pool::new(2);
    let mut w = World::new();
    let floor = w.add_material(library::concrete());
    let gel = w.add_material(library::gelatin());
    w.add_fixed(
        "floor",
        floor,
        Shape::cuboid(vec3(4.0, 1.0, 4.0)),
        Pose::at(vec3(0.0, -0.5, 0.0)),
    );
    w.add_solid(
        "jelly",
        gel,
        core_world::TetMesh::cuboid(Vec3::splat(0.2), 0.1).quadratic(),
        Pose::at(vec3(0.0, 0.12, 0.0)),
    );
    let mut s = Mechanics::new(Settings::default());
    s.prepare(&pool, &w);
    let fresh = s.checkpoint(&w);
    assert_eq!(fresh.encoded_len(), fresh.to_bytes().len());
    for _ in 0..8 {
        assert!(!s.step(&pool, &mut w, 1.0 / 60.0).failed());
    }
    let saved = s.checkpoint(&w);
    assert_eq!(saved.encoded_len(), saved.to_bytes().len());
    assert!(!s.step(&pool, &mut w, 1.0 / 60.0).failed());
    assert!(!s.step(&pool, &mut w, 1.0 / 60.0).failed());
    let expected = s.checkpoint(&w).to_bytes();
    s.restore_checkpoint(&mut w, &saved);
    assert!(!s.step(&pool, &mut w, 1.0 / 60.0).failed());
    assert!(!s.step(&pool, &mut w, 1.0 / 60.0).failed());
    assert_eq!(s.checkpoint(&w).to_bytes(), expected);
}

#[test]
fn cached_energy_matches_reference_and_preserves_history() {
    use core_world::TetMesh;
    let (mut world, mut mechanics, pool) = scene();
    let m = world.add_material(library::rubber());
    let id = world.add_solid(
        "solid",
        m,
        TetMesh::cuboid(Vec3::ONE, 0.5).quadratic(),
        Pose::at(vec3(3.0, 0.0, 0.0)),
    );
    let core_world::Representation::Solid(s) = &mut world.object_mut(id).representation else {
        unreachable!()
    };
    for p in &mut s.nodes.position {
        p.y *= 1.05;
    }
    mechanics.prepare(&pool, &world);
    let before = mechanics.checkpoint(&world).to_bytes();
    let reference = phys_mechanics::strain_energy(&world);
    for _ in 0..3 {
        assert_eq!(mechanics.strain_energy(&pool, &world), reference);
    }
    assert_eq!(mechanics.checkpoint(&world).to_bytes(), before);
}
