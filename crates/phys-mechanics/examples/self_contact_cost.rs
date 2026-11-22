//! Contact-rich deformable benchmark; no rendering or training output.
use core_math::{vec3, Pose, Vec3};
use core_runtime::Pool;
use core_world::{Material, ObjectId, Representation, TetMesh, World};
use phys_mechanics::{Mechanics, Settings};

/// Two separated pieces stored as one mesh/object. They must interact just
/// as they would if authored as two objects; object grouping is not physics.
fn approaching_parts(quadratic: bool) -> (World, ObjectId, usize) {
    let part = TetMesh::cuboid(vec3(0.4, 0.12, 0.4), 0.1);
    let part = if quadratic { part.quadratic() } else { part };
    let first = part.rest.len();
    let mut mesh = TetMesh::default();
    for shift in [-0.16, 0.16] {
        let offset = mesh.rest.len() as u32;
        mesh.rest
            .extend(part.rest.iter().map(|p| *p + Vec3::Y * shift));
        mesh.tets
            .extend(part.tets.iter().map(|t| t.map(|n| n + offset)));
        mesh.midpoints
            .extend(part.midpoints.iter().map(|t| t.map(|n| n + offset)));
    }
    let mut world = World::new();
    world.gravity = Vec3::ZERO;
    let material = world.add_material(Material::new("elastic", 1000.0, 1e5, 0.3, 0.3, 0.0));
    let id = world.add_solid("two parts, one mesh", material, mesh, Pose::IDENTITY);
    let Representation::Solid(solid) = &mut world.objects_mut()[id.0 as usize].representation
    else {
        unreachable!()
    };
    for (i, v) in solid.nodes.velocity.iter_mut().enumerate() {
        *v = if i < first { Vec3::Y } else { -Vec3::Y };
    }
    (world, id, first)
}

fn main() {
    let (mut world, id, _) = approaching_parts(true);
    let threads = std::env::var("SELF_THREADS")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(2);
    let steps = std::env::var("SELF_STEPS")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(24);
    let pool = Pool::new(threads);
    let mut solver = Mechanics::new(Settings {
        contact_precision: Some(1e-3),
        ..Default::default()
    });
    solver.prepare(&pool, &world);
    let (mut newton, mut cg, mut splits, mut impacts, mut failed) = (0, 0, 0, 0, 0);
    let start = std::time::Instant::now();
    for _ in 0..steps {
        let r = solver.step(&pool, &mut world, 1.0 / 120.0);
        newton += r.iterations;
        cg += r.linear_iterations;
        splits += r.splits;
        impacts += r.impacts;
        failed += usize::from(r.failed());
    }
    let elapsed = start.elapsed().as_secs_f64();
    let Representation::Solid(s) = &world.object(id).representation else {
        unreachable!()
    };
    println!("{} nodes, {threads} threads, {steps} steps: {newton} Newton, {cg} CG, {splits} splits, {impacts} impacts, {failed} failed, {elapsed:.6} s", s.nodes.len());
    assert_eq!(failed, 0, "benchmark contains rejected steps");
    if let Some(path) = std::env::var_os("SELF_CHECKPOINT") {
        std::fs::write(path, solver.checkpoint(&world).to_bytes()).unwrap();
    }
}
