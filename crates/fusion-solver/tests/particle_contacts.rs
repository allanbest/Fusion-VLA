use fusion_core::{BodyDesc, ParticleSet, World};
use fusion_math::{vec3, Vec3};
use fusion_solver::soft::solve_particle_body_contacts;

fn solve(p: &mut ParticleSet, w: &mut World) {
    let mut dp = vec![Vec3::ZERO; p.len()];
    let mut db = vec![Vec3::ZERO; w.bodies.len()];
    let mut dr = db.clone();
    solve_particle_body_contacts(
        p,
        &mut w.bodies,
        &mut dp,
        &mut db,
        &mut dr,
        0.001,
        0.0,
        1.0 / 1920.0,
    );
}

#[test]
fn later_particles_see_the_corrected_body_pose() {
    let mut w = World::new();
    w.add(BodyDesc::sphere(1.0).with_mass(1.0));
    let mut p = ParticleSet::new();
    p.push(vec3(0.9, 0.0, 0.0), 1.0, 0.1);
    p.push(vec3(0.9, 0.0, 0.0), 1.0, 0.1);
    solve(&mut p, &mut w);
    assert!(
        (w.bodies.position[0].x + 0.15).abs() < 1e-6,
        "{:?}",
        w.bodies.position[0]
    );
    assert!((p.position[1].x - 0.95).abs() < 1e-6);
}

#[test]
fn a_particle_impact_wakes_a_sleeping_dynamic_body() {
    let mut w = World::new();
    w.add(BodyDesc::sphere(1.0).with_mass(1.0));
    w.bodies.sleeping[0] = true;
    let mut p = ParticleSet::new();
    p.push(vec3(0.9, 0.0, 0.0), 1.0, 0.1);
    p.velocity[0] = -Vec3::X;
    solve(&mut p, &mut w);
    assert!(!w.bodies.sleeping[0]);
    assert!(w.bodies.position[0].x < 0.0);
}

#[test]
fn particle_contacts_are_translation_invariant_at_large_coordinates() {
    let run = |offset: f64| {
        let mut w = World::new();
        w.add(BodyDesc::cuboid(Vec3::ONE).fixed());
        w.bodies.position[0].x = offset;
        let mut p = ParticleSet::new();
        p.push(Vec3::ZERO, 1.0, 0.01);
        p.position[0].x = offset + 0.509;
        solve(&mut p, &mut w);
        p.position[0].x - offset
    };
    assert!((run(0.0) - run(100_000.0)).abs() < 1e-6);
}
