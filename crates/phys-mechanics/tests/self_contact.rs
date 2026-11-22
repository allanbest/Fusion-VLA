use core_math::{vec3, Pose, Vec3};
use core_runtime::Pool;
use core_space::closest_on_triangle;
use core_world::{Material, ObjectId, Representation, TetMesh, World};
use phys_mechanics::{Integrator, Mechanics, Settings};

/// Two separated pieces stored as one mesh/object. They must interact just
/// as they would if authored as two objects; object grouping is not physics.
fn approaching_parts(quadratic: bool) -> (World, ObjectId, usize) {
    let part = TetMesh::cuboid(vec3(0.4, 0.12, 0.4), 0.4);
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
    let material = world.add_material(Material::new("elastic", 1000.0, 1e5, 0.3, 0.0, 0.0));
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

fn vertical_gap(world: &World, id: ObjectId, first: usize) -> f64 {
    let Representation::Solid(solid) = &world.object(id).representation else {
        unreachable!()
    };
    let lower = solid.nodes.position[..first]
        .iter()
        .map(|p| p.y)
        .fold(f64::NEG_INFINITY, f64::max);
    let upper = solid.nodes.position[first..]
        .iter()
        .map(|p| p.y)
        .fold(f64::INFINITY, f64::min);
    upper - lower
}

/// Detect proper edge/face crossings between the two components. Their
/// axis-aligned bounds can overlap harmlessly once the slabs bend.
fn surfaces_intersect(world: &World, id: ObjectId, first: usize) -> bool {
    let Representation::Solid(solid) = &world.object(id).representation else {
        unreachable!()
    };
    for &[i, j] in &solid.surface.edges {
        for face in &solid.surface.faces {
            if face.contains(&i)
                || face.contains(&j)
                || (first != 0 && (i < first as u32) == (face[0] < first as u32))
            {
                continue;
            }
            let [a, b, c] = face.map(|n| solid.nodes.position[n as usize]);
            let [p, q] = [i, j].map(|n| solid.nodes.position[n as usize]);
            let normal = (b - a).cross(c - a);
            let d0 = normal.dot(p - a);
            let d1 = normal.dot(q - a);
            if d0 * d1 < 0.0 {
                let point = p + (q - p) * (d0 / (d0 - d1));
                let (closest, _) = closest_on_triangle(point, a, b, c);
                if (point - closest).length() < 1e-10 {
                    return true;
                }
            }
        }
    }
    false
}

#[test]
fn separated_parts_of_one_solid_do_not_pass_through_each_other() {
    for quadratic in [false, true] {
        let (mut world, id, first) = approaching_parts(quadratic);
        let pool = Pool::new(2);
        let mut solver = Mechanics::new(Settings {
            contact_precision: Some(1e-3),
            ..Default::default()
        });
        for step in 0..24 {
            let report = solver.step(&pool, &mut world, 1.0 / 120.0);
            assert!(
                !report.failed(),
                "quadratic={quadratic}, step {step}: {report:?}"
            );
            assert!(
                !surfaces_intersect(&world, id, first),
                "quadratic={quadratic}, step {step}: surfaces crossed (bounds gap {})",
                vertical_gap(&world, id, first)
            );
        }
    }
}

#[test]
fn disabling_self_contact_reproduces_the_missing_collision() {
    let (mut world, id, first) = approaching_parts(false);
    let pool = Pool::new(1);
    let mut solver = Mechanics::new(Settings {
        self_contact: false,
        ..Default::default()
    });
    let mut crossed = false;
    for _ in 0..24 {
        assert!(!solver.step(&pool, &mut world, 1.0 / 120.0).failed());
        crossed |= surfaces_intersect(&world, id, first);
    }
    assert!(crossed, "control never exercised surface interpenetration");
}

#[test]
fn self_contact_momentum_energy_and_replay() {
    for integrator in [Integrator::BackwardEuler, Integrator::TrBdf2] {
        let mut expected = None;
        for threads in [1, 4] {
            let (mut world, id, first) = approaching_parts(true);
            let pool = Pool::new(threads);
            let before = world.totals();
            let mut solver = Mechanics::new(Settings {
                contact_precision: Some(1e-3),
                integrator,
                tolerance: 1e-6,
                ..Default::default()
            });
            for _ in 0..8 {
                assert!(!solver.step(&pool, &mut world, 1.0 / 120.0).failed());
            }
            let saved = solver.checkpoint(&world);
            for _ in 0..16 {
                assert!(!solver.step(&pool, &mut world, 1.0 / 120.0).failed());
                assert!(!surfaces_intersect(&world, id, first));
            }
            let bytes = solver.checkpoint(&world).to_bytes();
            if let Some(expected) = &expected {
                assert_eq!(&bytes, expected);
            } else {
                expected = Some(bytes.clone());
            }
            let after = world.totals();
            assert!(
                (after.momentum - before.momentum).length() < 1e-5,
                "{integrator:?}: linear momentum drift {:?}",
                after.momentum - before.momentum
            );
            // An implicit Euler impact stage dissipates angular momentum;
            // temporal refinement is checked separately by the accuracy bench.
            assert!(
                (after.angular_momentum - before.angular_momentum).length() < 0.003,
                "{integrator:?}: angular momentum drift {:?}",
                after.angular_momentum - before.angular_momentum
            );
            let energy = after.kinetic_energy + solver.strain_energy(&pool, &world);
            assert!(
                energy < before.kinetic_energy * 1.02,
                "unforced contact added energy: {energy}"
            );
            solver.restore_checkpoint(&mut world, &saved);
            for _ in 0..16 {
                assert!(!solver.step(&pool, &mut world, 1.0 / 120.0).failed());
            }
            assert_eq!(solver.checkpoint(&world).to_bytes(), bytes);
        }
    }
}

#[test]
fn self_contact_gradient_matches_its_energy() {
    for quadratic in [false, true] {
        let (mut world, id, first) = approaching_parts(quadratic);
        let Representation::Solid(solid) = &mut world.objects_mut()[id.0 as usize].representation
        else {
            unreachable!()
        };
        for (i, p) in solid.nodes.position.iter_mut().enumerate() {
            p.y += if i < first { 0.0998 } else { -0.0998 };
        }
        let pool = Pool::new(2);
        let mut solver = Mechanics::new(Settings {
            contact_precision: Some(1e-3),
            ..Default::default()
        });
        let mut rng = core_math::Rng::new(2828);
        // Stay away from closest-feature switches on the initially aligned faces.
        let shifts: Vec<_> = (0..solid.nodes.len())
            .map(|_| rng.direction() * 3e-5)
            .collect();
        let (error, energy) = solver.gradient_check(&pool, &world, 1.0 / 120.0, |b| shifts[b]);
        assert!(
            energy.is_finite() && error < 1e-4,
            "quadratic={quadratic}: {error}"
        );
    }
}

/// A single connected U, with its base held and its arms closing inward.
fn closing_u() -> (World, ObjectId) {
    let mut mesh = TetMesh::cuboid(vec3(0.6, 0.8, 0.2), 0.2);
    mesh.tets.retain(|t| {
        let centre = t.iter().map(|&n| mesh.rest[n as usize]).sum::<Vec3>() / 4.0;
        centre.x.abs() > 0.1 || centre.y < -0.2
    });
    let mut remap = vec![u32::MAX; mesh.rest.len()];
    let mut rest = Vec::new();
    for t in &mut mesh.tets {
        for n in t {
            if remap[*n as usize] == u32::MAX {
                remap[*n as usize] = rest.len() as u32;
                rest.push(mesh.rest[*n as usize]);
            }
            *n = remap[*n as usize];
        }
    }
    mesh.rest = rest;
    let mut world = World::new();
    world.gravity = Vec3::ZERO;
    let mat = world.add_material(Material::new("soft", 1000.0, 1e3, 0.3, 0.0, 0.0));
    let id = world.add_solid("connected U", mat, mesh.quadratic(), Pose::IDENTITY);
    world.anchor(id, |p| p.y < -0.3);
    let Representation::Solid(s) = &mut world.objects_mut()[id.0 as usize].representation else {
        unreachable!()
    };
    for (p, v) in s.nodes.position.iter().zip(&mut s.nodes.velocity) {
        *v = Vec3::X * (-p.x.signum() * 3.0 * ((p.y + 0.3) / 0.6).clamp(0.0, 1.0));
    }
    (world, id)
}

#[test]
fn connected_solid_arms_collide_instead_of_passing_through() {
    for enabled in [true, false] {
        let (mut world, id) = closing_u();
        assert!(!surfaces_intersect(&world, id, 0));
        let pool = Pool::new(2);
        let mut solver = Mechanics::new(Settings {
            self_contact: enabled,
            contact_precision: Some(1e-3),
            ..Default::default()
        });
        let mut crossed = false;
        for step in 0..30 {
            let report = solver.step(&pool, &mut world, 1.0 / 120.0);
            assert!(
                !report.failed(),
                "enabled={enabled}, step {step}: {report:?}"
            );
            crossed |= surfaces_intersect(&world, id, 0);
            if crossed {
                break;
            }
        }
        assert_eq!(
            crossed, !enabled,
            "enabled={enabled}: control must exercise actual surface intersection"
        );
    }
}

#[test]
fn coincident_nonincident_self_surfaces_reject_without_advancing() {
    let (mut world, id, first) = approaching_parts(true);
    let Representation::Solid(s) = &mut world.objects_mut()[id.0 as usize].representation else {
        unreachable!()
    };
    for (i, p) in s.nodes.position.iter_mut().enumerate() {
        p.y += if i < first { 0.1 } else { -0.1 };
    }
    let pool = Pool::new(2);
    let mut solver = Mechanics::new(Settings {
        contact_precision: Some(1e-3),
        ..Default::default()
    });
    solver.prepare(&pool, &world);
    let before = solver.checkpoint(&world).to_bytes();
    let report = solver.step(&pool, &mut world, 1.0 / 120.0);
    assert!(report.failed() && report.invalid, "{report:?}");
    assert_eq!(solver.checkpoint(&world).to_bytes(), before);
}
