//! Size-weighted solve work, wall time and isolated-motion error. No app,
//! dataset generation or timing-based pass/fail threshold is involved.
use core_math::{vec3, Pose, Vec3};
use core_runtime::Pool;
use core_space::Shape;
use core_world::{Material, ObjectId, Representation, TetMesh, World};
use phys_mechanics::{Mechanics, Settings};
use std::time::Instant;

fn scene(copies: usize, contact: bool) -> (World, ObjectId) {
    let mut w = World::new();
    w.gravity = Vec3::ZERO;
    let m = w.add_material(Material::new("elastic", 1000.0, 1e5, 0.25, 0.4, 0.0));
    let mut first = None;
    for k in 0..copies {
        let centre = vec3(k as f64 * 2.0, 2.0, 0.0);
        let id = w.add_solid(
            "oscillator",
            m,
            TetMesh::cuboid(Vec3::splat(0.2), 0.05).quadratic(),
            Pose::at(centre),
        );
        first.get_or_insert(id);
        let Representation::Solid(s) = &mut w.object_mut(id).representation else {
            unreachable!()
        };
        for (p, v) in s.nodes.position.iter_mut().zip(&mut s.nodes.velocity) {
            p.x = centre.x + (p.x - centre.x) * 1.002;
            v.x = 0.03 * ((p.x - centre.x) * std::f64::consts::PI / 0.2).sin();
        }
    }
    if contact {
        let x = -10.0;
        w.add_fixed("support", m, Shape::ground(0.0), Pose::IDENTITY);
        let id = w.add_rigid(
            "striker",
            m,
            Shape::cuboid(Vec3::splat(0.2)),
            Pose::at(vec3(x, 0.15, 0.0)),
        );
        w.set_velocity(id, vec3(1.0, -2.0, 0.0), Vec3::ZERO);
    }
    (w, first.unwrap())
}

fn main() -> Result<(), String> {
    let threads = std::env::var("ISLAND_THREADS")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(1);
    let pool = Pool::new(threads);
    let h = 1.0 / 120.0;
    let steps = 30;
    let settings = Settings {
        contact_precision: Some(1e-4),
        ..Default::default()
    };
    let (mut alone, ai) = scene(1, false);
    let mut reference = Mechanics::new(settings);
    reference.profile.set_enabled(false);
    let mut trajectory = Vec::new();
    for _ in 0..steps {
        let r = reference.step(&pool, &mut alone, h);
        if r.failed() {
            return Err(format!("reference: {r:?}"));
        }
        let Representation::Solid(s) = &alone.object(ai).representation else {
            unreachable!()
        };
        trajectory.push(s.nodes.position.clone());
    }
    println!("mode,copies,threads,steps,total_blocks,largest_solve_blocks,max_islands,partition_retries,newton_iterations,cg_iterations,cg_block_iterations,max_isolated_position_error_m,seconds");
    let modes = if std::env::var_os("ISLAND_FIRST").is_some() {
        [true, false]
    } else {
        [false, true]
    };
    for copies in [1, 8, 24] {
        for islands in modes {
            let (mut w, id) = scene(copies, true);
            let mut s = Mechanics::new(Settings {
                islands,
                ..settings
            });
            s.profile.set_enabled(false);
            s.prepare(&pool, &w);
            let (mut newton, mut cg, mut work, mut retries, mut groups, mut largest) =
                (0, 0, 0, 0, 0, 0);
            let (mut seconds, mut error) = (0.0, 0.0_f64);
            for expected in &trajectory {
                let start = Instant::now();
                let r = s.step(&pool, &mut w, h);
                seconds += start.elapsed().as_secs_f64();
                if r.failed() {
                    return Err(format!("copies={copies}, islands={islands}: {r:?}"));
                }
                newton += r.iterations;
                cg += r.linear_iterations;
                work += r.linear_block_iterations;
                retries += r.island_rebuilds;
                groups = groups.max(r.islands);
                largest = largest.max(r.largest_island_blocks);
                let Representation::Solid(solid) = &w.object(id).representation else {
                    unreachable!()
                };
                for (&p, &q) in solid.nodes.position.iter().zip(expected) {
                    error = error.max((p - q).length());
                }
            }
            println!("{},{copies},{threads},{steps},{},{largest},{groups},{retries},{newton},{cg},{work},{error:.12e},{seconds:.6}",
                if islands { "islands" } else { "coupled" }, s.integration_shape().0);
        }
    }
    Ok(())
}
