//! Accuracy and compute tradeoffs. All timing is stepping time, not preparation,
//! measurement/output or pool construction. No training or editor code.
mod support {
    #[allow(dead_code)]
    pub mod accuracy;
    pub mod efficiency;
}
use core_math::{Pose, Vec3};
use core_runtime::Pool;
use core_space::Shape;
use core_world::{library, World};
use phys_mechanics::{Integrator, Mechanics, Settings};
use support::{accuracy, efficiency};
fn ordered<T, const N: usize>(mut values: [T; N]) -> [T; N] {
    if std::env::var_os("FUSION_BENCH_REVERSE").is_some() {
        values.reverse();
    }
    values
}
fn main() -> Result<(), String> {
    println!("kind,mode,edge_or_dt,threads,copies,energy_ratio,frequency_error,angular_error,newton,cg,matrix_products,refinements_or_sleeps,seconds");
    for dt in [1.0 / 60.0, 1.0 / 240.0] {
        for startup in ordered([false, true]) {
            let r = accuracy::longitudinal_options(0.1, dt, 1, startup, 1)?;
            println!(
                "vibration,{startup},{dt},1,1,{:.9e},{:.9e},0,{},{},{},{},{:.6}",
                r.energy_ratio,
                r.relative_error(),
                r.newton,
                r.cg,
                r.products,
                r.accuracy_retries,
                r.seconds
            );
        }
    }
    for angular in ordered([None, Some(1e-4)]) {
        let r = accuracy::self_impact_options(
            1.0 / 60.0,
            Integrator::TrBdf2,
            1,
            Vec3::ZERO,
            angular,
            1e-7,
        )?;
        println!(
            "contact,{},{},1,1,{:.9e},0,{:.9e},{},{},{},{},{:.6}",
            if angular.is_some() {
                "refined"
            } else {
                "unchecked"
            },
            1.0 / 60.0,
            r.energy_ratio,
            r.angular_error,
            r.newton,
            r.cg,
            r.products,
            r.accuracy_retries,
            r.seconds
        );
    }
    for edge in [0.075, 0.05] {
        for patch in ordered([1, 4, 8]) {
            let c = efficiency::run(
                efficiency::assembly(1, edge),
                Settings {
                    solid_block_size: patch,
                    ..Settings::default()
                },
                1,
                6,
            )?;
            println!(
                "patch,{patch},{edge},1,1,0,0,0,{},{},{},0,{:.6}",
                c.newton, c.cg, c.products, c.seconds
            );
        }
    }
    let w = efficiency::assembly(8, 0.075);
    let mut reference = None;
    for parallel in ordered([false, true]) {
        let c = efficiency::run(
            w.clone(),
            Settings {
                parallel_islands: parallel,
                ..Settings::default()
            },
            4,
            6,
        )?;
        if let Some(ref expected) = reference {
            if *expected != c.bytes {
                return Err("parallel state/history differs".into());
            }
        } else {
            reference = Some(c.bytes.clone());
        }
        println!(
            "parallel,{parallel},0.075,4,8,0,0,0,{},{},{},0,{:.6}",
            c.newton, c.cg, c.products, c.seconds
        );
    }
    // Exact stationary-state reuse, compared against the identical active map.
    let mut w = World::new();
    w.gravity = Vec3::ZERO;
    let m = w.add_material(library::rubber());
    for i in 0..128 {
        w.add_rigid(
            "rest",
            m,
            Shape::Sphere { radius: 0.1 },
            Pose::at(Vec3::X * i as f64),
        );
    }
    let pool = Pool::new(4);
    let mut previous = None;
    for sleeping in ordered([false, true]) {
        let mut world = w.clone();
        let mut s = Mechanics::new(Settings {
            sleeping,
            ..Settings::default()
        });
        s.profile.set_enabled(false);
        let mut seconds = 0.0;
        let (mut newton, mut cg, mut products, mut slept) = (0, 0, 0, 0);
        for _ in 0..60 {
            let start = std::time::Instant::now();
            let r = s.step(&pool, &mut world, 0.01);
            seconds += start.elapsed().as_secs_f64();
            if r.failed() {
                return Err(format!("{r:?}"));
            }
            newton += r.iterations;
            cg += r.linear_iterations;
            products += r.linear_products;
            slept += r.sleeping_islands;
        }
        let state = s.checkpoint(&world).to_bytes();
        if let Some(ref expected) = previous {
            if *expected != state {
                return Err("sleeping state/history differs".into());
            }
        } else {
            previous = Some(state);
        }
        println!("sleep,{sleeping},0.01,4,128,0,0,0,{newton},{cg},{products},{slept},{seconds:.6}");
    }
    Ok(())
}
