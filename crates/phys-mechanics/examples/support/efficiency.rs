use core_math::{vec3, Pose, Vec3};
use core_runtime::Pool;
use core_world::{Material, Representation, TetMesh, World};
use phys_mechanics::{Mechanics, Settings};
use std::time::Instant;

pub fn assembly(copies: usize, edge: f64) -> World {
    let mut w = World::new();
    w.gravity = vec3(0.0, -0.5, 0.0);
    let m = w.add_material(Material::new("stiff elastic", 1000.0, 1e7, 0.35, 0.0, 0.0));
    for k in 0..copies {
        let centre = Vec3::X * (k as f64 * 2.0);
        let id = w.add_solid(
            "clamped",
            m,
            TetMesh::cuboid(vec3(0.5, 0.3, 0.25), edge).quadratic(),
            Pose::at(centre),
        );
        w.anchor(id, |p| p.x < centre.x - 0.25 + 1e-8);
        let Representation::Solid(s) = &mut w.object_mut(id).representation else {
            unreachable!()
        };
        for (p, v) in s.nodes.position.iter_mut().zip(&mut s.nodes.velocity) {
            let t = (p.x - centre.x + 0.25) / 0.5;
            p.y += 0.001 * t * t;
            v.z = 0.002 * t * t;
        }
    }
    w
}

pub struct Cost {
    pub newton: usize,
    pub cg: usize,
    pub products: usize,
    pub seconds: f64,
    pub bytes: Vec<u8>,
}
pub fn run(
    mut world: World,
    settings: Settings,
    threads: usize,
    steps: usize,
) -> Result<Cost, String> {
    let pool = Pool::new(threads);
    let mut s = Mechanics::new(settings);
    s.profile.set_enabled(false);
    s.prepare(&pool, &world);
    let mut cost = Cost {
        newton: 0,
        cg: 0,
        products: 0,
        seconds: 0.0,
        bytes: Vec::new(),
    };
    for _ in 0..steps {
        let start = Instant::now();
        let r = s.step(&pool, &mut world, 0.01);
        cost.seconds += start.elapsed().as_secs_f64();
        if r.failed() {
            return Err(format!("{r:?}"));
        }
        cost.newton += r.iterations;
        cost.cg += r.linear_iterations;
        cost.products += r.linear_products;
    }
    cost.bytes = s.checkpoint(&world).to_bytes();
    Ok(cost)
}
