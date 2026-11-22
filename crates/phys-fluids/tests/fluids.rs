use core_math::{vec3, Pose, Vec3};
use core_runtime::Pool;
use core_space::{Aabb, Shape};
use core_world::{Material, Representation, World};
use phys_fluids::{FluidParticles, FluidSettings, FluidSolver};

#[test]
fn hydrostatic_column_settles_with_pressure_gradient_and_thread_determinism() {
    let run = |threads: usize| {
        let pool = Pool::new(threads);
        let spacing = 0.04;
        let domain = Aabb::new(Vec3::ZERO, vec3(0.32, 0.5, 0.32));
        let block = Aabb::new(vec3(0.02, 0.02, 0.02), vec3(0.30, 0.22, 0.30));
        let mut fluid = FluidParticles::block(block, spacing, 1000.0);
        let mut solver = FluidSolver::new(FluidSettings {
            rest_density: 1000.0,
            smoothing_radius: 0.088,
            sound_speed: 18.0,
            viscosity: 0.05,
            xsph: 0.08,
            domain: Some(domain),
            ..FluidSettings::default()
        });
        let gravity = vec3(0.0, -9.81, 0.0);
        let mut last = Default::default();
        for _ in 0..80 {
            last = solver.step(&pool, &mut fluid, gravity, 0.005);
        }
        assert!(last.max_density > 990.0 && last.max_density < 1200.0);
        assert!(last.max_pressure > 0.0);
        // Bottom half pressure exceeds top half pressure (hydrostatic gradient).
        let mid_y = 0.12;
        let (mut sum_bot, mut cnt_bot, mut sum_top, mut cnt_top) = (0.0, 0, 0.0, 0);
        for (&p, &pr) in fluid.positions.iter().zip(&fluid.pressures) {
            if p.y < mid_y {
                sum_bot += pr;
                cnt_bot += 1;
            } else {
                sum_top += pr;
                cnt_top += 1;
            }
        }
        assert!(cnt_bot > 0 && cnt_top > 0);
        assert!(
            sum_bot / cnt_bot as f64 > sum_top / cnt_top as f64,
            "expected bottom pressure > top pressure"
        );
        fluid
    };

    let f1 = run(1);
    let f4 = run(4);
    assert_eq!(f1.positions, f4.positions);
    assert_eq!(f1.velocities, f4.velocities);
}

#[test]
fn two_way_fsi_buoyancy_supports_light_rigid_body() {
    let pool = Pool::new(2);
    let spacing = 0.04;
    let domain = Aabb::new(Vec3::ZERO, vec3(0.36, 0.5, 0.36));
    let pool_bounds = Aabb::new(vec3(0.02, 0.02, 0.02), vec3(0.34, 0.22, 0.34));
    let mut fluid = FluidParticles::block(pool_bounds, spacing, 1000.0);
    let mut solver = FluidSolver::new(FluidSettings {
        rest_density: 1000.0,
        smoothing_radius: 0.09,
        sound_speed: 20.0,
        viscosity: 0.04,
        xsph: 0.05,
        domain: Some(domain),
        fsi_stiffness: 1.0,
        ..FluidSettings::default()
    });

    // First settle the water pool slightly.
    for _ in 0..30 {
        solver.step(&pool, &mut fluid, vec3(0.0, -9.81, 0.0), 0.004);
    }

    // Drop a half-density wooden box (ρ = 400 kg/m³ < 1000 kg/m³) onto the water surface.
    let mut world = World::new();
    let wood = world.add_material(Material::new("wood", 400.0, 1e6, 0.3, 0.4, 0.0));
    let box_id = world.add_rigid(
        "float",
        wood,
        Shape::cuboid(vec3(0.14, 0.14, 0.14)),
        Pose::at(vec3(0.18, 0.24, 0.18)),
    );

    let dt = 0.004;
    let grav = world.gravity;
    for _ in 0..120 {
        // Explicit gravity on the rigid body staggered with two-way FSI impulse.
        if let Representation::Rigid(r) = &mut world.object_mut(box_id).representation {
            r.velocity += grav * dt;
            r.pose.position += r.velocity * dt;
        }
        solver.step_coupled(&pool, &mut fluid, &mut world, dt);
    }

    let Representation::Rigid(r) = &world.object(box_id).representation else {
        unreachable!()
    };
    assert!(
        r.pose.position.y > 0.11,
        "half-density box sank to the floor (y = {}), expected buoyancy",
        r.pose.position.y
    );
    assert!(
        r.pose.position.y < 0.35,
        "box shot out of water unrealistically (y = {})",
        r.pose.position.y
    );
}
