use core_math::constants::ROOM_TEMPERATURE;
use core_math::{vec3, Pose, Vec3};
use core_runtime::Pool;
use core_world::{library, Column, Material, Property, Representation, TetMesh, Thermal, World};
use phys_mechanics::{Integrator, Mechanics, Settings};
use phys_thermal::{HeatConduction, Process, ThermalBoundary, ThermalSettings};

#[test]
fn uniform_heating_produces_stress_free_isotropic_thermal_expansion() {
    for &quadratic in &[false, true] {
        let mut world = World::new();
        world.gravity = Vec3::ZERO;
        let alpha = 2.0e-4_f64;
        let base = Material::new("test", 1000.0, 1e6, 0.25, 0.5, 0.0).with_thermal(Thermal {
            specific_heat: 900.0,
            conductivity: 50.0,
            expansion: alpha,
            emissivity: 0.5,
            reference_temperature: ROOM_TEMPERATURE,
            ..Thermal::default()
        });
        let mat = world.add_material(base);
        let size = vec3(0.4, 0.1, 0.1);
        let mesh = TetMesh::cuboid(size, 0.1);
        let mesh = if quadratic { mesh.quadratic() } else { mesh };
        let bar = world.add_solid("bar", mat, mesh, Pose::IDENTITY);
        let dt = 100.0_f64;
        {
            let Representation::Solid(s) = &mut world.object_mut(bar).representation else {
                unreachable!()
            };
            s.nodes
                .columns
                .ensure(Column::TEMPERATURE, ROOM_TEMPERATURE + dt);
        }

        let pool = Pool::new(2);
        let mut solver = Mechanics::new(Settings {
            integrator: Integrator::BackwardEuler,
            energy_accounting: true,
            ..Default::default()
        });
        for _ in 0..30 {
            let r = solver.step(&pool, &mut world, 0.01);
            assert!(!r.failed(), "quadratic={quadratic}: {r:?}");
        }
        let gamma = 1.0 + alpha * dt;
        let expected_half_x = 0.5 * size.x * gamma;
        let Representation::Solid(s) = &world.object(bar).representation else {
            unreachable!()
        };
        let min_x = s
            .nodes
            .position
            .iter()
            .map(|p| p.x)
            .fold(f64::INFINITY, f64::min);
        let max_x = s
            .nodes
            .position
            .iter()
            .map(|p| p.x)
            .fold(f64::NEG_INFINITY, f64::max);
        assert!(
            (max_x - expected_half_x).abs() < 5e-5 && (min_x + expected_half_x).abs() < 5e-5,
            "quadratic={quadratic}: expanded half-x {max_x} vs expected {expected_half_x}"
        );
        let residual_strain_energy = solver.strain_energy(&pool, &world);
        assert!(
            residual_strain_energy < 1e-8,
            "free thermal expansion should be stress-free, got {residual_strain_energy}"
        );
    }
}

#[test]
fn temperature_dependent_stiffness_and_constrained_gradient_match_finite_differences() {
    for &quadratic in &[false, true] {
        let mut world = World::new();
        let steel = library::structural_steel();
        let mat = world.add_material(Material {
            youngs: Property::Table(vec![
                (ROOM_TEMPERATURE, 2e6),
                (ROOM_TEMPERATURE + 200.0, 1e6),
            ]),
            poisson: Property::Constant(0.25),
            ..steel
        });
        let mesh = TetMesh::cuboid(vec3(0.4, 0.1, 0.1), 0.1);
        let mesh = if quadratic { mesh.quadratic() } else { mesh };
        let bar = world.add_solid("bar", mat, mesh, Pose::IDENTITY);
        world.anchor(bar, |p| p.x < -0.19);
        {
            let Representation::Solid(s) = &mut world.object_mut(bar).representation else {
                unreachable!()
            };
            let t = s
                .nodes
                .columns
                .ensure(Column::TEMPERATURE, ROOM_TEMPERATURE);
            for (i, p) in s.nodes.position.iter().enumerate() {
                t[i] = ROOM_TEMPERATURE + 150.0 * (p.x + 0.2) / 0.4;
            }
        }
        let pool = Pool::new(2);
        let mut solver = Mechanics::new(Settings::default());
        let (err, _) = solver.gradient_check(&pool, &world, 0.01, |b| {
            let k = (b % 5) as f64;
            vec3(2e-4 * (k + 1.0), -1e-4 * k, 1.5e-4)
        });
        assert!(
            err < 1e-7,
            "thermo-mechanical FD gradient failed (quadratic={quadratic}): {err}"
        );

        // Verify softening effect: anchored bar sagging under gravity deflects more when hot.
        let tip_y = |temp: f64| {
            let mut w = World::new();
            let steel = library::structural_steel();
            let m = w.add_material(Material {
                youngs: Property::Table(vec![
                    (ROOM_TEMPERATURE, 1e6),
                    (ROOM_TEMPERATURE + 200.0, 2e5),
                ]),
                poisson: Property::Constant(0.25),
                thermal: None,
                ..steel
            });
            let mut msh = TetMesh::cuboid(vec3(0.4, 0.1, 0.1), 0.1);
            if quadratic {
                msh = msh.quadratic();
            }
            let b = w.add_solid("beam", m, msh, Pose::IDENTITY);
            w.anchor(b, |p| p.x < -0.19);
            let Representation::Solid(s) = &mut w.object_mut(b).representation else {
                unreachable!()
            };
            s.nodes.columns.ensure(Column::TEMPERATURE, temp);
            let mut ms = Mechanics::new(Settings {
                integrator: Integrator::BackwardEuler,
                ..Default::default()
            });
            for _ in 0..20 {
                assert!(!ms.step(&pool, &mut w, 0.01).failed());
            }
            let Representation::Solid(s) = &w.object(b).representation else {
                unreachable!()
            };
            s.nodes
                .position
                .iter()
                .map(|p| p.y)
                .fold(f64::INFINITY, f64::min)
        };
        assert!(tip_y(ROOM_TEMPERATURE + 200.0) < tip_y(ROOM_TEMPERATURE) - 1e-3);
    }
}

#[test]
fn coupled_heat_conduction_and_thermal_expansion_staggers_cleanly() {
    let mut world = World::new();
    world.gravity = Vec3::ZERO;
    let alpha = 5e-4_f64;
    let mut base = library::aluminium();
    base.thermal = Some(Thermal {
        specific_heat: 900.0,
        conductivity: 5000.0, // fast diffusion for compact coupled test duration
        expansion: alpha,
        emissivity: 0.2,
        reference_temperature: ROOM_TEMPERATURE,
        ..Thermal::default()
    });
    base.youngs = Property::Constant(5e6);
    let mat = world.add_material(base);
    let length = 0.6_f64;
    let bar = world.add_solid(
        "bar",
        mat,
        TetMesh::cuboid(vec3(length, 0.1, 0.1), 0.1),
        Pose::at(vec3(0.5 * length, 0.0, 0.0)),
    );
    // Pin left face x = 0.0 and fix its temperature higher by +120 K.
    world.anchor(bar, |p| p.x < 1e-5);
    let left_nodes: Vec<u32> = {
        let Representation::Solid(s) = &mut world.object_mut(bar).representation else {
            unreachable!()
        };
        s.nodes
            .columns
            .ensure(Column::TEMPERATURE, ROOM_TEMPERATURE);
        s.nodes
            .position
            .iter()
            .enumerate()
            .filter(|(_, p)| p.x < 1e-5)
            .map(|(i, _)| i as u32)
            .collect()
    };

    let pool = Pool::new(2);
    let mut heat = HeatConduction::new(ThermalSettings {
        boundaries: vec![ThermalBoundary::Dirichlet {
            object: bar,
            nodes: left_nodes,
            temperature: ROOM_TEMPERATURE + 120.0,
        }],
        ..Default::default()
    });
    let mut mech = Mechanics::new(Settings {
        integrator: Integrator::BackwardEuler,
        ..Default::default()
    });

    let tip_x = |w: &World| {
        let Representation::Solid(s) = &w.object(bar).representation else {
            unreachable!()
        };
        s.nodes
            .position
            .iter()
            .map(|p| p.x)
            .fold(f64::NEG_INFINITY, f64::max)
    };

    let initial_tip = tip_x(&world);
    let mut mid_tip = initial_tip;
    for step in 0..25 {
        let tr = heat.step(&pool, &mut world, 10.0);
        assert!(tr.converged);
        let mr = mech.step(&pool, &mut world, 0.05);
        assert!(!mr.failed());
        if step == 5 {
            mid_tip = tip_x(&world);
        }
    }
    let final_tip = tip_x(&world);
    assert!(mid_tip > initial_tip + 0.005);
    assert!(final_tip > mid_tip);
    assert!(
        final_tip > initial_tip + 0.8 * length * alpha * 120.0,
        "coupled heating did not expand bar sufficiently: {initial_tip} -> {final_tip}"
    );
}

#[test]
fn dissipative_plastic_and_frictional_power_heats_solid_through_thermal_solver() {
    let mut world = World::new();
    world.gravity = vec3(0.0, -9.81, 0.0);
    let mut mat_data = Material::new("ductile_warm", 1200.0, 2.5e5, 0.3, 0.7, 0.0)
        .with_plasticity(4e3, 1e4)
        .with_thermal(Thermal {
            specific_heat: 400.0,
            conductivity: 40.0,
            expansion: 1e-5,
            emissivity: 0.1,
            reference_temperature: ROOM_TEMPERATURE,
            ..Thermal::default()
        });
    mat_data.friction = 0.8;
    let mat = world.add_material(mat_data);
    world.add_fixed(
        "floor",
        mat,
        core_space::Shape::HalfSpace {
            normal: Vec3::Y,
            offset: 0.0,
        },
        Pose::IDENTITY,
    );
    let block = world.add_solid(
        "slider",
        mat,
        TetMesh::cuboid(vec3(0.2, 0.2, 0.2), 0.2),
        Pose::at(vec3(0.0, 0.105, 0.0)),
    );
    if let Representation::Solid(s) = &mut world.object_mut(block).representation {
        s.nodes
            .columns
            .ensure(Column::TEMPERATURE, ROOM_TEMPERATURE);
    }
    world.set_velocity(block, vec3(6.0, -3.0, 0.0), Vec3::ZERO);

    let pool = Pool::new(2);
    let mut mech = Mechanics::new(Settings {
        integrator: Integrator::BackwardEuler,
        friction_updates: 2,
        ..Default::default()
    });
    let mut heat = HeatConduction::new(ThermalSettings::default());
    let h = 0.01;
    let mut max_power = 0.0_f64;
    for _ in 0..15 {
        let mr = mech.step(&pool, &mut world, h);
        assert!(!mr.failed());
        if let Some(power) = mech.dissipative_thermal_power(block) {
            max_power = max_power.max(power.iter().sum::<f64>());
            heat.set_nodal_power(block, power);
        }
        let tr = heat.step(&pool, &mut world, h);
        assert!(tr.converged);
    }
    let Representation::Solid(s) = &world.object(block).representation else {
        unreachable!()
    };
    let temps = s.nodes.columns.get(Column::TEMPERATURE).unwrap();
    let max_t = temps.iter().copied().fold(f64::NEG_INFINITY, f64::max);
    assert!(
        max_power > 1.0,
        "expected noticeable dissipative power, got {max_power} W"
    );
    assert!(
        max_t > ROOM_TEMPERATURE + 1e-4,
        "dissipative heating should warm solid above ambient {ROOM_TEMPERATURE}, got {max_t}"
    );
}
