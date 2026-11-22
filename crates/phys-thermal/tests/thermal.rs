use core_math::constants::{ROOM_TEMPERATURE, STEFAN_BOLTZMANN};
use core_math::{vec3, Pose};
use core_runtime::Pool;
use core_world::{library, Column, ObjectId, Representation, TetMesh, Thermal, World};
use phys_thermal::{HeatConduction, Process, ThermalBoundary, ThermalIntegrator, ThermalSettings};
use std::f64::consts::PI;

fn bar_mesh(quadratic: bool, length: f64, width: f64, edge: f64) -> TetMesh {
    let base = TetMesh::cuboid(vec3(length, width, width), edge);
    if quadratic {
        base.quadratic()
    } else {
        base
    }
}

fn bar_world(
    quadratic: bool,
    mat: core_world::Material,
    length: f64,
    width: f64,
    edge: f64,
) -> (World, ObjectId) {
    let mut world = World::new();
    let m = world.add_material(mat);
    let id = world.add_solid(
        "bar",
        m,
        bar_mesh(quadratic, length, width, edge),
        Pose::IDENTITY,
    );
    (world, id)
}

#[test]
fn steady_state_bar_reproduces_linear_temperature_gradient() {
    for &quadratic in &[false, true] {
        let length = 1.0;
        let (mut world, id) = bar_world(quadratic, library::structural_steel(), length, 0.2, 0.2);
        let (left, right): (Vec<u32>, Vec<u32>) = {
            let Representation::Solid(s) = &world.object(id).representation else {
                unreachable!()
            };
            let l = s
                .nodes
                .position
                .iter()
                .enumerate()
                .filter(|(_, p)| p.x < -0.5 * length + 1e-6)
                .map(|(i, _)| i as u32)
                .collect();
            let r = s
                .nodes
                .position
                .iter()
                .enumerate()
                .filter(|(_, p)| p.x > 0.5 * length - 1e-6)
                .map(|(i, _)| i as u32)
                .collect();
            (l, r)
        };
        let t_left = 300.0;
        let t_right = 400.0;
        let mut heat = HeatConduction::new(ThermalSettings {
            integrator: ThermalIntegrator::BackwardEuler,
            boundaries: vec![
                ThermalBoundary::Dirichlet {
                    object: id,
                    nodes: left,
                    temperature: t_left,
                },
                ThermalBoundary::Dirichlet {
                    object: id,
                    nodes: right,
                    temperature: t_right,
                },
            ],
            ..Default::default()
        });
        let pool = Pool::new(2);
        // Multiple large implicit steps converge to exact Laplace steady state.
        let mut r = heat.step(&pool, &mut world, 1e8);
        assert!(r.converged);
        r = heat.step(&pool, &mut world, 1e8);
        assert!(r.converged);
        let Representation::Solid(s) = &world.object(id).representation else {
            unreachable!()
        };
        let temp = s.nodes.columns.get(Column::TEMPERATURE).unwrap();
        for (i, p) in s.nodes.position.iter().enumerate() {
            let expected = t_left + (t_right - t_left) * (p.x / length + 0.5);
            assert!(
                (temp[i] - expected).abs() < 1e-4,
                "quadratic={quadratic} node {i} at x={}: got {} expected {expected}",
                p.x,
                temp[i]
            );
        }
    }
}

#[test]
fn transient_fourier_sine_mode_decays_at_analytical_rate() {
    let length = 1.0;
    let width = 0.2;
    let mut mat = library::aluminium();
    let th = mat.thermal.unwrap();
    let alpha = th.conductivity / (mat.density * th.specific_heat);
    mat.thermal = Some(Thermal {
        reference_temperature: 0.0,
        ..th
    });
    let (mut world, id) = bar_world(false, mat, length, width, 0.1);
    let (left, right): (Vec<u32>, Vec<u32>) = {
        let Representation::Solid(s) = &mut world.object_mut(id).representation else {
            unreachable!()
        };
        let t0 = s.nodes.columns.ensure(Column::TEMPERATURE, 0.0);
        for (i, p) in s.nodes.position.iter().enumerate() {
            t0[i] = (PI * (p.x / length + 0.5)).sin();
        }
        let l = s
            .nodes
            .position
            .iter()
            .enumerate()
            .filter(|(_, p)| p.x < -0.5 * length + 1e-6)
            .map(|(i, _)| i as u32)
            .collect();
        let r = s
            .nodes
            .position
            .iter()
            .enumerate()
            .filter(|(_, p)| p.x > 0.5 * length - 1e-6)
            .map(|(i, _)| i as u32)
            .collect();
        (l, r)
    };
    let mut heat = HeatConduction::new(ThermalSettings {
        integrator: ThermalIntegrator::CrankNicolson,
        boundaries: vec![
            ThermalBoundary::Dirichlet {
                object: id,
                nodes: left,
                temperature: 0.0,
            },
            ThermalBoundary::Dirichlet {
                object: id,
                nodes: right,
                temperature: 0.0,
            },
        ],
        ..Default::default()
    });
    let pool = Pool::new(2);
    let duration = 200.0;
    let steps = 40;
    let h = duration / steps as f64;
    for _ in 0..steps {
        let r = heat.step(&pool, &mut world, h);
        assert!(r.converged);
    }
    let Representation::Solid(s) = &world.object(id).representation else {
        unreachable!()
    };
    let temp = s.nodes.columns.get(Column::TEMPERATURE).unwrap();
    let decay = (-alpha * (PI / length).powi(2) * duration).exp();
    let max_t = temp.iter().copied().fold(0.0_f64, f64::max);
    assert!(
        (max_t - decay).abs() < 0.02,
        "Fourier mode amplitude {max_t} vs analytical {decay}"
    );
}

#[test]
fn adiabatic_solid_conserves_thermal_energy_to_machine_precision() {
    let run = |threads| {
        let pool = Pool::new(threads);
        let (mut world, id) = bar_world(true, library::structural_steel(), 0.4, 0.2, 0.2);
        {
            let Representation::Solid(s) = &mut world.object_mut(id).representation else {
                unreachable!()
            };
            let t = s
                .nodes
                .columns
                .ensure(Column::TEMPERATURE, ROOM_TEMPERATURE);
            for (i, p) in s.nodes.position.iter().enumerate() {
                t[i] = ROOM_TEMPERATURE + 100.0 * p.x + 40.0 * p.y;
            }
        }
        let mut heat = HeatConduction::new(ThermalSettings::default());
        let first = heat.step(&pool, &mut world, 0.5);
        assert!(first.converged);
        let e0 = first.thermal_energy;
        let mut last = first;
        for _ in 0..20 {
            last = heat.step(&pool, &mut world, 0.5);
            assert!(last.converged);
            assert!(
                (last.thermal_energy - e0).abs() / e0 < 1e-12,
                "thermal energy drifted: {} vs {e0}",
                last.thermal_energy
            );
        }
        assert!(last.max_temperature < first.max_temperature);
        assert!(last.min_temperature > first.min_temperature);
        let Representation::Solid(s) = &world.object(id).representation else {
            unreachable!()
        };
        s.nodes.columns.get(Column::TEMPERATURE).unwrap().to_vec()
    };

    assert_eq!(run(1), run(4));
}

#[test]
fn convection_and_radiation_cool_monotonically_to_ambient() {
    // 1. Lumped convective cooling matches analytical rate when Biot << 1.
    let (mut world, id) = bar_world(false, library::aluminium(), 0.1, 0.1, 0.1);
    {
        let Representation::Solid(s) = &mut world.object_mut(id).representation else {
            unreachable!()
        };
        s.nodes.columns.ensure(Column::TEMPERATURE, 400.0);
    }
    let coeff = 25.0;
    let ambient = 300.0;
    let area = 6.0 * 0.1 * 0.1;
    let mat = library::aluminium();
    let mass = world.object(id).mass();
    let cap = mass * mat.thermal.unwrap().specific_heat;
    let mut heat = HeatConduction::new(ThermalSettings {
        boundaries: vec![ThermalBoundary::Convection {
            object: id,
            coefficient: coeff,
            ambient,
        }],
        ..Default::default()
    });
    let pool = Pool::new(1);
    let h = 0.5;
    let mut last_report = Default::default();
    for _ in 0..20 {
        let r = heat.step(&pool, &mut world, h);
        assert!(r.converged);
        last_report = r;
    }
    let expected = ambient + (400.0 - ambient) * (-coeff * area / cap * 10.0).exp();
    let avg = last_report.thermal_energy / cap;
    assert!(
        (avg - expected).abs() < 0.01,
        "convective temperature {avg} differed from analytical {expected}"
    );

    // 2. Stefan-Boltzmann radiative cooling cools hot object monotonically toward ambient.
    let (mut world, id) = bar_world(false, library::granite(), 0.2, 0.2, 0.2);
    {
        let Representation::Solid(s) = &mut world.object_mut(id).representation else {
            unreachable!()
        };
        s.nodes.columns.ensure(Column::TEMPERATURE, 900.0);
    }
    let _ = STEFAN_BOLTZMANN;
    let mut rad = HeatConduction::new(ThermalSettings {
        boundaries: vec![ThermalBoundary::Radiation {
            object: id,
            ambient: 300.0,
        }],
        ..Default::default()
    });
    let mut prev = 900.0;
    for _ in 0..10 {
        let r = rad.step(&pool, &mut world, 50.0);
        assert!(r.converged);
        assert!(r.max_temperature < prev && r.min_temperature > 300.0);
        prev = r.max_temperature;
    }
}

#[test]
fn latent_heat_phase_change_slows_temperature_rise_across_melting_window() {
    let base_mat = library::aluminium();
    let th_base = base_mat.thermal.unwrap();
    let mut pc_mat = base_mat.clone();
    pc_mat.thermal = Some(th_base.with_phase_change(500.0, 200_000.0, 20.0));

    let run = |mat: core_world::Material| -> f64 {
        let (mut world, id) = bar_world(false, mat, 0.2, 0.2, 0.2);
        {
            let Representation::Solid(s) = &mut world.object_mut(id).representation else {
                unreachable!()
            };
            s.nodes.columns.ensure(Column::TEMPERATURE, 492.0);
        }
        let mut heat = HeatConduction::new(ThermalSettings {
            boundaries: vec![ThermalBoundary::VolumetricSource {
                object: id,
                rate: 5.0e7,
            }],
            ..Default::default()
        });
        let pool = Pool::new(2);
        let mut r = Default::default();
        for _ in 0..40 {
            r = heat.step(&pool, &mut world, 0.05);
            assert!(r.converged);
        }
        r.max_temperature
    };

    let t_no_latent = run(base_mat);
    let t_with_latent = run(pc_mat);
    assert!(
        t_no_latent > 520.0,
        "sensible baseline should pass melting band: {t_no_latent}"
    );
    assert!(
        t_with_latent < t_no_latent - 20.0 && t_with_latent > 495.0,
        "latent heat plateau should hold temperature near 500K: with={t_with_latent}, without={t_no_latent}"
    );
}

#[test]
fn nodal_power_source_conserves_injected_energy_and_replays_across_threads() {
    let run = |threads| {
        let pool = Pool::new(threads);
        let (mut world, id) = bar_world(false, library::aluminium(), 0.2, 0.2, 0.2);
        let n_nodes = {
            let Representation::Solid(s) = &mut world.object_mut(id).representation else {
                unreachable!()
            };
            s.nodes
                .columns
                .ensure(Column::TEMPERATURE, ROOM_TEMPERATURE);
            s.nodes.len()
        };
        let mut heat = HeatConduction::new(ThermalSettings::default());
        let first = heat.step(&pool, &mut world, 0.5);
        assert!(first.converged);
        let e0 = first.thermal_energy;

        // Replace nodal power repeatedly to ensure set_nodal_power overwrites cleanly.
        let per_node = 500.0 / n_nodes as f64;
        heat.set_nodal_power(id, vec![9999.0; n_nodes]);
        heat.set_nodal_power(id, vec![per_node; n_nodes]);
        let mut last = first;
        for _ in 0..10 {
            last = heat.step(&pool, &mut world, 0.5);
            assert!(last.converged);
        }
        let injected = 500.0 * 5.0; // 500 W * 5 s = 2500 J
        assert!(
            ((last.thermal_energy - e0) - injected).abs() < 1e-6,
            "dE={} vs injected={injected}",
            last.thermal_energy - e0
        );
        let Representation::Solid(s) = &world.object(id).representation else {
            unreachable!()
        };
        s.nodes.columns.get(Column::TEMPERATURE).unwrap().to_vec()
    };

    assert_eq!(run(1), run(4));
}
