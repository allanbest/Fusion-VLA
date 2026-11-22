//! Physical acceptance criteria. No timing limit depends on the test machine.
#[path = "../examples/support/accuracy.rs"]
mod accuracy;
use core_math::{vec3, Vec3};
use phys_mechanics::Integrator;

fn show(report: &accuracy::Report) {
    println!("{}\n{}", accuracy::CSV_HEADER, report.csv());
}

#[test]
fn cantilever_deflection_matches_beam_theory_and_improves_with_resolution() {
    let coarse = accuracy::cantilever_sag(0.2, 1).unwrap();
    let fine = accuracy::cantilever_sag(0.1, 1).unwrap();
    show(&coarse);
    show(&fine);
    assert!(coarse.relative_error() < 0.02, "{coarse:?}");
    assert!(fine.relative_error() < 0.01, "{fine:?}");
    assert!(fine.relative_error() < coarse.relative_error());
}

#[test]
fn rod_frequency_and_energy_converge_with_time_resolution() {
    let mut previous_error = f64::INFINITY;
    let mut previous_energy = 0.0;
    for dt in [1.0 / 60.0, 1.0 / 120.0, 1.0 / 240.0] {
        let r = accuracy::longitudinal_vibration(0.1, dt, 1).unwrap();
        show(&r);
        assert!(
            r.relative_error() < 0.02 && r.relative_error() < previous_error,
            "{r:?}"
        );
        assert!(
            r.energy_ratio <= 1.001 && r.energy_ratio > previous_energy,
            "{r:?}"
        );
        assert!(
            r.amplitude_per_period > 0.99 && r.amplitude_per_period < 1.001,
            "{r:?}"
        );
        assert!(r.momentum_error < 1e-9);
        assert_eq!(r.impacts + r.restarts, 0);
        previous_error = r.relative_error();
        previous_energy = r.energy_ratio;
    }
    assert!(previous_error < 0.005);
    assert!(previous_energy > 0.97);
}

#[test]
fn rod_frequency_improves_with_mesh_resolution() {
    let coarse = accuracy::longitudinal_vibration(0.2, 1.0 / 240.0, 1).unwrap();
    let fine = accuracy::longitudinal_vibration(0.1, 1.0 / 240.0, 1).unwrap();
    show(&coarse);
    show(&fine);
    assert!(fine.relative_error() < coarse.relative_error());
    assert!(fine.relative_error() < 0.005);
}

#[test]
fn bending_mode_matches_cantilever_theory() {
    let coarse = accuracy::bending_vibration(0.2, 1.0 / 60.0, 1).unwrap();
    let fine = accuracy::bending_vibration(0.1, 1.0 / 60.0, 1).unwrap();
    for r in [&coarse, &fine] {
        show(r);
        assert!(r.relative_error() < 0.02, "{r:?}");
        assert!(r.energy_ratio > 0.9 && r.energy_ratio <= 1.001, "{r:?}");
        assert!(
            r.amplitude_per_period > 0.99 && r.amplitude_per_period < 1.001,
            "{r:?}"
        );
        assert_eq!(r.impacts + r.restarts, 0);
    }
    assert!((fine.measured - coarse.measured).abs() / fine.reference < 0.005);
}

#[test]
fn moving_self_contact_respects_momentum_and_does_not_create_energy() {
    for dt in [1.0 / 60.0, 1.0 / 120.0, 1.0 / 240.0] {
        let r = accuracy::self_impact(dt, Integrator::TrBdf2, 1, Vec3::ZERO).unwrap();
        show(&r);
        // 38.4 kg at order-one velocities: < 0.003% momentum drift.
        // Absolute nonlinear position convergence here is 1e-7 m.
        assert!(r.momentum_error < 1e-3, "{r:?}");
        // Angular momentum is not exact for Euler contact stages. Require
        // <0.1% of the fixture's characteristic m*v*length (SI).
        assert!(r.angular_error < 1e-3 * 38.4 * 1.0 * 0.4, "{r:?}");
        assert!(r.energy_ratio > 0.0 && r.energy_ratio <= 1.01, "{r:?}");
    }
}

#[test]
fn all_integrators_respect_known_gravity_impulse_during_self_contact() {
    for integrator in [
        Integrator::BackwardEuler,
        Integrator::Bdf2,
        Integrator::Newmark,
        Integrator::TrBdf2,
    ] {
        let r = accuracy::self_impact(1.0 / 120.0, integrator, 1, vec3(0.0, -0.5, 0.0)).unwrap();
        show(&r);
        assert!(r.momentum_error < 1e-3, "{integrator:?}: {r:?}");
        assert!(
            r.energy_ratio > 0.0 && r.energy_ratio <= 1.01,
            "{integrator:?}: {r:?}"
        );
    }
}
