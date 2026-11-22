#[allow(dead_code)]
#[path = "../examples/support/accuracy.rs"]
mod accuracy;
use core_math::Vec3;
use phys_mechanics::Integrator;
#[test]
fn force_consistent_startup_preserves_resolved_vibration_energy() {
    let old = accuracy::longitudinal_options(0.1, 1.0 / 60.0, 1, false, 1).unwrap();
    let new = accuracy::longitudinal_options(0.1, 1.0 / 60.0, 1, true, 1).unwrap();
    println!("startup off {}\nstartup on {}", old.csv(), new.csv());
    assert!(new.energy_ratio > old.energy_ratio + 0.15);
    assert!(new.energy_ratio > 0.93 && new.energy_ratio <= 1.001);
    assert!(new.relative_error() < 0.02);
}
#[test]
fn tighter_angular_budget_refines_contact_instead_of_projecting_global_momentum() {
    let old =
        accuracy::self_impact_options(1.0 / 60.0, Integrator::TrBdf2, 1, Vec3::ZERO, None, 1e-7)
            .unwrap();
    let new = accuracy::self_impact_options(
        1.0 / 60.0,
        Integrator::TrBdf2,
        1,
        Vec3::ZERO,
        Some(1e-4),
        1e-7,
    )
    .unwrap();
    println!(
        "angular unchecked {}\nangular refined {}\nrefinements {}",
        old.csv(),
        new.csv(),
        new.accuracy_retries
    );
    assert!(new.accuracy_retries > 0);
    assert!(new.angular_error < old.angular_error * 0.75);
    assert!(new.momentum_error < 1e-5);
    assert!(new.energy_ratio > 0.0 && new.energy_ratio <= 1.01);
}
#[test]
fn solid_preconditioning_patches_keep_known_frequency_and_energy() {
    let r = accuracy::longitudinal_options(0.1, 1.0 / 240.0, 4, true, 4).unwrap();
    println!("patch four {}", r.csv());
    assert!(r.relative_error() < 0.005);
    assert!(r.energy_ratio > 0.99 && r.energy_ratio <= 1.001);
    assert!(r.momentum_error < 1e-9);
}
