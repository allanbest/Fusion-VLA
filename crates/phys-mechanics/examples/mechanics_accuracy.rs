//! Reproducible accuracy AND cost measurements, independent of the app.
mod support {
    pub mod accuracy;
}
use support::accuracy;
fn main() -> Result<(), String> {
    let threads = std::env::var("ACCURACY_THREADS")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(1);
    println!("{}", accuracy::CSV_HEADER);
    for edge in [0.2, 0.1] {
        println!("{}", accuracy::cantilever_sag(edge, threads)?.csv());
    }
    for dt in [1.0 / 60.0, 1.0 / 120.0, 1.0 / 240.0] {
        println!(
            "{}",
            accuracy::longitudinal_vibration(0.1, dt, threads)?.csv()
        );
    }
    for edge in [0.2, 0.1] {
        println!(
            "{}",
            accuracy::bending_vibration(edge, 1.0 / 60.0, threads)?.csv()
        );
    }
    println!(
        "{}",
        accuracy::longitudinal_vibration(0.2, 1.0 / 240.0, threads)?.csv()
    );
    for dt in [1.0 / 60.0, 1.0 / 120.0, 1.0 / 240.0] {
        println!(
            "{}",
            accuracy::self_impact(
                dt,
                phys_mechanics::Integrator::TrBdf2,
                threads,
                core_math::Vec3::ZERO
            )?
            .csv()
        );
    }
    Ok(())
}
