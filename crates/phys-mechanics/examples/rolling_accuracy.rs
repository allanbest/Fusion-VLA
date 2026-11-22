mod support {
    #[allow(dead_code)]
    pub mod rolling;
}
fn main() -> Result<(), String> {
    println!("case,dt,velocity_error,spin_error,slip,energy_ratio,expected_energy_ratio,newton,cg,products,linear_momentum_error,angular_momentum_error,seconds");
    for dt in [1.0 / 60.0, 1.0 / 120.0, 1.0 / 240.0] {
        for (name, spin) in support::rolling::launches() {
            let r = support::rolling::level(dt, 1, spin, false)?;
            println!(
                "{name},{dt},{:.9e},{:.9e},{:.9e},{:.9e},{:.9e},{},{},{},0,0,{:.6}",
                (r.velocity - r.expected_velocity).length(),
                (r.spin - r.expected_spin).length(),
                r.slip,
                r.energy_ratio,
                r.expected_energy_ratio,
                r.newton,
                r.cg,
                r.products,
                r.seconds
            );
        }
    }
    for dt in [1.0 / 240.0, 1.0 / 480.0, 1.0 / 960.0] {
        for e in [0.0, 0.5] {
            let r = support::rolling::oblique(dt, 1, e)?;
            println!(
                "oblique_{e},{dt},{:.9e},{:.9e},0,{:.9e},{:.9e},{},{},{},{:.9e},{:.9e},{:.6}",
                r.velocity_error,
                r.spin_error,
                r.energy_ratio,
                r.expected_energy_ratio,
                r.newton,
                r.cg,
                r.products,
                r.momentum_error,
                r.angular_error,
                r.seconds
            );
        }
    }
    Ok(())
}
