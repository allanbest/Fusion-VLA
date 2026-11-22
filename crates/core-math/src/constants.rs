//! Physical constants, SI, CODATA 2018 (exact where the SI defines them).

/// Newtonian constant of gravitation, m³ kg⁻¹ s⁻².
pub const GRAVITATIONAL: f64 = 6.674_30e-11;
/// Standard acceleration of gravity at the Earth's surface, m/s².
pub const STANDARD_GRAVITY: f64 = 9.806_65;
/// Boltzmann constant, J/K (exact).
pub const BOLTZMANN: f64 = 1.380_649e-23;
/// Avogadro constant, 1/mol (exact).
pub const AVOGADRO: f64 = 6.022_140_76e23;
/// Molar gas constant, J/(mol K) (exact).
pub const GAS: f64 = BOLTZMANN * AVOGADRO;
/// Stefan–Boltzmann constant, W m⁻² K⁻⁴.
pub const STEFAN_BOLTZMANN: f64 = 5.670_374_419e-8;
/// Speed of light in vacuum, m/s (exact).
pub const SPEED_OF_LIGHT: f64 = 299_792_458.0;
/// Elementary charge, C (exact).
pub const ELEMENTARY_CHARGE: f64 = 1.602_176_634e-19;
/// Vacuum magnetic permeability, N/A².
pub const VACUUM_PERMEABILITY: f64 = 1.256_637_062_12e-6;
/// Vacuum electric permittivity, F/m.
pub const VACUUM_PERMITTIVITY: f64 = 8.854_187_812_8e-12;
/// Atomic mass constant (one dalton), kg.
pub const ATOMIC_MASS: f64 = 1.660_539_066_60e-27;
/// Standard atmospheric pressure, Pa (exact).
pub const STANDARD_ATMOSPHERE: f64 = 101_325.0;
/// 0 °C, K (exact).
pub const ZERO_CELSIUS: f64 = 273.15;
/// The temperature material properties are quoted at unless stated: 20 °C, K.
pub const ROOM_TEMPERATURE: f64 = 293.15;
