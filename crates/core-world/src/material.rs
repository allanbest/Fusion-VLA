//! What things are made of.

use core_math::constants::ROOM_TEMPERATURE;

/// A material property: a constant, or a table over temperature.
#[derive(Clone, Debug, PartialEq)]
pub enum Property {
    Constant(f64),
    /// `(temperature K, value)`, by increasing temperature; linear between
    /// entries, held at the ends.
    Table(Vec<(f64, f64)>),
}

impl Property {
    /// The value at `temperature`, K.
    pub fn at(&self, temperature: f64) -> f64 {
        match self {
            Self::Constant(v) => *v,
            Self::Table(t) => {
                let i = t.partition_point(|&(x, _)| x < temperature);
                if i == 0 {
                    t[0].1
                } else if i == t.len() {
                    t[t.len() - 1].1
                } else {
                    let ((x0, y0), (x1, y1)) = (t[i - 1], t[i]);
                    y0 + (y1 - y0) * (temperature - x0) / (x1 - x0)
                }
            }
        }
    }

    /// The value at room temperature.
    pub fn nominal(&self) -> f64 {
        self.at(ROOM_TEMPERATURE)
    }
}

impl From<f64> for Property {
    fn from(v: f64) -> Self {
        Self::Constant(v)
    }
}

/// Rate-independent J2 plasticity with linear isotropic hardening, SI.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Plasticity {
    /// Initial uniaxial yield stress `σ_y0`, Pa.
    pub yield_stress: f64,
    /// Isotropic hardening modulus `H` (`σ_y = σ_y0 + H ε̄_p`), Pa.
    pub hardening: f64,
}

/// Continuum cohesive damage and fracture parameters, SI.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Fracture {
    /// Critical tensile volumetric strain energy density `ψ_c`, J/m³.
    pub toughness: f64,
    /// Post-peak softening strain-energy scale `ψ_s`, J/m³.
    pub softening: f64,
    /// Minimum residual stiffness factor `k_min ∈ (0, 1]` at complete damage `d = 1`.
    pub residual_stiffness: f64,
}

/// Isotropic thermal parameters, SI.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Thermal {
    /// Specific heat capacity at constant volume/pressure, J/(kg·K).
    pub specific_heat: f64,
    /// Isotropic thermal conductivity, W/(m·K).
    pub conductivity: f64,
    /// Linear coefficient of thermal expansion, 1/K.
    pub expansion: f64,
    /// Surface radiative emissivity (0 to 1).
    pub emissivity: f64,
    /// Stress-free reference temperature, K.
    pub reference_temperature: f64,
    /// Optional equilibrium melting temperature, K.
    pub melting_point: Option<f64>,
    /// Latent heat of fusion, J/kg.
    pub latent_heat: f64,
    /// Smooth enthalpy phase-change temperature width, K.
    pub phase_interval: f64,
}

impl Default for Thermal {
    fn default() -> Self {
        Self {
            specific_heat: 900.0,
            conductivity: 50.0,
            expansion: 1.2e-5,
            emissivity: 0.8,
            reference_temperature: ROOM_TEMPERATURE,
            melting_point: None,
            latent_heat: 0.0,
            phase_interval: 1.0,
        }
    }
}

impl Thermal {
    /// Attaches latent-heat melting/solidification phase-change parameters.
    pub fn with_phase_change(
        mut self,
        melting_point: f64,
        latent_heat: f64,
        phase_interval: f64,
    ) -> Self {
        self.melting_point = Some(melting_point);
        self.latent_heat = latent_heat;
        self.phase_interval = phase_interval.max(1e-6);
        self
    }

    /// Apparent specific heat capacity `c_p(T) + L dH/dT`, J/(kg·K),
    /// using a compactly supported `C¹` bell over
    /// `[T_m - phase_interval/2, T_m + phase_interval/2]`.
    pub fn effective_specific_heat(&self, temperature: f64) -> f64 {
        let Some(tm) = self.melting_point else {
            return self.specific_heat;
        };
        if self.latent_heat <= 0.0 || self.phase_interval <= 0.0 {
            return self.specific_heat;
        }
        let half = 0.5 * self.phase_interval;
        let u = (temperature - tm) / half;
        if u.abs() >= 1.0 {
            return self.specific_heat;
        }
        let kernel = (15.0 / 16.0) * (1.0 - u * u).powi(2) / half;
        self.specific_heat + self.latent_heat * kernel
    }
}

/// A material: physical data, SI. Properties a module needs and the
/// material lacks are an error at load, never a silent default.
#[derive(Clone, Debug, PartialEq)]
pub struct Material {
    pub name: String,
    /// kg/m³, in the reference state. Mass is conserved, so density is not
    /// a function of temperature: expansion changes volume instead.
    pub density: f64,
    /// Young's modulus, Pa.
    pub youngs: Property,
    /// Poisson's ratio.
    pub poisson: Property,
    /// Coulomb friction coefficient; a pair uses the geometric mean of its
    /// two materials'.
    pub friction: f64,
    /// Coefficient of restitution at contact, 0 (dead) to 1 (elastic): the
    /// energy an impact loses in deformation too small or fast to model. A
    /// pair uses the less elastic of its two materials'.
    pub restitution: f64,
    /// Optional J2 isotropic hardening plasticity law for solids.
    pub plasticity: Option<Plasticity>,
    /// Optional cohesive damage and fracture parameters for solids.
    pub fracture: Option<Fracture>,
    /// Optional thermal conduction, expansion, radiation and phase-change properties.
    pub thermal: Option<Thermal>,
}

impl Material {
    pub fn new(
        name: &str,
        density: f64,
        youngs: f64,
        poisson: f64,
        friction: f64,
        restitution: f64,
    ) -> Self {
        Self {
            name: name.into(),
            density,
            youngs: youngs.into(),
            poisson: poisson.into(),
            friction,
            restitution,
            plasticity: None,
            fracture: None,
            thermal: None,
        }
    }

    pub fn with_plasticity(mut self, yield_stress: f64, hardening: f64) -> Self {
        self.plasticity = Some(Plasticity {
            yield_stress,
            hardening,
        });
        self
    }

    pub fn with_fracture(
        mut self,
        toughness: f64,
        softening: f64,
        residual_stiffness: f64,
    ) -> Self {
        self.fracture = Some(Fracture {
            toughness,
            softening,
            residual_stiffness: residual_stiffness.clamp(1e-6, 1.0),
        });
        self
    }

    pub fn with_thermal(mut self, thermal: Thermal) -> Self {
        self.thermal = Some(thermal);
        self
    }

    /// Lamé's parameters `(μ, λ)` at `temperature`, Pa.
    pub fn lame(&self, temperature: f64) -> (f64, f64) {
        let (e, nu) = (self.youngs.at(temperature), self.poisson.at(temperature));
        (
            e / (2.0 * (1.0 + nu)),
            e * nu / ((1.0 + nu) * (1.0 - 2.0 * nu)),
        )
    }

    /// Speed of a compression wave in a bar, m/s — what sets explicit
    /// stability and how fast stress travels.
    pub fn bar_wave_speed(&self, temperature: f64) -> f64 {
        (self.youngs.at(temperature) / self.density).sqrt()
    }

    /// The friction coefficient between two materials.
    pub fn friction_with(&self, other: &Material) -> f64 {
        (self.friction * other.friction).sqrt()
    }

    /// The coefficient of restitution between two materials.
    pub fn restitution_with(&self, other: &Material) -> f64 {
        self.restitution.min(other.restitution)
    }
}

/// Engineering materials at room temperature, from handbook values
/// (isotropic approximations; wood along the grain; restitution typical of
/// the material against a hard surface).
pub mod library {
    use super::{Material, Thermal, ROOM_TEMPERATURE};

    fn thermal(specific_heat: f64, conductivity: f64, expansion: f64, emissivity: f64) -> Thermal {
        Thermal {
            specific_heat,
            conductivity,
            expansion,
            emissivity,
            reference_temperature: ROOM_TEMPERATURE,
            melting_point: None,
            latent_heat: 0.0,
            phase_interval: 1.0,
        }
    }

    pub fn structural_steel() -> Material {
        Material::new("structural steel", 7850.0, 200e9, 0.30, 0.5, 0.6)
            .with_thermal(thermal(490.0, 50.0, 1.2e-5, 0.8))
    }
    pub fn aluminium() -> Material {
        Material::new("aluminium 6061", 2700.0, 68.9e9, 0.33, 0.5, 0.5)
            .with_thermal(thermal(896.0, 167.0, 2.36e-5, 0.2))
    }
    pub fn concrete() -> Material {
        Material::new("concrete", 2400.0, 30e9, 0.20, 0.6, 0.4)
            .with_thermal(thermal(880.0, 1.7, 1.0e-5, 0.9))
    }
    pub fn glass() -> Material {
        Material::new("soda-lime glass", 2500.0, 70e9, 0.22, 0.5, 0.6)
            .with_thermal(thermal(840.0, 1.05, 8.5e-6, 0.92))
    }
    pub fn granite() -> Material {
        Material::new("granite", 2700.0, 50e9, 0.25, 0.6, 0.5)
            .with_thermal(thermal(790.0, 2.9, 8.0e-6, 0.9))
    }
    pub fn oak() -> Material {
        Material::new("oak", 750.0, 11e9, 0.35, 0.5, 0.5)
            .with_thermal(thermal(2000.0, 0.17, 5.0e-6, 0.9))
    }
    pub fn ice() -> Material {
        Material::new("ice", 917.0, 9e9, 0.33, 0.05, 0.3)
            .with_thermal(thermal(2050.0, 2.2, 5.1e-5, 0.97))
    }
    pub fn rubber() -> Material {
        Material::new("natural rubber", 920.0, 5e6, 0.49, 0.9, 0.8)
            .with_thermal(thermal(2000.0, 0.16, 1.5e-4, 0.95))
    }
    pub fn gelatin() -> Material {
        Material::new("gelatin", 1050.0, 20e3, 0.45, 0.6, 0.5)
            .with_thermal(thermal(4000.0, 0.55, 7.0e-5, 0.95))
    }
}
