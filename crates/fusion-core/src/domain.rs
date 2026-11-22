//! Domains: a local origin and a system of units.
//!
//! PLAN.md §7. FP64 carries ~16 significant digits; Fusion's stated span is 31
//! orders of magnitude. No single coordinate frame holds both ends, and that is
//! arithmetic rather than a performance problem. Every simulation therefore
//! runs in domain-local natural units and converts only at boundaries.
//!
//! v0.1 uses SI throughout and never crosses a boundary. The type exists now
//! because retrofitting an origin and a unit scale into a coordinate system
//! after scenes exist means touching every scene ever authored.

use fusion_math::DVec3;

/// A coordinate frame plus the units measured in it.
///
/// The scale fields answer "how many SI units is one local unit?", so
/// `to_si_length(x) == x * length_scale`.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Domain {
    /// Position of this domain's origin in the global frame, in SI metres.
    pub origin: DVec3,
    /// Metres per domain length unit.
    pub length_scale: f64,
    /// Seconds per domain time unit.
    pub time_scale: f64,
    /// Kilograms per domain mass unit.
    pub mass_scale: f64,
}

impl Default for Domain {
    fn default() -> Self {
        Self::SI
    }
}

impl Domain {
    /// Metres, seconds, kilograms. Fusion Basic never leaves this.
    pub const SI: Self = Self {
        origin: DVec3::ZERO,
        length_scale: 1.0,
        time_scale: 1.0,
        mass_scale: 1.0,
    };

    /// Angstroms, femtoseconds, atomic mass units — the natural units of
    /// molecular dynamics (PLAN.md §15).
    pub const ATOMIC: Self = Self {
        origin: DVec3::ZERO,
        length_scale: 1e-10,
        time_scale: 1e-15,
        mass_scale: 1.660_539_066_60e-27,
    };

    /// Astronomical units, years, solar masses (PLAN.md §14).
    pub const ORBITAL: Self = Self {
        origin: DVec3::ZERO,
        length_scale: 1.495_978_707e11,
        time_scale: 3.155_76e7,
        mass_scale: 1.988_47e30,
    };

    /// Parsecs, megayears, solar masses.
    pub const GALACTIC: Self = Self {
        origin: DVec3::ZERO,
        length_scale: 3.085_677_581e16,
        time_scale: 3.155_76e13,
        mass_scale: 1.988_47e30,
    };

    /// Builder: put the domain origin here, in SI metres from the world
    /// origin.
    pub fn with_origin(mut self, origin: DVec3) -> Self {
        self.origin = origin;
        self
    }

    /// Domain length to metres.
    #[inline]
    pub fn to_si_length(&self, v: f64) -> f64 {
        v * self.length_scale
    }

    /// Metres to domain length.
    #[inline]
    pub fn from_si_length(&self, v: f64) -> f64 {
        v / self.length_scale
    }

    /// Domain time to seconds.
    #[inline]
    pub fn to_si_time(&self, v: f64) -> f64 {
        v * self.time_scale
    }

    /// Seconds to domain time.
    #[inline]
    pub fn from_si_time(&self, v: f64) -> f64 {
        v / self.time_scale
    }

    /// Domain mass to kilograms.
    #[inline]
    pub fn to_si_mass(&self, v: f64) -> f64 {
        v * self.mass_scale
    }

    /// Local point to the global SI frame.
    #[inline]
    pub fn point_to_global(&self, p: DVec3) -> DVec3 {
        p * self.length_scale + self.origin
    }

    /// Global SI point into this domain's local units.
    #[inline]
    pub fn point_from_global(&self, p: DVec3) -> DVec3 {
        (p - self.origin) / self.length_scale
    }

    /// Derived scale for velocity, m/s per local unit.
    #[inline]
    pub fn velocity_scale(&self) -> f64 {
        self.length_scale / self.time_scale
    }
}
