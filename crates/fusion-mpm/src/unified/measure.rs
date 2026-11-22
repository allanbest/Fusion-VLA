//! Optional physical measurements; no extra scans on the live stepping path.
use super::{control, StepFailure};
use fusion_core::{FluidSet, SolidSet};
use fusion_math::{to_f64, DVec3, Mat3};

/// Continuum totals measured in double precision. Rigid bodies and external
/// work are deliberately separate; these totals alone are not an energy balance.
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct ContinuumTotals {
    /// Material-point mass, kg.
    pub mass: f64,
    /// Sum of current solid and fluid volumes, m³.
    pub volume: f64,
    /// Linear momentum, kg·m/s.
    pub momentum: DVec3,
    /// Angular momentum about the world origin, including APIC affine spin.
    pub angular_momentum: DVec3,
    /// Translational plus APIC affine kinetic energy, J; excludes strain energy.
    pub kinetic: f64,
}

impl ContinuumTotals {
    /// Measure the state using the same quadratic-grid moment matrix as the
    /// engine: `D = cell²/4 I`. Legacy empty fluid APIC arrays represent zero
    /// affine velocity and unit volume ratio. Invalid input is rejected.
    pub fn measure(
        solids: &[SolidSet],
        fluids: &[FluidSet],
        cell: f32,
    ) -> Result<Self, StepFailure> {
        control::validate(solids, fluids, cell)?;
        let mut out = Self::default();
        let d = (cell as f64).powi(2) / 4.0;
        let mut add = |mass: f64, volume: f64, p: DVec3, v: fusion_math::Vec3, c: Mat3| {
            let v = to_f64(v);
            let x = to_f64(c.x_axis);
            let y = to_f64(c.y_axis);
            let z = to_f64(c.z_axis);
            // sum_i r_i cross (C r_i), with sum w r r^T = D.
            let spin = DVec3::new(y.z - z.y, z.x - x.z, x.y - y.x) * d;
            out.mass += mass;
            out.volume += volume;
            out.momentum += v * mass;
            out.angular_momentum += (p.cross(v) + spin) * mass;
            out.kinetic += 0.5
                * mass
                * (v.length_squared()
                    + d * (x.length_squared() + y.length_squared() + z.length_squared()));
        };
        for s in solids {
            for i in 0..s.len() {
                add(
                    s.params.point_mass() as f64,
                    s.volume[i] as f64
                        * s.deformation[i].determinant() as f64
                        * s.plastic[i] as f64,
                    s.particles.position[i],
                    s.particles.velocity[i],
                    s.affine[i],
                );
            }
        }
        for f in fluids {
            for i in 0..f.len() {
                add(
                    f.params.particle_mass() as f64,
                    (f.params.spacing as f64).powi(3)
                        * f.volume_ratio.get(i).copied().unwrap_or(1.0) as f64,
                    f.particles.position[i],
                    f.particles.velocity[i],
                    f.affine.get(i).copied().unwrap_or(Mat3::ZERO),
                );
            }
        }
        if !out.mass.is_finite()
            || !out.volume.is_finite()
            || !out.kinetic.is_finite()
            || !out.momentum.is_finite()
            || !out.angular_momentum.is_finite()
        {
            return Err(StepFailure::InvalidState);
        }
        Ok(out)
    }
}
