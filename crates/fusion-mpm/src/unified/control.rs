//! Numerical contracts for the explicit Unified continuum solver.
//!
//! A stability estimate is a necessary condition, not an accuracy certificate.
//! The caller must still validate physical observables under refinement.

use fusion_core::{FluidSet, SolidSet};
use fusion_math::{Mat3, Vec3};

/// Why an outer step was rejected. Rejection never commits a partial interval.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum StepFailure {
    /// Invalid timestep, grid spacing or solver configuration.
    InvalidConfiguration,
    /// Inconsistent arrays, non-finite state or inadmissible material data.
    InvalidState,
    /// Completing the interval would exceed the configured work budget.
    SubstepBudget,
    /// The sparse grid would exceed the configured block budget.
    GridBudget,
    /// Arithmetic produced a non-finite state.
    NonFiniteUpdate,
    /// Strict mode encountered a numerical repair or clamp.
    RepairRequired,
}

impl std::fmt::Display for StepFailure {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(match self {
            Self::InvalidConfiguration => "invalid solver settings",
            Self::InvalidState => "invalid material state",
            Self::SubstepBudget => "more internal steps are needed",
            Self::GridBudget => "grid memory limit reached",
            Self::NonFiniteUpdate => "numerical overflow",
            Self::RepairRequired => "accuracy mode rejected a numerical repair",
        })
    }
}

impl std::error::Error for StepFailure {}

/// Policy for the preview engine's legacy numerical repairs.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum RepairPolicy {
    /// Preserve preview behaviour and count every repair in the report.
    #[default]
    Record,
    /// Reject the entire interval instead of accepting a repaired state.
    Reject,
}

/// Numerical interventions, counted over attempted substeps, including rejected ones.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct Repairs {
    /// Deformation gradients repaired after inversion or near-collapse.
    pub deformation: usize,
    /// Fluid volume ratios limited to the preview's admissible range.
    pub volume: usize,
    /// Plastic volume ratios limited to the preview's admissible range.
    pub plastic: usize,
}

impl Repairs {
    /// Total number of interventions (one point may have more than one).
    pub fn total(self) -> usize {
        self.deformation + self.volume + self.plastic
    }
}

/// Mechanism that limited the explicit timestep.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum LimitKind {
    /// The requested interval needed no smaller step.
    #[default]
    Interval,
    /// Elastic or acoustic propagation, including advection.
    Wave,
    /// Explicit viscous diffusion.
    Viscosity,
    /// Local affine deformation rate.
    Deformation,
    /// Acceleration must not carry a point across a significant part of a cell.
    Acceleration,
}

/// Stable-step estimate in seconds and the mechanism that determined it.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct StepLimit {
    /// Largest estimated admissible substep, seconds.
    pub dt: f32,
    /// Dominant restriction.
    pub kind: LimitKind,
}

impl StepLimit {
    fn restrict(&mut self, dt: f32, kind: LimitKind) {
        if dt < self.dt {
            self.dt = dt;
            self.kind = kind;
        }
    }
}

pub(super) fn matrix_finite(m: Mat3) -> bool {
    m.x_axis.is_finite() && m.y_axis.is_finite() && m.z_axis.is_finite()
}

pub(super) fn norm(m: Mat3) -> f32 {
    // Accumulate in double so a large finite input cannot overflow the norm.
    let squared = |v: Vec3| (v.x as f64).powi(2) + (v.y as f64).powi(2) + (v.z as f64).powi(2);
    (squared(m.x_axis) + squared(m.y_axis) + squared(m.z_axis)).sqrt() as f32
}

fn positive(x: f32) -> bool {
    x.is_finite() && x > 0.0
}

/// Validate the state used by this engine, without mutating or allocating it.
/// Missing fluid APIC arrays are allowed only when both are empty (legacy input).
pub fn validate(solids: &[SolidSet], fluids: &[FluidSet], cell: f32) -> Result<(), StepFailure> {
    if !positive(cell) {
        return Err(StepFailure::InvalidConfiguration);
    }
    // A bounded index also avoids integer overflow in stencil/block arithmetic.
    let positions_ok = |p: &fusion_core::ParticleSet| {
        p.velocity.len() == p.len()
            && p.position.iter().all(|x| {
                x.is_finite() && x.x.abs().max(x.y.abs()).max(x.z.abs()) / (cell as f64) < 1e12
            })
            && p.velocity.iter().all(|v| v.is_finite())
    };
    for s in solids.iter().filter(|s| !s.is_empty()) {
        let n = s.len();
        if !positions_ok(&s.particles)
            || !positive(s.params.cell_size)
            || !positive(s.params.point_mass())
            || !s.params.material.wave_speed().is_finite()
            || s.deformation.len() != n
            || s.affine.len() != n
            || s.volume.len() != n
            || s.plastic.len() != n
            || s.damage.len() != n
            || s.deformation
                .iter()
                .any(|&f| !matrix_finite(f) || !positive(f.determinant()))
            || s.affine.iter().any(|&c| !matrix_finite(c))
            || s.volume.iter().any(|&v| !positive(v))
            || s.plastic.iter().any(|&v| !positive(v))
            || s.damage
                .iter()
                .any(|&d| !d.is_finite() || !(0.0..=1.0).contains(&d))
            || s.params
                .fibre
                .is_some_and(|f| !f.stiffness.is_finite() || f.stiffness < 0.0)
        {
            return Err(StepFailure::InvalidState);
        }
    }
    for f in fluids.iter().filter(|f| !f.is_empty()) {
        let n = f.len();
        let legacy = f.affine.is_empty() && f.volume_ratio.is_empty();
        if !positions_ok(&f.particles)
            || !positive(f.params.spacing)
            || !positive(f.params.rest_density)
            || !positive(f.params.particle_mass())
            || !f.params.viscosity.is_finite()
            || f.params.viscosity < 0.0
            || f.params
                .dynamic_viscosity
                .is_some_and(|v| !v.is_finite() || v < 0.0)
            || (!legacy && (f.affine.len() != n || f.volume_ratio.len() != n))
            || f.affine.iter().any(|&c| !matrix_finite(c))
            || f.volume_ratio.iter().any(|&j| !positive(j))
        {
            return Err(StepFailure::InvalidState);
        }
    }
    Ok(())
}

/// Compute the explicit engine's current wave, viscosity, deformation and
/// gravity restrictions. Recompute after every accepted substep; compression
/// changes Tait's tangent sound speed. Inputs must pass [`validate`] first.
pub fn estimate(
    solids: &[SolidSet],
    fluids: &[FluidSet],
    cell: f32,
    sound: f32,
    gravity: Vec3,
    implicit: bool,
) -> StepLimit {
    let mut limit = StepLimit {
        dt: f32::INFINITY,
        kind: LimitKind::Interval,
    };
    // The explicit limit the implicit solids are excused from.
    let mut solid_waves = f32::INFINITY;
    for s in solids.iter().filter(|s| !s.is_empty()) {
        let material = s.params.material;
        let mut wave = material.wave_speed();
        if matches!(
            material.model,
            fusion_core::solid::Constitutive::Snow { .. }
        ) {
            let (mu, lambda) = material.lame();
            for &plastic in &s.plastic {
                let (mu, lambda) = crate::model::hardened(material.model, mu, lambda, plastic);
                wave = wave.max(((lambda + 2.0 * mu) / material.density).sqrt());
            }
        }
        if let Some(fibre) = s.params.fibre {
            wave *= (1.0 + fibre.stiffness).sqrt();
        }
        // Solved implicitly, a solid's waves need not set the step: only
        // how far its points move in one, and how fast it deforms. Fibres
        // are not in the implicit stiffness and keep their explicit limit.
        let waves = if implicit && s.params.fibre.is_none() {
            0.0
        } else {
            wave
        };
        for (v, &c) in s.particles.velocity.iter().zip(&s.affine) {
            let rate = norm(c);
            limit.restrict(
                0.4 * cell / (waves + v.length() + cell * rate),
                LimitKind::Wave,
            );
            if waves == 0.0 {
                solid_waves = solid_waves.min(0.4 * cell / (wave + v.length() + cell * rate));
            }
            limit.restrict(0.1 / rate, LimitKind::Deformation);
        }
    }
    for f in fluids.iter().filter(|f| !f.is_empty()) {
        // A conservative 3D explicit diffusion guard. This is an estimate for
        // this transfer discretisation, not a universal MPM stability theorem.
        let largest_j = f.volume_ratio.iter().copied().fold(1.0_f32, f32::max);
        let nu = super::viscosity_of(&f.params) * largest_j / f.params.rest_density;
        limit.restrict(cell * cell / (12.0 * nu), LimitKind::Viscosity);
        for i in 0..f.len() {
            let j = f.volume_ratio.get(i).copied().unwrap_or(1.0).min(1.0);
            // p = rho0*c0²/7 * (J^-7 - 1), rho=rho0/J:
            // sqrt(dp/drho) = c0 * J^-3 in compression.
            let wave = sound / (j * j * j);
            let rate = f.affine.get(i).copied().map(norm).unwrap_or(0.0);
            let speed = f.particles.velocity[i].length();
            limit.restrict(0.4 * cell / (wave + speed + cell * rate), LimitKind::Wave);
            limit.restrict(0.1 / rate, LimitKind::Deformation);
        }
    }
    if gravity.length_squared() > 0.0 {
        limit.restrict(
            (0.4 * cell / gravity.length()).sqrt(),
            LimitKind::Acceleration,
        );
    }
    // Only where it at least halves the substeps: an implicit solve costs a
    // few explicit substeps' work, and a solid barely past its limit — water
    // beside it already asking for six — made the step twice as slow.
    if 2.0 * solid_waves >= limit.dt {
        limit.restrict(solid_waves, LimitKind::Wave);
    }
    limit
}
