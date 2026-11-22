//! Experimental conservative finite-volume pressure projection.
//!
//! This is a tested engine prototype, NOT the live MPM fluid path. A scene
//! adapter must supply a compatible face field, surface geometry and coupled
//! moving-boundary treatment before this can replace weak compressibility.
//! Topology is explicit and independent of the sparse block allocator.

use crate::linear::{Pcg, SolveFailure, SolveReport, SolveSettings};

/// Oriented flux face between a fluid cell and another cell or atmosphere.
#[derive(Clone, Copy, Debug)]
pub struct PressureFace {
    /// Upstream cell index (orientation only, not flow direction).
    pub minus: usize,
    /// Downstream cell; `None` means zero gauge pressure at atmosphere.
    pub plus: Option<usize>,
    /// `dt * face_area / (density * centre_distance)`, m³/(Pa·s).
    /// Use the appropriate boundary distance for atmospheric faces.
    pub mobility: f64,
    /// Predicted volume flux, m³/s, positive from minus towards plus.
    pub flux: f64,
}

/// Projection failures leave all face fluxes unchanged.
#[derive(Clone, Debug, PartialEq)]
pub enum ProjectionFailure {
    /// Invalid cell volume, face connectivity or non-finite data.
    InvalidInput,
    /// Pressure iterations failed; no velocity correction was committed.
    Linear(SolveFailure),
}

/// Quantitative outcome of the prototype projection.
#[derive(Clone, Debug)]
pub struct ProjectionReport {
    /// Cell gauge pressures, Pa. Closed components use a pinned gauge.
    pub pressure: Vec<f64>,
    /// True linear residual and iteration count.
    pub solve: SolveReport,
    /// Maximum absolute cell divergence before projection, 1/s.
    pub divergence_before: f64,
    /// Maximum absolute cell divergence after projection, 1/s.
    pub divergence_after: f64,
}

/// Reusable linear workspace for pressure solves.
#[derive(Clone, Debug, Default)]
pub struct PressureProjection {
    linear: Pcg,
}

fn root(parent: &mut [usize], mut i: usize) -> usize {
    while parent[i] != i {
        parent[i] = parent[parent[i]];
        i = parent[i];
    }
    i
}

impl PressureProjection {
    /// Project face fluxes onto zero cell divergence using a symmetric graph
    /// Laplacian. Omit stationary impermeable faces (zero flux). Moving walls,
    /// prescribed inflow, surface tension and solid inertia are not supported.
    /// Closed connected components have one pressure pinned, without deleting
    /// any physical face; their net predicted flux is zero by construction.
    pub fn project(
        &mut self,
        volumes: &[f64],
        faces: &mut [PressureFace],
        settings: SolveSettings,
    ) -> Result<ProjectionReport, ProjectionFailure> {
        let n = volumes.len();
        if volumes.iter().any(|&v| !v.is_finite() || v <= 0.0)
            || faces.iter().any(|f| {
                f.minus >= n
                    || f.plus.is_some_and(|j| j >= n || j == f.minus)
                    || !f.mobility.is_finite()
                    || f.mobility <= 0.0
                    || !f.flux.is_finite()
            })
        {
            return Err(ProjectionFailure::InvalidInput);
        }
        let mut parent: Vec<_> = (0..n).collect();
        for f in faces.iter() {
            if let Some(j) = f.plus {
                let a = root(&mut parent, f.minus);
                let b = root(&mut parent, j);
                parent[b] = a;
            }
        }
        let mut open = vec![false; n];
        for f in faces.iter().filter(|f| f.plus.is_none()) {
            let r = root(&mut parent, f.minus);
            open[r] = true;
        }
        let mut pinned = vec![false; n];
        for i in 0..n {
            let r = root(&mut parent, i);
            if !open[r] {
                pinned[r] = true;
            }
        }
        let mut rhs = vec![0.0; n];
        let mut diagonal = vec![0.0; n];
        for f in faces.iter() {
            rhs[f.minus] -= f.flux;
            diagonal[f.minus] += f.mobility;
            if let Some(j) = f.plus {
                rhs[j] += f.flux;
                diagonal[j] += f.mobility;
            }
        }
        if rhs.iter().chain(&diagonal).any(|v| !v.is_finite()) {
            return Err(ProjectionFailure::Linear(SolveFailure::NonFinite));
        }
        let divergence_before = rhs
            .iter()
            .zip(volumes)
            .map(|(d, v)| (d / v).abs())
            .fold(0.0_f64, f64::max);
        for i in 0..n {
            if pinned[i] {
                rhs[i] = 0.0;
                diagonal[i] = 1.0;
            }
        }
        let mut pressure = vec![0.0; n];
        let solve = self
            .linear
            .solve(&rhs, &diagonal, &mut pressure, settings, |x, y| {
                y.fill(0.0);
                for f in faces.iter() {
                    let a = if pinned[f.minus] { 0.0 } else { x[f.minus] };
                    let b = f.plus.filter(|&j| !pinned[j]).map(|j| x[j]).unwrap_or(0.0);
                    let flux = f.mobility * (a - b);
                    if !pinned[f.minus] {
                        y[f.minus] += flux;
                    }
                    if let Some(j) = f.plus {
                        if !pinned[j] {
                            y[j] -= flux;
                        }
                    }
                }
                for i in 0..n {
                    if pinned[i] {
                        y[i] = x[i];
                    }
                }
            })
            .map_err(ProjectionFailure::Linear)?;
        let corrected: Vec<_> = faces
            .iter()
            .map(|f| {
                f.flux
                    + f.mobility * (pressure[f.minus] - f.plus.map(|j| pressure[j]).unwrap_or(0.0))
            })
            .collect();
        if corrected.iter().any(|q| !q.is_finite()) {
            return Err(ProjectionFailure::Linear(SolveFailure::NonFinite));
        }
        let mut divergence = vec![0.0; n];
        for (f, &q) in faces.iter().zip(&corrected) {
            divergence[f.minus] += q;
            if let Some(j) = f.plus {
                divergence[j] -= q;
            }
        }
        if divergence.iter().any(|v| !v.is_finite()) {
            return Err(ProjectionFailure::Linear(SolveFailure::NonFinite));
        }
        let divergence_after = divergence
            .iter()
            .zip(volumes)
            .map(|(d, v)| (d / v).abs())
            .fold(0.0_f64, f64::max);
        if !divergence_before.is_finite() || !divergence_after.is_finite() {
            return Err(ProjectionFailure::Linear(SolveFailure::NonFinite));
        }
        for (f, q) in faces.iter_mut().zip(corrected) {
            f.flux = q;
        }
        Ok(ProjectionReport {
            pressure,
            solve,
            divergence_before,
            divergence_after,
        })
    }
}
