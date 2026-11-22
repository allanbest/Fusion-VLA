//! Constitutive models: the one thing that differs between materials.
//!
//! PLAN.md §11: "A steel bar and a puddle of water differ only in their
//! constitutive law." Everything else in this crate is the same code for every
//! material — the transfers, the grid solve, the boundary handling. This file
//! is the difference, and it is two functions: what stress a deformation
//! implies, and what part of that deformation the material cannot come back
//! from.
//!
//! ## Stress, and which one
//!
//! [`stress`] returns the **first Piola–Kirchhoff** stress `P`, not the Cauchy
//! stress. The distinction matters and gets glossed: `P` is measured against
//! the material's *rest* configuration, Cauchy against its current one, and
//! MLS-MPM's force term wants the former because the volume it multiplies by
//! is the rest volume. Mixing them gives a solid whose stiffness drifts with
//! its own compression, which looks like a plausible material and is not one.

use fusion_core::solid::Constitutive;
use fusion_math::{Mat3, Vec3};

/// The principal stresses of a deformation, given its singular values.
///
/// Every model in this file is isotropic, which has a consequence worth
/// exploiting rather than merely noting: an isotropic stress shares its
/// principal directions with the deformation, so `P = U diag(p) Vᵀ` where `U`
/// and `V` come from the SVD of `F` and `p` depends only on the singular
/// values. Fixed corotated is the clearest case —
///
/// ```text
///   P = 2μ(F − R) + λ(J−1)J F⁻ᵀ
///     = 2μ(U Σ Vᵀ − U Vᵀ) + λ(J−1)J U Σ⁻¹ Vᵀ
///     = U [2μ(Σ − I) + λ(J−1)J Σ⁻¹] Vᵀ
/// ```
///
/// — and the bracket is this function.
///
/// Three things fall out of writing it this way rather than as matrices. The
/// polar decomposition and the matrix inverse both disappear, since `R` and
/// `F⁻ᵀ` are `U Vᵀ` and `U Σ⁻¹ Vᵀ`. The **stress measure a damage model needs**
/// is right here rather than requiring an eigendecomposition afterwards. And
/// damage can be applied to the tensile principal stresses alone, which is the
/// difference between a crack and a hole: a crack carries no tension across it
/// and still transmits compression when it closes.
pub fn principal_stress(model: Constitutive, sigma: Vec3, mu: f32, lambda: f32) -> Vec3 {
    let j = sigma.x * sigma.y * sigma.z;
    // A singular value at zero is a point squashed flat, where `1/σ` is
    // infinite and the model has no answer. Clamped to a millionth, which
    // leaves the volumetric term enormous and restoring — which is what should
    // happen to a point that has been flattened.
    let inverse = Vec3::new(
        1.0 / sigma.x.abs().max(1e-6) * sigma.x.signum(),
        1.0 / sigma.y.abs().max(1e-6) * sigma.y.signum(),
        1.0 / sigma.z.abs().max(1e-6) * sigma.z.signum(),
    );
    match model {
        Constitutive::FixedCorotated | Constitutive::Snow { .. } => {
            let volumetric = lambda * (j - 1.0) * j;
            Vec3::new(
                2.0 * mu * (sigma.x - 1.0) + volumetric * inverse.x,
                2.0 * mu * (sigma.y - 1.0) + volumetric * inverse.y,
                2.0 * mu * (sigma.z - 1.0) + volumetric * inverse.z,
            )
        }
        // Von Mises yields in shear only, so what is left after its return
        // map is a small-shear elastic deformation of any volume: the
        // neo-Hookean law, which resists being crushed, holds it.
        Constitutive::NeoHookean | Constitutive::VonMises { .. } => {
            // A point that has inverted has `J <= 0` and no logarithm. The
            // model is genuinely undefined there rather than merely awkward,
            // so it is clamped to the smallest positive value the volumetric
            // term can still push back from — which it does, hard, because
            // `ln` of something tiny is a large negative number and the term
            // is a restoring one.
            let volumetric = lambda * j.max(1e-4).ln();
            Vec3::new(
                mu * (sigma.x - inverse.x) + volumetric * inverse.x,
                mu * (sigma.y - inverse.y) + volumetric * inverse.y,
                mu * (sigma.z - inverse.z) + volumetric * inverse.z,
            )
        }
    }
}

/// The strain energy a deformation stores, per unit rest volume, J/m³.
///
/// The potential the stress is the gradient of: `P = dPsi/dF`, and in
/// principal form `p_i = dPsi/dsigma_i`, which is the relationship
/// [`crate::principal_stress`] and this function are two halves of. Writing
/// both from the same principal values is what keeps them the same model —
/// the test suite differentiates this one numerically and compares it against
/// that one, which is a check no amount of reading the two functions gives.
///
/// It exists because a continuum's energy is not its kinetic plus its
/// potential. A block dropped on the floor turns all of its gravitational
/// energy into *this*, hands it back on the way up, and an energy ledger that
/// cannot see the term watches a solid create and destroy joules from nothing.
pub fn energy_density(model: Constitutive, sigma: Vec3, mu: f32, lambda: f32) -> f32 {
    let j = sigma.x * sigma.y * sigma.z;
    match model {
        Constitutive::FixedCorotated | Constitutive::Snow { .. } => {
            let deviatoric = (sigma.x - 1.0) * (sigma.x - 1.0)
                + (sigma.y - 1.0) * (sigma.y - 1.0)
                + (sigma.z - 1.0) * (sigma.z - 1.0);
            mu * deviatoric + 0.5 * lambda * (j - 1.0) * (j - 1.0)
        }
        Constitutive::NeoHookean | Constitutive::VonMises { .. } => {
            // The same clamp `principal_stress` puts on `J`, for the same
            // reason: an inverted point has no logarithm, and the energy of
            // one has to stay large and positive rather than becoming NaN.
            let log_j = j.max(1e-4).ln();
            let first_invariant = sigma.x * sigma.x + sigma.y * sigma.y + sigma.z * sigma.z - 3.0;
            0.5 * mu * first_invariant - mu * log_j + 0.5 * lambda * log_j * log_j
        }
    }
}

/// First Piola–Kirchhoff stress for a deformation gradient.
///
/// `mu` and `lambda` are the Lamé parameters *after* whatever hardening the
/// model applies, so this function has no memory and no state — the same `f`
/// and the same parameters give the same answer, which is what lets it be
/// called from anywhere including a test.
pub fn stress(model: Constitutive, f: Mat3, mu: f32, lambda: f32) -> Mat3 {
    // Neo-Hookean in matrix form, `μ (F − F⁻ᵀ) + λ ln J F⁻ᵀ`, is the
    // principal form below exactly wherever every stretch is positive — and
    // needs no decomposition. The decomposition was the single most expensive
    // thing in a step of four jelly blocks, more than moving their points to
    // the grid and back. A point near flat or inverted still takes the
    // principal path, which clamps what the matrix form cannot.
    if matches!(
        model,
        Constitutive::NeoHookean | Constitutive::VonMises { .. }
    ) {
        let j = f.determinant();
        if j >= HEALTHY {
            let inverse_t = f.inverse().transpose();
            return (f - inverse_t) * mu + inverse_t * (lambda * j.ln());
        }
    }
    let (u, sigma, v) = f.svd();
    assemble(u, principal_stress(model, sigma, mu, lambda), v)
}

/// The strain energy per unit rest volume at deformation `f`: as
/// [`energy_density`], without a decomposition where the model allows.
pub fn energy_at(model: Constitutive, f: Mat3, mu: f32, lambda: f32) -> f32 {
    if matches!(
        model,
        Constitutive::NeoHookean | Constitutive::VonMises { .. }
    ) {
        let j = f.determinant();
        if j >= HEALTHY {
            let log_j = j.ln();
            let first_invariant = f.double_dot(&f) - 3.0;
            return 0.5 * mu * first_invariant - mu * log_j + 0.5 * lambda * log_j * log_j;
        }
    }
    let (_, sigma, _) = f.svd();
    energy_density(model, sigma, mu, lambda)
}

/// A volume ratio above which the matrix forms are used: well clear of the
/// clamps the principal forms put on near-flat points.
const HEALTHY: f32 = 1e-3;

/// The same, with a damage scalar applied to the tensile part.
///
/// `damage` runs from zero (intact) to one (parted). It multiplies the
/// principal stresses that are **positive** — the ones pulling the material
/// apart — and leaves the negative ones alone.
///
/// That asymmetry is the whole model. Scaling the stress uniformly would give
/// a broken point that also stops resisting compression, so a fractured solid
/// would collapse into itself rather than falling into pieces that pile up:
/// the crack would be a hole rather than a crack. Real cracked material carries
/// compression across a closed crack perfectly well, and the discontinuity is
/// in tension only.
pub fn stress_with_damage(model: Constitutive, f: Mat3, mu: f32, lambda: f32, damage: f32) -> Mat3 {
    let (u, sigma, v) = f.svd();
    let mut p = principal_stress(model, sigma, mu, lambda);
    let intact = (1.0 - damage.clamp(0.0, 1.0)).max(RESIDUAL);
    for axis in 0..3 {
        if p[axis] > 0.0 {
            p[axis] *= intact;
        }
    }
    assemble(u, p, v)
}

/// Tensile stiffness a fully parted point keeps.
///
/// Not zero, and the difference is the difference between a fracture model and
/// a hole in the simulation. A point with no tensile stiffness at all has
/// nothing stopping it stretching: its deformation gradient grows without
/// bound, it drifts away from its neighbours, and there is no restoring force
/// anywhere in the model to notice. Measured before this constant existed, a
/// shattered block's fragments were spread over fifteen million metres.
///
/// Two percent is small enough that a parted point reads as separated — a
/// fiftieth of the stiffness will not hold a fragment against gravity — and
/// large enough to keep `F` bounded. It plays the role a cohesive zone plays
/// in a fracture model that has one.
const RESIDUAL: f32 = 0.02;

/// `U diag(p) Vᵀ`, without forming the diagonal matrix.
#[inline]
fn assemble(u: Mat3, p: Vec3, v: Mat3) -> Mat3 {
    Mat3::from_cols(u.x_axis * p.x, u.y_axis * p.y, u.z_axis * p.z) * v.transpose()
}

/// The Lamé parameters this material point currently has, after hardening.
///
/// Only the plastic models use `plastic`; the elastic ones ignore it and
/// return what they were given. Written as one function rather than a branch
/// at each call site because "which materials harden" is a property of the
/// model and belongs next to the model.
#[inline]
pub fn hardened(model: Constitutive, mu: f32, lambda: f32, plastic: f32) -> (f32, f32) {
    match model {
        Constitutive::Snow { hardening, .. } => {
            // Packed snow is stiffer than fresh snow. `plastic` is the volume
            // ratio the material has permanently lost, so it falls below one
            // as the point compacts and the exponent rises.
            let scale = (hardening * (1.0 - plastic)).exp().clamp(0.05, 20.0);
            (mu * scale, lambda * scale)
        }
        _ => (mu, lambda),
    }
}

/// Split a deformation into the part the material keeps and the part it does
/// not, returning the elastic remainder and the plastic volume ratio to
/// multiply into the point's own.
///
/// Elastic models return `(f, 1.0)` — they come back from everything, which is
/// what elastic means.
///
/// ## Why the clamp is on singular values and not on the matrix
///
/// The yield condition is about *stretch*, and stretch is what the singular
/// values are. Clamping entries of `F` would clamp a rotation, so a snowball
/// that had merely turned around would be told it had yielded. This is the
/// second place in v0.8 where the SVD is not optional.
pub fn return_map(model: Constitutive, f: Mat3) -> (Mat3, f32) {
    match model {
        Constitutive::Snow {
            critical_compression,
            critical_stretch,
            ..
        } => {
            let (u, sigma, v) = f.svd();
            let low = (1.0 - critical_compression).max(1e-3);
            let high = 1.0 + critical_stretch;
            let clamped = fusion_math::Vec3::new(
                sigma.x.clamp(low, high),
                sigma.y.clamp(low, high),
                sigma.z.clamp(low, high),
            );
            let before = sigma.x * sigma.y * sigma.z;
            let after = clamped.x * clamped.y * clamped.z;
            let elastic = u * Mat3::from_diagonal(clamped) * v.transpose();
            // The ratio the point has given up permanently. One where nothing
            // was clamped, which is the common case and costs an SVD to find
            // out — the price of a model that yields only where it should.
            let plastic = if after.abs() > 1e-6 {
                (before / after).clamp(0.05, 20.0)
            } else {
                1.0
            };
            (elastic, plastic)
        }
        Constitutive::VonMises { yield_strain } => {
            // In logarithmic strain, where the volumetric and deviatoric
            // parts separate exactly: the mean is the volume change and is
            // left alone, and the rest is pulled back onto the yield surface
            // if it is outside it. Exponentiated back, the point keeps its
            // volume and gives up only shape.
            let (u, sigma, v) = f.svd();
            let magnitude = sigma.abs().max(Vec3::splat(MIN_STRETCH));
            let strain = Vec3::new(magnitude.x.ln(), magnitude.y.ln(), magnitude.z.ln());
            let mean = (strain.x + strain.y + strain.z) / 3.0;
            let deviatoric = strain - Vec3::splat(mean);
            let excess = deviatoric.length();
            let radius = yield_strain.max(1e-5);
            if excess <= radius {
                return (f, 1.0);
            }
            let kept = deviatoric * (radius / excess) + Vec3::splat(mean);
            let clamped = Vec3::new(
                kept.x.exp() * sigma.x.signum(),
                kept.y.exp() * sigma.y.signum(),
                kept.z.exp() * sigma.z.signum(),
            );
            let elastic = u * Mat3::from_diagonal(clamped) * v.transpose();
            // Not a volume ratio here — volume is what this law keeps — but
            // the plastic shear given up, as a factor, so that "how much has
            // yielded" reads the same way for every plastic law.
            (elastic, 1.0 + (excess - radius))
        }
        _ => (f, 1.0),
    }
}

/// The least stretch a material point is allowed to keep along any axis.
///
/// A hundredth, and the number was chosen by measuring what a larger one does
/// rather than by taste. The floor exists to keep `det F` **positive**, not to
/// hold material up: at a twentieth it also stopped a fixed-corotated block
/// from being crushed past its softening branch, which is a real defect of that
/// model that the test suite documents deliberately, and a guard that quietly
/// repairs a modelling error is worse than the error. At a hundredth the
/// corotated block collapses exactly as it did before and the inversions are
/// still gone.
///
/// A point squashed to a hundredth of its rest length along one axis is already
/// a hundred times past anything a solid comes back from. Nothing physical is
/// being clamped away here; what is being clamped away is the sign flip.
pub const MIN_STRETCH: f32 = 0.01;

/// Pull a deformation back from inversion, or `None` if it never went there.
///
/// ## What goes wrong without it
///
/// The deformation update is `F ← (I + h ∇v) F`, and nothing in that expression
/// knows what a solid is. Where the velocity gradient is large enough that
/// `I + h ∇v` is itself singular or worse, `det F` passes through zero and
/// comes out **negative** — the material point has been turned inside out.
/// Both models here have a term in `1/σ`, and at that point they are answering
/// a question about a configuration that does not exist: measured on a block of
/// 300 Pa jelly dropped on a slab, `min det F` reached `-0.000` and the solid's
/// total energy rose by 77 J out of nothing, which is the signature of a force
/// that is no longer the gradient of any potential.
///
/// ## Why a clamp and not a smaller step
///
/// A smaller step is the honest fix and it is not available: the strain rate
/// that inverts a point is known only once the substep has run, so a substep
/// count chosen ahead of time can bound the wave speed and the advection —
/// which [`crate::MpmSolver::substeps_for`] does — and cannot bound this. The
/// clamp is a floor under an answer that would otherwise be meaningless, in
/// the same spirit as the plastic return map: it **changes the state**, so the
/// solver counts every point it touches and reports the count rather than
/// quietly repairing the material.
///
/// The rotations are kept and only the stretches are clamped, for the reason
/// [`return_map`] gives: clamping entries of `F` would clamp a rotation.
pub fn uninvert(f: Mat3) -> Option<Mat3> {
    // The cheap gate, and it is exact rather than heuristic. If some singular
    // value is below `MIN_STRETCH` then
    //
    //   |det F| = σ₁σ₂σ₃ < MIN · σ₁σ₂ ≤ MIN · (σ₁² + σ₂²)/2 ≤ MIN · ‖F‖²/2
    //
    // so a determinant at or above `MIN · ‖F‖²/2` proves every singular value
    // is at or above `MIN`, with no SVD taken. Points that need the clamp are
    // rare and points that do not are every point every substep, which is what
    // makes the direction of this test the right way round.
    let det = f.determinant();
    if det >= MIN_STRETCH * f.double_dot(&f) * 0.5 {
        return None;
    }
    let (u, sigma, v) = f.svd();
    let clamped = Vec3::new(
        sigma.x.max(MIN_STRETCH),
        sigma.y.max(MIN_STRETCH),
        sigma.z.max(MIN_STRETCH),
    );
    if clamped == sigma {
        return None;
    }
    Some(u * Mat3::from_diagonal(clamped) * v.transpose())
}
