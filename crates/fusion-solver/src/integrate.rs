//! Integration.
//!
//! Semi-implicit (symplectic) Euler: velocity first from the forces, then
//! position from the *new* velocity. It costs the same as explicit Euler and
//! does not pump energy into an orbit the way explicit Euler does, which is
//! why PLAN.md §9 lists it and not forward Euler.
//!
//! ## Why velocities are recovered from corrections, not from positions
//!
//! The textbook XPBD step ends with `v = (x - x_prev) / h`. That is correct
//! algebra and a numerical trap in FP32: it subtracts two nearly equal numbers.
//! At a coordinate of 100 m the f32 spacing is 7.6e-6 m, while one substep of
//! a body starting from rest moves about 1e-5 m — barely one representable
//! step. The difference is then quantized to a few ULPs, and a body dropped at
//! y = 100 falls at 7.0 m/s^2 instead of 9.81.
//!
//! So Fusion never differences absolute positions. The solver accumulates the
//! corrections it applies, and velocity is recovered as
//! `v = v_predicted + correction / h`, which is the same quantity written
//! without the cancellation. It is also cheaper.
//!
//! This is a small instance of the problem PLAN.md §7 raises about coordinate
//! range, arriving at 100 m rather than at 10^21.

use fusion_core::BodySet;
use fusion_math::{Mat3, Vec3};

/// Apply gravity, damping and the gyroscopic term to velocities. `h` is the
/// substep, not the frame.
pub fn integrate_velocities(
    bodies: &mut BodySet,
    gravity: Vec3,
    linear_damping: f32,
    angular_damping: f32,
    h: f32,
    gyroscopic: bool,
) {
    // Implicit damping: `v /= 1 + c*h` rather than `v *= 1 - c*h`. The second
    // form goes negative — and then explodes — for large `c*h`.
    let linear_decay = 1.0 / (1.0 + linear_damping * h);
    let angular_decay = 1.0 / (1.0 + angular_damping * h);

    for i in 0..bodies.len() {
        if !bodies.is_movable(i) {
            continue;
        }
        bodies.linear_velocity[i] = (bodies.linear_velocity[i] + gravity * h) * linear_decay;

        if gyroscopic {
            let orientation = bodies.orientation[i];
            let inertia = bodies.inertia_local[i];
            // Isotropic inertia — every sphere, and every cube — has no
            // gyroscopic torque at all, so skip the work entirely.
            if !is_isotropic(inertia) {
                let body_frame = orientation.rotate_inverse(bodies.angular_velocity[i]);
                let solved = solve_gyroscopic(body_frame, inertia, h);
                bodies.angular_velocity[i] = orientation.rotate(solved);
            }
        }
        bodies.angular_velocity[i] *= angular_decay;
    }
}

#[inline]
fn is_isotropic(inertia: Vec3) -> bool {
    let scale = inertia.max_element();
    if scale <= 0.0 {
        return true;
    }
    let tolerance = scale * 1e-4;
    (inertia.x - inertia.y).abs() < tolerance && (inertia.y - inertia.z).abs() < tolerance
}

/// The gyroscopic term of Euler's equations, integrated implicitly.
///
/// Without torque, it is **angular momentum** that is conserved, not angular
/// velocity: `I_world` turns with the body, so `omega = I_world^-1 L` has to
/// turn too. Carrying `omega` forward unchanged is exactly right for a sphere
/// and exactly wrong for anything else — and the error is not subtle. It is
/// the difference between a thrown book tumbling and a thrown book gliding.
///
/// Solved in the body frame, where the inertia tensor is a constant diagonal:
///
/// ```text
/// f(w2) = I (w2 - w1) + h * w2 x (I w2) = 0
/// ```
///
/// One Newton step is enough at solver timesteps, and being implicit it stays
/// stable at high spin rates where the explicit form `w -= h I^-1 (w x I w)`
/// diverges.
fn solve_gyroscopic(w1: Vec3, inertia: Vec3, h: f32) -> Vec3 {
    let inertia_matrix = Mat3::from_diagonal(inertia);
    let mut w2 = w1;

    let angular_momentum = inertia_matrix * w2;
    let residual = inertia_matrix * (w2 - w1) + w2.cross(angular_momentum) * h;
    let jacobian = inertia_matrix
        + (Mat3::skew_symmetric(w2) * inertia_matrix - Mat3::skew_symmetric(angular_momentum)) * h;

    let correction = jacobian.inverse() * residual;
    w2 -= correction;
    // A singular Jacobian yields a zero inverse by convention, which leaves
    // `w2` untouched — degrading to the previous behaviour rather than to NaN.
    if w2.is_finite() {
        w2
    } else {
        w1
    }
}

/// Advance poses by the current velocities, ignoring every constraint. The
/// constraint solver then corrects the result.
pub fn predict_positions(bodies: &mut BodySet, h: f32) {
    for i in 0..bodies.len() {
        if !bodies.is_movable(i) {
            continue;
        }
        let step = bodies.linear_velocity[i] * h;
        bodies.displace(i, step);
        bodies.orientation[i] = bodies.orientation[i].integrate(bodies.angular_velocity[i], h);
    }
}

/// Fold the constraint solver's positional corrections back into velocity.
///
/// `delta_position` and `delta_rotation` are what the solver moved each body
/// by during this substep — a translation and a rotation vector. Dividing by
/// the substep turns them into the velocity change that the corrections
/// represent, which is what makes XPBD unconditionally stable: a correction
/// can never inject more velocity than the correction itself.
pub fn recover_velocities(
    bodies: &mut BodySet,
    delta_position: &[Vec3],
    delta_rotation: &[Vec3],
    h: f32,
) {
    let inv_h = 1.0 / h;
    for i in 0..bodies.len() {
        if !bodies.is_movable(i) {
            continue;
        }
        bodies.linear_velocity[i] += delta_position[i] * inv_h;
        bodies.angular_velocity[i] += delta_rotation[i] * inv_h;
    }
}
