//! The two XPBD primitives every constraint is built from.
//!
//! PLAN.md §10: "one mechanism covers contacts, joints, cloth and soft bodies;
//! fewer moving parts to own". This module is that mechanism. A contact is a
//! positional constraint along a normal; a ball joint is three positional
//! constraints; a hinge is a ball joint plus an angular constraint. None of
//! them needs solver code of its own.
//!
//! Both primitives return the multiplier increment they applied, so the caller
//! can accumulate, clamp, and warm-start it.

use fusion_core::BodySet;
use fusion_math::{Mat3, Vec3};

/// Inverse mass as the solver sees it: zero for a static body and zero for a
/// sleeping one, so both are immovable through the same arithmetic rather than
/// through a branch.
#[inline]
pub fn effective_inv_mass(bodies: &BodySet, i: usize) -> f32 {
    if bodies.is_movable(i) {
        bodies.inv_mass[i]
    } else {
        0.0
    }
}

/// `w = 1/m + (r x n) . I^-1 (r x n)` — how far a body moves along `n` per
/// unit impulse applied at offset `r`, rotation included.
#[inline]
pub fn positional_inverse_mass(inv_mass: f32, inv_inertia: &Mat3, r: Vec3, n: Vec3) -> f32 {
    if inv_mass <= 0.0 {
        return 0.0;
    }
    let rn = r.cross(n);
    inv_mass + rn.dot(*inv_inertia * rn)
}

/// `w = n . I^-1 n` — the angular counterpart, for constraints that resist
/// rotation without resisting translation.
#[inline]
pub fn angular_inverse_mass(inv_inertia: &Mat3, n: Vec3) -> f32 {
    n.dot(*inv_inertia * n)
}

/// The correction one constraint solve applied.
#[derive(Clone, Copy, Debug, Default)]
pub struct Applied {
    /// Increment to the constraint's Lagrange multiplier.
    pub delta_lambda: f32,
}

/// The multiplier increment a positional constraint *would* apply, without
/// applying it.
///
/// Constraints with an inequality or a Coulomb bound — contacts, friction,
/// joint limits — have to clamp the increment before it takes effect, so the
/// computation and the application are separate steps.
#[allow(clippy::too_many_arguments)]
pub fn positional_delta(
    bodies: &BodySet,
    a: usize,
    b: usize,
    r_a: Vec3,
    r_b: Vec3,
    direction: Vec3,
    error: f32,
    compliance: f32,
    lambda: f32,
    h: f32,
) -> (f32, f32) {
    let inv_i_a = bodies.inv_inertia_world(a);
    let inv_i_b = bodies.inv_inertia_world(b);
    let w = positional_inverse_mass(effective_inv_mass(bodies, a), &inv_i_a, r_a, direction)
        + positional_inverse_mass(effective_inv_mass(bodies, b), &inv_i_b, r_b, direction);
    if w <= 0.0 {
        return (0.0, 0.0);
    }
    let alpha = compliance / (h * h);
    ((-error - alpha * lambda) / (w + alpha), w)
}

/// Apply an equal and opposite positional impulse to a pair of bodies. `B`
/// receives `+impulse`, `A` receives `-impulse`, which is what makes every
/// constraint in Fusion momentum-conserving by construction.
#[allow(clippy::too_many_arguments)]
pub fn apply_positional_pair(
    bodies: &mut BodySet,
    a: usize,
    b: usize,
    r_a: Vec3,
    r_b: Vec3,
    impulse: Vec3,
    delta_position: &mut [Vec3],
    delta_rotation: &mut [Vec3],
) {
    apply_positional_pair_scaled(
        bodies,
        a,
        b,
        r_a,
        r_b,
        impulse,
        1.0,
        delta_position,
        delta_rotation,
    );
}

/// As [`apply_positional_pair`], but only `record_scale` of the correction is
/// reported to the velocity recovery.
///
/// The distinction exists because a contact correction does two different jobs
/// at once. Undoing the overlap a body just created by moving *is* its
/// collision response, and must become velocity. Undoing overlap that was
/// already there — because detection runs once per step, or because the scene
/// was authored badly — is bookkeeping, and must not.
///
/// Conflating the two is the single most productive bug in this solver. It
/// launches deeply-overlapped bodies at a hundred metres per second, it turns
/// a 6 m/s impact into a 15 m/s rebound, and every attempt to catch it
/// afterwards in the velocity pass fails, because by then the two are the same
/// number.
#[allow(clippy::too_many_arguments)]
pub fn apply_positional_pair_scaled(
    bodies: &mut BodySet,
    a: usize,
    b: usize,
    r_a: Vec3,
    r_b: Vec3,
    impulse: Vec3,
    record_scale: f32,
    delta_position: &mut [Vec3],
    delta_rotation: &mut [Vec3],
) {
    let inv_i_b = bodies.inv_inertia_world(b);
    apply_positional_impulse_scaled(
        bodies,
        b,
        r_b,
        impulse,
        record_scale,
        &inv_i_b,
        delta_position,
        delta_rotation,
    );
    let inv_i_a = bodies.inv_inertia_world(a);
    apply_positional_impulse_scaled(
        bodies,
        a,
        r_a,
        -impulse,
        record_scale,
        &inv_i_a,
        delta_position,
        delta_rotation,
    );
}

/// Apply an equal and opposite velocity impulse to a pair of bodies.
pub fn apply_velocity_pair(
    bodies: &mut BodySet,
    a: usize,
    b: usize,
    r_a: Vec3,
    r_b: Vec3,
    impulse: Vec3,
) {
    let inv_i_b = bodies.inv_inertia_world(b);
    apply_velocity_impulse(bodies, b, r_b, impulse, &inv_i_b);
    let inv_i_a = bodies.inv_inertia_world(a);
    apply_velocity_impulse(bodies, a, r_a, -impulse, &inv_i_a);
}

/// Relative velocity of the two bodies at the given contact offsets, `B`
/// relative to `A`.
#[inline]
pub fn relative_velocity(bodies: &BodySet, a: usize, b: usize, r_a: Vec3, r_b: Vec3) -> Vec3 {
    (bodies.linear_velocity[b] + bodies.angular_velocity[b].cross(r_b))
        - (bodies.linear_velocity[a] + bodies.angular_velocity[a].cross(r_a))
}

/// Solve one scalar positional constraint between two bodies.
///
/// The constraint is `error` metres of violation along the unit `direction`,
/// measured at world offsets `r_a` and `r_b` from the two centres of mass.
/// Positive `error` means the bodies must be drawn *together* along
/// `direction`; the sign convention matches contact separation, where a
/// negative value means overlap.
///
/// `compliance` is the inverse stiffness in m/N. Zero is perfectly rigid, and
/// XPBD stays stable there — that is the whole point of the formulation.
/// Dividing it by `h^2` is what makes the resulting stiffness independent of
/// the timestep.
#[allow(clippy::too_many_arguments)]
pub fn solve_positional(
    bodies: &mut BodySet,
    a: usize,
    b: usize,
    r_a: Vec3,
    r_b: Vec3,
    direction: Vec3,
    error: f32,
    compliance: f32,
    lambda: f32,
    h: f32,
    delta_position: &mut [Vec3],
    delta_rotation: &mut [Vec3],
) -> Applied {
    let inv_i_a = bodies.inv_inertia_world(a);
    let inv_i_b = bodies.inv_inertia_world(b);
    let w_a = positional_inverse_mass(effective_inv_mass(bodies, a), &inv_i_a, r_a, direction);
    let w_b = positional_inverse_mass(effective_inv_mass(bodies, b), &inv_i_b, r_b, direction);
    let w = w_a + w_b;
    if w <= 0.0 {
        return Applied::default();
    }

    let alpha = compliance / (h * h);
    let delta_lambda = (-error - alpha * lambda) / (w + alpha);
    let impulse = direction * delta_lambda;
    apply_positional_impulse(
        bodies,
        b,
        r_b,
        impulse,
        &inv_i_b,
        delta_position,
        delta_rotation,
    );
    apply_positional_impulse(
        bodies,
        a,
        r_a,
        -impulse,
        &inv_i_a,
        delta_position,
        delta_rotation,
    );
    Applied { delta_lambda }
}

/// Solve one angular constraint. `error_axis` is a rotation vector: its
/// direction is the axis to turn about and its length the angle to remove.
#[allow(clippy::too_many_arguments)]
pub fn solve_angular(
    bodies: &mut BodySet,
    a: usize,
    b: usize,
    error_axis: Vec3,
    compliance: f32,
    lambda: f32,
    h: f32,
    delta_rotation: &mut [Vec3],
) -> Applied {
    let (direction, error) = error_axis.normalize_and_length();
    if error < 1e-8 {
        return Applied::default();
    }
    let inv_i_a = bodies.inv_inertia_world(a);
    let inv_i_b = bodies.inv_inertia_world(b);
    let w = angular_inverse_mass(&inv_i_a, direction) + angular_inverse_mass(&inv_i_b, direction);
    if w <= 0.0 {
        return Applied::default();
    }

    let alpha = compliance / (h * h);
    let delta_lambda = (-error - alpha * lambda) / (w + alpha);
    let impulse = direction * delta_lambda;
    apply_angular_impulse(bodies, b, impulse, &inv_i_b, delta_rotation);
    apply_angular_impulse(bodies, a, -impulse, &inv_i_a, delta_rotation);
    Applied { delta_lambda }
}

/// Move and turn a body by a positional impulse applied at `r`, recording the
/// correction so the velocity pass can recover it without differencing
/// absolute positions (see the `integrate` module).
#[inline]
pub fn apply_positional_impulse(
    bodies: &mut BodySet,
    i: usize,
    r: Vec3,
    impulse: Vec3,
    inv_inertia: &Mat3,
    delta_position: &mut [Vec3],
    delta_rotation: &mut [Vec3],
) {
    apply_positional_impulse_scaled(
        bodies,
        i,
        r,
        impulse,
        1.0,
        inv_inertia,
        delta_position,
        delta_rotation,
    );
}

/// Apply a positional impulse to one body, scaling the rotational part.
///
/// The scale is how much of the correction is allowed to become rotation,
/// which is what keeps a contact from spinning a body it should only be
/// pushing.
#[inline]
#[allow(clippy::too_many_arguments)]
pub fn apply_positional_impulse_scaled(
    bodies: &mut BodySet,
    i: usize,
    r: Vec3,
    impulse: Vec3,
    record_scale: f32,
    inv_inertia: &Mat3,
    delta_position: &mut [Vec3],
    delta_rotation: &mut [Vec3],
) {
    if !bodies.is_movable(i) {
        return;
    }
    let inv_mass = bodies.inv_mass[i];
    let translation = impulse * inv_mass;
    let rotation = *inv_inertia * r.cross(impulse);

    bodies.displace(i, translation);
    bodies.orientation[i] = bodies.orientation[i].integrate(rotation, 1.0);
    delta_position[i] += translation * record_scale;
    delta_rotation[i] += rotation * record_scale;
}

/// Turn a body by a pure angular impulse.
#[inline]
pub fn apply_angular_impulse(
    bodies: &mut BodySet,
    i: usize,
    impulse: Vec3,
    inv_inertia: &Mat3,
    delta_rotation: &mut [Vec3],
) {
    if bodies.inv_mass[i] <= 0.0 {
        return;
    }
    let rotation = *inv_inertia * impulse;
    bodies.orientation[i] = bodies.orientation[i].integrate(rotation, 1.0);
    delta_rotation[i] += rotation;
}

/// Apply a velocity-level impulse at `r`.
#[inline]
pub fn apply_velocity_impulse(
    bodies: &mut BodySet,
    i: usize,
    r: Vec3,
    impulse: Vec3,
    inv_inertia: &Mat3,
) {
    if !bodies.is_movable(i) {
        return;
    }
    let inv_mass = bodies.inv_mass[i];
    let delta_w = *inv_inertia * r.cross(impulse);
    bodies.linear_velocity[i] += impulse * inv_mass;
    bodies.angular_velocity[i] += delta_w;
}
