//! Deformables: cloth and soft bodies.
//!
//! No new solver. A distance constraint between two particles is the same
//! positional constraint a contact uses, with a different error function and
//! no lever arms — particles have no orientation, so the generalized inverse
//! mass collapses to `1/m` and the whole thing gets cheaper. That collapse is
//! why ten thousand particles are affordable and ten thousand rigid bodies are
//! not.
//!
//! What is deliberately *not* here is cloth self-collision. It costs more than
//! everything else in this file combined, it needs its own spatial hash, and
//! draping cloth over geometry — which is what people actually want first —
//! does not need it.

use fusion_collide::narrow::point_vs_shape;
use fusion_collide::{ray_vs_shape, Ray};
use fusion_core::{BendingConstraint, BodySet, DistanceConstraint, ParticleSet};
use fusion_math::{to_f32, Aabb, Isometry, Vec3};

use crate::constraint::apply_velocity_impulse;

/// Gravity and damping on every particle.
pub fn integrate_particles(particles: &mut ParticleSet, gravity: Vec3, damping: f32, h: f32) {
    let decay = 1.0 / (1.0 + damping * h);
    for i in 0..particles.len() {
        if !particles.is_movable(i) {
            continue;
        }
        particles.velocity[i] = (particles.velocity[i] + gravity * h) * decay;
    }
}

/// Advance particles by their velocities, ignoring constraints.
pub fn predict_particles(particles: &mut ParticleSet, h: f32) {
    for i in 0..particles.len() {
        if !particles.is_movable(i) {
            continue;
        }
        let step = particles.velocity[i] * h;
        particles.displace(i, step);
    }
}

/// Recover velocity from the corrections the solve applied.
///
/// Same argument as the rigid path: never difference absolute positions.
pub fn recover_particle_velocities(particles: &mut ParticleSet, delta: &[Vec3], h: f32) {
    let inv_h = 1.0 / h;
    for (i, correction) in delta.iter().enumerate() {
        if !particles.is_movable(i) {
            continue;
        }
        particles.velocity[i] += *correction * inv_h;
    }
}

/// One pass over the distance constraints.
pub fn solve_distance(
    particles: &mut ParticleSet,
    constraints: &[DistanceConstraint],
    lambda: &mut [f32],
    delta: &mut [Vec3],
    h: f32,
) {
    for (index, c) in constraints.iter().enumerate() {
        let (a, b) = (c.a as usize, c.b as usize);
        let w = particles.inv_mass[a] + particles.inv_mass[b];
        if w <= 0.0 {
            continue;
        }
        let offset = particles.relative(a, b);
        let (direction, distance) = offset.normalize_and_length();
        if distance < 1e-9 {
            continue;
        }
        let mut error = distance - c.rest;
        if error < 0.0 {
            // Fabric buckles rather than resisting compression. Scaling the
            // error rather than dropping it keeps the constraint continuous.
            error *= c.compression_scale;
        }
        let alpha = c.compliance / (h * h);
        let delta_lambda = (-error - alpha * lambda[index]) / (w + alpha);
        lambda[index] += delta_lambda;

        let impulse = direction * delta_lambda;
        apply_particle_impulse(particles, b, impulse, delta);
        apply_particle_impulse(particles, a, -impulse, delta);
    }
}

/// One pass over the bending constraints.
pub fn solve_bending(
    particles: &mut ParticleSet,
    constraints: &[BendingConstraint],
    lambda: &mut [f32],
    delta: &mut [Vec3],
    h: f32,
) {
    for (index, c) in constraints.iter().enumerate() {
        let (a, b) = (c.a as usize, c.b as usize);
        let w = particles.inv_mass[a] + particles.inv_mass[b];
        if w <= 0.0 {
            continue;
        }
        let offset = particles.relative(a, b);
        let (direction, distance) = offset.normalize_and_length();
        if distance < 1e-9 {
            continue;
        }
        let error = distance - c.rest;
        let alpha = c.compliance / (h * h);
        let delta_lambda = (-error - alpha * lambda[index]) / (w + alpha);
        lambda[index] += delta_lambda;

        let impulse = direction * delta_lambda;
        apply_particle_impulse(particles, b, impulse, delta);
        apply_particle_impulse(particles, a, -impulse, delta);
    }
}

#[inline]
fn apply_particle_impulse(
    particles: &mut ParticleSet,
    i: usize,
    impulse: Vec3,
    delta: &mut [Vec3],
) {
    let inv_mass = particles.inv_mass[i];
    if inv_mass <= 0.0 {
        return;
    }
    let translation = impulse * inv_mass;
    particles.displace(i, translation);
    delta[i] += translation;
}

/// Where a particle travelling from `start` to `end` (both in the body's
/// frame, centred on it) first touched `shape`, one radius short — or `None`
/// where its path did not enter it from outside. See [`sweep_particles`].
fn entry(
    start: Vec3,
    end: Vec3,
    radius: f32,
    shape: &fusion_core::Shape,
    pose: &Isometry,
) -> Option<Vec3> {
    let travel = end - start;
    let length = travel.length();
    if length <= 1e-9 || point_vs_shape(start, 0.0, shape, pose, 0.0).is_some() {
        return None;
    }
    let ray = Ray::new(start, travel);
    let t = ray_vs_shape(&ray, shape, pose)?;
    (t < length).then(|| start + ray.direction * (t - radius).max(0.0))
}

/// Catch the particles whose path this substep crossed into a body, and put
/// each back just short of where it went in. Call once a substep, after
/// [`predict_particles`] and before the iterations; returns how many it
/// caught.
///
/// The contact pass sees where a particle *is*, not how it got there. A
/// particle that crossed a 1 cm pane in one substep is found on the far
/// side, clear, and nothing stops it; one that went more than halfway in is
/// pushed out through the far face, which is nearer. Water through a thin
/// wall and cloth through a fast paddle were both that. So the path is
/// rebuilt in the body's frame — the particle at the start of the substep
/// against where the body was then — and cast against the shape.
///
/// Left alone: a particle that moved less than its radius, which the contact
/// pass sees in time; one that started inside, which is the contact pass's
/// to push out; and one ending in contact that the contact pass would push
/// back the way it came. Sliding along a surface is never caught.
///
/// The body takes the particle's push back, as a contact gives it.
// One index into the positions, velocities, radii and corrections alike.
#[allow(clippy::too_many_arguments, clippy::needless_range_loop)]
pub fn sweep_particles(
    particles: &mut ParticleSet,
    bodies: &mut BodySet,
    delta: &mut [Vec3],
    body_delta_position: &mut [Vec3],
    body_delta_rotation: &mut [Vec3],
    h: f32,
) -> usize {
    let mut caught = 0;
    for body in 0..bodies.len() {
        if bodies.is_removed(body) {
            continue;
        }
        let moving = bodies.is_movable(body) || bodies.is_kinematic(body);
        let (v_body, w_body) = if moving {
            (bodies.linear_velocity[body], bodies.angular_velocity[body])
        } else {
            (Vec3::ZERO, Vec3::ZERO)
        };
        let rotation = bodies.orientation[body];
        let before = rotation.integrate(-w_body * h, 1.0).normalize();
        // The particle's start, as seen from the body now: where it was
        // relative to the body then, turned with the body since.
        let carry = rotation * before.inverse();
        let mut pose = Isometry::new(Vec3::ZERO, rotation);
        let mut bounds = bodies.shape[body].world_aabb(&pose);
        let inv_mass_body = if bodies.is_dynamic(body) {
            bodies.inv_mass[body]
        } else {
            0.0
        };
        for i in 0..particles.len() {
            if !particles.is_movable(i) {
                continue;
            }
            let radius = particles.radius[i];
            let end = to_f32(particles.position[i] - bodies.position[body]);
            let start = carry * (end - particles.velocity[i] * h + v_body * h);
            let travel = end - start;
            let length = travel.length();
            if length <= radius {
                continue;
            }
            let swept = Aabb {
                min: start.min(end),
                max: start.max(end),
            }
            .expanded(radius);
            if !bounds.overlaps(&swept) {
                continue;
            }
            let shape = &bodies.shape[body];
            if let Some((normal, _)) = point_vs_shape(end, radius, shape, &pose, 0.0) {
                if normal.dot(travel) <= 0.0 {
                    continue;
                }
            }
            let Some(target) = entry(start, end, radius, shape, &pose) else {
                continue;
            };
            let correction = target - end;
            particles.displace(i, correction);
            delta[i] += correction;
            caught += 1;
            if inv_mass_body > 0.0 && particles.inv_mass[i] > 0.0 {
                let reaction = -correction / particles.inv_mass[i];
                let lever = end;
                let inv_i = bodies.inv_inertia_world(body);
                let translation = reaction * inv_mass_body;
                let turn = inv_i * lever.cross(reaction);
                bodies.wake(body);
                bodies.position[body] += fusion_math::to_f64(translation);
                bodies.orientation[body] = bodies.orientation[body].integrate(turn, 1.0);
                body_delta_position[body] += translation;
                body_delta_rotation[body] += turn;
                pose.rotation = bodies.orientation[body];
                bounds = bodies.shape[body].world_aabb(&pose);
            }
        }
    }
    caught
}

/// Push particles out of rigid bodies, and push the bodies back.
///
/// Two-way by construction: the impulse that moves a particle is applied in
/// reverse to the body at the contact point, so a heavy curtain does swing a
/// light rod. Making it one-way would be a line shorter and would quietly
/// break conservation of momentum, which the diagnostics would then report as
/// a mystery.
///
/// The body's share of the correction is recorded into `body_delta_position`
/// and `body_delta_rotation`, which is what turns it into velocity when
/// `recover_velocities` runs. Recording it is not optional bookkeeping —
/// **buoyancy is entirely this**. Fusion has no buoyancy model and should not
/// have one: a floating body is a body being pushed up by more fluid pressure
/// underneath it than above. If the push moves the body but never becomes
/// velocity, gravity wins every substep and a cork sinks. It did, until this
/// was recorded; the cork now floats at the draft its density implies, with no
/// code anywhere that knows what buoyancy is.
/// Returns the number of particle-body contacts it found, which is what
/// `StepReport::particle_contacts` reports. Counting here rather than in a
/// separate pass is the only cheap place to do it: the rejection test that
/// decides whether a particle touches a body is this loop, and repeating it
/// for a diagnostic would cost a second sweep over every particle.
#[allow(clippy::too_many_arguments)]
pub fn solve_particle_body_contacts(
    particles: &mut ParticleSet,
    bodies: &mut BodySet,
    delta: &mut [Vec3],
    body_delta_position: &mut [Vec3],
    body_delta_rotation: &mut [Vec3],
    tolerance: f32,
    friction: f32,
    h: f32,
) -> usize {
    let mut touching = 0;
    for body in 0..bodies.len() {
        if bodies.is_removed(body) {
            continue;
        }
        // Work in a body-relative frame: subtract positions in f64 before
        // narrowing to collision precision. Refresh the frame after reactions.
        let mut pose = Isometry::new(Vec3::ZERO, bodies.orientation[body]);
        let mut bounds = bodies.shape[body].world_aabb(&pose);
        let inv_mass_body = if bodies.is_dynamic(body) {
            bodies.inv_mass[body]
        } else {
            0.0
        };
        let moving = bodies.is_movable(body) || bodies.is_kinematic(body);
        let (v_body, w_body) = if moving {
            (bodies.linear_velocity[body], bodies.angular_velocity[body])
        } else {
            (Vec3::ZERO, Vec3::ZERO)
        };
        let carry = bodies.orientation[body]
            * bodies.orientation[body]
                .integrate(-w_body * h, 1.0)
                .normalize()
                .inverse();

        for i in 0..particles.len() {
            if !particles.is_movable(i) {
                continue;
            }
            let radius = particles.radius[i];
            let point = to_f32(particles.position[i] - bodies.position[body]);
            // Cheap rejection before the shape test. Bodies are few and
            // particles are many, so this loop order keeps the pose and bounds
            // in registers across the whole particle sweep.
            //
            // Expanded by the tolerance as well as the radius, because a
            // rejection tighter than the test it feeds is a rejection that
            // hides answers. `point_vs_shape` reports any contact closer than
            // `tolerance`; a particle resting on a face sits at a separation of
            // about 1e-9, which is a nanometre *outside* a box expanded by the
            // radius alone. Nothing was corrected wrongly — a resting particle
            // needs no correction — but nothing could see it either, and
            // `particle_contacts` reported an empty floor under a settled pool.
            if !bounds
                .expanded(radius + tolerance.max(0.0))
                .contains_point(point)
            {
                continue;
            }
            let Some((normal, separation)) =
                point_vs_shape(point, radius, &bodies.shape[body], &pose, tolerance)
            else {
                continue;
            };
            // Counted here, where the particle is *touching* — within the
            // slop — rather than below, where it is overlapping enough to be
            // worth a correction. Those are different numbers, and the first
            // is the one that matches what `contact_points` reports for rigid
            // bodies: a detected contact, whether or not this iteration had
            // anything left to fix. Counting corrections instead reports zero
            // for a pool that has settled, which is precisely when a person
            // looking at the number wants to see it standing on the floor.
            touching += 1;
            if separation >= 0.0 {
                continue;
            }

            // Out the side it came in by. The nearest face of a thin body is
            // the far one once a particle is past its middle — pushed there
            // by its neighbours in the cloth, or by its own speed — and out
            // through it is through the pane. Where the particle was at the
            // start of the substep says which side it belongs on.
            let (normal, separation) = {
                let start = carry * (point - particles.velocity[i] * h - delta[i] + v_body * h);
                match (normal.dot(point - start) > 0.0)
                    .then(|| entry(start, point, radius, &bodies.shape[body], &pose))
                    .flatten()
                {
                    Some(target) => {
                        // Out towards the entry, as far as it is.
                        let (direction, distance) = (target - point).normalize_and_length();
                        (direction, -distance)
                    }
                    None => (normal, separation),
                }
            };

            let lever = point - pose.translation;
            let inv_i = bodies.inv_inertia_world(body);
            let rn = lever.cross(normal);
            let w_body = inv_mass_body + rn.dot(inv_i * rn);
            let w = particles.inv_mass[i] + w_body;
            if w <= 0.0 {
                continue;
            }

            let magnitude = -separation / w;
            let impulse = normal * magnitude;
            apply_particle_impulse(particles, i, impulse, delta);
            if inv_mass_body > 0.0 {
                let reaction = -impulse;
                let translation = reaction * inv_mass_body;
                let rotation = inv_i * lever.cross(reaction);
                bodies.wake(body);
                bodies.position[body] += fusion_math::to_f64(translation);
                bodies.orientation[body] = bodies.orientation[body].integrate(rotation, 1.0);
                body_delta_position[body] += translation;
                body_delta_rotation[body] += rotation;
            }

            // Friction, at the velocity level: remove the tangential motion a
            // Coulomb cone would not permit. Cloth that slides freely over
            // everything looks like plastic sheeting.
            if friction > 0.0 {
                let surface =
                    bodies.linear_velocity[body] + bodies.angular_velocity[body].cross(lever);
                let relative = particles.velocity[i] - surface;
                let tangential = relative.reject_from_unit(normal);
                let (direction, speed) = tangential.normalize_and_length();
                if speed > 1e-6 {
                    let normal_speed = relative.dot(normal).abs();
                    let removable = speed.min(friction * normal_speed.max(speed * 0.5));
                    particles.velocity[i] -= direction * removable;
                    if inv_mass_body > 0.0 {
                        let restore = direction * (removable / particles.inv_mass[i].max(1e-6));
                        apply_velocity_impulse(bodies, body, lever, restore, &inv_i);
                    }
                }
            }
            if inv_mass_body > 0.0 {
                pose.rotation = bodies.orientation[body];
                bounds = bodies.shape[body].world_aabb(&pose);
            }
        }
    }
    touching
}
