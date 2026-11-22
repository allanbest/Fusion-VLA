//! Contact solving: non-penetration, friction, and restitution.
//!
//! PLAN.md §10 is specific about where friction goes — "Coulomb cone projected
//! inside the same XPBD loop" — and about why: keeping position and velocity
//! corrections from fighting each other is a classic source of phantom energy.
//! So static friction is a *positional* constraint solved beside the normal
//! constraint, and only the sliding part is handled at the velocity level.
//!
//! The split matters physically as well as numerically. Static friction is a
//! constraint — the surfaces are stuck, and the contact point should not move
//! at all — while dynamic friction is a force, bounded by the Coulomb cone.
//! A solver that models both as velocity damping cannot hold a box on a slope;
//! it can only make it slide slowly.

use fusion_collide::contact::MAX_CONTACT_POINTS;
use fusion_collide::ContactManifold;
use fusion_core::BodySet;
use fusion_math::{Isometry, Vec3};

use crate::constraint::{
    apply_positional_impulse_scaled, apply_positional_pair, apply_positional_pair_scaled,
    apply_velocity_pair, effective_inv_mass, positional_delta, positional_inverse_mass,
    relative_velocity,
};

/// Lever arm from a body's centre of mass to a contact anchor.
///
/// Just the rotated local anchor. Computing it as `world_point - centre` would
/// build a large number and immediately subtract it away again, and in f32
/// that is not free: at a coordinate of 500 m the spacing is 3e-5 m, so a
/// 0.1 mm tangential slip — exactly the quantity static friction measures —
/// survives the round trip as noise.
#[inline]
fn lever(pose: &Isometry, local: Vec3) -> Vec3 {
    pose.rotation.rotate(local)
}

/// Offset from anchor A to anchor B, formed without ever building either
/// anchor's absolute position.
///
/// The centres are differenced in double precision by
/// [`BodySet::relative_translation`] and the lever arms are small and exact,
/// so the result carries full precision no matter where in the world the pair
/// happens to be. The obvious spelling — `world_b - world_a` — rounds each
/// anchor to the spacing of the world coordinate first, and throws away the
/// fine structure before the subtraction can use it.
#[inline]
fn anchor_offset(bodies: &BodySet, a: usize, b: usize, r_a: Vec3, r_b: Vec3) -> Vec3 {
    bodies.relative_translation(a, b) + (r_b - r_a)
}

/// Per-contact-point solver state. Lives in the working set, not in the
/// contact geometry.
#[derive(Clone, Copy, Debug, Default)]
pub struct ContactScratch {
    /// Accumulated normal multiplier for the current substep.
    pub normal_lambda: f32,
    /// Accumulated magnitude of the static-friction multiplier, bounded by
    /// the Coulomb cone at `friction * normal_lambda`.
    pub tangent_lambda: f32,

    /// Relative normal velocity at the start of the substep, before anything
    /// was solved. Negative while closing. Restitution is defined against
    /// this, and it has to be captured before the position solve overwrites
    /// the velocity that produced it.
    pub approach_velocity: f32,
    /// Metres of separation already credited to the velocity recovery this
    /// substep. Bounded by how far the bodies actually closed, so that fixing
    /// old overlap never counts as motion.
    pub recorded_separation: f32,
    /// The part of [`Self::normal_lambda`] that went on *tidying* rather than
    /// on stopping anything — pushing a body out of an overlap it should not
    /// have been in.
    ///
    /// The position solve already draws this line for the velocity credit:
    /// "everything beyond that is pre-existing overlap being tidied up, and
    /// tidying is not a collision". The Coulomb cone was budgeted from the
    /// whole multiplier anyway, so a landing that had to be pushed 3 mm out of
    /// the floor handed friction the tidying as well.
    ///
    /// Measured, a 1 m box crossing a floor at 6 m/s and landing from 5 cm up:
    /// it left the impact at **3.1 m/s** where Coulomb allows 5.5, and stopped
    /// dead in a metre. That is what made a thrown box feel like it had landed
    /// in sand.
    ///
    /// Accumulated from the iteration corrections only. The warm start applies
    /// the multiplier the *previous* substep converged to — a load already
    /// classified, and a real one, since a resting contact's `normal_lambda`
    /// is `m g h^2` to the digit — so re-classifying it here would count the
    /// same judgement twice and leave a resting box with almost no cone at
    /// all. Measured that way it slid for ever.
    pub tidied_lambda: f32,
    /// Whether this contact took a real impact at any point during the step.
    ///
    /// A fact about the step rather than the substep, like
    /// [`Self::speculative_impact`]: the substep that catches a landing is the
    /// one that sees the approach velocity, and the depenetration it starts
    /// goes on being paid off by the substeps after it. Read the flag per
    /// substep and only the first of eight has its cone corrected.
    pub impacting: bool,
    /// Anchor-to-anchor offset at the start of the substep.
    ///
    /// Captured and not read by static friction, which holds a contact to
    /// [`Self::stick_local_a`] instead. Measuring slip against this — since the
    /// substep — was tried: it removed the settled-box creep and let a ten-box
    /// stack's drift random-walk, +42 J and 88 mm over ten settled seconds,
    /// because a reference that forgets every substep never pulls accumulated
    /// drift back.
    pub previous_offset: Vec3,
    /// Approach speed a speculative clamp took off this contact, kept across
    /// substeps.
    ///
    /// Speculative contact and restitution work against each other: the clamp
    /// slows a body smoothly to zero approach speed as it reaches the surface,
    /// so by the time the surfaces actually touch there is no impact left to
    /// bounce from, and a ball with restitution 0.8 returns to a quarter of
    /// the height it should. Remembering what the clamp removed lets the
    /// bounce be computed from the speed the body really arrived with.
    pub speculative_impact: f32,
    /// Friction impulse already applied by the velocity pass this substep.
    /// The Coulomb bound is a budget for the substep as a whole, not a fresh
    /// allowance for each sweep — with two sweeps and no accounting, a sliding
    /// block decelerates at exactly twice `mu * g`.
    pub applied_friction: f32,
    /// The normal multiplier this contact converged to last substep, or —
    /// on the first substep of a step — last *step*, carried across by
    /// `XpbdWorkingSet`'s cache.
    ///
    /// This is the warm start. See [`warm_start`] for what is done with it and
    /// why a contact solved from zero every substep is the expensive way to
    /// get the same answer.
    pub warm_normal_lambda: f32,
    /// Where this contact last stuck: a point on body A, in A's frame, that
    /// static friction holds B's anchor to.
    ///
    /// Re-seeded to B's anchor every substep the surfaces slide, frozen once
    /// they stick, and carried across steps in the solver's contact cache. The
    /// two references it replaces each failed one way. Measured *since the
    /// contact was born*, a box that slid two metres and stopped was hauled
    /// back towards where it first touched — the settled-box creep, 150 mm in
    /// three seconds. Measured *since the substep*, nothing remembered
    /// anything, and a ten-box stack's micro-drift random-walked with no
    /// restoring pull: +42 J and 88 mm over ten settled seconds, against +5 J
    /// and 7 mm. Since *it stuck* is what "stuck" means.
    ///
    /// In A's frame rather than the world's, so a box resting on a turning
    /// platform is stuck to the platform, and so the reference is the same in
    /// a rotated scene.
    pub stick_local_a: Vec3,
    /// Normal impulse the velocity pass has applied at this touching point
    /// this substep, summed over every sweep. Negative is a pull.
    ///
    /// Summed so it can be clamped as a total rather than per sweep, which is
    /// what lets a manifold's corners share their load instead of the first
    /// one visited taking it — see [`solve_velocities`]. Clamped below at
    /// `-normal_lambda / h`, so over a substep this point never pulls harder
    /// than the position solve pushed.
    pub velocity_normal_impulse: f32,
    /// Speculative arresting impulse applied at this point this substep,
    /// summed over every sweep and clamped at zero: a corner that took more
    /// than its share of a landing hands the excess back.
    pub speculative_lambda: f32,
    /// Whether the position solve pushed on this point during this substep.
    ///
    /// Distinct from `normal_lambda > 0`, because a later iteration can
    /// legitimately retract the multiplier to zero while the push — and the
    /// velocity it produced — has already happened.
    pub active: bool,
    /// Kinetic energy this point has taken out of the bodies over the whole
    /// step, joules. Never reset between substeps.
    ///
    /// This is where the heat comes from. Friction sliding a block to a stop,
    /// a ball landing dead, a bounce that comes back lower than it fell — each
    /// is kinetic energy the contact removed, and a rigid body has nowhere
    /// else to put it. The heat solve in `fusion-sim` deposits it at the
    /// contact.
    ///
    /// Counted exactly where the solver applies an impulse — an impulse `j`
    /// at relative velocity `v` changes the pair's kinetic energy by
    /// `j v + w j^2 / 2` — so a restitution impulse, which *returns* energy,
    /// counts negative, and a step's total for one ball bounce comes out as
    /// `m v^2 (1 - e^2) / 2`. The impact itself is stopped by the position
    /// solve, where impulses are multipliers rather than velocities; see
    /// [`record_impact_dissipation`] for how that part is counted.
    pub dissipated: f32,
    /// Impulse this point applied to body B over the whole step, world frame,
    /// N s. Body A received the opposite. Never reset between substeps.
    ///
    /// What the stress solve loads a body with. Every push the solver makes
    /// is counted where it makes it: the position solve's normal multiplier
    /// over `h` once a substep (less what went on tidying an old overlap,
    /// which moves a body without being a force anything applied), static
    /// friction's corrections the same way, and the velocity pass's normal,
    /// speculative and friction impulses as they are applied. A box resting on
    /// a floor reports `m g dt` here.
    pub impulse: Vec3,
    /// The part of [`Self::impulse`], along the normal, that stopped or
    /// reversed an impact rather than holding a load: what the position solve
    /// credited to velocity on a contact closing faster than an impact, and
    /// the velocity pass's normal impulses on one. N s.
    ///
    /// Kept apart because the two load a body differently. A resting load is
    /// steady and is spread over the step; an impact is over in a fraction of
    /// a millisecond, and its force is its impulse over that.
    pub impact_impulse: f32,
    /// Fastest this contact closed at any point in the step, m/s: the
    /// approach speed at the start of each substep, or the speed a
    /// speculative catch took off it.
    pub impact_speed: f32,
}

/// Tunables shared by the position and velocity passes.
#[derive(Clone, Copy, Debug)]
pub struct ContactParameters {
    /// Inverse stiffness, in metres per newton. Zero is a rigid contact.
    pub compliance: f32,
    /// Overlap left uncorrected, metres. A small allowance stops resting
    /// contacts from jittering between touching and separated.
    pub penetration_slop: f32,
    /// Fraction of the remaining overlap removed per iteration.
    pub correction_rate: f32,
    /// Ceiling on how fast depenetration may push, m/s. Without it, two
    /// bodies spawned inside each other leave at speed.
    pub max_depenetration_speed: f32,
    /// Solve static friction as a position-level constraint in addition to the
    /// velocity-level Coulomb bound.
    pub static_friction: bool,
    /// Credit only the motion made this substep to the velocity recovery.
    pub budget_depenetration: bool,
    /// Tangential speed above which a contact counts as sliding, m/s. Static
    /// friction does not apply there — that is what dynamic friction is for.
    pub sliding_threshold: f32,
    /// Stop bodies at surfaces they would otherwise pass through this substep.
    pub speculative: bool,
    /// Impacts slower than this bounce with restitution zero regardless of
    /// material. Without the cutoff, a resting body's own gravity-induced
    /// approach speed is treated as an impact and it buzzes against the floor
    /// forever.
    pub restitution_threshold: f32,
    /// The substep length, carried here so [`warm_start`] can apply the same
    /// depenetration budget the iterations use.
    pub substep: f32,
}

/// Closing speed at which a contact stops being a resting one and becomes an
/// impact, m/s.
///
/// A resting or sliding contact closes only by what gravity adds in a substep
/// — five millimetres a second at 1920 Hz — and a landing closes at metres per
/// second, so anything in between separates them. Five centimetres a second is
/// the same figure [`ContactParameters::sliding_threshold`] uses for the
/// tangent direction, and for the same reason: it is well above the noise of a
/// body that is not going anywhere.
pub const IMPACT_SPEED: f32 = 0.05;

/// Record the incoming relative normal velocity at every contact point.
pub fn capture_approach_velocities(
    bodies: &BodySet,
    manifolds: &[ContactManifold],
    offsets: &[u32],
    scratch: &mut [ContactScratch],
) {
    for (m_index, manifold) in manifolds.iter().enumerate() {
        let (ia, ib) = (manifold.body_a as usize, manifold.body_b as usize);
        let pose_a = bodies.pose(ia);
        let pose_b = bodies.pose(ib);
        let base = offsets[m_index] as usize;
        for (k, point) in manifold.points().iter().enumerate() {
            let normal = pose_a.transform_direction(point.local_normal_a);
            let v = relative_velocity(
                bodies,
                ia,
                ib,
                lever(&pose_a, point.local_a),
                lever(&pose_b, point.local_b),
            );
            let s = &mut scratch[base + k];
            s.approach_velocity = v.dot(normal);
            s.previous_offset = anchor_offset(
                bodies,
                ia,
                ib,
                lever(&pose_a, point.local_a),
                lever(&pose_b, point.local_b),
            );
            // Rotate the converged multiplier into the warm-start slot before
            // clearing it. On the first substep this is what `prepare` seeded
            // from the previous step's cache; afterwards it is what the
            // previous substep converged to.
            s.warm_normal_lambda = s.normal_lambda;
            s.normal_lambda = 0.0;
            s.tangent_lambda = 0.0;
            s.recorded_separation = 0.0;
            s.tidied_lambda = 0.0;
            s.impacting |= s.approach_velocity < -IMPACT_SPEED;
            s.impact_speed = s
                .impact_speed
                .max(-s.approach_velocity)
                .max(s.speculative_impact);
            s.applied_friction = 0.0;
            s.velocity_normal_impulse = 0.0;
            s.speculative_lambda = 0.0;
            s.active = false;
            // `speculative_impact` deliberately survives: it is a fact about
            // this contact over the whole step, not about one substep.
        }
    }
}

/// Apply each contact's remembered impulse before the first iteration.
///
/// ## Why a contact solved from zero is the expensive way
///
/// A resting stack is in equilibrium: the contact under the bottom box
/// supports the weight of everything above it, and it supported the same
/// weight last substep and will support it next substep. Solving from zero
/// throws that away eight times per step and rediscovers it by Gauss-Seidel
/// iteration, one contact deep per sweep — which is exactly the
/// "iteration-count problem" PLAN.md §10 names, and why a twenty-box tower
/// needs more iterations than it is given.
///
/// Starting from the previous answer is the standard fix and the one §10 and
/// §29 both list. The impulse is *applied*, not merely assigned: the stored
/// multiplier represents a push that has to actually be on the bodies for the
/// iterations that follow to be refining it rather than adding to it a second
/// time.
///
/// ## What the energy ledger says this is worth
///
/// Before: a resting ten-box tower cycled 108 kJ through the position solve
/// every four seconds and the velocity pass took 126 kJ back out, in a stack
/// that was not visibly moving. That churn is the residual contact jitter that
/// keeps deep piles awake. The numbers after are in the benchmark.
#[allow(clippy::too_many_arguments)]
pub fn warm_start(
    bodies: &mut BodySet,
    manifolds: &[ContactManifold],
    offsets: &[u32],
    scratch: &mut [ContactScratch],
    params: &ContactParameters,
    fraction: f32,
    delta_position: &mut [Vec3],
    delta_rotation: &mut [Vec3],
    impulses: &mut [f32],
) {
    if fraction <= 0.0 {
        return;
    }
    for (m_index, manifold) in manifolds.iter().enumerate() {
        let ia = manifold.body_a as usize;
        let ib = manifold.body_b as usize;
        if !bodies.is_movable(ia) && !bodies.is_movable(ib) {
            continue;
        }
        let base = offsets[m_index] as usize;
        for (k, point) in manifold.points().iter().enumerate() {
            // Only a contact that is the *same* contact as last time. A point
            // the pipeline could not match to one from the previous step is a
            // new contact — an impact — and an impact has no history to
            // resume. Warm-starting one would apply a force before the
            // surfaces have touched.
            if !point.inherited {
                scratch[base + k].warm_normal_lambda = 0.0;
                continue;
            }
            let lambda = scratch[base + k].warm_normal_lambda * fraction;
            if lambda <= 0.0 {
                continue;
            }

            let pose_a = bodies.pose(ia);
            let pose_b = bodies.pose(ib);
            let normal = pose_a.transform_direction(point.local_normal_a);
            let r_a = lever(&pose_a, point.local_a);
            let r_b = lever(&pose_b, point.local_b);

            // A warm start must not resurrect a contact that has separated.
            // The remembered impulse is only valid while the surfaces are
            // still in contact; applied to a box that has been picked up it
            // would yank it back down.
            let separation = anchor_offset(bodies, ia, ib, r_a, r_b).dot(normal);
            if separation > params.penetration_slop {
                scratch[base + k].warm_normal_lambda = 0.0;
                continue;
            }

            let s = &mut scratch[base + k];
            s.normal_lambda = lambda;
            s.active = true;
            impulses[m_index] += lambda;

            // Credited to velocity on the same budget a fresh correction gets.
            // The alternative — crediting it in full — would add the whole
            // supporting impulse to the stack's velocity every substep, which
            // is a rocket rather than a floor.
            let moved = lambda
                * positional_inverse_mass(
                    effective_inv_mass(bodies, ia),
                    &bodies.inv_inertia_world(ia),
                    r_a,
                    normal,
                )
                + lambda
                    * positional_inverse_mass(
                        effective_inv_mass(bodies, ib),
                        &bodies.inv_inertia_world(ib),
                        r_b,
                        normal,
                    );
            let budget =
                ((-s.approach_velocity).max(0.0) * params.substep - s.recorded_separation).max(0.0);
            let credited = moved.clamp(0.0, budget);
            s.recorded_separation += credited;
            let record_scale = if moved.abs() > 1e-12 {
                credited / moved
            } else {
                0.0
            };

            apply_positional_pair_scaled(
                bodies,
                ia,
                ib,
                r_a,
                r_b,
                normal * lambda,
                record_scale,
                delta_position,
                delta_rotation,
            );
        }
    }
}

/// One position-solve pass over every contact point.
///
/// Gauss-Seidel: each point is evaluated against the poses left by the
/// previous point, not against a snapshot taken at the top of the pass. Jacobi
/// within a manifold looks tempting — one pose fetch instead of four — and it
/// over-corrects by roughly the number of points, because every point
/// independently solves the whole penetration. A four-point resting manifold
/// then applies four times the impulse it needs and the box leaps off the
/// floor.
#[allow(clippy::too_many_arguments)]
pub fn solve_positions(
    bodies: &mut BodySet,
    manifolds: &[ContactManifold],
    offsets: &[u32],
    scratch: &mut [ContactScratch],
    params: &ContactParameters,
    h: f32,
    delta_position: &mut [Vec3],
    delta_rotation: &mut [Vec3],
    impulses: &mut [f32],
) {
    for (m_index, manifold) in manifolds.iter().enumerate() {
        let ia = manifold.body_a as usize;
        let ib = manifold.body_b as usize;
        if !bodies.is_movable(ia) && !bodies.is_movable(ib) {
            continue;
        }
        let base = offsets[m_index] as usize;
        let friction = manifold.friction;

        for (k, point) in manifold.points().iter().enumerate() {
            let scratch = &mut scratch[base + k];

            let pose_a = bodies.pose(ia);
            let pose_b = bodies.pose(ib);
            // The normal rides A's orientation, so it stays correct as A turns.
            let normal = pose_a.transform_direction(point.local_normal_a);
            let r_a = lever(&pose_a, point.local_a);
            let r_b = lever(&pose_b, point.local_b);
            let offset = anchor_offset(bodies, ia, ib, r_a, r_b);

            // ---- non-penetration ----
            let separation = offset.dot(normal);
            let violation = separation + params.penetration_slop;
            // A point that is apart *and* carries no accumulated impulse has
            // nothing to do. One that is apart but has been pushed already may
            // need that push taken back.
            if violation < 0.0 || scratch.normal_lambda > 0.0 {
                let (_, w) = positional_delta(
                    bodies,
                    ia,
                    ib,
                    r_a,
                    r_b,
                    normal,
                    violation,
                    params.compliance,
                    scratch.normal_lambda,
                    h,
                );
                if w > 0.0 {
                    // The depenetration budget is per *substep*, not per
                    // iteration: applying the cap to each iteration separately
                    // would let N iterations inject N times the intended
                    // speed. `separation` changes by `lambda * w`, so the
                    // multiplier already spent converts back into metres.
                    //
                    // Measured from where the contact was at the start of the
                    // substep, not from the multiplier this contact has spent.
                    // The two agree when nothing else is pushing; they part
                    // when something is. A 0.6 kg shard under a 200 kg block
                    // needs the floor to push it, each substep, as far as the
                    // block squeezes it, and counting that against the cap let
                    // the block push the shard through the floor. The cap is
                    // for overlap that was already there at the start of the
                    // substep; motion the substep itself made is undone in full.
                    //
                    // Bounded, though: undone in full, contacts squeezing one
                    // piece of rubble from several sides each undid the
                    // others' squeeze and pumped the pile — a hit stone tower
                    // gained 78 J in one step. So the squeeze is undone up to
                    // [`SQUEEZE`] times the cap beyond it, which holds a shard
                    // under a block three hundred times its mass and leaves a
                    // pile no room to feed on itself.
                    let start = scratch.previous_offset.dot(normal);
                    let cap = params.max_depenetration_speed * h;
                    let spent = scratch.normal_lambda * w;
                    let from_start = start.min(-params.penetration_slop) + cap - separation;
                    let allowance = from_start.min(cap * (1.0 + SQUEEZE) - spent).max(0.0);
                    let error = (violation * params.correction_rate).max(-allowance);
                    let (delta, _) = positional_delta(
                        bodies,
                        ia,
                        ib,
                        r_a,
                        r_b,
                        normal,
                        error,
                        params.compliance,
                        scratch.normal_lambda,
                        h,
                    );
                    // Clamp the *accumulated* multiplier at zero, not each
                    // increment. A contact may only push overall, but within a
                    // step it must be able to take back a push it already made
                    // — otherwise corrections ratchet upward and every extra
                    // iteration adds energy instead of converging.
                    let new_lambda = (scratch.normal_lambda + delta).max(0.0);
                    let applied = new_lambda - scratch.normal_lambda;
                    scratch.normal_lambda = new_lambda;
                    if applied != 0.0 {
                        scratch.active |= applied > 0.0;
                        impulses[m_index] += applied;

                        // Only the part of this correction that undoes motion
                        // made during *this* substep may become velocity. The
                        // bodies closed by `-approach * h` metres; everything
                        // beyond that is pre-existing overlap being tidied up,
                        // and tidying is not a collision.
                        let moved = applied * w;
                        let budget = ((-scratch.approach_velocity).max(0.0) * h
                            - scratch.recorded_separation)
                            .max(0.0);
                        let credited = moved.clamp(0.0, budget);
                        scratch.recorded_separation += credited;
                        let record_scale = if moved.abs() > 1e-12 {
                            credited / moved
                        } else {
                            0.0
                        };
                        // What this correction spent on tidying: the same
                        // split, kept as a multiplier so the Coulomb cone can
                        // be told about it.
                        scratch.tidied_lambda += applied * (1.0 - record_scale);

                        apply_positional_pair_scaled(
                            bodies,
                            ia,
                            ib,
                            r_a,
                            r_b,
                            normal * applied,
                            record_scale,
                            delta_position,
                            delta_rotation,
                        );
                    }
                }
            }

            // ---- static friction ----
            //
            // The two anchors were coincident in the tangent plane when the
            // contact was found, so any tangential offset between them now is
            // exactly how far the surfaces have slipped since. Driving it back
            // to zero is what "stuck" means, and it is why a box can sit on a
            // slope instead of creeping down it.
            // Static friction only for a contact old enough to have a
            // reference. A first-step contact is an impact, and an impacting
            // surface is sliding — which is the velocity pass's business, not
            // this one's.
            if !params.static_friction
                || !point.inherited
                || friction <= 0.0
                || scratch.normal_lambda <= 0.0
            {
                continue;
            }
            // And only while the surfaces are actually stuck. Applying a
            // positional stick constraint to a sliding contact stacks it on
            // top of the velocity-level Coulomb bound, and the block
            // decelerates at 11 m/s^2 where mu*g says 3.9.
            // B's anchor, expressed as a point on A — where this contact would
            // be held if it stuck right now. Built from the anchor offset,
            // which is small and exact, rather than from world positions,
            // which are not.
            let reseed = |bodies: &BodySet| {
                let (pose_a, pose_b) = (bodies.pose(ia), bodies.pose(ib));
                let offset = anchor_offset(
                    bodies,
                    ia,
                    ib,
                    lever(&pose_a, point.local_a),
                    lever(&pose_b, point.local_b),
                );
                point.local_a + pose_a.rotation.inverse().rotate(offset)
            };
            {
                let v = relative_velocity(bodies, ia, ib, r_a, r_b);
                if v.reject_from_unit(normal).length() > params.sliding_threshold {
                    // Sliding: the reference travels with the contact, so the
                    // substep it sticks is the one it is held from.
                    scratch.stick_local_a = reseed(bodies);
                    continue;
                }
            }
            let pose_a = bodies.pose(ia);
            let pose_b = bodies.pose(ib);
            let normal = pose_a.transform_direction(point.local_normal_a);
            let r_a = lever(&pose_a, point.local_a);
            let r_b = lever(&pose_b, point.local_b);

            // Slip accumulated **since the contact last stuck**, in the tangent
            // plane: B's anchor against `stick_local_a`, a point on A that the
            // sliding gate above re-seeds and nothing else moves.
            //
            // Two other references were built and both failed. *Since the
            // contact was born* hauled a box that slid and stopped back towards
            // where it first touched — backwards at five centimetres a second,
            // 150 mm in three seconds. *Since the substep* removed that and let
            // a ten-box stack's drift random-walk: +42 J and 88 mm over ten
            // settled seconds. See `ContactScratch::stick_local_a`.
            let r_stick = lever(&pose_a, scratch.stick_local_a);
            let offset = anchor_offset(bodies, ia, ib, r_stick, r_b);
            let tangential = offset.reject_from_unit(normal);
            let (direction, slip) = tangential.normalize_and_length();
            if slip < 1e-9 {
                continue;
            }

            let (delta, _) = positional_delta(
                bodies,
                ia,
                ib,
                r_a,
                r_b,
                direction,
                slip,
                params.compliance,
                -scratch.tangent_lambda,
                h,
            );
            // `delta` is negative: it pulls the anchors back together.
            let wanted = -delta;
            if wanted <= 0.0 {
                continue;
            }
            // Coulomb cone: the tangential multiplier may not exceed the
            // normal one scaled by friction. Beyond that the surfaces slide,
            // and sliding is the velocity pass's business.
            // Static friction only ever runs on an inherited contact (see the
            // guard above), so its cone is the plain multiplier.
            let budget = (friction * scratch.normal_lambda - scratch.tangent_lambda).max(0.0);
            let applied = wanted.min(budget);
            // A cone that cannot hold the full correction does *not* move the
            // reference. The surfaces slipped a little this substep, and in a
            // jittering stack that happens now and then at every contact; a
            // reference that followed each one would ratchet, re-anchoring to
            // the drifted position every time so the drift is never pulled
            // back. Measured: an 8-box stack leaning 1 mm drifted 13 mm where
            // it drifted 1. Real sliding re-seeds through the velocity gate.
            if applied <= 0.0 {
                continue;
            }
            scratch.tangent_lambda += applied;
            scratch.impulse += direction * (-applied / h);

            // Friction corrections are credited to velocity in full, unlike
            // the normal ones.
            //
            // The symmetry is tempting — budget the credit by the tangential
            // motion made this substep, exactly as the normal direction
            // budgets it by the closing motion — and it is wrong. The normal
            // budget has something to fall back on: penetration that is not
            // credited still gets *fixed*, and the contact stops the body
            // regardless. Friction has nothing to fall back on. Uncredited
            // tangential correction moves the body without removing the
            // velocity that is sliding it, so the surface hauls it back
            // positionally, forever, while it keeps accelerating. A ten-box
            // stack that stands still under full credit tears itself apart at
            // 11 m/s under a budgeted one, over about twelve seconds.
            //
            // The price was residual jitter at resting contacts — around
            // 0.1 m/s in a tall stack, which kept stacks deeper than four
            // boxes awake. Most of it was the iterations losing a stack's
            // weight between its boxes, and finishing each substep from the
            // ground up (`propagate_shock`) took it out: a ten-box tower now
            // settles to rest and sleeps.
            apply_positional_pair(
                bodies,
                ia,
                ib,
                r_a,
                r_b,
                direction * -applied,
                delta_position,
                delta_rotation,
            );
        }
    }
}

/// How many contacts below the ground each body is: zero for anything that
/// does not move, one for a body resting on one, and so on up a stack.
/// Bodies no contact chain reaches are [`u32::MAX`]. See [`propagate_shock`].
pub fn support_levels(bodies: &BodySet, manifolds: &[ContactManifold], levels: &mut Vec<u32>) {
    levels.clear();
    levels.extend((0..bodies.len()).map(|i| if bodies.is_movable(i) { u32::MAX } else { 0 }));
    // Relaxed to a fixed point rather than searched breadth-first: the
    // manifolds are a flat list, and a stack is a few dozen deep at most.
    // Only through contacts that touch. A speculative one is a body that
    // will be near something by the end of the step, not one resting on it,
    // and counted as support it made a 0.4 kg shard tumbling past a floor
    // the ground under a 4.7 kg sliver it then hit: held still for the
    // sliver, it handed it its own surface speed, 47 m/s, and drove it 43 cm
    // into the floor.
    let touching = |m: &&ContactManifold| m.points().iter().any(|p| p.separation <= 0.0);
    // And only up from a body that is itself fairly still. Holding a shard
    // spinning at 150 rad/s still for the steel block on it handed the
    // block the shard's motion. A support is something that holds things
    // up, and [`SUPPORT_SPEED`] says how still that is.
    let steady = |i: usize| {
        !bodies.is_movable(i)
            || bodies.linear_velocity[i].length()
                + bodies.angular_velocity[i].length() * bodies.shape[i].bounding_radius()
                < SUPPORT_SPEED
    };
    loop {
        let mut changed = false;
        for m in manifolds.iter().filter(touching) {
            let (a, b) = (m.body_a as usize, m.body_b as usize);
            let (la, lb) = (levels[a], levels[b]);
            if la.saturating_add(1) < lb && steady(a) {
                levels[b] = la + 1;
                changed = true;
            } else if lb.saturating_add(1) < la && steady(b) {
                levels[a] = lb + 1;
                changed = true;
            }
        }
        if !changed {
            break;
        }
    }
}

/// One last pass over the contacts, from the ground up, in which each pair's
/// lower body does not move: shock propagation (Guendelman, Bridson and
/// Fedkiw, 2003).
///
/// Gauss-Seidel trades a contact's error to its neighbours and converges at a
/// rate set by the mass ratio. A 1 kg board under a 1000 kg block is pushed
/// up by the floor and straight back down by the block, iteration after
/// iteration, and it went the whole of its 5 cm into the floor and stayed
/// there; a 10 kg block landing at 5 m/s sank a 1 kg board 13 mm. Iterations
/// cannot fix that, a hundred or a thousand. Ordering can: solved upwards
/// with the body underneath held, each contact's correction is final, and an
/// overlap is pushed up the stack to the body that caused it rather than
/// down into the floor.
///
/// Only overlap is corrected, with the same per-substep budget as the
/// iterations, and only the part that undoes this substep's closing becomes
/// velocity, so the pass stops what the iterations could not without
/// launching anything. It is not momentum-conserving on its own — the body
/// held does not feel the push — which is why it is the last word on
/// position and not the solve.
#[allow(clippy::too_many_arguments)]
pub fn propagate_shock(
    bodies: &mut BodySet,
    manifolds: &[ContactManifold],
    offsets: &[u32],
    scratch: &mut [ContactScratch],
    params: &ContactParameters,
    h: f32,
    levels: &[u32],
    order: &[u32],
    delta_position: &mut [Vec3],
    delta_rotation: &mut [Vec3],
) {
    for &m_index in order {
        let m_index = m_index as usize;
        let manifold = &manifolds[m_index];
        let (ia, ib) = (manifold.body_a as usize, manifold.body_b as usize);
        let (la, lb) = (levels[ia], levels[ib]);
        if la == lb {
            continue;
        }
        // The body further from the ground moves; along the normal for B,
        // against it for A.
        let (upper, sign) = if la < lb { (ib, 1.0) } else { (ia, -1.0) };
        if !bodies.is_movable(upper) {
            continue;
        }
        let base = offsets[m_index] as usize;
        let points = manifold.points();
        let inv_inertia = bodies.inv_inertia_world(upper);
        let inv_mass = bodies.inv_mass[upper];
        // Each point's budget, from where it stood at the start of the
        // substep and where it stands now: this substep's closing in full,
        // and older overlap at the depenetration speed — the iterations'
        // budget. A total, so the sweeps below may share it out and take
        // some back.
        let cap = params.max_depenetration_speed * h;
        let mut budget = [0.0f32; MAX_CONTACT_POINTS];
        for (k, point) in points.iter().enumerate() {
            let pose_a = bodies.pose(ia);
            let pose_b = bodies.pose(ib);
            let normal = pose_a.transform_direction(point.local_normal_a);
            let separation = anchor_offset(
                bodies,
                ia,
                ib,
                lever(&pose_a, point.local_a),
                lever(&pose_b, point.local_b),
            )
            .dot(normal);
            let start = scratch[base + k].previous_offset.dot(normal);
            budget[k] = (start.min(-params.penetration_slop) + cap - separation).max(0.0);
        }
        // Every point's push worked out from the same pose and applied at
        // once, a share each, and swept to convergence. Corner after corner,
        // each box tipped the way its first corner pushed, and a ten-box
        // tower walked six centimetres.
        let mut total = [0.0f32; MAX_CONTACT_POINTS];
        let mut credit = [0.0f32; MAX_CONTACT_POINTS];
        for _ in 0..BLOCK_SWEEPS {
            let pose_a = bodies.pose(ia);
            let pose_b = bodies.pose(ib);
            let mut normals = [Vec3::ZERO; MAX_CONTACT_POINTS];
            let mut staged = [0.0f32; MAX_CONTACT_POINTS];
            let mut levers = [Vec3::ZERO; MAX_CONTACT_POINTS];
            let mut weights = [0.0f32; MAX_CONTACT_POINTS];
            let mut count = 0usize;
            let mut largest = 0.0f32;
            for (k, point) in points.iter().enumerate() {
                if budget[k] <= 0.0 && total[k] <= 0.0 {
                    continue;
                }
                let r_a = lever(&pose_a, point.local_a);
                let r_b = lever(&pose_b, point.local_b);
                normals[k] = pose_a.transform_direction(point.local_normal_a);
                let normal = normals[k];
                levers[k] = if upper == ib { r_b } else { r_a };
                weights[k] = positional_inverse_mass(inv_mass, &inv_inertia, levers[k], normal);
                if weights[k] <= 0.0 {
                    continue;
                }
                let separation = anchor_offset(bodies, ia, ib, r_a, r_b).dot(normal);
                let violation = separation + params.penetration_slop;
                let next = (total[k] - violation / weights[k]).clamp(0.0, budget[k] / weights[k]);
                staged[k] = next - total[k];
                if staged[k] != 0.0 {
                    count += 1;
                }
                largest = largest.max(next);
            }
            if count == 0 {
                break;
            }
            let share = 1.0 / count as f32;
            let mut largest_change = 0.0f32;
            for k in 0..points.len() {
                if staged[k] == 0.0 {
                    continue;
                }
                let applied = staged[k] * share;
                total[k] += applied;
                largest_change = largest_change.max(applied.abs());
                let s = &mut scratch[base + k];
                // Not added to the multiplier. It was sized for moving one
                // body, and the warm start and the velocity pass apply a
                // multiplier to both: carried there, a correction sized for a
                // 1000 kg block went to the 1 kg board under it and launched
                // it ten metres.
                s.active = true;
                // Only this substep's closing becomes velocity, and a push
                // taken back takes back what it was credited.
                let moved = applied * weights[k];
                let room = ((-s.approach_velocity).max(0.0) * h - s.recorded_separation).max(0.0);
                let credited = moved.clamp(-credit[k], room);
                credit[k] += credited;
                s.recorded_separation += credited;
                let record_scale = credited / moved;
                s.tidied_lambda += applied * (1.0 - record_scale);
                apply_positional_impulse_scaled(
                    bodies,
                    upper,
                    levers[k],
                    normals[k] * (sign * applied),
                    record_scale,
                    &inv_inertia,
                    delta_position,
                    delta_rotation,
                );
            }
            if largest_change <= BLOCK_TOLERANCE * largest {
                break;
            }
        }
    }
}

/// The velocity half of [`propagate_shock`]: after the velocity passes, from
/// the ground up, each pair's upper body alone loses whatever it still has
/// closing on the body under it.
///
/// Without it the position pass held a 1 kg board out of the floor under a
/// 1000 kg block and the velocity pass, Gauss-Seidel again, could not take
/// the block's fall out through the board: both sat still on the floor
/// carrying two metres a second downwards, and gaining.
///
/// Only closing speed is removed, never added — nothing here pulls, and a
/// contact the velocity pass has set bouncing is already separating and is
/// left alone. Only at points touching or within `SHOCK_REACH` of it: the
/// speculative catch of a point further off is the velocity pass's, and done
/// here as well it set a twenty-box tower walking fifteen centimetres.
#[allow(clippy::too_many_arguments)]
pub fn propagate_shock_velocities(
    bodies: &mut BodySet,
    manifolds: &[ContactManifold],
    offsets: &[u32],
    scratch: &mut [ContactScratch],
    h: f32,
    levels: &[u32],
    order: &[u32],
) {
    for &m_index in order {
        let m_index = m_index as usize;
        let manifold = &manifolds[m_index];
        let (ia, ib) = (manifold.body_a as usize, manifold.body_b as usize);
        let (la, lb) = (levels[ia], levels[ib]);
        if la == lb {
            continue;
        }
        let (upper, sign) = if la < lb { (ib, 1.0) } else { (ia, -1.0) };
        if !bodies.is_movable(upper) {
            continue;
        }
        let pose_a = bodies.pose(ia);
        let pose_b = bodies.pose(ib);
        let inv_inertia = bodies.inv_inertia_world(upper);
        let inv_mass = bodies.inv_mass[upper];
        let base = offsets[m_index] as usize;
        let points = manifold.points();
        let mut r_a = [Vec3::ZERO; MAX_CONTACT_POINTS];
        let mut r_b = [Vec3::ZERO; MAX_CONTACT_POINTS];
        let mut w = [0.0f32; MAX_CONTACT_POINTS];
        let mut floor = [0.0f32; MAX_CONTACT_POINTS];
        let mut normals = [Vec3::ZERO; MAX_CONTACT_POINTS];
        for (k, point) in points.iter().enumerate() {
            normals[k] = pose_a.transform_direction(point.local_normal_a);
            let normal = normals[k];
            r_a[k] = lever(&pose_a, point.local_a);
            r_b[k] = lever(&pose_b, point.local_b);
            let separation = anchor_offset(bodies, ia, ib, r_a[k], r_b[k]).dot(normal);
            if separation > SHOCK_REACH {
                continue;
            }
            let r = if upper == ib { r_b[k] } else { r_a[k] };
            w[k] = positional_inverse_mass(inv_mass, &inv_inertia, r, normal);
            // Within reach of touching it may close the rest of the gap.
            floor[k] = -separation.max(0.0) / h;
        }
        // Together, a share each, for the reason the position half is.
        let mut total = [0.0f32; MAX_CONTACT_POINTS];
        for _ in 0..BLOCK_SWEEPS {
            let mut staged = [0.0f32; MAX_CONTACT_POINTS];
            let mut count = 0usize;
            let mut largest = 0.0f32;
            for k in 0..points.len() {
                if w[k] <= 0.0 {
                    continue;
                }
                let v = relative_velocity(bodies, ia, ib, r_a[k], r_b[k]).dot(normals[k]);
                let next = (total[k] + (floor[k] - v) / w[k]).max(0.0);
                staged[k] = next - total[k];
                if staged[k] != 0.0 {
                    count += 1;
                }
                largest = largest.max(next);
            }
            if count == 0 {
                break;
            }
            let share = 1.0 / count as f32;
            let mut largest_change = 0.0f32;
            for k in 0..points.len() {
                if staged[k] == 0.0 {
                    continue;
                }
                let impulse = staged[k] * share;
                total[k] += impulse;
                largest_change = largest_change.max(impulse.abs());
                let r = if upper == ib { r_b[k] } else { r_a[k] };
                crate::constraint::apply_velocity_impulse(
                    bodies,
                    upper,
                    r,
                    normals[k] * (sign * impulse),
                    &inv_inertia,
                );
                scratch[base + k].impulse += normals[k] * impulse;
            }
            if largest_change <= BLOCK_TOLERANCE * largest {
                break;
            }
        }
    }
}

/// Fastest a body's surface may move and it still hold others up in
/// [`support_levels`], m/s. Measured over sixteen dropped piles of mixed
/// pieces: at one, a tumbling shard a falling block landed on was no
/// support and was driven into the floor; with no limit at all, the worst
/// overlaps were a third deeper. Two and a half had the least overlap on
/// average, 1.1 mm.
const SUPPORT_SPEED: f32 = 2.5;

/// How far apart two surfaces may be and still count as touching for
/// [`propagate_shock_velocities`], metres. A resting pair sits within the
/// slop either side of zero, so it has to reach past it.
const SHOCK_REACH: f32 = 0.0;

/// Largest number of sweeps a manifold's normal block may take in one pass.
///
/// Measured on the scorecard's hardest stack, rotated and leaning, 98% of
/// manifolds converge to [`BLOCK_TOLERANCE`] in five sweeps and none needs
/// more than six. The cap is a guard, not a budget.
const BLOCK_SWEEPS: u32 = 16;

/// How much squeeze — motion other contacts made this substep — a contact may
/// undo beyond the depenetration cap, as a multiple of the cap.
const SQUEEZE: f32 = 1.0;

/// A manifold's normal block has converged when no sweep changes any of its
/// points by more than this fraction of the largest impulse it carries.
///
/// Measured: at 1e-4, 1e-5, 1e-6 and 1e-7 every contact behaviour on the
/// scorecard agrees, where a fixed four sweeps (about 1e-4) and eight did not
/// — which is the evidence that convergence, rather than a lucky count, is
/// what changed the answers.
const BLOCK_TOLERANCE: f32 = 1e-5;

/// Count the kinetic energy the position solve took out of every impact this
/// substep. Call once per substep, after the iterations and before the
/// velocity passes.
///
/// The position solve stops a closing contact by moving the bodies, and the
/// velocity follows from the motion; so the impulse it applied at a point is
/// the separation it credited to velocity there, over `w h`. A point closing
/// at `v` and credited `d` metres went from `v` to `v - d / h` under that
/// impulse, which removes `J (v - d / (2 h))` of kinetic energy — exactly
/// `m_eff v^2 / 2` for a contact brought to rest. Only the credited part
/// counts: tidying an old overlap moves a body without changing its velocity,
/// and so dissipates nothing.
pub fn record_impact_dissipation(
    bodies: &BodySet,
    manifolds: &[ContactManifold],
    offsets: &[u32],
    scratch: &mut [ContactScratch],
    h: f32,
) {
    for (m_index, manifold) in manifolds.iter().enumerate() {
        let ia = manifold.body_a as usize;
        let ib = manifold.body_b as usize;
        let base = offsets[m_index] as usize;
        for (k, point) in manifold.points().iter().enumerate() {
            let s = &mut scratch[base + k];
            let pose_a = bodies.pose(ia);
            let pose_b = bodies.pose(ib);
            let normal = pose_a.transform_direction(point.local_normal_a);
            // The position solve's push this substep, as an impulse. Here
            // because this runs once a substep, after the iterations have
            // settled the multiplier and before the velocity pass adds to it.
            s.impulse += normal * ((s.normal_lambda - s.tidied_lambda).max(0.0) / h);
            let credited = s.recorded_separation;
            let closing = (-s.approach_velocity).max(0.0);
            // A resting contact closes at `g h` every substep, because gravity
            // is integrated before the contact holds it — and stopping that is
            // not dissipation, it is the splitting undoing its own work. Counted,
            // it heated a block sitting on a floor by 24 W for ever. Below
            // the impact threshold nothing has landed.
            if credited <= 0.0 || closing <= IMPACT_SPEED {
                continue;
            }
            let w = positional_inverse_mass(
                effective_inv_mass(bodies, ia),
                &bodies.inv_inertia_world(ia),
                lever(&pose_a, point.local_a),
                normal,
            ) + positional_inverse_mass(
                effective_inv_mass(bodies, ib),
                &bodies.inv_inertia_world(ib),
                lever(&pose_b, point.local_b),
                normal,
            );
            if w <= 0.0 {
                continue;
            }
            let impulse = credited / (w * h);
            s.dissipated += impulse * (closing - 0.5 * credited / h);
            s.impact_impulse += impulse;
        }
    }
}

/// One velocity-solve pass: restitution, then dynamic friction.
///
/// The restitution half also does the job the v0.1 solver did on its own —
/// removing the separating velocity that the position solve invented. That is
/// simply the `restitution = 0` case of the same equation. A body that arrives
/// already 4 mm deep would otherwise leave at 3 m/s, an artifact entirely of
/// *when* the contact was detected rather than of any physics.
///
/// ## A manifold's corners are solved together
///
/// Each manifold's normal impulses — speculative arrest and restitution — are
/// solved as a block: swept over its points until no sweep changes them, with
/// each point's impulse summed across sweeps and the *sum* clamped. Friction
/// runs once the normal block has settled.
///
/// Solved point by point instead, the answer depended on which corner was
/// visited first. The first corner of a landing box took the whole impact;
/// friction tipping a sliding box lifted its back edge in one sweep and the
/// next sweep pulled it back down, so a tall block that must fall stood still
/// — the floor was glue. And a rotated scene visits corners in a different
/// order, so it gave a different answer: the invariance scene's mismatch was
/// 28 mm. Converged, it is 1 mm; a leaning ten-box stack that drifted 15 mm and
/// gained 67 J over ten seconds drifts 0.2 mm and gains 0.03 J; and a block
/// sliding at mu = 0.6 or 0.9 falls over as it should.
pub fn solve_velocities(
    bodies: &mut BodySet,
    manifolds: &[ContactManifold],
    offsets: &[u32],
    scratch: &mut [ContactScratch],
    params: &ContactParameters,
    h: f32,
) {
    for (m_index, manifold) in manifolds.iter().enumerate() {
        let ia = manifold.body_a as usize;
        let ib = manifold.body_b as usize;
        if !bodies.is_movable(ia) && !bodies.is_movable(ib) {
            continue;
        }
        let pose_a = bodies.pose(ia);
        let pose_b = bodies.pose(ib);
        let base = offsets[m_index] as usize;
        let friction = manifold.friction;
        let restitution = manifold.restitution;
        let points = manifold.points();
        let inv_i_a = bodies.inv_inertia_world(ia);
        let inv_i_b = bodies.inv_inertia_world(ib);

        // The geometry is fixed for the pass — only velocities change here —
        // so it is worked out once rather than once per sweep.
        let mut r_a = [Vec3::ZERO; MAX_CONTACT_POINTS];
        let mut r_b = [Vec3::ZERO; MAX_CONTACT_POINTS];
        let mut w_normal = [0.0f32; MAX_CONTACT_POINTS];
        // Whether each point is being solved before the surfaces meet.
        let mut speculative = [false; MAX_CONTACT_POINTS];
        let mut separation = [0.0f32; MAX_CONTACT_POINTS];
        // Each point's own normal, riding A's orientation.
        let mut normals = [Vec3::ZERO; MAX_CONTACT_POINTS];
        for (k, point) in points.iter().enumerate() {
            normals[k] = pose_a.transform_direction(point.local_normal_a);
            let normal = normals[k];
            r_a[k] = lever(&pose_a, point.local_a);
            r_b[k] = lever(&pose_b, point.local_b);
            separation[k] = anchor_offset(bodies, ia, ib, r_a[k], r_b[k]).dot(normal);
            speculative[k] = separation[k] > 0.0;
            w_normal[k] =
                positional_inverse_mass(effective_inv_mass(bodies, ia), &inv_i_a, r_a[k], normal)
                    + positional_inverse_mass(
                        effective_inv_mass(bodies, ib),
                        &inv_i_b,
                        r_b[k],
                        normal,
                    );
        }

        // ---- normal block ----
        for _ in 0..BLOCK_SWEEPS {
            let mut largest_change = 0.0f32;
            let mut largest_impulse = 0.0f32;
            // ---- the points not touching yet, together ----
            //
            // Every speculative point's impulse is worked out from the same
            // velocities and they are applied at once, a share each, rather
            // than one after another. One after another, the first corner of a
            // box landing flat took the whole of the landing, tipped the box,
            // and it bounced onto a corner and walked eleven centimetres. Taken
            // together, a symmetric landing stays symmetric, whichever corner
            // is listed first; the sweeps converge the shares.
            if params.speculative {
                let mut staged = [0.0f32; MAX_CONTACT_POINTS];
                let mut count = 0usize;
                for k in 0..points.len() {
                    let normal = normals[k];
                    if !speculative[k] || w_normal[k] <= 0.0 {
                        continue;
                    }
                    let s = &mut scratch[base + k];
                    let v = relative_velocity(bodies, ia, ib, r_a[k], r_b[k]);
                    let normal_velocity = v.dot(normal);
                    // Negative is closing; this is the fastest approach that
                    // still lands on the surface rather than through it.
                    let allowed = -separation[k] / h;
                    if normal_velocity < allowed {
                        s.speculative_impact = s.speculative_impact.max(-normal_velocity);
                    }
                    let total =
                        (s.speculative_lambda + (allowed - normal_velocity) / w_normal[k]).max(0.0);
                    staged[k] = total - s.speculative_lambda;
                    if staged[k] != 0.0 {
                        count += 1;
                    }
                }
                let share = 1.0 / count.max(1) as f32;
                for k in 0..points.len() {
                    let normal = normals[k];
                    if staged[k] == 0.0 {
                        continue;
                    }
                    let s = &mut scratch[base + k];
                    let impulse = staged[k] * share;
                    let v = relative_velocity(bodies, ia, ib, r_a[k], r_b[k]);
                    let normal_velocity = v.dot(normal);
                    s.speculative_lambda += impulse;
                    largest_change = largest_change.max(impulse.abs());
                    largest_impulse = largest_impulse.max(s.speculative_lambda);
                    if s.speculative_impact > IMPACT_SPEED {
                        s.dissipated -= impulse * (normal_velocity + 0.5 * w_normal[k] * impulse);
                        s.impact_impulse += impulse;
                    }
                    apply_velocity_pair(bodies, ia, ib, r_a[k], r_b[k], normal * impulse);
                    s.impulse += normal * impulse;
                    // A real normal impulse — it is what arrests the fall — so
                    // it counts as pressure for the Coulomb cone: a landing
                    // caught speculatively still pays friction (v1.0). The
                    // factor of `h` converts it to the positional multiplier
                    // the friction pass sizes itself from.
                    s.normal_lambda += impulse * h;
                }
            }
            for k in 0..points.len() {
                let normal = normals[k];
                let s = &mut scratch[base + k];
                if w_normal[k] <= 0.0 {
                    continue;
                }
                let v = relative_velocity(bodies, ia, ib, r_a[k], r_b[k]);
                let normal_velocity = v.dot(normal);

                // A contact that is not touching yet is still worth solving,
                // if the bodies would close the gap and more within this
                // substep.
                //
                // This is speculative contact, and it is what PLAN.md §29
                // lists for v0.3 as "speculative CCD". Rather than sweeping
                // swept volumes, the narrow phase reports pairs that are merely
                // *near*, and the constraint caps the approach speed at
                // `separation / h` — exactly fast enough to arrive at the
                // surface and no faster. A bullet is stopped at the wall
                // instead of being discovered on the far side of it.
                //
                // The cost is one comparison per contact point. The
                // alternative, conservative advancement, costs a root-find per
                // pair.
                if speculative[k] {
                    // Solved together, above.
                    continue;
                }

                // A stale manifold whose bodies are flying apart must be left
                // alone.
                if !s.active {
                    continue;
                }

                // ---- restitution ----
                //
                // The target separating velocity, in one rule that covers
                // every case:
                //
                // | arriving at        | target            | meaning        |
                // |--------------------|-------------------|----------------|
                // | fast, closing      | `-e * approach`   | it bounces     |
                // | slowly closing     | `0`               | it lands dead  |
                // | resting            | `0`               | it stays put   |
                // | already separating | `approach`        | leave it alone |
                //
                // The last row is the one that is easy to get wrong in both
                // directions. *Skipping* separating contacts lets the position
                // solve keep adding depenetration velocity substep after
                // substep — eight substeps at a 2 m/s cap turn a 6 m/s impact
                // into a 15 m/s launch. *Zeroing* them instead takes a
                // legitimate bounce straight back off: the ball leaves the
                // floor in one substep and is stopped dead in the next.
                // Holding the velocity the contact arrived with does neither.
                //
                // The speed the body genuinely arrived with, which is not the
                // speed it has left if a speculative clamp already slowed it on
                // the way in.
                let approach = s.approach_velocity.min(-s.speculative_impact);
                // Only a contact that was *closing* at the start of the substep
                // is driven. One already moving apart has bounced, and is left
                // alone. This is only safe because depenetration is
                // budget-limited at the source — with an ungated position solve
                // free to inject velocity every substep, skipping separating
                // contacts let twelve coincident spheres reach a hundred metres
                // per second.
                if approach > 0.0 {
                    continue;
                }
                let target = if restitution > 0.0 && approach < -params.restitution_threshold {
                    -restitution * approach
                } else {
                    0.0
                };
                // Driven towards the target, pulling as well as pushing — but
                // never pulling harder, over the substep, than the position
                // solve pushed here.
                //
                // The pull is what damps a stack: it removes the separating
                // velocity a position correction leaves behind, which is why a
                // resting pile goes quiet instead of buzzing. Forbidding it
                // costs exactly that damping — measured, sliding friction came
                // out 18 to 28% weak and a ten-box stack walks itself apart.
                //
                // Unbounded, it was glue. A push of impulse `P` invents at most
                // `P w` of separating velocity at this point, so taking back
                // exactly that costs a pull of at most `P`, which is
                // `normal_lambda / h`. Anything beyond is a pull on separation
                // the position solve did not cause — the back edge of a box
                // that friction is tipping over. It is the sequential-impulse
                // clamp, lambda >= 0, taken over both passes of the substep.
                let floor = -s.normal_lambda / h;
                let total = (s.velocity_normal_impulse + (target - normal_velocity) / w_normal[k])
                    .max(floor);
                let impulse = total - s.velocity_normal_impulse;
                s.velocity_normal_impulse = total;
                largest_change = largest_change.max(impulse.abs());
                largest_impulse = largest_impulse.max(total.abs()).max(s.normal_lambda / h);
                if impulse != 0.0 {
                    // Only at an impact. At a resting contact these impulses
                    // take back separating velocity the position solve itself
                    // made, which is the solver tidying up after its own
                    // splitting rather than anything a surface felt; counted,
                    // it warmed a block sitting on a floor by about a watt.
                    if s.impacting {
                        s.dissipated -= impulse * (normal_velocity + 0.5 * w_normal[k] * impulse);
                        s.impact_impulse += impulse;
                    }
                    apply_velocity_pair(bodies, ia, ib, r_a[k], r_b[k], normal * impulse);
                    s.impulse += normal * impulse;
                }
            }
            if largest_change <= BLOCK_TOLERANCE * largest_impulse {
                break;
            }
        }

        // ---- dynamic friction ----
        for k in 0..points.len() {
            let normal = normals[k];
            let s = &mut scratch[base + k];
            if speculative[k] && !params.speculative {
                continue;
            }
            if !speculative[k] && !s.active {
                continue;
            }
            if w_normal[k] <= 0.0 || friction <= 0.0 || s.normal_lambda <= 0.0 {
                continue;
            }
            let v = relative_velocity(bodies, ia, ib, r_a[k], r_b[k]);
            let tangential = v.reject_from_unit(normal);
            let (direction, speed) = tangential.normalize_and_length();
            if speed < 1e-6 {
                continue;
            }
            let w_tangent = positional_inverse_mass(
                effective_inv_mass(bodies, ia),
                &inv_i_a,
                r_a[k],
                direction,
            ) + positional_inverse_mass(
                effective_inv_mass(bodies, ib),
                &inv_i_b,
                r_b[k],
                direction,
            );
            if w_tangent <= 0.0 {
                continue;
            }
            // The impulse that would stop the sliding outright, bounded by
            // what the Coulomb cone still permits. `lambda / h` converts a
            // positional multiplier into an impulse.
            //
            // The velocity pass gets the full Coulomb budget, not what the
            // position pass left of it.
            //
            // Netting the two off looks right — static friction has already
            // spent part of the cone, and that correction has already become
            // velocity — but it starves the only damping in the tangent
            // direction exactly when static friction is most active. The
            // positional constraint is a stiff spring with nothing opposing
            // it, so a box that has landed and stopped keeps rocking gently
            // and creeping: 4 cm from the impact itself, and 30 more over the
            // next three seconds.
            //
            // A little double-counting costs some accuracy in the deceleration
            // rate. An undamped limit cycle costs the whole result.
            //
            // The Coulomb cone, from the normal impulse this contact actually
            // transmitted rather than from its multiplier.
            //
            // The two are not the same and the difference is a bug somebody
            // feels rather than reads: `normal_lambda` includes depenetration
            // — pushing a body out of an overlap it should not have been in —
            // and the position solve already knows that is not a collision
            // ("everything beyond that is pre-existing overlap being tidied
            // up, and tidying is not a collision"). Budgeting the cone from it
            // hands friction the tidying as well.
            //
            // Measured, a 1 m box crossing a floor at 6 m/s and landing from
            // 5 cm up: it left the impact at **3.1 m/s** where Coulomb allows
            // 5.5, and stopped dead in a metre. That is what made thrown boxes
            // feel like they were landing in sand.
            //
            // The impulse is `m_eff * dv_n`: the change in relative normal
            // velocity across the substep, over the point's own inverse mass.
            // A resting contact transmits `m g h` — it cancels gravity every
            // substep — so a box on a slope still sticks; an impact transmits
            // `m dv`; and tidying transmits nothing, because the position
            // solve does not credit it to velocity.
            let stopping = speed / w_tangent;
            // Only an *impact* has its cone corrected.
            //
            // A contact the pipeline matched to last step's is one whose
            // multiplier is already the load it converged to — for a resting
            // box, `m g h^2` to the digit — and subtracting a tidying share
            // from that both under-states the cone and makes it depend on how
            // deep the body happened to be, which is the one quantity a
            // rotated scene reproduces least exactly. Measured: the
            // orientation-invariance test's mismatch went from 3 cm to 28.
            //
            // And the test for an impact is whether the surfaces were closing
            // — not whether the contact is new. A contact re-anchors whenever
            // it has slid past `ANCHOR_MATCH_DISTANCE`, so a fast slide
            // produces new contacts continually; treating those as impacts
            // gave them a weak cone every time they re-anchored, and the box
            // crept backwards at five centimetres a second for as long as the
            // test ran. A resting or sliding contact closes only by what
            // gravity adds in a substep, which is far below that threshold; a
            // landing closes at metres per second.
            let pressing = if s.impacting {
                (s.normal_lambda - s.tidied_lambda).max(0.0)
            } else {
                s.normal_lambda
            };
            let limit = (friction * pressing / h - s.applied_friction).max(0.0);
            let magnitude = stopping.min(limit);
            s.applied_friction += magnitude;
            s.dissipated += magnitude * (speed - 0.5 * w_tangent * magnitude);
            apply_velocity_pair(bodies, ia, ib, r_a[k], r_b[k], direction * -magnitude);
            s.impulse += direction * -magnitude;
        }
    }
}
