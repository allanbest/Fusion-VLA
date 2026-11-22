//! Joint solving.
//!
//! Every joint here is expressed as some combination of the two primitives in
//! `constraint`: a positional correction along a direction, and an angular
//! correction about an axis. Nothing in this file knows how to move a body —
//! it only decides what error to remove.
//!
//! That is the boundary PLAN.md §5 warns about defending. A hinge that grew
//! its own private impulse code would be a hinge that could not later be
//! coupled to a soft body, or run at FP64, or reported on by the energy
//! ledger.

use fusion_core::{BodySet, Joint, JointKind};
use fusion_math::{Quat, Vec3};

use crate::constraint::{solve_angular, solve_positional};

/// Accumulated multipliers for one joint over the current substep.
#[derive(Clone, Copy, Debug, Default)]
pub struct JointScratch {
    /// Multiplier for the positional (anchor-to-anchor) constraint.
    pub positional_lambda: f32,
    /// Multiplier for the angular constraint that aligns the joint frames.
    pub angular_lambda: f32,
    /// Multiplier for whichever limit is being enforced this substep.
    pub limit_lambda: f32,
}

/// One position-solve pass over every joint.
pub fn solve_positions(
    bodies: &mut BodySet,
    joints: &[Joint],
    scratch: &mut [JointScratch],
    h: f32,
    delta_position: &mut [Vec3],
    delta_rotation: &mut [Vec3],
) {
    for (index, joint) in joints.iter().enumerate() {
        if !joint.enabled {
            continue;
        }
        let a = joint.body_a.index();
        let b = joint.body_b.index();
        if !bodies.is_movable(a) && !bodies.is_movable(b) {
            continue;
        }
        let s = &mut scratch[index];

        // Angular part first. Orientation feeds into where the anchors are, so
        // fixing rotation before position leaves less for position to undo.
        match joint.kind {
            JointKind::Fixed | JointKind::Slider { .. } => {
                let error = frame_alignment_error(bodies, joint);
                let applied = solve_angular(
                    bodies,
                    a,
                    b,
                    error,
                    joint.angular_compliance,
                    s.angular_lambda,
                    h,
                    delta_rotation,
                );
                s.angular_lambda += applied.delta_lambda;
            }
            JointKind::Hinge => {
                // Only the two degrees of freedom off the hinge axis are
                // constrained. The cross product of the two axes is exactly
                // the rotation vector that brings them into line, and it
                // vanishes — correctly — when they already agree.
                let pose_a = bodies.pose(a);
                let pose_b = bodies.pose(b);
                let (frame_a, frame_b) = joint.world_frames(&pose_a, &pose_b);
                let axis_a = frame_a.rotate(Vec3::X);
                let axis_b = frame_b.rotate(Vec3::X);
                let applied = solve_angular(
                    bodies,
                    a,
                    b,
                    axis_a.cross(axis_b),
                    joint.angular_compliance,
                    s.angular_lambda,
                    h,
                    delta_rotation,
                );
                s.angular_lambda += applied.delta_lambda;
            }
            JointKind::Ball | JointKind::Distance { .. } => {}
        }

        // Positional part.
        //
        // Lever arms are the rotated local anchors, and the offset between the
        // two anchors is the double-precision centre difference plus those
        // levers. Building the two absolute anchor points and subtracting them
        // would work at the origin and quietly stop working a few hundred
        // metres out.
        let pose_a = bodies.pose(a);
        let pose_b = bodies.pose(b);
        let r_a = pose_a.rotation.rotate(joint.local_anchor_a);
        let r_b = pose_b.rotation.rotate(joint.local_anchor_b);
        let offset = bodies.relative_translation(a, b) + (r_b - r_a);

        match joint.kind {
            JointKind::Ball | JointKind::Fixed | JointKind::Hinge => {
                let (direction, error) = offset.normalize_and_length();
                if error > 1e-7 {
                    let applied = solve_positional(
                        bodies,
                        a,
                        b,
                        r_a,
                        r_b,
                        direction,
                        error,
                        joint.compliance,
                        s.positional_lambda,
                        h,
                        delta_position,
                        delta_rotation,
                    );
                    s.positional_lambda += applied.delta_lambda;
                }
            }
            JointKind::Distance { min, max } => {
                let (direction, distance) = offset.normalize_and_length();
                if direction == Vec3::ZERO {
                    continue;
                }
                // Inside the range there is nothing to do — a rope is slack
                // until it is taut, and that "until" is the whole point of an
                // inequality constraint.
                let error = if distance > max {
                    distance - max
                } else if distance < min {
                    distance - min
                } else {
                    0.0
                };
                if error != 0.0 {
                    let applied = solve_positional(
                        bodies,
                        a,
                        b,
                        r_a,
                        r_b,
                        direction,
                        error,
                        joint.compliance,
                        s.positional_lambda,
                        h,
                        delta_position,
                        delta_rotation,
                    );
                    s.positional_lambda += applied.delta_lambda;
                }
            }
            JointKind::Slider { min, max } => {
                let (frame_a, _) = joint.world_frames(&pose_a, &pose_b);
                let axis = frame_a.rotate(Vec3::X);

                // Everything off the axis is rigidly constrained.
                let perpendicular = offset.reject_from_unit(axis);
                let (direction, error) = perpendicular.normalize_and_length();
                if error > 1e-7 {
                    let applied = solve_positional(
                        bodies,
                        a,
                        b,
                        r_a,
                        r_b,
                        direction,
                        error,
                        joint.compliance,
                        s.positional_lambda,
                        h,
                        delta_position,
                        delta_rotation,
                    );
                    s.positional_lambda += applied.delta_lambda;
                }

                // Along the axis, only the ends of the travel are constrained.
                let along = offset.dot(axis);
                let limit_error = if along > max {
                    along - max
                } else if along < min {
                    along - min
                } else {
                    0.0
                };
                if limit_error != 0.0 {
                    let applied = solve_positional(
                        bodies,
                        a,
                        b,
                        r_a,
                        r_b,
                        axis,
                        limit_error,
                        joint.compliance,
                        s.limit_lambda,
                        h,
                        delta_position,
                        delta_rotation,
                    );
                    s.limit_lambda += applied.delta_lambda;
                }
            }
        }
    }
}

/// Rotation vector taking body A's joint frame onto body B's.
///
/// The quaternion `q_b * q_a^-1` is the relative rotation; twice its vector
/// part is the small-angle rotation vector, which is what the angular
/// constraint consumes. The `w < 0` branch is the double cover — the same
/// rotation expressed the long way round, which would otherwise be corrected
/// by turning almost all the way around instead of barely at all.
fn frame_alignment_error(bodies: &BodySet, joint: &Joint) -> Vec3 {
    let pose_a = bodies.pose(joint.body_a.index());
    let pose_b = bodies.pose(joint.body_b.index());
    let (frame_a, frame_b) = joint.world_frames(&pose_a, &pose_b);
    let relative: Quat = frame_b * frame_a.conjugate();
    let vector = relative.vector_part() * 2.0;
    if relative.w < 0.0 {
        -vector
    } else {
        vector
    }
}
