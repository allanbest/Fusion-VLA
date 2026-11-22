//! Joints.
//!
//! PLAN.md §10: "generic constraint functions `C(x)=0` / `C(x)>=0` — hinge,
//! ball, slider, fixed, distance all fall out of one machinery." So a joint
//! here is *description*, not behaviour: two anchors, two frames, and a kind.
//! The solver reads it and expresses it as positional and angular corrections,
//! which are the same two primitives contacts use.
//!
//! Every joint is defined by anchors and frames in **body-local** coordinates,
//! for the same reason contact anchors are: they then move with the bodies for
//! free, and stay meaningful across a reset.

use fusion_math::{Isometry, Quat, Vec3};

use crate::body::BodyHandle;

/// Reference to a joint in a [`crate::World`].
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct JointHandle(pub u32);

impl JointHandle {
    /// The index this handle refers to.
    #[inline]
    pub fn index(self) -> usize {
        self.0 as usize
    }
}

/// Which degrees of freedom a joint removes.
///
/// The convention throughout: the joint's local frame has **+X as its
/// principal axis** — the hinge axis, the slider axis. Picking one and
/// documenting it beats a per-joint axis field that half the constructors
/// would fill in wrong.
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum JointKind {
    /// Anchors coincide; rotation is free. Three degrees of freedom removed.
    Ball,
    /// Anchors coincide and frames align. All six removed — a weld.
    Fixed,
    /// Anchors coincide; rotation is allowed only about the shared +X axis.
    Hinge,
    /// Anchors held between `min` and `max` apart. One degree removed when the
    /// range is a single value, otherwise it is an inequality pair.
    Distance {
        /// Closest the anchors may come, metres.
        min: f32,
        /// Furthest they may separate, metres.
        max: f32,
    },
    /// Translation allowed only along the shared +X axis, between `min` and
    /// `max`; no relative rotation.
    Slider {
        /// Lower travel limit along the axis, metres.
        min: f32,
        /// Upper travel limit, metres.
        max: f32,
    },
}

/// One joint: two bodies, a pair of anchors, and a kind.
#[derive(Clone, Copy, Debug)]
pub struct Joint {
    /// First body. Anchors and frames below are indexed A then B.
    pub body_a: BodyHandle,
    /// Second body.
    pub body_b: BodyHandle,
    /// Anchor point in each body's local frame.
    pub local_anchor_a: Vec3,
    /// Anchor on B.
    pub local_anchor_b: Vec3,
    /// Joint frame in each body's local space. +X is the principal axis.
    pub local_frame_a: Quat,
    /// Joint frame on B.
    pub local_frame_b: Quat,
    /// Which degrees of freedom this joint removes.
    pub kind: JointKind,
    /// Inverse stiffness of the positional part, m/N. Zero is perfectly rigid.
    /// A small positive value gives a joint that visibly stretches under load,
    /// which is often what a soft mount or a rope wants.
    pub compliance: f32,
    /// Inverse stiffness of the angular part, rad/(N m).
    pub angular_compliance: f32,
    /// Set false to disable the joint without removing it.
    pub enabled: bool,
    /// Whether the two jointed bodies also collide with each other.
    ///
    /// Off by default, and that default matters: parts that are bolted
    /// together usually overlap at the join, and letting the contact solver
    /// fight the joint over that overlap produces a ragdoll that shakes itself
    /// apart. Turn it on for the cases that need it — a piston in a cylinder,
    /// a hinge with a hard stop.
    pub collide_connected: bool,
}

impl Joint {
    /// A joint of `kind` between two bodies, anchored at their centres until
    /// [`Joint::at_world_point`] says otherwise.
    pub fn new(body_a: BodyHandle, body_b: BodyHandle, kind: JointKind) -> Self {
        Self {
            body_a,
            body_b,
            local_anchor_a: Vec3::ZERO,
            local_anchor_b: Vec3::ZERO,
            local_frame_a: Quat::IDENTITY,
            local_frame_b: Quat::IDENTITY,
            kind,
            compliance: 0.0,
            angular_compliance: 0.0,
            enabled: true,
            collide_connected: false,
        }
    }

    /// Build a joint pinned at a world-space point, given the two bodies'
    /// current poses. This is how joints are almost always authored: place the
    /// parts, then say where they connect.
    pub fn at_world_point(
        body_a: BodyHandle,
        pose_a: &Isometry,
        body_b: BodyHandle,
        pose_b: &Isometry,
        point: Vec3,
        kind: JointKind,
    ) -> Self {
        let mut joint = Self::new(body_a, body_b, kind);
        joint.local_anchor_a = pose_a.inverse_transform_point(point);
        joint.local_anchor_b = pose_b.inverse_transform_point(point);
        // Frames are captured from the current orientations, so the joint's
        // rest state is whatever the scene was authored in.
        joint.local_frame_a = pose_a.rotation.conjugate();
        joint.local_frame_b = pose_b.rotation.conjugate();
        joint
    }

    /// Same, with the principal axis (hinge or slider) given in world space.
    pub fn at_world_point_with_axis(
        body_a: BodyHandle,
        pose_a: &Isometry,
        body_b: BodyHandle,
        pose_b: &Isometry,
        point: Vec3,
        axis: Vec3,
        kind: JointKind,
    ) -> Self {
        let mut joint = Self::at_world_point(body_a, pose_a, body_b, pose_b, point, kind);
        // A frame whose +X lies along `axis`, expressed in each body.
        let world_frame = frame_from_x_axis(axis);
        joint.local_frame_a = pose_a.rotation.conjugate() * world_frame;
        joint.local_frame_b = pose_b.rotation.conjugate() * world_frame;
        joint
    }

    /// Builder: soften the positional constraint, m/N. Zero is rigid.
    pub fn with_compliance(mut self, compliance: f32) -> Self {
        self.compliance = compliance;
        self
    }

    /// Builder: soften the angular constraint.
    pub fn with_angular_compliance(mut self, compliance: f32) -> Self {
        self.angular_compliance = compliance;
        self
    }

    /// Builder: let the two jointed bodies collide with each other. Off by
    /// default, because a hinge whose halves overlap would fight itself.
    pub fn colliding(mut self) -> Self {
        self.collide_connected = true;
        self
    }

    /// The joint's anchor points and frames in world space, given both poses.
    #[inline]
    pub fn world_anchors(&self, pose_a: &Isometry, pose_b: &Isometry) -> (Vec3, Vec3) {
        (
            pose_a.transform_point(self.local_anchor_a),
            pose_b.transform_point(self.local_anchor_b),
        )
    }

    /// The two joint frames in world space, given the bodies' poses.
    #[inline]
    pub fn world_frames(&self, pose_a: &Isometry, pose_b: &Isometry) -> (Quat, Quat) {
        (
            pose_a.rotation * self.local_frame_a,
            pose_b.rotation * self.local_frame_b,
        )
    }
}

/// A rotation whose +X axis points along `axis`.
pub fn frame_from_x_axis(axis: Vec3) -> Quat {
    let x = axis.normalize_or_zero();
    if x == Vec3::ZERO {
        return Quat::IDENTITY;
    }
    // Shortest rotation taking +X onto `x`.
    let dot = Vec3::X.dot(x);
    if dot > 1.0 - 1e-6 {
        Quat::IDENTITY
    } else if dot < -1.0 + 1e-6 {
        Quat::from_axis_angle(Vec3::Y, std::f32::consts::PI)
    } else {
        let cross = Vec3::X.cross(x);
        Quat::from_xyzw(cross.x, cross.y, cross.z, 1.0 + dot).normalize()
    }
}
