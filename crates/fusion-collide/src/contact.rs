//! Contact data.
//!
//! Contacts are found once per step but solved every substep (PLAN.md §10:
//! substepping is default-on because the stability it buys is worth more than
//! the iterations it costs). Between those substeps the bodies move, so a
//! contact cannot be stored as a world-space point and a world-space depth —
//! both would be stale by the time they are used.
//!
//! Instead each contact stores its two anchors in the *body frames* of the
//! bodies involved, and the normal in the body frame of A. Anchors and normal
//! then move with the bodies for free, and the current separation is recovered
//! by transforming them with the current poses. This is the same reason the
//! warm-starting cache in v0.2 will key on local anchors.

use fusion_math::Vec3;

/// Maximum contact points kept per pair. Four is enough to hold a resting box
/// face flat; more points buy nothing and cost solver iterations.
pub const MAX_CONTACT_POINTS: usize = 4;

/// One point of contact, as *geometry only*.
///
/// Everything the solver needs to remember about a contact — multipliers,
/// activity, incoming velocity — lives in the solver's working set instead.
/// PLAN.md §9 defines a solver as `prepare(State) -> WorkingSet`, and this is
/// that boundary being honoured: a contact point is a fact about where two
/// surfaces meet, not a scratch pad for whichever method happens to be running.
#[derive(Clone, Copy, Debug, Default)]
pub struct ContactPoint {
    /// Anchor on body A, in A's body frame.
    pub local_a: Vec3,
    /// Anchor on body B, in B's body frame.
    pub local_b: Vec3,
    /// This point's normal in A's body frame, from A towards B.
    ///
    /// A point's own, not the pair's. A box tumbling against another can
    /// touch it with a corner one moment and swing a face into it the next,
    /// and a manifold that could only hold one normal for both held the
    /// corner's, and the face went 25 mm in. The pair's
    /// [`ContactManifold::local_normal_a`] is its first point's.
    pub local_normal_a: Vec3,
    /// Separation along the normal at detection time. Negative means
    /// penetrating, and the solver's job is to drive it to zero.
    pub separation: f32,
    /// Whether these anchors were carried over from the previous step.
    ///
    /// A contact in its first step has anchors that are coincident in the
    /// tangent plane by construction, so it has no history to measure slip
    /// against — and static friction, asked for an opinion anyway, ends up
    /// reacting to the tilt the normal solve introduces while resolving the
    /// impact. Only an inherited contact carries a reference worth pulling
    /// back to.
    pub inherited: bool,
}

/// All contact points between one pair of bodies, sharing a normal.
#[derive(Clone, Copy, Debug)]
pub struct ContactManifold {
    /// The lower-indexed body of the pair. Normals point from A towards B.
    pub body_a: u32,
    /// The higher-indexed body.
    pub body_b: u32,
    /// The first point's normal in A's body frame, pointing from A towards
    /// B — the pair's, for whatever wants one. Each point carries its own,
    /// [`ContactPoint::local_normal_a`], which is what the solver uses.
    pub local_normal_a: Vec3,
    /// World normal at detection time. Kept for diagnostics and rendering.
    pub world_normal: Vec3,
    /// Combined friction and restitution for the pair. Carried but unused in
    /// v0.1; the solver that reads them arrives in v0.2.
    pub friction: f32,
    /// Combined restitution for the pair, the maximum of the two materials'.
    pub restitution: f32,
    points: [ContactPoint; MAX_CONTACT_POINTS],
    count: u8,
}

impl ContactManifold {
    /// An empty manifold for a pair, with the normal already established.
    /// Points are added by [`ContactManifold::push`].
    pub fn new(body_a: u32, body_b: u32, local_normal_a: Vec3, world_normal: Vec3) -> Self {
        Self {
            body_a,
            body_b,
            local_normal_a,
            world_normal,
            friction: 0.0,
            restitution: 0.0,
            points: [ContactPoint::default(); MAX_CONTACT_POINTS],
            count: 0,
        }
    }

    /// Add a point, silently dropping it if the manifold is full. Callers that
    /// can produce more than four points reduce first.
    pub fn push(&mut self, p: ContactPoint) -> bool {
        if (self.count as usize) < MAX_CONTACT_POINTS {
            self.points[self.count as usize] = p;
            self.count += 1;
            true
        } else {
            false
        }
    }

    /// The contact points, which is at most [`MAX_CONTACT_POINTS`] of them.
    #[inline]
    pub fn points(&self) -> &[ContactPoint] {
        &self.points[..self.count as usize]
    }

    /// The same, mutably — the solver writes converged multipliers back
    /// through this.
    #[inline]
    pub fn points_mut(&mut self) -> &mut [ContactPoint] {
        &mut self.points[..self.count as usize]
    }

    /// How many contact points this manifold holds.
    #[inline]
    pub fn len(&self) -> usize {
        self.count as usize
    }

    /// True where the pair was found by the broad phase and turned out not
    /// to touch.
    #[inline]
    pub fn is_empty(&self) -> bool {
        self.count == 0
    }

    /// Deepest penetration in the manifold, as a positive depth.
    pub fn max_depth(&self) -> f32 {
        self.points()
            .iter()
            .fold(0.0f32, |d, p| d.max(-p.separation))
    }
}

/// Result of a narrow-phase query in world space, before it is converted into
/// body-local anchors.
#[derive(Clone, Copy, Debug)]
pub struct WorldContact {
    /// Point on A's surface.
    pub point_a: Vec3,
    /// Point on B's surface.
    pub point_b: Vec3,
    /// Unit normal, from A towards B.
    pub normal: Vec3,
    /// Negative when penetrating.
    pub separation: f32,
}
