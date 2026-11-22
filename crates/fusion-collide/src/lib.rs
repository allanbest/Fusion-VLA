//! Fusion L3 — collision detection.
//!
//! Broad phase narrows a quadratic problem to a list of candidate pairs;
//! narrow phase turns each candidate into a contact manifold. Nothing here
//! integrates state. Optional velocity-based reach predicts candidate contacts
//! over the requested interval; the solver decides how to resolve them.

#![deny(unsafe_code)]
// The public API freeze (PLAN.md §29, `docs/API.md`): an undocumented public
// item is one nobody agreed to freeze.
#![deny(missing_docs)]

pub mod broad;
pub mod broad_grid;
pub mod contact;
pub mod grid;
pub mod narrow;
pub mod ray;

use fusion_core::BodySet;
use fusion_math::{Aabb, Isometry, Vec3};

pub use broad::{BroadPair, SweepAndPrune};
pub use broad_grid::Grid;
pub use contact::{ContactManifold, ContactPoint, WorldContact, MAX_CONTACT_POINTS};
pub use grid::SpatialHash;
pub use ray::{pick, ray_vs_shape, Ray};

/// Reusable scratch for one collision pass. Held across frames so the pipeline
/// stops allocating after the first step.
#[derive(Clone, Debug)]
pub struct CollisionPipeline {
    broad_phase: SweepAndPrune,
    /// The other broad phase, for scenes with many bodies of a size. Which one
    /// runs is decided per step from the bounds themselves; both produce the
    /// same sorted list, so the choice is invisible below this line. See
    /// [`broad_grid::Grid::suits`].
    grid: broad_grid::Grid,
    /// True where the last step used the grid. Reported by
    /// [`CollisionPipeline::used_grid`] so a benchmark can say which structure
    /// it measured.
    used_grid: bool,
    aabbs: Vec<Aabb>,
    dynamic: Vec<bool>,
    pairs: Vec<BroadPair>,
    contacts: Vec<WorldContact>,
    /// Body pairs that never collide, sorted, each stored with the lower index
    /// first. Populated from the joints each step.
    exclusions: Vec<(u32, u32)>,
    /// Last step's manifolds, sorted by body pair.
    ///
    /// Contacts are *found* fresh every step but not *born* fresh: a box that
    /// has been resting on the floor for a second is in the same contact it
    /// was in a second ago, and its anchors should say so.
    previous: Vec<ContactManifold>,
    /// Broad-phase margin. Bounds are grown by this much so that a pair which
    /// is about to touch is already in the list.
    pub margin: f32,
    /// The step the detected contacts will be solved over, seconds. Each
    /// body's bounds and each pair's contact tolerance grow by how far the
    /// body can move in it, so a pair that will meet *during* the step has a
    /// contact *before* it — speculative contact, as far as it needs to reach.
    /// Zero keeps the fixed tolerance alone.
    pub sweep_dt: f32,
    /// Maximum speculative travel per body, metres. Set by the simulation.
    pub max_sweep_distance: f32,
    sweeps: Vec<f32>,
    /// Narrow-phase tolerance: how far apart two surfaces may be and still
    /// generate a contact point. See
    /// [`narrow::primitives::DEFAULT_CONTACT_TOLERANCE`].
    pub contact_tolerance: f32,
}

impl Default for CollisionPipeline {
    fn default() -> Self {
        Self {
            broad_phase: SweepAndPrune::default(),
            grid: broad_grid::Grid::default(),
            used_grid: false,
            aabbs: Vec::new(),
            dynamic: Vec::new(),
            pairs: Vec::new(),
            contacts: Vec::new(),
            exclusions: Vec::new(),
            previous: Vec::new(),
            margin: 0.02,
            sweep_dt: 0.0,
            max_sweep_distance: 2.0,
            sweeps: Vec::new(),
            contact_tolerance: narrow::primitives::DEFAULT_CONTACT_TOLERANCE,
        }
    }
}

impl CollisionPipeline {
    /// A pipeline with the default margins. Keep one and reuse it: the
    /// buffers and the previous step's manifolds it holds are what make
    /// contact anchors survive from one step to the next.
    pub fn new() -> Self {
        Self::default()
    }

    /// Run broad phase and narrow phase, filling `out` with one manifold per
    /// touching pair.
    ///
    /// Manifolds store body-local anchors, so they stay valid as the solver
    /// moves bodies during its substeps.
    pub fn detect(&mut self, bodies: &BodySet, out: &mut Vec<ContactManifold>) {
        out.clear();
        let n = bodies.len();
        self.aabbs.clear();
        self.dynamic.clear();
        self.aabbs.reserve(n);
        self.dynamic.reserve(n);
        self.sweeps.clear();
        self.sweeps.reserve(n);
        for i in 0..n {
            // How far this body can travel this step: its speed, and its spin
            // times the farthest any of it is from its centre. A body at 8 m/s
            // covers 3 cm of a 1/240 s step, six times the fixed tolerance —
            // it was found already 3 cm into the floor, and a 2 cm pane was
            // stepped over entirely.
            let sweep = sweep_of(bodies, i, self.sweep_dt, self.max_sweep_distance);
            self.sweeps.push(sweep);
            self.aabbs
                .push(bodies.aabb(i).expanded(self.margin + sweep));
            self.dynamic.push(bodies.is_dynamic(i));
        }

        // Whichever structure suits the scene. Both emit the same pairs in
        // the same order — sorted, ascending, no duplicates — so this decision
        // changes the cost of the step and nothing else about it. That is
        // deliberate: without it, a scene would change behaviour on the step
        // it crossed a body count, which is the worst possible way to find out
        // an optimization exists.
        self.used_grid = broad_grid::Grid::suits(&self.aabbs);
        if self.used_grid {
            self.grid
                .find_pairs(&self.aabbs, &self.dynamic, &mut self.pairs);
        } else {
            self.broad_phase
                .find_pairs(&self.aabbs, &self.dynamic, &mut self.pairs);
            self.pairs.sort_unstable();
        }

        for index in 0..self.pairs.len() {
            let (a, b) = self.pairs[index];
            if self.excluded(a, b) || bodies.is_removed(a as usize) || bodies.is_removed(b as usize)
            {
                continue;
            }
            let (ia, ib) = (a as usize, b as usize);
            let tolerance = self.contact_tolerance + self.sweeps[ia] + self.sweeps[ib];
            let Some(mut manifold) = collide_pair(
                bodies,
                ia,
                ib,
                tolerance,
                self.contact_tolerance,
                self.sweep_dt,
                &mut self.contacts,
            ) else {
                continue;
            };
            if let Ok(index) = self
                .previous
                .binary_search_by_key(&(a, b), |m| (m.body_a, m.body_b))
            {
                inherit_anchors(
                    &self.previous[index],
                    &mut manifold,
                    bodies,
                    anchor_reach(bodies, ia, ib),
                    true,
                );
            }
            out.push(manifold);
        }

        // Keep this step's manifolds for the next one, sorted so the lookup is
        // a binary search. Sorting also makes the inheritance independent of
        // the order the broad phase happened to emit pairs in, which matters
        // for determinism (PLAN.md §34).
        self.previous.clear();
        self.previous.extend_from_slice(out);
        self.previous.sort_unstable_by_key(|m| (m.body_a, m.body_b));
    }

    /// True where the last detection used the uniform grid rather than the
    /// sweep.
    pub fn used_grid(&self) -> bool {
        self.used_grid
    }

    /// Pairs the last broad phase emitted, before the narrow phase rejected
    /// any of them.
    pub fn pair_count(&self) -> usize {
        self.pairs.len()
    }

    /// How far an anchor may drift and still be considered the same contact
    /// point. Beyond this the surfaces have genuinely slid, and the contact is
    /// re-anchored where they touch now.
    pub const ANCHOR_MATCH_DISTANCE: f32 = 0.05;

    /// The manifolds kept from the last detection, which the next one
    /// carries anchors over from. For a state file to save.
    pub fn remembered(&self) -> &[ContactManifold] {
        &self.previous
    }

    /// Replace the manifolds kept from the last detection: what a run
    /// resumed from a state file had when it was saved.
    pub fn remember(&mut self, manifolds: impl IntoIterator<Item = ContactManifold>) {
        self.previous.clear();
        self.previous.extend(manifolds);
        self.previous.sort_unstable_by_key(|m| (m.body_a, m.body_b));
    }

    /// Forget every contact a body carried over from earlier steps.
    ///
    /// For a body whose shape has just changed under it — a piece broken off,
    /// or the piece left behind. Its remembered anchors are points on a shape
    /// it no longer has, and matching new contacts against them would hand
    /// the solver a history that never happened.
    pub fn forget_body(&mut self, body: u32) {
        self.previous
            .retain(|m| m.body_a != body && m.body_b != body);
    }

    /// Replace the set of pairs that must never generate contacts.
    ///
    /// Sorted on the way in so the lookup is a binary search rather than a
    /// hash: the list is small, rebuilt every step, and looked up once per
    /// broad-phase pair.
    pub fn set_exclusions(&mut self, pairs: impl IntoIterator<Item = (u32, u32)>) {
        self.exclusions.clear();
        self.exclusions.extend(
            pairs
                .into_iter()
                .map(|(a, b)| if a <= b { (a, b) } else { (b, a) }),
        );
        self.exclusions.sort_unstable();
        self.exclusions.dedup();
    }

    #[inline]
    fn excluded(&self, a: u32, b: u32) -> bool {
        !self.exclusions.is_empty() && self.exclusions.binary_search(&(a, b)).is_ok()
    }
}

/// How far body `i` can travel in `dt`: its speed, and its spin times the
/// farthest any of it is from its centre, capped at `cap` metres. Zero for a
/// fixed or removed body, and for `dt` zero.
pub fn sweep_of(bodies: &BodySet, i: usize, dt: f32, cap: f32) -> f32 {
    if bodies.is_removed(i) || !(bodies.is_movable(i) || bodies.is_kinematic(i)) || dt <= 0.0 {
        return 0.0;
    }
    let reach = bodies.shape[i].bounding_radius();
    ((bodies.linear_velocity[i].length() + bodies.angular_velocity[i].length() * reach) * dt)
        .min(cap.max(0.0))
}

/// The narrow phase for one pair, at the poses the bodies have now: a
/// manifold of at most four points, or `None` where nothing is within
/// `tolerance`. Anchors are fresh; [`inherit_anchors`] carries history over.
///
/// `near` is the base tolerance: points that will be within it by the end of
/// `sweep_dt` are the ones kept first. Used by [`CollisionPipeline::detect`]
/// and by a solver re-detecting a pair part-way through a step.
pub fn collide_pair(
    bodies: &BodySet,
    ia: usize,
    ib: usize,
    tolerance: f32,
    near: f32,
    sweep_dt: f32,
    scratch: &mut Vec<WorldContact>,
) -> Option<ContactManifold> {
    let pose_a = bodies.pose(ia);
    let pose_b = bodies.pose(ib);

    // Narrow phase runs in a frame centred on A, never in world space.
    //
    // Contact geometry is millimetre-scale work, and f32 has 15 microns of
    // resolution at a coordinate of 200 m. Generating the four corners of a
    // resting box face at absolute coordinates rounds each of them
    // independently, and the solver reads the disagreement as a tilt of about
    // 1.5e-5 radians. It then corrects the tilt that is not there, the box
    // rocks, and friction on a rocking box is roughly twice friction on a
    // flat one: the same slide that travels 1.7 m at the origin stops after
    // 0.85 m at 200 m up.
    //
    // Subtracting the two centres first is exact — they are close together,
    // so the subtraction is — and everything after it happens at the scale of
    // the bodies rather than the scale of the world. This is PLAN.md §7's
    // domain-local principle applied at the smallest scale it matters: one
    // contact pair.
    let local_a = Isometry::new(Vec3::ZERO, pose_a.rotation);
    // Differenced in double precision by the body store: at 20 km the f32
    // spacing is 2 mm, so subtracting two f32 centres would put a 2 mm error
    // into every contact before the narrow phase even runs.
    let local_b = Isometry::new(bodies.relative_translation(ia, ib), pose_b.rotation);

    scratch.clear();
    narrow::primitives::collide(
        &bodies.shape[ia],
        &local_a,
        &bodies.shape[ib],
        &local_b,
        tolerance,
        scratch,
    );
    if scratch.is_empty() {
        return None;
    }
    // Kept by where each point will be at the end of the step, not where it
    // is now: a tumbling shard's corner that is about to swing into the floor
    // is the one that matters, and chosen by present depth it was dropped for
    // four resting corners and went eleven centimetres in.
    let towards_b = local_b.translation;
    let (va, wa) = (bodies.linear_velocity[ia], bodies.angular_velocity[ia]);
    let (vb, wb) = (bodies.linear_velocity[ib], bodies.angular_velocity[ib]);
    reduce_contacts(scratch, near, |c| {
        let at_a = va + wa.cross(c.point_a);
        let at_b = vb + wb.cross(c.point_b - towards_b);
        c.separation + sweep_dt * (at_b - at_a).dot(c.normal)
    });

    // The normal is a direction, so the shifted frame leaves it unchanged.
    let world_normal = scratch[0].normal;
    let mut manifold = ContactManifold::new(
        ia as u32,
        ib as u32,
        pose_a.inverse_transform_direction(world_normal),
        world_normal,
    );
    // Pair-combining rules. Geometric mean for friction and the maximum for
    // restitution are the usual conventions.
    manifold.friction = (bodies.material[ia].friction * bodies.material[ib].friction).sqrt();
    manifold.restitution = bodies.material[ia]
        .restitution
        .max(bodies.material[ib].restitution);
    for c in scratch.iter() {
        manifold.push(ContactPoint {
            local_a: local_a.inverse_transform_point(c.point_a),
            local_b: local_b.inverse_transform_point(c.point_b),
            local_normal_a: pose_a.inverse_transform_direction(c.normal),
            separation: c.separation,
            inherited: false,
        });
    }
    (!manifold.is_empty()).then_some(manifold)
}

/// How far an anchor may drift between two detections of a pair and still
/// be the same contact point. Small bodies match over a small distance: a
/// shard's corners are a few centimetres apart, and matched over five a
/// corner deep in the floor took a shallow neighbour's anchors every step and
/// was never pushed out.
pub fn anchor_reach(bodies: &BodySet, ia: usize, ib: usize) -> f32 {
    let thinnest = |i: usize| {
        let a = bodies.shape[i].local_aabb();
        (a.max - a.min).min_element()
    };
    CollisionPipeline::ANCHOR_MATCH_DISTANCE.min(0.3 * thinnest(ia).min(thinnest(ib)))
}

/// Match `manifold`'s points to `old`'s, one to one, nearest anchor on B
/// first. Returns, for each point of `manifold`, the index of the point of
/// `old` it is the same contact as.
///
/// With `adopt`, a matched point takes the old anchors and is marked
/// inherited — but only where they still say the same depth (see below).
/// That is what makes static friction mean anything. A contact re-derived
/// from scratch every step has its two anchors coincident in the tangent
/// plane by construction, so "slip since detection" is always zero and the
/// only tangential signal left for friction to see is the tilt the normal
/// solve introduces while pushing bodies apart — which is why a box dropped
/// flat used to slide 0.4 m on landing, correcting a slide that never
/// happened. Kept anchors accumulate real slip instead. Once the drift
/// exceeds `reach` the surfaces have slid for real, and the contact
/// re-anchors rather than dragging its history along.
///
/// Without `adopt`, a matched point keeps its fresh anchors and takes only
/// the old one's inherited flag: a re-detection inside a step, whose whole
/// point is that the old anchors have gone stale.
pub fn inherit_anchors(
    old: &ContactManifold,
    manifold: &mut ContactManifold,
    bodies: &BodySet,
    reach: f32,
    adopt: bool,
) -> [Option<usize>; contact::MAX_CONTACT_POINTS] {
    let (ia, ib) = (manifold.body_a as usize, manifold.body_b as usize);
    let pose_a = Isometry::new(Vec3::ZERO, bodies.pose(ia).rotation);
    let pose_b = Isometry::new(
        bodies.relative_translation(ia, ib),
        bodies.pose(ib).rotation,
    );
    let mut taken = [false; contact::MAX_CONTACT_POINTS];
    let mut matches = [None; contact::MAX_CONTACT_POINTS];
    for (slot, point) in matches.iter_mut().zip(manifold.points_mut()) {
        let matched = old
            .points()
            .iter()
            .enumerate()
            .filter(|(k, _)| !taken[*k])
            .min_by(|(_, x), (_, y)| {
                x.local_b
                    .distance_squared(point.local_b)
                    .total_cmp(&y.local_b.distance_squared(point.local_b))
            });
        let Some((k, previous)) = matched else {
            continue;
        };
        if previous.local_b.distance(point.local_b) >= reach {
            continue;
        }
        if adopt {
            // The old anchors as they stand now, along the normal: the same
            // contact only if they say the same depth. Carried anchors are
            // for friction's sideways memory; they must never hide how deep
            // the contact has gone.
            let normal = pose_a.transform_direction(point.local_normal_a);
            let then = (pose_b.transform_point(previous.local_b)
                - pose_a.transform_point(previous.local_a))
            .dot(normal);
            if (then - point.separation).abs() >= ANCHOR_DEPTH_AGREEMENT {
                continue;
            }
            point.local_a = previous.local_a;
            point.local_b = previous.local_b;
            point.inherited = true;
        } else {
            point.inherited = previous.inherited;
        }
        taken[k] = true;
        *slot = Some(k);
    }
    matches
}

/// How far a carried-over contact's depth may disagree with the one just
/// measured and still be the same contact, metres.
///
/// A tenth of a millimetre. At two, a rolling ball sank three millimetres and
/// popped back, over and over: its anchor is a point of the ball, carried
/// from step to step as the ball turns, and it rises off the floor as it goes
/// round, so the ball reads as clear of the floor it is sinking into. At half
/// a millimetre it still sank one. A resting or sliding contact's carried
/// anchors agree with the fresh ones to microns, so nothing that should keep
/// its history loses it.
const ANCHOR_DEPTH_AGREEMENT: f32 = 1e-4;

/// Down to [`contact::MAX_CONTACT_POINTS`], keeping the ones that matter.
///
/// A manifold holds four points, and it took the first four it was given.
/// The box routines never give more; a hull gives one per vertex in contact,
/// in vertex order, so a shard of broken glass with one corner 36 mm into
/// the floor and five more resting on it kept four resting ones and dropped
/// the corner — and lay in the floor for as long as anyone watched. So: the
/// deepest by the end of the step first, then the point farthest from it, then the one that spans
/// the largest triangle with those two, then the one farthest from all three.
/// The deepest is never lost, and the four spread as wide as the patch does.
fn reduce_contacts(
    contacts: &mut Vec<contact::WorldContact>,
    near: f32,
    predicted: impl Fn(&contact::WorldContact) -> f32,
) {
    use contact::MAX_CONTACT_POINTS;
    if contacts.len() <= MAX_CONTACT_POINTS {
        return;
    }
    let at = |c: &contact::WorldContact| c.point_a;
    // The points that will be touching by the end of the step come first.
    // A wide speculative search finds corners that are near but clear, and
    // spreading the four slots over those left a light shard under a heavy
    // block one real support and pushed it through the floor.
    let mut pool: Vec<usize> = (0..contacts.len())
        .filter(|&k| predicted(&contacts[k]) <= near)
        .collect();
    if pool.len() < MAX_CONTACT_POINTS {
        // Too few: all of them, then the nearest of the rest.
        let mut rest: Vec<usize> = (0..contacts.len()).filter(|k| !pool.contains(k)).collect();
        rest.sort_by(|&a, &b| predicted(&contacts[a]).total_cmp(&predicted(&contacts[b])));
        pool.extend(rest.into_iter().take(MAX_CONTACT_POINTS - pool.len()));
        let kept: Vec<contact::WorldContact> = pool.iter().map(|&k| contacts[k]).collect();
        contacts.clear();
        contacts.extend(kept);
        return;
    }
    let mut chosen: Vec<usize> = Vec::with_capacity(MAX_CONTACT_POINTS);
    // Deepest by the end of the step: the one that will be furthest in if
    // nothing stops it.
    let deepest = pool
        .iter()
        .copied()
        .min_by(|&a, &b| predicted(&contacts[a]).total_cmp(&predicted(&contacts[b])))
        .unwrap_or(0);
    chosen.push(deepest);
    let p0 = at(&contacts[deepest]);
    let far = pool
        .iter()
        .copied()
        .filter(|k| !chosen.contains(k))
        .max_by(|&a, &b| {
            (at(&contacts[a]) - p0)
                .length_squared()
                .total_cmp(&(at(&contacts[b]) - p0).length_squared())
        });
    if let Some(far) = far {
        chosen.push(far);
        let p1 = at(&contacts[far]);
        let wide = pool
            .iter()
            .copied()
            .filter(|k| !chosen.contains(k))
            .max_by(|&a, &b| {
                let area = |k: usize| (p1 - p0).cross(at(&contacts[k]) - p0).length_squared();
                area(a).total_cmp(&area(b))
            });
        if let Some(wide) = wide {
            chosen.push(wide);
            let p2 = at(&contacts[wide]);
            let last = pool
                .iter()
                .copied()
                .filter(|k| !chosen.contains(k))
                .max_by(|&a, &b| {
                    let reach = |k: usize| {
                        let q = at(&contacts[k]);
                        (q - p0)
                            .length_squared()
                            .min((q - p1).length_squared())
                            .min((q - p2).length_squared())
                    };
                    reach(a).total_cmp(&reach(b))
                });
            if let Some(last) = last {
                chosen.push(last);
            }
        }
    }
    let kept: Vec<contact::WorldContact> = chosen.iter().map(|&k| contacts[k]).collect();
    contacts.clear();
    contacts.extend(kept);
}
