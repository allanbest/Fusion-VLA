//! Islands and sleeping.
//!
//! PLAN.md §26 puts sleeping at the top of the adaptive-resolution list, and
//! §10 calls stacking "mostly a persistence-caching and iteration-count
//! problem". Sleeping addresses both: a settled pile stops costing anything,
//! and — because a sleeping body cannot be nudged by solver noise — it also
//! stops the slow creep that eventually topples a stack nobody is touching.
//!
//! The unit of sleep is the **island**, not the body. Bodies joined by
//! contacts or joints share a fate: putting one box of a stack to sleep while
//! its neighbour is still settling would have the sleeping box act as an
//! immovable wall, and the pile would lock into whatever shape it happened to
//! be in. So an island sleeps only when every one of its members has been
//! still for long enough, and wakes the instant any of them is disturbed.

use fusion_collide::ContactManifold;
use fusion_core::{BodySet, Joint};

/// When a body counts as "not moving".
#[derive(Clone, Copy, Debug)]
pub struct SleepPolicy {
    /// Whether bodies may sleep at all. On by default; turning it off is a
    /// way to check whether a settling problem is the sleep rule's doing.
    pub enabled: bool,
    /// Speed below which a body is a candidate for sleep, m/s.
    pub linear_threshold: f32,
    /// Angular speed likewise, rad/s.
    pub angular_threshold: f32,
    /// How long it must stay that way, seconds.
    pub time_to_sleep: f32,
    /// Penetration depth that keeps an island awake regardless of how slowly
    /// it is moving, metres.
    ///
    /// Depenetration is deliberately rate-limited, so bodies that start
    /// overlapped push apart slowly — slowly enough to look asleep. Freezing
    /// them there would leave the overlap permanently, and a solver that
    /// resolves a problem by declining to look at it is worse than one that
    /// resolves it slowly.
    pub max_sleeping_penetration: f32,
}

impl Default for SleepPolicy {
    fn default() -> Self {
        Self {
            enabled: true,
            // Two centimetres a second. Deliberately tight: a looser
            // threshold once let deeper piles sleep through their jitter,
            // but it also freezes a body that is still settling — at 5 cm/s a box dropped flat goes
            // to sleep 6 cm from where it would have come to rest. Sleeping is
            // an optimization, and an optimization that changes the answer is
            // not one.
            linear_threshold: 0.02,
            angular_threshold: 0.05,
            time_to_sleep: 0.5,
            max_sleeping_penetration: 2e-3,
        }
    }
}

/// Union-find over bodies, rebuilt each step from the current contacts and
/// joints.
#[derive(Clone, Debug, Default)]
pub struct IslandSet {
    parent: Vec<u32>,
    /// Scratch: the smallest sleep timer seen in each island so far.
    island_timer: Vec<f32>,
    island_count: usize,
}

impl IslandSet {
    /// An empty set, to be filled by the step that uses it.
    pub fn new() -> Self {
        Self::default()
    }

    /// Connected groups of bodies found by the last build. Bodies in
    /// different islands cannot affect each other this step, which is what
    /// makes islands the unit sleeping and (one day) threading work on.
    pub fn island_count(&self) -> usize {
        self.island_count
    }

    fn find(&mut self, mut x: u32) -> u32 {
        while self.parent[x as usize] != x {
            // Path halving: keeps the trees flat without a second pass.
            let grandparent = self.parent[self.parent[x as usize] as usize];
            self.parent[x as usize] = grandparent;
            x = grandparent;
        }
        x
    }

    fn union(&mut self, a: u32, b: u32) {
        let (ra, rb) = (self.find(a), self.find(b));
        if ra != rb {
            // Always attach the larger index under the smaller, so the result
            // does not depend on the order pairs arrive in. Determinism (§34)
            // has to survive the bookkeeping as well as the physics.
            let (lo, hi) = if ra < rb { (ra, rb) } else { (rb, ra) };
            self.parent[hi as usize] = lo;
        }
    }

    /// Rebuild islands, update sleep timers, and put settled islands to sleep.
    ///
    /// Static bodies deliberately do not join islands. Every dynamic body in a
    /// scene touches the ground eventually, and if the ground linked them they
    /// would all be one island and nothing would ever sleep.
    pub fn update(
        &mut self,
        bodies: &mut BodySet,
        manifolds: &[ContactManifold],
        joints: &[Joint],
        policy: &SleepPolicy,
        dt: f32,
    ) {
        let n = bodies.len();
        self.parent.clear();
        self.parent.extend(0..n as u32);

        if !policy.enabled {
            self.island_count = n;
            for i in 0..n {
                if bodies.sleeping[i] {
                    bodies.wake(i);
                }
            }
            return;
        }

        for m in manifolds {
            let (a, b) = (m.body_a as usize, m.body_b as usize);
            if bodies.is_dynamic(a) && bodies.is_dynamic(b) {
                self.union(m.body_a, m.body_b);
            }
            // Still resolving an overlap: not settled, however slow it looks.
            if m.max_depth() > policy.max_sleeping_penetration {
                if bodies.is_dynamic(a) {
                    bodies.wake(a);
                }
                if bodies.is_dynamic(b) {
                    bodies.wake(b);
                }
            }
        }
        for j in joints.iter().filter(|j| j.enabled) {
            let (a, b) = (j.body_a.index(), j.body_b.index());
            if bodies.is_dynamic(a) && bodies.is_dynamic(b) {
                self.union(j.body_a.0, j.body_b.0);
            }
        }

        // Per-body: how long has this one been still?
        for i in 0..n {
            if !bodies.is_dynamic(i) {
                continue;
            }
            let still = bodies.linear_velocity[i].length() < policy.linear_threshold
                && bodies.angular_velocity[i].length() < policy.angular_threshold;
            if still {
                bodies.sleep_timer[i] += dt;
            } else {
                bodies.sleep_timer[i] = 0.0;
                if bodies.sleeping[i] {
                    bodies.wake(i);
                }
            }
        }

        // Per-island: the island is only as settled as its least settled member.
        self.island_timer.clear();
        self.island_timer.resize(n, f32::INFINITY);
        let mut roots = 0;
        for i in 0..n {
            if !bodies.is_dynamic(i) {
                continue;
            }
            let root = self.find(i as u32) as usize;
            if self.island_timer[root].is_infinite() {
                roots += 1;
            }
            self.island_timer[root] = self.island_timer[root].min(bodies.sleep_timer[i]);
        }
        self.island_count = roots;

        for i in 0..n {
            if !bodies.is_dynamic(i) {
                continue;
            }
            let root = self.find(i as u32) as usize;
            let settled = self.island_timer[root] >= policy.time_to_sleep;
            if settled && !bodies.sleeping[i] {
                bodies.sleep(i);
            } else if !settled && bodies.sleeping[i] {
                bodies.wake(i);
            }
        }
    }
}
