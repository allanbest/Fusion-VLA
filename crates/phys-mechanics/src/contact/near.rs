//! The pairs that can be in contact, remembered while things stay near
//! where they were.
//!
//! A solve asks for the contacts at one state after another, each a small
//! move from the last: a line search's trials, a Newton iteration's result.
//! Searching all of space each time finds mostly the same thing — that
//! nearly every pair of primitives is far too far apart to matter.
//!
//! So at a *reference* state the search is done once, a little wider than
//! contact needs: every pair whose gap is within a `margin` of contact is
//! *listed*, with how much room it has. A gap is a distance, so it can
//! shrink by no more than its two sides move towards each other. While no
//! object has moved, relative to the others, by more than the margin
//! allows, every pair that is not listed is certainly still out of contact,
//! and a listed pair with more room than its objects have moved is too. What
//! remains is evaluated exactly as the full search would have evaluated it,
//! in the same order — so the pairs found are the same ones, to the bit.
//! When something has moved too far, the list is made again from there.
//!
//! How far objects have moved is measured per object, as a shared
//! translation (which changes no distance within the object) and a spread
//! about it: a block falling a long way without deforming keeps its list.

use core_math::Vec3;
use core_space::Aabb;

use super::{Attach, Contact, Context, Edge, Other, Point, Triangle, POINTS_PER_CHUNK};

/// The margin, in contact precisions: pairs with a gap below `(1 + this) ε`
/// are listed. A wider list holds for longer and costs more to go through,
/// and which is better depends on the scene (how fast things deform and
/// turn, how close together their primitives are), so the margin adapts
/// between these bounds: it doubles when motion outruns a list that was
/// short, and halves when a list comes out long. The margin changes what a
/// search costs and never what it finds, so it is not simulation state.
const MARGIN: (f64, f64) = (4.0, 256.0);

/// A list that served fewer states than this before motion outran it was
/// too narrow.
const SERVES: u32 = 12;

/// A list is long when it has more pairs than this many times the
/// primitives it is over, and short enough to widen below this many times.
const LONG: usize = 4;
const SHORT: usize = 1;

/// A primitive's points move, over a step, by their object's shared
/// translation and at most its spread more. A swept test measures a side by
/// the mean of its points and how far they are from that mean: for a
/// triangle at most three spreads in all, for an edge or a point fewer.
const SIDES: f64 = 3.0;

/// The share of a margin motion may use up before the list is made again;
/// the rest is room for rounding.
const USE: f64 = 0.8;

/// Objects beyond which the room between each pair of them is not worked
/// out (it is quadratic in their number): each gets the margin alone.
const MOST_OBJECTS: usize = 512;

#[derive(Clone, Debug, Default)]
pub(super) struct Near {
    /// The lists stand for the reference state of the stage under way.
    valid: bool,
    precision: f64,
    self_contact: bool,
    margin: f64,
    /// The margin to list with next, in precisions (zero: not yet chosen),
    /// and how many states the current list has served.
    width: f64,
    served: u32,
    /// Per point, where it was.
    reference: Vec<Vec3>,
    /// Per point `i`, its listed triangles are
    /// `triangles[triangle_start[i]..triangle_start[i + 1]]`, ascending,
    /// each with its room: how far its gap was above ε.
    triangle_start: Vec<u32>,
    triangles: Vec<u32>,
    triangle_room: Vec<f64>,
    /// Likewise per edge `i`, the listed edges after it.
    edge_start: Vec<u32>,
    edges: Vec<u32>,
    edge_room: Vec<f64>,
    /// Per object: whether it has points that move, and whether pairs
    /// within it are searched.
    object: Vec<Kind>,
    /// Per object: the least room any unlisted pair between it and another
    /// object has.
    isolation: Vec<f64>,
    /// Per object, since the reference state: the mean displacement of its
    /// points, the furthest any is from that mean, and that plus the mean's
    /// own distance from the mean of all moving objects.
    shift: Vec<Vec3>,
    spread: Vec<f64>,
    drift: Vec<f64>,
    count: Vec<u32>,
    /// Per object, over the step being limited: the mean displacement of
    /// its points and the furthest any point of it goes from that mean.
    step_shift: Vec<Vec3>,
    step_spread: Vec<f64>,
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
enum Kind {
    /// No points or edges: takes no part.
    #[default]
    Absent,
    Still,
    Moving,
    /// Moving, and its own primitives are paired with one another.
    MovingSelf,
}

impl Near {
    /// Whether the lists still stand with the points at `positions`:
    /// measures how far every object has moved since the reference state.
    pub(super) fn holds(
        &mut self,
        points: &[Point],
        positions: &[Vec3],
        precision: f64,
        self_contact: bool,
    ) -> bool {
        if !self.valid
            || self.precision.to_bits() != precision.to_bits()
            || self.self_contact != self_contact
            || self.reference.len() != positions.len()
        {
            return false;
        }
        let held = self.measure(points, positions);
        if held {
            self.served += 1;
        } else if self.served < SERVES
            && self.triangles.len() + self.edges.len()
                <= SHORT * (points.len() + self.edge_start.len())
        {
            self.width = (2.0 * self.width).min(MARGIN.1);
        }
        held
    }

    /// As [`Near::holds`], for a caller that only asks.
    pub(super) fn stands(
        &mut self,
        points: &[Point],
        positions: &[Vec3],
        precision: f64,
        self_contact: bool,
    ) -> bool {
        self.valid
            && self.precision.to_bits() == precision.to_bits()
            && self.self_contact == self_contact
            && self.reference.len() == positions.len()
            && self.measure(points, positions)
    }

    /// Whether, from a state the lists stand at, a step can bring only
    /// listed pairs close. `motion` is a moving point's displacement over
    /// the step as a translation and a bound on what more it moves (a rigid
    /// body's centre and its turning).
    ///
    /// A swept test passes a pair over when its sides cannot close the gap:
    /// when they move towards each other by less than nine tenths of it.
    /// Sides of objects `a` and `b` close by at most `closing(a, b)`, and an
    /// unlisted pair's gap is above ε by its room less what has closed
    /// already; so if every object's own part of both stays within its part
    /// of the room, no unlisted pair can matter.
    pub(super) fn sweeps(
        &mut self,
        points: &[Point],
        motion: impl Fn(&Point) -> (Vec3, f64),
    ) -> bool {
        let n = self.object.len();
        self.step_shift.clear();
        self.step_shift.resize(n, Vec3::ZERO);
        self.step_spread.clear();
        self.step_spread.resize(n, 0.0);
        let moves = |p: &Point| !matches!(p.attach, Attach::Still(_));
        for point in points.iter().filter(|p| moves(p)) {
            self.step_shift[point.owner] += motion(point).0;
        }
        let (mut common, mut movers) = (Vec3::ZERO, 0usize);
        for o in 0..n {
            if self.count[o] > 0 {
                self.step_shift[o] = self.step_shift[o] / self.count[o] as f64;
                common += self.step_shift[o];
                movers += 1;
            }
        }
        if movers > 0 {
            common /= movers as f64;
        }
        for point in points.iter().filter(|p| moves(p)) {
            let (translation, more) = motion(point);
            let apart = (translation - self.step_shift[point.owner]).length() + more;
            let spread = &mut self.step_spread[point.owner];
            *spread = spread.max(apart);
        }
        for o in 0..n {
            if self.object[o] == Kind::Absent {
                continue;
            }
            let reach = SIDES * self.step_spread[o] + (self.step_shift[o] - common).length();
            if !within(reach + 0.9 * self.drift[o], 0.5 * USE * self.isolation[o]) {
                return false;
            }
            if self.object[o] == Kind::MovingSelf
                && !within(
                    2.0 * SIDES * self.step_spread[o] + 0.9 * 2.0 * self.spread[o],
                    USE * self.margin,
                )
            {
                return false;
            }
        }
        true
    }

    /// A bound on how far a primitive of object `a` and one of `b` move
    /// towards each other over the step.
    #[inline]
    fn closing(&self, a: usize, b: usize) -> f64 {
        SIDES * (self.step_spread[a] + self.step_spread[b])
            + (self.step_shift[a] - self.step_shift[b]).length()
    }

    /// Whether the step certainly cannot bring a listed pair with `room` at
    /// the reference state close enough for a swept test to act on it.
    #[inline]
    pub(super) fn out_of_reach(&self, room: f64, a: usize, b: usize) -> bool {
        0.9 * (room - self.closed(a, b)) - self.closing(a, b) > (1.0 - USE) * 0.5 * self.margin
    }

    fn measure(&mut self, points: &[Point], positions: &[Vec3]) -> bool {
        let n = self.object.len();
        self.shift.clear();
        self.shift.resize(n, Vec3::ZERO);
        self.spread.clear();
        self.spread.resize(n, 0.0);
        self.count.clear();
        self.count.resize(n, 0);
        let moves = |p: &Point| !matches!(p.attach, Attach::Still(_));
        let mut finite = true;
        for (i, point) in points.iter().enumerate().filter(|(_, p)| moves(p)) {
            let d = positions[i] - self.reference[i];
            finite &= d.is_finite();
            self.shift[point.owner] += d;
            self.count[point.owner] += 1;
        }
        if !finite {
            return false;
        }
        let (mut common, mut movers) = (Vec3::ZERO, 0usize);
        for o in 0..n {
            if self.count[o] > 0 {
                self.shift[o] = self.shift[o] / self.count[o] as f64;
                common += self.shift[o];
                movers += 1;
            }
        }
        if movers > 0 {
            common /= movers as f64;
        }
        for (i, point) in points.iter().enumerate().filter(|(_, p)| moves(p)) {
            let apart = (positions[i] - self.reference[i] - self.shift[point.owner]).length();
            let spread = &mut self.spread[point.owner];
            *spread = spread.max(apart);
        }
        self.drift.clear();
        self.drift.resize(n, 0.0);
        for o in 0..n {
            if self.object[o] == Kind::Absent {
                continue;
            }
            self.drift[o] = self.spread[o] + (self.shift[o] - common).length();
            // Pairs of two objects: each may use half of the room between them.
            if !within(self.drift[o], 0.5 * USE * self.isolation[o]) {
                return false;
            }
            // Pairs within one object close by twice its spread at most.
            if self.object[o] == Kind::MovingSelf
                && !within(2.0 * self.spread[o], USE * self.margin)
            {
                return false;
            }
        }
        true
    }

    /// How far a pair of primitives of objects `a` and `b` can have moved
    /// towards each other since the reference state.
    #[inline]
    fn closed(&self, a: usize, b: usize) -> f64 {
        self.spread[a] + self.spread[b] + (self.shift[a] - self.shift[b]).length()
    }

    /// Whether a listed pair with `room` at the reference state is certainly
    /// still out of contact. (Not so if anything here is not a number.)
    #[inline]
    pub(super) fn clear_of_contact(&self, room: f64, a: usize, b: usize) -> bool {
        room - self.closed(a, b) > (1.0 - USE) * 0.5 * self.margin
    }

    #[inline]
    pub(super) fn triangles_of(&self, point: usize) -> impl Iterator<Item = (u32, f64)> + '_ {
        let range = self.triangle_start[point] as usize..self.triangle_start[point + 1] as usize;
        self.triangles[range.clone()]
            .iter()
            .copied()
            .zip(self.triangle_room[range].iter().copied())
    }

    #[inline]
    pub(super) fn edges_of(&self, edge: usize) -> impl Iterator<Item = (u32, f64)> + '_ {
        let range = self.edge_start[edge] as usize..self.edge_start[edge + 1] as usize;
        self.edges[range.clone()]
            .iter()
            .copied()
            .zip(self.edge_room[range].iter().copied())
    }
}

/// Whether `value` is at most `bound`: not so if either is not a number,
/// which is what every test of the lists wants (a state that cannot be
/// measured is one the lists do not stand for).
#[inline]
fn within(value: f64, bound: f64) -> bool {
    value <= bound
}

/// One chunk's part of a list: per item its number of partners, then the
/// partners and their room.
type Listed = (Vec<u32>, Vec<u32>, Vec<f64>);

fn join(mut a: Listed, b: Listed) -> Listed {
    a.0.extend(b.0);
    a.1.extend(b.1);
    a.2.extend(b.2);
    a
}

fn starts(counts: &[u32], out: &mut Vec<u32>) {
    out.clear();
    out.push(0);
    let mut total = 0;
    for &c in counts {
        total += c;
        out.push(total);
    }
}

/// Whether every moving end of every edge and corner of every triangle is
/// one of its object's contact points — what lets the points' motion stand
/// for everything's.
pub(super) fn points_cover(points: &[Point], edges: &[Edge], triangles: &[Triangle]) -> bool {
    let key = |attach: &Attach| match *attach {
        Attach::Node { solid, node } => Some([0, solid as u64, node as u64, 0, 0]),
        Attach::Rigid { rigid, local } => Some([
            1,
            rigid as u64,
            local.x.to_bits(),
            local.y.to_bits(),
            local.z.to_bits(),
        ]),
        Attach::Still(_) => None,
    };
    let mut known: Vec<([u64; 5], usize)> = points
        .iter()
        .filter_map(|p| key(&p.attach).map(|k| (k, p.owner)))
        .collect();
    known.sort_unstable();
    let covered = |attach: &Attach, owner: usize| match key(attach) {
        None => true,
        Some(k) => known.binary_search(&(k, owner)).is_ok(),
    };
    edges
        .iter()
        .all(|e| e.ends.iter().all(|a| covered(a, e.owner)))
        && triangles.iter().all(|t| {
            t.nodes.iter().all(|&node| {
                covered(
                    &Attach::Node {
                        solid: t.solid,
                        node,
                    },
                    t.owner,
                )
            })
        })
}

impl Contact {
    /// Lists the pairs near contact at `x`, where [`Contact::positions`]
    /// already are: `x` becomes the reference state.
    pub(super) fn list_near(&mut self, ctx: &Context, x: &[f64]) {
        let precision = self.precision;
        if self.near.width == 0.0 {
            self.near.width = 4.0 * MARGIN.0;
        }
        let margin = self.near.width * precision;
        // Every pair with a gap below ε + margin has boxes that overlap
        // when each is grown by half the margin more than a search in place
        // grows it.
        self.fit_triangles(ctx, x, &[], 0.5 * margin, true);
        self.fit_edges(ctx, x, None, 0.5 * margin, true);
        let this = &*self;
        // Listed unless certainly beyond the margin (so also if not a number).
        let beyond = |s: f64| s >= precision + margin;
        let (counts, triangles, triangle_room) = ctx.pool.map_reduce(
            self.points.len(),
            POINTS_PER_CHUNK,
            Listed::default(),
            |range| {
                let mut out = Listed::default();
                for i in range {
                    let before = out.1.len();
                    this.triangle_candidates(i, 0.5 * margin, |t| {
                        let s = this.gap(ctx, i, Other::Triangle(t), x, this.positions[i]).s;
                        if !beyond(s) {
                            out.1.push(t);
                            out.2.push(s - precision);
                        }
                    });
                    out.0.push((out.1.len() - before) as u32);
                }
                out
            },
            join,
        );
        let (edge_counts, edges, edge_room) = ctx.pool.map_reduce(
            self.edges.len(),
            POINTS_PER_CHUNK,
            Listed::default(),
            |range| {
                let mut out = Listed::default();
                for i in range {
                    let before = out.1.len();
                    this.edge_candidates(i, |j| {
                        let s = this.edge_gap(ctx, i, j, x).s;
                        if !beyond(s) {
                            out.1.push(j as u32);
                            out.2.push(s - precision);
                        }
                    });
                    out.0.push((out.1.len() - before) as u32);
                }
                out
            },
            join,
        );

        // The room between objects: a pair of primitives is at least as far
        // apart as what bounds them, less their radii. A moving object is
        // bounded by its points; a fixed one is taken primitive by primitive
        // (a wide floor's corners and edges are far from what stands on it).
        let n = ctx.world.objects().len();
        let mut object = vec![Kind::Absent; n];
        let mut bounds = vec![Aabb::EMPTY; n];
        let mut radius = vec![0.0f64; n];
        let mut still: Vec<(usize, Aabb, f64)> = Vec::new();
        let mut extent = 0.0f64;
        for (i, point) in self.points.iter().enumerate() {
            let o = point.owner;
            let p = self.positions[i];
            extent = extent.max(p.abs().max_element());
            match point.attach {
                Attach::Still(_) => {
                    still.push((o, Aabb::new(p, p), point.radius));
                    object[o] = object[o].at_least(Kind::Still);
                }
                Attach::Rigid { .. } | Attach::Node { .. } => {
                    bounds[o] = bounds[o].including(p);
                    radius[o] = radius[o].max(point.radius);
                    let selfish = self.self_contact && matches!(point.attach, Attach::Node { .. });
                    object[o] = object[o].at_least(if selfish {
                        Kind::MovingSelf
                    } else {
                        Kind::Moving
                    });
                }
            }
        }
        for edge in &self.edges {
            let o = edge.owner;
            if edge.fixed {
                let ends = edge.ends.map(|a| a.position(ctx.layout, ctx.frame, x));
                extent = extent.max(ends[0].abs().max_element());
                extent = extent.max(ends[1].abs().max_element());
                still.push((o, Aabb::from_points(ends), edge.radius));
                object[o] = object[o].at_least(Kind::Still);
            } else {
                radius[o] = radius[o].max(edge.radius);
            }
        }
        let moving: Vec<usize> = (0..n)
            .filter(|&o| matches!(object[o], Kind::Moving | Kind::MovingSelf))
            .collect();
        let mut isolation = vec![f64::INFINITY; n];
        let between = |p: &Aabb, q: &Aabb| (p.min - q.max).max(q.min - p.max).max_element();
        // An unlisted pair has more room than the margin; two objects' pairs
        // have at least what lies between the objects.
        let mut part = |a: usize, b: usize, room: f64| {
            let room = margin.max(room);
            isolation[a] = isolation[a].min(room);
            isolation[b] = isolation[b].min(room);
        };
        if moving.len() <= MOST_OBJECTS {
            for (k, &a) in moving.iter().enumerate() {
                for &b in &moving[k + 1..] {
                    let room = between(&bounds[a], &bounds[b]) - radius[a] - radius[b];
                    part(a, b, room - precision);
                }
            }
        } else {
            for &a in &moving {
                part(a, a, 0.0);
            }
        }
        let pairwise = moving.len() * still.len() <= MOST_OBJECTS * MOST_OBJECTS;
        for &(f, primitive, r) in &still {
            for &a in &moving {
                let room = if pairwise {
                    between(&bounds[a], &primitive) - radius[a] - r - precision
                } else {
                    0.0
                };
                part(a, f, room);
            }
        }

        // Rounding in any of this is far below the unused part of a margin
        // unless the scene is astronomically far from its origin for its
        // precision; then, and where points do not stand for every vertex,
        // the list serves this state only.
        let sound = self.tracked && (1.0 - USE) * 0.25 * margin >= 1e-9 * extent;
        let near = &mut self.near;
        near.valid = sound;
        near.precision = precision;
        near.self_contact = self.self_contact;
        near.margin = margin;
        near.reference.clear();
        near.reference.extend_from_slice(&self.positions);
        starts(&counts, &mut near.triangle_start);
        near.triangles = triangles;
        near.triangle_room = triangle_room;
        starts(&edge_counts, &mut near.edge_start);
        near.edges = edges;
        near.edge_room = edge_room;
        near.served = 0;
        if near.triangles.len() + near.edges.len() > LONG * (self.points.len() + self.edges.len()) {
            near.width = (0.5 * near.width).max(MARGIN.0);
        }
        near.object = object;
        near.isolation = isolation;
        // Nothing has moved since.
        near.shift.clear();
        near.shift.resize(n, Vec3::ZERO);
        near.spread.clear();
        near.spread.resize(n, 0.0);
        near.drift.clear();
        near.drift.resize(n, 0.0);
    }
}

impl Kind {
    fn at_least(self, other: Kind) -> Kind {
        let rank = |k: Kind| match k {
            Kind::Absent => 0,
            Kind::Still => 1,
            Kind::Moving => 2,
            Kind::MovingSelf => 3,
        };
        if rank(other) > rank(self) {
            other
        } else {
            self
        }
    }
}
