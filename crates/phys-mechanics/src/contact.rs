//! Contact and friction, as barrier and dissipation potentials (in the
//! manner of Incremental Potential Contact, Li et al. 2020).
//!
//! **Geometry.** Contact is checked between *points* — a solid's surface
//! nodes, a rigid body's corners or centre (with a radius) — and either a
//! *shape* (a fixed or rigid body's signed distance) or another solid's
//! *surface triangles*. Edges of boxes, capsule centrelines and unique mesh
//! boundary edges are also checked against one another, including fixed
//! bodies. A deformable solid also checks its own non-incident surface
//! primitives; shared vertices are topological neighbours, not contacts.
//! Self-contact uses the true positive distance as its barrier gap, so
//! initially disjoint parts cannot cross through the surface tolerance.
//!
//! **Barrier.** For an inter-object pair at distance `d`, the gap is
//! `s = d + ε/2` (self-contact instead uses `s = d`), and
//! the energy `κ (s − ε)² ln(ε/s)` acts for `s < ε`, rising without bound as
//! `s → 0`. Surfaces so meet within `±ε/2` of where they truly are — `ε` is
//! the contact precision — and a scene may start with things exactly
//! touching. `κ = m / h²` (a mass at the contact over the step squared)
//! makes the barrier as stiff as the inertia it stops, whatever the masses
//! and the step; it is one physical spring, the same in every stage of a
//! step.
//! A gap below a millionth of ε counts as touching (infinite energy): a
//! solve may never end that close, where rounding alone could flip its
//! sign.
//!
//! **No tunnelling.** Every step is limited so that no gap can close: the
//! gap of each pair falls at most by how far its two sides can move.
//!
//! **Impacts.** Contact is non-smooth: an impact faster than the step is
//! over within it. The barrier then acts as a perfectly inelastic stop, and
//! restitution is Newton's impact law applied to velocities afterwards
//! ([`Contact::impacts`]), in the manner of Moreau's time stepping.
//!
//! **Friction.** Smoothed Coulomb friction (`μ λ f₀(|u|)` in the tangential
//! slip `u` over the step), with the normal force `λ` and the tangent plane
//! taken from the contacts at the start of the step (lagged), and updated
//! when the solver asks.

use core_math::{Mat3, Vec3};
use core_numerics::{BlockMatrix, Triplet, Triplets};
use core_space::{closest_on_triangle, closest_segments, Aabb, Bvh, Shape};
use core_world::{Link, Representation};

use crate::dofs::{Along, Attach, Owner, Unknowns};
use crate::rigid::{contact_edges, contact_points, CORNER_ROUNDING};
use crate::term::{Context, Term};

mod near;
use near::Near;

#[derive(Clone, Copy, Debug)]
struct Point {
    owner: usize,
    attach: Attach,
    radius: f64,
    mass: f64,
}

#[derive(Clone, Copy, Debug)]
struct Triangle {
    owner: usize,
    solid: u32,
    nodes: [u32; 3],
    mass: f64,
}

#[derive(Clone, Copy, Debug)]
struct Edge {
    owner: usize,
    ends: [Attach; 2],
    radius: f64,
    mass: f64,
    fixed: bool,
}

/// Stable primitive identities, independent of changing closest-point weights.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Pair {
    PointBody(u32, usize),
    PointTriangle(u32, u32),
    Edges(u32, u32),
}

/// A body with a signed distance.
#[derive(Clone, Copy, Debug)]
enum Body {
    Fixed(usize),
    Rigid(usize),
}

/// The other side of a pair.
#[derive(Clone, Copy, Debug)]
enum Other {
    Body(Body),
    Triangle(u32),
}

/// A pair, evaluated: its gap, normal (pointing from the other side to the
/// point), the stiffness mass, and the material points involved with their
/// weights in the gap's gradient (`∂s/∂pₖ = wₖ distance_gradient`).
#[derive(Clone, Copy, Debug)]
struct Gap {
    s: f64,
    /// Unit physical normal for friction/impact velocity resolution.
    normal: Vec3,
    /// Exact world-space gap gradient. Affine SDF transforms need A⁻ᵀ n,
    /// whose length is generally not one before the rigid projection.
    distance_gradient: Vec3,
    mass: f64,
    sides: [(Attach, f64); 4],
    count: usize,
    friction: f64,
    restitution: f64,
    owner: usize,
    pair: Pair,
}

/// Parallel-edge smoothing for every edge-edge pair. The maps differentiate
/// w=(edge_a cross edge_b)/sqrt(epsilon_cross), with a rest-length scale.
struct EdgeSmoothing {
    value: f64,
    w: Vec3,
    maps: [Mat3; 4],
    slope: f64,
    ends: [Attach; 4],
}

/// A contact's friction, frozen for the step.
#[derive(Clone, Copy, Debug)]
struct Grip {
    pair: Pair,
    sides: [(Attach, f64); 4],
    starts: [Vec3; 4],
    count: usize,
    normal: Vec3,
    /// `μ λ`, N.
    limit: f64,
    /// Newly formed at a friction refresh, starting from unbraked slip.
    unbraked: bool,
}

pub(crate) struct FrictionState(Vec<Grip>);

/// A pair in contact, for the impact law: the material points on its two
/// sides with their weights (the relative velocity is `Σ wₖ vₖ`), the
/// normal, and the coefficient of restitution.
#[derive(Clone, Copy, Debug)]
pub struct Impact {
    pub sides: [(Attach, f64); 4],
    pub count: usize,
    pub normal: Vec3,
    pub restitution: f64,
    /// Solver-only barrier potential stored in this pair at the evaluated
    /// state, J. It is not physical strain energy for rigid bodies.
    pub(crate) potential: f64,
}

/// Up to eight `(block, G)`: how a weighted sum of material points moves
/// with each block (two rigid bodies' four blocks each at most).
type Combined = ([(u32, Mat3); 8], usize);

#[derive(Clone, Debug, Default)]
pub struct Contact {
    /// Include non-incident primitive pairs of the same deformable solid.
    pub self_contact: bool,
    /// ε, m.
    pub precision: f64,
    /// Slip speed below which friction acts as a stiff spring rather than
    /// at its full Coulomb value, m/s.
    pub stick_speed: f64,
    points: Vec<Point>,
    triangles: Vec<Triangle>,
    edges: Vec<Edge>,
    bodies: Vec<Body>,
    body_cache: Bodies,
    grips: Vec<Grip>,
    /// The pairs in contact where [`Contact::active`] last looked.
    gaps: Vec<Gap>,
    /// Where that was, while a stage's solve is under way.
    seen: Seen,
    /// The pairs that can be in contact near there.
    near: Near,
    /// Whether the contact points' motion stands for every vertex's.
    tracked: bool,
    /// Scratch: the derivatives' shares, a chunk of pairs or grips each.
    shares: Vec<Share>,
    /// The surface triangles' hierarchy: rebuilt when contacts are
    /// refreshed, refit to the triangles' current boxes otherwise.
    bvh: Bvh,
    edge_bvh: Bvh,
    // Scratch.
    positions: Vec<Vec3>,
    edge_positions: Vec<[Vec3; 2]>,
    edge_boxes: Vec<Aabb>,
    edge_owners: Vec<u32>,
    triangle_owners: Vec<u32>,
    triangle_boxes: Vec<Aabb>,
    excluded_pairs: Vec<(usize, usize)>,
}

/// The unknowns [`Contact::gaps`] were found at. A solve asks for the
/// contacts at one state several times over (the energy of an accepted
/// trial, then its derivatives, then the impact check and friction of
/// what it converged to); the pairs depend on the unknowns alone while a
/// stage's constants stand, so they are found once.
#[derive(Clone, Debug, Default)]
struct Seen {
    /// Between a stage's [`Term::begin_step`] and the next
    /// [`Contact::forget`]: outside a stage nothing is remembered.
    in_stage: bool,
    valid: bool,
    x: Vec<f64>,
    precision: f64,
    self_contact: bool,
}

/// A gap below this fraction of ε counts as touching.
const MIN_GAP: f64 = 1e-6;

/// Points per parallel chunk of contact work.
const POINTS_PER_CHUNK: usize = 64;

/// A body's place for candidate search: the body, its object, its bounds
/// grown by how far anything can reach.
type Placed = (Body, usize, Aabb);

/// Every body placed for one search: the list, a hierarchy over the
/// bounded ones, and the unbounded ones (half-spaces) apart.
#[derive(Clone, Debug, Default)]
struct Bodies {
    placed: Vec<Placed>,
    bvh: Bvh,
    /// Indices into `placed` of bodies with infinite bounds.
    unbounded: Vec<usize>,
    /// Indices into `placed` of the bounded bodies, as the hierarchy numbers them.
    bounded: Vec<usize>,
    boxes: Vec<Aabb>,
}

impl Contact {
    pub(crate) fn boundary_contact(&mut self, ctx: &Context, x: &[f64]) -> bool {
        self.impacts(ctx, x).iter().any(|i| {
            i.sides[..i.count].iter().any(|&(a, _)| match a {
                Attach::Still(_) => true,
                Attach::Node { solid, node } => {
                    ctx.layout.solids[solid as usize].block[node as usize] == crate::dofs::HELD
                }
                Attach::Rigid { .. } => false,
            })
        })
    }

    pub(crate) fn friction_state(&self) -> FrictionState {
        FrictionState(self.grips.clone())
    }
    pub(crate) fn restore_friction(&mut self, state: FrictionState) {
        self.grips = state.0;
    }

    /// Endpoint force work on the same affine/nodal paths as the potential.
    pub(crate) fn discrete_work(&mut self, ctx: &Context, x: &[f64]) -> (f64, f64) {
        self.active(ctx, x, false);
        let scale = Self::scale(ctx) / ctx.frame.weight;
        let mut normal_work = 0.0;
        for gap in &self.gaps {
            let (energy, db, _) = self.barrier(gap.s);
            let mut displacement = Vec3::ZERO;
            for &(a, w) in &gap.sides[..gap.count] {
                displacement +=
                    (a.position(ctx.layout, ctx.frame, x) - a.start(ctx.layout, ctx.frame)) * w;
            }
            let smoothing = self.edge_smoothing(ctx, gap, x);
            normal_work -= db
                * scale
                * gap.mass
                * smoothing.as_ref().map_or(1.0, |m| m.value)
                * gap.distance_gradient.dot(displacement);
            if let Some(m) = smoothing {
                let mut dw = Vec3::ZERO;
                for k in 0..4 {
                    let a = m.ends[k];
                    dw += m.maps[k]
                        * (a.position(ctx.layout, ctx.frame, x) - a.start(ctx.layout, ctx.frame));
                }
                normal_work -= energy * scale * gap.mass * m.slope * m.w.dot(dw);
            }
        }
        let e = self.stick_speed * ctx.frame.h;
        let friction_work = -self
            .grips
            .iter()
            .map(|g| {
                let d = self.displacement(ctx, g, x);
                let y = (d - g.normal * g.normal.dot(d)).length();
                g.limit * friction_profile(y, e).1 * y
            })
            .sum::<f64>();
        (normal_work, friction_work)
    }

    /// Contact between everything in the world, at precision `precision`.
    pub fn new(ctx: &Context, precision: f64) -> Self {
        let mut c = Self {
            self_contact: true,
            precision,
            stick_speed: 10.0 * precision,
            ..Default::default()
        };
        let world = ctx.world;
        for (slot, owner) in ctx.layout.objects.iter().enumerate() {
            let index = ctx.layout.object_index(slot);
            match *owner {
                Owner::Fixed(f) => {
                    c.bodies.push(Body::Fixed(f));
                    let body = &ctx.layout.fixed[f];
                    c.points
                        .extend(
                            contact_points(&body.shape)
                                .into_iter()
                                .map(|(p, radius)| Point {
                                    owner: index,
                                    attach: Attach::Still(body.pose.transform_point(p)),
                                    radius,
                                    mass: f64::INFINITY,
                                }),
                        );
                    c.edges.extend(
                        contact_edges(&body.shape)
                            .into_iter()
                            .map(|(ends, radius)| Edge {
                                owner: index,
                                ends: ends.map(|p| Attach::Still(body.pose.transform_point(p))),
                                radius,
                                mass: f64::INFINITY,
                                fixed: true,
                            }),
                    );
                }
                Owner::Rigid(r) => {
                    c.bodies.push(Body::Rigid(r));
                    let Representation::Rigid(rigid) = &world.objects()[index].representation
                    else {
                        unreachable!()
                    };
                    let points = contact_points(&rigid.shape);
                    let mass = rigid.mass / points.len() as f64;
                    c.edges.extend(contact_edges(&rigid.shape).into_iter().map(
                        |(ends, radius)| Edge {
                            owner: index,
                            ends: ends.map(|local| Attach::Rigid {
                                rigid: r as u32,
                                local,
                            }),
                            radius,
                            mass,
                            fixed: false,
                        },
                    ));
                    c.points
                        .extend(points.into_iter().map(|(local, radius)| Point {
                            owner: index,
                            attach: Attach::Rigid {
                                rigid: r as u32,
                                local,
                            },
                            radius,
                            mass,
                        }));
                }
                Owner::Solid(s) => {
                    let Representation::Solid(solid) = &world.objects()[index].representation
                    else {
                        unreachable!()
                    };
                    let mass = &solid.nodes.mass;
                    c.points
                        .extend(solid.surface.nodes.iter().map(|&node| Point {
                            owner: index,
                            attach: Attach::Node {
                                solid: s as u32,
                                node,
                            },
                            radius: 0.0,
                            mass: mass[node as usize],
                        }));
                    c.triangles
                        .extend(solid.surface.faces.iter().map(|&nodes| Triangle {
                            owner: index,
                            solid: s as u32,
                            nodes,
                            mass: nodes.iter().map(|&n| mass[n as usize]).sum::<f64>() / 3.0,
                        }));
                    // Each surface edge once, in node order, independent of
                    // face orientation and hash iteration order.
                    let mut edges: Vec<[u32; 2]> = solid
                        .surface
                        .faces
                        .iter()
                        .flat_map(|f| {
                            [[f[0], f[1]], [f[1], f[2]], [f[2], f[0]]].map(|mut e| {
                                e.sort_unstable();
                                e
                            })
                        })
                        .collect();
                    edges.sort_unstable();
                    edges.dedup();
                    c.edges.extend(edges.into_iter().map(|nodes| Edge {
                        owner: index,
                        ends: nodes.map(|node| Attach::Node {
                            solid: s as u32,
                            node,
                        }),
                        radius: 0.0,
                        mass: 0.5 * (mass[nodes[0] as usize] + mass[nodes[1] as usize]),
                        fixed: false,
                    }));
                }
            }
        }
        for link in world.links() {
            let pair = match *link {
                Link::BallJoint { a, b, .. } => (a.object(), b.object()),
                Link::Weld { a, b, .. }
                | Link::Hinge { a, b, .. }
                | Link::Slider { a, b, .. }
                | Link::Cylindrical { a, b, .. }
                | Link::Universal { a, b, .. }
                | Link::Planar { a, b, .. } => (Some(a), b),
                Link::Anchor { .. } | Link::Spring { .. } | Link::Tendon { .. } => (None, None),
            };
            if let (Some(oa), Some(ob)) = pair {
                let (ia, ib) = (oa.0 as usize, ob.0 as usize);
                if ia != ib {
                    c.excluded_pairs.push((ia.min(ib), ia.max(ib)));
                }
            }
        }
        for &(oa, ob) in world.excluded_contacts() {
            let (ia, ib) = (oa.0 as usize, ob.0 as usize);
            if ia != ib {
                c.excluded_pairs.push((ia.min(ib), ia.max(ib)));
            }
        }
        c.excluded_pairs.sort_unstable();
        c.excluded_pairs.dedup();
        c.edge_owners = c.edges.iter().map(|e| e.owner as u32).collect();
        c.triangle_owners = c.triangles.iter().map(|t| t.owner as u32).collect();
        c.tracked = near::points_cover(&c.points, &c.edges, &c.triangles);
        c
    }

    #[inline]
    fn is_excluded(&self, a: usize, b: usize) -> bool {
        if a == b || self.excluded_pairs.is_empty() {
            return false;
        }
        let key = (a.min(b), a.max(b));
        self.excluded_pairs.binary_search(&key).is_ok()
    }

    /// A thousandth of the smallest moving object's diagonal, capped at
    /// twice its thickness. Preserve the established scale for compact
    /// objects without letting a long bar's length set its contact error.
    pub fn default_precision(ctx: &Context) -> f64 {
        let mut smallest = f64::INFINITY;
        for slot in 0..ctx.layout.objects.len() {
            let object = &ctx.world.objects()[ctx.layout.object_index(slot)];
            let size = match &object.representation {
                Representation::Fixed(_) => continue,
                Representation::Rigid(r) => r.shape.local_bounds().size(),
                Representation::Solid(s) => Aabb::from_points(s.mesh.rest.iter().copied()).size(),
            };
            smallest = smallest.min(size.length().min(2.0 * size.min_element()));
        }
        if smallest.is_finite() {
            1e-3 * smallest
        } else {
            1e-3
        }
    }

    fn body_owner(&self, ctx: &Context, body: Body) -> usize {
        match body {
            Body::Fixed(f) => ctx.layout.fixed[f].object,
            Body::Rigid(r) => ctx.layout.rigids[r].object,
        }
    }

    /// A body's shape, centre and matrix (a rotation, or for a rigid body
    /// in the solve, the nearly-orthogonal `A`).
    fn body_shape(&self, ctx: &Context, body: Body, x: impl Unknowns) -> (Shape, Vec3, Mat3) {
        match body {
            Body::Fixed(f) => {
                let b = &ctx.layout.fixed[f];
                (b.shape, b.pose.position, b.pose.rotation.to_mat3())
            }
            Body::Rigid(r) => {
                let (c, a) = ctx.frame.rigid(ctx.layout, r, x);
                (ctx.frame.rigids[r].shape, c, a)
            }
        }
    }

    fn friction(&self, ctx: &Context, a: usize, b: usize) -> f64 {
        let world = ctx.world;
        let (ma, mb) = (world.objects()[a].material, world.objects()[b].material);
        world.material(ma).friction_with(world.material(mb))
    }

    fn restitution(&self, ctx: &Context, a: usize, b: usize) -> f64 {
        let world = ctx.world;
        let (ma, mb) = (world.objects()[a].material, world.objects()[b].material);
        world.material(ma).restitution_with(world.material(mb))
    }

    fn update_positions(&mut self, ctx: &Context, x: &[f64]) {
        self.positions.resize(self.points.len(), Vec3::ZERO);
        let points = &self.points;
        ctx.pool
            .for_chunks_mut(&mut self.positions, 256, |offset, part| {
                for (k, p) in part.iter_mut().enumerate() {
                    *p = points[offset + k].attach.position(ctx.layout, ctx.frame, x);
                }
            });
    }

    fn triangle_corners(&self, ctx: &Context, t: &Triangle, x: impl Unknowns) -> [Vec3; 3] {
        t.nodes
            .map(|n| ctx.frame.node(ctx.layout, t.solid as usize, n as usize, x))
    }

    /// IPC's polynomial mollifier removes the non-differentiable closest
    /// feature switch of nearly parallel edges, on every edge-edge pair:
    /// mesh edges, rigid bodies' edges and fixed edges alike. Where two edges
    /// are parallel the pair fades out, and the point barriers at their ends
    /// (vertex/triangle, or a corner against the other body's signed distance)
    /// carry the contact. All swept edge limits remain present.
    ///
    /// Without it on rigid edges, two boxes resting corner on corner put an
    /// exact closest-feature switch where Newton converges: the gradient
    /// alternated between two values and the line search stalled, which is
    /// what stopped stacked and piled boxes.
    fn edge_smoothing(&self, ctx: &Context, gap: &Gap, x: &[f64]) -> Option<EdgeSmoothing> {
        let Pair::Edges(i, j) = gap.pair else {
            return None;
        };
        let (a, b) = (&self.edges[i as usize], &self.edges[j as usize]);
        let ends = [a.ends[0], a.ends[1], b.ends[0], b.ends[1]];
        // Where each end sits in its own undeformed frame: lengths there are
        // the edges' rest lengths.
        let rest = ends.map(|attach| match attach {
            Attach::Node { solid, node } => {
                let Representation::Solid(s) =
                    &ctx.world.objects()[ctx.layout.solids[solid as usize].object].representation
                else {
                    unreachable!()
                };
                s.mesh.rest[node as usize]
            }
            Attach::Rigid { local, .. } => local,
            Attach::Still(position) => position,
        });
        let scale2 =
            1e-3 * (rest[1] - rest[0]).length_squared() * (rest[3] - rest[2]).length_squared();
        if scale2 <= 0.0 {
            return None;
        }
        let p = ends.map(|a| a.position(ctx.layout, ctx.frame, x));
        let (u, v) = (p[1] - p[0], p[3] - p[2]);
        let cross = u.cross(v);
        if cross.length_squared() >= scale2 {
            return None;
        }
        let inverse_scale = 1.0 / scale2.sqrt();
        let w = cross * inverse_scale;
        let z = w.length_squared();
        let du = Mat3::skew(v) * (-inverse_scale);
        let dv = Mat3::skew(u) * inverse_scale;
        Some(EdgeSmoothing {
            value: z * (2.0 - z),
            w,
            maps: [du * -1.0, du, dv * -1.0, dv],
            slope: 4.0 * (1.0 - z),
            ends,
        })
    }

    fn edge_gap(&self, ctx: &Context, i: usize, j: usize, x: impl Unknowns) -> Gap {
        let (a, b) = (&self.edges[i], &self.edges[j]);
        let [a0, a1] = a.ends.map(|p| p.position(ctx.layout, ctx.frame, x));
        let [b0, b1] = b.ends.map(|p| p.position(ctx.layout, ctx.frame, x));
        let (p, q, [s, t]) = closest_segments(a0, a1, b0, b1);
        let d = (p - q).length();
        let normal = if d > 0.0 {
            (p - q) / d
        } else {
            (a1 - a0).cross(b1 - b0).normalize_or_zero()
        };
        Gap {
            s: d - a.radius - b.radius
                + if a.owner == b.owner {
                    0.0
                } else {
                    0.5 * self.precision
                },
            normal,
            distance_gradient: normal,
            mass: a.mass.min(b.mass),
            sides: [
                (a.ends[0], 1.0 - s),
                (a.ends[1], s),
                (b.ends[0], t - 1.0),
                (b.ends[1], -t),
            ],
            count: 4,
            friction: self.friction(ctx, a.owner, b.owner),
            restitution: self.restitution(ctx, a.owner, b.owner),
            owner: if a.fixed { b.owner } else { a.owner },
            pair: Pair::Edges(i as u32, j as u32),
        }
    }

    /// Exact endpoint sweep bounds: endpoint positions are affine in the
    /// Newton unknowns, for rigid bodies as well as finite-element nodes.
    fn fit_edges(&mut self, ctx: &Context, x: &[f64], dx: Option<&[f64]>, pad: f64, rebuild: bool) {
        self.edge_positions
            .resize(self.edges.len(), [Vec3::ZERO; 2]);
        self.edge_boxes.resize(self.edges.len(), Aabb::EMPTY);
        let edges = &self.edges;
        ctx.pool
            .for_chunks_mut(&mut self.edge_positions, 256, |offset, part| {
                for (k, p) in part.iter_mut().enumerate() {
                    *p = edges[offset + k]
                        .ends
                        .map(|a| a.position(ctx.layout, ctx.frame, x));
                }
            });
        let positions = &self.edge_positions;
        let precision = self.precision;
        ctx.pool
            .for_chunks_mut(&mut self.edge_boxes, 256, |offset, part| {
                for (k, bounds) in part.iter_mut().enumerate() {
                    let i = offset + k;
                    *bounds = Aabb::from_points(positions[i]);
                    if let Some(dx) = dx {
                        for a in edges[i].ends {
                            *bounds = bounds.including(a.position(
                                ctx.layout,
                                ctx.frame,
                                Along { x, dx, t: 1.0 },
                            ));
                        }
                    }
                    *bounds = bounds.grown(edges[i].radius + precision + pad);
                }
            });
        if rebuild || self.edge_bvh.len() != self.edges.len() {
            self.edge_bvh = Bvh::build_grouped(&self.edge_boxes, &self.edge_owners);
        } else {
            self.edge_bvh.refit(&self.edge_boxes);
        }
    }

    /// Edge `i`'s box at `x` as [`Contact::fit_edges`] fits it for a search
    /// in place: around its ends, grown by its radius and ε.
    fn edge_bounds(&self, ctx: &Context, i: usize, x: &[f64]) -> Aabb {
        let edge = &self.edges[i];
        Aabb::from_points(edge.ends.map(|a| a.position(ctx.layout, ctx.frame, x)))
            .grown(edge.radius + self.precision)
    }

    /// Edge `i`'s ends at `x`, and its box over the step `dx` as
    /// [`Contact::fit_edges`] fits it for a swept search: around its ends at
    /// both ends of the step, grown by its radius and ε.
    fn edge_sweep(&self, ctx: &Context, i: usize, x: &[f64], dx: &[f64]) -> ([Vec3; 2], Aabb) {
        let edge = &self.edges[i];
        let ends = edge.ends.map(|a| a.position(ctx.layout, ctx.frame, x));
        let mut bounds = Aabb::from_points(ends);
        for a in edge.ends {
            bounds = bounds.including(a.position(ctx.layout, ctx.frame, Along { x, dx, t: 1.0 }));
        }
        (ends, bounds.grown(edge.radius + self.precision))
    }

    fn edge_candidates(&self, i: usize, mut visit: impl FnMut(usize)) {
        let a = &self.edges[i];
        let mut found = Vec::new();
        let include_self = self.self_contact && matches!(a.ends[0], Attach::Node { .. });
        let mut candidate = |j: usize| {
            let b = &self.edges[j];
            if j > i
                && !(a.fixed && b.fixed)
                && !self.is_excluded(a.owner, b.owner)
                && (a.owner != b.owner
                    || (include_self && !a.ends.iter().any(|p| b.ends.contains(p))))
            {
                found.push(j);
            }
        };
        if include_self {
            self.edge_bvh.query(&self.edge_boxes[i], &mut candidate);
        } else {
            self.edge_bvh
                .query_excluding(&self.edge_boxes[i], a.owner as u32, &mut candidate);
        }
        found.sort_unstable();
        for j in found {
            visit(j);
        }
    }

    /// The gap of point `i` against `other`, at `x` (where the point is at
    /// `p`).
    fn gap(&self, ctx: &Context, i: usize, other: Other, x: impl Unknowns, p: Vec3) -> Gap {
        let point = &self.points[i];
        let half = 0.5 * self.precision;
        match other {
            Other::Body(body) => {
                let (shape, centre, a) = self.body_shape(ctx, body, x);
                let inverse = a.inverse();
                let local = inverse * (p - centre);
                let (d, n) = contact_distance(&shape, local);
                let distance_gradient = inverse.transpose() * n;
                let normal = distance_gradient.normalize();
                let owner = self.body_owner(ctx, body);
                let (other_side, mass) = match body {
                    Body::Fixed(_) => ((Attach::Still(p), 0.0), point.mass),
                    Body::Rigid(r) => {
                        let m = ctx.frame.rigids[r].mass
                            / ctx.frame.rigids[r].points.len().max(1) as f64;
                        (
                            (
                                Attach::Rigid {
                                    rigid: r as u32,
                                    local,
                                },
                                -1.0,
                            ),
                            point.mass.min(m),
                        )
                    }
                };
                Gap {
                    s: d - point.radius + half,
                    normal,
                    distance_gradient,
                    mass,
                    sides: [
                        (point.attach, 1.0),
                        other_side,
                        (Attach::Still(p), 0.0),
                        (Attach::Still(p), 0.0),
                    ],
                    count: 2,
                    friction: self.friction(ctx, point.owner, owner),
                    restitution: self.restitution(ctx, point.owner, owner),
                    owner: if matches!(point.attach, Attach::Still(_)) {
                        owner
                    } else {
                        point.owner
                    },
                    pair: Pair::PointBody(i as u32, owner),
                }
            }
            Other::Triangle(t) => {
                let tri = &self.triangles[t as usize];
                let [a, b, c] = self.triangle_corners(ctx, tri, x);
                let (q, w) = closest_on_triangle(p, a, b, c);
                let d = (p - q).length();
                let normal = if d > 0.0 {
                    (p - q) / d
                } else {
                    (b - a).cross(c - a).normalize_or_zero()
                };
                let node = |k: usize| {
                    (
                        Attach::Node {
                            solid: tri.solid,
                            node: tri.nodes[k],
                        },
                        -w[k],
                    )
                };
                Gap {
                    s: d - point.radius + if point.owner == tri.owner { 0.0 } else { half },
                    normal,
                    distance_gradient: normal,
                    mass: point.mass.min(tri.mass),
                    sides: [(point.attach, 1.0), node(0), node(1), node(2)],
                    count: 4,
                    friction: self.friction(ctx, point.owner, tri.owner),
                    restitution: self.restitution(ctx, point.owner, tri.owner),
                    owner: point.owner,
                    pair: Pair::PointTriangle(i as u32, t),
                }
            }
        }
    }

    /// Physical material points for tangential slip and impact velocities.
    /// Barrier derivatives use the original geometric gap: a rounded point's
    /// centre/edge is convenient for distance, but friction must act on its
    /// surface to produce the correct torque (notably for spheres/capsules).
    fn surface_sides(&self, ctx: &Context, gap: &Gap, x: &[f64]) -> [(Attach, f64); 4] {
        let mut sides = gap.sides;
        let shift = |attach: Attach, offset: Vec3| {
            if offset == Vec3::ZERO {
                return attach;
            }
            match attach {
                Attach::Rigid { rigid, local } => {
                    let (_, a) = ctx.frame.rigid(ctx.layout, rigid as usize, x);
                    Attach::Rigid {
                        rigid,
                        local: local + a.inverse() * offset,
                    }
                }
                Attach::Still(p) => Attach::Still(p + offset),
                Attach::Node { .. } => attach, // mesh vertices have zero radius
            }
        };
        match gap.pair {
            Pair::PointBody(i, _) => {
                let point = self.points[i as usize];
                sides[0].0 = shift(sides[0].0, gap.normal * -point.radius);
                match sides[1].0 {
                    Attach::Rigid { rigid, local } => {
                        let (d, n) =
                            contact_distance(&ctx.frame.rigids[rigid as usize].shape, local);
                        sides[1].0 = Attach::Rigid {
                            rigid,
                            local: local - n * d,
                        };
                    }
                    Attach::Still(p) => {
                        let distance = gap.s + point.radius - 0.5 * self.precision;
                        sides[1].0 = Attach::Still(p - gap.normal * distance);
                    }
                    Attach::Node { .. } => unreachable!(),
                }
            }
            Pair::PointTriangle(i, _) => {
                sides[0].0 = shift(sides[0].0, gap.normal * -self.points[i as usize].radius);
            }
            Pair::Edges(i, j) => {
                for side in &mut sides[..2] {
                    side.0 = shift(side.0, gap.normal * -self.edges[i as usize].radius);
                }
                for side in &mut sides[2..] {
                    side.0 = shift(side.0, gap.normal * self.edges[j as usize].radius);
                }
            }
        }
        sides
    }

    /// Refit cached analytic-body bounds; no polar decomposition is needed.
    fn fit_bodies(&mut self, ctx: &Context, x: &[f64], reach: &dyn Fn(Body) -> f64, rebuild: bool) {
        let changed = self.body_cache.placed.len() != self.bodies.len();
        if changed {
            self.body_cache.placed.clear();
            self.body_cache.bounded.clear();
            self.body_cache.unbounded.clear();
        }
        for i in 0..self.bodies.len() {
            let body = self.bodies[i];
            let (shape, centre, a) = self.body_shape(ctx, body, x);
            let bounds = shape
                .affine_bounds(centre, a)
                .grown(2.0 * self.precision + reach(body));
            let placed = (body, self.body_owner(ctx, body), bounds);
            if changed {
                self.body_cache.placed.push(placed);
                if shape.is_bounded() {
                    self.body_cache.bounded.push(i);
                } else {
                    self.body_cache.unbounded.push(i);
                }
            } else {
                self.body_cache.placed[i] = placed;
            }
        }
        self.body_cache.boxes.clear();
        self.body_cache.boxes.extend(
            self.body_cache
                .bounded
                .iter()
                .map(|&k| self.body_cache.placed[k].2),
        );
        if rebuild || changed {
            self.body_cache.bvh = Bvh::build(&self.body_cache.boxes);
        } else {
            self.body_cache.bvh.refit(&self.body_cache.boxes);
        }
    }

    /// Fits the triangle hierarchy to the triangles at `x`, each grown by ε
    /// and by how far its corners may move (`pad` where `reach` says
    /// nothing) — rebuilt if `rebuild`, refit otherwise.
    fn fit_triangles(&mut self, ctx: &Context, x: &[f64], reach: &[f64], pad: f64, rebuild: bool) {
        if self.triangles.is_empty() {
            return;
        }
        self.triangle_boxes
            .resize(self.triangles.len(), Aabb::EMPTY);
        let triangles = &self.triangles;
        let precision = self.precision;
        ctx.pool
            .for_chunks_mut(&mut self.triangle_boxes, 256, |offset, part| {
                for (k, b) in part.iter_mut().enumerate() {
                    let t = offset + k;
                    let tri = &triangles[t];
                    *b = Aabb::from_points(tri.nodes.map(|n| {
                        ctx.frame
                            .node(ctx.layout, tri.solid as usize, n as usize, x)
                    }))
                    .grown(precision + reach.get(t).copied().unwrap_or(pad));
                }
            });
        if rebuild || self.bvh.len() != self.triangle_boxes.len() {
            self.bvh = Bvh::build_grouped(&self.triangle_boxes, &self.triangle_owners);
        } else {
            self.bvh.refit(&self.triangle_boxes);
        }
    }

    /// Calls `visit` with every candidate pair of point `i`, which may move
    /// by `reach`, in a fixed order: bodies first, then triangles.
    fn point_candidates(
        &self,
        bodies: &Bodies,
        i: usize,
        reach: f64,
        scratch: &mut Vec<usize>,
        mut visit: impl FnMut(Other),
    ) {
        self.body_candidates(bodies, i, reach, scratch, |body| visit(Other::Body(body)));
        self.triangle_candidates(i, reach, |t| visit(Other::Triangle(t)));
    }

    /// The bodies whose bounds point `i` is within `reach` of, in body
    /// order. `scratch` is the caller's, to be used again point after point.
    fn body_candidates(
        &self,
        bodies: &Bodies,
        i: usize,
        reach: f64,
        near: &mut Vec<usize>,
        mut visit: impl FnMut(Body),
    ) {
        let point = &self.points[i];
        let p = self.positions[i];
        let query = Aabb::around(p, Vec3::splat(point.radius + reach));
        near.clear();
        near.extend_from_slice(&bodies.unbounded);
        bodies.bvh.query(&query, |k| near.push(bodies.bounded[k]));
        // In body order, whatever the hierarchy's shape.
        near.sort_unstable();
        for &k in near.iter() {
            let (body, owner, bounds) = bodies.placed[k];
            if point.owner != owner
                && !self.is_excluded(point.owner, owner)
                && !(matches!(point.attach, Attach::Still(_)) && matches!(body, Body::Fixed(_)))
                && bounds.grown(point.radius + reach).contains(p)
            {
                visit(body);
            }
        }
    }

    /// Whether point `i` and triangle `t` are a pair at all: not the same
    /// object's unless it collides with itself, and never a triangle the
    /// point is a corner of.
    fn point_meets_triangle(&self, i: usize, t: usize) -> bool {
        let point = &self.points[i];
        let tri = &self.triangles[t];
        let include_self = self.self_contact && matches!(point.attach, Attach::Node { .. });
        !self.is_excluded(point.owner, tri.owner)
            && (tri.owner != point.owner || include_self)
            && !matches!(point.attach, Attach::Node { solid, node } if solid == tri.solid && tri.nodes.contains(&node))
    }

    /// The triangles whose fitted boxes are within `reach` of point `i`, in
    /// triangle order.
    fn triangle_candidates(&self, i: usize, reach: f64, mut visit: impl FnMut(u32)) {
        if self.triangles.is_empty() {
            return;
        }
        let point = &self.points[i];
        let p = self.positions[i];
        let query = Aabb::around(p, Vec3::splat(point.radius + reach));
        let mut found = Vec::new();
        let include_self = self.self_contact && matches!(point.attach, Attach::Node { .. });
        let mut candidate = |t: usize| {
            if self.point_meets_triangle(i, t) {
                found.push(t as u32);
            }
        };
        if include_self {
            self.bvh.query(&query, &mut candidate);
        } else {
            self.bvh
                .query_excluding(&query, point.owner as u32, &mut candidate);
        }
        // The hierarchy's traversal order depends on its shape; sort, so the
        // pairs come in an order that does not.
        found.sort_unstable();
        for t in found {
            visit(t);
        }
    }

    /// The stage's constants are about to change (or have): what was found
    /// under the old ones is no longer the answer.
    pub(crate) fn forget(&mut self) {
        self.seen.in_stage = false;
        self.seen.valid = false;
    }

    /// Every pair in contact (gap below ε) at `x`, in point order, into
    /// [`Contact::gaps`]. Asked again at the very same unknowns within a
    /// stage, it keeps what it found.
    fn active(&mut self, ctx: &Context, x: &[f64], rebuild: bool) {
        let seen = &self.seen;
        if seen.valid
            && seen.precision.to_bits() == self.precision.to_bits()
            && seen.self_contact == self.self_contact
            && seen.x.len() == x.len()
            && seen
                .x
                .iter()
                .zip(x)
                .all(|(a, b)| a.to_bits() == b.to_bits())
        {
            return;
        }
        self.seen.valid = false;
        self.update_positions(ctx, x);
        self.fit_bodies(ctx, x, &|_| 0.0, rebuild);
        // The pairs listed near the reference state are every pair that can
        // be in contact while nothing has moved too far from it; once
        // something has, list again from here.
        if !(self.seen.in_stage
            && self.near.holds(
                &self.points,
                &self.positions,
                self.precision,
                self.self_contact,
            ))
        {
            self.list_near(ctx, x);
        }
        let this = &*self;
        let near = &this.near;
        let mut gaps = ctx.pool.map_reduce(
            self.points.len(),
            POINTS_PER_CHUNK,
            Vec::new(),
            |range| {
                let mut out = Vec::new();
                let mut scratch = Vec::new();
                for i in range {
                    let p = this.positions[i];
                    this.body_candidates(&this.body_cache, i, 0.0, &mut scratch, |body| {
                        let gap = this.gap(ctx, i, Other::Body(body), x, p);
                        if gap.s < this.precision {
                            out.push(gap);
                        }
                    });
                    let point = &this.points[i];
                    for (t, room) in near.triangles_of(i) {
                        let tri = &this.triangles[t as usize];
                        if near.clear_of_contact(room, point.owner, tri.owner) {
                            continue;
                        }
                        // The search this list stands in for looked only at
                        // triangles whose boxes, grown by ε, reach the point.
                        let query = Aabb::around(p, Vec3::splat(point.radius));
                        let bounds = Aabb::from_points(this.triangle_corners(ctx, tri, x))
                            .grown(this.precision);
                        if !bounds.overlaps(&query) {
                            continue;
                        }
                        let gap = this.gap(ctx, i, Other::Triangle(t), x, p);
                        if gap.s < this.precision {
                            out.push(gap);
                        }
                    }
                }
                out
            },
            |mut a, b| {
                a.extend(b);
                a
            },
        );
        let edge_gaps = ctx.pool.map_reduce(
            self.edges.len(),
            POINTS_PER_CHUNK,
            Vec::new(),
            |range| {
                let mut out = Vec::new();
                for i in range {
                    let a = &this.edges[i];
                    let mut bounds = None;
                    for (j, room) in near.edges_of(i) {
                        let j = j as usize;
                        if near.clear_of_contact(room, a.owner, this.edges[j].owner) {
                            continue;
                        }
                        // As above: only where the edges' boxes overlap.
                        let query = *bounds.get_or_insert_with(|| this.edge_bounds(ctx, i, x));
                        if !this.edge_bounds(ctx, j, x).overlaps(&query) {
                            continue;
                        }
                        let gap = this.edge_gap(ctx, i, j, x);
                        if gap.s < this.precision {
                            out.push(gap);
                        }
                    }
                }
                out
            },
            |mut a, b| {
                a.extend(b);
                a
            },
        );
        gaps.extend(edge_gaps);
        self.gaps = gaps;
        if self.seen.in_stage {
            self.seen.x.clear();
            self.seen.x.extend_from_slice(x);
            self.seen.precision = self.precision;
            self.seen.self_contact = self.self_contact;
            self.seen.valid = true;
        }
    }

    /// What the barrier `m b(s)` is multiplied by in the incremental
    /// potential: `W / H²`, the stage's weight over the step's length
    /// squared.
    fn scale(ctx: &Context) -> f64 {
        ctx.frame.weight / (ctx.frame.step * ctx.frame.step)
    }

    /// `(b, b′, b″)` of the barrier at gap `s`, per unit mass.
    fn barrier(&self, s: f64) -> (f64, f64, f64) {
        let e = self.precision;
        if s >= e {
            return (0.0, 0.0, 0.0);
        }
        if s <= MIN_GAP * e {
            return (f64::INFINITY, f64::NEG_INFINITY, f64::INFINITY);
        }
        let l = (s / e).ln();
        let t = s - e;
        (
            -t * t * l,
            -2.0 * t * l - t * t / s,
            -2.0 * l - 4.0 * t / s + t * t / (s * s),
        )
    }

    /// Takes the contacts at `x` as the ones whose friction acts until the
    /// next call. Returns how much the friction bounds changed from the
    /// last call: the largest relative change, or infinity if the contacts
    /// are not the same ones.
    pub fn grip(&mut self, ctx: &Context, x: &[f64]) -> f64 {
        let step2 = ctx.frame.step * ctx.frame.step;
        let mut grips = Vec::new();
        self.active(ctx, x, true);
        for gap in &self.gaps {
            let (_, db, _) = self.barrier(gap.s);
            let smoothing = self.edge_smoothing(ctx, gap, x).map_or(1.0, |m| m.value);
            let force = gap.mass * -db / step2 * smoothing * gap.distance_gradient.length();
            if gap.friction <= 0.0 || force <= 0.0 {
                continue;
            }
            let sides = self.surface_sides(ctx, gap, x);
            let mut starts = [Vec3::ZERO; 4];
            for (start, side) in starts.iter_mut().zip(&sides[..gap.count]) {
                *start = side.0.start(ctx.layout, ctx.frame);
            }
            let mut grip = Grip {
                pair: gap.pair,
                sides,
                starts,
                count: gap.count,
                normal: gap.normal,
                limit: gap.friction * force,
                unbraked: false,
            };
            if !self.grips.iter().any(|old| old.pair == gap.pair) {
                let d = self.displacement(ctx, &grip, x);
                let slip = (d - grip.normal * grip.normal.dot(d)).length();
                grip.unbraked = slip >= self.stick_speed * ctx.frame.h;
            }
            grips.push(grip);
        }
        let change = if grips.len() != self.grips.len() {
            f64::INFINITY
        } else {
            let total: f64 = grips
                .iter()
                .map(|g| g.limit)
                .sum::<f64>()
                .max(f64::MIN_POSITIVE);
            grips
                .iter()
                .zip(&self.grips)
                .map(|(a, b)| {
                    if a.pair == b.pair {
                        (a.limit - b.limit).abs() / total * grips.len() as f64
                    } else {
                        f64::INFINITY
                    }
                })
                .fold(0.0, f64::max)
        };
        self.grips = grips;
        change
    }

    /// Pairs whose gap at `x` is not positive — a state no solve may start
    /// from — with their object and gap.
    pub fn penetrations(&mut self, ctx: &Context, x: &[f64]) -> Vec<(usize, f64)> {
        self.active(ctx, x, false);
        self.gaps
            .iter()
            .filter(|g| g.s <= MIN_GAP * self.precision)
            .map(|g| (g.owner, g.s))
            .collect()
    }

    /// The pairs in contact at `x`, for the impact law.
    pub fn impacts(&mut self, ctx: &Context, x: &[f64]) -> Vec<Impact> {
        let step2 = ctx.frame.step * ctx.frame.step;
        self.active(ctx, x, true);
        self.gaps
            .iter()
            .map(|gap| Impact {
                sides: self.surface_sides(ctx, gap, x),
                count: gap.count,
                normal: gap.normal,
                restitution: gap.restitution,
                potential: gap.mass * self.barrier(gap.s).0 / step2
                    * self.edge_smoothing(ctx, gap, x).map_or(1.0, |m| m.value),
            })
            .collect()
    }

    /// Frictional contacts in use.
    pub fn grips(&self) -> usize {
        self.grips.len()
    }

    /// Evaluates contact forces on `object` inside a local tactile patch sphere
    /// `(world_origin, radius)` aligned with outward `world_normal`.
    /// Returns `(normal_force, total_force, center_of_pressure, active_contacts)`.
    pub(crate) fn tactile_patch(
        &mut self,
        ctx: &Context,
        x: &[f64],
        object: usize,
        world_origin: Vec3,
        world_normal: Vec3,
        radius: f64,
    ) -> (f64, Vec3, Option<Vec3>, usize) {
        self.active(ctx, x, false);
        let step2 = (ctx.frame.step * ctx.frame.step).max(1e-24);
        let wn = world_normal.try_normalize();
        let e = self.stick_speed * ctx.frame.h;
        let r2 = radius * radius;
        let mut normal_force = 0.0;
        let mut total_force = Vec3::ZERO;
        let mut cop_accum = Vec3::ZERO;
        let mut count = 0usize;

        for gap in &self.gaps {
            let (_, db, _) = self.barrier(gap.s);
            let smoothing = self.edge_smoothing(ctx, gap, x).map_or(1.0, |m| m.value);
            let f_n = gap.mass * -db / step2 * smoothing * gap.distance_gradient.length();
            if f_n <= 0.0 {
                continue;
            }
            let (owner_a, owner_b) = match gap.pair {
                Pair::PointBody(p, b) => (
                    self.points[p as usize].owner,
                    self.body_owner(ctx, self.bodies[b]),
                ),
                Pair::PointTriangle(p, t) => (
                    self.points[p as usize].owner,
                    self.triangles[t as usize].owner,
                ),
                Pair::Edges(ea, eb) => {
                    (self.edges[ea as usize].owner, self.edges[eb as usize].owner)
                }
            };
            let is_a = owner_a == object;
            let is_b = owner_b == object;
            if !is_a && !is_b {
                continue;
            }
            let pos = if matches!(gap.pair, Pair::Edges(_, _)) {
                let mut p = Vec3::ZERO;
                for &(att, w) in &gap.sides[..gap.count] {
                    if (is_a && w > 0.0) || (!is_a && w < 0.0) {
                        p += att.position(ctx.layout, ctx.frame, x) * w.abs();
                    }
                }
                p
            } else {
                let p_a = match gap.pair {
                    Pair::PointBody(p, _) | Pair::PointTriangle(p, _) => self.points[p as usize]
                        .attach
                        .position(ctx.layout, ctx.frame, x),
                    Pair::Edges(_, _) => unreachable!(),
                };
                if is_a {
                    p_a
                } else {
                    p_a - gap.normal * gap.s
                }
            };
            if (pos - world_origin).length_squared() > r2 {
                continue;
            }
            // Outward unit direction from `object` toward the contacting counterparty:
            let outward = if is_a { -gap.normal } else { gap.normal };
            let proj = match wn {
                Some(n) => {
                    let dot = outward.dot(n);
                    if dot <= 0.0 {
                        continue;
                    }
                    dot
                }
                None => 1.0,
            };
            let contrib = f_n * proj;
            normal_force += contrib;
            total_force += (-outward) * f_n;
            cop_accum += pos * contrib;
            count += 1;

            if let Some(grip) = self.grips.iter().find(|g| g.pair == gap.pair) {
                let d = self.displacement(ctx, grip, x);
                let u = d - grip.normal * grip.normal.dot(d);
                let y = u.length();
                if y > 1e-12 {
                    let ft_mag = grip.limit * friction_profile(y, e).1;
                    let ft_on_a = -u * (ft_mag / y);
                    total_force += if is_a { ft_on_a } else { -ft_on_a };
                }
            }
        }

        let cop = (normal_force > 0.0).then(|| cop_accum / normal_force);
        (normal_force, total_force, cop, count)
    }

    /// Frictional power dissipated over the stage (W), distributing half onto
    /// each contacting deformable solid node when present.
    pub(crate) fn friction_power(
        &self,
        ctx: &Context,
        x: &[f64],
        solid_power: &mut [Vec<f64>],
    ) -> f64 {
        if ctx.frame.h <= 0.0 {
            return 0.0;
        }
        let e = self.stick_speed * ctx.frame.h;
        let inv_h = 1.0 / ctx.frame.h;
        let mut total = 0.0;
        for g in &self.grips {
            let d = self.displacement(ctx, g, x);
            let y = (d - g.normal * g.normal.dot(d)).length();
            let power = g.limit * friction_profile(y, e).1 * y * inv_h;
            if power <= 0.0 {
                continue;
            }
            total += power;
            for &(att, w) in &g.sides[..g.count] {
                if let Attach::Node { solid, node } = att {
                    if let Some(dst) = solid_power
                        .get_mut(solid as usize)
                        .and_then(|v| v.get_mut(node as usize))
                    {
                        *dst += 0.5 * power * w.abs();
                    }
                }
            }
        }
        total
    }

    fn combine(ctx: &Context, sides: &[(Attach, f64)], x: &[f64]) -> Combined {
        let mut out = [(0u32, Mat3::ZERO); 8];
        let mut len = 0;
        for &(attach, weight) in sides {
            if weight == 0.0 {
                continue;
            }
            for &(block, j) in attach.jacobian(ctx.layout, ctx.frame, x).iter() {
                match out[..len].iter_mut().find(|(b, _)| *b == block) {
                    Some((_, g)) => *g += j * weight,
                    None => {
                        out[len] = (block, j * weight);
                        len += 1;
                    }
                }
            }
        }
        (out, len)
    }

    /// How far a grip's two sides moved apart over the step, at `x`.
    fn displacement(&self, ctx: &Context, grip: &Grip, x: &[f64]) -> Vec3 {
        let mut d = Vec3::ZERO;
        for k in 0..grip.count {
            let (attach, weight) = grip.sides[k];
            if weight != 0.0 {
                d += (attach.position(ctx.layout, ctx.frame, x) - grip.starts[k]) * weight;
            }
        }
        d
    }
}

/// A shape's signed distance as contact sees it: a box with its edges and
/// corners rounded as its contact points are ([`CORNER_ROUNDING`]) — a box
/// shrunk by `ρ`, then grown by `ρ` all round. Its distance is smooth
/// everywhere within `ρ` of the surface: a sharp box's normal jumps across
/// every edge, and a pile of boxes puts corners on edges everywhere, where
/// Newton's quadratic model keeps breaking.
fn contact_distance(shape: &Shape, p: Vec3) -> (f64, Vec3) {
    match *shape {
        Shape::Cuboid { half } => {
            let rounding = CORNER_ROUNDING * half.min_element();
            let (d, n) = Shape::Cuboid {
                half: half - Vec3::splat(rounding),
            }
            .distance(p);
            (d - rounding, n)
        }
        _ => shape.distance(p),
    }
}

/// The smoothed friction profile: `f₀` (energy per unit `μλ`) and `f₁ = f₀′`.
fn friction_profile(y: f64, e: f64) -> (f64, f64) {
    if y >= e {
        (y, 1.0)
    } else {
        (
            y * y / e - y * y * y / (3.0 * e * e) + e / 3.0,
            2.0 * y / e - y * y / (e * e),
        )
    }
}

/// One chunk of pairs' or grips' share of the gradient and Hessian, before
/// it is added in. Kept from one evaluation to the next: a pile's shares
/// are megabytes, and asking for that memory afresh every Newton iteration
/// cost as much as filling it.
#[derive(Clone, Debug, Default)]
struct Share {
    gradient: Vec<(u32, Vec3)>,
    hessian: Vec<Triplet>,
}

impl Term for Contact {
    fn name(&self) -> &'static str {
        "contact"
    }

    fn begin_step(&mut self, _ctx: &Context) {
        // A stage's constants now stand until `forget`: within it, the
        // pairs in contact are a function of the unknowns alone.
        self.seen.in_stage = true;
        self.seen.valid = false;
    }

    fn energy(&mut self, ctx: &Context, x: &[f64]) -> f64 {
        self.active(ctx, x, false);
        let gaps = &self.gaps;
        let barrier = gaps
            .iter()
            .map(|gap| {
                let energy = gap.mass * self.barrier(gap.s).0;
                if energy.is_finite() {
                    energy * self.edge_smoothing(ctx, gap, x).map_or(1.0, |m| m.value)
                } else {
                    energy
                }
            })
            .sum::<f64>()
            * Self::scale(ctx);
        let h2 = ctx.frame.weight;
        let e = self.stick_speed * ctx.frame.h;
        let this = &*self;
        let dissipation = ctx.pool.map_reduce(
            self.grips.len(),
            POINTS_PER_CHUNK,
            0.0,
            |range| {
                range
                    .map(|k| {
                        let g = &this.grips[k];
                        let d = this.displacement(ctx, g, x);
                        let slip = d - g.normal * g.normal.dot(d);
                        h2 * g.limit * friction_profile(slip.length(), e).0
                    })
                    .sum::<f64>()
            },
            |a, b| a + b,
        );
        barrier + dissipation
    }

    fn external_forces(&mut self, ctx: &Context, x: &[f64]) -> bool {
        self.grips.iter().any(|g| {
            g.sides[..g.count]
                .iter()
                .any(|(a, _)| matches!(a, Attach::Still(_)))
        }) || self.boundary_contact(ctx, x)
    }

    fn derivatives(
        &mut self,
        ctx: &Context,
        x: &[f64],
        gradient: &mut [f64],
        _fixed: &mut BlockMatrix,
        changing: &mut Triplets,
    ) {
        self.assemble(ctx, x, gradient, Some(changing));
    }

    fn step_limit(&mut self, ctx: &Context, x: &[f64], dx: &[f64]) -> f64 {
        self.update_positions(ctx, x);
        let point_reach: Vec<f64> = self
            .points
            .iter()
            .map(|p| p.attach.reach(ctx.layout, ctx.frame, x, dx))
            .collect();
        let node_reach = |solid: u32, node: u32| {
            Attach::Node { solid, node }.reach(ctx.layout, ctx.frame, x, dx)
        };
        let triangle_reach = |t: &Triangle| {
            t.nodes
                .iter()
                .map(|&n| node_reach(t.solid, n))
                .fold(0.0, f64::max)
        };
        // A rigid body's point at distance `arm` from its centre moves by at
        // most |δc| + arm Σₖ |δsₖ| / r.
        let turn = |r: usize| {
            let b = ctx.layout.rigids[r].block;
            (1..4).map(|k| Vec3::read(dx, b + k).length()).sum::<f64>() / ctx.frame.rigids[r].radius
        };
        let rigid_reach = |r: usize, arm: f64| {
            Vec3::read(dx, ctx.layout.rigids[r].block).length() + turn(r) * arm
        };
        let body_reach = |body: Body| match body {
            Body::Fixed(_) => 0.0,
            Body::Rigid(r) => rigid_reach(r, ctx.frame.rigids[r].reach),
        };
        self.fit_bodies(ctx, x, &body_reach, false);
        // How far a pair's two sides can move *relative to each other*: the
        // shared motion (two boxes falling together, a node riding on a
        // moving face) cannot close a gap, so it is taken out (as in
        // additive CCD). Each side moves by a translation `T` plus at most
        // `e` more (rotation, or a triangle's corners differing from their
        // mean); the gap closes by at most `|T_a − T_b| + e_a + e_b`.
        let node_motion = |solid: u32, node: u32| {
            let b = ctx.layout.solids[solid as usize].block[node as usize];
            if b == crate::dofs::HELD {
                Vec3::ZERO
            } else {
                Vec3::read(dx, b as usize)
            }
        };
        let rigid_motion =
            |r: usize, arm: f64| (Vec3::read(dx, ctx.layout.rigids[r].block), turn(r) * arm);
        // Whether the pairs listed near the reference state are every pair
        // this step can bring close: the list must stand here, and no
        // object may move, relative to another, by more than the room every
        // unlisted pair still has. Otherwise search along the whole sweep.
        let listed = self.seen.in_stage
            && self.near.stands(
                &self.points,
                &self.positions,
                self.precision,
                self.self_contact,
            )
            && self.near.sweeps(&self.points, |point| match point.attach {
                Attach::Node { solid, node } => (node_motion(solid, node), 0.0),
                Attach::Rigid { rigid, .. } => {
                    rigid_motion(rigid as usize, ctx.frame.rigids[rigid as usize].reach)
                }
                Attach::Still(_) => (Vec3::ZERO, 0.0),
            });
        if !listed {
            let reach: Vec<f64> = self.triangles.iter().map(triangle_reach).collect();
            self.fit_triangles(ctx, x, &reach, 0.0, false);
            self.fit_edges(ctx, x, Some(dx), 0.0, false);
        }
        let this = &*self;
        let near = &this.near;
        // One pair of a point: the limit it leaves of `alpha`.
        let point_limit = |i: usize, other: Other, alpha: f64| {
            let p = this.positions[i];
            let s0 = this.gap(ctx, i, other, x, p).s;
            let (ta, ea) = match this.points[i].attach {
                Attach::Node { solid, node } => (node_motion(solid, node), 0.0),
                Attach::Rigid { rigid, local } => rigid_motion(rigid as usize, local.length()),
                Attach::Still(_) => (Vec3::ZERO, 0.0),
            };
            let (tb, eb) = match other {
                Other::Body(Body::Fixed(_)) => (Vec3::ZERO, 0.0),
                Other::Body(Body::Rigid(r)) => {
                    let arm = (p - ctx.frame.rigid(ctx.layout, r, x).0).length()
                        + s0.max(0.0)
                        + point_reach[i]
                        + this.precision;
                    rigid_motion(r, arm)
                }
                Other::Triangle(t) => {
                    let tri = &this.triangles[t as usize];
                    let d = tri.nodes.map(|n| node_motion(tri.solid, n));
                    let mean = (d[0] + d[1] + d[2]) / 3.0;
                    (
                        mean,
                        d.iter().map(|v| (*v - mean).length()).fold(0.0, f64::max),
                    )
                }
            };
            let reach = (ta - tb).length() + ea + eb;
            if reach <= 0.9 * s0 {
                return alpha; // cannot close even over the whole step
            }
            alpha.min(match other {
                Other::Body(Body::Fixed(_)) => this.advance_signed(ctx, i, other, x, dx, s0, alpha),
                // The inverse of a changing affine matrix is not
                // linear in t. Its signed distance need not be
                // convex: an initially opening gap can close.
                Other::Body(Body::Rigid(_)) | Other::Triangle(_) => {
                    this.advance(ctx, i, other, x, dx, reach, alpha)
                }
            })
        };
        // Each chunk of points keeps its own running limit; the chunks are
        // fixed, so the result does not depend on the thread count.
        let alpha = ctx.pool.map_reduce(
            self.points.len(),
            POINTS_PER_CHUNK,
            1.0f64,
            |range| {
                let mut alpha = 1.0f64;
                let mut scratch = Vec::new();
                for i in range {
                    let reach = point_reach[i];
                    if !listed {
                        this.point_candidates(&this.body_cache, i, reach, &mut scratch, |other| {
                            alpha = point_limit(i, other, alpha);
                        });
                        continue;
                    }
                    this.body_candidates(&this.body_cache, i, reach, &mut scratch, |body| {
                        alpha = point_limit(i, Other::Body(body), alpha);
                    });
                    let point = &this.points[i];
                    for (t, room) in near.triangles_of(i) {
                        let tri = &this.triangles[t as usize];
                        if near.out_of_reach(room, point.owner, tri.owner) {
                            continue;
                        }
                        // The search this list stands in for looked only at
                        // triangles whose swept boxes reach the point's.
                        let query =
                            Aabb::around(this.positions[i], Vec3::splat(point.radius + reach));
                        let bounds = Aabb::from_points(this.triangle_corners(ctx, tri, x))
                            .grown(this.precision + triangle_reach(tri));
                        if bounds.overlaps(&query) {
                            alpha = point_limit(i, Other::Triangle(t), alpha);
                        }
                    }
                }
                alpha
            },
            f64::min,
        );
        let alpha = ctx.pool.map_reduce(
            self.edges.len(),
            POINTS_PER_CHUNK,
            alpha,
            |range| {
                let mut limit = alpha;
                for i in range {
                    if !listed {
                        this.edge_candidates(i, |j| {
                            let (pa, pb) = (this.edge_positions[i], this.edge_positions[j]);
                            limit = limit.min(this.advance_edges(ctx, i, j, pa, pb, x, dx, limit));
                        });
                        continue;
                    }
                    let a = &this.edges[i];
                    let mut swept = None;
                    for (j, room) in near.edges_of(i) {
                        let j = j as usize;
                        if near.out_of_reach(room, a.owner, this.edges[j].owner) {
                            continue;
                        }
                        // As above: only where the edges' swept boxes overlap.
                        let (pa, query) =
                            *swept.get_or_insert_with(|| this.edge_sweep(ctx, i, x, dx));
                        let (pb, bounds) = this.edge_sweep(ctx, j, x, dx);
                        if bounds.overlaps(&query) {
                            limit = limit.min(this.advance_edges(ctx, i, j, pa, pb, x, dx, limit));
                        }
                    }
                }
                limit
            },
            f64::min,
        );
        alpha.max(0.0)
    }
}

impl Contact {
    /// Adds the gradient at `x` to `gradient`: the forces of contact and
    /// friction, without the work of their Hessian.
    pub(crate) fn gradient(&mut self, ctx: &Context, x: &[f64], gradient: &mut [f64]) {
        self.assemble(ctx, x, gradient, None);
    }

    /// The gradient, and the Hessian's entries if asked for: pair by pair
    /// and grip by grip in parallel chunks, each into its own share, added
    /// in chunk order — barriers, then friction.
    fn assemble(
        &mut self,
        ctx: &Context,
        x: &[f64],
        gradient: &mut [f64],
        mut changing: Option<&mut Triplets>,
    ) {
        self.active(ctx, x, false);
        let scale = Self::scale(ctx);
        let hessian = changing.is_some();
        let mut shares = std::mem::take(&mut self.shares);
        let this = &*self;
        let gaps = &this.gaps;
        let pair_chunks = gaps.len().div_ceil(POINTS_PER_CHUNK);
        let grip_chunks = this.grips.len().div_ceil(POINTS_PER_CHUNK);
        if shares.len() < pair_chunks + grip_chunks {
            shares.resize_with(pair_chunks + grip_chunks, Share::default);
        }
        let (of_pairs, of_grips) = shares.split_at_mut(pair_chunks);
        ctx.pool.for_chunks_mut(of_pairs, 1, |chunk, share| {
            let share = &mut share[0];
            share.gradient.clear();
            share.hessian.clear();
            let first = chunk * POINTS_PER_CHUNK;
            for gap in &gaps[first..(first + POINTS_PER_CHUNK).min(gaps.len())] {
                let (energy, db, ddb) = this.barrier(gap.s);
                let smoothing = this.edge_smoothing(ctx, gap, x);
                let weight = smoothing.as_ref().map_or(1.0, |m| m.value);
                let factor = scale * gap.mass;
                let (db, ddb) = (db * factor * weight, ddb * factor * weight);
                if let Some(m) = smoothing {
                    // Exact product-rule gradient; a positive Gauss-Newton
                    // angular block supplements the distance Hessian below.
                    // Omitting indefinite cross/geometric blocks affects
                    // only the Newton approximation, never the energy/force.
                    let mut maps = [(0u32, Mat3::ZERO); 16];
                    let mut count = 0;
                    for k in 0..4 {
                        for &(b, j) in m.ends[k].jacobian(ctx.layout, ctx.frame, x).iter() {
                            maps[count] = (b, m.maps[k] * j);
                            count += 1;
                        }
                    }
                    let coefficient = energy * factor * m.slope;
                    for &(b, j) in &maps[..count] {
                        let transposed = j.transpose();
                        share.gradient.push((b, transposed * m.w * coefficient));
                        if hessian {
                            for &(c, k) in &maps[..count] {
                                share.hessian.push((b, c, transposed * k * coefficient));
                            }
                        }
                    }
                }
                let (blocks, len) = Self::combine(ctx, &gap.sides[..gap.count], x);
                let mut g = [(0u32, Vec3::ZERO); 8];
                for (out, &(b, j)) in g.iter_mut().zip(&blocks[..len]) {
                    *out = (b, j.transpose() * gap.distance_gradient);
                }
                for &(b, gi) in &g[..len] {
                    share.gradient.push((b, gi * db));
                    if hessian {
                        for &(c, gj) in &g[..len] {
                            share.hessian.push((b, c, Mat3::outer(gi, gj) * ddb));
                        }
                    }
                }
                if hessian && db < 0.0 && db.is_finite() {
                    if let (Pair::PointBody(_, _), Attach::Rigid { rigid, local }) =
                        (gap.pair, gap.sides[1].0)
                    {
                        let (_, a) = ctx.frame.rigid(ctx.layout, rigid as usize, x);
                        let inv_a = a.inverse();
                        let n = (a.transpose() * gap.distance_gradient).normalize_or_zero();
                        let qn = local.length();
                        if let Some(qh) = local.try_normalize() {
                            let nq = n.dot(qh) * qn;
                            let cross = n.cross(qh);
                            let k_w = if cross.length_squared() > 1e-24 {
                                let e3 = cross.normalize();
                                let e2 = (n - qh).normalize();
                                let lam3 = nq.max(0.0);
                                let lam2 = (0.5 * (qn + nq)).max(0.0);
                                (Mat3::outer(e3, e3) * lam3 + Mat3::outer(e2, e2) * lam2)
                                    * (2.0 * (-db))
                            } else if nq > 0.0 {
                                (Mat3::IDENTITY - Mat3::outer(n, n)) * (2.0 * (-db) * nq)
                            } else {
                                Mat3::ZERO
                            };
                            if k_w != Mat3::ZERO {
                                let scale_r = 1.0 / ctx.frame.rigids[rigid as usize].radius;
                                let rb = ctx.layout.rigids[rigid as usize].block as u32;
                                let (r0, r1, r2) = (inv_a.row(0), inv_a.row(1), inv_a.row(2));
                                let b_mats = [
                                    Mat3::from_rows(
                                        Vec3::ZERO,
                                        r2 * (-0.5 * scale_r),
                                        r1 * (0.5 * scale_r),
                                    ),
                                    Mat3::from_rows(
                                        r2 * (0.5 * scale_r),
                                        Vec3::ZERO,
                                        r0 * (-0.5 * scale_r),
                                    ),
                                    Mat3::from_rows(
                                        r1 * (-0.5 * scale_r),
                                        r0 * (0.5 * scale_r),
                                        Vec3::ZERO,
                                    ),
                                ];
                                for (k, bk) in b_mats.iter().enumerate() {
                                    let bkt_kw = bk.transpose() * k_w;
                                    for (l, bl) in b_mats.iter().enumerate() {
                                        share.hessian.push((
                                            rb + 1 + k as u32,
                                            rb + 1 + l as u32,
                                            bkt_kw * *bl,
                                        ));
                                    }
                                }
                            }
                        }
                    }
                }
            }
        });
        let h2 = ctx.frame.weight;
        let e = self.stick_speed * ctx.frame.h;
        ctx.pool
            .for_chunks_mut(&mut of_grips[..grip_chunks], 1, |chunk, share| {
                let share = &mut share[0];
                share.gradient.clear();
                share.hessian.clear();
                let first = chunk * POINTS_PER_CHUNK;
                for grip in &this.grips[first..(first + POINTS_PER_CHUNK).min(this.grips.len())] {
                    let d = this.displacement(ctx, grip, x);
                    let u = d - grip.normal * grip.normal.dot(d);
                    let y = u.length();
                    let (_, f1) = friction_profile(y, e);
                    let plane = Mat3::IDENTITY - Mat3::outer(grip.normal, grip.normal);
                    let scale = h2 * grip.limit;
                    // ∇ᵤD and a positive semi-definite ∇²ᵤD, in 3-D within the plane.
                    let (force, stiffness) = if grip.unbraked && y > 0.0 {
                        let dir = u / y;
                        let dir_outer = Mat3::outer(dir, dir);
                        let s = scale * f1 / y;
                        let s_radial = if y < e { s } else { 0.55 * s };
                        (u * s, (plane - dir_outer) * s + dir_outer * s_radial)
                    } else if y >= e {
                        let dir = u / y;
                        (dir * scale, (plane - Mat3::outer(dir, dir)) * (scale / y))
                    } else if y > 0.0 {
                        (
                            u * (scale * f1 / y),
                            plane * (scale * (2.0 / e - y / (e * e)))
                                - Mat3::outer(u, u) * (scale / (e * e * y)),
                        )
                    } else {
                        (Vec3::ZERO, plane * (scale * 2.0 / e))
                    };
                    let (blocks, len) = Self::combine(ctx, &grip.sides[..grip.count], x);
                    // The slip's derivative is the plane projection of G.
                    let mut projected = [Mat3::ZERO; 8];
                    for (out, &(_, g)) in projected.iter_mut().zip(&blocks[..len]) {
                        *out = plane * g;
                    }
                    for (&(b, _), pb) in blocks[..len].iter().zip(&projected) {
                        let transposed = pb.transpose();
                        share.gradient.push((b, transposed * force));
                        if hessian {
                            let left = transposed * stiffness;
                            for (&(c, _), pc) in blocks[..len].iter().zip(&projected) {
                                share.hessian.push((b, c, left * *pc));
                            }
                        }
                    }
                }
            });
        for share in &mut shares[..pair_chunks + grip_chunks] {
            for &(b, g) in &share.gradient {
                g.add_to(gradient, b as usize);
            }
            if let Some(changing) = changing.as_deref_mut() {
                changing.append(&mut share.hessian);
            }
        }
        self.shares = shares;
    }

    /// Additive CCD on four linearly moving endpoints (Li et al. 2021).
    /// Subtract common translation from the speed bound, so a pair moving
    /// together does not constrain the step. Recompute closest parameters
    /// along the sweep: endpoint/interior transitions must not be frozen.
    #[allow(clippy::too_many_arguments)]
    fn advance_edges(
        &self,
        ctx: &Context,
        i: usize,
        j: usize,
        pa: [Vec3; 2],
        pb: [Vec3; 2],
        x: &[f64],
        dx: &[f64],
        limit: f64,
    ) -> f64 {
        let (a, b) = (&self.edges[i], &self.edges[j]);
        let end = Along { x, dx, t: 1.0 };
        let da = std::array::from_fn::<_, 2, _>(|k| {
            a.ends[k].position(ctx.layout, ctx.frame, end) - pa[k]
        });
        let db = std::array::from_fn::<_, 2, _>(|k| {
            b.ends[k].position(ctx.layout, ctx.frame, end) - pb[k]
        });
        let ma = (da[0] + da[1]) * 0.5;
        let mb = (db[0] + db[1]) * 0.5;
        let reach = (ma - mb).length()
            + (da[0] - ma).length().max((da[1] - ma).length())
            + (db[0] - mb).length().max((db[1] - mb).length());
        let closest_at = |t: f64| {
            let (p, q, _) = closest_segments(
                pa[0] + da[0] * t,
                pa[1] + da[1] * t,
                pb[0] + db[0] * t,
                pb[1] + db[1] * t,
            );
            (p, q, (p - q).length())
        };
        let (p0, q0, d0) = closest_at(0.0);
        let offset = a.radius + b.radius
            - if a.owner == b.owner {
                0.0
            } else {
                0.5 * self.precision
            };
        // Also retain positive geometric separation for initially separated
        // zero-thickness mesh edges, whose barrier permits ε/2 overlap.
        let min_gap = MIN_GAP * self.precision;
        let min_floor = (2.0 * min_gap).min(0.5 * ((d0 - offset).max(0.0) + min_gap));
        let floor = (offset + (0.1 * (d0 - offset)).max(min_floor)).max(0.1 * d0);
        if reach == 0.0 || (limit * reach <= d0 - floor) {
            return limit;
        }
        if d0 == 0.0 && offset < 0.0 {
            // Exactly touching thin edges have no side or unique normal.
            // Preserve the existing ε/2 contact tolerance at initialization.
            return limit;
        }
        let stop_margin = 0.05 * (d0 - floor).max(0.0);
        let mut t = 0.0;
        for iter in 0..1000 {
            let (p, q, d) = if iter == 0 {
                (p0, q0, d0)
            } else {
                closest_at(t)
            };
            if d <= floor + stop_margin {
                return t;
            }
            if (limit - t) * reach <= d - floor {
                return limit;
            }
            let mut advance = 0.9 * (d - floor) / reach;
            // Exact conservative 4-corner separating-plane jump: since every
            // point on each edge is a convex combination of its endpoints, if
            // n . (a_u(t') - b_v(t')) > floor for all (u, v) in {0, 1}^2,
            // then the minimum segment distance exceeds floor across [t, t'].
            if d > 0.0 {
                let n = (p - q) / d;
                let mut plane = limit - t;
                let mut valid = true;
                'pairs: for u in 0..2 {
                    for v in 0..2 {
                        let start = pa[u] + da[u] * t - (pb[v] + db[v] * t);
                        let margin = n.dot(start) - floor;
                        if margin <= 0.0 {
                            valid = false;
                            break 'pairs;
                        }
                        let rate = n.dot(da[u] - db[v]);
                        if rate < 0.0 {
                            plane = plane.min(margin / (-rate));
                        }
                    }
                }
                if valid {
                    let safe = 0.9 * plane;
                    if safe >= limit - t {
                        return limit;
                    }
                    advance = advance.max(safe);
                }
            }
            if t > 0.0 && advance < 1e-8 * limit {
                return t;
            }
            t = (t + advance).min(limit);
            if t >= limit {
                return limit;
            }
        }
        t
    }

    /// The safe fraction of `x + t dx`, `t ∈ [0, limit]`, for a point against
    /// a body, keeping the gap above a tenth of where it started. The gap is
    /// signed and, for a point moving linearly relative to a convex shape
    /// (a node or a rigid body's point, against anything fixed), convex in
    /// `t`: if it opens at `t = 0` the whole step is safe; if it ends below
    /// the floor, the crossing is found by bisection; otherwise its least
    /// value is found, and bisected to if it dips too low — which also
    /// catches a step that would carry the point right through.
    #[allow(clippy::too_many_arguments)]
    fn advance_signed(
        &self,
        ctx: &Context,
        i: usize,
        other: Other,
        x: &[f64],
        dx: &[f64],
        s0: f64,
        limit: f64,
    ) -> f64 {
        let point = &self.points[i];
        let gap = |t: f64| {
            let along = Along { x, dx, t };
            self.gap(
                ctx,
                i,
                other,
                along,
                point.attach.position(ctx.layout, ctx.frame, along),
            )
            .s
        };
        let min_gap = MIN_GAP * self.precision;
        let min_floor = (2.0 * min_gap).min(0.5 * (s0 + min_gap));
        let floor = (0.1 * s0).max(min_floor);
        // The gap's rate at t = 0: Σ wₖ n · (∂pₖ/∂x dx).
        let start = self.gap(ctx, i, other, x, self.positions[i]);
        let (blocks, len) = Self::combine(ctx, &start.sides[..start.count], x);
        let slope: f64 = blocks[..len]
            .iter()
            .map(|&(b, g)| start.distance_gradient.dot(g * Vec3::read(dx, b as usize)))
            .sum();
        if slope >= 0.0 {
            return limit;
        }
        let mut high = limit;
        if gap(limit) > floor {
            // Find the least gap on [0, limit] (golden-section search).
            let r = 0.5 * (5f64.sqrt() - 1.0);
            let (mut a, mut b) = (0.0, limit);
            let (mut c, mut d) = (b - r * (b - a), a + r * (b - a));
            let (mut gc, mut gd) = (gap(c), gap(d));
            for _ in 0..40 {
                if gc < gd {
                    b = d;
                    d = c;
                    gd = gc;
                    c = b - r * (b - a);
                    gc = gap(c);
                } else {
                    a = c;
                    c = d;
                    gc = gd;
                    d = a + r * (b - a);
                    gd = gap(d);
                }
                if b - a < 1e-6 * limit {
                    break;
                }
            }
            let least = 0.5 * (a + b);
            if gap(least) > floor {
                return limit;
            }
            high = least;
        }
        // Bisect for where the gap reaches the floor; keep the safe side.
        let mut low = 0.0;
        for _ in 0..50 {
            let mid = 0.5 * (low + high);
            if gap(mid) > floor {
                low = mid;
            } else {
                high = mid;
            }
            if high - low < 1e-9 * limit {
                break;
            }
        }
        low
    }

    /// Conservative advancement of one pair along `x + t dx`, `t ∈ [0,
    /// limit]`, keeping the gap above a tenth of where it started (as
    /// additive CCD does, Li et al. 2021): the gap falls by at most
    /// `reach · Δt` over `Δt`, so each advance by `(s − s_min) / reach` (with
    /// `s` measured where the march stands) can never cross `s_min`. Motion
    /// that does not close the gap — sliding along a surface — costs a few
    /// cheap evaluations, not a shorter step. Returns the safe fraction.
    #[allow(clippy::too_many_arguments)]
    fn advance(
        &self,
        ctx: &Context,
        i: usize,
        other: Other,
        x: &[f64],
        dx: &[f64],
        reach: f64,
        limit: f64,
    ) -> f64 {
        let point = &self.points[i];
        let gap_at = |t: f64| {
            let along = Along { x, dx, t };
            self.gap(
                ctx,
                i,
                other,
                along,
                point.attach.position(ctx.layout, ctx.frame, along),
            )
        };
        let start = self.gap(ctx, i, other, x, self.positions[i]);
        let s_start = start.s;
        let slope = if s_start < self.precision {
            let (blocks, len) = Self::combine(ctx, &start.sides[..start.count], x);
            blocks[..len]
                .iter()
                .map(|&(b, g)| start.distance_gradient.dot(g * Vec3::read(dx, b as usize)))
                .sum()
        } else {
            -1.0
        };
        let keep = if s_start < self.precision && slope >= 0.0 {
            0.8
        } else {
            0.1
        };
        let min_gap = MIN_GAP * self.precision;
        let min_floor = (2.0 * min_gap).min(0.5 * (s_start + min_gap));
        let floor = (keep * s_start).max(min_floor);
        let required = (point.radius - 0.5 * self.precision + floor).max(0.0);
        let stop_margin = 0.05 * (s_start - floor).max(0.0);
        let mut t = 0.0;
        for iter in 0..1000 {
            let g = if iter == 0 { start } else { gap_at(t) };
            let s = g.s;
            if s <= floor + stop_margin {
                break;
            }
            if (limit - t) * reach <= s - floor {
                t = limit;
                break;
            }
            let mut advance = (s - floor) / reach;
            if let Other::Body(Body::Rigid(r)) = other {
                if self.plane_margin(ctx, i, r, g.normal, required, x, dx, t) > 0.0 {
                    if self.plane_margin(ctx, i, r, g.normal, required, x, dx, limit) > 0.0 {
                        t = limit;
                        break;
                    }
                    let (mut low, mut high) = (t, limit);
                    for _ in 0..12 {
                        let mid = 0.5 * (low + high);
                        if self.plane_margin(ctx, i, r, g.normal, required, x, dx, mid) > 0.0 {
                            low = mid;
                        } else {
                            high = mid;
                        }
                    }
                    advance = advance.max(0.9 * (low - t));
                }
            }
            if t > 0.0 && advance < 1e-6 * limit {
                break;
            }
            t = (t + advance).min(limit);
            if t >= limit {
                break;
            }
        }
        t
    }

    /// Concave world-space separating-plane margin at `t` for a fixed unit
    /// `normal` and local distance threshold `required`. Because a convex
    /// body's support and `required * |A(t)ᵀ n|` are both convex in `t` while
    /// `n · (p(t) − c(t))` is affine, this margin is concave in `t`: if it is
    /// positive at both endpoints of an interval `[t₀, t₁]`, the true gap
    /// stays above `floor` throughout `[t₀, t₁]`.
    #[allow(clippy::too_many_arguments)]
    fn plane_margin(
        &self,
        ctx: &Context,
        i: usize,
        rigid: usize,
        normal: Vec3,
        required: f64,
        x: &[f64],
        dx: &[f64],
        t: f64,
    ) -> f64 {
        let point = &self.points[i];
        let shape = ctx.frame.rigids[rigid].shape;
        let along = Along { x, dx, t };
        let (centre, matrix) = ctx.frame.rigid(ctx.layout, rigid, along);
        let direction = matrix.transpose() * normal;
        let len = direction.length();
        let support = match shape {
            Shape::Sphere { radius } => radius * len,
            Shape::Capsule {
                radius,
                half_length,
            } => radius * len + half_length * direction.y.abs(),
            Shape::Cuboid { half } => {
                let rounding = CORNER_ROUNDING * half.min_element();
                (half - Vec3::splat(rounding)).dot(direction.abs()) + rounding * len
            }
            Shape::HalfSpace { .. } => f64::INFINITY,
        };
        let p = point.attach.position(ctx.layout, ctx.frame, along);
        normal.dot(p - centre) - support - required * len
    }

    #[cfg(test)]
    fn separating_plane_safe(
        &self,
        ctx: &Context,
        i: usize,
        rigid: usize,
        x: &[f64],
        dx: &[f64],
        limit: f64,
    ) -> bool {
        let g = self.gap(
            ctx,
            i,
            Other::Body(Body::Rigid(rigid)),
            x,
            self.positions[i],
        );
        let floor = (0.1 * g.s).max(2.0 * MIN_GAP * self.precision);
        if g.s <= floor {
            return false;
        }
        let required = (self.points[i].radius - 0.5 * self.precision + floor).max(0.0);
        self.plane_margin(ctx, i, rigid, g.normal, required, x, dx, 0.0) > 0.0
            && self.plane_margin(ctx, i, rigid, g.normal, required, x, dx, limit) > 0.0
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::dofs::{Frame, Layout, RigidFrame, SolidDofs};
    use core_math::{vec3, Pose, Quat};
    use core_runtime::Pool;
    use core_world::World;

    fn posed_frame(world: &World, layout: &Layout) -> Frame {
        let mut frame = Frame {
            h: 0.01,
            step: 0.01,
            weight: 0.0001,
            start: vec![0.0; 3 * layout.blocks],
            ..Default::default()
        };
        for d in &layout.rigids {
            let Representation::Rigid(body) = &world.objects()[d.object].representation else {
                unreachable!()
            };
            let radius = crate::rigid::gyration_radius(body.mass, &body.inertia);
            let a = body.pose.rotation.to_mat3();
            body.pose.position.write(&mut frame.start, d.block);
            for k in 0..3 {
                (a.col(k) * radius).write(&mut frame.start, d.block + 1 + k);
            }
            frame.rigids.push(RigidFrame {
                mass: body.mass,
                inertia: body.inertia,
                moment: crate::rigid::second_moment(&body.inertia),
                radius,
                centre_start: body.pose.position,
                rotation_start: body.pose.rotation,
                matrix_start: a,
                predicted: body.pose.rotation,
                world_inertia: body.world_inertia(),
                shape: body.shape,
                points: contact_points(&body.shape),
                reach: body.shape.local_bounds().size().length(),
            });
        }
        frame
    }

    #[test]
    fn analytic_body_barriers_and_frozen_surface_friction_have_exact_affine_gradients() {
        let pool = Pool::new(2);
        for (shape, local) in [
            (Shape::Sphere { radius: 1.0 }, vec3(1.002, 0.0, 0.0)),
            (
                Shape::Capsule {
                    radius: 0.5,
                    half_length: 0.4,
                },
                vec3(0.502, 0.1, 0.0),
            ),
            (Shape::cuboid(Vec3::splat(2.0)), vec3(1.002, 0.1, 0.0)),
        ] {
            let mut world = World::new();
            let m = world.add_material(core_world::Material::new(
                "test", 1000.0, 1e6, 0.3, 0.4, 0.0,
            ));
            let rotation = Quat::from_axis_angle(vec3(1.0, 2.0, 3.0).normalize(), 0.4);
            let a = rotation.to_mat3()
                * Mat3::from_cols(vec3(1.03, 0.02, 0.0), vec3(0.01, 0.98, 0.0), Vec3::Z);
            world.add_fixed(
                "point",
                m,
                Shape::Sphere { radius: 1e-5 },
                Pose::at(a * local),
            );
            world.add_rigid("body", m, shape, Pose::new(Vec3::ZERO, rotation));
            let layout = Layout::new(&world);
            let mut frame = posed_frame(&world, &layout);
            frame.rigids[0].matrix_start = a;
            for k in 0..3 {
                (a.col(k) * frame.rigids[0].radius).write(&mut frame.start, k + 1);
            }
            let ctx = Context {
                pool: &pool,
                world: &world,
                layout: &layout,
                frame: &frame,
            };
            let mut contact = Contact::new(&ctx, 0.01);
            // Isolate fixed point -> affine SDF, without corner/edge terms.
            contact.points.retain(|p| p.owner == 0);
            contact.bodies.retain(|b| matches!(b, Body::Rigid(_)));
            contact.edges.clear();
            contact.edge_owners.clear();
            let mut x = frame.start.clone();
            for friction in [false, true] {
                if friction {
                    contact.grip(&ctx, &x);
                    assert!(contact.grips() > 0);
                    let t = rotation.rotate(Vec3::Y) * 1e-4;
                    t.write(&mut x, 0);
                }
                let mut gradient = vec![0.0; x.len()];
                contact.derivatives(
                    &ctx,
                    &x,
                    &mut gradient,
                    &mut BlockMatrix::default(),
                    &mut Triplets::new(layout.blocks),
                );
                let error = core_numerics::check::gradient_error(
                    |q| contact.energy(&ctx, q),
                    &x,
                    &gradient,
                    1e-7,
                );
                assert!(error < 1e-7, "{shape:?}, friction={friction}: {error}");
            }
        }
    }

    // Four independent nodal blocks exercise the same linear endpoint path
    // used by deformable and affine rigid edges, without unrelated terms.
    fn edge_limit(points: [Vec3; 4], motion: [Vec3; 4], radius: f64) -> f64 {
        edge_limit_with_self(points, motion, radius, false)
    }

    fn edge_limit_with_self(
        points: [Vec3; 4],
        motion: [Vec3; 4],
        radius: f64,
        same_owner: bool,
    ) -> f64 {
        let world = World::new();
        let pool = Pool::new(2);
        let layout = Layout {
            blocks: 4,
            solids: vec![SolidDofs {
                object: 0,
                block: vec![0, 1, 2, 3],
                held_at: vec![Vec3::ZERO; 4],
            }],
            ..Default::default()
        };
        let frame = Frame::default();
        let ctx = Context {
            world: &world,
            pool: &pool,
            layout: &layout,
            frame: &frame,
        };
        let mut contact = Contact {
            precision: 0.01,
            self_contact: same_owner,
            edges: (0..2)
                .map(|i| Edge {
                    owner: if same_owner { 0 } else { i },
                    ends: [0, 1].map(|k| Attach::Node {
                        solid: 0,
                        node: (2 * i + k) as u32,
                    }),
                    radius,
                    mass: 1.0,
                    fixed: false,
                })
                .collect(),
            edge_owners: if same_owner { vec![0, 0] } else { vec![0, 1] },
            ..Default::default()
        };
        let x: Vec<f64> = points.into_iter().flat_map(|p| [p.x, p.y, p.z]).collect();
        let dx: Vec<f64> = motion.into_iter().flat_map(|p| [p.x, p.y, p.z]).collect();
        contact.step_limit(&ctx, &x, &dx)
    }

    #[test]
    fn self_edge_energy_is_smooth_through_parallel_alignment() {
        let mut world = World::new();
        let material = world.add_material(core_world::library::rubber());
        world.add_solid(
            "mesh",
            material,
            core_world::TetMesh::cuboid(Vec3::ONE, 1.0),
            Pose::IDENTITY,
        );
        let pool = Pool::new(1);
        let layout = Layout {
            blocks: 4,
            solids: vec![SolidDofs {
                object: 0,
                block: vec![0, 1, 2, 3],
                held_at: vec![Vec3::ZERO; 4],
            }],
            ..Default::default()
        };
        let frame = Frame {
            h: 1.0,
            weight: 1.0,
            step: 1.0,
            ..Default::default()
        };
        let ctx = Context {
            pool: &pool,
            world: &world,
            layout: &layout,
            frame: &frame,
        };
        let mut contact = Contact {
            precision: 0.01,
            self_contact: true,
            edges: (0..2)
                .map(|i| Edge {
                    owner: 0,
                    ends: [0, 1].map(|k| Attach::Node {
                        solid: 0,
                        node: 2 * i + k,
                    }),
                    radius: 0.0,
                    mass: 1.0,
                    fixed: false,
                })
                .collect(),
            edge_owners: vec![0, 0],
            ..Default::default()
        };
        for tilt in [0.0, 1e-5, 0.01, 0.1] {
            let points = [
                vec3(-0.5, 0.0, 0.0),
                vec3(0.5, 0.0, 0.0),
                vec3(-0.25, 0.004, -tilt * 0.5),
                vec3(0.75, 0.004, tilt * 0.5),
            ];
            let x: Vec<_> = points.into_iter().flat_map(|p| [p.x, p.y, p.z]).collect();
            let mut gradient = vec![0.0; 12];
            let mut matrix = BlockMatrix::default();
            let mut changing = Triplets::new(4);
            contact.derivatives(&ctx, &x, &mut gradient, &mut matrix, &mut changing);
            let error = core_numerics::check::gradient_error(
                |p| contact.energy(&ctx, p),
                &x,
                &gradient,
                // Resolve the narrowing closest-feature region near parallel.
                (tilt * 1e-5_f64).clamp(1e-10, 1e-7),
            );
            if tilt != 0.0 {
                assert!(error < 1e-5, "tilt={tilt}: {error}");
            } else {
                // Relative error is undefined when the true gradient is zero.
                let mut probe = x.clone();
                for k in 0..x.len() {
                    probe[k] += 1e-7;
                    let up = contact.energy(&ctx, &probe);
                    probe[k] -= 2e-7;
                    let down = contact.energy(&ctx, &probe);
                    probe[k] = x[k];
                    assert!(((up - down) / 2e-7).abs() < 1e-10);
                }
            }
            if tilt == 0.0 {
                assert_eq!(contact.energy(&ctx, &x), 0.0);
                assert!(gradient.iter().all(|g| *g == 0.0));
                // Smoothing must not disable geometric collision prevention.
                let dx: Vec<_> = [Vec3::ZERO, Vec3::ZERO, -Vec3::Y * 0.008, -Vec3::Y * 0.008]
                    .into_iter()
                    .flat_map(|p| [p.x, p.y, p.z])
                    .collect();
                let limit = contact.step_limit(&ctx, &x, &dx);
                assert!((0.0..0.5).contains(&limit));
            }
        }
    }

    fn crossed() -> [Vec3; 4] {
        [vec3(0.0, 1.0, -1.0), vec3(0.0, 1.0, 1.0), -Vec3::X, Vec3::X]
    }

    #[test]
    fn swept_edge_bounds_catch_crossing_with_clear_endpoints() {
        let points = crossed();
        let motion = [-Vec3::Y * 2.0, -Vec3::Y * 2.0, Vec3::ZERO, Vec3::ZERO];
        let alpha = edge_limit(points, motion, 0.05);
        assert!((0.35..0.45).contains(&alpha), "{alpha}");
        for k in 0..=100 {
            let t = alpha * k as f64 / 100.0;
            let p = std::array::from_fn::<_, 4, _>(|i| points[i] + motion[i] * t);
            let (a, b, _) = closest_segments(p[0], p[1], p[2], p[3]);
            assert!((a - b).length() > 0.095);
        }
    }

    #[test]
    fn swept_self_edges_stop_before_an_interior_crossing() {
        let points = crossed();
        let motion = [-Vec3::Y * 2.0, -Vec3::Y * 2.0, Vec3::ZERO, Vec3::ZERO];
        let alpha = edge_limit_with_self(points, motion, 0.0, true);
        assert!((0.4..0.5).contains(&alpha), "{alpha}");
        let translation = vec3(1000.0, -800.0, 400.0);
        let shifted = edge_limit_with_self(points, motion.map(|p| p + translation), 0.0, true);
        assert!((alpha - shifted).abs() < 1e-10);
        assert_eq!(
            edge_limit_with_self(points, [translation; 4], 0.0, true),
            1.0
        );
    }

    #[test]
    fn common_translation_does_not_shorten_edge_ccd() {
        let points = crossed();
        let motion = [-Vec3::Y * 2.0, -Vec3::Y * 2.0, Vec3::ZERO, Vec3::ZERO];
        let translation = vec3(1000.0, -800.0, 400.0);
        let alpha = edge_limit(points, motion, 0.05);
        let shifted = edge_limit(points, motion.map(|v| v + translation), 0.05);
        assert!((alpha - shifted).abs() < 1e-10, "{alpha} vs {shifted}");
        assert_eq!(edge_limit(points, [translation; 4], 0.05), 1.0);
    }

    #[test]
    fn deforming_thin_edges_retain_geometric_separation() {
        let points = crossed();
        let motion = [
            vec3(0.3, -2.0, 0.0),
            vec3(-0.3, -2.0, 0.0),
            Vec3::ZERO,
            Vec3::ZERO,
        ];
        let alpha = edge_limit(points, motion, 0.0);
        assert!((0.35..0.5).contains(&alpha), "{alpha}");
        for k in 0..=100 {
            let t = alpha * k as f64 / 100.0;
            let p = std::array::from_fn::<_, 4, _>(|i| points[i] + motion[i] * t);
            let (a, b, _) = closest_segments(p[0], p[1], p[2], p[3]);
            assert!((a - b).length() >= 0.1);
        }
    }

    #[test]
    fn an_initially_opening_gap_can_close_against_a_changing_rigid_matrix() {
        let mut world = World::new();
        let material = world.add_material(core_world::library::oak());
        let p = vec3(1.00001, 0.002, 0.0);
        world.add_fixed(
            "point",
            material,
            Shape::Sphere { radius: 1e-6 },
            Pose::at(p),
        );
        let shape = Shape::cuboid(Vec3::splat(2.0));
        world.add_rigid("box", material, shape, Pose::IDENTITY);
        let layout = Layout::new(&world);
        let frame = Frame {
            rigids: vec![RigidFrame {
                mass: 1.0,
                inertia: Mat3::IDENTITY,
                moment: Mat3::IDENTITY,
                radius: 1.0,
                centre_start: Vec3::ZERO,
                rotation_start: Quat::IDENTITY,
                matrix_start: Mat3::IDENTITY,
                predicted: Quat::IDENTITY,
                world_inertia: Mat3::IDENTITY,
                shape,
                points: contact_points(&shape),
                reach: 3f64.sqrt(),
            }],
            ..Default::default()
        };
        let pool = Pool::new(1);
        let ctx = Context {
            world: &world,
            pool: &pool,
            layout: &layout,
            frame: &frame,
        };
        let mut contact = Contact::new(&ctx, 1e-5);
        let mut x = vec![0.0; 12];
        for (k, axis) in [Vec3::X, Vec3::Y, Vec3::Z].into_iter().enumerate() {
            axis.write(&mut x, k + 1);
        }
        let mut dx = vec![0.0; 12];
        (Vec3::Y * 0.01).write(&mut dx, 1);
        (-Vec3::X * 0.01).write(&mut dx, 2);
        let other = Other::Body(Body::Rigid(0));
        let initial = contact.gap(&ctx, 0, other, &x[..], p).s;
        // A positive initial slope made the old convex-distance shortcut
        // accept the full step, although that step ends inside the box.
        assert!(
            contact
                .gap(
                    &ctx,
                    0,
                    other,
                    Along {
                        x: &x,
                        dx: &dx,
                        t: 1e-4
                    },
                    p
                )
                .s
                > initial
        );
        assert!(
            contact
                .gap(
                    &ctx,
                    0,
                    other,
                    Along {
                        x: &x,
                        dx: &dx,
                        t: 1.0
                    },
                    p
                )
                .s
                < 0.0
        );
        let alpha = contact.step_limit(&ctx, &x, &dx);
        assert!((0.2..0.6).contains(&alpha), "{alpha}");
        assert!(
            contact
                .gap(
                    &ctx,
                    0,
                    other,
                    Along {
                        x: &x,
                        dx: &dx,
                        t: alpha
                    },
                    p
                )
                .s
                > 0.0
        );
        let mut rng = core_math::Rng::new(840);
        let mut certified = 0;
        for _ in 0..200 {
            for k in 0..4 {
                (rng.direction() * rng.range(0.0, 1e-3)).write(&mut dx, k);
            }
            if contact.separating_plane_safe(&ctx, 0, 0, &x, &dx, 1.0) {
                certified += 1;
                for k in 0..=100 {
                    let gap = contact
                        .gap(
                            &ctx,
                            0,
                            other,
                            Along {
                                x: &x,
                                dx: &dx,
                                t: k as f64 / 100.0,
                            },
                            p,
                        )
                        .s;
                    assert!(gap > 0.1 * initial, "certified an unsafe path: {gap}");
                }
            }
        }
        assert!(certified > 20);
    }
}
