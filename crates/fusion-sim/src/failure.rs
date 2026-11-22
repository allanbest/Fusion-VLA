//! What a rigid body does when it is loaded past its strength.
//!
//! The stress solve ([`crate::stress`]) says how hard every part of a body is
//! being worked. A body whose [`Failure`] allows it acts on that, once per
//! step, after the stress is known:
//!
//! - **[`Failure::Breaks`]** — the body cracks where its load-to-strength ratio
//!   first passes one, across the direction it was being pulled apart in, and
//!   becomes rigid pieces. A beam bent past its strength snaps at the face in
//!   tension; a block crushed past its compressive strength splits along the
//!   load, the way a concrete cylinder in a test rig does. The further past
//!   its strength it was, the more pieces: two for a crack, up to eight for a
//!   shattering.
//! - **[`Failure::Deforms`]** — the body stops being rigid the moment it yields
//!   (ductile), cracks (brittle) or reaches its melting point, and carries on
//!   as a continuum solid: the material-point method, which can bend, dent,
//!   tear and melt.
//!
//! ## What a break conserves
//!
//! The pieces are the Voronoi cells of a few seeds, cut from the body's
//! convex shape, so their volumes add up to the body's to the precision of a
//! convex hull; mass is shared at the body's own density, so it adds up to the
//! body's mass. Each piece moves with the velocity the rigid body had at its
//! centre of mass, which conserves linear momentum exactly and angular
//! momentum to the extent the pieces' inertias add up to the whole's — exactly
//! for a box, and to the faceting of a sphere's polyhedron otherwise. Each
//! piece carries the temperature of the material it was cut from.
//!
//! The energy a real crack absorbs (the fracture energy, a few joules a
//! square metre for glass) is not taken out of the motion.
//!
//! ## What deforming costs, and what it is not
//!
//! A continuum is explicit, so its timestep is set by how fast waves cross it,
//! and real steel's 5 km/s is out of reach — the same limit the continuum's own
//! material library documents. A body that deforms is therefore handed to the
//! continuum as an *analogue*: the same density, heat and strength ratios, at
//! the stiffest modulus that runs in about a dozen substeps. It is a softer
//! material than the one that yielded, and it behaves like one; see
//! [`continuum_analogue`].

use fusion_collide::CollisionPipeline;
use fusion_core::solid::{Constitutive, Fracture, SolidDesc, SolidMaterial, SolidParams};
use fusion_core::{
    Assessment, BodyHeat, BodySet, BodyStress, BodyType, ConvexHull, Failure, Material, Parting,
    Pattern, Shape, World,
};
use fusion_math::{Isometry, Quat, Vec3};
use fusion_solver::XpbdWorkingSet;

/// What happened to a body that failed.
#[derive(Clone, Debug, PartialEq)]
pub enum Outcome {
    /// It broke. The body kept its index as the largest piece, and the rest
    /// were added as new bodies at these indices.
    Broke {
        /// Indices of the pieces other than the body itself.
        pieces: Vec<usize>,
    },
    /// It became a continuum solid, at this index in `World::solids`, and the
    /// rigid body was removed.
    Deformed {
        /// Index of the new solid.
        solid: usize,
        /// Whether it was heat rather than load: it reached its melting point.
        melted: bool,
    },
}

/// A failure, as it happened.
#[derive(Clone, Debug, PartialEq)]
pub struct FailureEvent {
    /// Solver step it happened on.
    pub step: u64,
    /// The body that failed.
    pub body: usize,
    /// Its load-to-strength ratio when it did.
    pub utilisation: f32,
    /// How it parted: cracked, sheared, split along its grain or snapped
    /// across it.
    pub parting: fusion_core::Parting,
    /// The material's pattern: how the pieces were cut.
    pub pattern: Pattern,
    /// The largest von Mises stress in it as it went, Pa.
    pub peak_stress: f32,
    /// What became of it.
    pub outcome: Outcome,
}

/// Most bodies that may fail in one step. A cascade — a stack of glass
/// shattering — spreads over a few steps rather than all at once.
const MAX_FAILURES_PER_STEP: usize = 8;

/// Most pieces one break makes.
const MAX_PIECES: usize = 12;

/// Most shards glass shatters into at once.
const MAX_SHARDS: usize = 14;

/// Smallest shard, as a fraction of the volume of the scene body it came from.
const SMALLEST_SHARD: f32 = 1.0 / 96.0;

/// How long a piece is left to settle after the break that made it, before it
/// can break again, seconds. A body broken by a landing is still landing for
/// a few steps, and judged afresh each one its pieces broke again and again
/// from the one blow — a concrete block dropped 30 cm crumbled into seven.
/// The energy that would have done it went into the first crack.
const SETTLE_SECONDS: f32 = 0.10;

/// Below this fraction of a stone body overloaded, it loses a chip; above
/// it, it breaks into chunks.
const CHIP_SPREAD: f32 = 0.2;

/// Smallest piece a break may make, as a fraction of the volume of the scene
/// body it all came from. Below it, a body is gravel and stops breaking.
const SMALLEST_PIECE: f32 = 1.0 / 32.0;

/// Ceiling on bodies in the world, broken pieces included.
const MAX_BODIES: usize = 4096;

/// A body that may deform, as it was before the rigid solve of the step.
/// See [`step`].
#[derive(Clone, Copy, Debug)]
pub struct Before {
    /// Which body.
    pub body: usize,
    /// Where its centre was.
    pub position: fusion_math::DVec3,
    /// Which way it faced.
    pub orientation: fusion_math::Quat,
    /// How fast it was going.
    pub linear_velocity: Vec3,
    /// How fast it was turning.
    pub angular_velocity: Vec3,
}

impl Before {
    /// Every body in `world` that becomes a continuum when it yields, as it
    /// stands now.
    pub fn capture(world: &World, out: &mut Vec<Before>) {
        out.clear();
        let bodies = &world.bodies;
        out.extend(
            (0..bodies.len())
                .filter(|&i| {
                    bodies.failure[i] == Failure::Deforms
                        && bodies.advanced[i]
                        && !bodies.is_removed(i)
                        && bodies.body_type[i] == BodyType::Dynamic
                })
                .map(|i| Before {
                    body: i,
                    position: bodies.position[i],
                    orientation: bodies.orientation[i],
                    linear_velocity: bodies.linear_velocity[i],
                    angular_velocity: bodies.angular_velocity[i],
                }),
        );
    }
}

/// Act on every body that has been loaded past its strength.
///
/// `before` holds the deformable ones as they were before this step's rigid
/// solve, and `reports` what each contact did in it. A body that yields is
/// made a continuum from where it was and how it was moving *before* the
/// blow, and what its contacts pushed on everything else this step is taken
/// back: the continuum is handed the impact to deform under, rather than a
/// ball the rigid solve had already bounced off the thing it yielded on.
/// It used to be handed the bounce — a steel ball that yielded landing at
/// 30 m/s came off whole and undented, and went back up six metres.
#[allow(clippy::too_many_arguments)]
pub fn step(
    world: &mut World,
    pipeline: &mut CollisionPipeline,
    working: &mut XpbdWorkingSet,
    step_index: u64,
    dt: f32,
    before: &[Before],
    reports: &[fusion_solver::ContactReport],
    events: &mut Vec<FailureEvent>,
) {
    let n = world.bodies.len();
    let mut failures = 0;
    // A piece is left to settle for a moment after the break that made it.
    for stress in &mut world.bodies.stress {
        stress.settling = stress.settling.saturating_sub(1);
    }
    let settle = (SETTLE_SECONDS / dt.max(1e-6)).ceil() as u32;
    for i in 0..n {
        if failures >= MAX_FAILURES_PER_STEP {
            break;
        }
        let bodies = &world.bodies;
        let failure = bodies.failure[i];
        if !failure.can_fail()
            || !bodies.advanced[i]
            || bodies.is_removed(i)
            || bodies.body_type[i] != BodyType::Dynamic
        {
            continue;
        }
        let material = bodies.material[i];
        let stress = &bodies.stress[i];
        let heat = &bodies.heat[i];
        match failure {
            Failure::Never => {}
            Failure::Breaks => {
                if !stress.is_built() || stress.fracture < 1.0 || stress.settling > 0 {
                    continue;
                }
                let ratio = stress.fracture;
                let peak_stress = stress.peak_von_mises;
                let assessment = Assessment {
                    utilisation: stress.utilisation,
                    fracture: ratio,
                    normal: stress.worst_normal,
                    mode: stress.worst_parting,
                };
                let pattern = material.strength.pattern;
                if let Some(pieces) = fracture(world, i, assessment, step_index, settle) {
                    for &k in std::iter::once(&i).chain(&pieces) {
                        pipeline.forget_body(k as u32);
                        working.forget_body(k as u32);
                    }
                    events.push(FailureEvent {
                        step: step_index,
                        body: i,
                        utilisation: ratio,
                        parting: assessment.mode,
                        pattern,
                        peak_stress,
                        outcome: Outcome::Broke { pieces },
                    });
                    failures += 1;
                }
            }
            Failure::Deforms => {
                let melted = material.thermal.melts()
                    && heat.grid.is_built()
                    && heat.range().1 >= material.thermal.melt_point as f64;
                let overloaded = stress.is_built() && stress.utilisation >= 1.0;
                if !(melted || overloaded) {
                    continue;
                }
                let utilisation = stress.utilisation;
                let parting = stress.worst_parting;
                let peak_stress = stress.peak_von_mises;
                if let Some(was) = before.iter().find(|b| b.body == i) {
                    undo_blow(world, i, was, reports);
                }
                if let Some(solid) = promote(world, i, dt) {
                    pipeline.forget_body(i as u32);
                    working.forget_body(i as u32);
                    events.push(FailureEvent {
                        step: step_index,
                        body: i,
                        utilisation,
                        parting,
                        pattern: material.strength.pattern,
                        peak_stress,
                        outcome: Outcome::Deformed { solid, melted },
                    });
                    failures += 1;
                }
            }
        }
    }
}

// ---------------------------------------------------------------------------
// Breaking
// ---------------------------------------------------------------------------

/// A body's shape as a convex polyhedron, in its own frame.
///
/// A sphere becomes a 130-vertex polyhedron scaled out to the sphere's own
/// volume, so the pieces weigh what the ball did.
fn polyhedron(shape: &Shape) -> Option<ConvexHull> {
    match shape {
        Shape::Hull(hull) => Some((**hull).clone()),
        Shape::Cuboid { half_extents: h } => {
            let corners: Vec<Vec3> = (0..8)
                .map(|k| {
                    Vec3::new(
                        if k & 1 == 0 { -h.x } else { h.x },
                        if k & 2 == 0 { -h.y } else { h.y },
                        if k & 4 == 0 { -h.z } else { h.z },
                    )
                })
                .collect();
            ConvexHull::from_points(&corners)
        }
        Shape::Sphere { radius } => {
            let (rings, around) = (8, 16);
            let mut points = vec![Vec3::Y * *radius, -Vec3::Y * *radius];
            for r in 1..rings {
                let polar = std::f32::consts::PI * r as f32 / rings as f32;
                for a in 0..around {
                    let azimuth =
                        std::f32::consts::TAU * (a as f32 + 0.5 * (r % 2) as f32) / around as f32;
                    points.push(
                        Vec3::new(
                            polar.sin() * azimuth.cos(),
                            polar.cos(),
                            polar.sin() * azimuth.sin(),
                        ) * *radius,
                    );
                }
            }
            let hull = ConvexHull::from_points(&points)?;
            let scale = (shape.volume() / hull.volume()).cbrt();
            let points: Vec<Vec3> = hull.vertices().iter().map(|p| *p * scale).collect();
            ConvexHull::from_points(&points)
        }
    }
}

/// The part of `hull` on the side of the plane `n · x <= offset`.
fn clip(hull: &ConvexHull, n: Vec3, offset: f32) -> Option<ConvexHull> {
    let vertices = hull.vertices();
    let d: Vec<f32> = vertices.iter().map(|p| p.dot(n) - offset).collect();
    if d.iter().all(|x| *x <= 1e-6) {
        return Some(hull.clone());
    }
    if d.iter().all(|x| *x >= -1e-6) {
        return None;
    }
    let mut points: Vec<Vec3> = vertices
        .iter()
        .zip(&d)
        .filter(|(_, d)| **d <= 0.0)
        .map(|(p, _)| *p)
        .collect();
    for face in hull.faces() {
        for (x, y) in [(0, 1), (1, 2), (2, 0)] {
            let (a, b) = (face.indices[x] as usize, face.indices[y] as usize);
            if (d[a] < 0.0) != (d[b] < 0.0) {
                let t = d[a] / (d[a] - d[b]);
                points.push(vertices[a] + (vertices[b] - vertices[a]) * t);
            }
        }
    }
    ConvexHull::from_points(&points)
}

/// The Voronoi cells of `seeds`, cut from `hull`. `None` where any cell came
/// out empty or degenerate.
fn voronoi(hull: &ConvexHull, seeds: &[Vec3]) -> Option<Vec<ConvexHull>> {
    let mut pieces = Vec::with_capacity(seeds.len());
    for (i, &s) in seeds.iter().enumerate() {
        let mut piece = hull.clone();
        for (j, &t) in seeds.iter().enumerate() {
            if i == j {
                continue;
            }
            let n = (t - s).normalize_or_zero();
            if n == Vec3::ZERO {
                return None;
            }
            piece = clip(&piece, n, n.dot((s + t) * 0.5))?;
        }
        pieces.push(piece);
    }
    Some(pieces)
}

/// A rotation whose columns are an orthonormal, right-handed basis.
fn quat_from_basis(x: Vec3, y: Vec3, z: Vec3) -> Quat {
    let (m00, m11, m22) = (x.x, y.y, z.z);
    let trace = m00 + m11 + m22;
    let q = if trace > 0.0 {
        let s = (trace + 1.0).sqrt() * 2.0;
        Quat::from_xyzw((y.z - z.y) / s, (z.x - x.z) / s, (x.y - y.x) / s, 0.25 * s)
    } else if m00 > m11 && m00 > m22 {
        let s = (1.0 + m00 - m11 - m22).sqrt() * 2.0;
        Quat::from_xyzw(0.25 * s, (y.x + x.y) / s, (z.x + x.z) / s, (y.z - z.y) / s)
    } else if m11 > m22 {
        let s = (1.0 + m11 - m00 - m22).sqrt() * 2.0;
        Quat::from_xyzw((y.x + x.y) / s, 0.25 * s, (z.y + y.z) / s, (z.x - x.z) / s)
    } else {
        let s = (1.0 + m22 - m00 - m11).sqrt() * 2.0;
        Quat::from_xyzw((z.x + x.z) / s, (z.y + y.z) / s, 0.25 * s, (x.y - y.x) / s)
    };
    q.normalize()
}

/// A piece, ready to be a body.
struct Piece {
    /// Its shape in its own principal frame, centred on its centre of mass.
    shape: Shape,
    /// Its centre of mass, in the parent's frame.
    centre: Vec3,
    /// Its principal frame, in the parent's frame.
    rotation: Quat,
    mass: f32,
    inertia: Vec3,
    volume: f32,
}

fn piece(hull: &ConvexHull, density: f32) -> Option<Piece> {
    let volume = hull.volume();
    let mass = density * volume;
    if !(volume > 0.0 && mass > 0.0) {
        return None;
    }
    let centre = hull.centroid();
    let (moments, axes) = fusion_core::stress::principal(hull.inertia_tensor(mass));
    let (x, y) = (axes[0].normalize_or_zero(), axes[1].normalize_or_zero());
    let z = x.cross(y);
    let local: Vec<Vec3> = hull
        .vertices()
        .iter()
        .map(|p| {
            let d = *p - centre;
            Vec3::new(d.dot(x), d.dot(y), d.dot(z))
        })
        .collect();
    let shape = Shape::hull(&local)?;
    Some(Piece {
        shape,
        centre,
        rotation: quat_from_basis(x, y, z),
        mass,
        inertia: Vec3::new(moments[0], moments[1], moments[2]).max(Vec3::splat(1e-12)),
        volume,
    })
}

/// Break body `i` the way `how` says, through its most loaded point. Returns
/// the indices of the new pieces, or `None` where the body is too small to
/// break or the cut came out degenerate.
fn fracture(
    world: &mut World,
    i: usize,
    how: Assessment,
    step_index: u64,
    settle: u32,
) -> Option<Vec<usize>> {
    let normal = how.normal;
    let bodies = &world.bodies;
    let source = bodies.source[i] as usize;
    let source_volume = world
        .scene()
        .get(source)
        .map(|desc| desc.shape.volume())
        .unwrap_or_else(|| bodies.shape[i].volume());
    let material = bodies.material[i];
    let grain = bodies.grain[i];
    // Splinters need fibres; without a grain the material cracks.
    let pattern = match material.strength.pattern {
        Pattern::Splinter if material.strength.grain.is_none() => Pattern::Crack,
        p => p,
    };
    // Glass leaves shards a fraction of the size anything else would.
    let smallest = source_volume
        * if pattern == Pattern::Shatter {
            SMALLEST_SHARD
        } else {
            SMALLEST_PIECE
        };
    let shape = bodies.shape[i].clone();
    let volume = shape.volume();
    if volume < 2.0 * smallest {
        return None;
    }
    let hull = polyhedron(&shape)?;
    let density = bodies.mass(i) / hull.volume();
    let stress = &bodies.stress[i];
    let origin = stress.worst_point;
    let gap = 0.25 * stress.grid.size.min_element().max(1e-4);

    // How many pieces. More the more of it is overloaded: a local overload —
    // a corner that took a landing — makes one crack, and an impact that
    // overloads most of the body breaks it up. Counted by how much rather
    // than how badly: by the worst ratio, a concrete block dropped 30 cm on
    // its corner, overloaded in one cell three and a half times over, became
    // four pieces. Glass is the exception: a crack in it branches as it runs,
    // and the harder the blow the more it branches, so how badly counts too.
    //
    // And how badly counts as well, for everything. A 25 t ball landing on
    // a wooden cube overloads the top of it eight times over and little of
    // the rest — it was split in two, like a wedge would split it, where a
    // blow with a hundred times a wedge's energy behind it breaks it up. The
    // logarithm, because energy is what a fracture spends, and each crack
    // through the body takes about the same whatever drove it.
    let room = MAX_BODIES.saturating_sub(bodies.len()) + 1;
    let overloaded = stress.overloaded;
    let severity = (how.fracture.max(1.0).ln() * 3.0) as usize;
    let wanted = match pattern {
        Pattern::Crack => 2 + (overloaded * 6.0) as usize + severity,
        Pattern::Tear => 2,
        Pattern::Chip => 2 + (overloaded * 4.0) as usize + severity,
        Pattern::Splinter => 2 + (overloaded * 2.0 * MAX_PIECES as f32) as usize + severity,
        Pattern::Shatter => {
            4 + ((how.fracture - 1.0).max(0.0) * 3.0) as usize
                + (overloaded * 3.0 * MAX_SHARDS as f32) as usize
        }
    };
    let most = match pattern {
        Pattern::Crack => 8,
        Pattern::Tear => 2,
        Pattern::Chip => 6,
        Pattern::Splinter => MAX_PIECES,
        Pattern::Shatter => MAX_SHARDS,
    };
    let wanted = wanted
        .clamp(2, most)
        .min((volume / smallest) as usize)
        .min(room);
    if wanted < 2 {
        return None;
    }
    let mut rng = world.rng(0xB2EA_4000 ^ ((i as u64) << 24) ^ step_index);
    let aabb = shape.local_aabb();
    if pattern == Pattern::Splinter && how.mode == Parting::Snap {
        let mut pieces = snap(&hull, origin, grain, wanted, smallest, &mut rng)?
            .iter()
            .map(|c| piece(c, density))
            .collect::<Option<Vec<Piece>>>()?;
        pieces.sort_by(|a, b| b.volume.total_cmp(&a.volume));
        return Some(place(world, i, &pieces, smallest, settle));
    }
    // Stone loaded at one place loses a chip there, and only goes through
    // when much of it is overloaded.
    if pattern == Pattern::Chip && overloaded < CHIP_SPREAD {
        if let Some(cells) = chip(&hull, origin, normal, gap, smallest, &mut rng) {
            let pieces: Option<Vec<Piece>> = cells.iter().map(|c| piece(c, density)).collect();
            if let Some(mut pieces) = pieces {
                pieces.sort_by(|a, b| b.volume.total_cmp(&a.volume));
                return Some(place(world, i, &pieces, smallest, settle));
            }
        }
    }
    let reach = 0.6 * (aabb.max - aabb.min).max_element();
    // A body far thinner one way than the others — a pane, a slab — and
    // which way that is.
    let extent = aabb.max - aabb.min;
    let thin = (extent.min_element() < 0.25 * extent.max_element()).then_some(
        if extent.x <= extent.y && extent.x <= extent.z {
            Vec3::X
        } else if extent.y <= extent.z {
            Vec3::Y
        } else {
            Vec3::Z
        },
    );
    // Seeds kept apart, so the pieces come out comparable. Thrown at
    // random, two would land close, the cell between them would be a
    // sliver under the smallest piece, and the search below would give up
    // seed after seed until it reached two: a 25 t ball on a wooden cube
    // split it in half. Spaced at a fraction of what each piece would get
    // of the body evenly shared — of its cross-section, where every seed is
    // on one sheet.
    let flat = (pattern == Pattern::Splinter && how.mode == Parting::Split)
        .then_some(grain)
        .or(thin);
    let apart = 0.55
        * match flat {
            Some(axis) => {
                let depth = extent.dot(axis.abs()).max(1e-4);
                (volume / depth / wanted as f32).sqrt()
            }
            None => (volume / wanted as f32).cbrt(),
        };
    let firsts = [origin - normal * gap, origin + normal * gap];
    let mut extra: Vec<Vec3> = Vec::new();
    for _ in 0..(wanted - 2) * 32 {
        if extra.len() + 2 >= wanted {
            break;
        }
        let p = if pattern == Pattern::Shatter {
            // Shards smallest where it was struck: seeds packed around the
            // point it broke at, thinning out with the square of the
            // distance.
            let d = Vec3::new(
                rng.range(-1.0, 1.0),
                rng.range(-1.0, 1.0),
                rng.range(-1.0, 1.0),
            );
            let r = rng.range(0.0, 1.0);
            origin + d.normalize_or_zero() * reach * r.powf(1.5)
        } else {
            Vec3::new(
                rng.range(aabb.min.x, aabb.max.x),
                rng.range(aabb.min.y, aabb.max.y),
                rng.range(aabb.min.z, aabb.max.z),
            )
        };
        // A split runs the length of the grain: every seed on the one
        // cross-section, so every cut between two of them runs along it and
        // the pieces are slats, the way a log splits.
        let p = if pattern == Pattern::Splinter && how.mode == Parting::Split {
            p - grain * grain.dot(p - origin)
        } else if let Some(across) = thin {
            // A pane breaks through its thickness, never into layers: every
            // seed on the one sheet, so every cut goes straight through.
            p - across * across.dot(p - origin)
        } else {
            p
        };
        let crowded = pattern != Pattern::Shatter
            && firsts
                .iter()
                .chain(&extra)
                .any(|q| q.distance_squared(p) < apart * apart);
        if shape.signed_distance_local(p) < 0.0 && !crowded {
            extra.push(p);
        }
    }

    // Fewer seeds until every piece is big enough to be one.
    let mut pieces = None;
    for count in (2..=2 + extra.len()).rev() {
        let mut seeds = vec![origin - normal * gap, origin + normal * gap];
        seeds.extend_from_slice(&extra[..count - 2]);
        let Some(cells) = voronoi(&hull, &seeds) else {
            continue;
        };
        // A shard may be smaller than a piece: it is gravel at once.
        let floor = if pattern == Pattern::Shatter {
            0.25 * smallest
        } else {
            smallest
        };
        if cells.iter().any(|c| c.volume() < floor) {
            continue;
        }
        let built: Option<Vec<Piece>> = cells.iter().map(|c| piece(c, density)).collect();
        if let Some(built) = built {
            pieces = Some(built);
            break;
        }
    }
    let mut pieces = pieces?;
    pieces.sort_by(|a, b| b.volume.total_cmp(&a.volume));
    Some(place(world, i, &pieces, smallest, settle))
}

/// A chip off a body, where it was loaded.
///
/// Stone and concrete are grains in a matrix, and a crack in them wanders
/// between the grains and stops; what a blow on a corner takes off is that
/// corner. The chip is a cap cut by a plane facing out from the body's centre
/// towards the point it broke at, tilted a little at random, and placed to
/// take between a twentieth and a seventh of the body — never more than a
/// third, and always the point that broke.
fn chip(
    hull: &ConvexHull,
    origin: Vec3,
    normal: Vec3,
    gap: f32,
    smallest: f32,
    rng: &mut fusion_core::Rng,
) -> Option<Vec<ConvexHull>> {
    let centre = hull.centroid();
    let out = (origin - centre).normalize_or_zero();
    let out = if out == Vec3::ZERO { normal } else { out };
    let wobble = Vec3::new(
        rng.range(-1.0, 1.0),
        rng.range(-1.0, 1.0),
        rng.range(-1.0, 1.0),
    ) * 0.35;
    let out = (out + wobble).normalize_or_zero();
    if out == Vec3::ZERO {
        return None;
    }
    let volume = hull.volume();
    let cap = |d: f32| clip(hull, -out, -d).map(|c| c.volume()).unwrap_or(0.0);
    let far = hull
        .vertices()
        .iter()
        .map(|v| v.dot(out))
        .fold(f32::MIN, f32::max);
    // The plane taking `fraction` of the body, by bisection: the cap shrinks
    // as the plane moves out.
    let plane_for = |fraction: f32| {
        let (mut lo, mut hi) = (out.dot(centre), far);
        for _ in 0..24 {
            let mid = 0.5 * (lo + hi);
            if cap(mid) > fraction * volume {
                lo = mid;
            } else {
                hi = mid;
            }
        }
        0.5 * (lo + hi)
    };
    let wanted = plane_for(rng.range(0.05, 0.14));
    let largest = plane_for(0.33);
    let d = wanted.min(out.dot(origin) - gap).max(largest);
    let main = clip(hull, out, d)?;
    let chip = clip(hull, -out, -d)?;
    if chip.volume() < 0.25 * smallest || main.volume() < smallest {
        return None;
    }
    Some(vec![main, chip])
}

/// A body with a grain, snapped across it.
///
/// Wood bent past its strength does not crack straight across like glass:
/// the fibres on the stretched face part one at a time, at different
/// lengths, and the break runs off along the grain before it gets through.
/// What is left are two halves with long ragged points and a few splinters.
///
/// Pieces here are convex, so the ragged end is a slant: a cut tilted so it
/// runs along the grain about twice as far as it goes through the body,
/// across its thinnest side, the way a plank laid flat and bent breaks. When
/// the break wants more than two pieces, the slanted points come off as
/// splinters, the first pair whole and a third and more by splitting those
/// along the grain.
fn snap(
    hull: &ConvexHull,
    origin: Vec3,
    grain: Vec3,
    wanted: usize,
    smallest: f32,
    rng: &mut fusion_core::Rng,
) -> Option<Vec<ConvexHull>> {
    let g = grain.normalize_or_zero();
    if g == Vec3::ZERO {
        return None;
    }
    // Through the thinnest side across the grain.
    let extent = |d: Vec3| {
        let (lo, hi) = hull
            .vertices()
            .iter()
            .fold((f32::MAX, f32::MIN), |(lo, hi), p| {
                (lo.min(p.dot(d)), hi.max(p.dot(d)))
            });
        hi - lo
    };
    let e1 = g.any_orthonormal();
    let e2 = g.cross(e1);
    let (mut thin, mut wide) = (e1, e2);
    let mut best = f32::MAX;
    for k in 0..12 {
        let a = std::f32::consts::PI * k as f32 / 12.0;
        let d = e1 * a.cos() + e2 * a.sin();
        let x = extent(d);
        if x < best {
            best = x;
            thin = d;
            wide = g.cross(d);
        }
    }
    let slant = if rng.range(0.0, 1.0) < 0.5 { -1.0 } else { 1.0 } * rng.range(0.85, 1.35);
    let m = (g + thin * slant).normalize();
    let c = m.dot(origin);
    let behind = clip(hull, m, c)?;
    let ahead = clip(hull, -m, -c)?;
    // Where the slanted face starts and ends along the grain.
    let size = extent(g).max(extent(thin)).max(1e-6);
    let on_face: Vec<f32> = behind
        .vertices()
        .iter()
        .filter(|p| (p.dot(m) - c).abs() < 1e-4 * size)
        .map(|p| p.dot(g))
        .collect();
    if on_face.len() < 3 {
        return Some(vec![behind, ahead]);
    }
    let lo = on_face.iter().copied().fold(f32::MAX, f32::min);
    let hi = on_face.iter().copied().fold(f32::MIN, f32::max);
    // Splinters are allowed smaller than a piece: they are gravel at once.
    let sliver = smallest * 0.125;
    let mut out = Vec::new();
    let mut tips = Vec::new();
    for (half, body_side, tip_side) in [(&behind, (g, lo), (-g, -lo)), (&ahead, (-g, -hi), (g, hi))]
    {
        if 2 + tips.len() >= wanted {
            out.push(half.clone());
            continue;
        }
        let main = clip(half, body_side.0, body_side.1);
        let tip = clip(half, tip_side.0, tip_side.1);
        match (main, tip) {
            (Some(main), Some(tip)) if tip.volume() >= sliver && main.volume() >= smallest => {
                out.push(main);
                tips.push(tip);
            }
            _ => out.push(half.clone()),
        }
    }
    // More still: each point split down its length.
    let mut spare = wanted.saturating_sub(out.len() + tips.len());
    for tip in tips {
        if spare > 0 {
            let w = wide.dot(tip.centroid());
            if let (Some(a), Some(b)) = (clip(&tip, wide, w), clip(&tip, -wide, -w)) {
                if a.volume() >= sliver && b.volume() >= sliver {
                    out.push(a);
                    out.push(b);
                    spare -= 1;
                    continue;
                }
            }
        }
        out.push(tip);
    }
    Some(out)
}

/// Put `pieces` (largest first) in the world in place of body `i`: the
/// largest keeps the body's index, so whatever refers to the body follows the
/// bulk of it. Returns the indices of the rest.
fn place(world: &mut World, i: usize, pieces: &[Piece], smallest: f32, settle: u32) -> Vec<usize> {
    let bodies = &mut world.bodies;
    let pose = bodies.pose(i);
    let velocity = bodies.linear_velocity[i];
    let spin = bodies.angular_velocity[i];
    let parent_heat = bodies.heat[i].clone();
    let material = bodies.material[i];
    let failure = bodies.failure[i];
    let body_source = bodies.source[i];
    let grain = bodies.grain[i];

    let mut placed = Vec::with_capacity(pieces.len());
    for (k, p) in pieces.iter().enumerate() {
        let position = pose.transform_point(p.centre);
        let orientation = (pose.rotation * p.rotation).normalize();
        let v = velocity + spin.cross(pose.rotation.rotate(p.centre));
        let inv_inertia = Vec3::new(1.0 / p.inertia.x, 1.0 / p.inertia.y, 1.0 / p.inertia.z);
        let heat = {
            let mut heat = BodyHeat::uniform(parent_heat.mean() as f32);
            heat.uniform = parent_heat.mean();
            if parent_heat.is_built() {
                heat.build(&p.shape);
                for c in 0..heat.temperature.len() {
                    let local = p.centre + p.rotation.rotate(heat.cell_centre(c));
                    heat.temperature[c] = parent_heat.at_local(local);
                }
            }
            heat
        };
        let index = if k == 0 {
            bodies.set_pose(i, Isometry::new(position, orientation));
            bodies.linear_velocity[i] = v;
            bodies.angular_velocity[i] = spin;
            bodies.inv_mass[i] = 1.0 / p.mass;
            bodies.inv_inertia_local[i] = inv_inertia;
            bodies.inertia_local[i] = p.inertia;
            bodies.shape[i] = p.shape.clone();
            bodies.wake(i);
            i
        } else {
            let handle = bodies.push(
                Isometry::new(position, orientation),
                v,
                spin,
                1.0 / p.mass,
                inv_inertia,
                p.shape.clone(),
                material,
                BodyType::Dynamic,
            );
            let j = handle.index();
            bodies.failure[j] = failure;
            bodies.source[j] = body_source;
            placed.push(j);
            j
        };
        bodies.heat[index] = heat;
        bodies.stress[index] = BodyStress {
            settling: settle,
            ..BodyStress::default()
        };
        bodies.reshaped[index] = true;
        // The grain runs where it ran in the whole, in the piece's own frame.
        bodies.grain[index] = p.rotation.inverse().rotate(grain).normalize_or_zero();
        // A piece too small to break again is gravel: it stays whole, and
        // stops paying for a stress solve it could never act on.
        if p.volume < 2.0 * smallest {
            bodies.failure[index] = Failure::Never;
        }
    }

    reattach_joints(world, i, &pose, &placed);
    placed
}

/// Joints on a body that broke follow whichever piece holds their anchor.
fn reattach_joints(world: &mut World, i: usize, old_pose: &Isometry, pieces: &[usize]) {
    if !world
        .joints()
        .iter()
        .any(|j| j.body_a.index() == i || j.body_b.index() == i)
    {
        return;
    }
    let candidates: Vec<usize> = std::iter::once(i).chain(pieces.iter().copied()).collect();
    let bodies = world.bodies.clone();
    for joint in world.joints_storage_mut().iter_mut() {
        for side in 0..2 {
            let (body, anchor, frame) = if side == 0 {
                (
                    &mut joint.body_a,
                    &mut joint.local_anchor_a,
                    &mut joint.local_frame_a,
                )
            } else {
                (
                    &mut joint.body_b,
                    &mut joint.local_anchor_b,
                    &mut joint.local_frame_b,
                )
            };
            if body.index() != i {
                continue;
            }
            let world_anchor = old_pose.transform_point(*anchor);
            let world_frame = old_pose.rotation * *frame;
            let best = candidates
                .iter()
                .copied()
                .min_by(|&a, &b| {
                    let da = bodies.shape[a].signed_distance_local(
                        bodies.pose(a).inverse_transform_point(world_anchor),
                    );
                    let db = bodies.shape[b].signed_distance_local(
                        bodies.pose(b).inverse_transform_point(world_anchor),
                    );
                    da.total_cmp(&db)
                })
                .unwrap_or(i);
            let pose = bodies.pose(best);
            *body = fusion_core::BodyHandle(best as u32);
            *anchor = pose.inverse_transform_point(world_anchor);
            *frame = (pose.rotation.inverse() * world_frame).normalize();
        }
    }
}

// ---------------------------------------------------------------------------
// Deforming
// ---------------------------------------------------------------------------

/// Substeps a deformed body's continuum is sized to need at 240 Hz. Half the
/// continuum's ceiling, so a body that deforms does not by itself make the
/// step report that it ran out of substeps.
const ANALOGUE_SUBSTEPS: f32 = 12.0;

/// The continuum material a rigid one becomes when it deforms.
///
/// Density, Poisson's ratio, the thermal numbers and the *ratios* of strength
/// to stiffness are the rigid material's. Young's modulus is the smaller of the
/// real one and the stiffest the continuum can step at `dt` in
/// `ANALOGUE_SUBSTEPS` substeps with cells of `cell` metres. How it fails is
/// the material's character:
///
/// - **ductile** (a metal) becomes elastoplastic: it springs back up to its
///   yield strain, never less than 0.2% so it can stand under its own weight
///   at the softer modulus, and flows past it;
/// - **brittle** (glass, concrete, ice) becomes an elastic solid that tears at
///   its tensile strain, never less than 0.3%;
/// - anything else (rubber) stays elastic.
///
/// Anything that melts softens towards its melting point, as the continuum's
/// own alloy does.
pub fn continuum_analogue(material: &Material, cell: f32, dt: f32) -> SolidMaterial {
    let strength = material.strength;
    let nu = strength.poissons_ratio.clamp(0.0, 0.45);
    let density = material.density.max(1e-3);
    let fastest = ANALOGUE_SUBSTEPS * 0.4 * cell / dt.max(1e-6);
    let ceiling = fastest * fastest * density * (1.0 + nu) * (1.0 - 2.0 * nu) / (1.0 - nu);
    let youngs = strength.youngs_modulus.min(ceiling).max(1.0);
    let real = strength.youngs_modulus.max(1.0);
    // A metal yields in shear and keeps its volume: von Mises. It was given
    // the snow law, which compacts, and a steel ball that yielded landing at
    // 60 m/s was crushed to a 2 cm sheet no wider than itself.
    //
    // At the material's own yield *stress*, not its yield strain. The
    // analogue is softer than the metal, and the real strain at that
    // softness is a yield stress hundreds of times too low: the ball flowed
    // like water, eight metres across. How much a dent takes out of a blow
    // is set by the stress it yields at, so that is what is kept.
    let (model, fracture) = if strength.is_ductile() {
        let yield_strain = (strength.yield_stress / youngs).clamp(0.002, 0.2);
        (Constitutive::VonMises { yield_strain }, Fracture::TOUGH)
    } else if strength.tensile.is_finite() && strength.youngs_modulus > 1e8 {
        let onset = 1.0 + (strength.tensile / real).max(0.003);
        (
            Constitutive::FixedCorotated,
            Fracture {
                onset,
                failure: onset + 0.02,
            },
        )
    } else {
        (Constitutive::NeoHookean, Fracture::TOUGH)
    };
    let mut thermal = material.thermal;
    if thermal.melts() {
        thermal.softening = thermal.softening.max(0.95);
    }
    SolidMaterial {
        density,
        youngs_modulus: youngs,
        poissons_ratio: nu,
        model,
        thermal,
        fracture,
    }
}

/// Most material points a deformed body becomes.
const MAX_POINTS: f32 = 30_000.0;

/// Put body `i` back as it was before the step's rigid solve, and take back
/// from every other body what its contacts gave them in it.
fn undo_blow(world: &mut World, i: usize, was: &Before, reports: &[fusion_solver::ContactReport]) {
    let bodies = &mut world.bodies;
    for r in reports {
        let (a, b) = (r.body_a as usize, r.body_b as usize);
        // `impulse` is what B received; A received the opposite.
        let (other, given, anchor) = if a == i {
            (b, r.impulse, r.local_b)
        } else if b == i {
            (a, -r.impulse, r.local_a)
        } else {
            continue;
        };
        if bodies.is_dynamic(other) {
            let at = bodies.pose(other).transform_point(anchor);
            bodies.apply_impulse_at(other, -given, at);
        }
    }
    bodies.position[i] = was.position;
    bodies.orientation[i] = was.orientation;
    bodies.linear_velocity[i] = was.linear_velocity;
    bodies.angular_velocity[i] = was.angular_velocity;
}

/// Turn body `i` into a continuum solid, moving and at the temperature it was.
fn promote(world: &mut World, i: usize, dt: f32) -> Option<usize> {
    let bodies = &world.bodies;
    let shape = bodies.shape[i].clone();
    let pose = bodies.pose(i);
    let aabb = shape.local_aabb();
    let longest = (aabb.max - aabb.min).max_element();
    let spacing = (longest / 12.0)
        .max((shape.volume() / MAX_POINTS).cbrt())
        .max(0.005);
    let material = bodies.material[i];
    let mut params = SolidParams::at(spacing);
    params.material = continuum_analogue(&material, params.cell_size, dt);
    // The rigid body's mass, which a mass override may have set apart from its
    // density.
    let mass = bodies.mass(i);
    let heat = bodies.heat[i].clone();
    let mut desc = SolidDesc::new(shape).with_params(params);
    desc.pose = pose;
    desc.temperature = Some(heat.mean() as f32);
    let mut solid = fusion_core::build_solid(&desc);
    if solid.is_empty() {
        return None;
    }
    if mass.is_finite() && mass > 0.0 {
        let points = solid.len() as f32;
        let wanted = mass / (points * solid.params.point_volume());
        if (wanted - solid.params.material.density).abs() > 1e-3 * wanted {
            solid.params.material.density = wanted;
            let inv = 1.0 / solid.params.point_mass();
            for m in &mut solid.particles.inv_mass {
                *m = inv;
            }
        }
    }
    let velocity = bodies.linear_velocity[i];
    let spin = bodies.angular_velocity[i];
    let centre = pose.translation;
    let thermal = solid.params.material.thermal;
    for k in 0..solid.len() {
        let p = solid.particles.translation(k);
        solid.particles.velocity[k] = velocity + spin.cross(p - centre);
        // Sensible heat only: the rigid body never paid any latent heat, so a
        // point above its melting point starts part-way through melting
        // rather than fully molten with energy nobody supplied.
        let t = heat.at_local(pose.inverse_transform_point(p));
        solid.enthalpy[k] = thermal.specific_heat as f64 * t;
    }
    world.solids.push(solid);
    world.bodies.remove(i);
    for joint in world.joints_storage_mut().iter_mut() {
        if joint.body_a.index() == i || joint.body_b.index() == i {
            joint.enabled = false;
        }
    }
    Some(world.solids.len() - 1)
}

/// Whether anything in `bodies` could fail, so a scene where nothing can
/// skips the pass.
pub fn any_can_fail(bodies: &BodySet) -> bool {
    bodies
        .failure
        .iter()
        .zip(&bodies.advanced)
        .any(|(f, &advanced)| advanced && f.can_fail())
}
