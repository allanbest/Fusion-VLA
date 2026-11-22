//! Independent contact components. Plans are speculative; every evaluated
//! configuration is enclosed before accepting a step. New coupling rolls the
//! whole interval back and monotonically merges the plan, never a partial state.
//! Current links are anchors and gravity is uniform. Future joints, fields or
//! other cross-object forces must contribute coupling edges to this planner.
use super::{EnergyLedger, Mechanics, StepReport};
use crate::dofs::{Layout, Owner, HELD};
use crate::term::Context;
use core_math::{Mat3, Vec3};
use core_runtime::Pool;
use core_space::{Aabb, Bvh, Shape};
use core_world::{Link, Representation, World};
use std::collections::BTreeMap;

/// Cached solve storage, not simulation state. History always comes from the
/// parent's checkpointable vectors, including after a merge, split or restore.
pub(super) struct Worker {
    members: Vec<usize>,
    supports: Vec<usize>,
    blocks: Vec<usize>,
    solids: Vec<usize>,
    rigids: Vec<usize>,
    solver: Mechanics,
    world: World,
    indices: Vec<usize>,
    report: StepReport,
    last_h: Option<f64>,
    /// Bitwise stationary: the last solve reproduced its own input.
    asleep: bool,
    /// At rest within [`Settings::settle_time`]: frozen this step.
    ///
    /// [`Settings::settle_time`]: super::Settings::settle_time
    settled: bool,
}

#[derive(Default)]
pub(super) struct Sweep {
    enabled: bool,
    /// Solids, then rigids, in local layout order.
    bounds: Vec<Aabb>,
    margin: f64,
}

/// Covers every rotation as well as affine Newton trial configurations and the
/// final polar projection. ||A||₂² <= 1 + ||AᵀA-I||F; rotations have scale one.
fn radius_bound(shape: Shape, a: Mat3, margin: f64) -> f64 {
    let radius = match shape {
        Shape::Sphere { radius } => radius,
        Shape::Capsule {
            radius,
            half_length,
        } => radius + half_length,
        Shape::Cuboid { half } => half.length(),
        Shape::HalfSpace { .. } => unreachable!("dynamic shapes are bounded"),
    };
    let c = a.transpose() * a - Mat3::IDENTITY;
    let defect = c.frobenius_squared().sqrt();
    (radius + margin) * (1.0 + defect).sqrt()
}

fn sphere(centre: Vec3, radius: f64) -> Aabb {
    if centre.is_finite() && radius.is_finite() {
        Aabb::around(centre, Vec3::splat(radius))
    } else {
        Aabb::EVERYTHING
    }
}

fn include(bounds: &mut Aabb, p: Vec3) {
    *bounds = if p.is_finite() {
        bounds.including(p)
    } else {
        Aabb::EVERYTHING
    };
}

impl Sweep {
    fn reset(&mut self, layout: &Layout, precision: f64) {
        self.margin = 4.0 * precision;
        self.enabled = true;
        self.bounds.clear();
        self.bounds
            .resize(layout.solids.len() + layout.rigids.len(), Aabb::EMPTY);
    }

    /// Called before any energy term, even when an earlier term rejects the
    /// trial. Endpoints enclose each linear Newton/CCD path between them.
    pub(super) fn record(&mut self, ctx: &Context, x: &[f64]) {
        if !self.enabled {
            return;
        }
        for (s, dofs) in ctx.layout.solids.iter().enumerate() {
            for node in 0..dofs.block.len() {
                include(&mut self.bounds[s], ctx.frame.node(ctx.layout, s, node, x));
            }
        }
        for (r, _) in ctx.layout.rigids.iter().enumerate() {
            let (centre, a) = ctx.frame.rigid(ctx.layout, r, x);
            let bounds = sphere(
                centre,
                radius_bound(ctx.frame.rigids[r].shape, a, self.margin),
            );
            let k = ctx.layout.solids.len() + r;
            self.bounds[k] = self.bounds[k].union(&bounds);
        }
    }
}

#[derive(Clone, PartialEq, Eq, PartialOrd, Ord)]
struct Group {
    members: Vec<usize>,
    supports: Vec<usize>,
}

struct Plan {
    dynamic: Vec<usize>,
    bounds: Vec<Aabb>,
    margins: Vec<f64>,
    roots: Vec<usize>,
    fixed: Vec<usize>,
    fixed_bvh: Bvh,
    unbounded: Vec<(usize, Vec3, f64)>,
    linked_supports: Vec<Vec<usize>>,
}

fn aabb_overlaps_halfspace(bounds: &Aabb, normal: Vec3, offset: f64) -> bool {
    let corner = Vec3::new(
        if normal.x >= 0.0 {
            bounds.min.x
        } else {
            bounds.max.x
        },
        if normal.y >= 0.0 {
            bounds.min.y
        } else {
            bounds.max.y
        },
        if normal.z >= 0.0 {
            bounds.min.z
        } else {
            bounds.max.z
        },
    );
    normal.dot(corner) <= offset
}

impl Plan {
    fn new(parent: &Mechanics, world: &World, h: f64) -> Self {
        let mut plan = Self {
            dynamic: Vec::new(),
            bounds: Vec::new(),
            margins: Vec::new(),
            roots: Vec::new(),
            fixed: Vec::new(),
            fixed_bvh: Bvh::default(),
            unbounded: Vec::new(),
            linked_supports: Vec::new(),
        };
        let mut fixed_boxes = Vec::new();
        let gravity = world.gravity.length() * h * h;
        for (index, object) in world.objects().iter().enumerate() {
            let (bounds, size) = match &object.representation {
                Representation::Fixed(f) => {
                    if f.shape.is_bounded() {
                        plan.fixed.push(index);
                        fixed_boxes.push(f.shape.bounds(&f.pose));
                    } else if let Shape::HalfSpace { normal, offset } = f.shape {
                        let world_normal = f.pose.rotation.rotate(normal);
                        let world_offset = offset + world_normal.dot(f.pose.position);
                        plan.unbounded.push((index, world_normal, world_offset));
                    } else {
                        plan.unbounded.push((index, Vec3::ZERO, f64::INFINITY));
                    }
                    continue;
                }
                Representation::Solid(s) => {
                    let mut bounds = Aabb::EMPTY;
                    for (&p, &v) in s.nodes.position.iter().zip(&s.nodes.velocity) {
                        include(&mut bounds, p);
                        include(&mut bounds, p + v * h);
                    }
                    (
                        bounds.grown(gravity),
                        Aabb::from_points(s.mesh.rest.iter().copied()).size(),
                    )
                }
                Representation::Rigid(r) => {
                    let Owner::Rigid(global) = parent.layout.objects[index] else {
                        unreachable!()
                    };
                    let a = parent.affine[global]
                        .filter(|(q, _)| *q == r.pose.rotation)
                        .map_or_else(|| r.pose.rotation.to_mat3(), |(_, a)| a);
                    let size = r.shape.local_bounds().size();
                    let precision = parent
                        .settings
                        .contact_precision
                        .unwrap_or_else(|| 1e-3 * size.length().min(2.0 * size.min_element()));
                    let radius = radius_bound(r.shape, a, 4.0 * precision);
                    let start = sphere(r.pose.position, radius);
                    (
                        start
                            .union(&sphere(r.pose.position + r.velocity * h, radius))
                            .grown(gravity),
                        r.shape.local_bounds().size(),
                    )
                }
            };
            let precision = parent
                .settings
                .contact_precision
                .unwrap_or_else(|| 1e-3 * size.length().min(2.0 * size.min_element()));
            // Contact's analytic broad phase grows by 2ε. Extra padding also
            // encloses point radii and the point/triangle and edge barriers.
            let margin = 4.0 * precision;
            plan.dynamic.push(index);
            plan.bounds.push(bounds.grown(margin));
            plan.margins.push(margin);
        }
        plan.roots = (0..plan.dynamic.len()).collect();
        if parent.settings.mutual_gravity.is_some() && plan.dynamic.len() > 1 {
            for i in 1..plan.dynamic.len() {
                plan.merge(0, i);
            }
        }
        plan.linked_supports
            .resize_with(plan.dynamic.len(), Vec::new);
        plan.fixed_bvh = Bvh::build(&fixed_boxes);
        for link in world.links() {
            let pair = match link {
                Link::Anchor { .. } => continue,
                Link::Spring { a, b, .. } | Link::BallJoint { a, b, .. } => (
                    a.object().map(|o| o.0 as usize),
                    b.object().map(|o| o.0 as usize),
                ),
                Link::Weld { a, b, .. }
                | Link::Hinge { a, b, .. }
                | Link::Slider { a, b, .. }
                | Link::Cylindrical { a, b, .. }
                | Link::Universal { a, b, .. }
                | Link::Planar { a, b, .. } => (Some(a.0 as usize), b.map(|o| o.0 as usize)),
                Link::Tendon { waypoints, .. } => {
                    for win in waypoints.windows(2) {
                        if let (Some(oa), Some(ob)) = (
                            win[0].object().map(|o| o.0 as usize),
                            win[1].object().map(|o| o.0 as usize),
                        ) {
                            match (
                                plan.dynamic.binary_search(&oa),
                                plan.dynamic.binary_search(&ob),
                            ) {
                                (Ok(da), Ok(db)) => plan.merge(da, db),
                                (Ok(da), Err(_)) => plan.linked_supports[da].push(ob),
                                (Err(_), Ok(db)) => plan.linked_supports[db].push(oa),
                                _ => {}
                            }
                        }
                    }
                    continue;
                }
            };
            if let (Some(oa), Some(ob)) = pair {
                match (
                    plan.dynamic.binary_search(&oa),
                    plan.dynamic.binary_search(&ob),
                ) {
                    (Ok(da), Ok(db)) => plan.merge(da, db),
                    (Ok(da), Err(_)) => plan.linked_supports[da].push(ob),
                    (Err(_), Ok(db)) => plan.linked_supports[db].push(oa),
                    _ => {}
                }
            }
        }
        plan
    }

    fn root(&mut self, mut k: usize) -> usize {
        while self.roots[k] != k {
            self.roots[k] = self.roots[self.roots[k]];
            k = self.roots[k];
        }
        k
    }

    fn merge(&mut self, a: usize, b: usize) {
        let (a, b) = (self.root(a), self.root(b));
        if a != b {
            self.roots[a.max(b)] = a.min(b);
        }
    }

    fn groups(&mut self) -> Vec<Group> {
        let roots: Vec<u32> = (0..self.dynamic.len())
            .map(|i| self.root(i) as u32)
            .collect();
        if roots
            .first()
            .is_some_and(|first| roots.iter().any(|r| r != first))
        {
            let bvh = Bvh::build_grouped(&self.bounds, &roots);
            for (i, &root) in roots.iter().enumerate() {
                let bounds = self.bounds[i];
                bvh.query_excluding(&bounds, root, |j| {
                    if i < j {
                        self.merge(i, j);
                    }
                });
            }
        }
        let mut groups: Vec<Group> = Vec::new();
        let mut slots = vec![usize::MAX; self.dynamic.len()];
        for i in 0..self.dynamic.len() {
            let object = self.dynamic[i];
            let root = self.root(i);
            if slots[root] == usize::MAX {
                slots[root] = groups.len();
                groups.push(Group {
                    members: Vec::new(),
                    supports: Vec::new(),
                });
            }
            let group = &mut groups[slots[root]];
            group.members.push(object);
            self.fixed_bvh
                .query(&self.bounds[i], |j| group.supports.push(self.fixed[j]));
            for &(support, normal, offset) in &self.unbounded {
                if aabb_overlaps_halfspace(&self.bounds[i], normal, offset) {
                    group.supports.push(support);
                }
            }
            group.supports.extend_from_slice(&self.linked_supports[i]);
        }
        for group in &mut groups {
            group.supports.sort_unstable();
            group.supports.dedup();
        }
        groups
    }

    fn enclose(&mut self, worker: &Worker) {
        for (i, dofs) in worker.solver.layout.solids.iter().enumerate() {
            self.enclose_object(worker.indices[dofs.object], worker.solver.sweep.bounds[i]);
        }
        for (i, dofs) in worker.solver.layout.rigids.iter().enumerate() {
            self.enclose_object(
                worker.indices[dofs.object],
                worker.solver.sweep.bounds[worker.solver.layout.solids.len() + i],
            );
        }
    }

    fn enclose_object(&mut self, object: usize, bounds: Aabb) {
        let slot = self.dynamic.binary_search(&object).expect("dynamic object");
        self.bounds[slot] = self.bounds[slot].union(&bounds.grown(self.margins[slot]));
    }
}

impl Worker {
    fn new(parent: &Mechanics, world: &World, group: &Group) -> Self {
        let mut indices = group.members.clone();
        indices.extend_from_slice(&group.supports);
        indices.sort_unstable();
        let mut layout = Layout::selected(world, indices.clone());
        let mut blocks = vec![0; layout.blocks];
        let mut solids = Vec::new();
        let mut rigids = Vec::new();
        for dofs in &layout.solids {
            let Owner::Solid(s) = parent.layout.objects[dofs.object] else {
                unreachable!()
            };
            solids.push(s);
            for (&local, &global) in dofs.block.iter().zip(&parent.layout.solids[s].block) {
                if local != HELD {
                    blocks[local as usize] = global as usize;
                }
            }
        }
        for dofs in &layout.rigids {
            let Owner::Rigid(r) = parent.layout.objects[dofs.object] else {
                unreachable!()
            };
            rigids.push(r);
            for k in 0..4 {
                blocks[dofs.block + k] = parent.layout.rigids[r].block + k;
            }
        }
        let local_world = world.subset(&indices);
        for d in &mut layout.solids {
            d.object = indices.binary_search(&d.object).unwrap();
        }
        for d in &mut layout.rigids {
            d.object = indices.binary_search(&d.object).unwrap();
        }
        for d in &mut layout.fixed {
            d.object = indices.binary_search(&d.object).unwrap();
        }
        layout.indices.clear();
        let mut solver = Mechanics::new(parent.settings);
        solver.built_for = (local_world.objects().len(), local_world.links().len());
        solver.layout = layout;
        solver.dissipative_energy = solver
            .layout
            .solids
            .iter()
            .map(|d| {
                let Representation::Solid(s) = &local_world.objects()[d.object].representation
                else {
                    unreachable!()
                };
                vec![0.0; s.nodes.position.len()]
            })
            .collect();
        solver.dissipative_power = solver.dissipative_energy.clone();
        Self {
            members: group.members.clone(),
            supports: group.supports.clone(),
            blocks,
            solids,
            rigids,
            solver,
            world: local_world,
            indices,
            report: StepReport::default(),
            last_h: None,
            asleep: false,
            settled: false,
        }
    }

    /// Whether this island rests on a fixed support and every point of it
    /// moved slower than the contact precision per `settle_time` (and less
    /// than the precision) over the last step, measured from where it
    /// started that step. Reads only checkpointed state: the world and the
    /// integration history.
    ///
    /// Displacement rather than the stored velocity: a resting stack on the
    /// trapezoidal stage of TR-BDF2 can carry a steady velocity of several
    /// mm/s that its positions do not follow. A body in free flight moves at
    /// least g h²/2 a step, so nothing settles at the top of its arc.
    fn at_rest(&self, parent: &Mechanics, world: &World, settle_time: f64) -> bool {
        let (Some((earlier, _)), Some(h), Some(accel)) =
            (&parent.earlier, parent.previous_h, &parent.acceleration)
        else {
            return false;
        };
        // Resting means resting on something: an island near no fixed support
        // or damped world-mounted joint is floating or oscillating freely, and
        // a slow drift or small vibration there is motion. An in-place joint
        // target or parameter change also prevents settling so actuated joints
        // wake immediately.
        let grounded = !self.supports.is_empty()
            || self.world.links().iter().any(|link| match *link {
                Link::Weld { b: None, .. } | Link::Planar { b: None, .. } => true,
                Link::Hinge { b: None, drive, .. } | Link::Slider { b: None, drive, .. } => {
                    drive.damping > 0.0
                }
                Link::Cylindrical {
                    b: None,
                    linear_drive,
                    angular_drive,
                    ..
                } => linear_drive.damping > 0.0 || angular_drive.damping > 0.0,
                _ => false,
            });
        if !grounded || !self.world.subset_links_match(world, &self.indices) {
            return false;
        }
        // Never more than the precision in one step, however long the step.
        let reach = parent.contact.precision * (h / settle_time).min(1.0);
        let moved = |now: Vec3, block: usize| (now - Vec3::read(earlier, block)).length() <= reach;
        let accel_ok = |block: usize| Vec3::read(accel, block).length() * (0.5 * h * h) <= reach;
        // Already asleep: stopped with no history (zero acceleration and an
        // Euler restart). Any velocity or rotation edit wakes it.
        let asleep = self
            .blocks
            .iter()
            .all(|&b| parent.restart[b] && Vec3::read(accel, b) == Vec3::ZERO);
        self.members.iter().all(|&m| {
            match (&world.objects()[m].representation, parent.layout.objects[m]) {
                (Representation::Rigid(r), Owner::Rigid(g)) => {
                    // Spin in place moves no centre; measure full-step rotation
                    // displacement from `earlier` when available so stage-to-stage
                    // TR-BDF2 micro-spin does not keep a stationary arm awake.
                    let radius = 0.5 * r.shape.local_bounds().size().length();
                    let block = parent.layout.rigids[g].block;
                    let r_earlier = Mat3::from_cols(
                        Vec3::read(earlier, block + 1),
                        Vec3::read(earlier, block + 2),
                        Vec3::read(earlier, block + 3),
                    );
                    let spin_ok = if r_earlier != Mat3::ZERO {
                        (r.pose.rotation.to_mat3() - r_earlier)
                            .frobenius_squared()
                            .sqrt()
                            * (0.5 * radius)
                            <= reach
                    } else {
                        r.spin.length() * radius * h <= reach
                    };
                    let rotated = asleep
                        && !parent.affine[g]
                            .is_none_or(|(q, _)| super::sleep::same_quat(q, r.pose.rotation));
                    moved(r.pose.position, block)
                        && accel_ok(block)
                        && spin_ok
                        && !(asleep && (r.velocity != Vec3::ZERO || r.spin != Vec3::ZERO))
                        && !rotated
                }
                (Representation::Solid(s), Owner::Solid(i)) => parent.layout.solids[i]
                    .block
                    .iter()
                    .zip(s.nodes.position.iter().zip(&s.nodes.velocity))
                    .all(|(&b, (&p, &v))| {
                        b == HELD
                            || (moved(p, b as usize)
                                && accel_ok(b as usize)
                                && !(asleep && v != Vec3::ZERO))
                    }),
                _ => true,
            }
        })
    }

    /// Stop every point of a settled island, in the worker's own world.
    /// Committed like a solve's result.
    fn freeze(&mut self) -> f64 {
        let mut kinetic = 0.0;
        for object in self.world.objects_mut() {
            kinetic += object.kinetic_energy();
            match &mut object.representation {
                Representation::Rigid(r) => {
                    r.velocity = Vec3::ZERO;
                    r.spin = Vec3::ZERO;
                }
                Representation::Solid(s) => s.nodes.velocity.fill(Vec3::ZERO),
                Representation::Fixed(_) => {}
            }
        }
        kinetic
    }

    fn load(&mut self, parent: &Mechanics, pool: &Pool, world: &World, h: f64) {
        let mut desired = parent.settings;
        desired.islands = false;
        self.asleep = parent.settings.sleeping
            && self.solver.fixed_point
            && self.last_h.is_some_and(|old| old.to_bits() == h.to_bits())
            && self.solver.settings == desired
            && super::sleep::same_vec(self.world.gravity, world.gravity)
            && self.world.subset_dynamics_match(world, &self.indices)
            && self
                .solver
                .mapped_history_matches(parent, &self.blocks, &self.rigids, &self.solids);
        // Settled: at rest under the gravity it came to rest in. Decided from
        // checkpointed state alone (never this cached worker), so a restored
        // run decides exactly as the original did.
        self.settled = !self.asleep
            && parent
                .gravity_seen
                .is_some_and(|g| super::sleep::same_vec(g, world.gravity))
            && parent
                .settings
                .settle_time
                .is_some_and(|t| self.at_rest(parent, world, t));
        if self.settled {
            self.world.refresh_subset(world, &self.indices);
        }
        if self.asleep || self.settled {
            self.world.time = world.time;
            self.solver.profile.set_enabled(parent.profile.is_enabled());
            self.solver.profile.clear();
            return;
        }
        self.world.refresh_subset(world, &self.indices);
        self.solver.gravity_seen = parent.gravity_seen;
        self.solver.previous_h = parent.previous_h;
        self.solver.settings = parent.settings;
        self.solver.settings.islands = false;
        if let Some(a) = &parent.acceleration {
            Self::gather(
                &self.blocks,
                a,
                self.solver.acceleration.get_or_insert_with(Vec::new),
            );
        } else {
            self.solver.acceleration = None;
        }
        if let Some((x, v)) = &parent.earlier {
            let (local_x, local_v) = self
                .solver
                .earlier
                .get_or_insert_with(|| (Vec::new(), Vec::new()));
            Self::gather(&self.blocks, x, local_x);
            Self::gather(&self.blocks, v, local_v);
        } else {
            self.solver.earlier = None;
        }
        self.solver.restart.resize(self.blocks.len(), false);
        for (flag, &b) in self.solver.restart.iter_mut().zip(&self.blocks) {
            *flag = parent.restart[b];
        }
        self.solver.affine.resize(self.rigids.len(), None);
        for (affine, &r) in self.solver.affine.iter_mut().zip(&self.rigids) {
            *affine = parent.affine[r];
        }
        self.solver.unloading.resize(self.rigids.len(), 0.0);
        for (unloading, &r) in self.solver.unloading.iter_mut().zip(&self.rigids) {
            *unloading = parent.unloading[r];
        }
        self.solver
            .plastic
            .resize_with(self.solids.len(), Default::default);
        for (plastic, &s) in self.solver.plastic.iter_mut().zip(&self.solids) {
            *plastic = parent.plastic[s].clone();
        }
        self.solver.ensure_terms(pool, &self.world);
        for v in &mut self.solver.dissipative_energy {
            v.fill(0.0);
        }
        for v in &mut self.solver.dissipative_power {
            v.fill(0.0);
        }
        self.solver
            .elastic
            .sync_state(&self.world, &self.solver.layout, &self.solver.plastic);
        self.solver.joints.sync_world(&self.world);
        if !self.asleep {
            self.solver
                .sweep
                .reset(&self.solver.layout, self.solver.contact.precision);
        }
        self.solver.profile.set_enabled(parent.profile.is_enabled());
        self.solver.profile.clear();
    }

    /// Proactive multi-rate substep count for this island over outer interval `h`.
    fn recommended_substeps(&self, h: f64) -> usize {
        let cap = 1usize << super::MAX_SPLITS;
        let mut substeps = 1usize;
        if let Some(max_dt) = self.solver.settings.max_substep {
            if max_dt > 0.0 && max_dt.is_finite() {
                substeps = substeps.max((h / max_dt).ceil() as usize);
            }
        }
        if let Some(cfl) = self.solver.settings.cfl_subcycling {
            if cfl > 0.0 && cfl.is_finite() {
                let mut max_rate = 0.0f64;
                let eps = self.solver.contact.precision.max(1e-6);
                for object in self.world.objects() {
                    match &object.representation {
                        Representation::Rigid(r) => {
                            let size = r.shape.local_bounds().size();
                            let char_len = size.min_element().max(0.25 * size.length()).max(eps);
                            let radius = 0.5 * size.length();
                            let speed = r.velocity.length() + r.spin.length() * radius;
                            max_rate = max_rate.max(speed / char_len);
                        }
                        Representation::Solid(s) => {
                            let size = Aabb::from_points(s.mesh.rest.iter().copied()).size();
                            let char_len = size.min_element().max(0.25 * size.length()).max(eps);
                            let max_v = s
                                .nodes
                                .velocity
                                .iter()
                                .map(|v| v.length())
                                .fold(0.0, f64::max);
                            max_rate = max_rate.max(max_v / char_len);
                        }
                        Representation::Fixed(_) => {}
                    }
                }
                let omega2 = self
                    .solver
                    .joints
                    .max_frequency_squared(&self.world, &self.solver.layout);
                if omega2 > 0.0 && omega2.is_finite() {
                    max_rate = max_rate.max(omega2.sqrt() / std::f64::consts::TAU);
                }
                if max_rate > 0.0 && max_rate.is_finite() {
                    substeps = substeps.max(((h * max_rate) / cfl).ceil() as usize);
                }
            }
        }
        substeps.clamp(1, cap)
    }

    fn gather(blocks: &[usize], global: &[f64], local: &mut Vec<f64>) {
        local.resize(3 * blocks.len(), 0.0);
        for (i, &b) in blocks.iter().enumerate() {
            local[3 * i..3 * i + 3].copy_from_slice(&global[3 * b..3 * b + 3]);
        }
    }

    fn store(&self, parent: &mut Mechanics) {
        for (local, &global) in self.blocks.iter().enumerate() {
            parent.acceleration.as_mut().unwrap()[3 * global..3 * global + 3].copy_from_slice(
                &self.solver.acceleration.as_ref().unwrap()[3 * local..3 * local + 3],
            );
            let (x, v) = parent.earlier.as_mut().unwrap();
            if let Some((lx, lv)) = &self.solver.earlier {
                x[3 * global..3 * global + 3].copy_from_slice(&lx[3 * local..3 * local + 3]);
                v[3 * global..3 * global + 3].copy_from_slice(&lv[3 * local..3 * local + 3]);
            }
            // Split BDF2 intervals invalidate only this island's two-step
            // history. A checkpointed Euler restart ignores its placeholders.
            parent.restart[global] = self.solver.restart[local] || self.solver.earlier.is_none();
        }
        for (local, &global) in self.rigids.iter().enumerate() {
            parent.affine[global] = self.solver.affine[local];
            parent.unloading[global] = self.solver.unloading[local];
        }
        for (local, &global) in self.solids.iter().enumerate() {
            parent.plastic[global] = self.solver.plastic[local].clone();
            parent.dissipative_power[global] = self.solver.dissipative_power[local].clone();
        }
    }

    pub(super) fn sample_wrench(
        &mut self,
        pool: &Pool,
        parent_world: &World,
        link_index: usize,
    ) -> Option<(Vec3, Vec3)> {
        let links = parent_world.links();
        links.get(link_index)?.remap(&self.indices)?;
        let local_idx = links[..link_index]
            .iter()
            .filter(|l| l.remap(&self.indices).is_some())
            .count();
        Some(self.solver.link_wrench_for(pool, &self.world, local_idx))
    }

    pub(super) fn sample_tactile(
        &mut self,
        pool: &Pool,
        object: usize,
        world_origin: Vec3,
        world_normal: Vec3,
        radius: f64,
    ) -> Option<(f64, Vec3, Option<Vec3>, usize)> {
        let local_obj = self.indices.iter().position(|&idx| idx == object)?;
        Some(self.solver.tactile_for(
            pool,
            &self.world,
            local_obj,
            world_origin,
            world_normal,
            radius,
        ))
    }
}

impl Mechanics {
    pub(super) fn step_islands(&mut self, pool: &Pool, world: &mut World, h: f64) -> StepReport {
        // A prior reference solve or derivative check may own a global matrix.
        // Release it before allocating local matrices; history is kept separately.
        if self.terms_ready {
            self.release_terms();
        }
        let original = self.snapshot(world);
        let planning = self.profile.is_enabled().then(std::time::Instant::now);
        let mut plan = Plan::new(self, world, h);
        self.contact.precision = self.settings.contact_precision.unwrap_or_else(|| {
            plan.margins
                .iter()
                .copied()
                .reduce(f64::min)
                .map_or(1e-3, |m| m * 0.25)
        });
        let mut groups = plan.groups();
        if let Some(start) = planning {
            self.profile.add("islands: planning", start.elapsed());
        }
        let mut cache: BTreeMap<Group, Worker> = std::mem::take(&mut self.workers)
            .into_iter()
            .map(|w| {
                (
                    Group {
                        members: w.members.clone(),
                        supports: w.supports.clone(),
                    },
                    w,
                )
            })
            .collect();
        let mut total = StepReport::default();
        loop {
            // Exact keys retain reusable local matrices; no quadratic cache search
            // and no unbounded collection of historical merge/split layouts.
            cache.retain(|g, _| groups.binary_search(g).is_ok());
            let mut active = Vec::with_capacity(groups.len());
            let mut result = StepReport::default();
            for group in &groups {
                let mut worker = cache
                    .remove(group)
                    .unwrap_or_else(|| Worker::new(self, world, group));
                worker.load(self, pool, world, h);
                worker.world.time = original.time;
                active.push(worker);
            }
            let accounting = self.settings.energy_accounting;
            let advance = |worker: &mut Worker| {
                if worker.asleep || worker.settled {
                    let old = worker.report;
                    let mut energy = old.energy;
                    if worker.settled {
                        let before =
                            accounting.then(|| worker.solver.sample_energy(pool, &worker.world));
                        let kinetic = worker.freeze();
                        energy = before.map(|before| {
                            let mut after = before;
                            after.kinetic = 0.0;
                            let mut ledger = EnergyLedger {
                                before,
                                after,
                                settling_work: -kinetic,
                                ..EnergyLedger::default()
                            };
                            ledger.finish();
                            ledger
                        });
                    }
                    worker.report = StepReport {
                        grips: old.grips,
                        boundary_forces: old.boundary_forces,
                        energy,
                        sleeping_islands: 1,
                        max_island_substeps: 0,
                        ..StepReport::default()
                    };
                    worker.report.newton.converged = true;
                    worker.world.time += h;
                } else {
                    let substeps = worker.recommended_substeps(h);
                    if substeps <= 1 {
                        worker.report = worker.solver.step_within(pool, &mut worker.world, h, 0);
                        worker.report.max_island_substeps = 1;
                    } else {
                        let h_sub = h / substeps as f64;
                        let outer_saved = worker.solver.snapshot(&worker.world);
                        let outer_start = worker.solver.state(&worker.world);
                        let mut combined = StepReport::default();
                        for sub in 0..substeps {
                            let r = worker.solver.step_within(pool, &mut worker.world, h_sub, 0);
                            if sub == 0 {
                                combined = r;
                            } else {
                                combined.solves += r.solves;
                                combined.iterations += r.iterations;
                                combined.linear_iterations += r.linear_iterations;
                                combined.linear_block_iterations += r.linear_block_iterations;
                                combined.linear_products += r.linear_products;
                                combined.linear_block_products += r.linear_block_products;
                                combined.impacts += r.impacts;
                                combined.contact_restarts += r.contact_restarts;
                                combined.accuracy_retries += r.accuracy_retries;
                                combined.angular_error += r.angular_error;
                                combined.boundary_forces |= r.boundary_forces;
                                combined.accuracy_failed = r.accuracy_failed;
                                combined.splits += r.splits;
                                combined.refreshes_discarded += r.refreshes_discarded;
                                combined.unconverged = r.unconverged;
                                combined.invalid = r.invalid;
                                combined.grips = r.grips;
                                combined.newton = r.newton;
                                EnergyLedger::serial(&mut combined.energy, r.energy);
                            }
                            if combined.failed() {
                                break;
                            }
                        }
                        if combined.failed() {
                            combined.energy = None;
                            worker.solver.restore(&mut worker.world, outer_saved);
                            worker.solver.fixed_point = false;
                        } else {
                            if worker.solver.settings.integrator == super::Integrator::Bdf2 {
                                worker.solver.earlier = None;
                            } else {
                                worker.solver.earlier = Some(outer_start);
                            }
                            worker.solver.previous_h = Some(h);
                            worker.solver.fixed_point =
                                worker.solver.stationary_match(&worker.world, &outer_saved);
                        }
                        combined.max_island_substeps = substeps;
                        worker.report = combined;
                    }
                    if !worker.report.failed() {
                        let inv_h = 1.0 / h;
                        for (pow, eng) in worker
                            .solver
                            .dissipative_power
                            .iter_mut()
                            .zip(&worker.solver.dissipative_energy)
                        {
                            for (p, &e) in pow.iter_mut().zip(eng) {
                                *p = e * inv_h;
                            }
                        }
                    }
                }
                worker.last_h = Some(h);
            };
            if self.settings.parallel_islands && active.len() > 1 {
                // Nested element/vector jobs run serially on this same pool;
                // fixed reduction chunks still give identical floating results.
                pool.for_chunks_mut(&mut active, 1, |_, part| advance(&mut part[0]));
            } else {
                for worker in &mut active {
                    advance(worker);
                }
            }
            for worker in &active {
                let report = worker.report;
                self.profile.merge(&worker.solver.profile);
                total.solves += report.solves;
                total.iterations += report.iterations;
                total.linear_iterations += report.linear_iterations;
                total.linear_block_iterations += report.linear_block_iterations;
                total.linear_products += report.linear_products;
                total.linear_block_products += report.linear_block_products;
                total.contact_restarts += report.contact_restarts;
                total.accuracy_retries += report.accuracy_retries;
                total.splits += report.splits;
                total.refreshes_discarded += report.refreshes_discarded;
                total.island_solves += usize::from(!worker.asleep && !worker.settled);
                total.max_island_substeps =
                    total.max_island_substeps.max(report.max_island_substeps);
                result.sleeping_islands += report.sleeping_islands;
                if !result.failed() {
                    result.newton = report.newton;
                }
                result.unconverged |= report.unconverged;
                result.invalid |= report.invalid;
                result.accuracy_failed |= report.accuracy_failed;
                result.angular_error += report.angular_error;
                result.boundary_forces |= report.boundary_forces;
                result.grips += report.grips;
                result.impacts += report.impacts;
                EnergyLedger::spatial(&mut result.energy, report.energy);
                // A settled island does not move: its planned bounds, from
                // where it is, already enclose it.
                if !worker.settled {
                    plan.enclose(worker);
                }
            }
            let planning = self.profile.is_enabled().then(std::time::Instant::now);
            let next = plan.groups();
            if let Some(start) = planning {
                self.profile.add("islands: verification", start.elapsed());
            }
            if next != groups {
                self.restore(world, original.clone());
                // Drop obsolete matrices now; retain exact-key storage only.
                cache = active
                    .into_iter()
                    .chain(cache.into_values())
                    .map(|w| {
                        (
                            Group {
                                members: w.members.clone(),
                                supports: w.supports.clone(),
                            },
                            w,
                        )
                    })
                    .collect();
                groups = next;
                total.island_rebuilds += 1;
                continue;
            }
            total.newton = result.newton;
            total.unconverged = result.unconverged;
            total.invalid = result.invalid;
            total.accuracy_failed = result.accuracy_failed;
            total.angular_error = result.angular_error;
            total.boundary_forces = result.boundary_forces;
            total.sleeping_islands = result.sleeping_islands;
            total.grips = result.grips;
            total.impacts = result.impacts;
            total.energy = result.energy;
            if self.settings.energy_accounting && groups.is_empty() {
                total.energy = Some(EnergyLedger::default());
            }
            total.islands = groups.len();
            total.largest_island_blocks = active
                .iter()
                .map(|w| w.solver.layout.blocks)
                .max()
                .unwrap_or(0);
            if total.failed() {
                total.energy = None;
                self.restore(world, original);
            } else {
                let n = 3 * self.layout.blocks;
                self.acceleration.get_or_insert_with(|| vec![0.0; n]);
                self.earlier
                    .get_or_insert_with(|| (vec![0.0; n], vec![0.0; n]));
                for worker in &active {
                    if worker.settled {
                        // At rest with no memory: zero velocity, acceleration,
                        // and unloading budget, and an Euler start when it wakes.
                        world.commit_subset(&worker.world, &worker.indices);
                        let acceleration = self.acceleration.as_mut().unwrap();
                        let (earlier_x, earlier_v) = self.earlier.as_mut().unwrap();
                        for &b in &worker.blocks {
                            acceleration[3 * b..3 * b + 3].fill(0.0);
                            earlier_v[3 * b..3 * b + 3].fill(0.0);
                            self.restart[b] = true;
                        }
                        for &m in &worker.members {
                            match (&world.objects()[m].representation, self.layout.objects[m]) {
                                (Representation::Rigid(r), Owner::Rigid(g)) => {
                                    let block = self.layout.rigids[g].block;
                                    r.pose.position.write(earlier_x, block);
                                    let rot = r.pose.rotation.to_mat3();
                                    for k in 0..3 {
                                        rot.col(k).write(earlier_x, block + 1 + k);
                                    }
                                }
                                (Representation::Solid(s), Owner::Solid(i)) => {
                                    for (node, &b) in self.layout.solids[i].block.iter().enumerate()
                                    {
                                        if b != HELD {
                                            s.nodes.position[node].write(earlier_x, b as usize);
                                        }
                                    }
                                }
                                _ => {}
                            }
                        }
                        for &r in &worker.rigids {
                            self.unloading[r] = 0.0;
                        }
                    } else if !worker.asleep {
                        world.commit_subset(&worker.world, &worker.indices);
                        worker.store(self);
                    }
                }
                self.gravity_seen = Some(world.gravity);
                self.previous_h = Some(h);
                world.time = original.time + h;
            }
            self.workers = active;
            return total;
        }
    }
}
