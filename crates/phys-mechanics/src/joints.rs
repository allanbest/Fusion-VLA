//! Mechanical joints, springs, welds, revolute hinges, prismatic sliders,
//! cylindrical/universal/planar joints, and 1D routed tendons.
//!
//! Each link adds a conservative elastic/actuator potential `U(x)` (scaled by
//! `ctx.frame.weight` inside the Newton objective) and, for damped springs or
//! joint drives, an implicit step-local Rayleigh/Kelvin–Voigt dissipation
//! potential with nonpositive discrete damping work when unactuated.
//!
//! All joint Hessian blocks have fixed block topology for a prepared layout
//! and bind directly into the fixed-pattern [`BlockMatrix`] slots. Bilateral
//! constraints also support optional outer Augmented-Lagrangian multiplier
//! updates within a stage.

use core_math::{Mat3, Pose, Quat, Vec3};
use core_numerics::{BlockMatrix, Triplets};
use core_world::{ortho_basis, JointDrive, Link, LinkPoint, ObjectId, Representation, World};

use crate::dofs::{Attach, Layout, Owner, HELD};
use crate::term::{Context, Term};

#[derive(Clone, Copy, Debug)]
enum DampMode {
    None,
    Full,
    Axis(Vec3),
}

#[derive(Clone, Debug)]
struct PointConstraint {
    link_index: usize,
    is_spring: bool,
    a: Attach,
    b: Attach,
    rest_length: f64,
    stiffness: f64,
    damping: f64,
    damp: DampMode,
    start_d: Vec3,
    lambda: Vec3,
    external: bool,
    blocks: [u32; 8],
    block_count: usize,
    slots: [usize; 64],
}

#[derive(Clone, Copy, Debug)]
enum AxisSource {
    World(Vec3),
    Rigid { rigid: usize, local_axis: Vec3 },
}

#[derive(Clone, Debug)]
struct SliderConstraint {
    link_index: usize,
    planar_only: bool,
    a: Attach,
    b: Attach,
    axis_source: AxisSource,
    stiffness: f64,
    limits: Option<(f64, f64)>,
    drive: JointDrive,
    u0: Vec3,
    s0: f64,
    lambda: Vec3,
    external: bool,
    blocks: [u32; 8],
    block_count: usize,
    slots: [usize; 64],
}

#[derive(Clone, Copy, Debug)]
enum WeldTarget {
    World(Mat3),
    Rigid { rigid: usize, rel: Mat3 },
    Fixed { object: ObjectId, invert: bool },
}

#[derive(Clone, Debug)]
struct WeldRot {
    link_index: usize,
    rigid: usize,
    target: WeldTarget,
    rel_mat: Mat3,
    stiffness: f64,
    lambda: Mat3,
    external: bool,
    blocks: [usize; 6],
    block_count: usize,
    slots: [usize; 36],
}

#[derive(Clone, Copy, Debug)]
enum HingeTarget {
    World(Mat3),
    Rigid(usize),
    Fixed { object: ObjectId },
}

#[derive(Clone, Debug)]
struct HingeRot {
    link_index: usize,
    rigid: usize,
    target: HingeTarget,
    swapped: bool,
    u_a: Vec3,
    v_a: Vec3,
    w_a: Vec3,
    u_b: Vec3,
    v_b: Vec3,
    w_b: Vec3,
    angular_stiffness: f64,
    limits: Option<(f64, f64)>,
    drive: JointDrive,
    theta0: f64,
    n_b0: Vec3,
    t_world0: Vec3,
    base_proj: f64,
    lambda_u: Vec3,
    external: bool,
    blocks: [usize; 6],
    block_count: usize,
    slots: [usize; 36],
}

#[derive(Clone, Copy, Debug)]
enum UniversalTarget {
    World(Vec3),
    Rigid { rigid: usize, axis_b: Vec3 },
    Fixed { object: ObjectId, axis_b: Vec3 },
}

#[derive(Clone, Debug)]
struct UniversalRot {
    link_index: usize,
    rigid: usize,
    axis_a: Vec3,
    target: UniversalTarget,
    stiffness: f64,
    lambda: f64,
    external: bool,
    blocks: [usize; 6],
    block_count: usize,
    slots: [usize; 36],
}

#[derive(Clone, Debug)]
struct TendonConstraint {
    link_index: usize,
    waypoints: Vec<Attach>,
    stiffness: f64,
    damping: f64,
    slack_only: bool,
    drive: JointDrive,
    start_u: Vec<Vec3>,
    start_d: Vec<Vec3>,
    external: bool,
    blocks: Vec<u32>,
    slots: Vec<usize>,
}

#[derive(Clone, Debug, Default)]
pub struct Joints {
    constraints: Vec<PointConstraint>,
    sliders: Vec<SliderConstraint>,
    welds: Vec<WeldRot>,
    hinges: Vec<HingeRot>,
    universals: Vec<UniversalRot>,
    tendons: Vec<TendonConstraint>,
    external: bool,
    touched_rigids: Vec<bool>,
    rigid_pairs: Vec<(usize, usize)>,
}

/// Smooth C^∞ pseudo-Huber proportional actuator potential:
/// `U(e) = (f_max² / k) (sqrt(1 + (k e / f_max)²) - 1)` when `max_force` is set,
/// or `½ k e²` when unlimited. Returns `(U, dU/de, d²U/de²)` with `d²U/de² >= 0`.
#[inline]
fn drive_potential(err: f64, stiffness: f64, max_force: Option<f64>) -> (f64, f64, f64) {
    if stiffness <= 0.0 {
        return (0.0, 0.0, 0.0);
    }
    if let Some(f_max) = max_force {
        if f_max <= 0.0 {
            return (0.0, 0.0, 0.0);
        }
        let r = (stiffness * err) / f_max;
        let q2 = 1.0 + r * r;
        let q = q2.sqrt();
        let u = (f_max * f_max / stiffness) * (q - 1.0);
        let du = (stiffness * err) / q;
        let d2u = stiffness / (q2 * q);
        (u, du, d2u)
    } else {
        (0.5 * stiffness * err * err, stiffness * err, stiffness)
    }
}

#[inline]
fn limit_potential(coord: f64, limits: Option<(f64, f64)>, stiffness: f64) -> (f64, f64, f64) {
    if stiffness <= 0.0 {
        return (0.0, 0.0, 0.0);
    }
    let Some((lo, hi)) = limits else {
        return (0.0, 0.0, 0.0);
    };
    let err = if coord < lo {
        coord - lo
    } else if coord > hi {
        coord - hi
    } else {
        0.0
    };
    if err != 0.0 {
        (0.5 * stiffness * err * err, stiffness * err, stiffness)
    } else {
        (0.0, 0.0, 0.0)
    }
}

fn attach_point(layout: &Layout, world: &World, point: LinkPoint) -> Attach {
    match point {
        LinkPoint::World(p) => Attach::Still(p),
        LinkPoint::Body { object, local } => match layout.objects[object.0 as usize] {
            Owner::Fixed(_) => {
                let Representation::Fixed(f) = &world.objects()[object.0 as usize].representation
                else {
                    unreachable!()
                };
                Attach::Still(f.pose.transform_point(local))
            }
            Owner::Rigid(rigid) => Attach::Rigid {
                rigid: rigid as u32,
                local,
            },
            Owner::Solid(_) => panic!("LinkPoint::Body requires a rigid or fixed object"),
        },
        LinkPoint::Node { object, node } => match layout.objects[object.0 as usize] {
            Owner::Solid(solid) => Attach::Node {
                solid: solid as u32,
                node,
            },
            _ => panic!("LinkPoint::Node requires a deformable solid object"),
        },
    }
}

fn attach_is_external(layout: &Layout, attach: &Attach) -> bool {
    match *attach {
        Attach::Still(_) => true,
        Attach::Node { solid, node } => layout.solids[solid as usize].block[node as usize] == HELD,
        Attach::Rigid { .. } => false,
    }
}

fn attach_velocity(world: &World, layout: &Layout, attach: &Attach) -> Vec3 {
    match *attach {
        Attach::Still(_) => Vec3::ZERO,
        Attach::Node { solid, node } => {
            let obj = layout.solids[solid as usize].object;
            let Representation::Solid(s) = &world.objects()[obj].representation else {
                unreachable!()
            };
            s.nodes.velocity[node as usize]
        }
        Attach::Rigid { rigid, local } => {
            let obj = layout.rigids[rigid as usize].object;
            let Representation::Rigid(r) = &world.objects()[obj].representation else {
                unreachable!()
            };
            r.velocity + r.spin.cross(r.pose.rotation.rotate(local))
        }
    }
}

fn attach_blocks(layout: &Layout, a: &Attach, b: &Attach) -> ([u32; 8], usize) {
    let mut blocks = [0u32; 8];
    let mut len = 0;
    let mut push = |blk: u32| {
        if !blocks[..len].contains(&blk) {
            blocks[len] = blk;
            len += 1;
        }
    };
    for attach in [a, b] {
        match *attach {
            Attach::Still(_) => {}
            Attach::Node { solid, node } => {
                let blk = layout.solids[solid as usize].block[node as usize];
                if blk != HELD {
                    push(blk);
                }
            }
            Attach::Rigid { rigid, .. } => {
                let base = layout.rigids[rigid as usize].block as u32;
                for k in 0..4 {
                    push(base + k);
                }
            }
        }
    }
    (blocks, len)
}

fn combine(ctx: &Context, a: &Attach, b: &Attach, x: &[f64]) -> ([(u32, f64, Mat3); 8], usize) {
    let mut entries = [(0u32, 0.0f64, Mat3::ZERO); 8];
    let mut len = 0;
    for (sign, attach) in [(1.0, a), (-1.0, b)] {
        for &(block, jac) in attach.jacobian(ctx.layout, ctx.frame, x).iter() {
            if let Some(slot) = entries[..len].iter_mut().find(|e| e.0 == block) {
                slot.1 += sign;
                slot.2 += jac * sign;
            } else {
                entries[len] = (block, sign, jac * sign);
                len += 1;
            }
        }
    }
    (entries, len)
}

fn world_point(world: &World, point: LinkPoint) -> Vec3 {
    match point {
        LinkPoint::World(p) => p,
        LinkPoint::Body { object, local } => match &world.objects()[object.0 as usize]
            .representation
        {
            Representation::Fixed(f) => f.pose.transform_point(local),
            Representation::Rigid(r) => r.pose.transform_point(local),
            Representation::Solid(_) => panic!("LinkPoint::Body requires a rigid or fixed object"),
        },
        LinkPoint::Node { object, node } => {
            match &world.objects()[object.0 as usize].representation {
                Representation::Solid(s) => s.nodes.position[node as usize],
                _ => panic!("LinkPoint::Node requires a deformable solid object"),
            }
        }
    }
}

fn rotation_arc(from: Vec3, to: Vec3) -> Quat {
    let dot = from.dot(to).clamp(-1.0, 1.0);
    if dot > 1.0 - 1e-12 {
        return Quat::IDENTITY;
    }
    if dot < -1.0 + 1e-12 {
        let (_, perp, _) = ortho_basis(from);
        return Quat::from_axis_angle(perp, std::f64::consts::PI);
    }
    let cross = from.cross(to);
    Quat::from_xyzw(cross.x, cross.y, cross.z, 1.0 + dot).normalize()
}

fn body_pose(world: &World, object: ObjectId) -> Pose {
    match &world.objects()[object.0 as usize].representation {
        Representation::Fixed(f) => f.pose,
        Representation::Rigid(r) => r.pose,
        Representation::Solid(_) => panic!("rigid joint requires fixed or rigid objects"),
    }
}

fn rot_blocks(layout: &Layout, ra: usize, rb: Option<usize>) -> ([usize; 6], usize) {
    let ba = layout.rigids[ra].block;
    let mut blocks = [ba + 1, ba + 2, ba + 3, 0, 0, 0];
    let mut count = 3;
    if let Some(rb) = rb {
        if rb != ra {
            let bb = layout.rigids[rb].block;
            blocks[3] = bb + 1;
            blocks[4] = bb + 2;
            blocks[5] = bb + 3;
            count = 6;
        }
    }
    (blocks, count)
}

impl Joints {
    pub fn new(ctx: &Context) -> Self {
        let mut constraints = Vec::new();
        let mut sliders = Vec::new();
        let mut welds = Vec::new();
        let mut hinges = Vec::new();
        let mut universals = Vec::new();
        let mut tendons = Vec::new();
        let mut rigid_pairs = Vec::new();
        let mut record_pair = |oa: Option<usize>, ob: Option<usize>| {
            if let (Some(ia), Some(ib)) = (oa, ob) {
                if let (Owner::Rigid(ra), Owner::Rigid(rb)) =
                    (ctx.layout.objects[ia], ctx.layout.objects[ib])
                {
                    if ra != rb {
                        rigid_pairs.push((ra.min(rb), ra.max(rb)));
                    }
                }
            }
        };

        for (link_index, link) in ctx.world.links().iter().enumerate() {
            match link {
                Link::Anchor { .. } => {}
                &Link::Spring {
                    a,
                    b,
                    rest_length,
                    stiffness,
                    damping,
                } => {
                    record_pair(
                        a.object().map(|o| o.0 as usize),
                        b.object().map(|o| o.0 as usize),
                    );
                    let a = attach_point(ctx.layout, ctx.world, a);
                    let b = attach_point(ctx.layout, ctx.world, b);
                    let external =
                        attach_is_external(ctx.layout, &a) || attach_is_external(ctx.layout, &b);
                    let (blocks, block_count) = attach_blocks(ctx.layout, &a, &b);
                    constraints.push(PointConstraint {
                        link_index,
                        is_spring: true,
                        a,
                        b,
                        rest_length,
                        stiffness,
                        damping,
                        damp: DampMode::None,
                        start_d: Vec3::ZERO,
                        lambda: Vec3::ZERO,
                        external,
                        blocks,
                        block_count,
                        slots: [0; 64],
                    });
                }
                &Link::BallJoint { a, b, stiffness } => {
                    record_pair(
                        a.object().map(|o| o.0 as usize),
                        b.object().map(|o| o.0 as usize),
                    );
                    let a = attach_point(ctx.layout, ctx.world, a);
                    let b = attach_point(ctx.layout, ctx.world, b);
                    let external =
                        attach_is_external(ctx.layout, &a) || attach_is_external(ctx.layout, &b);
                    let (blocks, block_count) = attach_blocks(ctx.layout, &a, &b);
                    constraints.push(PointConstraint {
                        link_index,
                        is_spring: false,
                        a,
                        b,
                        rest_length: 0.0,
                        stiffness,
                        damping: 0.0,
                        damp: DampMode::None,
                        start_d: Vec3::ZERO,
                        lambda: Vec3::ZERO,
                        external,
                        blocks,
                        block_count,
                        slots: [0; 64],
                    });
                }
                &Link::Weld {
                    a,
                    b,
                    local_a,
                    local_b,
                    rest_rotation,
                    stiffness,
                    rotational_stiffness,
                } => {
                    record_pair(Some(a.0 as usize), b.map(|o| o.0 as usize));
                    let pt_a = LinkPoint::Body {
                        object: a,
                        local: local_a,
                    };
                    let pt_b = match b {
                        Some(obj) => LinkPoint::Body {
                            object: obj,
                            local: local_b,
                        },
                        None => LinkPoint::World(local_b),
                    };
                    let aa = attach_point(ctx.layout, ctx.world, pt_a);
                    let ab = attach_point(ctx.layout, ctx.world, pt_b);
                    let external =
                        attach_is_external(ctx.layout, &aa) || attach_is_external(ctx.layout, &ab);
                    let (blocks, block_count) = attach_blocks(ctx.layout, &aa, &ab);
                    constraints.push(PointConstraint {
                        link_index,
                        is_spring: false,
                        a: aa,
                        b: ab,
                        rest_length: 0.0,
                        stiffness,
                        damping: 0.0,
                        damp: DampMode::None,
                        start_d: Vec3::ZERO,
                        lambda: Vec3::ZERO,
                        external,
                        blocks,
                        block_count,
                        slots: [0; 64],
                    });
                    Self::push_weld_rot(
                        ctx,
                        &mut welds,
                        link_index,
                        a,
                        b,
                        rest_rotation,
                        rotational_stiffness,
                    );
                }
                &Link::Hinge {
                    a,
                    b,
                    local_a,
                    local_b,
                    axis_a,
                    rest_rotation,
                    stiffness,
                    angular_stiffness,
                    limits,
                    drive,
                    ..
                } => {
                    record_pair(Some(a.0 as usize), b.map(|o| o.0 as usize));
                    let pt_a = LinkPoint::Body {
                        object: a,
                        local: local_a,
                    };
                    let pt_b = match b {
                        Some(obj) => LinkPoint::Body {
                            object: obj,
                            local: local_b,
                        },
                        None => LinkPoint::World(local_b),
                    };
                    let aa = attach_point(ctx.layout, ctx.world, pt_a);
                    let ab = attach_point(ctx.layout, ctx.world, pt_b);
                    let external =
                        attach_is_external(ctx.layout, &aa) || attach_is_external(ctx.layout, &ab);
                    let (blocks, block_count) = attach_blocks(ctx.layout, &aa, &ab);
                    constraints.push(PointConstraint {
                        link_index,
                        is_spring: false,
                        a: aa,
                        b: ab,
                        rest_length: 0.0,
                        stiffness,
                        damping: 0.0,
                        damp: DampMode::None,
                        start_d: Vec3::ZERO,
                        lambda: Vec3::ZERO,
                        external,
                        blocks,
                        block_count,
                        slots: [0; 64],
                    });
                    Self::push_hinge_rot(
                        ctx,
                        &mut hinges,
                        link_index,
                        a,
                        b,
                        axis_a,
                        rest_rotation,
                        angular_stiffness,
                        limits,
                        drive,
                    );
                }
                &Link::Slider {
                    a,
                    b,
                    local_a,
                    local_b,
                    axis_b,
                    rest_rotation,
                    stiffness,
                    rotational_stiffness,
                    limits,
                    drive,
                } => {
                    record_pair(Some(a.0 as usize), b.map(|o| o.0 as usize));
                    let pt_a = LinkPoint::Body {
                        object: a,
                        local: local_a,
                    };
                    let pt_b = match b {
                        Some(obj) => LinkPoint::Body {
                            object: obj,
                            local: local_b,
                        },
                        None => LinkPoint::World(local_b),
                    };
                    let aa = attach_point(ctx.layout, ctx.world, pt_a);
                    let ab = attach_point(ctx.layout, ctx.world, pt_b);
                    let external =
                        attach_is_external(ctx.layout, &aa) || attach_is_external(ctx.layout, &ab);
                    let axis_norm = axis_b.normalize();
                    let axis_source = match b.map(|obj| (obj, ctx.layout.objects[obj.0 as usize])) {
                        None => AxisSource::World(axis_norm),
                        Some((obj, Owner::Fixed(_))) => {
                            let rot = body_pose(ctx.world, obj).rotation;
                            AxisSource::World(rot.rotate(axis_norm))
                        }
                        Some((_, Owner::Rigid(rb))) => AxisSource::Rigid {
                            rigid: rb,
                            local_axis: axis_norm,
                        },
                        Some((_, Owner::Solid(_))) => {
                            panic!("Link::Slider requires rigid or fixed objects")
                        }
                    };
                    let (blocks, block_count) = attach_blocks(ctx.layout, &aa, &ab);
                    sliders.push(SliderConstraint {
                        link_index,
                        planar_only: false,
                        a: aa,
                        b: ab,
                        axis_source,
                        stiffness,
                        limits,
                        drive,
                        u0: Vec3::X,
                        s0: 0.0,
                        lambda: Vec3::ZERO,
                        external,
                        blocks,
                        block_count,
                        slots: [0; 64],
                    });
                    Self::push_weld_rot(
                        ctx,
                        &mut welds,
                        link_index,
                        a,
                        b,
                        rest_rotation,
                        rotational_stiffness,
                    );
                }
                &Link::Cylindrical {
                    a,
                    b,
                    local_a,
                    local_b,
                    axis_a,
                    axis_b,
                    rest_rotation,
                    stiffness,
                    angular_stiffness,
                    linear_limits,
                    angular_limits,
                    linear_drive,
                    angular_drive,
                } => {
                    record_pair(Some(a.0 as usize), b.map(|o| o.0 as usize));
                    let pt_a = LinkPoint::Body {
                        object: a,
                        local: local_a,
                    };
                    let pt_b = match b {
                        Some(obj) => LinkPoint::Body {
                            object: obj,
                            local: local_b,
                        },
                        None => LinkPoint::World(local_b),
                    };
                    let aa = attach_point(ctx.layout, ctx.world, pt_a);
                    let ab = attach_point(ctx.layout, ctx.world, pt_b);
                    let external =
                        attach_is_external(ctx.layout, &aa) || attach_is_external(ctx.layout, &ab);
                    let axis_norm = axis_b.normalize();
                    let axis_source = match b.map(|obj| (obj, ctx.layout.objects[obj.0 as usize])) {
                        None => AxisSource::World(axis_norm),
                        Some((obj, Owner::Fixed(_))) => {
                            let rot = body_pose(ctx.world, obj).rotation;
                            AxisSource::World(rot.rotate(axis_norm))
                        }
                        Some((_, Owner::Rigid(rb))) => AxisSource::Rigid {
                            rigid: rb,
                            local_axis: axis_norm,
                        },
                        Some((_, Owner::Solid(_))) => {
                            panic!("Link::Cylindrical requires rigid or fixed objects")
                        }
                    };
                    let (blocks, block_count) = attach_blocks(ctx.layout, &aa, &ab);
                    sliders.push(SliderConstraint {
                        link_index,
                        planar_only: false,
                        a: aa,
                        b: ab,
                        axis_source,
                        stiffness,
                        limits: linear_limits,
                        drive: linear_drive,
                        u0: Vec3::X,
                        s0: 0.0,
                        lambda: Vec3::ZERO,
                        external,
                        blocks,
                        block_count,
                        slots: [0; 64],
                    });
                    Self::push_hinge_rot(
                        ctx,
                        &mut hinges,
                        link_index,
                        a,
                        b,
                        axis_a,
                        rest_rotation,
                        angular_stiffness,
                        angular_limits,
                        angular_drive,
                    );
                }
                &Link::Universal {
                    a,
                    b,
                    local_a,
                    local_b,
                    axis_a,
                    axis_b,
                    stiffness,
                    angular_stiffness,
                } => {
                    record_pair(Some(a.0 as usize), b.map(|o| o.0 as usize));
                    let pt_a = LinkPoint::Body {
                        object: a,
                        local: local_a,
                    };
                    let pt_b = match b {
                        Some(obj) => LinkPoint::Body {
                            object: obj,
                            local: local_b,
                        },
                        None => LinkPoint::World(local_b),
                    };
                    let aa = attach_point(ctx.layout, ctx.world, pt_a);
                    let ab = attach_point(ctx.layout, ctx.world, pt_b);
                    let external =
                        attach_is_external(ctx.layout, &aa) || attach_is_external(ctx.layout, &ab);
                    let (blocks, block_count) = attach_blocks(ctx.layout, &aa, &ab);
                    constraints.push(PointConstraint {
                        link_index,
                        is_spring: false,
                        a: aa,
                        b: ab,
                        rest_length: 0.0,
                        stiffness,
                        damping: 0.0,
                        damp: DampMode::None,
                        start_d: Vec3::ZERO,
                        lambda: Vec3::ZERO,
                        external,
                        blocks,
                        block_count,
                        slots: [0; 64],
                    });
                    Self::push_universal_rot(
                        ctx,
                        &mut universals,
                        link_index,
                        a,
                        b,
                        axis_a,
                        axis_b,
                        angular_stiffness,
                    );
                }
                &Link::Planar {
                    a,
                    b,
                    local_a,
                    local_b,
                    normal_a,
                    normal_b,
                    stiffness,
                    angular_stiffness,
                } => {
                    record_pair(Some(a.0 as usize), b.map(|o| o.0 as usize));
                    let pt_a = LinkPoint::Body {
                        object: a,
                        local: local_a,
                    };
                    let pt_b = match b {
                        Some(obj) => LinkPoint::Body {
                            object: obj,
                            local: local_b,
                        },
                        None => LinkPoint::World(local_b),
                    };
                    let aa = attach_point(ctx.layout, ctx.world, pt_a);
                    let ab = attach_point(ctx.layout, ctx.world, pt_b);
                    let external =
                        attach_is_external(ctx.layout, &aa) || attach_is_external(ctx.layout, &ab);
                    let n_norm = normal_b.normalize();
                    let axis_source = match b.map(|obj| (obj, ctx.layout.objects[obj.0 as usize])) {
                        None => AxisSource::World(n_norm),
                        Some((obj, Owner::Fixed(_))) => {
                            let rot = body_pose(ctx.world, obj).rotation;
                            AxisSource::World(rot.rotate(n_norm))
                        }
                        Some((_, Owner::Rigid(rb))) => AxisSource::Rigid {
                            rigid: rb,
                            local_axis: n_norm,
                        },
                        Some((_, Owner::Solid(_))) => {
                            panic!("Link::Planar requires rigid or fixed objects")
                        }
                    };
                    let (blocks, block_count) = attach_blocks(ctx.layout, &aa, &ab);
                    sliders.push(SliderConstraint {
                        link_index,
                        planar_only: true,
                        a: aa,
                        b: ab,
                        axis_source,
                        stiffness,
                        limits: None,
                        drive: JointDrive::NONE,
                        u0: Vec3::Y,
                        s0: 0.0,
                        lambda: Vec3::ZERO,
                        external,
                        blocks,
                        block_count,
                        slots: [0; 64],
                    });
                    // Align normal_a with normal_b with free twist rotation around normal.
                    let q_align = rotation_arc(normal_a.normalize(), n_norm);
                    Self::push_hinge_rot(
                        ctx,
                        &mut hinges,
                        link_index,
                        a,
                        b,
                        normal_a,
                        q_align,
                        angular_stiffness,
                        None,
                        JointDrive::NONE,
                    );
                }
                Link::Tendon {
                    waypoints,
                    stiffness,
                    damping,
                    slack_only,
                    drive,
                } => {
                    for pair in waypoints.windows(2) {
                        record_pair(
                            pair[0].object().map(|o| o.0 as usize),
                            pair[1].object().map(|o| o.0 as usize),
                        );
                    }
                    let attaches: Vec<Attach> = waypoints
                        .iter()
                        .map(|&pt| attach_point(ctx.layout, ctx.world, pt))
                        .collect();
                    let external = attaches.iter().any(|a| attach_is_external(ctx.layout, a));
                    let mut blocks = Vec::new();
                    for a in &attaches {
                        match *a {
                            Attach::Still(_) => {}
                            Attach::Node { solid, node } => {
                                let blk = ctx.layout.solids[solid as usize].block[node as usize];
                                if blk != HELD && !blocks.contains(&blk) {
                                    blocks.push(blk);
                                }
                            }
                            Attach::Rigid { rigid, .. } => {
                                let base = ctx.layout.rigids[rigid as usize].block as u32;
                                for k in 0..4 {
                                    if !blocks.contains(&(base + k)) {
                                        blocks.push(base + k);
                                    }
                                }
                            }
                        }
                    }
                    blocks.sort_unstable();
                    let n = blocks.len();
                    let m_segs = attaches.len().saturating_sub(1);
                    tendons.push(TendonConstraint {
                        link_index,
                        waypoints: attaches,
                        stiffness: *stiffness,
                        damping: *damping,
                        slack_only: *slack_only,
                        drive: *drive,
                        start_u: vec![Vec3::ZERO; m_segs],
                        start_d: vec![Vec3::ZERO; m_segs],
                        external,
                        blocks,
                        slots: vec![0; n * n],
                    });
                }
            }
        }

        rigid_pairs.sort_unstable();
        rigid_pairs.dedup();

        let mut touched_rigids = vec![false; ctx.layout.rigids.len()];
        for c in &constraints {
            for attach in [&c.a, &c.b] {
                if let Attach::Rigid { rigid, .. } = *attach {
                    touched_rigids[rigid as usize] = true;
                }
            }
        }
        for s in &sliders {
            for attach in [&s.a, &s.b] {
                if let Attach::Rigid { rigid, .. } = *attach {
                    touched_rigids[rigid as usize] = true;
                }
            }
        }
        for w in &welds {
            touched_rigids[w.rigid] = true;
            if let WeldTarget::Rigid { rigid, .. } = w.target {
                touched_rigids[rigid] = true;
            }
        }
        for h in &hinges {
            touched_rigids[h.rigid] = true;
            if let HingeTarget::Rigid(rigid) = h.target {
                touched_rigids[rigid] = true;
            }
        }
        for u in &universals {
            touched_rigids[u.rigid] = true;
            if let UniversalTarget::Rigid { rigid, .. } = u.target {
                touched_rigids[rigid] = true;
            }
        }
        for t in &tendons {
            for attach in &t.waypoints {
                if let Attach::Rigid { rigid, .. } = *attach {
                    touched_rigids[rigid as usize] = true;
                }
            }
        }
        let external = constraints.iter().any(|c| c.external)
            || sliders.iter().any(|s| s.external)
            || welds.iter().any(|w| w.external)
            || hinges.iter().any(|h| h.external)
            || universals.iter().any(|u| u.external)
            || tendons.iter().any(|t| t.external);
        Self {
            constraints,
            sliders,
            welds,
            hinges,
            universals,
            tendons,
            external,
            touched_rigids,
            rigid_pairs,
        }
    }

    fn push_weld_rot(
        ctx: &Context,
        welds: &mut Vec<WeldRot>,
        link_index: usize,
        a: ObjectId,
        b: Option<ObjectId>,
        rest_rotation: Quat,
        rotational_stiffness: f64,
    ) {
        let rel = rest_rotation.normalize().to_mat3();
        let owner_a = ctx.layout.objects[a.0 as usize];
        let owner_b = b.map(|obj| ctx.layout.objects[obj.0 as usize]);
        match (owner_a, owner_b) {
            (Owner::Rigid(ra), Some(Owner::Rigid(rb))) => {
                let (blocks, block_count) = rot_blocks(ctx.layout, ra, Some(rb));
                welds.push(WeldRot {
                    link_index,
                    rigid: ra,
                    target: WeldTarget::Rigid { rigid: rb, rel },
                    rel_mat: rel,
                    stiffness: rotational_stiffness,
                    lambda: Mat3::ZERO,
                    external: false,
                    blocks,
                    block_count,
                    slots: [0; 36],
                });
            }
            (Owner::Rigid(ra), None) => {
                let (blocks, block_count) = rot_blocks(ctx.layout, ra, None);
                welds.push(WeldRot {
                    link_index,
                    rigid: ra,
                    target: WeldTarget::World(rel),
                    rel_mat: rel,
                    stiffness: rotational_stiffness,
                    lambda: Mat3::ZERO,
                    external: true,
                    blocks,
                    block_count,
                    slots: [0; 36],
                });
            }
            (Owner::Rigid(ra), Some(Owner::Fixed(_))) => {
                let Some(b_id) = b else { unreachable!() };
                let (blocks, block_count) = rot_blocks(ctx.layout, ra, None);
                welds.push(WeldRot {
                    link_index,
                    rigid: ra,
                    target: WeldTarget::Fixed {
                        object: b_id,
                        invert: false,
                    },
                    rel_mat: rel,
                    stiffness: rotational_stiffness,
                    lambda: Mat3::ZERO,
                    external: true,
                    blocks,
                    block_count,
                    slots: [0; 36],
                });
            }
            (Owner::Fixed(_), Some(Owner::Rigid(rb))) => {
                let (blocks, block_count) = rot_blocks(ctx.layout, rb, None);
                welds.push(WeldRot {
                    link_index,
                    rigid: rb,
                    target: WeldTarget::Fixed {
                        object: a,
                        invert: true,
                    },
                    rel_mat: rel,
                    stiffness: rotational_stiffness,
                    lambda: Mat3::ZERO,
                    external: true,
                    blocks,
                    block_count,
                    slots: [0; 36],
                });
            }
            _ => {}
        }
    }

    #[allow(clippy::too_many_arguments)]
    fn push_hinge_rot(
        ctx: &Context,
        hinges: &mut Vec<HingeRot>,
        link_index: usize,
        a: ObjectId,
        b: Option<ObjectId>,
        axis_a: Vec3,
        rest_rotation: Quat,
        angular_stiffness: f64,
        limits: Option<(f64, f64)>,
        drive: JointDrive,
    ) {
        let (u_a, v_a, w_a) = ortho_basis(axis_a);
        let r_rest = rest_rotation.normalize().to_mat3();
        let (u_b, v_b, w_b) = (r_rest * u_a, r_rest * v_a, r_rest * w_a);
        let owner_a = ctx.layout.objects[a.0 as usize];
        let owner_b = b.map(|obj| ctx.layout.objects[obj.0 as usize]);
        match (owner_a, owner_b) {
            (Owner::Rigid(ra), Some(Owner::Rigid(rb))) => {
                let (blocks, block_count) = rot_blocks(ctx.layout, ra, Some(rb));
                hinges.push(HingeRot {
                    link_index,
                    rigid: ra,
                    target: HingeTarget::Rigid(rb),
                    swapped: false,
                    u_a,
                    v_a,
                    w_a,
                    u_b,
                    v_b,
                    w_b,
                    angular_stiffness,
                    limits,
                    drive,
                    theta0: 0.0,
                    n_b0: v_b,
                    t_world0: w_b,
                    base_proj: 0.0,
                    lambda_u: Vec3::ZERO,
                    external: false,
                    blocks,
                    block_count,
                    slots: [0; 36],
                });
            }
            (Owner::Rigid(ra), None) => {
                let (blocks, block_count) = rot_blocks(ctx.layout, ra, None);
                hinges.push(HingeRot {
                    link_index,
                    rigid: ra,
                    target: HingeTarget::World(Mat3::IDENTITY),
                    swapped: false,
                    u_a,
                    v_a,
                    w_a,
                    u_b,
                    v_b,
                    w_b,
                    angular_stiffness,
                    limits,
                    drive,
                    theta0: 0.0,
                    n_b0: v_b,
                    t_world0: w_b,
                    base_proj: 0.0,
                    lambda_u: Vec3::ZERO,
                    external: true,
                    blocks,
                    block_count,
                    slots: [0; 36],
                });
            }
            (Owner::Rigid(ra), Some(Owner::Fixed(_))) => {
                let Some(b_id) = b else { unreachable!() };
                let (blocks, block_count) = rot_blocks(ctx.layout, ra, None);
                hinges.push(HingeRot {
                    link_index,
                    rigid: ra,
                    target: HingeTarget::Fixed { object: b_id },
                    swapped: false,
                    u_a,
                    v_a,
                    w_a,
                    u_b,
                    v_b,
                    w_b,
                    angular_stiffness,
                    limits,
                    drive,
                    theta0: 0.0,
                    n_b0: v_b,
                    t_world0: w_b,
                    base_proj: 0.0,
                    lambda_u: Vec3::ZERO,
                    external: true,
                    blocks,
                    block_count,
                    slots: [0; 36],
                });
            }
            (Owner::Fixed(_), Some(Owner::Rigid(rb))) => {
                let (blocks, block_count) = rot_blocks(ctx.layout, rb, None);
                // Swap A and B so `rigid` is always the dynamic rigid body:
                // angle_swapped = -theta.
                hinges.push(HingeRot {
                    link_index,
                    rigid: rb,
                    target: HingeTarget::Fixed { object: a },
                    swapped: true,
                    u_a: u_b,
                    v_a: v_b,
                    w_a: w_b,
                    u_b: u_a,
                    v_b: v_a,
                    w_b: w_a,
                    angular_stiffness,
                    limits: limits.map(|(lo, hi)| (-hi, -lo)),
                    drive: JointDrive {
                        target: -drive.target,
                        target_velocity: -drive.target_velocity,
                        ..drive
                    },
                    theta0: 0.0,
                    n_b0: v_a,
                    t_world0: w_a,
                    base_proj: 0.0,
                    lambda_u: Vec3::ZERO,
                    external: true,
                    blocks,
                    block_count,
                    slots: [0; 36],
                });
            }
            _ => {}
        }
    }

    #[allow(clippy::too_many_arguments)]
    fn push_universal_rot(
        ctx: &Context,
        universals: &mut Vec<UniversalRot>,
        link_index: usize,
        a: ObjectId,
        b: Option<ObjectId>,
        axis_a: Vec3,
        axis_b: Vec3,
        stiffness: f64,
    ) {
        let ax_a = axis_a.normalize();
        let ax_b = axis_b.normalize();
        let owner_a = ctx.layout.objects[a.0 as usize];
        let owner_b = b.map(|obj| ctx.layout.objects[obj.0 as usize]);
        match (owner_a, owner_b) {
            (Owner::Rigid(ra), Some(Owner::Rigid(rb))) => {
                let (blocks, block_count) = rot_blocks(ctx.layout, ra, Some(rb));
                universals.push(UniversalRot {
                    link_index,
                    rigid: ra,
                    axis_a: ax_a,
                    target: UniversalTarget::Rigid {
                        rigid: rb,
                        axis_b: ax_b,
                    },
                    stiffness,
                    lambda: 0.0,
                    external: false,
                    blocks,
                    block_count,
                    slots: [0; 36],
                });
            }
            (Owner::Rigid(ra), None) => {
                let (blocks, block_count) = rot_blocks(ctx.layout, ra, None);
                universals.push(UniversalRot {
                    link_index,
                    rigid: ra,
                    axis_a: ax_a,
                    target: UniversalTarget::World(ax_b),
                    stiffness,
                    lambda: 0.0,
                    external: true,
                    blocks,
                    block_count,
                    slots: [0; 36],
                });
            }
            (Owner::Rigid(ra), Some(Owner::Fixed(_))) => {
                let Some(b_id) = b else { unreachable!() };
                let (blocks, block_count) = rot_blocks(ctx.layout, ra, None);
                universals.push(UniversalRot {
                    link_index,
                    rigid: ra,
                    axis_a: ax_a,
                    target: UniversalTarget::Fixed {
                        object: b_id,
                        axis_b: ax_b,
                    },
                    stiffness,
                    lambda: 0.0,
                    external: true,
                    blocks,
                    block_count,
                    slots: [0; 36],
                });
            }
            (Owner::Fixed(_), Some(Owner::Rigid(rb))) => {
                let (blocks, block_count) = rot_blocks(ctx.layout, rb, None);
                universals.push(UniversalRot {
                    link_index,
                    rigid: rb,
                    axis_a: ax_b,
                    target: UniversalTarget::Fixed {
                        object: a,
                        axis_b: ax_a,
                    },
                    stiffness,
                    lambda: 0.0,
                    external: true,
                    blocks,
                    block_count,
                    slots: [0; 36],
                });
            }
            _ => {}
        }
    }

    /// Synchronize mutable joint parameters (drives, targets, limits, stiffness,
    /// rest lengths and rest rotations) from `world` without reallocating or
    /// invalidating the bound [`BlockMatrix`] sparsity pattern.
    pub(crate) fn sync_world(&mut self, world: &World) {
        let links = world.links();
        for c in &mut self.constraints {
            let Some(link) = links.get(c.link_index) else {
                continue;
            };
            match *link {
                Link::Spring {
                    rest_length,
                    stiffness,
                    damping,
                    ..
                } if c.is_spring => {
                    c.rest_length = rest_length;
                    c.stiffness = stiffness;
                    c.damping = damping;
                }
                Link::BallJoint { stiffness, .. }
                | Link::Weld { stiffness, .. }
                | Link::Hinge { stiffness, .. }
                | Link::Universal { stiffness, .. } => {
                    c.stiffness = stiffness;
                }
                _ => {}
            }
        }
        for s in &mut self.sliders {
            let Some(link) = links.get(s.link_index) else {
                continue;
            };
            let (axis_b, stiffness, limits, drive, b) = match *link {
                Link::Slider {
                    axis_b,
                    stiffness,
                    limits,
                    drive,
                    b,
                    ..
                } => (axis_b, stiffness, limits, drive, b),
                Link::Cylindrical {
                    axis_b,
                    stiffness,
                    linear_limits,
                    linear_drive,
                    b,
                    ..
                } => (axis_b, stiffness, linear_limits, linear_drive, b),
                Link::Planar {
                    normal_b,
                    stiffness,
                    b,
                    ..
                } => (normal_b, stiffness, None, JointDrive::NONE, b),
                _ => continue,
            };
            s.stiffness = stiffness;
            s.limits = limits;
            s.drive = drive;
            let axis_norm = axis_b.normalize();
            match &mut s.axis_source {
                AxisSource::World(w) => {
                    *w = match b {
                        Some(obj) => body_pose(world, obj).rotation.rotate(axis_norm),
                        None => axis_norm,
                    };
                }
                AxisSource::Rigid { local_axis, .. } => {
                    *local_axis = axis_norm;
                }
            }
        }
        for w in &mut self.welds {
            let Some(link) = links.get(w.link_index) else {
                continue;
            };
            let (rest_rotation, rot_stiffness) = match *link {
                Link::Weld {
                    rest_rotation,
                    rotational_stiffness,
                    ..
                }
                | Link::Slider {
                    rest_rotation,
                    rotational_stiffness,
                    ..
                } => (rest_rotation, rotational_stiffness),
                _ => continue,
            };
            w.rel_mat = rest_rotation.normalize().to_mat3();
            w.stiffness = rot_stiffness;
            if let WeldTarget::Rigid { rel, .. } = &mut w.target {
                *rel = w.rel_mat;
            } else if let WeldTarget::World(r) = &mut w.target {
                *r = w.rel_mat;
            }
        }
        for h in &mut self.hinges {
            let Some(link) = links.get(h.link_index) else {
                continue;
            };
            let (axis_a, rest_rotation, angular_stiffness, limits, drive) = match *link {
                Link::Hinge {
                    axis_a,
                    rest_rotation,
                    angular_stiffness,
                    limits,
                    drive,
                    ..
                } => (axis_a, rest_rotation, angular_stiffness, limits, drive),
                Link::Cylindrical {
                    axis_a,
                    rest_rotation,
                    angular_stiffness,
                    angular_limits,
                    angular_drive,
                    ..
                } => (
                    axis_a,
                    rest_rotation,
                    angular_stiffness,
                    angular_limits,
                    angular_drive,
                ),
                Link::Planar {
                    normal_a,
                    normal_b,
                    angular_stiffness,
                    ..
                } => (
                    normal_a,
                    rotation_arc(normal_a.normalize(), normal_b.normalize()),
                    angular_stiffness,
                    None,
                    JointDrive::NONE,
                ),
                _ => continue,
            };
            let (u_a, v_a, w_a) = ortho_basis(axis_a);
            let r_rest = rest_rotation.normalize().to_mat3();
            let (u_b, v_b, w_b) = (r_rest * u_a, r_rest * v_a, r_rest * w_a);
            h.angular_stiffness = angular_stiffness;
            if h.swapped {
                h.u_a = u_b;
                h.v_a = v_b;
                h.w_a = w_b;
                h.u_b = u_a;
                h.v_b = v_a;
                h.w_b = w_a;
                h.limits = limits.map(|(lo, hi)| (-hi, -lo));
                h.drive = JointDrive {
                    target: -drive.target,
                    target_velocity: -drive.target_velocity,
                    ..drive
                };
            } else {
                h.u_a = u_a;
                h.v_a = v_a;
                h.w_a = w_a;
                h.u_b = u_b;
                h.v_b = v_b;
                h.w_b = w_b;
                h.limits = limits;
                h.drive = drive;
            }
        }
        for u in &mut self.universals {
            if let Some(Link::Universal {
                axis_a,
                axis_b,
                stiffness,
                ..
            }) = links.get(u.link_index)
            {
                u.stiffness = *stiffness;
                let _ = (axis_a, axis_b);
            }
        }
        for t in &mut self.tendons {
            if let Some(Link::Tendon {
                stiffness,
                damping,
                slack_only,
                drive,
                ..
            }) = links.get(t.link_index)
            {
                t.stiffness = *stiffness;
                t.damping = *damping;
                t.slack_only = *slack_only;
                t.drive = *drive;
            }
        }
    }

    pub(crate) fn clear_multipliers(&mut self) {
        for c in &mut self.constraints {
            c.lambda = Vec3::ZERO;
        }
        for s in &mut self.sliders {
            s.lambda = Vec3::ZERO;
        }
        for w in &mut self.welds {
            w.lambda = Mat3::ZERO;
        }
        for h in &mut self.hinges {
            h.lambda_u = Vec3::ZERO;
        }
        for u in &mut self.universals {
            u.lambda = 0.0;
        }
    }

    /// Augmented-Lagrangian dual ascent update `λ <- λ + k * c(x*)` on all
    /// bilateral joint constraints. Returns max position constraint violation (m).
    pub(crate) fn update_multipliers(&mut self, ctx: &Context, x: &[f64]) -> f64 {
        let mut max_err = 0.0_f64;
        for c in &mut self.constraints {
            if c.is_spring || c.stiffness <= 0.0 {
                continue;
            }
            let d = c.a.position(ctx.layout, ctx.frame, x) - c.b.position(ctx.layout, ctx.frame, x);
            max_err = max_err.max(d.length());
            c.lambda += d * c.stiffness;
        }
        for s in &mut self.sliders {
            if s.stiffness <= 0.0 {
                continue;
            }
            let d = s.a.position(ctx.layout, ctx.frame, x) - s.b.position(ctx.layout, ctx.frame, x);
            let coord = s.u0.dot(d);
            if s.planar_only {
                max_err = max_err.max(coord.abs());
                s.lambda.x += s.stiffness * coord;
            } else {
                let d_perp = d - s.u0 * coord;
                max_err = max_err.max(d_perp.length());
                s.lambda += d_perp * s.stiffness;
                s.lambda -= s.u0 * s.u0.dot(s.lambda);
            }
        }
        for w in &mut self.welds {
            if w.stiffness <= 0.0 {
                continue;
            }
            let (_, a_a) = ctx.frame.rigid(ctx.layout, w.rigid, x);
            let target = match w.target {
                WeldTarget::Rigid { rigid, rel } => ctx.frame.rigid(ctx.layout, rigid, x).1 * rel,
                _ => Self::weld_world_target(ctx, w).unwrap(),
            };
            w.lambda += (a_a - target) * w.stiffness;
        }
        for h in &mut self.hinges {
            if h.angular_stiffness <= 0.0 {
                continue;
            }
            let (_, a_a) = ctx.frame.rigid(ctx.layout, h.rigid, x);
            let a_b = Self::hinge_base_matrix(ctx, h, x);
            let err_u = a_a * h.u_a - a_b * h.u_b;
            h.lambda_u += err_u * h.angular_stiffness;
        }
        for u in &mut self.universals {
            if u.stiffness <= 0.0 {
                continue;
            }
            let (_, a_a) = ctx.frame.rigid(ctx.layout, u.rigid, x);
            let p_a = a_a * u.axis_a;
            let p_b = Self::universal_target_vec(ctx, u, x);
            let c = p_a.dot(p_b);
            u.lambda += u.stiffness * c;
        }
        max_err
    }

    /// Physical constraint force & torque (N, N·m) carried through link `link_index`.
    pub(crate) fn link_wrench(
        &self,
        ctx: &Context,
        world: &World,
        link_index: usize,
        x: &[f64],
    ) -> (Vec3, Vec3) {
        let mut force = Vec3::ZERO;
        let mut torque = Vec3::ZERO;
        for c in &self.constraints {
            if c.link_index != link_index {
                continue;
            }
            let d = c.a.position(ctx.layout, ctx.frame, x) - c.b.position(ctx.layout, ctx.frame, x);
            if c.is_spring {
                let len = d.length();
                if len > 1e-12 && c.stiffness > 0.0 {
                    force += (d / len) * (c.stiffness * (len - c.rest_length));
                }
            } else {
                force += d * c.stiffness + c.lambda;
            }
        }
        for s in &self.sliders {
            if s.link_index != link_index {
                continue;
            }
            let d = s.a.position(ctx.layout, ctx.frame, x) - s.b.position(ctx.layout, ctx.frame, x);
            let coord = s.u0.dot(d);
            if s.planar_only {
                force += s.u0 * (s.stiffness * coord + s.lambda.x);
            } else {
                let d_perp = d - s.u0 * coord;
                force += d_perp * s.stiffness + s.lambda;
                let f_ax = limit_potential(coord, s.limits, s.stiffness).1
                    + drive_potential(coord - s.drive.target, s.drive.stiffness, s.drive.max_force)
                        .1;
                force += s.u0 * f_ax;
            }
        }
        for w in &self.welds {
            if w.link_index != link_index || w.stiffness <= 0.0 {
                continue;
            }
            let (_, a_a) = ctx.frame.rigid(ctx.layout, w.rigid, x);
            let target = match w.target {
                WeldTarget::Rigid { rigid, rel } => ctx.frame.rigid(ctx.layout, rigid, x).1 * rel,
                _ => Self::weld_world_target(ctx, w).unwrap(),
            };
            let eff = (a_a - target) * w.stiffness + w.lambda;
            for col in 0..3 {
                torque += a_a.col(col).cross(eff.col(col)) * 0.5;
            }
        }
        for h in &self.hinges {
            if h.link_index != link_index {
                continue;
            }
            let (_, a_a) = ctx.frame.rigid(ctx.layout, h.rigid, x);
            let a_b = Self::hinge_base_matrix(ctx, h, x);
            if h.angular_stiffness > 0.0 {
                let f_u = (a_a * h.u_a - a_b * h.u_b) * h.angular_stiffness + h.lambda_u;
                torque += (a_a * h.u_a).cross(f_u);
            }
            let theta = Self::hinge_angle(ctx, h, x);
            let tau_theta = limit_potential(theta, h.limits, h.angular_stiffness).1
                + drive_potential(theta - h.drive.target, h.drive.stiffness, h.drive.max_force).1;
            let axis_w = a_b * h.u_b;
            torque += axis_w * tau_theta;
        }
        for u in &self.universals {
            if u.link_index != link_index || u.stiffness <= 0.0 {
                continue;
            }
            let (_, a_a) = ctx.frame.rigid(ctx.layout, u.rigid, x);
            let p_a = a_a * u.axis_a;
            let p_b = Self::universal_target_vec(ctx, u, x);
            let c = p_a.dot(p_b);
            torque += p_a.cross(p_b) * (u.stiffness * c + u.lambda);
        }
        for t in &self.tendons {
            if t.link_index != link_index {
                continue;
            }
            let mut total_len = 0.0;
            let mut dir0 = Vec3::ZERO;
            for (seg, pair) in t.waypoints.windows(2).enumerate() {
                let d = pair[1].position(ctx.layout, ctx.frame, x)
                    - pair[0].position(ctx.layout, ctx.frame, x);
                let len = d.length();
                if seg == 0 && len > 1e-12 {
                    dir0 = d / len;
                }
                total_len += len;
            }
            let err = total_len - t.drive.target;
            let e_eff = if t.slack_only { err.max(0.0) } else { err };
            let f_t = if t.slack_only && err <= 0.0 {
                0.0
            } else {
                (if t.stiffness > 0.0 {
                    t.stiffness * e_eff
                } else {
                    0.0
                }) + drive_potential(e_eff, t.drive.stiffness, t.drive.max_force).1
            };
            force += dir0 * f_t;
        }
        let _ = world;
        (force, torque)
    }

    /// Mixed IFT parameter sensitivities `-wᵀ (∂²E / ∂x ∂θ)` over all links.
    pub(crate) fn adjoint_link_gradients(
        &self,
        ctx: &Context,
        world: &World,
        x: &[f64],
        w: &[f64],
        d_target: &mut [f64],
        d_stiffness: &mut [f64],
    ) {
        let scale = ctx.frame.weight;
        let links = world.links();
        let partial_drive_stiff = |err: f64, k: f64, max_force: Option<f64>| -> f64 {
            if let Some(f_max) = max_force {
                if f_max <= 0.0 {
                    return 0.0;
                }
                let r = (k * err) / f_max;
                let q2 = 1.0 + r * r;
                err / (q2 * q2.sqrt())
            } else {
                err
            }
        };

        for c in &self.constraints {
            let d = c.a.position(ctx.layout, ctx.frame, x) - c.b.position(ctx.layout, ctx.frame, x);
            let (entries, count) = combine(ctx, &c.a, &c.b, x);
            let mut dw = Vec3::ZERO;
            for &(b, _, jac) in &entries[..count] {
                dw += jac * Vec3::read(w, b as usize);
            }
            if c.is_spring {
                if c.rest_length == 0.0 {
                    d_stiffness[c.link_index] -= scale * d.dot(dw);
                } else {
                    let len = d.length();
                    if len > 1e-12 {
                        let u = d / len;
                        let u_dw = u.dot(dw);
                        d_target[c.link_index] += scale * c.stiffness * u_dw;
                        d_stiffness[c.link_index] -= scale * (len - c.rest_length) * u_dw;
                    }
                }
            } else if matches!(
                links.get(c.link_index),
                Some(Link::BallJoint { .. } | Link::Weld { .. } | Link::Universal { .. })
            ) {
                d_stiffness[c.link_index] -= scale * d.dot(dw);
            }
        }

        for s in &self.sliders {
            let d = s.a.position(ctx.layout, ctx.frame, x) - s.b.position(ctx.layout, ctx.frame, x);
            let coord = s.u0.dot(d);
            let (entries, count) = combine(ctx, &s.a, &s.b, x);
            let mut dw = Vec3::ZERO;
            for &(b, _, jac) in &entries[..count] {
                dw += jac * Vec3::read(w, b as usize);
            }
            let d_coord_w = s.u0.dot(dw);
            if s.planar_only {
                d_stiffness[s.link_index] -= scale * coord * d_coord_w;
            } else {
                let d2drv =
                    drive_potential(coord - s.drive.target, s.drive.stiffness, s.drive.max_force).2;
                let g_k_drv = partial_drive_stiff(
                    coord - s.drive.target,
                    s.drive.stiffness,
                    s.drive.max_force,
                );
                d_target[s.link_index] += scale * d2drv * d_coord_w;
                if matches!(
                    links.get(s.link_index),
                    Some(Link::Slider { .. } | Link::Cylindrical { .. })
                ) {
                    d_stiffness[s.link_index] -= scale * g_k_drv * d_coord_w;
                }
            }
        }

        for w_rot in &self.welds {
            let ba = ctx.layout.rigids[w_rot.rigid].block;
            let ra = ctx.frame.rigids[w_rot.rigid].radius;
            let (_, a_a) = ctx.frame.rigid(ctx.layout, w_rot.rigid, x);
            let (err, mut term) = match w_rot.target {
                WeldTarget::Rigid { rigid: rb_id, rel } if rb_id != w_rot.rigid => {
                    let bb = ctx.layout.rigids[rb_id].block;
                    let rb = ctx.frame.rigids[rb_id].radius;
                    let (_, a_b) = ctx.frame.rigid(ctx.layout, rb_id, x);
                    let err = a_a - a_b * rel;
                    let mut sum = 0.0;
                    for col in 0..3 {
                        sum += (err.col(col) * (0.5 / ra)).dot(Vec3::read(w, ba + 1 + col));
                    }
                    let err_rt = err * rel.transpose();
                    for j in 0..3 {
                        sum -= (err_rt.col(j) * (0.5 / rb)).dot(Vec3::read(w, bb + 1 + j));
                    }
                    (err, sum)
                }
                _ => {
                    if let Some(r0) = Self::weld_world_target(ctx, w_rot) {
                        let err = a_a - r0;
                        let mut sum = 0.0;
                        for col in 0..3 {
                            sum += (err.col(col) * (0.5 / ra)).dot(Vec3::read(w, ba + 1 + col));
                        }
                        (err, sum)
                    } else {
                        (Mat3::ZERO, 0.0)
                    }
                }
            };
            let _ = err;
            term *= scale;
            d_stiffness[w_rot.link_index] -= term;
        }

        for h in &self.hinges {
            let ra = ctx.frame.rigids[h.rigid].radius;
            let mut beta = [0.0f64; 6];
            for (c, b_slot) in beta.iter_mut().enumerate().take(3) {
                *b_slot = h.v_a[c] / ra;
            }
            if let HingeTarget::Rigid(rb_id) = h.target {
                if rb_id != h.rigid {
                    let rb = ctx.frame.rigids[rb_id].radius;
                    for (c, b_slot) in beta[3..6].iter_mut().enumerate() {
                        *b_slot = -h.n_b0[c] / rb;
                    }
                }
            }
            let mut d_theta_w = 0.0;
            for (&b_val, &block_id) in beta.iter().zip(h.blocks.iter()).take(h.block_count) {
                d_theta_w += (h.t_world0 * b_val).dot(Vec3::read(w, block_id));
            }
            let theta = Self::hinge_angle(ctx, h, x);
            let err = theta - h.drive.target;
            let d2drv = drive_potential(err, h.drive.stiffness, h.drive.max_force).2;
            let g_k_drv = partial_drive_stiff(err, h.drive.stiffness, h.drive.max_force);
            let sign = if h.swapped { -1.0 } else { 1.0 };
            if matches!(links.get(h.link_index), Some(Link::Hinge { .. })) {
                d_target[h.link_index] += sign * scale * d2drv * d_theta_w;
                d_stiffness[h.link_index] -= scale * g_k_drv * d_theta_w;
            }
        }

        for u in &self.universals {
            let ra = ctx.frame.rigids[u.rigid].radius;
            let (_, a_a) = ctx.frame.rigid(ctx.layout, u.rigid, x);
            let p_a = a_a * u.axis_a;
            let p_b = Self::universal_target_vec(ctx, u, x);
            let c = p_a.dot(p_b);
            let mut d_cw = 0.0;
            for col in 0..3 {
                d_cw += (p_b * (u.axis_a[col] / ra)).dot(Vec3::read(w, u.blocks[col]));
            }
            if let UniversalTarget::Rigid {
                rigid: rb_id,
                axis_b,
            } = u.target
            {
                if rb_id != u.rigid {
                    let rb = ctx.frame.rigids[rb_id].radius;
                    for col in 0..3 {
                        d_cw += (p_a * (axis_b[col] / rb)).dot(Vec3::read(w, u.blocks[3 + col]));
                    }
                }
            }
            d_stiffness[u.link_index] -= scale * c * d_cw;
        }

        for t in &self.tendons {
            if t.waypoints.len() < 2 {
                continue;
            }
            let mut total_len = 0.0;
            let mut d_len_w = 0.0;
            for pair in t.waypoints.windows(2) {
                let d = pair[1].position(ctx.layout, ctx.frame, x)
                    - pair[0].position(ctx.layout, ctx.frame, x);
                let len = d.length();
                total_len += len;
                if len > 1e-12 {
                    let dir = d / len;
                    let (entries, count) = combine(ctx, &pair[1], &pair[0], x);
                    for &(b, _, jac) in &entries[..count] {
                        let dw = jac * Vec3::read(w, b as usize);
                        d_len_w += dir.dot(dw);
                    }
                }
            }
            let err = total_len - t.drive.target;
            if !t.slack_only || err > 0.0 {
                let e_eff = if t.slack_only { err.max(0.0) } else { err };
                let d2drv = drive_potential(e_eff, t.drive.stiffness, t.drive.max_force).2;
                let mut h_t = 0.0;
                if t.stiffness > 0.0 {
                    h_t += t.stiffness;
                }
                h_t += d2drv;
                d_target[t.link_index] += scale * h_t * d_len_w;
                if t.drive.stiffness > 0.0 {
                    d_stiffness[t.link_index] -= scale
                        * partial_drive_stiff(e_eff, t.drive.stiffness, t.drive.max_force)
                        * d_len_w;
                } else {
                    d_stiffness[t.link_index] -= scale * e_eff * d_len_w;
                }
            }
        }
    }

    pub(crate) fn touches_rigid(&self, rigid: usize) -> bool {
        self.touched_rigids.get(rigid).copied().unwrap_or(false)
    }

    pub(crate) fn touches_boundary(&self) -> bool {
        self.external
    }

    /// Pairs of rigid-body indices directly connected by a joint or spring.
    pub(crate) fn coupled_rigid_pairs(&self) -> &[(usize, usize)] {
        &self.rigid_pairs
    }

    /// Characteristic maximum natural angular frequency squared `ω² = k / m`
    /// (rad²/s²) across active springs and joint drives in `world`.
    pub(crate) fn max_frequency_squared(&self, world: &World, layout: &Layout) -> f64 {
        let attach_mass = |a: &Attach| -> f64 {
            match *a {
                Attach::Still(_) => f64::INFINITY,
                Attach::Node { solid, node } => {
                    let obj = layout.solids[solid as usize].object;
                    let Representation::Solid(s) = &world.objects()[obj].representation else {
                        unreachable!()
                    };
                    s.nodes.mass[node as usize].max(1e-12)
                }
                Attach::Rigid { rigid, .. } => {
                    let obj = layout.rigids[rigid as usize].object;
                    world.objects()[obj].mass().max(1e-12)
                }
            }
        };
        let mut max_w2 = 0.0_f64;
        for c in &self.constraints {
            if c.is_spring && c.stiffness > 0.0 {
                let inv_m = 1.0 / attach_mass(&c.a) + 1.0 / attach_mass(&c.b);
                max_w2 = max_w2.max(c.stiffness * inv_m);
            }
        }
        for s in &self.sliders {
            if s.drive.stiffness > 0.0 {
                let inv_m = 1.0 / attach_mass(&s.a) + 1.0 / attach_mass(&s.b);
                max_w2 = max_w2.max(s.drive.stiffness * inv_m);
            }
        }
        for h in &self.hinges {
            if h.drive.stiffness > 0.0 {
                let obj_a = layout.rigids[h.rigid].object;
                let Representation::Rigid(ra) = &world.objects()[obj_a].representation else {
                    unreachable!()
                };
                let ia = (ra.inertia.trace() / 3.0).max(1e-12);
                let mut inv_i = 1.0 / ia;
                if let HingeTarget::Rigid(rb_idx) = h.target {
                    let obj_b = layout.rigids[rb_idx].object;
                    let Representation::Rigid(rb) = &world.objects()[obj_b].representation else {
                        unreachable!()
                    };
                    inv_i += 1.0 / (rb.inertia.trace() / 3.0).max(1e-12);
                }
                max_w2 = max_w2.max(h.drive.stiffness * inv_i);
            }
        }
        for t in &self.tendons {
            let k_eff = t.stiffness + t.drive.stiffness;
            if k_eff > 0.0 {
                let min_m = t
                    .waypoints
                    .iter()
                    .map(&attach_mass)
                    .fold(f64::INFINITY, f64::min)
                    .max(1e-12);
                if min_m.is_finite() {
                    max_w2 = max_w2.max(2.0 * k_eff / min_m);
                }
            }
        }
        max_w2
    }

    /// Physical conservative energy of all joints in `world` (no damping), J.
    pub(crate) fn world_energy(world: &World) -> f64 {
        let mut energy = 0.0;
        for link in world.links() {
            match link {
                Link::Anchor { .. } => {}
                &Link::Spring {
                    a,
                    b,
                    rest_length,
                    stiffness,
                    ..
                } => {
                    if stiffness <= 0.0 {
                        continue;
                    }
                    let d = world_point(world, a) - world_point(world, b);
                    if rest_length == 0.0 {
                        energy += 0.5 * stiffness * d.length_squared();
                    } else {
                        let diff = d.length() - rest_length;
                        energy += 0.5 * stiffness * diff * diff;
                    }
                }
                &Link::BallJoint { a, b, stiffness } => {
                    if stiffness > 0.0 {
                        let d = world_point(world, a) - world_point(world, b);
                        energy += 0.5 * stiffness * d.length_squared();
                    }
                }
                &Link::Weld {
                    a,
                    b,
                    local_a,
                    local_b,
                    rest_rotation,
                    stiffness,
                    rotational_stiffness,
                } => {
                    if stiffness > 0.0 {
                        let pa = body_pose(world, a).transform_point(local_a);
                        let pb = match b {
                            Some(obj) => body_pose(world, obj).transform_point(local_b),
                            None => local_b,
                        };
                        energy += 0.5 * stiffness * (pa - pb).length_squared();
                    }
                    if rotational_stiffness > 0.0 {
                        let rot_a = body_pose(world, a).rotation.to_mat3();
                        let rot_b = match b {
                            Some(obj) => body_pose(world, obj).rotation.to_mat3(),
                            None => Mat3::IDENTITY,
                        };
                        let err = rot_a - rot_b * rest_rotation.normalize().to_mat3();
                        energy += 0.25 * rotational_stiffness * err.frobenius_squared();
                    }
                }
                &Link::Hinge {
                    a,
                    b,
                    local_a,
                    local_b,
                    axis_a,
                    rest_rotation,
                    stiffness,
                    angular_stiffness,
                    limits,
                    drive,
                    ..
                } => {
                    let pose_a = body_pose(world, a);
                    let (pose_b, pb) = match b {
                        Some(obj) => {
                            let p = body_pose(world, obj);
                            (p, p.transform_point(local_b))
                        }
                        None => (Pose::IDENTITY, local_b),
                    };
                    if stiffness > 0.0 {
                        let pa = pose_a.transform_point(local_a);
                        energy += 0.5 * stiffness * (pa - pb).length_squared();
                    }
                    let (u_a, v_a, w_a) = ortho_basis(axis_a);
                    let r_rest = rest_rotation.normalize().to_mat3();
                    let rot_a = pose_a.rotation.to_mat3();
                    let rot_b = pose_b.rotation.to_mat3();
                    if angular_stiffness > 0.0 {
                        let ua = rot_a * u_a;
                        let ub = rot_b * (r_rest * u_a);
                        energy += 0.5 * angular_stiffness * (ua - ub).length_squared();
                    }
                    let va = rot_a * v_a;
                    let vb = rot_b * (r_rest * v_a);
                    let wb = rot_b * (r_rest * w_a);
                    let theta = va.dot(wb).atan2(va.dot(vb));
                    energy += limit_potential(theta, limits, angular_stiffness).0;
                    energy +=
                        drive_potential(theta - drive.target, drive.stiffness, drive.max_force).0;
                }
                &Link::Slider {
                    a,
                    b,
                    local_a,
                    local_b,
                    axis_b,
                    rest_rotation,
                    stiffness,
                    rotational_stiffness,
                    limits,
                    drive,
                } => {
                    let pose_a = body_pose(world, a);
                    let (pose_b, pb) = match b {
                        Some(obj) => {
                            let p = body_pose(world, obj);
                            (p, p.transform_point(local_b))
                        }
                        None => (Pose::IDENTITY, local_b),
                    };
                    let pa = pose_a.transform_point(local_a);
                    let d = pa - pb;
                    let u = pose_b.rotation.rotate(axis_b.normalize());
                    let s = u.dot(d);
                    if stiffness > 0.0 {
                        let d_perp = d - u * s;
                        energy += 0.5 * stiffness * d_perp.length_squared();
                    }
                    energy += limit_potential(s, limits, stiffness).0;
                    energy += drive_potential(s - drive.target, drive.stiffness, drive.max_force).0;
                    if rotational_stiffness > 0.0 {
                        let rot_a = pose_a.rotation.to_mat3();
                        let rot_b = pose_b.rotation.to_mat3();
                        let err = rot_a - rot_b * rest_rotation.normalize().to_mat3();
                        energy += 0.25 * rotational_stiffness * err.frobenius_squared();
                    }
                }
                &Link::Cylindrical {
                    a,
                    b,
                    local_a,
                    local_b,
                    axis_a,
                    axis_b,
                    rest_rotation,
                    stiffness,
                    angular_stiffness,
                    linear_limits,
                    angular_limits,
                    linear_drive,
                    angular_drive,
                } => {
                    let pose_a = body_pose(world, a);
                    let (pose_b, pb) = match b {
                        Some(obj) => {
                            let p = body_pose(world, obj);
                            (p, p.transform_point(local_b))
                        }
                        None => (Pose::IDENTITY, local_b),
                    };
                    let d = pose_a.transform_point(local_a) - pb;
                    let u = pose_b.rotation.rotate(axis_b.normalize());
                    let s = u.dot(d);
                    if stiffness > 0.0 {
                        let d_perp = d - u * s;
                        energy += 0.5 * stiffness * d_perp.length_squared();
                    }
                    energy += limit_potential(s, linear_limits, stiffness).0;
                    energy += drive_potential(
                        s - linear_drive.target,
                        linear_drive.stiffness,
                        linear_drive.max_force,
                    )
                    .0;
                    let (u_a, v_a, w_a) = ortho_basis(axis_a);
                    let r_rest = rest_rotation.normalize().to_mat3();
                    let rot_a = pose_a.rotation.to_mat3();
                    let rot_b = pose_b.rotation.to_mat3();
                    if angular_stiffness > 0.0 {
                        let ua = rot_a * u_a;
                        let ub = rot_b * (r_rest * u_a);
                        energy += 0.5 * angular_stiffness * (ua - ub).length_squared();
                    }
                    let va = rot_a * v_a;
                    let vb = rot_b * (r_rest * v_a);
                    let wb = rot_b * (r_rest * w_a);
                    let theta = va.dot(wb).atan2(va.dot(vb));
                    energy += limit_potential(theta, angular_limits, angular_stiffness).0;
                    energy += drive_potential(
                        theta - angular_drive.target,
                        angular_drive.stiffness,
                        angular_drive.max_force,
                    )
                    .0;
                }
                &Link::Universal {
                    a,
                    b,
                    local_a,
                    local_b,
                    axis_a,
                    axis_b,
                    stiffness,
                    angular_stiffness,
                } => {
                    let pose_a = body_pose(world, a);
                    let (pose_b, pb) = match b {
                        Some(obj) => {
                            let p = body_pose(world, obj);
                            (p, p.transform_point(local_b))
                        }
                        None => (Pose::IDENTITY, local_b),
                    };
                    if stiffness > 0.0 {
                        energy += 0.5
                            * stiffness
                            * (pose_a.transform_point(local_a) - pb).length_squared();
                    }
                    if angular_stiffness > 0.0 {
                        let pa = pose_a.rotation.rotate(axis_a.normalize());
                        let pb_ax = pose_b.rotation.rotate(axis_b.normalize());
                        let c = pa.dot(pb_ax);
                        energy += 0.5 * angular_stiffness * c * c;
                    }
                }
                &Link::Planar {
                    a,
                    b,
                    local_a,
                    local_b,
                    normal_a,
                    normal_b,
                    stiffness,
                    angular_stiffness,
                } => {
                    let pose_a = body_pose(world, a);
                    let (pose_b, pb) = match b {
                        Some(obj) => {
                            let p = body_pose(world, obj);
                            (p, p.transform_point(local_b))
                        }
                        None => (Pose::IDENTITY, local_b),
                    };
                    let n_b = pose_b.rotation.rotate(normal_b.normalize());
                    if stiffness > 0.0 {
                        let s = n_b.dot(pose_a.transform_point(local_a) - pb);
                        energy += 0.5 * stiffness * s * s;
                    }
                    if angular_stiffness > 0.0 {
                        let n_a = pose_a.rotation.rotate(normal_a.normalize());
                        energy += 0.5 * angular_stiffness * (n_a - n_b).length_squared();
                    }
                }
                Link::Tendon {
                    waypoints,
                    stiffness,
                    slack_only,
                    drive,
                    ..
                } => {
                    let mut len = 0.0;
                    for pair in waypoints.windows(2) {
                        len += (world_point(world, pair[1]) - world_point(world, pair[0])).length();
                    }
                    let err = len - drive.target;
                    let e_eff = if *slack_only { err.max(0.0) } else { err };
                    if *stiffness > 0.0 {
                        energy += 0.5 * stiffness * e_eff * e_eff;
                    }
                    energy += drive_potential(e_eff, drive.stiffness, drive.max_force).0;
                }
            }
        }
        energy
    }

    /// Discrete Kelvin–Voigt damping work over the completed stage, J.
    pub(crate) fn damping_work(&self, ctx: &Context, x: &[f64]) -> f64 {
        if ctx.frame.h <= 0.0 {
            return 0.0;
        }
        let mut work = 0.0;
        for c in &self.constraints {
            if c.damping <= 0.0 {
                continue;
            }
            let d = c.a.position(ctx.layout, ctx.frame, x) - c.b.position(ctx.layout, ctx.frame, x);
            let dd = d - c.start_d;
            let sq = match c.damp {
                DampMode::None => 0.0,
                DampMode::Full => dd.length_squared(),
                DampMode::Axis(n) => {
                    let s = n.dot(dd);
                    s * s
                }
            };
            work -= (c.damping / ctx.frame.h) * sq;
        }
        for s in &self.sliders {
            if s.drive.damping <= 0.0 {
                continue;
            }
            let d = s.a.position(ctx.layout, ctx.frame, x) - s.b.position(ctx.layout, ctx.frame, x);
            let ds = s.u0.dot(d) - s.s0;
            let err_v = ds - s.drive.target_velocity * ctx.frame.h;
            work -= (s.drive.damping / ctx.frame.h) * err_v * ds;
        }
        for h in &self.hinges {
            if h.drive.damping <= 0.0 {
                continue;
            }
            let dtheta = Self::hinge_angle(ctx, h, x) - h.theta0;
            let err_v = dtheta - h.drive.target_velocity * ctx.frame.h;
            work -= (h.drive.damping / ctx.frame.h) * err_v * dtheta;
        }
        for t in &self.tendons {
            let c_tot = t.damping + t.drive.damping;
            if c_tot <= 0.0 {
                continue;
            }
            let mut d_lin = 0.0;
            for (seg, pair) in t.waypoints.windows(2).enumerate() {
                let d = pair[1].position(ctx.layout, ctx.frame, x)
                    - pair[0].position(ctx.layout, ctx.frame, x);
                d_lin += t.start_u[seg].dot(d - t.start_d[seg]);
            }
            let err_v = d_lin - t.drive.target_velocity * ctx.frame.h;
            work -= (c_tot / ctx.frame.h) * err_v * d_lin;
        }
        work
    }

    fn universal_target_vec(ctx: &Context, u: &UniversalRot, x: &[f64]) -> Vec3 {
        match u.target {
            UniversalTarget::World(ax) => ax,
            UniversalTarget::Fixed { object, axis_b } => {
                body_pose(ctx.world, object).rotation.rotate(axis_b)
            }
            UniversalTarget::Rigid { rigid, axis_b } => {
                ctx.frame.rigid(ctx.layout, rigid, x).1 * axis_b
            }
        }
    }

    /// Physical generalized gradient (in N) at step start `(x, v)` for acceleration
    /// initialization in TR-BDF2 / Newmark.
    pub(crate) fn startup_gradient(&self, ctx: &Context, x: &[f64], out: &mut [f64]) {
        for c in &self.constraints {
            let d = c.a.position(ctx.layout, ctx.frame, x) - c.b.position(ctx.layout, ctx.frame, x);
            let mut grad = if c.stiffness <= 0.0 {
                Vec3::ZERO
            } else if c.rest_length == 0.0 {
                d * c.stiffness
            } else {
                let len = d.length();
                if len > 1e-12 {
                    d * (c.stiffness * (len - c.rest_length) / len)
                } else {
                    Vec3::ZERO
                }
            };
            if c.damping > 0.0 {
                let dv = attach_velocity(ctx.world, ctx.layout, &c.a)
                    - attach_velocity(ctx.world, ctx.layout, &c.b);
                match c.damp {
                    DampMode::None => {}
                    DampMode::Full => grad += dv * c.damping,
                    DampMode::Axis(n) => grad += n * (c.damping * n.dot(dv)),
                }
            }
            if grad != Vec3::ZERO {
                let (entries, len) = combine(ctx, &c.a, &c.b, x);
                for &(b, _, jac) in &entries[..len] {
                    (jac.transpose() * grad).add_to(out, b as usize);
                }
            }
        }
        for s in &self.sliders {
            let d = s.a.position(ctx.layout, ctx.frame, x) - s.b.position(ctx.layout, ctx.frame, x);
            let coord = s.u0.dot(d);
            let grad = if s.planar_only {
                if s.stiffness > 0.0 {
                    s.u0 * (s.stiffness * coord)
                } else {
                    Vec3::ZERO
                }
            } else {
                let d_perp = d - s.u0 * coord;
                let mut g = if s.stiffness > 0.0 {
                    d_perp * s.stiffness
                } else {
                    Vec3::ZERO
                };
                let mut g_s = limit_potential(coord, s.limits, s.stiffness).1
                    + drive_potential(coord - s.drive.target, s.drive.stiffness, s.drive.max_force)
                        .1;
                if s.drive.damping > 0.0 {
                    let dv = attach_velocity(ctx.world, ctx.layout, &s.a)
                        - attach_velocity(ctx.world, ctx.layout, &s.b);
                    g_s += s.drive.damping * (s.u0.dot(dv) - s.drive.target_velocity);
                }
                g += s.u0 * g_s;
                g
            };
            if grad != Vec3::ZERO {
                let (entries, len) = combine(ctx, &s.a, &s.b, x);
                for &(b, _, jac) in &entries[..len] {
                    (jac.transpose() * grad).add_to(out, b as usize);
                }
            }
        }
        for t in &self.tendons {
            let mut total_len = 0.0;
            for pair in t.waypoints.windows(2) {
                total_len += (pair[1].position(ctx.layout, ctx.frame, x)
                    - pair[0].position(ctx.layout, ctx.frame, x))
                .length();
            }
            let err = total_len - t.drive.target;
            let e_eff = if t.slack_only { err.max(0.0) } else { err };
            let g_tens = (if t.stiffness > 0.0 {
                t.stiffness * e_eff
            } else {
                0.0
            }) + drive_potential(e_eff, t.drive.stiffness, t.drive.max_force).1;
            let c_tot = t.damping + t.drive.damping;
            let mut g_damp = 0.0;
            if c_tot > 0.0 {
                let mut dl_dt = 0.0;
                for (seg, pair) in t.waypoints.windows(2).enumerate() {
                    let dv = attach_velocity(ctx.world, ctx.layout, &pair[1])
                        - attach_velocity(ctx.world, ctx.layout, &pair[0]);
                    dl_dt += t.start_u[seg].dot(dv);
                }
                g_damp = c_tot * (dl_dt - t.drive.target_velocity);
            }
            for (seg, pair) in t.waypoints.windows(2).enumerate() {
                let d = pair[1].position(ctx.layout, ctx.frame, x)
                    - pair[0].position(ctx.layout, ctx.frame, x);
                let len = d.length();
                let mut force = Vec3::ZERO;
                if len > 1e-12 && g_tens != 0.0 {
                    force += (d / len) * g_tens;
                }
                if g_damp != 0.0 {
                    force += t.start_u[seg] * g_damp;
                }
                if force != Vec3::ZERO {
                    let (entries, count) = combine(ctx, &pair[1], &pair[0], x);
                    for &(b, _, jac) in &entries[..count] {
                        (jac.transpose() * force).add_to(out, b as usize);
                    }
                }
            }
        }
    }

    fn weld_world_target(ctx: &Context, w: &WeldRot) -> Option<Mat3> {
        match w.target {
            WeldTarget::World(r) => Some(r),
            WeldTarget::Fixed { object, invert } => {
                let r_fixed = body_pose(ctx.world, object).rotation.to_mat3();
                Some(if invert {
                    r_fixed * w.rel_mat.transpose()
                } else {
                    r_fixed * w.rel_mat
                })
            }
            WeldTarget::Rigid { .. } => None,
        }
    }

    fn hinge_base_matrix(ctx: &Context, h: &HingeRot, x: &[f64]) -> Mat3 {
        match h.target {
            HingeTarget::World(r) => r,
            HingeTarget::Fixed { object, .. } => body_pose(ctx.world, object).rotation.to_mat3(),
            HingeTarget::Rigid(rb) => ctx.frame.rigid(ctx.layout, rb, x).1,
        }
    }

    fn hinge_angle(ctx: &Context, h: &HingeRot, x: &[f64]) -> f64 {
        let (_, a_a) = ctx.frame.rigid(ctx.layout, h.rigid, x);
        let a_b = Self::hinge_base_matrix(ctx, h, x);
        h.theta0 + h.t_world0.dot(a_a * h.v_a - a_b * h.n_b0) - h.base_proj
    }
}

impl Term for Joints {
    fn name(&self) -> &'static str {
        "joints"
    }

    fn pattern(&self, _ctx: &Context, entries: &mut Vec<(usize, usize)>) {
        for c in &self.constraints {
            for r in 0..c.block_count {
                for col in 0..c.block_count {
                    entries.push((c.blocks[r] as usize, c.blocks[col] as usize));
                }
            }
        }
        for s in &self.sliders {
            for r in 0..s.block_count {
                for col in 0..s.block_count {
                    entries.push((s.blocks[r] as usize, s.blocks[col] as usize));
                }
            }
        }
        for w in &self.welds {
            for r in 0..w.block_count {
                for col in 0..w.block_count {
                    entries.push((w.blocks[r], w.blocks[col]));
                }
            }
        }
        for h in &self.hinges {
            for r in 0..h.block_count {
                for col in 0..h.block_count {
                    entries.push((h.blocks[r], h.blocks[col]));
                }
            }
        }
        for u in &self.universals {
            for r in 0..u.block_count {
                for col in 0..u.block_count {
                    entries.push((u.blocks[r], u.blocks[col]));
                }
            }
        }
        for t in &self.tendons {
            for &row in &t.blocks {
                for &col in &t.blocks {
                    entries.push((row as usize, col as usize));
                }
            }
        }
    }

    fn bind(&mut self, _ctx: &Context, matrix: &BlockMatrix) {
        for c in &mut self.constraints {
            for r in 0..c.block_count {
                for col in 0..c.block_count {
                    c.slots[r * c.block_count + col] = matrix
                        .slot(c.blocks[r] as usize, c.blocks[col] as usize)
                        .unwrap();
                }
            }
        }
        for s in &mut self.sliders {
            for r in 0..s.block_count {
                for col in 0..s.block_count {
                    s.slots[r * s.block_count + col] = matrix
                        .slot(s.blocks[r] as usize, s.blocks[col] as usize)
                        .unwrap();
                }
            }
        }
        for w in &mut self.welds {
            for r in 0..w.block_count {
                for col in 0..w.block_count {
                    w.slots[r * w.block_count + col] =
                        matrix.slot(w.blocks[r], w.blocks[col]).unwrap();
                }
            }
        }
        for h in &mut self.hinges {
            for r in 0..h.block_count {
                for col in 0..h.block_count {
                    h.slots[r * h.block_count + col] =
                        matrix.slot(h.blocks[r], h.blocks[col]).unwrap();
                }
            }
        }
        for u in &mut self.universals {
            for r in 0..u.block_count {
                for col in 0..u.block_count {
                    u.slots[r * u.block_count + col] =
                        matrix.slot(u.blocks[r], u.blocks[col]).unwrap();
                }
            }
        }
        for t in &mut self.tendons {
            let n = t.blocks.len();
            for r in 0..n {
                for col in 0..n {
                    t.slots[r * n + col] = matrix
                        .slot(t.blocks[r] as usize, t.blocks[col] as usize)
                        .unwrap();
                }
            }
        }
    }

    fn begin_step(&mut self, ctx: &Context) {
        self.sync_world(ctx.world);
        self.clear_multipliers();
        for c in &mut self.constraints {
            if c.damping <= 0.0 {
                c.damp = DampMode::None;
                continue;
            }
            c.start_d = c.a.start(ctx.layout, ctx.frame) - c.b.start(ctx.layout, ctx.frame);
            c.damp = if c.rest_length == 0.0 {
                DampMode::Full
            } else if let Some(n) = c.start_d.try_normalize() {
                DampMode::Axis(n)
            } else {
                DampMode::None
            };
        }
        for s in &mut self.sliders {
            let start_d = s.a.start(ctx.layout, ctx.frame) - s.b.start(ctx.layout, ctx.frame);
            s.u0 = match s.axis_source {
                AxisSource::World(u) => u.normalize(),
                AxisSource::Rigid { rigid, local_axis } => ctx.frame.rigids[rigid]
                    .rotation_start
                    .rotate(local_axis)
                    .normalize(),
            };
            s.s0 = s.u0.dot(start_d);
        }
        for h in &mut self.hinges {
            let a_a0 = ctx.frame.rigids[h.rigid].matrix_start;
            let a_b0 = match h.target {
                HingeTarget::World(r) => r,
                HingeTarget::Fixed { object, .. } => {
                    body_pose(ctx.world, object).rotation.to_mat3()
                }
                HingeTarget::Rigid(rb) => ctx.frame.rigids[rb].matrix_start,
            };
            let va0 = a_a0 * h.v_a;
            let vb0 = a_b0 * h.v_b;
            let wb0 = a_b0 * h.w_b;
            h.theta0 = va0.dot(wb0).atan2(va0.dot(vb0));
            let (sin_t, cos_t) = h.theta0.sin_cos();
            h.n_b0 = h.v_b * cos_t + h.w_b * sin_t;
            let t_b0 = -h.v_b * sin_t + h.w_b * cos_t;
            h.t_world0 = a_b0 * t_b0;
            h.base_proj = h.t_world0.dot(va0 - a_b0 * h.n_b0);
        }
        for t in &mut self.tendons {
            for (seg, pair) in t.waypoints.windows(2).enumerate() {
                let d = pair[1].start(ctx.layout, ctx.frame) - pair[0].start(ctx.layout, ctx.frame);
                t.start_d[seg] = d;
                t.start_u[seg] = d.try_normalize().unwrap_or(Vec3::ZERO);
            }
        }
    }

    fn energy(&mut self, ctx: &Context, x: &[f64]) -> f64 {
        let scale = ctx.frame.weight;
        let mut total = 0.0;
        for c in &self.constraints {
            let d = c.a.position(ctx.layout, ctx.frame, x) - c.b.position(ctx.layout, ctx.frame, x);
            if c.stiffness > 0.0 {
                if c.rest_length == 0.0 {
                    total += scale * 0.5 * c.stiffness * d.length_squared();
                } else {
                    let diff = d.length() - c.rest_length;
                    total += scale * 0.5 * c.stiffness * diff * diff;
                }
            }
            if !c.is_spring {
                total += scale * c.lambda.dot(d);
            }
            if c.damping > 0.0 && ctx.frame.h > 0.0 {
                let dd = d - c.start_d;
                let sq = match c.damp {
                    DampMode::None => 0.0,
                    DampMode::Full => dd.length_squared(),
                    DampMode::Axis(n) => {
                        let s = n.dot(dd);
                        s * s
                    }
                };
                total += scale * 0.5 * (c.damping / ctx.frame.h) * sq;
            }
        }
        for s in &self.sliders {
            let d = s.a.position(ctx.layout, ctx.frame, x) - s.b.position(ctx.layout, ctx.frame, x);
            let coord = s.u0.dot(d);
            if s.planar_only {
                if s.stiffness > 0.0 {
                    total += scale * (0.5 * s.stiffness * coord * coord + s.lambda.x * coord);
                }
            } else {
                if s.stiffness > 0.0 {
                    let d_perp = d - s.u0 * coord;
                    total += scale
                        * (0.5 * s.stiffness * d_perp.length_squared() + s.lambda.dot(d_perp));
                }
                total += scale * limit_potential(coord, s.limits, s.stiffness).0;
                total += scale
                    * drive_potential(coord - s.drive.target, s.drive.stiffness, s.drive.max_force)
                        .0;
                if s.drive.damping > 0.0 && ctx.frame.h > 0.0 {
                    let s_ref = s.s0 + s.drive.target_velocity * ctx.frame.h;
                    let ds = coord - s_ref;
                    total += scale * 0.5 * (s.drive.damping / ctx.frame.h) * ds * ds;
                }
            }
        }
        for w in &self.welds {
            if w.stiffness <= 0.0 {
                continue;
            }
            let (_, a_a) = ctx.frame.rigid(ctx.layout, w.rigid, x);
            let target = match w.target {
                WeldTarget::Rigid { rigid, rel } => ctx.frame.rigid(ctx.layout, rigid, x).1 * rel,
                _ => Self::weld_world_target(ctx, w).unwrap(),
            };
            let err = a_a - target;
            let mut l_dot = 0.0;
            for col in 0..3 {
                l_dot += w.lambda.col(col).dot(err.col(col));
            }
            total += scale * (0.25 * w.stiffness * err.frobenius_squared() + 0.5 * l_dot);
        }
        for h in &self.hinges {
            let (_, a_a) = ctx.frame.rigid(ctx.layout, h.rigid, x);
            let a_b = Self::hinge_base_matrix(ctx, h, x);
            if h.angular_stiffness > 0.0 {
                let err_u = a_a * h.u_a - a_b * h.u_b;
                total += scale
                    * (0.5 * h.angular_stiffness * err_u.length_squared() + h.lambda_u.dot(err_u));
            }
            let theta = Self::hinge_angle(ctx, h, x);
            total += scale * limit_potential(theta, h.limits, h.angular_stiffness).0;
            total += scale
                * drive_potential(theta - h.drive.target, h.drive.stiffness, h.drive.max_force).0;
            if h.drive.damping > 0.0 && ctx.frame.h > 0.0 {
                let t_ref = h.theta0 + h.drive.target_velocity * ctx.frame.h;
                let dt = theta - t_ref;
                total += scale * 0.5 * (h.drive.damping / ctx.frame.h) * dt * dt;
            }
        }
        for u in &self.universals {
            if u.stiffness <= 0.0 {
                continue;
            }
            let (_, a_a) = ctx.frame.rigid(ctx.layout, u.rigid, x);
            let p_a = a_a * u.axis_a;
            let p_b = Self::universal_target_vec(ctx, u, x);
            let c = p_a.dot(p_b);
            total += scale * (0.5 * u.stiffness * c * c + u.lambda * c);
        }
        for t in &self.tendons {
            if t.waypoints.len() < 2 {
                continue;
            }
            let mut total_len = 0.0;
            let mut d_lin = 0.0;
            for (seg, pair) in t.waypoints.windows(2).enumerate() {
                let d = pair[1].position(ctx.layout, ctx.frame, x)
                    - pair[0].position(ctx.layout, ctx.frame, x);
                total_len += d.length();
                d_lin += t.start_u[seg].dot(d - t.start_d[seg]);
            }
            let err = total_len - t.drive.target;
            let e_eff = if t.slack_only { err.max(0.0) } else { err };
            if !t.slack_only || err > 0.0 {
                if t.stiffness > 0.0 {
                    total += scale * 0.5 * t.stiffness * e_eff * e_eff;
                }
                total += scale * drive_potential(e_eff, t.drive.stiffness, t.drive.max_force).0;
            }
            let c_tot = t.damping + t.drive.damping;
            if c_tot > 0.0 && ctx.frame.h > 0.0 {
                let err_d = d_lin - t.drive.target_velocity * ctx.frame.h;
                total += scale * 0.5 * (c_tot / ctx.frame.h) * err_d * err_d;
            }
        }
        total
    }

    fn derivatives(
        &mut self,
        ctx: &Context,
        x: &[f64],
        gradient: &mut [f64],
        fixed: &mut BlockMatrix,
        _changing: &mut Triplets,
    ) {
        let scale = ctx.frame.weight;
        let values = fixed.values_mut();
        let find_idx = |blocks: &[u32], len: usize, blk: u32| -> usize {
            blocks[..len].iter().position(|&b| b == blk).unwrap()
        };

        for c in &self.constraints {
            let d = c.a.position(ctx.layout, ctx.frame, x) - c.b.position(ctx.layout, ctx.frame, x);
            let mut g_d = Vec3::ZERO;
            let mut h_d = Mat3::ZERO;
            if c.stiffness > 0.0 {
                if c.rest_length == 0.0 {
                    g_d += d * (scale * c.stiffness);
                    h_d += Mat3::scalar(scale * c.stiffness);
                } else {
                    let len = d.length();
                    if len > 1e-12 {
                        let u = d / len;
                        let k = scale * c.stiffness;
                        g_d += u * (k * (len - c.rest_length));
                        let outer = Mat3::outer(u, u);
                        let radial = outer * k;
                        let stretch_ratio = (len - c.rest_length) / len;
                        h_d += if stretch_ratio > 0.0 {
                            radial + (Mat3::IDENTITY - outer) * (k * stretch_ratio)
                        } else {
                            radial
                        };
                    }
                }
            }
            if !c.is_spring {
                g_d += c.lambda * scale;
            }
            if c.damping > 0.0 && ctx.frame.h > 0.0 {
                let kd = scale * (c.damping / ctx.frame.h);
                let dd = d - c.start_d;
                match c.damp {
                    DampMode::None => {}
                    DampMode::Full => {
                        g_d += dd * kd;
                        h_d += Mat3::scalar(kd);
                    }
                    DampMode::Axis(n) => {
                        g_d += n * (kd * n.dot(dd));
                        h_d += Mat3::outer(n, n) * kd;
                    }
                }
            }
            let (entries, len) = combine(ctx, &c.a, &c.b, x);
            for &(b, _, jac) in &entries[..len] {
                (jac.transpose() * g_d).add_to(gradient, b as usize);
            }
            for &(row, _, jac_r) in &entries[..len] {
                let r_idx = find_idx(&c.blocks, c.block_count, row);
                let left = jac_r.transpose() * h_d;
                for &(col, _, jac_c) in &entries[..len] {
                    let c_idx = find_idx(&c.blocks, c.block_count, col);
                    values[c.slots[r_idx * c.block_count + c_idx]] += left * jac_c;
                }
            }
        }

        for s in &self.sliders {
            let d = s.a.position(ctx.layout, ctx.frame, x) - s.b.position(ctx.layout, ctx.frame, x);
            let coord = s.u0.dot(d);
            let u_outer = Mat3::outer(s.u0, s.u0);
            let mut g_d = Vec3::ZERO;
            let mut h_d = Mat3::ZERO;
            if s.planar_only {
                if s.stiffness > 0.0 {
                    let ks = scale * s.stiffness;
                    g_d += s.u0 * (ks * coord + scale * s.lambda.x);
                    h_d += u_outer * ks;
                }
            } else {
                if s.stiffness > 0.0 {
                    let kt = scale * s.stiffness;
                    let d_perp = d - s.u0 * coord;
                    g_d += d_perp * kt + s.lambda * scale;
                    h_d += (Mat3::IDENTITY - u_outer) * kt;
                }
                let (_, lim_g, lim_h) = limit_potential(coord, s.limits, s.stiffness);
                let (_, drv_g, drv_h) =
                    drive_potential(coord - s.drive.target, s.drive.stiffness, s.drive.max_force);
                let mut g_s = scale * (lim_g + drv_g);
                let mut h_s = scale * (lim_h + drv_h);
                if s.drive.damping > 0.0 && ctx.frame.h > 0.0 {
                    let kd = scale * (s.drive.damping / ctx.frame.h);
                    let s_ref = s.s0 + s.drive.target_velocity * ctx.frame.h;
                    g_s += kd * (coord - s_ref);
                    h_s += kd;
                }
                g_d += s.u0 * g_s;
                h_d += u_outer * h_s;
            }

            let (entries, len) = combine(ctx, &s.a, &s.b, x);
            for &(b, _, jac) in &entries[..len] {
                (jac.transpose() * g_d).add_to(gradient, b as usize);
            }
            for &(row, _, jac_r) in &entries[..len] {
                let r_idx = find_idx(&s.blocks, s.block_count, row);
                let left = jac_r.transpose() * h_d;
                for &(col, _, jac_c) in &entries[..len] {
                    let c_idx = find_idx(&s.blocks, s.block_count, col);
                    values[s.slots[r_idx * s.block_count + c_idx]] += left * jac_c;
                }
            }
        }

        for w in &self.welds {
            if w.stiffness <= 0.0 {
                continue;
            }
            let ba = ctx.layout.rigids[w.rigid].block;
            let ra = ctx.frame.rigids[w.rigid].radius;
            let (_, a_a) = ctx.frame.rigid(ctx.layout, w.rigid, x);
            let k = scale * w.stiffness;
            match w.target {
                WeldTarget::Rigid { rigid: rb_id, rel } if rb_id != w.rigid => {
                    let bb = ctx.layout.rigids[rb_id].block;
                    let rb = ctx.frame.rigids[rb_id].radius;
                    let (_, a_b) = ctx.frame.rigid(ctx.layout, rb_id, x);
                    let eff = (a_a - a_b * rel) * k + w.lambda * scale;
                    let ga = 0.5 / ra;
                    let gb = -0.5 / rb;
                    let ha = Mat3::scalar(k / (2.0 * ra * ra));
                    let hb = Mat3::scalar(k / (2.0 * rb * rb));
                    let hab = -k / (2.0 * ra * rb);
                    let n = w.block_count;
                    for col in 0..3 {
                        (eff.col(col) * ga).add_to(gradient, ba + 1 + col);
                        values[w.slots[col * n + col]] += ha;
                        values[w.slots[(3 + col) * n + (3 + col)]] += hb;
                    }
                    let eff_rt = eff * rel.transpose();
                    for j in 0..3 {
                        (eff_rt.col(j) * gb).add_to(gradient, bb + 1 + j);
                        for col in 0..3 {
                            let block = Mat3::scalar(hab * rel[(j, col)]);
                            values[w.slots[col * n + (3 + j)]] += block;
                            values[w.slots[(3 + j) * n + col]] += block;
                        }
                    }
                }
                _ => {
                    if let Some(r0) = Self::weld_world_target(ctx, w) {
                        let eff = (a_a - r0) * k + w.lambda * scale;
                        let ga = 0.5 / ra;
                        let ha = Mat3::scalar(k / (2.0 * ra * ra));
                        let n = w.block_count;
                        for col in 0..3 {
                            (eff.col(col) * ga).add_to(gradient, ba + 1 + col);
                            values[w.slots[col * n + col]] += ha;
                        }
                    }
                }
            }
        }

        for h in &self.hinges {
            let ra = ctx.frame.rigids[h.rigid].radius;
            let (_, a_a) = ctx.frame.rigid(ctx.layout, h.rigid, x);
            let a_b = Self::hinge_base_matrix(ctx, h, x);
            let n = h.block_count;

            let mut alpha = [0.0f64; 6];
            let mut beta = [0.0f64; 6];
            for c in 0..3 {
                alpha[c] = h.u_a[c] / ra;
                beta[c] = h.v_a[c] / ra;
            }
            if let HingeTarget::Rigid(rb_id) = h.target {
                if rb_id != h.rigid {
                    let rb = ctx.frame.rigids[rb_id].radius;
                    for c in 0..3 {
                        alpha[3 + c] = -h.u_b[c] / rb;
                        beta[3 + c] = -h.n_b0[c] / rb;
                    }
                }
            }

            let k_axis = scale * h.angular_stiffness.max(0.0);
            let err_u = a_a * h.u_a - a_b * h.u_b;
            let eff_u = if h.angular_stiffness > 0.0 {
                err_u * k_axis + h.lambda_u * scale
            } else {
                Vec3::ZERO
            };

            let theta = Self::hinge_angle(ctx, h, x);
            let (_, lim_g, lim_h) = limit_potential(theta, h.limits, h.angular_stiffness);
            let (_, drv_g, drv_h) =
                drive_potential(theta - h.drive.target, h.drive.stiffness, h.drive.max_force);
            let mut g_theta = scale * (lim_g + drv_g);
            let mut h_theta = scale * (lim_h + drv_h);
            if h.drive.damping > 0.0 && ctx.frame.h > 0.0 {
                let kd = scale * (h.drive.damping / ctx.frame.h);
                let t_ref = h.theta0 + h.drive.target_velocity * ctx.frame.h;
                g_theta += kd * (theta - t_ref);
                h_theta += kd;
            }
            let t_outer = Mat3::outer(h.t_world0, h.t_world0);

            for r in 0..n {
                let g = eff_u * alpha[r] + h.t_world0 * (g_theta * beta[r]);
                g.add_to(gradient, h.blocks[r]);
                for col in 0..n {
                    let block = Mat3::scalar(k_axis * alpha[r] * alpha[col])
                        + t_outer * (h_theta * beta[r] * beta[col]);
                    values[h.slots[r * n + col]] += block;
                }
            }
        }

        for u in &self.universals {
            if u.stiffness <= 0.0 {
                continue;
            }
            let ra = ctx.frame.rigids[u.rigid].radius;
            let (_, a_a) = ctx.frame.rigid(ctx.layout, u.rigid, x);
            let p_a = a_a * u.axis_a;
            let p_b = Self::universal_target_vec(ctx, u, x);
            let c = p_a.dot(p_b);
            let mut j = [Vec3::ZERO; 6];
            for (col, j_slot) in j.iter_mut().enumerate().take(3) {
                *j_slot = p_b * (u.axis_a[col] / ra);
            }
            if let UniversalTarget::Rigid {
                rigid: rb_id,
                axis_b,
            } = u.target
            {
                if rb_id != u.rigid {
                    let rb = ctx.frame.rigids[rb_id].radius;
                    for (col, j_slot) in j[3..6].iter_mut().enumerate() {
                        *j_slot = p_a * (axis_b[col] / rb);
                    }
                }
            }
            let g_c = scale * (u.stiffness * c + u.lambda);
            let k_c = scale * u.stiffness;
            let n = u.block_count;
            for r in 0..n {
                (j[r] * g_c).add_to(gradient, u.blocks[r]);
                for col in 0..n {
                    values[u.slots[r * n + col]] += Mat3::outer(j[r], j[col]) * k_c;
                }
            }
        }

        for t in &self.tendons {
            let n = t.blocks.len();
            if n == 0 || t.waypoints.len() < 2 {
                continue;
            }
            let mut total_len = 0.0;
            let mut d_lin = 0.0;
            let mut j_len = vec![Vec3::ZERO; n];
            let c_tot = t.damping + t.drive.damping;
            let damp_active = c_tot > 0.0 && ctx.frame.h > 0.0;
            let mut j_damp = if damp_active {
                vec![Vec3::ZERO; n]
            } else {
                Vec::new()
            };
            for (seg, pair) in t.waypoints.windows(2).enumerate() {
                let d = pair[1].position(ctx.layout, ctx.frame, x)
                    - pair[0].position(ctx.layout, ctx.frame, x);
                let len = d.length();
                total_len += len;
                d_lin += t.start_u[seg].dot(d - t.start_d[seg]);
                let dir = if len > 1e-12 { d / len } else { Vec3::ZERO };
                let (entries, count) = combine(ctx, &pair[1], &pair[0], x);
                for &(b, _, jac) in &entries[..count] {
                    let idx = t.blocks.binary_search(&b).unwrap();
                    if dir != Vec3::ZERO {
                        j_len[idx] += jac.transpose() * dir;
                    }
                    if damp_active {
                        j_damp[idx] += jac.transpose() * t.start_u[seg];
                    }
                }
            }
            let err = total_len - t.drive.target;
            let e_eff = if t.slack_only { err.max(0.0) } else { err };
            if !t.slack_only || err > 0.0 {
                let (_, drv_g, drv_h) =
                    drive_potential(e_eff, t.drive.stiffness, t.drive.max_force);
                let g_tens_phys = (if t.stiffness > 0.0 {
                    t.stiffness * e_eff
                } else {
                    0.0
                }) + drv_g;
                let g_tens = scale * g_tens_phys;
                let h_tens = scale * ((if t.stiffness > 0.0 { t.stiffness } else { 0.0 }) + drv_h);
                for r in 0..n {
                    let blk_r = t.blocks[r] as usize;
                    (j_len[r] * g_tens).add_to(gradient, blk_r);
                    for col in 0..n {
                        values[t.slots[r * n + col]] += Mat3::outer(j_len[r], j_len[col]) * h_tens;
                    }
                }
                if g_tens > 0.0 {
                    for pair in t.waypoints.windows(2) {
                        let d = pair[1].position(ctx.layout, ctx.frame, x)
                            - pair[0].position(ctx.layout, ctx.frame, x);
                        let len = d.length();
                        if len > 1e-12 {
                            let u = d / len;
                            let h_geo = (Mat3::IDENTITY - Mat3::outer(u, u)) * (g_tens / len);
                            let (entries, count) = combine(ctx, &pair[1], &pair[0], x);
                            for &(row, _, jac_r) in &entries[..count] {
                                let r_idx = t.blocks.binary_search(&row).unwrap();
                                let left = jac_r.transpose() * h_geo;
                                for &(col, _, jac_c) in &entries[..count] {
                                    let c_idx = t.blocks.binary_search(&col).unwrap();
                                    values[t.slots[r_idx * n + c_idx]] += left * jac_c;
                                }
                            }
                        }
                    }
                }
            }
            if damp_active {
                let kd = scale * (c_tot / ctx.frame.h);
                let err_d = d_lin - t.drive.target_velocity * ctx.frame.h;
                let g_damp = kd * err_d;
                for r in 0..n {
                    let blk_r = t.blocks[r] as usize;
                    (j_damp[r] * g_damp).add_to(gradient, blk_r);
                    for col in 0..n {
                        values[t.slots[r * n + col]] += Mat3::outer(j_damp[r], j_damp[col]) * kd;
                    }
                }
            }
        }
    }

    fn external_forces(&mut self, _ctx: &Context, _x: &[f64]) -> bool {
        self.external
    }
}
