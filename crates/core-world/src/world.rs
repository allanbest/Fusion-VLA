//! The world: everything a simulation knows, as data.

use core_math::constants::STANDARD_GRAVITY;
use core_math::{Pose, Quat, Vec3};
use core_space::Shape;
use std::sync::Arc;

use crate::material::Material;
use crate::mesh::TetMesh;
use crate::object::{Fixed, Object, Representation, Rigid, Solid};

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct MaterialId(pub u32);

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct ObjectId(pub u32);

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct LinkId(pub u32);

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct ProbeId(pub u32);

/// Virtual robotics and instrumentation sensors attached to [`World`].
#[derive(Clone, Debug, PartialEq)]
pub enum Probe {
    /// Joint encoder reading scalar coordinate and generalized velocity of `link`.
    JointState { link: LinkId },
    /// 6-axis constraint force/torque sensor reading world-frame reaction `(force, torque)` on `link`.
    Wrench { link: LinkId },
    /// Distributed tactile contact-patch sensor on `object` (centered at `local_origin`,
    /// outward surface normal `local_normal`, radius `radius` m).
    Tactile {
        object: ObjectId,
        local_origin: Vec3,
        local_normal: Vec3,
        radius: f64,
    },
    /// 6-axis IMU mounted at `local` in `object`'s body frame reporting proper linear
    /// acceleration (`a - g`, m/s²) and body-frame angular velocity (rad/s).
    Imu { object: ObjectId, local: Vec3 },
}

/// Instantaneous sensor sample returned by `Mechanics::sample_probes`.
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum ProbeReading {
    JointState {
        coordinate: f64,
        velocity: f64,
    },
    Wrench {
        force: Vec3,
        torque: Vec3,
    },
    Tactile {
        normal_force: f64,
        total_force: Vec3,
        center_of_pressure: Option<Vec3>,
        active_contacts: usize,
    },
    Imu {
        linear_acceleration: Vec3,
        angular_velocity: Vec3,
    },
}

/// Where a mechanical link attaches on a body, deformable node, or fixed world point.
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum LinkPoint {
    World(Vec3),
    Body { object: ObjectId, local: Vec3 },
    Node { object: ObjectId, node: u32 },
}

impl LinkPoint {
    pub fn object(&self) -> Option<ObjectId> {
        match *self {
            Self::World(_) => None,
            Self::Body { object, .. } | Self::Node { object, .. } => Some(object),
        }
    }

    fn remap(self, indices: &[usize]) -> Option<Self> {
        match self {
            Self::World(p) => Some(Self::World(p)),
            Self::Body { object, local } => {
                indices
                    .binary_search(&(object.0 as usize))
                    .ok()
                    .map(|i| Self::Body {
                        object: ObjectId(i as u32),
                        local,
                    })
            }
            Self::Node { object, node } => {
                indices
                    .binary_search(&(object.0 as usize))
                    .ok()
                    .map(|i| Self::Node {
                        object: ObjectId(i as u32),
                        node,
                    })
            }
        }
    }
}

/// Proportional-derivative (PD) actuator or motor target on a 1-DOF joint.
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct JointDrive {
    /// Target joint coordinate (radians for [`Link::Hinge`], metres for [`Link::Slider`]).
    pub target: f64,
    /// Target joint velocity (rad/s for [`Link::Hinge`], m/s for [`Link::Slider`]).
    pub target_velocity: f64,
    /// Proportional stiffness `k_p` (N·m/rad for [`Link::Hinge`], N/m for [`Link::Slider`]).
    pub stiffness: f64,
    /// Derivative damping `k_d` (N·m·s/rad for [`Link::Hinge`], N·s/m for [`Link::Slider`]).
    pub damping: f64,
    /// Optional maximum actuator force/torque magnitude (N·m for [`Link::Hinge`], N for [`Link::Slider`]).
    pub max_force: Option<f64>,
}

impl JointDrive {
    pub const NONE: Self = Self {
        target: 0.0,
        target_velocity: 0.0,
        stiffness: 0.0,
        damping: 0.0,
        max_force: None,
    };

    pub const fn pd(target: f64, stiffness: f64, damping: f64) -> Self {
        Self {
            target,
            target_velocity: 0.0,
            stiffness,
            damping,
            max_force: None,
        }
    }

    pub const fn with_velocity(mut self, target_velocity: f64) -> Self {
        self.target_velocity = target_velocity;
        self
    }

    pub const fn with_max_force(mut self, max_force: f64) -> Self {
        self.max_force = Some(max_force);
        self
    }

    fn same_bits(&self, other: &Self) -> bool {
        self.target.to_bits() == other.target.to_bits()
            && self.target_velocity.to_bits() == other.target_velocity.to_bits()
            && self.stiffness.to_bits() == other.stiffness.to_bits()
            && self.damping.to_bits() == other.damping.to_bits()
            && self.max_force.map(f64::to_bits) == other.max_force.map(f64::to_bits)
    }
}

/// A connection between objects, or between an object and the world.
#[derive(Clone, Debug, PartialEq)]
pub enum Link {
    /// Holds nodes of a solid at fixed places.
    Anchor {
        object: ObjectId,
        nodes: Vec<u32>,
        at: Vec<Vec3>,
    },
    /// Kelvin–Voigt elastic spring between two points (`stiffness` in N/m,
    /// `damping` in N·s/m).
    Spring {
        a: LinkPoint,
        b: LinkPoint,
        rest_length: f64,
        stiffness: f64,
        damping: f64,
    },
    /// Spherical ball-and-socket joint penalizing separation (`stiffness` in N/m).
    BallJoint {
        a: LinkPoint,
        b: LinkPoint,
        stiffness: f64,
    },
    /// Rigid weld holding `local_a` on `a` to `local_b` on `b` (or world
    /// coordinates when `b` is `None`) with relative orientation
    /// `A_a ≈ A_b * rest_rotation` (`rotational_stiffness` in N·m/rad).
    Weld {
        a: ObjectId,
        b: Option<ObjectId>,
        local_a: Vec3,
        local_b: Vec3,
        rest_rotation: Quat,
        stiffness: f64,
        rotational_stiffness: f64,
    },
    /// Revolute (1-DOF rotational) hinge holding `local_a` on `a` to `local_b`
    /// on `b` (or world when `b` is `None`), aligning `axis_a` in `a` with
    /// `axis_b` in `b`, with optional angle `limits` (rad) and PD `drive`.
    Hinge {
        a: ObjectId,
        b: Option<ObjectId>,
        local_a: Vec3,
        local_b: Vec3,
        axis_a: Vec3,
        axis_b: Vec3,
        rest_rotation: Quat,
        stiffness: f64,
        angular_stiffness: f64,
        limits: Option<(f64, f64)>,
        drive: JointDrive,
    },
    /// Prismatic (1-DOF linear) slider guiding `local_a` on `a` along unit
    /// `axis_b` in `b`'s frame (or world when `b` is `None`) through `local_b`,
    /// locking relative orientation to `rest_rotation`, with optional position
    /// `limits` (m) and PD `drive`.
    Slider {
        a: ObjectId,
        b: Option<ObjectId>,
        local_a: Vec3,
        local_b: Vec3,
        axis_b: Vec3,
        rest_rotation: Quat,
        stiffness: f64,
        rotational_stiffness: f64,
        limits: Option<(f64, f64)>,
        drive: JointDrive,
    },
    /// Cylindrical (2-DOF: axial translation + rotation about `axis_a`/`axis_b`) joint.
    Cylindrical {
        a: ObjectId,
        b: Option<ObjectId>,
        local_a: Vec3,
        local_b: Vec3,
        axis_a: Vec3,
        axis_b: Vec3,
        rest_rotation: Quat,
        stiffness: f64,
        angular_stiffness: f64,
        linear_limits: Option<(f64, f64)>,
        angular_limits: Option<(f64, f64)>,
        linear_drive: JointDrive,
        angular_drive: JointDrive,
    },
    /// Universal / Hooke (2-DOF rotational, twist-free) joint holding `local_a` on `a`
    /// to `local_b` on `b` (or world when `b` is `None`) and keeping cross-pin axes
    /// `axis_a` (in `a`) and `axis_b` (in `b`) perpendicular.
    Universal {
        a: ObjectId,
        b: Option<ObjectId>,
        local_a: Vec3,
        local_b: Vec3,
        axis_a: Vec3,
        axis_b: Vec3,
        stiffness: f64,
        angular_stiffness: f64,
    },
    /// Planar (3-DOF SE(2)) joint confining `local_a` on `a` to the plane through
    /// `local_b` on `b` with unit normal `normal_b`, aligning `normal_a` with `normal_b`.
    Planar {
        a: ObjectId,
        b: Option<ObjectId>,
        local_a: Vec3,
        local_b: Vec3,
        normal_a: Vec3,
        normal_b: Vec3,
        stiffness: f64,
        angular_stiffness: f64,
    },
    /// 1D routed tendon/cable passing sequentially through `waypoints`, penalizing
    /// total path length `L = Σ |p_{k+1} - p_k|` (optionally unilateral `slack_only`
    /// tension) with Kelvin–Voigt damping and [`JointDrive`].
    Tendon {
        waypoints: Vec<LinkPoint>,
        stiffness: f64,
        damping: f64,
        slack_only: bool,
        drive: JointDrive,
    },
}

fn vec_bits_eq(a: Vec3, b: Vec3) -> bool {
    a.to_array().map(f64::to_bits) == b.to_array().map(f64::to_bits)
}

fn quat_bits_eq(a: Quat, b: Quat) -> bool {
    [a.x, a.y, a.z, a.w].map(f64::to_bits) == [b.x, b.y, b.z, b.w].map(f64::to_bits)
}

fn limits_bits_eq(a: Option<(f64, f64)>, b: Option<(f64, f64)>) -> bool {
    a.map(|(lo, hi)| (lo.to_bits(), hi.to_bits())) == b.map(|(lo, hi)| (lo.to_bits(), hi.to_bits()))
}

fn point_bits_eq(a: LinkPoint, b: LinkPoint) -> bool {
    match (a, b) {
        (LinkPoint::World(pa), LinkPoint::World(pb)) => vec_bits_eq(pa, pb),
        (
            LinkPoint::Body {
                object: oa,
                local: la,
            },
            LinkPoint::Body {
                object: ob,
                local: lb,
            },
        ) => oa == ob && vec_bits_eq(la, lb),
        (
            LinkPoint::Node {
                object: oa,
                node: na,
            },
            LinkPoint::Node {
                object: ob,
                node: nb,
            },
        ) => oa == ob && na == nb,
        _ => false,
    }
}

impl Link {
    pub fn remap(&self, indices: &[usize]) -> Option<Self> {
        match self {
            Self::Anchor { object, nodes, at } => indices
                .binary_search(&(object.0 as usize))
                .ok()
                .map(|i| Self::Anchor {
                    object: ObjectId(i as u32),
                    nodes: nodes.clone(),
                    at: at.clone(),
                }),
            Self::Spring {
                a,
                b,
                rest_length,
                stiffness,
                damping,
            } => Some(Self::Spring {
                a: a.remap(indices)?,
                b: b.remap(indices)?,
                rest_length: *rest_length,
                stiffness: *stiffness,
                damping: *damping,
            }),
            Self::BallJoint { a, b, stiffness } => Some(Self::BallJoint {
                a: a.remap(indices)?,
                b: b.remap(indices)?,
                stiffness: *stiffness,
            }),
            Self::Weld {
                a,
                b,
                local_a,
                local_b,
                rest_rotation,
                stiffness,
                rotational_stiffness,
            } => {
                let a = indices.binary_search(&(a.0 as usize)).ok()?;
                let b = match b {
                    Some(b) => Some(ObjectId(indices.binary_search(&(b.0 as usize)).ok()? as u32)),
                    None => None,
                };
                Some(Self::Weld {
                    a: ObjectId(a as u32),
                    b,
                    local_a: *local_a,
                    local_b: *local_b,
                    rest_rotation: *rest_rotation,
                    stiffness: *stiffness,
                    rotational_stiffness: *rotational_stiffness,
                })
            }
            Self::Hinge {
                a,
                b,
                local_a,
                local_b,
                axis_a,
                axis_b,
                rest_rotation,
                stiffness,
                angular_stiffness,
                limits,
                drive,
            } => {
                let a = indices.binary_search(&(a.0 as usize)).ok()?;
                let b = match b {
                    Some(b) => Some(ObjectId(indices.binary_search(&(b.0 as usize)).ok()? as u32)),
                    None => None,
                };
                Some(Self::Hinge {
                    a: ObjectId(a as u32),
                    b,
                    local_a: *local_a,
                    local_b: *local_b,
                    axis_a: *axis_a,
                    axis_b: *axis_b,
                    rest_rotation: *rest_rotation,
                    stiffness: *stiffness,
                    angular_stiffness: *angular_stiffness,
                    limits: *limits,
                    drive: *drive,
                })
            }
            Self::Slider {
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
                let a = indices.binary_search(&(a.0 as usize)).ok()?;
                let b = match b {
                    Some(b) => Some(ObjectId(indices.binary_search(&(b.0 as usize)).ok()? as u32)),
                    None => None,
                };
                Some(Self::Slider {
                    a: ObjectId(a as u32),
                    b,
                    local_a: *local_a,
                    local_b: *local_b,
                    axis_b: *axis_b,
                    rest_rotation: *rest_rotation,
                    stiffness: *stiffness,
                    rotational_stiffness: *rotational_stiffness,
                    limits: *limits,
                    drive: *drive,
                })
            }
            Self::Cylindrical {
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
                let a = indices.binary_search(&(a.0 as usize)).ok()?;
                let b = match b {
                    Some(b) => Some(ObjectId(indices.binary_search(&(b.0 as usize)).ok()? as u32)),
                    None => None,
                };
                Some(Self::Cylindrical {
                    a: ObjectId(a as u32),
                    b,
                    local_a: *local_a,
                    local_b: *local_b,
                    axis_a: *axis_a,
                    axis_b: *axis_b,
                    rest_rotation: *rest_rotation,
                    stiffness: *stiffness,
                    angular_stiffness: *angular_stiffness,
                    linear_limits: *linear_limits,
                    angular_limits: *angular_limits,
                    linear_drive: *linear_drive,
                    angular_drive: *angular_drive,
                })
            }
            Self::Universal {
                a,
                b,
                local_a,
                local_b,
                axis_a,
                axis_b,
                stiffness,
                angular_stiffness,
            } => {
                let a = indices.binary_search(&(a.0 as usize)).ok()?;
                let b = match b {
                    Some(b) => Some(ObjectId(indices.binary_search(&(b.0 as usize)).ok()? as u32)),
                    None => None,
                };
                Some(Self::Universal {
                    a: ObjectId(a as u32),
                    b,
                    local_a: *local_a,
                    local_b: *local_b,
                    axis_a: *axis_a,
                    axis_b: *axis_b,
                    stiffness: *stiffness,
                    angular_stiffness: *angular_stiffness,
                })
            }
            Self::Planar {
                a,
                b,
                local_a,
                local_b,
                normal_a,
                normal_b,
                stiffness,
                angular_stiffness,
            } => {
                let a = indices.binary_search(&(a.0 as usize)).ok()?;
                let b = match b {
                    Some(b) => Some(ObjectId(indices.binary_search(&(b.0 as usize)).ok()? as u32)),
                    None => None,
                };
                Some(Self::Planar {
                    a: ObjectId(a as u32),
                    b,
                    local_a: *local_a,
                    local_b: *local_b,
                    normal_a: *normal_a,
                    normal_b: *normal_b,
                    stiffness: *stiffness,
                    angular_stiffness: *angular_stiffness,
                })
            }
            Self::Tendon {
                waypoints,
                stiffness,
                damping,
                slack_only,
                drive,
            } => {
                let mut remapped = Vec::with_capacity(waypoints.len());
                for pt in waypoints {
                    remapped.push(pt.remap(indices)?);
                }
                Some(Self::Tendon {
                    waypoints: remapped,
                    stiffness: *stiffness,
                    damping: *damping,
                    slack_only: *slack_only,
                    drive: *drive,
                })
            }
        }
    }

    pub fn same_bits(&self, other: &Self) -> bool {
        match (self, other) {
            (
                Self::Anchor {
                    object: oa,
                    nodes: na,
                    at: ata,
                },
                Self::Anchor {
                    object: ob,
                    nodes: nb,
                    at: atb,
                },
            ) => {
                oa == ob
                    && na == nb
                    && ata.len() == atb.len()
                    && ata.iter().zip(atb).all(|(&x, &y)| vec_bits_eq(x, y))
            }
            (
                Self::Spring {
                    a: aa,
                    b: ba,
                    rest_length: ra,
                    stiffness: sa,
                    damping: da,
                },
                Self::Spring {
                    a: ab,
                    b: bb,
                    rest_length: rb,
                    stiffness: sb,
                    damping: db,
                },
            ) => {
                point_bits_eq(*aa, *ab)
                    && point_bits_eq(*ba, *bb)
                    && ra.to_bits() == rb.to_bits()
                    && sa.to_bits() == sb.to_bits()
                    && da.to_bits() == db.to_bits()
            }
            (
                Self::BallJoint {
                    a: aa,
                    b: ba,
                    stiffness: sa,
                },
                Self::BallJoint {
                    a: ab,
                    b: bb,
                    stiffness: sb,
                },
            ) => point_bits_eq(*aa, *ab) && point_bits_eq(*ba, *bb) && sa.to_bits() == sb.to_bits(),
            (
                Self::Weld {
                    a: aa,
                    b: ba,
                    local_a: laa,
                    local_b: lba,
                    rest_rotation: qa,
                    stiffness: sa,
                    rotational_stiffness: rsa,
                },
                Self::Weld {
                    a: ab,
                    b: bb,
                    local_a: lab,
                    local_b: lbb,
                    rest_rotation: qb,
                    stiffness: sb,
                    rotational_stiffness: rsb,
                },
            ) => {
                aa == ab
                    && ba == bb
                    && vec_bits_eq(*laa, *lab)
                    && vec_bits_eq(*lba, *lbb)
                    && quat_bits_eq(*qa, *qb)
                    && sa.to_bits() == sb.to_bits()
                    && rsa.to_bits() == rsb.to_bits()
            }
            (
                Self::Hinge {
                    a: aa,
                    b: ba,
                    local_a: laa,
                    local_b: lba,
                    axis_a: axa,
                    axis_b: bxa,
                    rest_rotation: qa,
                    stiffness: sa,
                    angular_stiffness: asa,
                    limits: lima,
                    drive: dra,
                },
                Self::Hinge {
                    a: ab,
                    b: bb,
                    local_a: lab,
                    local_b: lbb,
                    axis_a: axb,
                    axis_b: bxb,
                    rest_rotation: qb,
                    stiffness: sb,
                    angular_stiffness: asb,
                    limits: limb,
                    drive: drb,
                },
            ) => {
                aa == ab
                    && ba == bb
                    && vec_bits_eq(*laa, *lab)
                    && vec_bits_eq(*lba, *lbb)
                    && vec_bits_eq(*axa, *axb)
                    && vec_bits_eq(*bxa, *bxb)
                    && quat_bits_eq(*qa, *qb)
                    && sa.to_bits() == sb.to_bits()
                    && asa.to_bits() == asb.to_bits()
                    && limits_bits_eq(*lima, *limb)
                    && dra.same_bits(drb)
            }
            (
                Self::Slider {
                    a: aa,
                    b: ba,
                    local_a: laa,
                    local_b: lba,
                    axis_b: bxa,
                    rest_rotation: qa,
                    stiffness: sa,
                    rotational_stiffness: rsa,
                    limits: lima,
                    drive: dra,
                },
                Self::Slider {
                    a: ab,
                    b: bb,
                    local_a: lab,
                    local_b: lbb,
                    axis_b: bxb,
                    rest_rotation: qb,
                    stiffness: sb,
                    rotational_stiffness: rsb,
                    limits: limb,
                    drive: drb,
                },
            ) => {
                aa == ab
                    && ba == bb
                    && vec_bits_eq(*laa, *lab)
                    && vec_bits_eq(*lba, *lbb)
                    && vec_bits_eq(*bxa, *bxb)
                    && quat_bits_eq(*qa, *qb)
                    && sa.to_bits() == sb.to_bits()
                    && rsa.to_bits() == rsb.to_bits()
                    && limits_bits_eq(*lima, *limb)
                    && dra.same_bits(drb)
            }
            (
                Self::Cylindrical {
                    a: aa,
                    b: ba,
                    local_a: laa,
                    local_b: lba,
                    axis_a: axa,
                    axis_b: bxa,
                    rest_rotation: qa,
                    stiffness: sa,
                    angular_stiffness: asa,
                    linear_limits: lla,
                    angular_limits: ala,
                    linear_drive: lda,
                    angular_drive: ada,
                },
                Self::Cylindrical {
                    a: ab,
                    b: bb,
                    local_a: lab,
                    local_b: lbb,
                    axis_a: axb,
                    axis_b: bxb,
                    rest_rotation: qb,
                    stiffness: sb,
                    angular_stiffness: asb,
                    linear_limits: llb,
                    angular_limits: alb,
                    linear_drive: ldb,
                    angular_drive: adb,
                },
            ) => {
                aa == ab
                    && ba == bb
                    && vec_bits_eq(*laa, *lab)
                    && vec_bits_eq(*lba, *lbb)
                    && vec_bits_eq(*axa, *axb)
                    && vec_bits_eq(*bxa, *bxb)
                    && quat_bits_eq(*qa, *qb)
                    && sa.to_bits() == sb.to_bits()
                    && asa.to_bits() == asb.to_bits()
                    && limits_bits_eq(*lla, *llb)
                    && limits_bits_eq(*ala, *alb)
                    && lda.same_bits(ldb)
                    && ada.same_bits(adb)
            }
            (
                Self::Universal {
                    a: aa,
                    b: ba,
                    local_a: laa,
                    local_b: lba,
                    axis_a: axa,
                    axis_b: bxa,
                    stiffness: sa,
                    angular_stiffness: asa,
                },
                Self::Universal {
                    a: ab,
                    b: bb,
                    local_a: lab,
                    local_b: lbb,
                    axis_a: axb,
                    axis_b: bxb,
                    stiffness: sb,
                    angular_stiffness: asb,
                },
            ) => {
                aa == ab
                    && ba == bb
                    && vec_bits_eq(*laa, *lab)
                    && vec_bits_eq(*lba, *lbb)
                    && vec_bits_eq(*axa, *axb)
                    && vec_bits_eq(*bxa, *bxb)
                    && sa.to_bits() == sb.to_bits()
                    && asa.to_bits() == asb.to_bits()
            }
            (
                Self::Planar {
                    a: aa,
                    b: ba,
                    local_a: laa,
                    local_b: lba,
                    normal_a: naa,
                    normal_b: nba,
                    stiffness: sa,
                    angular_stiffness: asa,
                },
                Self::Planar {
                    a: ab,
                    b: bb,
                    local_a: lab,
                    local_b: lbb,
                    normal_a: nab,
                    normal_b: nbb,
                    stiffness: sb,
                    angular_stiffness: asb,
                },
            ) => {
                aa == ab
                    && ba == bb
                    && vec_bits_eq(*laa, *lab)
                    && vec_bits_eq(*lba, *lbb)
                    && vec_bits_eq(*naa, *nab)
                    && vec_bits_eq(*nba, *nbb)
                    && sa.to_bits() == sb.to_bits()
                    && asa.to_bits() == asb.to_bits()
            }
            (
                Self::Tendon {
                    waypoints: wa,
                    stiffness: sa,
                    damping: da,
                    slack_only: sla,
                    drive: dra,
                },
                Self::Tendon {
                    waypoints: wb,
                    stiffness: sb,
                    damping: db,
                    slack_only: slb,
                    drive: drb,
                },
            ) => {
                wa.len() == wb.len()
                    && wa.iter().zip(wb).all(|(&x, &y)| point_bits_eq(x, y))
                    && sa.to_bits() == sb.to_bits()
                    && da.to_bits() == db.to_bits()
                    && sla == slb
                    && dra.same_bits(drb)
            }
            _ => false,
        }
    }
}

/// The scene: materials, objects, links, probes, and the environment.
#[derive(Clone, Debug)]
pub struct World {
    /// Uniform gravitational acceleration, m/s².
    pub gravity: Vec3,
    /// Simulated time, s.
    pub time: f64,
    materials: Arc<Vec<Material>>,
    objects: Vec<Object>,
    links: Vec<Link>,
    excluded_contacts: Vec<(ObjectId, ObjectId)>,
    probes: Vec<Probe>,
}

impl Default for World {
    fn default() -> Self {
        Self::new()
    }
}

impl World {
    /// An empty world under standard gravity along −y.
    pub fn new() -> Self {
        Self {
            gravity: Vec3::new(0.0, -STANDARD_GRAVITY, 0.0),
            time: 0.0,
            materials: Arc::new(Vec::new()),
            objects: Vec::new(),
            links: Vec::new(),
            excluded_contacts: Vec::new(),
            probes: Vec::new(),
        }
    }

    /// A compact independently writable view. Geometry/materials are shared
    /// through Arc; selected object state and remapped anchors are copied.
    /// Indices must be sorted and unique. Object/link IDs are local to the view.
    pub fn subset(&self, indices: &[usize]) -> Self {
        assert!(indices.windows(2).all(|w| w[0] < w[1]));
        let links = self
            .links
            .iter()
            .filter_map(|link| link.remap(indices))
            .collect();
        let excluded_contacts = self
            .excluded_contacts
            .iter()
            .filter_map(|&(a, b)| {
                let ia = ObjectId(indices.binary_search(&(a.0 as usize)).ok()? as u32);
                let ib = ObjectId(indices.binary_search(&(b.0 as usize)).ok()? as u32);
                Some(if ia <= ib { (ia, ib) } else { (ib, ia) })
            })
            .collect();
        Self {
            gravity: self.gravity,
            time: self.time,
            materials: self.materials.clone(),
            objects: indices.iter().map(|&i| self.objects[i].clone()).collect(),
            links,
            excluded_contacts,
            probes: Vec::new(),
        }
    }

    /// Refresh an existing subset without copying geometry or material data.
    /// As with solver layouts, geometry/material edits require invalidation.
    pub fn refresh_subset(&mut self, source: &Self, indices: &[usize]) {
        assert_eq!(self.objects.len(), indices.len());
        self.gravity = source.gravity;
        self.time = source.time;
        for (object, &i) in self.objects.iter_mut().zip(indices) {
            match (
                &mut object.representation,
                &source.objects[i].representation,
            ) {
                (Representation::Rigid(a), Representation::Rigid(b)) => {
                    a.pose = b.pose;
                    a.velocity = b.velocity;
                    a.spin = b.spin;
                }
                (Representation::Solid(a), Representation::Solid(b)) => {
                    a.nodes.position.clone_from(&b.nodes.position);
                    a.nodes.velocity.clone_from(&b.nodes.velocity);
                    a.nodes.columns.clone_from_columns(&b.nodes.columns);
                    a.elements.clone_from_columns(&b.elements);
                }
                (Representation::Fixed(_), Representation::Fixed(_)) => {}
                _ => panic!("subset layout changed without invalidation"),
            }
        }
        let mut slot = 0;
        for link in &source.links {
            if let Some(remapped) = link.remap(indices) {
                assert!(
                    slot < self.links.len(),
                    "subset link count changed without invalidation"
                );
                if !self.links[slot].same_bits(&remapped) {
                    self.links[slot] = remapped;
                }
                slot += 1;
            }
        }
        assert_eq!(
            slot,
            self.links.len(),
            "subset link count changed without invalidation"
        );
    }

    /// Whether the remapped links of `source` match `self.links` bitwise.
    pub fn subset_links_match(&self, source: &Self, indices: &[usize]) -> bool {
        let mut iter = source.links.iter().filter_map(|link| link.remap(indices));
        for local in &self.links {
            let Some(remapped) = iter.next() else {
                return false;
            };
            if !local.same_bits(&remapped) {
                return false;
            }
        }
        iter.next().is_none()
    }

    /// Bitwise physical state comparison for an existing compact view.
    pub fn subset_dynamics_match(&self, source: &Self, indices: &[usize]) -> bool {
        let vec_eq =
            |a: Vec3, b: Vec3| a.to_array().map(f64::to_bits) == b.to_array().map(f64::to_bits);
        let quat_eq = |a: core_math::Quat, b: core_math::Quat| {
            [a.x, a.y, a.z, a.w].map(f64::to_bits) == [b.x, b.y, b.z, b.w].map(f64::to_bits)
        };
        self.subset_links_match(source, indices)
            && self.objects.iter().zip(indices).all(|(a, &i)| {
                match (&a.representation, &source.objects[i].representation) {
                    (Representation::Solid(a), Representation::Solid(b)) => {
                        a.nodes
                            .position
                            .iter()
                            .zip(&b.nodes.position)
                            .all(|(&a, &b)| vec_eq(a, b))
                            && a.nodes
                                .velocity
                                .iter()
                                .zip(&b.nodes.velocity)
                                .all(|(&a, &b)| vec_eq(a, b))
                            && a.nodes.columns == b.nodes.columns
                            && a.elements == b.elements
                    }
                    (Representation::Rigid(a), Representation::Rigid(b)) => {
                        vec_eq(a.pose.position, b.pose.position)
                            && quat_eq(a.pose.rotation, b.pose.rotation)
                            && vec_eq(a.velocity, b.velocity)
                            && vec_eq(a.spin, b.spin)
                    }
                    (Representation::Fixed(a), Representation::Fixed(b)) => {
                        a.shape == b.shape
                            && vec_eq(a.pose.position, b.pose.position)
                            && quat_eq(a.pose.rotation, b.pose.rotation)
                    }
                    _ => false,
                }
            })
    }

    /// Publish only dynamic state from a completed subset. The caller decides
    /// whether the entire requested interval can be committed atomically.
    pub fn commit_subset(&mut self, subset: &Self, indices: &[usize]) {
        assert_eq!(subset.objects.len(), indices.len());
        for (object, &i) in subset.objects.iter().zip(indices) {
            match (&object.representation, &mut self.objects[i].representation) {
                (Representation::Rigid(a), Representation::Rigid(b)) => {
                    b.pose = a.pose;
                    b.velocity = a.velocity;
                    b.spin = a.spin;
                }
                (Representation::Solid(a), Representation::Solid(b)) => {
                    b.nodes.position.clone_from(&a.nodes.position);
                    b.nodes.velocity.clone_from(&a.nodes.velocity);
                    b.nodes.columns.clone_from_columns(&a.nodes.columns);
                    b.elements.clone_from_columns(&a.elements);
                }
                (Representation::Fixed(_), Representation::Fixed(_)) => {}
                _ => panic!("subset layout changed without invalidation"),
            }
        }
    }

    pub fn add_material(&mut self, material: Material) -> MaterialId {
        Arc::make_mut(&mut self.materials).push(material);
        MaterialId(self.materials.len() as u32 - 1)
    }

    fn add(
        &mut self,
        name: &str,
        material: MaterialId,
        representation: Representation,
    ) -> ObjectId {
        assert!(
            (material.0 as usize) < self.materials.len(),
            "unknown material"
        );
        self.objects.push(Object {
            name: name.into(),
            material,
            representation,
        });
        ObjectId(self.objects.len() as u32 - 1)
    }

    /// Something that does not move.
    pub fn add_fixed(
        &mut self,
        name: &str,
        material: MaterialId,
        shape: Shape,
        pose: Pose,
    ) -> ObjectId {
        self.add(name, material, Representation::Fixed(Fixed { shape, pose }))
    }

    /// A rigid body with its centre of mass at `pose`, at rest.
    pub fn add_rigid(
        &mut self,
        name: &str,
        material: MaterialId,
        shape: Shape,
        pose: Pose,
    ) -> ObjectId {
        let unit = shape
            .unit_mass()
            .expect("a rigid body needs a bounded shape");
        let density = self.material(material).density;
        let rigid = Rigid {
            shape,
            pose,
            velocity: Vec3::ZERO,
            spin: Vec3::ZERO,
            mass: density * unit.volume,
            inertia: unit.inertia * density,
        };
        self.add(name, material, Representation::Rigid(rigid))
    }

    /// A deformable solid: `mesh` placed at `pose`, at rest.
    pub fn add_solid(
        &mut self,
        name: &str,
        material: MaterialId,
        mesh: TetMesh,
        pose: Pose,
    ) -> ObjectId {
        let solid = Solid::new(mesh, &pose, self.material(material).density);
        self.add(name, material, Representation::Solid(solid))
    }

    /// Sets an object moving as a rigid whole: `linear` at its centre of
    /// mass, turning at `angular` (rad/s) about it.
    pub fn set_velocity(&mut self, id: ObjectId, linear: Vec3, angular: Vec3) {
        let centre = self.object(id).centre_of_mass();
        match &mut self.objects[id.0 as usize].representation {
            Representation::Fixed(_) => panic!("a fixed object cannot move"),
            Representation::Rigid(r) => {
                r.velocity = linear;
                r.spin = angular;
            }
            Representation::Solid(s) => {
                let centre = centre.unwrap();
                for (v, p) in s.nodes.velocity.iter_mut().zip(&s.nodes.position) {
                    *v = linear + angular.cross(*p - centre);
                }
            }
        }
    }

    /// Holds the nodes of a solid that `select` accepts (by current
    /// position) where they are. `None` if it accepts none.
    pub fn anchor(&mut self, id: ObjectId, select: impl Fn(Vec3) -> bool) -> Option<LinkId> {
        let Representation::Solid(s) = &self.object(id).representation else {
            panic!("only a solid's nodes can be anchored; make the object fixed instead");
        };
        let nodes: Vec<u32> = (0..s.nodes.len() as u32)
            .filter(|&i| select(s.nodes.position[i as usize]))
            .collect();
        if nodes.is_empty() {
            return None;
        }
        let at = nodes
            .iter()
            .map(|&i| s.nodes.position[i as usize])
            .collect();
        self.links.push(Link::Anchor {
            object: id,
            nodes,
            at,
        });
        Some(LinkId(self.links.len() as u32 - 1))
    }

    pub fn add_spring(
        &mut self,
        a: LinkPoint,
        b: LinkPoint,
        rest_length: f64,
        stiffness: f64,
        damping: f64,
    ) -> LinkId {
        self.links.push(Link::Spring {
            a,
            b,
            rest_length,
            stiffness,
            damping,
        });
        LinkId(self.links.len() as u32 - 1)
    }

    pub fn add_ball_joint(&mut self, a: LinkPoint, b: LinkPoint, stiffness: f64) -> LinkId {
        self.links.push(Link::BallJoint { a, b, stiffness });
        LinkId(self.links.len() as u32 - 1)
    }

    #[allow(clippy::too_many_arguments)]
    pub fn add_weld(
        &mut self,
        a: ObjectId,
        b: Option<ObjectId>,
        local_a: Vec3,
        local_b: Vec3,
        rest_rotation: Quat,
        stiffness: f64,
        rotational_stiffness: f64,
    ) -> LinkId {
        self.links.push(Link::Weld {
            a,
            b,
            local_a,
            local_b,
            rest_rotation,
            stiffness,
            rotational_stiffness,
        });
        LinkId(self.links.len() as u32 - 1)
    }

    /// Adds a 1-DOF revolute hinge between `a` and `b` (`None` for the world).
    /// `axis_a` is the hinge axis in `a`'s local frame; at zero hinge angle
    /// (`q_a = q_b * rest_rotation`), `b`'s axis is `rest_rotation.rotate(axis_a)`.
    #[allow(clippy::too_many_arguments)]
    pub fn add_hinge(
        &mut self,
        a: ObjectId,
        b: Option<ObjectId>,
        local_a: Vec3,
        local_b: Vec3,
        axis_a: Vec3,
        rest_rotation: Quat,
        stiffness: f64,
        angular_stiffness: f64,
        limits: Option<(f64, f64)>,
        drive: JointDrive,
    ) -> LinkId {
        let axis_a = axis_a.normalize();
        let rest_rotation = rest_rotation.normalize();
        let axis_b = rest_rotation.rotate(axis_a);
        self.links.push(Link::Hinge {
            a,
            b,
            local_a,
            local_b,
            axis_a,
            axis_b,
            rest_rotation,
            stiffness,
            angular_stiffness,
            limits,
            drive,
        });
        LinkId(self.links.len() as u32 - 1)
    }

    /// Adds a 1-DOF prismatic slider guiding `local_a` on `a` along `axis_b`
    /// in `b`'s local frame (or world frame when `b` is `None`).
    #[allow(clippy::too_many_arguments)]
    pub fn add_slider(
        &mut self,
        a: ObjectId,
        b: Option<ObjectId>,
        local_a: Vec3,
        local_b: Vec3,
        axis_b: Vec3,
        rest_rotation: Quat,
        stiffness: f64,
        rotational_stiffness: f64,
        limits: Option<(f64, f64)>,
        drive: JointDrive,
    ) -> LinkId {
        self.links.push(Link::Slider {
            a,
            b,
            local_a,
            local_b,
            axis_b: axis_b.normalize(),
            rest_rotation: rest_rotation.normalize(),
            stiffness,
            rotational_stiffness,
            limits,
            drive,
        });
        LinkId(self.links.len() as u32 - 1)
    }

    /// Adds a 2-DOF cylindrical joint (translation along and rotation around `axis_a`/`axis_b`).
    #[allow(clippy::too_many_arguments)]
    pub fn add_cylindrical(
        &mut self,
        a: ObjectId,
        b: Option<ObjectId>,
        local_a: Vec3,
        local_b: Vec3,
        axis_a: Vec3,
        rest_rotation: Quat,
        stiffness: f64,
        angular_stiffness: f64,
        linear_limits: Option<(f64, f64)>,
        angular_limits: Option<(f64, f64)>,
        linear_drive: JointDrive,
        angular_drive: JointDrive,
    ) -> LinkId {
        let axis_a = axis_a.normalize();
        let rest_rotation = rest_rotation.normalize();
        let axis_b = rest_rotation.rotate(axis_a);
        self.links.push(Link::Cylindrical {
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
        });
        LinkId(self.links.len() as u32 - 1)
    }

    /// Adds a 2-DOF universal (Hooke) joint holding `local_a` on `a` to `local_b` on `b`
    /// (`None` for world) and constraining orthogonal cross pins `axis_a` and `axis_b`.
    #[allow(clippy::too_many_arguments)]
    pub fn add_universal(
        &mut self,
        a: ObjectId,
        b: Option<ObjectId>,
        local_a: Vec3,
        local_b: Vec3,
        axis_a: Vec3,
        axis_b: Vec3,
        stiffness: f64,
        angular_stiffness: f64,
    ) -> LinkId {
        self.links.push(Link::Universal {
            a,
            b,
            local_a,
            local_b,
            axis_a: axis_a.normalize(),
            axis_b: axis_b.normalize(),
            stiffness,
            angular_stiffness,
        });
        LinkId(self.links.len() as u32 - 1)
    }

    /// Adds a 3-DOF planar SE(2) joint confining `local_a` on `a` to the plane
    /// through `local_b` on `b` (`None` for world) with normal `normal_b`,
    /// aligning `normal_a` with `normal_b`.
    #[allow(clippy::too_many_arguments)]
    pub fn add_planar(
        &mut self,
        a: ObjectId,
        b: Option<ObjectId>,
        local_a: Vec3,
        local_b: Vec3,
        normal_a: Vec3,
        normal_b: Vec3,
        stiffness: f64,
        angular_stiffness: f64,
    ) -> LinkId {
        self.links.push(Link::Planar {
            a,
            b,
            local_a,
            local_b,
            normal_a: normal_a.normalize(),
            normal_b: normal_b.normalize(),
            stiffness,
            angular_stiffness,
        });
        LinkId(self.links.len() as u32 - 1)
    }

    /// Adds a 1D routed tendon/cable passing through `waypoints` (`waypoints.len() >= 2`).
    pub fn add_tendon(
        &mut self,
        waypoints: Vec<LinkPoint>,
        stiffness: f64,
        damping: f64,
        slack_only: bool,
        drive: JointDrive,
    ) -> LinkId {
        assert!(waypoints.len() >= 2, "tendon requires at least 2 waypoints");
        self.links.push(Link::Tendon {
            waypoints,
            stiffness,
            damping,
            slack_only,
            drive,
        });
        LinkId(self.links.len() as u32 - 1)
    }

    /// Adds a virtual instrumentation [`Probe`] to the world.
    pub fn add_probe(&mut self, probe: Probe) -> ProbeId {
        self.probes.push(probe);
        ProbeId(self.probes.len() as u32 - 1)
    }

    pub fn probes(&self) -> &[Probe] {
        &self.probes
    }

    pub fn probe(&self, id: ProbeId) -> &Probe {
        &self.probes[id.0 as usize]
    }

    pub fn link(&self, id: LinkId) -> &Link {
        &self.links[id.0 as usize]
    }

    pub fn link_mut(&mut self, id: LinkId) -> &mut Link {
        &mut self.links[id.0 as usize]
    }

    /// Updates the target coordinate of a driven hinge (rad), slider (m),
    /// tendon target length (m), or spring rest length (m) in place without
    /// rebuilding solver topology.
    pub fn set_joint_target(&mut self, id: LinkId, target: f64) {
        match &mut self.links[id.0 as usize] {
            Link::Hinge { drive, .. } | Link::Slider { drive, .. } | Link::Tendon { drive, .. } => {
                drive.target = target;
            }
            Link::Cylindrical { linear_drive, .. } => {
                linear_drive.target = target;
            }
            Link::Spring { rest_length, .. } => {
                *rest_length = target;
            }
            _ => panic!(
                "set_joint_target requires a Hinge, Slider, Cylindrical, Tendon, or Spring link"
            ),
        }
    }

    /// Updates both the linear (m) and angular (rad) targets of a [`Link::Cylindrical`].
    pub fn set_cylindrical_targets(&mut self, id: LinkId, linear_target: f64, angular_target: f64) {
        let Link::Cylindrical {
            linear_drive,
            angular_drive,
            ..
        } = &mut self.links[id.0 as usize]
        else {
            panic!("set_cylindrical_targets requires a Cylindrical link");
        };
        linear_drive.target = linear_target;
        angular_drive.target = angular_target;
    }

    /// Replaces the PD drive of a [`Link::Hinge`], [`Link::Slider`], or [`Link::Tendon`] in place.
    pub fn set_joint_drive(&mut self, id: LinkId, new_drive: JointDrive) {
        match &mut self.links[id.0 as usize] {
            Link::Hinge { drive, .. } | Link::Slider { drive, .. } | Link::Tendon { drive, .. } => {
                *drive = new_drive;
            }
            _ => panic!("set_joint_drive requires a Hinge, Slider, or Tendon link"),
        }
    }

    /// Updates a spring's rest length in place.
    pub fn set_spring_rest_length(&mut self, id: LinkId, new_rest_length: f64) {
        let Link::Spring { rest_length, .. } = &mut self.links[id.0 as usize] else {
            panic!("set_spring_rest_length requires a Spring link");
        };
        *rest_length = new_rest_length;
    }

    /// Updates the relative target orientation of a [`Link::Weld`], [`Link::Hinge`],
    /// or [`Link::Slider`] in place.
    pub fn set_weld_rotation(&mut self, id: LinkId, rotation: Quat) {
        let q = rotation.normalize();
        match &mut self.links[id.0 as usize] {
            Link::Weld { rest_rotation, .. } | Link::Slider { rest_rotation, .. } => {
                *rest_rotation = q;
            }
            Link::Hinge {
                axis_a,
                axis_b,
                rest_rotation,
                ..
            } => {
                *rest_rotation = q;
                *axis_b = q.rotate(*axis_a);
            }
            _ => panic!("set_weld_rotation requires a Weld, Hinge, or Slider link"),
        }
    }

    fn point_position(&self, pt: LinkPoint) -> Vec3 {
        match pt {
            LinkPoint::World(p) => p,
            LinkPoint::Body { object, local } => {
                match &self.objects[object.0 as usize].representation {
                    Representation::Fixed(f) => f.pose.transform_point(local),
                    Representation::Rigid(r) => r.pose.transform_point(local),
                    Representation::Solid(_) => {
                        panic!("LinkPoint::Body requires fixed or rigid object")
                    }
                }
            }
            LinkPoint::Node { object, node } => {
                match &self.objects[object.0 as usize].representation {
                    Representation::Solid(s) => s.nodes.position[node as usize],
                    _ => panic!("LinkPoint::Node requires a deformable solid"),
                }
            }
        }
    }

    fn point_velocity(&self, pt: LinkPoint) -> Vec3 {
        match pt {
            LinkPoint::World(_) => Vec3::ZERO,
            LinkPoint::Body { object, local } => {
                match &self.objects[object.0 as usize].representation {
                    Representation::Fixed(_) => Vec3::ZERO,
                    Representation::Rigid(r) => {
                        r.velocity + r.spin.cross(r.pose.rotation.rotate(local))
                    }
                    Representation::Solid(_) => {
                        panic!("LinkPoint::Body requires fixed or rigid object")
                    }
                }
            }
            LinkPoint::Node { object, node } => {
                match &self.objects[object.0 as usize].representation {
                    Representation::Solid(s) => s.nodes.velocity[node as usize],
                    _ => panic!("LinkPoint::Node requires a deformable solid"),
                }
            }
        }
    }

    fn object_pose_spin(&self, id: ObjectId) -> (Pose, Vec3) {
        match &self.objects[id.0 as usize].representation {
            Representation::Fixed(f) => (f.pose, Vec3::ZERO),
            Representation::Rigid(r) => (r.pose, r.spin),
            Representation::Solid(_) => panic!("joint requires fixed or rigid object"),
        }
    }

    /// Current scalar joint coordinate: hinge angle in `(-π, π]` (rad), slider
    /// or cylindrical axial displacement (m), tendon total path length (m), or
    /// spring elongation `|p_a - p_b| - L_0` (m).
    pub fn joint_coordinate(&self, id: LinkId) -> Option<f64> {
        match self.links.get(id.0 as usize)? {
            Link::Hinge {
                a,
                b,
                axis_a,
                rest_rotation,
                ..
            } => {
                let (pose_a, _) = self.object_pose_spin(*a);
                let pose_b = b.map_or(Pose::IDENTITY, |obj| self.object_pose_spin(obj).0);
                let (_, v_a, w_a) = ortho_basis(*axis_a);
                let r_rest = rest_rotation.normalize();
                let v_b = pose_b.rotation.rotate(r_rest.rotate(v_a));
                let w_b = pose_b.rotation.rotate(r_rest.rotate(w_a));
                let va_world = pose_a.rotation.rotate(v_a);
                Some(va_world.dot(w_b).atan2(va_world.dot(v_b)))
            }
            Link::Slider {
                a,
                b,
                local_a,
                local_b,
                axis_b,
                ..
            }
            | Link::Cylindrical {
                a,
                b,
                local_a,
                local_b,
                axis_b,
                ..
            } => {
                let pa = self.point_position(LinkPoint::Body {
                    object: *a,
                    local: *local_a,
                });
                let (pb, pose_b) = match *b {
                    Some(obj) => (
                        self.point_position(LinkPoint::Body {
                            object: obj,
                            local: *local_b,
                        }),
                        self.object_pose_spin(obj).0,
                    ),
                    None => (*local_b, Pose::IDENTITY),
                };
                let u_world = pose_b.rotation.rotate(axis_b.normalize());
                Some(u_world.dot(pa - pb))
            }
            Link::Tendon { waypoints, .. } => {
                let mut total = 0.0;
                for pair in waypoints.windows(2) {
                    total += (self.point_position(pair[1]) - self.point_position(pair[0])).length();
                }
                Some(total)
            }
            Link::Spring {
                a, b, rest_length, ..
            } => {
                let d = self.point_position(*a) - self.point_position(*b);
                Some(d.length() - *rest_length)
            }
            _ => None,
        }
    }

    /// Current angular coordinate (rad) of a [`Link::Cylindrical`] or [`Link::Hinge`].
    pub fn cylindrical_angle(&self, id: LinkId) -> Option<f64> {
        match self.links.get(id.0 as usize)? {
            Link::Hinge {
                a,
                b,
                axis_a,
                rest_rotation,
                ..
            }
            | Link::Cylindrical {
                a,
                b,
                axis_a,
                rest_rotation,
                ..
            } => {
                let (pose_a, _) = self.object_pose_spin(*a);
                let pose_b = b.map_or(Pose::IDENTITY, |obj| self.object_pose_spin(obj).0);
                let (_, v_a, w_a) = ortho_basis(*axis_a);
                let r_rest = rest_rotation.normalize();
                let v_b = pose_b.rotation.rotate(r_rest.rotate(v_a));
                let w_b = pose_b.rotation.rotate(r_rest.rotate(w_a));
                let va_world = pose_a.rotation.rotate(v_a);
                Some(va_world.dot(w_b).atan2(va_world.dot(v_b)))
            }
            _ => None,
        }
    }

    /// Current scalar joint velocity: hinge angular rate (rad/s), slider or
    /// cylindrical axial rate (m/s), tendon elongation rate (m/s), or spring
    /// elongation rate (m/s).
    pub fn joint_velocity(&self, id: LinkId) -> Option<f64> {
        match self.links.get(id.0 as usize)? {
            Link::Hinge { a, b, axis_b, .. } => {
                let (_, spin_a) = self.object_pose_spin(*a);
                let (pose_b, spin_b) = b.map_or((Pose::IDENTITY, Vec3::ZERO), |obj| {
                    self.object_pose_spin(obj)
                });
                let u_world = pose_b.rotation.rotate(axis_b.normalize());
                Some((spin_a - spin_b).dot(u_world))
            }
            Link::Slider {
                a,
                b,
                local_a,
                local_b,
                axis_b,
                ..
            }
            | Link::Cylindrical {
                a,
                b,
                local_a,
                local_b,
                axis_b,
                ..
            } => {
                let va = self.point_velocity(LinkPoint::Body {
                    object: *a,
                    local: *local_a,
                });
                let (vb, pose_b) = match *b {
                    Some(obj) => (
                        self.point_velocity(LinkPoint::Body {
                            object: obj,
                            local: *local_b,
                        }),
                        self.object_pose_spin(obj).0,
                    ),
                    None => (Vec3::ZERO, Pose::IDENTITY),
                };
                let u_world = pose_b.rotation.rotate(axis_b.normalize());
                Some(u_world.dot(va - vb))
            }
            Link::Tendon { waypoints, .. } => {
                let mut rate = 0.0;
                for pair in waypoints.windows(2) {
                    let d = self.point_position(pair[1]) - self.point_position(pair[0]);
                    let dv = self.point_velocity(pair[1]) - self.point_velocity(pair[0]);
                    rate += d.try_normalize().map_or(0.0, |u| u.dot(dv));
                }
                Some(rate)
            }
            Link::Spring { a, b, .. } => {
                let d = self.point_position(*a) - self.point_position(*b);
                let dv = self.point_velocity(*a) - self.point_velocity(*b);
                Some(d.try_normalize().map_or(0.0, |u| u.dot(dv)))
            }
            _ => None,
        }
    }

    pub fn material(&self, id: MaterialId) -> &Material {
        &self.materials[id.0 as usize]
    }

    pub fn material_mut(&mut self, id: MaterialId) -> &mut Material {
        &mut Arc::make_mut(&mut self.materials)[id.0 as usize]
    }

    pub fn materials(&self) -> &[Material] {
        &self.materials
    }

    pub fn materials_mut(&mut self) -> &mut [Material] {
        Arc::make_mut(&mut self.materials).as_mut_slice()
    }

    pub fn object(&self, id: ObjectId) -> &Object {
        &self.objects[id.0 as usize]
    }

    pub fn object_mut(&mut self, id: ObjectId) -> &mut Object {
        &mut self.objects[id.0 as usize]
    }

    pub fn objects(&self) -> &[Object] {
        &self.objects
    }

    pub fn objects_mut(&mut self) -> &mut [Object] {
        &mut self.objects
    }

    pub fn ids(&self) -> impl Iterator<Item = ObjectId> {
        (0..self.objects.len() as u32).map(ObjectId)
    }

    pub fn links(&self) -> &[Link] {
        &self.links
    }

    /// Excludes collision detection between objects `a` and `b`.
    pub fn exclude_contact(&mut self, a: ObjectId, b: ObjectId) {
        if a == b {
            return;
        }
        let pair = if a < b { (a, b) } else { (b, a) };
        if !self.excluded_contacts.contains(&pair) {
            self.excluded_contacts.push(pair);
        }
    }

    pub fn excluded_contacts(&self) -> &[(ObjectId, ObjectId)] {
        &self.excluded_contacts
    }

    /// Conserved quantities of everything that moves.
    pub fn totals(&self) -> Totals {
        let moving = self.objects.iter().filter(|o| o.is_dynamic());
        let mut t = Totals::default();
        for o in moving {
            t.mass += o.mass();
            t.momentum += o.momentum();
            t.angular_momentum += o.angular_momentum();
            t.kinetic_energy += o.kinetic_energy();
            t.gravitational_energy += o.gravitational_energy(self.gravity);
        }
        t
    }
}

/// Deterministic right-handed orthonormal basis `(u, v, w)` with `u = axis.normalize()`.
pub fn ortho_basis(axis: Vec3) -> (Vec3, Vec3, Vec3) {
    let u = axis.normalize();
    let seed = if u.x.abs() < 0.9 { Vec3::X } else { Vec3::Y };
    let v = u.cross(seed).normalize();
    let w = u.cross(v);
    (u, v, w)
}

/// The audit: what physics says is conserved, or changes only by known
/// amounts.
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct Totals {
    /// kg.
    pub mass: f64,
    /// kg m/s.
    pub momentum: Vec3,
    /// About the origin, kg m²/s.
    pub angular_momentum: Vec3,
    /// J.
    pub kinetic_energy: f64,
    /// In the uniform field, relative to the origin, J.
    pub gravitational_energy: f64,
}
