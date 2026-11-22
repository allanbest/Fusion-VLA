//! The unknowns of a mechanics step, and how world state maps onto them.
//!
//! Every unknown is a 3-block, and every 3-block is a length:
//! - a solid's free node: its position;
//! - a rigid body: its centre of mass `c`, and the three columns of a matrix
//!   `A` times its radius of gyration `r` (affine body dynamics, Lan et al.
//!   2022). A point of the body is at `c + A X` — *linear* in the unknowns,
//!   so contact on a rigid body is as well-conditioned as on a mesh node,
//!   where a rotation's curvature would make Newton crawl. A stiff energy
//!   keeps `A` a rotation; the pose takes the nearest rotation after each
//!   stage.
//!
//! Anchored nodes and fixed objects are not unknowns.
//!
//! An [`Attach`] is a material point — a node, a point fixed in a rigid
//! body, a fixed place — whose position is a function of the unknowns; the
//! terms that act at points (contact, friction) work through it.

use core_math::{Mat3, Pose, Quat, Vec3};
use core_space::Shape;
use core_world::{Link, Representation, World};

/// No block: the node is held.
pub const HELD: u32 = u32::MAX;

/// Which unknowns belong to which object.
#[derive(Clone, Debug, Default)]
pub struct Layout {
    pub blocks: usize,
    pub solids: Vec<SolidDofs>,
    pub rigids: Vec<RigidDofs>,
    pub fixed: Vec<FixedBody>,
    /// For each selected object, where it went. Full layouts use world order.
    pub objects: Vec<Owner>,
    /// Selected world indices; empty for a full layout.
    pub indices: Vec<usize>,
    /// Each rigid body's blocks, `(first, 4)`, solved as one group.
    pub rigid_blocks: Vec<(usize, usize)>,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Owner {
    Fixed(usize),
    Rigid(usize),
    Solid(usize),
}

#[derive(Clone, Debug)]
pub struct SolidDofs {
    pub object: usize,
    /// Per node, its block or [`HELD`].
    pub block: Vec<u32>,
    /// Per node, where it is held (meaningful for held nodes only).
    pub held_at: Vec<Vec3>,
}

#[derive(Clone, Debug)]
pub struct RigidDofs {
    pub object: usize,
    /// The centre's block; `A`'s three columns are the next three.
    pub block: usize,
}

/// Blocks per rigid body.
pub const RIGID_BLOCKS: usize = 4;

#[derive(Clone, Debug)]
pub struct FixedBody {
    pub object: usize,
    pub shape: Shape,
    pub pose: Pose,
}

impl Layout {
    pub fn new(world: &World) -> Self {
        Self::from_indices(world, 0..world.objects().len())
    }

    /// A compact layout over a sorted subset, without copying world geometry.
    pub(crate) fn selected(world: &World, indices: Vec<usize>) -> Self {
        let mut layout = Self::from_indices(world, indices.iter().copied());
        layout.indices = indices;
        layout
    }

    pub(crate) fn object_index(&self, slot: usize) -> usize {
        if self.indices.is_empty() {
            slot
        } else {
            self.indices[slot]
        }
    }

    fn from_indices(world: &World, indices: impl Iterator<Item = usize>) -> Self {
        let mut layout = Self::default();
        for index in indices {
            let object = &world.objects()[index];
            let owner = match &object.representation {
                Representation::Fixed(f) => {
                    layout.fixed.push(FixedBody {
                        object: index,
                        shape: f.shape,
                        pose: f.pose,
                    });
                    Owner::Fixed(layout.fixed.len() - 1)
                }
                Representation::Rigid(_) => {
                    layout.rigids.push(RigidDofs {
                        object: index,
                        block: layout.blocks,
                    });
                    layout.rigid_blocks.push((layout.blocks, RIGID_BLOCKS));
                    layout.blocks += RIGID_BLOCKS;
                    Owner::Rigid(layout.rigids.len() - 1)
                }
                Representation::Solid(s) => {
                    let n = s.nodes.len();
                    let mut held_at = vec![Vec3::ZERO; n];
                    let mut held = vec![false; n];
                    for link in world.links() {
                        if let Link::Anchor { object, nodes, at } = link {
                            if object.0 as usize == index {
                                for (&i, &p) in nodes.iter().zip(at) {
                                    held[i as usize] = true;
                                    held_at[i as usize] = p;
                                }
                            }
                        }
                    }
                    let block = (0..n)
                        .map(|i| {
                            if held[i] {
                                HELD
                            } else {
                                layout.blocks += 1;
                                (layout.blocks - 1) as u32
                            }
                        })
                        .collect();
                    layout.solids.push(SolidDofs {
                        object: index,
                        block,
                        held_at,
                    });
                    Owner::Solid(layout.solids.len() - 1)
                }
            };
            layout.objects.push(owner);
        }
        layout
    }
}

/// Values of the unknowns: a vector, or a point along a step `x + t dx`
/// (evaluated block by block, without forming it).
pub trait Unknowns: Copy {
    fn block(self, b: usize) -> Vec3;
}

impl Unknowns for &[f64] {
    #[inline]
    fn block(self, b: usize) -> Vec3 {
        Vec3::read(self, b)
    }
}

/// `x + t dx`.
#[derive(Clone, Copy, Debug)]
pub struct Along<'a> {
    pub x: &'a [f64],
    pub dx: &'a [f64],
    pub t: f64,
}

impl Unknowns for Along<'_> {
    #[inline]
    fn block(self, b: usize) -> Vec3 {
        Vec3::read(self.x, b) + Vec3::read(self.dx, b) * self.t
    }
}

/// What a rigid body looks like during one stage.
#[derive(Clone, Debug)]
pub struct RigidFrame {
    pub mass: f64,
    /// Body-frame inertia about the centre of mass.
    pub inertia: Mat3,
    /// Body-frame second moment of mass `∫ρ X Xᵀ`, kg m².
    pub moment: Mat3,
    /// Radius of gyration: the unknowns hold `r A`.
    pub radius: f64,
    pub centre_start: Vec3,
    pub rotation_start: Quat,
    /// The matrix `A` the stage starts from (a rotation to within the
    /// rigidity's tolerance).
    pub matrix_start: Mat3,
    /// The orientation after torque-free motion over the stage.
    pub predicted: Quat,
    /// World inertia at the predicted orientation.
    pub world_inertia: Mat3,
    pub shape: Shape,
    /// Contact points in the body frame, with radii.
    pub points: Vec<(Vec3, f64)>,
    /// Largest distance from the centre of mass to the surface.
    pub reach: f64,
}

/// The constants of one step, beside the [`Layout`].
#[derive(Clone, Debug, Default)]
pub struct Frame {
    /// The stage's length, s.
    pub h: f64,
    /// What physical potentials are multiplied by in the incremental
    /// potential (`h²` for a whole implicit step; differs between the
    /// stages of a multi-stage integrator).
    pub weight: f64,
    /// The whole step's length, s — what contact stiffness is scaled to,
    /// so it is the same physical spring in every stage.
    pub step: f64,
    /// Per block: the unknowns at the start of the step.
    pub start: Vec<f64>,
    pub rigids: Vec<RigidFrame>,
}

impl Frame {
    /// A solid node's position.
    #[inline]
    pub fn node(&self, layout: &Layout, solid: usize, node: usize, x: impl Unknowns) -> Vec3 {
        let b = layout.solids[solid].block[node];
        if b == HELD {
            layout.solids[solid].held_at[node]
        } else {
            x.block(b as usize)
        }
    }

    /// A rigid body's centre and matrix `A`.
    #[inline]
    pub fn rigid(&self, layout: &Layout, rigid: usize, x: impl Unknowns) -> (Vec3, Mat3) {
        let b = layout.rigids[rigid].block;
        let scale = 1.0 / self.rigids[rigid].radius;
        (
            x.block(b),
            Mat3::from_cols(
                x.block(b + 1) * scale,
                x.block(b + 2) * scale,
                x.block(b + 3) * scale,
            ),
        )
    }
}

/// A material point whose place depends on the unknowns.
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum Attach {
    /// Node `node` of solid `solid`.
    Node { solid: u32, node: u32 },
    /// The point at `local` in rigid body `rigid`'s own frame.
    Rigid { rigid: u32, local: Vec3 },
    /// Does not move.
    Still(Vec3),
}

/// Up to four `(block, ∂position/∂block)` pairs.
#[derive(Clone, Copy, Debug)]
pub struct Jacobian {
    pub entries: [(u32, Mat3); 4],
    pub len: usize,
}

impl Jacobian {
    pub const NONE: Self = Self {
        entries: [(0, Mat3::ZERO); 4],
        len: 0,
    };

    pub fn iter(&self) -> impl Iterator<Item = &(u32, Mat3)> {
        self.entries[..self.len].iter()
    }
}

impl Attach {
    pub fn position(&self, layout: &Layout, frame: &Frame, x: impl Unknowns) -> Vec3 {
        match *self {
            Attach::Node { solid, node } => frame.node(layout, solid as usize, node as usize, x),
            Attach::Rigid { rigid, local } => {
                let (c, a) = frame.rigid(layout, rigid as usize, x);
                c + a * local
            }
            Attach::Still(p) => p,
        }
    }

    /// Where it was at the start of the step.
    pub fn start(&self, layout: &Layout, frame: &Frame) -> Vec3 {
        match *self {
            Attach::Node { .. } => self.position(layout, frame, &frame.start[..]),
            Attach::Rigid { rigid, local } => {
                let f = &frame.rigids[rigid as usize];
                f.centre_start + f.matrix_start * local
            }
            Attach::Still(p) => p,
        }
    }

    /// How its position changes with the unknowns, at `x`.
    pub fn jacobian(&self, layout: &Layout, frame: &Frame, x: &[f64]) -> Jacobian {
        match *self {
            Attach::Node { solid, node } => {
                let b = layout.solids[solid as usize].block[node as usize];
                if b == HELD {
                    Jacobian::NONE
                } else {
                    let mut entries = [(0, Mat3::ZERO); 4];
                    entries[0] = (b, Mat3::IDENTITY);
                    Jacobian { entries, len: 1 }
                }
            }
            Attach::Rigid { rigid, local } => {
                // p = c + Σₖ Xₖ (sₖ / r): constant.
                let _ = x;
                let b = layout.rigids[rigid as usize].block as u32;
                let scale = 1.0 / frame.rigids[rigid as usize].radius;
                let entries = [
                    (b, Mat3::IDENTITY),
                    (b + 1, Mat3::scalar(local.x * scale)),
                    (b + 2, Mat3::scalar(local.y * scale)),
                    (b + 3, Mat3::scalar(local.z * scale)),
                ];
                Jacobian { entries, len: 4 }
            }
            Attach::Still(_) => Jacobian::NONE,
        }
    }

    /// A bound on how far it moves when the unknowns move by `dx` (any
    /// fraction of `dx` moves it by at most that fraction of this).
    pub fn reach(&self, layout: &Layout, frame: &Frame, x: &[f64], dx: &[f64]) -> f64 {
        match *self {
            Attach::Node { solid, node } => {
                let b = layout.solids[solid as usize].block[node as usize];
                if b == HELD {
                    0.0
                } else {
                    Vec3::read(dx, b as usize).length()
                }
            }
            Attach::Rigid { rigid, local } => {
                // Linear: δp = δc + Σₖ Xₖ δsₖ / r.
                let _ = x;
                let b = layout.rigids[rigid as usize].block;
                let scale = 1.0 / frame.rigids[rigid as usize].radius;
                Vec3::read(dx, b).length()
                    + scale
                        * (local.x.abs() * Vec3::read(dx, b + 1).length()
                            + local.y.abs() * Vec3::read(dx, b + 2).length()
                            + local.z.abs() * Vec3::read(dx, b + 3).length())
            }
            Attach::Still(_) => 0.0,
        }
    }
}
