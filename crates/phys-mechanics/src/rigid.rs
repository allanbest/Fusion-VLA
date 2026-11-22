//! Rigid-body pieces: the torque-free rotation, and the points a body
//! touches the world with.

use core_math::{Mat3, Quat, Vec3};
use core_numerics::{BlockMatrix, Triplets};
use core_space::Shape;

use crate::term::{Context, Term};

/// The orientation after `h` seconds of torque-free motion with world
/// angular momentum `momentum`, for a body of `inertia` (its own frame).
///
/// Symplectic splitting of the free rigid body into rotations about its
/// principal axes (Dullweber, Leimkuhler & McLachlan 1997), in the order
/// 1 2 3 2 1: angular momentum is conserved exactly, energy to within a
/// bounded oscillation, for any step. A body spinning about a principal
/// axis turns at exactly its rate.
pub fn free_rotation(rotation: Quat, momentum: Vec3, inertia: &Mat3, h: f64) -> Quat {
    let (moments, axes) = inertia.symmetric_eigen();
    // Work in the principal frame: R = R_p Pᵀ with P the principal axes.
    let principal = Quat::from_mat3(&if axes.determinant() < 0.0 {
        Mat3::from_cols(axes.col(0), axes.col(1), -axes.col(2))
    } else {
        axes
    });
    let mut frame = rotation * principal; // body-principal to world
    let mut body = frame.rotate_inverse(momentum); // Π in the principal frame
    let turn = |frame: &mut Quat, body: &mut Vec3, axis: usize, dt: f64| {
        if moments[axis] <= 0.0 {
            return;
        }
        let mut e = Vec3::ZERO;
        e[axis] = 1.0;
        let angle = body[axis] / moments[axis] * dt;
        // The body turns by `angle` about its axis; Π turns the other way in
        // the body frame, so world momentum stays put.
        *frame = (*frame * Quat::from_axis_angle(e, angle)).normalize();
        *body = Quat::from_axis_angle(e, -angle).rotate(*body);
    };
    turn(&mut frame, &mut body, 0, 0.5 * h);
    turn(&mut frame, &mut body, 1, 0.5 * h);
    turn(&mut frame, &mut body, 2, h);
    turn(&mut frame, &mut body, 1, 0.5 * h);
    turn(&mut frame, &mut body, 0, 0.5 * h);
    (frame * principal.conjugate()).normalize()
}

/// A box's corners are rounded by this fraction of its smallest half-size.
pub const CORNER_ROUNDING: f64 = 0.02;

/// The points of a shape, in its own frame, that contact is checked at,
/// each with a radius: the surface is everything within the radius.
pub fn contact_points(shape: &Shape) -> Vec<(Vec3, f64)> {
    match *shape {
        Shape::Sphere { radius } => vec![(Vec3::ZERO, radius)],
        Shape::Capsule {
            radius,
            half_length,
        } => {
            vec![
                (Vec3::new(0.0, -half_length, 0.0), radius),
                (Vec3::new(0.0, half_length, 0.0), radius),
            ]
        }
        Shape::Cuboid { half } => {
            // The corners of a slightly smaller box, each with a radius: a
            // minimally rounded box. A sharp corner resting exactly on
            // another box's corner or edge has no defined normal (and an
            // aligned stack puts every corner there); a rounded one sits
            // clearly over the face below.
            let rounding = CORNER_ROUNDING * half.min_element();
            let inner = half - Vec3::splat(rounding);
            (0..8)
                .map(|c| {
                    (
                        Vec3::new(
                            sign(c & 1) * inner.x,
                            sign(c & 2) * inner.y,
                            sign(c & 4) * inner.z,
                        ),
                        rounding,
                    )
                })
                .collect()
        }
        Shape::HalfSpace { .. } => Vec::new(),
    }
}

fn sign(bit: usize) -> f64 {
    if bit != 0 {
        1.0
    } else {
        -1.0
    }
}

/// Contact edges in the body's frame, with the same rounding as its points
/// and signed distance. Spheres have no edges; a capsule has its centreline.
pub fn contact_edges(shape: &Shape) -> Vec<([Vec3; 2], f64)> {
    match *shape {
        Shape::Cuboid { .. } => {
            let corners = contact_points(shape);
            let mut edges = Vec::with_capacity(12);
            for a in 0..8 {
                for axis in [1, 2, 4] {
                    if a & axis == 0 {
                        edges.push(([corners[a].0, corners[a | axis].0], corners[a].1));
                    }
                }
            }
            edges
        }
        Shape::Capsule {
            radius,
            half_length,
        } if half_length > 0.0 => vec![(
            [
                Vec3::new(0.0, -half_length, 0.0),
                Vec3::new(0.0, half_length, 0.0),
            ],
            radius,
        )],
        _ => Vec::new(),
    }
}

/// A length characteristic of how far the body's points are from its centre
/// of mass: the radius of gyration. Rotation unknowns are scaled by it so
/// that every unknown in the solve is a length.
pub fn gyration_radius(mass: f64, inertia: &Mat3) -> f64 {
    (inertia.trace() / (2.0 * mass)).sqrt()
}

/// How much stiffer than its own rotational inertia a rigid body's
/// orthogonality energy is: enough that the matrix `A` stays a rotation to
/// well within the contact precision, and is projected onto one after each
/// stage.
const ORTHOGONALITY: f64 = ORTHOGONALITY_DEFAULT;
const ORTHOGONALITY_DEFAULT: f64 = 3e2;

/// The rigid bodies' own terms, on the matrix part `A` of each body's
/// unknowns (its centre's inertia is in [`crate::Inertia`]):
///
/// - **inertia** `(W/d²) ½ tr((A − Ã) J (A − Ã)ᵀ)`, with `Ã` the
///   torque-free orientation and `J = ∫ρ X Xᵀ` — for a small turn `δθ` off
///   `Ã` this is exactly `½ δθᵀ I δθ`;
/// - **rigidity** `k Σᵢⱼ (aᵢ·aⱼ − δᵢⱼ)²`, keeping `A` orthogonal.
#[derive(Clone, Debug, Default)]
pub struct RigidBodies {
    /// Per body: the slots of its 3×3 group of matrix-part blocks.
    slots: Vec<[usize; 9]>,
    pub(crate) exact_hessian: bool,
}

impl RigidBodies {
    /// `(W/d²)` and the physical rigidity weight `k` for body `r`.
    /// Scaled to `frame.step` rather than stage length `h` so TR-BDF2 stages
    /// share the same physical rigidity stiffness and rest without phantom
    /// stage-to-stage vertical velocity.
    fn weights(ctx: &Context, r: usize) -> (f64, f64) {
        let f = &ctx.frame.rigids[r];
        let inertial = ctx.frame.weight / (ctx.frame.h * ctx.frame.h);
        let step = if ctx.frame.step > 0.0 {
            ctx.frame.step
        } else {
            ctx.frame.h
        };
        let physical = ctx.frame.weight / (step * step);
        (inertial, ORTHOGONALITY * physical * f.moment.trace() / 3.0)
    }

    /// Body `r`'s `A` and `Ã`.
    fn matrices(ctx: &Context, r: usize, x: &[f64]) -> (Mat3, Mat3) {
        let (_, a) = ctx.frame.rigid(ctx.layout, r, x);
        (a, ctx.frame.rigids[r].predicted.to_mat3())
    }
}

impl Term for RigidBodies {
    fn name(&self) -> &'static str {
        "rigid bodies"
    }

    fn pattern(&self, ctx: &Context, entries: &mut Vec<(usize, usize)>) {
        for d in &ctx.layout.rigids {
            for k in 1..4 {
                for l in 1..4 {
                    entries.push((d.block + k, d.block + l));
                }
            }
        }
    }

    fn bind(&mut self, ctx: &Context, matrix: &BlockMatrix) {
        self.slots = ctx
            .layout
            .rigids
            .iter()
            .map(|d| {
                std::array::from_fn(|kl| {
                    matrix
                        .slot(d.block + 1 + kl / 3, d.block + 1 + kl % 3)
                        .unwrap()
                })
            })
            .collect();
    }

    fn energy(&mut self, ctx: &Context, x: &[f64]) -> f64 {
        let mut total = 0.0;
        for r in 0..ctx.layout.rigids.len() {
            let (inertial, rigidity) = Self::weights(ctx, r);
            let (a, predicted) = Self::matrices(ctx, r, x);
            let d = a - predicted;
            total += 0.5 * inertial * (d * ctx.frame.rigids[r].moment * d.transpose()).trace();
            let e = a.transpose() * a - Mat3::IDENTITY;
            total += rigidity * e.frobenius_squared();
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
        // Each body's gradient and 9×9 Hessian, in parallel; added in body
        // order.
        let exact_hessian = self.exact_hessian;
        let shares = ctx.pool.map_reduce(
            ctx.layout.rigids.len(),
            4,
            Vec::new(),
            |range| {
                range
                    .map(|r| Self::body(ctx, r, x, exact_hessian))
                    .collect::<Vec<_>>()
            },
            |mut a, b| {
                a.extend(b);
                a
            },
        );
        let values = fixed.values_mut();
        for (r, (g, h)) in shares.into_iter().enumerate() {
            let block = ctx.layout.rigids[r].block;
            for (k, gk) in g.iter().enumerate() {
                gk.add_to(gradient, block + 1 + k);
            }
            for kl in 0..9 {
                values[self.slots[r][kl]] += h[kl];
            }
        }
    }
}

impl RigidBodies {
    /// Body `r`'s orthogonality potential per unit stage weight, J: the
    /// solver-only elastic energy stored by a non-orthogonal `A`.
    pub(crate) fn rigidity_potential(ctx: &Context, r: usize, x: &[f64]) -> f64 {
        let (_, rigidity) = Self::weights(ctx, r);
        let (a, _) = Self::matrices(ctx, r, x);
        rigidity * (a.transpose() * a - Mat3::IDENTITY).frobenius_squared() / ctx.frame.weight
    }

    /// Body `r`'s gradient (per column block) and Hessian blocks.
    #[allow(clippy::needless_range_loop)] // k indexes columns, gradient and Hessian rows alike
    fn body(ctx: &Context, r: usize, x: &[f64], exact_hessian: bool) -> ([Vec3; 3], [Mat3; 9]) {
        let f = &ctx.frame.rigids[r];
        let (inertial, rigidity) = Self::weights(ctx, r);
        let (a, predicted) = Self::matrices(ctx, r, x);
        let j = f.moment;
        let e = a.transpose() * a - Mat3::IDENTITY;
        let aat = a * a.transpose();
        // With respect to the columns aₖ (dimensionless), then scaled to the
        // unknowns sₖ = r aₖ.
        let scale = 1.0 / f.radius;
        let mut gradient = [Vec3::ZERO; 3];
        let mut hessian = [[0.0; 9]; 9];
        for k in 0..3 {
            // Inertia: ∂/∂aₖ = inertial Σₗ Jₖₗ (aₗ − ãₗ); rigidity:
            // 4k Σⱼ eₖⱼ aⱼ.
            let mut g = Vec3::ZERO;
            for l in 0..3 {
                g += (a.col(l) - predicted.col(l)) * (inertial * j[(k, l)])
                    + a.col(l) * (4.0 * rigidity * e[(k, l)]);
            }
            gradient[k] = g * scale;
            for l in 0..3 {
                // ∂²/∂aₖ∂aₗ: inertial Jₖₗ I + 4k (δₖₗ AAᵀ + aₗ aₖᵀ + eₖₗ I).
                let mut block = Mat3::scalar(inertial * j[(k, l)] + 4.0 * rigidity * e[(k, l)])
                    + Mat3::outer(a.col(l), a.col(k)) * (4.0 * rigidity);
                if k == l {
                    block += aat * (4.0 * rigidity);
                }
                for i in 0..3 {
                    for m in 0..3 {
                        hessian[3 * k + i][3 * l + m] = block[(i, m)];
                    }
                }
            }
        }
        if !exact_hessian {
            core_numerics::dense::project_positive(&mut hessian);
        }
        let blocks = std::array::from_fn(|kl| {
            let (k, l) = (kl / 3, kl % 3);
            let mut block = Mat3::ZERO;
            for i in 0..3 {
                for m in 0..3 {
                    block[(i, m)] = hessian[3 * k + i][3 * l + m] * scale * scale;
                }
            }
            block
        });
        (gradient, blocks)
    }
}

/// `J = ∫ρ X Xᵀ` from the inertia tensor: `J = ½ tr(I) 1 − I`.
pub fn second_moment(inertia: &Mat3) -> Mat3 {
    Mat3::scalar(0.5 * inertia.trace()) - *inertia
}
