//! Heat on the same grid the momentum solve uses.
//!
//! PLAN.md §11: "Because the substrate is already a lattice, adding physics is
//! adding fields, not adding systems." This file is the first cash of that
//! claim, and it is worth noticing how little there is of it. There is no new
//! spatial structure, no new neighbour search, and no new transfer — the
//! B-spline weights that carry momentum carry enthalpy, and the Laplacian is
//! six neighbours on a grid that already exists.
//!
//! ## Enthalpy, not temperature
//!
//! The field integrated here is specific enthalpy, `J/kg`. Temperature is
//! derived from it. That is not a stylistic choice: a melting material absorbs
//! energy at constant temperature, so temperature is *flat* through a phase
//! change and a scheme that integrates it has nowhere to put the latent heat.
//! Enthalpy stays monotonic in energy through the whole of it, and the phase
//! falls out. §11 names the enthalpy formulation for this reason.
//!
//! ## What is explicit and what that costs
//!
//! Forward Euler on the diffusion, which is stable only while
//! `dt < dx² / (6α)`. For everything in the material library that bound is tens
//! of seconds against a mechanical step of milliseconds — heat is *slow* — so
//! it is free in practice and it is still enforced, by sub-cycling, because
//! "free in practice" is a property of today's material library rather than of
//! the code.
//!
//! §11's table says "FVM/FDM diffusion, implicit". Implicit would remove the
//! bound entirely and costs a linear solve per step; with the explicit bound
//! three orders of magnitude away from the step already being taken, that solve
//! would buy nothing this version can measure. It is the right answer for a
//! conduction-dominated scene at a coarse timestep, and that scene is not one
//! Fusion runs yet.

use fusion_core::SolidSet;
use fusion_math::Vec3;

use crate::grid::{weights, Grid};

/// The diffusion number an explicit step is held below.
///
/// The 3D stability limit is 1/6. Half of it, because the bound is derived for
/// a uniform medium and the grid here has a free boundary running through it.
const SAFE_FOURIER: f32 = 0.5 / 6.0;

/// Diffuse heat through one solid for `h` seconds.
///
/// Called with the grid the mechanical passes have already filled, so the
/// masses are the ones the momentum solve used and the stencil is the one the
/// particles were at when it ran. Returns the sub-cycles taken.
pub fn diffuse(solid: &mut SolidSet, grid: &mut Grid, h: f32) -> u32 {
    let thermal = solid.params.material.thermal;
    let cell = grid.cell();
    if solid.is_empty() || grid.is_empty() || h <= 0.0 || thermal.conductivity <= 0.0 {
        return 0;
    }
    let alpha = thermal.diffusivity(solid.params.material.density);
    if alpha <= 0.0 {
        return 0;
    }

    // --- points to grid -------------------------------------------------
    //
    // Mass-weighted, so a node holds the specific enthalpy of the material
    // around it rather than a sum of it. Divided by the same mass the momentum
    // transfer used, which is what makes the two fields agree about where the
    // material is.
    for value in grid.enthalpy.iter_mut() {
        *value = 0.0;
    }
    for value in grid.enthalpy_mass.iter_mut() {
        *value = 0.0;
    }
    let point_mass = solid.params.point_mass();
    for p in 0..solid.len() {
        let local = grid.to_local(solid.particles.position[p]);
        let (base, fx) = grid.stencil(local);
        if !grid.holds(base) {
            continue;
        }
        let (wx, wy, wz) = (weights(fx.x), weights(fx.y), weights(fx.z));
        let enthalpy = solid.enthalpy[p];
        for (k, wk) in wz.iter().enumerate() {
            for (j, wj) in wy.iter().enumerate() {
                for (i, wi) in wx.iter().enumerate() {
                    let node = grid.index(base[0] + i, base[1] + j, base[2] + k);
                    // One summand, used twice. See [`Grid::enthalpy_mass`]:
                    // the numerator and the denominator of a weighted mean
                    // have to be accumulated from the same number, or the
                    // quotient carries the difference between them as a
                    // temperature.
                    let carried = (wi * wj * wk * point_mass) as f64;
                    grid.enthalpy[node] += carried * enthalpy;
                    grid.enthalpy_mass[node] += carried;
                }
            }
        }
    }
    for node in 0..grid.len() {
        if grid.has_mass(node) {
            grid.enthalpy[node] /= grid.enthalpy_mass[node];
        } else {
            grid.enthalpy[node] = 0.0;
        }
    }

    // How full each node is, as a fraction of the material a whole cell holds.
    //
    // A node where a solid meets the air holds a sliver of a cell, and both
    // halves of its heat balance shrink with it: less capacity to warm, and
    // less cross-section to conduct through. Scaling the conductance by the
    // emptier of the two nodes keeps them in step, and that is not only
    // tidiness — the ratio is what the stability bound is made of, so a
    // conductance that ignored it would make the timestep depend on how thin
    // the thinnest sliver in the scene happened to be.
    //
    // Left out, the bar in the test suite diffused its temperature mode to
    // nothing in a second where the heat equation says it should have kept a
    // third of it: the sliver nodes along its thin axis went unstable and the
    // noise ate the mode.
    let full = (solid.params.material.density * cell * cell * cell) as f64;
    let fullness = |node: usize| (grid.enthalpy_mass[node] / full.max(1e-12)).min(1.0);

    // Sub-cycled to stay inside the explicit bound, which with the scaling
    // above is the textbook `dx² / 6α` and independent of how thin the
    // material is.
    let conductance = (thermal.conductivity * cell) as f64;
    let limit = SAFE_FOURIER * cell * cell / alpha;
    let cycles = (h / limit).ceil().clamp(1.0, 64.0) as u32;
    let sub = (h / cycles as f32) as f64;

    // --- diffuse --------------------------------------------------------
    //
    // In **joules**, not in kelvin.
    //
    // The obvious form — add `alpha dt/dx² (T_j - T_i)` to each node's
    // temperature — moves the same number of kelvin between two nodes however
    // much material each of them holds, so a heavy node trading with a light
    // one creates energy out of nothing. It is a plausible-looking
    // discretization, it is what this was written as first, and it leaked half
    // a percent of a solid's heat per second in a scene that was doing nothing
    // but conducting.
    //
    // As a flux, each pair exchanges equal and opposite joules and the sum
    // over the grid is exactly zero. Dividing by the nodal mass afterwards
    // returns the specific enthalpy the points are carried in — and because
    // that is the same mass the transfer used, the round trip to the points
    // conserves as well: `sum_p m_p sum_i w_ip (Q_i / m_i)` is `sum_i Q_i`,
    // which is zero.
    grid.enthalpy_start.copy_from_slice(&grid.enthalpy);
    let dims = grid.dims();
    for _ in 0..cycles {
        for value in grid.enthalpy_delta.iter_mut() {
            *value = 0.0;
        }
        for k in 1..dims[2].saturating_sub(1) {
            for j in 1..dims[1].saturating_sub(1) {
                for i in 1..dims[0].saturating_sub(1) {
                    let node = grid.index(i, j, k);
                    if !grid.has_mass(node) {
                        continue;
                    }
                    let (here, _) = thermal.state_of(grid.enthalpy[node]);
                    let mut joules = 0.0;
                    for (di, dj, dk) in [
                        (1isize, 0isize, 0isize),
                        (-1, 0, 0),
                        (0, 1, 0),
                        (0, -1, 0),
                        (0, 0, 1),
                        (0, 0, -1),
                    ] {
                        let n = grid.index(
                            (i as isize + di) as usize,
                            (j as isize + dj) as usize,
                            (k as isize + dk) as usize,
                        );
                        // A node with no mass is not cold, it is *absent*. The
                        // free surface is insulated — no flux — which is what
                        // a solid in vacuum does. Treating an empty node as
                        // ambient instead would quietly hold the whole scene
                        // at room temperature and make the hottest point in a
                        // struck bar its coolest.
                        if !grid.has_mass(n) {
                            continue;
                        }
                        let (there, _) = thermal.state_of(grid.enthalpy[n]);
                        // `k A / dx` with `A = dx²` is `k dx`. Each pair is
                        // visited from both ends and receives equal and
                        // opposite joules, which is where the conservation
                        // comes from — it is a property of the expression
                        // rather than something to check afterwards.
                        let share = fullness(node).min(fullness(n));
                        joules += conductance * share * (there - here) * sub;
                    }
                    grid.enthalpy_delta[node] = joules;
                }
            }
        }
        for node in 0..grid.len() {
            if grid.has_mass(node) {
                grid.enthalpy[node] += grid.enthalpy_delta[node] / grid.enthalpy_mass[node];
            }
        }
    }

    // --- grid to points -------------------------------------------------
    //
    // The *change*, not the value. Transferring the value would round-trip the
    // field through the grid every substep, and each round trip is a smoothing
    // — a hot spot would spread at a rate set by the transfer rather than by
    // the conductivity, which is the same failure PIC has with velocity and
    // has the same fix.
    // The total across every sub-cycle, not the last one's — differenced
    // against the snapshot taken before the loop.
    for node in 0..grid.len() {
        grid.enthalpy_delta[node] = if grid.has_mass(node) {
            grid.enthalpy[node] - grid.enthalpy_start[node]
        } else {
            0.0
        };
    }
    for p in 0..solid.len() {
        let local = grid.to_local(solid.particles.position[p]);
        let (base, fx) = grid.stencil(local);
        if !grid.holds(base) {
            continue;
        }
        let (wx, wy, wz) = (weights(fx.x), weights(fx.y), weights(fx.z));
        let mut change = 0.0f64;
        for (k, wk) in wz.iter().enumerate() {
            for (j, wj) in wy.iter().enumerate() {
                for (i, wi) in wx.iter().enumerate() {
                    let node = grid.index(base[0] + i, base[1] + j, base[2] + k);
                    change += (wi * wj * wk) as f64 * grid.enthalpy_delta[node];
                }
            }
        }
        solid.enthalpy[p] += change;
    }
    cycles
}

/// Heat one material point by the work its yielding dissipated.
///
/// The Taylor–Quinney term: almost all the work spent on permanent deformation
/// leaves as heat rather than as stored dislocation energy. It is what makes
/// §11's demo work without anything being authored — a bar struck hard enough
/// to yield heats *because* it yielded, at the place it yielded, by an amount
/// the mechanics decided.
///
/// `P : ΔF_p` is a work density in the reference configuration, so multiplying
/// by the rest volume gives joules and dividing by the point's mass gives the
/// specific enthalpy this function returns.
///
/// Clamped at zero. The increment is a first-order estimate of an integral over
/// the substep, and a return map that *un*-yields a point — which the clamp on
/// singular values can do when a point springs back through the yield surface —
/// would otherwise report negative dissipation and cool the material down.
/// Plastic work is dissipation and dissipation has a sign.
#[inline]
pub fn plastic_heat(
    stress: fusion_math::Mat3,
    plastic_increment: fusion_math::Mat3,
    rest_volume: f32,
    point_mass: f32,
    coefficient: f32,
) -> f64 {
    if coefficient <= 0.0 || point_mass <= 0.0 {
        return 0.0;
    }
    let work = stress.double_dot(&plastic_increment).max(0.0);
    // Computed narrow and returned wide. The work itself is a well-scaled
    // `f32` product; what needs the width is the *sum* it is about to join,
    // which is an enthalpy five orders of magnitude larger than a substep's
    // worth of dissipation.
    (coefficient * work * rest_volume) as f64 / point_mass as f64
}

/// The thermal part of the deformation gradient, as a scalar stretch.
///
/// Isotropic expansion splits multiplicatively out of `F`: what the
/// constitutive model should see is `F / (1 + αΔT)`, because the material is
/// not under stress merely for having grown. Without the split a heated block
/// clamped at neither end still develops stress, which is the ghost force that
/// makes thermal expansion look like a bug.
#[inline]
pub fn thermal_stretch(expansion: f32, temperature: f32, reference: f32) -> f32 {
    (1.0 + expansion * (temperature - reference)).max(1e-3)
}

/// Where the heat in a solid is, summed for a diagnostic.
#[inline]
pub fn centre_of_heat(solid: &SolidSet) -> Vec3 {
    let mut weighted = Vec3::ZERO;
    let mut total = 0.0;
    for i in 0..solid.len() {
        let above_ambient = (solid.temperature(i) - fusion_core::solid::AMBIENT).max(0.0);
        weighted += solid.particles.translation(i) * above_ambient;
        total += above_ambient;
    }
    if total > 1e-6 {
        weighted / total
    } else {
        Vec3::ZERO
    }
}
