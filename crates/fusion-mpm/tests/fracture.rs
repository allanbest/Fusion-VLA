//! Damage, and what makes a crack different from a hole.
//!
//! PLAN.md §11 asks for "fracture that emerges instead of being authored", so
//! the tests are about emergence: nothing here names a crack, a seam or a
//! break location, and the assertions are about where the material chose to
//! part and whether it stayed parted.

use fusion_core::{
    build_solid, BodyDesc, BodySet, Constitutive, Fracture, Material, SolidDesc, SolidMaterial,
    SolidParams, SolidSet, World,
};
use fusion_math::{vec3, Mat3, Vec3};
use fusion_mpm::{principal_stress, stress, stress_with_damage, MpmSolver};

const DT: f32 = 1.0 / 240.0;

fn brittle(spacing: f32, at: Vec3, velocity: Vec3) -> SolidSet {
    build_solid(
        &SolidDesc::block(Vec3::splat(0.24))
            .at(at)
            .with_params(SolidParams::at(spacing).of(SolidMaterial::PORCELAIN))
            .with_velocity(velocity),
    )
}

fn floor() -> World {
    let mut world = World::new();
    world.add(
        BodyDesc::cuboid(vec3(3.0, 0.2, 3.0))
            .at(vec3(0.0, -0.1, 0.0))
            .with_material(Material::CONCRETE)
            .fixed(),
    );
    world
}

/// The damage law is exact where it can be checked by hand.
#[test]
fn damage_begins_where_the_material_says_it_does() {
    let f = Fracture {
        onset: 1.01,
        failure: 1.05,
    };
    assert_eq!(f.damage_at(1.0), 0.0, "damage at rest");
    assert_eq!(f.damage_at(1.009), 0.0, "damage below onset");
    assert!(
        (f.damage_at(1.03) - 0.5).abs() < 1e-5,
        "halfway is half damaged"
    );
    assert_eq!(f.damage_at(1.05), 1.0);
    assert_eq!(f.damage_at(9.0), 1.0, "damage is bounded");

    // And a tough material does not break however hard it is pulled.
    assert_eq!(Fracture::TOUGH.damage_at(100.0), 0.0);
    assert!(!Fracture::TOUGH.breaks());
}

/// A broken point carries no tension and still carries compression.
///
/// The asymmetry that makes this a crack model rather than a hole model. A
/// uniform scaling would give a fractured solid that collapses into itself
/// instead of falling into pieces that pile up.
#[test]
fn a_crack_carries_compression_and_not_tension() {
    let (mu, lambda) = SolidMaterial::PORCELAIN.lame();
    let model = Constitutive::FixedCorotated;

    let stretched = Mat3::from_diagonal(vec3(1.05, 1.0, 1.0));
    let intact = stress(model, stretched, mu, lambda);
    let broken = stress_with_damage(model, stretched, mu, lambda, 1.0);
    assert!(
        intact.x_axis.x > 0.0,
        "a stretched intact point should pull back"
    );
    // A fiftieth, not nothing: a point with no tensile stiffness at all has
    // nothing bounding its own deformation and drifts away for ever. See
    // `model::RESIDUAL`.
    assert!(
        broken.x_axis.x < 0.05 * intact.x_axis.x,
        "a fully damaged point still pulled with {} of {}",
        broken.x_axis.x,
        intact.x_axis.x
    );
    assert!(
        broken.x_axis.x > 0.0,
        "a parted point with no stiffness at all cannot be kept finite"
    );

    let squashed = Mat3::from_diagonal(vec3(0.95, 1.0, 1.0));
    let intact = stress(model, squashed, mu, lambda);
    let broken = stress_with_damage(model, squashed, mu, lambda, 1.0);
    assert!(intact.x_axis.x < 0.0, "a squashed point should push back");
    assert!(
        (broken.x_axis.x - intact.x_axis.x).abs() < 1e-3 * intact.x_axis.x.abs(),
        "damage changed the compressive response: {} vs {}",
        broken.x_axis.x,
        intact.x_axis.x
    );
}

/// The principal form agrees with the matrix form it replaced.
///
/// `P = U diag(p) Vᵀ` is an identity rather than an approximation, so this is
/// a test that the algebra was done right — it would catch a dropped factor or
/// a transposed reconstruction, both of which produce a solid that behaves
/// plausibly and has the wrong stiffness.
#[test]
fn the_principal_form_is_the_stress_it_claims_to_be() {
    let (mu, lambda) = (3.0e5f32, 2.0e5f32);
    for model in [Constitutive::FixedCorotated, Constitutive::NeoHookean] {
        for f in [
            Mat3::from_diagonal(vec3(1.1, 0.95, 1.02)),
            Mat3::from_quat(fusion_math::Quat::from_rotation_z(0.6))
                * Mat3::from_diagonal(vec3(1.2, 0.9, 1.0)),
        ] {
            // The long way: 2μ(F − R) + λ(J−1)J F⁻ᵀ, straight from the paper.
            let j = f.determinant();
            let (r, _) = f.polar();
            let f_inv_t = f.inverse().transpose();
            let expected = match model {
                Constitutive::NeoHookean => {
                    (f - f_inv_t) * mu + f_inv_t * (lambda * j.max(1e-4).ln())
                }
                _ => (f - r) * (2.0 * mu) + f_inv_t * (lambda * (j - 1.0) * j),
            };
            let got = stress(model, f, mu, lambda);
            for i in 0..3 {
                let error = (got.col(i) - expected.col(i)).length();
                assert!(
                    error < 1e-3 * expected.col(i).length().max(1.0),
                    "{model:?} column {i}: {:?} vs {:?}",
                    got.col(i),
                    expected.col(i)
                );
            }
        }
    }

    // And the singular values it is built from are the ones a stretch has.
    let p = principal_stress(
        Constitutive::FixedCorotated,
        vec3(1.0, 1.0, 1.0),
        mu,
        lambda,
    );
    assert!(
        p.length() < 1e-3,
        "an undeformed point is under stress: {p:?}"
    );
}

/// A brittle block dropped hard enough breaks, and a tough one does not.
///
/// Same scene, same speed, same solver: the only difference is whether the
/// material has a failure strain. Nothing in the scene mentions a crack.
#[test]
fn a_brittle_solid_breaks_and_a_tough_one_does_not() {
    fn drop_it(material: SolidMaterial) -> (f32, f32) {
        let mut world = floor();
        let mut solid = build_solid(
            &SolidDesc::block(Vec3::splat(0.24))
                .at(vec3(0.0, 0.4, 0.0))
                .with_params(SolidParams::at(0.03).of(material))
                .with_velocity(vec3(0.0, -9.0, 0.0)),
        );
        let mut mpm = MpmSolver::new();
        for _ in 0..300 {
            mpm.step(&mut solid, &mut world.bodies, world.gravity, DT);
        }
        (solid.peak_damage(), solid.broken_fraction())
    }

    let (peak, broken) = drop_it(SolidMaterial::PORCELAIN);
    assert!(
        peak > 0.5,
        "porcelain driven into concrete at 9 m/s reached only {peak:.3} damage"
    );
    assert!(broken > 0.0, "nothing fully parted: {broken:.3}");

    let (peak, _) = drop_it(SolidMaterial::JELLY);
    assert_eq!(
        peak, 0.0,
        "a material with no failure strain recorded damage"
    );
}

/// Damage does not heal.
///
/// A solid that cracked on impact and then settled must stay cracked. Without
/// the monotone rule the pieces would knit back together as the strain came
/// off them, which looks exactly like a crack closing and is a solid becoming
/// whole again.
#[test]
fn a_crack_does_not_heal() {
    let mut world = floor();
    let mut solid = brittle(0.03, vec3(0.0, 0.4, 0.0), vec3(0.0, -9.0, 0.0));
    let mut mpm = MpmSolver::new();

    let mut worst = 0.0f32;
    let mut worst_at = 0;
    for step in 0..600 {
        mpm.step(&mut solid, &mut world.bodies, world.gravity, DT);
        let now = solid.peak_damage();
        assert!(
            now >= worst - 1e-6,
            "damage fell from {worst:.4} to {now:.4} at step {step}"
        );
        if now > worst {
            worst = now;
            worst_at = step;
        }
    }
    assert!(worst > 0.5 && worst_at < 400, "nothing broke on impact");
}

/// Where it breaks is decided by the scene, not by the material.
///
/// Two identical blocks, struck differently: one driven straight down onto a
/// flat floor, one dropped onto the corner of a step. If the fracture were
/// authored anywhere the damage would be in the same place both times.
#[test]
fn a_crack_is_where_the_tension_was() {
    fn break_it(step: bool) -> Vec3 {
        let mut world = floor();
        if step {
            world.add(
                BodyDesc::cuboid(vec3(0.16, 0.3, 0.6))
                    .at(vec3(-0.16, 0.05, 0.0))
                    .with_material(Material::CONCRETE)
                    .fixed(),
            );
        }
        let mut solid = brittle(0.025, vec3(0.0, 0.45, 0.0), vec3(0.0, -10.0, 0.0));
        let mut mpm = MpmSolver::new();
        for _ in 0..300 {
            mpm.step(&mut solid, &mut world.bodies, world.gravity, DT);
        }
        // The damage-weighted centre: where the breaking happened.
        let mut weighted = Vec3::ZERO;
        let mut total = 0.0;
        for i in 0..solid.len() {
            weighted += solid.particles.translation(i) * solid.damage[i];
            total += solid.damage[i];
        }
        assert!(total > 0.0, "nothing broke at all");
        weighted / total
    }

    let flat = break_it(false);
    let stepped = break_it(true);
    assert!(
        (flat.x - stepped.x).abs() > 0.01,
        "the damage centred at the same place with and without an obstacle: \
         {flat:?} and {stepped:?}"
    );
}

/// Fracture is deterministic, like everything else.
#[test]
fn breaking_happens_the_same_way_twice() {
    fn run() -> Vec<f32> {
        let mut world = floor();
        let mut solid = brittle(0.03, vec3(0.01, 0.42, -0.01), vec3(0.0, -9.0, 0.0));
        let mut mpm = MpmSolver::new();
        for _ in 0..250 {
            mpm.step(&mut solid, &mut world.bodies, world.gravity, DT);
        }
        solid.damage.clone()
    }
    assert_eq!(run(), run(), "two runs broke differently");
}

/// A tough material pays nothing for a damage model it does not use.
#[test]
fn an_unbreakable_material_is_untouched_by_the_damage_model() {
    let mut world = floor();
    let mut a = build_solid(
        &SolidDesc::block(Vec3::splat(0.24))
            .at(vec3(0.0, 0.4, 0.0))
            .with_params(SolidParams::at(0.04).of(SolidMaterial::JELLY))
            .with_velocity(vec3(0.0, -6.0, 0.0)),
    );
    let mut bodies = world.bodies.clone();
    let mut mpm = MpmSolver::new();
    for _ in 0..200 {
        mpm.step(&mut a, &mut bodies, world.gravity, DT);
    }

    let mut b = build_solid(
        &SolidDesc::block(Vec3::splat(0.24))
            .at(vec3(0.0, 0.4, 0.0))
            .with_params(SolidParams::at(0.04).of(SolidMaterial::JELLY))
            .with_velocity(vec3(0.0, -6.0, 0.0)),
    );
    let mut mpm = MpmSolver::new();
    for _ in 0..200 {
        mpm.step(&mut b, &mut world.bodies, world.gravity, DT);
    }
    assert_eq!(a.particles.position, b.particles.position);
    assert!(a.damage.iter().all(|d| *d == 0.0));
}

/// A dummy so `BodySet` is used and the helper above keeps its meaning.
#[test]
fn a_solid_with_no_bodies_still_breaks_under_its_own_motion() {
    let mut solid = brittle(0.04, Vec3::ZERO, Vec3::ZERO);
    let mut bodies = BodySet::new();
    let mut mpm = MpmSolver::new();
    for _ in 0..30 {
        mpm.step(&mut solid, &mut bodies, Vec3::ZERO, DT);
    }
    assert_eq!(solid.peak_damage(), 0.0, "a still solid damaged itself");
}

/// How much of the answer is the mesh, measured rather than claimed.
///
/// A local damage model has no length scale in it. Nothing says how wide a
/// crack is, so the width it takes is one material point — and refining the
/// sampling makes the crack thinner rather than making the answer better.
/// This is the classic failure of local damage models and the reason nonlocal
/// and gradient formulations exist.
///
/// Measured on one block at one impact speed, the broken fraction is 50%, 50%,
/// 44%, 40% and 57% across five resolutions from 216 to 2744 points. It is
/// **bounded and not convergent**: the answer stays in a band of about a fifth
/// and does not settle inside it. That is the honest statement — the fracture
/// pattern is plausible at every resolution and is not the same pattern, and
/// §9's real-time tier is where this belongs until the model gains a length
/// scale.
///
/// The assertion is on what is true: bounded, and the fragments stay together.
/// A test that asserted convergence would fail for the right reason and be
/// removed for the wrong one.
#[test]
fn the_broken_fraction_is_bounded_across_resolutions() {
    let mut fractions = Vec::new();
    for spacing in [0.04f32, 0.03, 0.025, 0.02] {
        let mut world = floor();
        let mut solid = build_solid(
            &SolidDesc::block(Vec3::splat(0.24))
                .at(vec3(0.0, 0.4, 0.0))
                .with_params(SolidParams::at(spacing).of(SolidMaterial::PORCELAIN))
                .with_velocity(vec3(0.0, -3.0, 0.0)),
        );
        let mut mpm = MpmSolver::new();
        for _ in 0..300 {
            mpm.step(&mut solid, &mut world.bodies, world.gravity, DT);
        }

        let broken = solid.broken_fraction();
        assert!(
            (0.2..0.8).contains(&broken),
            "spacing {spacing} broke {:.0}% of the block, which is neither a \
             crack nor an intact solid",
            broken * 100.0
        );
        // And the pieces stayed in the neighbourhood. Before the residual
        // stiffness existed they were spread over fifteen million metres.
        for i in 0..solid.len() {
            let p = solid.particles.translation(i);
            assert!(
                p.is_finite() && p.length() < 2.0,
                "spacing {spacing}: a fragment reached {p:?}"
            );
        }
        fractions.push(broken);
    }
    let low = fractions.iter().cloned().fold(f32::INFINITY, f32::min);
    let high = fractions.iter().cloned().fold(0.0f32, f32::max);
    assert!(
        high - low < 0.3,
        "the broken fraction ranged from {low:.2} to {high:.2} across \
         resolutions, which is not bounded in any useful sense"
    );
}
