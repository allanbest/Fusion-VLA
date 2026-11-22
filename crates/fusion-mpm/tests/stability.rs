//! What a solid does when it is asked for more than the model can give.
//!
//! v1.0's row in PLAN.md §29 says "both modes stable", and stable does not
//! mean the answer is right — a 300 Pa block genuinely cannot hold itself up,
//! and spreading into a puddle is the correct answer for a material whose
//! shear modulus is a fortieth of the pressure at its own base. It means the
//! solver stays inside the set of states its models are defined on, and says
//! so when it has to work to keep it there.
//!
//! These are the cases that came from someone running the editor and reporting
//! that soft solids "break apart into weird shapes". Two separate faults were
//! under that sentence, and only one of them is here: the *coming apart* was
//! `fusion_core::fill` sampling a block at an eighth of the points it was asked
//! for (`fusion-core/tests/sampling.rs`), and a solid under-resolved that badly
//! separates into one clump per lattice column. What is left once the block is
//! sampled correctly is this — points turning inside out, quietly, while the
//! picture still looks like a solid.

use fusion_core::{BodyDesc, Material, SolidDesc, SolidMaterial, SolidParams, World};
use fusion_math::{vec3, Vec3};
use fusion_mpm::{ContinuumEnergy, MpmSolver};

const DT: f32 = 1.0 / 240.0;

/// A block of the given stiffness, dropped 20 cm onto a slab.
fn drop_block(youngs: f32, poisson: f32, spacing: f32, steps: usize) -> Outcome {
    let mut world = World::new();
    world.add(
        BodyDesc::cuboid(vec3(20.0, 1.0, 20.0))
            .at(vec3(0.0, -0.5, 0.0))
            .with_material(Material::CONCRETE)
            .fixed(),
    );
    let material = SolidMaterial {
        youngs_modulus: youngs,
        poissons_ratio: poisson,
        ..SolidMaterial::JELLY
    };
    let mut solid = fusion_core::build_solid(
        &SolidDesc::block(Vec3::splat(0.4))
            .at(vec3(0.0, 0.6, 0.0))
            .with_params(SolidParams::at(spacing).of(material)),
    );
    let gravity = vec3(0.0, -9.81, 0.0);
    let start = ContinuumEnergy::of(&solid, gravity);
    let mut outcome = Outcome {
        least_volume: 1.0,
        peak_energy: start.mechanical(),
        start_energy: start.mechanical(),
        uninverted: 0,
    };
    let mut solver = MpmSolver::new();
    for _ in 0..steps {
        let report = solver.step(&mut solid, &mut world.bodies, gravity, DT);
        outcome.least_volume = outcome.least_volume.min(report.least_volume);
        outcome.uninverted += report.uninverted;
        let energy = ContinuumEnergy::of(&solid, gravity);
        outcome.peak_energy = outcome.peak_energy.max(energy.mechanical());
    }
    outcome
}

struct Outcome {
    least_volume: f32,
    peak_energy: f64,
    start_energy: f64,
    uninverted: usize,
}

/// No material point ever ends a step turned inside out.
///
/// `det F <= 0` is not a soft material, it is a point that has passed through
/// itself, and every model in `fusion_mpm::model` has a term in `1/sigma` that
/// is answering a question about a configuration that does not exist once it
/// happens. Measured before [`fusion_mpm::model::uninvert`] existed, the 300 Pa
/// row below reached `min det F = -0.0000`.
#[test]
fn a_soft_solid_never_turns_itself_inside_out() {
    for youngs in [1.0e4f32, 3.0e3, 1.0e3, 3.0e2, 1.0e2, 3.0e1] {
        for poisson in [0.3f32, 0.49] {
            let out = drop_block(youngs, poisson, 0.4 / 6.0, 480);
            assert!(
                out.least_volume >= 0.0,
                "{youngs} Pa at nu = {poisson}: min det F is {:.4}, and a \
                 negative one is a point that has been turned inside out",
                out.least_volume
            );
        }
    }
}

/// And it does not gain energy while being crushed.
///
/// The same measurement, one level up: a force that is the gradient of a
/// potential cannot do net work on a closed system, so a solid falling under
/// gravity onto a fixed slab has nowhere to get energy from. Before the clamp,
/// a 300 Pa block gained **228 J** out of nothing over two seconds, on a
/// starting budget of 377 J.
#[test]
fn a_crushed_solid_creates_no_energy() {
    for youngs in [1.0e4f32, 1.0e3, 3.0e2, 1.0e2] {
        let out = drop_block(youngs, 0.3, 0.4 / 6.0, 480);
        // A tenth of a joule of headroom on a starting total of about 110 J:
        // the transfer is dissipative, so the honest expectation is a fall.
        assert!(
            out.peak_energy <= out.start_energy + 0.1,
            "{youngs} Pa gained {:.2} J from nothing (started at {:.2} J)",
            out.peak_energy - out.start_energy,
            out.start_energy
        );
    }
}

/// The clamp does nothing at all to a solid that is not in trouble.
///
/// The gate in [`fusion_mpm::model::uninvert`] is a determinant against a
/// Frobenius norm, which is an *exact* sufficient condition rather than a
/// heuristic — it never takes the SVD on a point whose singular values are all
/// above the floor. This is the test that the floor is not quietly propping up
/// ordinary scenes: a jelly block at its library stiffness is touched zero
/// times.
#[test]
fn a_healthy_solid_is_never_clamped() {
    let out = drop_block(1.0e4, 0.3, 0.4 / 10.0, 480);
    assert_eq!(
        out.uninverted, 0,
        "the inversion clamp fired {} times on a scene that is nowhere near \
         inverting",
        out.uninverted
    );
    assert!(out.least_volume > 0.15, "min det F {:.3}", out.least_volume);
}
