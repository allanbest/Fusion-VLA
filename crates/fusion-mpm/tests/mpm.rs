//! The material point method, against things that are true whatever method
//! computes them: free fall, conservation, and the difference between a
//! material that comes back and one that does not.
//!
//! PLAN.md §9 puts this solver in the real-time tier — plausible, not a
//! measurement — so these are chosen the way the fluid's are: the ones a
//! real-time continuum still has to pass, with the tolerance stated rather
//! than tuned until it goes green.

use fusion_core::{BodyDesc, BodySet, Material, SolidDesc, SolidMaterial, SolidParams, World};
use fusion_math::{vec3, Mat3, Vec3};
use fusion_mpm::{stress, MpmSolver};

const DT: f32 = 1.0 / 240.0;

/// A cube of material, alone in the world.
fn block(material: SolidMaterial, size: f32, at: Vec3) -> fusion_core::SolidSet {
    fusion_core::build_solid(
        &SolidDesc::block(Vec3::splat(size))
            .at(at)
            .with_params(SolidParams::at(size / 6.0).of(material)),
    )
}

/// Centre of mass of a solid.
fn centre(solid: &fusion_core::SolidSet) -> Vec3 {
    let n = solid.len() as f32;
    let mut sum = Vec3::ZERO;
    for i in 0..solid.len() {
        sum += solid.particles.translation(i);
    }
    sum / n
}

/// An undeformed material point is under no stress.
///
/// The first thing to check about a constitutive model and the easiest to get
/// wrong: a sign error in either term leaves a solid that explodes the instant
/// it is created, with nothing having touched it.
#[test]
fn nothing_at_rest_is_under_stress() {
    for material in SolidMaterial::LIBRARY.map(|(_, m)| m) {
        let (mu, lambda) = material.lame();
        let p = stress(material.model, Mat3::IDENTITY, mu, lambda);
        for i in 0..3 {
            assert!(
                p.col(i).length() < 1e-3,
                "{:?} is under stress at rest: {p:?}",
                material.model
            );
        }
    }
}

/// And a compressed one pushes back, while a stretched one pulls.
#[test]
fn stress_has_the_sign_that_restores() {
    for material in SolidMaterial::LIBRARY.map(|(_, m)| m) {
        let (mu, lambda) = material.lame();
        // Uniform compression to 90%: the stress has to act to expand.
        let squashed = Mat3::from_diagonal(vec3(0.9, 0.9, 0.9));
        let p = stress(material.model, squashed, mu, lambda);
        assert!(
            p.x_axis.x < 0.0,
            "{:?} did not resist compression: {}",
            material.model,
            p.x_axis.x
        );

        let stretched = Mat3::from_diagonal(vec3(1.1, 1.1, 1.1));
        let p = stress(material.model, stretched, mu, lambda);
        assert!(
            p.x_axis.x > 0.0,
            "{:?} did not resist stretching: {}",
            material.model,
            p.x_axis.x
        );
    }
}

/// A solid in free fall falls at g, and arrives undeformed.
///
/// The transfer's own test, because free fall is the one case where the answer
/// is known exactly: no boundary, no stress, and every material point doing
/// the same thing. A transfer that loses momentum shows up here as a solid
/// falling slowly, and one that has the affine term wrong shows up as a block
/// that deforms on the way down while nothing is touching it.
#[test]
fn a_solid_in_free_fall_falls_at_g() {
    let mut solid = block(SolidMaterial::JELLY, 0.3, vec3(0.0, 2.0, 0.0));
    let mut bodies = BodySet::new();
    let mut mpm = MpmSolver::new();
    let gravity = vec3(0.0, -9.81, 0.0);

    let start = centre(&solid);
    let steps = 120;
    for _ in 0..steps {
        mpm.step(&mut solid, &mut bodies, gravity, DT);
    }
    let elapsed = steps as f32 * DT;

    let fallen = start.y - centre(&solid).y;
    let expected = 0.5 * 9.81 * elapsed * elapsed;
    assert!(
        (fallen - expected).abs() < 0.02 * expected.max(1e-3),
        "fell {fallen:.4} m in {elapsed:.3} s, free fall says {expected:.4}"
    );

    let speed = solid.particles.velocity[0].y;
    assert!(
        (speed + 9.81 * elapsed).abs() < 0.05,
        "reached {speed:.3} m/s, free fall says {:.3}",
        -9.81 * elapsed
    );

    assert!(
        solid.peak_strain() < 1e-3,
        "deformed by {:.4} while nothing was touching it",
        solid.peak_strain()
    );
}

/// Mass is conserved exactly, which for MPM is a claim about the weights.
///
/// The B-spline weights sum to one, so every gram a particle hands the grid is
/// a gram the grid hands back. This is worth a test rather than an assertion
/// in a comment because it is the property that breaks first when a stencil
/// runs off the edge of the grid — which is why the grid carries two cells of
/// margin and not one.
#[test]
fn the_grid_gives_back_every_gram_it_was_given() {
    let mut solid = block(SolidMaterial::JELLY, 0.3, vec3(0.0, 1.0, 0.0));
    let mut bodies = BodySet::new();
    let mut mpm = MpmSolver::new();
    let expected = solid.mass();

    for _ in 0..60 {
        mpm.step(&mut solid, &mut bodies, vec3(0.0, -9.81, 0.0), DT);
        let carried: f32 = mpm.grid.mass.iter().sum();
        assert!(
            (carried - expected).abs() < 1e-4 * expected,
            "the grid held {carried:.6} kg of a {expected:.6} kg solid"
        );
    }
}

/// A block dropped on the floor stays on the floor, and stays a block.
///
/// The whole method end to end: transfer, stress, a rigid boundary, and a
/// hundred frames for anything wrong to accumulate. The tolerances are loose
/// on purpose — where it lands is a real-time answer — and the assertions are
/// about the things that must be true of *any* answer: it is above the floor,
/// it is still finite, and it has not deformed past recognition.
#[test]
fn a_block_dropped_on_the_floor_settles_on_it() {
    let mut world = World::new();
    world.add(
        BodyDesc::cuboid(vec3(2.0, 0.2, 2.0))
            .at(vec3(0.0, -0.1, 0.0))
            .with_material(Material::CONCRETE)
            .fixed(),
    );
    let mut solid = block(SolidMaterial::JELLY, 0.24, vec3(0.0, 0.4, 0.0));
    let mut mpm = MpmSolver::new();

    for _ in 0..600 {
        mpm.step(&mut solid, &mut world.bodies, world.gravity, DT);
    }

    let mut lowest = f32::INFINITY;
    let mut highest = f32::NEG_INFINITY;
    for i in 0..solid.len() {
        let p = solid.particles.translation(i);
        assert!(p.is_finite(), "a material point left: {p:?}");
        lowest = lowest.min(p.y);
        highest = highest.max(p.y);
    }
    assert!(
        lowest > -0.1,
        "the block sank through the floor to {lowest:.3} m"
    );
    assert!(
        highest < 0.5,
        "the block was still where it was dropped, or above it: {highest:.3} m"
    );
    assert!(
        solid.peak_strain() < 0.6,
        "the block was crushed to {:.2} of its volume",
        1.0 - solid.peak_strain()
    );

    // And it came to rest rather than jittering forever.
    let speed = solid
        .particles
        .velocity
        .iter()
        .fold(0.0f32, |m, v| m.max(v.length()));
    assert!(speed < 0.6, "still moving at {speed:.3} m/s after 2.5 s");
}

/// Elastic comes back and plastic does not, which is the whole of v0.8's
/// "elastic + elastoplastic".
///
/// Both blocks are dropped from the same height onto the same floor. The
/// elastic one may deform as much as it likes on the way; what it may not do
/// is *remember*. The plastic one must.
#[test]
fn plastic_remembers_and_elastic_does_not() {
    fn drop_it(material: SolidMaterial) -> (f32, f32) {
        let mut world = World::new();
        world.add(
            BodyDesc::cuboid(vec3(2.0, 0.2, 2.0))
                .at(vec3(0.0, -0.1, 0.0))
                .fixed(),
        );
        let mut solid = fusion_core::build_solid(
            &SolidDesc::block(Vec3::splat(0.24))
                .at(vec3(0.0, 0.6, 0.0))
                .with_params(SolidParams::at(0.04).of(material)),
        );
        let mut mpm = MpmSolver::new();
        for _ in 0..500 {
            mpm.step(&mut solid, &mut world.bodies, world.gravity, DT);
        }
        let yielded = solid
            .plastic
            .iter()
            .fold(0.0f32, |m, p| m.max((1.0 - p).abs()));
        (yielded, solid.peak_strain())
    }

    let (elastic_yield, _) = drop_it(SolidMaterial::JELLY);
    assert!(
        elastic_yield < 1e-6,
        "an elastic solid recorded {elastic_yield:.4} of permanent deformation"
    );

    let (plastic_yield, _) = drop_it(SolidMaterial::SNOW);
    assert!(
        plastic_yield > 1e-3,
        "snow landed and forgot about it: {plastic_yield:.6}"
    );
}

/// The CFL bound is reported rather than discovered.
///
/// §2's third principle: make the domain of applicability legible. Real steel
/// is outside it, and the solver says so instead of producing a shivering
/// mess and leaving the user to guess why.
#[test]
fn a_material_too_stiff_to_run_says_so() {
    let mpm = MpmSolver::new();

    let jelly = block(SolidMaterial::JELLY, 0.3, Vec3::ZERO);
    let (substeps, unstable) = mpm.substeps_for(&jelly, DT);
    assert!(!unstable, "jelly needed {substeps} substeps at 240 Hz");

    let steel = SolidMaterial {
        density: 7850.0,
        youngs_modulus: 200.0e9,
        poissons_ratio: 0.3,
        model: fusion_core::Constitutive::FixedCorotated,
        thermal: fusion_core::Thermal::INERT,
        fracture: fusion_core::Fracture::TOUGH,
    };
    let bar = block(steel, 0.3, Vec3::ZERO);
    let (_, unstable) = mpm.substeps_for(&bar, DT);
    assert!(
        unstable,
        "real steel was reported as runnable at a graphics timestep"
    );

    // And the bound itself is the physical one: a wave crossing a cell.
    let cell = 0.05;
    let dt = SolidMaterial::JELLY.max_stable_dt(cell);
    let crossing = cell / SolidMaterial::JELLY.wave_speed();
    assert!(
        dt < crossing && dt > 0.1 * crossing,
        "the CFL bound {dt:.6} is not a fraction of the crossing time {crossing:.6}"
    );
}

/// Two runs of the same scene produce the same state, to the bit.
///
/// §22, on the newest solver in the engine. MPM's transfers are a scatter, so
/// this is not free: it holds because the scatter runs in particle order and
/// nothing in the step consults a clock, a hash map, or an allocation address.
#[test]
fn the_same_scene_runs_the_same_way_twice() {
    fn run() -> Vec<[f64; 3]> {
        let mut world = World::new();
        world.add(
            BodyDesc::cuboid(vec3(2.0, 0.2, 2.0))
                .at(vec3(0.0, -0.1, 0.0))
                .fixed(),
        );
        let mut solid = block(SolidMaterial::SNOW, 0.24, vec3(0.03, 0.5, -0.02));
        let mut mpm = MpmSolver::new();
        for _ in 0..200 {
            mpm.step(&mut solid, &mut world.bodies, world.gravity, DT);
        }
        solid
            .particles
            .position
            .iter()
            .map(|p| [p.x, p.y, p.z])
            .collect()
    }
    assert_eq!(run(), run(), "two runs of one scene disagreed");
}

/// A solid pushes back on what it rests on.
///
/// One-way coupling is what the roadmap asks for; this is the accumulation
/// that makes it two-way, and without it a block of jelly can sit on a plank
/// that never notices. The plank is free to fall, so the only thing that can
/// push it down is the solid.
#[test]
fn a_solid_pushes_on_the_body_it_rests_on() {
    let mut world = World::new();
    let plank = world.add(
        BodyDesc::cuboid(vec3(0.6, 0.06, 0.6))
            .at(vec3(0.0, 0.0, 0.0))
            .with_material(Material::new(400.0, 0.2, 0.4)),
    );
    let mut solid = block(SolidMaterial::JELLY, 0.2, vec3(0.0, 0.25, 0.0));
    let mut mpm = MpmSolver::new();
    let gravity = vec3(0.0, -9.81, 0.0);

    let mut pushed = 0.0f32;
    for _ in 0..200 {
        // The plank is held up by hand so that gravity is not what moves it:
        // whatever velocity it has at the end came from the solid.
        world.bodies.set_translation(plank.index(), Vec3::ZERO);
        world.bodies.linear_velocity[plank.index()] = Vec3::ZERO;
        mpm.step(&mut solid, &mut world.bodies, gravity, DT);
        pushed = pushed.min(world.bodies.linear_velocity[plank.index()].y);
    }
    assert!(
        pushed < -1e-4,
        "the plank never felt the solid resting on it: {pushed:e} m/s"
    );
}

/// A plastic solid settles into a yield surface rather than flowing forever.
///
/// The bound on the accumulated plastic ratio is a guard, and a guard that the
/// shipped materials rely on is not a guard, it is the behaviour. This asserts
/// nothing reaches it: a material that yields on every substep in the same
/// direction is a material with no tensile strength, which is a liquid rather
/// than a solid however it is labelled.
///
/// Found by the demo reporting a plastic ratio of infinity, and then — after
/// the ratio was bounded — of exactly the bound, on half of one block's points.
#[test]
fn a_plastic_solid_stops_yielding_when_it_settles() {
    for (name, material) in SolidMaterial::LIBRARY {
        let mut world = World::new();
        world.add(
            BodyDesc::cuboid(vec3(2.0, 0.2, 2.0))
                .at(vec3(0.0, -0.1, 0.0))
                .fixed(),
        );
        let mut solid = block(material, 0.3, vec3(0.0, 0.5, 0.0));
        let mut mpm = MpmSolver::new();
        for _ in 0..900 {
            mpm.step(&mut solid, &mut world.bodies, world.gravity, DT);
        }

        let extreme = solid.plastic.iter().fold(1.0f32, |m, p| {
            if (p - 1.0).abs() > (m - 1.0).abs() {
                *p
            } else {
                m
            }
        });
        assert!(
            extreme > 0.1 && extreme < 10.0,
            "{name} ran its plastic ratio to {extreme:.3} in under four seconds"
        );
        assert!(
            solid.peak_strain() < 0.6,
            "{name} was crushed past recognition: {:.2}",
            solid.peak_strain()
        );
    }
}

/// A solid left alone stays exactly where it is, at any sampling spacing.
///
/// The regression for a NaN that shipped in v0.8 and depended on where the
/// sampling lattice happened to fall against the grid. A point sitting exactly
/// on a half-cell boundary gives its outermost node a B-spline weight near
/// `1e-14`; cubed, the node's mass is a denormal; `Vec3 / f32` multiplies by
/// the reciprocal, which overflows to infinity; and zero momentum times
/// infinity is NaN. One node poisons eight points by the next step and the
/// whole solid within a second.
///
/// Nothing about it is visible at a glance and nothing about it is
/// intermittent — it is perfectly deterministic and depends only on arithmetic
/// alignment, which is why it wants a test that sweeps the alignment rather
/// than one scene that happens to be safe.
#[test]
fn a_solid_at_rest_stays_at_rest_however_it_is_sampled() {
    for spacing in [0.02f32, 0.025, 0.03, 0.04, 0.05, 0.06] {
        let mut solid = fusion_core::build_solid(
            &SolidDesc::block(Vec3::splat(0.3))
                .with_params(SolidParams::at(spacing).of(SolidMaterial::JELLY)),
        );
        let before = solid.particles.position.clone();
        let mut bodies = BodySet::new();
        let mut mpm = MpmSolver::new();
        for _ in 0..30 {
            mpm.step(&mut solid, &mut bodies, Vec3::ZERO, DT);
        }
        for (i, (now, then)) in solid.particles.position.iter().zip(&before).enumerate() {
            assert!(
                now.x.is_finite() && now.y.is_finite() && now.z.is_finite(),
                "spacing {spacing}: point {i} went non-finite with nothing acting on it"
            );
            let moved = (*now - *then).length();
            assert!(
                moved < 1e-6,
                "spacing {spacing}: point {i} drifted {moved:.2e} m in a still scene"
            );
        }
        assert!(
            (solid.peak_strain()).abs() < 1e-6,
            "spacing {spacing}: a solid at rest deformed by {:.2e}",
            solid.peak_strain()
        );
    }
}

/// A soft material can be crushed past the point its model can push back from,
/// and which model it is decides whether that happens.
///
/// The volumetric pressure of fixed corotated elasticity is `λ(J − 1)J`, which
/// **peaks at half volume and falls back to zero** as the point flattens. Past
/// that peak the material resists less the further it is compressed, which is
/// a softening branch and therefore a one-way trip. Neo-Hookean's `λ ln J`
/// diverges instead, and holds.
///
/// The two runs below are the same block, the same 10 kPa stiffness, the same
/// drop, and one term of difference. This is why [`SolidMaterial::JELLY`] — the
/// softest thing in the library and so the one most likely to be crushed past
/// the peak — is neo-Hookean.
#[test]
fn a_model_without_a_volumetric_barrier_can_be_crushed_flat() {
    use fusion_core::solid::Constitutive;

    let height = |model: Constitutive| {
        let mut world = World::new();
        world.add(
            BodyDesc::cuboid(vec3(4.0, 0.4, 4.0))
                .at(vec3(0.0, -0.2, 0.0))
                .with_material(Material::CONCRETE)
                .fixed(),
        );
        let material = SolidMaterial {
            model,
            ..SolidMaterial::JELLY
        };
        let mut solid = fusion_core::build_solid(
            &SolidDesc::block(vec3(0.3, 0.3, 0.3))
                .at(vec3(0.0, 1.0, 0.0))
                .with_params(SolidParams::at(0.04).of(material)),
        );
        let mut mpm = MpmSolver::new();
        for _ in 0..900 {
            mpm.step(&mut solid, &mut world.bodies, world.gravity, 1.0 / 240.0);
        }
        let (mut low, mut high) = (f32::MAX, f32::MIN);
        for i in 0..solid.len() {
            let y = solid.particles.translation(i).y;
            low = low.min(y);
            high = high.max(y);
        }
        (high - low, solid.least_volume())
    };

    let (crushed, volume) = height(Constitutive::FixedCorotated);
    assert!(
        crushed < 0.1,
        "corotated survived at {crushed:.3} m — if the model has grown a \
         barrier, this test is the place to say so"
    );
    assert!(volume < 0.1, "and it is crushed: min det F is {volume:.4}");

    let (standing, volume) = height(Constitutive::NeoHookean);
    assert!(
        standing > 0.2,
        "neo-Hookean flattened to {standing:.3} m, which it should not"
    );
    assert!(volume > 0.3, "min det F {volume:.4}");
}

#[test]
fn a_nearby_noncontacting_body_receives_no_grid_reaction() {
    let mut world = World::new();
    // The sphere's expanded AABB contains grid nodes, but its surface does not.
    world.add(BodyDesc::sphere(0.5));
    world.add(BodyDesc::cuboid(vec3(4.0, 1.0, 4.0)).at(vec3(0.6, 0.0, 0.6)));
    let mut solid = block(SolidMaterial::JELLY, 0.06, vec3(0.48, 0.51, 0.48));
    solid.particles.velocity.fill(-Vec3::Y);
    let before = solid.particles.linear_momentum() + world.bodies.linear_momentum();
    let mut solver = MpmSolver::new();
    solver.step(&mut solid, &mut world.bodies, Vec3::ZERO, 0.001);
    assert_eq!(world.bodies.linear_velocity[0], Vec3::ZERO);
    assert!(world.bodies.linear_velocity[1].y < 0.0);
    let after = solid.particles.linear_momentum() + world.bodies.linear_momentum();
    assert!((after - before).length() < 2e-6, "{before:?} -> {after:?}");
}

/// Von Mises plasticity yields in shape and never in volume.
#[test]
fn metal_plasticity_keeps_volume() {
    use fusion_core::solid::Constitutive;
    let model = Constitutive::VonMises { yield_strain: 0.01 };
    for f in [
        fusion_math::Mat3::from_diagonal(fusion_math::vec3(0.6, 1.3, 1.25)),
        fusion_math::Mat3::from_cols(
            fusion_math::vec3(1.4, 0.2, 0.0),
            fusion_math::vec3(0.1, 0.7, 0.3),
            fusion_math::vec3(0.0, -0.2, 0.9),
        ),
    ] {
        let (elastic, plastic) = fusion_mpm::model::return_map(model, f);
        assert!(plastic > 1.0, "that far past yield, it yielded");
        assert!(
            (elastic.determinant() - f.determinant()).abs() < 1e-4 * f.determinant().abs(),
            "volume {} became {}",
            f.determinant(),
            elastic.determinant()
        );
    }
}

#[test]
fn fast_rigid_ball_compresses_foam_without_tunneling() {
    let mut world = World::new();
    world.add(
        BodyDesc::cuboid(vec3(0.8, 0.04, 0.8))
            .at(vec3(0.0, -0.02, 0.0))
            .with_material(Material::STEEL)
            .fixed(),
    );
    let ball = world
        .add(
            BodyDesc::sphere(0.025)
                .at(vec3(0.0, 0.13, 0.0))
                .with_velocity(vec3(0.0, -5.0, 0.0))
                .with_material(Material::STEEL),
        )
        .index();

    let mut params = SolidParams::at(0.01).of(SolidMaterial::FOAM);
    params.cell_size = 0.02;
    let mut foam = fusion_core::build_solid(
        &SolidDesc::block(Vec3::splat(0.1))
            .at(vec3(0.0, 0.05, 0.0))
            .with_params(params),
    );
    let mut mpm = MpmSolver::new();
    let gravity = vec3(0.0, -9.81, 0.0);

    for _ in 0..14 {
        world.bodies.linear_velocity[ball] += gravity * DT;
        world.bodies.position[ball] += fusion_math::to_f64(world.bodies.linear_velocity[ball] * DT);
        // Simulate the fixed steel floor contact beneath the ball (y = 0, radius = 0.025).
        if world.bodies.position[ball].y < 0.025 {
            world.bodies.position[ball].y = 0.025;
            world.bodies.linear_velocity[ball].y = world.bodies.linear_velocity[ball].y.max(0.0);
        }
        let report = mpm.step(&mut foam, &mut world.bodies, gravity, DT);
        assert!(!report.refused);
    }

    let ball_y = world.bodies.position[ball].y as f32;
    let vy = world.bodies.linear_velocity[ball].y;
    assert!(
        vy > -5.1,
        "foam reaction should slow the descending ball, got vy = {vy}"
    );
    for i in 0..foam.len() {
        let p = foam.particles.translation(i);
        let radial = p.x.hypot(p.z);
        if radial < 0.025 * 0.7 {
            assert!(
                p.y < ball_y,
                "foam column particle at r={radial:.4} popped above ball center: p.y={}, ball_y={ball_y}",
                p.y
            );
        }
    }
}
