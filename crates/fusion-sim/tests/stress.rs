//! Stress in rigid bodies, against answers that need no simulation.
//!
//! Each case is one a structural engineer would check by hand: a column under
//! its own weight, a beam on two supports, a bar spinning about its middle, a
//! bar landing on its end, a block heated unevenly. The solve is linear
//! elasticity on a coarse grid, so the answers are compared to the textbook
//! where the textbook is exact, and the tolerances are what that grid earns.

use fusion_core::{BodyDesc, Material, World};
use fusion_math::{vec3, Vec3};
use fusion_sim::{state_hash, HeatSettings, Simulation, StressSettings};

fn floor(world: &mut World) {
    world.add(
        BodyDesc::cuboid(vec3(40.0, 1.0, 40.0))
            .at(vec3(0.0, -0.5, 0.0))
            .with_material(Material::CONCRETE)
            .fixed(),
    );
}

/// Mean of one stress component over a horizontal layer of cells.
fn layer(sim: &Simulation, body: usize, y: usize, component: usize) -> (f32, f32) {
    let stress = &sim.world.bodies.stress[body];
    let grid = &stress.grid;
    let mut sum = 0.0;
    let mut n = 0.0;
    for z in 0..grid.dims[2] {
        for x in 0..grid.dims[0] {
            let i = grid.index(x, y, z);
            if grid.inside[i] {
                sum += stress.stress[i][component];
                n += 1.0;
            }
        }
    }
    (grid.cell_centre(grid.index(0, y, 0)).y, sum / n)
}

/// The contact impulses the solver reports for one body, summed, as a force.
fn contact_force(sim: &Simulation, body: usize) -> Vec3 {
    let mut j = Vec3::ZERO;
    for r in sim.contact_reports() {
        if r.body_b as usize == body {
            j += r.impulse;
        }
        if r.body_a as usize == body {
            j -= r.impulse;
        }
    }
    j / sim.solver_dt
}

/// A block on a floor feels exactly its weight through the floor, and a
/// sliding one exactly `mu` times that against its motion — steel on
/// concrete, whose `mu` is the geometric mean of 0.6 and 0.8.
#[test]
fn the_contacts_report_the_weight_and_the_friction() {
    for (speed, drag) in [(0.0, 0.0), (4.0, (0.6f32 * 0.8).sqrt())] {
        let mut world = World::new();
        floor(&mut world);
        let b = world.add(
            BodyDesc::cuboid(Vec3::splat(0.5))
                .at(vec3(0.0, 0.25, 0.0))
                .with_velocity(vec3(speed, 0.0, 0.0))
                .with_material(Material::STEEL),
        );
        let mut sim = Simulation::new(world);
        for _ in 0..30 {
            sim.step();
        }
        let weight = sim.world.bodies.mass(b.index()) * 9.81;
        let f = contact_force(&sim, b.index()) / weight;
        assert!((f.y - 1.0).abs() < 1e-3, "{f:?}");
        assert!((f.x + drag).abs() < 2e-3, "{f:?}");
    }
}

/// A column standing on a floor: the compression at height `y` is the weight
/// of what is above it, `ρ g (H - y)`. Every layer, to a percent.
#[test]
fn a_column_carries_its_own_weight() {
    let mut world = World::new();
    floor(&mut world);
    let b = world.add(
        BodyDesc::cuboid(vec3(1.0, 2.0, 1.0))
            .at(vec3(0.0, 1.0, 0.0))
            .with_material(Material::STEEL),
    );
    let mut sim = Simulation::new(world);
    sim.sleep.enabled = false;
    sim.stress = StressSettings::on();
    for _ in 0..30 {
        sim.step();
    }
    let rho_g = 7850.0 * 9.81;
    let dims = sim.world.bodies.stress[b.index()].grid.dims;
    assert_eq!(dims, [4, 8, 4]);
    for y in 0..dims[1] {
        let (height, syy) = layer(&sim, b.index(), y, 1);
        let expected = -rho_g * (1.0 - height);
        assert!(
            (syy - expected).abs() < 0.01 * rho_g * 2.0,
            "layer {y}: {syy} against {expected}"
        );
    }
}

/// A bar spinning about its middle is pulled apart by its own centripetal
/// load: `σ(x) = ρ ω² (L²/4 - x²) / 2`, largest at the centre.
#[test]
fn a_spinning_bar_is_pulled_apart_at_its_middle() {
    let mut world = World::new().with_gravity(Vec3::ZERO);
    let omega = 50.0;
    let b = world.add(
        BodyDesc::cuboid(vec3(2.0, 0.25, 0.25))
            .with_angular_velocity(vec3(0.0, omega, 0.0))
            .with_material(Material::STEEL),
    );
    let mut sim = Simulation::new(world);
    sim.stress = StressSettings::on();
    sim.step();
    let stress = &sim.world.bodies.stress[b.index()];
    let grid = &stress.grid;
    for x in 0..grid.dims[0] {
        let i = grid.index(x, 1, 1);
        let c = grid.cell_centre(i).x;
        let expected = 7850.0 * omega * omega * (1.0 - c * c) / 2.0;
        let got = stress.stress[i][0];
        assert!(
            (got - expected).abs() < 0.02 * 7850.0 * omega * omega / 2.0,
            "x = {c}: {got} against {expected}"
        );
    }
}

/// Gravity accelerates every part of a falling body alike and stresses none
/// of it.
#[test]
fn a_falling_body_is_unstressed() {
    let mut world = World::new();
    let b = world.add(
        BodyDesc::cuboid(Vec3::splat(0.5))
            .at(vec3(0.0, 5.0, 0.0))
            .with_material(Material::STEEL),
    );
    let mut sim = Simulation::new(world);
    sim.stress = StressSettings::on();
    for _ in 0..10 {
        sim.step();
    }
    // Nothing loads it, so nothing is solved at all.
    assert_eq!(sim.body_stress(b), 0.0);
}

/// A beam on two supports under its own weight bends: tension along its
/// underside, compression along its top, and the surface stress is the beam
/// formula's `M / Z` with `M = q L² / 8`, to the accuracy a grid four cells
/// deep has.
#[test]
fn a_beam_on_two_supports_bends() {
    let mut world = World::new();
    floor(&mut world);
    for x in [-1.8f32, 1.8] {
        world.add(
            BodyDesc::cuboid(vec3(0.4, 1.0, 0.6))
                .at(vec3(x, 0.5, 0.0))
                .with_material(Material::CONCRETE)
                .fixed(),
        );
    }
    let b = world.add(
        BodyDesc::cuboid(vec3(4.0, 0.4, 0.4))
            .at(vec3(0.0, 1.2, 0.0))
            .with_material(Material::CONCRETE),
    );
    let mut sim = Simulation::new(world);
    sim.sleep.enabled = false;
    sim.stress = StressSettings::on();
    for _ in 0..120 {
        sim.step();
    }
    let stress = &sim.world.bodies.stress[b.index()];
    let grid = &stress.grid;
    assert_eq!(grid.dims, [40, 4, 4]);
    let mid = grid.dims[0] / 2;
    let bottom = stress.stress[grid.index(mid, 0, 2)][0];
    let top = stress.stress[grid.index(mid, 3, 2)][0];
    assert!(bottom > 0.0 && top < 0.0, "{bottom} {top}");
    // Cell centres sit at three quarters of the half-depth; bending is linear.
    let surface = bottom / 0.75;
    let q = 2400.0 * 9.81 * 0.4 * 0.4;
    let expected = q * 3.6f32.powi(2) / 8.0 / (0.4 * 0.4 * 0.4 / 6.0);
    assert!(
        (surface / expected - 1.0).abs() < 0.10,
        "{surface} against {expected}"
    );
    // A steady load is cheap once it has been solved.
    assert!(sim.stress_effort().0 <= 5, "{:?}", sim.stress_effort());
}

/// A bar landing on its end: the stress at the struck face is the textbook
/// elastic impact stress `ρ c Δv / 2`, with `c` the bar wave speed.
///
/// On a steel floor. On concrete the bar's share of the impact is smaller —
/// the softer floor gives, and takes most of the energy — and the face stress
/// comes out 12% lower, which is the right direction for the physics and not
/// what this formula is about.
#[test]
fn an_end_on_landing_carries_the_impact_stress() {
    let mut world = World::new();
    world.add(
        BodyDesc::cuboid(vec3(40.0, 1.0, 40.0))
            .at(vec3(0.0, -0.5, 0.0))
            .with_material(Material::STEEL)
            .fixed(),
    );
    let b = world.add(
        BodyDesc::cuboid(vec3(0.2, 0.6, 0.2))
            .at(vec3(0.0, 1.3, 0.0))
            .with_material(Material::STEEL),
    );
    let mut sim = Simulation::new(world);
    sim.stress = StressSettings::on();
    for _ in 0..400 {
        let before = sim.world.bodies.linear_velocity[b.index()].y;
        sim.step();
        let after = sim.world.bodies.linear_velocity[b.index()].y;
        if after - before > 1.0 {
            let (_, face) = layer(&sim, b.index(), 0, 1);
            let c = (200e9f32 / 7850.0).sqrt();
            let expected = -7850.0 * c * (after - before) / 2.0;
            assert!(
                (face / expected - 1.0).abs() < 0.10,
                "{face} against {expected}"
            );
            return;
        }
    }
    panic!("it never landed");
}

/// A free block whose temperature rises linearly across it expands into a
/// shape it is free to take, and carries almost no stress; one hot at the
/// core and cold at the skin cannot, and puts its skin in tension.
#[test]
fn only_an_uneven_temperature_stresses_a_free_body() {
    let mut peaks = Vec::new();
    for hot_core in [false, true] {
        let mut world = World::new().with_gravity(Vec3::ZERO);
        let b = world.add(BodyDesc::cuboid(Vec3::ONE).with_material(Material::GLASS));
        let mut sim = Simulation::new(world);
        sim.heat = HeatSettings::insulated();
        sim.stress = StressSettings::on();
        let shape = sim.world.bodies.shape[b.index()].clone();
        let heat = &mut sim.world.bodies.heat[b.index()];
        heat.build(&shape);
        for i in 0..heat.temperature.len() {
            let c = heat.cell_centre(i);
            heat.temperature[i] = if hot_core {
                400.0 - 400.0 * c.length_squared() as f64 / 0.75
            } else {
                300.0 + 100.0 * c.x as f64
            };
        }
        sim.step();
        let stress = &sim.world.bodies.stress[b.index()];
        let grid = &stress.grid;
        let [nx, ny, nz] = grid.dims;
        let skin = stress.stress[grid.index(0, ny / 2, nz / 2)];
        let core = stress.stress[grid.index(nx / 2, ny / 2, nz / 2)];
        if hot_core {
            assert!(core[0] < 0.0 && core[1] < 0.0, "{core:?}");
            assert!(skin[1] > 0.0 && skin[2] > 0.0, "{skin:?}");
        }
        peaks.push(stress.peak_von_mises);
    }
    // The scale of thermal stress here, E α ΔT / (1 - ν).
    let scale = 70e9 * 9e-6 * 100.0 / 0.78;
    assert!(peaks[0] < 0.05 * scale, "{peaks:?}");
    assert!(peaks[1] > 0.5 * scale, "{peaks:?}");
}

/// Measuring stress changes nothing about the motion.
#[test]
fn measuring_stress_does_not_move_anything() {
    let run = |on: bool| {
        let mut world = World::new();
        floor(&mut world);
        for k in 0..4 {
            world.add(
                BodyDesc::cuboid(Vec3::splat(0.5))
                    .at(vec3(0.1 * k as f32, 0.3 + 0.6 * k as f32, 0.0))
                    .with_material(Material::WOOD),
            );
        }
        let mut sim = Simulation::new(world);
        if on {
            sim.stress = StressSettings::on();
        }
        for _ in 0..240 {
            sim.step();
        }
        state_hash(&sim)
    };
    assert_eq!(run(false), run(true));
}

/// Hot steel is weaker steel, and hot glass is not until it flows.
#[test]
fn heat_weakens_metals_and_not_glass() {
    let steel = Material::STEEL;
    let at = |m: &Material, t: f32| m.strength.retained(&m.thermal, t);
    assert_eq!(at(&steel, 293.0), 1.0);
    // Structural steel keeps about half its strength at 600 C.
    let k = at(&steel, 873.0);
    assert!((0.3..0.7).contains(&k), "{k}");
    assert_eq!(at(&steel, 1800.0), 0.0);
    let glass = Material::GLASS;
    assert_eq!(at(&glass, 900.0), 1.0);
    // Ice near its melting point is as strong as ice gets.
    assert_eq!(at(&Material::ICE, 263.0), 1.0);
}
