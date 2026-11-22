//! Shared physical benchmarks for the test gate and the standalone report.
//! SI units. References are small-strain rod/beam theory, not another solver.
use core_math::{vec3, Pose, Vec3};
use core_runtime::Pool;
use core_world::{Material, ObjectId, Representation, TetMesh, World};
use phys_mechanics::{Integrator, Mechanics, Settings};
use std::time::Instant;

#[derive(Debug)]
pub struct Report {
    pub case: &'static str,
    pub edge: f64,
    pub dt: f64,
    pub nodes: usize,
    pub measured: f64,
    pub reference: f64,
    pub energy_ratio: f64,
    pub amplitude_per_period: f64,
    pub momentum_error: f64,
    pub angular_error: f64,
    pub steps: usize,
    pub newton: usize,
    pub cg: usize,
    pub products: usize,
    pub accuracy_retries: usize,
    pub splits: usize,
    pub restarts: usize,
    pub impacts: usize,
    pub seconds: f64,
}

impl Report {
    fn new(case: &'static str, edge: f64, dt: f64, nodes: usize, reference: f64) -> Self {
        Self {
            case,
            edge,
            dt,
            nodes,
            measured: 0.0,
            reference,
            energy_ratio: 0.0,
            amplitude_per_period: 0.0,
            momentum_error: 0.0,
            angular_error: 0.0,
            steps: 0,
            newton: 0,
            cg: 0,
            products: 0,
            accuracy_retries: 0,
            splits: 0,
            restarts: 0,
            impacts: 0,
            seconds: 0.0,
        }
    }
    pub fn relative_error(&self) -> f64 {
        (self.measured / self.reference - 1.0).abs()
    }
    pub fn csv(&self) -> String {
        let error = if self.reference == 0.0 {
            String::new()
        } else {
            format!("{:.8e}", self.relative_error())
        };
        format!(
            "{},{},{},{},{:.12e},{:.12e},{},{:.8e},{:.8e},{:.8e},{:.8e},{},{},{},{},{},{},{:.6}",
            self.case,
            self.edge,
            self.dt,
            self.nodes,
            self.measured,
            self.reference,
            error,
            self.energy_ratio,
            self.amplitude_per_period,
            self.momentum_error,
            self.angular_error,
            self.steps,
            self.newton,
            self.cg,
            self.splits,
            self.restarts,
            self.impacts,
            self.seconds
        )
    }
    fn step(
        &mut self,
        solver: &mut Mechanics,
        pool: &Pool,
        world: &mut World,
    ) -> Result<(), String> {
        let start = Instant::now();
        let r = solver.step(pool, world, self.dt);
        self.seconds += start.elapsed().as_secs_f64();
        self.steps += 1;
        self.newton += r.iterations;
        self.cg += r.linear_iterations;
        self.products += r.linear_products;
        self.accuracy_retries += r.accuracy_retries;
        self.splits += r.splits;
        self.restarts += r.contact_restarts;
        self.impacts += r.impacts;
        if r.failed() {
            return Err(format!("{} at step {}: {r:?}", self.case, self.steps));
        }
        Ok(())
    }
}

pub const CSV_HEADER: &str = "case,edge_m,dt_s,nodes,measured,reference,relative_error,final_energy_ratio,amplitude_per_period,momentum_error_kg_m_s,angular_error_kg_m2_s,steps,newton,cg,splits,contact_restarts,impacts,solve_seconds";

fn solid(world: &World, id: ObjectId) -> &core_world::Solid {
    let Representation::Solid(s) = &world.object(id).representation else {
        unreachable!()
    };
    s
}

fn engine(pool: &Pool, world: &World, integrator: Integrator) -> Mechanics {
    let mut s = Mechanics::new(Settings {
        integrator,
        contact_precision: Some(1e-4),
        tolerance: 1e-4,
        ..Default::default()
    });
    s.prepare(pool, world);
    s.profile.set_enabled(false);
    s
}

fn beam(edge: f64) -> (World, ObjectId) {
    let mut world = World::new();
    world.gravity = vec3(0.0, -0.01, 0.0);
    let mat = world.add_material(Material::new("isotropic beam", 1000.0, 1e6, 0.0, 0.0, 0.0));
    let id = world.add_solid(
        "cantilever",
        mat,
        TetMesh::cuboid(vec3(1.0, 0.1, 0.1), edge).quadratic(),
        Pose::IDENTITY,
    );
    world.anchor(id, |p| p.x < -0.5 + 1e-9);
    (world, id)
}

/// Euler-Bernoulli tip sag for a cantilever under its own weight:
/// q L^4 / (8 E I) plus Timoshenko shear q L^2 / (2 k G A),
/// k=5/6 for a rectangle at nu=0. The 3D clamp/finite-strain corrections
/// still warrant a theory tolerance rather than exact equality.
pub fn cantilever_sag(edge: f64, threads: usize) -> Result<Report, String> {
    let (mut world, id) = beam(edge);
    let pool = Pool::new(threads);
    let mut solver = engine(&pool, &world, Integrator::BackwardEuler);
    let target = 1000.0 * 0.01 * 0.1_f64.powi(2) / (8.0 * 1e6 * 0.1_f64.powi(4) / 12.0)
        + 1000.0 * 0.01 / (2.0 * (5.0 / 6.0) * (1e6 / 2.0));
    let mut report = Report::new(
        "cantilever_sag_m",
        edge,
        0.25,
        solid(&world, id).nodes.len(),
        target,
    );
    let mut settled = 0;
    for _ in 0..64 {
        report.step(&mut solver, &pool, &mut world)?;
        let speed = solid(&world, id)
            .nodes
            .velocity
            .iter()
            .map(|v| v.length())
            .fold(0.0, f64::max);
        settled = if speed < 1e-7 { settled + 1 } else { 0 };
        if settled >= 4 {
            break;
        }
    }
    let s = solid(&world, id);
    let tips: Vec<_> = s
        .mesh
        .rest
        .iter()
        .enumerate()
        .filter(|(_, p)| p.x > 0.5 - 1e-9)
        .collect();
    report.measured = tips
        .iter()
        .map(|(i, p)| p.y - s.nodes.position[*i].y)
        .sum::<f64>()
        / tips.len() as f64;
    // Equilibrium must actually be reached; a mean over an unfinished transient
    // could accidentally agree with the static answer.
    let speed = s
        .nodes
        .velocity
        .iter()
        .map(|v| v.length())
        .fold(0.0, f64::max);
    if speed > 1e-7 || settled < 4 {
        return Err(format!("beam has not settled: {speed} m/s"));
    }
    Ok(report)
}

fn crossing(samples: &[(f64, f64, f64)]) -> Result<(f64, f64), String> {
    let mut crossings = Vec::new();
    for w in samples.windows(2) {
        let [(ta, a, va), (tb, b, vb)] = [w[0], w[1]];
        if a > 0.0 && b <= 0.0 {
            let f = a / (a - b);
            crossings.push((ta + (tb - ta) * f, (va + (vb - va) * f).abs()));
        }
    }
    if crossings.len() < 3 {
        return Err("fewer than three measured oscillation crossings".into());
    }
    let (first, last) = (crossings[0], *crossings.last().unwrap());
    let periods = (crossings.len() - 1) as f64;
    Ok((
        periods / (last.0 - first.0),
        (last.1 / first.1).powf(1.0 / periods),
    ))
}

/// Modal projection suppresses unrelated vibration rather than measuring a
/// single node or assuming the deformed extrema remain the same nodes.
fn modal_sample(world: &World, id: ObjectId, mode: &[Vec3]) -> (f64, f64, f64) {
    let s = solid(world, id);
    let norm: f64 = mode
        .iter()
        .zip(&s.nodes.mass)
        .map(|(m, mass)| mass * m.length_squared())
        .sum();
    let (mut q, mut v) = (0.0, 0.0);
    for (i, m) in mode.iter().enumerate() {
        q += s.nodes.mass[i] * m.dot(s.nodes.position[i] - s.mesh.rest[i]);
        v += s.nodes.mass[i] * m.dot(s.nodes.velocity[i]);
    }
    (world.time, q / norm, v / norm)
}

pub fn longitudinal_vibration(edge: f64, dt: f64, threads: usize) -> Result<Report, String> {
    longitudinal_options(edge, dt, threads, true, 1)
}

pub fn longitudinal_options(
    edge: f64,
    dt: f64,
    threads: usize,
    startup: bool,
    patch: usize,
) -> Result<Report, String> {
    let mut world = World::new();
    world.gravity = Vec3::ZERO;
    let mat = world.add_material(Material::new("free rod", 1000.0, 1e5, 0.0, 0.0, 0.0));
    let id = world.add_solid(
        "rod",
        mat,
        TetMesh::cuboid(vec3(1.0, 0.1, 0.1), edge).quadratic(),
        Pose::IDENTITY,
    );
    let Representation::Solid(s) = &mut world.object_mut(id).representation else {
        unreachable!()
    };
    let mode: Vec<_> = s
        .mesh
        .rest
        .iter()
        .map(|p| Vec3::X * (std::f64::consts::PI * p.x).sin())
        .collect();
    for (v, m) in s.nodes.velocity.iter_mut().zip(&mode) {
        *v = *m * 0.001;
    }
    let pool = Pool::new(threads);
    let mut solver = engine(&pool, &world, Integrator::TrBdf2);
    solver.settings.force_startup = startup;
    solver.settings.solid_block_size = patch;
    let before = world.totals();
    let mut report = Report::new("longitudinal_frequency_hz", edge, dt, mode.len(), 5.0);
    let mut samples = vec![modal_sample(&world, id, &mode)];
    for _ in 0..(1.2 / dt).round() as usize {
        report.step(&mut solver, &pool, &mut world)?;
        samples.push(modal_sample(&world, id, &mode));
    }
    (report.measured, report.amplitude_per_period) = crossing(&samples)?;
    let after = world.totals();
    report.energy_ratio =
        (after.kinetic_energy + solver.strain_energy(&pool, &world)) / before.kinetic_energy;
    report.momentum_error = (after.momentum - before.momentum).length();
    Ok(report)
}

pub fn bending_vibration(edge: f64, dt: f64, threads: usize) -> Result<Report, String> {
    let (mut world, id) = beam(edge);
    world.gravity = Vec3::ZERO;
    let Representation::Solid(s) = &mut world.object_mut(id).representation else {
        unreachable!()
    };
    let mode: Vec<_> = s
        .mesh
        .rest
        .iter()
        .map(|p| {
            let x = p.x + 0.5;
            let y = x * x * (6.0 - 4.0 * x + x * x) / 3.0;
            let slope = (12.0 * x - 12.0 * x * x + 4.0 * x * x * x) / 3.0;
            vec3(-p.y * slope, y, 0.0)
        })
        .collect();
    for (p, m) in s.nodes.position.iter_mut().zip(&mode) {
        *p += *m * 1e-4;
    }
    let pool = Pool::new(threads);
    let mut solver = engine(&pool, &world, Integrator::TrBdf2);
    let initial_energy = solver.strain_energy(&pool, &world);
    let target = 1.875104068711961_f64.powi(2) / (2.0 * std::f64::consts::PI)
        * (1e6 * 0.1_f64.powi(4) / (12.0 * 1000.0 * 0.1_f64.powi(2))).sqrt();
    let mut report = Report::new("bending_frequency_hz", edge, dt, mode.len(), target);
    let mut samples = vec![modal_sample(&world, id, &mode)];
    for _ in 0..(6.0 / dt).round() as usize {
        report.step(&mut solver, &pool, &mut world)?;
        samples.push(modal_sample(&world, id, &mode));
    }
    (report.measured, report.amplitude_per_period) = crossing(&samples)?;
    report.energy_ratio =
        (world.totals().kinetic_energy + solver.strain_energy(&pool, &world)) / initial_energy;
    Ok(report)
}

/// Two approaching pieces belong to one solid. A nonzero common velocity
/// prevents a spurious zero-momentum reset from passing this check. Gravity
/// has an exactly known external impulse; no supports are present.
pub fn self_impact(
    dt: f64,
    integrator: Integrator,
    threads: usize,
    gravity: Vec3,
) -> Result<Report, String> {
    self_impact_options(dt, integrator, threads, gravity, None, 1e-4)
}

pub fn self_impact_options(
    dt: f64,
    integrator: Integrator,
    threads: usize,
    gravity: Vec3,
    angular_tolerance: Option<f64>,
    solve_tolerance: f64,
) -> Result<Report, String> {
    let part = TetMesh::cuboid(vec3(0.4, 0.12, 0.4), 0.4).quadratic();
    let first = part.rest.len();
    let mut mesh = TetMesh::default();
    for y in [-0.16, 0.16] {
        let offset = mesh.rest.len() as u32;
        mesh.rest.extend(part.rest.iter().map(|p| *p + Vec3::Y * y));
        mesh.tets
            .extend(part.tets.iter().map(|t| t.map(|n| n + offset)));
        mesh.midpoints
            .extend(part.midpoints.iter().map(|t| t.map(|n| n + offset)));
    }
    let mut world = World::new();
    world.gravity = gravity;
    let mat = world.add_material(Material::new("elastic impact", 1000.0, 1e5, 0.3, 0.0, 0.0));
    let id = world.add_solid(
        "one solid, two approaching pieces",
        mat,
        mesh,
        Pose::IDENTITY,
    );
    let Representation::Solid(s) = &mut world.object_mut(id).representation else {
        unreachable!()
    };
    for (i, v) in s.nodes.velocity.iter_mut().enumerate() {
        *v = vec3(0.4, 0.2, -0.3) + if i < first { Vec3::Y } else { -Vec3::Y };
    }
    let pool = Pool::new(threads);
    let mut solver = Mechanics::new(Settings {
        integrator,
        contact_precision: Some(1e-3),
        // Absolute position convergence = solve_tolerance × contact precision.
        // The default comparison uses 1e-7 m; refinement uses 1e-10 m.
        tolerance: solve_tolerance,
        angular_tolerance,
        ..Default::default()
    });
    solver.prepare(&pool, &world);
    solver.profile.set_enabled(false);
    let before = world.totals();
    let mut report = Report::new("self_impact_momentum_kg_m_s", 0.4, dt, first * 2, 0.0);
    for _ in 0..(0.2 / dt).round() as usize {
        report.step(&mut solver, &pool, &mut world)?;
        let expected = before.momentum + gravity * (before.mass * world.time);
        report.momentum_error = report
            .momentum_error
            .max((world.totals().momentum - expected).length());
    }
    let after = world.totals();
    report.measured = report.momentum_error;
    report.angular_error = (after.angular_momentum - before.angular_momentum).length();
    report.energy_ratio =
        (after.kinetic_energy + after.gravitational_energy + solver.strain_energy(&pool, &world))
            / (before.kinetic_energy + before.gravitational_energy);
    if report.impacts == 0 && report.restarts == 0 {
        return Err("impact check never exercised contact".into());
    }
    Ok(report)
}
