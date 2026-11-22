//! Independent rigid-sphere sliding/rolling references, in SI units.
use core_math::{vec3, Pose, Vec3};
use core_runtime::Pool;
use core_space::Shape;
use core_world::{Material, Representation, World};
use phys_mechanics::{Mechanics, Settings};

pub struct Report {
    pub velocity: Vec3,
    pub spin: Vec3,
    pub expected_velocity: Vec3,
    pub expected_spin: Vec3,
    pub energy_ratio: f64,
    pub expected_energy_ratio: f64,
    pub slip: f64,
    pub friction_work: f64,
    pub initial_energy: f64,
    pub seconds: f64,
    pub newton: usize,
    pub cg: usize,
    pub products: usize,
    pub bytes: Vec<u8>,
}

/// A homogeneous sphere launched along a level floor. With no rolling
/// resistance, v_roll = v0 - 2/7 * initial_slip. Friction changes translation
/// and spin until the contact material point has zero tangential velocity.
pub fn level(dt: f64, threads: usize, spin: Vec3, ledger: bool) -> Result<Report, String> {
    let mut w = World::new();
    let (radius, mu, speed) = (0.1, 0.4, 1.0);
    let m = w.add_material(Material::new("rolling", 1000.0, 1e6, 0.3, mu, 0.0));
    w.add_fixed("floor", m, Shape::ground(0.0), Pose::IDENTITY);
    let id = w.add_rigid(
        "sphere",
        m,
        Shape::Sphere { radius },
        Pose::at(Vec3::Y * radius),
    );
    let v0 = Vec3::X * speed;
    w.set_velocity(id, v0, spin);
    let before = w.totals().kinetic_energy;
    let initial_slip = v0 + spin.cross(-Vec3::Y * radius);
    let expected_velocity = v0 - initial_slip * (2.0 / 7.0);
    let expected_spin = spin + Vec3::Y.cross(initial_slip) * (5.0 / (7.0 * radius));
    let expected_energy_ratio = (0.5 * expected_velocity.length_squared()
        + 0.2 * radius * radius * expected_spin.length_squared())
        / (0.5 * v0.length_squared() + 0.2 * radius * radius * spin.length_squared());
    let pool = Pool::new(threads);
    let mut solver = Mechanics::new(Settings {
        contact_precision: Some(1e-4),
        tolerance: 1e-5,
        energy_accounting: ledger,
        ..Default::default()
    });
    solver.prepare(&pool, &w);
    solver.profile.set_enabled(false);
    let (mut seconds, mut friction_work) = (0.0, 0.0);
    let (mut newton, mut cg, mut products) = (0, 0, 0);
    for _ in 0..(0.5 / dt).round() as usize {
        let start = std::time::Instant::now();
        let r = solver.step(&pool, &mut w, dt);
        seconds += start.elapsed().as_secs_f64();
        if r.failed() {
            return Err(format!("time {}: {r:?}", w.time));
        }
        newton += r.iterations;
        cg += r.linear_iterations;
        products += r.linear_products;
        if let Some(e) = r.energy {
            friction_work += e.friction_work;
        }
    }
    let Representation::Rigid(body) = &w.object(id).representation else {
        unreachable!()
    };
    Ok(Report {
        velocity: body.velocity,
        spin: body.spin,
        expected_velocity,
        expected_spin,
        energy_ratio: w.totals().kinetic_energy / before,
        expected_energy_ratio,
        slip: (body.velocity + body.spin.cross(-Vec3::Y * radius) - Vec3::Y * body.velocity.y)
            .length(),
        friction_work,
        initial_energy: before,
        seconds,
        newton,
        cg,
        products,
        bytes: solver.checkpoint(&w).to_bytes(),
    })
}

pub fn launches() -> [(&'static str, Vec3); 3] {
    [
        ("slide", Vec3::ZERO),
        ("roll", vec3(0.0, 0.0, -10.0)),
        ("side_spin", vec3(5.0, 0.0, 3.0)),
    ]
}

pub struct ImpactReport {
    pub velocity_error: f64,
    pub spin_error: f64,
    pub momentum_error: f64,
    pub angular_error: f64,
    pub energy_ratio: f64,
    pub expected_energy_ratio: f64,
    pub seconds: f64,
    pub newton: usize,
    pub cg: usize,
    pub products: usize,
    pub bytes: Vec<u8>,
}

/// Two identical spheres: closed-form first hard-sphere collision followed
/// by Newton normal restitution and a sticking Coulomb tangential impulse.
/// Unlike a zero-momentum symmetric check, the common velocity is nonzero.
pub fn oblique(dt: f64, threads: usize, restitution: f64) -> Result<ImpactReport, String> {
    let (radius, mu) = (0.1, 0.4);
    let mut w = World::new();
    w.gravity = Vec3::ZERO;
    let m = w.add_material(Material::new("oblique", 1000.0, 1e6, 0.3, mu, restitution));
    let common = vec3(0.1, 0.2, -0.3);
    let positions = [-Vec3::X * 0.11, Vec3::X * 0.11];
    let velocities = [common + vec3(1.0, 0.0, 0.25), common - vec3(1.0, 0.0, 0.25)];
    for i in 0..2 {
        let id = w.add_rigid(
            "sphere",
            m,
            Shape::Sphere { radius },
            Pose::at(positions[i]),
        );
        w.set_velocity(id, velocities[i], Vec3::ZERO);
    }
    let before = w.totals();
    let d = positions[0] - positions[1];
    let v = velocities[0] - velocities[1];
    let a = v.length_squared();
    let b = d.dot(v);
    let c = d.length_squared() - 4.0 * radius * radius;
    let hit_time = (-b - (b * b - a * c).sqrt()) / a;
    let normal = (d + v * hit_time).normalize();
    let vn = normal.dot(v);
    let tangent = v - normal * vn;
    // J_n / m and J_t / m. Two solid spheres have tangential inverse mass
    // 2 * (1/m + r²/I) = 7/m. This fixture has enough friction to stick.
    let jn = -(1.0 + restitution) * vn * 0.5;
    let jt = tangent.length() / 7.0;
    assert!(jt < mu * jn);
    let impulse_per_mass = normal * jn - tangent * (1.0 / 7.0);
    let expected_v = [
        velocities[0] + impulse_per_mass,
        velocities[1] - impulse_per_mass,
    ];
    let expected_spin = (-normal).cross(impulse_per_mass) * (2.5 / radius);
    let mass = before.mass * 0.5;
    let expected_energy =
        0.5 * mass * (expected_v[0].length_squared() + expected_v[1].length_squared())
            + 0.4 * mass * radius * radius * expected_spin.length_squared();
    let pool = Pool::new(threads);
    let mut solver = Mechanics::new(Settings {
        contact_precision: Some(1e-5),
        tolerance: 1e-6,
        ..Default::default()
    });
    solver.prepare(&pool, &w);
    solver.profile.set_enabled(false);
    let mut seconds = 0.0;
    let (mut newton, mut cg, mut products) = (0, 0, 0);
    for _ in 0..(0.04 / dt).round() as usize {
        let start = std::time::Instant::now();
        let r = solver.step(&pool, &mut w, dt);
        seconds += start.elapsed().as_secs_f64();
        if r.failed() {
            return Err(format!("time {}: {r:?}", w.time));
        }
        newton += r.iterations;
        cg += r.linear_iterations;
        products += r.linear_products;
    }
    let (mut velocity_error, mut spin_error) = (0.0_f64, 0.0_f64);
    for (i, o) in w.objects().iter().enumerate() {
        let Representation::Rigid(r) = &o.representation else {
            unreachable!()
        };
        velocity_error = velocity_error.max((r.velocity - expected_v[i]).length());
        spin_error = spin_error.max((r.spin - expected_spin).length());
    }
    let after = w.totals();
    Ok(ImpactReport {
        velocity_error,
        spin_error,
        momentum_error: (after.momentum - before.momentum).length(),
        angular_error: (after.angular_momentum - before.angular_momentum).length(),
        energy_ratio: after.kinetic_energy / before.kinetic_energy,
        expected_energy_ratio: expected_energy / before.kinetic_energy,
        seconds,
        newton,
        cg,
        products,
        bytes: solver.checkpoint(&w).to_bytes(),
    })
}
