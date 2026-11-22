use core_math::{vec3, Mat3, Pose, Vec3};
use core_runtime::Pool;
use core_space::Shape;
use core_world::{library, Column, Material, Representation, TetMesh, World};
use phys_mechanics::law::{NeoHookean, Warm};
use phys_mechanics::{Integrator, Mechanics, MechanicsState, Settings};

#[test]
fn hencky_j2_return_yields_hardens_preserves_volume_and_reduces_energy() {
    let youngs = 1e6;
    let poisson = 0.3;
    let yield_stress = 5e4;
    let hardening = 1e5;
    let law = NeoHookean::from_youngs(youngs, poisson);
    let mut warm = Warm::default();

    // Below yield: 1% uniaxial stretch -> sigma_eff < yield_stress => purely elastic.
    let f_small = Mat3::from_diagonal(vec3(1.01, 0.997, 0.997));
    assert!(law
        .hencky_j2_return(&f_small, 0.0, yield_stress, hardening, &mut warm)
        .is_none());

    // Beyond yield: 15% volume-preserving deviatoric stretch + shear.
    let f_trial = Mat3::from_cols(
        vec3(1.15, 0.05, 0.0),
        vec3(0.0, 0.93, 0.02),
        vec3(0.0, 0.0, 1.0 / (1.15 * 0.93)),
    );
    let (fe_new, delta_fp_inv, eps_p_new) = law
        .hencky_j2_return(&f_trial, 0.0, yield_stress, hardening, &mut warm)
        .expect("must yield");

    assert!(eps_p_new > 0.0);
    assert!(
        (delta_fp_inv.determinant() - 1.0).abs() < 1e-12,
        "plastic flow must preserve volume: det={}",
        delta_fp_inv.determinant()
    );
    assert!(
        law.energy(&fe_new) < law.energy(&f_trial),
        "return mapping must strictly decrease stored elastic strain energy"
    );

    // Hardened yield stress equals yield_stress + hardening * eps_p_new,
    // so evaluating again at fe_new with eps_p_new sits on the yield surface (no re-yield).
    let recheck = law.hencky_j2_return(&fe_new, eps_p_new, yield_stress, hardening, &mut warm);
    assert!(
        recheck.is_none(),
        "returned state should satisfy updated yield condition"
    );

    // Unloading to stress-free intermediate configuration F = F_p leaves permanent strain F_p != I.
    let fp = delta_fp_inv.inverse();
    let f_unloaded = fp * delta_fp_inv;
    assert!((f_unloaded - Mat3::IDENTITY).frobenius_squared() < 1e-24);
    assert!((fp - Mat3::IDENTITY).frobenius_squared() > 1e-4);
}

fn dent_world(quadratic: bool) -> (World, core_world::ObjectId) {
    let mut world = World::new();
    let mat = world.add_material(
        Material::new("soft-metal", 2000.0, 2e6, 0.3, 0.3, 0.0).with_plasticity(4e4, 2e4),
    );
    let floor_mat = world.add_material(library::granite());
    world.add_fixed(
        "floor",
        floor_mat,
        Shape::cuboid(vec3(2.0, 0.4, 2.0)),
        Pose::at(vec3(0.0, -0.2, 0.0)),
    );
    let mesh = TetMesh::cuboid(vec3(0.16, 0.16, 0.16), 0.08);
    let mesh = if quadratic { mesh.quadratic() } else { mesh };
    let block = world.add_solid("block", mat, mesh, Pose::at(vec3(0.0, 0.15, 0.0)));
    world.set_velocity(block, vec3(0.0, -5.0, 0.0), Vec3::ZERO);
    (world, block)
}

#[test]
fn plastic_impact_accumulates_strain_dissipates_energy_and_keeps_permanent_dent() {
    for &quadratic in &[false, true] {
        let (mut world, block) = dent_world(quadratic);
        let pool = Pool::new(2);
        let mut solver = Mechanics::new(Settings {
            integrator: Integrator::BackwardEuler,
            energy_accounting: true,
            ..Default::default()
        });
        let mut total_plastic_work = 0.0;
        for _ in 0..35 {
            let report = solver.step(&pool, &mut world, 1.0 / 240.0);
            assert!(!report.failed(), "quadratic={quadratic}: {report:?}");
            let ledger = report.energy.unwrap();
            assert!(
                ledger.plastic_work <= 0.0,
                "plastic dissipation must be nonpositive: {}",
                ledger.plastic_work
            );
            total_plastic_work += ledger.plastic_work;
        }
        assert!(
            total_plastic_work < -1e-2,
            "expected substantial plastic dissipation, got {total_plastic_work}"
        );
        let Representation::Solid(solid) = &world.object(block).representation else {
            unreachable!()
        };
        let eps = solid
            .elements
            .get(Column::PLASTIC_STRAIN)
            .expect("PLASTIC_STRAIN column must exist");
        let max_eps = eps.iter().copied().fold(0.0, f64::max);
        assert!(
            max_eps > 0.01,
            "expected plastic strain > 0.01, got {max_eps}"
        );

        // After plastic yielding (nontrivial F_p^-1), FD gradient must still match energy.
        let (fd_err, _) = solver.gradient_check(&pool, &world, 1.0 / 240.0, |b| {
            let k = (b % 5) as f64;
            vec3(1e-4 * (k + 1.0), -1.2e-4 * k, 0.8e-4 * (2.0 - k))
        });
        assert!(
            fd_err < 1e-7,
            "plastic FD gradient check failed (quadratic={quadratic}): {fd_err}"
        );
    }
}

#[test]
fn plastic_state_survives_v4_checkpoints_and_thread_counts() {
    let run = |threads| {
        let pool = Pool::new(threads);
        let (mut world, block) = dent_world(false);
        let mut solver = Mechanics::new(Settings::default());
        for _ in 0..10 {
            let r = solver.step(&pool, &mut world, 1.0 / 240.0);
            assert!(!r.failed());
        }
        let mid = solver.checkpoint(&world);
        let bytes = mid.to_bytes();
        assert_eq!(bytes.len(), mid.encoded_len());

        // Restore mid-impact plastic state into a fresh solver and finish.
        let mut restored_world = dent_world(false).0;
        let mut resumed = Mechanics::new(Settings::default());
        resumed.prepare(&pool, &restored_world);
        let template = resumed.checkpoint(&restored_world);
        let decoded = MechanicsState::from_bytes(&bytes, &template).unwrap();
        assert_eq!(decoded.to_bytes(), bytes);
        resumed.restore_checkpoint(&mut restored_world, &decoded);

        for _ in 10..25 {
            let r1 = solver.step(&pool, &mut world, 1.0 / 240.0);
            let r2 = resumed.step(&pool, &mut restored_world, 1.0 / 240.0);
            assert!(!r1.failed() && !r2.failed());
        }
        let final_bytes = solver.checkpoint(&world).to_bytes();
        assert_eq!(final_bytes, resumed.checkpoint(&restored_world).to_bytes());
        let Representation::Solid(s) = &world.object(block).representation else {
            unreachable!()
        };
        let eps = s.elements.get(Column::PLASTIC_STRAIN).unwrap().to_vec();
        (final_bytes, eps)
    };

    let (b1, eps1) = run(1);
    let (b4, eps4) = run(4);
    assert_eq!(b1, b4);
    assert_eq!(eps1, eps4);

    // Reject negative equivalent plastic strain in corrupted v4 bytes.
    let mut bad = b1;
    let tail = bad.len() - 8;
    bad[tail..].copy_from_slice(&(-0.5_f64).to_le_bytes());
    let (w, _) = dent_world(false);
    let pool = Pool::new(1);
    let mut s = Mechanics::new(Settings::default());
    s.prepare(&pool, &w);
    assert!(MechanicsState::from_bytes(&bad, &s.checkpoint(&w)).is_err());
}

#[test]
fn cohesive_fracture_degrades_stiffness_populates_damage_and_closes_energy_budget() {
    let build = || {
        let mut world = World::new();
        world.gravity = Vec3::ZERO;
        let mat = world.add_material(
            Material::new("brittle", 1200.0, 2e5, 0.25, 0.3, 0.0).with_fracture(150.0, 350.0, 0.05),
        );
        let bar = world.add_solid(
            "bar",
            mat,
            TetMesh::cuboid(vec3(0.4, 0.15, 0.15), 0.15),
            Pose::IDENTITY,
        );
        // Clamp left face at rest and pull right half gently at resolved timestep.
        world.anchor(bar, |p| p.x < -0.19);
        if let Representation::Solid(s) = &mut world.object_mut(bar).representation {
            for (p, v) in s.nodes.position.iter().zip(&mut s.nodes.velocity) {
                v.x = ((p.x + 0.2) / 0.4).clamp(0.0, 1.0) * 1.6;
            }
        }
        (world, bar)
    };

    let run = |threads: usize| {
        let pool = Pool::new(threads);
        let (mut world, bar) = build();
        let mut solver = Mechanics::new(Settings {
            integrator: Integrator::TrBdf2,
            energy_accounting: true,
            ..Default::default()
        });
        let mut total_fracture_work = 0.0;
        let mut max_residual = 0.0_f64;
        for step in 0..48 {
            if step == 24 {
                // Save & verify v5 checkpoint roundtrip with active damage state.
                let cp = solver.checkpoint(&world);
                let bytes = cp.to_bytes();
                assert_eq!(bytes.len(), cp.encoded_len());
                let mut other = Mechanics::new(solver.settings);
                let mut clone_w = build().0;
                other.prepare(&pool, &clone_w);
                let decoded =
                    MechanicsState::from_bytes(&bytes, &other.checkpoint(&clone_w)).unwrap();
                other.restore_checkpoint(&mut clone_w, &decoded);
                assert_eq!(other.checkpoint(&clone_w).to_bytes(), bytes);
            }
            let r = solver.step(&pool, &mut world, 1.0 / 360.0);
            assert!(!r.failed());
            let ledger = r.energy.unwrap();
            assert!(ledger.fracture_work <= 0.0);
            total_fracture_work += ledger.fracture_work;
            max_residual = max_residual.max(ledger.residual.abs());
        }
        let Representation::Solid(s) = &world.object(bar).representation else {
            unreachable!()
        };
        let damage = s
            .elements
            .get(Column::DAMAGE)
            .expect("Column::DAMAGE should be populated");
        let max_d = damage.iter().copied().fold(0.0_f64, f64::max);
        assert!(max_d > 0.1, "expected cohesive damage > 0.1, got {max_d}");
        assert!(
            total_fracture_work < -1e-4,
            "expected negative fracture work, got {total_fracture_work}"
        );
        assert!(
            max_residual < 0.05,
            "energy ledger residual with fracture work too large: {max_residual}"
        );
        solver.checkpoint(&world).to_bytes()
    };

    assert_eq!(run(1), run(4));
}
