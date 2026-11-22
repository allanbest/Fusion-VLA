//! Acceptance tests for the Unified continuum's transactional foundation.
use fusion_core::{BodySet, FluidParams, FluidSet, SolidSet};
use fusion_math::{vec3, Mat3, Vec3};
use fusion_mpm::unified::{control, LimitKind, RepairPolicy, StepFailure, UnifiedSolver};

fn fluid() -> FluidSet {
    let mut f = FluidSet::new(FluidParams::water_at(0.1));
    f.push(vec3(0.2, 0.2, 0.2));
    f.affine.resize(f.len(), Mat3::ZERO);
    f.volume_ratio.resize(f.len(), 1.0);
    f
}

#[test]
fn budget_rejection_does_not_move_any_state() {
    let mut solver = UnifiedSolver::new();
    solver.max_substeps = 1;
    let mut fluids = vec![fluid()];
    let before = fluids[0].clone();
    let r = solver.step(
        &mut [],
        &mut fluids,
        &mut BodySet::new(),
        Vec3::Y * -9.81,
        1.0,
    );
    assert_eq!(r.failure, Some(StepFailure::SubstepBudget));
    assert_eq!(r.advanced_dt, 0.0);
    assert_eq!(r.substeps, 0);
    assert_eq!(fluids[0].particles.position, before.particles.position);
    assert_eq!(fluids[0].affine, before.affine);
    assert_eq!(fluids[0].volume_ratio, before.volume_ratio);
}

#[test]
fn late_grid_failure_rolls_back_earlier_substeps() {
    let mut solver = UnifiedSolver::new();
    solver.max_blocks = 1;
    solver.max_substeps = 100;
    let mut fluids = vec![fluid()];
    fluids[0].particles.velocity[0] = Vec3::X * 10.0;
    let before = fluids[0].clone();
    let r = solver.step(&mut [], &mut fluids, &mut BodySet::new(), Vec3::ZERO, 0.06);
    assert_eq!(r.failure, Some(StepFailure::GridBudget));
    assert!(r.attempted_substeps > 1, "{r:?}");
    assert_eq!(r.substeps, 0);
    assert_eq!(fluids[0].particles.position, before.particles.position);
    assert_eq!(fluids[0].particles.velocity, before.particles.velocity);
    assert_eq!(fluids[0].affine, before.affine);
}

#[test]
fn strict_mode_rejects_clamps_and_preview_counts_them() {
    let mut f = fluid();
    f.volume_ratio[0] = 6.0;
    let mut fluids = vec![f];
    let mut solver = UnifiedSolver::new();
    solver.repair_policy = RepairPolicy::Reject;
    let r = solver.step(&mut [], &mut fluids, &mut BodySet::new(), Vec3::ZERO, 0.001);
    assert_eq!(r.failure, Some(StepFailure::RepairRequired));
    assert_eq!(r.repairs.volume, 1);
    assert_eq!(fluids[0].volume_ratio[0], 6.0);
    solver.repair_policy = RepairPolicy::Record;
    let r = solver.step(&mut [], &mut fluids, &mut BodySet::new(), Vec3::ZERO, 0.001);
    assert_eq!(r.failure, None);
    assert_eq!(r.repairs.volume, 1);
    assert_eq!(fluids[0].volume_ratio[0], 5.0);
}

#[test]
fn invalid_state_and_configuration_are_explicit_failures() {
    let mut solver = UnifiedSolver::new();
    for dt in [f32::NAN, f32::INFINITY, -1.0] {
        let r = solver.step(&mut [], &mut [fluid()], &mut BodySet::new(), Vec3::ZERO, dt);
        assert_eq!(r.failure, Some(StepFailure::InvalidConfiguration));
    }
    let mut f = fluid();
    f.volume_ratio[0] = f32::NAN;
    let r = solver.step(&mut [], &mut [f], &mut BodySet::new(), Vec3::ZERO, 0.001);
    assert_eq!(r.failure, Some(StepFailure::InvalidState));
    let mut f = fluid();
    f.affine.clear(); // An inconsistent partial APIC state must not be erased.
    let r = solver.step(&mut [], &mut [f], &mut BodySet::new(), Vec3::ZERO, 0.001);
    assert_eq!(r.failure, Some(StepFailure::InvalidState));
}

#[test]
fn compression_and_viscosity_tighten_the_timestep() {
    let mut fluids = vec![fluid()];
    let estimate = |f: &[FluidSet]| control::estimate(&[], f, 0.2, 20.0, Vec3::ZERO, false);
    let rest = estimate(&fluids);
    fluids[0].volume_ratio[0] = 0.5;
    let compressed = estimate(&fluids);
    assert!((compressed.dt / rest.dt - 0.125).abs() < 1e-6);
    fluids[0].volume_ratio[0] = 1.0;
    fluids[0].params.rest_density = 0.01;
    fluids[0].params.viscosity = 2.0;
    let viscous = estimate(&fluids);
    assert_eq!(viscous.kind, LimitKind::Viscosity);
    assert!(viscous.dt < rest.dt * 0.01);
}

#[test]
fn force_free_translation_is_preserved() {
    let mut f = fluid();
    for x in 0..4 {
        for y in 0..4 {
            for z in 0..4 {
                f.push(vec3(x as f32, y as f32, z as f32) * 0.1);
            }
        }
    }
    f.affine.resize(f.len(), Mat3::ZERO);
    f.volume_ratio.resize(f.len(), 1.0);
    let velocity = vec3(0.5, -0.2, 0.3);
    f.particles.velocity.fill(velocity);
    let mut fluids = vec![f];
    let before = fluids[0].particles.position.clone();
    let mut solver = UnifiedSolver::new();
    solver.repair_policy = RepairPolicy::Reject;
    for _ in 0..40 {
        let r = solver.step(&mut [], &mut fluids, &mut BodySet::new(), Vec3::ZERO, 0.002);
        assert_eq!(r.failure, None, "{r:?}");
        assert_eq!(r.repairs.total(), 0);
    }
    for (i, p) in before.iter().enumerate() {
        assert!((fluids[0].particles.velocity[i] - velocity).length() < 2e-4);
        let expected = *p + fusion_math::to_f64(velocity * 0.08);
        assert!((fluids[0].particles.position[i] - expected).length() < 1e-5);
    }
}

#[test]
fn empty_and_zero_intervals_are_well_defined() {
    let mut solver = UnifiedSolver::new();
    let r = solver.step(&mut [], &mut [], &mut BodySet::new(), Vec3::ZERO, 0.01);
    assert_eq!(r.failure, None);
    assert_eq!(r.advanced_dt, 0.01);
    let r = solver.step(
        &mut [],
        &mut [fluid()],
        &mut BodySet::new(),
        Vec3::ZERO,
        0.0,
    );
    assert_eq!(r.failure, None);
    assert_eq!(r.substeps, 0);
}

#[test]
fn huge_affine_field_is_rejected_without_nan_commit() {
    let mut f = fluid();
    f.affine[0] = Mat3::IDENTITY * 1e30;
    let r = UnifiedSolver::new().step(
        &mut [] as &mut [SolidSet],
        &mut [f],
        &mut BodySet::new(),
        Vec3::ZERO,
        0.01,
    );
    assert_eq!(r.failure, Some(StepFailure::SubstepBudget));
    assert_eq!(r.advanced_dt, 0.0);
}

#[test]
fn reactions_go_to_the_contacting_body_not_a_nearby_bounding_box() {
    use fusion_core::{BodyDesc, Shape, World};
    let mut world = World::new();
    // Its AABB overlaps the stencil; its curved surface does not.
    world.add(BodyDesc::new(Shape::Sphere { radius: 0.5 }));
    world.add(BodyDesc::cuboid(vec3(4.0, 1.0, 4.0)).at(vec3(0.6, 0.0, 0.6)));
    let mut f = fluid();
    f.particles.position[0] = fusion_math::to_f64(Vec3::splat(0.6));
    f.particles.velocity[0] = -Vec3::Y;
    let mut fluids = vec![f];
    let before = fluids[0].particles.linear_momentum() + world.bodies.linear_momentum();
    let mut solver = UnifiedSolver::new();
    let r = solver.step(&mut [], &mut fluids, &mut world.bodies, Vec3::ZERO, 0.001);
    assert_eq!(r.failure, None);
    assert_eq!(world.bodies.linear_velocity[0], Vec3::ZERO);
    assert!(world.bodies.linear_velocity[1].y < 0.0);
    let after = fluids[0].particles.linear_momentum() + world.bodies.linear_momentum();
    assert!((after - before).length() < 2e-6, "{before:?} -> {after:?}");
}

#[test]
fn free_rotation_preserves_apic_angular_momentum() {
    use fusion_mpm::unified::ContinuumTotals;
    let mut f = FluidSet::new(FluidParams::water_at(0.1));
    let omega = vec3(0.2, -0.3, 0.5);
    let c = Mat3::from_cols(
        omega.cross(Vec3::X),
        omega.cross(Vec3::Y),
        omega.cross(Vec3::Z),
    );
    for x in -2..=2 {
        for y in -2..=2 {
            for z in -2..=2 {
                let p = vec3(x as f32, y as f32, z as f32) * 0.1;
                f.push(p);
                *f.particles.velocity.last_mut().unwrap() = omega.cross(p);
            }
        }
    }
    f.affine.resize(f.len(), c);
    f.volume_ratio.resize(f.len(), 1.0);
    let mut fluids = vec![f];
    let before = ContinuumTotals::measure(&[], &fluids, 0.2).unwrap();
    let r = UnifiedSolver::new().step(&mut [], &mut fluids, &mut BodySet::new(), Vec3::ZERO, 0.001);
    assert_eq!(r.failure, None);
    let after = ContinuumTotals::measure(&[], &fluids, 0.2).unwrap();
    assert_eq!(before.mass, after.mass);
    assert!((after.angular_momentum - before.angular_momentum).length() < 1e-6);
    assert!((after.momentum - before.momentum).length() < 1e-6);
    assert!((after.kinetic - before.kinetic).abs() < 1e-6);
}

#[test]
fn appended_fluid_points_preserve_existing_material_history() {
    let mut f = fluid();
    f.volume_ratio[0] = 0.95;
    f.affine[0] = Mat3::IDENTITY * 0.01;
    f.push(Vec3::splat(0.3));
    assert_eq!(f.volume_ratio, [0.95, 1.0]);
    assert_eq!(f.affine[0], Mat3::IDENTITY * 0.01);
    assert_eq!(f.affine[1], Mat3::ZERO);
    assert!(control::validate(&[], &[f], 0.2).is_ok());
}

#[test]
fn compacted_snow_and_expanded_viscous_fluid_use_current_material_scales() {
    use fusion_core::{SolidMaterial, SolidParams};
    let mut s = SolidSet::new(SolidParams::default().of(SolidMaterial::SNOW));
    s.push(Vec3::ZERO);
    let mut solids = vec![s];
    let rest = control::estimate(&solids, &[], 0.2, 20.0, Vec3::ZERO, false);
    solids[0].plastic[0] = 0.5;
    let packed = control::estimate(&solids, &[], 0.2, 20.0, Vec3::ZERO, false);
    assert!(packed.dt < rest.dt * 0.5);
    let mut fluids = vec![fluid()];
    fluids[0].params.rest_density = 0.01;
    fluids[0].params.viscosity = 2.0;
    let rest = control::estimate(&[], &fluids, 0.2, 20.0, Vec3::ZERO, false);
    fluids[0].volume_ratio[0] = 4.0;
    let expanded = control::estimate(&[], &fluids, 0.2, 20.0, Vec3::ZERO, false);
    assert!((expanded.dt / rest.dt - 0.25).abs() < 1e-6);
}
