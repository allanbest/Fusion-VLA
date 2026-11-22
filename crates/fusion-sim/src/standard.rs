//! The Standard engine's step.
//!
//! XPBD rigid bodies, cloth, soft bodies and position-based fluids, then the
//! MPM continuum — each solid on its own grid (Classic) or every solid and
//! fluid on one shared grid (Unified) — then heat, stress and failure. This
//! is the loop from PLAN.md §31:
//!
//! ```text
//! INPUTS     components, actions, scripts
//! BROAD      SAP over AABBs
//! NARROW     analytic fast paths / GJK-EPA, contact manifolds
//! ISLANDS    sleep and wake
//! SOLVE      XPBD, N substeps x M iterations
//! CONTINUUM  MLS-MPM, Classic or Unified
//! HEAT       contact heating, conduction, cooling
//! STRESS     inertia relief in rigid bodies
//! FAILURE    breaking, yielding, melting
//! OUTPUT     energy, diagnostics
//! ```
//!
//! Nothing here knows about Fusion Core; see [`crate::core_engine`] for the
//! other engine behind [`Simulation`].

use std::time::Instant;

use fusion_mpm::MpmReport;
use fusion_solver::{RigidSolver, SolverScene, StepContext};

use crate::{
    control, failure, heat, script, stress, ContinuumSolver, Simulation, StepReport,
    MAX_KEPT_FAILURES,
};

impl Simulation {
    /// One Standard-engine step of `dt` seconds. The caller has checked `dt`.
    pub(crate) fn step_standard(&mut self, dt: f32) -> StepReport {
        // The rigid solve, inputs and scripts precede the continuum. Keep an
        // exact checkpoint so a rejected continuum interval cannot leave the
        // rigid world or clock one interval ahead of its material points.
        let rollback = (self.continuum == ContinuumSolver::Unified
            && (!self.world.solids.is_empty() || !self.world.fluids.is_empty()))
        .then(|| (self.rigid_checkpoint(), self.probe_samples.clone()));

        control::apply(
            &mut self.world,
            &self.components,
            &self.inputs,
            self.step_index,
            dt,
        );

        // --- SCRIPTS ---
        //
        // First, before anything looks at where a body is. A script that moves
        // a kinematic body has to move it *before* collision detection, or the
        // contacts are found against last step's pose and a paddle sweeps
        // through the water it was supposed to push.
        //
        // Taken out of `self` for the call because a script needs `&mut World`
        // and the scripts live beside it. A `Vec` swap, not a copy — the same
        // trade the joints make below, for the same borrow.
        if !self.scripts.is_empty() {
            let mut scripts = std::mem::take(&mut self.scripts);
            let failures = script::run_with_inputs(
                &mut scripts,
                &mut self.world,
                self.time,
                dt,
                self.step_index,
                &self.inputs,
            );
            self.scripts = scripts;
            for (index, error) in failures {
                self.script_errors.push((index, error));
            }
        }

        // --- BROAD + NARROW ---
        // Keep the base tolerance fixed. The pipeline grows each body's
        // bounds by its own motion; a scene-wide speed inflates unrelated
        // resting pairs and duplicates the per-body speculative reach.
        self.pipeline.contact_tolerance =
            fusion_collide::narrow::primitives::DEFAULT_CONTACT_TOLERANCE;
        self.pipeline.margin = 0.02;
        self.pipeline.max_sweep_distance = self.max_speculative_distance.max(0.0);
        // Jointed bodies usually overlap where they join, so by default they
        // are kept out of the contact solve entirely rather than left to fight
        // their own joint over that overlap.
        self.pipeline.set_exclusions(
            self.world
                .joints()
                .iter()
                .filter(|j| !j.collide_connected)
                .map(|j| (j.body_a.0, j.body_b.0)),
        );
        let collide_start = Instant::now();
        // Contacts reach as far as each body can travel this step, so a fast
        // body meets what it will hit before it is inside it.
        // Only when the solver solves speculative contacts: without that a
        // contact found before the surfaces meet is one nothing acts on.
        self.pipeline.sweep_dt = if self.solver.speculative { dt } else { 0.0 };
        self.pipeline
            .detect(&self.world.bodies, &mut self.manifolds);
        let collision_time = collide_start.elapsed();

        // Islands are rebuilt from the contacts just found, so sleep decisions
        // always reflect what is actually touching this step. Joints are moved
        // aside for the same borrow reason as in the solve below.
        let joints = std::mem::take(self.world.joints_storage_mut());
        self.islands.update(
            &mut self.world.bodies,
            &self.manifolds,
            &joints,
            &self.sleep,
            dt,
        );
        *self.world.joints_storage_mut() = joints;

        // --- SOLVE + INTEGRATE ---
        let solve_start = Instant::now();
        let ctx = StepContext {
            dt,
            gravity: self.world.gravity,
            linear_damping: self.world.linear_damping,
            angular_damping: self.world.angular_damping,
            air: self.air,
            energy_ledger: self.energy_ledger,
            contact_tolerance: self.pipeline.contact_tolerance,
            max_sweep_distance: self.pipeline.max_sweep_distance,
            close_dissipation: self.heat.enabled,
        };
        // The borrow checker will not let the solver hold `&world.joints` and
        // `&mut world.bodies` at once, and splitting `World` apart to please it
        // would put the scene and the state in different places for no
        // physical reason. Moving the joints aside for the duration of the
        // step is the cheaper trade — it is a `Vec` swap, not a copy.
        let joints = std::mem::take(self.world.joints_storage_mut());
        // What may yield into a continuum, as it was before the blow: see
        // `failure::step`.
        if failure::any_can_fail(&self.world.bodies) {
            failure::Before::capture(&self.world, &mut self.before_solve);
        } else {
            self.before_solve.clear();
        }
        // On the unified path the fluids are material points on the shared
        // grid, not a position-based solve: set aside while the rigid solver
        // runs, so it neither steps them nor couples to them.
        let unified = self.continuum == ContinuumSolver::Unified;
        let set_aside = if unified {
            std::mem::take(&mut self.world.fluids)
        } else {
            Vec::new()
        };
        {
            let world = &mut self.world;
            let mut scene = SolverScene {
                bodies: &mut world.bodies,
                particles: &mut world.particles,
                joints: &joints,
                distance: &world.distance_constraints,
                bending: &world.bending_constraints,
                triangles: &world.triangles,
                fluids: &mut world.fluids,
            };
            self.solver
                .prepare(&scene, &self.manifolds, &mut self.working);
            self.solver.step(&mut scene, &mut self.working, &ctx);
        }
        *self.world.joints_storage_mut() = joints;

        // --- CONTINUUM ---
        //
        // After the rigid solve, for the reason the fluid's boundary pass is
        // after it: the grid's boundary conditions read body poses, and a
        // solid should be pushed out of the wall the wall has finished moving
        // to rather than the one it started the step in.
        //
        // Each solid gets the whole step rather than being interleaved with
        // the rigid substeps. MPM substeps on its own CFL bound, which is set
        // by the material's wave speed and has nothing to do with how many
        // times a contact wants re-linearizing.
        self.mpm_report = MpmReport::default();
        self.unified_report = fusion_mpm::UnifiedReport::default();
        if unified {
            self.world.fluids = set_aside;
            let mut solids = std::mem::take(&mut self.world.solids);
            let mut fluids = std::mem::take(&mut self.world.fluids);
            let report =
                if solids.iter().all(|s| s.is_empty()) && fluids.iter().all(|f| f.is_empty()) {
                    // No continuum work or settings apply to this rigid-only step.
                    fusion_mpm::UnifiedReport {
                        advanced_dt: dt,
                        ..Default::default()
                    }
                } else {
                    self.unified.step(
                        &mut solids,
                        &mut fluids,
                        &mut self.world.bodies,
                        self.world.gravity,
                        dt,
                    )
                };
            self.world.solids = solids;
            self.world.fluids = fluids;
            self.unified_report = report;
            self.mpm_report.points = report.points;
            self.mpm_report.active_nodes = report.active_nodes;
            self.mpm_report.substeps = report.substeps;
            self.mpm_report.unstable = report.unstable;
            self.mpm_report.refused = report.refused;
            if let Some(failure) = report.failure {
                if let Some((checkpoint, probes)) = &rollback {
                    self.restore_rigid(checkpoint);
                    self.probe_samples.clone_from(probes);
                }
                return StepReport {
                    step_index: self.step_index,
                    time: self.time,
                    dt: 0.0,
                    failure: Some(failure.into()),
                    ..Default::default()
                };
            }
            for solid in &self.world.solids {
                self.mpm_report.peak_strain = self.mpm_report.peak_strain.max(solid.peak_strain());
                self.mpm_report.least_volume =
                    self.mpm_report.least_volume.min(solid.least_volume());
                self.mpm_report.peak_damage = self.mpm_report.peak_damage.max(solid.peak_damage());
                self.mpm_report.broken = self.mpm_report.broken.max(solid.broken_fraction());
            }
        } else if !self.world.solids.is_empty() {
            let mut solids = std::mem::take(&mut self.world.solids);
            for solid in &mut solids {
                let report = self
                    .mpm
                    .step(solid, &mut self.world.bodies, self.world.gravity, dt);
                self.mpm_report.points += report.points;
                self.mpm_report.active_nodes += report.active_nodes;
                self.mpm_report.substeps = self.mpm_report.substeps.max(report.substeps);
                self.mpm_report.unstable |= report.unstable;
                self.mpm_report.refused |= report.refused;
                self.mpm_report.peak_strain = self.mpm_report.peak_strain.max(report.peak_strain);
                self.mpm_report.least_volume =
                    self.mpm_report.least_volume.min(report.least_volume);
                self.mpm_report.peak_temperature = self
                    .mpm_report
                    .peak_temperature
                    .max(report.peak_temperature);
                self.mpm_report.melted = self.mpm_report.melted.max(report.melted);
                self.mpm_report.thermal_cycles =
                    self.mpm_report.thermal_cycles.max(report.thermal_cycles);
                self.mpm_report.peak_damage = self.mpm_report.peak_damage.max(report.peak_damage);
                self.mpm_report.broken = self.mpm_report.broken.max(report.broken);
            }
            self.world.solids = solids;
        }

        // --- HEAT ---
        //
        // After every solve, because its sources are what the contacts did
        // this step.
        // Collected every step: a pass over the contact points, and what a
        // caller reads contact forces from whether or not heat is on.
        self.contact_reports.clear();
        self.contact_reports.extend(self.working.contact_reports());
        if self.heat.enabled {
            heat::step(
                &mut self.world.bodies,
                &self.contact_reports,
                &self.heat,
                &mut self.heat_working,
                dt,
            );
        }

        // --- STRESS ---
        //
        // After heat, so thermal strain reads this step's temperatures.
        stress::step(
            &mut self.world.bodies,
            &self.contact_reports,
            &self.stress,
            &mut self.stress_working,
            dt,
        );

        // --- FAILURE ---
        //
        // Last, because it changes what bodies there are: a body that breaks
        // or melts does so between steps, never inside one.
        if failure::any_can_fail(&self.world.bodies) {
            failure::step(
                &mut self.world,
                &mut self.pipeline,
                &mut self.working,
                self.step_index,
                dt,
                &self.before_solve,
                &self.contact_reports,
                &mut self.failures,
            );
            // Kept for whoever reads them, but not forever: a run nobody
            // drains — a script, a headless batch — would otherwise hold
            // every break it ever made.
            if self.failures.len() > MAX_KEPT_FAILURES {
                let extra = self.failures.len() - MAX_KEPT_FAILURES;
                self.failures.drain(..extra);
            }
        }
        let solve_time = solve_start.elapsed();

        self.time += dt as f64;
        self.step_index += 1;
        self.sample_probes();

        // --- OUTPUT ---
        //
        // Everything that has energy, not only the rigid bodies. Reporting
        // bodies alone made `energy_delta` meaningless for any scene with
        // cloth or fluid in it — the number moved when a box did and sat still
        // while ten thousand particles sloshed — and it was listed as a known
        // defect through v0.4.
        //
        // Summed in double by the ledger's routine, which is also what makes
        // the two agree: a step report whose energy disagreed with the ledger
        // measuring the same step would be worse than having neither.
        let kinetic = self.world.bodies.kinetic_energy()
            + self.world.particles.kinetic_energy()
            + self
                .world
                .fluids
                .iter()
                .map(|f| f.particles.kinetic_energy())
                .sum::<f32>()
            + self
                .world
                .solids
                .iter()
                .map(|s| s.particles.kinetic_energy())
                .sum::<f32>();
        let gravity = self.world.gravity;
        let potential = self.world.bodies.potential_energy(gravity)
            + self.world.particles.potential_energy(gravity)
            + self
                .world
                .fluids
                .iter()
                .map(|f| f.particles.potential_energy(gravity))
                .sum::<f32>()
            + self
                .world
                .solids
                .iter()
                .map(|s| s.particles.potential_energy(gravity))
                .sum::<f32>();
        // The continuum's own budget, added to the ledger's. The ledger covers
        // the XPBD world — bodies, cloth, soft bodies, fluid — and cannot
        // reach the strain energy, which is a constitutive question and lives
        // a crate away. A scene with a solid in it was reporting the energy of
        // everything *except* the solid, which is worse than reporting
        // nothing: the number looked like an answer.
        let continuum = self.standard_continuum_energy();
        let total = self.standard_mechanical_energy(&continuum) as f32;
        let energy_delta = match self.last_energy {
            Some(prev) => total - prev,
            None => 0.0,
        };
        self.last_energy = Some(total);

        let report = StepReport {
            step_index: self.step_index,
            time: self.time,
            dt,
            failure: None,
            broad_phase_pairs: self.pipeline.pair_count(),
            manifolds: self.manifolds.len(),
            contact_points: self.working.contact_count(),
            contact_refreshes: self.working.refreshed_manifolds(),
            particle_contacts: self.working.particle_contact_count(),
            total_normal_impulse: self.working.impulses.iter().sum(),
            kinetic_energy: kinetic,
            potential_energy: potential,
            strain_energy: continuum.strain as f32,
            heat: continuum.heat as f32,
            energy_delta,
            collision_time,
            solve_time,
            sleeping_bodies: self.world.bodies.sleeping_count(),
            islands: self.islands.island_count(),
            fluid_particles: self.world.fluid_particle_count(),
            peak_fluid_neighbours: self.working.peak_fluid_neighbours(),
            fluid_compression: self.working.fluid_compression(),
            energy_stages: if self.working.ledger.is_enabled() {
                Some(std::array::from_fn(|i| {
                    self.working.ledger.stage(fusion_solver::Stage::ALL[i])
                }))
            } else {
                None
            },
        };
        self.diagnostics.observe(&report);
        report
    }

    /// The continuum's energy budget on the Standard engine, summed over
    /// every solid.
    pub(crate) fn standard_continuum_energy(&self) -> fusion_mpm::ContinuumEnergy {
        let gravity = self.world.gravity;
        self.world
            .solids
            .iter()
            .fold(fusion_mpm::ContinuumEnergy::default(), |sum, solid| {
                sum + fusion_mpm::ContinuumEnergy::of(solid, gravity)
            })
    }

    /// Total mechanical energy on the Standard engine, J: the ledger's sum
    /// over bodies, particles and fluids, plus `continuum`'s.
    ///
    /// The ledger covers the XPBD world and cannot reach the strain energy,
    /// which is a constitutive question and lives a crate away. Leaving it
    /// out made a scene with a solid report the energy of everything except
    /// the solid. Summed in double, because a diagnostic that is no finer
    /// than the thing it measures reports its own rounding.
    pub(crate) fn standard_mechanical_energy(
        &self,
        continuum: &fusion_mpm::ContinuumEnergy,
    ) -> f64 {
        fusion_solver::ledger::mechanical_energy(
            &self.world.bodies,
            &self.world.particles,
            &self.world.fluids,
            self.world.gravity,
        ) + continuum.mechanical()
    }
}
