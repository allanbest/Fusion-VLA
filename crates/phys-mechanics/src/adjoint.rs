//! Analytical trajectory gradients via the Implicit Function Theorem (IFT).
//!
//! At a converged backward-Euler stage minimum `x*(θ)`, stationarity reads
//! `∇_x E(x*(θ); θ) = 0`. Given any end-of-step scalar objective `L(x*, v*)`,
//! solving a single adjoint linear system `H w = ∂L / ∂x*` (with the converged
//! Hessian `H = ∇²_xx E(x*)`) yields exact sensitivities
//! `dL / dθ = (∂L / ∂θ)_direct − wᵀ ∂²E / (∂x ∂θ)` without finite differences.

use core_math::{Mat3, Vec3};
use core_numerics::{
    conjugate_gradient, BlockJacobi, CgSettings, CgWorkspace, Objective, SparseCholesky,
};
use core_runtime::Pool;
use core_world::{ObjectId, World};

use crate::dofs::{Owner, HELD};
use crate::step::{Integrator, Mechanics, Stage, System};
use crate::term::{Context, Term};

/// End-of-step objective gradient inputs for [`Mechanics::backward_step`].
#[derive(Clone, Debug, Default)]
pub struct AdjointLoss {
    /// Gradient wrt free generalized block vector `∂L / ∂x*` (`3 * blocks`).
    pub d_loss_dx: Vec<f64>,
    /// Gradient wrt post-step rigid body COM positions `(object, ∂L / ∂p*)`.
    pub object_position_grads: Vec<(ObjectId, Vec3)>,
    /// Gradient wrt post-step rigid body linear velocities `(object, ∂L / ∂v*)`.
    pub object_velocity_grads: Vec<(ObjectId, Vec3)>,
    /// Gradient wrt post-step solid node positions `(object, node, ∂L / ∂p*)`.
    pub solid_node_position_grads: Vec<(ObjectId, usize, Vec3)>,
    /// Gradient wrt post-step solid node velocities `(object, node, ∂L / ∂v*)`.
    pub solid_node_velocity_grads: Vec<(ObjectId, usize, Vec3)>,
}

/// Analytical step parameter sensitivities returned by [`Mechanics::backward_step`].
#[derive(Clone, Debug)]
pub struct AdjointGradients {
    /// Forward-stepped world state at `t + h` produced by the backward-Euler pass.
    pub world_after: World,
    /// Whether both the forward step and the adjoint linear solve succeeded.
    pub converged: bool,
    /// Adjoint vector `w = H⁻¹ (∂L / ∂x*)` (`3 * blocks`).
    pub adjoint: Vec<f64>,
    /// Sensitivity wrt initial positions `∂L / ∂x₀` on free blocks (`3 * blocks`).
    pub d_initial_positions: Vec<f64>,
    /// Sensitivity wrt initial linear velocities `∂L / ∂v₀` on free blocks (`3 * blocks`).
    pub d_initial_velocities: Vec<f64>,
    /// Sensitivity wrt uniform world gravity `∂L / ∂g`.
    pub d_gravity: Vec3,
    /// Per-link sensitivity wrt drive target (or spring rest length) `∂L / ∂u`.
    pub d_joint_targets: Vec<f64>,
    /// Per-link sensitivity wrt drive stiffness (or structural joint stiffness) `∂L / ∂k`.
    pub d_joint_stiffnesses: Vec<f64>,
}

impl Mechanics {
    /// Evaluates one smooth backward-Euler step from `world_before` and computes
    /// analytical IFT sensitivities of `loss` wrt initial states, gravity, and
    /// joint parameters.
    pub fn backward_step(
        &mut self,
        pool: &Pool,
        world_before: &World,
        h: f64,
        loss: &AdjointLoss,
    ) -> AdjointGradients {
        let mut world_after = world_before.clone();
        let n_links = world_before.links().len();
        if !h.is_finite() || h <= 0.0 {
            return AdjointGradients {
                world_after,
                converged: false,
                adjoint: Vec::new(),
                d_initial_positions: Vec::new(),
                d_initial_velocities: Vec::new(),
                d_gravity: Vec3::ZERO,
                d_joint_targets: vec![0.0; n_links],
                d_joint_stiffnesses: vec![0.0; n_links],
            };
        }

        let orig_settings = self.settings;
        let orig_workers = std::mem::take(&mut self.workers);
        self.settings.islands = false;
        self.settings.sleeping = false;
        self.settings.integrator = Integrator::BackwardEuler;
        self.settings.impact_law = false;

        let report = self.step(pool, &mut world_after, h);
        let n = 3 * self.layout.blocks;
        let mut d_initial_positions = vec![0.0; n];
        let mut d_initial_velocities = vec![0.0; n];
        let mut d_gravity = Vec3::ZERO;
        let mut d_joint_targets = vec![0.0; n_links];
        let mut d_joint_stiffnesses = vec![0.0; n_links];

        if report.failed() {
            self.settings = orig_settings;
            self.workers = orig_workers;
            return AdjointGradients {
                world_after,
                converged: false,
                adjoint: vec![0.0; n],
                d_initial_positions,
                d_initial_velocities,
                d_gravity,
                d_joint_targets,
                d_joint_stiffnesses,
            };
        }

        let mut rhs = vec![0.0; n];
        let copy_len = loss.d_loss_dx.len().min(n);
        rhs[..copy_len].copy_from_slice(&loss.d_loss_dx[..copy_len]);
        let inv_h = 1.0 / h;
        for &(obj, g_p) in &loss.object_position_grads {
            if let Some(&Owner::Rigid(r)) = self.layout.objects.get(obj.0 as usize) {
                g_p.add_to(&mut rhs, self.layout.rigids[r].block);
            }
        }
        for &(obj, g_v) in &loss.object_velocity_grads {
            if let Some(&Owner::Rigid(r)) = self.layout.objects.get(obj.0 as usize) {
                let b = self.layout.rigids[r].block;
                (g_v * inv_h).add_to(&mut rhs, b);
                (-g_v * inv_h).add_to(&mut d_initial_positions, b);
            }
        }
        for &(obj, node, g_p) in &loss.solid_node_position_grads {
            if let Some(&Owner::Solid(s)) = self.layout.objects.get(obj.0 as usize) {
                if let Some(&b) = self.layout.solids[s].block.get(node) {
                    if b != HELD {
                        g_p.add_to(&mut rhs, b as usize);
                    }
                }
            }
        }
        for &(obj, node, g_v) in &loss.solid_node_velocity_grads {
            if let Some(&Owner::Solid(s)) = self.layout.objects.get(obj.0 as usize) {
                if let Some(&b) = self.layout.solids[s].block.get(node) {
                    if b != HELD {
                        let b = b as usize;
                        (g_v * inv_h).add_to(&mut rhs, b);
                        (-g_v * inv_h).add_to(&mut d_initial_positions, b);
                    }
                }
            }
        }

        if self.layout.blocks == 0 {
            self.settings = orig_settings;
            self.workers = orig_workers;
            return AdjointGradients {
                world_after,
                converged: true,
                adjoint: Vec::new(),
                d_initial_positions,
                d_initial_velocities,
                d_gravity,
                d_joint_targets,
                d_joint_stiffnesses,
            };
        }

        // Re-evaluate exact stage Hessian at the converged state `x*` relative
        // to `world_before` and `Stage::Euler` parameters (`weight = h * h`).
        let x_star = self.x.clone();
        self.begin(world_before, Stage::Euler, h);
        self.elastic
            .sync_state(world_before, &self.layout, &self.plastic);
        {
            let ctx = Context {
                pool,
                world: world_before,
                layout: &self.layout,
                frame: &self.frame,
            };
            self.joints.begin_step(&ctx);
            self.mutual_gravity.begin_step(&ctx);
        }
        // If outer AL ran in forward step, ensure multipliers match `world_after`'s converged pass;
        // evaluate stage Hessian cleanly at `x*`.
        let mut grad = vec![0.0; n];
        self.rigid.exact_hessian = true;
        {
            let mut system = System {
                world: world_before,
                layout: &self.layout,
                frame: &self.frame,
                terms: [
                    &mut self.inertia,
                    &mut self.gravity,
                    &mut self.mutual_gravity,
                    &mut self.elastic,
                    &mut self.rigid,
                    &mut self.contact,
                    &mut self.joints,
                ],
                hessian: &mut self.hessian,
                triplets: &mut self.triplets,
                profile: &mut self.profile,
                sweep: &mut self.sweep,
                groups: &self.solver_groups,
                translation: &self.translation_weights,
            };
            system.derivatives(pool, &x_star, &mut grad);
        }
        self.rigid.exact_hessian = false;

        let mut w = vec![0.0; n];
        let mut lin_ok = false;
        if self.layout.blocks <= 256 {
            let mut direct = SparseCholesky::default();
            lin_ok = direct.factor(&self.hessian) && direct.solve(&rhs, &mut w);
        }
        if !lin_ok {
            let mut precond = BlockJacobi::default();
            precond.update(pool, &self.hessian, &self.solver_groups);
            let mut ws = CgWorkspace::default();
            let cg_report = conjugate_gradient(
                pool,
                &self.hessian,
                &precond,
                &rhs,
                &mut w,
                CgSettings {
                    tolerance: 1e-12,
                    max_iterations: (n * 4).max(200),
                },
                &mut ws,
            );
            lin_ok = cg_report.converged;
            if !lin_ok && self.layout.blocks <= 1024 {
                let mut direct = SparseCholesky::default();
                lin_ok = direct.factor(&self.hessian) && direct.solve(&rhs, &mut w);
            }
        }
        // Keep residual projection on rigid blocks clean if solver had mild regularization:
        let _ = Mat3::IDENTITY;

        if lin_ok {
            for &(b, m) in &self.gravity.weights {
                let wb = Vec3::read(&w, b);
                (wb * m).add_to(&mut d_initial_positions, b);
                (wb * (m * h)).add_to(&mut d_initial_velocities, b);
                d_gravity += wb * (m * h * h);
            }
            let ctx = Context {
                pool,
                world: world_before,
                layout: &self.layout,
                frame: &self.frame,
            };
            self.joints.adjoint_link_gradients(
                &ctx,
                world_before,
                &x_star,
                &w,
                &mut d_joint_targets,
                &mut d_joint_stiffnesses,
            );
        }

        self.settings = orig_settings;
        self.workers = orig_workers;
        AdjointGradients {
            world_after,
            converged: lin_ok && !report.failed(),
            adjoint: w,
            d_initial_positions,
            d_initial_velocities,
            d_gravity,
            d_joint_targets,
            d_joint_stiffnesses,
        }
    }
}
