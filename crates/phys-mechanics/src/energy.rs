//! Discrete mechanical work ledger. Diagnostics do not modify the trajectory.
use super::Mechanics;
use core_runtime::Pool;
use core_world::World;

/// Physical energies in joules. Contact barriers are solver potentials, not
/// stored material strain energy or physical heat.
#[derive(Clone, Copy, Debug, Default)]
pub struct MechanicalEnergy {
    pub kinetic: f64,
    pub elastic: f64,
    pub gravitational: f64,
    pub joint: f64,
}
impl MechanicalEnergy {
    pub fn total(self) -> f64 {
        self.kinetic + self.elastic + self.gravitational + self.joint
    }
    fn add(&mut self, other: Self) {
        self.kinetic += other.kinetic;
        self.elastic += other.elastic;
        self.gravitational += other.gravitational;
        self.joint += other.joint;
    }
}

/// Accepted-interval accounting in joules; absent on a failed interval.
/// Forces use their discrete endpoint work, not an exact continuous integral.
#[derive(Clone, Copy, Debug, Default)]
pub struct EnergyLedger {
    pub before: MechanicalEnergy,
    pub after: MechanicalEnergy,
    /// Uniform gravity work, exactly minus its potential-energy change.
    pub gravity_work: f64,
    /// Work of the numerical normal barrier, including self-edge smoothing.
    pub barrier_work: f64,
    /// Lagged Coulomb force dotted with discrete slip; nonpositive.
    pub friction_work: f64,
    /// Discrete viscous damping work in springs; nonpositive.
    pub damping_work: f64,
    /// Discrete plastic dissipation work; nonpositive.
    pub plastic_work: f64,
    /// Discrete cohesive fracture dissipation work; nonpositive.
    pub fracture_work: f64,
    /// Kinetic-energy change of restitution impulses at fixed positions.
    /// This may return energy removed by the inelastic position solve.
    pub restitution_work: f64,
    /// Kinetic energy removed by putting settled islands to sleep
    /// ([`crate::Settings::settle_time`]); nonpositive.
    pub settling_work: f64,
    /// Unresolved discrete balance: integration/force quadrature, nonlinear
    /// tolerance, rigid projection and any unaccounted constraint work.
    /// Signed; it must not be mislabeled as measured frictional heat.
    pub residual: f64,
}
impl EnergyLedger {
    pub(super) fn finish(&mut self) {
        self.gravity_work = self.before.gravitational - self.after.gravitational;
        self.residual = (self.after.kinetic + self.after.elastic + self.after.joint)
            - (self.before.kinetic + self.before.elastic + self.before.joint)
            - self.gravity_work
            - self.barrier_work
            - self.friction_work
            - self.damping_work
            - self.plastic_work
            - self.fracture_work
            - self.restitution_work
            - self.settling_work;
    }
    pub(super) fn serial(acc: &mut Option<Self>, next: Option<Self>) {
        if let Some(next) = next {
            if let Some(acc) = acc {
                acc.after = next.after;
                acc.barrier_work += next.barrier_work;
                acc.friction_work += next.friction_work;
                acc.damping_work += next.damping_work;
                acc.plastic_work += next.plastic_work;
                acc.fracture_work += next.fracture_work;
                acc.restitution_work += next.restitution_work;
                acc.settling_work += next.settling_work;
                acc.finish();
            } else {
                *acc = Some(next);
            }
        }
    }
    pub(super) fn spatial(acc: &mut Option<Self>, next: Option<Self>) {
        if let Some(next) = next {
            if let Some(acc) = acc {
                acc.before.add(next.before);
                acc.after.add(next.after);
                acc.barrier_work += next.barrier_work;
                acc.friction_work += next.friction_work;
                acc.damping_work += next.damping_work;
                acc.plastic_work += next.plastic_work;
                acc.fracture_work += next.fracture_work;
                acc.restitution_work += next.restitution_work;
                acc.settling_work += next.settling_work;
                acc.finish();
            } else {
                *acc = Some(next);
            }
        }
    }
}
impl Mechanics {
    pub(super) fn kinetic_energy(&self, world: &World) -> f64 {
        self.layout
            .solids
            .iter()
            .map(|d| world.objects()[d.object].kinetic_energy())
            .sum::<f64>()
            + self
                .layout
                .rigids
                .iter()
                .map(|d| world.objects()[d.object].kinetic_energy())
                .sum::<f64>()
    }
    pub(super) fn sample_energy(&mut self, pool: &Pool, world: &World) -> MechanicalEnergy {
        let gravitational = self
            .layout
            .solids
            .iter()
            .map(|d| world.objects()[d.object].gravitational_energy(world.gravity))
            .sum::<f64>()
            + self
                .layout
                .rigids
                .iter()
                .map(|d| world.objects()[d.object].gravitational_energy(world.gravity))
                .sum::<f64>()
            + crate::gravity::MutualGravity::world_energy(
                self.settings.mutual_gravity,
                world,
                &self.layout,
            );
        MechanicalEnergy {
            kinetic: self.kinetic_energy(world),
            elastic: self.strain_energy(pool, world),
            gravitational,
            joint: crate::joints::Joints::world_energy(world),
        }
    }
}
