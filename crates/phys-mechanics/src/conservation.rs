//! Local, translation-invariant angular audit. Uniform gravity has no torque
//! about the centre of mass. Boundary/anchor contact is deliberately excluded.
use super::Mechanics;
use core_math::Vec3;
use core_world::{Representation, World};

#[derive(Default)]
pub(super) struct AngularState {
    pub momentum: Vec3,
    pub scale: f64,
    pub mass_radius: f64,
    pub mass: f64,
}
impl Mechanics {
    pub(super) fn angular_state(&self, world: &World) -> AngularState {
        let mut mass = 0.0;
        let mut centre = Vec3::ZERO;
        let mut momentum = Vec3::ZERO;
        for &object in self
            .layout
            .solids
            .iter()
            .map(|d| &d.object)
            .chain(self.layout.rigids.iter().map(|d| &d.object))
        {
            let o = &world.objects()[object];
            let m = o.mass();
            mass += m;
            centre += o.centre_of_mass().unwrap() * m;
            momentum += o.momentum();
        }
        if mass == 0.0 {
            return AngularState::default();
        }
        centre *= 1.0 / mass;
        let velocity = momentum / mass;
        let mut state = AngularState {
            mass,
            ..AngularState::default()
        };
        for d in &self.layout.solids {
            let Representation::Solid(s) = &world.objects()[d.object].representation else {
                unreachable!()
            };
            for ((&x, &v), &m) in s
                .nodes
                .position
                .iter()
                .zip(&s.nodes.velocity)
                .zip(&s.nodes.mass)
            {
                let r = x - centre;
                let p = (v - velocity) * m;
                state.momentum += r.cross(p);
                state.scale += r.length() * p.length();
                state.mass_radius += m * r.length();
            }
        }
        for d in &self.layout.rigids {
            let Representation::Rigid(r) = &world.objects()[d.object].representation else {
                unreachable!()
            };
            let arm = r.pose.position - centre;
            let p = (r.velocity - velocity) * r.mass;
            state.momentum += arm.cross(p) + r.angular_momentum();
            state.scale += arm.length() * p.length() + r.angular_momentum().length();
            state.mass_radius +=
                r.mass * (arm.length() + r.shape.local_bounds().size().length() * 0.5);
        }
        state
    }
}
