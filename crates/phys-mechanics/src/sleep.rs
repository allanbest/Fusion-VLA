//! Exact equilibrium reuse, with no velocity threshold or approximate freezing.
use super::{Mechanics, MechanicsState};
use crate::elastic::SolidPlastic;
use core_math::{Mat3, Quat, Vec3};
use core_world::{Representation, World};

pub(super) fn same_vec(a: Vec3, b: Vec3) -> bool {
    a.to_array().map(f64::to_bits) == b.to_array().map(f64::to_bits)
}
fn same_slice(a: &[f64], b: &[f64]) -> bool {
    a.len() == b.len() && a.iter().zip(b).all(|(a, b)| a.to_bits() == b.to_bits())
}
fn same_mat(a: &Mat3, b: &Mat3) -> bool {
    a.cols.iter().zip(&b.cols).all(|(&x, &y)| same_vec(x, y))
}
pub(super) fn same_plastic(a: &SolidPlastic, b: &SolidPlastic) -> bool {
    a.fp_inv.len() == b.fp_inv.len()
        && same_slice(&a.eps_p, &b.eps_p)
        && a.fp_inv.iter().zip(&b.fp_inv).all(|(x, y)| same_mat(x, y))
}
fn same_option(a: &Option<Vec<f64>>, b: &Option<Vec<f64>>) -> bool {
    match (a, b) {
        (Some(a), Some(b)) => same_slice(a, b),
        (None, None) => true,
        _ => false,
    }
}
fn same_affine(a: &Option<(Quat, Mat3)>, b: &Option<(Quat, Mat3)>) -> bool {
    match (a, b) {
        (Some((aq, am)), Some((bq, bm))) => {
            same_quat(*aq, *bq) && am.cols.iter().zip(&bm.cols).all(|(&a, &b)| same_vec(a, b))
        }
        (None, None) => true,
        _ => false,
    }
}
pub(super) fn same_quat(a: Quat, b: Quat) -> bool {
    [a.x, a.y, a.z, a.w].map(f64::to_bits) == [b.x, b.y, b.z, b.w].map(f64::to_bits)
}

impl Mechanics {
    pub(super) fn stationary_match(&self, world: &World, state: &MechanicsState) -> bool {
        if self.previous_h.map(f64::to_bits) != state.previous_h.map(f64::to_bits)
            || self.gravity_seen.map(|v| v.to_array().map(f64::to_bits))
                != state.gravity_seen.map(|v| v.to_array().map(f64::to_bits))
        {
            return false;
        }
        if !self
            .acceleration
            .as_ref()
            .is_some_and(|a| a.iter().all(|&v| v == 0.0))
            || !same_option(&self.acceleration, &state.acceleration)
            || self.restart != state.restart
            || !same_slice(&self.unloading, &state.unloading)
            || self.plastic.len() != state.plastic.len()
            || !self
                .plastic
                .iter()
                .zip(&state.plastic)
                .all(|(a, b)| same_plastic(a, b))
            || !self
                .affine
                .iter()
                .zip(&state.affine)
                .all(|(a, b)| same_affine(a, b))
        {
            return false;
        }
        match (&self.earlier, &state.earlier) {
            (Some((ax, av)), Some((bx, bv))) if same_slice(ax, bx) && same_slice(av, bv) => {}
            (None, None) => {}
            _ => return false,
        }
        for (d, (p, v)) in self.layout.solids.iter().zip(&state.nodes) {
            let Representation::Solid(s) = &world.objects()[d.object].representation else {
                unreachable!()
            };
            if !s
                .nodes
                .position
                .iter()
                .zip(p)
                .all(|(&a, &b)| same_vec(a, b))
                || !s
                    .nodes
                    .velocity
                    .iter()
                    .zip(v)
                    .all(|(&a, &b)| a == Vec3::ZERO && same_vec(a, b))
            {
                return false;
            }
        }
        for (d, (p, v, w)) in self.layout.rigids.iter().zip(&state.rigids) {
            let Representation::Rigid(r) = &world.objects()[d.object].representation else {
                unreachable!()
            };
            if r.velocity != Vec3::ZERO
                || r.spin != Vec3::ZERO
                || !same_vec(r.pose.position, p.position)
                || !same_quat(r.pose.rotation, p.rotation)
                || !same_vec(r.velocity, *v)
                || !same_vec(r.spin, *w)
            {
                return false;
            }
        }
        true
    }

    pub(super) fn mapped_history_matches(
        &self,
        parent: &Self,
        blocks: &[usize],
        rigids: &[usize],
        solids: &[usize],
    ) -> bool {
        let mapped = |a: &Option<Vec<f64>>, b: &Option<Vec<f64>>| match (a, b) {
            (Some(a), Some(b)) => blocks
                .iter()
                .enumerate()
                .all(|(i, &j)| same_slice(&a[3 * i..3 * i + 3], &b[3 * j..3 * j + 3])),
            (None, None) => true,
            _ => false,
        };
        if !mapped(&self.acceleration, &parent.acceleration) {
            return false;
        }
        match (&self.earlier, &parent.earlier) {
            (Some((ax, av)), Some((bx, bv))) => {
                if !blocks.iter().enumerate().all(|(i, &j)| {
                    same_slice(&ax[3 * i..3 * i + 3], &bx[3 * j..3 * j + 3])
                        && same_slice(&av[3 * i..3 * i + 3], &bv[3 * j..3 * j + 3])
                }) {
                    return false;
                }
            }
            (None, None) => {}
            _ => return false,
        }
        if self.previous_h.map(f64::to_bits) != parent.previous_h.map(f64::to_bits)
            || self.gravity_seen.map(|v| v.to_array().map(f64::to_bits))
                != parent.gravity_seen.map(|v| v.to_array().map(f64::to_bits))
        {
            return false;
        }
        blocks
            .iter()
            .enumerate()
            .all(|(i, &b)| self.restart[i] == parent.restart[b])
            && rigids.iter().enumerate().all(|(i, &r)| {
                same_affine(&self.affine[i], &parent.affine[r])
                    && self.unloading[i].to_bits() == parent.unloading[r].to_bits()
            })
            && solids
                .iter()
                .enumerate()
                .all(|(i, &s)| same_plastic(&self.plastic[i], &parent.plastic[s]))
    }
}
