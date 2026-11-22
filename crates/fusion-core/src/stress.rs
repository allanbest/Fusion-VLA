//! Stress inside a rigid body, and what a body does when it is too much.
//!
//! A rigid body does not deform, but it is still loaded: the floor pushes up
//! on a block's foot and its own weight pulls down through all of it, a ball
//! landing on concrete is decelerated by a force concentrated where it
//! touches, a hot core pushes on a cold skin. Small elastic strains carry those
//! loads, and they are small enough that the motion can stay rigid while the
//! stress is worked out on the side — the quasi-static "inertia relief"
//! analysis structural engineers use for a free-flying body.
//!
//! This module is the state: one stress tensor per cell of the body's
//! [`Voxels`], the same grid temperature uses, and the per-body choice of
//! what happens when the stress exceeds what the material can carry. The
//! solve lives in `fusion-sim`, because it needs the contacts.

use fusion_math::{Mat3, Vec3};

use crate::voxels::Voxels;

/// What a rigid body does when the load on it is more than its material can
/// carry.
///
/// A choice per object, not per material, because it is a choice about the
/// *scene*: the same steel is a beam that should be allowed to fail in one
/// shot and a floor that must never move in another.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash)]
pub enum Failure {
    /// Stays whole whatever happens. Its stress is still measured.
    #[default]
    Never,
    /// Cracks into rigid pieces where its stress first exceeds its strength,
    /// across the direction it was being pulled apart in. The harder it was
    /// overloaded, the more pieces.
    Breaks,
    /// Stops being rigid the moment it yields, cracks or reaches its melting
    /// point, and carries on as a deformable continuum that can bend, dent,
    /// tear and melt.
    Deforms,
}

impl Failure {
    /// Every option, in the order an editor offers them, with a label.
    pub const ALL: [(Self, &'static str); 3] = [
        (Self::Never, "Stays whole"),
        (Self::Breaks, "Breaks into pieces"),
        (Self::Deforms, "Deforms and melts"),
    ];

    /// True for anything but [`Failure::Never`].
    #[inline]
    pub fn can_fail(self) -> bool {
        self != Self::Never
    }
}

/// The stress field of one rigid body, in its own frame.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct BodyStress {
    /// The cells. Empty until the body is first loaded.
    pub grid: Voxels,
    /// Cauchy stress per cell, Pa, body frame, as
    /// `[xx, yy, zz, xy, yz, zx]`. Tension is positive.
    pub stress: Vec<[f32; 6]>,
    /// Node displacements from the last solve, scaled by Young's modulus so
    /// the solve does not care how stiff the material is. Kept to start the
    /// next solve from, which is most of what makes a steady load cheap.
    pub displacement: Vec<f64>,
    /// The steady load vector that displacement answers, kept so a step whose
    /// loads have not moved can keep its answer rather than re-derive it.
    /// Empty after an impact, which has to be solved again.
    pub load: Vec<f64>,
    /// The displacement per unit impulse of the last impact, kept to start
    /// the next one from: a body knocked the same way step after step — a
    /// shard settling on a pile — then costs a few iterations rather than a
    /// cold solve each time.
    pub impact_displacement: Vec<f64>,
    /// Largest von Mises stress in the body, Pa.
    pub peak_von_mises: f32,
    /// Largest ratio of load to strength anywhere in the body. One is the
    /// point at which the material first gives way: yields if it is ductile,
    /// cracks if it is brittle.
    pub utilisation: f32,
    /// The cell nearest to parting: where the load over the strength the
    /// material breaks at is greatest. The same cell as
    /// [`BodyStress::utilisation`]'s for a brittle material; for a metal,
    /// which yields well before it parts, or wood crushed across its grain,
    /// which dents, it can be another.
    pub worst: usize,
    /// Where in the body it is reached, body-local: one of that cell's
    /// sample points, which for a bent beam is nearer its surface than the
    /// cell's centre.
    pub worst_point: Vec3,
    /// The stress there, `[xx, yy, zz, xy, yz, zx]`, Pa. Its largest
    /// principal direction is the one a crack opens across.
    pub worst_stress: [f32; 6],
    /// Fraction of the body's cells loaded past the strength they part at somewhere: how
    /// much of it is failing, where [`BodyStress::utilisation`] says how badly
    /// the worst of it is. A corner chipped by a landing is a few percent; a
    /// pane hit square by a hammer is most of it.
    pub overloaded: f32,
    /// Load over the strength the material parts at, at [`BodyStress::worst`].
    /// One or more and a body that breaks, breaks.
    pub fracture: f32,
    /// The plane it would part on there: its normal, body frame.
    pub worst_normal: Vec3,
    /// How it would part there.
    pub worst_parting: crate::Parting,
    /// Steps left before it may break again: a piece just broken off settles
    /// first. Kept when the grid is rebuilt.
    pub settling: u32,
}

impl BodyStress {
    /// Whether the grid exists.
    #[inline]
    pub fn is_built(&self) -> bool {
        self.grid.is_built()
    }

    /// The stress tensor of a cell as a matrix.
    pub fn tensor(&self, i: usize) -> Mat3 {
        let [xx, yy, zz, xy, yz, zx] = self.stress[i];
        Mat3::from_cols(
            Vec3::new(xx, xy, zx),
            Vec3::new(xy, yy, yz),
            Vec3::new(zx, yz, zz),
        )
    }

    /// Von Mises equivalent stress of a cell, Pa: the single number a ductile
    /// material yields on.
    pub fn von_mises(&self, i: usize) -> f32 {
        von_mises(self.stress[i])
    }

    /// Principal stresses of a cell, largest first, with their directions.
    pub fn principal(&self, i: usize) -> ([f32; 3], [Vec3; 3]) {
        principal(self.stress[i])
    }

    /// Von Mises stress at a body-local point: the nearest cell's, or zero for
    /// a body that has never been loaded.
    pub fn von_mises_at_local(&self, p: Vec3) -> f32 {
        match self.grid.nearest_inside(p) {
            Some(i) if i < self.stress.len() => self.von_mises(i),
            _ => 0.0,
        }
    }

    /// Back to unloaded, keeping the grid.
    pub fn clear(&mut self) {
        for s in &mut self.stress {
            *s = [0.0; 6];
        }
        self.peak_von_mises = 0.0;
        self.utilisation = 0.0;
        self.worst = 0;
        self.worst_point = Vec3::ZERO;
        self.worst_stress = [0.0; 6];
        self.overloaded = 0.0;
        self.fracture = 0.0;
        self.worst_normal = Vec3::X;
        self.worst_parting = crate::Parting::Crack;
    }
}

/// Von Mises equivalent stress of a tensor in `[xx, yy, zz, xy, yz, zx]`.
pub fn von_mises(s: [f32; 6]) -> f32 {
    let [xx, yy, zz, xy, yz, zx] = s.map(|v| v as f64);
    let j2 = ((xx - yy).powi(2) + (yy - zz).powi(2) + (zz - xx).powi(2)) / 6.0
        + xy * xy
        + yz * yz
        + zx * zx;
    (3.0 * j2).max(0.0).sqrt() as f32
}

/// Principal stresses alone, largest first: the eigenvalues of a symmetric
/// tensor in `[xx, yy, zz, xy, yz, zx]`, in closed form.
///
/// The trigonometric solution of the characteristic cubic, in double. Used
/// where only the values matter — a failure check at every sample point of
/// every cell, every step — and [`principal`] where the directions do too.
pub fn principal_values(s: [f32; 6]) -> [f32; 3] {
    let [xx, yy, zz, xy, yz, zx] = s.map(|v| v as f64);
    let off = xy * xy + yz * yz + zx * zx;
    let mean = (xx + yy + zz) / 3.0;
    let scale = xx.abs().max(yy.abs()).max(zz.abs()).max(off.sqrt());
    if off <= 1e-24 * scale * scale {
        let mut v = [xx, yy, zz];
        v.sort_by(|a, b| b.total_cmp(a));
        return v.map(|x| x as f32);
    }
    let (a, b, c) = (xx - mean, yy - mean, zz - mean);
    let p2 = a * a + b * b + c * c + 2.0 * off;
    let p = (p2 / 6.0).sqrt();
    // det((A - mean I) / p) / 2, clamped against rounding past ±1.
    let det = a * (b * c - yz * yz) - xy * (xy * c - yz * zx) + zx * (xy * yz - b * zx);
    let r = (det / (2.0 * p * p * p)).clamp(-1.0, 1.0);
    let phi = r.acos() / 3.0;
    let largest = mean + 2.0 * p * phi.cos();
    let smallest = mean + 2.0 * p * (phi + 2.0 * std::f64::consts::FRAC_PI_3).cos();
    let middle = 3.0 * mean - largest - smallest;
    [largest as f32, middle as f32, smallest as f32]
}

/// Eigenvalues and eigenvectors of a symmetric tensor in
/// `[xx, yy, zz, xy, yz, zx]`, largest first.
///
/// Cyclic Jacobi rotations in double precision. A closed-form cubic is
/// shorter and loses every digit when two principal stresses are nearly
/// equal, which for a block resting on a floor is the usual case rather than
/// an edge one.
// Index loops, because a Jacobi rotation is written in indices everywhere it
// is written down, and an iterator over one row of three would hide which row.
#[allow(clippy::needless_range_loop)]
pub fn principal(s: [f32; 6]) -> ([f32; 3], [Vec3; 3]) {
    let [xx, yy, zz, xy, yz, zx] = s.map(|v| v as f64);
    let mut a = [[xx, xy, zx], [xy, yy, yz], [zx, yz, zz]];
    let mut v = [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0f64]];
    for _ in 0..32 {
        let off = a[0][1].abs() + a[1][2].abs() + a[0][2].abs();
        let scale = a[0][0].abs() + a[1][1].abs() + a[2][2].abs() + off;
        if off <= 1e-14 * scale.max(1e-300) {
            break;
        }
        for (p, q) in [(0, 1), (1, 2), (0, 2)] {
            if a[p][q].abs() <= 1e-300 {
                continue;
            }
            let theta = (a[q][q] - a[p][p]) / (2.0 * a[p][q]);
            let t = theta.signum() / (theta.abs() + (theta * theta + 1.0).sqrt());
            let t = if theta == 0.0 { 1.0 } else { t };
            let c = 1.0 / (t * t + 1.0).sqrt();
            let sn = t * c;
            for k in 0..3 {
                let (akp, akq) = (a[k][p], a[k][q]);
                a[k][p] = c * akp - sn * akq;
                a[k][q] = sn * akp + c * akq;
            }
            for k in 0..3 {
                let (apk, aqk) = (a[p][k], a[q][k]);
                a[p][k] = c * apk - sn * aqk;
                a[q][k] = sn * apk + c * aqk;
            }
            for row in &mut v {
                let (vp, vq) = (row[p], row[q]);
                row[p] = c * vp - sn * vq;
                row[q] = sn * vp + c * vq;
            }
        }
    }
    let mut order = [0usize, 1, 2];
    order.sort_by(|&i, &j| a[j][j].total_cmp(&a[i][i]));
    let values = order.map(|i| a[i][i] as f32);
    let vectors = order.map(|i| Vec3::new(v[0][i] as f32, v[1][i] as f32, v[2][i] as f32));
    (values, vectors)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn uniaxial_tension_is_its_own_von_mises() {
        assert!((von_mises([5.0, 0.0, 0.0, 0.0, 0.0, 0.0]) - 5.0).abs() < 1e-5);
        // Hydrostatic pressure has no deviatoric part at all.
        assert!(von_mises([-3.0, -3.0, -3.0, 0.0, 0.0, 0.0]).abs() < 1e-5);
    }

    #[test]
    fn pure_shear_has_principal_stresses_at_45_degrees() {
        let (values, vectors) = principal([0.0, 0.0, 0.0, 2.0, 0.0, 0.0]);
        assert!((values[0] - 2.0).abs() < 1e-5, "{values:?}");
        assert!(values[1].abs() < 1e-5, "{values:?}");
        assert!((values[2] + 2.0).abs() < 1e-5, "{values:?}");
        let d = vectors[0];
        assert!(
            (d.x.abs() - d.y.abs()).abs() < 1e-4 && d.z.abs() < 1e-4,
            "{d:?}"
        );
        // Pure shear's von Mises is sqrt(3) times the shear.
        assert!((von_mises([0.0, 0.0, 0.0, 2.0, 0.0, 0.0]) - 2.0 * 3f32.sqrt()).abs() < 1e-4);
    }

    #[test]
    fn the_closed_form_agrees_with_the_rotations() {
        for s in [
            [3.0, -1.0, 2.0, 0.5, -0.7, 1.2],
            [1.0, 1.0, 1.0, 0.0, 0.0, 0.0],
            [5.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 2.0, 0.0, 0.0],
            [1e8, -2e7, 3e6, 4e6, -5e5, 1e3],
        ] {
            let (jacobi, _) = principal(s);
            let closed = principal_values(s);
            let scale = s.iter().fold(0.0f32, |m, v| m.max(v.abs()));
            for k in 0..3 {
                assert!(
                    (jacobi[k] - closed[k]).abs() <= 1e-5 * scale,
                    "{s:?}: {jacobi:?} {closed:?}"
                );
            }
        }
    }

    #[test]
    fn principal_values_reconstruct_the_tensor() {
        let s = [3.0, -1.0, 2.0, 0.5, -0.7, 1.2];
        let (values, vectors) = principal(s);
        let mut m = Mat3::ZERO;
        for k in 0..3 {
            m += fusion_math::outer_product(vectors[k], vectors[k]) * values[k];
        }
        let back = [
            m.x_axis.x, m.y_axis.y, m.z_axis.z, m.y_axis.x, m.z_axis.y, m.x_axis.z,
        ];
        for k in 0..6 {
            assert!((back[k] - s[k]).abs() < 1e-4, "{back:?} vs {s:?}");
        }
    }
}
