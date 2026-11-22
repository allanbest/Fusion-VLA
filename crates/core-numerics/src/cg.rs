//! Preconditioned conjugate gradients, for symmetric positive (semi-)
//! definite systems — the linear solve inside every Newton step.
//!
//! Each iteration is a matrix product and three parallel passes over the
//! vectors, each fusing what can be fused: `p·Ap`; the updates of `x` and
//! `r` with the preconditioner and both dot products; the new direction.
//! Sums are taken over fixed chunks in order, so the result does not depend
//! on the number of threads.

use std::sync::Mutex;

use core_math::{Mat3, Vec3};
use core_runtime::Pool;

use crate::vector::CHUNK;
use crate::{BlockOperator, LinearOperator};

/// An approximate inverse, applied to a residual. During one CG solve it
/// must define a fixed symmetric positive-definite map.
pub trait Preconditioner: Sync {
    /// `z = M⁻¹ r` on the part of the vectors starting at `offset` (a
    /// multiple of 3).
    fn apply_part(&self, offset: usize, r: &[f64], z: &mut [f64]);
}

/// No preconditioning.
pub struct Identity;

impl Preconditioner for Identity {
    fn apply_part(&self, _: usize, r: &[f64], z: &mut [f64]) {
        z.copy_from_slice(r);
    }
}

/// The inverse of each diagonal 3×3 block. Cheap, parallel, and it removes
/// the scale differences between degrees of freedom (a heavy node beside a
/// light one, a stiff element beside a soft one).
///
/// Groups of consecutive blocks that belong together — a rigid body's
/// position and orientation, strongly coupled through its lever arms — can
/// be inverted as one dense block instead.
#[derive(Debug, Default)]
pub struct BlockJacobi {
    inverse: Vec<Mat3>,
    groups: Vec<CoupledInverse>,
    grouped: Vec<bool>,
    scratch: Vec<Mutex<FactorScratch>>,
}

/// Buffers are retained across Newton iterations; each body factors independently.
#[derive(Clone, Debug, Default)]
struct CoupledInverse {
    first: usize,
    count: usize,
    inverse: Vec<f64>,
    valid: bool,
}

/// One scratch buffer per fixed batch, rather than per body. Applying the
/// preconditioner then touches only the compact inverse data.
#[derive(Clone, Debug, Default)]
struct FactorScratch {
    matrix: Vec<f64>,
    scale: Vec<f64>,
}

const GROUPS_PER_CHUNK: usize = 8;

impl Clone for BlockJacobi {
    fn clone(&self) -> Self {
        Self {
            inverse: self.inverse.clone(),
            groups: self.groups.clone(),
            grouped: self.grouped.clone(),
            scratch: self
                .scratch
                .iter()
                .map(|s| Mutex::new(s.lock().unwrap().clone()))
                .collect(),
        }
    }
}

impl BlockJacobi {
    pub fn new(pool: &Pool, matrix: &impl BlockOperator) -> Self {
        let mut p = Self::default();
        p.update(pool, matrix, &[]);
        p
    }

    /// Recomputes for a new matrix, retaining buffers. Each nonempty,
    /// nonoverlapping `(first, count)` group is inverted as one SPD block.
    /// A group that cannot be safely factored falls back to its 3x3 blocks.
    pub fn update(&mut self, pool: &Pool, matrix: &impl BlockOperator, groups: &[(usize, usize)]) {
        self.update_blocks(pool, matrix);
        self.grouped.clear();
        self.grouped.resize(self.inverse.len(), false);
        self.groups
            .resize_with(groups.len(), CoupledInverse::default);
        for (group, &(first, count)) in self.groups.iter_mut().zip(groups) {
            assert!(
                count > 0
                    && first
                        .checked_add(count)
                        .is_some_and(|end| end <= self.inverse.len()),
                "invalid preconditioner group"
            );
            group.first = first;
            group.count = count;
        }
        self.groups.sort_unstable_by_key(|g| g.first);
        assert!(
            self.groups
                .windows(2)
                .all(|g| g[0].first + g[0].count <= g[1].first),
            "overlapping preconditioner groups"
        );
        self.scratch
            .resize_with(self.groups.len().div_ceil(GROUPS_PER_CHUNK), Mutex::default);
        let scratch = &self.scratch;
        pool.for_chunks_mut(&mut self.groups, GROUPS_PER_CHUNK, |offset, part| {
            let mut workspace = scratch[offset / GROUPS_PER_CHUNK].lock().unwrap();
            let FactorScratch {
                matrix: buffer,
                scale,
            } = &mut *workspace;
            for group in part {
                let n = 3 * group.count;
                buffer.resize(n * n, 0.0);
                group.inverse.resize(n * n, 0.0);
                scale.resize(n, 0.0);
                for r in 0..group.count {
                    for c in 0..group.count {
                        let block = matrix.block(group.first + r, group.first + c);
                        for i in 0..3 {
                            for j in 0..3 {
                                buffer[(3 * r + i) * n + 3 * c + j] = block[(i, j)];
                            }
                        }
                    }
                }
                group.valid = invert_positive(n, buffer, &mut group.inverse, scale);
            }
        });
        for group in &self.groups {
            if group.valid {
                self.grouped[group.first..group.first + group.count].fill(true);
            }
        }
    }

    fn update_blocks(&mut self, pool: &Pool, matrix: &impl BlockOperator) {
        self.inverse.clear();
        self.inverse.resize(matrix.dimension() / 3, Mat3::ZERO);
        matrix.add_diagonal(&mut self.inverse);
        pool.for_chunks_mut(&mut self.inverse, CHUNK / 3, |_, part| {
            for d in part {
                let mut m = [0.0; 9];
                for i in 0..3 {
                    for j in 0..3 {
                        m[3 * i + j] = d[(i, j)];
                    }
                }
                let mut inverse = [0.0; 9];
                if invert_positive(3, &mut m, &mut inverse, &mut [0.0; 3]) {
                    *d = Mat3::from_rows(
                        Vec3::new(inverse[0], inverse[1], inverse[2]),
                        Vec3::new(inverse[3], inverse[4], inverse[5]),
                        Vec3::new(inverse[6], inverse[7], inverse[8]),
                    );
                } else {
                    // Always keep the approximate inverse positive and finite;
                    // an indefinite local block must not reverse the residual.
                    let diag = d.diagonal();
                    let pick = |v: f64| {
                        let inverse = 1.0 / v;
                        if inverse > 0.0 && inverse.is_finite() {
                            inverse
                        } else {
                            1.0
                        }
                    };
                    *d = Mat3::from_diagonal(Vec3::new(pick(diag.x), pick(diag.y), pick(diag.z)));
                }
            }
        });
    }
}

impl Preconditioner for BlockJacobi {
    fn apply_part(&self, offset: usize, r: &[f64], z: &mut [f64]) {
        let first = offset / 3;
        let last = first + z.len() / 3;
        for (k, out) in z.chunks_exact_mut(3).enumerate() {
            if !self.grouped[first + k] {
                out.copy_from_slice(&(self.inverse[first + k] * Vec3::read(r, k)).to_array());
            }
        }
        // Sorted groups allow each vector chunk to visit just its own bodies.
        let start = self.groups.partition_point(|g| g.first + g.count <= first);
        for group in self.groups[start..].iter().take_while(|g| g.first < last) {
            if !group.valid {
                continue;
            }
            let (b, count, inverse) = (group.first, group.count, &group.inverse);
            if b >= first && b + count <= last {
                let n = 3 * count;
                let base = 3 * (b - first);
                for i in 0..n {
                    z[base + i] = (0..n).map(|j| inverse[i * n + j] * r[base + j]).sum();
                }
            } else {
                // A group straddling a fixed CG chunk uses its 3x3 inverses
                // on both sides, preserving a fixed symmetric preconditioner.
                for k in b.max(first)..(b + count).min(last) {
                    let out = &mut z[3 * (k - first)..3 * (k - first) + 3];
                    out.copy_from_slice(&(self.inverse[k] * Vec3::read(r, k - first)).to_array());
                }
            }
        }
    }
}

/// Diagonally equilibrated Cholesky inverse of an SPD matrix. Operates in
/// caller-owned buffers. Scaling makes the pivot test independent of mass
/// units and avoids rejecting sound blocks solely for disparate scales.
/// The factorization and triangular inverse follow the POTRF/POTRI route:
/// https://www.netlib.org/lapack/explore-html/d4/d12/group__potri_gaced925926290482532c433f73b30b9f1.html
fn invert_positive(n: usize, a: &mut [f64], inverse: &mut [f64], scale: &mut [f64]) -> bool {
    for i in 0..n {
        let diagonal = a[i * n + i];
        if diagonal <= 0.0 || !diagonal.is_finite() {
            return false;
        }
        scale[i] = 1.0 / diagonal.sqrt();
    }
    for i in 0..n {
        for j in 0..=i {
            let lower = (a[i * n + j] * scale[i]) * scale[j];
            let upper = (a[j * n + i] * scale[j]) * scale[i];
            if !lower.is_finite()
                || !upper.is_finite()
                || (lower - upper).abs() > 1e-10 * lower.abs().max(upper.abs()).max(1.0)
            {
                return false;
            }
            a[i * n + j] = 0.5 * (lower + upper);
        }
    }
    // A = L L^T, with unit-scale pivots after equilibration.
    for i in 0..n {
        for j in 0..=i {
            let mut value = a[i * n + j];
            for k in 0..j {
                value -= a[i * n + k] * a[j * n + k];
            }
            if j == i {
                if value <= 64.0 * f64::EPSILON || !value.is_finite() {
                    return false;
                }
                a[i * n + j] = value.sqrt();
            } else {
                a[i * n + j] = value / a[j * n + j];
            }
        }
    }
    inverse.fill(0.0);
    // L^-1, by forward substitution.
    for i in 0..n {
        inverse[i * n + i] = 1.0 / a[i * n + i];
        for j in 0..i {
            let mut value = 0.0;
            for k in j..i {
                value -= a[i * n + k] * inverse[k * n + j];
            }
            inverse[i * n + j] = value / a[i * n + i];
        }
    }
    // D L^-T L^-1 D; write the same rounded value on both sides.
    for i in 0..n {
        for j in 0..=i {
            let mut value = 0.0;
            for k in i..n {
                value += inverse[k * n + i] * inverse[k * n + j];
            }
            value = (value * scale[i]) * scale[j];
            if !value.is_finite() {
                return false;
            }
            a[i * n + j] = value;
            a[j * n + i] = value;
        }
    }
    inverse.copy_from_slice(a);
    true
}

/// When to stop.
#[derive(Clone, Copy, Debug)]
pub struct CgSettings {
    /// Stop when `|r| ≤ tolerance |b|`.
    pub tolerance: f64,
    pub max_iterations: usize,
}

impl Default for CgSettings {
    fn default() -> Self {
        Self {
            tolerance: 1e-8,
            max_iterations: 10_000,
        }
    }
}

/// What a solve did.
#[derive(Clone, Copy, Debug, Default)]
pub struct CgReport {
    pub iterations: usize,
    /// True `|b-Ax| / |b|` at return (not just the recursive estimate).
    pub residual: f64,
    pub converged: bool,
    /// Explicit equation-residual evaluations beyond the initial one.
    pub residual_checks: usize,
    /// Direction restarts after a recursive residual falsely appeared converged.
    pub residual_restarts: usize,
    /// Matrix-vector products, including initial/true-residual evaluations.
    pub matrix_products: usize,
    /// The matrix showed a direction of zero or negative curvature, so the
    /// solve stopped early (truncated); `x` is still a descent direction.
    pub indefinite: bool,
    /// The right-hand side or the iteration produced a non-finite number.
    pub not_finite: bool,
}

/// Reusable memory for [`conjugate_gradient`].
#[derive(Clone, Debug, Default)]
pub struct CgWorkspace {
    r: Vec<f64>,
    z: Vec<f64>,
    p: Vec<f64>,
    ap: Vec<f64>,
}

/// Solves `A x = b` from the starting guess in `x`.
pub fn conjugate_gradient(
    pool: &Pool,
    a: &impl LinearOperator,
    preconditioner: &impl Preconditioner,
    b: &[f64],
    x: &mut [f64],
    settings: CgSettings,
    work: &mut CgWorkspace,
) -> CgReport {
    let n = b.len();
    debug_assert_eq!(a.dimension(), n);
    for v in [&mut work.r, &mut work.z, &mut work.p, &mut work.ap] {
        v.clear();
        v.resize(n, 0.0);
    }
    let CgWorkspace { r, z, p, ap } = work;
    let add = |x: (f64, f64), y: (f64, f64)| (x.0 + y.0, x.1 + y.1);

    let b_norm = crate::vector::norm_squared(pool, b).sqrt();
    if !b_norm.is_finite() {
        return CgReport {
            not_finite: true,
            residual: f64::NAN,
            ..Default::default()
        };
    }
    if b_norm == 0.0 {
        x.fill(0.0);
        return CgReport {
            converged: true,
            ..Default::default()
        };
    }

    let (mut rz, mut rr, initial_products) = if x.iter().all(|&v| v == 0.0) {
        let (rz, rr) = pool.split_reduce(
            (&mut r[..], &mut z[..], &mut p[..]),
            CHUNK,
            (0.0, 0.0),
            |offset, (r, z, p)| {
                r.copy_from_slice(&b[offset..offset + r.len()]);
                preconditioner.apply_part(offset, r, z);
                p.copy_from_slice(z);
                (dot(r, z), dot(r, r))
            },
            add,
        );
        (rz, rr, 0)
    } else {
        // r = b − A x, z = M⁻¹ r, p = z; returns (r·z, r·r).
        a.apply(pool, x, r);
        let (rz, rr) = pool.split_reduce(
            (&mut r[..], &mut z[..], &mut p[..]),
            CHUNK,
            (0.0, 0.0),
            |offset, (r, z, p)| {
                for (ri, bi) in r.iter_mut().zip(&b[offset..]) {
                    *ri = bi - *ri;
                }
                preconditioner.apply_part(offset, r, z);
                p.copy_from_slice(z);
                (dot(r, z), dot(r, r))
            },
            add,
        );
        (rz, rr, 1)
    };

    let target = settings.tolerance * b_norm;
    let mut report = CgReport {
        matrix_products: initial_products,
        ..CgReport::default()
    };
    for k in 0..settings.max_iterations {
        report.residual = rr.sqrt() / b_norm;
        if !rr.is_finite() {
            report.not_finite = true;
            return report;
        }
        if rr.sqrt() <= target {
            // Cancellation can make the recursive residual optimistic on
            // ill-conditioned stiffness/mass systems. Verify before accepting.
            a.apply(pool, x, ap);
            report.residual_checks += 1;
            report.matrix_products += 1;
            (rz, rr) = pool.split_reduce(
                (&mut r[..], &mut z[..], &mut p[..]),
                CHUNK,
                (0.0, 0.0),
                |offset, (r, z, p)| {
                    for (k, ri) in r.iter_mut().enumerate() {
                        *ri = b[offset + k] - ap[offset + k];
                    }
                    preconditioner.apply_part(offset, r, z);
                    p.copy_from_slice(z);
                    (dot(r, z), dot(r, r))
                },
                add,
            );
            report.residual = rr.sqrt() / b_norm;
            if !rr.is_finite() {
                report.not_finite = true;
                return report;
            }
            if rr.sqrt() <= target {
                report.converged = true;
                return report;
            }
            report.residual_restarts += 1;
        }
        // ap = A p, then p·Ap. The product is cut finer than the sum: each
        // row stands alone, while the sum is taken over the fixed chunks in
        // order (so it is the same number however the product was shared).
        let p_ref: &[f64] = p;
        report.matrix_products += 1;
        a.apply(pool, p_ref, ap);
        let ap_ref: &[f64] = ap;
        let curvature = pool.map_reduce(
            n,
            CHUNK,
            0.0,
            |range| dot(&p_ref[range.clone()], &ap_ref[range]),
            |x, y| x + y,
        );
        if curvature <= 0.0 || !curvature.is_finite() {
            report.indefinite = curvature.is_finite();
            report.not_finite = !curvature.is_finite();
            if k == 0 && curvature.is_finite() {
                // Nothing learnt yet: the preconditioned residual is still
                // a descent direction.
                for (xi, pi) in x.iter_mut().zip(p.iter()) {
                    *xi += pi;
                }
            }
            a.apply(pool, x, ap);
            report.residual_checks += 1;
            report.matrix_products += 1;
            for (r, (&b, &ax)) in r.iter_mut().zip(b.iter().zip(ap.iter())) {
                *r = b - ax;
            }
            report.residual = crate::vector::norm_squared(pool, r).sqrt() / b_norm;
            report.not_finite |= !report.residual.is_finite();
            return report;
        }
        let alpha = rz / curvature;
        // x += α p, r −= α Ap, z = M⁻¹ r; returns (r·z, r·r).
        let ap_ref: &[f64] = ap;
        let (rz_next, rr_next) = pool.split_reduce(
            (&mut x[..], &mut r[..], &mut z[..]),
            CHUNK,
            (0.0, 0.0),
            |offset, (x, r, z)| {
                let len = x.len();
                for k in 0..len {
                    x[k] += alpha * p_ref[offset + k];
                    r[k] -= alpha * ap_ref[offset + k];
                }
                preconditioner.apply_part(offset, r, z);
                (dot(r, z), dot(r, r))
            },
            add,
        );
        let beta = rz_next / rz;
        rz = rz_next;
        rr = rr_next;
        // p = z + β p
        pool.split_reduce(
            (&mut p[..], &z[..]),
            CHUNK,
            (),
            |_, (p, z)| {
                for (pi, zi) in p.iter_mut().zip(z) {
                    *pi = zi + beta * *pi;
                }
            },
            |_, _| (),
        );
        report.iterations = k + 1;
    }
    a.apply(pool, x, ap);
    report.residual_checks += 1;
    report.matrix_products += 1;
    for (r, (&b, &ax)) in r.iter_mut().zip(b.iter().zip(ap.iter())) {
        *r = b - ax;
    }
    report.residual = crate::vector::norm_squared(pool, r).sqrt() / b_norm;
    report.not_finite |= !report.residual.is_finite();
    report.converged = !report.not_finite && report.residual <= settings.tolerance;
    report
}

/// A dot product over one chunk, with four running sums so that it
/// vectorises; their order is fixed, so it is deterministic.
#[inline]
pub(crate) fn dot(a: &[f64], b: &[f64]) -> f64 {
    let mut s = [0.0; 4];
    let split = a.len() / 4 * 4;
    for (x, y) in a[..split].chunks_exact(4).zip(b[..split].chunks_exact(4)) {
        for k in 0..4 {
            s[k] += x[k] * y[k];
        }
    }
    let tail: f64 = a[split..].iter().zip(&b[split..]).map(|(x, y)| x * y).sum();
    (s[0] + s[1]) + (s[2] + s[3]) + tail
}
