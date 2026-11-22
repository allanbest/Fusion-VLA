//! Small dense symmetric matrices, on the stack: an element's 12×12
//! Hessian, a rigid body's 12×12 block.

/// Eigenvalues and orthonormal eigenvectors (column `k` of the second
/// result belongs to value `k`) of a symmetric matrix, by cyclic Jacobi
/// rotations — slow in `N³` per sweep but exact to rounding, for the small
/// `N` it is used at.
#[allow(clippy::needless_range_loop)] // rotations read two entries per index
pub fn symmetric_eigen<const N: usize>(matrix: &[[f64; N]; N]) -> ([f64; N], [[f64; N]; N]) {
    let mut a = *matrix;
    let mut v = [[0.0; N]; N];
    for (i, row) in v.iter_mut().enumerate() {
        row[i] = 1.0;
    }
    for _ in 0..64 {
        let mut off = 0.0;
        let mut diagonal = 0.0;
        for i in 0..N {
            diagonal += a[i][i] * a[i][i];
            for j in i + 1..N {
                off += a[i][j] * a[i][j];
            }
        }
        if off <= 1e-30 * diagonal || off == 0.0 {
            break;
        }
        for p in 0..N {
            for q in p + 1..N {
                if a[p][q] == 0.0 {
                    continue;
                }
                let theta = (a[q][q] - a[p][p]) / (2.0 * a[p][q]);
                let t = theta.signum() / (theta.abs() + (theta * theta + 1.0).sqrt());
                let c = 1.0 / (t * t + 1.0).sqrt();
                let s = t * c;
                for row in a.iter_mut() {
                    let (akp, akq) = (row[p], row[q]);
                    row[p] = c * akp - s * akq;
                    row[q] = s * akp + c * akq;
                }
                for k in 0..N {
                    let (apk, aqk) = (a[p][k], a[q][k]);
                    a[p][k] = c * apk - s * aqk;
                    a[q][k] = s * apk + c * aqk;
                }
                for row in v.iter_mut() {
                    let (vp, vq) = (row[p], row[q]);
                    row[p] = c * vp - s * vq;
                    row[q] = s * vp + c * vq;
                }
            }
        }
    }
    let mut values = [0.0; N];
    for (i, value) in values.iter_mut().enumerate() {
        *value = a[i][i];
    }
    (values, v)
}

/// Cholesky positive-definiteness check with diagonal scaling: succeeds for
/// strictly SPD matrices without running full cyclic Jacobi eigen-sweeps.
#[allow(clippy::needless_range_loop)]
fn is_positive_definite<const N: usize>(matrix: &[[f64; N]; N]) -> bool {
    let mut scale = [0.0; N];
    for i in 0..N {
        let diag = matrix[i][i];
        if diag <= 0.0 || !diag.is_finite() {
            return false;
        }
        scale[i] = 1.0 / diag.sqrt();
    }
    let mut l = [[0.0; N]; N];
    for i in 0..N {
        for j in 0..=i {
            let a_ij = matrix[i][j] * scale[i] * scale[j];
            let a_ji = matrix[j][i] * scale[j] * scale[i];
            if !a_ij.is_finite() || (a_ij - a_ji).abs() > 1e-12 * a_ij.abs().max(1.0) {
                return false;
            }
            let mut sum = 0.5 * (a_ij + a_ji);
            for k in 0..j {
                sum -= l[i][k] * l[j][k];
            }
            if i == j {
                if sum <= 64.0 * f64::EPSILON || !sum.is_finite() {
                    return false;
                }
                l[i][j] = sum.sqrt();
            } else {
                l[i][j] = sum / l[j][j];
            }
        }
    }
    true
}

/// Replaces a symmetric matrix by the nearest positive semi-definite one:
/// negative eigenvalues set to zero. Newton then always steps downhill.
pub fn project_positive<const N: usize>(matrix: &mut [[f64; N]; N]) {
    if is_positive_definite(matrix) {
        return;
    }
    let (values, vectors) = symmetric_eigen(matrix);
    if values.iter().all(|&v| v >= 0.0) {
        return;
    }
    for i in 0..N {
        for j in 0..N {
            matrix[i][j] = (0..N)
                .map(|k| vectors[i][k] * values[k].max(0.0) * vectors[j][k])
                .sum();
        }
    }
}
