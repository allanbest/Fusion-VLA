//! Checking derivatives against finite differences — every energy term's
//! gradient and Hessian is tested this way before it is trusted.

/// The largest error of `gradient` against central differences of `energy`
/// at `x`, relative to the gradient's largest component.
pub fn gradient_error(
    mut energy: impl FnMut(&[f64]) -> f64,
    x: &[f64],
    gradient: &[f64],
    step: f64,
) -> f64 {
    let mut probe = x.to_vec();
    let scale = gradient
        .iter()
        .fold(0.0f64, |m, g| m.max(g.abs()))
        .max(f64::MIN_POSITIVE);
    let mut worst = 0.0f64;
    for i in 0..x.len() {
        probe[i] = x[i] + step;
        let up = energy(&probe);
        probe[i] = x[i] - step;
        let down = energy(&probe);
        probe[i] = x[i];
        worst = worst.max(((up - down) / (2.0 * step) - gradient[i]).abs());
    }
    worst / scale
}

/// The largest error of a dense `hessian` (row-major, `n × n`) against
/// central differences of `gradient` at `x`, relative to the Hessian's
/// largest entry.
pub fn hessian_error(
    mut gradient: impl FnMut(&[f64], &mut [f64]),
    x: &[f64],
    hessian: &[f64],
    step: f64,
) -> f64 {
    let n = x.len();
    let mut probe = x.to_vec();
    let (mut up, mut down) = (vec![0.0; n], vec![0.0; n]);
    let scale = hessian
        .iter()
        .fold(0.0f64, |m, h| m.max(h.abs()))
        .max(f64::MIN_POSITIVE);
    let mut worst = 0.0f64;
    for j in 0..n {
        probe[j] = x[j] + step;
        up.fill(0.0);
        gradient(&probe, &mut up);
        probe[j] = x[j] - step;
        down.fill(0.0);
        gradient(&probe, &mut down);
        probe[j] = x[j];
        for i in 0..n {
            worst = worst.max(((up[i] - down[i]) / (2.0 * step) - hessian[i * n + j]).abs());
        }
    }
    worst / scale
}
