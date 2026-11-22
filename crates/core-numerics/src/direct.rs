//! Regularized variable-band (skyline) sparse direct `LDLᵀ` solver for
//! symmetric positive-definite and regularized block operators.

use std::collections::VecDeque;

use core_math::{Mat3, Vec3};

use crate::BlockOperator;

/// Variable-band symmetric `LDLᵀ` factorization over a [`BlockOperator`],
/// using Reverse Cuthill-McKee block ordering to minimize fill-in.
#[derive(Clone, Debug, Default)]
pub struct SparseCholesky {
    blocks: usize,
    n: usize,
    /// `new_block -> old_block`
    perm: Vec<usize>,
    /// `old_block -> new_block`
    inv_perm: Vec<usize>,
    first_col: Vec<usize>,
    row_start: Vec<usize>,
    lower: Vec<f64>,
    diag: Vec<f64>,
    scale: Vec<f64>,
    had_regularization: bool,
}

impl SparseCholesky {
    /// Factors `matrix` into `P A Pᵀ ≈ L D Lᵀ`, clamping any non-positive pivots
    /// to a small positive floor so indefinite or rank-deficient systems yield
    /// a valid SPD descent direction. Returns `false` if non-finite entries
    /// are encountered.
    pub fn factor(&mut self, matrix: &impl BlockOperator) -> bool {
        self.n = matrix.dimension();
        self.blocks = self.n / 3;
        self.had_regularization = false;
        if self.n == 0 {
            self.perm.clear();
            self.inv_perm.clear();
            self.first_col.clear();
            self.row_start.clear();
            self.lower.clear();
            self.diag.clear();
            self.scale.clear();
            return true;
        }
        debug_assert_eq!(self.n, 3 * self.blocks);

        // 1. Build symmetric block adjacency graph to compute RCM ordering and
        //    skyline lower envelope.
        let mut adj: Vec<Vec<usize>> = vec![Vec::new(); self.blocks];
        let mut off_pairs: Vec<usize> = Vec::new();
        for r in 0..self.blocks {
            off_pairs.clear();
            matrix.for_each_row_block(r, &mut |c, mat| {
                if c != r && mat != Mat3::ZERO {
                    off_pairs.push(c);
                }
            });
            for &c in &off_pairs {
                adj[r].push(c);
                adj[c].push(r);
            }
        }
        for row in &mut adj {
            row.sort_unstable();
            row.dedup();
        }

        self.perm.clear();
        self.perm.reserve(self.blocks);
        let mut visited = vec![false; self.blocks];
        let mut queue = VecDeque::new();
        let mut comp = Vec::new();
        for start in 0..self.blocks {
            if visited[start] {
                continue;
            }
            // Pick lowest-degree node in this connected component as BFS seed.
            let mut seed = start;
            let mut best_deg = adj[start].len();
            comp.clear();
            visited[start] = true;
            queue.push_back(start);
            while let Some(u) = queue.pop_front() {
                comp.push(u);
                if adj[u].len() < best_deg {
                    best_deg = adj[u].len();
                    seed = u;
                }
                for &v in &adj[u] {
                    if !visited[v] {
                        visited[v] = true;
                        queue.push_back(v);
                    }
                }
            }
            for &u in &comp {
                visited[u] = false;
            }
            // Cuthill-McKee BFS from `seed` with ascending-degree neighbor order,
            // then reverse per component (RCM).
            let comp_start = self.perm.len();
            visited[seed] = true;
            queue.push_back(seed);
            let mut neighbors = Vec::new();
            while let Some(u) = queue.pop_front() {
                self.perm.push(u);
                neighbors.clear();
                for &v in &adj[u] {
                    if !visited[v] {
                        visited[v] = true;
                        neighbors.push(v);
                    }
                }
                neighbors.sort_unstable_by_key(|&v| (adj[v].len(), v));
                for &v in &neighbors {
                    queue.push_back(v);
                }
            }
            self.perm[comp_start..].reverse();
        }

        self.inv_perm.resize(self.blocks, 0);
        for (new_b, &old_b) in self.perm.iter().enumerate() {
            self.inv_perm[old_b] = new_b;
        }

        // 2. Compute scalar envelope `first_col` and CSR-style `row_start`.
        let mut block_first: Vec<usize> = (0..self.blocks).collect();
        for (&nr, row_adj) in self.inv_perm.iter().zip(&adj) {
            for &old_c in row_adj {
                let nc = self.inv_perm[old_c];
                if nc < nr {
                    block_first[nr] = block_first[nr].min(nc);
                }
            }
        }
        self.first_col.resize(self.n, 0);
        self.row_start.resize(self.n + 1, 0);
        self.row_start[0] = 0;
        for (nr, &bf) in block_first.iter().enumerate() {
            let fc = 3 * bf;
            for dr in 0..3 {
                let i = 3 * nr + dr;
                self.first_col[i] = fc;
                self.row_start[i + 1] = self.row_start[i] + (i - fc);
            }
        }

        let nnz_lower = self.row_start[self.n];
        self.lower.clear();
        self.lower.resize(nnz_lower, 0.0);
        self.diag.clear();
        self.diag.resize(self.n, 0.0);

        // 3. Scatter matrix entries into the permuted skyline lower triangle.
        for old_r in 0..self.blocks {
            let nr = self.inv_perm[old_r];
            matrix.for_each_row_block(old_r, &mut |old_c, mat| {
                if mat == Mat3::ZERO {
                    return;
                }
                let nc = self.inv_perm[old_c];
                if nc == nr {
                    let sym = (mat + mat.transpose()) * 0.5;
                    for dr in 0..3 {
                        let i = 3 * nr + dr;
                        self.diag[i] += sym[(dr, dr)];
                        let r0 = self.row_start[i];
                        let fi = self.first_col[i];
                        for dc in 0..dr {
                            let j = 3 * nr + dc;
                            self.lower[r0 + (j - fi)] += sym[(dr, dc)];
                        }
                    }
                } else if nc < nr {
                    for dr in 0..3 {
                        let i = 3 * nr + dr;
                        let r0 = self.row_start[i];
                        let fi = self.first_col[i];
                        for dc in 0..3 {
                            let j = 3 * nc + dc;
                            self.lower[r0 + (j - fi)] += mat[(dr, dc)];
                        }
                    }
                }
            });
        }

        if self.diag.iter().any(|v| !v.is_finite()) || self.lower.iter().any(|v| !v.is_finite()) {
            return false;
        }
        // Symmetric diagonal equilibration: A_hat = S A S with unit-magnitude diagonal.
        self.scale.resize(self.n, 1.0);
        for (i, &d) in self.diag.iter().enumerate() {
            self.scale[i] = if d > 0.0 { 1.0 / d.sqrt() } else { 1.0 };
        }
        for i in 0..self.n {
            let si = self.scale[i];
            self.diag[i] *= si * si;
            let fi = self.first_col[i];
            let r0 = self.row_start[i];
            for j in fi..i {
                self.lower[r0 + (j - fi)] *= si * self.scale[j];
            }
        }
        let min_pivot = 1e-12;

        // 4. Skyline Crout LDLᵀ factorization in place.
        for i in 0..self.n {
            let fi = self.first_col[i];
            let r0 = self.row_start[i];
            for j in fi..i {
                let fj = self.first_col[j];
                let k0 = fi.max(fj);
                let mut sum = self.lower[r0 + (j - fi)];
                let si = r0 + (k0 - fi);
                let sj = self.row_start[j] + (k0 - fj);
                let len = j - k0;
                let (head, _tail) = self.lower.split_at(r0);
                let row_j = &head[sj..sj + len];
                // Split borrow: `_tail` starts at `r0`, and `si >= r0`.
                let row_i = &_tail[(si - r0)..(si - r0) + len];
                for (&wik, &ljk) in row_i.iter().zip(row_j) {
                    sum -= wik * ljk;
                }
                self.lower[r0 + (j - fi)] = sum;
            }
            let mut di = self.diag[i];
            for j in fi..i {
                let idx = r0 + (j - fi);
                let wd = self.lower[idx];
                let l_ij = wd / self.diag[j];
                self.lower[idx] = l_ij;
                di -= l_ij * wd;
            }
            if !di.is_finite() {
                return false;
            }
            if di <= min_pivot {
                self.had_regularization = true;
                self.diag[i] = min_pivot;
            } else {
                self.diag[i] = di;
            }
        }
        true
    }

    /// Whether positive pivot clamping was triggered during the last [`Self::factor`].
    pub fn was_regularized(&self) -> bool {
        self.had_regularization
    }

    /// Solves `A out = rhs` using the factored `L D Lᵀ` decomposition.
    pub fn solve(&self, rhs: &[f64], out: &mut [f64]) -> bool {
        debug_assert_eq!(rhs.len(), self.n);
        debug_assert_eq!(out.len(), self.n);
        if self.n == 0 {
            return true;
        }
        let mut z = vec![0.0; self.n];
        for nr in 0..self.blocks {
            Vec3::read(rhs, self.perm[nr]).write(&mut z, nr);
        }
        for (zi, &si) in z.iter_mut().zip(&self.scale) {
            *zi *= si;
        }
        for i in 0..self.n {
            let fi = self.first_col[i];
            let row = &self.lower[self.row_start[i]..self.row_start[i + 1]];
            let mut sum = z[i];
            for (&l_ij, &y_j) in row.iter().zip(&z[fi..i]) {
                sum -= l_ij * y_j;
            }
            z[i] = sum;
        }
        for (v, &d) in z.iter_mut().zip(&self.diag) {
            *v /= d;
        }
        for i in (0..self.n).rev() {
            let xi = z[i];
            let fi = self.first_col[i];
            let row = &self.lower[self.row_start[i]..self.row_start[i + 1]];
            for (&l_ij, dst) in row.iter().zip(&mut z[fi..i]) {
                *dst -= l_ij * xi;
            }
        }
        for (zi, &si) in z.iter_mut().zip(&self.scale) {
            *zi *= si;
        }
        for nr in 0..self.blocks {
            Vec3::read(&z, nr).write(out, self.perm[nr]);
        }
        out.iter().all(|v| v.is_finite())
    }
}
