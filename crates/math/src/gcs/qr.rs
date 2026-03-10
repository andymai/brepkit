//! Householder QR factorization with column pivoting.
//!
//! Rank-revealing factorization used by the DogLeg solver for least-squares
//! and by DOF analysis for rank detection. Column pivoting selects the
//! column with the largest remaining norm at each step, ensuring the
//! diagonal of R is non-increasing in magnitude.

/// Result of a QR factorization with column pivoting.
///
/// Stores the factored matrix in compact form: Householder vectors below
/// the diagonal, R on and above the diagonal.
pub struct QrResult {
    /// Row-major storage: Householder vectors below diagonal, R on/above.
    data: Vec<f64>,
    /// Householder scaling factors (one per column processed).
    tau: Vec<f64>,
    /// Column permutation: `perm[k]` is the original column index of
    /// the k-th pivot column.
    perm: Vec<usize>,
    /// Number of rows.
    m: usize,
    /// Number of columns.
    n: usize,
}

impl QrResult {
    /// Perform QR factorization with column pivoting on an m×n matrix.
    ///
    /// `data` is row-major, length `m * n`. It is modified in-place.
    #[allow(clippy::too_many_lines)]
    pub fn factorize(data: &mut [f64], m: usize, n: usize) -> Self {
        assert!(
            data.len() >= m * n,
            "data length {} < m*n = {}",
            data.len(),
            m * n
        );

        let k = m.min(n);
        let mut tau = vec![0.0; k];
        let mut perm: Vec<usize> = (0..n).collect();

        // Column norms for pivoting
        let mut col_norms = vec![0.0; n];
        for j in 0..n {
            let mut s = 0.0;
            for i in 0..m {
                let v = data[i * n + j];
                s += v * v;
            }
            col_norms[j] = s;
        }

        for step in 0..k {
            // Column pivoting: find column with largest remaining norm
            let mut best_col = step;
            let mut best_norm = col_norms[step];
            for j in (step + 1)..n {
                if col_norms[j] > best_norm {
                    best_norm = col_norms[j];
                    best_col = j;
                }
            }

            // Swap columns if needed
            if best_col != step {
                for i in 0..m {
                    data.swap(i * n + step, i * n + best_col);
                }
                col_norms.swap(step, best_col);
                perm.swap(step, best_col);
            }

            // Compute Householder reflector for column `step`, rows step..m
            let mut norm_sq = 0.0;
            for i in step..m {
                let v = data[i * n + step];
                norm_sq += v * v;
            }

            if norm_sq < 1e-300 {
                tau[step] = 0.0;
                continue;
            }

            let norm = norm_sq.sqrt();
            let alpha = data[step * n + step];
            let beta = if alpha >= 0.0 { -norm } else { norm };
            tau[step] = (beta - alpha) / beta;
            let scale = 1.0 / (alpha - beta);

            for i in (step + 1)..m {
                data[i * n + step] *= scale;
            }
            data[step * n + step] = beta;

            // Apply reflector to remaining columns
            for j in (step + 1)..n {
                let mut dot = data[step * n + j];
                for i in (step + 1)..m {
                    dot += data[i * n + step] * data[i * n + j];
                }
                let t = tau[step] * dot;
                data[step * n + j] -= t;
                for i in (step + 1)..m {
                    data[i * n + j] -= data[i * n + step] * t;
                }
            }

            // Update remaining column norms (downdate)
            for j in (step + 1)..n {
                let v = data[step * n + j];
                col_norms[j] -= v * v;
                // Clamp to avoid negative due to rounding
                if col_norms[j] < 0.0 {
                    col_norms[j] = 0.0;
                }
            }
        }

        Self {
            data: data.to_vec(),
            tau,
            perm,
            m,
            n,
        }
    }

    /// Numerical rank, counting diagonal elements of R with
    /// `|R[i,i]| > tol * |R[0,0]|`.
    #[must_use]
    pub fn rank(&self, tol: f64) -> usize {
        let k = self.m.min(self.n);
        if k == 0 {
            return 0;
        }
        let r00 = self.data[0].abs();
        if r00 < 1e-300 {
            return 0;
        }
        let threshold = tol * r00;
        let mut rank = 0;
        for i in 0..k {
            if self.data[i * self.n + i].abs() > threshold {
                rank += 1;
            } else {
                break;
            }
        }
        rank
    }

    /// Compute Q^T * b.
    pub fn qt_mul(&self, b: &[f64]) -> Vec<f64> {
        let mut result = b.to_vec();
        let k = self.m.min(self.n);
        for step in 0..k {
            if self.tau[step].abs() < 1e-300 {
                continue;
            }
            let mut dot = result[step];
            for i in (step + 1)..self.m {
                dot += self.data[i * self.n + step] * result[i];
            }
            let t = self.tau[step] * dot;
            result[step] -= t;
            for i in (step + 1)..self.m {
                result[i] -= self.data[i * self.n + step] * t;
            }
        }
        result
    }

    /// Solve the least-squares problem min ||Jx - b|| via back-substitution
    /// on the R factor, then unpermute.
    pub fn solve_least_squares(&self, b: &[f64]) -> Vec<f64> {
        let qtb = self.qt_mul(b);
        let k = self.m.min(self.n);

        // Back-substitute R * z = qtb[0..k]
        let mut z = vec![0.0; self.n];
        for i in (0..k).rev() {
            let rii = self.data[i * self.n + i];
            if rii.abs() < 1e-300 {
                continue;
            }
            let mut s = qtb[i];
            for j in (i + 1)..k.min(self.n) {
                s -= self.data[i * self.n + j] * z[j];
            }
            z[i] = s / rii;
        }

        // Unpermute
        let mut x = vec![0.0; self.n];
        for (i, &pi) in self.perm.iter().enumerate() {
            x[pi] = z[i];
        }
        x
    }
}

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::expect_used)]
mod tests {
    use super::*;

    #[test]
    fn identity_3x3() {
        let mut data = vec![1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0];
        let qr = QrResult::factorize(&mut data, 3, 3);
        assert_eq!(qr.rank(1e-10), 3);
    }

    #[test]
    fn known_3x3() {
        // A = [[1, 2, 3], [4, 5, 6], [7, 8, 10]]
        let mut data = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 10.0];
        let qr = QrResult::factorize(&mut data, 3, 3);
        assert_eq!(qr.rank(1e-10), 3);

        // Solve Ax = [1, 1, 1]
        let x = qr.solve_least_squares(&[1.0, 1.0, 1.0]);
        // Verify Ax ≈ [1, 1, 1]
        let a = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 10.0];
        for i in 0..3 {
            let row_sum = a[i * 3] * x[0] + a[i * 3 + 1] * x[1] + a[i * 3 + 2] * x[2];
            assert!((row_sum - 1.0).abs() < 1e-10, "row {i}: {row_sum} != 1.0");
        }
    }

    #[test]
    fn rank_deficient() {
        // Rows 2 = row 0 + row 1, so rank = 2
        let mut data = vec![1.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0, 1.0];
        let qr = QrResult::factorize(&mut data, 2, 4);
        assert_eq!(qr.rank(1e-10), 2);
    }

    #[test]
    fn overdetermined_least_squares() {
        // 3 equations, 2 unknowns: x + y = 1, x - y = 0, x = 0.5
        // Solution: x = 0.5, y = 0.5
        let mut data = vec![1.0, 1.0, 1.0, -1.0, 1.0, 0.0];
        let qr = QrResult::factorize(&mut data, 3, 2);
        let x = qr.solve_least_squares(&[1.0, 0.0, 0.5]);
        assert!((x[0] - 0.5).abs() < 1e-10, "x[0] = {}", x[0]);
        assert!((x[1] - 0.5).abs() < 1e-10, "x[1] = {}", x[1]);
    }

    #[test]
    fn empty_matrix() {
        let mut data = vec![];
        let qr = QrResult::factorize(&mut data, 0, 0);
        assert_eq!(qr.rank(1e-10), 0);
    }

    #[test]
    fn single_element() {
        let mut data = vec![5.0];
        let qr = QrResult::factorize(&mut data, 1, 1);
        assert_eq!(qr.rank(1e-10), 1);
        let x = qr.solve_least_squares(&[10.0]);
        assert!((x[0] - 2.0).abs() < 1e-10);
    }
}
