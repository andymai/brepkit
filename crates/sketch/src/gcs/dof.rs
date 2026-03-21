//! Degrees-of-freedom analysis via QR rank detection.

use super::qr::QrResult;

/// Result of DOF analysis on the constraint system.
#[derive(Debug, Clone, Copy)]
pub struct DofAnalysis {
    /// Degrees of freedom remaining (under-constrained dimensions).
    pub dof: usize,
    /// Rank of the Jacobian matrix.
    pub rank: usize,
    /// Total number of solver parameters.
    pub num_params: usize,
    /// Total number of constraint equations.
    pub num_equations: usize,
}

/// Analyze degrees of freedom from a Jacobian matrix.
///
/// DOF = `num_params - rank(J)`. A fully constrained system has DOF = 0.
/// Over-constrained systems have `num_equations > num_params` with DOF = 0
/// (or negative if constraints are contradictory, though we report 0).
pub fn analyze(jacobian: &[f64], m: usize, n: usize) -> DofAnalysis {
    let rank = if m == 0 || n == 0 {
        0
    } else {
        let mut data = jacobian.to_vec();
        let qr = QrResult::factorize(&mut data, m, n);
        qr.rank(1e-10)
    };
    DofAnalysis {
        dof: n.saturating_sub(rank),
        rank,
        num_params: n,
        num_equations: m,
    }
}

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::expect_used)]
mod tests {
    use super::*;

    #[test]
    fn fully_constrained() {
        // 2 equations, 2 unknowns, full rank
        let j = vec![1.0, 0.0, 0.0, 1.0];
        let result = analyze(&j, 2, 2);
        assert_eq!(result.dof, 0);
        assert_eq!(result.rank, 2);
    }

    #[test]
    fn under_constrained() {
        // 1 equation, 2 unknowns → DOF = 1
        let j = vec![1.0, 1.0];
        let result = analyze(&j, 1, 2);
        assert_eq!(result.dof, 1);
        assert_eq!(result.rank, 1);
    }

    #[test]
    fn over_constrained_redundant() {
        // 3 equations, 2 unknowns, but row 3 = row 1 → rank 2
        let j = vec![1.0, 0.0, 0.0, 1.0, 1.0, 0.0];
        let result = analyze(&j, 3, 2);
        assert_eq!(result.dof, 0);
        assert_eq!(result.rank, 2);
    }

    #[test]
    fn empty_system() {
        let result = analyze(&[], 0, 0);
        assert_eq!(result.dof, 0);
        assert_eq!(result.rank, 0);
    }

    #[test]
    fn free_point_has_two_dof() {
        // No constraints, 2 params → DOF = 2
        let result = analyze(&[], 0, 2);
        assert_eq!(result.dof, 2);
    }
}
