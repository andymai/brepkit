//! 2D constraint solver for sketch-mode parametric design.
//!
//! Implements a basic geometric constraint system that solves for point
//! positions satisfying distance, angle, coincidence, and alignment
//! constraints. Uses Newton-Raphson iteration on the constraint residuals.
//!
//! This is the foundation for parametric sketch mode, similar to
//! `FreeCAD`'s [`PlaneGCS`] or [`SolveSpace`].

use brepkit_math::MathError;

/// A 2D point variable in the constraint system.
#[derive(Debug, Clone, Copy)]
pub struct SketchPoint {
    /// X coordinate.
    pub x: f64,
    /// Y coordinate.
    pub y: f64,
    /// Whether this point is fixed (not adjusted by the solver).
    pub fixed: bool,
}

impl SketchPoint {
    /// Creates a new free (movable) point.
    #[must_use]
    pub const fn new(x: f64, y: f64) -> Self {
        Self { x, y, fixed: false }
    }

    /// Creates a new fixed point.
    #[must_use]
    pub const fn fixed(x: f64, y: f64) -> Self {
        Self { x, y, fixed: true }
    }
}

/// Index into the sketch's point array.
pub type PointIdx = usize;

/// A geometric constraint in the sketch.
#[derive(Debug, Clone)]
pub enum Constraint {
    /// Two points must be at the same location.
    Coincident(PointIdx, PointIdx),
    /// Distance between two points must equal the given value.
    Distance(PointIdx, PointIdx, f64),
    /// A point's X coordinate must equal the given value.
    FixX(PointIdx, f64),
    /// A point's Y coordinate must equal the given value.
    FixY(PointIdx, f64),
    /// Two points must have the same X coordinate (vertical alignment).
    Vertical(PointIdx, PointIdx),
    /// Two points must have the same Y coordinate (horizontal alignment).
    Horizontal(PointIdx, PointIdx),
    /// The angle of the line from p1 to p2 must equal the given value (radians).
    Angle(PointIdx, PointIdx, f64),
    /// The line p1-p2 must be perpendicular to line p3-p4.
    Perpendicular(PointIdx, PointIdx, PointIdx, PointIdx),
    /// The line p1-p2 must be parallel to line p3-p4.
    Parallel(PointIdx, PointIdx, PointIdx, PointIdx),
}

/// A 2D constraint sketch that can be solved.
#[derive(Debug, Default)]
pub struct Sketch {
    /// The points in the sketch.
    pub points: Vec<SketchPoint>,
    /// The constraints to satisfy.
    pub constraints: Vec<Constraint>,
}

/// Result of solving the constraint system.
#[derive(Debug)]
pub struct SolveResult {
    /// Whether the solver converged.
    pub converged: bool,
    /// Number of iterations used.
    pub iterations: usize,
    /// Maximum constraint residual after solving.
    pub max_residual: f64,
}

impl Sketch {
    /// Creates a new empty sketch.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Adds a point and returns its index.
    pub fn add_point(&mut self, point: SketchPoint) -> PointIdx {
        let idx = self.points.len();
        self.points.push(point);
        idx
    }

    /// Adds a constraint.
    pub fn add_constraint(&mut self, constraint: Constraint) {
        self.constraints.push(constraint);
    }

    /// Solve the constraint system using Newton-Raphson iteration.
    ///
    /// Modifies point positions in-place to satisfy constraints.
    ///
    /// # Errors
    /// Returns `MathError::ConvergenceFailure` if the solver doesn't converge.
    pub fn solve(
        &mut self,
        max_iterations: usize,
        tolerance: f64,
    ) -> Result<SolveResult, MathError> {
        // Build variable vector (only free point coordinates)
        let var_map = self.build_variable_map();
        let num_vars = var_map.len();

        if num_vars == 0 {
            // All points fixed — just check constraints
            let residual = self.max_residual();
            return Ok(SolveResult {
                converged: residual < tolerance,
                iterations: 0,
                max_residual: residual,
            });
        }

        let mut vars = self.extract_variables(&var_map);

        for iteration in 0..max_iterations {
            self.apply_variables(&var_map, &vars);

            let residuals = self.compute_residuals();
            let max_res = residuals.iter().fold(0.0_f64, |a, &b| a.max(b.abs()));

            if max_res < tolerance {
                return Ok(SolveResult {
                    converged: true,
                    iterations: iteration,
                    max_residual: max_res,
                });
            }

            // Compute Jacobian
            let jacobian = self.compute_jacobian(&var_map, &vars);

            // Solve J * delta = -residuals using least-squares (J^T J delta = -J^T r)
            let delta = solve_least_squares(&jacobian, &residuals, num_vars);

            // Update variables
            for (i, d) in delta.iter().enumerate() {
                vars[i] += d;
            }
        }

        self.apply_variables(&var_map, &vars);
        let max_res = self.max_residual();

        if max_res < tolerance {
            Ok(SolveResult {
                converged: true,
                iterations: max_iterations,
                max_residual: max_res,
            })
        } else {
            Err(MathError::ConvergenceFailure {
                iterations: max_iterations,
            })
        }
    }

    /// Build a mapping from variable index to (`point_idx`, `is_y`).
    fn build_variable_map(&self) -> Vec<(usize, bool)> {
        let mut map = Vec::new();
        for (i, p) in self.points.iter().enumerate() {
            if !p.fixed {
                map.push((i, false)); // x
                map.push((i, true)); // y
            }
        }
        map
    }

    /// Extract current variable values.
    fn extract_variables(&self, var_map: &[(usize, bool)]) -> Vec<f64> {
        var_map
            .iter()
            .map(|&(pi, is_y)| {
                if is_y {
                    self.points[pi].y
                } else {
                    self.points[pi].x
                }
            })
            .collect()
    }

    /// Apply variable values back to points.
    fn apply_variables(&mut self, var_map: &[(usize, bool)], vars: &[f64]) {
        for (i, &(pi, is_y)) in var_map.iter().enumerate() {
            if is_y {
                self.points[pi].y = vars[i];
            } else {
                self.points[pi].x = vars[i];
            }
        }
    }

    /// Compute constraint residuals.
    fn compute_residuals(&self) -> Vec<f64> {
        self.constraints
            .iter()
            .map(|c| self.constraint_residual(c))
            .collect()
    }

    /// Maximum absolute residual.
    fn max_residual(&self) -> f64 {
        self.compute_residuals()
            .iter()
            .fold(0.0_f64, |a, &b| a.max(b.abs()))
    }

    /// Compute the residual of a single constraint.
    fn constraint_residual(&self, c: &Constraint) -> f64 {
        match c {
            Constraint::Coincident(a, b) => {
                let pa = &self.points[*a];
                let pb = &self.points[*b];
                let dx = pa.x - pb.x;
                let dy = pa.y - pb.y;
                dx.hypot(dy)
            }
            Constraint::Distance(a, b, d) => {
                let pa = &self.points[*a];
                let pb = &self.points[*b];
                let dx = pa.x - pb.x;
                let dy = pa.y - pb.y;
                dx.hypot(dy) - d
            }
            Constraint::FixX(a, val) => self.points[*a].x - val,
            Constraint::FixY(a, val) => self.points[*a].y - val,
            Constraint::Vertical(a, b) => self.points[*a].x - self.points[*b].x,
            Constraint::Horizontal(a, b) => self.points[*a].y - self.points[*b].y,
            Constraint::Angle(a, b, angle) => {
                let pa = &self.points[*a];
                let pb = &self.points[*b];
                let actual = (pb.y - pa.y).atan2(pb.x - pa.x);
                // Normalize angle difference to [-π, π]
                let diff = actual - angle;
                diff.sin() // Use sin for smooth zero-crossing
            }
            Constraint::Perpendicular(a, b, c, d) => {
                let dx1 = self.points[*b].x - self.points[*a].x;
                let dy1 = self.points[*b].y - self.points[*a].y;
                let dx2 = self.points[*d].x - self.points[*c].x;
                let dy2 = self.points[*d].y - self.points[*c].y;
                dx1.mul_add(dx2, dy1 * dy2) // dot product = 0 for perpendicular
            }
            Constraint::Parallel(a, b, c, d) => {
                let dx1 = self.points[*b].x - self.points[*a].x;
                let dy1 = self.points[*b].y - self.points[*a].y;
                let dx2 = self.points[*d].x - self.points[*c].x;
                let dy2 = self.points[*d].y - self.points[*c].y;
                dx1.mul_add(dy2, -(dy1 * dx2)) // cross product = 0 for parallel
            }
        }
    }

    /// Compute the Jacobian matrix using finite differences.
    fn compute_jacobian(&mut self, var_map: &[(usize, bool)], vars: &[f64]) -> Vec<Vec<f64>> {
        let num_constraints = self.constraints.len();
        let num_vars = vars.len();
        let eps = 1e-8;

        let mut jacobian = vec![vec![0.0; num_vars]; num_constraints];

        let base_residuals = self.compute_residuals();

        for j in 0..num_vars {
            let (pi, is_y) = var_map[j];
            let original = if is_y {
                self.points[pi].y
            } else {
                self.points[pi].x
            };

            // Perturb variable
            if is_y {
                self.points[pi].y = original + eps;
            } else {
                self.points[pi].x = original + eps;
            }

            let perturbed = self.compute_residuals();

            for i in 0..num_constraints {
                jacobian[i][j] = (perturbed[i] - base_residuals[i]) / eps;
            }

            // Restore
            if is_y {
                self.points[pi].y = original;
            } else {
                self.points[pi].x = original;
            }
        }

        jacobian
    }
}

/// Solve a least-squares system J * delta = -residuals.
///
/// Uses the normal equations: (J^T J) delta = -J^T residuals.
/// For small systems this is efficient; for larger systems, QR would be better.
fn solve_least_squares(jacobian: &[Vec<f64>], residuals: &[f64], num_vars: usize) -> Vec<f64> {
    let num_constraints = residuals.len();

    // Compute J^T J
    let mut jtj = vec![vec![0.0; num_vars]; num_vars];
    let mut jtr = vec![0.0; num_vars];

    for i in 0..num_constraints {
        for j in 0..num_vars {
            jtr[j] -= jacobian[i][j] * residuals[i];
            for k in 0..num_vars {
                jtj[j][k] += jacobian[i][j] * jacobian[i][k];
            }
        }
    }

    // Add small regularization for numerical stability
    for (i, row) in jtj.iter_mut().enumerate() {
        row[i] += 1e-12;
    }

    // Solve via Gaussian elimination
    gauss_solve(&mut jtj, &mut jtr)
}

/// Solve a linear system Ax = b via Gaussian elimination with partial pivoting.
#[allow(clippy::needless_range_loop)]
fn gauss_solve(a: &mut [Vec<f64>], b: &mut [f64]) -> Vec<f64> {
    let n = b.len();

    // Forward elimination with partial pivoting
    for col in 0..n {
        // Find pivot
        let mut max_val = a[col][col].abs();
        let mut max_row = col;
        for row in (col + 1)..n {
            if a[row][col].abs() > max_val {
                max_val = a[row][col].abs();
                max_row = row;
            }
        }

        // Swap rows
        if max_row != col {
            a.swap(col, max_row);
            b.swap(col, max_row);
        }

        let pivot = a[col][col];
        if pivot.abs() < 1e-15 {
            continue; // Skip singular column
        }

        for row in (col + 1)..n {
            let factor = a[row][col] / pivot;
            for k in col..n {
                a[row][k] -= factor * a[col][k];
            }
            b[row] -= factor * b[col];
        }
    }

    // Back substitution
    let mut x = vec![0.0; n];
    for col in (0..n).rev() {
        if a[col][col].abs() < 1e-15 {
            continue;
        }
        let mut sum = b[col];
        for k in (col + 1)..n {
            sum -= a[col][k] * x[k];
        }
        x[col] = sum / a[col][col];
    }

    x
}

#[cfg(test)]
#[allow(clippy::unwrap_used)]
mod tests {
    use super::*;

    const TOL: f64 = 1e-6;

    #[test]
    fn fixed_points_no_solve() {
        let mut sketch = Sketch::new();
        sketch.add_point(SketchPoint::fixed(0.0, 0.0));
        sketch.add_point(SketchPoint::fixed(1.0, 0.0));
        sketch.add_constraint(Constraint::Distance(0, 1, 1.0));

        let result = sketch.solve(100, TOL).unwrap();
        assert!(result.converged);
        assert_eq!(result.iterations, 0);
    }

    #[test]
    fn distance_constraint() {
        let mut sketch = Sketch::new();
        let p0 = sketch.add_point(SketchPoint::fixed(0.0, 0.0));
        let p1 = sketch.add_point(SketchPoint::new(0.5, 0.0)); // start close

        sketch.add_constraint(Constraint::Distance(p0, p1, 3.0));

        let result = sketch.solve(100, TOL).unwrap();
        assert!(result.converged);

        let dx = sketch.points[p1].x - sketch.points[p0].x;
        let dy = sketch.points[p1].y - sketch.points[p0].y;
        let dist = dx.hypot(dy);
        assert!(
            (dist - 3.0).abs() < TOL,
            "distance should be 3.0, got {dist}"
        );
    }

    #[test]
    fn coincident_constraint() {
        let mut sketch = Sketch::new();
        let p0 = sketch.add_point(SketchPoint::fixed(1.0, 2.0));
        let p1 = sketch.add_point(SketchPoint::new(3.0, 4.0));

        sketch.add_constraint(Constraint::Coincident(p0, p1));

        let result = sketch.solve(100, TOL).unwrap();
        assert!(result.converged);

        assert!((sketch.points[p1].x - 1.0).abs() < TOL);
        assert!((sketch.points[p1].y - 2.0).abs() < TOL);
    }

    #[test]
    fn horizontal_constraint() {
        let mut sketch = Sketch::new();
        let p0 = sketch.add_point(SketchPoint::fixed(0.0, 1.0));
        let p1 = sketch.add_point(SketchPoint::new(5.0, 3.0));

        sketch.add_constraint(Constraint::Horizontal(p0, p1));

        let result = sketch.solve(100, TOL).unwrap();
        assert!(result.converged);

        assert!(
            (sketch.points[p1].y - sketch.points[p0].y).abs() < TOL,
            "points should have same Y"
        );
    }

    #[test]
    fn vertical_constraint() {
        let mut sketch = Sketch::new();
        let p0 = sketch.add_point(SketchPoint::fixed(2.0, 0.0));
        let p1 = sketch.add_point(SketchPoint::new(5.0, 7.0));

        sketch.add_constraint(Constraint::Vertical(p0, p1));

        let result = sketch.solve(100, TOL).unwrap();
        assert!(result.converged);

        assert!(
            (sketch.points[p1].x - sketch.points[p0].x).abs() < TOL,
            "points should have same X"
        );
    }

    #[test]
    fn perpendicular_constraint() {
        let mut sketch = Sketch::new();
        let p0 = sketch.add_point(SketchPoint::fixed(0.0, 0.0));
        let p1 = sketch.add_point(SketchPoint::fixed(1.0, 0.0));
        let p2 = sketch.add_point(SketchPoint::fixed(0.0, 0.0));
        let p3 = sketch.add_point(SketchPoint::new(0.5, 0.5)); // should move to (0, y)

        sketch.add_constraint(Constraint::Perpendicular(p0, p1, p2, p3));

        let result = sketch.solve(100, TOL).unwrap();
        assert!(result.converged);

        // Line p0-p1 is along X. Perpendicular line p2-p3 should be along Y.
        let dx2 = sketch.points[p3].x - sketch.points[p2].x;
        let dot = dx2; // dot product with (1,0) direction
        assert!(dot.abs() < TOL, "lines should be perpendicular, dot={dot}");
    }

    #[test]
    fn parallel_constraint() {
        let mut sketch = Sketch::new();
        let p0 = sketch.add_point(SketchPoint::fixed(0.0, 0.0));
        let p1 = sketch.add_point(SketchPoint::fixed(1.0, 1.0));
        let p2 = sketch.add_point(SketchPoint::fixed(2.0, 0.0));
        let p3 = sketch.add_point(SketchPoint::new(3.0, 0.5)); // should adjust

        sketch.add_constraint(Constraint::Parallel(p0, p1, p2, p3));

        let result = sketch.solve(100, TOL).unwrap();
        assert!(result.converged);

        // Line p0-p1 has direction (1,1). Line p2-p3 should be parallel.
        let dx2 = sketch.points[p3].x - sketch.points[p2].x;
        let dy2 = sketch.points[p3].y - sketch.points[p2].y;
        let cross = dy2 - dx2; // cross product with (1,1)
        assert!(cross.abs() < TOL, "lines should be parallel, cross={cross}");
    }

    #[test]
    fn fix_xy_constraint() {
        let mut sketch = Sketch::new();
        let p = sketch.add_point(SketchPoint::new(5.0, 7.0));

        sketch.add_constraint(Constraint::FixX(p, 2.0));
        sketch.add_constraint(Constraint::FixY(p, 3.0));

        let result = sketch.solve(100, TOL).unwrap();
        assert!(result.converged);

        assert!((sketch.points[p].x - 2.0).abs() < TOL);
        assert!((sketch.points[p].y - 3.0).abs() < TOL);
    }

    #[test]
    fn multi_constraint_triangle() {
        let mut sketch = Sketch::new();
        let p0 = sketch.add_point(SketchPoint::fixed(0.0, 0.0));
        let p1 = sketch.add_point(SketchPoint::new(1.0, 0.0));
        let p2 = sketch.add_point(SketchPoint::new(0.5, 1.0));

        // Fix p1 on x-axis
        sketch.add_constraint(Constraint::Horizontal(p0, p1));
        // Distance p0-p1 = 3
        sketch.add_constraint(Constraint::Distance(p0, p1, 3.0));
        // Distance p0-p2 = 4
        sketch.add_constraint(Constraint::Distance(p0, p2, 4.0));
        // Distance p1-p2 = 5
        sketch.add_constraint(Constraint::Distance(p1, p2, 5.0));

        let result = sketch.solve(200, TOL).unwrap();
        assert!(result.converged, "triangle constraints should converge");

        // Verify distances
        let d01 = (sketch.points[p1].x - sketch.points[p0].x)
            .hypot(sketch.points[p1].y - sketch.points[p0].y);
        assert!((d01 - 3.0).abs() < TOL, "d01 should be 3, got {d01}");
    }
}
