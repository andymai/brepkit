//! 2D constraint solver for sketch-mode parametric design.
//!
//! Implements a geometric constraint system that solves for point
//! positions satisfying distance, angle, coincidence, alignment,
//! and curve-intersection constraints. Uses Newton-Raphson iteration
//! on the constraint residuals with graph-based decomposition into
//! independent connected components for scalability.
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

/// A reference to a 2D curve for intersection constraints.
#[derive(Debug, Clone)]
pub enum CurveRef {
    /// Line through two points: signed distance = 0.
    Line(PointIdx, PointIdx),
    /// Circle centered at a point with given radius.
    Circle(PointIdx, f64),
}

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
    /// The point must lie at the intersection of two 2D curves.
    /// Produces two residual equations (one per curve).
    CurveIntersection {
        /// The point constrained to lie on both curves.
        point: PointIdx,
        /// First curve.
        curve1: CurveRef,
        /// Second curve.
        curve2: CurveRef,
    },
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

/// A connected component of the constraint graph.
struct ConstraintComponent {
    /// Indices into `Sketch::constraints` for this component's constraints.
    constraint_indices: Vec<usize>,
    /// All point indices involved in this component.
    point_indices: Vec<usize>,
}

/// Extract all point indices referenced by a constraint.
fn constraint_points(c: &Constraint) -> Vec<PointIdx> {
    match c {
        Constraint::Coincident(a, b)
        | Constraint::Distance(a, b, _)
        | Constraint::Vertical(a, b)
        | Constraint::Horizontal(a, b)
        | Constraint::Angle(a, b, _) => vec![*a, *b],
        Constraint::FixX(a, _) | Constraint::FixY(a, _) => vec![*a],
        Constraint::Perpendicular(a, b, c, d) | Constraint::Parallel(a, b, c, d) => {
            vec![*a, *b, *c, *d]
        }
        Constraint::CurveIntersection {
            point,
            curve1,
            curve2,
        } => {
            let mut pts = vec![*point];
            collect_curve_ref_points(curve1, &mut pts);
            collect_curve_ref_points(curve2, &mut pts);
            pts
        }
    }
}

/// Collect point indices referenced by a `CurveRef`.
fn collect_curve_ref_points(cr: &CurveRef, pts: &mut Vec<PointIdx>) {
    match cr {
        CurveRef::Line(a, b) => {
            pts.push(*a);
            pts.push(*b);
        }
        CurveRef::Circle(c, _) => {
            pts.push(*c);
        }
    }
}

/// Number of residual equations produced by a constraint.
const fn constraint_residual_count(c: &Constraint) -> usize {
    match c {
        Constraint::CurveIntersection { .. } => 2,
        _ => 1,
    }
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
    /// Decomposes the constraint graph into independent connected components
    /// and solves each separately for improved scalability.
    ///
    /// Modifies point positions in-place to satisfy constraints.
    ///
    /// # Errors
    /// Returns `MathError::ConvergenceFailure` if the solver doesn't converge.
    #[allow(clippy::too_many_lines)]
    pub fn solve(
        &mut self,
        max_iterations: usize,
        tolerance: f64,
    ) -> Result<SolveResult, MathError> {
        let components = self.decompose_components();

        // If there's only one component (or zero), use it directly.
        // If multiple, solve each independently.
        let mut total_iterations = 0;
        let mut total_max_residual = 0.0_f64;
        let mut all_converged = true;

        for comp in &components {
            let result = self.solve_component(comp, max_iterations, tolerance)?;
            total_iterations = total_iterations.max(result.iterations);
            total_max_residual = total_max_residual.max(result.max_residual);
            if !result.converged {
                all_converged = false;
            }
        }

        // Handle case with no constraints
        if components.is_empty() {
            return Ok(SolveResult {
                converged: true,
                iterations: 0,
                max_residual: 0.0,
            });
        }

        Ok(SolveResult {
            converged: all_converged,
            iterations: total_iterations,
            max_residual: total_max_residual,
        })
    }

    /// Decompose the constraint system into independent connected components
    /// using union-find.
    fn decompose_components(&self) -> Vec<ConstraintComponent> {
        let n = self.points.len();
        if n == 0 {
            return Vec::new();
        }

        // Union-Find with path compression and union by rank
        let mut parent: Vec<usize> = (0..n).collect();
        let mut rank: Vec<usize> = vec![0; n];

        // For each constraint, union all points it references
        for constraint in &self.constraints {
            let pts = constraint_points(constraint);
            for i in 1..pts.len() {
                union(&mut parent, &mut rank, pts[0], pts[i]);
            }
        }

        // Group constraints by component root
        let mut component_map: std::collections::HashMap<usize, Vec<usize>> =
            std::collections::HashMap::new();

        for (ci, constraint) in self.constraints.iter().enumerate() {
            let pts = constraint_points(constraint);
            if let Some(&first_pt) = pts.first() {
                let root = find(&mut parent, first_pt);
                component_map.entry(root).or_default().push(ci);
            }
        }

        // Build components
        let mut components = Vec::new();
        for (_, constraint_indices) in component_map {
            // Collect unique point indices for this component
            let mut point_set = std::collections::HashSet::new();
            for &ci in &constraint_indices {
                let pts = constraint_points(&self.constraints[ci]);
                for p in pts {
                    point_set.insert(p);
                }
            }
            let mut point_indices: Vec<usize> = point_set.into_iter().collect();
            point_indices.sort_unstable();

            components.push(ConstraintComponent {
                constraint_indices,
                point_indices,
            });
        }

        components
    }

    /// Solve a single connected component of the constraint graph.
    fn solve_component(
        &mut self,
        comp: &ConstraintComponent,
        max_iterations: usize,
        tolerance: f64,
    ) -> Result<SolveResult, MathError> {
        // Build variable map: only free points in this component
        let var_map: Vec<(usize, bool)> = comp
            .point_indices
            .iter()
            .filter(|&&pi| !self.points[pi].fixed)
            .flat_map(|&pi| [(pi, false), (pi, true)])
            .collect();

        let num_vars = var_map.len();

        if num_vars == 0 {
            // All points in this component are fixed — just check residuals
            let residual = self.max_residual_for(&comp.constraint_indices);
            return Ok(SolveResult {
                converged: residual < tolerance,
                iterations: 0,
                max_residual: residual,
            });
        }

        let mut vars = self.extract_variables(&var_map);

        for iteration in 0..max_iterations {
            self.apply_variables(&var_map, &vars);

            let residuals = self.compute_residuals_for(&comp.constraint_indices);
            let max_res = residuals.iter().fold(0.0_f64, |a, &b| a.max(b.abs()));

            if max_res < tolerance {
                return Ok(SolveResult {
                    converged: true,
                    iterations: iteration,
                    max_residual: max_res,
                });
            }

            // Compute Jacobian
            let jacobian = self.compute_jacobian_for(&comp.constraint_indices, &var_map, &vars);

            // Solve J * delta = -residuals using least-squares (J^T J delta = -J^T r)
            let delta = solve_least_squares(&jacobian, &residuals, num_vars);

            // Update variables
            for (i, d) in delta.iter().enumerate() {
                vars[i] += d;
            }
        }

        self.apply_variables(&var_map, &vars);
        let max_res = self.max_residual_for(&comp.constraint_indices);

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

    /// Compute constraint residuals for a subset of constraints (by index).
    fn compute_residuals_for(&self, constraint_indices: &[usize]) -> Vec<f64> {
        constraint_indices
            .iter()
            .flat_map(|&ci| self.constraint_residuals(&self.constraints[ci]))
            .collect()
    }

    /// Maximum absolute residual for a subset of constraints.
    fn max_residual_for(&self, constraint_indices: &[usize]) -> f64 {
        self.compute_residuals_for(constraint_indices)
            .iter()
            .fold(0.0_f64, |a, &b| a.max(b.abs()))
    }

    /// Compute the residuals of a single constraint.
    ///
    /// Most constraints produce one residual; `CurveIntersection` produces two.
    fn constraint_residuals(&self, c: &Constraint) -> Vec<f64> {
        match c {
            Constraint::Coincident(a, b) => {
                let pa = &self.points[*a];
                let pb = &self.points[*b];
                let dx = pa.x - pb.x;
                let dy = pa.y - pb.y;
                vec![dx.hypot(dy)]
            }
            Constraint::Distance(a, b, d) => {
                let pa = &self.points[*a];
                let pb = &self.points[*b];
                let dx = pa.x - pb.x;
                let dy = pa.y - pb.y;
                vec![dx.hypot(dy) - d]
            }
            Constraint::FixX(a, val) => vec![self.points[*a].x - val],
            Constraint::FixY(a, val) => vec![self.points[*a].y - val],
            Constraint::Vertical(a, b) => vec![self.points[*a].x - self.points[*b].x],
            Constraint::Horizontal(a, b) => vec![self.points[*a].y - self.points[*b].y],
            Constraint::Angle(a, b, angle) => {
                let pa = &self.points[*a];
                let pb = &self.points[*b];
                let actual = (pb.y - pa.y).atan2(pb.x - pa.x);
                let diff = actual - angle;
                vec![diff.sin()]
            }
            Constraint::Perpendicular(a, b, c, d) => {
                let dx1 = self.points[*b].x - self.points[*a].x;
                let dy1 = self.points[*b].y - self.points[*a].y;
                let dx2 = self.points[*d].x - self.points[*c].x;
                let dy2 = self.points[*d].y - self.points[*c].y;
                vec![dx1.mul_add(dx2, dy1 * dy2)]
            }
            Constraint::Parallel(a, b, c, d) => {
                let dx1 = self.points[*b].x - self.points[*a].x;
                let dy1 = self.points[*b].y - self.points[*a].y;
                let dx2 = self.points[*d].x - self.points[*c].x;
                let dy2 = self.points[*d].y - self.points[*c].y;
                vec![dx1.mul_add(dy2, -(dy1 * dx2))]
            }
            Constraint::CurveIntersection {
                point,
                curve1,
                curve2,
            } => {
                let p = &self.points[*point];
                vec![
                    self.curve_ref_residual(curve1, p.x, p.y),
                    self.curve_ref_residual(curve2, p.x, p.y),
                ]
            }
        }
    }

    /// Evaluate the implicit curve function for a `CurveRef` at point (px, py).
    fn curve_ref_residual(&self, cr: &CurveRef, px: f64, py: f64) -> f64 {
        match cr {
            CurveRef::Line(a, b) => {
                let pa = &self.points[*a];
                let pb = &self.points[*b];
                // Signed distance (un-normalized) from p to line through a, b:
                // (b.y - a.y) * (p.x - a.x) - (b.x - a.x) * (p.y - a.y)
                (pb.y - pa.y).mul_add(px - pa.x, -(pb.x - pa.x) * (py - pa.y))
            }
            CurveRef::Circle(c, r) => {
                let pc = &self.points[*c];
                let dx = px - pc.x;
                let dy = py - pc.y;
                dx.mul_add(dx, dy * dy) - r * r
            }
        }
    }

    /// Compute the Jacobian matrix using finite differences for a subset of constraints.
    fn compute_jacobian_for(
        &mut self,
        constraint_indices: &[usize],
        var_map: &[(usize, bool)],
        vars: &[f64],
    ) -> Vec<Vec<f64>> {
        let num_residuals: usize = constraint_indices
            .iter()
            .map(|&ci| constraint_residual_count(&self.constraints[ci]))
            .sum();
        let num_vars = vars.len();
        let eps = 1e-8;

        let mut jacobian = vec![vec![0.0; num_vars]; num_residuals];

        let base_residuals = self.compute_residuals_for(constraint_indices);

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

            let perturbed = self.compute_residuals_for(constraint_indices);

            for i in 0..num_residuals {
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

/// Union-Find: find with path compression.
fn find(parent: &mut [usize], i: usize) -> usize {
    if parent[i] != i {
        parent[i] = find(parent, parent[i]);
    }
    parent[i]
}

/// Union-Find: union by rank.
fn union(parent: &mut [usize], rank: &mut [usize], a: usize, b: usize) {
    let ra = find(parent, a);
    let rb = find(parent, b);
    if ra == rb {
        return;
    }
    match rank[ra].cmp(&rank[rb]) {
        std::cmp::Ordering::Less => parent[ra] = rb,
        std::cmp::Ordering::Greater => parent[rb] = ra,
        std::cmp::Ordering::Equal => {
            parent[rb] = ra;
            rank[ra] += 1;
        }
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

    #[test]
    fn constraint_decomposition_two_groups() {
        // Two completely disconnected groups of points with constraints.
        let mut sketch = Sketch::new();

        // Group A: p0 (fixed), p1 (free) with distance = 5
        let p0 = sketch.add_point(SketchPoint::fixed(0.0, 0.0));
        let p1 = sketch.add_point(SketchPoint::new(1.0, 0.0));
        sketch.add_constraint(Constraint::Distance(p0, p1, 5.0));
        sketch.add_constraint(Constraint::Horizontal(p0, p1));

        // Group B: p2 (fixed), p3 (free) with FixX and FixY
        let _p2 = sketch.add_point(SketchPoint::fixed(10.0, 10.0));
        let p3 = sketch.add_point(SketchPoint::new(20.0, 20.0));
        sketch.add_constraint(Constraint::FixX(p3, 7.0));
        sketch.add_constraint(Constraint::FixY(p3, 8.0));

        // Verify decomposition finds two components
        let components = sketch.decompose_components();
        assert_eq!(components.len(), 2, "should have 2 independent components");

        // Solve and verify both groups converge
        let result = sketch.solve(100, TOL).unwrap();
        assert!(result.converged);

        // Group A: p1 should be at distance 5 from origin on x-axis
        let d01 = sketch.points[p1].x.hypot(sketch.points[p1].y);
        assert!(
            (d01 - 5.0).abs() < TOL,
            "group A distance should be 5, got {d01}"
        );
        assert!(
            (sketch.points[p1].y).abs() < TOL,
            "group A should be horizontal"
        );

        // Group B: p3 should be at (7, 8)
        assert!((sketch.points[p3].x - 7.0).abs() < TOL);
        assert!((sketch.points[p3].y - 8.0).abs() < TOL);
    }

    #[test]
    fn constraint_decomposition_single_group() {
        // All points connected — should produce a single component
        // and behave identically to the original monolithic solver.
        let mut sketch = Sketch::new();
        let p0 = sketch.add_point(SketchPoint::fixed(0.0, 0.0));
        let p1 = sketch.add_point(SketchPoint::new(0.5, 0.0));

        sketch.add_constraint(Constraint::Distance(p0, p1, 3.0));

        let components = sketch.decompose_components();
        assert_eq!(components.len(), 1, "should have 1 component");

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
    fn curve_intersection_line_circle() {
        // Line through (0,0) and (10,0) — the x-axis
        // Circle centered at (5,0) with radius 3
        // Intersection points: (2, 0) and (8, 0)
        let mut sketch = Sketch::new();

        let line_a = sketch.add_point(SketchPoint::fixed(0.0, 0.0));
        let line_b = sketch.add_point(SketchPoint::fixed(10.0, 0.0));
        let circle_center = sketch.add_point(SketchPoint::fixed(5.0, 0.0));

        // Start the intersection point near one of the solutions
        let p = sketch.add_point(SketchPoint::new(2.5, 0.5));

        sketch.add_constraint(Constraint::CurveIntersection {
            point: p,
            curve1: CurveRef::Line(line_a, line_b),
            curve2: CurveRef::Circle(circle_center, 3.0),
        });

        let result = sketch.solve(100, TOL).unwrap();
        assert!(result.converged, "line-circle intersection should converge");

        // The point should be on the x-axis (line constraint)
        assert!(
            sketch.points[p].y.abs() < TOL,
            "point should be on x-axis, y = {}",
            sketch.points[p].y
        );

        // The point should be on the circle
        let dx = sketch.points[p].x - 5.0;
        let dy = sketch.points[p].y;
        let dist_from_center = dx.hypot(dy);
        assert!(
            (dist_from_center - 3.0).abs() < TOL,
            "point should be on circle, dist = {dist_from_center}"
        );

        // Should converge to (2, 0) since we started near it
        assert!(
            (sketch.points[p].x - 2.0).abs() < TOL,
            "should converge to x=2, got x={}",
            sketch.points[p].x
        );
    }

    #[test]
    fn rectangle_30x20() {
        // Build a 30x20 rectangle from 4 points with distance, horizontal,
        // and vertical constraints. Pin one corner to the origin.
        let mut sketch = Sketch::new();

        // Four corners — start with rough initial guesses
        let p0 = sketch.add_point(SketchPoint::new(0.0, 0.0)); // bottom-left
        let p1 = sketch.add_point(SketchPoint::new(25.0, 1.0)); // bottom-right
        let p2 = sketch.add_point(SketchPoint::new(26.0, 18.0)); // top-right
        let p3 = sketch.add_point(SketchPoint::new(1.0, 17.0)); // top-left

        // Pin p0 at origin
        sketch.add_constraint(Constraint::FixX(p0, 0.0));
        sketch.add_constraint(Constraint::FixY(p0, 0.0));

        // Bottom edge: p0-p1 horizontal, length 30
        sketch.add_constraint(Constraint::Horizontal(p0, p1));
        sketch.add_constraint(Constraint::Distance(p0, p1, 30.0));

        // Right edge: p1-p2 vertical, length 20
        sketch.add_constraint(Constraint::Vertical(p1, p2));
        sketch.add_constraint(Constraint::Distance(p1, p2, 20.0));

        // Top edge: p2-p3 horizontal, length 30
        sketch.add_constraint(Constraint::Horizontal(p2, p3));
        sketch.add_constraint(Constraint::Distance(p2, p3, 30.0));

        // Left edge: p3-p0 vertical, length 20
        sketch.add_constraint(Constraint::Vertical(p3, p0));
        sketch.add_constraint(Constraint::Distance(p3, p0, 20.0));

        let result = sketch.solve(200, TOL).unwrap();
        assert!(result.converged, "rectangle should converge");

        // Verify corner positions
        let eps = 1e-4;
        assert!((sketch.points[p0].x).abs() < eps, "p0.x = 0");
        assert!((sketch.points[p0].y).abs() < eps, "p0.y = 0");
        assert!((sketch.points[p1].x - 30.0).abs() < eps, "p1.x = 30");
        assert!((sketch.points[p1].y).abs() < eps, "p1.y = 0");
        assert!((sketch.points[p2].x - 30.0).abs() < eps, "p2.x = 30");
        assert!((sketch.points[p2].y - 20.0).abs() < eps, "p2.y = 20");
        assert!((sketch.points[p3].x).abs() < eps, "p3.x = 0");
        assert!((sketch.points[p3].y - 20.0).abs() < eps, "p3.y = 20");

        // Verify edge lengths via Euclidean distance
        let d01 = (sketch.points[p1].x - sketch.points[p0].x)
            .hypot(sketch.points[p1].y - sketch.points[p0].y);
        let d12 = (sketch.points[p2].x - sketch.points[p1].x)
            .hypot(sketch.points[p2].y - sketch.points[p1].y);
        let d23 = (sketch.points[p3].x - sketch.points[p2].x)
            .hypot(sketch.points[p3].y - sketch.points[p2].y);
        let d30 = (sketch.points[p0].x - sketch.points[p3].x)
            .hypot(sketch.points[p0].y - sketch.points[p3].y);

        assert!((d01 - 30.0).abs() < eps, "bottom edge = 30, got {d01}");
        assert!((d12 - 20.0).abs() < eps, "right edge = 20, got {d12}");
        assert!((d23 - 30.0).abs() < eps, "top edge = 30, got {d23}");
        assert!((d30 - 20.0).abs() < eps, "left edge = 20, got {d30}");
    }
}
