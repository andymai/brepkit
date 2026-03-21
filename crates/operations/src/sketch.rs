//! 2D constraint solver for sketch-mode parametric design.
//!
//! Thin wrapper around `brepkit-sketch`. Provides the `Sketch` struct
//! for backwards compatibility and re-exports the full GCS API for new code.
//!
//! # Migration Guide
//! - **Old API**: `Sketch` with `SketchPoint` and `Constraint` (point-index based)
//! - **New API**: `GcsSystem` with `PointId`, `LineId`, `CircleId` handles
//!
//! The old `Sketch` API is preserved for backward compatibility. New code
//! should use `GcsSystem` directly for full entity and constraint support.

use brepkit_sketch::SketchError;

// Re-export the full GCS API for new code.
pub use brepkit_sketch::{
    CircleData, CircleId, Constraint as GcsConstraint, ConstraintId, DofAnalysis, GcsSystem,
    LineData, LineId, PointData, PointId, SolveResult,
};

/// A 2D point variable in the constraint system (legacy API).
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

/// Index into the sketch's point array (legacy API).
pub type PointIdx = usize;

/// A geometric constraint in the sketch (legacy API).
///
/// This uses point indices for backward compatibility.
/// For new code, use [`GcsConstraint`] with entity handles.
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
    /// The angle between line p1-p2 and line p3-p4 must equal the given value (radians).
    Angle(PointIdx, PointIdx, PointIdx, PointIdx, f64),
    /// The line p1-p2 must be perpendicular to line p3-p4.
    Perpendicular(PointIdx, PointIdx, PointIdx, PointIdx),
    /// The line p1-p2 must be parallel to line p3-p4.
    Parallel(PointIdx, PointIdx, PointIdx, PointIdx),
}

/// A 2D constraint sketch using the legacy point-index API.
///
/// Internally delegates to [`GcsSystem`].
#[derive(Debug, Default)]
pub struct Sketch {
    /// The points in the sketch.
    pub points: Vec<SketchPoint>,
    /// The constraints to satisfy.
    pub constraints: Vec<Constraint>,
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

    /// Solve the constraint system.
    ///
    /// Converts to a `GcsSystem`, solves, and writes positions back.
    ///
    /// # Errors
    /// Returns `SketchError` if an entity handle is invalid.
    pub fn solve(
        &mut self,
        max_iterations: usize,
        tolerance: f64,
    ) -> Result<SolveResult, SketchError> {
        let mut sys = GcsSystem::new();

        // Add points
        let point_ids: Vec<PointId> = self
            .points
            .iter()
            .map(|p| {
                sys.add_point(PointData {
                    x: p.x,
                    y: p.y,
                    fixed: p.fixed,
                })
            })
            .collect();

        // Track implicit lines created for point-pair constraints
        let mut line_cache: std::collections::HashMap<(usize, usize), LineId> =
            std::collections::HashMap::new();

        let mut get_or_create_line = |sys: &mut GcsSystem,
                                      ids: &[PointId],
                                      a: usize,
                                      b: usize|
         -> Result<LineId, SketchError> {
            let key = (a, b);
            if let Some(&lid) = line_cache.get(&key) {
                return Ok(lid);
            }
            let lid = sys.add_line(ids[a], ids[b])?;
            line_cache.insert(key, lid);
            Ok(lid)
        };

        // Convert constraints
        for c in &self.constraints {
            match c {
                Constraint::Coincident(a, b) => {
                    sys.add_constraint(GcsConstraint::Coincident(point_ids[*a], point_ids[*b]))?;
                }
                Constraint::Distance(a, b, d) => {
                    sys.add_constraint(GcsConstraint::Distance(point_ids[*a], point_ids[*b], *d))?;
                }
                Constraint::FixX(p, v) => {
                    sys.add_constraint(GcsConstraint::FixX(point_ids[*p], *v))?;
                }
                Constraint::FixY(p, v) => {
                    sys.add_constraint(GcsConstraint::FixY(point_ids[*p], *v))?;
                }
                Constraint::Vertical(a, b) => {
                    let l = get_or_create_line(&mut sys, &point_ids, *a, *b)?;
                    sys.add_constraint(GcsConstraint::Vertical(l))?;
                }
                Constraint::Horizontal(a, b) => {
                    let l = get_or_create_line(&mut sys, &point_ids, *a, *b)?;
                    sys.add_constraint(GcsConstraint::Horizontal(l))?;
                }
                Constraint::Angle(a, b, c, d, theta) => {
                    let l1 = get_or_create_line(&mut sys, &point_ids, *a, *b)?;
                    let l2 = get_or_create_line(&mut sys, &point_ids, *c, *d)?;
                    sys.add_constraint(GcsConstraint::Angle(l1, l2, *theta))?;
                }
                Constraint::Perpendicular(a, b, c, d) => {
                    let l1 = get_or_create_line(&mut sys, &point_ids, *a, *b)?;
                    let l2 = get_or_create_line(&mut sys, &point_ids, *c, *d)?;
                    sys.add_constraint(GcsConstraint::Perpendicular(l1, l2))?;
                }
                Constraint::Parallel(a, b, c, d) => {
                    let l1 = get_or_create_line(&mut sys, &point_ids, *a, *b)?;
                    let l2 = get_or_create_line(&mut sys, &point_ids, *c, *d)?;
                    sys.add_constraint(GcsConstraint::Parallel(l1, l2))?;
                }
            }
        }

        let result = sys.solve(max_iterations, tolerance)?;

        // Write solved positions back
        for (i, pid) in point_ids.iter().enumerate() {
            if let Some(data) = sys.point(*pid) {
                self.points[i].x = data.x;
                self.points[i].y = data.y;
            }
        }

        Ok(result)
    }
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
        let p1 = sketch.add_point(SketchPoint::new(0.5, 0.0));

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
        let p3 = sketch.add_point(SketchPoint::new(0.5, 0.5));

        sketch.add_constraint(Constraint::Perpendicular(p0, p1, p2, p3));

        let result = sketch.solve(100, TOL).unwrap();
        assert!(result.converged);

        let dx2 = sketch.points[p3].x - sketch.points[p2].x;
        let dot = dx2;
        assert!(dot.abs() < TOL, "lines should be perpendicular, dot={dot}");
    }

    #[test]
    fn parallel_constraint() {
        let mut sketch = Sketch::new();
        let p0 = sketch.add_point(SketchPoint::fixed(0.0, 0.0));
        let p1 = sketch.add_point(SketchPoint::fixed(1.0, 1.0));
        let p2 = sketch.add_point(SketchPoint::fixed(2.0, 0.0));
        let p3 = sketch.add_point(SketchPoint::new(3.0, 0.5));

        sketch.add_constraint(Constraint::Parallel(p0, p1, p2, p3));

        let result = sketch.solve(100, TOL).unwrap();
        assert!(result.converged);

        let dx2 = sketch.points[p3].x - sketch.points[p2].x;
        let dy2 = sketch.points[p3].y - sketch.points[p2].y;
        let cross = dy2 - dx2;
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

        sketch.add_constraint(Constraint::Horizontal(p0, p1));
        sketch.add_constraint(Constraint::Distance(p0, p1, 3.0));
        sketch.add_constraint(Constraint::Distance(p0, p2, 4.0));
        sketch.add_constraint(Constraint::Distance(p1, p2, 5.0));

        let result = sketch.solve(200, TOL).unwrap();
        assert!(result.converged, "triangle constraints should converge");

        let d01 = (sketch.points[p1].x - sketch.points[p0].x)
            .hypot(sketch.points[p1].y - sketch.points[p0].y);
        assert!((d01 - 3.0).abs() < TOL, "d01 should be 3, got {d01}");
    }

    #[test]
    fn rectangle_30x20() {
        let mut sketch = Sketch::new();

        let p0 = sketch.add_point(SketchPoint::new(0.0, 0.0));
        let p1 = sketch.add_point(SketchPoint::new(25.0, 1.0));
        let p2 = sketch.add_point(SketchPoint::new(26.0, 18.0));
        let p3 = sketch.add_point(SketchPoint::new(1.0, 17.0));

        sketch.add_constraint(Constraint::FixX(p0, 0.0));
        sketch.add_constraint(Constraint::FixY(p0, 0.0));

        sketch.add_constraint(Constraint::Horizontal(p0, p1));
        sketch.add_constraint(Constraint::Distance(p0, p1, 30.0));

        sketch.add_constraint(Constraint::Vertical(p1, p2));
        sketch.add_constraint(Constraint::Distance(p1, p2, 20.0));

        sketch.add_constraint(Constraint::Horizontal(p2, p3));
        sketch.add_constraint(Constraint::Distance(p2, p3, 30.0));

        sketch.add_constraint(Constraint::Vertical(p3, p0));
        sketch.add_constraint(Constraint::Distance(p3, p0, 20.0));

        let result = sketch.solve(200, TOL).unwrap();
        assert!(result.converged, "rectangle should converge");

        let eps = 1e-4;
        assert!((sketch.points[p0].x).abs() < eps, "p0.x = 0");
        assert!((sketch.points[p0].y).abs() < eps, "p0.y = 0");
        assert!((sketch.points[p1].x - 30.0).abs() < eps, "p1.x = 30");
        assert!((sketch.points[p1].y).abs() < eps, "p1.y = 0");
        assert!((sketch.points[p2].x - 30.0).abs() < eps, "p2.x = 30");
        assert!((sketch.points[p2].y - 20.0).abs() < eps, "p2.y = 20");
        assert!((sketch.points[p3].x).abs() < eps, "p3.x = 0");
        assert!((sketch.points[p3].y - 20.0).abs() < eps, "p3.y = 20");
    }
}
