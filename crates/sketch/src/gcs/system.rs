//! GCS system: CRUD operations, parameter management, and solve orchestration.

use std::collections::HashMap;

use crate::SketchError;

use super::constraint::{
    Constraint, ConstraintEntry, ConstraintId, EntitySnapshot, JacobianWriter, eval_jacobian,
    eval_residuals, residual_count,
};
use super::dof::{self, DofAnalysis};
use super::entity::{
    CircleData, CircleId, GenArena, LineData, LineId, ParamRef, PointData, PointId,
};
use super::solver::{self, SolveResult};

/// The geometric constraint system.
///
/// Owns all entities (points, lines, circles) and constraints.
/// Provides CRUD operations and orchestrates the solver.
pub struct GcsSystem {
    points: GenArena<PointData>,
    lines: GenArena<LineData>,
    circles: GenArena<CircleData>,
    constraints: GenArena<ConstraintEntry>,
    /// Cached parameter map (rebuilt when dirty).
    param_map: Vec<ParamRef>,
    /// Map from `ParamRef` to index in param_map.
    param_index: HashMap<ParamRef, usize>,
    /// Whether the param map needs rebuilding.
    dirty: bool,
}

impl Clone for GcsSystem {
    fn clone(&self) -> Self {
        Self {
            points: self.points.clone(),
            lines: self.lines.clone(),
            circles: self.circles.clone(),
            constraints: self.constraints.clone(),
            param_map: self.param_map.clone(),
            param_index: self.param_index.clone(),
            dirty: self.dirty,
        }
    }
}

impl Default for GcsSystem {
    fn default() -> Self {
        Self::new()
    }
}

impl GcsSystem {
    /// Create a new empty GCS.
    #[must_use]
    pub fn new() -> Self {
        Self {
            points: GenArena::new(),
            lines: GenArena::new(),
            circles: GenArena::new(),
            constraints: GenArena::new(),
            param_map: Vec::new(),
            param_index: HashMap::new(),
            dirty: false,
        }
    }

    // ── Entity CRUD ─────────────────────────────────────────────────

    /// Add a point. Returns its handle.
    pub fn add_point(&mut self, data: PointData) -> PointId {
        self.dirty = true;
        self.points.insert(data)
    }

    /// Get a point by handle.
    #[must_use]
    pub fn point(&self, id: PointId) -> Option<&PointData> {
        self.points.get(id)
    }

    /// Get a mutable reference to a point.
    pub fn point_mut(&mut self, id: PointId) -> Option<&mut PointData> {
        self.points.get_mut(id)
    }

    /// Remove a point. Fails if referenced by any line, circle, or constraint.
    ///
    /// # Errors
    ///
    /// Returns `SketchError::EntityInUse` if the point is referenced by a line,
    /// circle, or constraint. Returns `SketchError::InvalidHandle` if the handle
    /// is stale or invalid.
    pub fn remove_point(&mut self, id: PointId) -> Result<PointData, SketchError> {
        // Check lines
        for (_, line) in self.lines.iter() {
            if line.p1 == id || line.p2 == id {
                return Err(SketchError::EntityInUse);
            }
        }
        // Check circles
        for (_, circle) in self.circles.iter() {
            if circle.center == id {
                return Err(SketchError::EntityInUse);
            }
        }
        // Check constraints
        for (_, entry) in self.constraints.iter() {
            if constraint_references_point(&entry.constraint, id) {
                return Err(SketchError::EntityInUse);
            }
        }
        self.dirty = true;
        self.points.remove(id).ok_or(SketchError::InvalidHandle)
    }

    /// Add a line between two existing points.
    ///
    /// # Errors
    ///
    /// Returns `SketchError::InvalidHandle` if either point handle is invalid.
    pub fn add_line(&mut self, p1: PointId, p2: PointId) -> Result<LineId, SketchError> {
        if !self.points.contains(p1) || !self.points.contains(p2) {
            return Err(SketchError::InvalidHandle);
        }
        Ok(self.lines.insert(LineData { p1, p2 }))
    }

    /// Get a line by handle.
    #[must_use]
    pub fn line(&self, id: LineId) -> Option<&LineData> {
        self.lines.get(id)
    }

    /// Remove a line. Fails if referenced by any constraint.
    ///
    /// # Errors
    ///
    /// Returns `SketchError::EntityInUse` if the line is referenced by a constraint.
    /// Returns `SketchError::InvalidHandle` if the handle is stale or invalid.
    pub fn remove_line(&mut self, id: LineId) -> Result<LineData, SketchError> {
        for (_, entry) in self.constraints.iter() {
            if constraint_references_line(&entry.constraint, id) {
                return Err(SketchError::EntityInUse);
            }
        }
        self.lines.remove(id).ok_or(SketchError::InvalidHandle)
    }

    /// Add a circle with a center point and radius.
    ///
    /// # Errors
    ///
    /// Returns `SketchError::InvalidHandle` if the center point handle is invalid.
    pub fn add_circle(&mut self, center: PointId, radius: f64) -> Result<CircleId, SketchError> {
        if !self.points.contains(center) {
            return Err(SketchError::InvalidHandle);
        }
        self.dirty = true;
        Ok(self.circles.insert(CircleData { center, radius }))
    }

    /// Get a circle by handle.
    #[must_use]
    pub fn circle(&self, id: CircleId) -> Option<&CircleData> {
        self.circles.get(id)
    }

    /// Remove a circle. Fails if referenced by any constraint.
    ///
    /// # Errors
    ///
    /// Returns `SketchError::EntityInUse` if the circle is referenced by a constraint.
    /// Returns `SketchError::InvalidHandle` if the handle is stale or invalid.
    pub fn remove_circle(&mut self, id: CircleId) -> Result<CircleData, SketchError> {
        for (_, entry) in self.constraints.iter() {
            if constraint_references_circle(&entry.constraint, id) {
                return Err(SketchError::EntityInUse);
            }
        }
        self.dirty = true;
        self.circles.remove(id).ok_or(SketchError::InvalidHandle)
    }

    // ── Constraint CRUD ─────────────────────────────────────────────

    /// Add a constraint. Validates that all referenced entities exist.
    ///
    /// # Errors
    ///
    /// Returns `SketchError::InvalidHandle` if any entity referenced by the
    /// constraint does not exist.
    pub fn add_constraint(&mut self, constraint: Constraint) -> Result<ConstraintId, SketchError> {
        self.validate_constraint(&constraint)?;
        self.dirty = true;
        Ok(self.constraints.insert(ConstraintEntry { constraint }))
    }

    /// Remove a constraint by handle.
    ///
    /// # Errors
    ///
    /// Returns `SketchError::InvalidHandle` if the handle is stale or invalid.
    pub fn remove_constraint(&mut self, id: ConstraintId) -> Result<(), SketchError> {
        self.constraints
            .remove(id)
            .map(|_| {
                self.dirty = true;
            })
            .ok_or(SketchError::InvalidHandle)
    }

    /// Get a constraint by handle.
    #[must_use]
    pub fn constraint(&self, id: ConstraintId) -> Option<&Constraint> {
        self.constraints.get(id).map(|e| &e.constraint)
    }

    /// Number of constraints.
    #[must_use]
    pub fn constraint_count(&self) -> usize {
        self.constraints.len()
    }

    /// Number of points.
    #[must_use]
    pub fn point_count(&self) -> usize {
        self.points.len()
    }

    /// Number of lines.
    #[must_use]
    pub fn line_count(&self) -> usize {
        self.lines.len()
    }

    /// Number of circles.
    #[must_use]
    pub fn circle_count(&self) -> usize {
        self.circles.len()
    }

    // ── Solve ───────────────────────────────────────────────────────

    /// Solve the constraint system.
    ///
    /// Modifies entity positions in-place to satisfy all constraints.
    ///
    /// # Errors
    ///
    /// Returns `SketchError` if the system parameters are in an invalid state.
    /// The `Result` wrapper is retained for future error paths (e.g. singular
    /// Jacobian detection).
    #[allow(clippy::unnecessary_wraps)]
    pub fn solve(
        &mut self,
        max_iterations: usize,
        tolerance: f64,
    ) -> Result<SolveResult, SketchError> {
        self.rebuild_if_dirty();

        let n = self.param_map.len();
        let m: usize = self
            .constraints
            .iter()
            .map(|(_, e)| residual_count(&e.constraint))
            .sum();

        if n == 0 {
            // No free params — just check residuals
            let snap = self.build_snapshot();
            let mut residuals = Vec::with_capacity(m);
            for (_, entry) in self.constraints.iter() {
                eval_residuals(&entry.constraint, &snap, &mut residuals);
            }
            let max_r = residuals.iter().fold(0.0_f64, |a, &b| a.max(b.abs()));
            return Ok(SolveResult {
                converged: max_r < tolerance,
                iterations: 0,
                max_residual: max_r,
            });
        }

        // Extract params
        let mut params = self.extract_params();
        let param_index = self.param_index.clone();
        let param_map = self.param_map.clone();

        // Collect constraints for closures
        let constraints: Vec<Constraint> = self
            .constraints
            .iter()
            .map(|(_, e)| e.constraint.clone())
            .collect();

        let residual_fn = |p: &[f64]| -> Vec<f64> {
            let snap = build_snapshot_from_params(p, &param_map, &param_index, self);
            let mut r = Vec::with_capacity(m);
            for c in &constraints {
                eval_residuals(c, &snap, &mut r);
            }
            r
        };

        let jacobian_fn = |p: &[f64]| -> Vec<f64> {
            let snap = build_snapshot_from_params(p, &param_map, &param_index, self);
            let mut jac = vec![0.0; m * n];
            let mut row = 0;
            {
                let mut jw = JacobianWriter {
                    data: &mut jac,
                    ncols: n,
                    param_index: &param_index,
                };
                for c in &constraints {
                    eval_jacobian(c, &snap, &mut jw, row);
                    row += residual_count(c);
                }
            }
            jac
        };

        let result = solver::solve_dogleg(
            &mut params,
            &residual_fn,
            &jacobian_fn,
            m,
            max_iterations,
            tolerance,
        );

        // Write back solved params
        self.write_params(&params);

        Ok(result)
    }

    /// Analyze degrees of freedom in the current system.
    pub fn dof(&mut self) -> DofAnalysis {
        self.rebuild_if_dirty();

        let n = self.param_map.len();
        let m: usize = self
            .constraints
            .iter()
            .map(|(_, e)| residual_count(&e.constraint))
            .sum();

        if n == 0 || m == 0 {
            return DofAnalysis {
                dof: n,
                rank: 0,
                num_params: n,
                num_equations: m,
            };
        }

        let params = self.extract_params();
        let snap = self.build_snapshot();
        let mut jac = vec![0.0; m * n];
        let mut row = 0;
        {
            let mut jw = JacobianWriter {
                data: &mut jac,
                ncols: n,
                param_index: &self.param_index,
            };
            for (_, entry) in self.constraints.iter() {
                eval_jacobian(&entry.constraint, &snap, &mut jw, row);
                row += residual_count(&entry.constraint);
            }
        }
        let _ = params; // params were needed to build snapshot

        dof::analyze(&jac, m, n)
    }

    /// Iterate over all points.
    pub fn points(&self) -> impl Iterator<Item = (PointId, &PointData)> {
        self.points.iter()
    }

    /// Iterate over all lines.
    pub fn lines(&self) -> impl Iterator<Item = (LineId, &LineData)> {
        self.lines.iter()
    }

    /// Iterate over all circles.
    pub fn circles(&self) -> impl Iterator<Item = (CircleId, &CircleData)> {
        self.circles.iter()
    }

    // ── Internal ────────────────────────────────────────────────────

    /// Rebuild parameter map if dirty.
    fn rebuild_if_dirty(&mut self) {
        if !self.dirty {
            return;
        }
        self.param_map.clear();
        self.param_index.clear();

        // Free point params
        for (id, data) in self.points.iter() {
            if !data.fixed {
                let idx = self.param_map.len();
                self.param_map.push(ParamRef::PointX(id));
                self.param_index.insert(ParamRef::PointX(id), idx);
                let idx = self.param_map.len();
                self.param_map.push(ParamRef::PointY(id));
                self.param_index.insert(ParamRef::PointY(id), idx);
            }
        }

        // Circle radius params
        for (id, _) in self.circles.iter() {
            let idx = self.param_map.len();
            self.param_map.push(ParamRef::CircleRadius(id));
            self.param_index.insert(ParamRef::CircleRadius(id), idx);
        }

        self.dirty = false;
    }

    /// Extract parameter values from entities.
    fn extract_params(&self) -> Vec<f64> {
        self.param_map
            .iter()
            .map(|pr| match pr {
                ParamRef::PointX(id) => self.points.get(*id).map_or(0.0, |p| p.x),
                ParamRef::PointY(id) => self.points.get(*id).map_or(0.0, |p| p.y),
                ParamRef::CircleRadius(id) => self.circles.get(*id).map_or(0.0, |c| c.radius),
            })
            .collect()
    }

    /// Write parameter values back to entities.
    fn write_params(&mut self, params: &[f64]) {
        for (i, pr) in self.param_map.iter().enumerate() {
            match pr {
                ParamRef::PointX(id) => {
                    if let Some(p) = self.points.get_mut(*id) {
                        p.x = params[i];
                    }
                }
                ParamRef::PointY(id) => {
                    if let Some(p) = self.points.get_mut(*id) {
                        p.y = params[i];
                    }
                }
                ParamRef::CircleRadius(id) => {
                    if let Some(c) = self.circles.get_mut(*id) {
                        c.radius = params[i];
                    }
                }
            }
        }
    }

    /// Build an entity snapshot for residual/Jacobian evaluation.
    fn build_snapshot(&self) -> EntitySnapshot {
        EntitySnapshot {
            points: self.points.iter().map(|(id, d)| (id, (d.x, d.y))).collect(),
            lines: self
                .lines
                .iter()
                .map(|(id, d)| (id, (d.p1, d.p2)))
                .collect(),
            circles: self
                .circles
                .iter()
                .map(|(id, d)| (id, (d.center, d.radius)))
                .collect(),
        }
    }

    /// Validate all entity references in a constraint.
    fn validate_constraint(&self, c: &Constraint) -> Result<(), SketchError> {
        match c {
            Constraint::Coincident(p1, p2) | Constraint::Distance(p1, p2, _) => {
                self.check_point(*p1)?;
                self.check_point(*p2)?;
            }
            Constraint::PointLineDistance(pt, line, _) => {
                self.check_point(*pt)?;
                self.check_line(*line)?;
            }
            Constraint::FixX(p, _) | Constraint::FixY(p, _) => {
                self.check_point(*p)?;
            }
            Constraint::Horizontal(line) | Constraint::Vertical(line) => {
                self.check_line(*line)?;
            }
            Constraint::Angle(l1, l2, _)
            | Constraint::Perpendicular(l1, l2)
            | Constraint::Parallel(l1, l2) => {
                self.check_line(*l1)?;
                self.check_line(*l2)?;
            }
        }
        Ok(())
    }

    fn check_point(&self, id: PointId) -> Result<(), SketchError> {
        if self.points.contains(id) {
            Ok(())
        } else {
            Err(SketchError::InvalidHandle)
        }
    }

    fn check_line(&self, id: LineId) -> Result<(), SketchError> {
        if self.lines.contains(id) {
            Ok(())
        } else {
            Err(SketchError::InvalidHandle)
        }
    }
}

/// Build a snapshot from parameter values (used in solver closures).
fn build_snapshot_from_params(
    params: &[f64],
    _param_map: &[ParamRef],
    param_index: &HashMap<ParamRef, usize>,
    sys: &GcsSystem,
) -> EntitySnapshot {
    let points = sys
        .points
        .iter()
        .map(|(id, data)| {
            let x = param_index
                .get(&ParamRef::PointX(id))
                .map_or(data.x, |&i| params[i]);
            let y = param_index
                .get(&ParamRef::PointY(id))
                .map_or(data.y, |&i| params[i]);
            (id, (x, y))
        })
        .collect();

    let lines = sys.lines.iter().map(|(id, d)| (id, (d.p1, d.p2))).collect();

    let circles = sys
        .circles
        .iter()
        .map(|(id, data)| {
            let r = param_index
                .get(&ParamRef::CircleRadius(id))
                .map_or(data.radius, |&i| params[i]);
            (id, (data.center, r))
        })
        .collect();

    EntitySnapshot {
        points,
        lines,
        circles,
    }
}

/// Check if a constraint references a specific point.
fn constraint_references_point(c: &Constraint, id: PointId) -> bool {
    match c {
        Constraint::Coincident(p1, p2) | Constraint::Distance(p1, p2, _) => *p1 == id || *p2 == id,
        Constraint::PointLineDistance(pt, _, _) => *pt == id,
        Constraint::FixX(p, _) | Constraint::FixY(p, _) => *p == id,
        Constraint::Horizontal(_)
        | Constraint::Vertical(_)
        | Constraint::Angle(_, _, _)
        | Constraint::Perpendicular(_, _)
        | Constraint::Parallel(_, _) => false,
    }
}

/// Check if a constraint references a specific line.
fn constraint_references_line(c: &Constraint, id: LineId) -> bool {
    match c {
        Constraint::Horizontal(l) | Constraint::Vertical(l) => *l == id,
        Constraint::PointLineDistance(_, l, _) => *l == id,
        Constraint::Angle(l1, l2, _)
        | Constraint::Perpendicular(l1, l2)
        | Constraint::Parallel(l1, l2) => *l1 == id || *l2 == id,
        Constraint::Coincident(_, _)
        | Constraint::Distance(_, _, _)
        | Constraint::FixX(_, _)
        | Constraint::FixY(_, _) => false,
    }
}

/// Check if a constraint references a specific circle.
fn constraint_references_circle(c: &Constraint, _id: CircleId) -> bool {
    // PR1 has no circle-specific constraints yet
    match c {
        Constraint::Coincident(_, _)
        | Constraint::Distance(_, _, _)
        | Constraint::PointLineDistance(_, _, _)
        | Constraint::FixX(_, _)
        | Constraint::FixY(_, _)
        | Constraint::Horizontal(_)
        | Constraint::Vertical(_)
        | Constraint::Angle(_, _, _)
        | Constraint::Perpendicular(_, _)
        | Constraint::Parallel(_, _) => false,
    }
}

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::expect_used)]
mod tests {
    use super::*;

    const TOL: f64 = 1e-10;

    #[test]
    fn fix_xy_converges() {
        let mut sys = GcsSystem::new();
        let p = sys.add_point(PointData {
            x: 5.0,
            y: 7.0,
            fixed: false,
        });
        sys.add_constraint(Constraint::FixX(p, 2.0)).unwrap();
        sys.add_constraint(Constraint::FixY(p, 3.0)).unwrap();

        let result = sys.solve(100, TOL).unwrap();
        assert!(result.converged);
        let pt = sys.point(p).unwrap();
        assert!((pt.x - 2.0).abs() < TOL);
        assert!((pt.y - 3.0).abs() < TOL);
    }

    #[test]
    fn distance_constraint() {
        let mut sys = GcsSystem::new();
        let p0 = sys.add_point(PointData {
            x: 0.0,
            y: 0.0,
            fixed: true,
        });
        let p1 = sys.add_point(PointData {
            x: 0.5,
            y: 0.0,
            fixed: false,
        });
        sys.add_constraint(Constraint::Distance(p0, p1, 3.0))
            .unwrap();

        let result = sys.solve(100, TOL).unwrap();
        assert!(result.converged, "max_r = {}", result.max_residual);
        let pt0 = sys.point(p0).unwrap();
        let pt1 = sys.point(p1).unwrap();
        let dist = ((pt1.x - pt0.x).powi(2) + (pt1.y - pt0.y).powi(2)).sqrt();
        assert!(
            (dist - 3.0).abs() < 1e-6,
            "distance should be 3.0, got {dist}"
        );
    }

    #[test]
    fn coincident_constraint() {
        let mut sys = GcsSystem::new();
        let p0 = sys.add_point(PointData {
            x: 1.0,
            y: 2.0,
            fixed: true,
        });
        let p1 = sys.add_point(PointData {
            x: 3.0,
            y: 4.0,
            fixed: false,
        });
        sys.add_constraint(Constraint::Coincident(p0, p1)).unwrap();

        let result = sys.solve(100, TOL).unwrap();
        assert!(result.converged);
        let pt = sys.point(p1).unwrap();
        assert!((pt.x - 1.0).abs() < TOL);
        assert!((pt.y - 2.0).abs() < TOL);
    }

    #[test]
    fn horizontal_line() {
        let mut sys = GcsSystem::new();
        let p0 = sys.add_point(PointData {
            x: 0.0,
            y: 1.0,
            fixed: true,
        });
        let p1 = sys.add_point(PointData {
            x: 5.0,
            y: 3.0,
            fixed: false,
        });
        let l = sys.add_line(p0, p1).unwrap();
        sys.add_constraint(Constraint::Horizontal(l)).unwrap();

        let result = sys.solve(100, TOL).unwrap();
        assert!(result.converged);
        assert!((sys.point(p1).unwrap().y - sys.point(p0).unwrap().y).abs() < TOL);
    }

    #[test]
    fn vertical_line() {
        let mut sys = GcsSystem::new();
        let p0 = sys.add_point(PointData {
            x: 2.0,
            y: 0.0,
            fixed: true,
        });
        let p1 = sys.add_point(PointData {
            x: 5.0,
            y: 7.0,
            fixed: false,
        });
        let l = sys.add_line(p0, p1).unwrap();
        sys.add_constraint(Constraint::Vertical(l)).unwrap();

        let result = sys.solve(100, TOL).unwrap();
        assert!(result.converged);
        assert!((sys.point(p1).unwrap().x - sys.point(p0).unwrap().x).abs() < TOL);
    }

    #[test]
    fn perpendicular_lines() {
        let mut sys = GcsSystem::new();
        let p0 = sys.add_point(PointData {
            x: 0.0,
            y: 0.0,
            fixed: true,
        });
        let p1 = sys.add_point(PointData {
            x: 1.0,
            y: 0.0,
            fixed: true,
        });
        let p2 = sys.add_point(PointData {
            x: 0.0,
            y: 0.0,
            fixed: true,
        });
        let p3 = sys.add_point(PointData {
            x: 0.5,
            y: 0.5,
            fixed: false,
        });
        let l1 = sys.add_line(p0, p1).unwrap();
        let l2 = sys.add_line(p2, p3).unwrap();
        sys.add_constraint(Constraint::Perpendicular(l1, l2))
            .unwrap();

        let result = sys.solve(100, TOL).unwrap();
        assert!(result.converged);
        let pt3 = sys.point(p3).unwrap();
        // Line p0-p1 is along X. Perpendicular means p3.x - p2.x = 0
        assert!(pt3.x.abs() < TOL, "p3.x = {}", pt3.x);
    }

    #[test]
    fn parallel_lines() {
        let mut sys = GcsSystem::new();
        let p0 = sys.add_point(PointData {
            x: 0.0,
            y: 0.0,
            fixed: true,
        });
        let p1 = sys.add_point(PointData {
            x: 1.0,
            y: 1.0,
            fixed: true,
        });
        let p2 = sys.add_point(PointData {
            x: 2.0,
            y: 0.0,
            fixed: true,
        });
        let p3 = sys.add_point(PointData {
            x: 3.0,
            y: 0.5,
            fixed: false,
        });
        let l1 = sys.add_line(p0, p1).unwrap();
        let l2 = sys.add_line(p2, p3).unwrap();
        sys.add_constraint(Constraint::Parallel(l1, l2)).unwrap();

        let result = sys.solve(100, TOL).unwrap();
        assert!(result.converged);
        let pt3 = sys.point(p3).unwrap();
        let dy = pt3.y - 0.0; // p2.y = 0
        let dx = pt3.x - 2.0; // p2.x = 2
        // Cross with (1,1) should be 0: dy - dx = 0
        assert!((dy - dx).abs() < TOL, "not parallel: dy={dy}, dx={dx}");
    }

    #[test]
    fn rectangle_30x20() {
        let mut sys = GcsSystem::new();

        let p0 = sys.add_point(PointData {
            x: 0.0,
            y: 0.0,
            fixed: false,
        });
        let p1 = sys.add_point(PointData {
            x: 25.0,
            y: 1.0,
            fixed: false,
        });
        let p2 = sys.add_point(PointData {
            x: 26.0,
            y: 18.0,
            fixed: false,
        });
        let p3 = sys.add_point(PointData {
            x: 1.0,
            y: 17.0,
            fixed: false,
        });

        let bottom = sys.add_line(p0, p1).unwrap();
        let right = sys.add_line(p1, p2).unwrap();
        let top = sys.add_line(p2, p3).unwrap();
        let left = sys.add_line(p3, p0).unwrap();

        // Pin origin
        sys.add_constraint(Constraint::FixX(p0, 0.0)).unwrap();
        sys.add_constraint(Constraint::FixY(p0, 0.0)).unwrap();

        // Bottom: horizontal, length 30
        sys.add_constraint(Constraint::Horizontal(bottom)).unwrap();
        sys.add_constraint(Constraint::Distance(p0, p1, 30.0))
            .unwrap();

        // Right: vertical, length 20
        sys.add_constraint(Constraint::Vertical(right)).unwrap();
        sys.add_constraint(Constraint::Distance(p1, p2, 20.0))
            .unwrap();

        // Top: horizontal, length 30
        sys.add_constraint(Constraint::Horizontal(top)).unwrap();
        sys.add_constraint(Constraint::Distance(p2, p3, 30.0))
            .unwrap();

        // Left: vertical, length 20
        sys.add_constraint(Constraint::Vertical(left)).unwrap();
        sys.add_constraint(Constraint::Distance(p3, p0, 20.0))
            .unwrap();

        let result = sys.solve(200, 1e-8).unwrap();
        assert!(
            result.converged,
            "rectangle: max_r = {}",
            result.max_residual
        );

        let eps = 1e-4;
        let pt0 = sys.point(p0).unwrap();
        let pt1 = sys.point(p1).unwrap();
        let pt2 = sys.point(p2).unwrap();
        let pt3 = sys.point(p3).unwrap();

        assert!(pt0.x.abs() < eps, "p0.x = {}", pt0.x);
        assert!(pt0.y.abs() < eps, "p0.y = {}", pt0.y);
        assert!((pt1.x - 30.0).abs() < eps, "p1.x = {}", pt1.x);
        assert!(pt1.y.abs() < eps, "p1.y = {}", pt1.y);
        assert!((pt2.x - 30.0).abs() < eps, "p2.x = {}", pt2.x);
        assert!((pt2.y - 20.0).abs() < eps, "p2.y = {}", pt2.y);
        assert!(pt3.x.abs() < eps, "p3.x = {}", pt3.x);
        assert!((pt3.y - 20.0).abs() < eps, "p3.y = {}", pt3.y);
    }

    #[test]
    fn dof_analysis() {
        let mut sys = GcsSystem::new();
        let p = sys.add_point(PointData {
            x: 0.0,
            y: 0.0,
            fixed: false,
        });

        // Free point: 2 DOF
        let dof = sys.dof();
        assert_eq!(dof.dof, 2);

        // Fix X: 1 DOF
        let cx = sys.add_constraint(Constraint::FixX(p, 0.0)).unwrap();
        let dof = sys.dof();
        assert_eq!(dof.dof, 1);

        // Fix Y: 0 DOF
        sys.add_constraint(Constraint::FixY(p, 0.0)).unwrap();
        let dof = sys.dof();
        assert_eq!(dof.dof, 0);

        // Remove FixX: back to 1 DOF
        sys.remove_constraint(cx).unwrap();
        let dof = sys.dof();
        assert_eq!(dof.dof, 1);
    }

    #[test]
    fn remove_point_in_use_fails() {
        let mut sys = GcsSystem::new();
        let p0 = sys.add_point(PointData {
            x: 0.0,
            y: 0.0,
            fixed: false,
        });
        let p1 = sys.add_point(PointData {
            x: 1.0,
            y: 0.0,
            fixed: false,
        });
        let _l = sys.add_line(p0, p1).unwrap();

        assert!(sys.remove_point(p0).is_err());
    }

    #[test]
    fn remove_line_in_use_fails() {
        let mut sys = GcsSystem::new();
        let p0 = sys.add_point(PointData {
            x: 0.0,
            y: 0.0,
            fixed: false,
        });
        let p1 = sys.add_point(PointData {
            x: 1.0,
            y: 0.0,
            fixed: false,
        });
        let l = sys.add_line(p0, p1).unwrap();
        sys.add_constraint(Constraint::Horizontal(l)).unwrap();

        assert!(sys.remove_line(l).is_err());
    }

    #[test]
    fn stale_constraint_handle() {
        let mut sys = GcsSystem::new();
        let p = sys.add_point(PointData {
            x: 0.0,
            y: 0.0,
            fixed: false,
        });
        let c = sys.add_constraint(Constraint::FixX(p, 0.0)).unwrap();
        sys.remove_constraint(c).unwrap();
        assert!(sys.remove_constraint(c).is_err());
    }

    #[test]
    fn solve_after_removal() {
        let mut sys = GcsSystem::new();
        let p = sys.add_point(PointData {
            x: 5.0,
            y: 7.0,
            fixed: false,
        });
        let _cx = sys.add_constraint(Constraint::FixX(p, 2.0)).unwrap();
        let cy = sys.add_constraint(Constraint::FixY(p, 3.0)).unwrap();

        // Solve, then remove FixY, re-solve
        let r1 = sys.solve(100, TOL).unwrap();
        assert!(r1.converged);

        sys.remove_constraint(cy).unwrap();
        let r2 = sys.solve(100, TOL).unwrap();
        assert!(r2.converged);
        // X should still be at 2.0, Y should be unchanged from last solve
        assert!((sys.point(p).unwrap().x - 2.0).abs() < TOL);
    }

    #[test]
    fn add_constraint_with_invalid_handle_fails() {
        let mut sys = GcsSystem::new();
        let p = sys.add_point(PointData {
            x: 0.0,
            y: 0.0,
            fixed: false,
        });
        sys.remove_point(p).unwrap();
        assert!(sys.add_constraint(Constraint::FixX(p, 0.0)).is_err());
    }

    #[test]
    fn triangle_345() {
        let mut sys = GcsSystem::new();
        let p0 = sys.add_point(PointData {
            x: 0.0,
            y: 0.0,
            fixed: true,
        });
        let p1 = sys.add_point(PointData {
            x: 1.0,
            y: 0.0,
            fixed: false,
        });
        let p2 = sys.add_point(PointData {
            x: 0.5,
            y: 1.0,
            fixed: false,
        });

        let bottom = sys.add_line(p0, p1).unwrap();
        sys.add_constraint(Constraint::Horizontal(bottom)).unwrap();
        sys.add_constraint(Constraint::Distance(p0, p1, 3.0))
            .unwrap();
        sys.add_constraint(Constraint::Distance(p0, p2, 4.0))
            .unwrap();
        sys.add_constraint(Constraint::Distance(p1, p2, 5.0))
            .unwrap();

        let result = sys.solve(200, 1e-8).unwrap();
        assert!(
            result.converged,
            "triangle: max_r = {}",
            result.max_residual
        );

        let pt1 = sys.point(p1).unwrap();
        let d01 = (pt1.x.powi(2) + pt1.y.powi(2)).sqrt();
        assert!((d01 - 3.0).abs() < 1e-4, "d01 = {d01}");
    }

    #[test]
    fn fixed_points_no_solve_needed() {
        let mut sys = GcsSystem::new();
        let p0 = sys.add_point(PointData {
            x: 0.0,
            y: 0.0,
            fixed: true,
        });
        let p1 = sys.add_point(PointData {
            x: 1.0,
            y: 0.0,
            fixed: true,
        });
        sys.add_constraint(Constraint::Distance(p0, p1, 1.0))
            .unwrap();

        let result = sys.solve(100, TOL).unwrap();
        assert!(result.converged);
        assert_eq!(result.iterations, 0);
    }
}
