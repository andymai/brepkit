//! GCS system: CRUD operations, parameter management, and solve orchestration.

use std::collections::HashMap;

use crate::SketchError;

use super::constraint::{
    Constraint, ConstraintEntry, ConstraintId, EntitySnapshot, JacobianWriter, eval_jacobian,
    eval_residuals, residual_count,
};
use super::dof::{self, DofAnalysis};
use super::entity::{
    ArcData, ArcId, CircleData, CircleId, GenArena, LineData, LineId, ParamRef, PointData, PointId,
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
    arcs: GenArena<ArcData>,
    constraints: GenArena<ConstraintEntry>,
    /// Internal constraints auto-added by `add_arc` (center–end distance).
    /// Keyed by `ArcId` so they can be removed with the arc.
    arc_internal_constraints: HashMap<ArcId, ConstraintId>,
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
            arcs: self.arcs.clone(),
            constraints: self.constraints.clone(),
            arc_internal_constraints: self.arc_internal_constraints.clone(),
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
            arcs: GenArena::new(),
            constraints: GenArena::new(),
            arc_internal_constraints: HashMap::new(),
            param_map: Vec::new(),
            param_index: HashMap::new(),
            dirty: false,
        }
    }

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
        for (_, line) in self.lines.iter() {
            if line.p1 == id || line.p2 == id {
                return Err(SketchError::EntityInUse);
            }
        }
        for (_, circle) in self.circles.iter() {
            if circle.center == id {
                return Err(SketchError::EntityInUse);
            }
        }
        for (_, arc) in self.arcs.iter() {
            if arc.center == id || arc.start == id || arc.end == id {
                return Err(SketchError::EntityInUse);
            }
        }
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

    /// Add an arc defined by center, start, and end points.
    ///
    /// Auto-adds an internal `PointOnArc(end, arc)` constraint so that
    /// `dist(center, end) == dist(center, start)` is maintained dynamically
    /// as the start point moves.
    ///
    /// # Errors
    ///
    /// Returns `SketchError::InvalidHandle` if any point handle is invalid.
    pub fn add_arc(
        &mut self,
        center: PointId,
        start: PointId,
        end: PointId,
    ) -> Result<ArcId, SketchError> {
        self.check_point(center)?;
        self.check_point(start)?;
        self.check_point(end)?;

        let arc_id = self.arcs.insert(ArcData { center, start, end });

        // Internal constraint: end point must lie on the arc's circle
        // (dynamically tracks dist(center, start) rather than a frozen radius)
        let cid = self.constraints.insert(ConstraintEntry {
            constraint: Constraint::PointOnArc(end, arc_id),
        });
        self.arc_internal_constraints.insert(arc_id, cid);

        self.dirty = true;
        Ok(arc_id)
    }

    /// Get an arc by handle.
    #[must_use]
    pub fn arc(&self, id: ArcId) -> Option<&ArcData> {
        self.arcs.get(id)
    }

    /// Get a mutable reference to an arc.
    pub fn arc_mut(&mut self, id: ArcId) -> Option<&mut ArcData> {
        self.arcs.get_mut(id)
    }

    /// Remove an arc. Fails if referenced by any user constraint.
    ///
    /// Also removes the internal distance constraint that was auto-added
    /// by [`add_arc`](Self::add_arc).
    ///
    /// # Errors
    ///
    /// Returns `SketchError::EntityInUse` if the arc is referenced by a constraint.
    /// Returns `SketchError::InvalidHandle` if the handle is stale or invalid.
    pub fn remove_arc(&mut self, id: ArcId) -> Result<ArcData, SketchError> {
        for (cid, entry) in self.constraints.iter() {
            if self.arc_internal_constraints.get(&id) == Some(&cid) {
                continue;
            }
            if constraint_references_arc(&entry.constraint, id) {
                return Err(SketchError::EntityInUse);
            }
        }

        if let Some(cid) = self.arc_internal_constraints.remove(&id) {
            self.constraints.remove(cid);
        }

        self.dirty = true;
        self.arcs.remove(id).ok_or(SketchError::InvalidHandle)
    }

    /// Number of arcs.
    #[must_use]
    pub fn arc_count(&self) -> usize {
        self.arcs.len()
    }

    /// Iterate over all arcs.
    pub fn arcs(&self) -> impl Iterator<Item = (ArcId, &ArcData)> {
        self.arcs.iter()
    }

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

    /// Number of constraints (includes internal arc constraints).
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

        let mut params = self.extract_params();
        let param_index = self.param_index.clone();
        let param_map = self.param_map.clone();

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

    /// Rebuild parameter map if dirty.
    fn rebuild_if_dirty(&mut self) {
        if !self.dirty {
            return;
        }
        self.param_map.clear();
        self.param_index.clear();

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
            arcs: self
                .arcs
                .iter()
                .map(|(id, d)| (id, (d.center, d.start, d.end)))
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
            Constraint::PointOnCircle(pt, circ) => {
                self.check_point(*pt)?;
                self.check_circle(*circ)?;
            }
            Constraint::PointOnArc(pt, arc) => {
                self.check_point(*pt)?;
                self.check_arc(*arc)?;
            }
            Constraint::TangentLineArc(line, arc, shared) => {
                self.check_line(*line)?;
                self.check_arc(*arc)?;
                self.check_point(*shared)?;
            }
            Constraint::TangentArcArc(arc1, arc2, shared) => {
                self.check_arc(*arc1)?;
                self.check_arc(*arc2)?;
                self.check_point(*shared)?;
            }
            Constraint::EqualRadiusArcArc(arc1, arc2) => {
                self.check_arc(*arc1)?;
                self.check_arc(*arc2)?;
            }
            Constraint::EqualRadiusArcCircle(arc, circ) => {
                self.check_arc(*arc)?;
                self.check_circle(*circ)?;
            }
            Constraint::ArcLength(arc, _) => {
                self.check_arc(*arc)?;
            }
            Constraint::ConcentricArcArc(arc1, arc2) => {
                self.check_arc(*arc1)?;
                self.check_arc(*arc2)?;
            }
            Constraint::ConcentricArcCircle(arc, circ) => {
                self.check_arc(*arc)?;
                self.check_circle(*circ)?;
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

    fn check_circle(&self, id: CircleId) -> Result<(), SketchError> {
        if self.circles.contains(id) {
            Ok(())
        } else {
            Err(SketchError::InvalidHandle)
        }
    }

    fn check_arc(&self, id: ArcId) -> Result<(), SketchError> {
        if self.arcs.contains(id) {
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

    let arcs = sys
        .arcs
        .iter()
        .map(|(id, d)| (id, (d.center, d.start, d.end)))
        .collect();

    EntitySnapshot {
        points,
        lines,
        circles,
        arcs,
    }
}

/// Check if a constraint references a specific point.
fn constraint_references_point(c: &Constraint, id: PointId) -> bool {
    match c {
        Constraint::Coincident(p1, p2) | Constraint::Distance(p1, p2, _) => *p1 == id || *p2 == id,
        Constraint::PointLineDistance(pt, _, _)
        | Constraint::PointOnCircle(pt, _)
        | Constraint::PointOnArc(pt, _) => *pt == id,
        Constraint::FixX(p, _) | Constraint::FixY(p, _) => *p == id,
        Constraint::TangentLineArc(_, _, shared) | Constraint::TangentArcArc(_, _, shared) => {
            *shared == id
        }
        Constraint::Horizontal(_)
        | Constraint::Vertical(_)
        | Constraint::Angle(_, _, _)
        | Constraint::Perpendicular(_, _)
        | Constraint::Parallel(_, _)
        | Constraint::EqualRadiusArcArc(_, _)
        | Constraint::EqualRadiusArcCircle(_, _)
        | Constraint::ArcLength(_, _)
        | Constraint::ConcentricArcArc(_, _)
        | Constraint::ConcentricArcCircle(_, _) => false,
    }
}

/// Check if a constraint references a specific line.
fn constraint_references_line(c: &Constraint, id: LineId) -> bool {
    match c {
        Constraint::Horizontal(l) | Constraint::Vertical(l) => *l == id,
        Constraint::PointLineDistance(_, l, _) => *l == id,
        Constraint::TangentLineArc(l, _, _) => *l == id,
        Constraint::Angle(l1, l2, _)
        | Constraint::Perpendicular(l1, l2)
        | Constraint::Parallel(l1, l2) => *l1 == id || *l2 == id,
        Constraint::Coincident(_, _)
        | Constraint::Distance(_, _, _)
        | Constraint::FixX(_, _)
        | Constraint::FixY(_, _)
        | Constraint::PointOnCircle(_, _)
        | Constraint::PointOnArc(_, _)
        | Constraint::TangentArcArc(_, _, _)
        | Constraint::EqualRadiusArcArc(_, _)
        | Constraint::EqualRadiusArcCircle(_, _)
        | Constraint::ArcLength(_, _)
        | Constraint::ConcentricArcArc(_, _)
        | Constraint::ConcentricArcCircle(_, _) => false,
    }
}

/// Check if a constraint references a specific circle.
fn constraint_references_circle(c: &Constraint, id: CircleId) -> bool {
    match c {
        Constraint::PointOnCircle(_, circ) => *circ == id,
        Constraint::EqualRadiusArcCircle(_, circ) | Constraint::ConcentricArcCircle(_, circ) => {
            *circ == id
        }
        Constraint::Coincident(_, _)
        | Constraint::Distance(_, _, _)
        | Constraint::PointLineDistance(_, _, _)
        | Constraint::FixX(_, _)
        | Constraint::FixY(_, _)
        | Constraint::Horizontal(_)
        | Constraint::Vertical(_)
        | Constraint::Angle(_, _, _)
        | Constraint::Perpendicular(_, _)
        | Constraint::Parallel(_, _)
        | Constraint::PointOnArc(_, _)
        | Constraint::TangentLineArc(_, _, _)
        | Constraint::TangentArcArc(_, _, _)
        | Constraint::EqualRadiusArcArc(_, _)
        | Constraint::ArcLength(_, _)
        | Constraint::ConcentricArcArc(_, _) => false,
    }
}

/// Check if a constraint references a specific arc.
fn constraint_references_arc(c: &Constraint, id: ArcId) -> bool {
    match c {
        Constraint::PointOnArc(_, arc) | Constraint::ArcLength(arc, _) => *arc == id,
        Constraint::TangentLineArc(_, arc, _) => *arc == id,
        Constraint::TangentArcArc(a1, a2, _)
        | Constraint::EqualRadiusArcArc(a1, a2)
        | Constraint::ConcentricArcArc(a1, a2) => *a1 == id || *a2 == id,
        Constraint::EqualRadiusArcCircle(arc, _) | Constraint::ConcentricArcCircle(arc, _) => {
            *arc == id
        }
        Constraint::Coincident(_, _)
        | Constraint::Distance(_, _, _)
        | Constraint::PointLineDistance(_, _, _)
        | Constraint::FixX(_, _)
        | Constraint::FixY(_, _)
        | Constraint::Horizontal(_)
        | Constraint::Vertical(_)
        | Constraint::Angle(_, _, _)
        | Constraint::Perpendicular(_, _)
        | Constraint::Parallel(_, _)
        | Constraint::PointOnCircle(_, _) => false,
    }
}

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::expect_used)]
mod tests;
