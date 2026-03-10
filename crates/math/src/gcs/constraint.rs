//! Constraint types with analytic residuals and Jacobians.

use std::collections::HashMap;

use super::entity::{CircleId, Handle, LineId, ParamRef, PointData, PointId};

/// Internal storage for a constraint.
#[derive(Debug, Clone)]
pub struct ConstraintEntry {
    /// The constraint.
    pub constraint: Constraint,
}

/// Handle to a constraint in the GCS.
pub type ConstraintId = Handle<ConstraintEntry>;

/// A geometric constraint in the GCS.
///
/// Each variant knows how to compute its residual(s) and analytic
/// Jacobian entries. Constraints reference entities by handle, so they
/// are validated at add-time and invalidated if an entity is removed.
#[derive(Debug, Clone)]
pub enum Constraint {
    /// Two points must coincide. Produces 2 residuals: `[p1.x - p2.x, p1.y - p2.y]`.
    Coincident(PointId, PointId),

    /// Distance between two points must equal `d`. Produces 1 residual
    /// using the squared form: `dx² + dy² - d²`, normalized by `max(1, 2d)`.
    Distance(PointId, PointId, f64),

    /// Signed distance from a point to a line must equal `d`. Produces 1 residual.
    /// When `d = 0`, this is "point on line".
    PointLineDistance(PointId, LineId, f64),

    /// Fix the X coordinate of a point. Produces 1 residual: `p.x - value`.
    FixX(PointId, f64),

    /// Fix the Y coordinate of a point. Produces 1 residual: `p.y - value`.
    FixY(PointId, f64),

    /// A line must be horizontal. Produces 1 residual: `p2.y - p1.y`.
    Horizontal(LineId),

    /// A line must be vertical. Produces 1 residual: `p2.x - p1.x`.
    Vertical(LineId),

    /// Angle between two lines. Uses the cross-cos form to avoid atan2
    /// discontinuities: `cross·cos(θ) - dot·sin(θ)`.
    Angle(LineId, LineId, f64),

    /// Two lines must be perpendicular. Produces 1 residual: `dot(d1, d2)`.
    Perpendicular(LineId, LineId),

    /// Two lines must be parallel. Produces 1 residual: `cross(d1, d2)`.
    Parallel(LineId, LineId),
}

/// Entity data snapshot used during residual/Jacobian evaluation.
/// Avoids borrowing the GcsSystem during computation.
///
/// Uses `HashMap` for O(1) point and line lookups instead of linear search.
#[derive(Debug)]
pub struct EntitySnapshot {
    /// Point positions keyed by handle.
    pub points: HashMap<PointId, (f64, f64)>,
    /// Line endpoint pairs keyed by handle.
    pub lines: HashMap<LineId, (PointId, PointId)>,
    /// Circle definitions keyed by handle.
    /// Not yet used in PR1 constraints but needed for PR2.
    #[allow(dead_code)]
    pub circles: HashMap<CircleId, (PointId, f64)>,
}

impl EntitySnapshot {
    /// Look up a point's (x, y) by handle.
    fn point(&self, id: PointId) -> (f64, f64) {
        self.points.get(&id).copied().unwrap_or((0.0, 0.0))
    }

    /// Look up a line's endpoint IDs.
    fn line(&self, id: LineId) -> (PointId, PointId) {
        self.lines.get(&id).copied().unwrap_or_else(|| {
            // Should never happen if constraints are validated
            let dummy = PointId::dummy();
            (dummy, dummy)
        })
    }
}

impl Handle<PointData> {
    /// Create a dummy handle (only for unreachable fallback paths).
    fn dummy() -> Self {
        Self {
            index: u32::MAX,
            generation: u32::MAX,
            _marker: std::marker::PhantomData,
        }
    }
}

/// Number of residual equations a constraint produces.
pub const fn residual_count(c: &Constraint) -> usize {
    match c {
        Constraint::Coincident(_, _) => 2,
        Constraint::Distance(_, _, _)
        | Constraint::PointLineDistance(_, _, _)
        | Constraint::FixX(_, _)
        | Constraint::FixY(_, _)
        | Constraint::Horizontal(_)
        | Constraint::Vertical(_)
        | Constraint::Angle(_, _, _)
        | Constraint::Perpendicular(_, _)
        | Constraint::Parallel(_, _) => 1,
    }
}

/// Compute residuals for a constraint, appending to `out`.
pub fn eval_residuals(c: &Constraint, snap: &EntitySnapshot, out: &mut Vec<f64>) {
    match c {
        Constraint::Coincident(p1, p2) => {
            let (x1, y1) = snap.point(*p1);
            let (x2, y2) = snap.point(*p2);
            out.push(x1 - x2);
            out.push(y1 - y2);
        }
        Constraint::Distance(p1, p2, d) => {
            let (x1, y1) = snap.point(*p1);
            let (x2, y2) = snap.point(*p2);
            let dx = x1 - x2;
            let dy = y1 - y2;
            let scale = 1.0_f64.max(2.0 * d);
            out.push((dx * dx + dy * dy - d * d) / scale);
        }
        Constraint::PointLineDistance(pt, line, d) => {
            let (px, py) = snap.point(*pt);
            let (lp1, lp2) = snap.line(*line);
            let (x1, y1) = snap.point(lp1);
            let (x2, y2) = snap.point(lp2);
            let ldx = x2 - x1;
            let ldy = y2 - y1;
            let len = ldx.hypot(ldy);
            if len < 1e-300 {
                out.push(0.0);
                return;
            }
            // Signed distance: cross(line_dir, pt - p1) / |line_dir| - d
            let cross = ldx * (py - y1) - ldy * (px - x1);
            out.push(cross / len - d);
        }
        Constraint::FixX(p, v) => {
            let (x, _) = snap.point(*p);
            out.push(x - v);
        }
        Constraint::FixY(p, v) => {
            let (_, y) = snap.point(*p);
            out.push(y - v);
        }
        Constraint::Horizontal(line) => {
            let (p1, p2) = snap.line(*line);
            let (_, y1) = snap.point(p1);
            let (_, y2) = snap.point(p2);
            out.push(y2 - y1);
        }
        Constraint::Vertical(line) => {
            let (p1, p2) = snap.line(*line);
            let (x1, _) = snap.point(p1);
            let (x2, _) = snap.point(p2);
            out.push(x2 - x1);
        }
        Constraint::Angle(l1, l2, theta) => {
            let (l1p1, l1p2) = snap.line(*l1);
            let (l2p1, l2p2) = snap.line(*l2);
            let (x1a, y1a) = snap.point(l1p1);
            let (x1b, y1b) = snap.point(l1p2);
            let (x2a, y2a) = snap.point(l2p1);
            let (x2b, y2b) = snap.point(l2p2);
            let d1x = x1b - x1a;
            let d1y = y1b - y1a;
            let d2x = x2b - x2a;
            let d2y = y2b - y2a;
            let cross = d1x * d2y - d1y * d2x;
            let dot = d1x * d2x + d1y * d2y;
            let (sin_t, cos_t) = theta.sin_cos();
            // cross·cos(θ) - dot·sin(θ) = |d1||d2| sin(α - θ)
            out.push(cross * cos_t - dot * sin_t);
        }
        Constraint::Perpendicular(l1, l2) => {
            let (l1p1, l1p2) = snap.line(*l1);
            let (l2p1, l2p2) = snap.line(*l2);
            let (x1a, y1a) = snap.point(l1p1);
            let (x1b, y1b) = snap.point(l1p2);
            let (x2a, y2a) = snap.point(l2p1);
            let (x2b, y2b) = snap.point(l2p2);
            let dot = (x1b - x1a) * (x2b - x2a) + (y1b - y1a) * (y2b - y2a);
            out.push(dot);
        }
        Constraint::Parallel(l1, l2) => {
            let (l1p1, l1p2) = snap.line(*l1);
            let (l2p1, l2p2) = snap.line(*l2);
            let (x1a, y1a) = snap.point(l1p1);
            let (x1b, y1b) = snap.point(l1p2);
            let (x2a, y2a) = snap.point(l2p1);
            let (x2b, y2b) = snap.point(l2p2);
            let cross = (x1b - x1a) * (y2b - y2a) - (y1b - y1a) * (x2b - x2a);
            out.push(cross);
        }
    }
}

/// Writer that places Jacobian entries into a row-major dense matrix.
pub struct JacobianWriter<'a> {
    /// Row-major Jacobian, m rows × n cols.
    pub data: &'a mut [f64],
    /// Number of columns (= number of parameters).
    pub ncols: usize,
    /// Map from `ParamRef` to column index.
    pub param_index: &'a HashMap<ParamRef, usize>,
}

impl JacobianWriter<'_> {
    /// Write a value to row `row`, parameter `param_ref`.
    fn set(&mut self, row: usize, pr: ParamRef, val: f64) {
        if let Some(&col) = self.param_index.get(&pr) {
            self.data[row * self.ncols + col] = val;
        }
    }

    /// Add a value (accumulate) to row `row`, parameter `param_ref`.
    fn add(&mut self, row: usize, pr: ParamRef, val: f64) {
        if let Some(&col) = self.param_index.get(&pr) {
            self.data[row * self.ncols + col] += val;
        }
    }
}

/// Write analytic Jacobian entries for a constraint.
///
/// `row_offset` is the first residual row for this constraint.
#[allow(clippy::too_many_lines)]
pub fn eval_jacobian(
    c: &Constraint,
    snap: &EntitySnapshot,
    jw: &mut JacobianWriter<'_>,
    row_offset: usize,
) {
    match c {
        Constraint::Coincident(p1, p2) => {
            // r0 = x1 - x2  → ∂r0/∂x1 = 1, ∂r0/∂x2 = -1
            // r1 = y1 - y2  → ∂r1/∂y1 = 1, ∂r1/∂y2 = -1
            jw.set(row_offset, ParamRef::PointX(*p1), 1.0);
            jw.set(row_offset, ParamRef::PointX(*p2), -1.0);
            jw.set(row_offset + 1, ParamRef::PointY(*p1), 1.0);
            jw.set(row_offset + 1, ParamRef::PointY(*p2), -1.0);
        }
        Constraint::Distance(p1, p2, d) => {
            // r = (dx² + dy² - d²) / scale, where scale = max(1, 2d)
            // ∂r/∂x1 = 2dx/scale, ∂r/∂y1 = 2dy/scale
            let (x1, y1) = snap.point(*p1);
            let (x2, y2) = snap.point(*p2);
            let dx = x1 - x2;
            let dy = y1 - y2;
            let scale = 1.0_f64.max(2.0 * d);
            let gx = 2.0 * dx / scale;
            let gy = 2.0 * dy / scale;
            jw.set(row_offset, ParamRef::PointX(*p1), gx);
            jw.set(row_offset, ParamRef::PointY(*p1), gy);
            jw.set(row_offset, ParamRef::PointX(*p2), -gx);
            jw.set(row_offset, ParamRef::PointY(*p2), -gy);
        }
        Constraint::PointLineDistance(pt, line, _d) => {
            // r = cross(line_dir, pt - p1) / |line_dir| - d
            // where cross = ldx*(py-y1) - ldy*(px-x1), L = |line_dir|
            // This has 6 nonzero partials: px, py, x1, y1, x2, y2
            let (px, py) = snap.point(*pt);
            let (lp1, lp2) = snap.line(*line);
            let (x1, y1) = snap.point(lp1);
            let (x2, y2) = snap.point(lp2);
            let ldx = x2 - x1;
            let ldy = y2 - y1;
            let len = ldx.hypot(ldy);
            if len < 1e-300 {
                return;
            }
            let inv_l = 1.0 / len;
            let cross = ldx * (py - y1) - ldy * (px - x1);

            // ∂r/∂px = -ldy / L
            jw.set(row_offset, ParamRef::PointX(*pt), -ldy * inv_l);
            // ∂r/∂py = ldx / L
            jw.set(row_offset, ParamRef::PointY(*pt), ldx * inv_l);

            // For the line endpoints, we need the full derivative.
            // Let f = cross / L where cross = ldx*(py-y1) - ldy*(px-x1)
            // Use the quotient rule: ∂f/∂var = (L * ∂cross/∂var - cross * ∂L/∂var) / L²
            let l_sq = len * len;
            let inv_l_sq = 1.0 / l_sq;

            // ∂cross/∂x1 = -ldy + (partial due to ldx change) = ...
            // Actually, x1 affects both ldx (through ldx = x2-x1) and (px-x1).
            // ∂cross/∂x1 = ∂/∂x1 [ (x2-x1)(py-y1) - (y2-y1)(px-x1) ]
            //             = -(py-y1) + (y2-y1) = -(py - y2)
            let dc_dx1 = -(py - y2);
            let dl_dx1 = -ldx * inv_l;
            let dr_dx1 = (len * dc_dx1 - cross * dl_dx1) * inv_l_sq;

            // ∂cross/∂y1 = -(x2-x1) + ... Wait, let me redo.
            // cross = (x2-x1)(py-y1) - (y2-y1)(px-x1)
            // ∂cross/∂y1 = (x2-x1)*(-1) - 0 = -(x2-x1) = -ldx
            // But y1 also doesn't appear in (y2-y1) for the second term? Wait, y1 does:
            // (y2-y1) → ∂/∂y1 = -1, so second term: -(-1)*(px-x1) = (px-x1)
            // ∂cross/∂y1 = -ldx + (px - x1)
            // Actually: cross = ldx*(py-y1) - ldy*(px-x1)
            // ∂cross/∂y1 = ldx*(-1) - (-1)*(px-x1) = -ldx + (px-x1)
            let dc_dy1 = -ldx + (px - x1);
            let dl_dy1 = -ldy * inv_l;
            let dr_dy1 = (len * dc_dy1 - cross * dl_dy1) * inv_l_sq;

            // ∂cross/∂x2: ldx = x2-x1, so ∂ldx/∂x2 = 1
            // ∂cross/∂x2 = 1*(py-y1) - 0 = (py-y1)
            let dc_dx2 = py - y1;
            let dl_dx2 = ldx * inv_l;
            let dr_dx2 = (len * dc_dx2 - cross * dl_dx2) * inv_l_sq;

            // ∂cross/∂y2: ldy = y2-y1, so ∂ldy/∂y2 = 1
            // ∂cross/∂y2 = 0 - 1*(px-x1) = -(px-x1)
            let dc_dy2 = -(px - x1);
            let dl_dy2 = ldy * inv_l;
            let dr_dy2 = (len * dc_dy2 - cross * dl_dy2) * inv_l_sq;

            jw.add(row_offset, ParamRef::PointX(lp1), dr_dx1);
            jw.add(row_offset, ParamRef::PointY(lp1), dr_dy1);
            jw.add(row_offset, ParamRef::PointX(lp2), dr_dx2);
            jw.add(row_offset, ParamRef::PointY(lp2), dr_dy2);
        }
        Constraint::FixX(p, _) => {
            jw.set(row_offset, ParamRef::PointX(*p), 1.0);
        }
        Constraint::FixY(p, _) => {
            jw.set(row_offset, ParamRef::PointY(*p), 1.0);
        }
        Constraint::Horizontal(line) => {
            let (p1, p2) = snap.line(*line);
            // r = y2 - y1 → ∂r/∂y2 = 1, ∂r/∂y1 = -1
            jw.set(row_offset, ParamRef::PointY(p2), 1.0);
            jw.set(row_offset, ParamRef::PointY(p1), -1.0);
        }
        Constraint::Vertical(line) => {
            let (p1, p2) = snap.line(*line);
            // r = x2 - x1 → ∂r/∂x2 = 1, ∂r/∂x1 = -1
            jw.set(row_offset, ParamRef::PointX(p2), 1.0);
            jw.set(row_offset, ParamRef::PointX(p1), -1.0);
        }
        Constraint::Angle(l1, l2, theta) => {
            // r = cross·cos(θ) - dot·sin(θ)
            // where cross = d1x*d2y - d1y*d2x, dot = d1x*d2x + d1y*d2y
            let (l1p1, l1p2) = snap.line(*l1);
            let (l2p1, l2p2) = snap.line(*l2);
            let (x1a, y1a) = snap.point(l1p1);
            let (x1b, y1b) = snap.point(l1p2);
            let (x2a, y2a) = snap.point(l2p1);
            let (x2b, y2b) = snap.point(l2p2);
            let d1x = x1b - x1a;
            let d1y = y1b - y1a;
            let d2x = x2b - x2a;
            let d2y = y2b - y2a;
            let (sin_t, cos_t) = theta.sin_cos();

            // ∂cross/∂d1x = d2y, ∂cross/∂d1y = -d2x, ∂cross/∂d2x = -d1y, ∂cross/∂d2y = d1x
            // ∂dot/∂d1x = d2x, ∂dot/∂d1y = d2y, ∂dot/∂d2x = d1x, ∂dot/∂d2y = d1y
            // ∂r/∂d1x = d2y*cos_t - d2x*sin_t
            // ∂r/∂d1y = -d2x*cos_t - d2y*sin_t
            // ∂r/∂d2x = -d1y*cos_t - d1x*sin_t
            // ∂r/∂d2y = d1x*cos_t - d1y*sin_t
            let dr_d1x = d2y * cos_t - d2x * sin_t;
            let dr_d1y = -d2x * cos_t - d2y * sin_t;
            let dr_d2x = -d1y * cos_t - d1x * sin_t;
            let dr_d2y = d1x * cos_t - d1y * sin_t;

            // d1x = x1b - x1a → ∂/∂x1a = -1, ∂/∂x1b = 1
            jw.set(row_offset, ParamRef::PointX(l1p1), -dr_d1x);
            jw.set(row_offset, ParamRef::PointX(l1p2), dr_d1x);
            jw.set(row_offset, ParamRef::PointY(l1p1), -dr_d1y);
            jw.set(row_offset, ParamRef::PointY(l1p2), dr_d1y);
            // Handle shared points between l1 and l2 with add() not set()
            jw.add(row_offset, ParamRef::PointX(l2p1), -dr_d2x);
            jw.add(row_offset, ParamRef::PointX(l2p2), dr_d2x);
            jw.add(row_offset, ParamRef::PointY(l2p1), -dr_d2y);
            jw.add(row_offset, ParamRef::PointY(l2p2), dr_d2y);
        }
        Constraint::Perpendicular(l1, l2) => {
            // r = dot(d1, d2) = d1x*d2x + d1y*d2y
            let (l1p1, l1p2) = snap.line(*l1);
            let (l2p1, l2p2) = snap.line(*l2);
            let (x1a, y1a) = snap.point(l1p1);
            let (x1b, y1b) = snap.point(l1p2);
            let (x2a, y2a) = snap.point(l2p1);
            let (x2b, y2b) = snap.point(l2p2);
            let d2x = x2b - x2a;
            let d2y = y2b - y2a;
            let d1x = x1b - x1a;
            let d1y = y1b - y1a;

            // ∂r/∂x1a = -d2x, ∂r/∂x1b = d2x, ∂r/∂y1a = -d2y, ∂r/∂y1b = d2y
            // ∂r/∂x2a = -d1x, ∂r/∂x2b = d1x, ∂r/∂y2a = -d1y, ∂r/∂y2b = d1y
            jw.set(row_offset, ParamRef::PointX(l1p1), -d2x);
            jw.set(row_offset, ParamRef::PointX(l1p2), d2x);
            jw.set(row_offset, ParamRef::PointY(l1p1), -d2y);
            jw.set(row_offset, ParamRef::PointY(l1p2), d2y);
            jw.add(row_offset, ParamRef::PointX(l2p1), -d1x);
            jw.add(row_offset, ParamRef::PointX(l2p2), d1x);
            jw.add(row_offset, ParamRef::PointY(l2p1), -d1y);
            jw.add(row_offset, ParamRef::PointY(l2p2), d1y);
        }
        Constraint::Parallel(l1, l2) => {
            // r = cross(d1, d2) = d1x*d2y - d1y*d2x
            let (l1p1, l1p2) = snap.line(*l1);
            let (l2p1, l2p2) = snap.line(*l2);
            let (x1a, y1a) = snap.point(l1p1);
            let (x1b, y1b) = snap.point(l1p2);
            let (x2a, y2a) = snap.point(l2p1);
            let (x2b, y2b) = snap.point(l2p2);
            let d1x = x1b - x1a;
            let d1y = y1b - y1a;
            let d2x = x2b - x2a;
            let d2y = y2b - y2a;

            // ∂r/∂d1x = d2y, ∂r/∂d1y = -d2x, ∂r/∂d2x = -d1y, ∂r/∂d2y = d1x
            jw.set(row_offset, ParamRef::PointX(l1p1), -d2y);
            jw.set(row_offset, ParamRef::PointX(l1p2), d2y);
            jw.set(row_offset, ParamRef::PointY(l1p1), d2x);
            jw.set(row_offset, ParamRef::PointY(l1p2), -d2x);
            jw.add(row_offset, ParamRef::PointX(l2p1), d1y);
            jw.add(row_offset, ParamRef::PointX(l2p2), -d1y);
            jw.add(row_offset, ParamRef::PointY(l2p1), -d1x);
            jw.add(row_offset, ParamRef::PointY(l2p2), d1x);
        }
    }
}

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::expect_used)]
mod tests {
    use super::*;

    /// Build a simple snapshot with two points at given positions.
    fn two_point_snap(x1: f64, y1: f64, x2: f64, y2: f64) -> (PointId, PointId, EntitySnapshot) {
        use super::super::entity::GenArena;
        use super::super::entity::PointData;
        let mut arena = GenArena::new();
        let p1 = arena.insert(PointData {
            x: x1,
            y: y1,
            fixed: false,
        });
        let p2 = arena.insert(PointData {
            x: x2,
            y: y2,
            fixed: false,
        });
        let snap = EntitySnapshot {
            points: [(p1, (x1, y1)), (p2, (x2, y2))].into_iter().collect(),
            lines: HashMap::new(),
            circles: HashMap::new(),
        };
        (p1, p2, snap)
    }

    #[test]
    fn coincident_at_solution() {
        let (p1, p2, snap) = two_point_snap(3.0, 4.0, 3.0, 4.0);
        let c = Constraint::Coincident(p1, p2);
        let mut r = Vec::new();
        eval_residuals(&c, &snap, &mut r);
        assert_eq!(r.len(), 2);
        assert!((r[0]).abs() < 1e-15);
        assert!((r[1]).abs() < 1e-15);
    }

    #[test]
    fn coincident_away_from_solution() {
        let (p1, p2, snap) = two_point_snap(0.0, 0.0, 1.0, 2.0);
        let c = Constraint::Coincident(p1, p2);
        let mut r = Vec::new();
        eval_residuals(&c, &snap, &mut r);
        assert!((r[0] - (-1.0)).abs() < 1e-15);
        assert!((r[1] - (-2.0)).abs() < 1e-15);
    }

    #[test]
    fn distance_at_solution() {
        let (p1, p2, snap) = two_point_snap(0.0, 0.0, 3.0, 4.0);
        let c = Constraint::Distance(p1, p2, 5.0);
        let mut r = Vec::new();
        eval_residuals(&c, &snap, &mut r);
        assert!(r[0].abs() < 1e-14, "residual = {}", r[0]);
    }

    #[test]
    fn fix_x_residual() {
        let (p1, _, snap) = two_point_snap(7.0, 3.0, 0.0, 0.0);
        let c = Constraint::FixX(p1, 5.0);
        let mut r = Vec::new();
        eval_residuals(&c, &snap, &mut r);
        assert!((r[0] - 2.0).abs() < 1e-15);
    }

    /// Verify analytic Jacobian against finite differences for a constraint.
    fn check_jacobian_fd(c: &Constraint, snap: &EntitySnapshot, params: &[ParamRef]) {
        let param_index: HashMap<ParamRef, usize> =
            params.iter().enumerate().map(|(i, p)| (*p, i)).collect();
        let n = params.len();
        let m = residual_count(c);

        // Analytic Jacobian
        let mut jac = vec![0.0; m * n];
        let mut jw = JacobianWriter {
            data: &mut jac,
            ncols: n,
            param_index: &param_index,
        };
        eval_jacobian(c, snap, &mut jw, 0);

        // Finite-difference Jacobian
        let eps = 1e-7;
        let mut r0 = Vec::new();
        eval_residuals(c, snap, &mut r0);

        for (col, pr) in params.iter().enumerate() {
            let mut perturbed_points = snap.points.clone();
            match pr {
                ParamRef::PointX(pid) => {
                    if let Some(xy) = perturbed_points.get_mut(pid) {
                        xy.0 += eps;
                    }
                }
                ParamRef::PointY(pid) => {
                    if let Some(xy) = perturbed_points.get_mut(pid) {
                        xy.1 += eps;
                    }
                }
                ParamRef::CircleRadius(_) => {}
            }
            let perturbed_snap = EntitySnapshot {
                points: perturbed_points,
                lines: snap.lines.clone(),
                circles: snap.circles.clone(),
            };
            let mut r1 = Vec::new();
            eval_residuals(c, &perturbed_snap, &mut r1);

            for row in 0..m {
                let fd = (r1[row] - r0[row]) / eps;
                let analytic = jac[row * n + col];
                let err = (fd - analytic).abs();
                let scale = 1.0_f64.max(analytic.abs());
                assert!(
                    err < 1e-5 * scale + 1e-8,
                    "Jacobian mismatch at ({row},{col}): analytic={analytic}, fd={fd}, err={err}"
                );
            }
        }
    }

    #[test]
    fn jacobian_coincident() {
        let (p1, p2, snap) = two_point_snap(1.0, 2.0, 3.0, 5.0);
        let c = Constraint::Coincident(p1, p2);
        let params = vec![
            ParamRef::PointX(p1),
            ParamRef::PointY(p1),
            ParamRef::PointX(p2),
            ParamRef::PointY(p2),
        ];
        check_jacobian_fd(&c, &snap, &params);
    }

    #[test]
    fn jacobian_distance() {
        let (p1, p2, snap) = two_point_snap(1.0, 2.0, 4.0, 6.0);
        let c = Constraint::Distance(p1, p2, 5.0);
        let params = vec![
            ParamRef::PointX(p1),
            ParamRef::PointY(p1),
            ParamRef::PointX(p2),
            ParamRef::PointY(p2),
        ];
        check_jacobian_fd(&c, &snap, &params);
    }

    #[test]
    fn jacobian_fix_xy() {
        let (p1, _, snap) = two_point_snap(3.0, 7.0, 0.0, 0.0);
        check_jacobian_fd(&Constraint::FixX(p1, 5.0), &snap, &[ParamRef::PointX(p1)]);
        check_jacobian_fd(&Constraint::FixY(p1, 2.0), &snap, &[ParamRef::PointY(p1)]);
    }

    #[test]
    fn jacobian_horizontal_vertical() {
        use super::super::entity::GenArena;
        use super::super::entity::{LineData, PointData};
        let mut pts = GenArena::new();
        let p1 = pts.insert(PointData {
            x: 1.0,
            y: 3.0,
            fixed: false,
        });
        let p2 = pts.insert(PointData {
            x: 5.0,
            y: 7.0,
            fixed: false,
        });
        let mut lines = GenArena::new();
        let l = lines.insert(LineData { p1, p2 });

        let snap = EntitySnapshot {
            points: [(p1, (1.0, 3.0)), (p2, (5.0, 7.0))].into_iter().collect(),
            lines: std::iter::once((l, (p1, p2))).collect(),
            circles: HashMap::new(),
        };
        let params = vec![
            ParamRef::PointX(p1),
            ParamRef::PointY(p1),
            ParamRef::PointX(p2),
            ParamRef::PointY(p2),
        ];
        check_jacobian_fd(&Constraint::Horizontal(l), &snap, &params);
        check_jacobian_fd(&Constraint::Vertical(l), &snap, &params);
    }

    #[test]
    fn jacobian_parallel_perpendicular() {
        use super::super::entity::GenArena;
        use super::super::entity::{LineData, PointData};
        let mut pts = GenArena::new();
        let p1 = pts.insert(PointData {
            x: 0.0,
            y: 0.0,
            fixed: false,
        });
        let p2 = pts.insert(PointData {
            x: 3.0,
            y: 1.0,
            fixed: false,
        });
        let p3 = pts.insert(PointData {
            x: 1.0,
            y: 2.0,
            fixed: false,
        });
        let p4 = pts.insert(PointData {
            x: 4.0,
            y: 5.0,
            fixed: false,
        });
        let mut lines = GenArena::new();
        let l1 = lines.insert(LineData { p1, p2 });
        let l2 = lines.insert(LineData { p1: p3, p2: p4 });

        let snap = EntitySnapshot {
            points: [
                (p1, (0.0, 0.0)),
                (p2, (3.0, 1.0)),
                (p3, (1.0, 2.0)),
                (p4, (4.0, 5.0)),
            ]
            .into_iter()
            .collect(),
            lines: [(l1, (p1, p2)), (l2, (p3, p4))].into_iter().collect(),
            circles: HashMap::new(),
        };
        let params = vec![
            ParamRef::PointX(p1),
            ParamRef::PointY(p1),
            ParamRef::PointX(p2),
            ParamRef::PointY(p2),
            ParamRef::PointX(p3),
            ParamRef::PointY(p3),
            ParamRef::PointX(p4),
            ParamRef::PointY(p4),
        ];
        check_jacobian_fd(&Constraint::Parallel(l1, l2), &snap, &params);
        check_jacobian_fd(&Constraint::Perpendicular(l1, l2), &snap, &params);
    }

    #[test]
    fn jacobian_angle() {
        use super::super::entity::GenArena;
        use super::super::entity::{LineData, PointData};
        let mut pts = GenArena::new();
        let p1 = pts.insert(PointData {
            x: 0.0,
            y: 0.0,
            fixed: false,
        });
        let p2 = pts.insert(PointData {
            x: 3.0,
            y: 1.0,
            fixed: false,
        });
        let p3 = pts.insert(PointData {
            x: 1.0,
            y: 0.0,
            fixed: false,
        });
        let p4 = pts.insert(PointData {
            x: 2.0,
            y: 4.0,
            fixed: false,
        });
        let mut lines = GenArena::new();
        let l1 = lines.insert(LineData { p1, p2 });
        let l2 = lines.insert(LineData { p1: p3, p2: p4 });

        let snap = EntitySnapshot {
            points: [
                (p1, (0.0, 0.0)),
                (p2, (3.0, 1.0)),
                (p3, (1.0, 0.0)),
                (p4, (2.0, 4.0)),
            ]
            .into_iter()
            .collect(),
            lines: [(l1, (p1, p2)), (l2, (p3, p4))].into_iter().collect(),
            circles: HashMap::new(),
        };
        let params = vec![
            ParamRef::PointX(p1),
            ParamRef::PointY(p1),
            ParamRef::PointX(p2),
            ParamRef::PointY(p2),
            ParamRef::PointX(p3),
            ParamRef::PointY(p3),
            ParamRef::PointX(p4),
            ParamRef::PointY(p4),
        ];
        check_jacobian_fd(&Constraint::Angle(l1, l2, 0.5), &snap, &params);
    }

    #[test]
    fn jacobian_point_line_distance() {
        use super::super::entity::GenArena;
        use super::super::entity::{LineData, PointData};
        let mut pts = GenArena::new();
        let pt = pts.insert(PointData {
            x: 2.0,
            y: 3.0,
            fixed: false,
        });
        let lp1 = pts.insert(PointData {
            x: 0.0,
            y: 0.0,
            fixed: false,
        });
        let lp2 = pts.insert(PointData {
            x: 4.0,
            y: 1.0,
            fixed: false,
        });
        let mut lines = GenArena::new();
        let l = lines.insert(LineData { p1: lp1, p2: lp2 });

        let snap = EntitySnapshot {
            points: [(pt, (2.0, 3.0)), (lp1, (0.0, 0.0)), (lp2, (4.0, 1.0))]
                .into_iter()
                .collect(),
            lines: std::iter::once((l, (lp1, lp2))).collect(),
            circles: HashMap::new(),
        };
        let params = vec![
            ParamRef::PointX(pt),
            ParamRef::PointY(pt),
            ParamRef::PointX(lp1),
            ParamRef::PointY(lp1),
            ParamRef::PointX(lp2),
            ParamRef::PointY(lp2),
        ];
        check_jacobian_fd(&Constraint::PointLineDistance(pt, l, 1.5), &snap, &params);
    }
}
