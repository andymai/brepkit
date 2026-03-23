//! Constraint types with analytic residuals and Jacobians.

use std::collections::HashMap;

use super::entity::{ArcId, CircleId, Handle, LineId, ParamRef, PointData, PointId};

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

    /// Point must lie on a circle. Produces 1 residual: `dist(pt, center) - radius`.
    PointOnCircle(PointId, CircleId),

    /// Point must lie on an arc's circle. Produces 1 residual:
    /// `dist(pt, center) - dist(start, center)`.
    PointOnArc(PointId, ArcId),

    /// A line must be tangent to an arc at a shared point.
    /// Produces 1 residual: `cross(line_dir, arc_tangent)` at the shared point.
    TangentLineArc(LineId, ArcId, PointId),

    /// Two arcs must be tangent at a shared point.
    /// Produces 1 residual: `cross(tangent1, tangent2)` at the shared point.
    TangentArcArc(ArcId, ArcId, PointId),

    /// Two arcs must have equal radius. Produces 1 residual:
    /// `dist(c1, s1) - dist(c2, s2)`.
    EqualRadiusArcArc(ArcId, ArcId),

    /// Arc and circle must have equal radius. Produces 1 residual:
    /// `dist(center_arc, start_arc) - circle_radius`.
    EqualRadiusArcCircle(ArcId, CircleId),

    /// Arc length must equal a target value. Produces 1 residual:
    /// `r * theta - target` where `r = dist(center, start)` and
    /// `theta = |atan2(cross, dot)|` of start/end vectors from center.
    ArcLength(ArcId, f64),

    /// Two arcs must be concentric. Produces 2 residuals: `[c1.x - c2.x, c1.y - c2.y]`.
    ConcentricArcArc(ArcId, ArcId),

    /// Arc and circle must be concentric. Produces 2 residuals:
    /// `[c_arc.x - c_circ.x, c_arc.y - c_circ.y]`.
    ConcentricArcCircle(ArcId, CircleId),
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
    /// Circle definitions keyed by handle: `(center_id, radius)`.
    pub circles: HashMap<CircleId, (PointId, f64)>,
    /// Arc definitions keyed by handle: `(center, start, end)`.
    pub arcs: HashMap<ArcId, (PointId, PointId, PointId)>,
}

impl EntitySnapshot {
    /// Look up a point's (x, y) by handle.
    ///
    /// Returns `(NaN, NaN)` for stale/missing handles. NaN propagates
    /// through residual and Jacobian arithmetic, causing the solver to
    /// detect non-convergence rather than silently using wrong values.
    fn point(&self, id: PointId) -> (f64, f64) {
        self.points
            .get(&id)
            .copied()
            .unwrap_or((f64::NAN, f64::NAN))
    }

    /// Look up a line's endpoint IDs.
    ///
    /// Returns dummy IDs for stale handles. When those IDs are subsequently
    /// looked up via [`Self::point`], NaN is returned, poisoning downstream
    /// arithmetic.
    fn line(&self, id: LineId) -> (PointId, PointId) {
        self.lines.get(&id).copied().unwrap_or_else(|| {
            let dummy = PointId::dummy();
            (dummy, dummy)
        })
    }

    /// Look up a circle's `(center_id, radius)`.
    ///
    /// Returns a dummy center ID and NaN radius for stale handles.
    fn circle(&self, id: CircleId) -> (PointId, f64) {
        self.circles
            .get(&id)
            .copied()
            .unwrap_or_else(|| (PointId::dummy(), f64::NAN))
    }

    /// Look up an arc's `(center_id, start_id, end_id)`.
    ///
    /// Returns dummy IDs for stale handles.
    fn arc(&self, id: ArcId) -> (PointId, PointId, PointId) {
        self.arcs.get(&id).copied().unwrap_or_else(|| {
            let dummy = PointId::dummy();
            (dummy, dummy, dummy)
        })
    }
}

impl Handle<PointData> {
    /// Create a dummy handle for unreachable fallback paths.
    ///
    /// Looking up a dummy handle in [`EntitySnapshot::point`] returns NaN,
    /// which propagates through the solver to signal an error.
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
        Constraint::Coincident(_, _)
        | Constraint::ConcentricArcArc(_, _)
        | Constraint::ConcentricArcCircle(_, _) => 2,
        Constraint::Distance(_, _, _)
        | Constraint::PointLineDistance(_, _, _)
        | Constraint::FixX(_, _)
        | Constraint::FixY(_, _)
        | Constraint::Horizontal(_)
        | Constraint::Vertical(_)
        | Constraint::Angle(_, _, _)
        | Constraint::Perpendicular(_, _)
        | Constraint::Parallel(_, _)
        | Constraint::PointOnCircle(_, _)
        | Constraint::PointOnArc(_, _)
        | Constraint::TangentLineArc(_, _, _)
        | Constraint::TangentArcArc(_, _, _)
        | Constraint::EqualRadiusArcArc(_, _)
        | Constraint::EqualRadiusArcCircle(_, _)
        | Constraint::ArcLength(_, _) => 1,
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
        Constraint::PointOnCircle(pt, circ) => {
            let (px, py) = snap.point(*pt);
            let (center_id, radius) = snap.circle(*circ);
            let (cx, cy) = snap.point(center_id);
            let dx = px - cx;
            let dy = py - cy;
            let dist = dx.hypot(dy);
            out.push(dist - radius);
        }
        Constraint::PointOnArc(pt, arc) => {
            let (center_id, start_id, _end_id) = snap.arc(*arc);
            let (px, py) = snap.point(*pt);
            let (cx, cy) = snap.point(center_id);
            let (sx, sy) = snap.point(start_id);
            let dist_pt = (px - cx).hypot(py - cy);
            let dist_ref = (sx - cx).hypot(sy - cy);
            out.push(dist_pt - dist_ref);
        }
        Constraint::TangentLineArc(line, arc, shared) => {
            let (lp1, lp2) = snap.line(*line);
            let (center_id, _start_id, _end_id) = snap.arc(*arc);
            let (x1, y1) = snap.point(lp1);
            let (x2, y2) = snap.point(lp2);
            let (cx, cy) = snap.point(center_id);
            let (sx, sy) = snap.point(*shared);
            let line_dx = x2 - x1;
            let line_dy = y2 - y1;
            // Arc tangent at shared point: perpendicular to radius
            let arc_tx = -(sy - cy);
            let arc_ty = sx - cx;
            out.push(line_dx * arc_ty - line_dy * arc_tx);
        }
        Constraint::TangentArcArc(arc1, arc2, shared) => {
            let (c1_id, _s1, _e1) = snap.arc(*arc1);
            let (c2_id, _s2, _e2) = snap.arc(*arc2);
            let (c1x, c1y) = snap.point(c1_id);
            let (c2x, c2y) = snap.point(c2_id);
            let (sx, sy) = snap.point(*shared);
            let t1x = -(sy - c1y);
            let t1y = sx - c1x;
            let t2x = -(sy - c2y);
            let t2y = sx - c2x;
            out.push(t1x * t2y - t1y * t2x);
        }
        Constraint::EqualRadiusArcArc(arc1, arc2) => {
            let (c1_id, s1_id, _e1) = snap.arc(*arc1);
            let (c2_id, s2_id, _e2) = snap.arc(*arc2);
            let (c1x, c1y) = snap.point(c1_id);
            let (s1x, s1y) = snap.point(s1_id);
            let (c2x, c2y) = snap.point(c2_id);
            let (s2x, s2y) = snap.point(s2_id);
            let r1 = (s1x - c1x).hypot(s1y - c1y);
            let r2 = (s2x - c2x).hypot(s2y - c2y);
            out.push(r1 - r2);
        }
        Constraint::EqualRadiusArcCircle(arc, circ) => {
            let (center_id, start_id, _end_id) = snap.arc(*arc);
            let (cx, cy) = snap.point(center_id);
            let (sx, sy) = snap.point(start_id);
            let (_circ_center, radius) = snap.circle(*circ);
            let r_arc = (sx - cx).hypot(sy - cy);
            out.push(r_arc - radius);
        }
        Constraint::ArcLength(arc, target) => {
            let (center_id, start_id, end_id) = snap.arc(*arc);
            let (cx, cy) = snap.point(center_id);
            let (sx, sy) = snap.point(start_id);
            let (ex, ey) = snap.point(end_id);
            let dsx = sx - cx;
            let dsy = sy - cy;
            let dex = ex - cx;
            let dey = ey - cy;
            let r = dsx.hypot(dsy);
            let cross = dsx * dey - dsy * dex;
            let dot = dsx * dex + dsy * dey;
            let theta = cross.atan2(dot).abs();
            out.push(r * theta - target);
        }
        Constraint::ConcentricArcArc(arc1, arc2) => {
            let (c1_id, _s1, _e1) = snap.arc(*arc1);
            let (c2_id, _s2, _e2) = snap.arc(*arc2);
            let (c1x, c1y) = snap.point(c1_id);
            let (c2x, c2y) = snap.point(c2_id);
            out.push(c1x - c2x);
            out.push(c1y - c2y);
        }
        Constraint::ConcentricArcCircle(arc, circ) => {
            let (arc_center, _s, _e) = snap.arc(*arc);
            let (circ_center, _radius) = snap.circle(*circ);
            let (acx, acy) = snap.point(arc_center);
            let (ccx, ccy) = snap.point(circ_center);
            out.push(acx - ccx);
            out.push(acy - ccy);
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
        Constraint::PointOnCircle(pt, circ) => {
            // r = dist(pt, center) - radius
            // dist = sqrt((px-cx)^2 + (py-cy)^2)
            let (px, py) = snap.point(*pt);
            let (center_id, _radius) = snap.circle(*circ);
            let (cx, cy) = snap.point(center_id);
            let dx = px - cx;
            let dy = py - cy;
            let dist = dx.hypot(dy);
            if dist < 1e-300 {
                // Degenerate: point is at center, gradient undefined → 0
                jw.set(row_offset, ParamRef::CircleRadius(*circ), -1.0);
                return;
            }
            let inv_d = 1.0 / dist;
            let nx = dx * inv_d;
            let ny = dy * inv_d;
            jw.set(row_offset, ParamRef::PointX(*pt), nx);
            jw.set(row_offset, ParamRef::PointY(*pt), ny);
            jw.add(row_offset, ParamRef::PointX(center_id), -nx);
            jw.add(row_offset, ParamRef::PointY(center_id), -ny);
            jw.add(row_offset, ParamRef::CircleRadius(*circ), -1.0);
        }
        Constraint::PointOnArc(pt, arc) => {
            // r = dist(pt, center) - dist(start, center)
            let (center_id, start_id, _end_id) = snap.arc(*arc);
            let (px, py) = snap.point(*pt);
            let (cx, cy) = snap.point(center_id);
            let (sx, sy) = snap.point(start_id);
            let dpx = px - cx;
            let dpy = py - cy;
            let dsx = sx - cx;
            let dsy = sy - cy;
            let dist_pt = dpx.hypot(dpy);
            let dist_ref = dsx.hypot(dsy);
            // Partials for dist_pt w.r.t. px, py, cx, cy
            let (nx_pt, ny_pt) = if dist_pt < 1e-300 {
                (0.0, 0.0)
            } else {
                (dpx / dist_pt, dpy / dist_pt)
            };
            // Partials for dist_ref w.r.t. sx, sy, cx, cy
            let (nx_ref, ny_ref) = if dist_ref < 1e-300 {
                (0.0, 0.0)
            } else {
                (dsx / dist_ref, dsy / dist_ref)
            };
            // ∂r/∂px = nx_pt, ∂r/∂py = ny_pt
            jw.set(row_offset, ParamRef::PointX(*pt), nx_pt);
            jw.set(row_offset, ParamRef::PointY(*pt), ny_pt);
            // ∂r/∂sx = -nx_ref, ∂r/∂sy = -ny_ref
            jw.set(row_offset, ParamRef::PointX(start_id), -nx_ref);
            jw.set(row_offset, ParamRef::PointY(start_id), -ny_ref);
            // ∂r/∂cx = -nx_pt + nx_ref, ∂r/∂cy = -ny_pt + ny_ref
            jw.add(row_offset, ParamRef::PointX(center_id), -nx_pt + nx_ref);
            jw.add(row_offset, ParamRef::PointY(center_id), -ny_pt + ny_ref);
        }
        Constraint::TangentLineArc(line, arc, shared) => {
            // r = line_dx * arc_ty - line_dy * arc_tx
            // where line_dx = x2-x1, line_dy = y2-y1
            // arc_tx = -(sy - cy), arc_ty = sx - cx
            let (lp1, lp2) = snap.line(*line);
            let (center_id, _start_id, _end_id) = snap.arc(*arc);
            let (x1, y1) = snap.point(lp1);
            let (x2, y2) = snap.point(lp2);
            let (cx, cy) = snap.point(center_id);
            let (sx, sy) = snap.point(*shared);
            let line_dx = x2 - x1;
            let line_dy = y2 - y1;
            let arc_tx = -(sy - cy);
            let arc_ty = sx - cx;

            // r = line_dx * arc_ty - line_dy * arc_tx
            //   = (x2-x1)*(sx-cx) - (y2-y1)*(-(sy-cy))
            //   = (x2-x1)*(sx-cx) + (y2-y1)*(sy-cy)

            // ∂r/∂x1 = -arc_ty = -(sx-cx)
            jw.set(row_offset, ParamRef::PointX(lp1), -arc_ty);
            // ∂r/∂y1 = arc_tx = -(sy-cy)  ... wait
            // ∂r/∂y1 = -(-arc_tx) ... let me be careful
            // r = line_dx * arc_ty - line_dy * arc_tx
            // ∂r/∂y1 = 0 - (-1)*arc_tx = arc_tx
            jw.set(row_offset, ParamRef::PointY(lp1), arc_tx);
            // ∂r/∂x2 = arc_ty
            jw.set(row_offset, ParamRef::PointX(lp2), arc_ty);
            // ∂r/∂y2 = -arc_tx
            jw.set(row_offset, ParamRef::PointY(lp2), -arc_tx);

            // ∂r/∂sx: arc_ty = sx - cx → ∂arc_ty/∂sx = 1
            //   ∂r/∂sx = line_dx * 1 - line_dy * 0 = line_dx
            jw.add(row_offset, ParamRef::PointX(*shared), line_dx);
            // ∂r/∂sy: arc_tx = -(sy - cy) → ∂arc_tx/∂sy = -1
            //   ∂r/∂sy = line_dx * 0 - line_dy * (-1) = line_dy
            jw.add(row_offset, ParamRef::PointY(*shared), line_dy);

            // ∂r/∂cx: arc_ty = sx - cx → ∂arc_ty/∂cx = -1
            //   arc_tx = -(sy - cy) → ∂arc_tx/∂cx = 0
            //   ∂r/∂cx = line_dx * (-1) - line_dy * 0 = -line_dx
            jw.add(row_offset, ParamRef::PointX(center_id), -line_dx);
            // ∂r/∂cy: arc_tx = -(sy - cy) → ∂arc_tx/∂cy = 1
            //   arc_ty = sx - cx → ∂arc_ty/∂cy = 0
            //   ∂r/∂cy = line_dx * 0 - line_dy * 1 = -line_dy
            jw.add(row_offset, ParamRef::PointY(center_id), -line_dy);
        }
        Constraint::TangentArcArc(arc1, arc2, shared) => {
            // r = t1x * t2y - t1y * t2x
            // t1 = (-(sy-c1y), sx-c1x), t2 = (-(sy-c2y), sx-c2x)
            let (c1_id, _s1, _e1) = snap.arc(*arc1);
            let (c2_id, _s2, _e2) = snap.arc(*arc2);
            let (c1x, c1y) = snap.point(c1_id);
            let (c2x, c2y) = snap.point(c2_id);
            let (sx, sy) = snap.point(*shared);
            let t1x = -(sy - c1y);
            let t1y = sx - c1x;
            let t2x = -(sy - c2y);
            let t2y = sx - c2x;

            // r = t1x*t2y - t1y*t2x
            // Expand: (-(sy-c1y))*(sx-c2x) - (sx-c1x)*(-(sy-c2y))
            //       = -(sy-c1y)*(sx-c2x) + (sx-c1x)*(sy-c2y)

            // ∂r/∂sx: ∂t1y/∂sx=1, ∂t2y/∂sx=1 (t1x,t2x don't depend on sx)
            //   = t1x*1 - 1*t2x = t1x - t2x
            jw.set(row_offset, ParamRef::PointX(*shared), t1x - t2x);
            // ∂r/∂sy: ∂t1x/∂sy=-1, ∂t2x/∂sy=-1 (t1y,t2y don't depend on sy)
            //   = (-1)*t2y - t1y*(-1) = -t2y + t1y
            jw.set(row_offset, ParamRef::PointY(*shared), t1y - t2y);

            // ∂r/∂c1x: ∂t1y/∂c1x = -1, others 0
            //   = 0 - (-1)*t2x = t2x
            jw.add(row_offset, ParamRef::PointX(c1_id), t2x);
            // ∂r/∂c1y: ∂t1x/∂c1y = 1, others 0
            //   = 1*t2y - 0 = t2y
            jw.add(row_offset, ParamRef::PointY(c1_id), t2y);

            // ∂r/∂c2x: ∂t2y/∂c2x = -1, others 0
            //   = t1x*(-1) - 0 = -t1x
            jw.add(row_offset, ParamRef::PointX(c2_id), -t1x);
            // ∂r/∂c2y: ∂t2x/∂c2y = 1, others 0
            //   = 0 - t1y*1 = -t1y
            jw.add(row_offset, ParamRef::PointY(c2_id), -t1y);
        }
        Constraint::EqualRadiusArcArc(arc1, arc2) => {
            // r = dist(c1, s1) - dist(c2, s2)
            let (c1_id, s1_id, _e1) = snap.arc(*arc1);
            let (c2_id, s2_id, _e2) = snap.arc(*arc2);
            let (c1x, c1y) = snap.point(c1_id);
            let (s1x, s1y) = snap.point(s1_id);
            let (c2x, c2y) = snap.point(c2_id);
            let (s2x, s2y) = snap.point(s2_id);
            let d1x = s1x - c1x;
            let d1y = s1y - c1y;
            let d2x = s2x - c2x;
            let d2y = s2y - c2y;
            let r1 = d1x.hypot(d1y);
            let r2 = d2x.hypot(d2y);
            let (n1x, n1y) = if r1 < 1e-300 {
                (0.0, 0.0)
            } else {
                (d1x / r1, d1y / r1)
            };
            let (n2x, n2y) = if r2 < 1e-300 {
                (0.0, 0.0)
            } else {
                (d2x / r2, d2y / r2)
            };
            jw.set(row_offset, ParamRef::PointX(s1_id), n1x);
            jw.set(row_offset, ParamRef::PointY(s1_id), n1y);
            jw.add(row_offset, ParamRef::PointX(c1_id), -n1x);
            jw.add(row_offset, ParamRef::PointY(c1_id), -n1y);
            jw.add(row_offset, ParamRef::PointX(s2_id), -n2x);
            jw.add(row_offset, ParamRef::PointY(s2_id), -n2y);
            jw.add(row_offset, ParamRef::PointX(c2_id), n2x);
            jw.add(row_offset, ParamRef::PointY(c2_id), n2y);
        }
        Constraint::EqualRadiusArcCircle(arc, circ) => {
            // r = dist(center_arc, start_arc) - circle_radius
            let (center_id, start_id, _end_id) = snap.arc(*arc);
            let (cx, cy) = snap.point(center_id);
            let (sx, sy) = snap.point(start_id);
            let dx = sx - cx;
            let dy = sy - cy;
            let dist = dx.hypot(dy);
            let (nx, ny) = if dist < 1e-300 {
                (0.0, 0.0)
            } else {
                (dx / dist, dy / dist)
            };
            jw.set(row_offset, ParamRef::PointX(start_id), nx);
            jw.set(row_offset, ParamRef::PointY(start_id), ny);
            jw.add(row_offset, ParamRef::PointX(center_id), -nx);
            jw.add(row_offset, ParamRef::PointY(center_id), -ny);
            jw.add(row_offset, ParamRef::CircleRadius(*circ), -1.0);
        }
        Constraint::ArcLength(arc, _target) => {
            // r = r * theta - target
            // r = dist(center, start), theta = |atan2(cross, dot)|
            // cross = dsx*dey - dsy*dex, dot = dsx*dex + dsy*dey
            let (center_id, start_id, end_id) = snap.arc(*arc);
            let (cx, cy) = snap.point(center_id);
            let (sx, sy) = snap.point(start_id);
            let (ex, ey) = snap.point(end_id);
            let dsx = sx - cx;
            let dsy = sy - cy;
            let dex = ex - cx;
            let dey = ey - cy;
            let r = dsx.hypot(dsy);
            let cross = dsx * dey - dsy * dex;
            let dot = dsx * dex + dsy * dey;
            let raw_theta = cross.atan2(dot);
            let theta = raw_theta.abs();
            let sign_theta = if raw_theta >= 0.0 { 1.0 } else { -1.0 };

            // dr/dparam = dr_dparam * theta + r * dtheta_dparam
            // where dr is the partial of dist(center, start) and dtheta is
            // the partial of |atan2(cross, dot)|

            // atan2(y,x) → ∂/∂y = x/(x²+y²), ∂/∂x = -y/(x²+y²)
            let denom = cross * cross + dot * dot;
            if r < 1e-300 || denom < 1e-300 {
                return;
            }
            let inv_r = 1.0 / r;
            let inv_denom = 1.0 / denom;

            // Partials of cross and dot w.r.t. each variable:
            // cross = dsx*dey - dsy*dex
            // dot   = dsx*dex + dsy*dey

            // Helper: for a variable v, dtheta/dv = sign_theta * (dot*dcross_dv - cross*ddot_dv) / denom
            // and dr/dv for radius partials

            // sx partials: dsx = sx-cx, ∂dsx/∂sx = 1
            //   ∂cross/∂sx = dey, ∂dot/∂sx = dex
            //   ∂r/∂sx = dsx/r
            let dr_dsx = dsx * inv_r;
            let dtheta_dsx = sign_theta * (dot * dey - cross * dex) * inv_denom;
            jw.set(
                row_offset,
                ParamRef::PointX(start_id),
                dr_dsx * theta + r * dtheta_dsx,
            );

            // sy partials: dsy = sy-cy, ∂dsy/∂sy = 1
            //   ∂cross/∂sy = -dex, ∂dot/∂sy = dey
            //   ∂r/∂sy = dsy/r
            let dr_dsy = dsy * inv_r;
            let dtheta_dsy = sign_theta * (dot * (-dex) - cross * dey) * inv_denom;
            jw.set(
                row_offset,
                ParamRef::PointY(start_id),
                dr_dsy * theta + r * dtheta_dsy,
            );

            // ex partials: dex = ex-cx, ∂dex/∂ex = 1
            //   ∂cross/∂ex = -dsy, ∂dot/∂ex = dsx
            //   ∂r/∂ex = 0 (r doesn't depend on end)
            let dtheta_dex = sign_theta * (dot * (-dsy) - cross * dsx) * inv_denom;
            jw.set(row_offset, ParamRef::PointX(end_id), r * dtheta_dex);

            // ey partials: dey = ey-cy, ∂dey/∂ey = 1
            //   ∂cross/∂ey = dsx, ∂dot/∂ey = dsy
            //   ∂r/∂ey = 0
            let dtheta_dey = sign_theta * (dot * dsx - cross * dsy) * inv_denom;
            jw.set(row_offset, ParamRef::PointY(end_id), r * dtheta_dey);

            // cx partials:
            //   Derivatives w.r.t. cx: ∂dsx/∂cx = -1, ∂dsy/∂cx = 0, ∂dex/∂cx = -1, ∂dey/∂cx = 0
            //   ∂cross/∂cx = ∂dsx/∂cx·dey + dsx·∂dey/∂cx - ∂dsy/∂cx·dex - dsy·∂dex/∂cx
            //              = (-1)·dey + 0 - 0 - dsy·(-1) = -dey + dsy
            let dcross_dcx = -dey + dsy;
            //   ∂dot/∂cx = ∂dsx/∂cx·dex + dsx·∂dex/∂cx + ∂dsy/∂cx·dey + dsy·∂dey/∂cx
            //            = (-1)·dex + dsx·(-1) + 0 + 0 = -dex - dsx
            let ddot_dcx = -dex - dsx;
            let dr_dcx = -dsx * inv_r; // ∂r/∂cx = -dsx/r
            let dtheta_dcx = sign_theta * (dot * dcross_dcx - cross * ddot_dcx) * inv_denom;
            jw.add(
                row_offset,
                ParamRef::PointX(center_id),
                dr_dcx * theta + r * dtheta_dcx,
            );

            // cy partials:
            //   Derivatives w.r.t. cy: ∂dsx/∂cy = 0, ∂dsy/∂cy = -1, ∂dex/∂cy = 0, ∂dey/∂cy = -1
            //   ∂cross/∂cy = ∂dsx/∂cy·dey + dsx·∂dey/∂cy - ∂dsy/∂cy·dex - dsy·∂dex/∂cy
            //              = 0 + dsx·(-1) - (-1)·dex - 0 = -dsx + dex
            let dcross_dcy = -dsx + dex;
            //   ∂dot/∂cy = ∂dsx/∂cy·dex + dsx·∂dex/∂cy + ∂dsy/∂cy·dey + dsy·∂dey/∂cy
            //            = 0 + 0 + (-1)·dey + dsy·(-1) = -dey - dsy
            let ddot_dcy = -dey - dsy;
            let dr_dcy = -dsy * inv_r;
            let dtheta_dcy = sign_theta * (dot * dcross_dcy - cross * ddot_dcy) * inv_denom;
            jw.add(
                row_offset,
                ParamRef::PointY(center_id),
                dr_dcy * theta + r * dtheta_dcy,
            );
        }
        Constraint::ConcentricArcArc(arc1, arc2) => {
            let (c1_id, _s1, _e1) = snap.arc(*arc1);
            let (c2_id, _s2, _e2) = snap.arc(*arc2);
            // r0 = c1x - c2x, r1 = c1y - c2y
            jw.set(row_offset, ParamRef::PointX(c1_id), 1.0);
            jw.add(row_offset, ParamRef::PointX(c2_id), -1.0);
            jw.set(row_offset + 1, ParamRef::PointY(c1_id), 1.0);
            jw.add(row_offset + 1, ParamRef::PointY(c2_id), -1.0);
        }
        Constraint::ConcentricArcCircle(arc, circ) => {
            let (arc_center, _s, _e) = snap.arc(*arc);
            let (circ_center, _radius) = snap.circle(*circ);
            // r0 = acx - ccx, r1 = acy - ccy
            jw.set(row_offset, ParamRef::PointX(arc_center), 1.0);
            jw.add(row_offset, ParamRef::PointX(circ_center), -1.0);
            jw.set(row_offset + 1, ParamRef::PointY(arc_center), 1.0);
            jw.add(row_offset + 1, ParamRef::PointY(circ_center), -1.0);
        }
    }
}

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::expect_used, clippy::iter_on_single_items)]
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
            arcs: HashMap::new(),
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
                ParamRef::CircleRadius(cid) => {
                    // Perturb circle radius — need a mutable copy of circles
                    let mut perturbed_circles = snap.circles.clone();
                    if let Some(entry) = perturbed_circles.get_mut(cid) {
                        entry.1 += eps;
                    }
                    let perturbed_snap_circ = EntitySnapshot {
                        points: perturbed_points,
                        lines: snap.lines.clone(),
                        circles: perturbed_circles,
                        arcs: snap.arcs.clone(),
                    };
                    let mut r1 = Vec::new();
                    eval_residuals(c, &perturbed_snap_circ, &mut r1);
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
                    continue;
                }
            }
            let perturbed_snap = EntitySnapshot {
                points: perturbed_points,
                lines: snap.lines.clone(),
                circles: snap.circles.clone(),
                arcs: snap.arcs.clone(),
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
            arcs: HashMap::new(),
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
            arcs: HashMap::new(),
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
            arcs: HashMap::new(),
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
    fn jacobian_point_on_circle() {
        use super::super::entity::GenArena;
        use super::super::entity::{CircleData, PointData};
        let mut pts = GenArena::new();
        let center = pts.insert(PointData {
            x: 1.0,
            y: 2.0,
            fixed: false,
        });
        let pt = pts.insert(PointData {
            x: 4.0,
            y: 6.0,
            fixed: false,
        });
        let mut circles = GenArena::new();
        let circ = circles.insert(CircleData {
            center,
            radius: 3.0,
        });
        let snap = EntitySnapshot {
            points: [(center, (1.0, 2.0)), (pt, (4.0, 6.0))]
                .into_iter()
                .collect(),
            lines: HashMap::new(),
            circles: [(circ, (center, 3.0))].into_iter().collect(),
            arcs: HashMap::new(),
        };
        let c = Constraint::PointOnCircle(pt, circ);
        let params = vec![
            ParamRef::PointX(pt),
            ParamRef::PointY(pt),
            ParamRef::PointX(center),
            ParamRef::PointY(center),
            ParamRef::CircleRadius(circ),
        ];
        check_jacobian_fd(&c, &snap, &params);
    }

    #[test]
    fn jacobian_point_on_arc() {
        use super::super::entity::GenArena;
        use super::super::entity::{ArcData, PointData};
        let mut pts = GenArena::new();
        let center = pts.insert(PointData {
            x: 0.0,
            y: 0.0,
            fixed: false,
        });
        let start = pts.insert(PointData {
            x: 2.0,
            y: 0.0,
            fixed: false,
        });
        let end = pts.insert(PointData {
            x: 0.0,
            y: 2.0,
            fixed: false,
        });
        let pt = pts.insert(PointData {
            x: 1.5,
            y: 1.5,
            fixed: false,
        });
        let mut arcs = GenArena::new();
        let arc = arcs.insert(ArcData { center, start, end });
        let snap = EntitySnapshot {
            points: [
                (center, (0.0, 0.0)),
                (start, (2.0, 0.0)),
                (end, (0.0, 2.0)),
                (pt, (1.5, 1.5)),
            ]
            .into_iter()
            .collect(),
            lines: HashMap::new(),
            circles: HashMap::new(),
            arcs: [(arc, (center, start, end))].into_iter().collect(),
        };
        let c = Constraint::PointOnArc(pt, arc);
        let params = vec![
            ParamRef::PointX(pt),
            ParamRef::PointY(pt),
            ParamRef::PointX(center),
            ParamRef::PointY(center),
            ParamRef::PointX(start),
            ParamRef::PointY(start),
        ];
        check_jacobian_fd(&c, &snap, &params);
    }

    #[test]
    fn jacobian_tangent_line_arc() {
        use super::super::entity::GenArena;
        use super::super::entity::{ArcData, LineData, PointData};
        let mut pts = GenArena::new();
        let p1 = pts.insert(PointData {
            x: 0.0,
            y: 0.0,
            fixed: false,
        });
        let p2 = pts.insert(PointData {
            x: 2.0,
            y: 0.0,
            fixed: false,
        });
        let center = pts.insert(PointData {
            x: 2.0,
            y: 1.0,
            fixed: false,
        });
        let start = pts.insert(PointData {
            x: 2.0,
            y: 0.0,
            fixed: false,
        });
        let end = pts.insert(PointData {
            x: 3.0,
            y: 1.0,
            fixed: false,
        });
        // shared point is p2 (same position as start)
        let mut lines = GenArena::new();
        let line = lines.insert(LineData { p1, p2 });
        let mut arcs = GenArena::new();
        let arc = arcs.insert(ArcData { center, start, end });
        let snap = EntitySnapshot {
            points: [
                (p1, (0.0, 0.0)),
                (p2, (2.0, 0.0)),
                (center, (2.0, 1.0)),
                (start, (2.0, 0.0)),
                (end, (3.0, 1.0)),
            ]
            .into_iter()
            .collect(),
            lines: [(line, (p1, p2))].into_iter().collect(),
            circles: HashMap::new(),
            arcs: [(arc, (center, start, end))].into_iter().collect(),
        };
        let c = Constraint::TangentLineArc(line, arc, p2);
        let params = vec![
            ParamRef::PointX(p1),
            ParamRef::PointY(p1),
            ParamRef::PointX(p2),
            ParamRef::PointY(p2),
            ParamRef::PointX(center),
            ParamRef::PointY(center),
        ];
        check_jacobian_fd(&c, &snap, &params);
    }

    #[test]
    fn jacobian_tangent_arc_arc() {
        use super::super::entity::GenArena;
        use super::super::entity::{ArcData, PointData};
        let mut pts = GenArena::new();
        let c1 = pts.insert(PointData {
            x: 0.0,
            y: 1.0,
            fixed: false,
        });
        let c2 = pts.insert(PointData {
            x: 2.0,
            y: 1.0,
            fixed: false,
        });
        let shared = pts.insert(PointData {
            x: 1.0,
            y: 0.0,
            fixed: false,
        });
        let s1 = pts.insert(PointData {
            x: 1.0,
            y: 0.0,
            fixed: false,
        });
        let e1 = pts.insert(PointData {
            x: -1.0,
            y: 1.0,
            fixed: false,
        });
        let s2 = pts.insert(PointData {
            x: 1.0,
            y: 0.0,
            fixed: false,
        });
        let e2 = pts.insert(PointData {
            x: 3.0,
            y: 1.0,
            fixed: false,
        });
        let mut arcs = GenArena::new();
        let arc1 = arcs.insert(ArcData {
            center: c1,
            start: s1,
            end: e1,
        });
        let arc2 = arcs.insert(ArcData {
            center: c2,
            start: s2,
            end: e2,
        });
        let snap = EntitySnapshot {
            points: [
                (c1, (0.0, 1.0)),
                (c2, (2.0, 1.0)),
                (shared, (1.0, 0.0)),
                (s1, (1.0, 0.0)),
                (e1, (-1.0, 1.0)),
                (s2, (1.0, 0.0)),
                (e2, (3.0, 1.0)),
            ]
            .into_iter()
            .collect(),
            lines: HashMap::new(),
            circles: HashMap::new(),
            arcs: [(arc1, (c1, s1, e1)), (arc2, (c2, s2, e2))]
                .into_iter()
                .collect(),
        };
        let c = Constraint::TangentArcArc(arc1, arc2, shared);
        let params = vec![
            ParamRef::PointX(shared),
            ParamRef::PointY(shared),
            ParamRef::PointX(c1),
            ParamRef::PointY(c1),
            ParamRef::PointX(c2),
            ParamRef::PointY(c2),
        ];
        check_jacobian_fd(&c, &snap, &params);
    }

    #[test]
    fn jacobian_equal_radius_arc_arc() {
        use super::super::entity::GenArena;
        use super::super::entity::{ArcData, PointData};
        let mut pts = GenArena::new();
        let c1 = pts.insert(PointData {
            x: 0.0,
            y: 0.0,
            fixed: false,
        });
        let s1 = pts.insert(PointData {
            x: 2.0,
            y: 0.0,
            fixed: false,
        });
        let e1 = pts.insert(PointData {
            x: 0.0,
            y: 2.0,
            fixed: false,
        });
        let c2 = pts.insert(PointData {
            x: 5.0,
            y: 0.0,
            fixed: false,
        });
        let s2 = pts.insert(PointData {
            x: 8.0,
            y: 0.0,
            fixed: false,
        });
        let e2 = pts.insert(PointData {
            x: 5.0,
            y: 3.0,
            fixed: false,
        });
        let mut arcs = GenArena::new();
        let arc1 = arcs.insert(ArcData {
            center: c1,
            start: s1,
            end: e1,
        });
        let arc2 = arcs.insert(ArcData {
            center: c2,
            start: s2,
            end: e2,
        });
        let snap = EntitySnapshot {
            points: [
                (c1, (0.0, 0.0)),
                (s1, (2.0, 0.0)),
                (e1, (0.0, 2.0)),
                (c2, (5.0, 0.0)),
                (s2, (8.0, 0.0)),
                (e2, (5.0, 3.0)),
            ]
            .into_iter()
            .collect(),
            lines: HashMap::new(),
            circles: HashMap::new(),
            arcs: [(arc1, (c1, s1, e1)), (arc2, (c2, s2, e2))]
                .into_iter()
                .collect(),
        };
        let c = Constraint::EqualRadiusArcArc(arc1, arc2);
        let params = vec![
            ParamRef::PointX(c1),
            ParamRef::PointY(c1),
            ParamRef::PointX(s1),
            ParamRef::PointY(s1),
            ParamRef::PointX(c2),
            ParamRef::PointY(c2),
            ParamRef::PointX(s2),
            ParamRef::PointY(s2),
        ];
        check_jacobian_fd(&c, &snap, &params);
    }

    #[test]
    fn jacobian_equal_radius_arc_circle() {
        use super::super::entity::GenArena;
        use super::super::entity::{ArcData, CircleData, PointData};
        let mut pts = GenArena::new();
        let ac = pts.insert(PointData {
            x: 0.0,
            y: 0.0,
            fixed: false,
        });
        let as_ = pts.insert(PointData {
            x: 2.0,
            y: 0.0,
            fixed: false,
        });
        let ae = pts.insert(PointData {
            x: 0.0,
            y: 2.0,
            fixed: false,
        });
        let cc = pts.insert(PointData {
            x: 5.0,
            y: 5.0,
            fixed: false,
        });
        let mut arcs = GenArena::new();
        let arc = arcs.insert(ArcData {
            center: ac,
            start: as_,
            end: ae,
        });
        let mut circles = GenArena::new();
        let circ = circles.insert(CircleData {
            center: cc,
            radius: 3.0,
        });
        let snap = EntitySnapshot {
            points: [
                (ac, (0.0, 0.0)),
                (as_, (2.0, 0.0)),
                (ae, (0.0, 2.0)),
                (cc, (5.0, 5.0)),
            ]
            .into_iter()
            .collect(),
            lines: HashMap::new(),
            circles: [(circ, (cc, 3.0))].into_iter().collect(),
            arcs: [(arc, (ac, as_, ae))].into_iter().collect(),
        };
        let c = Constraint::EqualRadiusArcCircle(arc, circ);
        let params = vec![
            ParamRef::PointX(ac),
            ParamRef::PointY(ac),
            ParamRef::PointX(as_),
            ParamRef::PointY(as_),
            ParamRef::CircleRadius(circ),
        ];
        check_jacobian_fd(&c, &snap, &params);
    }

    #[test]
    fn jacobian_arc_length() {
        use super::super::entity::GenArena;
        use super::super::entity::{ArcData, PointData};
        let mut pts = GenArena::new();
        let center = pts.insert(PointData {
            x: 0.0,
            y: 0.0,
            fixed: false,
        });
        let start = pts.insert(PointData {
            x: 2.0,
            y: 0.0,
            fixed: false,
        });
        let end = pts.insert(PointData {
            x: 0.0,
            y: 2.0,
            fixed: false,
        });
        let mut arcs = GenArena::new();
        let arc = arcs.insert(ArcData { center, start, end });
        let snap = EntitySnapshot {
            points: [(center, (0.0, 0.0)), (start, (2.0, 0.0)), (end, (0.0, 2.0))]
                .into_iter()
                .collect(),
            lines: HashMap::new(),
            circles: HashMap::new(),
            arcs: [(arc, (center, start, end))].into_iter().collect(),
        };
        let target = std::f64::consts::PI; // 90 degrees * r=2
        let c = Constraint::ArcLength(arc, target);
        let params = vec![
            ParamRef::PointX(center),
            ParamRef::PointY(center),
            ParamRef::PointX(start),
            ParamRef::PointY(start),
            ParamRef::PointX(end),
            ParamRef::PointY(end),
        ];
        check_jacobian_fd(&c, &snap, &params);
    }

    #[test]
    fn jacobian_concentric_arc_arc() {
        use super::super::entity::GenArena;
        use super::super::entity::{ArcData, PointData};
        let mut pts = GenArena::new();
        let c1 = pts.insert(PointData {
            x: 1.0,
            y: 2.0,
            fixed: false,
        });
        let s1 = pts.insert(PointData {
            x: 3.0,
            y: 2.0,
            fixed: false,
        });
        let e1 = pts.insert(PointData {
            x: 1.0,
            y: 4.0,
            fixed: false,
        });
        let c2 = pts.insert(PointData {
            x: 3.0,
            y: 4.0,
            fixed: false,
        });
        let s2 = pts.insert(PointData {
            x: 4.0,
            y: 4.0,
            fixed: false,
        });
        let e2 = pts.insert(PointData {
            x: 3.0,
            y: 5.0,
            fixed: false,
        });
        let mut arcs = GenArena::new();
        let arc1 = arcs.insert(ArcData {
            center: c1,
            start: s1,
            end: e1,
        });
        let arc2 = arcs.insert(ArcData {
            center: c2,
            start: s2,
            end: e2,
        });
        let snap = EntitySnapshot {
            points: [
                (c1, (1.0, 2.0)),
                (s1, (3.0, 2.0)),
                (e1, (1.0, 4.0)),
                (c2, (3.0, 4.0)),
                (s2, (4.0, 4.0)),
                (e2, (3.0, 5.0)),
            ]
            .into_iter()
            .collect(),
            lines: HashMap::new(),
            circles: HashMap::new(),
            arcs: [(arc1, (c1, s1, e1)), (arc2, (c2, s2, e2))]
                .into_iter()
                .collect(),
        };
        let c = Constraint::ConcentricArcArc(arc1, arc2);
        let params = vec![
            ParamRef::PointX(c1),
            ParamRef::PointY(c1),
            ParamRef::PointX(c2),
            ParamRef::PointY(c2),
        ];
        check_jacobian_fd(&c, &snap, &params);
    }

    #[test]
    fn jacobian_concentric_arc_circle() {
        use super::super::entity::GenArena;
        use super::super::entity::{ArcData, CircleData, PointData};
        let mut pts = GenArena::new();
        let ac = pts.insert(PointData {
            x: 1.0,
            y: 2.0,
            fixed: false,
        });
        let as_ = pts.insert(PointData {
            x: 3.0,
            y: 2.0,
            fixed: false,
        });
        let ae = pts.insert(PointData {
            x: 1.0,
            y: 4.0,
            fixed: false,
        });
        let cc = pts.insert(PointData {
            x: 3.0,
            y: 4.0,
            fixed: false,
        });
        let mut arcs = GenArena::new();
        let arc = arcs.insert(ArcData {
            center: ac,
            start: as_,
            end: ae,
        });
        let mut circles = GenArena::new();
        let circ = circles.insert(CircleData {
            center: cc,
            radius: 2.0,
        });
        let snap = EntitySnapshot {
            points: [
                (ac, (1.0, 2.0)),
                (as_, (3.0, 2.0)),
                (ae, (1.0, 4.0)),
                (cc, (3.0, 4.0)),
            ]
            .into_iter()
            .collect(),
            lines: HashMap::new(),
            circles: [(circ, (cc, 2.0))].into_iter().collect(),
            arcs: [(arc, (ac, as_, ae))].into_iter().collect(),
        };
        let c = Constraint::ConcentricArcCircle(arc, circ);
        let params = vec![
            ParamRef::PointX(ac),
            ParamRef::PointY(ac),
            ParamRef::PointX(cc),
            ParamRef::PointY(cc),
        ];
        check_jacobian_fd(&c, &snap, &params);
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
            arcs: HashMap::new(),
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
