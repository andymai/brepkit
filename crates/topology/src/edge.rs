//! Edge — a curve bounded by two vertices.

use brepkit_math::curves::{Circle3D, Ellipse3D};
use brepkit_math::nurbs::curve::NurbsCurve;
use brepkit_math::traits::ParametricCurve;
use brepkit_math::vec::{Point3, Vec3};

use crate::arena;
use crate::vertex::VertexId;

/// Typed handle for an [`Edge`] stored in an [`Arena`](crate::Arena).
pub type EdgeId = arena::Id<Edge>;

/// The geometric curve associated with an edge.
#[derive(Debug, Clone)]
pub enum EdgeCurve {
    /// A straight line segment (geometry is fully determined by the vertices).
    Line,
    /// A NURBS curve defining the edge geometry.
    NurbsCurve(NurbsCurve),
    /// A circular arc (or full circle when the edge is closed).
    Circle(Circle3D),
    /// An elliptical arc (or full ellipse when the edge is closed).
    Ellipse(Ellipse3D),
}

impl EdgeCurve {
    /// Evaluate the curve at parameter `t`.
    ///
    /// `Line` has no stored geometry, so it linearly interpolates between
    /// `start` and `end` with `t` in `[0, 1]`. Circle, Ellipse, and NURBS
    /// dispatch to their [`ParametricCurve`] implementations.
    #[must_use]
    pub fn evaluate_with_endpoints(&self, t: f64, start: Point3, end: Point3) -> Point3 {
        match self {
            Self::Line => start + (end - start) * t,
            Self::Circle(c) => ParametricCurve::evaluate(c, t),
            Self::Ellipse(e) => ParametricCurve::evaluate(e, t),
            Self::NurbsCurve(n) => ParametricCurve::evaluate(n, t),
        }
    }

    /// Tangent vector at parameter `t`.
    ///
    /// For `Line`, returns the normalized `start → end` direction. For curves
    /// with stored geometry, dispatches to [`ParametricCurve::tangent`].
    #[must_use]
    pub fn tangent_with_endpoints(&self, t: f64, start: Point3, end: Point3) -> Vec3 {
        match self {
            Self::Line => {
                let dir = end - start;
                dir.normalize().unwrap_or(Vec3::new(1.0, 0.0, 0.0))
            }
            Self::Circle(c) => ParametricCurve::tangent(c, t),
            Self::Ellipse(e) => ParametricCurve::tangent(e, t),
            Self::NurbsCurve(n) => ParametricCurve::tangent(n, t),
        }
    }

    /// Parameter domain of this curve.
    ///
    /// `Line` uses `[0, 1]`. Closed Circle and Ellipse edges (`start ≈ end`)
    /// use the full `[0, 2π]` domain. Open arcs project both endpoints onto
    /// the curve and return the CCW angular range `[a₀, a₁]` with `a₁ > a₀`,
    /// so sampling the domain traces exactly the trimmed arc rather than the
    /// full curve. NURBS edges whose endpoints sit at the curve's natural
    /// ends (either orientation), or whose endpoint projections fail to
    /// validate as a forward interior sub-span, use the full knot span; a
    /// validated open sub-span returns the projected `[t₀, t₁]` so the edge
    /// samples only its own piece of a shared curve.
    #[must_use]
    pub fn domain_with_endpoints(&self, start: Point3, end: Point3) -> (f64, f64) {
        const TAU: f64 = std::f64::consts::TAU;
        // Below this chord the endpoints are considered coincident and the
        // edge is treated as a closed (full) curve.
        const CLOSED_EPS: f64 = 1e-9;
        // NURBS whole-edge match band: split vertices on marched/fit curves sit
        // up to the fit error (~1e-6) off the exact curve, well above vertex
        // tolerance.
        const END_EPS: f64 = 1e-6;
        // NURBS sub-span on-curve band (the weld scale).
        const WELD_EPS: f64 = 1e-5;
        match self {
            Self::Line => (0.0, 1.0),
            Self::Circle(c) => {
                if (start - end).length() < CLOSED_EPS {
                    ParametricCurve::domain(c)
                } else {
                    let a0 = c.project(start);
                    let delta = (c.project(end) - a0).rem_euclid(TAU);
                    let delta = if delta < 1e-12 { TAU } else { delta };
                    (a0, a0 + delta)
                }
            }
            Self::Ellipse(e) => {
                if (start - end).length() < CLOSED_EPS {
                    ParametricCurve::domain(e)
                } else {
                    let a0 = e.project(start);
                    let delta = (e.project(end) - a0).rem_euclid(TAU);
                    let delta = if delta < 1e-12 { TAU } else { delta };
                    (a0, a0 + delta)
                }
            }
            Self::NurbsCurve(n) => {
                let (d0, d1) = ParametricCurve::domain(n);
                if (start - end).length() < CLOSED_EPS {
                    return (d0, d1);
                }
                let p0 = ParametricCurve::evaluate(n, d0);
                let p1 = ParametricCurve::evaluate(n, d1);
                if ((p0 - start).length() < END_EPS && (p1 - end).length() < END_EPS)
                    || ((p0 - end).length() < END_EPS && (p1 - start).length() < END_EPS)
                {
                    return (d0, d1);
                }
                let proj = |p| brepkit_math::nurbs::projection::project_point_to_curve(n, p, 1e-9);
                if let (Ok(pa), Ok(pb)) = (proj(start), proj(end)) {
                    // Accept only a forward, non-degenerate, on-curve span.
                    // Everything else — reversed pairs (on a closed curve these
                    // are usually seam-crossing forward sub-arcs, where
                    // interpolating backward would trace the complement arc;
                    // downstream arrangement consumers also mint degenerate
                    // sliver loops from reversed spans), degenerate spans,
                    // off-curve endpoints — keeps the historical full-domain
                    // behaviour.
                    if pa.distance < WELD_EPS
                        && pb.distance < WELD_EPS
                        && pb.parameter - pa.parameter > 1e-6 * (d1 - d0)
                    {
                        return (pa.parameter, pb.parameter);
                    }
                }
                (d0, d1)
            }
        }
    }

    /// Type tag string for debugging and serialization.
    #[must_use]
    pub const fn type_tag(&self) -> &'static str {
        match self {
            Self::Line => "line",
            Self::Circle(_) => "circle",
            Self::Ellipse(_) => "ellipse",
            Self::NurbsCurve(_) => "nurbs_curve",
        }
    }
}

/// A topological edge: a curve bounded by a start and end vertex.
///
/// An edge where `start == end` is a closed (degenerate) edge such as
/// a full circle.
#[derive(Debug, Clone)]
pub struct Edge {
    /// The vertex at the start of the edge.
    start: VertexId,
    /// The vertex at the end of the edge.
    end: VertexId,
    /// The geometric curve of the edge.
    curve: EdgeCurve,
    /// Optional edge-specific tolerance. When `None`, the edge inherits the
    /// tolerance from its bounding vertices.
    tolerance: Option<f64>,
}

impl Edge {
    /// Creates a new edge between two vertices with the given curve.
    ///
    /// The edge tolerance defaults to `None`, meaning the edge inherits
    /// tolerance from its bounding vertices.
    #[must_use]
    pub const fn new(start: VertexId, end: VertexId, curve: EdgeCurve) -> Self {
        Self {
            start,
            end,
            curve,
            tolerance: None,
        }
    }

    /// Creates a new edge with an explicit tolerance.
    ///
    /// Pass `None` to inherit the vertex tolerance, or `Some(tol)` to set
    /// an edge-specific tolerance.
    #[must_use]
    pub const fn with_tolerance(
        start: VertexId,
        end: VertexId,
        curve: EdgeCurve,
        tol: Option<f64>,
    ) -> Self {
        Self {
            start,
            end,
            curve,
            tolerance: tol,
        }
    }

    /// Returns the start vertex of this edge.
    #[must_use]
    pub const fn start(&self) -> VertexId {
        self.start
    }

    /// Returns the end vertex of this edge.
    #[must_use]
    pub const fn end(&self) -> VertexId {
        self.end
    }

    /// Returns a reference to the curve geometry of this edge.
    #[must_use]
    pub const fn curve(&self) -> &EdgeCurve {
        &self.curve
    }

    /// Returns `true` if the edge is closed (start equals end).
    #[must_use]
    pub fn is_closed(&self) -> bool {
        self.start == self.end
    }

    /// Sets the start vertex of this edge.
    pub fn set_start(&mut self, start: VertexId) {
        self.start = start;
    }

    /// Sets the end vertex of this edge.
    pub fn set_end(&mut self, end: VertexId) {
        self.end = end;
    }

    /// Sets the curve geometry of this edge.
    pub fn set_curve(&mut self, curve: EdgeCurve) {
        self.curve = curve;
    }

    /// Returns the edge-specific tolerance, or `None` if the edge inherits
    /// tolerance from its bounding vertices.
    #[must_use]
    pub const fn tolerance(&self) -> Option<f64> {
        self.tolerance
    }

    /// Sets the edge-specific tolerance.
    ///
    /// Pass `None` to revert to inheriting the vertex tolerance.
    pub fn set_tolerance(&mut self, tol: Option<f64>) {
        self.tolerance = tol;
    }

    /// Returns the effective tolerance for this edge.
    ///
    /// If the edge has its own tolerance, that value is returned. Otherwise
    /// the provided `vertex_tol` (typically the maximum of the two bounding
    /// vertex tolerances) is used as a fallback.
    #[must_use]
    pub fn effective_tolerance(&self, vertex_tol: f64) -> f64 {
        self.tolerance.unwrap_or(vertex_tol)
    }
}

#[cfg(test)]
mod tests {
    #![allow(clippy::unwrap_used, clippy::expect_used)]

    use super::*;
    use crate::arena::Arena;
    use crate::vertex::Vertex;
    use brepkit_math::vec::Point3;

    fn make_test_vertices() -> (VertexId, VertexId) {
        let mut arena: Arena<Vertex> = Arena::new();
        let v0 = arena.alloc(Vertex::new(Point3::new(0.0, 0.0, 0.0), 1e-7));
        let v1 = arena.alloc(Vertex::new(Point3::new(1.0, 0.0, 0.0), 1e-7));
        (v0, v1)
    }

    #[test]
    fn new_defaults_tolerance_to_none() {
        let (v0, v1) = make_test_vertices();
        let edge = Edge::new(v0, v1, EdgeCurve::Line);
        assert!(edge.tolerance().is_none());
    }

    #[test]
    fn with_tolerance_stores_value() {
        let (v0, v1) = make_test_vertices();
        let edge = Edge::with_tolerance(v0, v1, EdgeCurve::Line, Some(1e-5));
        assert_eq!(edge.tolerance(), Some(1e-5));
    }

    #[test]
    fn with_tolerance_none() {
        let (v0, v1) = make_test_vertices();
        let edge = Edge::with_tolerance(v0, v1, EdgeCurve::Line, None);
        assert!(edge.tolerance().is_none());
    }

    #[test]
    fn set_tolerance_round_trip() {
        let (v0, v1) = make_test_vertices();
        let mut edge = Edge::new(v0, v1, EdgeCurve::Line);

        edge.set_tolerance(Some(0.001));
        assert_eq!(edge.tolerance(), Some(0.001));

        edge.set_tolerance(None);
        assert!(edge.tolerance().is_none());
    }

    #[test]
    fn effective_tolerance_uses_own_when_set() {
        let (v0, v1) = make_test_vertices();
        let edge = Edge::with_tolerance(v0, v1, EdgeCurve::Line, Some(1e-5));
        assert!((edge.effective_tolerance(1e-7) - 1e-5).abs() < f64::EPSILON);
    }

    #[test]
    fn effective_tolerance_falls_back_to_vertex_tol() {
        let (v0, v1) = make_test_vertices();
        let edge = Edge::new(v0, v1, EdgeCurve::Line);
        assert!((edge.effective_tolerance(1e-7) - 1e-7).abs() < f64::EPSILON);
    }

    fn open_nurbs() -> EdgeCurve {
        let pts = [
            Point3::new(0.0, 0.0, 0.0),
            Point3::new(1.0, 1.0, 0.0),
            Point3::new(2.0, 1.5, 0.0),
            Point3::new(3.0, 1.0, 0.0),
            Point3::new(4.0, 0.0, 0.0),
        ];
        EdgeCurve::NurbsCurve(brepkit_math::nurbs::fitting::interpolate(&pts, 3).unwrap())
    }

    #[test]
    fn nurbs_domain_whole_edge_keeps_full_span_both_orientations() {
        let curve = open_nurbs();
        let EdgeCurve::NurbsCurve(n) = &curve else {
            unreachable!()
        };
        let (d0, d1) = brepkit_math::traits::ParametricCurve::domain(n);
        let p0 = brepkit_math::traits::ParametricCurve::evaluate(n, d0);
        let p1 = brepkit_math::traits::ParametricCurve::evaluate(n, d1);
        assert_eq!(curve.domain_with_endpoints(p0, p1), (d0, d1));
        assert_eq!(curve.domain_with_endpoints(p1, p0), (d0, d1));
    }

    #[test]
    fn nurbs_domain_forward_sub_span_is_trimmed() {
        let curve = open_nurbs();
        let EdgeCurve::NurbsCurve(n) = &curve else {
            unreachable!()
        };
        let (d0, d1) = brepkit_math::traits::ParametricCurve::domain(n);
        let ta = d0 + 0.25 * (d1 - d0);
        let tb = d0 + 0.7 * (d1 - d0);
        let pa = brepkit_math::traits::ParametricCurve::evaluate(n, ta);
        let pb = brepkit_math::traits::ParametricCurve::evaluate(n, tb);
        let (t0, t1) = curve.domain_with_endpoints(pa, pb);
        assert!((t0 - ta).abs() < 1e-6, "t0={t0} expected {ta}");
        assert!((t1 - tb).abs() < 1e-6, "t1={t1} expected {tb}");
        // The trimmed domain evaluates back to the endpoints, so a consumer
        // sampling it traces only the edge's own piece of the shared curve.
        let s = curve.evaluate_with_endpoints(t0, pa, pb);
        let e = curve.evaluate_with_endpoints(t1, pa, pb);
        assert!((s - pa).length() < 1e-6);
        assert!((e - pb).length() < 1e-6);
    }

    #[test]
    fn nurbs_domain_reversed_sub_span_falls_back_to_full_domain() {
        let curve = open_nurbs();
        let EdgeCurve::NurbsCurve(n) = &curve else {
            unreachable!()
        };
        let (d0, d1) = brepkit_math::traits::ParametricCurve::domain(n);
        let ta = d0 + 0.25 * (d1 - d0);
        let tb = d0 + 0.7 * (d1 - d0);
        let pa = brepkit_math::traits::ParametricCurve::evaluate(n, ta);
        let pb = brepkit_math::traits::ParametricCurve::evaluate(n, tb);
        assert_eq!(curve.domain_with_endpoints(pb, pa), (d0, d1));
    }

    #[test]
    fn nurbs_domain_off_curve_endpoints_fall_back_to_full_domain() {
        let curve = open_nurbs();
        let EdgeCurve::NurbsCurve(n) = &curve else {
            unreachable!()
        };
        let (d0, d1) = brepkit_math::traits::ParametricCurve::domain(n);
        let ta = d0 + 0.25 * (d1 - d0);
        let tb = d0 + 0.7 * (d1 - d0);
        let off = Vec3::new(0.0, 0.0, 1.0) * 0.5;
        let pa = brepkit_math::traits::ParametricCurve::evaluate(n, ta) + off;
        let pb = brepkit_math::traits::ParametricCurve::evaluate(n, tb) + off;
        assert_eq!(curve.domain_with_endpoints(pa, pb), (d0, d1));
    }
}
