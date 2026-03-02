//! `PCurve` — 2D parametric curves on surfaces.
//!
//! A pcurve represents an edge's geometry projected into a face's surface
//! parameter space (u, v). `PCurves` are essential for exact boolean operations,
//! surface trimming, and proper I/O with STEP/IGES formats.
//!
//! `PCurves` are stored in a central registry on [`Topology`](crate::Topology),
//! keyed by (edge, face) pairs. This avoids modifying the `OrientedEdge`
//! struct (which is `Copy`) and follows a relational design.

use std::collections::HashMap;

use brepkit_math::curves2d::Curve2D;

use crate::edge::EdgeId;
use crate::face::FaceId;

/// Key identifying a pcurve: which edge on which face.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct PCurveKey {
    /// The edge this pcurve belongs to.
    pub edge: EdgeId,
    /// The face whose surface parameter space the pcurve lives in.
    pub face: FaceId,
}

impl PCurveKey {
    /// Creates a new pcurve key.
    #[must_use]
    pub const fn new(edge: EdgeId, face: FaceId) -> Self {
        Self { edge, face }
    }
}

/// A 2D parametric curve on a surface, with parameter bounds.
///
/// The curve is defined in the face's surface (u, v) parameter space.
/// The parameter range `[t_start, t_end]` maps to the edge's 3D start
/// and end vertices (respecting orientation).
#[derive(Debug, Clone)]
pub struct PCurve {
    /// The 2D curve in surface parameter space.
    curve: Curve2D,
    /// Start parameter on the 2D curve.
    t_start: f64,
    /// End parameter on the 2D curve.
    t_end: f64,
}

impl PCurve {
    /// Creates a new pcurve.
    #[must_use]
    pub const fn new(curve: Curve2D, t_start: f64, t_end: f64) -> Self {
        Self {
            curve,
            t_start,
            t_end,
        }
    }

    /// Returns a reference to the 2D curve.
    #[must_use]
    pub const fn curve(&self) -> &Curve2D {
        &self.curve
    }

    /// Returns the start parameter.
    #[must_use]
    pub const fn t_start(&self) -> f64 {
        self.t_start
    }

    /// Returns the end parameter.
    #[must_use]
    pub const fn t_end(&self) -> f64 {
        self.t_end
    }

    /// Evaluates the pcurve at parameter `t`.
    #[must_use]
    pub fn evaluate(&self, t: f64) -> brepkit_math::vec::Point2 {
        self.curve.evaluate(t)
    }
}

/// Registry of pcurves, mapping (edge, face) pairs to 2D curves.
///
/// Each edge on a face can have an associated pcurve that represents
/// the edge's geometry in the face's surface parameter space.
#[derive(Debug, Default, Clone)]
pub struct PCurveRegistry {
    curves: HashMap<PCurveKey, PCurve>,
}

impl PCurveRegistry {
    /// Creates a new, empty pcurve registry.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Sets the pcurve for an (edge, face) pair.
    pub fn set(&mut self, edge: EdgeId, face: FaceId, pcurve: PCurve) {
        self.curves.insert(PCurveKey::new(edge, face), pcurve);
    }

    /// Gets the pcurve for an (edge, face) pair, if one exists.
    #[must_use]
    pub fn get(&self, edge: EdgeId, face: FaceId) -> Option<&PCurve> {
        self.curves.get(&PCurveKey::new(edge, face))
    }

    /// Returns true if a pcurve exists for the given (edge, face) pair.
    #[must_use]
    pub fn contains(&self, edge: EdgeId, face: FaceId) -> bool {
        self.curves.contains_key(&PCurveKey::new(edge, face))
    }

    /// Removes the pcurve for an (edge, face) pair.
    pub fn remove(&mut self, edge: EdgeId, face: FaceId) -> Option<PCurve> {
        self.curves.remove(&PCurveKey::new(edge, face))
    }

    /// Returns the number of pcurves in the registry.
    #[must_use]
    pub fn len(&self) -> usize {
        self.curves.len()
    }

    /// Returns true if the registry is empty.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.curves.is_empty()
    }

    /// Returns all pcurves for a given face.
    #[must_use]
    pub fn pcurves_for_face(&self, face: FaceId) -> Vec<(EdgeId, &PCurve)> {
        self.curves
            .iter()
            .filter(|(k, _)| k.face == face)
            .map(|(k, v)| (k.edge, v))
            .collect()
    }

    /// Returns all pcurves for a given edge (typically 1-2 per edge).
    #[must_use]
    pub fn pcurves_for_edge(&self, edge: EdgeId) -> Vec<(FaceId, &PCurve)> {
        self.curves
            .iter()
            .filter(|(k, _)| k.edge == edge)
            .map(|(k, v)| (k.face, v))
            .collect()
    }
}

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::expect_used)]
mod tests {
    use brepkit_math::curves2d::{Curve2D, Line2D, NurbsCurve2D};
    use brepkit_math::vec::{Point2, Point3, Vec2};

    use crate::edge::{Edge, EdgeCurve};
    use crate::face::{Face, FaceSurface};
    use crate::topology::Topology;
    use crate::vertex::Vertex;
    use crate::wire::{OrientedEdge, Wire};

    use super::*;

    fn make_simple_topology() -> (Topology, EdgeId, FaceId) {
        let mut topo = Topology::new();

        let v0 = topo
            .vertices
            .alloc(Vertex::new(Point3::new(0.0, 0.0, 0.0), 1e-7));
        let v1 = topo
            .vertices
            .alloc(Vertex::new(Point3::new(1.0, 0.0, 0.0), 1e-7));
        let v2 = topo
            .vertices
            .alloc(Vertex::new(Point3::new(1.0, 1.0, 0.0), 1e-7));

        let e0 = topo.edges.alloc(Edge::new(v0, v1, EdgeCurve::Line));
        let e1 = topo.edges.alloc(Edge::new(v1, v2, EdgeCurve::Line));
        let e2 = topo.edges.alloc(Edge::new(v2, v0, EdgeCurve::Line));

        let wire = Wire::new(
            vec![
                OrientedEdge::new(e0, true),
                OrientedEdge::new(e1, true),
                OrientedEdge::new(e2, true),
            ],
            true,
        )
        .unwrap();
        let wire_id = topo.wires.alloc(wire);

        let face = Face::new(
            wire_id,
            vec![],
            FaceSurface::Plane {
                normal: brepkit_math::vec::Vec3::new(0.0, 0.0, 1.0),
                d: 0.0,
            },
        );
        let face_id = topo.faces.alloc(face);

        (topo, e0, face_id)
    }

    #[test]
    fn set_and_get_pcurve() {
        let (mut topo, edge_id, face_id) = make_simple_topology();

        let line = Line2D::new(Point2::new(0.0, 0.0), Vec2::new(1.0, 0.0)).unwrap();
        let pcurve = PCurve::new(Curve2D::Line(line), 0.0, 1.0);

        topo.pcurves.set(edge_id, face_id, pcurve);

        assert!(topo.pcurves.contains(edge_id, face_id));
        let pc = topo.pcurves.get(edge_id, face_id).unwrap();
        assert!((pc.t_start() - 0.0).abs() < f64::EPSILON);
        assert!((pc.t_end() - 1.0).abs() < f64::EPSILON);
    }

    #[test]
    fn pcurve_evaluate() {
        // Line direction is normalized, so (1,0) stays (1,0)
        let line = Line2D::new(Point2::new(0.0, 0.0), Vec2::new(1.0, 0.0)).unwrap();
        let pcurve = PCurve::new(Curve2D::Line(line), 0.0, 1.0);

        let p = pcurve.evaluate(0.5);
        assert!((p.x() - 0.5).abs() < 1e-10);
        assert!((p.y() - 0.0).abs() < 1e-10);
    }

    #[test]
    fn pcurve_registry_empty() {
        let registry = PCurveRegistry::new();
        assert!(registry.is_empty());
        assert_eq!(registry.len(), 0);
    }

    #[test]
    fn pcurve_remove() {
        let (mut topo, edge_id, face_id) = make_simple_topology();

        let line = Line2D::new(Point2::new(0.0, 0.0), Vec2::new(1.0, 0.0)).unwrap();
        topo.pcurves
            .set(edge_id, face_id, PCurve::new(Curve2D::Line(line), 0.0, 1.0));

        assert!(topo.pcurves.contains(edge_id, face_id));
        topo.pcurves.remove(edge_id, face_id);
        assert!(!topo.pcurves.contains(edge_id, face_id));
    }

    #[test]
    fn pcurves_for_face() {
        let (mut topo, edge_id, face_id) = make_simple_topology();

        let line = Line2D::new(Point2::new(0.0, 0.0), Vec2::new(1.0, 0.0)).unwrap();
        topo.pcurves
            .set(edge_id, face_id, PCurve::new(Curve2D::Line(line), 0.0, 1.0));

        let pcurves = topo.pcurves.pcurves_for_face(face_id);
        assert_eq!(pcurves.len(), 1);
        assert_eq!(pcurves[0].0, edge_id);
    }

    #[test]
    fn nurbs_pcurve() {
        let curve = NurbsCurve2D::from_line(Point2::new(0.0, 0.0), Point2::new(1.0, 1.0)).unwrap();
        let pcurve = PCurve::new(Curve2D::Nurbs(curve), 0.0, 1.0);

        let p = pcurve.evaluate(0.5);
        assert!((p.x() - 0.5).abs() < 1e-10);
        assert!((p.y() - 0.5).abs() < 1e-10);
    }
}
