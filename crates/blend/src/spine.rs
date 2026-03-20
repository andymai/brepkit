//! Spine: ordered edge chain with arc-length parameterization.
//!
//! A spine represents the guideline along which a fillet or chamfer is
//! computed. It may consist of multiple edges forming a G1-continuous chain.

use brepkit_math::vec::{Point3, Vec3};
use brepkit_topology::Topology;
use brepkit_topology::edge::EdgeId;

use crate::BlendError;

/// An ordered chain of edges forming the fillet guideline.
#[derive(Debug, Clone)]
pub struct Spine {
    /// Ordered edge IDs in the chain.
    edges: Vec<EdgeId>,
    /// Cumulative arc-length at each edge boundary.
    /// `params[0] = 0`, `params[i]` = cumulative length through edge `i-1`.
    params: Vec<f64>,
    /// Total arc length of the spine.
    length: f64,
    /// Whether the chain forms a closed loop.
    is_closed: bool,
}

/// Get the chord-length endpoints of an edge.
fn edge_endpoints(topo: &Topology, edge_id: EdgeId) -> Result<(Point3, Point3), BlendError> {
    let edge = topo.edge(edge_id)?;
    let p_start = topo.vertex(edge.start())?.point();
    let p_end = topo.vertex(edge.end())?.point();
    Ok((p_start, p_end))
}

impl Spine {
    /// Build a spine from a single edge.
    ///
    /// # Errors
    /// Returns `BlendError` if the edge or its vertices cannot be found.
    pub fn from_single_edge(topo: &Topology, edge_id: EdgeId) -> Result<Self, BlendError> {
        let (p_start, p_end) = edge_endpoints(topo, edge_id)?;
        let length = (p_end - p_start).length();

        Ok(Self {
            edges: vec![edge_id],
            params: vec![0.0, length],
            length,
            is_closed: false,
        })
    }

    /// Build a spine from an ordered chain of edges.
    ///
    /// Edges must be G1-continuous (verified by caller).
    ///
    /// # Errors
    /// Returns `BlendError` if any edge or vertex cannot be found.
    pub fn from_chain(topo: &Topology, edges: Vec<EdgeId>) -> Result<Self, BlendError> {
        let mut params = Vec::with_capacity(edges.len() + 1);
        params.push(0.0);
        let mut cumulative = 0.0;

        for &eid in &edges {
            let (p_start, p_end) = edge_endpoints(topo, eid)?;
            cumulative += (p_end - p_start).length();
            params.push(cumulative);
        }

        let is_closed = if edges.len() >= 2 {
            let first = topo.edge(edges[0])?;
            let last = topo.edge(edges[edges.len() - 1])?;
            first.start() == last.end()
        } else {
            false
        };

        Ok(Self {
            edges,
            params,
            length: cumulative,
            is_closed,
        })
    }

    /// Total arc length.
    #[must_use]
    pub fn length(&self) -> f64 {
        self.length
    }

    /// Number of edges in the chain.
    #[must_use]
    pub fn edge_count(&self) -> usize {
        self.edges.len()
    }

    /// Whether the spine forms a closed loop.
    #[must_use]
    pub fn is_closed(&self) -> bool {
        self.is_closed
    }

    /// The edges in order.
    #[must_use]
    pub fn edges(&self) -> &[EdgeId] {
        &self.edges
    }

    /// Map a global spine parameter `s in [0, length]` to `(edge_index, local_t in [0,1])`.
    #[must_use]
    pub fn locate(&self, s: f64) -> (usize, f64) {
        let s_clamped = s.clamp(0.0, self.length);
        for i in 0..self.edges.len() {
            let s0 = self.params[i];
            let s1 = self.params[i + 1];
            if s_clamped <= s1 || i == self.edges.len() - 1 {
                let edge_len = s1 - s0;
                let t = if edge_len > f64::EPSILON {
                    (s_clamped - s0) / edge_len
                } else {
                    0.0
                };
                return (i, t.clamp(0.0, 1.0));
            }
        }
        (self.edges.len() - 1, 1.0)
    }

    /// Evaluate the 3D point on the spine at global parameter `s`.
    ///
    /// # Errors
    /// Returns `BlendError` if topology lookups fail.
    pub fn evaluate(&self, topo: &Topology, s: f64) -> Result<Point3, BlendError> {
        let (idx, t) = self.locate(s);
        let (p0, p1) = edge_endpoints(topo, self.edges[idx])?;
        // Linear interpolation for now; curved edges need curve evaluation
        Ok(p0 + (p1 - p0) * t)
    }

    /// Evaluate the tangent direction on the spine at global parameter `s`.
    ///
    /// Returns the unit tangent, or a fallback Z-axis if the edge is degenerate.
    ///
    /// # Errors
    /// Returns `BlendError` if topology lookups fail.
    pub fn tangent(&self, topo: &Topology, s: f64) -> Result<Vec3, BlendError> {
        let (idx, _t) = self.locate(s);
        let (p0, p1) = edge_endpoints(topo, self.edges[idx])?;
        Ok((p1 - p0).normalize().unwrap_or(Vec3::new(0.0, 0.0, 1.0)))
    }
}

#[cfg(test)]
mod tests {
    #![allow(clippy::unwrap_used, clippy::expect_used)]
    use super::*;
    use brepkit_topology::Topology;
    use brepkit_topology::edge::{Edge, EdgeCurve};
    use brepkit_topology::vertex::Vertex;

    fn make_line_edge(topo: &mut Topology, a: Point3, b: Point3) -> EdgeId {
        let v0 = topo.add_vertex(Vertex::new(a, 1e-7));
        let v1 = topo.add_vertex(Vertex::new(b, 1e-7));
        topo.add_edge(Edge::new(v0, v1, EdgeCurve::Line))
    }

    #[test]
    fn single_edge_spine_length() {
        let mut topo = Topology::new();
        let eid = make_line_edge(
            &mut topo,
            Point3::new(0.0, 0.0, 0.0),
            Point3::new(10.0, 0.0, 0.0),
        );
        let spine = Spine::from_single_edge(&topo, eid).unwrap();
        assert!((spine.length() - 10.0).abs() < 1e-10);
        assert_eq!(spine.edge_count(), 1);
        assert!(!spine.is_closed());
    }

    #[test]
    fn locate_maps_parameter_correctly() {
        let mut topo = Topology::new();
        let eid = make_line_edge(
            &mut topo,
            Point3::new(0.0, 0.0, 0.0),
            Point3::new(10.0, 0.0, 0.0),
        );
        let spine = Spine::from_single_edge(&topo, eid).unwrap();
        let (idx, t) = spine.locate(5.0);
        assert_eq!(idx, 0);
        assert!((t - 0.5).abs() < 1e-10);
    }

    #[test]
    fn evaluate_midpoint() {
        let mut topo = Topology::new();
        let eid = make_line_edge(
            &mut topo,
            Point3::new(0.0, 0.0, 0.0),
            Point3::new(10.0, 0.0, 0.0),
        );
        let spine = Spine::from_single_edge(&topo, eid).unwrap();
        let mid = spine.evaluate(&topo, 5.0).unwrap();
        assert!((mid - Point3::new(5.0, 0.0, 0.0)).length() < 1e-10);
    }
}
