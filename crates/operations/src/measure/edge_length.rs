//! Edge and wire length computation.

use brepkit_topology::Topology;
use brepkit_topology::face::FaceId;

/// Compute the length of a single edge.
///
/// For line edges, returns the Euclidean distance between endpoints.
/// For NURBS curve edges, uses numerical integration (Simpson's rule).
///
/// # Errors
///
/// Returns an error if the edge lookup fails.
pub fn edge_length(
    topo: &Topology,
    edge_id: brepkit_topology::edge::EdgeId,
) -> Result<f64, crate::OperationsError> {
    let edge = topo.edge(edge_id)?;
    match edge.curve() {
        brepkit_topology::edge::EdgeCurve::Line => {
            let start = topo.vertex(edge.start())?.point();
            let end = topo.vertex(edge.end())?.point();
            Ok((end - start).length())
        }
        brepkit_topology::edge::EdgeCurve::NurbsCurve(curve) => Ok(curve.arc_length(50)),
        brepkit_topology::edge::EdgeCurve::Circle(circle) => {
            if edge.is_closed() {
                Ok(circle.circumference())
            } else {
                let start = topo.vertex(edge.start())?.point();
                let end = topo.vertex(edge.end())?.point();
                let t0 = circle.project(start);
                let t1 = circle.project(end);
                let mut angle = t1 - t0;
                if angle < 0.0 {
                    angle += std::f64::consts::TAU;
                }
                Ok(angle * circle.radius())
            }
        }
        brepkit_topology::edge::EdgeCurve::Ellipse(ellipse) => {
            if edge.is_closed() {
                Ok(ellipse.approximate_circumference())
            } else {
                // Approximate arc length via sampling
                let start = topo.vertex(edge.start())?.point();
                let end = topo.vertex(edge.end())?.point();
                let t0 = ellipse.project(start);
                let t1 = ellipse.project(end);
                let mut angle = t1 - t0;
                if angle < 0.0 {
                    angle += std::f64::consts::TAU;
                }
                let n = 50;
                let dt = angle / n as f64;
                let mut length = 0.0;
                let mut prev = ellipse.evaluate(t0);
                for i in 1..=n {
                    let t = t0 + dt * i as f64;
                    let curr = ellipse.evaluate(t);
                    length += (curr - prev).length();
                    prev = curr;
                }
                Ok(length)
            }
        }
    }
}

/// Compute the total length (perimeter) of a wire.
///
/// Sums the length of all edges in the wire.
///
/// # Errors
///
/// Returns an error if any edge lookup fails.
pub fn wire_length(
    topo: &Topology,
    wire_id: brepkit_topology::wire::WireId,
) -> Result<f64, crate::OperationsError> {
    let wire = topo.wire(wire_id)?;
    let mut total = 0.0;
    for oe in wire.edges() {
        total += edge_length(topo, oe.edge())?;
    }
    Ok(total)
}

/// Compute the perimeter of a face (outer wire length).
///
/// # Errors
///
/// Returns an error if topology lookups fail.
pub fn face_perimeter(topo: &Topology, face_id: FaceId) -> Result<f64, crate::OperationsError> {
    let face = topo.face(face_id)?;
    wire_length(topo, face.outer_wire())
}
