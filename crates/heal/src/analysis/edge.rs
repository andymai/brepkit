//! Edge analysis — vertex-curve deviation, degeneracy, arc length.

use brepkit_math::tolerance::Tolerance;
use brepkit_topology::Topology;
use brepkit_topology::edge::EdgeId;

use crate::HealError;
use crate::status::Status;

/// Number of sample points used to approximate arc length.
const ARC_LENGTH_SAMPLES: usize = 16;

/// Result of analyzing a single edge.
#[derive(Debug, Clone)]
pub struct EdgeAnalysis {
    /// Whether the edge has a 3D curve (non-Line variants always do).
    pub has_curve_3d: bool,
    /// Whether the edge is degenerate (start == end vertex and zero-length curve).
    pub is_degenerate: bool,
    /// Distance from the edge's 3D curve start to the start vertex position.
    pub vertex_start_deviation: f64,
    /// Distance from the edge's 3D curve end to the end vertex position.
    pub vertex_end_deviation: f64,
    /// Approximate arc length computed by chord-length summation.
    pub curve_length_approx: f64,
    /// Outcome status flags.
    pub status: Status,
}

/// Returns `true` if the edge has stored 3D curve geometry.
///
/// `Line` edges derive geometry from their vertices, so this returns
/// `false` for them. All other variants (`NurbsCurve`, `Circle`,
/// `Ellipse`) carry explicit geometry.
///
/// # Errors
///
/// Returns [`HealError`] if the edge ID is invalid.
pub fn has_curve_3d(topo: &Topology, edge_id: EdgeId) -> Result<bool, HealError> {
    let edge = topo.edge(edge_id)?;
    Ok(!matches!(
        edge.curve(),
        brepkit_topology::edge::EdgeCurve::Line
    ))
}

/// Returns `true` if a pcurve exists for the given edge on the given face.
///
/// # Errors
///
/// Returns [`HealError`] if the edge or face ID is invalid.
pub fn has_pcurve(
    topo: &Topology,
    edge_id: EdgeId,
    face_id: brepkit_topology::face::FaceId,
) -> Result<bool, HealError> {
    // Validate that both entities exist.
    let _ = topo.edge(edge_id)?;
    let _ = topo.face(face_id)?;
    Ok(topo.pcurves().contains(edge_id, face_id))
}

/// Returns `true` if the edge is a seam edge on the given face.
///
/// A seam edge appears both forward and reverse in the same face's wires.
///
/// # Errors
///
/// Returns [`HealError`] if entity lookups fail.
pub fn is_seam(
    topo: &Topology,
    edge_id: EdgeId,
    face_id: brepkit_topology::face::FaceId,
) -> Result<bool, HealError> {
    let face = topo.face(face_id)?;
    let mut forward_count = 0u32;
    let mut reverse_count = 0u32;

    let wire_ids: Vec<_> = std::iter::once(face.outer_wire())
        .chain(face.inner_wires().iter().copied())
        .collect();

    for wid in wire_ids {
        let wire = topo.wire(wid)?;
        for oe in wire.edges() {
            if oe.edge() == edge_id {
                if oe.is_forward() {
                    forward_count += 1;
                } else {
                    reverse_count += 1;
                }
            }
        }
    }

    Ok(forward_count > 0 && reverse_count > 0)
}

/// Computes the deviation between vertex positions and the curve endpoints.
///
/// Returns `(start_deviation, end_deviation)` as Euclidean distances.
///
/// # Errors
///
/// Returns [`HealError`] if entity lookups fail.
pub fn vertex_curve_deviation(topo: &Topology, edge_id: EdgeId) -> Result<(f64, f64), HealError> {
    let edge = topo.edge(edge_id)?;
    let start_pos = topo.vertex(edge.start())?.point();
    let end_pos = topo.vertex(edge.end())?.point();

    let (t_min, t_max) = edge.curve().domain_with_endpoints(start_pos, end_pos);
    let curve_start = edge
        .curve()
        .evaluate_with_endpoints(t_min, start_pos, end_pos);
    let curve_end = edge
        .curve()
        .evaluate_with_endpoints(t_max, start_pos, end_pos);

    Ok((
        (curve_start - start_pos).length(),
        (curve_end - end_pos).length(),
    ))
}

/// Approximate the arc length of an edge by chord-length summation.
fn approx_arc_length(topo: &Topology, edge_id: EdgeId) -> Result<f64, HealError> {
    let edge = topo.edge(edge_id)?;
    let start_pos = topo.vertex(edge.start())?.point();
    let end_pos = topo.vertex(edge.end())?.point();
    let (t_min, t_max) = edge.curve().domain_with_endpoints(start_pos, end_pos);

    let mut length = 0.0;
    let mut prev = edge
        .curve()
        .evaluate_with_endpoints(t_min, start_pos, end_pos);
    for i in 1..=ARC_LENGTH_SAMPLES {
        let t = t_min + (t_max - t_min) * (i as f64 / ARC_LENGTH_SAMPLES as f64);
        let pt = edge.curve().evaluate_with_endpoints(t, start_pos, end_pos);
        length += (pt - prev).length();
        prev = pt;
    }
    Ok(length)
}

/// Perform a full analysis of the given edge.
///
/// # Errors
///
/// Returns [`HealError`] if entity lookups fail.
pub fn analyze_edge(
    topo: &Topology,
    edge_id: EdgeId,
    tolerance: &Tolerance,
) -> Result<EdgeAnalysis, HealError> {
    let has_3d = has_curve_3d(topo, edge_id)?;
    let (start_dev, end_dev) = vertex_curve_deviation(topo, edge_id)?;
    let arc_len = approx_arc_length(topo, edge_id)?;

    let edge = topo.edge(edge_id)?;
    let is_degenerate = edge.is_closed() && arc_len < tolerance.linear;

    let mut status = Status::OK;
    if start_dev > tolerance.linear || end_dev > tolerance.linear {
        status = status.merge(Status::DONE1);
    }
    if is_degenerate {
        status = status.merge(Status::DONE2);
    }

    Ok(EdgeAnalysis {
        has_curve_3d: has_3d,
        is_degenerate,
        vertex_start_deviation: start_dev,
        vertex_end_deviation: end_dev,
        curve_length_approx: arc_len,
        status,
    })
}
