//! Wire analysis — edge ordering, closure, gaps, self-intersection.

use brepkit_math::tolerance::Tolerance;
use brepkit_topology::Topology;
use brepkit_topology::edge::EdgeId;
use brepkit_topology::wire::WireId;

use crate::HealError;
use crate::status::Status;

/// Number of samples per edge for self-intersection checks.
const SELF_INTERSECT_SAMPLES: usize = 8;

/// Result of analyzing a single wire.
#[derive(Debug, Clone)]
pub struct WireAnalysis {
    /// Number of oriented edges in the wire.
    pub edge_count: usize,
    /// Whether the last edge endpoint connects back to the first edge start.
    pub is_closed: bool,
    /// Whether consecutive edges share matching vertices.
    pub is_ordered: bool,
    /// Gaps between consecutive edges (non-matching endpoints).
    pub gaps: Vec<WireGap>,
    /// Edges shorter than the tolerance.
    pub small_edges: Vec<SmallEdge>,
    /// Pairs of non-adjacent edge indices whose sample points overlap.
    pub self_intersections: Vec<(usize, usize)>,
    /// Indices of degenerate edges (closed with zero length).
    pub degenerate_edges: Vec<usize>,
    /// Outcome status flags.
    pub status: Status,
}

/// A gap between two consecutive edges in a wire.
#[derive(Debug, Clone)]
pub struct WireGap {
    /// Index of the first edge in the gap pair.
    pub edge_index: usize,
    /// Index of the second edge in the gap pair.
    pub next_edge_index: usize,
    /// Distance between the endpoint of the first edge and the start
    /// of the second edge.
    pub distance: f64,
}

/// An edge that is shorter than the analysis tolerance.
#[derive(Debug, Clone)]
pub struct SmallEdge {
    /// Index of this edge in the wire's edge list.
    pub edge_index: usize,
    /// The edge's topology handle.
    pub edge_id: EdgeId,
    /// Approximate arc length of the edge.
    pub length: f64,
}

/// Analyze a wire for ordering, closure, gaps, small/degenerate edges,
/// and self-intersection.
///
/// # Errors
///
/// Returns [`HealError`] if entity lookups fail.
#[allow(clippy::too_many_lines)]
pub fn analyze_wire(
    topo: &Topology,
    wire_id: WireId,
    tolerance: &Tolerance,
) -> Result<WireAnalysis, HealError> {
    let wire = topo.wire(wire_id)?;
    let oe_list = wire.edges();
    let edge_count = oe_list.len();

    if edge_count == 0 {
        return Ok(WireAnalysis {
            edge_count: 0,
            is_closed: false,
            is_ordered: true,
            gaps: Vec::new(),
            small_edges: Vec::new(),
            self_intersections: Vec::new(),
            degenerate_edges: Vec::new(),
            status: Status::OK,
        });
    }

    // Collect oriented start/end vertex positions for each edge.
    let mut starts = Vec::with_capacity(edge_count);
    let mut ends = Vec::with_capacity(edge_count);
    let mut lengths = Vec::with_capacity(edge_count);

    for oe in oe_list {
        let edge = topo.edge(oe.edge())?;
        let s = topo.vertex(oe.oriented_start(edge))?.point();
        let e = topo.vertex(oe.oriented_end(edge))?.point();
        starts.push(s);
        ends.push(e);

        // Approximate arc length.
        let start_pos = topo.vertex(edge.start())?.point();
        let end_pos = topo.vertex(edge.end())?.point();
        let (t_min, t_max) = edge.curve().domain_with_endpoints(start_pos, end_pos);
        let samples = 16;
        let mut len = 0.0;
        let mut prev = edge
            .curve()
            .evaluate_with_endpoints(t_min, start_pos, end_pos);
        for i in 1..=samples {
            let t = t_min + (t_max - t_min) * (i as f64 / samples as f64);
            let pt = edge.curve().evaluate_with_endpoints(t, start_pos, end_pos);
            len += (pt - prev).length();
            prev = pt;
        }
        lengths.push(len);
    }

    // Check ordering and gaps.
    let mut is_ordered = true;
    let mut gaps = Vec::new();
    for i in 0..edge_count.saturating_sub(1) {
        let dist = (starts[i + 1] - ends[i]).length();
        if dist > tolerance.linear {
            is_ordered = false;
            gaps.push(WireGap {
                edge_index: i,
                next_edge_index: i + 1,
                distance: dist,
            });
        }
    }

    // Check closure (last end -> first start).
    let closure_dist = (starts[0] - ends[edge_count - 1]).length();
    let is_closed = closure_dist <= tolerance.linear;
    if !is_closed && wire.is_closed() {
        is_ordered = false;
        gaps.push(WireGap {
            edge_index: edge_count - 1,
            next_edge_index: 0,
            distance: closure_dist,
        });
    }

    // Small edges.
    let mut small_edges = Vec::new();
    for (i, oe) in oe_list.iter().enumerate() {
        if lengths[i] < tolerance.linear {
            small_edges.push(SmallEdge {
                edge_index: i,
                edge_id: oe.edge(),
                length: lengths[i],
            });
        }
    }

    // Degenerate edges (closed vertex + zero length).
    let mut degenerate_edges = Vec::new();
    for (i, oe) in oe_list.iter().enumerate() {
        let edge = topo.edge(oe.edge())?;
        if edge.is_closed() && lengths[i] < tolerance.linear {
            degenerate_edges.push(i);
        }
    }

    // Self-intersection: sample each edge and check for proximity between
    // non-adjacent edge pairs.
    let self_intersections = detect_self_intersections(topo, oe_list, edge_count, tolerance)?;

    let mut status = Status::OK;
    if !gaps.is_empty() {
        status = status.merge(Status::DONE1);
    }
    if !small_edges.is_empty() {
        status = status.merge(Status::DONE2);
    }
    if !self_intersections.is_empty() {
        status = status.merge(Status::DONE3);
    }
    if !degenerate_edges.is_empty() {
        status = status.merge(Status::DONE4);
    }

    Ok(WireAnalysis {
        edge_count,
        is_closed,
        is_ordered,
        gaps,
        small_edges,
        self_intersections,
        degenerate_edges,
        status,
    })
}

/// Detect self-intersections by sampling edges and checking proximity.
fn detect_self_intersections(
    topo: &Topology,
    oe_list: &[brepkit_topology::wire::OrientedEdge],
    edge_count: usize,
    tolerance: &Tolerance,
) -> Result<Vec<(usize, usize)>, HealError> {
    if edge_count < 3 {
        return Ok(Vec::new());
    }

    // Sample points for each edge.
    let mut edge_samples: Vec<Vec<brepkit_math::vec::Point3>> = Vec::with_capacity(edge_count);
    for oe in oe_list {
        let edge = topo.edge(oe.edge())?;
        let start_pos = topo.vertex(edge.start())?.point();
        let end_pos = topo.vertex(edge.end())?.point();
        let (t_min, t_max) = edge.curve().domain_with_endpoints(start_pos, end_pos);

        let mut samples = Vec::with_capacity(SELF_INTERSECT_SAMPLES);
        for j in 0..SELF_INTERSECT_SAMPLES {
            let t = t_min + (t_max - t_min) * (j as f64 / (SELF_INTERSECT_SAMPLES - 1) as f64);
            samples.push(edge.curve().evaluate_with_endpoints(t, start_pos, end_pos));
        }
        edge_samples.push(samples);
    }

    let mut intersections = Vec::new();
    let tol_sq = tolerance.linear * tolerance.linear;

    for i in 0..edge_count {
        // Skip adjacent edges (they share a vertex naturally).
        for j in (i + 2)..edge_count {
            // Also skip the wrap-around adjacency for closed wires.
            if i == 0 && j == edge_count - 1 {
                continue;
            }
            if samples_overlap(&edge_samples[i], &edge_samples[j], tol_sq) {
                intersections.push((i, j));
            }
        }
    }

    Ok(intersections)
}

/// Check if any sample point from edge A is within `tol_sq` of any sample
/// point from edge B (excluding shared endpoint proximity).
fn samples_overlap(
    a: &[brepkit_math::vec::Point3],
    b: &[brepkit_math::vec::Point3],
    tol_sq: f64,
) -> bool {
    // Skip first and last samples (they are at the edge endpoints, which
    // might coincide with shared vertices in a closed wire).
    let a_inner = if a.len() > 2 { &a[1..a.len() - 1] } else { a };
    let b_inner = if b.len() > 2 { &b[1..b.len() - 1] } else { b };

    for pa in a_inner {
        for pb in b_inner {
            if (*pa - *pb).length_squared() < tol_sq {
                return true;
            }
        }
    }
    false
}
