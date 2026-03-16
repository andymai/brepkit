//! 2D wire construction from edge soup via angular-sorting traversal.
//!
//! Given an unordered set of [`OrientedPCurveEdge`]s on a face, builds
//! minimal closed wire loops by traversing edges using the minimum
//! clockwise angle rule at each vertex.
//!
//! Port of OCCT's `BOPAlgo_WireSplitter_1.cxx` algorithm, simplified
//! for the boolean_v2 pipeline.

#![allow(dead_code)] // Used by later boolean_v2 pipeline stages.

use std::collections::HashMap;
use std::f64::consts::TAU;

use brepkit_math::vec::Point2;

use super::pipeline::OrientedPCurveEdge;

// ---------------------------------------------------------------------------
// Types
// ---------------------------------------------------------------------------

/// An entry in the vertex adjacency map.
struct VertexEntry {
    /// Index into the input edge slice.
    edge_idx: usize,
    /// `true` if this vertex is the START of the edge (outgoing).
    outgoing: bool,
    /// Tangent angle at this vertex in \[0, 2π).
    angle: f64,
}

// ---------------------------------------------------------------------------
// Public API
// ---------------------------------------------------------------------------

/// Build closed wire loops from an unordered set of 2D edges.
///
/// Each edge has start/end UV coordinates. The algorithm:
/// 1. Builds a vertex adjacency map (quantized UV endpoints).
/// 2. Computes outgoing angles for each edge at each vertex.
/// 3. Traverses edges using the minimum clockwise angle rule.
/// 4. Returns a list of closed loops, each a `Vec<OrientedPCurveEdge>`.
///
/// `tol` is the UV-space tolerance for vertex deduplication.
/// `_u_periodic` and `_v_periodic` are reserved for future periodic surfaces.
pub fn build_wire_loops(
    edges: &[OrientedPCurveEdge],
    tol: f64,
    _u_periodic: bool,
    _v_periodic: bool,
) -> Vec<Vec<OrientedPCurveEdge>> {
    if edges.is_empty() {
        return Vec::new();
    }

    // 1. Build vertex adjacency map.
    let mut adj: HashMap<(i64, i64), Vec<VertexEntry>> = HashMap::new();
    for (idx, edge) in edges.iter().enumerate() {
        let start_key = quantize_uv(edge.start_uv, tol);
        let end_key = quantize_uv(edge.end_uv, tol);
        let angle_out = edge_angle_at_vertex(edge, true);
        let angle_in = edge_angle_at_vertex(edge, false);
        adj.entry(start_key).or_default().push(VertexEntry {
            edge_idx: idx,
            outgoing: true,
            angle: angle_out,
        });
        adj.entry(end_key).or_default().push(VertexEntry {
            edge_idx: idx,
            outgoing: false,
            angle: angle_in,
        });
    }

    // 2. Greedy traversal.
    let mut used = vec![false; edges.len()];
    let mut loops = Vec::new();

    loop {
        // Find first unused edge to start a new loop.
        let Some(start_idx) = used.iter().position(|u| !u) else {
            break;
        };
        used[start_idx] = true;

        let start_vertex = quantize_uv(edges[start_idx].start_uv, tol);
        let mut current_loop = vec![edges[start_idx].clone()];
        let mut current_idx = start_idx;

        // Walk edges until we close the loop.
        loop {
            let current_edge = &edges[current_idx];
            let end_vertex = quantize_uv(current_edge.end_uv, tol);

            // Check for loop closure.
            if end_vertex == start_vertex {
                loops.push(current_loop);
                break;
            }

            // Incoming angle at the end vertex.
            let incoming_angle = edge_angle_at_vertex(current_edge, false);
            let arriving_start = quantize_uv(current_edge.start_uv, tol);

            // Find best outgoing edge at end_vertex.
            let Some(entries) = adj.get(&end_vertex) else {
                // Dead end — discard this incomplete loop.
                break;
            };

            let mut best_cw = f64::MAX;
            let mut best_idx = None;
            for entry in entries {
                if !entry.outgoing || used[entry.edge_idx] {
                    continue;
                }
                // Skip the reverse of the arriving edge (prevents U-turns
                // along section edges that appear as forward+backward pairs).
                let candidate = &edges[entry.edge_idx];
                if quantize_uv(candidate.end_uv, tol) == arriving_start {
                    // Check if this is truly the reverse edge (same line, opposite direction).
                    let angle_diff = (entry.angle - incoming_angle).abs();
                    let is_reverse = angle_diff < 0.1 || (angle_diff - TAU).abs() < 0.1;
                    if is_reverse {
                        continue;
                    }
                }

                let cw = clockwise_angle(incoming_angle, entry.angle);
                if cw < best_cw {
                    best_cw = cw;
                    best_idx = Some(entry.edge_idx);
                }
            }

            let Some(next_idx) = best_idx else {
                // No way out (all edges used or only reverse available).
                // Try again without the reverse-edge skip.
                let fallback = entries
                    .iter()
                    .filter(|e| e.outgoing && !used[e.edge_idx])
                    .min_by(|a, b| {
                        clockwise_angle(incoming_angle, a.angle)
                            .partial_cmp(&clockwise_angle(incoming_angle, b.angle))
                            .unwrap_or(std::cmp::Ordering::Equal)
                    });
                if let Some(fb) = fallback {
                    used[fb.edge_idx] = true;
                    current_loop.push(edges[fb.edge_idx].clone());
                    current_idx = fb.edge_idx;
                    continue;
                }
                // Truly no way out — discard loop.
                break;
            };

            used[next_idx] = true;
            current_loop.push(edges[next_idx].clone());
            current_idx = next_idx;
        }
    }

    loops
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

/// Quantize a 2D point to an integer key for vertex deduplication.
fn quantize_uv(p: Point2, tol: f64) -> (i64, i64) {
    let resolution = 1.0 / tol;
    (
        (p.x() * resolution).round() as i64,
        (p.y() * resolution).round() as i64,
    )
}

/// Compute the outgoing angle of an edge at a vertex in UV space.
///
/// Returns the angle in \[0, 2π) of the tangent direction at the vertex.
///
/// - `at_start = true`: outgoing direction (start → end).
/// - `at_start = false`: incoming direction (end → start, pointing back along the edge).
fn edge_angle_at_vertex(edge: &OrientedPCurveEdge, at_start: bool) -> f64 {
    let (dx, dy) = if at_start {
        (
            edge.end_uv.x() - edge.start_uv.x(),
            edge.end_uv.y() - edge.start_uv.y(),
        )
    } else {
        (
            edge.start_uv.x() - edge.end_uv.x(),
            edge.start_uv.y() - edge.end_uv.y(),
        )
    };
    let angle = dy.atan2(dx);
    if angle < 0.0 { angle + TAU } else { angle }
}

/// Compute the clockwise sweep angle from `angle_in` to `angle_out`.
///
/// `angle_in` is the incoming edge's angle at the vertex (points back along
/// the arriving edge). `angle_out` is the candidate outgoing edge's angle.
///
/// The formula computes the CCW angle from `angle_out` to the travel
/// direction (`angle_in + π`). Minimum value = rightmost turn = traces
/// minimal enclosed regions.
///
/// Returns a value in (0, 2π].
fn clockwise_angle(angle_in: f64, angle_out: f64) -> f64 {
    // Travel direction = opposite of incoming (which points backward).
    let travel = (angle_in + std::f64::consts::PI).rem_euclid(TAU);
    let da = (travel - angle_out).rem_euclid(TAU);
    if da < 1e-14 {
        TAU // Exact continuation → maximum angle.
    } else {
        da
    }
}

#[cfg(test)]
mod tests {
    #![allow(clippy::unwrap_used, clippy::expect_used)]
    use super::*;
    use brepkit_math::curves2d::{Curve2D, Line2D};
    use brepkit_math::vec::Vec2;
    use brepkit_topology::edge::EdgeCurve;

    fn make_line_edge(start: Point2, end: Point2) -> OrientedPCurveEdge {
        let dir = Vec2::new(end.x() - start.x(), end.y() - start.y());
        let pcurve =
            Curve2D::Line(Line2D::new(start, dir).expect("non-zero direction for test edge"));
        OrientedPCurveEdge {
            curve_3d: EdgeCurve::Line,
            pcurve,
            start_uv: start,
            end_uv: end,
            start_3d: brepkit_math::vec::Point3::new(start.x(), start.y(), 0.0),
            end_3d: brepkit_math::vec::Point3::new(end.x(), end.y(), 0.0),
            forward: true,
        }
    }

    #[test]
    fn square_cut_by_vertical_line_produces_two_loops() {
        // Square boundary [0,10]×[0,10] split by vertical line at x=5.
        let edges = vec![
            make_line_edge(Point2::new(0.0, 0.0), Point2::new(5.0, 0.0)),
            make_line_edge(Point2::new(5.0, 0.0), Point2::new(10.0, 0.0)),
            make_line_edge(Point2::new(10.0, 0.0), Point2::new(10.0, 10.0)),
            make_line_edge(Point2::new(10.0, 10.0), Point2::new(5.0, 10.0)),
            make_line_edge(Point2::new(5.0, 10.0), Point2::new(0.0, 10.0)),
            make_line_edge(Point2::new(0.0, 10.0), Point2::new(0.0, 0.0)),
            // Cutting line in both directions
            make_line_edge(Point2::new(5.0, 0.0), Point2::new(5.0, 10.0)),
            make_line_edge(Point2::new(5.0, 10.0), Point2::new(5.0, 0.0)),
        ];

        let loops = build_wire_loops(&edges, 1e-7, false, false);

        assert_eq!(loops.len(), 2, "expected 2 loops, got {}", loops.len());
        for (i, lp) in loops.iter().enumerate() {
            assert_eq!(
                lp.len(),
                4,
                "loop {i} should have 4 edges, got {}",
                lp.len()
            );
        }
    }

    #[test]
    fn single_square_returns_one_loop() {
        let edges = vec![
            make_line_edge(Point2::new(0.0, 0.0), Point2::new(10.0, 0.0)),
            make_line_edge(Point2::new(10.0, 0.0), Point2::new(10.0, 10.0)),
            make_line_edge(Point2::new(10.0, 10.0), Point2::new(0.0, 10.0)),
            make_line_edge(Point2::new(0.0, 10.0), Point2::new(0.0, 0.0)),
        ];

        let loops = build_wire_loops(&edges, 1e-7, false, false);

        assert_eq!(loops.len(), 1, "expected 1 loop, got {}", loops.len());
        assert_eq!(loops[0].len(), 4);
    }

    #[test]
    fn clockwise_angle_basics() {
        use std::f64::consts::PI;
        // Incoming from left (angle_in = π, points leftward).
        // Travel direction = π + π = 0 (rightward).
        //
        // Outgoing UP (π/2): dA = 0 - π/2 = -π/2 → 3π/2.
        let cw = clockwise_angle(PI, PI / 2.0);
        assert!((cw - 3.0 * PI / 2.0).abs() < 1e-10, "cw = {cw}");

        // Outgoing DOWN (3π/2): dA = 0 - 3π/2 = -3π/2 → π/2.
        let cw2 = clockwise_angle(PI, 3.0 * PI / 2.0);
        assert!((cw2 - PI / 2.0).abs() < 1e-10, "cw2 = {cw2}");

        // Outgoing RIGHT (0) = same as travel: dA = 0 → 2π.
        let cw3 = clockwise_angle(PI, 0.0);
        assert!((cw3 - TAU).abs() < 1e-10, "cw3 = {cw3}");
    }

    #[test]
    fn triangle_returns_one_loop() {
        let edges = vec![
            make_line_edge(Point2::new(0.0, 0.0), Point2::new(10.0, 0.0)),
            make_line_edge(Point2::new(10.0, 0.0), Point2::new(5.0, 10.0)),
            make_line_edge(Point2::new(5.0, 10.0), Point2::new(0.0, 0.0)),
        ];

        let loops = build_wire_loops(&edges, 1e-7, false, false);
        assert_eq!(loops.len(), 1);
        assert_eq!(loops[0].len(), 3);
    }
}
