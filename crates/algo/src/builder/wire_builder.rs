//! 2D wire construction from edge soup via angular-sorting traversal.
//!
//! Given an unordered set of [`OrientedPCurveEdge`]s on a face, builds
//! minimal closed wire loops by traversing edges using the minimum
//! clockwise angle rule at each vertex.
//!
//! Uses the minimum clockwise angle traversal algorithm for face splitting.

use std::collections::HashMap;
use std::f64::consts::TAU;

use brepkit_math::vec::Point2;

use super::split_types::OrientedPCurveEdge;

// ---------------------------------------------------------------------------
// Types
// ---------------------------------------------------------------------------

/// An entry in the vertex adjacency map.
struct VertexEntry {
    /// Index into the input edge slice.
    edge_idx: usize,
    /// `true` if this vertex is the START of the edge (outgoing).
    outgoing: bool,
    /// Tangent angle at this vertex in \[0, 2pi).
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
/// `u_periodic` and `v_periodic` indicate whether the surface parameter wraps
/// (e.g. cylinder u in \[0, 2pi)). When `true`, UV coordinates are normalized
/// into \[0, period) before quantizing, and edge angles correct for seam-crossing.
pub fn build_wire_loops(
    edges: &[OrientedPCurveEdge],
    tol: f64,
    u_periodic: bool,
    v_periodic: bool,
) -> Vec<Vec<OrientedPCurveEdge>> {
    if edges.is_empty() {
        return Vec::new();
    }

    let u_period = if u_periodic { Some(TAU) } else { None };
    let v_period = if v_periodic { Some(TAU) } else { None };

    // 1. Build vertex adjacency map.
    let mut adj: HashMap<(i64, i64), Vec<VertexEntry>> = HashMap::new();
    for (idx, edge) in edges.iter().enumerate() {
        let start_key = quantize_uv_periodic(edge.start_uv, tol, u_period, v_period);
        let end_key = quantize_uv_periodic(edge.end_uv, tol, u_period, v_period);
        let angle_out = edge_angle_at_vertex_periodic(edge, true, u_period, v_period);
        let angle_in = edge_angle_at_vertex_periodic(edge, false, u_period, v_period);
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

        let start_vertex = quantize_uv_periodic(edges[start_idx].start_uv, tol, u_period, v_period);
        let mut current_loop = vec![edges[start_idx].clone()];
        let mut current_idx = start_idx;

        // Walk edges until we close the loop.
        loop {
            let current_edge = &edges[current_idx];
            let end_vertex = quantize_uv_periodic(current_edge.end_uv, tol, u_period, v_period);

            // Check for loop closure: quantized keys must match AND the raw
            // 2D distance must be small. On periodic surfaces, seam-opposite
            // vertices quantize to the same key but have large UV distance
            // (~2pi). Check raw UV distance to reject seam-boundary false closures.
            let is_closed = if end_vertex == start_vertex {
                let raw_du = (current_edge.end_uv.x() - edges[start_idx].start_uv.x()).abs();
                let raw_dv = (current_edge.end_uv.y() - edges[start_idx].start_uv.y()).abs();
                let seam_threshold = std::f64::consts::PI;
                // Reject seam-boundary false closures: vertices on opposite sides
                // of a periodic seam quantize to the same key but have large UV distance.
                !(u_periodic && raw_du > seam_threshold || v_periodic && raw_dv > seam_threshold)
            } else {
                false
            };
            if is_closed {
                loops.push(current_loop);
                break;
            }

            // Incoming angle at the end vertex.
            let incoming_angle =
                edge_angle_at_vertex_periodic(current_edge, false, u_period, v_period);
            let arriving_start =
                quantize_uv_periodic(current_edge.start_uv, tol, u_period, v_period);

            // Find best outgoing edge at end_vertex.
            let Some(entries) = adj.get(&end_vertex) else {
                // Dead end -- discard this incomplete loop.
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
                if quantize_uv_periodic(candidate.end_uv, tol, u_period, v_period) == arriving_start
                {
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
                // Truly no way out -- discard loop.
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
#[allow(dead_code)]
fn quantize_uv(p: Point2, tol: f64) -> (i64, i64) {
    quantize_uv_periodic(p, tol, None, None)
}

/// Quantize a 2D point with periodic normalization.
///
/// When `u_period`/`v_period` is `Some`, the coordinate is normalized into
/// `[0, period)` via `rem_euclid` before quantizing. This ensures vertices
/// near the seam (e.g. u=6.28 and u=0.001 on a cylinder) hash to the same key.
fn quantize_uv_periodic(
    p: Point2,
    tol: f64,
    u_period: Option<f64>,
    v_period: Option<f64>,
) -> (i64, i64) {
    // Guard against non-positive or non-finite tolerance.
    let safe_tol = if tol <= 0.0 || !tol.is_finite() {
        1e-7
    } else {
        tol
    };
    let resolution = 1.0 / safe_tol;
    (
        quantize_coord(p.x(), resolution, u_period),
        quantize_coord(p.y(), resolution, v_period),
    )
}

/// Quantize a single coordinate, wrapping at the period boundary so that
/// values near `0` and near `period` hash to the same key.
fn quantize_coord(val: f64, resolution: f64, period: Option<f64>) -> i64 {
    if let Some(p) = period {
        let normalized = val.rem_euclid(p);
        let q = (normalized * resolution).round() as i64;
        let period_q = (p * resolution).round() as i64;
        if q >= period_q { 0 } else { q }
    } else {
        (val * resolution).round() as i64
    }
}

/// Compute the outgoing angle of an edge at a vertex in UV space.
///
/// Returns the angle in \[0, 2pi) of the tangent direction at the vertex.
///
/// - `at_start = true`: outgoing direction (start -> end).
/// - `at_start = false`: incoming direction (end -> start, pointing back along the edge).
#[allow(dead_code)]
fn edge_angle_at_vertex(edge: &OrientedPCurveEdge, at_start: bool) -> f64 {
    edge_angle_at_vertex_periodic(edge, at_start, None, None)
}

/// Compute the outgoing angle of an edge at a vertex, with periodic wrapping.
///
/// For curved edges (NURBS pcurve), evaluates the pcurve near the endpoint
/// to get the true tangent direction. For straight edges, uses the chord.
///
/// When a period is set and the raw `dx` exceeds half the period, the
/// difference is wrapped so seam-crossing edges get correct tangent angles.
fn edge_angle_at_vertex_periodic(
    edge: &OrientedPCurveEdge,
    at_start: bool,
    u_period: Option<f64>,
    v_period: Option<f64>,
) -> f64 {
    let (mut dx, mut dy) = pcurve_tangent_at_endpoint(edge, at_start);
    if let Some(period) = u_period {
        let half = period * 0.5;
        if dx.abs() > half {
            dx -= dx.signum() * period;
        }
    }
    if let Some(period) = v_period {
        let half = period * 0.5;
        if dy.abs() > half {
            dy -= dy.signum() * period;
        }
    }
    let angle = dy.atan2(dx);
    if angle < 0.0 { angle + TAU } else { angle }
}

/// Compute the tangent direction at an edge endpoint in UV space.
///
/// For `Line2D` pcurves, returns the chord direction (exact).
/// For `NurbsCurve2D` pcurves, evaluates the pcurve near the endpoint
/// to approximate the true tangent -- important for half-circle arcs where
/// the chord direction can be perpendicular to the actual tangent.
fn pcurve_tangent_at_endpoint(edge: &OrientedPCurveEdge, at_start: bool) -> (f64, f64) {
    use brepkit_math::curves2d::Curve2D;

    // For NURBS pcurves, sample near the endpoint for tangent direction.
    // Reverse edges reuse the same pcurve -- swap t0/tn to match the
    // oriented edge direction.
    if let Curve2D::Nurbs(ref nurbs) = edge.pcurve {
        let knots = nurbs.knots();
        if knots.len() >= 2 {
            let t0_raw = knots[0];
            let tn_raw = knots[knots.len() - 1];
            // For reverse edges, the pcurve's t0 corresponds to the edge's
            // end and tn corresponds to the edge's start.
            let (t_start, t_end) = if edge.forward {
                (t0_raw, tn_raw)
            } else {
                (tn_raw, t0_raw)
            };
            let span = (t_end - t_start).abs();
            let delta = span * 0.01;

            if at_start {
                let p0 = nurbs.evaluate(t_start);
                let p1 = nurbs.evaluate(t_start + (t_end - t_start).signum() * delta);
                return (p1.x() - p0.x(), p1.y() - p0.y());
            }
            // at_end: incoming direction (from end back toward start).
            let p0 = nurbs.evaluate(t_end);
            let p1 = nurbs.evaluate(t_end - (t_end - t_start).signum() * delta);
            return (p1.x() - p0.x(), p1.y() - p0.y());
        }
    }

    // For Line2D and fallback: use chord direction.
    if at_start {
        (
            edge.end_uv.x() - edge.start_uv.x(),
            edge.end_uv.y() - edge.start_uv.y(),
        )
    } else {
        (
            edge.start_uv.x() - edge.end_uv.x(),
            edge.start_uv.y() - edge.end_uv.y(),
        )
    }
}

/// Compute the clockwise sweep angle from `angle_in` to `angle_out`.
///
/// `angle_in` is the incoming edge's angle at the vertex (points back along
/// the arriving edge). `angle_out` is the candidate outgoing edge's angle.
///
/// The formula computes the CCW angle from `angle_out` to the travel
/// direction (`angle_in + pi`). Minimum value = rightmost turn = traces
/// minimal enclosed regions.
///
/// Returns a value in (0, 2pi].
fn clockwise_angle(angle_in: f64, angle_out: f64) -> f64 {
    // Travel direction = opposite of incoming (which points backward).
    let travel = (angle_in + std::f64::consts::PI).rem_euclid(TAU);
    let da = (travel - angle_out).rem_euclid(TAU);
    if da < 1e-14 {
        TAU // Exact continuation -> maximum angle.
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
            source_edge_idx: None,
            pave_block_id: None,
            start_vid: None,
            end_vid: None,
        }
    }

    #[test]
    fn square_cut_by_vertical_line_produces_two_loops() {
        let edges = vec![
            make_line_edge(Point2::new(0.0, 0.0), Point2::new(5.0, 0.0)),
            make_line_edge(Point2::new(5.0, 0.0), Point2::new(10.0, 0.0)),
            make_line_edge(Point2::new(10.0, 0.0), Point2::new(10.0, 10.0)),
            make_line_edge(Point2::new(10.0, 10.0), Point2::new(5.0, 10.0)),
            make_line_edge(Point2::new(5.0, 10.0), Point2::new(0.0, 10.0)),
            make_line_edge(Point2::new(0.0, 10.0), Point2::new(0.0, 0.0)),
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
        let cw = clockwise_angle(PI, PI / 2.0);
        assert!((cw - 3.0 * PI / 2.0).abs() < 1e-10, "cw = {cw}");

        let cw2 = clockwise_angle(PI, 3.0 * PI / 2.0);
        assert!((cw2 - PI / 2.0).abs() < 1e-10, "cw2 = {cw2}");

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

    #[test]
    fn quantize_uv_periodic_wraps_near_seam() {
        let p1 = Point2::new(TAU + 0.001, 1.0);
        let p2 = Point2::new(0.001, 1.0);
        let tol = 1e-7;
        let k1 = quantize_uv_periodic(p1, tol, Some(TAU), None);
        let k2 = quantize_uv_periodic(p2, tol, Some(TAU), None);
        assert_eq!(
            k1, k2,
            "points near seam should hash equal: {k1:?} vs {k2:?}"
        );

        let p3 = Point2::new(TAU, 1.0);
        let p4 = Point2::new(0.0, 1.0);
        let k3 = quantize_uv_periodic(p3, tol, Some(TAU), None);
        let k4 = quantize_uv_periodic(p4, tol, Some(TAU), None);
        assert_eq!(k3, k4, "u=TAU should wrap to u=0: {k3:?} vs {k4:?}");

        let k5 = quantize_uv_periodic(p1, tol, None, None);
        let k6 = quantize_uv_periodic(p2, tol, None, None);
        assert_ne!(k5, k6, "non-periodic should keep them distinct");
    }

    #[test]
    fn edge_angle_periodic_wraps_large_du() {
        let edge = make_line_edge(Point2::new(6.0, 0.0), Point2::new(0.3, 0.0));
        let angle = edge_angle_at_vertex_periodic(&edge, true, Some(TAU), None);
        assert!(
            !(0.5..=TAU - 0.5).contains(&angle),
            "angle should be near 0 (rightward), got {angle}"
        );

        let angle_raw = edge_angle_at_vertex(&edge, true);
        assert!(
            (angle_raw - std::f64::consts::PI).abs() < 0.5,
            "non-periodic angle should be near pi, got {angle_raw}"
        );
    }

    #[test]
    fn wire_loop_crossing_seam() {
        let tau_05 = TAU + 0.5;
        let edges = vec![
            make_line_edge(Point2::new(5.5, 0.0), Point2::new(tau_05, 0.0)),
            make_line_edge(Point2::new(tau_05, 0.0), Point2::new(tau_05, 5.0)),
            make_line_edge(Point2::new(tau_05, 5.0), Point2::new(5.5, 5.0)),
            make_line_edge(Point2::new(5.5, 5.0), Point2::new(5.5, 0.0)),
        ];
        let loops = build_wire_loops(&edges, 1e-7, true, false);
        assert_eq!(loops.len(), 1, "expected 1 loop, got {}", loops.len());
        assert_eq!(loops[0].len(), 4);
    }
}
