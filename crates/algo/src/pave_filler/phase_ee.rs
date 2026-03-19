//! Phase EE: Edge-edge intersection detection.
//!
//! Finds intersection points and overlapping segments between edges
//! from the two solids. Creates new vertices at crossing points and
//! adds extra paves to both edges.

use crate::ds::{GfaArena, Interference, Pave};
use brepkit_math::tolerance::Tolerance;
use brepkit_math::vec::Point3;
use brepkit_topology::Topology;
use brepkit_topology::edge::{EdgeCurve, EdgeId};
use brepkit_topology::solid::SolidId;
use brepkit_topology::vertex::Vertex;

use super::helpers::{add_pave_to_edge, find_nearby_pave_vertex};
use crate::error::AlgoError;

/// Detect edge-edge intersections between the two solids.
///
/// For each `(ea, eb)` pair where `ea` belongs to `solid_a` and `eb` to
/// `solid_b`, find intersection points. When a crossing coincides with
/// an existing vertex, add paves to both edges. When no existing vertex
/// is near, record the interference for the later `MakeSplitEdges` phase.
///
/// # Errors
///
/// Returns [`AlgoError`] if any topology lookup fails.
#[allow(clippy::too_many_lines)]
pub fn perform(
    topo: &mut Topology,
    solid_a: SolidId,
    solid_b: SolidId,
    tol: Tolerance,
    arena: &mut GfaArena,
) -> Result<(), AlgoError> {
    let edges_a = brepkit_topology::explorer::solid_edges(topo, solid_a)?;
    let edges_b = brepkit_topology::explorer::solid_edges(topo, solid_b)?;

    // Collect edge data up front to avoid repeated lookups
    let data_a = collect_edge_data(topo, &edges_a)?;
    let data_b = collect_edge_data(topo, &edges_b)?;

    for (idx_a, (&ea_id, ea_data)) in edges_a.iter().zip(data_a.iter()).enumerate() {
        for (idx_b, (&eb_id, eb_data)) in edges_b.iter().zip(data_b.iter()).enumerate() {
            // Avoid duplicates when both edges are in both solids
            if ea_id == eb_id {
                continue;
            }

            // Quick AABB rejection
            if !aabbs_overlap(
                &ea_data.bbox_min,
                &ea_data.bbox_max,
                &eb_data.bbox_min,
                &eb_data.bbox_max,
                tol.linear,
            ) {
                continue;
            }

            // Find intersection points
            let crossings = find_edge_edge_crossings(topo, ea_id, ea_data, eb_id, eb_data, tol)?;

            for (t_a, t_b, point) in crossings {
                // Check if the intersection point is already a known vertex
                let existing = find_nearby_pave_vertex(topo, arena, point, tol);

                let vertex_id = if let Some(vid) = existing {
                    vid
                } else {
                    // No existing vertex near this point — create one.
                    topo.add_vertex(Vertex::new(point, tol.linear))
                };

                // Add extra paves to both edges
                add_pave_to_edge(arena, ea_id, Pave::new(vertex_id, t_a));
                add_pave_to_edge(arena, eb_id, Pave::new(vertex_id, t_b));

                arena.interference.ee.push(Interference::EE {
                    e1: ea_id,
                    e2: eb_id,
                    new_vertex: Some(vertex_id),
                    common_pave_block: None,
                });

                log::debug!(
                    "EE: edges {ea_id:?}[{idx_a}] and {eb_id:?}[{idx_b}] cross at \
                     t_a={t_a:.6}, t_b={t_b:.6}",
                );
            }
        }
    }

    Ok(())
}

/// Pre-computed edge data for fast intersection checks.
struct EdgeData {
    /// Start vertex position.
    start_pos: Point3,
    /// End vertex position.
    end_pos: Point3,
    /// Start of parameter domain.
    t0: f64,
    /// End of parameter domain.
    t1: f64,
    /// Axis-aligned bounding box minimum corner.
    bbox_min: Point3,
    /// Axis-aligned bounding box maximum corner.
    bbox_max: Point3,
}

/// Collect pre-computed data for a set of edges.
fn collect_edge_data(topo: &Topology, edges: &[EdgeId]) -> Result<Vec<EdgeData>, AlgoError> {
    let mut data = Vec::with_capacity(edges.len());
    for &eid in edges {
        let edge = topo.edge(eid)?;
        let start_pos = topo.vertex(edge.start())?.point();
        let end_pos = topo.vertex(edge.end())?.point();
        let (t0, t1) = edge.curve().domain_with_endpoints(start_pos, end_pos);

        // Compute AABB by sampling
        let n: usize = 16;
        let mut min = start_pos;
        let mut max = start_pos;
        for i in 0..=n {
            let t = t0 + (t1 - t0) * (i as f64 / n as f64);
            let pt = edge.curve().evaluate_with_endpoints(t, start_pos, end_pos);
            min = Point3::new(
                min.x().min(pt.x()),
                min.y().min(pt.y()),
                min.z().min(pt.z()),
            );
            max = Point3::new(
                max.x().max(pt.x()),
                max.y().max(pt.y()),
                max.z().max(pt.z()),
            );
        }

        data.push(EdgeData {
            start_pos,
            end_pos,
            t0,
            t1,
            bbox_min: min,
            bbox_max: max,
        });
    }
    Ok(data)
}

/// Check if two AABBs overlap with tolerance padding.
fn aabbs_overlap(min_a: &Point3, max_a: &Point3, min_b: &Point3, max_b: &Point3, tol: f64) -> bool {
    min_a.x() <= max_b.x() + tol
        && max_a.x() >= min_b.x() - tol
        && min_a.y() <= max_b.y() + tol
        && max_a.y() >= min_b.y() - tol
        && min_a.z() <= max_b.z() + tol
        && max_a.z() >= min_b.z() - tol
}

/// Find crossing points between two edges.
///
/// Uses algebraic line-line intersection when both edges are lines,
/// and segment-pair sampling otherwise.
#[allow(clippy::too_many_lines)]
fn find_edge_edge_crossings(
    topo: &Topology,
    ea_id: EdgeId,
    ea: &EdgeData,
    eb_id: EdgeId,
    eb: &EdgeData,
    tol: Tolerance,
) -> Result<Vec<(f64, f64, Point3)>, AlgoError> {
    let edge_a = topo.edge(ea_id)?;
    let edge_b = topo.edge(eb_id)?;

    // For Line-Line: algebraic intersection
    if matches!(edge_a.curve(), EdgeCurve::Line) && matches!(edge_b.curve(), EdgeCurve::Line) {
        return Ok(line_line_intersection(ea, eb, tol));
    }

    // General case: sample both edges and find close segment pairs
    let n: usize = 32;
    let mut crossings = Vec::new();

    // Sample edge A
    let pts_a: Vec<SegmentEndpoint> = (0..=n)
        .map(|i| {
            let t = ea.t0 + (ea.t1 - ea.t0) * (i as f64 / n as f64);
            let pos = edge_a
                .curve()
                .evaluate_with_endpoints(t, ea.start_pos, ea.end_pos);
            SegmentEndpoint { t, pos }
        })
        .collect();

    // Sample edge B
    let pts_b: Vec<SegmentEndpoint> = (0..=n)
        .map(|i| {
            let t = eb.t0 + (eb.t1 - eb.t0) * (i as f64 / n as f64);
            let pos = edge_b
                .curve()
                .evaluate_with_endpoints(t, eb.start_pos, eb.end_pos);
            SegmentEndpoint { t, pos }
        })
        .collect();

    // Find closest approach between segment pairs
    let domain_a = (ea.t0 - tol.linear)..=(ea.t1 + tol.linear);
    let domain_b = (eb.t0 - tol.linear)..=(eb.t1 + tol.linear);

    for i in 0..n {
        for j in 0..n {
            // Quick distance check: if minimum endpoint distance exceeds
            // the sum of segment lengths + tolerance, skip.
            let min_dist = (pts_a[i].pos - pts_b[j].pos)
                .length()
                .min((pts_a[i].pos - pts_b[j + 1].pos).length())
                .min((pts_a[i + 1].pos - pts_b[j].pos).length())
                .min((pts_a[i + 1].pos - pts_b[j + 1].pos).length());

            let seg_len_a = (pts_a[i + 1].pos - pts_a[i].pos).length();
            let seg_len_b = (pts_b[j + 1].pos - pts_b[j].pos).length();

            if min_dist > seg_len_a + seg_len_b + tol.linear {
                continue;
            }

            // Refine: find closest approach between segments
            if let Some((t_a, t_b, pt)) = closest_segment_pair(
                [&pts_a[i], &pts_a[i + 1]],
                [&pts_b[j], &pts_b[j + 1]],
                tol.linear,
            ) {
                // Ensure within domain
                if domain_a.contains(&t_a) && domain_b.contains(&t_b) {
                    // Deduplicate: skip if too close to existing crossing
                    let is_dup = crossings
                        .iter()
                        .any(|&(ct_a, ct_b, _): &(f64, f64, Point3)| {
                            (t_a - ct_a).abs() < 1e-6 && (t_b - ct_b).abs() < 1e-6
                        });
                    if !is_dup {
                        crossings.push((t_a, t_b, pt));
                    }
                }
            }
        }
    }

    Ok(crossings)
}

/// Algebraic line-line intersection.
///
/// Computes the closest approach between two line segments. If the
/// segments are within tolerance at that point, returns the crossing.
fn line_line_intersection(ea: &EdgeData, eb: &EdgeData, tol: Tolerance) -> Vec<(f64, f64, Point3)> {
    let da = ea.end_pos - ea.start_pos;
    let db = eb.end_pos - eb.start_pos;
    let w = ea.start_pos - eb.start_pos;

    let a = da.dot(da);
    let b = da.dot(db);
    let c = db.dot(db);
    let d = da.dot(w);
    let e = db.dot(w);

    let denom = a.mul_add(c, -(b * b));

    // Parallel lines — 1e-20 checks for mathematical degeneracy
    // (near-zero determinant), not geometric tolerance.
    if denom.abs() < 1e-20 {
        return Vec::new();
    }

    let s = b.mul_add(e, -(c * d)) / denom;
    let t = a.mul_add(e, -(b * d)) / denom;

    // Check if within edge domains [0, 1] for lines
    let range = -tol.linear..=1.0 + tol.linear;
    if !range.contains(&s) || !range.contains(&t) {
        return Vec::new();
    }

    let pt_a = ea.start_pos + da * s;
    let pt_b = eb.start_pos + db * t;
    let dist = (pt_a - pt_b).length();

    if dist <= tol.linear {
        let midpoint = Point3::new(
            f64::midpoint(pt_a.x(), pt_b.x()),
            f64::midpoint(pt_a.y(), pt_b.y()),
            f64::midpoint(pt_a.z(), pt_b.z()),
        );
        // Map s,t from [0,1] to actual edge parameter domains
        let param_a = s.mul_add(ea.t1 - ea.t0, ea.t0);
        let param_b = t.mul_add(eb.t1 - eb.t0, eb.t0);
        vec![(param_a, param_b, midpoint)]
    } else {
        Vec::new()
    }
}

/// A parameterized sample point on an edge segment.
struct SegmentEndpoint {
    /// Parameter value on the edge curve.
    t: f64,
    /// 3D position at this parameter.
    pos: Point3,
}

/// Find closest approach between two line segments.
///
/// Returns `(param_a, param_b, midpoint)` if distance is within tolerance.
#[allow(clippy::similar_names)]
fn closest_segment_pair(
    seg_a: [&SegmentEndpoint; 2],
    seg_b: [&SegmentEndpoint; 2],
    tol: f64,
) -> Option<(f64, f64, Point3)> {
    let da = seg_a[1].pos - seg_a[0].pos;
    let db = seg_b[1].pos - seg_b[0].pos;
    let w = seg_a[0].pos - seg_b[0].pos;

    let a = da.dot(da);
    let b = da.dot(db);
    let c = db.dot(db);
    let d = da.dot(w);
    let e = db.dot(w);

    let denom = a.mul_add(c, -(b * b));

    // 1e-20 checks for mathematical degeneracy (near-zero determinant),
    // not geometric tolerance.
    let (s, t) = if denom.abs() < 1e-20 {
        // Parallel segments — use midpoints
        (0.5, 0.5)
    } else {
        let s = (b.mul_add(e, -(c * d)) / denom).clamp(0.0, 1.0);
        let t = (a.mul_add(e, -(b * d)) / denom).clamp(0.0, 1.0);
        (s, t)
    };

    let pt_a = seg_a[0].pos + da * s;
    let pt_b = seg_b[0].pos + db * t;
    let dist = (pt_a - pt_b).length();

    if dist <= tol {
        let param_a = s.mul_add(seg_a[1].t - seg_a[0].t, seg_a[0].t);
        let param_b = t.mul_add(seg_b[1].t - seg_b[0].t, seg_b[0].t);
        let midpoint = Point3::new(
            f64::midpoint(pt_a.x(), pt_b.x()),
            f64::midpoint(pt_a.y(), pt_b.y()),
            f64::midpoint(pt_a.z(), pt_b.z()),
        );
        Some((param_a, param_b, midpoint))
    } else {
        None
    }
}
