//! Phase VE: Vertex-on-edge interference detection.
//!
//! For each (vertex, edge) pair across solids, checks if the vertex
//! lies on the edge. If so, adds an extra pave to the edge's pave block.

use brepkit_math::tolerance::Tolerance;
use brepkit_math::vec::Point3;
use brepkit_topology::Topology;
use brepkit_topology::edge::EdgeId;
use brepkit_topology::solid::SolidId;
use brepkit_topology::vertex::VertexId;

use crate::ds::{GfaArena, Interference, Pave};
use crate::error::AlgoError;

/// Detect vertices lying on edges between the two solids.
///
/// Checks vertices of A against edges of B, and vertices of B against
/// edges of A. When a vertex lies on an edge (within tolerance), an
/// extra pave is inserted into the edge's pave block for later splitting.
///
/// # Errors
///
/// Returns [`AlgoError`] if any topology lookup fails.
pub fn perform(
    topo: &Topology,
    solid_a: SolidId,
    solid_b: SolidId,
    tol: Tolerance,
    arena: &mut GfaArena,
) -> Result<(), AlgoError> {
    let verts_a = brepkit_topology::explorer::solid_vertices(topo, solid_a)?;
    let verts_b = brepkit_topology::explorer::solid_vertices(topo, solid_b)?;
    let edges_a = brepkit_topology::explorer::solid_edges(topo, solid_a)?;
    let edges_b = brepkit_topology::explorer::solid_edges(topo, solid_b)?;

    // Check vertices of A against edges of B
    check_vertex_edge_pairs(topo, &verts_a, &edges_b, tol, arena)?;

    // Check vertices of B against edges of A
    check_vertex_edge_pairs(topo, &verts_b, &edges_a, tol, arena)?;

    Ok(())
}

/// Check each vertex against each edge and record VE interferences.
#[allow(clippy::too_many_lines)]
fn check_vertex_edge_pairs(
    topo: &Topology,
    vertices: &[VertexId],
    edges: &[EdgeId],
    tol: Tolerance,
    arena: &mut GfaArena,
) -> Result<(), AlgoError> {
    for &vid in vertices {
        let resolved_vid = arena.resolve_vertex(vid);
        let vertex = topo.vertex(resolved_vid)?;
        let pos = vertex.point();
        let vtol = vertex.tolerance();

        for &eid in edges {
            let edge = topo.edge(eid)?;

            // Skip if vertex is already an endpoint of this edge
            let start_v = arena.resolve_vertex(edge.start());
            let end_v = arena.resolve_vertex(edge.end());
            if resolved_vid == start_v || resolved_vid == end_v {
                continue;
            }

            // Get edge domain
            let start_pos = topo.vertex(edge.start())?.point();
            let end_pos = topo.vertex(edge.end())?.point();
            let (t0, t1) = edge.curve().domain_with_endpoints(start_pos, end_pos);

            // Project vertex onto edge curve
            let param = project_point_on_edge(topo, eid, pos)?;

            // Check if the parameter is within the edge domain
            if param < t0 - 1e-10 || param > t1 + 1e-10 {
                continue;
            }

            // Evaluate edge at the found parameter
            let edge_pt = edge
                .curve()
                .evaluate_with_endpoints(param, start_pos, end_pos);
            let dist = (pos - edge_pt).length();

            let combined_tol = vtol + tol.linear;
            if dist <= combined_tol {
                // Add extra pave to the pave blocks that contain this parameter
                let pave = Pave::new(resolved_vid, param);
                if let Some(pb_ids) = arena.edge_pave_blocks.get(&eid) {
                    let pb_ids_copy: Vec<_> = pb_ids.clone();
                    for pb_id in pb_ids_copy {
                        if let Some(pb) = arena.pave_blocks.get_mut(pb_id) {
                            let (pb_start, pb_end) = pb.parameter_range();
                            if param > pb_start + 1e-10 && param < pb_end - 1e-10 {
                                pb.add_extra_pave(pave);
                            }
                        }
                    }
                }

                // Record interference
                arena.interference.ve.push(Interference::VE {
                    vertex: resolved_vid,
                    edge: eid,
                    parameter: param,
                });

                log::debug!(
                    "VE: vertex {resolved_vid:?} on edge {eid:?} at t={param:.6} (dist={dist:.2e})",
                );
            }
        }
    }

    Ok(())
}

/// Project a point onto an edge curve, returning the closest parameter.
///
/// Uses coarse sampling followed by ternary search refinement for
/// robustness across all edge curve types.
fn project_point_on_edge(
    topo: &Topology,
    edge_id: EdgeId,
    point: Point3,
) -> Result<f64, AlgoError> {
    let edge = topo.edge(edge_id)?;
    let start_pos = topo.vertex(edge.start())?.point();
    let end_pos = topo.vertex(edge.end())?.point();
    let (t0, t1) = edge.curve().domain_with_endpoints(start_pos, end_pos);

    // Sample the edge at N points and find the closest
    let n_samples: usize = 32;
    let mut best_t = t0;
    let mut best_dist_sq = f64::MAX;

    for i in 0..=n_samples {
        let t = t0 + (t1 - t0) * (i as f64 / n_samples as f64);
        let pt = edge.curve().evaluate_with_endpoints(t, start_pos, end_pos);
        let d_sq = (point - pt).length_squared();
        if d_sq < best_dist_sq {
            best_dist_sq = d_sq;
            best_t = t;
        }
    }

    // Refine with ternary search in the neighborhood
    let dt = (t1 - t0) / n_samples as f64;
    let mut lo = (best_t - dt).max(t0);
    let mut hi = (best_t + dt).min(t1);

    for _ in 0..20 {
        let m1 = lo + (hi - lo) / 3.0;
        let m2 = hi - (hi - lo) / 3.0;
        let d1 =
            (point - edge.curve().evaluate_with_endpoints(m1, start_pos, end_pos)).length_squared();
        let d2 =
            (point - edge.curve().evaluate_with_endpoints(m2, start_pos, end_pos)).length_squared();
        if d1 < d2 {
            hi = m2;
        } else {
            lo = m1;
        }
    }

    Ok(f64::midpoint(lo, hi))
}
