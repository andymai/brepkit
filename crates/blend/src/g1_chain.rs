//! G1 continuity chain expansion for fillet edge propagation.
//!
//! Given a set of seed edges, iteratively expands along manifold edges
//! that share the same face pair and are tangent-continuous at the
//! shared vertex.

use std::collections::{HashMap, HashSet};

use brepkit_math::tolerance::Tolerance;
use brepkit_math::traits::ParametricCurve;
use brepkit_math::vec::{Point3, Vec3};
use brepkit_topology::Topology;
use brepkit_topology::edge::{EdgeCurve, EdgeId};
use brepkit_topology::face::FaceId;
use brepkit_topology::solid::SolidId;

/// Sample the tangent of an edge curve at normalized parameter `t` in `[0, 1]`.
///
/// Maps from the `[0, 1]` interval to the curve's native parameter range,
/// accounting for circle/ellipse angle wrapping.
fn sample_edge_tangent(curve: &EdgeCurve, p_start: Point3, p_end: Point3, t: f64) -> Vec3 {
    match curve {
        EdgeCurve::Line => p_end - p_start,
        EdgeCurve::Circle(circle) => {
            let ts = circle.project(p_start);
            let mut te = circle.project(p_end);
            if te <= ts {
                te += std::f64::consts::TAU;
            }
            ParametricCurve::tangent(circle, ts + (te - ts) * t)
        }
        EdgeCurve::Ellipse(ellipse) => {
            let ts = ellipse.project(p_start);
            let mut te = ellipse.project(p_end);
            if te <= ts {
                te += std::f64::consts::TAU;
            }
            ParametricCurve::tangent(ellipse, ts + (te - ts) * t)
        }
        EdgeCurve::NurbsCurve(nurbs) => {
            let (u0, u1) = nurbs.domain();
            let u = u0 + (u1 - u0) * t;
            let d = nurbs.derivatives(u, 1);
            d[1]
        }
    }
}

/// Expand a seed edge set by G1 (tangent-continuity) chain propagation.
///
/// Starting from `seed_edges`, iteratively adds any manifold edge that:
/// 1. Shares a vertex with an edge already in the set.
/// 2. Has the same pair of adjacent faces (same ridgeline).
/// 3. Is tangent-continuous at the shared vertex (< 10 deg deviation).
///
/// # Errors
///
/// Returns `BlendError::Topology` if any topology lookup fails.
#[allow(clippy::too_many_lines)]
pub fn expand_g1_chain(
    topo: &Topology,
    solid: SolidId,
    seed_edges: &[EdgeId],
    tol: Tolerance,
) -> Result<Vec<EdgeId>, crate::BlendError> {
    let solid_data = topo.solid(solid)?;
    let shell = topo.shell(solid_data.outer_shell())?;
    let shell_face_ids: Vec<FaceId> = shell.faces().to_vec();

    // Build edge->faces and vertex->edges maps for the full shell.
    let mut edge_to_faces: HashMap<usize, Vec<FaceId>> = HashMap::new();
    let mut vertex_to_edges: HashMap<usize, Vec<EdgeId>> = HashMap::new();
    let mut edge_ids: HashMap<usize, EdgeId> = HashMap::new();

    for &fid in &shell_face_ids {
        let face = topo.face(fid)?;
        let wire_ids: Vec<_> = std::iter::once(face.outer_wire())
            .chain(face.inner_wires().iter().copied())
            .collect();
        for wid in wire_ids {
            let wire = topo.wire(wid)?;
            for oe in wire.edges() {
                let eid = oe.edge();
                edge_to_faces.entry(eid.index()).or_default().push(fid);
                edge_ids.insert(eid.index(), eid);
                let edge = topo.edge(eid)?;
                vertex_to_edges
                    .entry(edge.start().index())
                    .or_default()
                    .push(eid);
                vertex_to_edges
                    .entry(edge.end().index())
                    .or_default()
                    .push(eid);
            }
        }
    }
    // Deduplicate vertex_to_edges (each edge appears once per adjacent face).
    for edges in vertex_to_edges.values_mut() {
        edges.sort_unstable_by_key(|e: &EdgeId| e.index());
        edges.dedup_by_key(|e: &mut EdgeId| e.index());
    }

    // Iterative BFS expansion.
    let mut expanded: HashSet<usize> = seed_edges.iter().map(|e| e.index()).collect();
    let mut queue: Vec<EdgeId> = seed_edges.to_vec();

    while let Some(current) = queue.pop() {
        // Face pair for current edge (sorted for comparison).
        let Some(cf) = edge_to_faces.get(&current.index()) else {
            continue;
        };
        if cf.len() != 2 {
            continue;
        }
        let (cf1, cf2) = {
            let (a, b) = (cf[0].index(), cf[1].index());
            if a < b { (a, b) } else { (b, a) }
        };

        let cur_edge = topo.edge(current)?;
        let cur_start = topo.vertex(cur_edge.start())?.point();
        let cur_end = topo.vertex(cur_edge.end())?.point();

        for &shared_vid in &[cur_edge.start(), cur_edge.end()] {
            // "Away from vertex" tangent for the current edge at this vertex.
            let t_cur = {
                let t_raw = if shared_vid == cur_edge.start() {
                    // Forward tangent at start points away from vertex -- correct sign.
                    sample_edge_tangent(cur_edge.curve(), cur_start, cur_end, 0.0)
                } else {
                    // Forward tangent at end points INTO vertex; negate for "away".
                    -sample_edge_tangent(cur_edge.curve(), cur_start, cur_end, 1.0)
                };
                let len = t_raw.length();
                if len < tol.linear {
                    continue;
                }
                t_raw * (1.0 / len)
            };

            let Some(neighbors) = vertex_to_edges.get(&shared_vid.index()) else {
                continue;
            };
            for &nb in neighbors {
                if expanded.contains(&nb.index()) {
                    continue;
                }
                // Must be manifold (exactly 2 adjacent faces).
                let Some(nf) = edge_to_faces.get(&nb.index()) else {
                    continue;
                };
                if nf.len() != 2 {
                    continue;
                }
                // Must share the same face pair.
                let (nf1, nf2) = {
                    let (a, b) = (nf[0].index(), nf[1].index());
                    if a < b { (a, b) } else { (b, a) }
                };
                if (cf1, cf2) != (nf1, nf2) {
                    continue;
                }

                // "Away from vertex" tangent for the neighbor edge at the shared vertex.
                let nb_edge = topo.edge(nb)?;
                let nb_start = topo.vertex(nb_edge.start())?.point();
                let nb_end = topo.vertex(nb_edge.end())?.point();
                let t_nb = {
                    let t_raw = if shared_vid == nb_edge.start() {
                        sample_edge_tangent(nb_edge.curve(), nb_start, nb_end, 0.0)
                    } else {
                        -sample_edge_tangent(nb_edge.curve(), nb_start, nb_end, 1.0)
                    };
                    let len = t_raw.length();
                    if len < tol.linear {
                        continue;
                    }
                    t_raw * (1.0 / len)
                };

                // G1 continuity: "away" tangents must be anti-parallel (< ~10 deg deviation).
                // cos(170 deg) ~ -0.985.  This is strict: a true G1 joint has dot = -1.0.
                if t_cur.dot(t_nb) < -0.985 {
                    expanded.insert(nb.index());
                    queue.push(nb);
                }
            }
        }
    }

    let mut result: Vec<EdgeId> = expanded
        .iter()
        .filter_map(|idx| edge_ids.get(idx).copied())
        .collect();
    result.sort_unstable_by_key(|e| e.index());
    Ok(result)
}
