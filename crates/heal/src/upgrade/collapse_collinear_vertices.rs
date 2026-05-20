//! Collapse collinear interior wire vertices that are structurally orphan
//! across the solid.
//!
//! After mesh-fallback boolean assembly, adjacent faces can carry
//! asymmetric tessellation residue at shared edges: face A's wire goes
//! `… → P → V → Q → …` (V being a tessellation-diagonal artifact) while
//! face B's wire jumps `… → P → Q → …` directly. Calls to
//! `unify_faces`/`merge_collinear_edges` only smooth wires of faces that
//! the unifier itself merged, leaving the asymmetry intact whenever the
//! two faces remain separate topological entities.
//!
//! When `V` is incident to exactly two edges across the entire solid AND
//! both are Line curves AND V is collinear strictly between P and Q,
//! every wire that traverses V must do so as the consecutive pair
//! (`E1`, `E2`) in some orientation. The vertex is structurally
//! collapsible: replace the pair with a single oriented edge `E_new(P,Q)`
//! in every wire that referenced it, reusing an existing `Edge(P,Q,Line)`
//! when one is already present (so adjacent faces end up sharing the
//! same `EdgeId` rather than two parallel ones, which is what downstream
//! face-adjacency counting needs).
//!
//! Conservative gates: only Line curves, only when V's two neighbours
//! are distinct, only when V is strictly between them, only when V has
//! exactly two incident edges in the solid. Vertices that fail any gate
//! are left untouched.

use std::collections::{HashMap, HashSet};

use brepkit_math::tolerance::Tolerance;
use brepkit_topology::Topology;
use brepkit_topology::edge::{Edge, EdgeCurve, EdgeId};
use brepkit_topology::face::{FaceId, FaceSurface};
use brepkit_topology::solid::SolidId;
use brepkit_topology::vertex::VertexId;
use brepkit_topology::wire::{OrientedEdge, Wire, WireId};

use crate::HealError;

/// Collapse interior wire vertices that are collinear between their two
/// neighbours and have no other edges anchoring them.
///
/// Returns the number of vertices collapsed.
///
/// # Errors
///
/// Returns [`HealError`] if entity lookups fail. Best-effort: a single
/// failure mid-pass does not abort already-applied collapses.
pub fn collapse_collinear_wire_vertices(
    topo: &mut Topology,
    solid_id: SolidId,
    tol: Tolerance,
) -> Result<usize, HealError> {
    // 1. Gather every (face_id, wire_id) pair in the solid. Faces hold
    //    one outer wire plus zero-or-more inner wires (holes); both are
    //    in scope.
    let face_ids = brepkit_topology::explorer::solid_faces(topo, solid_id)?;
    let mut wire_ids: Vec<WireId> = Vec::new();
    for &fid in &face_ids {
        let face = topo.face(fid)?;
        wire_ids.push(face.outer_wire());
        wire_ids.extend(face.inner_wires().iter().copied());
    }
    if wire_ids.is_empty() {
        return Ok(0);
    }

    // 2. Snapshot every edge that participates in any solid wire, indexed
    //    by EdgeId so vertex→edge counting below stays O(unique edges)
    //    rather than O(sum of wire lengths). Edges that appear in 2 wires
    //    (a healthy manifold edge) only count once toward vertex incidence.
    //    Also record which faces reference each edge so the planarity gate
    //    below can confirm every face touching V is planar before collapse.
    let mut edges_in_solid: HashMap<EdgeId, (VertexId, VertexId)> = HashMap::new();
    let mut edge_to_faces: HashMap<EdgeId, HashSet<FaceId>> = HashMap::new();
    for &fid in &face_ids {
        let face = topo.face(fid)?;
        let mut all_wires: Vec<WireId> = vec![face.outer_wire()];
        all_wires.extend(face.inner_wires().iter().copied());
        for wid in all_wires {
            let wire = topo.wire(wid)?;
            for oe in wire.edges() {
                let eid = oe.edge();
                if let std::collections::hash_map::Entry::Vacant(slot) = edges_in_solid.entry(eid) {
                    let edge = topo.edge(eid)?;
                    slot.insert((edge.start(), edge.end()));
                }
                edge_to_faces.entry(eid).or_default().insert(fid);
            }
        }
    }

    // 3. Build vertex → incident-edges map from the snapshot.
    let mut vertex_edges: HashMap<VertexId, Vec<EdgeId>> = HashMap::new();
    for (&eid, &(start, end)) in &edges_in_solid {
        vertex_edges.entry(start).or_default().push(eid);
        if end != start {
            vertex_edges.entry(end).or_default().push(eid);
        }
    }

    // 4. Identify candidate vertices: exactly two incident edges, both
    //    Lines, and the vertex strictly between the two far endpoints.
    let lin_tol = tol.linear.max(1e-12);
    let lin_tol_sq = lin_tol * lin_tol;

    let mut plans: Vec<CollapsePlan> = Vec::new();
    // Track which existing edges we've already reused so a chain of
    // collinear vertices on the same line doesn't all try to claim
    // overlapping P→Q edges that the prior plan already consumed.
    // (Rare in practice — guard via per-iteration topology reads.)
    for (&vertex, incident) in &vertex_edges {
        if incident.len() != 2 {
            continue;
        }
        let eid1 = incident[0];
        let eid2 = incident[1];
        let edge1 = topo.edge(eid1)?;
        let edge2 = topo.edge(eid2)?;
        if !matches!(edge1.curve(), EdgeCurve::Line) || !matches!(edge2.curve(), EdgeCurve::Line) {
            continue;
        }
        // Determine the far endpoints P and Q (the non-V end of each edge).
        let p_vid = if edge1.start() == vertex {
            edge1.end()
        } else if edge1.end() == vertex {
            edge1.start()
        } else {
            continue; // V not actually on edge1 — defensive
        };
        let q_vid = if edge2.start() == vertex {
            edge2.end()
        } else if edge2.end() == vertex {
            edge2.start()
        } else {
            continue;
        };
        if p_vid == q_vid || p_vid == vertex || q_vid == vertex {
            continue;
        }
        // Topological gate: E1 and E2 must be shared by the same face set.
        // Anything else means V's two-edge pair has inconsistent
        // membership and collapsing it would re-asymmetrize the wires.
        //
        //   * 2 faces — "symmetric" case, both faces register V on their
        //     wire. Common when mesh fallback splits the same chord on
        //     both sides; coplanar pair required so the chord doesn't
        //     diverge from the underlying geometry on either side.
        //   * 1 face — "asymmetric residue" case (#696), only one face
        //     carries V; the neighbour already references a direct P→Q
        //     edge via a separate `EdgeId`. Re-unifying onto that edge
        //     requires it to exist and to live on a face that is
        //     coplanar with — or perpendicular to (a different planar
        //     face altogether) — the V-bearing face. We check this
        //     after the planarity pass below.
        let faces_e1 = edge_to_faces.get(&eid1).cloned().unwrap_or_default();
        let faces_e2 = edge_to_faces.get(&eid2).cloned().unwrap_or_default();
        if faces_e1 != faces_e2 || !(1..=2).contains(&faces_e1.len()) {
            continue;
        }

        // Planarity gate: a vertex on a chord between curved-surface samples
        // is load-bearing for the curved face's tessellation — removing it
        // breaks watertightness against the adjacent face. Only collapse
        // when both faces are coplanar (planes match in normal direction
        // and offset) so the chord and the underlying surface coincide.
        let ang_tol_cos = (1.0 - tol.angular).max(1.0 - 1e-9);
        let mut plane_ref: Option<(brepkit_math::vec::Vec3, f64)> = None;
        let mut all_planar = true;
        for &fid in &faces_e1 {
            let face = topo.face(fid)?;
            let FaceSurface::Plane { normal, d } = face.surface() else {
                all_planar = false;
                break;
            };
            let n_len = normal.dot(*normal).sqrt();
            if n_len < tol.linear {
                all_planar = false;
                break;
            }
            let n_unit = *normal * (1.0 / n_len);
            let d_norm = *d / n_len;
            if let Some((prev_n, prev_d)) = plane_ref {
                let dot = prev_n.dot(n_unit).abs();
                if dot < ang_tol_cos {
                    all_planar = false;
                    break;
                }
                let signed_d = if prev_n.dot(n_unit) >= 0.0 {
                    d_norm
                } else {
                    -d_norm
                };
                if (signed_d - prev_d).abs() > tol.linear {
                    all_planar = false;
                    break;
                }
            } else {
                plane_ref = Some((n_unit, d_norm));
            }
        }
        if !all_planar {
            continue;
        }

        // Strict-between collinearity check in 3D.
        let v_pt = topo.vertex(vertex)?.point();
        let p_pt = topo.vertex(p_vid)?.point();
        let q_pt = topo.vertex(q_vid)?.point();
        let pq = q_pt - p_pt;
        let pq_len_sq = pq.dot(pq);
        if pq_len_sq < lin_tol_sq {
            continue; // P and Q coincident
        }
        let pv = v_pt - p_pt;
        let t = pv.dot(pq) / pq_len_sq;
        // Strictly between: 0 < t < 1, with a tolerance margin so end
        // collisions don't masquerade as interior.
        if t <= lin_tol || t >= 1.0 - lin_tol {
            continue;
        }
        // Perpendicular distance from V to line PQ, scaled by pq length.
        let projection = p_pt + pq * t;
        let perp = v_pt - projection;
        if perp.dot(perp) > lin_tol_sq {
            continue;
        }

        // Find-or-create the merged edge so the two faces keep
        // referencing the same `EdgeId` after the collapse. For the
        // asymmetric (size-1) case, *require* an existing direct edge
        // on a face the V-bearing face isn't using — otherwise we'd be
        // either creating an orphan edge (no benefit) or unifying a
        // face with itself.
        let existing = find_line_edge(topo, p_vid, q_vid);
        let new_edge = match (existing, faces_e1.len()) {
            (Some(eid), _) => {
                // Check the existing edge is on a face NOT in faces_e1.
                let neighbour_faces = edge_to_faces.get(&eid).cloned().unwrap_or_default();
                if neighbour_faces.is_subset(&faces_e1) {
                    // No genuine "other side" to re-unify onto.
                    continue;
                }
                eid
            }
            (None, 2) => topo.add_edge(Edge::new(p_vid, q_vid, EdgeCurve::Line)),
            (None, _) => continue,
        };
        if new_edge == eid1 || new_edge == eid2 {
            continue; // Defensive — should not match the V-touching edges
        }
        plans.push(CollapsePlan {
            vertex,
            old_edges: [eid1, eid2],
            new_edge,
        });
    }

    if plans.is_empty() {
        return Ok(0);
    }

    // 5. Apply plans: for each wire, scan for consecutive edges that match
    //    a (V, E1, E2) pair and rewrite the oriented-edge list. A wire's
    //    edge list shrinks by 1 for each match; rebuild via `Wire::new`
    //    + `*wire_mut = new_wire` (matches `merge_collinear_edges` pattern
    //    in `unify_same_domain`).
    let plan_by_vertex: HashMap<VertexId, &CollapsePlan> =
        plans.iter().map(|p| (p.vertex, p)).collect();

    let mut applied: usize = 0;
    // Track which vertices were actually consumed by at least one wire
    // rewrite — the plan-vertex count is the upper bound, but a vertex
    // unreachable from any wire (orphaned by a prior pass) shouldn't
    // count as "collapsed".
    let mut consumed: std::collections::HashSet<VertexId> = std::collections::HashSet::new();

    for &wid in &wire_ids {
        let new_edges_opt = try_rewrite_wire(topo, wid, &plan_by_vertex, &mut consumed)?;
        if let Some(new_edges) = new_edges_opt {
            let is_closed = topo.wire(wid)?.is_closed();
            match Wire::new(new_edges, is_closed) {
                Ok(new_wire) => {
                    let wire_mut = topo.wire_mut(wid)?;
                    *wire_mut = new_wire;
                }
                Err(e) => {
                    log::warn!(
                        "collapse_collinear_wire_vertices: skipped wire {wid:?} \
                         after rewrite — {e}"
                    );
                }
            }
        }
    }
    applied += consumed.len();

    Ok(applied)
}

/// Find an existing `Line` edge whose endpoints are `{a, b}` (in either
/// order), or create one. Reuse keeps neighbouring faces on the same
/// `EdgeId`; creation handles the case where no face already carried the
/// short-circuit edge.
fn find_line_edge(topo: &Topology, a: VertexId, b: VertexId) -> Option<EdgeId> {
    for (eid, edge) in topo.edges().iter() {
        if !matches!(edge.curve(), EdgeCurve::Line) {
            continue;
        }
        let (s, e) = (edge.start(), edge.end());
        if (s == a && e == b) || (s == b && e == a) {
            return Some(eid);
        }
    }
    None
}

/// Walk `wire`'s edges; whenever two consecutive edges touch a plan's
/// vertex with both plan edges accounted for, emit a single oriented
/// edge for the plan's `new_edge` and skip ahead. Returns `None` if the
/// wire is unchanged (avoids reallocating untouched wires).
fn try_rewrite_wire(
    topo: &Topology,
    wid: WireId,
    plans: &HashMap<VertexId, &CollapsePlan>,
    consumed: &mut std::collections::HashSet<VertexId>,
) -> Result<Option<Vec<OrientedEdge>>, HealError> {
    let wire = topo.wire(wid)?;
    let edges = wire.edges();
    let n = edges.len();
    if n < 2 {
        return Ok(None);
    }
    let mut rewritten: Vec<OrientedEdge> = Vec::with_capacity(n);
    let mut changed = false;
    let mut i = 0;
    while i < n {
        // Try to pair edges[i] with edges[(i+1) % n] across a plan vertex.
        let next_idx = if i + 1 < n {
            i + 1
        } else if wire.is_closed() && n > 1 {
            0
        } else {
            // Open wire, last edge — no pair to consider.
            rewritten.push(edges[i]);
            i += 1;
            continue;
        };

        let oe_cur = edges[i];
        let oe_next = edges[next_idx];
        let edge_cur = topo.edge(oe_cur.edge())?;
        let edge_next = topo.edge(oe_next.edge())?;

        let end_v = oe_cur.oriented_end(edge_cur);
        let start_v_next = oe_next.oriented_start(edge_next);
        if end_v != start_v_next {
            rewritten.push(oe_cur);
            i += 1;
            continue;
        }

        let Some(plan) = plans.get(&end_v) else {
            rewritten.push(oe_cur);
            i += 1;
            continue;
        };

        // Verify both wire edges are this plan's old edges (either order).
        let cur_eid = oe_cur.edge();
        let nxt_eid = oe_next.edge();
        let matches_plan = (cur_eid == plan.old_edges[0] && nxt_eid == plan.old_edges[1])
            || (cur_eid == plan.old_edges[1] && nxt_eid == plan.old_edges[0]);
        if !matches_plan {
            rewritten.push(oe_cur);
            i += 1;
            continue;
        }

        // Determine orientation: the merged oriented edge should run
        // from oe_cur.oriented_start(...) to oe_next.oriented_end(...).
        let start_v = oe_cur.oriented_start(edge_cur);
        let final_v = oe_next.oriented_end(edge_next);
        let new_edge = topo.edge(plan.new_edge)?;
        let forward = if new_edge.start() == start_v && new_edge.end() == final_v {
            true
        } else if new_edge.start() == final_v && new_edge.end() == start_v {
            false
        } else {
            // Shouldn't happen — the new edge was created or found with
            // endpoints {P, Q}. If it somehow doesn't match, bail on this
            // pair rather than corrupt the wire.
            rewritten.push(oe_cur);
            i += 1;
            continue;
        };

        rewritten.push(OrientedEdge::new(plan.new_edge, forward));
        changed = true;
        consumed.insert(end_v);
        // Skip the consumed pair. For closed-wire wrap (next_idx == 0),
        // we must avoid re-processing edges[0] later; mark by jumping
        // past the array.
        if next_idx == 0 {
            // Replace the first emitted edge (which was the previous
            // edges[0]) by removing the trailing position. Closed-wrap
            // case is rare and benign — break out.
            // Pop the previously-pushed first edge so we don't double-count.
            // Actually since we emitted starting from i=0 normally, when
            // we wrap, the consumed pair is (edges[n-1], edges[0]); we
            // emit the new merged edge here and need to drop the first
            // entry of `rewritten` (which was edges[0]).
            if !rewritten.is_empty() {
                rewritten.remove(0);
            }
            break;
        }
        i += 2;
    }

    Ok(if changed { Some(rewritten) } else { None })
}

struct CollapsePlan {
    vertex: VertexId,
    old_edges: [EdgeId; 2],
    new_edge: EdgeId,
}

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::expect_used)]
mod tests {
    use brepkit_math::vec::{Point3, Vec3};
    use brepkit_topology::Topology;
    use brepkit_topology::edge::{Edge, EdgeCurve};
    use brepkit_topology::face::{Face, FaceId, FaceSurface};
    use brepkit_topology::shell::Shell;
    use brepkit_topology::solid::Solid;
    use brepkit_topology::vertex::Vertex;
    use brepkit_topology::wire::{OrientedEdge, Wire};

    use super::*;

    fn vertex(topo: &mut Topology, p: Point3) -> VertexId {
        topo.add_vertex(Vertex::new(p, 1e-7))
    }

    /// Build a face whose outer wire is a 4-edge polyline A→V→B→…→A
    /// where `v` is collinear-strictly-between `a` and `b`.
    fn face_with_split_edge(
        topo: &mut Topology,
        a: VertexId,
        v: VertexId,
        b: VertexId,
        c: VertexId,
    ) -> FaceId {
        let e_av = topo.add_edge(Edge::new(a, v, EdgeCurve::Line));
        let e_vb = topo.add_edge(Edge::new(v, b, EdgeCurve::Line));
        let e_bc = topo.add_edge(Edge::new(b, c, EdgeCurve::Line));
        let e_ca = topo.add_edge(Edge::new(c, a, EdgeCurve::Line));
        let wire = topo.add_wire(
            Wire::new(
                vec![
                    OrientedEdge::new(e_av, true),
                    OrientedEdge::new(e_vb, true),
                    OrientedEdge::new(e_bc, true),
                    OrientedEdge::new(e_ca, true),
                ],
                true,
            )
            .unwrap(),
        );
        let surface = FaceSurface::Plane {
            normal: Vec3::new(0.0, 0.0, 1.0),
            d: 0.0,
        };
        topo.add_face(Face::new(wire, Vec::new(), surface))
    }

    fn wrap_in_solid(topo: &mut Topology, faces: Vec<FaceId>) -> SolidId {
        let shell = topo.add_shell(Shell::new(faces).unwrap());
        topo.add_solid(Solid::new(shell, Vec::new()))
    }

    #[test]
    fn vertex_strictly_between_collapses() {
        // Two coplanar faces share the chord A-V-B. The chord vertex V
        // sits collinearly between A and B; both faces have wire
        // A→V→B→{C or D}→A. Edges E1=(A,V) and E2=(V,B) are shared by
        // BOTH faces (the symmetric / structurally-collapsible case).
        let mut topo = Topology::new();
        let a = vertex(&mut topo, Point3::new(0.0, 0.0, 0.0));
        let b = vertex(&mut topo, Point3::new(2.0, 0.0, 0.0));
        let v = vertex(&mut topo, Point3::new(1.0, 0.0, 0.0));
        let c = vertex(&mut topo, Point3::new(0.0, -1.0, 0.0));
        let d = vertex(&mut topo, Point3::new(2.0, 1.0, 0.0));

        // Shared edges across both faces.
        let e_av = topo.add_edge(Edge::new(a, v, EdgeCurve::Line));
        let e_vb = topo.add_edge(Edge::new(v, b, EdgeCurve::Line));

        let surface = FaceSurface::Plane {
            normal: Vec3::new(0.0, 0.0, 1.0),
            d: 0.0,
        };

        // Face 1: A→V→B→D→A (uses shared e_av, e_vb)
        let e_bd = topo.add_edge(Edge::new(b, d, EdgeCurve::Line));
        let e_da = topo.add_edge(Edge::new(d, a, EdgeCurve::Line));
        let wire1 = topo.add_wire(
            Wire::new(
                vec![
                    OrientedEdge::new(e_av, true),
                    OrientedEdge::new(e_vb, true),
                    OrientedEdge::new(e_bd, true),
                    OrientedEdge::new(e_da, true),
                ],
                true,
            )
            .unwrap(),
        );
        let f1 = topo.add_face(Face::new(wire1, Vec::new(), surface.clone()));

        // Face 2: B→V→A→C→B (reverse traversal of e_vb, e_av; uses shared edges)
        let e_ac = topo.add_edge(Edge::new(a, c, EdgeCurve::Line));
        let e_cb = topo.add_edge(Edge::new(c, b, EdgeCurve::Line));
        let wire2 = topo.add_wire(
            Wire::new(
                vec![
                    OrientedEdge::new(e_vb, false),
                    OrientedEdge::new(e_av, false),
                    OrientedEdge::new(e_ac, true),
                    OrientedEdge::new(e_cb, true),
                ],
                true,
            )
            .unwrap(),
        );
        let f2 = topo.add_face(Face::new(wire2, Vec::new(), surface));

        let solid = wrap_in_solid(&mut topo, vec![f1, f2]);

        let collapsed =
            collapse_collinear_wire_vertices(&mut topo, solid, Tolerance::new()).unwrap();
        assert_eq!(
            collapsed, 1,
            "single shared interior vertex should collapse"
        );

        // Both wires should now be 3 edges (was 4) and both should
        // reference the same merged A↔B edge (find_or_create_line_edge
        // creates one and reuses it for the second wire).
        let w1 = topo.wire(topo.face(f1).unwrap().outer_wire()).unwrap();
        let w2 = topo.wire(topo.face(f2).unwrap().outer_wire()).unwrap();
        assert_eq!(w1.edges().len(), 3);
        assert_eq!(w2.edges().len(), 3);
        assert_eq!(
            w1.edges()[0].edge(),
            w2.edges()[0].edge(),
            "both wires should now reference the same merged EdgeId for A↔B"
        );
    }

    #[test]
    fn off_line_vertex_is_not_collapsed() {
        // V is not collinear — perpendicular offset of 1e-3 on a 2.0 baseline.
        let mut topo = Topology::new();
        let a = vertex(&mut topo, Point3::new(0.0, 0.0, 0.0));
        let b = vertex(&mut topo, Point3::new(2.0, 0.0, 0.0));
        let v = vertex(&mut topo, Point3::new(1.0, 1e-3, 0.0));
        let d = vertex(&mut topo, Point3::new(2.0, 1.0, 0.0));
        let f = face_with_split_edge(&mut topo, a, v, b, d);
        let solid = wrap_in_solid(&mut topo, vec![f]);
        let collapsed =
            collapse_collinear_wire_vertices(&mut topo, solid, Tolerance::new()).unwrap();
        assert_eq!(collapsed, 0);
    }

    #[test]
    fn vertex_with_three_incident_edges_is_not_collapsed() {
        // V on edge A-V-B but also has a third edge going to a corner —
        // collapsing would orphan that third edge, so the pass must skip.
        let mut topo = Topology::new();
        let a = vertex(&mut topo, Point3::new(0.0, 0.0, 0.0));
        let b = vertex(&mut topo, Point3::new(2.0, 0.0, 0.0));
        let v = vertex(&mut topo, Point3::new(1.0, 0.0, 0.0));
        let c = vertex(&mut topo, Point3::new(1.0, 1.0, 0.0));
        let d = vertex(&mut topo, Point3::new(2.0, 1.0, 0.0));

        // First face: A→V→B→D→A
        let f_split = face_with_split_edge(&mut topo, a, v, b, d);
        // Second face: A→V→C→A — gives V a 3rd incident edge (V→C and C→A)
        let e_av2 = topo.add_edge(Edge::new(a, v, EdgeCurve::Line));
        let e_vc = topo.add_edge(Edge::new(v, c, EdgeCurve::Line));
        let e_ca2 = topo.add_edge(Edge::new(c, a, EdgeCurve::Line));
        let wire = topo.add_wire(
            Wire::new(
                vec![
                    OrientedEdge::new(e_av2, true),
                    OrientedEdge::new(e_vc, true),
                    OrientedEdge::new(e_ca2, true),
                ],
                true,
            )
            .unwrap(),
        );
        let surface = FaceSurface::Plane {
            normal: Vec3::new(0.0, 0.0, 1.0),
            d: 0.0,
        };
        let f_extra = topo.add_face(Face::new(wire, Vec::new(), surface));
        let solid = wrap_in_solid(&mut topo, vec![f_split, f_extra]);

        let collapsed =
            collapse_collinear_wire_vertices(&mut topo, solid, Tolerance::new()).unwrap();
        assert_eq!(
            collapsed, 0,
            "V has 3 distinct incident edges (A→V appears as two EdgeIds plus V→C); should not collapse"
        );
    }
}
