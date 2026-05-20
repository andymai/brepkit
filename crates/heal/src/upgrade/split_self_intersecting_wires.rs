//! Split self-intersecting (figure-8 / pinched) wires into simple cycles.
//!
//! After mesh-fallback boolean assembly, a face's inner wire can pass
//! through the same vertex more than once — a "figure-8" topology that
//! occurs when the assembler glues two physically-separate holes
//! together via diagonal bridge edges across the gap material between
//! them. The slab-top in the dovetail / 4×4-pocket scenario (#696) is
//! the canonical example: a single 19-edge inner wire walks pocket
//! #1's perimeter, bridges via a diagonal to pocket #2, walks pocket
//! #2's perimeter, then bridges back — visiting the bridge-anchor
//! vertex twice.
//!
//! A wire visiting the same `VertexId` twice is mathematically a
//! figure-8 cycle, not a simple cycle. Most downstream code (wire
//! traversal in tessellation, STEP export, validation, set-membership
//! classification) assumes wires are simple — so the figure-8 is a
//! ticking time bomb even when wire-closure checks pass it (because
//! it does close, just at the pinch vertex).
//!
//! This pass walks every wire in the solid, identifies pinch vertices
//! (any vertex that appears 2+ times as the start of an oriented
//! edge), and splits the wire into the natural sub-cycles formed
//! between pinch occurrences. Each sub-cycle becomes a separate
//! `WireId`; for **inner** wires the new wires all become additional
//! inner wires on the same face. **Outer** wires with pinches are
//! left alone — that case is rarer (would mean the face's outer
//! boundary itself self-intersects, which usually indicates an
//! upstream geometry bug rather than topology cleanup work).
//!
//! Note: this pass changes wire topology but does **not** create or
//! remove edges. Faces that had a figure-8 inner wire end up with
//! correctly-structured separate inner wires that still reference
//! the original (potentially spurious) bridge edges. Downstream
//! boundary-edge counting and `unify_faces` are unchanged; this is
//! defensive structural cleanup so a later pass can layer
//! bridge-edge removal on top of well-formed cycles.

use std::collections::{HashMap, HashSet};

use brepkit_topology::Topology;
use brepkit_topology::solid::SolidId;
use brepkit_topology::vertex::VertexId;
use brepkit_topology::wire::{OrientedEdge, Wire, WireId};

use crate::HealError;

/// Split every self-intersecting inner wire on every face of the
/// given solid into simple sub-cycles.
///
/// Returns the number of inner wires that were replaced (one count
/// per wire that had at least one pinch).
///
/// # Errors
///
/// Returns [`HealError`] if entity lookups fail.
pub fn split_self_intersecting_inner_wires(
    topo: &mut Topology,
    solid_id: SolidId,
) -> Result<usize, HealError> {
    let face_ids = brepkit_topology::explorer::solid_faces(topo, solid_id)?;
    let mut wires_split: usize = 0;

    for fid in face_ids {
        // Snapshot inner wires; we'll rebuild `inner_wires_mut` only if a
        // split actually fires for this face (avoids touching faces whose
        // wires are already simple).
        let inner_wire_ids: Vec<WireId> = topo.face(fid)?.inner_wires().to_vec();
        let mut new_inner_wires: Vec<WireId> = Vec::new();
        let mut face_changed = false;

        for wid in inner_wire_ids {
            let split_outcome = try_split_wire(topo, wid)?;
            match split_outcome {
                None => new_inner_wires.push(wid),
                Some(sub_cycles) => {
                    face_changed = true;
                    wires_split += 1;
                    for cycle_edges in sub_cycles {
                        let new_wire = Wire::new(cycle_edges, true)?;
                        new_inner_wires.push(topo.add_wire(new_wire));
                    }
                }
            }
        }

        if face_changed {
            // The face's old figure-8 `WireId` is no longer referenced
            // here. `Topology` does not currently support entity
            // removal, so the old wire entry stays live in the arena
            // but becomes unreachable through any face — any later
            // pass that enumerates wires by walking faces (the
            // standard pattern, e.g. `solid_faces` + `face.inner_wires()`)
            // will skip it. Direct arena scans should not act on
            // unreachable wires.
            let face_mut = topo.face_mut(fid)?;
            let iw = face_mut.inner_wires_mut();
            iw.clear();
            iw.extend(new_inner_wires);
        }
    }

    Ok(wires_split)
}

/// Try to split a closed wire at any pinch vertex. Returns `None` if
/// the wire has no pinches (no work needed); `Some(cycles)` otherwise,
/// where each `cycles` entry is a simple sub-cycle (a `Vec<OrientedEdge>`
/// suitable for `Wire::new(_, closed = true)`).
fn try_split_wire(
    topo: &Topology,
    wid: WireId,
) -> Result<Option<Vec<Vec<OrientedEdge>>>, HealError> {
    let wire = topo.wire(wid)?;
    if !wire.is_closed() {
        return Ok(None);
    }
    let edges = wire.edges();
    let n = edges.len();
    if n < 3 {
        return Ok(None);
    }

    // Build the start-vertex sequence: vertex_seq[i] = oriented_start of
    // edges[i]. For a closed wire of `n` edges, this sequence has length
    // `n` and edges[n-1].oriented_end == vertex_seq[0] (the wire closes
    // back to its starting vertex).
    let mut vertex_seq: Vec<VertexId> = Vec::with_capacity(n);
    for oe in edges {
        let edge = topo.edge(oe.edge())?;
        vertex_seq.push(oe.oriented_start(edge));
    }

    // Quick scan: does any vertex appear twice? If not, we're done.
    let mut first_seen: HashSet<VertexId> = HashSet::with_capacity(n);
    let mut has_pinch = false;
    for &v in &vertex_seq {
        if !first_seen.insert(v) {
            has_pinch = true;
            break;
        }
    }
    if !has_pinch {
        return Ok(None);
    }

    // Stack-based pinch resolution. Walk vertices in order; whenever we
    // hit a vertex already on the stack, peel off everything from that
    // earlier occurrence onward as a sub-cycle, then continue. This is
    // the standard "extract simple cycles from a self-intersecting
    // closed walk" algorithm: it terminates with all detected loops in
    // `cycles` and the residual outer cycle in `current`.
    let mut cycles: Vec<Vec<OrientedEdge>> = Vec::new();
    let mut current: Vec<OrientedEdge> = Vec::with_capacity(n);
    let mut stack_pos: HashMap<VertexId, usize> = HashMap::new();

    for (i, oe) in edges.iter().enumerate() {
        let start_v = vertex_seq[i];
        if let Some(&peel_from) = stack_pos.get(&start_v) {
            // Extract inner cycle: edges from `peel_from` to end of
            // `current` form a closed loop at `start_v`.
            let inner: Vec<OrientedEdge> = current.split_off(peel_from);
            if !inner.is_empty() {
                cycles.push(inner);
            }
            // Vertices that were peeled off must be removed from the
            // position map (they're no longer on the active walk).
            // Rebuild the position map from the surviving prefix.
            stack_pos.clear();
            for (j, oe2) in current.iter().enumerate() {
                let edge2 = topo.edge(oe2.edge())?;
                stack_pos.insert(oe2.oriented_start(edge2), j);
            }
        }
        stack_pos.insert(start_v, current.len());
        current.push(*oe);
    }

    if !current.is_empty() {
        cycles.push(current);
    }

    // If we only extracted one cycle (the original), no actual split
    // happened — fall back to "no change". Otherwise return the split.
    if cycles.len() < 2 {
        return Ok(None);
    }
    Ok(Some(cycles))
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
    use brepkit_topology::vertex::{Vertex, VertexId};
    use brepkit_topology::wire::{OrientedEdge, Wire, WireId};

    use super::*;

    fn vertex(topo: &mut Topology, p: Point3) -> VertexId {
        topo.add_vertex(Vertex::new(p, 1e-7))
    }

    fn edge_line(topo: &mut Topology, a: VertexId, b: VertexId) -> brepkit_topology::edge::EdgeId {
        topo.add_edge(Edge::new(a, b, EdgeCurve::Line))
    }

    fn wrap_in_solid(topo: &mut Topology, face: FaceId) -> SolidId {
        let shell = topo.add_shell(Shell::new(vec![face]).unwrap());
        topo.add_solid(Solid::new(shell, Vec::new()))
    }

    fn make_face_with_inner_wire(
        topo: &mut Topology,
        outer_wire: WireId,
        inner_wire: WireId,
    ) -> FaceId {
        let surface = FaceSurface::Plane {
            normal: Vec3::new(0.0, 0.0, 1.0),
            d: 0.0,
        };
        topo.add_face(Face::new(outer_wire, vec![inner_wire], surface))
    }

    /// Two overlapping square loops sharing a pinch vertex V.
    /// Loop 1: V → A → B → V (3 edges)
    /// Loop 2: V → C → D → V (3 edges)
    /// Combined as one wire: V → A → B → V → C → D → V → (close)
    /// The wire has 6 edges and vertex V appears at positions 0 and 3.
    /// After split: 2 sub-cycles of 3 edges each.
    #[test]
    fn figure_8_inner_wire_splits_into_two_cycles() {
        let mut topo = Topology::new();
        let v = vertex(&mut topo, Point3::new(0.0, 0.0, 0.0));
        let a = vertex(&mut topo, Point3::new(1.0, 0.0, 0.0));
        let b = vertex(&mut topo, Point3::new(1.0, 1.0, 0.0));
        let c = vertex(&mut topo, Point3::new(-1.0, 0.0, 0.0));
        let d = vertex(&mut topo, Point3::new(-1.0, -1.0, 0.0));

        let e_va = edge_line(&mut topo, v, a);
        let e_ab = edge_line(&mut topo, a, b);
        let e_bv = edge_line(&mut topo, b, v);
        let e_vc = edge_line(&mut topo, v, c);
        let e_cd = edge_line(&mut topo, c, d);
        let e_dv = edge_line(&mut topo, d, v);

        let figure8 = topo.add_wire(
            Wire::new(
                vec![
                    OrientedEdge::new(e_va, true),
                    OrientedEdge::new(e_ab, true),
                    OrientedEdge::new(e_bv, true),
                    OrientedEdge::new(e_vc, true),
                    OrientedEdge::new(e_cd, true),
                    OrientedEdge::new(e_dv, true),
                ],
                true,
            )
            .unwrap(),
        );

        // Build a plausible outer wire (a square enclosing both loops).
        let p = vertex(&mut topo, Point3::new(-5.0, -5.0, 0.0));
        let q = vertex(&mut topo, Point3::new(5.0, -5.0, 0.0));
        let r = vertex(&mut topo, Point3::new(5.0, 5.0, 0.0));
        let s = vertex(&mut topo, Point3::new(-5.0, 5.0, 0.0));
        let e_pq = edge_line(&mut topo, p, q);
        let e_qr = edge_line(&mut topo, q, r);
        let e_rs = edge_line(&mut topo, r, s);
        let e_sp = edge_line(&mut topo, s, p);
        let outer = topo.add_wire(
            Wire::new(
                vec![
                    OrientedEdge::new(e_pq, true),
                    OrientedEdge::new(e_qr, true),
                    OrientedEdge::new(e_rs, true),
                    OrientedEdge::new(e_sp, true),
                ],
                true,
            )
            .unwrap(),
        );

        let fid = make_face_with_inner_wire(&mut topo, outer, figure8);
        let solid = wrap_in_solid(&mut topo, fid);

        let splits = split_self_intersecting_inner_wires(&mut topo, solid).unwrap();
        assert_eq!(splits, 1, "one figure-8 wire should split");

        let face = topo.face(fid).unwrap();
        assert_eq!(
            face.inner_wires().len(),
            2,
            "after split, the face should have 2 inner wires"
        );

        // Each new inner wire should be a simple closed 3-edge loop.
        for &iw_id in face.inner_wires() {
            let iw = topo.wire(iw_id).unwrap();
            assert_eq!(
                iw.edges().len(),
                3,
                "each sub-cycle should have 3 edges (V→X→Y→V)"
            );
            assert!(iw.is_closed(), "sub-cycles must close");
        }
    }

    /// Simple non-pinched inner wire (a plain square hole) — no split,
    /// no change. Verifies the pass is a no-op on well-formed wires.
    #[test]
    fn simple_inner_wire_unchanged() {
        let mut topo = Topology::new();
        // Inner square hole.
        let a = vertex(&mut topo, Point3::new(-1.0, -1.0, 0.0));
        let b = vertex(&mut topo, Point3::new(1.0, -1.0, 0.0));
        let c = vertex(&mut topo, Point3::new(1.0, 1.0, 0.0));
        let d = vertex(&mut topo, Point3::new(-1.0, 1.0, 0.0));
        let e_ab = edge_line(&mut topo, a, b);
        let e_bc = edge_line(&mut topo, b, c);
        let e_cd = edge_line(&mut topo, c, d);
        let e_da = edge_line(&mut topo, d, a);
        let inner = topo.add_wire(
            Wire::new(
                vec![
                    OrientedEdge::new(e_ab, true),
                    OrientedEdge::new(e_bc, true),
                    OrientedEdge::new(e_cd, true),
                    OrientedEdge::new(e_da, true),
                ],
                true,
            )
            .unwrap(),
        );

        // Outer square.
        let p = vertex(&mut topo, Point3::new(-5.0, -5.0, 0.0));
        let q = vertex(&mut topo, Point3::new(5.0, -5.0, 0.0));
        let r = vertex(&mut topo, Point3::new(5.0, 5.0, 0.0));
        let s = vertex(&mut topo, Point3::new(-5.0, 5.0, 0.0));
        let e_pq = edge_line(&mut topo, p, q);
        let e_qr = edge_line(&mut topo, q, r);
        let e_rs = edge_line(&mut topo, r, s);
        let e_sp = edge_line(&mut topo, s, p);
        let outer = topo.add_wire(
            Wire::new(
                vec![
                    OrientedEdge::new(e_pq, true),
                    OrientedEdge::new(e_qr, true),
                    OrientedEdge::new(e_rs, true),
                    OrientedEdge::new(e_sp, true),
                ],
                true,
            )
            .unwrap(),
        );

        let fid = make_face_with_inner_wire(&mut topo, outer, inner);
        let solid = wrap_in_solid(&mut topo, fid);

        let splits = split_self_intersecting_inner_wires(&mut topo, solid).unwrap();
        assert_eq!(splits, 0, "well-formed wires should not split");
        let face = topo.face(fid).unwrap();
        assert_eq!(face.inner_wires().len(), 1);
    }
}
