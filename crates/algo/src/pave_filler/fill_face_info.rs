//! Populate [`FaceInfo`] with classified pave blocks.
//!
//! For each face involved in the boolean, collects:
//! - `pave_blocks_on`: boundary edges that were split (from the face's wires)
//! - `pave_blocks_sc`: section edges from FF intersection curves
//! - `pave_blocks_in`: edges from the opposing solid that lie inside this face
//!
//! [`FaceInfo`]: crate::ds::FaceInfo

use std::collections::HashSet;

use brepkit_topology::Topology;
use brepkit_topology::edge::EdgeId;
use brepkit_topology::face::FaceId;
use brepkit_topology::vertex::VertexId;

use crate::ds::{GfaArena, Interference, PaveBlockId};
use crate::error::AlgoError;

/// Populate [`FaceInfo`] for all faces with their classified pave blocks.
///
/// - `pave_blocks_on`: split boundary edges of each face
/// - `pave_blocks_sc`: section edges from FF intersections
/// - `pave_blocks_in`: edges from the other solid inside this face
///
/// # Errors
///
/// Returns [`AlgoError`] if a topology lookup fails.
pub fn perform(topo: &Topology, arena: &mut GfaArena) -> Result<(), AlgoError> {
    fill_boundary_on(topo, arena)?;
    fill_section_sc(arena);
    fill_ef_in(arena);
    Ok(())
}

/// For each face, find its boundary edges and map their leaf pave blocks
/// into `pave_blocks_on`.
fn fill_boundary_on(topo: &Topology, arena: &mut GfaArena) -> Result<(), AlgoError> {
    // Collect all faces that have face_info entries or appear in FF interference
    let mut all_faces: HashSet<FaceId> = arena.face_info.keys().copied().collect();
    for interf in &arena.interference.ff {
        if let Interference::FF { f1, f2, .. } = interf {
            all_faces.insert(*f1);
            all_faces.insert(*f2);
        }
    }

    for fid in all_faces {
        let face = topo.face(fid)?;

        // Collect all boundary edge IDs from outer + inner wires
        let mut boundary_edges: HashSet<EdgeId> = HashSet::new();

        let outer_wire = topo.wire(face.outer_wire())?;
        for oe in outer_wire.edges() {
            boundary_edges.insert(oe.edge());
        }
        for &inner_wid in face.inner_wires() {
            if let Ok(inner_wire) = topo.wire(inner_wid) {
                for oe in inner_wire.edges() {
                    boundary_edges.insert(oe.edge());
                }
            }
        }

        // Map each boundary edge's leaf pave blocks into ON.
        // Snapshot pave block data first to avoid aliasing arena borrows.
        let mut on_entries: Vec<(PaveBlockId, VertexId, VertexId)> = Vec::new();
        for eid in boundary_edges {
            if let Some(pb_ids) = arena.edge_pave_blocks.get(&eid).cloned() {
                let leaves = collect_leaves(arena, &pb_ids);
                for leaf_id in leaves {
                    if let Some(pb) = arena.pave_blocks.get(leaf_id) {
                        let sv = arena.resolve_vertex(pb.start.vertex);
                        let ev = arena.resolve_vertex(pb.end.vertex);
                        on_entries.push((leaf_id, sv, ev));
                    }
                }
            }
        }
        let fi = arena.face_info_mut(fid);
        for (leaf_id, sv, ev) in on_entries {
            fi.pave_blocks_on.insert(leaf_id);
            fi.vertices_on.insert(sv);
            fi.vertices_on.insert(ev);
        }
    }

    Ok(())
}

/// Section edges from FF intersection curves go into `pave_blocks_sc`.
fn fill_section_sc(arena: &mut GfaArena) {
    // Snapshot curve data to avoid aliasing
    let curve_data: Vec<_> = arena
        .curves
        .iter()
        .map(|c| (c.face_a, c.face_b, c.pave_blocks.clone()))
        .collect();

    for (face_a, face_b, pb_ids) in curve_data {
        for &pb_id in &pb_ids {
            // Snapshot vertex IDs before borrowing face_info mutably
            let Some(pb) = arena.pave_blocks.get(pb_id) else {
                continue;
            };
            let sv = arena.resolve_vertex(pb.start.vertex);
            let ev = arena.resolve_vertex(pb.end.vertex);

            let fi_a = arena.face_info_mut(face_a);
            fi_a.pave_blocks_sc.insert(pb_id);
            fi_a.vertices_sc.insert(sv);
            fi_a.vertices_sc.insert(ev);

            let fi_b = arena.face_info_mut(face_b);
            fi_b.pave_blocks_sc.insert(pb_id);
            fi_b.vertices_sc.insert(sv);
            fi_b.vertices_sc.insert(ev);
        }
    }
}

/// Edges from EF interference go into the face's `pave_blocks_in`.
fn fill_ef_in(arena: &mut GfaArena) {
    // Snapshot EF data
    let ef_data: Vec<_> = arena
        .interference
        .ef
        .iter()
        .filter_map(|interf| {
            if let Interference::EF { edge, face, .. } = interf {
                Some((*edge, *face))
            } else {
                None
            }
        })
        .collect();

    for (edge_id, face_id) in ef_data {
        if let Some(pb_ids) = arena.edge_pave_blocks.get(&edge_id).cloned() {
            let leaves = collect_leaves(arena, &pb_ids);
            let fi = arena.face_info_mut(face_id);
            for leaf_id in leaves {
                fi.pave_blocks_in.insert(leaf_id);
            }
        }
    }
}

/// Recursively collect leaf pave blocks (those with no children).
fn collect_leaves(arena: &GfaArena, pb_ids: &[PaveBlockId]) -> Vec<PaveBlockId> {
    let mut leaves = Vec::new();
    for &pb_id in pb_ids {
        if let Some(pb) = arena.pave_blocks.get(pb_id) {
            if pb.children.is_empty() {
                leaves.push(pb_id);
            } else {
                leaves.extend(collect_leaves(arena, &pb.children));
            }
        }
    }
    leaves
}
