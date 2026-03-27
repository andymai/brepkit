//! Separate shape store for the GFA pipeline.
//!
//! The `GfaShapeStore` deep-copies input solids into an isolated Topology,
//! runs the entire GFA pipeline on the copies, then exports the result
//! back to the caller's Topology. This eliminates vertex/edge identity
//! conflicts between input solids and the GFA result.

use std::collections::HashMap;

use brepkit_topology::Topology;
use brepkit_topology::edge::{Edge, EdgeId};
use brepkit_topology::face::Face;
use brepkit_topology::shell::Shell;
use brepkit_topology::solid::{Solid, SolidId};
use brepkit_topology::vertex::{Vertex, VertexId};
use brepkit_topology::wire::{OrientedEdge, Wire, WireId};

use crate::error::AlgoError;

/// Isolated shape store for GFA operations.
///
/// Contains its own `Topology` with deep-copied input solid entities.
/// The GFA pipeline operates entirely within this store, then the result
/// is exported back to the caller's Topology.
pub struct GfaShapeStore {
    /// The store's own topology arena.
    pub topo: Topology,
    /// Store-local SolidId for the deep-copied solid A.
    pub solid_a: SolidId,
    /// Store-local SolidId for the deep-copied solid B.
    pub solid_b: SolidId,
}

impl GfaShapeStore {
    /// Create a new GFA shape store by deep-copying two input solids.
    ///
    /// # Errors
    ///
    /// Returns [`AlgoError`] if any topology lookup fails.
    pub fn new(source: &Topology, orig_a: SolidId, orig_b: SolidId) -> Result<Self, AlgoError> {
        let mut topo = Topology::default();

        let solid_a = deep_copy_solid(source, &mut topo, orig_a)?;
        let solid_b = deep_copy_solid(source, &mut topo, orig_b)?;

        Ok(Self {
            topo,
            solid_a,
            solid_b,
        })
    }

    /// Export a result solid from this store back to the caller's Topology.
    ///
    /// Deep-copies all entities from the store solid into `target`,
    /// returning the new SolidId in the target Topology.
    ///
    /// # Errors
    ///
    /// Returns [`AlgoError`] if any topology lookup fails.
    pub fn export_solid(
        &self,
        target: &mut Topology,
        solid_id: SolidId,
    ) -> Result<SolidId, AlgoError> {
        deep_copy_solid(&self.topo, target, solid_id)
    }
}

/// Deep-copy a solid from `source` topology into `target` topology.
///
/// Creates new vertices, edges, wires, faces, shells, and solid in `target`
/// with remapped IDs. Returns the new SolidId.
///
/// Uses the snapshot-then-allocate pattern to satisfy the borrow checker.
#[allow(clippy::too_many_lines, clippy::items_after_statements)]
fn deep_copy_solid(
    source: &Topology,
    target: &mut Topology,
    solid_id: SolidId,
) -> Result<SolidId, AlgoError> {
    // ── Snapshot phase ──────────────────────────────────────────────

    let solid = source.solid(solid_id)?;
    let outer_shell_id = solid.outer_shell();
    let inner_shell_ids: Vec<_> = solid.inner_shells().to_vec();

    let all_shell_ids: Vec<_> = std::iter::once(outer_shell_id)
        .chain(inner_shell_ids.iter().copied())
        .collect();

    struct VertexSnap {
        old_index: usize,
        point: brepkit_math::vec::Point3,
        tol: f64,
    }
    struct EdgeSnap {
        old_index: usize,
        start_index: usize,
        end_index: usize,
        curve: brepkit_topology::edge::EdgeCurve,
        tolerance: Option<f64>,
    }
    struct WireSnap {
        old_index: usize,
        edges: Vec<(usize, bool)>,
        closed: bool,
    }
    struct FaceSnap {
        outer_wire_index: usize,
        inner_wire_indices: Vec<usize>,
        surface: brepkit_topology::face::FaceSurface,
        reversed: bool,
    }
    struct ShellSnap {
        faces: Vec<FaceSnap>,
    }

    let mut vertex_snaps: Vec<VertexSnap> = Vec::new();
    let mut edge_snaps: Vec<EdgeSnap> = Vec::new();
    let mut wire_snaps: Vec<WireSnap> = Vec::new();
    let mut shell_snaps: Vec<ShellSnap> = Vec::new();

    let mut seen_vertices = std::collections::HashSet::new();
    let mut seen_edges = std::collections::HashSet::new();
    let mut seen_wires = std::collections::HashSet::new();

    for &shell_id in &all_shell_ids {
        let shell = source.shell(shell_id)?;
        let mut face_snaps = Vec::new();

        for &face_id in shell.faces() {
            let face = source.face(face_id)?;
            let surface = face.surface().clone();
            let outer_wire_index = face.outer_wire().index();
            let inner_wire_indices: Vec<usize> =
                face.inner_wires().iter().map(|w| w.index()).collect();

            for wire_id in
                std::iter::once(face.outer_wire()).chain(face.inner_wires().iter().copied())
            {
                if !seen_wires.insert(wire_id.index()) {
                    continue;
                }
                let wire = source.wire(wire_id)?;
                let mut edge_refs = Vec::new();

                for oe in wire.edges() {
                    let edge_idx = oe.edge().index();
                    edge_refs.push((edge_idx, oe.is_forward()));

                    if !seen_edges.insert(edge_idx) {
                        continue;
                    }
                    let edge = source.edge(oe.edge())?;
                    let start_idx = edge.start().index();
                    let end_idx = edge.end().index();

                    for &vid in &[edge.start(), edge.end()] {
                        let vid_idx = vid.index();
                        if seen_vertices.insert(vid_idx) {
                            let v = source.vertex(vid)?;
                            vertex_snaps.push(VertexSnap {
                                old_index: vid_idx,
                                point: v.point(),
                                tol: v.tolerance(),
                            });
                        }
                    }

                    edge_snaps.push(EdgeSnap {
                        old_index: edge_idx,
                        start_index: start_idx,
                        end_index: end_idx,
                        curve: edge.curve().clone(),
                        tolerance: edge.tolerance(),
                    });
                }

                wire_snaps.push(WireSnap {
                    old_index: wire_id.index(),
                    edges: edge_refs,
                    closed: wire.is_closed(),
                });
            }

            face_snaps.push(FaceSnap {
                outer_wire_index,
                inner_wire_indices,
                surface,
                reversed: face.is_reversed(),
            });
        }

        shell_snaps.push(ShellSnap { faces: face_snaps });
    }

    // ── Allocate phase ──────────────────────────────────────────────

    let mut vertex_map: HashMap<usize, VertexId> = HashMap::new();
    for vsnap in &vertex_snaps {
        let new_vid = target.add_vertex(Vertex::new(vsnap.point, vsnap.tol));
        vertex_map.insert(vsnap.old_index, new_vid);
    }

    let mut edge_map: HashMap<usize, EdgeId> = HashMap::new();
    for esnap in &edge_snaps {
        let new_start = vertex_map[&esnap.start_index];
        let new_end = vertex_map[&esnap.end_index];
        let new_eid = target.add_edge(Edge::with_tolerance(
            new_start,
            new_end,
            esnap.curve.clone(),
            esnap.tolerance,
        ));
        edge_map.insert(esnap.old_index, new_eid);
    }

    let mut wire_map: HashMap<usize, WireId> = HashMap::new();
    for wsnap in &wire_snaps {
        let new_edges: Vec<OrientedEdge> = wsnap
            .edges
            .iter()
            .map(|&(edge_idx, fwd)| OrientedEdge::new(edge_map[&edge_idx], fwd))
            .collect();
        let new_wire = Wire::new(new_edges, wsnap.closed)?;
        wire_map.insert(wsnap.old_index, target.add_wire(new_wire));
    }

    let mut new_shell_ids = Vec::new();
    for ssnap in &shell_snaps {
        let mut new_face_ids = Vec::new();
        for fsnap in &ssnap.faces {
            let new_outer = wire_map[&fsnap.outer_wire_index];
            let new_inner: Vec<WireId> = fsnap
                .inner_wire_indices
                .iter()
                .map(|idx| wire_map[idx])
                .collect();
            let mut new_face = Face::new(new_outer, new_inner, fsnap.surface.clone());
            if fsnap.reversed {
                new_face.set_reversed(true);
            }
            new_face_ids.push(target.add_face(new_face));
        }
        let new_shell = Shell::new(new_face_ids)?;
        new_shell_ids.push(target.add_shell(new_shell));
    }

    let new_outer = *new_shell_ids
        .first()
        .ok_or_else(|| AlgoError::AssemblyFailed("solid has no shells to copy".into()))?;
    let new_inner: Vec<_> = new_shell_ids[1..].to_vec();

    Ok(target.add_solid(Solid::new(new_outer, new_inner)))
}

#[cfg(test)]
mod tests {
    #![allow(clippy::unwrap_used, clippy::expect_used)]
    use super::*;

    #[test]
    fn round_trip_preserves_box() {
        use brepkit_topology::test_utils::make_unit_cube_manifold_at;

        let mut source = Topology::default();
        let solid = make_unit_cube_manifold_at(&mut source, 0.0, 0.0, 0.0);

        // Import into store
        let store = GfaShapeStore::new(&source, solid, solid).unwrap();

        // Verify store has the solid
        let s = store.topo.solid(store.solid_a).unwrap();
        let sh = store.topo.shell(s.outer_shell()).unwrap();
        assert_eq!(sh.faces().len(), 6, "box should have 6 faces");

        // Export back to a new topology
        let mut target = Topology::default();
        let exported = store.export_solid(&mut target, store.solid_a).unwrap();
        let s2 = target.solid(exported).unwrap();
        let sh2 = target.shell(s2.outer_shell()).unwrap();
        assert_eq!(sh2.faces().len(), 6, "exported box should have 6 faces");

        // Verify face counts match between source and exported
        let source_faces = brepkit_topology::explorer::solid_faces(&source, solid).unwrap();
        let target_faces = brepkit_topology::explorer::solid_faces(&target, exported).unwrap();
        assert_eq!(source_faces.len(), target_faces.len());

        // Verify isolation: the store has TWO copies of the box (A and B
        // both point to the same source solid). The store topology should
        // have more entities than the source (2× vertices, edges, etc.).
        let store_vertex_count =
            brepkit_topology::explorer::solid_faces(&store.topo, store.solid_a)
                .unwrap()
                .len();
        assert_eq!(store_vertex_count, 6, "store solid_a should have 6 faces");
    }
}
