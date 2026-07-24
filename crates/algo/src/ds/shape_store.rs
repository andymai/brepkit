//! Separate shape store for the GFA pipeline.
//!
//! The `GfaShapeStore` deep-copies input solids into an isolated Topology,
//! runs the entire GFA pipeline on the copies, then exports the result
//! back to the caller's Topology. This eliminates vertex/edge identity
//! conflicts between input solids and the GFA result.

use std::collections::HashMap;

use brepkit_topology::Topology;
use brepkit_topology::edge::{Edge, EdgeId};
use brepkit_topology::face::{Face, FaceId};
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
    /// Maps a store-local input face index back to the caller's original face
    /// index, for both operands — shape-evolution provenance across the copy.
    pub input_face_to_caller: HashMap<usize, usize>,
}

impl GfaShapeStore {
    /// Create a new GFA shape store by deep-copying two input solids.
    ///
    /// # Errors
    ///
    /// Returns [`AlgoError`] if any topology lookup fails.
    pub fn new(source: &Topology, orig_a: SolidId, orig_b: SolidId) -> Result<Self, AlgoError> {
        let mut topo = Topology::default();

        let (solid_a, map_a) = deep_copy_solid(source, &mut topo, orig_a)?;
        let (solid_b, map_b) = deep_copy_solid(source, &mut topo, orig_b)?;

        // Invert the caller-index -> store-face maps into store-face-index ->
        // caller-face-index so a store input face resolves to its caller origin.
        let mut input_face_to_caller = HashMap::new();
        for (caller_idx, store_face) in map_a.into_iter().chain(map_b) {
            input_face_to_caller.insert(store_face.index(), caller_idx);
        }

        Ok(Self {
            topo,
            solid_a,
            solid_b,
            input_face_to_caller,
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
        Ok(deep_copy_solid(&self.topo, target, solid_id)?.0)
    }

    /// Like [`Self::export_solid`], but also returns the map from each store
    /// result-face index to the new caller face it was copied to, for
    /// translating shape-evolution provenance out of the store.
    ///
    /// # Errors
    ///
    /// Returns [`AlgoError`] if any topology lookup fails.
    pub fn export_solid_with_face_map(
        &self,
        target: &mut Topology,
        solid_id: SolidId,
    ) -> Result<(SolidId, HashMap<usize, FaceId>), AlgoError> {
        deep_copy_solid(&self.topo, target, solid_id)
    }
}

/// Isolated shape store for an N-way GFA fuse.
///
/// Like [`GfaShapeStore`] but holds an arbitrary number of deep-copied source
/// solids in one store topology, for the N-way fuse pipeline (one arrangement
/// over all operands rather than sequential pairwise booleans). The 2-operand
/// [`GfaShapeStore`] is unchanged; this is additive.
///
/// Foundation for the N-way fuse: consumed by the forthcoming
/// `run_pave_filler_n` orchestration (see `project_nway-gfa-fuse`). Marked
/// `allow(dead_code)` only until that consumer lands in the next increment.
#[allow(dead_code)]
pub struct GfaShapeStoreN {
    /// The store's own topology arena.
    pub topo: Topology,
    /// Store-local `SolidId` for each deep-copied source, in input order.
    pub sources: Vec<SolidId>,
    /// Store-local input face → the source index (position in `sources`) it
    /// belongs to. The basis for per-sub-face source tagging downstream.
    pub face_source: HashMap<FaceId, usize>,
    /// Store-local input-face index → caller-topology face index, for
    /// shape-evolution provenance across the copy.
    pub input_face_to_caller: HashMap<usize, usize>,
}

#[allow(dead_code)] // Consumed by run_pave_filler_n in the next N-way increment.
impl GfaShapeStoreN {
    /// Create a store by deep-copying every solid in `origs` into one topology.
    ///
    /// # Errors
    ///
    /// Returns [`AlgoError`] if `origs` is empty or any topology lookup fails.
    pub fn new(source: &Topology, origs: &[SolidId]) -> Result<Self, AlgoError> {
        if origs.is_empty() {
            return Err(AlgoError::AssemblyFailed(
                "N-way fuse store needs at least one source solid".into(),
            ));
        }

        let mut topo = Topology::default();
        let mut sources = Vec::with_capacity(origs.len());
        let mut face_source = HashMap::new();
        let mut input_face_to_caller = HashMap::new();

        for (src_idx, &orig) in origs.iter().enumerate() {
            let (solid, map) = deep_copy_solid(source, &mut topo, orig)?;
            sources.push(solid);
            for (caller_idx, store_face) in map {
                face_source.insert(store_face, src_idx);
                input_face_to_caller.insert(store_face.index(), caller_idx);
            }
        }

        Ok(Self {
            topo,
            sources,
            face_source,
            input_face_to_caller,
        })
    }

    /// Export a result solid from this store back to the caller's topology.
    ///
    /// # Errors
    ///
    /// Returns [`AlgoError`] if any topology lookup fails.
    pub fn export_solid(
        &self,
        target: &mut Topology,
        solid_id: SolidId,
    ) -> Result<SolidId, AlgoError> {
        Ok(deep_copy_solid(&self.topo, target, solid_id)?.0)
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
) -> Result<(SolidId, HashMap<usize, FaceId>), AlgoError> {
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
        old_index: usize,
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
                old_index: face_id.index(),
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
    // Provenance: source face index -> the new face it was copied to.
    let mut face_map: HashMap<usize, FaceId> = HashMap::new();
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
            let new_fid = target.add_face(new_face);
            face_map.insert(fsnap.old_index, new_fid);
            new_face_ids.push(new_fid);
        }
        let new_shell = Shell::new(new_face_ids)?;
        new_shell_ids.push(target.add_shell(new_shell));
    }

    let new_outer = *new_shell_ids
        .first()
        .ok_or_else(|| AlgoError::AssemblyFailed("solid has no shells to copy".into()))?;
    let new_inner: Vec<_> = new_shell_ids[1..].to_vec();

    Ok((target.add_solid(Solid::new(new_outer, new_inner)), face_map))
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

        let store = GfaShapeStore::new(&source, solid, solid).unwrap();

        let s = store.topo.solid(store.solid_a).unwrap();
        let sh = store.topo.shell(s.outer_shell()).unwrap();
        assert_eq!(sh.faces().len(), 6, "box should have 6 faces");

        let mut target = Topology::default();
        let exported = store.export_solid(&mut target, store.solid_a).unwrap();
        let s2 = target.solid(exported).unwrap();
        let sh2 = target.shell(s2.outer_shell()).unwrap();
        assert_eq!(sh2.faces().len(), 6, "exported box should have 6 faces");

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

    #[test]
    fn nway_store_copies_all_sources_and_tags_faces() {
        use brepkit_topology::explorer::solid_faces;
        use brepkit_topology::test_utils::make_unit_cube_manifold_at;

        let mut source = Topology::default();
        let boxes = [
            make_unit_cube_manifold_at(&mut source, 0.0, 0.0, 0.0),
            make_unit_cube_manifold_at(&mut source, 0.5, 0.0, 0.0),
            make_unit_cube_manifold_at(&mut source, 1.0, 0.0, 0.0),
        ];

        let store = GfaShapeStoreN::new(&source, &boxes).unwrap();
        assert_eq!(store.sources.len(), 3, "three sources copied");

        // Every source is an independent 6-face box, and every one of its faces
        // is tagged with that source's index.
        for (src_idx, &sid) in store.sources.iter().enumerate() {
            let faces = solid_faces(&store.topo, sid).unwrap();
            assert_eq!(faces.len(), 6, "source {src_idx} is a 6-face box");
            for fid in faces {
                assert_eq!(
                    store.face_source.get(&fid).copied(),
                    Some(src_idx),
                    "face {fid:?} tagged to source {src_idx}"
                );
            }
        }

        // 3 independent boxes -> 18 distinct faces, all source-tagged.
        assert_eq!(store.face_source.len(), 18, "18 distinct tagged faces");

        // Export round-trips one source into a fresh topology.
        let mut target = Topology::default();
        let exported = store.export_solid(&mut target, store.sources[1]).unwrap();
        assert_eq!(solid_faces(&target, exported).unwrap().len(), 6);
    }

    #[test]
    fn nway_store_rejects_empty() {
        let source = Topology::default();
        assert!(GfaShapeStoreN::new(&source, &[]).is_err());
    }
}
