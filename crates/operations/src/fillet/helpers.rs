//! Helper types and functions for fillet operations.

use std::collections::HashMap;

use brepkit_math::vec::Point3;
use brepkit_math::vec::Vec3;
use brepkit_topology::edge::EdgeId;
use brepkit_topology::face::FaceId;
use brepkit_topology::vertex::VertexId;

/// Extract inner wire vertex positions from a face's topology.
///
/// Used for non-planar faces that don't have a `FacePolygon` (planar faces
/// store inner wires in `FacePolygon::inner_wires` instead).
pub(super) fn extract_inner_wire_positions(
    topo: &brepkit_topology::Topology,
    face: &brepkit_topology::face::Face,
) -> Result<Vec<Vec<Point3>>, crate::OperationsError> {
    let mut result = Vec::new();
    for &inner_wid in face.inner_wires() {
        let inner_wire = topo.wire(inner_wid)?;
        let mut iw_positions = Vec::new();
        for oe in inner_wire.edges() {
            let edge = topo.edge(oe.edge())?;
            let vid = oe.oriented_start(edge);
            iw_positions.push(topo.vertex(vid)?.point());
        }
        if !iw_positions.is_empty() {
            result.push(iw_positions);
        }
    }
    Ok(result)
}

// ── Internal data structures ───────────────────────────────────────

pub(super) struct FacePolygon {
    pub(super) vertex_ids: Vec<VertexId>,
    pub(super) positions: Vec<Point3>,
    pub(super) wire_edge_ids: Vec<EdgeId>,
    pub(super) normal: Vec3,
    #[allow(dead_code)]
    pub(super) d: f64,
    /// Inner wire vertex positions (holes in the face).
    pub(super) inner_wires: Vec<Vec<Point3>>,
}

pub(super) struct FilletEdgeData {
    points: HashMap<(usize, usize), Point3>,
}

impl FilletEdgeData {
    pub(super) fn new() -> Self {
        Self {
            points: HashMap::new(),
        }
    }

    pub(super) fn insert(&mut self, face_id: FaceId, vertex_id: VertexId, point: Point3) {
        self.points
            .insert((face_id.index(), vertex_id.index()), point);
    }

    pub(super) fn get_point(
        &self,
        face_id: FaceId,
        vertex_id: VertexId,
    ) -> Result<Point3, crate::OperationsError> {
        self.points
            .get(&(face_id.index(), vertex_id.index()))
            .copied()
            .ok_or_else(|| crate::OperationsError::InvalidInput {
                reason: format!(
                    "missing fillet point for face {} vertex {}",
                    face_id.index(),
                    vertex_id.index()
                ),
            })
    }
}

pub(super) fn record_fillet_point(
    data: &mut HashMap<usize, FilletEdgeData>,
    edge_index: usize,
    vertex_id: VertexId,
    face_id: FaceId,
    point: Point3,
) {
    data.entry(edge_index)
        .or_insert_with(FilletEdgeData::new)
        .insert(face_id, vertex_id, point);
}
