//! Shape query utilities.

use std::collections::HashMap;

use brepkit_topology::Topology;
use brepkit_topology::edge::EdgeId;
use brepkit_topology::face::{FaceId, FaceSurface};
use brepkit_topology::solid::SolidId;

use crate::OperationsError;

/// Filter edges to only those shared by two planar faces in a solid.
///
/// Given a solid and a set of edge IDs, returns only the edges
/// where both adjacent faces have a planar surface.
///
/// # Errors
///
/// Returns `OperationsError::Topology` if any entity ID is invalid.
pub fn filter_planar_edges(
    topo: &Topology,
    solid_id: SolidId,
    edge_ids: &[EdgeId],
) -> Result<Vec<EdgeId>, OperationsError> {
    let solid_data = topo.solid(solid_id)?;
    let shell = topo.shell(solid_data.outer_shell())?;

    let mut edge_faces: HashMap<usize, Vec<FaceId>> = HashMap::new();
    for &fid in shell.faces() {
        let face = topo.face(fid)?;
        let wire = topo.wire(face.outer_wire())?;
        for oe in wire.edges() {
            edge_faces.entry(oe.edge().index()).or_default().push(fid);
        }
    }

    let mut result = Vec::new();
    for &eid in edge_ids {
        if let Some(adj_faces) = edge_faces.get(&eid.index()) {
            let all_planar = adj_faces.iter().all(|&fid| {
                topo.face(fid)
                    .map(|f| matches!(f.surface(), FaceSurface::Plane { .. }))
                    .unwrap_or(false)
            });
            if all_planar {
                result.push(eid);
            }
        }
    }
    Ok(result)
}
