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

/// Filter edges to only those the blend engines can fillet: manifold edges
/// (shared by exactly two faces) whose **both** neighbor faces are analytic
/// (plane/cylinder/cone/sphere/torus, not NURBS).
///
/// The walking and rolling-ball fillet engines build the blend section against
/// analytic neighbor surfaces. An edge bordering a NURBS face (e.g. a previous
/// fillet's blend face) is unsupported — attempting it yields a silent no-op or
/// a self-intersecting solid — so callers should skip such edges rather than
/// feed them to the engine.
///
/// # Errors
///
/// Returns `OperationsError::Topology` if any entity ID is invalid.
pub fn filter_filletable_edges(
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
        let Some(adj_faces) = edge_faces.get(&eid.index()) else {
            continue;
        };
        if adj_faces.len() != 2 {
            continue;
        }
        let both_analytic = adj_faces.iter().all(|&fid| {
            topo.face(fid)
                .map(|f| f.surface().is_analytic())
                .unwrap_or(false)
        });
        if both_analytic {
            result.push(eid);
        }
    }
    Ok(result)
}

#[cfg(test)]
mod tests {
    #![allow(clippy::unwrap_used, clippy::expect_used)]

    use brepkit_topology::explorer::solid_edges;

    use super::*;

    #[test]
    fn filletable_edges_all_planar_box() {
        let mut topo = Topology::new();
        let cube = crate::primitives::make_box(&mut topo, 10.0, 10.0, 10.0).unwrap();
        let edges = solid_edges(&topo, cube).unwrap();
        let filletable = filter_filletable_edges(&topo, cube, &edges).unwrap();
        assert_eq!(
            filletable.len(),
            edges.len(),
            "every box edge is plane↔plane and filletable"
        );
        assert_eq!(edges.len(), 12);
    }

    #[test]
    fn filletable_edges_exclude_nurbs_blend_neighbors() {
        let mut topo = Topology::new();
        let cube = crate::primitives::make_box(&mut topo, 10.0, 10.0, 10.0).unwrap();
        let edges = solid_edges(&topo, cube).unwrap();
        // A single fillet introduces a NURBS blend face; its boundary edges
        // border a NURBS neighbor and must be excluded.
        let filleted = crate::blend_ops::fillet_v2(&mut topo, cube, &[edges[0]], 1.0)
            .unwrap()
            .solid;
        let r_edges = solid_edges(&topo, filleted).unwrap();
        let filletable = filter_filletable_edges(&topo, filleted, &r_edges).unwrap();
        assert!(
            filletable.len() < r_edges.len(),
            "edges bordering the NURBS blend must be excluded: {} of {} kept",
            filletable.len(),
            r_edges.len()
        );
        assert!(
            !filletable.is_empty(),
            "the remaining plane↔plane edges stay filletable"
        );
    }
}
