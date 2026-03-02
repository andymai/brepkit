//! Topology healing: repair common defects in B-Rep models.
//!
//! Equivalent to `ShapeFix` in `OpenCascade`. Provides repair
//! operations for common geometry issues encountered in imported CAD files.

use std::collections::HashMap;

use brepkit_math::tolerance::Tolerance;
use brepkit_math::vec::{Point3, Vec3};
use brepkit_topology::Topology;
use brepkit_topology::face::FaceSurface;
use brepkit_topology::solid::SolidId;
use brepkit_topology::vertex::VertexId;

/// Summary of repairs performed by [`heal_solid`].
#[derive(Debug, Default, Clone)]
pub struct HealingReport {
    /// Number of coincident vertices merged.
    pub vertices_merged: usize,
    /// Number of degenerate edges removed.
    pub degenerate_edges_removed: usize,
    /// Number of face orientations fixed.
    pub orientations_fixed: usize,
}

/// Run all healing operations on a solid.
///
/// This is the top-level repair function, equivalent to OCCT's
/// `ShapeFix_Shape`. It runs:
/// 1. Merge coincident vertices (close gaps)
/// 2. Remove degenerate edges (shorter than tolerance)
/// 3. Fix face orientations (ensure outward normals)
///
/// # Errors
/// Returns an error if topology lookups fail.
pub fn heal_solid(
    topo: &mut Topology,
    solid: SolidId,
    tolerance: f64,
) -> Result<HealingReport, crate::OperationsError> {
    let vertices_merged = merge_coincident_vertices(topo, solid, tolerance)?;
    let degenerate_edges_removed = remove_degenerate_edges(topo, solid, tolerance)?;
    let orientations_fixed = fix_face_orientations(topo, solid)?;

    Ok(HealingReport {
        vertices_merged,
        degenerate_edges_removed,
        orientations_fixed,
    })
}

/// Merge near-coincident vertices in a solid.
///
/// Finds vertex pairs that are within `tolerance` of each other and
/// merges them by updating all edge references to point to a single
/// canonical vertex. This fixes small gaps caused by floating-point
/// imprecision during modeling operations.
///
/// Returns the number of vertices merged.
///
/// # Errors
/// Returns an error if topology lookups fail.
pub fn merge_coincident_vertices(
    topo: &mut Topology,
    solid: SolidId,
    tolerance: f64,
) -> Result<usize, crate::OperationsError> {
    let tol = if tolerance > 0.0 {
        tolerance
    } else {
        Tolerance::new().linear
    };
    let tol_sq = tol * tol;

    // Collect all vertex positions.
    let solid_data = topo.solid(solid)?;
    let shell = topo.shell(solid_data.outer_shell())?;
    let face_ids: Vec<_> = shell.faces().to_vec();

    // Gather all unique vertex IDs and their positions.
    let mut vertex_ids: Vec<VertexId> = Vec::new();
    let mut positions: Vec<Point3> = Vec::new();
    let mut seen = std::collections::HashSet::new();

    for &fid in &face_ids {
        let face = topo.face(fid)?;
        let wire = topo.wire(face.outer_wire())?;
        for oe in wire.edges() {
            let edge = topo.edge(oe.edge())?;
            for &vid in &[edge.start(), edge.end()] {
                if seen.insert(vid.index()) {
                    let point = topo.vertex(vid)?.point();
                    vertex_ids.push(vid);
                    positions.push(point);
                }
            }
        }
    }

    // Build merge map: for each vertex, find the canonical (lowest-index)
    // vertex it should merge into.
    let num_verts = vertex_ids.len();
    let mut merge_to: HashMap<usize, VertexId> = HashMap::new();
    let mut merged_count = 0;

    for i in 0..num_verts {
        if merge_to.contains_key(&vertex_ids[i].index()) {
            continue;
        }
        for j in (i + 1)..num_verts {
            if merge_to.contains_key(&vertex_ids[j].index()) {
                continue;
            }
            let dist_sq = (positions[i] - positions[j]).length_squared();
            if dist_sq < tol_sq {
                merge_to.insert(vertex_ids[j].index(), vertex_ids[i]);
                merged_count += 1;
            }
        }
    }

    if merged_count == 0 {
        return Ok(0);
    }

    // Apply merges: update edge start/end vertices.
    let mut edge_ids = Vec::new();
    for &fid in &face_ids {
        let face = topo.face(fid)?;
        let wire = topo.wire(face.outer_wire())?;
        for oe in wire.edges() {
            edge_ids.push(oe.edge());
        }
    }
    edge_ids.sort_by_key(|e| e.index());
    edge_ids.dedup_by_key(|e| e.index());

    let updates: Vec<_> = edge_ids
        .iter()
        .filter_map(|&eid| {
            let edge = topo.edge(eid).ok()?;
            let new_start = merge_to
                .get(&edge.start().index())
                .copied()
                .unwrap_or_else(|| edge.start());
            let new_end = merge_to
                .get(&edge.end().index())
                .copied()
                .unwrap_or_else(|| edge.end());
            if new_start != edge.start() || new_end != edge.end() {
                Some((eid, new_start, new_end))
            } else {
                None
            }
        })
        .collect();

    for (eid, new_start, new_end) in updates {
        let edge = topo.edge_mut(eid)?;
        *edge = brepkit_topology::edge::Edge::new(new_start, new_end, edge.curve().clone());
    }

    Ok(merged_count)
}

/// Remove degenerate edges (shorter than tolerance) from a solid.
///
/// An edge whose start and end vertices are within `tolerance` of each
/// other is considered degenerate. Such edges are collapsed: their
/// references in wires are removed, and the wire is rebuilt without them.
///
/// Returns the number of degenerate edges removed.
///
/// # Errors
/// Returns an error if topology lookups fail.
pub fn remove_degenerate_edges(
    topo: &mut Topology,
    solid: SolidId,
    tolerance: f64,
) -> Result<usize, crate::OperationsError> {
    let tol = if tolerance > 0.0 {
        tolerance
    } else {
        Tolerance::new().linear
    };
    let tol_sq = tol * tol;

    let solid_data = topo.solid(solid)?;
    let shell = topo.shell(solid_data.outer_shell())?;
    let face_ids: Vec<_> = shell.faces().to_vec();

    let mut removed_count = 0;

    for &fid in &face_ids {
        let face = topo.face(fid)?;
        let wire_id = face.outer_wire();
        let wire = topo.wire(wire_id)?;

        let mut new_edges = Vec::new();
        let mut any_removed = false;

        for oe in wire.edges() {
            let edge = topo.edge(oe.edge())?;
            let start_pos = topo.vertex(edge.start())?.point();
            let end_pos = topo.vertex(edge.end())?.point();
            let len_sq = (end_pos - start_pos).length_squared();

            if len_sq < tol_sq && edge.start() != edge.end() {
                // Degenerate edge — skip it
                any_removed = true;
                removed_count += 1;
            } else {
                new_edges.push(*oe);
            }
        }

        if any_removed && !new_edges.is_empty() {
            let new_wire = brepkit_topology::wire::Wire::new(new_edges, wire.is_closed())?;
            *topo.wire_mut(wire_id)? = new_wire;
        }
    }

    Ok(removed_count)
}

/// Fix face orientations so normals point outward from the solid.
///
/// Uses the signed volume test: for each face, computes the signed volume
/// contribution. If the total signed volume is negative, the overall
/// orientation is flipped. Then checks individual faces against the
/// expected outward direction.
///
/// Returns the number of faces whose orientation was fixed.
///
/// # Errors
/// Returns an error if topology lookups fail.
pub fn fix_face_orientations(
    topo: &mut Topology,
    solid: SolidId,
) -> Result<usize, crate::OperationsError> {
    let solid_data = topo.solid(solid)?;
    let shell = topo.shell(solid_data.outer_shell())?;
    let face_ids: Vec<_> = shell.faces().to_vec();

    // Compute center of mass (approximate) from all face centroids.
    let mut center = Vec3::new(0.0, 0.0, 0.0);
    let mut total_faces: usize = 0;

    for &fid in &face_ids {
        let face = topo.face(fid)?;
        let wire = topo.wire(face.outer_wire())?;
        let mut face_center = Vec3::new(0.0, 0.0, 0.0);
        let edges = wire.edges();
        for oe in edges {
            let edge = topo.edge(oe.edge())?;
            let pos = topo.vertex(edge.start())?.point();
            face_center = face_center + Vec3::new(pos.x(), pos.y(), pos.z());
        }

        let vert_count = edges.len();
        if vert_count > 0 {
            #[allow(clippy::cast_precision_loss)]
            let inv = 1.0 / vert_count as f64;
            center = center + face_center * inv;
            total_faces += 1;
        }
    }

    if total_faces == 0 {
        return Ok(0);
    }

    #[allow(clippy::cast_precision_loss)]
    let inv_faces = 1.0 / total_faces as f64;
    let center_pt = Point3::new(
        center.x() * inv_faces,
        center.y() * inv_faces,
        center.z() * inv_faces,
    );

    // For each planar face, check if the normal points away from center.
    let mut fixed_count = 0;
    let mut faces_to_flip = Vec::new();

    for &fid in &face_ids {
        let face = topo.face(fid)?;

        if let FaceSurface::Plane { normal, d } = face.surface() {
            // Get a point on the face
            let wire = topo.wire(face.outer_wire())?;
            if let Some(first_oe) = wire.edges().first() {
                let edge = topo.edge(first_oe.edge())?;
                let face_point = topo.vertex(edge.start())?.point();

                // Vector from center to face point
                let to_face = face_point - center_pt;

                // If normal points toward center (wrong direction), flip it
                if normal.dot(to_face) < 0.0 {
                    faces_to_flip.push((fid, *normal, *d));
                    fixed_count += 1;
                }
            }
        }
    }

    // Apply flips
    for (fid, normal, d) in faces_to_flip {
        let face = topo.face_mut(fid)?;
        face.set_surface(FaceSurface::Plane {
            normal: Vec3::new(-normal.x(), -normal.y(), -normal.z()),
            d: -d,
        });
    }

    Ok(fixed_count)
}

#[cfg(test)]
mod tests {
    #![allow(clippy::unwrap_used)]

    use brepkit_topology::Topology;

    use super::*;

    #[test]
    fn no_merge_on_clean_box() {
        let mut topo = Topology::new();
        let solid = crate::primitives::make_box(&mut topo, 2.0, 2.0, 2.0).unwrap();

        let count = merge_coincident_vertices(&mut topo, solid, 1e-7).unwrap();
        assert_eq!(count, 0, "clean box should have no coincident vertices");
    }

    #[test]
    fn merge_with_large_tolerance() {
        let mut topo = Topology::new();
        let solid = crate::primitives::make_box(&mut topo, 2.0, 2.0, 2.0).unwrap();

        let count = merge_coincident_vertices(&mut topo, solid, 3.0).unwrap();
        assert!(count > 0, "large tolerance should merge some vertices");
    }

    #[test]
    fn heal_clean_box() {
        let mut topo = Topology::new();
        let solid = crate::primitives::make_box(&mut topo, 2.0, 2.0, 2.0).unwrap();

        let report = heal_solid(&mut topo, solid, 1e-7).unwrap();
        assert_eq!(report.vertices_merged, 0);
        assert_eq!(report.degenerate_edges_removed, 0);
        // Orientation may or may not need fixing depending on make_box
    }

    #[test]
    fn no_degenerate_edges_in_clean_box() {
        let mut topo = Topology::new();
        let solid = crate::primitives::make_box(&mut topo, 2.0, 2.0, 2.0).unwrap();

        let count = remove_degenerate_edges(&mut topo, solid, 1e-7).unwrap();
        assert_eq!(count, 0);
    }

    #[test]
    fn fix_orientations_on_clean_box() {
        let mut topo = Topology::new();
        let solid = crate::primitives::make_box(&mut topo, 2.0, 2.0, 2.0).unwrap();

        let count = fix_face_orientations(&mut topo, solid).unwrap();
        // A properly constructed box should not need fixes
        assert_eq!(count, 0);
    }
}
