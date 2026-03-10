//! Topology healing: repair common defects in B-Rep models.
//!
//! Equivalent to `ShapeFix` in `OpenCascade`. Provides repair
//! operations for common geometry issues encountered in imported CAD files.

use std::collections::HashMap;

use brepkit_math::tolerance::Tolerance;
use brepkit_math::vec::{Point3, Vec3};
use brepkit_topology::Topology;
use brepkit_topology::face::{FaceId, FaceSurface};
use brepkit_topology::solid::SolidId;
use brepkit_topology::vertex::VertexId;

/// Combined result of [`repair_solid`]: validation before, healing, validation after.
#[derive(Debug, Clone)]
pub struct RepairReport {
    /// Validation issues found before healing.
    pub before: crate::validate::ValidationReport,
    /// Healing actions performed.
    pub healing: HealingReport,
    /// Validation issues remaining after healing.
    pub after: crate::validate::ValidationReport,
}

impl RepairReport {
    /// Whether the solid is valid after repair (no remaining errors).
    #[must_use]
    pub fn is_valid_after(&self) -> bool {
        self.after.is_valid()
    }

    /// Total number of repairs performed.
    #[must_use]
    pub fn total_repairs(&self) -> usize {
        self.healing.vertices_merged
            + self.healing.degenerate_edges_removed
            + self.healing.orientations_fixed
            + self.healing.wire_gaps_closed
            + self.healing.small_faces_removed
            + self.healing.duplicate_faces_removed
    }
}

/// Validate, heal, and re-validate a solid in one pass.
///
/// This is the top-level convenience function for repairing imported models.
/// It chains: `validate_solid` → `heal_solid` → `validate_solid`, returning
/// all three reports so the caller can see what was found, what was fixed,
/// and what remains.
///
/// # Errors
/// Returns an error if topology lookups fail.
pub fn repair_solid(
    topo: &mut Topology,
    solid: SolidId,
    tolerance: f64,
) -> Result<RepairReport, crate::OperationsError> {
    let before = crate::validate::validate_solid(topo, solid)?;
    let healing = heal_solid(topo, solid, tolerance)?;
    let after = crate::validate::validate_solid(topo, solid)?;

    Ok(RepairReport {
        before,
        healing,
        after,
    })
}

/// Summary of repairs performed by [`heal_solid`].
#[derive(Debug, Default, Clone)]
pub struct HealingReport {
    /// Number of coincident vertices merged.
    pub vertices_merged: usize,
    /// Number of degenerate edges removed.
    pub degenerate_edges_removed: usize,
    /// Number of face orientations fixed.
    pub orientations_fixed: usize,
    /// Number of wire gaps closed.
    pub wire_gaps_closed: usize,
    /// Number of small faces removed.
    pub small_faces_removed: usize,
    /// Number of duplicate faces removed.
    pub duplicate_faces_removed: usize,
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
    // Pass 1: fix wire connectivity (must run before vertex merging).
    let wire_gaps_closed = close_wire_gaps(topo, solid, tolerance)?;
    // Pass 2: merge near-coincident vertices.
    let vertices_merged = merge_coincident_vertices(topo, solid, tolerance)?;
    // Pass 3: remove degenerate edges.
    let degenerate_edges_removed = remove_degenerate_edges(topo, solid, tolerance)?;
    // Pass 4: remove small faces.
    let small_faces_removed = remove_small_faces(topo, solid, tolerance)?;
    // Pass 5: remove duplicate faces.
    let duplicate_faces_removed = remove_duplicate_faces(topo, solid, tolerance)?;
    // Pass 6: fix face orientations (run last, after topology is clean).
    let orientations_fixed = fix_face_orientations(topo, solid)?;

    Ok(HealingReport {
        vertices_merged,
        degenerate_edges_removed,
        orientations_fixed,
        wire_gaps_closed,
        small_faces_removed,
        duplicate_faces_removed,
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
        let wire = topo.wire(face.outer_wire())?;
        let first_oe = match wire.edges().first() {
            Some(oe) => oe,
            None => continue,
        };
        let edge = topo.edge(first_oe.edge())?;
        let face_point = topo.vertex(edge.start())?.point();
        let to_face = face_point - center_pt;

        match face.surface() {
            FaceSurface::Plane { normal, d } => {
                if normal.dot(to_face) < 0.0 {
                    faces_to_flip.push((fid, *normal, *d));
                    fixed_count += 1;
                }
            }
            FaceSurface::Cylinder(cyl) => {
                // For cylinders, the outward radial direction should point away from center.
                let to_pt = Vec3::new(
                    face_point.x() - cyl.origin().x(),
                    face_point.y() - cyl.origin().y(),
                    face_point.z() - cyl.origin().z(),
                );
                let h = to_pt.dot(cyl.axis());
                let radial = to_pt - cyl.axis() * h;
                if radial.dot(to_face) < 0.0 {
                    // Cylinder orientation is wrong — but we can only flip planar faces.
                    // For analytic surfaces, orientation is inherent; skip.
                }
            }
            // Non-planar faces: orientation is determined by surface parameterization,
            // not a flippable normal. Skip for now.
            _ => {}
        }
    }

    // Apply flips (only planar faces can be flipped).
    for (fid, normal, d) in faces_to_flip {
        let face = topo.face_mut(fid)?;
        face.set_surface(FaceSurface::Plane {
            normal: Vec3::new(-normal.x(), -normal.y(), -normal.z()),
            d: -d,
        });
    }

    Ok(fixed_count)
}

/// Close gaps between consecutive edges in face wires.
///
/// When two consecutive edges in a wire don't share an endpoint (the end
/// of edge N doesn't match the start of edge N+1), this function closes
/// the gap by merging the mismatched vertices. This is common when
/// importing models from other CAD systems with different tolerances.
///
/// Returns the number of gaps closed.
///
/// # Errors
/// Returns an error if topology lookups fail.
pub fn close_wire_gaps(
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

    let mut gaps_closed = 0;

    for &fid in &face_ids {
        let face = topo.face(fid)?;

        // Check all wires (outer + inner).
        let wire_ids: Vec<_> = std::iter::once(face.outer_wire())
            .chain(face.inner_wires().iter().copied())
            .collect();

        for wire_id in wire_ids {
            let wire = topo.wire(wire_id)?;
            let edges_list: Vec<_> = wire.edges().to_vec();
            let n_edges = edges_list.len();

            if n_edges < 2 {
                continue;
            }

            // For each pair of consecutive edges, check if end of edge i
            // matches start of edge i+1.
            let mut merge_pairs: Vec<(VertexId, VertexId)> = Vec::new();

            for i in 0..n_edges {
                let next_i = (i + 1) % n_edges;

                let edge_i = topo.edge(edges_list[i].edge())?;
                let edge_next = topo.edge(edges_list[next_i].edge())?;

                // End vertex of current edge (accounting for orientation).
                let end_vid = if edges_list[i].is_forward() {
                    edge_i.end()
                } else {
                    edge_i.start()
                };

                // Start vertex of next edge (accounting for orientation).
                let start_vid = if edges_list[next_i].is_forward() {
                    edge_next.start()
                } else {
                    edge_next.end()
                };

                if end_vid == start_vid {
                    continue; // Already connected
                }

                let end_pos = topo.vertex(end_vid)?.point();
                let start_pos = topo.vertex(start_vid)?.point();
                let dist_sq = (end_pos - start_pos).length_squared();

                if dist_sq < tol_sq {
                    // Close the gap by merging the vertices.
                    merge_pairs.push((start_vid, end_vid)); // merge start into end
                }
            }

            // Apply merges using "snapshot then allocate" pattern.
            for (merge_from, merge_to) in &merge_pairs {
                // Snapshot: collect all edges that need updating.
                let solid_d = topo.solid(solid)?;
                let sh = topo.shell(solid_d.outer_shell())?;
                let fids: Vec<_> = sh.faces().to_vec();

                let mut updates = Vec::new();
                for &fid2 in &fids {
                    let f = topo.face(fid2)?;
                    let w = topo.wire(f.outer_wire())?;
                    for oe in w.edges() {
                        let edge = topo.edge(oe.edge())?;
                        let cur_start = edge.start();
                        let cur_end = edge.end();
                        let new_start = if cur_start == *merge_from {
                            *merge_to
                        } else {
                            cur_start
                        };
                        let new_end = if cur_end == *merge_from {
                            *merge_to
                        } else {
                            cur_end
                        };
                        if new_start != cur_start || new_end != cur_end {
                            let curve = edge.curve().clone();
                            updates.push((oe.edge(), new_start, new_end, curve));
                        }
                    }
                }

                // Allocate: apply the updates.
                for (eid, new_start, new_end, curve) in updates {
                    let em = topo.edge_mut(eid)?;
                    *em = brepkit_topology::edge::Edge::new(new_start, new_end, curve);
                }
                gaps_closed += 1;
            }
        }
    }

    Ok(gaps_closed)
}

/// Remove faces smaller than a minimum area threshold.
///
/// Faces with a bounding-box diagonal smaller than `tolerance` are
/// considered degenerate slivers and are removed from the shell.
/// This is common after boolean operations that produce micro-faces
/// at near-tangent intersections.
///
/// Returns the number of faces removed.
///
/// # Errors
/// Returns an error if topology lookups fail.
pub fn remove_small_faces(
    topo: &mut Topology,
    solid: SolidId,
    tolerance: f64,
) -> Result<usize, crate::OperationsError> {
    let tol = if tolerance > 0.0 {
        tolerance
    } else {
        Tolerance::new().linear
    };

    let solid_data = topo.solid(solid)?;
    let shell_id = solid_data.outer_shell();
    let shell = topo.shell(shell_id)?;
    let face_ids: Vec<_> = shell.faces().to_vec();

    let mut small_faces: Vec<FaceId> = Vec::new();

    for &fid in &face_ids {
        let face = topo.face(fid)?;
        let wire = topo.wire(face.outer_wire())?;

        // Compute bounding box of the face's outer wire.
        let mut min_pt = Vec3::new(f64::MAX, f64::MAX, f64::MAX);
        let mut max_pt = Vec3::new(f64::MIN, f64::MIN, f64::MIN);

        for oe in wire.edges() {
            let edge = topo.edge(oe.edge())?;
            for &vid in &[edge.start(), edge.end()] {
                let pos = topo.vertex(vid)?.point();
                min_pt = Vec3::new(
                    min_pt.x().min(pos.x()),
                    min_pt.y().min(pos.y()),
                    min_pt.z().min(pos.z()),
                );
                max_pt = Vec3::new(
                    max_pt.x().max(pos.x()),
                    max_pt.y().max(pos.y()),
                    max_pt.z().max(pos.z()),
                );
            }
        }

        let diagonal = (max_pt - min_pt).length();
        if diagonal < tol {
            small_faces.push(fid);
        }
    }

    if small_faces.is_empty() {
        return Ok(0);
    }

    let removed_count = small_faces.len();
    let small_set: std::collections::HashSet<usize> =
        small_faces.iter().map(|f| f.index()).collect();

    // Rebuild the shell without the small faces.
    let remaining: Vec<FaceId> = face_ids
        .into_iter()
        .filter(|f| !small_set.contains(&f.index()))
        .collect();

    if remaining.is_empty() {
        return Ok(0); // Don't remove ALL faces
    }

    let new_shell =
        brepkit_topology::shell::Shell::new(remaining).map_err(crate::OperationsError::Topology)?;
    *topo.shell_mut(shell_id)? = new_shell;

    Ok(removed_count)
}

/// Remove duplicate (coincident) faces from a solid.
///
/// Two faces are considered duplicates if their outward normals are
/// parallel (or anti-parallel) and all vertices of one face are within
/// `tolerance` of the other face's plane. This happens when boolean
/// operations create overlapping fragments.
///
/// Returns the number of duplicate faces removed.
///
/// # Errors
/// Returns an error if topology lookups fail.
pub fn remove_duplicate_faces(
    topo: &mut Topology,
    solid: SolidId,
    tolerance: f64,
) -> Result<usize, crate::OperationsError> {
    let tol = if tolerance > 0.0 {
        tolerance
    } else {
        Tolerance::new().linear
    };

    let solid_data = topo.solid(solid)?;
    let shell_id = solid_data.outer_shell();
    let shell = topo.shell(shell_id)?;
    let face_ids: Vec<_> = shell.faces().to_vec();

    // Collect face data for comparison.
    // Tuple: (centroid, normal, vertex_count)
    let mut face_data: Vec<(FaceId, Point3, Vec3, usize)> = Vec::new();

    for &fid in &face_ids {
        let face = topo.face(fid)?;
        let normal = match face.surface() {
            FaceSurface::Plane { normal, .. } => *normal,
            FaceSurface::Cylinder(cyl) => cyl.axis(),
            FaceSurface::Cone(cone) => cone.axis(),
            FaceSurface::Sphere(_) => Vec3::new(0.0, 0.0, 1.0), // placeholder for comparison
            FaceSurface::Torus(tor) => tor.z_axis(),
            FaceSurface::Nurbs(_) => continue, // NURBS dedup needs parameter-space comparison
        };

        let wire = topo.wire(face.outer_wire())?;
        let mut centroid = Vec3::new(0.0, 0.0, 0.0);
        let mut count = 0;

        for oe in wire.edges() {
            let edge = topo.edge(oe.edge())?;
            let pos = topo.vertex(edge.start())?.point();
            centroid = centroid + Vec3::new(pos.x(), pos.y(), pos.z());
            count += 1;
        }

        if count > 0 {
            #[allow(clippy::cast_precision_loss)]
            let inv = 1.0 / count as f64;
            centroid = centroid * inv;
        }

        let centroid_pt = Point3::new(centroid.x(), centroid.y(), centroid.z());
        face_data.push((fid, centroid_pt, normal, count));
    }

    // Find duplicate pairs: same vertex count, parallel normals, close centroids.
    let mut duplicates: std::collections::HashSet<usize> = std::collections::HashSet::new();

    for i in 0..face_data.len() {
        if duplicates.contains(&face_data[i].0.index()) {
            continue;
        }
        for j in (i + 1)..face_data.len() {
            if duplicates.contains(&face_data[j].0.index()) {
                continue;
            }

            let (_, centroid_a, normal_a, count_a) = &face_data[i];
            let (fid_j, centroid_b, normal_b, count_b) = &face_data[j];

            // Same vertex count.
            if count_a != count_b {
                continue;
            }

            // Normals parallel or anti-parallel.
            let dot = normal_a.dot(*normal_b).abs();
            if dot < 1.0 - tol {
                continue;
            }

            // Centroids close.
            let centroid_dist = (*centroid_a - *centroid_b).length();
            if centroid_dist < tol {
                duplicates.insert(fid_j.index());
            }
        }
    }

    if duplicates.is_empty() {
        return Ok(0);
    }

    let removed_count = duplicates.len();

    // Rebuild shell without duplicates.
    let remaining: Vec<FaceId> = face_ids
        .into_iter()
        .filter(|f| !duplicates.contains(&f.index()))
        .collect();

    if remaining.is_empty() {
        return Ok(0);
    }

    let new_shell =
        brepkit_topology::shell::Shell::new(remaining).map_err(crate::OperationsError::Topology)?;
    *topo.shell_mut(shell_id)? = new_shell;

    Ok(removed_count)
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

    // ── Wire gap closure tests ──────────────────────────

    #[test]
    fn close_wire_gaps_clean_box() {
        let mut topo = Topology::new();
        let solid = crate::primitives::make_box(&mut topo, 1.0, 1.0, 1.0).unwrap();

        let count = close_wire_gaps(&mut topo, solid, 1e-7).unwrap();
        assert_eq!(count, 0, "clean box should have no wire gaps");
    }

    // ── Small face removal tests ────────────────────────

    #[test]
    fn no_small_faces_in_box() {
        let mut topo = Topology::new();
        let solid = crate::primitives::make_box(&mut topo, 1.0, 1.0, 1.0).unwrap();

        let count = remove_small_faces(&mut topo, solid, 0.01).unwrap();
        assert_eq!(count, 0, "unit box should have no small faces");
    }

    // ── Duplicate face removal tests ────────────────────

    #[test]
    fn no_duplicates_in_box() {
        let mut topo = Topology::new();
        let solid = crate::primitives::make_box(&mut topo, 1.0, 1.0, 1.0).unwrap();

        let count = remove_duplicate_faces(&mut topo, solid, 1e-7).unwrap();
        assert_eq!(count, 0, "box should have no duplicate faces");
    }

    // ── Full heal pipeline tests ────────────────────────

    #[test]
    fn heal_report_all_fields() {
        let mut topo = Topology::new();
        let solid = crate::primitives::make_box(&mut topo, 1.0, 1.0, 1.0).unwrap();

        let report = heal_solid(&mut topo, solid, 1e-7).unwrap();

        // Clean box should need minimal or no repairs.
        assert_eq!(report.wire_gaps_closed, 0);
        assert_eq!(report.vertices_merged, 0);
        assert_eq!(report.degenerate_edges_removed, 0);
        assert_eq!(report.small_faces_removed, 0);
        assert_eq!(report.duplicate_faces_removed, 0);
    }

    #[test]
    fn heal_cylinder_no_crash() {
        let mut topo = Topology::new();
        let solid = crate::primitives::make_cylinder(&mut topo, 1.0, 2.0).unwrap();

        // Healing should not crash on cylinders (has non-planar faces).
        let report = heal_solid(&mut topo, solid, 1e-7).unwrap();
        assert_eq!(report.wire_gaps_closed, 0);
    }

    #[test]
    fn heal_clean_geometry_zero_repairs() {
        let mut topo = Topology::new();
        let solid = crate::primitives::make_box(&mut topo, 1.0, 1.0, 1.0).unwrap();

        let report = repair_solid(&mut topo, solid, 1e-7).unwrap();
        assert_eq!(
            report.total_repairs(),
            0,
            "clean box should need zero repairs, got {}",
            report.total_repairs()
        );
        assert!(
            report.is_valid_after(),
            "clean box should be valid after repair"
        );
    }

    #[test]
    fn heal_preserves_volume() {
        let mut topo = Topology::new();
        let solid = crate::primitives::make_box(&mut topo, 3.0, 4.0, 5.0).unwrap();

        let vol_before = crate::measure::solid_volume(&topo, solid, 0.1).unwrap();
        let _report = repair_solid(&mut topo, solid, 1e-7).unwrap();
        let vol_after = crate::measure::solid_volume(&topo, solid, 0.1).unwrap();

        assert!(
            (vol_before - vol_after).abs() < 0.01,
            "heal should preserve volume: before={vol_before}, after={vol_after}"
        );
    }

    #[test]
    fn heal_preserves_box_volume() {
        let mut topo = Topology::new();
        let solid = crate::primitives::make_box(&mut topo, 2.0, 3.0, 4.0).unwrap();

        let vol_before = crate::measure::solid_volume(&topo, solid, 0.1).unwrap();
        let _report = heal_solid(&mut topo, solid, 1e-7).unwrap();
        let vol_after = crate::measure::solid_volume(&topo, solid, 0.1).unwrap();

        assert!(
            (vol_before - vol_after).abs() < 0.1,
            "healing should preserve volume: before={vol_before}, after={vol_after}"
        );
    }
}
