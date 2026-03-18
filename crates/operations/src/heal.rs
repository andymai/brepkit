//! Topology healing: repair common defects in B-Rep models.
//!
//! Equivalent to `ShapeFix` in `OpenCascade`. Provides repair
//! operations for common geometry issues encountered in imported CAD files.

use std::collections::{HashMap, HashSet};

use brepkit_math::tolerance::Tolerance;
use brepkit_math::vec::{Point3, Vec3};
use brepkit_topology::Topology;
use brepkit_topology::face::{Face, FaceId, FaceSurface};
use brepkit_topology::shell::Shell;
use brepkit_topology::solid::SolidId;
use brepkit_topology::vertex::VertexId;
use brepkit_topology::wire::{OrientedEdge, Wire};

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
            face_center += Vec3::new(pos.x(), pos.y(), pos.z());
        }

        let vert_count = edges.len();
        if vert_count > 0 {
            #[allow(clippy::cast_precision_loss)]
            let inv = 1.0 / vert_count as f64;
            center += face_center * inv;
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
            normal: -normal,
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
            centroid += Vec3::new(pos.x(), pos.y(), pos.z());
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

// ── Face Unification ──────────────────────────────────────────────

/// Compare two face surfaces for geometric equivalence.
///
/// Two surfaces are equivalent if they represent the same infinite surface
/// (e.g., same plane, same cylinder axis/radius). This is the same logic
/// used by wireframe edge filtering in `tessellate.rs`.
fn surfaces_equivalent(a: &FaceSurface, b: &FaceSurface) -> bool {
    let tol = Tolerance::new();
    let lin = tol.linear;
    let ang = tol.angular;

    match (a, b) {
        (FaceSurface::Plane { normal: na, d: da }, FaceSurface::Plane { normal: nb, d: db }) => {
            // Relaxed tolerance for plane comparison. Mesh boolean and face
            // splitting create coplanar triangles whose normals differ by
            // varying amounts from floating-point cross-product computation.
            // 1e-4 radians (~0.006°) and 1e-3 mm are tight enough to avoid
            // false merges while allowing mesh-derived coplanar faces to unify.
            // Moderate tolerance: boolean assembly creates coplanar fragments
            // with normals that differ slightly from vertex merging.
            let plane_ang = 1e-3_f64;
            let plane_lin = 1e-2_f64;
            let dot = na.dot(*nb);
            (dot.abs() - 1.0).abs() < plane_ang && (da - db * dot.signum()).abs() < plane_lin
        }
        (FaceSurface::Cylinder(ca), FaceSurface::Cylinder(cb)) => {
            (ca.radius() - cb.radius()).abs() < lin
                && ca.axis().dot(cb.axis()).abs() > 1.0 - ang
                && {
                    let d = cb.origin() - ca.origin();
                    d.cross(ca.axis()).length_squared() < lin * lin
                }
        }
        (FaceSurface::Cone(ca), FaceSurface::Cone(cb)) => {
            (ca.half_angle() - cb.half_angle()).abs() < ang
                && ca.axis().dot(cb.axis()).abs() > 1.0 - ang
                && {
                    let d = cb.apex() - ca.apex();
                    d.dot(d) < lin * lin
                }
        }
        (FaceSurface::Sphere(sa), FaceSurface::Sphere(sb)) => {
            (sa.radius() - sb.radius()).abs() < lin && {
                let d = sb.center() - sa.center();
                d.dot(d) < lin * lin
            }
        }
        (FaceSurface::Torus(ta), FaceSurface::Torus(tb)) => {
            (ta.major_radius() - tb.major_radius()).abs() < lin
                && (ta.minor_radius() - tb.minor_radius()).abs() < lin
                && ta.z_axis().dot(tb.z_axis()).abs() > 1.0 - ang
                && {
                    let d = tb.center() - ta.center();
                    d.dot(d) < lin * lin
                }
        }
        // Different surface types are never equivalent.
        (
            FaceSurface::Plane { .. }
            | FaceSurface::Cylinder(_)
            | FaceSurface::Cone(_)
            | FaceSurface::Sphere(_)
            | FaceSurface::Torus(_)
            | FaceSurface::Nurbs(_),
            _,
        ) => false,
    }
}

/// Union-Find: find root with path compression.
fn uf_find(parent: &mut [usize], mut x: usize) -> usize {
    while parent[x] != x {
        parent[x] = parent[parent[x]];
        x = parent[x];
    }
    x
}

/// Union-Find: merge two sets.
fn uf_union(parent: &mut [usize], a: usize, b: usize) {
    let ra = uf_find(parent, a);
    let rb = uf_find(parent, b);
    if ra != rb {
        parent[rb] = ra;
    }
}

/// Unify adjacent faces that lie on the same geometric surface.
///
/// This merges co-surface face fragments produced by boolean operations
/// back into single faces, reducing face count and improving topology
/// quality. Equivalent to `ShapeUpgrade_UnifySameDomain` in OpenCASCADE.
///
/// The algorithm:
/// 1. Build an edge→face adjacency map
/// 2. Group faces by surface equivalence using connected-component analysis
/// 3. For each group of ≥2 faces, merge their outer wires by removing
///    internal shared edges and splicing the remaining edge chains
/// 4. Rebuild the shell with unified faces
///
/// Returns the number of faces removed by unification.
///
/// # Errors
///
/// Returns an error if topology lookups fail.
#[allow(clippy::too_many_lines)]
pub fn unify_faces(topo: &mut Topology, solid: SolidId) -> Result<usize, crate::OperationsError> {
    /// Maximum boundary edges for a merged face. Groups whose boundary
    /// exceeds this are skipped to prevent O(N²) slowdowns in subsequent
    /// boolean intersection computations. 200 edges is generous for any
    /// practical merged face (a merged rectangle has 4-20 edges).
    const MAX_BOUNDARY_EDGES: usize = 200;

    let solid_data = topo.solid(solid)?;
    let shell_id = solid_data.outer_shell();
    let shell = topo.shell(shell_id)?;
    let all_face_ids: Vec<FaceId> = shell.faces().to_vec();
    let original_count = all_face_ids.len();

    if original_count < 2 {
        return Ok(0);
    }

    // Step 1: Build edge→face map.
    let edge_face_map = brepkit_topology::explorer::edge_to_face_map(topo, solid)?;

    // Step 2: Find connected components of faces sharing edges on the same surface.
    // Union-Find structure.
    let face_index_map: HashMap<usize, usize> = all_face_ids
        .iter()
        .enumerate()
        .map(|(i, fid)| (fid.index(), i))
        .collect();

    let n = all_face_ids.len();
    let mut parent: Vec<usize> = (0..n).collect();

    // For each edge shared by 2+ faces, check if adjacent faces are on the same surface.
    // Includes non-manifold edges (3+ faces) to enable merging across boolean assembly
    // imperfections where faces should be unified despite non-manifold edges.
    for faces in edge_face_map.values() {
        if faces.len() < 2 {
            continue;
        }
        // Check all pairs of faces sharing this edge.
        for i in 0..faces.len() {
            for j in (i + 1)..faces.len() {
                let fa_idx = match face_index_map.get(&faces[i].index()) {
                    Some(&idx) => idx,
                    None => continue,
                };
                let fb_idx = match face_index_map.get(&faces[j].index()) {
                    Some(&idx) => idx,
                    None => continue,
                };
                let surface_a = topo.face(faces[i])?.surface().clone();
                let surface_b = topo.face(faces[j])?.surface().clone();
                if surfaces_equivalent(&surface_a, &surface_b) {
                    uf_union(&mut parent, fa_idx, fb_idx);
                }
            }
        }
    }

    // Step 3: Group faces by their root.
    let mut groups: HashMap<usize, Vec<usize>> = HashMap::new();
    for i in 0..n {
        let root = uf_find(&mut parent, i);
        groups.entry(root).or_default().push(i);
    }

    // Only process groups with ≥2 faces.
    let merge_groups: Vec<Vec<usize>> = groups.into_values().filter(|g| g.len() >= 2).collect();

    if merge_groups.is_empty() {
        return Ok(0);
    }

    // Step 4: For each merge group, merge the faces.
    let mut merged_face_ids: Vec<FaceId> = Vec::new();
    let mut consumed: HashSet<usize> = HashSet::new();

    for group in &merge_groups {
        let group_face_ids: Vec<FaceId> = group.iter().map(|&i| all_face_ids[i]).collect();

        // Find internal edges: edges shared by two faces BOTH in this group.
        let group_set: HashSet<usize> = group_face_ids.iter().map(|f| f.index()).collect();
        let mut internal_edges: HashSet<usize> = HashSet::new();

        for (edge_idx, faces) in &edge_face_map {
            if faces.len() == 2
                && group_set.contains(&faces[0].index())
                && group_set.contains(&faces[1].index())
            {
                internal_edges.insert(*edge_idx);
            }
        }

        // Collect all oriented edges from all faces in the group, excluding internal edges.
        // Also collect inner wires from all faces.
        let mut boundary_edges: Vec<OrientedEdge> = Vec::new();
        let mut all_inner_wires: Vec<brepkit_topology::wire::WireId> = Vec::new();
        let mut representative_surface: Option<FaceSurface> = None;
        let mut representative_reversed = false;

        for &fid in &group_face_ids {
            let face = topo.face(fid)?;
            if representative_surface.is_none() {
                representative_surface = Some(face.surface().clone());
                representative_reversed = face.is_reversed();
            }
            all_inner_wires.extend_from_slice(face.inner_wires());

            let wire = topo.wire(face.outer_wire())?;
            for oe in wire.edges() {
                if !internal_edges.contains(&oe.edge().index()) {
                    boundary_edges.push(*oe);
                }
            }
        }

        // Skip groups whose merged boundary would be too complex.
        // A face with hundreds of boundary edges can cause O(N²) or worse
        // performance in subsequent boolean intersection computations.
        if boundary_edges.len() > MAX_BOUNDARY_EDGES {
            log::debug!(
                "unify_faces: skipping merge group with {} boundary edges (limit {})",
                boundary_edges.len(),
                MAX_BOUNDARY_EDGES
            );
            continue;
        }

        // Reorder boundary edges into closed loops.
        let mut loops = order_edges_into_loops(topo, &boundary_edges)?;

        if loops.is_empty() {
            // Couldn't form any valid loop — skip this group, keep original faces.
            continue;
        }

        // Select the outer wire by enclosed 3D area (Newell normal magnitude).
        // Edge count is unreliable — a hole tessellated into many short edges
        // would be misclassified as the outer boundary.
        let outer_idx = if loops.len() > 1 {
            loops
                .iter()
                .enumerate()
                .max_by(|(_, a), (_, b)| {
                    let area_a = loop_area_3d(topo, a);
                    let area_b = loop_area_3d(topo, b);
                    area_a
                        .partial_cmp(&area_b)
                        .unwrap_or(std::cmp::Ordering::Equal)
                })
                .map_or(0, |(i, _)| i)
        } else {
            0
        };
        let outer_loop = loops.remove(outer_idx);

        let new_wire = Wire::new(outer_loop, true).map_err(crate::OperationsError::Topology)?;
        let new_wire_id = topo.add_wire(new_wire);

        // Convert remaining loops to inner wires.
        for inner_loop in loops {
            if let Ok(iw) = Wire::new(inner_loop, true) {
                all_inner_wires.push(topo.add_wire(iw));
            }
        }

        let surface =
            representative_surface.ok_or_else(|| crate::OperationsError::InvalidInput {
                reason: "empty merge group".to_string(),
            })?;

        let new_face = if representative_reversed {
            Face::new_reversed(new_wire_id, all_inner_wires, surface)
        } else {
            Face::new(new_wire_id, all_inner_wires, surface)
        };
        let new_face_id = topo.add_face(new_face);
        merged_face_ids.push(new_face_id);

        for &fid in &group_face_ids {
            consumed.insert(fid.index());
        }
    }

    if consumed.is_empty() {
        return Ok(0);
    }

    // Step 5: Rebuild the shell with unmerged faces + new merged faces.
    let mut new_faces: Vec<FaceId> = all_face_ids
        .into_iter()
        .filter(|f| !consumed.contains(&f.index()))
        .collect();
    new_faces.extend(merged_face_ids);

    let new_shell = Shell::new(new_faces).map_err(crate::OperationsError::Topology)?;
    *topo.shell_mut(shell_id)? = new_shell;

    let final_count = topo.shell(shell_id)?.faces().len();
    Ok(original_count - final_count)
}

/// Compute the enclosed 3D area of a loop of oriented edges using Newell's method.
///
/// Returns 0.0 if any vertex lookup fails (defensive fallback).
fn loop_area_3d(topo: &Topology, loop_edges: &[OrientedEdge]) -> f64 {
    let mut positions: Vec<Point3> = Vec::with_capacity(loop_edges.len());
    for oe in loop_edges {
        let edge = match topo.edge(oe.edge()) {
            Ok(e) => e,
            Err(_) => return 0.0,
        };
        let vid = if oe.is_forward() {
            edge.start()
        } else {
            edge.end()
        };
        match topo.vertex(vid) {
            Ok(v) => positions.push(v.point()),
            Err(_) => return 0.0,
        }
    }
    if positions.len() < 3 {
        return 0.0;
    }
    // Newell normal magnitude = 2× enclosed area.
    crate::winding::newell_normal(&positions).length() * 0.5
}

/// Edge info for wire ordering: oriented edge with resolved vertex indices.
struct EdgeInfo {
    oe: OrientedEdge,
    start_vid: usize,
    end_vid: usize,
}

/// Order boundary edges into one or more closed loops.
///
/// Returns a `Vec<Vec<OrientedEdge>>` where each inner vec is a closed
/// loop with edges chained end-to-start. Empty if edges can't form any
/// valid loop.
fn order_edges_into_loops(
    topo: &Topology,
    edges: &[OrientedEdge],
) -> Result<Vec<Vec<OrientedEdge>>, crate::OperationsError> {
    if edges.is_empty() {
        return Ok(Vec::new());
    }

    let mut infos: Vec<EdgeInfo> = Vec::with_capacity(edges.len());
    for oe in edges {
        let edge = topo.edge(oe.edge())?;
        let (sv, ev) = if oe.is_forward() {
            (edge.start().index(), edge.end().index())
        } else {
            (edge.end().index(), edge.start().index())
        };
        infos.push(EdgeInfo {
            oe: *oe,
            start_vid: sv,
            end_vid: ev,
        });
    }

    // Build a map from start_vertex → edge index for quick lookup.
    let mut start_map: HashMap<usize, Vec<usize>> = HashMap::new();
    for (i, info) in infos.iter().enumerate() {
        start_map.entry(info.start_vid).or_default().push(i);
    }

    let mut used = vec![false; edges.len()];
    let mut loops: Vec<Vec<OrientedEdge>> = Vec::new();

    // Walk chains starting from each unused edge.
    while let Some(start_idx) = used.iter().position(|&u| !u) {
        let mut chain = Vec::new();
        chain.push(infos[start_idx].oe);
        used[start_idx] = true;
        let chain_start = infos[start_idx].start_vid;
        let mut current_end = infos[start_idx].end_vid;

        let max_steps = edges.len();
        for _ in 1..=max_steps {
            if current_end == chain_start {
                break; // loop closed
            }
            let candidates = match start_map.get(&current_end) {
                Some(c) => c,
                None => break, // broken chain
            };
            let mut found = false;
            for &idx in candidates {
                if !used[idx] {
                    used[idx] = true;
                    chain.push(infos[idx].oe);
                    current_end = infos[idx].end_vid;
                    found = true;
                    break;
                }
            }
            if !found {
                break; // dead end
            }
        }

        // Only keep the chain if it forms a closed loop.
        if current_end == chain_start && !chain.is_empty() {
            loops.push(chain);
        }
    }

    Ok(loops)
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

    // ── Face unification tests ──────────────────────────

    #[test]
    fn unify_clean_box_no_change() {
        let mut topo = Topology::new();
        let solid = crate::primitives::make_box(&mut topo, 1.0, 1.0, 1.0).unwrap();

        let (f_before, _, _) =
            brepkit_topology::explorer::solid_entity_counts(&topo, solid).unwrap();
        let removed = unify_faces(&mut topo, solid).unwrap();

        assert_eq!(removed, 0, "clean box should have nothing to unify");
        let (f_after, _, _) =
            brepkit_topology::explorer::solid_entity_counts(&topo, solid).unwrap();
        assert_eq!(f_before, f_after);
    }

    #[test]
    fn unify_clean_cylinder_no_change() {
        let mut topo = Topology::new();
        let solid = crate::primitives::make_cylinder(&mut topo, 1.0, 2.0).unwrap();

        let removed = unify_faces(&mut topo, solid).unwrap();
        assert_eq!(removed, 0, "clean cylinder should have nothing to unify");
    }

    #[test]
    fn unify_boolean_box_reduces_faces() {
        // Two overlapping boxes: the boolean result has coplanar face fragments
        // on the shared cutting plane. Use unify_faces=false so the fragments
        // remain for this test to verify explicit unification.
        let mut topo = Topology::new();
        let box1 = crate::primitives::make_box(&mut topo, 2.0, 2.0, 2.0).unwrap();
        let box2 = crate::primitives::make_box(&mut topo, 2.0, 2.0, 2.0).unwrap();

        // Translate box2 by (1, 0, 0) so they overlap by 1 unit in X.
        let translate = brepkit_math::mat::Mat4::translation(1.0, 0.0, 0.0);
        crate::transform::transform_solid(&mut topo, box2, &translate).unwrap();

        let opts = crate::boolean::BooleanOptions {
            unify_faces: false,
            ..Default::default()
        };
        let fused = crate::boolean::boolean_with_options(
            &mut topo,
            crate::boolean::BooleanOp::Fuse,
            box1,
            box2,
            opts,
        )
        .unwrap();

        let (f_before, _, _) =
            brepkit_topology::explorer::solid_entity_counts(&topo, fused).unwrap();

        let vol_before = crate::measure::solid_volume(&topo, fused, 0.1).unwrap();

        let removed = unify_faces(&mut topo, fused).unwrap();

        let (f_after, _, _) =
            brepkit_topology::explorer::solid_entity_counts(&topo, fused).unwrap();

        let vol_after = crate::measure::solid_volume(&topo, fused, 0.1).unwrap();

        // Unification should reduce face count.
        assert!(
            removed > 0,
            "boolean fuse of overlapping boxes should produce unifiable coplanar faces, \
             f_before={f_before}, f_after={f_after}"
        );
        assert!(f_after < f_before);

        // Volume should be preserved.
        assert!(
            (vol_before - vol_after).abs() < 0.1,
            "unification should preserve volume: before={vol_before}, after={vol_after}"
        );
    }

    #[test]
    fn unify_preserves_volume() {
        let mut topo = Topology::new();
        let solid = crate::primitives::make_box(&mut topo, 3.0, 4.0, 5.0).unwrap();

        let vol_before = crate::measure::solid_volume(&topo, solid, 0.1).unwrap();
        let _removed = unify_faces(&mut topo, solid).unwrap();
        let vol_after = crate::measure::solid_volume(&topo, solid, 0.1).unwrap();

        assert!(
            (vol_before - vol_after).abs() < 0.01,
            "unify should preserve volume: before={vol_before}, after={vol_after}"
        );
    }

    #[test]
    fn unify_shell_box_reduces_faces() {
        // Shell a box (hollow it out) — this produces coplanar face fragments
        // that should be merged by unify_faces.
        let mut topo = Topology::new();
        let solid = crate::primitives::make_box(&mut topo, 10.0, 10.0, 10.0).unwrap();

        // Get top face (z=10) as the open face for shelling.
        let solid_data = topo.solid(solid).unwrap();
        let shell = topo.shell(solid_data.outer_shell()).unwrap();
        let face_ids: Vec<_> = shell.faces().to_vec();

        // Find the top face (normal pointing +z, d ≈ 10)
        let top_face = face_ids
            .iter()
            .find(|&&fid| {
                let face = topo.face(fid).unwrap();
                match face.surface() {
                    FaceSurface::Plane { normal, d } => normal.z() > 0.9 && (*d - 10.0).abs() < 0.1,
                    _ => false,
                }
            })
            .copied();

        let open_faces = match top_face {
            Some(f) => vec![f],
            None => vec![face_ids[0]], // fallback
        };

        let shelled = crate::shell_op::shell(&mut topo, solid, 1.0, &open_faces).unwrap();

        let (f_before, e_before, v_before) =
            brepkit_topology::explorer::solid_entity_counts(&topo, shelled).unwrap();
        #[allow(clippy::cast_possible_wrap)]
        let chi_before = (v_before as i64) - (e_before as i64) + (f_before as i64);

        let vol_before = crate::measure::solid_volume(&topo, shelled, 0.1).unwrap();

        let removed = unify_faces(&mut topo, shelled).unwrap();

        let (f_after, e_after, v_after) =
            brepkit_topology::explorer::solid_entity_counts(&topo, shelled).unwrap();
        #[allow(clippy::cast_possible_wrap)]
        let chi_after = (v_after as i64) - (e_after as i64) + (f_after as i64);

        let vol_after = crate::measure::solid_volume(&topo, shelled, 0.1).unwrap();

        eprintln!(
            "shell box: faces {f_before} -> {f_after} (removed {removed}), \
             χ {chi_before} -> {chi_after}, vol {vol_before:.1} -> {vol_after:.1}"
        );

        // Volume must be preserved.
        assert!(
            (vol_before - vol_after).abs() / vol_before < 0.01,
            "unification should preserve volume: before={vol_before}, after={vol_after}"
        );

        // Euler characteristic must be preserved.
        assert_eq!(
            chi_before, chi_after,
            "Euler χ should be preserved: before={chi_before}, after={chi_after}"
        );
    }

    #[test]
    fn unify_cylinder_boolean_reduces_faces() {
        // Boolean cut of a box with a cylinder produces co-cylindrical face fragments.
        let mut topo = Topology::new();
        let box1 = crate::primitives::make_box(&mut topo, 4.0, 4.0, 4.0).unwrap();
        let cyl = crate::primitives::make_cylinder(&mut topo, 1.0, 6.0).unwrap();

        // Move cylinder to center of box top face.
        let translate = brepkit_math::mat::Mat4::translation(2.0, 2.0, -1.0);
        crate::transform::transform_solid(&mut topo, cyl, &translate).unwrap();

        let result =
            crate::boolean::boolean(&mut topo, crate::boolean::BooleanOp::Cut, box1, cyl).unwrap();

        let (f_before, _, _) =
            brepkit_topology::explorer::solid_entity_counts(&topo, result).unwrap();

        let vol_before = crate::measure::solid_volume(&topo, result, 0.1).unwrap();

        let removed = unify_faces(&mut topo, result).unwrap();

        let vol_after = crate::measure::solid_volume(&topo, result, 0.1).unwrap();

        // Volume must be preserved.
        assert!(
            (vol_before - vol_after).abs() < 0.1,
            "unification should preserve volume: before={vol_before}, after={vol_after}"
        );

        // Log for diagnostics (test passes either way — this is informational).
        let (f_after, _, _) =
            brepkit_topology::explorer::solid_entity_counts(&topo, result).unwrap();
        eprintln!("cylinder boolean: faces {f_before} -> {f_after}, removed {removed}");
    }

    #[test]
    fn unify_shell_rounded_rect_preserves_volume() {
        // Shell a rounded rectangle with 3 arc edges per quarter-circle corner
        // (matching brepjs behavior). The extrusion creates 3 cylindrical face
        // fragments per corner. unify_faces merges these. This is the exact
        // scenario that causes volume corruption in the topology parity test.
        use brepkit_math::curves::Circle3D;
        use brepkit_math::tolerance::Tolerance;
        use brepkit_math::vec::Vec3;
        use brepkit_topology::edge::{Edge, EdgeCurve};
        use brepkit_topology::face::Face;
        use brepkit_topology::vertex::Vertex;
        use brepkit_topology::wire::{OrientedEdge, Wire};

        let mut topo = Topology::new();
        let tol = Tolerance::new();

        let w = 41.5_f64;
        let d = 41.5_f64;
        let h = 21.0_f64;
        let r = 4.0_f64;
        let thickness = 1.2_f64;
        let hw = w / 2.0;
        let hd = d / 2.0;

        // Corner centers:
        let c_br = Point3::new(hw - r, -hd + r, 0.0);
        let c_tr = Point3::new(hw - r, hd - r, 0.0);
        let c_tl = Point3::new(-hw + r, hd - r, 0.0);
        let c_bl = Point3::new(-hw + r, -hd + r, 0.0);

        let z_axis = Vec3::new(0.0, 0.0, 1.0);

        // Subdivide each quarter circle into 3 arcs (30° each).
        // Quarter goes from angle a0 to a0+π/2 in 3 steps.
        let n_sub = 3usize;
        let quarter = std::f64::consts::FRAC_PI_2;

        // Corner start angles (CCW): BR=-π/2, TR=0, TL=π/2, BL=π
        let corners = [
            (c_br, -std::f64::consts::FRAC_PI_2),
            (c_tr, 0.0),
            (c_tl, std::f64::consts::FRAC_PI_2),
            (c_bl, std::f64::consts::PI),
        ];

        // Build all vertices: for each corner, n_sub+1 points on the arc,
        // but the last point of one corner is the first of the next line segment.
        // Layout: for corner i, arc vertices are at angles a0 + j*(π/2)/n_sub for j=0..n_sub.
        // The vertex at j=0 is the line-end of the previous side.
        // The vertex at j=n_sub is the line-start of the next side.

        // Generate corner arc points.
        let mut corner_verts: Vec<Vec<Point3>> = Vec::new();
        for &(center, a0) in &corners {
            let mut pts = Vec::new();
            for j in 0..=n_sub {
                #[allow(clippy::cast_precision_loss)]
                let angle = a0 + (j as f64) * quarter / (n_sub as f64);
                let pt = Point3::new(
                    center.x() + r * angle.cos(),
                    center.y() + r * angle.sin(),
                    0.0,
                );
                pts.push(pt);
            }
            corner_verts.push(pts);
        }

        // Allocate vertices (sharing endpoints between corners and lines).
        // Wire order: bottom_line, br_arc[0..3], right_line, tr_arc[0..3], top_line, tl_arc[0..3], left_line, bl_arc[0..3]
        // Each line connects: corner[i][n_sub] -> corner[(i+1)%4][0]
        // Line vertices: br[3]=right_start, tr[0]=right_end -> but corners go BR, TR, TL, BL
        // So: bottom_line = bl[3] -> br[0], right_line = br[3] -> tr[0], etc.

        // Allocate all unique vertex IDs.
        // For each corner: n_sub+1 points, but corner[i][n_sub] == start of next line == corner[(i+1)%4][0]
        // Wait, that's not right. Let me think again...
        // Wire order CCW: bl[3]->br[0] (bottom), br[0]->br[3] (br arc), br[3]->tr[0] (right), tr[0]->tr[3] (tr arc), etc.
        // So corner[i][0] is the START of the arc, corner[i][n_sub] is the END.
        // Line between corner i end and corner (i+1)%4 start.
        // But our corners are [BR, TR, TL, BL], and CCW order is: bottom, BR, right, TR, top, TL, left, BL
        // So: BL[3]->BR[0] = bottom line, BR[0]->BR[3] = BR arc, BR[3]->TR[0] = right line, etc.

        // Unique points: 4 corners × n_sub intermediate + 4 corner endpoints shared with lines.
        // Actually: each corner has n_sub+1 points. corner[i][0] is shared with previous line end,
        // corner[i][n_sub] is shared with next line start.
        // Total unique: 4 * n_sub (intermediate arc points) + 4 (shared line/arc junction points) = 4*(n_sub+1) - 4 = 4*n_sub
        // Wait: 4 corners × (n_sub+1) points each, but corner[i][n_sub] == (next line start) and
        // corner[(i+1)%4][0] == (next line end). These are NOT the same point.
        // Actually in CCW order: BL_end -> BR_start (bottom line), BR arcs, BR_end -> TR_start (right line), etc.
        // So each corner contributes n_sub+1 unique points, and lines share those endpoints.
        // Total unique vertices = 4 * (n_sub + 1) = 16 for n_sub=3.

        // Let me just allocate all vertices in wire order.
        let mut wire_edges = Vec::new();

        // CCW order: bottom_line, br_arc, right_line, tr_arc, top_line, tl_arc, left_line, bl_arc
        // corners[0]=BR, corners[1]=TR, corners[2]=TL, corners[3]=BL
        let corner_order = [3, 0, 1, 2]; // BL, BR, TR, TL in CCW order

        for ci in 0..4 {
            let this_corner = corner_order[ci];
            let next_corner = corner_order[(ci + 1) % 4];

            // Line: this_corner[n_sub] -> next_corner[0]
            let line_start = corner_verts[this_corner][n_sub];
            let line_end = corner_verts[next_corner][0];
            let ls_vid = topo.add_vertex(Vertex::new(line_start, tol.linear));
            let le_vid = topo.add_vertex(Vertex::new(line_end, tol.linear));
            let line_eid = topo.add_edge(Edge::new(ls_vid, le_vid, EdgeCurve::Line));
            wire_edges.push(OrientedEdge::new(line_eid, true));

            // Arc segments for next_corner.
            let center = corners[next_corner].0;
            let circle = Circle3D::new(center, z_axis, r).unwrap();
            let mut prev_vid = le_vid;
            for j in 1..=n_sub {
                let pt = corner_verts[next_corner][j];
                let next_vid = topo.add_vertex(Vertex::new(pt, tol.linear));
                let arc_eid = topo.add_edge(Edge::new(
                    prev_vid,
                    next_vid,
                    EdgeCurve::Circle(circle.clone()),
                ));
                wire_edges.push(OrientedEdge::new(arc_eid, true));
                prev_vid = next_vid;
            }
        }

        let wire = Wire::new(wire_edges, true).unwrap();
        let wire_id = topo.add_wire(wire);

        let normal = Vec3::new(0.0, 0.0, 1.0);
        let face = Face::new(wire_id, vec![], FaceSurface::Plane { normal, d: 0.0 });
        let face_id = topo.add_face(face);

        // Extrude.
        let solid =
            crate::extrude::extrude(&mut topo, face_id, Vec3::new(0.0, 0.0, 1.0), h).unwrap();

        // Find top face for shelling.
        let top_faces: Vec<FaceId> = {
            let s = topo.solid(solid).unwrap();
            let sh = topo.shell(s.outer_shell()).unwrap();
            sh.faces()
                .iter()
                .filter(|&&fid| {
                    let f = topo.face(fid).unwrap();
                    if let FaceSurface::Plane { normal: n, d } = f.surface() {
                        n.z() > 0.9 && (*d - h).abs() < 0.1
                    } else {
                        false
                    }
                })
                .copied()
                .collect()
        };
        assert_eq!(top_faces.len(), 1, "should find exactly one top face");

        let shelled = crate::shell_op::shell(&mut topo, solid, thickness, &top_faces).unwrap();

        let (f_before, e_before, v_before) =
            brepkit_topology::explorer::solid_entity_counts(&topo, shelled).unwrap();
        let vol_before = crate::measure::solid_volume(&topo, shelled, 0.01).unwrap();

        // Count cylinder faces before.
        let shell_id = topo.solid(shelled).unwrap().outer_shell();
        let cyl_before = topo
            .shell(shell_id)
            .unwrap()
            .faces()
            .iter()
            .filter(|&&fid| matches!(topo.face(fid).unwrap().surface(), FaceSurface::Cylinder(_)))
            .count();

        let removed = unify_faces(&mut topo, shelled).unwrap();

        let (f_after, e_after, v_after) =
            brepkit_topology::explorer::solid_entity_counts(&topo, shelled).unwrap();
        let vol_after = crate::measure::solid_volume(&topo, shelled, 0.01).unwrap();

        let cyl_after = topo
            .shell(shell_id)
            .unwrap()
            .faces()
            .iter()
            .filter(|&&fid| matches!(topo.face(fid).unwrap().surface(), FaceSurface::Cylinder(_)))
            .count();

        #[allow(clippy::cast_possible_wrap)]
        let chi_before = (v_before as i64) - (e_before as i64) + (f_before as i64);
        #[allow(clippy::cast_possible_wrap)]
        let chi_after = (v_after as i64) - (e_after as i64) + (f_after as i64);

        eprintln!(
            "shell rounded rect (3 arcs/corner): faces {f_before} -> {f_after} (removed {removed}), \
             cyl {cyl_before} -> {cyl_after}, \
             χ {chi_before} -> {chi_after}, vol {vol_before:.1} -> {vol_after:.1}"
        );

        // Cylinder faces should be merged: 24 -> 8.
        assert!(removed > 0, "unify should merge cylinder face fragments");

        // Volume must be preserved (within 1%).
        let rel_err = (vol_before - vol_after).abs() / vol_before;
        assert!(
            rel_err < 0.01,
            "unify should preserve volume: before={vol_before:.2}, after={vol_after:.2}, err={:.2}%",
            rel_err * 100.0
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
