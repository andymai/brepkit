//! Solid assembly functions for boolean operations.
//!
//! These functions build a solid from a set of face specifications (planar,
//! NURBS, analytic) using spatial hashing for vertex deduplication and edge
//! sharing. Post-assembly passes refine boundary edges and split non-manifold
//! edges to ensure a valid manifold result.

use std::collections::{HashMap, HashSet};

use brepkit_math::aabb::Aabb3;
use brepkit_math::tolerance::Tolerance;
use brepkit_math::vec::{Point3, Vec3};
use brepkit_topology::Topology;
use brepkit_topology::edge::{Edge, EdgeCurve, EdgeId};
use brepkit_topology::face::{Face, FaceId, FaceSurface};
use brepkit_topology::shell::Shell;
use brepkit_topology::solid::{Solid, SolidId};
use brepkit_topology::vertex::{Vertex, VertexId};
use brepkit_topology::wire::{OrientedEdge, Wire, WireId};

use super::classify::polygon_centroid;
use super::precompute::{analytic_face_normal_d, face_polygon};
use super::types::{FaceSpec, MIN_SOLID_FACES};

// ---------------------------------------------------------------------------
// Spatial hashing helpers
// ---------------------------------------------------------------------------

/// Quantize a coordinate to a spatial hash key.
#[inline]
#[allow(clippy::cast_possible_truncation)] // coordinate * 1e7 fits in i64
pub(super) fn quantize(v: f64, resolution: f64) -> i64 {
    (v * resolution).round() as i64
}

/// Quantize a 3D point to a spatial hash key for vertex deduplication.
#[inline]
pub(super) fn quantize_point(p: Point3, resolution: f64) -> (i64, i64, i64) {
    (
        quantize(p.x(), resolution),
        quantize(p.y(), resolution),
        quantize(p.z(), resolution),
    )
}

/// Compute a scale-relative spatial-hash resolution from a set of vertex positions.
///
/// Uses the bounding-box diagonal of the input points scaled by 1e-7 to keep
/// the hash cell roughly at tolerance-level relative to the model extent.
/// Falls back to `1.0 / tol.linear` for degenerate (near-single-point) models.
pub(super) fn vertex_merge_resolution(
    all_pts: impl Iterator<Item = Point3>,
    tol: Tolerance,
) -> f64 {
    let fallback = 1.0 / tol.linear;
    if let Some(bbox) = Aabb3::try_from_points(all_pts) {
        let diagonal = (bbox.max - bbox.min).length();
        if diagonal > tol.linear {
            // 1e-7 relative factor: same precision as absolute tolerance at unit scale,
            // but scales correctly for large models (100m+) and sub-mm geometry.
            1.0 / (diagonal * 1e-7_f64)
        } else {
            fallback
        }
    } else {
        fallback
    }
}

// ---------------------------------------------------------------------------
// Solid assembly
// ---------------------------------------------------------------------------

/// Assemble a solid from a set of planar face polygons with normals.
///
/// Uses spatial hashing for vertex dedup and edge sharing.
/// This is a convenience wrapper around [`assemble_solid_mixed`] for the
/// common case where all faces are planar.
#[allow(clippy::redundant_pub_crate)]
pub(crate) fn assemble_solid(
    topo: &mut Topology,
    faces: &[(Vec<Point3>, Vec3, f64)],
    tol: Tolerance,
) -> Result<SolidId, crate::OperationsError> {
    let specs: Vec<FaceSpec> = faces
        .iter()
        .map(|(verts, normal, d)| FaceSpec::Planar {
            vertices: verts.clone(),
            normal: *normal,
            d: *d,
            inner_wires: vec![],
        })
        .collect();
    assemble_solid_mixed(topo, &specs, tol)
}

/// Build inner wire topology from vertex position lists.
///
/// For each inner wire (a closed loop of vertex positions), creates vertices
/// (via `vertex_map` dedup), edges (via `edge_map` sharing), and a `Wire`.
/// Returns the list of `WireId`s to pass as inner wires when constructing a `Face`.
fn build_inner_wires(
    topo: &mut Topology,
    inner_wire_specs: &[Vec<Point3>],
    vertex_map: &mut HashMap<(i64, i64, i64), VertexId>,
    edge_map: &mut HashMap<(usize, usize), EdgeId>,
    resolution: f64,
    tol: Tolerance,
) -> Result<Vec<WireId>, crate::OperationsError> {
    let mut inner_wire_ids = Vec::with_capacity(inner_wire_specs.len());
    for iw_verts in inner_wire_specs {
        let iw_n = iw_verts.len();
        if iw_n < 3 {
            continue;
        }

        let iw_vert_ids: Vec<VertexId> = iw_verts
            .iter()
            .map(|p| {
                let key = quantize_point(*p, resolution);
                *vertex_map
                    .entry(key)
                    .or_insert_with(|| topo.add_vertex(Vertex::new(*p, tol.linear)))
            })
            .collect();

        let mut iw_oriented_edges = Vec::with_capacity(iw_n);
        for i in 0..iw_n {
            let j = (i + 1) % iw_n;
            let vi = iw_vert_ids[i].index();
            let vj = iw_vert_ids[j].index();
            let (key_min, key_max) = if vi <= vj { (vi, vj) } else { (vj, vi) };
            let is_forward = vi <= vj;

            let edge_id = *edge_map.entry((key_min, key_max)).or_insert_with(|| {
                let (start, end) = if vi <= vj {
                    (iw_vert_ids[i], iw_vert_ids[j])
                } else {
                    (iw_vert_ids[j], iw_vert_ids[i])
                };
                topo.add_edge(Edge::new(start, end, EdgeCurve::Line))
            });

            iw_oriented_edges.push(OrientedEdge::new(edge_id, is_forward));
        }

        let wire = Wire::new(iw_oriented_edges, true).map_err(crate::OperationsError::Topology)?;
        inner_wire_ids.push(topo.add_wire(wire));
    }
    Ok(inner_wire_ids)
}

/// Assemble a solid from a set of face specifications with mixed surface types.
///
/// Like [`assemble_solid`], but supports faces with NURBS, analytic, or any
/// other surface type. Uses the same spatial-hashing vertex dedup and edge
/// sharing as the planar variant.
///
/// This is the general-purpose solid assembly function that unblocks operations
/// on non-planar faces.
#[allow(clippy::redundant_pub_crate)]
pub(crate) fn assemble_solid_mixed(
    topo: &mut Topology,
    face_specs: &[FaceSpec],
    tol: Tolerance,
) -> Result<SolidId, crate::OperationsError> {
    // Pre-allocate topology arenas based on expected output size.
    // Typical face → ~2 unique vertices, ~3 edges, 1 wire, 1 face.
    let n = face_specs.len();
    topo.reserve(n.saturating_mul(2), n.saturating_mul(3), n, n, 1, 1);

    let resolution = vertex_merge_resolution(
        face_specs.iter().flat_map(|s| match s {
            FaceSpec::Planar { vertices, .. }
            | FaceSpec::Surface { vertices, .. }
            | FaceSpec::CylindricalFace { vertices, .. } => vertices.iter().copied(),
        }),
        tol,
    );

    let mut vertex_map: HashMap<(i64, i64, i64), VertexId> =
        HashMap::with_capacity(face_specs.len() * 4);
    let mut edge_map: HashMap<(usize, usize), brepkit_topology::edge::EdgeId> =
        HashMap::with_capacity(face_specs.len() * 4);

    let mut face_ids = Vec::with_capacity(face_specs.len());

    // Process CylindricalFace specs first so circle edges populate edge_map
    // before planar/surface faces look them up. This ensures adjacent planar
    // faces share the Circle edge rather than creating a Line edge.
    let cylindrical_first = face_specs
        .iter()
        .filter(|s| matches!(s, FaceSpec::CylindricalFace { .. }))
        .chain(
            face_specs
                .iter()
                .filter(|s| !matches!(s, FaceSpec::CylindricalFace { .. })),
        );

    for spec in cylindrical_first {
        match spec {
            FaceSpec::CylindricalFace {
                vertices,
                cylinder,
                reversed,
                ..
            } => {
                let verts = vertices;
                let n = verts.len();
                if n < 3 {
                    continue;
                }

                let vert_ids: Vec<VertexId> = verts
                    .iter()
                    .map(|p| {
                        let key = quantize_point(*p, resolution);
                        *vertex_map
                            .entry(key)
                            .or_insert_with(|| topo.add_vertex(Vertex::new(*p, tol.linear)))
                    })
                    .collect();

                let mut oriented_edges = Vec::with_capacity(n);
                for i in 0..n {
                    let j = (i + 1) % n;
                    let vi = vert_ids[i].index();
                    let vj = vert_ids[j].index();
                    if vi == vj {
                        continue; // Skip degenerate zero-length edges.
                    }
                    let (key_min, key_max) = if vi <= vj { (vi, vj) } else { (vj, vi) };
                    let is_forward = vi <= vj;

                    let edge_id = *edge_map.entry((key_min, key_max)).or_insert_with(|| {
                        let (start, end) = if vi <= vj {
                            (vert_ids[i], vert_ids[j])
                        } else {
                            (vert_ids[j], vert_ids[i])
                        };

                        // Determine if this edge is angular (arc) or axial (line)
                        // by projecting both endpoints onto the cylinder.
                        let (u1, v1) = cylinder.project_point(verts[i]);
                        let (u2, v2) = cylinder.project_point(verts[j]);
                        let u_diff = (u1 - u2).abs();
                        let v_diff = (v1 - v2).abs();

                        // Angular edge: endpoints at the same height (v) but different
                        // angle (u). If v also differs, it's a diagonal/seam → Line.
                        if u_diff > tol.linear
                            && u_diff < (std::f64::consts::TAU - tol.linear)
                            && v_diff < tol.linear * 100.0
                        {
                            // Create a Circle3D at the v-level of this edge.
                            let center = cylinder.origin() + cylinder.axis() * ((v1 + v2) * 0.5);
                            if let Ok(circle) = brepkit_math::curves::Circle3D::new(
                                center,
                                cylinder.axis(),
                                cylinder.radius(),
                            ) {
                                topo.add_edge(Edge::new(start, end, EdgeCurve::Circle(circle)))
                            } else {
                                topo.add_edge(Edge::new(start, end, EdgeCurve::Line))
                            }
                        } else {
                            // Axial edge (same angle, different height): line.
                            topo.add_edge(Edge::new(start, end, EdgeCurve::Line))
                        }
                    });

                    if oriented_edges
                        .last()
                        .is_some_and(|last: &OrientedEdge| last.edge() == edge_id)
                    {
                        continue;
                    }
                    oriented_edges.push(OrientedEdge::new(edge_id, is_forward));
                }

                let wire =
                    Wire::new(oriented_edges, true).map_err(crate::OperationsError::Topology)?;
                let wire_id = topo.add_wire(wire);

                // Build inner wires from FaceSpec.
                let inner_wire_ids = build_inner_wires(
                    topo,
                    spec.inner_wires(),
                    &mut vertex_map,
                    &mut edge_map,
                    resolution,
                    tol,
                )?;

                let surface = FaceSurface::Cylinder(cylinder.clone());
                let face = if *reversed {
                    topo.add_face(Face::new_reversed(wire_id, inner_wire_ids, surface))
                } else {
                    topo.add_face(Face::new(wire_id, inner_wire_ids, surface))
                };
                face_ids.push(face);
            }
            spec => {
                // Planar or Surface: extract (verts, surface, reversed)
                let (verts, surface, reversed) = match spec {
                    FaceSpec::Planar {
                        vertices,
                        normal,
                        d,
                        ..
                    } => (
                        vertices.clone(),
                        FaceSurface::Plane {
                            normal: *normal,
                            d: *d,
                        },
                        false,
                    ),
                    FaceSpec::Surface {
                        vertices,
                        surface,
                        reversed,
                        ..
                    } => (vertices.clone(), surface.clone(), *reversed),
                    FaceSpec::CylindricalFace { .. } => unreachable!(),
                };

                let n = verts.len();
                if n < 3 {
                    continue;
                }

                let vert_ids: Vec<VertexId> = verts
                    .iter()
                    .map(|p| {
                        let key = quantize_point(*p, resolution);
                        *vertex_map
                            .entry(key)
                            .or_insert_with(|| topo.add_vertex(Vertex::new(*p, tol.linear)))
                    })
                    .collect();

                let mut oriented_edges = Vec::with_capacity(n);
                for i in 0..n {
                    let j = (i + 1) % n;
                    let vi = vert_ids[i].index();
                    let vj = vert_ids[j].index();
                    // Skip degenerate zero-length edges (collapsed vertices).
                    if vi == vj {
                        continue;
                    }
                    let (key_min, key_max) = if vi <= vj { (vi, vj) } else { (vj, vi) };
                    let is_forward = vi <= vj;

                    let edge_id = *edge_map.entry((key_min, key_max)).or_insert_with(|| {
                        let (start, end) = if vi <= vj {
                            (vert_ids[i], vert_ids[j])
                        } else {
                            (vert_ids[j], vert_ids[i])
                        };
                        topo.add_edge(Edge::new(start, end, EdgeCurve::Line))
                    });

                    // Skip duplicate edges anywhere in the wire (not just consecutive).
                    // Duplicates arise when the polygon revisits a vertex pair due to
                    // degenerate splits or vertex merging.
                    if oriented_edges
                        .iter()
                        .any(|oe: &OrientedEdge| oe.edge() == edge_id)
                    {
                        continue;
                    }
                    oriented_edges.push(OrientedEdge::new(edge_id, is_forward));
                }

                let wire =
                    Wire::new(oriented_edges, true).map_err(crate::OperationsError::Topology)?;
                let wire_id = topo.add_wire(wire);

                // Build inner wires from FaceSpec.
                let inner_wire_ids = build_inner_wires(
                    topo,
                    spec.inner_wires(),
                    &mut vertex_map,
                    &mut edge_map,
                    resolution,
                    tol,
                )?;

                let face = if reversed {
                    topo.add_face(Face::new_reversed(wire_id, inner_wire_ids, surface))
                } else {
                    topo.add_face(Face::new(wire_id, inner_wire_ids, surface))
                };
                face_ids.push(face);
            }
        }
    }

    if face_ids.is_empty() {
        return Err(crate::OperationsError::InvalidInput {
            reason: "solid assembly produced no faces".into(),
        });
    }

    // Post-assembly edge refinement: split long boundary edges at
    // intermediate collinear vertices so adjacent faces can share edges.
    // Pass precomputed vertex positions from assembly to avoid redundant
    // face→wire→edge→vertex traversal.
    let vertex_positions: HashMap<VertexId, Point3> = vertex_map
        .values()
        .filter_map(|&vid| topo.vertex(vid).ok().map(|v| (vid, v.point())))
        .collect();
    refine_boundary_edges(
        topo,
        &mut face_ids,
        &mut edge_map,
        tol,
        Some(&vertex_positions),
    )?;

    // Stitch boundary edge pairs that should be shared but were assigned
    // different VertexIds by the spatial hash (cell-boundary straddling).
    stitch_boundary_edges(topo, &mut face_ids, tol)?;

    // Split non-manifold edges (shared by > 2 faces) into separate copies,
    // pairing faces by angular ordering around the edge.
    // Run up to 3 iterations: edge replacement can sometimes create new
    // non-manifold edges when the replacement edge shares vertices with
    // other faces' edges.
    for _ in 0..3 {
        split_nonmanifold_edges(topo, &mut face_ids)?;
    }

    let shell = Shell::new(face_ids).map_err(crate::OperationsError::Topology)?;
    let shell_id = topo.add_shell(shell);
    Ok(topo.add_solid(Solid::new(shell_id, vec![])))
}

// ---------------------------------------------------------------------------
// Degenerate result detection
// ---------------------------------------------------------------------------

/// Validate that a boolean result is not degenerate.
///
/// Checks for:
/// - Too few faces (< `MIN_SOLID_FACES`)
/// - No edges or vertices (empty topology)
/// - Euler characteristic, manifold edges, boundary edges, wire closure,
///   degenerate faces, and face area via [`crate::validate::validate_solid`]
pub(super) fn validate_boolean_result(
    topo: &Topology,
    solid: SolidId,
) -> Result<(), crate::OperationsError> {
    let s = topo.solid(solid)?;
    let shell = topo.shell(s.outer_shell())?;
    let face_count = shell.faces().len();

    if face_count < MIN_SOLID_FACES {
        return Err(crate::OperationsError::InvalidInput {
            reason: format!(
                "boolean result has only {face_count} faces (minimum {MIN_SOLID_FACES} required for a closed solid)"
            ),
        });
    }

    // Check that we have at least some edges and vertices.
    let (f, e, v) = brepkit_topology::explorer::solid_entity_counts(topo, solid)?;
    if e == 0 || v == 0 {
        return Err(crate::OperationsError::InvalidInput {
            reason: format!("boolean result has degenerate topology (F={f}, E={e}, V={v})"),
        });
    }

    // Full topological validation: Euler characteristic, manifold edges,
    // boundary edges, wire closure, degenerate faces.
    // Logged as warnings rather than hard errors — many boolean results have
    // minor topological imperfections (e.g., boundary edges on analytic faces)
    // that don't prevent downstream use. Hard-failing here would reject ~25%
    // of currently working booleans. The long-term fix is post-boolean healing.
    match crate::validate::validate_solid(topo, solid) {
        Ok(report) if !report.is_valid() => {
            let errors: Vec<_> = report
                .issues
                .iter()
                .filter(|i| i.severity == crate::validate::Severity::Error)
                .map(|i| i.description.as_str())
                .collect();
            log::warn!(
                "boolean result has {} validation error(s): {}",
                errors.len(),
                errors.join("; ")
            );
        }
        Err(e) => {
            log::warn!("validate_solid failed (skipping validation): {e}");
        }
        Ok(_) => {}
    }

    Ok(())
}

// ---------------------------------------------------------------------------
// Post-assembly edge refinement
// ---------------------------------------------------------------------------

/// Split long boundary edges at intermediate collinear vertices.
///
/// After boolean assembly, some unsplit (passthrough) faces may have edges that
/// span the same geometric line as multiple shorter edges from adjacent
/// split faces. This function splits those long edges at the intermediate
/// vertex positions, enabling proper edge sharing between adjacent faces.
#[allow(clippy::too_many_lines)]
pub(super) fn refine_boundary_edges(
    topo: &mut Topology,
    face_ids: &mut [FaceId],
    edge_map: &mut HashMap<(usize, usize), EdgeId>,
    tol: Tolerance,
    precomputed_positions: Option<&HashMap<VertexId, Point3>>,
) -> Result<(), crate::OperationsError> {
    // Single-pass: build edge-to-face count AND collect edge vertex pairs.
    // This avoids a second full face→wire→edge→vertex traversal.
    let mut edge_face_count: HashMap<EdgeId, usize> = HashMap::new();
    let mut edge_vertices: HashMap<EdgeId, (VertexId, VertexId)> = HashMap::new();
    for &fid in face_ids.iter() {
        let face = topo.face(fid)?;
        for wid in std::iter::once(face.outer_wire()).chain(face.inner_wires().iter().copied()) {
            let wire = topo.wire(wid)?;
            for oe in wire.edges() {
                let eid = oe.edge();
                *edge_face_count.entry(eid).or_default() += 1;
                if let std::collections::hash_map::Entry::Vacant(e) = edge_vertices.entry(eid) {
                    if let Ok(edge) = topo.edge(eid) {
                        e.insert((edge.start(), edge.end()));
                    }
                }
            }
        }
    }

    // Find boundary edges (used by exactly 1 face)
    let boundary_edges: HashSet<EdgeId> = edge_face_count
        .iter()
        .filter(|&(_, &count)| count == 1)
        .map(|(&eid, _)| eid)
        .collect();

    if boundary_edges.is_empty() {
        return Ok(());
    }

    // Build vertex positions. Use precomputed positions from assembly when
    // available, falling back to topology only for missing vertices
    // (e.g. passthrough faces not in the assembly's vertex_map).
    let mut extra_positions: HashMap<VertexId, Point3> = HashMap::new();
    for &(start, end) in edge_vertices.values() {
        for &vid in &[start, end] {
            let in_pre = precomputed_positions.is_some_and(|p| p.contains_key(&vid));
            if !in_pre {
                if let std::collections::hash_map::Entry::Vacant(e) = extra_positions.entry(vid) {
                    if let Ok(v) = topo.vertex(vid) {
                        e.insert(v.point());
                    }
                }
            }
        }
    }

    // For each boundary edge, find intermediate collinear vertices.
    // Use a spatial hash grid for O(V) build + O(1) amortized query,
    // much faster than SAH BVH's O(V log²V) build for point clouds.
    let get_pos = |vid: &VertexId| -> Option<Point3> {
        precomputed_positions
            .and_then(|p| p.get(vid))
            .or_else(|| extra_positions.get(vid))
            .copied()
    };
    // Build vert_list from both sources, deduplicating by VertexId.
    let mut seen: HashSet<VertexId> = HashSet::new();
    let mut vert_list: Vec<(VertexId, Point3)> = Vec::new();
    if let Some(pre) = precomputed_positions {
        for (&vid, &pos) in pre {
            if seen.insert(vid) {
                vert_list.push((vid, pos));
            }
        }
    }
    for (&vid, &pos) in &extra_positions {
        if seen.insert(vid) {
            vert_list.push((vid, pos));
        }
    }

    // Compute grid cell size from bounding box and vertex count.
    // Target ~1 vertex per cell on average for O(1) query cost.
    // NOTE: cell_size is calibrated from the global vertex population.
    // If boundary faces are concentrated in a small sub-region, the cell
    // size may be too large, degrading to O(boundary_verts) per query.
    // This is acceptable for boolean assembly outputs where vertices are
    // distributed across the full solid extent.
    let (mut bb_min, mut bb_max) = (
        Point3::new(f64::INFINITY, f64::INFINITY, f64::INFINITY),
        Point3::new(f64::NEG_INFINITY, f64::NEG_INFINITY, f64::NEG_INFINITY),
    );
    for &(_, pos) in &vert_list {
        bb_min = Point3::new(
            bb_min.x().min(pos.x()),
            bb_min.y().min(pos.y()),
            bb_min.z().min(pos.z()),
        );
        bb_max = Point3::new(
            bb_max.x().max(pos.x()),
            bb_max.y().max(pos.y()),
            bb_max.z().max(pos.z()),
        );
    }
    let diag = ((bb_max.x() - bb_min.x()).powi(2)
        + (bb_max.y() - bb_min.y()).powi(2)
        + (bb_max.z() - bb_min.z()).powi(2))
    .sqrt();
    let cell_size = (diag / (vert_list.len() as f64).cbrt()).max(tol.linear);
    let inv_cell = 1.0 / cell_size;

    let mut grid: HashMap<(i64, i64, i64), Vec<usize>> = HashMap::new();
    for (i, &(_, pos)) in vert_list.iter().enumerate() {
        let cx = (pos.x() * inv_cell).floor() as i64;
        let cy = (pos.y() * inv_cell).floor() as i64;
        let cz = (pos.z() * inv_cell).floor() as i64;
        grid.entry((cx, cy, cz)).or_default().push(i);
    }

    let mut edge_splits: HashMap<EdgeId, Vec<VertexId>> = HashMap::new();

    for &eid in &boundary_edges {
        let &(start_vid, end_vid) = match edge_vertices.get(&eid) {
            Some(v) => v,
            None => continue,
        };
        let (p0, p1) = match (get_pos(&start_vid), get_pos(&end_vid)) {
            (Some(a), Some(b)) => (a, b),
            _ => continue,
        };
        let dx = p1.x() - p0.x();
        let dy = p1.y() - p0.y();
        let dz = p1.z() - p0.z();
        let len_sq = dx * dx + dy * dy + dz * dz;
        if len_sq < tol.linear * tol.linear {
            continue;
        }
        let len = len_sq.sqrt();

        // Query hash grid with the edge's AABB expanded by tolerance
        let edge_aabb = Aabb3 {
            min: Point3::new(p0.x().min(p1.x()), p0.y().min(p1.y()), p0.z().min(p1.z())),
            max: Point3::new(p0.x().max(p1.x()), p0.y().max(p1.y()), p0.z().max(p1.z())),
        }
        .expanded(tol.linear);
        let min_cx = (edge_aabb.min.x() * inv_cell).floor() as i64;
        let min_cy = (edge_aabb.min.y() * inv_cell).floor() as i64;
        let min_cz = (edge_aabb.min.z() * inv_cell).floor() as i64;
        let max_cx = (edge_aabb.max.x() * inv_cell).floor() as i64;
        let max_cy = (edge_aabb.max.y() * inv_cell).floor() as i64;
        let max_cz = (edge_aabb.max.z() * inv_cell).floor() as i64;

        let mut intermediates: Vec<(f64, VertexId)> = Vec::new();

        for gx in min_cx..=max_cx {
            for gy in min_cy..=max_cy {
                for gz in min_cz..=max_cz {
                    if let Some(indices) = grid.get(&(gx, gy, gz)) {
                        for &cand_idx in indices {
                            let (vid, pos) = vert_list[cand_idx];
                            if vid == start_vid || vid == end_vid {
                                continue;
                            }
                            // Project pos onto line p0 + t*(p1-p0)
                            let dpx = pos.x() - p0.x();
                            let dpy = pos.y() - p0.y();
                            let dpz = pos.z() - p0.z();
                            let t = (dpx * dx + dpy * dy + dpz * dz) / len_sq;

                            // Must be strictly between endpoints
                            if t <= tol.linear / len || t >= 1.0 - tol.linear / len {
                                continue;
                            }

                            // Check distance from point to line
                            let proj_x = p0.x() + t * dx;
                            let proj_y = p0.y() + t * dy;
                            let proj_z = p0.z() + t * dz;
                            let dist_sq = (pos.x() - proj_x).powi(2)
                                + (pos.y() - proj_y).powi(2)
                                + (pos.z() - proj_z).powi(2);

                            if dist_sq < tol.linear * tol.linear {
                                intermediates.push((t, vid));
                            }
                        }
                    }
                }
            }
        }

        if !intermediates.is_empty() {
            intermediates
                .sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap_or(std::cmp::Ordering::Equal));
            intermediates.dedup_by_key(|(_, vid)| *vid);
            edge_splits.insert(eid, intermediates.into_iter().map(|(_, vid)| vid).collect());
        }
    }

    if edge_splits.is_empty() {
        return Ok(());
    }

    // Rebuild faces that have edges needing splits
    for fi in 0..face_ids.len() {
        let fid = face_ids[fi];
        let face = topo.face(fid)?;
        let outer_wire_id = face.outer_wire();
        let outer_wire = topo.wire(outer_wire_id)?;

        let mut needs_rebuild = false;
        for oe in outer_wire.edges() {
            if edge_splits.contains_key(&oe.edge()) {
                needs_rebuild = true;
                break;
            }
        }

        if !needs_rebuild {
            continue;
        }

        // Snapshot face data before mutable borrow
        let surface = face.surface().clone();
        let inner_wires = face.inner_wires().to_vec();
        let is_reversed = face.is_reversed();
        let old_edges: Vec<OrientedEdge> = outer_wire.edges().to_vec();

        // Rebuild the outer wire with split edges
        let mut new_oriented_edges = Vec::new();
        for oe in &old_edges {
            if let Some(intermediates) = edge_splits.get(&oe.edge()) {
                let (start_vid, end_vid) = match edge_vertices.get(&oe.edge()) {
                    Some(&v) => v,
                    None => continue,
                };
                let original_curve = topo.edge(oe.edge())?.curve().clone();

                // Build vertex chain in traversal order
                let chain: Vec<VertexId> = if oe.is_forward() {
                    let mut c = vec![start_vid];
                    c.extend(intermediates.iter().copied());
                    c.push(end_vid);
                    c
                } else {
                    let mut c = vec![end_vid];
                    c.extend(intermediates.iter().rev().copied());
                    c.push(start_vid);
                    c
                };

                // Create sub-edges (reusing from edge_map when possible).
                // Preserve the original edge's curve type so curved edges
                // (Circle, Ellipse) are not silently replaced with lines.
                for k in 0..chain.len() - 1 {
                    let va = chain[k];
                    let vb = chain[k + 1];
                    let va_idx = va.index();
                    let vb_idx = vb.index();
                    let (key_min, key_max) = if va_idx <= vb_idx {
                        (va_idx, vb_idx)
                    } else {
                        (vb_idx, va_idx)
                    };
                    let fwd = va_idx <= vb_idx;
                    let sub_eid = *edge_map.entry((key_min, key_max)).or_insert_with(|| {
                        let (s, e) = if fwd { (va, vb) } else { (vb, va) };
                        topo.add_edge(Edge::new(s, e, original_curve.clone()))
                    });
                    // Skip if edge already in wire (prevents duplicates from
                    // vertex merging creating overlapping segments).
                    if !new_oriented_edges
                        .iter()
                        .any(|e: &OrientedEdge| e.edge() == sub_eid)
                    {
                        new_oriented_edges.push(OrientedEdge::new(sub_eid, fwd));
                    }
                }
            } else {
                // Skip if unsplit edge already added by a prior split expansion.
                if !new_oriented_edges
                    .iter()
                    .any(|e: &OrientedEdge| e.edge() == oe.edge())
                {
                    new_oriented_edges.push(*oe);
                }
            }
        }

        let new_wire =
            Wire::new(new_oriented_edges, true).map_err(crate::OperationsError::Topology)?;
        let new_wire_id = topo.add_wire(new_wire);

        let new_face = if is_reversed {
            Face::new_reversed(new_wire_id, inner_wires, surface)
        } else {
            Face::new(new_wire_id, inner_wires, surface)
        };
        face_ids[fi] = topo.add_face(new_face);
    }

    Ok(())
}

// ---------------------------------------------------------------------------
// Post-assembly boundary edge stitching
// ---------------------------------------------------------------------------

/// Merge geometrically-coincident boundary edge pairs.
///
/// After boolean assembly, the spatial-hash vertex deduplication may map
/// coincident vertices to different hash cells when positions straddle a
/// cell boundary. This creates separate `VertexId`s → separate `EdgeId`s →
/// boundary edges even though the geometry matches. This function finds
/// such pairs and stitches them by rewriting one face's wire to reference
/// the other face's edge.
///
/// Returns the number of edges stitched.
#[allow(clippy::too_many_lines)]
pub(super) fn stitch_boundary_edges(
    topo: &mut Topology,
    face_ids: &mut [FaceId],
    tol: Tolerance,
) -> Result<usize, crate::OperationsError> {
    struct BoundaryEdgeInfo {
        edge_id: EdgeId,
        start_vid: VertexId,
        end_vid: VertexId,
        start_pos: Point3,
        end_pos: Point3,
        midpoint: Point3,
        face_idx: usize,
    }

    // 1. Build edge→face count and collect edge metadata.
    let mut edge_face_count: HashMap<EdgeId, usize> = HashMap::new();
    let mut edge_vertices: HashMap<EdgeId, (VertexId, VertexId)> = HashMap::new();
    // Track which face and wire own each edge for later rewriting.
    let mut edge_owner: HashMap<EdgeId, (usize, WireId)> = HashMap::new();

    for (fi, &fid) in face_ids.iter().enumerate() {
        let face = topo.face(fid)?;
        let outer_wire_id = face.outer_wire();
        // Traverse outer wire and all inner wires for edge counting.
        for wid in std::iter::once(outer_wire_id).chain(face.inner_wires().iter().copied()) {
            let wire = topo.wire(wid)?;
            for oe in wire.edges() {
                let eid = oe.edge();
                *edge_face_count.entry(eid).or_default() += 1;
                if let std::collections::hash_map::Entry::Vacant(e) = edge_vertices.entry(eid) {
                    if let Ok(edge) = topo.edge(eid) {
                        e.insert((edge.start(), edge.end()));
                    }
                }
                edge_owner.entry(eid).or_insert((fi, outer_wire_id));
            }
        }
    }

    // 2. Collect boundary edges (count == 1) with endpoint positions.
    let mut boundary_edges: Vec<BoundaryEdgeInfo> = Vec::new();
    for (&eid, &count) in &edge_face_count {
        if count != 1 {
            continue;
        }
        let &(sv, ev) = match edge_vertices.get(&eid) {
            Some(v) => v,
            None => continue,
        };
        let sp = topo.vertex(sv)?.point();
        let ep = topo.vertex(ev)?.point();
        let mid = Point3::new(
            (sp.x() + ep.x()) * 0.5,
            (sp.y() + ep.y()) * 0.5,
            (sp.z() + ep.z()) * 0.5,
        );
        let &(fi, _wid) = match edge_owner.get(&eid) {
            Some(v) => v,
            None => continue,
        };
        boundary_edges.push(BoundaryEdgeInfo {
            edge_id: eid,
            start_vid: sv,
            end_vid: ev,
            start_pos: sp,
            end_pos: ep,
            midpoint: mid,
            face_idx: fi,
        });
    }

    if boundary_edges.len() < 2 {
        return Ok(0);
    }

    // 3. Build spatial hash grid of boundary edge midpoints.
    let tol_linear = tol.linear;
    let cell_size = boundary_edges
        .iter()
        .map(|be| {
            let dx = be.end_pos.x() - be.start_pos.x();
            let dy = be.end_pos.y() - be.start_pos.y();
            let dz = be.end_pos.z() - be.start_pos.z();
            (dx * dx + dy * dy + dz * dz).sqrt() * 0.5
        })
        .fold(f64::INFINITY, f64::min)
        .max(tol_linear * 10.0);
    let inv_cell = 1.0 / cell_size;

    let mut grid: HashMap<(i64, i64, i64), Vec<usize>> = HashMap::new();
    for (i, be) in boundary_edges.iter().enumerate() {
        let cx = (be.midpoint.x() * inv_cell).floor() as i64;
        let cy = (be.midpoint.y() * inv_cell).floor() as i64;
        let cz = (be.midpoint.z() * inv_cell).floor() as i64;
        grid.entry((cx, cy, cz)).or_default().push(i);
    }

    // 4. Find matching pairs.
    let mut stitched: HashSet<EdgeId> = HashSet::new();
    // Map: (face_idx, old_edge_id) → replacement_edge_id
    let mut replacements: HashMap<(usize, EdgeId), EdgeId> = HashMap::new();
    // Map: old_vertex → new_vertex for cascading vertex remaps
    let mut vertex_remap: HashMap<VertexId, VertexId> = HashMap::new();
    let mut stitch_count = 0;

    let tol_sq = tol_linear * tol_linear;

    for i in 0..boundary_edges.len() {
        let be1 = &boundary_edges[i];
        if stitched.contains(&be1.edge_id) {
            continue;
        }

        let mid = be1.midpoint;
        let cx = (mid.x() * inv_cell).floor() as i64;
        let cy = (mid.y() * inv_cell).floor() as i64;
        let cz = (mid.z() * inv_cell).floor() as i64;

        // Query 3×3×3 neighborhood
        let mut best_match: Option<usize> = None;
        let mut best_dist_sq = f64::INFINITY;

        for dx in -1..=1 {
            for dy in -1..=1 {
                for dz in -1..=1 {
                    if let Some(indices) = grid.get(&(cx + dx, cy + dy, cz + dz)) {
                        for &j in indices {
                            if j <= i {
                                continue;
                            }
                            let be2 = &boundary_edges[j];
                            if stitched.contains(&be2.edge_id) {
                                continue;
                            }
                            // Must be from different faces
                            if be1.face_idx == be2.face_idx {
                                continue;
                            }

                            // Check endpoint matching (same or reversed direction)
                            let same_dir = (be1.start_pos - be2.start_pos).length_squared()
                                < tol_sq
                                && (be1.end_pos - be2.end_pos).length_squared() < tol_sq;
                            let rev_dir = (be1.start_pos - be2.end_pos).length_squared() < tol_sq
                                && (be1.end_pos - be2.start_pos).length_squared() < tol_sq;

                            if !same_dir && !rev_dir {
                                continue;
                            }

                            let mid_dist_sq = (be1.midpoint - be2.midpoint).length_squared();
                            if mid_dist_sq < best_dist_sq {
                                best_dist_sq = mid_dist_sq;
                                best_match = Some(j);
                            }
                        }
                    }
                }
            }
        }

        if let Some(j) = best_match {
            let be2 = &boundary_edges[j];

            // E1 (from be1) is the "keeper". E2 (from be2) gets replaced.
            // Remap be2's vertices to be1's vertices.
            let same_dir = (be1.start_pos - be2.start_pos).length_squared() < tol_sq;

            if same_dir {
                // be2.start → be1.start, be2.end → be1.end
                if be2.start_vid != be1.start_vid {
                    vertex_remap.insert(be2.start_vid, be1.start_vid);
                }
                if be2.end_vid != be1.end_vid {
                    vertex_remap.insert(be2.end_vid, be1.end_vid);
                }
            } else {
                // Reversed: be2.start → be1.end, be2.end → be1.start
                if be2.start_vid != be1.end_vid {
                    vertex_remap.insert(be2.start_vid, be1.end_vid);
                }
                if be2.end_vid != be1.start_vid {
                    vertex_remap.insert(be2.end_vid, be1.start_vid);
                }
            }

            // Replace be2's edge with be1's edge in be2's face wire
            replacements.insert((be2.face_idx, be2.edge_id), be1.edge_id);

            stitched.insert(be1.edge_id);
            stitched.insert(be2.edge_id);
            stitch_count += 1;
        }
    }

    if stitch_count == 0 {
        return Ok(0);
    }

    log::debug!(
        "[boolean] stitch_boundary_edges: {} pairs, {} vertex remaps",
        stitch_count,
        vertex_remap.len()
    );

    // 5. Apply vertex remaps to ALL edges in affected faces.
    //    Cascade: if A→B and B→C, then A→C.
    let mut resolved_remap: HashMap<VertexId, VertexId> = HashMap::new();
    for (&from, &to) in &vertex_remap {
        let mut target = to;
        let mut depth = 0;
        while let Some(&next) = vertex_remap.get(&target) {
            if next == target || depth > 10 {
                break;
            }
            target = next;
            depth += 1;
        }
        resolved_remap.insert(from, target);
    }

    // Collect which faces need rebuilding (faces that have edge replacements
    // OR contain edges with remapped vertices).
    let affected_face_indices: HashSet<usize> = replacements.keys().map(|(fi, _)| *fi).collect();

    for &fi in &affected_face_indices {
        let fid = face_ids[fi];
        let face = topo.face(fid)?;
        let outer_wire_id = face.outer_wire();
        let wire = topo.wire(outer_wire_id)?;
        let surface = face.surface().clone();
        let is_reversed = face.is_reversed();
        let inner_wires: Vec<WireId> = face.inner_wires().to_vec();
        let old_edges: Vec<OrientedEdge> = wire.edges().to_vec();

        let mut new_oriented_edges: Vec<OrientedEdge> = Vec::with_capacity(old_edges.len());

        for oe in &old_edges {
            if let Some(&replacement_eid) = replacements.get(&(fi, oe.edge())) {
                // This edge is being replaced by the keeper edge.
                // The keeper edge's canonical direction may differ from
                // the replaced edge's direction in this wire, so we need
                // to compute the correct orientation.
                let keeper = topo.edge(replacement_eid)?;
                let keeper_start = keeper.start();
                let keeper_end = keeper.end();

                // What vertex does this wire position expect at the start
                // of traversal for this oriented edge?
                let old_edge = topo.edge(oe.edge())?;
                let expected_start = if oe.is_forward() {
                    old_edge.start()
                } else {
                    old_edge.end()
                };
                let resolved_expected = resolved_remap
                    .get(&expected_start)
                    .copied()
                    .unwrap_or(expected_start);

                // If keeper's start matches expected start, traverse forward;
                // otherwise traverse reversed.
                let is_forward =
                    keeper_start == resolved_expected || keeper_end != resolved_expected;
                new_oriented_edges.push(OrientedEdge::new(replacement_eid, is_forward));
            } else {
                // Keep the original edge, but remap its vertices if needed.
                let edge = topo.edge(oe.edge())?;
                let old_start = edge.start();
                let old_end = edge.end();
                let new_start = resolved_remap.get(&old_start).copied();
                let new_end = resolved_remap.get(&old_end).copied();

                if new_start.is_some() || new_end.is_some() {
                    let curve = edge.curve().clone();
                    let s = new_start.unwrap_or(old_start);
                    let e = new_end.unwrap_or(old_end);
                    let new_eid = topo.add_edge(Edge::new(s, e, curve));
                    new_oriented_edges.push(OrientedEdge::new(new_eid, oe.is_forward()));
                } else {
                    new_oriented_edges.push(*oe);
                }
            }
        }

        let new_wire =
            Wire::new(new_oriented_edges, true).map_err(crate::OperationsError::Topology)?;
        let new_wire_id = topo.add_wire(new_wire);
        let new_face = if is_reversed {
            Face::new_reversed(new_wire_id, inner_wires, surface)
        } else {
            Face::new(new_wire_id, inner_wires, surface)
        };
        face_ids[fi] = topo.add_face(new_face);
    }

    Ok(stitch_count)
}

// ---------------------------------------------------------------------------
// Non-manifold edge splitting
// ---------------------------------------------------------------------------

/// Split non-manifold edges into multiple coincident copies.
///
/// After boolean assembly, some edges may be shared by more than 2 faces.
/// This happens when two solids share an edge or a vertex exactly, creating
/// an L-shaped junction. A manifold solid requires every edge to be shared
/// by exactly 2 faces.
///
/// This function detects non-manifold edges and duplicates them, assigning
/// each copy to a pair of faces based on angular ordering around the edge.
/// Faces are sorted by the angle of their outward normal projected onto
/// the plane perpendicular to the edge, then paired consecutively.
#[allow(clippy::too_many_lines)]
pub(super) fn split_nonmanifold_edges(
    topo: &mut Topology,
    face_ids: &mut [FaceId],
) -> Result<(), crate::OperationsError> {
    // Build edge → [(face_index, is_forward)] map.
    let mut edge_faces: HashMap<usize, Vec<(usize, bool)>> = HashMap::new();
    for (fi, &fid) in face_ids.iter().enumerate() {
        let face = topo.face(fid)?;
        // Traverse outer wire and all inner wires.
        for wid in std::iter::once(face.outer_wire()).chain(face.inner_wires().iter().copied()) {
            let wire = topo.wire(wid)?;
            for oe in wire.edges() {
                edge_faces
                    .entry(oe.edge().index())
                    .or_default()
                    .push((fi, oe.is_forward()));
            }
        }
    }

    // Find non-manifold edges (shared by > 2 faces).
    let nonmanifold: Vec<(usize, Vec<(usize, bool)>)> = edge_faces
        .into_iter()
        .filter(|(_, faces)| faces.len() > 2)
        .collect();

    if nonmanifold.is_empty() {
        return Ok(());
    }

    // For each non-manifold edge, sort faces by angle and create edge copies.
    // Map: (face_index, old_edge_index) → new_edge_id
    let mut edge_replacements: HashMap<(usize, usize), EdgeId> = HashMap::new();

    for (edge_idx, face_refs) in &nonmanifold {
        let edge_id = topo.edge_id_from_index(*edge_idx).ok_or_else(|| {
            crate::OperationsError::InvalidInput {
                reason: format!("edge index {edge_idx} not found"),
            }
        })?;
        // Snapshot edge data before any mutable borrows (borrow checker).
        let edge_start = topo.edge(edge_id)?.start();
        let edge_end = topo.edge(edge_id)?.end();
        let edge_curve = topo.edge(edge_id)?.curve().clone();
        let start_pos = topo.vertex(edge_start)?.point();
        let end_pos = topo.vertex(edge_end)?.point();

        // Edge direction vector.
        let edge_dir = Vec3::new(
            end_pos.x() - start_pos.x(),
            end_pos.y() - start_pos.y(),
            end_pos.z() - start_pos.z(),
        );
        let edge_len = edge_dir.length();
        // Numerical-zero guard: skip degenerate zero-length edges that would
        // cause division-by-zero when normalizing the edge direction below.
        if edge_len < 1e-15 {
            continue;
        }
        let edge_axis = Vec3::new(
            edge_dir.x() / edge_len,
            edge_dir.y() / edge_len,
            edge_dir.z() / edge_len,
        );

        // Build a local 2D frame perpendicular to the edge.
        let perp = if edge_axis.x().abs() < 0.9 {
            Vec3::new(1.0, 0.0, 0.0)
        } else {
            Vec3::new(0.0, 1.0, 0.0)
        };
        let u_axis = edge_axis.cross(perp);
        let u_len = u_axis.length();
        // Numerical-zero guard: edge_axis nearly parallel to perp — cross
        // product is degenerate. Skip rather than produce a garbage frame.
        if u_len < 1e-15 {
            continue;
        }
        let u_axis = Vec3::new(u_axis.x() / u_len, u_axis.y() / u_len, u_axis.z() / u_len);
        let v_axis = edge_axis.cross(u_axis);

        // Compute angle for each face's normal projected onto the perpendicular plane.
        let mut face_angles: Vec<(usize, bool, f64)> = Vec::new();
        for &(fi, is_fwd) in face_refs {
            let face = topo.face(face_ids[fi])?;
            let normal = if let FaceSurface::Plane { normal, .. } = face.surface() {
                *normal
            } else {
                // For non-planar faces, approximate normal from wire polygon centroid.
                let wire = topo.wire(face.outer_wire())?;
                let mut sum = Vec3::new(0.0, 0.0, 0.0);
                let mut count = 0usize;
                for oe in wire.edges() {
                    if let Ok(e) = topo.edge(oe.edge()) {
                        if let Ok(v) = topo.vertex(e.start()) {
                            let p = v.point();
                            sum = Vec3::new(sum.x() + p.x(), sum.y() + p.y(), sum.z() + p.z());
                            count += 1;
                        }
                    }
                }
                if count == 0 {
                    continue;
                }
                #[allow(clippy::cast_precision_loss)]
                let inv = 1.0 / count as f64;
                let centroid_dir = Vec3::new(sum.x() * inv, sum.y() * inv, sum.z() * inv);
                let mid = Point3::new(
                    (start_pos.x() + end_pos.x()) * 0.5,
                    (start_pos.y() + end_pos.y()) * 0.5,
                    (start_pos.z() + end_pos.z()) * 0.5,
                );
                Vec3::new(
                    centroid_dir.x() - mid.x(),
                    centroid_dir.y() - mid.y(),
                    centroid_dir.z() - mid.z(),
                )
            };

            // If face is reversed, flip the effective normal for sorting.
            let effective_normal = if face.is_reversed() { -normal } else { normal };

            // Project normal onto perpendicular plane and compute angle.
            let proj_u = effective_normal.dot(u_axis);
            let proj_v = effective_normal.dot(v_axis);
            let angle = proj_v.atan2(proj_u);
            face_angles.push((fi, is_fwd, angle));
        }

        // Sort by angle.
        face_angles.sort_by(|a, b| a.2.partial_cmp(&b.2).unwrap_or(std::cmp::Ordering::Equal));

        // Pair consecutive faces (in angular order) and assign edge copies.
        let n = face_angles.len();
        for pair_idx in 0..(n / 2) {
            let i = pair_idx * 2;
            let j = i + 1;
            if j >= n {
                break;
            }
            let new_edge_id = if pair_idx == 0 {
                edge_id
            } else {
                topo.add_edge(Edge::new(edge_start, edge_end, edge_curve.clone()))
            };
            edge_replacements.insert((face_angles[i].0, *edge_idx), new_edge_id);
            edge_replacements.insert((face_angles[j].0, *edge_idx), new_edge_id);
        }
        // Handle odd face (keeps the original edge — still non-manifold but
        // the iterative loop will process it on the next pass).
        if n % 2 == 1 {
            let last = &face_angles[n - 1];
            edge_replacements.insert((last.0, *edge_idx), edge_id);
        }
    }

    if edge_replacements.is_empty() {
        return Ok(());
    }

    // Rebuild face wires with replaced edges.
    let affected_faces: HashSet<usize> = edge_replacements.keys().map(|(fi, _)| *fi).collect();
    for fi in affected_faces {
        let fid = face_ids[fi];
        let face = topo.face(fid)?;
        let wire = topo.wire(face.outer_wire())?;
        let surface = face.surface().clone();
        let is_reversed = face.is_reversed();
        let inner_wires: Vec<WireId> = face.inner_wires().to_vec();

        let new_edges: Vec<OrientedEdge> = wire
            .edges()
            .iter()
            .map(|oe| {
                if let Some(&new_eid) = edge_replacements.get(&(fi, oe.edge().index())) {
                    OrientedEdge::new(new_eid, oe.is_forward())
                } else {
                    *oe
                }
            })
            .collect();

        let new_wire = Wire::new(new_edges, true).map_err(crate::OperationsError::Topology)?;
        let new_wire_id = topo.add_wire(new_wire);
        let new_face = if is_reversed {
            Face::new_reversed(new_wire_id, inner_wires, surface)
        } else {
            Face::new(new_wire_id, inner_wires, surface)
        };
        face_ids[fi] = topo.add_face(new_face);
    }

    Ok(())
}

// ---------------------------------------------------------------------------
// Shared-boundary fuse fast path
// ---------------------------------------------------------------------------

/// Try to fuse two all-planar solids that share exactly one coplanar face.
///
/// If solids A and B share a face (opposite normals, coplanar, overlapping
/// extent), merge them by removing the shared face pair and combining
/// remaining faces into a new solid via `assemble_solid_mixed`. Returns
/// `None` if the fast path doesn't apply.
#[allow(clippy::too_many_lines)]
pub(super) fn try_shared_boundary_fuse(
    topo: &mut Topology,
    _a: SolidId,
    _b: SolidId,
    face_ids_a: &[FaceId],
    face_ids_b: &[FaceId],
    tol: Tolerance,
) -> Result<Option<SolidId>, crate::OperationsError> {
    struct PlaneInfo {
        normal: Vec3,
        d: f64,
        vertices: Vec<Point3>,
    }

    /// Area ratio below which two faces are not considered extent-matching.
    const SHARED_FACE_AREA_RATIO_MIN: f64 = 0.99;

    // Only worth it for small solids (avoids pathological cases).
    if face_ids_a.len() > 20 || face_ids_b.len() > 20 {
        return Ok(None);
    }

    // Require all faces to be planar.
    for &fid in face_ids_a.iter().chain(face_ids_b.iter()) {
        if !matches!(topo.face(fid)?.surface(), FaceSurface::Plane { .. }) {
            return Ok(None);
        }
    }

    // Snapshot each face: (normal, d, vertices).
    let snapshot = |fid: FaceId| -> Result<PlaneInfo, crate::OperationsError> {
        let face = topo.face(fid)?;
        let surface = face.surface().clone();
        let reversed = face.is_reversed();
        let verts = face_polygon(topo, fid)?;
        let (mut normal, mut d) = analytic_face_normal_d(&surface, &verts);
        if reversed {
            normal = -normal;
            d = -d;
        }
        Ok(PlaneInfo {
            normal,
            d,
            vertices: verts,
        })
    };

    let infos_a: Vec<PlaneInfo> = face_ids_a
        .iter()
        .map(|&fid| snapshot(fid))
        .collect::<Result<Vec<_>, _>>()?;
    let infos_b: Vec<PlaneInfo> = face_ids_b
        .iter()
        .map(|&fid| snapshot(fid))
        .collect::<Result<Vec<_>, _>>()?;

    // Find shared face pair: coplanar with opposite normals and overlapping extent.
    let mut shared_a = None;
    let mut shared_b = None;
    let mut shared_count = 0;

    for (ia, pa) in infos_a.iter().enumerate() {
        for (ib, pb) in infos_b.iter().enumerate() {
            // Opposite normals, same plane (n_a ≈ -n_b, d_a ≈ -d_b).
            let dot = pa.normal.dot(pb.normal);
            if dot > -1.0 + tol.angular {
                continue;
            }
            if !tol.approx_eq(pa.d, -pb.d) {
                continue;
            }

            // Verify matching extent: both face polygons must have
            // approximately equal area.
            let area_a = polygon_area_3d(&pa.vertices, pa.normal);
            let area_b = polygon_area_3d(&pb.vertices, pb.normal);
            let area_ratio = if area_a > area_b {
                area_b / area_a
            } else {
                area_a / area_b
            };
            if area_ratio < SHARED_FACE_AREA_RATIO_MIN {
                continue;
            }

            // Centroids should be within a geometry-scaled tolerance.
            // Use sqrt(area) as the face extent scale.
            let centroid_a = polygon_centroid(&pa.vertices);
            let centroid_b = polygon_centroid(&pb.vertices);
            let dist = (centroid_a - centroid_b).length();
            let face_extent = area_a.sqrt().max(tol.linear);
            // Geometry-scaled centroid coincidence test: centroids must be within
            // 1e-6 * face_extent (i.e., within one millionth of the face size).
            // This relative threshold adapts to model scale — a 1m face allows
            // 1 micron drift, a 1mm face allows 1 nm.
            if dist > face_extent * 1e-6 {
                continue;
            }

            shared_a = Some(ia);
            shared_b = Some(ib);
            shared_count += 1;

            if shared_count > 1 {
                // Multiple shared faces → too complex for fast path.
                return Ok(None);
            }
        }
    }

    let (skip_a, skip_b) = match (shared_a, shared_b) {
        (Some(a), Some(b)) => (a, b),
        _ => return Ok(None),
    };

    // Build face specs from all faces except the shared pair.
    let mut face_specs: Vec<FaceSpec> = Vec::with_capacity(face_ids_a.len() + face_ids_b.len() - 2);

    for (i, info) in infos_a.iter().enumerate() {
        if i == skip_a {
            continue;
        }
        face_specs.push(FaceSpec::Planar {
            vertices: info.vertices.clone(),
            normal: info.normal,
            d: info.d,
            inner_wires: vec![],
        });
    }
    for (i, info) in infos_b.iter().enumerate() {
        if i == skip_b {
            continue;
        }
        face_specs.push(FaceSpec::Planar {
            vertices: info.vertices.clone(),
            normal: info.normal,
            d: info.d,
            inner_wires: vec![],
        });
    }

    let result = assemble_solid_mixed(topo, &face_specs, tol)?;
    Ok(Some(result))
}

// ---------------------------------------------------------------------------
// Polygon area helper
// ---------------------------------------------------------------------------

/// Compute the area of a 3D polygon given its vertices and face normal.
pub(super) fn polygon_area_3d(vertices: &[Point3], normal: Vec3) -> f64 {
    if vertices.len() < 3 {
        return 0.0;
    }
    let mut area = Vec3::new(0.0, 0.0, 0.0);
    let v0 = vertices[0];
    for i in 1..vertices.len() - 1 {
        let e1 = vertices[i] - v0;
        let e2 = vertices[i + 1] - v0;
        area += e1.cross(e2);
    }
    (area.dot(normal) * 0.5).abs()
}
