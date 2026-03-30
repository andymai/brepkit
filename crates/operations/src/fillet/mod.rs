//! Edge filleting (rounding edges with a constant or variable radius).
//!
//! Replaces sharp edges with a smooth cylindrical fillet surface.
//! Supports edges between planar faces, analytic faces (cylinder, cone,
//! sphere, torus), and NURBS faces from a prior fillet operation.  Each
//! filleted edge is replaced by a true rolling-ball NURBS blend surface
//! with G1 tangent continuity.
//!
//! For NURBS adjacent faces the outward normal is computed by projecting
//! the edge sample point onto the surface, giving accurate cross-section
//! geometry (see `face_surface_normal_at`).  Non-planar faces containing
//! target edges are trimmed by offsetting boundary vertices at fillet
//! contact locations along face boundary directions.
//!
//! The rolling-ball algorithm:
//! 1. For each target edge, find the two adjacent planar faces
//! 2. Offset each face plane inward by radius R
//! 3. Intersect the offset planes to find the fillet center line (spine)
//! 4. Compute contact points where the rolling ball touches each face
//! 5. Build a degree (2,1) rational NURBS surface: circular arc cross-section
//!    swept along the edge
//! 6. Trim the adjacent faces along the contact lines
//! 7. Assemble the result with modified faces + NURBS fillet faces
//!
//! The NURBS fillet surface uses the exact rational circular arc
//! representation (3 control points, weights [1, cos(α/2), 1]),
//! giving mathematically exact G1 continuity with both adjacent faces.

mod g1_chain;
mod geometry;
mod helpers;
mod radius_law;
mod rolling_ball;
#[cfg(test)]
mod tests;

// Re-export public items.
pub use g1_chain::fillet_rolling_ball_propagate_g1;
pub(crate) use geometry::face_surface_normal_at;
pub use radius_law::FilletRadiusLaw;
#[allow(deprecated)]
pub use rolling_ball::fillet_rolling_ball;

use std::collections::{HashMap, HashSet};

use brepkit_math::tolerance::Tolerance;
use brepkit_math::vec::Point3;
use brepkit_topology::Topology;
use brepkit_topology::edge::EdgeId;
use brepkit_topology::face::FaceSurface;
use brepkit_topology::solid::SolidId;

use crate::boolean::FaceSpec;
use crate::dot_normal_point;

use helpers::{FacePolygon, FilletEdgeData, extract_inner_wire_positions, record_fillet_point};

/// Fillet one or more edges of a solid with a constant radius (flat chamfer).
///
/// **Deprecated**: This creates flat bevel faces, not rounded fillets.
/// Use [`fillet_rolling_ball`] for true G1-continuous NURBS blend surfaces.
///
/// Each target edge is replaced by a flat bevel face (chamfer-like
/// approximation of a fillet arc).
///
/// # Errors
///
/// Returns an error if:
/// - `radius` is non-positive
/// - `edges` is empty
/// - Any edge is not shared by exactly two faces
/// - A target edge is adjacent to a non-planar face
#[deprecated(
    since = "0.8.0",
    note = "Use fillet_rolling_ball for true rounded fillets"
)]
#[allow(clippy::too_many_lines)]
pub fn fillet(
    topo: &mut Topology,
    solid: SolidId,
    edges: &[EdgeId],
    radius: f64,
) -> Result<SolidId, crate::OperationsError> {
    let tol = Tolerance::new();

    if radius <= tol.linear {
        return Err(crate::OperationsError::InvalidInput {
            reason: format!("fillet radius must be positive, got {radius}"),
        });
    }
    if edges.is_empty() {
        return Err(crate::OperationsError::InvalidInput {
            reason: "no edges specified for fillet".into(),
        });
    }

    // Collect face data.
    let solid_data = topo.solid(solid)?;
    let shell = topo.shell(solid_data.outer_shell())?;
    let shell_face_ids: Vec<_> = shell.faces().to_vec();

    let mut edge_to_faces: HashMap<usize, Vec<_>> = HashMap::new();
    let mut face_polygons: HashMap<usize, FacePolygon> = HashMap::new();

    for &face_id in &shell_face_ids {
        let face = topo.face(face_id)?;

        let wire = topo.wire(face.outer_wire())?;
        let mut vertex_ids = Vec::with_capacity(wire.edges().len());
        let mut positions = Vec::with_capacity(wire.edges().len());
        let mut wire_edge_ids = Vec::with_capacity(wire.edges().len());

        for oe in wire.edges() {
            let edge = topo.edge(oe.edge())?;
            let vid = oe.oriented_start(edge);
            vertex_ids.push(vid);
            positions.push(topo.vertex(vid)?.point());
            wire_edge_ids.push(oe.edge());

            edge_to_faces
                .entry(oe.edge().index())
                .or_default()
                .push(face_id);
        }

        // Inner wire edges also contribute to adjacency: an edge shared
        // between a face's inner wire (hole boundary) and another face's
        // outer wire should be counted for both faces.
        // Also extract inner wire vertex positions for preservation.
        let mut face_inner_wires = Vec::new();
        for &inner_wid in face.inner_wires() {
            let inner_wire = topo.wire(inner_wid)?;
            let mut iw_positions = Vec::new();
            for oe in inner_wire.edges() {
                edge_to_faces
                    .entry(oe.edge().index())
                    .or_default()
                    .push(face_id);
                let edge = topo.edge(oe.edge())?;
                let vid = oe.oriented_start(edge);
                iw_positions.push(topo.vertex(vid)?.point());
            }
            if !iw_positions.is_empty() {
                face_inner_wires.push(iw_positions);
            }
        }

        // Only build polygon data for planar faces. Non-planar faces
        // will be passed through unchanged if they don't contain target edges.
        let normal = match face.surface() {
            FaceSurface::Plane { normal, .. } => *normal,
            _ => continue,
        };
        if positions.is_empty() {
            continue;
        }
        let d = dot_normal_point(normal, positions[0]);

        face_polygons.insert(
            face_id.index(),
            FacePolygon {
                vertex_ids,
                positions,
                wire_edge_ids,
                normal,
                d,
                inner_wires: face_inner_wires,
            },
        );
    }

    // Filter target edges: only keep manifold edges (shared by exactly 2 faces).
    // Non-manifold edges (boundary/seam) are silently skipped rather than causing
    // an error, so callers can pass "all edges" without pre-filtering.
    let filtered_edges: Vec<EdgeId> = edges
        .iter()
        .copied()
        .filter(|edge_id| {
            edge_to_faces
                .get(&edge_id.index())
                .is_some_and(|faces| faces.len() == 2)
        })
        .collect();

    if filtered_edges.is_empty() {
        return Err(crate::OperationsError::InvalidInput {
            reason: "no manifold edges to fillet (all edges are boundary or missing)".into(),
        });
    }

    let target_set: HashSet<usize> = filtered_edges.iter().map(|e| e.index()).collect();

    // Vertices at endpoints of filleted edges (used to detect side-face corners).
    let mut vertex_fillet_endpoints: HashSet<usize> = HashSet::new();
    for &edge_id in &filtered_edges {
        let edge = topo.edge(edge_id)?;
        vertex_fillet_endpoints.insert(edge.start().index());
        vertex_fillet_endpoints.insert(edge.end().index());
    }

    // Build modified face polygons and fillet faces.
    // Strategy: identical to chamfer but with more offset segments to
    // approximate the circular fillet.
    let mut fillet_data: HashMap<usize, FilletEdgeData> = HashMap::new();
    let mut result_specs: Vec<FaceSpec> = Vec::new();

    for &face_id in &shell_face_ids {
        // Non-planar faces pass through unchanged.
        let Some(poly) = face_polygons.get(&face_id.index()) else {
            let face = topo.face(face_id)?;
            let verts = crate::boolean::face_polygon(topo, face_id)?;
            let np_inner = extract_inner_wire_positions(topo, face)?;
            result_specs.push(FaceSpec::Surface {
                vertices: verts,
                surface: face.surface().clone(),
                reversed: false,
                inner_wires: np_inner,
            });
            continue;
        };
        let n = poly.positions.len();
        let mut new_verts: Vec<Point3> = Vec::with_capacity(n + target_set.len());

        for i in 0..n {
            let prev_i = if i == 0 { n - 1 } else { i - 1 };
            let next_i = (i + 1) % n;

            let before_filleted = target_set.contains(&poly.wire_edge_ids[prev_i].index());
            let after_filleted = target_set.contains(&poly.wire_edge_ids[i].index());

            let pos = poly.positions[i];
            let prev_pos = poly.positions[prev_i];
            let next_pos = poly.positions[next_i];

            // Check if vertex sits at a fillet endpoint even though neither
            // adjacent edge of THIS face is the filleted edge (side face case).
            let at_fillet_endpoint = vertex_fillet_endpoints.contains(&poly.vertex_ids[i].index());

            match (before_filleted, after_filleted, at_fillet_endpoint) {
                (false, false, false) => {
                    new_verts.push(pos);
                }
                (false, false, true) => {
                    // Side face corner: split into two contact points.
                    let dir_prev = (prev_pos - pos).normalize()?;
                    new_verts.push(pos + dir_prev * radius);

                    let dir_next = (next_pos - pos).normalize()?;
                    new_verts.push(pos + dir_next * radius);
                }
                (true, false, _) => {
                    let dir = (next_pos - pos).normalize()?;
                    let c = pos + dir * radius;
                    new_verts.push(c);
                    record_fillet_point(
                        &mut fillet_data,
                        poly.wire_edge_ids[prev_i].index(),
                        poly.vertex_ids[i],
                        face_id,
                        c,
                    );
                }
                (false, true, _) => {
                    let dir = (prev_pos - pos).normalize()?;
                    let c = pos + dir * radius;
                    new_verts.push(c);
                    record_fillet_point(
                        &mut fillet_data,
                        poly.wire_edge_ids[i].index(),
                        poly.vertex_ids[i],
                        face_id,
                        c,
                    );
                }
                (true, true, _) => {
                    let dir_prev = (prev_pos - pos).normalize()?;
                    let c_after = pos + dir_prev * radius;
                    new_verts.push(c_after);
                    record_fillet_point(
                        &mut fillet_data,
                        poly.wire_edge_ids[i].index(),
                        poly.vertex_ids[i],
                        face_id,
                        c_after,
                    );

                    let dir_next = (next_pos - pos).normalize()?;
                    let c_before = pos + dir_next * radius;
                    new_verts.push(c_before);
                    record_fillet_point(
                        &mut fillet_data,
                        poly.wire_edge_ids[prev_i].index(),
                        poly.vertex_ids[i],
                        face_id,
                        c_before,
                    );
                }
            }
        }

        let new_d = dot_normal_point(poly.normal, new_verts[0]);
        result_specs.push(FaceSpec::Planar {
            vertices: new_verts,
            normal: poly.normal,
            d: new_d,
            inner_wires: poly.inner_wires.clone(),
        });
    }

    // Build fillet faces (planar quads approximating the fillet arc).
    for &edge_id in &filtered_edges {
        let data = fillet_data.get(&edge_id.index()).ok_or_else(|| {
            crate::OperationsError::InvalidInput {
                reason: format!("failed to compute fillet data for edge {}", edge_id.index()),
            }
        })?;

        let edge = topo.edge(edge_id)?;
        let v_start = edge.start();
        let v_end = edge.end();

        let Some(face_list) = edge_to_faces.get(&edge_id.index()) else {
            return Err(crate::OperationsError::InvalidInput {
                reason: format!(
                    "fillet: edge {} not found in edge-to-face map",
                    edge_id.index()
                ),
            });
        };
        if face_list.len() < 2 {
            return Err(crate::OperationsError::InvalidInput {
                reason: format!(
                    "fillet: edge {} has {} adjacent faces, expected 2",
                    edge_id.index(),
                    face_list.len()
                ),
            });
        }
        let f1 = face_list[0];
        let f2 = face_list[1];

        let c1_start = data.get_point(f1, v_start)?;
        let c1_end = data.get_point(f1, v_end)?;
        let c2_start = data.get_point(f2, v_start)?;
        let c2_end = data.get_point(f2, v_end)?;

        let n1 = face_polygons[&f1.index()].normal;
        let n2 = face_polygons[&f2.index()].normal;
        let avg_normal = n1 + n2;

        let edge_a = c2_start - c1_start;
        let edge_b = c1_end - c1_start;
        let raw_normal = edge_a.cross(edge_b);

        let (quad, normal) = if raw_normal.dot(avg_normal) >= 0.0 {
            (
                vec![c1_start, c2_start, c2_end, c1_end],
                raw_normal.normalize()?,
            )
        } else {
            let flipped = edge_b.cross(edge_a);
            (
                vec![c1_start, c1_end, c2_end, c2_start],
                flipped.normalize()?,
            )
        };

        let d = dot_normal_point(normal, quad[0]);
        result_specs.push(FaceSpec::Planar {
            vertices: quad,
            normal,
            d,
            inner_wires: vec![],
        });
    }

    crate::boolean::assemble_solid_mixed(topo, &result_specs, tol)
}

/// Fillet edges with variable radius using canal surface generation.
///
/// Each edge gets a [`FilletRadiusLaw`] that defines how the radius
/// changes along the edge. The fillet surface is a canal surface:
/// the envelope of a sphere of varying radius moving along the edge.
///
/// The implementation samples the radius law at multiple points along
/// each edge, computes rolling-ball arc cross-sections at each sample,
/// and interpolates a NURBS surface through all cross-sections using
/// tensor-product surface fitting.
///
/// For constant radius, use `FilletRadiusLaw::Constant(r)` or the
/// simpler [`fillet_rolling_ball`] function.
///
/// # Errors
///
/// Returns errors similar to [`fillet_rolling_ball`].
#[allow(clippy::too_many_lines)]
pub fn fillet_variable(
    topo: &mut Topology,
    solid: SolidId,
    edge_laws: &[(EdgeId, FilletRadiusLaw)],
) -> Result<SolidId, crate::OperationsError> {
    let tol = Tolerance::new();

    if edge_laws.is_empty() {
        return Err(crate::OperationsError::InvalidInput {
            reason: "no edges specified for fillet".into(),
        });
    }

    // Validate all radii are positive.
    for (_, law) in edge_laws {
        for t in [0.0, 0.25, 0.5, 0.75, 1.0] {
            if law.evaluate(t) <= tol.linear {
                return Err(crate::OperationsError::InvalidInput {
                    reason: "fillet radius must be positive at all points".into(),
                });
            }
        }
    }

    // Collect face data (same as fillet_rolling_ball).
    let solid_data = topo.solid(solid)?;
    let shell = topo.shell(solid_data.outer_shell())?;
    let shell_face_ids: Vec<_> = shell.faces().to_vec();

    let mut edge_to_faces: std::collections::HashMap<usize, Vec<_>> =
        std::collections::HashMap::new();
    let mut face_polygons: std::collections::HashMap<usize, FacePolygon> =
        std::collections::HashMap::new();
    let mut face_surfaces: std::collections::HashMap<usize, FaceSurface> =
        std::collections::HashMap::new();
    let target_set: std::collections::HashSet<usize> =
        edge_laws.iter().map(|(e, _)| e.index()).collect();

    for &face_id in &shell_face_ids {
        let face = topo.face(face_id)?;
        face_surfaces.insert(face_id.index(), face.surface().clone());

        let wire = topo.wire(face.outer_wire())?;
        let mut vertex_ids = Vec::new();
        let mut positions = Vec::new();
        let mut wire_edge_ids = Vec::new();

        for oe in wire.edges() {
            let edge = topo.edge(oe.edge())?;
            let vid = oe.oriented_start(edge);
            vertex_ids.push(vid);
            positions.push(topo.vertex(vid)?.point());
            wire_edge_ids.push(oe.edge());
            edge_to_faces
                .entry(oe.edge().index())
                .or_default()
                .push(face_id);
        }

        // Extract inner wire vertex positions for preservation.
        let mut face_inner_wires = Vec::new();
        for &inner_wid in face.inner_wires() {
            let inner_wire = topo.wire(inner_wid)?;
            let mut iw_positions = Vec::new();
            for oe in inner_wire.edges() {
                edge_to_faces
                    .entry(oe.edge().index())
                    .or_default()
                    .push(face_id);
                let edge_data = topo.edge(oe.edge())?;
                let vid = oe.oriented_start(edge_data);
                iw_positions.push(topo.vertex(vid)?.point());
            }
            if !iw_positions.is_empty() {
                face_inner_wires.push(iw_positions);
            }
        }

        // Build polygon data for planar faces (used for trimming).
        let normal = match face.surface() {
            FaceSurface::Plane { normal, .. } => *normal,
            _ => continue,
        };

        face_polygons.insert(
            face_id.index(),
            FacePolygon {
                vertex_ids,
                positions,
                wire_edge_ids,
                normal,
                d: 0.0,
                inner_wires: face_inner_wires,
            },
        );
    }

    // Build a map from edge index to radius law for per-vertex radius lookup.
    // Each vertex adjacent to a filleted edge uses that edge's actual radius
    // at the vertex (start=0.0, end=1.0) instead of a global average.
    let edge_law_map: HashMap<usize, &FilletRadiusLaw> = edge_laws
        .iter()
        .map(|(eid, law)| (eid.index(), law))
        .collect();

    // Use the constant-radius trimming from the basic fillet for the planar faces.
    // The NURBS canal surface replaces the fillet face.
    let mut all_specs: Vec<FaceSpec> = Vec::new();

    for &face_id in &shell_face_ids {
        let Some(poly) = face_polygons.get(&face_id.index()) else {
            let face = topo.face(face_id)?;
            let verts = crate::boolean::face_polygon(topo, face_id)?;
            let np_inner = extract_inner_wire_positions(topo, face)?;
            all_specs.push(FaceSpec::Surface {
                vertices: verts,
                surface: face.surface().clone(),
                reversed: false,
                inner_wires: np_inner,
            });
            continue;
        };
        let n = poly.positions.len();

        // Skip polygon trimming for degenerate faces (e.g., disc caps).
        if n < 3 {
            all_specs.push(FaceSpec::Planar {
                vertices: poly.positions.clone(),
                normal: poly.normal,
                d: poly.d,
                inner_wires: poly.inner_wires.clone(),
            });
            continue;
        }

        let mut new_verts: Vec<Point3> = Vec::with_capacity(n + target_set.len());

        for i in 0..n {
            let prev_i = if i == 0 { n - 1 } else { i - 1 };
            let next_i = (i + 1) % n;
            let before_filleted = target_set.contains(&poly.wire_edge_ids[prev_i].index());
            let after_filleted = target_set.contains(&poly.wire_edge_ids[i].index());
            let pos = poly.positions[i];
            let prev_pos = poly.positions[prev_i];
            let next_pos = poly.positions[next_i];

            // Look up per-edge radius at this vertex:
            // - "before" edge (prev_i): vertex i is at its end → evaluate(1.0)
            // - "after" edge (i): vertex i is at its start → evaluate(0.0)
            let radius_before = edge_law_map
                .get(&poly.wire_edge_ids[prev_i].index())
                .map_or(0.0, |law| law.evaluate(1.0));
            let radius_after = edge_law_map
                .get(&poly.wire_edge_ids[i].index())
                .map_or(0.0, |law| law.evaluate(0.0));

            match (before_filleted, after_filleted) {
                (false, false) => new_verts.push(pos),
                (true, false) => {
                    let dir = (next_pos - pos).normalize()?;
                    new_verts.push(pos + dir * radius_before);
                }
                (false, true) => {
                    let dir = (prev_pos - pos).normalize()?;
                    new_verts.push(pos + dir * radius_after);
                }
                (true, true) => {
                    let dir_prev = (prev_pos - pos).normalize()?;
                    new_verts.push(pos + dir_prev * radius_before);
                    let dir_next = (next_pos - pos).normalize()?;
                    new_verts.push(pos + dir_next * radius_after);
                }
            }
        }

        let new_d = dot_normal_point(poly.normal, new_verts[0]);
        all_specs.push(FaceSpec::Planar {
            vertices: new_verts,
            normal: poly.normal,
            d: new_d,
            inner_wires: poly.inner_wires.clone(),
        });
    }

    // Build variable-radius NURBS canal surfaces for each edge.
    let n_samples = 5; // Number of cross-sections along each edge

    for (edge_id, law) in edge_laws {
        let edge = topo.edge(*edge_id)?;
        let p_start = topo.vertex(edge.start())?.point();
        let p_end = topo.vertex(edge.end())?.point();

        let Some(face_list) = edge_to_faces.get(&edge_id.index()) else {
            continue;
        };
        if face_list.len() < 2 {
            continue;
        }
        let f1 = face_list[0];
        let f2 = face_list[1];

        // Get face surfaces for normal evaluation on curved faces.
        let (Some(surf1), Some(surf2)) = (
            face_surfaces.get(&f1.index()),
            face_surfaces.get(&f2.index()),
        ) else {
            continue;
        };

        // Evaluate surface normals at the edge start point.
        let Some(n1_start) = face_surface_normal_at(surf1, p_start) else {
            continue;
        };
        let Some(n2_start) = face_surface_normal_at(surf2, p_start) else {
            continue;
        };

        let edge_curve = edge.curve().clone();

        let edge_tan = geometry::sample_edge_tangent(&edge_curve, p_start, p_end, 0.0);
        if edge_tan.length() < tol.linear {
            continue;
        }
        let edge_dir = edge_tan.normalize()?;

        // Reference cross-section at t=0 for fallback directions.
        let cross1_ref = edge_dir.cross(n1_start);
        let cross2_ref = edge_dir.cross(n2_start);
        let d1_ref = if cross1_ref.dot(n2_start) > 0.0 {
            cross1_ref
        } else {
            -cross1_ref
        };
        let d2_ref = if cross2_ref.dot(n1_start) > 0.0 {
            cross2_ref
        } else {
            -cross2_ref
        };
        let d1_ref = d1_ref.normalize().unwrap_or(d1_ref);
        let d2_ref = d2_ref.normalize().unwrap_or(d2_ref);
        let cos_half_ref = d1_ref.dot(d2_ref).clamp(-1.0, 1.0);
        let half_angle = cos_half_ref.acos() / 2.0;

        if half_angle.abs() < tol.angular {
            continue;
        }

        // Use more samples for curved faces or curved edges.
        let both_planar = matches!(surf1, FaceSurface::Plane { .. })
            && matches!(surf2, FaceSurface::Plane { .. });
        let n_v = if both_planar {
            geometry::edge_v_samples(&edge_curve).max(n_samples)
        } else {
            geometry::edge_v_samples(&edge_curve).max(n_samples).max(7)
        };

        // Build interpolation grid: n_v rows × 3 columns (arc CPs).
        let mut grid: Vec<Vec<Point3>> = Vec::with_capacity(n_v);
        let mut sample_weights: Vec<f64> = Vec::with_capacity(n_v);

        #[allow(clippy::cast_precision_loss)]
        for s in 0..n_v {
            let t = s as f64 / (n_v - 1).max(1) as f64;
            let r = law.evaluate(t);
            let p = geometry::sample_edge_point(&edge_curve, p_start, p_end, t);
            let tan = geometry::sample_edge_tangent(&edge_curve, p_start, p_end, t);
            let local_dir = tan.normalize().unwrap_or(edge_dir);

            // Evaluate surface normals at this sample point.
            let ln1 = face_surface_normal_at(surf1, p).unwrap_or(n1_start);
            let ln2 = face_surface_normal_at(surf2, p).unwrap_or(n2_start);

            // Vertex blend direction: uses original convention (toward other face's
            // outward normal). The vertex blend fills the gap between fillet strips
            // at a vertex, and its geometry depends on the edge fillet positions.
            // TODO(#260): vertex blend direction may need adjustment after edge
            // fillet direction fix — currently causes ~30% volume inflation on
            // all-edges fillet. Investigate vertex blend contact point computation.
            let c1 = local_dir.cross(ln1);
            let c2 = local_dir.cross(ln2);
            let ld1 = if c1.dot(ln2) > 0.0 { c1 } else { -c1 };
            let ld2 = if c2.dot(ln1) > 0.0 { c2 } else { -c2 };
            let ld1 = ld1.normalize().unwrap_or(d1_ref);
            let ld2 = ld2.normalize().unwrap_or(d2_ref);

            let local_cos = ld1.dot(ld2).clamp(-1.0, 1.0);
            let local_half = local_cos.acos() / 2.0;
            let bisector = (ld1 + ld2).normalize().unwrap_or(d1_ref);

            let contact1 = p + ld1 * r;
            let contact2 = p + ld2 * r;
            let mid_dist = r / local_half.cos().max(0.01);
            let mid_cp = p + bisector * mid_dist;

            sample_weights.push(local_half.cos().max(0.01));
            grid.push(vec![contact1, mid_cp, contact2]);
        }

        // Build a rational NURBS surface with exact circular arc cross-sections.
        // u-direction: degree 2, 3 CPs with weights [1, cos(α/2), 1]
        // v-direction: interpolated through sampled stations along the edge
        let degree_v = (n_v - 1).min(3);

        // Interpolate each of the 3 u-rows independently in v.
        let row_contact1: Vec<Point3> = (0..n_v).map(|i| grid[i][0]).collect();
        let row_mid: Vec<Point3> = (0..n_v).map(|i| grid[i][1]).collect();
        let row_contact2: Vec<Point3> = (0..n_v).map(|i| grid[i][2]).collect();

        let crv0 = brepkit_math::nurbs::fitting::interpolate(&row_contact1, degree_v)
            .map_err(crate::OperationsError::Math)?;
        let crv1 = brepkit_math::nurbs::fitting::interpolate(&row_mid, degree_v)
            .map_err(crate::OperationsError::Math)?;
        let crv2 = brepkit_math::nurbs::fitting::interpolate(&row_contact2, degree_v)
            .map_err(crate::OperationsError::Math)?;

        // All three curves share the same knot vector and degree since they
        // interpolate the same number of points with the same degree.
        let knots_v = crv0.knots().to_vec();
        let n_cp_v = crv0.control_points().len();

        // Per-station arc weights: interpolate sample_weights to match n_cp_v.
        #[allow(clippy::cast_precision_loss)]
        let mid_weights: Vec<f64> = if n_cp_v == sample_weights.len() {
            sample_weights.clone()
        } else {
            (0..n_cp_v)
                .map(|i| {
                    let t = i as f64 / (n_cp_v - 1).max(1) as f64;
                    let idx_f = t * (sample_weights.len() - 1).max(1) as f64;
                    let lo = (idx_f.floor() as usize).min(sample_weights.len() - 1);
                    let hi = (lo + 1).min(sample_weights.len() - 1);
                    let frac = idx_f - lo as f64;
                    sample_weights[lo] * (1.0 - frac) + sample_weights[hi] * frac
                })
                .collect()
        };

        let surface = brepkit_math::nurbs::surface::NurbsSurface::new(
            2,                                  // degree_u (circular arc)
            crv0.degree(),                      // degree_v
            vec![0.0, 0.0, 0.0, 1.0, 1.0, 1.0], // knots_u
            knots_v,
            vec![
                crv0.control_points().to_vec(),
                crv1.control_points().to_vec(),
                crv2.control_points().to_vec(),
            ],
            vec![vec![1.0; n_cp_v], mid_weights, vec![1.0; n_cp_v]],
        )
        .map_err(crate::OperationsError::Math)?;

        // Boundary vertices for the canal surface.
        let c1s = grid[0][0];
        let c2s = grid[0][2];
        let c1e = grid[n_v - 1][0];
        let c2e = grid[n_v - 1][2];

        all_specs.push(FaceSpec::Surface {
            vertices: vec![c1s, c2s, c2e, c1e],
            surface: FaceSurface::Nurbs(surface),
            reversed: false,
            inner_wires: vec![],
        });
    }

    crate::boolean::assemble_solid_mixed(topo, &all_specs, tol)
}
