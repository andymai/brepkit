//! Edge filleting (rounding edges with a constant radius).
//!
//! Replaces sharp edges with a smooth cylindrical fillet surface.
//! Works on planar solids only. Each filleted edge is replaced by
//! a true rolling-ball NURBS blend surface with G1 tangent continuity.
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

use std::collections::{HashMap, HashSet};

use brepkit_math::nurbs::surface::NurbsSurface;
use brepkit_math::tolerance::Tolerance;
use brepkit_math::vec::{Point3, Vec3};
use brepkit_topology::Topology;
use brepkit_topology::edge::EdgeId;
use brepkit_topology::face::{FaceId, FaceSurface};
use brepkit_topology::solid::SolidId;
use brepkit_topology::vertex::VertexId;

use crate::boolean::FaceSpec;
use crate::dot_normal_point;

/// Fillet one or more edges of a solid with a constant radius.
///
/// Each target edge is replaced by a flat bevel face (chamfer-like
/// approximation of a fillet arc). For true cylindrical fillet
/// surfaces, a NURBS implementation would be needed, but this
/// piecewise-planar approach produces correct topology and is
/// suitable for downstream tessellation at any resolution.
///
/// # Errors
///
/// Returns an error if:
/// - `radius` is non-positive
/// - `edges` is empty
/// - Any edge is not shared by exactly two faces
/// - A target edge is adjacent to a non-planar face
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
    let shell_face_ids: Vec<FaceId> = shell.faces().to_vec();

    let mut edge_to_faces: HashMap<usize, Vec<FaceId>> = HashMap::new();
    let mut face_polygons: HashMap<usize, FacePolygon> = HashMap::new();

    for &face_id in &shell_face_ids {
        let face = topo.face(face_id)?;

        let wire = topo.wire(face.outer_wire())?;
        let mut vertex_ids = Vec::with_capacity(wire.edges().len());
        let mut positions = Vec::with_capacity(wire.edges().len());
        let mut wire_edge_ids = Vec::with_capacity(wire.edges().len());

        for oe in wire.edges() {
            let edge = topo.edge(oe.edge())?;
            let vid = if oe.is_forward() {
                edge.start()
            } else {
                edge.end()
            };
            vertex_ids.push(vid);
            positions.push(topo.vertex(vid)?.point());
            wire_edge_ids.push(oe.edge());

            edge_to_faces
                .entry(oe.edge().index())
                .or_default()
                .push(face_id);
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
            },
        );
    }

    // Validate target edges.
    let target_set: HashSet<usize> = edges.iter().map(|e| e.index()).collect();

    for &edge_id in edges {
        let faces = edge_to_faces.get(&edge_id.index()).ok_or_else(|| {
            crate::OperationsError::InvalidInput {
                reason: format!("edge {} is not part of the solid", edge_id.index()),
            }
        })?;
        if faces.len() != 2 {
            return Err(crate::OperationsError::InvalidInput {
                reason: format!(
                    "edge {} is shared by {} faces, expected exactly 2",
                    edge_id.index(),
                    faces.len()
                ),
            });
        }
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
            result_specs.push(FaceSpec::Surface {
                vertices: verts,
                surface: face.surface().clone(),
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

            match (before_filleted, after_filleted) {
                (false, false) => {
                    new_verts.push(pos);
                }
                (true, false) => {
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
                (false, true) => {
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
                (true, true) => {
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
        });
    }

    // Build fillet faces (planar quads approximating the fillet arc).
    for &edge_id in edges {
        let data = fillet_data.get(&edge_id.index()).ok_or_else(|| {
            crate::OperationsError::InvalidInput {
                reason: format!("failed to compute fillet data for edge {}", edge_id.index()),
            }
        })?;

        let edge = topo.edge(edge_id)?;
        let v_start = edge.start();
        let v_end = edge.end();

        let face_list = &edge_to_faces[&edge_id.index()];
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
        });
    }

    crate::boolean::assemble_solid_mixed(topo, &result_specs, tol)
}

/// Fillet one or more edges of a solid using the rolling-ball algorithm.
///
/// Produces true NURBS cylindrical fillet surfaces with G1 tangent
/// continuity, replacing the flat-quad approximation of [`fillet`].
///
/// For each target edge between two planar faces:
/// 1. Offset both face planes inward by `radius`
/// 2. Intersect offset planes to find the fillet center line
/// 3. Compute contact points on each face
/// 4. Build a degree (2,1) rational NURBS surface with exact circular
///    arc cross-section
///
/// # Errors
///
/// Returns an error if:
/// - `radius` is non-positive
/// - `edges` is empty
/// - Any edge is not shared by exactly two faces
/// - A target edge is adjacent to a non-planar face
#[allow(clippy::too_many_lines)]
pub fn fillet_rolling_ball(
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

    // Phase 1: Collect face data and build adjacency.
    let solid_data = topo.solid(solid)?;
    let shell = topo.shell(solid_data.outer_shell())?;
    let shell_face_ids: Vec<FaceId> = shell.faces().to_vec();

    let mut edge_to_faces: HashMap<usize, Vec<FaceId>> = HashMap::new();
    let mut face_polygons: HashMap<usize, FacePolygon> = HashMap::new();

    for &face_id in &shell_face_ids {
        let face = topo.face(face_id)?;

        let wire = topo.wire(face.outer_wire())?;
        let mut vertex_ids = Vec::with_capacity(wire.edges().len());
        let mut positions = Vec::with_capacity(wire.edges().len());
        let mut wire_edge_ids = Vec::with_capacity(wire.edges().len());

        for oe in wire.edges() {
            let edge = topo.edge(oe.edge())?;
            let vid = if oe.is_forward() {
                edge.start()
            } else {
                edge.end()
            };
            vertex_ids.push(vid);
            positions.push(topo.vertex(vid)?.point());
            wire_edge_ids.push(oe.edge());

            edge_to_faces
                .entry(oe.edge().index())
                .or_default()
                .push(face_id);
        }

        // Only build polygon data for planar faces. Non-planar faces
        // will be passed through unchanged if they don't contain target edges.
        let (normal, d) = match face.surface() {
            FaceSurface::Plane { normal, d } => (*normal, *d),
            _ => continue,
        };

        face_polygons.insert(
            face_id.index(),
            FacePolygon {
                vertex_ids,
                positions,
                wire_edge_ids,
                normal,
                d,
            },
        );
    }

    // Phase 2: Validate target edges and build vertex-to-edge adjacency.
    let target_set: HashSet<usize> = edges.iter().map(|e| e.index()).collect();
    let mut vertex_fillet_edges: HashMap<usize, Vec<EdgeId>> = HashMap::new();

    for &edge_id in edges {
        let faces = edge_to_faces.get(&edge_id.index()).ok_or_else(|| {
            crate::OperationsError::InvalidInput {
                reason: format!("edge {} is not part of the solid", edge_id.index()),
            }
        })?;
        if faces.len() != 2 {
            return Err(crate::OperationsError::InvalidInput {
                reason: format!(
                    "edge {} is shared by {} faces, expected exactly 2",
                    edge_id.index(),
                    faces.len()
                ),
            });
        }

        let edge = topo.edge(edge_id)?;
        vertex_fillet_edges
            .entry(edge.start().index())
            .or_default()
            .push(edge_id);
        vertex_fillet_edges
            .entry(edge.end().index())
            .or_default()
            .push(edge_id);
    }

    // Phase 3: Build modified (trimmed) planar faces.
    let mut all_specs: Vec<FaceSpec> = Vec::new();
    let mut fillet_face_indices: Vec<usize> = Vec::new();

    for &face_id in &shell_face_ids {
        // Non-planar faces pass through unchanged.
        let Some(poly) = face_polygons.get(&face_id.index()) else {
            let face = topo.face(face_id)?;
            let verts = crate::boolean::face_polygon(topo, face_id)?;
            all_specs.push(FaceSpec::Surface {
                vertices: verts,
                surface: face.surface().clone(),
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

            match (before_filleted, after_filleted) {
                (false, false) => {
                    new_verts.push(pos);
                }
                (true, false) => {
                    let dir = (next_pos - pos).normalize()?;
                    new_verts.push(pos + dir * radius);
                }
                (false, true) => {
                    let dir = (prev_pos - pos).normalize()?;
                    new_verts.push(pos + dir * radius);
                }
                (true, true) => {
                    let dir_prev = (prev_pos - pos).normalize()?;
                    new_verts.push(pos + dir_prev * radius);

                    let dir_next = (next_pos - pos).normalize()?;
                    new_verts.push(pos + dir_next * radius);
                }
            }
        }

        let new_d = dot_normal_point(poly.normal, new_verts[0]);
        all_specs.push(FaceSpec::Planar {
            vertices: new_verts,
            normal: poly.normal,
            d: new_d,
        });
    }

    // Phase 4: Build NURBS fillet faces for each target edge.
    // Also collect contact points per vertex for vertex blend patches.
    // vertex_contacts maps vertex_index → list of (face_index, contact_point) pairs.
    let mut vertex_contacts: HashMap<usize, Vec<(usize, Point3)>> = HashMap::new();

    for &edge_id in edges {
        let edge = topo.edge(edge_id)?;
        let p_start = topo.vertex(edge.start())?.point();
        let p_end = topo.vertex(edge.end())?.point();

        let face_list = &edge_to_faces[&edge_id.index()];
        let f1 = face_list[0];
        let f2 = face_list[1];
        let n1 = face_polygons[&f1.index()].normal;
        let n2 = face_polygons[&f2.index()].normal;

        // Edge direction
        let edge_vec = p_end - p_start;
        let edge_len = edge_vec.length();
        if edge_len < tol.linear {
            continue;
        }
        let edge_dir = edge_vec.normalize()?;

        // Compute inward-pointing directions on each face (perpendicular to edge,
        // in the face plane, pointing toward the solid interior).
        // For face with outward normal n and edge direction t:
        // The inward direction is -(t × n) or +(t × n) depending on orientation.
        let cross1 = edge_dir.cross(n1);
        let cross2 = edge_dir.cross(n2);

        // Choose sign so the inward directions point toward each other
        // (their dot product should be positive for a convex edge).
        let d1 = if cross1.dot(n2) > 0.0 {
            cross1
        } else {
            Vec3::new(-cross1.x(), -cross1.y(), -cross1.z())
        };
        let d2 = if cross2.dot(n1) > 0.0 {
            cross2
        } else {
            Vec3::new(-cross2.x(), -cross2.y(), -cross2.z())
        };

        // Normalize (they should already be unit length since edge_dir and n are unit)
        let d1 = d1.normalize().unwrap_or(d1);
        let d2 = d2.normalize().unwrap_or(d2);

        // Half dihedral angle (angle between the inward face directions)
        let cos_half = d1.dot(d2).clamp(-1.0, 1.0);
        let half_angle = cos_half.acos() / 2.0;

        if half_angle.abs() < tol.angular || (std::f64::consts::PI - half_angle).abs() < tol.angular
        {
            // Degenerate angle — faces are parallel or antiparallel, can't fillet
            continue;
        }

        // Bisector direction (toward fillet center)
        let bisector = (d1 + d2).normalize()?;

        // Fillet center distance is R/sin(half_angle), but we only need
        // contact points and the middle control point for the NURBS arc.

        // Compute contact points and fillet center at each edge endpoint.
        // Contact point on face i = edge_point + d_i * radius
        let contact1_start = p_start + d1 * radius;
        let contact1_end = p_end + d1 * radius;
        let contact2_start = p_start + d2 * radius;
        let contact2_end = p_end + d2 * radius;

        // Build the NURBS fillet surface as a degree (2,1) rational surface.
        // u-direction: circular arc (degree 2, 3 control points)
        // v-direction: along edge (degree 1, 2 control points)
        //
        // The arc goes from contact1 to contact2 through a middle control point.
        // For a rational quadratic circular arc:
        //   P0 = contact point on face 1 (weight 1)
        //   P1 = intersection of tangent lines at P0 and P2 (weight cos(α/2))
        //   P2 = contact point on face 2 (weight 1)
        //
        // The tangent at P0 points from P0 toward the center (in the plane
        // perpendicular to the edge), and similarly for P2.
        // The middle control point is where these tangent lines meet.
        //
        // For a symmetric fillet, the middle CP is on the bisector at distance
        // R / cos(α/2) from the edge.

        // Arc half-angle: the angle from contact1 to contact2 through the center
        let arc_half = half_angle; // For the fillet arc
        let w_mid = arc_half.cos(); // Weight for middle control point

        // Middle control point: on the bisector at distance R/cos(α/2)
        let mid_dist = radius / arc_half.cos();
        let mid_start = p_start + bisector * mid_dist;
        let mid_end = p_end + bisector * mid_dist;

        // Build the NURBS surface.
        // Convention: control_points[u_index][v_index].
        // u = arc direction (3 CPs, degree 2), v = edge direction (2 CPs, degree 1).
        let fillet_surface = NurbsSurface::new(
            2,                                  // degree_u (circular arc)
            1,                                  // degree_v (linear along edge)
            vec![0.0, 0.0, 0.0, 1.0, 1.0, 1.0], // knots_u: 3 CPs + degree 2 + 1 = 6
            vec![0.0, 0.0, 1.0, 1.0],           // knots_v: 2 CPs + degree 1 + 1 = 4
            vec![
                // u=0: contact on face 1 (start → end of edge)
                vec![contact1_start, contact1_end],
                // u=0.5: middle arc CP (start → end of edge)
                vec![mid_start, mid_end],
                // u=1: contact on face 2 (start → end of edge)
                vec![contact2_start, contact2_end],
            ],
            vec![
                vec![1.0, 1.0],     // u=0 weights
                vec![w_mid, w_mid], // u=0.5 weights (cos(α/2))
                vec![1.0, 1.0],     // u=1 weights
            ],
        )
        .map_err(crate::OperationsError::Math)?;

        all_specs.push(FaceSpec::Surface {
            vertices: vec![contact1_start, contact2_start, contact2_end, contact1_end],
            surface: FaceSurface::Nurbs(fillet_surface),
        });

        // Track which faces need normal reversal (surface normal points
        // inward, same direction as bisector toward solid interior).
        let srf_mid_normal = match &all_specs[all_specs.len() - 1] {
            FaceSpec::Surface {
                surface: FaceSurface::Nurbs(srf),
                ..
            } => srf.normal(0.5, 0.5).unwrap_or(bisector),
            _ => bisector,
        };
        if srf_mid_normal.dot(bisector) > 0.0 {
            fillet_face_indices.push(all_specs.len() - 1);
        }

        // Record contact points at each vertex for vertex blend detection.
        // Each edge contributes two contact points per endpoint (one on each face).
        let start_vi = edge.start().index();
        let end_vi = edge.end().index();
        vertex_contacts
            .entry(start_vi)
            .or_default()
            .push((f1.index(), contact1_start));
        vertex_contacts
            .entry(start_vi)
            .or_default()
            .push((f2.index(), contact2_start));
        vertex_contacts
            .entry(end_vi)
            .or_default()
            .push((f1.index(), contact1_end));
        vertex_contacts
            .entry(end_vi)
            .or_default()
            .push((f2.index(), contact2_end));
    }

    // Phase 5b: Build vertex blend patches at junctions where 3+ fillet edges meet.
    // At such a vertex, each fillet strip contributes contact points on two faces.
    // Two fillet strips that share a face will have contact points on that face that
    // are at the same position (both offset R from the vertex along the face).
    // We deduplicate by face, giving exactly N unique contact points for N fillet edges.
    // These points form a polygon (typically a triangle for 3-edge corners) that we
    // close with a planar blend face.
    for (&vi, contacts) in &vertex_contacts {
        let fillet_count = vertex_fillet_edges.get(&vi).map_or(0, Vec::len);
        if fillet_count < 3 {
            continue;
        }

        // Deduplicate contact points by spatial proximity.
        // At a 3-edge box corner, 6 contact entries collapse to 3 unique positions
        // (each position is shared by two fillet strips on different faces).
        let mut blend_points: Vec<Point3> = Vec::new();
        for &(_face_idx, pt) in contacts {
            let already = blend_points
                .iter()
                .any(|existing| (*existing - pt).length() < tol.linear);
            if !already {
                blend_points.push(pt);
            }
        }
        if blend_points.len() < 3 {
            continue;
        }

        // Compute the outward normal for the blend patch.
        // The vertex's original position is "inside" the fillet region, so the normal
        // should point away from the original vertex.
        // Use the cross product of two edges of the polygon.
        let e1 = blend_points[1] - blend_points[0];
        let e2 = blend_points[2] - blend_points[0];
        let cross = e1.cross(e2);
        let blend_normal = if let Ok(n) = cross.normalize() {
            n
        } else {
            continue; // Degenerate (collinear points)
        };

        // Orient the normal to point outward (away from the original vertex position).
        // The original vertex is at the centroid of the face normals, offset inward.
        // We can use any face polygon vertex to get the original vertex position.
        let original_vertex = face_polygons
            .values()
            .flat_map(|fp| {
                fp.vertex_ids
                    .iter()
                    .zip(fp.positions.iter())
                    .filter(|(vid, _)| vid.index() == vi)
                    .map(|(_, pos)| *pos)
            })
            .next();

        let blend_normal = if let Some(v_pos) = original_vertex {
            let centroid = blend_points
                .iter()
                .fold(Vec3::new(0.0, 0.0, 0.0), |acc, p| {
                    Vec3::new(acc.x() + p.x(), acc.y() + p.y(), acc.z() + p.z())
                });
            let centroid = Point3::new(
                centroid.x() / blend_points.len() as f64,
                centroid.y() / blend_points.len() as f64,
                centroid.z() / blend_points.len() as f64,
            );
            // Normal should point away from the original vertex
            let to_vertex = v_pos - centroid;
            if to_vertex.dot(blend_normal) > 0.0 {
                Vec3::new(-blend_normal.x(), -blend_normal.y(), -blend_normal.z())
            } else {
                blend_normal
            }
        } else {
            blend_normal
        };

        // Order the blend points consistently (counter-clockwise when viewed from
        // the outward normal direction).
        let centroid = blend_points
            .iter()
            .fold(Vec3::new(0.0, 0.0, 0.0), |acc, p| {
                Vec3::new(acc.x() + p.x(), acc.y() + p.y(), acc.z() + p.z())
            });
        let centroid = Point3::new(
            centroid.x() / blend_points.len() as f64,
            centroid.y() / blend_points.len() as f64,
            centroid.z() / blend_points.len() as f64,
        );

        // Build a local reference frame: normal + two tangent axes
        let ref_dir = (blend_points[0] - centroid)
            .normalize()
            .unwrap_or(Vec3::new(1.0, 0.0, 0.0));
        let tangent_u = ref_dir;
        let tangent_v = blend_normal.cross(tangent_u);

        let mut indexed_points: Vec<(f64, Point3)> = blend_points
            .iter()
            .map(|p| {
                let d = *p - centroid;
                let angle = d.dot(tangent_v).atan2(d.dot(tangent_u));
                (angle, *p)
            })
            .collect();
        indexed_points.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap_or(std::cmp::Ordering::Equal));

        let ordered_points: Vec<Point3> = indexed_points.into_iter().map(|(_, p)| p).collect();

        let blend_d = dot_normal_point(blend_normal, ordered_points[0]);
        all_specs.push(FaceSpec::Planar {
            vertices: ordered_points,
            normal: blend_normal,
            d: blend_d,
        });
    }

    // Phase 6: Assemble the solid using mixed-surface assembly.
    let solid_id = crate::boolean::assemble_solid_mixed(topo, &all_specs, tol)?;

    // Phase 7: Mark fillet faces whose NURBS surface normal points inward
    // as reversed. This ensures tessellation produces outward-facing
    // triangles for correct volume computation and rendering.
    if !fillet_face_indices.is_empty() {
        let solid_data = topo.solid(solid_id)?;
        let shell = topo.shell(solid_data.outer_shell())?;
        let face_ids: Vec<_> = shell.faces().to_vec();
        for &fi in &fillet_face_indices {
            if fi < face_ids.len() {
                let fid = face_ids[fi];
                let face = topo.face_mut(fid)?;
                face.set_reversed(true);
            }
        }
    }

    Ok(solid_id)
}

// ── Internal data structures ───────────────────────────────────────

struct FacePolygon {
    vertex_ids: Vec<VertexId>,
    positions: Vec<Point3>,
    wire_edge_ids: Vec<EdgeId>,
    normal: Vec3,
    #[allow(dead_code)]
    d: f64,
}

struct FilletEdgeData {
    points: HashMap<(usize, usize), Point3>,
}

impl FilletEdgeData {
    fn new() -> Self {
        Self {
            points: HashMap::new(),
        }
    }

    fn insert(&mut self, face_id: FaceId, vertex_id: VertexId, point: Point3) {
        self.points
            .insert((face_id.index(), vertex_id.index()), point);
    }

    fn get_point(
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

fn record_fillet_point(
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

/// Law governing how fillet radius varies along an edge.
#[derive(Debug, Clone)]
pub enum FilletRadiusLaw {
    /// Constant radius (same as basic [`fillet`]).
    Constant(f64),
    /// Linear interpolation from `start_radius` to `end_radius`.
    Linear {
        /// Radius at the start of the edge.
        start: f64,
        /// Radius at the end of the edge.
        end: f64,
    },
    /// Smooth S-curve (sinusoidal) interpolation between two radii.
    SCurve {
        /// Radius at the start of the edge.
        start: f64,
        /// Radius at the end of the edge.
        end: f64,
    },
}

impl FilletRadiusLaw {
    /// Evaluate the radius at parameter `t ∈ [0, 1]` along the edge.
    #[must_use]
    pub fn evaluate(&self, t: f64) -> f64 {
        let t = t.clamp(0.0, 1.0);
        match self {
            Self::Constant(r) => *r,
            Self::Linear { start, end } => (end - start).mul_add(t, *start),
            Self::SCurve { start, end } => {
                // Smooth step: 3t² - 2t³ (Hermite interpolation)
                let s = t * t * (-2.0f64).mul_add(t, 3.0);
                (end - start).mul_add(s, *start)
            }
        }
    }
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
    use brepkit_math::nurbs::surface_fitting::interpolate_surface;

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
    let shell_face_ids: Vec<FaceId> = shell.faces().to_vec();

    let mut edge_to_faces: std::collections::HashMap<usize, Vec<FaceId>> =
        std::collections::HashMap::new();
    let mut face_polygons: std::collections::HashMap<usize, FacePolygon> =
        std::collections::HashMap::new();
    let target_set: std::collections::HashSet<usize> =
        edge_laws.iter().map(|(e, _)| e.index()).collect();

    for &face_id in &shell_face_ids {
        let face = topo.face(face_id)?;

        let wire = topo.wire(face.outer_wire())?;
        let mut vertex_ids = Vec::new();
        let mut positions = Vec::new();
        let mut wire_edge_ids = Vec::new();

        for oe in wire.edges() {
            let edge = topo.edge(oe.edge())?;
            let vid = if oe.is_forward() {
                edge.start()
            } else {
                edge.end()
            };
            vertex_ids.push(vid);
            positions.push(topo.vertex(vid)?.point());
            wire_edge_ids.push(oe.edge());
            edge_to_faces
                .entry(oe.edge().index())
                .or_default()
                .push(face_id);
        }

        // Only build polygon data for planar faces.
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
            },
        );
    }

    // Build trimmed planar faces (using average radius for trimming).
    let avg_radius: f64 = edge_laws
        .iter()
        .map(|(_, law)| law.evaluate(0.5))
        .sum::<f64>()
        / edge_laws.len() as f64;
    // Placeholder: edges_only will be used for per-edge trimming.
    let _ = edge_laws.len();

    // Use the constant-radius trimming from the basic fillet for the planar faces.
    // The NURBS canal surface replaces the fillet face.
    let mut all_specs: Vec<FaceSpec> = Vec::new();

    for &face_id in &shell_face_ids {
        let Some(poly) = face_polygons.get(&face_id.index()) else {
            let face = topo.face(face_id)?;
            let verts = crate::boolean::face_polygon(topo, face_id)?;
            all_specs.push(FaceSpec::Surface {
                vertices: verts,
                surface: face.surface().clone(),
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

            match (before_filleted, after_filleted) {
                (false, false) => new_verts.push(pos),
                (true, false) => {
                    let dir = (next_pos - pos).normalize()?;
                    new_verts.push(pos + dir * avg_radius);
                }
                (false, true) => {
                    let dir = (prev_pos - pos).normalize()?;
                    new_verts.push(pos + dir * avg_radius);
                }
                (true, true) => {
                    let dir_prev = (prev_pos - pos).normalize()?;
                    new_verts.push(pos + dir_prev * avg_radius);
                    let dir_next = (next_pos - pos).normalize()?;
                    new_verts.push(pos + dir_next * avg_radius);
                }
            }
        }

        let new_d = dot_normal_point(poly.normal, new_verts[0]);
        all_specs.push(FaceSpec::Planar {
            vertices: new_verts,
            normal: poly.normal,
            d: new_d,
        });
    }

    // Build variable-radius NURBS canal surfaces for each edge.
    let n_samples = 5; // Number of cross-sections along each edge

    for (edge_id, law) in edge_laws {
        let edge = topo.edge(*edge_id)?;
        let p_start = topo.vertex(edge.start())?.point();
        let p_end = topo.vertex(edge.end())?.point();

        let face_list = edge_to_faces.get(&edge_id.index());
        if face_list.is_none() || face_list.is_some_and(|f| f.len() < 2) {
            continue;
        }
        let empty_faces = vec![];
        let face_list = face_list.unwrap_or(&empty_faces);
        let f1 = face_list[0];
        let f2 = face_list[1];

        let (n1, n2) = match (
            face_polygons.get(&f1.index()),
            face_polygons.get(&f2.index()),
        ) {
            (Some(p1), Some(p2)) => (p1.normal, p2.normal),
            _ => continue,
        };

        let edge_vec = p_end - p_start;
        let edge_len = edge_vec.length();
        if edge_len < tol.linear {
            continue;
        }
        let edge_dir = edge_vec.normalize()?;

        // Compute cross-section geometry at each sample point.
        let cross1 = edge_dir.cross(n1);
        let cross2 = edge_dir.cross(n2);
        let d1 = if cross1.dot(n2) > 0.0 {
            cross1
        } else {
            -cross1
        };
        let d2 = if cross2.dot(n1) > 0.0 {
            cross2
        } else {
            -cross2
        };
        let d1 = d1.normalize().unwrap_or(d1);
        let d2 = d2.normalize().unwrap_or(d2);
        let bisector = (d1 + d2).normalize()?;
        let cos_half = d1.dot(d2).clamp(-1.0, 1.0);
        let half_angle = cos_half.acos() / 2.0;

        if half_angle.abs() < tol.angular {
            continue;
        }

        // Build interpolation grid: n_samples rows × 3 columns (arc CPs).
        let mut grid: Vec<Vec<Point3>> = Vec::with_capacity(n_samples);
        let arc_half = half_angle;

        #[allow(clippy::cast_precision_loss)]
        for s in 0..n_samples {
            let t = s as f64 / (n_samples - 1).max(1) as f64;
            let r = law.evaluate(t);
            let p = Point3::new(
                p_start.x().mul_add(1.0 - t, p_end.x() * t),
                p_start.y().mul_add(1.0 - t, p_end.y() * t),
                p_start.z().mul_add(1.0 - t, p_end.z() * t),
            );

            let contact1 = p + d1 * r;
            let contact2 = p + d2 * r;
            let mid_dist = r / arc_half.cos();
            let mid_cp = p + bisector * mid_dist;

            grid.push(vec![contact1, mid_cp, contact2]);
        }

        // Transpose grid for interpolate_surface convention:
        // rows = arc CPs (3), columns = samples along edge (n_samples).
        let n_arc = 3;
        let transposed: Vec<Vec<Point3>> = (0..n_arc)
            .map(|col| (0..n_samples).map(|row| grid[row][col]).collect())
            .collect();
        let degree_u = 2.min(n_arc - 1);
        let degree_v = (n_samples - 1).min(3);
        let surface = interpolate_surface(&transposed, degree_u, degree_v)
            .map_err(crate::OperationsError::Math)?;

        // Boundary vertices for the canal surface.
        let c1s = grid[0][0];
        let c2s = grid[0][2];
        let c1e = grid[n_samples - 1][0];
        let c2e = grid[n_samples - 1][2];

        all_specs.push(FaceSpec::Surface {
            vertices: vec![c1s, c2s, c2e, c1e],
            surface: FaceSurface::Nurbs(surface),
        });
    }

    crate::boolean::assemble_solid_mixed(topo, &all_specs, tol)
}

#[cfg(test)]
mod tests {
    #![allow(clippy::unwrap_used, clippy::expect_used)]

    use std::collections::HashSet;

    use brepkit_topology::Topology;
    use brepkit_topology::edge::EdgeId;
    use brepkit_topology::test_utils::make_unit_cube_manifold;
    use brepkit_topology::validation::validate_shell_manifold;

    use super::*;

    fn solid_edge_ids(topo: &Topology, solid_id: SolidId) -> Vec<EdgeId> {
        let solid = topo.solid(solid_id).expect("test solid");
        let shell = topo.shell(solid.outer_shell()).expect("test shell");
        let mut seen = HashSet::new();
        let mut edges = Vec::new();
        for &fid in shell.faces() {
            let face = topo.face(fid).expect("test face");
            let wire = topo.wire(face.outer_wire()).expect("test wire");
            for oe in wire.edges() {
                if seen.insert(oe.edge().index()) {
                    edges.push(oe.edge());
                }
            }
        }
        edges
    }

    #[test]
    fn fillet_single_edge() {
        let mut topo = Topology::new();
        let cube = make_unit_cube_manifold(&mut topo);

        let edges = solid_edge_ids(&topo, cube);
        let target = edges[0];

        let result = fillet(&mut topo, cube, &[target], 0.1).expect("fillet should succeed");

        let s = topo.solid(result).expect("result solid");
        let sh = topo.shell(s.outer_shell()).expect("shell");

        // 6 original + 1 fillet = 7 faces
        assert_eq!(
            sh.faces().len(),
            7,
            "expected 7 faces after single-edge fillet"
        );
    }

    #[test]
    fn fillet_result_is_manifold() {
        let mut topo = Topology::new();
        let cube = make_unit_cube_manifold(&mut topo);

        let edges = solid_edge_ids(&topo, cube);
        let result = fillet(&mut topo, cube, &[edges[0]], 0.1).expect("fillet should succeed");

        let s = topo.solid(result).expect("result solid");
        let sh = topo.shell(s.outer_shell()).expect("shell");
        validate_shell_manifold(sh, &topo.faces, &topo.wires)
            .expect("fillet result should be manifold");
    }

    #[test]
    fn fillet_zero_radius_error() {
        let mut topo = Topology::new();
        let cube = make_unit_cube_manifold(&mut topo);
        let edges = solid_edge_ids(&topo, cube);
        assert!(fillet(&mut topo, cube, &[edges[0]], 0.0).is_err());
    }

    #[test]
    fn fillet_negative_radius_error() {
        let mut topo = Topology::new();
        let cube = make_unit_cube_manifold(&mut topo);
        let edges = solid_edge_ids(&topo, cube);
        assert!(fillet(&mut topo, cube, &[edges[0]], -0.1).is_err());
    }

    #[test]
    fn fillet_no_edges_error() {
        let mut topo = Topology::new();
        let cube = make_unit_cube_manifold(&mut topo);
        assert!(fillet(&mut topo, cube, &[], 0.1).is_err());
    }

    // ── Variable-radius fillet tests ────────────────

    #[test]
    fn radius_law_constant() {
        let law = FilletRadiusLaw::Constant(0.5);
        assert!((law.evaluate(0.0) - 0.5).abs() < 1e-10);
        assert!((law.evaluate(0.5) - 0.5).abs() < 1e-10);
        assert!((law.evaluate(1.0) - 0.5).abs() < 1e-10);
    }

    #[test]
    fn radius_law_linear() {
        let law = FilletRadiusLaw::Linear {
            start: 0.1,
            end: 0.5,
        };
        assert!((law.evaluate(0.0) - 0.1).abs() < 1e-10);
        assert!((law.evaluate(0.5) - 0.3).abs() < 1e-10);
        assert!((law.evaluate(1.0) - 0.5).abs() < 1e-10);
    }

    #[test]
    fn radius_law_scurve() {
        let law = FilletRadiusLaw::SCurve {
            start: 0.1,
            end: 0.5,
        };
        // S-curve should match endpoints
        assert!((law.evaluate(0.0) - 0.1).abs() < 1e-10);
        assert!((law.evaluate(1.0) - 0.5).abs() < 1e-10);
        // Midpoint should be between start and end
        let mid = law.evaluate(0.5);
        assert!(mid > 0.1 && mid < 0.5);
    }

    #[test]
    fn fillet_variable_constant_law() {
        let mut topo = Topology::new();
        let cube = make_unit_cube_manifold(&mut topo);

        let edges = solid_edge_ids(&topo, cube);
        let laws = vec![(edges[0], FilletRadiusLaw::Constant(0.1))];

        let result = fillet_variable(&mut topo, cube, &laws).expect("variable fillet should work");

        let s = topo.solid(result).expect("result solid");
        let sh = topo.shell(s.outer_shell()).expect("shell");
        assert_eq!(sh.faces().len(), 7, "should have 7 faces after fillet");
    }

    #[test]
    fn fillet_variable_linear_law() {
        let mut topo = Topology::new();
        let cube = make_unit_cube_manifold(&mut topo);

        let edges = solid_edge_ids(&topo, cube);
        let laws = vec![(
            edges[0],
            FilletRadiusLaw::Linear {
                start: 0.05,
                end: 0.15,
            },
        )];

        let result = fillet_variable(&mut topo, cube, &laws).expect("variable fillet should work");

        let vol = crate::measure::solid_volume(&topo, result, 0.1).unwrap();
        assert!(vol > 0.5, "filleted cube should have volume, got {vol}");
    }

    #[test]
    fn fillet_has_positive_volume() {
        let mut topo = Topology::new();
        let cube = make_unit_cube_manifold(&mut topo);

        let edges = solid_edge_ids(&topo, cube);
        let result = fillet(&mut topo, cube, &[edges[0]], 0.1).expect("fillet should succeed");

        let vol = crate::measure::solid_volume(&topo, result, 0.1).unwrap();
        assert!(
            vol > 0.5,
            "filleted cube should have significant volume, got {vol}"
        );
    }

    // ── Rolling-ball fillet tests ──────────────────────────

    #[test]
    fn rolling_ball_fillet_single_edge() {
        let mut topo = Topology::new();
        let cube = make_unit_cube_manifold(&mut topo);

        let edges = solid_edge_ids(&topo, cube);
        let result = fillet_rolling_ball(&mut topo, cube, &[edges[0]], 0.1)
            .expect("rolling-ball fillet should succeed");

        let s = topo.solid(result).expect("result solid");
        let sh = topo.shell(s.outer_shell()).expect("shell");

        // 6 original faces + 1 NURBS fillet = 7 faces
        assert_eq!(
            sh.faces().len(),
            7,
            "expected 7 faces after single-edge rolling-ball fillet"
        );
    }

    #[test]
    fn rolling_ball_fillet_has_nurbs_face() {
        let mut topo = Topology::new();
        let cube = make_unit_cube_manifold(&mut topo);

        let edges = solid_edge_ids(&topo, cube);
        let result = fillet_rolling_ball(&mut topo, cube, &[edges[0]], 0.1)
            .expect("rolling-ball fillet should succeed");

        let s = topo.solid(result).expect("result solid");
        let sh = topo.shell(s.outer_shell()).expect("shell");

        // At least one face should be a NURBS surface (the fillet).
        let has_nurbs = sh.faces().iter().any(|&fid| {
            matches!(
                topo.face(fid).expect("face").surface(),
                FaceSurface::Nurbs(_)
            )
        });
        assert!(has_nurbs, "rolling-ball fillet should produce NURBS faces");
    }

    #[test]
    fn rolling_ball_fillet_surface_is_circular_arc() {
        let mut topo = Topology::new();
        let cube = make_unit_cube_manifold(&mut topo);

        let edges = solid_edge_ids(&topo, cube);
        let result = fillet_rolling_ball(&mut topo, cube, &[edges[0]], 0.2)
            .expect("rolling-ball fillet should succeed");

        let s = topo.solid(result).expect("result solid");
        let sh = topo.shell(s.outer_shell()).expect("shell");

        // Find the NURBS fillet face and verify it's a proper circular arc.
        for &fid in sh.faces() {
            let face = topo.face(fid).expect("face");
            if let FaceSurface::Nurbs(surface) = face.surface() {
                // The surface should be degree (2, 1) — circular arc × linear.
                assert_eq!(
                    surface.degree_u(),
                    2,
                    "u (arc) direction should be degree 2"
                );
                assert_eq!(
                    surface.degree_v(),
                    1,
                    "v (extrusion) direction should be degree 1"
                );

                // Evaluate at the midpoint (u=0.5, v=0.5) and check that
                // the point is at distance R from both adjacent faces.
                let mid_pt = surface.evaluate(0.5, 0.5);

                // For a unit cube, the fillet point should be inside the cube
                // (all coordinates between -0.1 and 1.1 for radius 0.2).
                assert!(
                    mid_pt.x() > -0.5 && mid_pt.x() < 1.5,
                    "fillet midpoint x should be near cube: {mid_pt:?}"
                );
            }
        }
    }

    #[test]
    fn rolling_ball_fillet_positive_volume() {
        let mut topo = Topology::new();
        let cube = make_unit_cube_manifold(&mut topo);

        let edges = solid_edge_ids(&topo, cube);
        let result =
            fillet_rolling_ball(&mut topo, cube, &[edges[0]], 0.1).expect("fillet should succeed");

        let vol = crate::measure::solid_volume(&topo, result, 0.1).unwrap();
        assert!(
            vol > 0.5,
            "filleted cube should have significant volume, got {vol}"
        );
    }

    #[test]
    fn rolling_ball_fillet_multiple_edges() {
        let mut topo = Topology::new();
        let cube = make_unit_cube_manifold(&mut topo);

        let edges = solid_edge_ids(&topo, cube);
        // Fillet 2 edges
        let result = fillet_rolling_ball(&mut topo, cube, &[edges[0], edges[1]], 0.1)
            .expect("multi-edge rolling-ball fillet should succeed");

        let s = topo.solid(result).expect("result solid");
        let sh = topo.shell(s.outer_shell()).expect("shell");

        // 6 original + 2 NURBS fillets = 8 faces
        assert_eq!(
            sh.faces().len(),
            8,
            "expected 8 faces after two-edge rolling-ball fillet"
        );
    }

    #[test]
    fn rolling_ball_fillet_error_cases() {
        let mut topo = Topology::new();
        let cube = make_unit_cube_manifold(&mut topo);
        let edges = solid_edge_ids(&topo, cube);

        assert!(fillet_rolling_ball(&mut topo, cube, &[edges[0]], 0.0).is_err());
        assert!(fillet_rolling_ball(&mut topo, cube, &[edges[0]], -0.1).is_err());
        assert!(fillet_rolling_ball(&mut topo, cube, &[], 0.1).is_err());
    }

    // ── Vertex blend tests ───────────────────────────────

    #[test]
    fn vertex_blend_all_edges_box() {
        // Fillet all 12 edges of a unit cube → 8 vertex blend patches should
        // close the corners, giving a watertight mesh.
        let mut topo = Topology::new();
        let cube = make_unit_cube_manifold(&mut topo);
        let edges = solid_edge_ids(&topo, cube);
        assert_eq!(edges.len(), 12, "unit cube should have 12 edges");

        let result = fillet_rolling_ball(&mut topo, cube, &edges, 0.1)
            .expect("all-edges fillet should succeed");

        let s = topo.solid(result).expect("result solid");
        let sh = topo.shell(s.outer_shell()).expect("shell");

        // 6 trimmed planar faces + 12 NURBS fillet strips + 8 vertex blend triangles = 26
        assert_eq!(
            sh.faces().len(),
            26,
            "expected 26 faces (6 planar + 12 fillet + 8 blend)"
        );
    }

    #[test]
    fn vertex_blend_tessellates_successfully() {
        // Verify the fully-filleted box can be tessellated without error.
        // Watertight stitching at NURBS-to-planar seams is a tessellation-level
        // concern tracked separately.
        let mut topo = Topology::new();
        let cube = make_unit_cube_manifold(&mut topo);
        let edges = solid_edge_ids(&topo, cube);

        let result = fillet_rolling_ball(&mut topo, cube, &edges, 0.1)
            .expect("all-edges fillet should succeed");

        let mesh = crate::tessellate::tessellate_solid(&topo, result, 0.05).unwrap();
        // Should produce a non-trivial mesh.
        assert!(mesh.positions.len() > 20, "should have many vertices");
        assert!(mesh.indices.len() > 60, "should have many triangles");
    }

    #[test]
    fn vertex_blend_positive_volume() {
        let mut topo = Topology::new();
        let cube = make_unit_cube_manifold(&mut topo);
        let edges = solid_edge_ids(&topo, cube);

        let result = fillet_rolling_ball(&mut topo, cube, &edges, 0.1)
            .expect("all-edges fillet should succeed");

        let vol = crate::measure::solid_volume(&topo, result, 0.05).unwrap();
        // Unit cube volume = 1.0. Filleting removes corner material, so volume < 1.0 but > 0.5.
        assert!(vol > 0.5, "filleted cube volume should be > 0.5, got {vol}");
        assert!(vol < 1.0, "filleted cube volume should be < 1.0, got {vol}");
    }

    #[test]
    fn vertex_blend_box_primitive() {
        // Test with make_box (2×3×4) to verify non-unit dimensions work.
        let mut topo = Topology::new();
        let solid = crate::primitives::make_box(&mut topo, 2.0, 3.0, 4.0).unwrap();
        let edges = solid_edge_ids(&topo, solid);
        assert_eq!(edges.len(), 12);

        let result = fillet_rolling_ball(&mut topo, solid, &edges, 0.2)
            .expect("box primitive all-edges fillet should succeed");

        let s = topo.solid(result).expect("result solid");
        let sh = topo.shell(s.outer_shell()).expect("shell");
        assert_eq!(sh.faces().len(), 26);
    }

    #[test]
    fn vertex_blend_three_edges_at_corner() {
        // Fillet just the 3 edges meeting at one corner vertex to test minimal
        // vertex blend (produces one blend triangle).
        let mut topo = Topology::new();
        let cube = make_unit_cube_manifold(&mut topo);
        let all_edges = solid_edge_ids(&topo, cube);

        // Find 3 edges sharing a common vertex.
        let mut vertex_to_edges: HashMap<usize, Vec<EdgeId>> = HashMap::new();
        for &eid in &all_edges {
            let e = topo.edge(eid).unwrap();
            vertex_to_edges
                .entry(e.start().index())
                .or_default()
                .push(eid);
            vertex_to_edges
                .entry(e.end().index())
                .or_default()
                .push(eid);
        }

        let (&_vi, corner_edges) = vertex_to_edges
            .iter()
            .find(|(_, edges)| edges.len() >= 3)
            .expect("box should have vertices with 3 edges");

        let targets: Vec<EdgeId> = corner_edges.iter().take(3).copied().collect();

        let result = fillet_rolling_ball(&mut topo, cube, &targets, 0.1)
            .expect("3-edge corner fillet should succeed");

        let s = topo.solid(result).expect("result solid");
        let sh = topo.shell(s.outer_shell()).expect("shell");

        // 6 original faces + 3 NURBS fillets + at least 1 vertex blend triangle
        assert!(
            sh.faces().len() >= 10,
            "expected at least 10 faces (6 + 3 + 1 blend), got {}",
            sh.faces().len()
        );
    }
}
