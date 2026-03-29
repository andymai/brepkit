//! Rolling-ball fillet algorithm producing G1-continuous NURBS blend surfaces.

use std::collections::{HashMap, HashSet};

use brepkit_math::nurbs::surface::NurbsSurface;
use brepkit_math::nurbs::surface_fitting::interpolate_surface;
use brepkit_math::tolerance::Tolerance;
use brepkit_math::vec::{Point3, Vec3};
use brepkit_topology::Topology;
use brepkit_topology::edge::EdgeId;
use brepkit_topology::face::{FaceId, FaceSurface};
use brepkit_topology::solid::SolidId;

use crate::boolean::FaceSpec;
use crate::dot_normal_point;

use super::g1_chain::expand_g1_chain;
use super::geometry::{
    edge_v_samples, face_surface_normal_at, sample_edge_point, sample_edge_tangent,
};
use super::helpers::{FacePolygon, extract_inner_wire_positions};

/// Fillet one or more edges of a solid using the rolling-ball algorithm.
///
/// Produces true NURBS cylindrical fillet surfaces with G1 tangent
/// continuity, replacing the flat-quad approximation of [`super::fillet`].
///
/// **G1 chain propagation**: the edge set is automatically expanded to
/// include all G1-continuous neighbors that share the same face pair
/// (< 10 degree tangent deviation).  This ensures that selecting one edge
/// from a smooth chain (e.g. a rounded-rectangle profile) fillets the
/// entire chain.
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
/// - Adjacent fillet strips overlap (on planar or curved faces)
/// - Fillet radius exceeds surface curvature of an adjacent face
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
    let mut face_surfaces: HashMap<usize, FaceSurface> = HashMap::new();

    for &face_id in &shell_face_ids {
        let face = topo.face(face_id)?;
        face_surfaces.insert(face_id.index(), face.surface().clone());

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

        // Inner wire edges also contribute to adjacency.
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

        // Build polygon data for planar faces (used for Phase 3 trimming).
        // Non-planar faces are stored in face_surfaces and passed through
        // untrimmed — their fillet geometry is still computed in Phase 4.
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
                inner_wires: face_inner_wires,
            },
        );
    }

    // Precompute edge → polygon entries for O(|filtered_edges|) Phase 2d lookup
    // instead of O(|filtered_edges| × |planar_faces|) nested iteration.
    let mut edge_to_poly_pos: HashMap<usize, Vec<(usize, usize)>> = HashMap::new();
    for (&face_key, poly) in &face_polygons {
        for (i, eid) in poly.wire_edge_ids.iter().enumerate() {
            edge_to_poly_pos
                .entry(eid.index())
                .or_default()
                .push((face_key, i));
        }
    }

    // Phase 2: Filter to manifold edges and build vertex-to-edge adjacency.
    let user_edges: Vec<EdgeId> = edges
        .iter()
        .copied()
        .filter(|edge_id| {
            edge_to_faces
                .get(&edge_id.index())
                .is_some_and(|faces| faces.len() == 2)
        })
        .collect();

    if user_edges.is_empty() {
        return Err(crate::OperationsError::InvalidInput {
            reason: "no manifold edges to fillet (all edges are boundary or missing)".into(),
        });
    }

    // Phase 2a: G1 chain propagation — automatically expand the edge set to
    // include all G1-continuous neighbors sharing the same face pair.
    let filtered_edges = expand_g1_chain(topo, solid, &user_edges, tol)?;
    if filtered_edges.len() > user_edges.len() {
        log::info!(
            "G1 chain: expanded {} edges to {} edges",
            user_edges.len(),
            filtered_edges.len()
        );
    }

    let target_set: HashSet<usize> = filtered_edges.iter().map(|e| e.index()).collect();
    let mut vertex_fillet_edges: HashMap<usize, Vec<EdgeId>> = HashMap::new();

    for &edge_id in &filtered_edges {
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

    // Phase 2b: Validate that the fillet radius fits within adjacent face geometry.
    // For each target edge on each adjacent face, the shortest non-target edge
    // from the shared vertices bounds how far the contact point can extend.
    for &edge_id in &filtered_edges {
        let edge = topo.edge(edge_id)?;
        let p_start = topo.vertex(edge.start())?.point();
        let p_end = topo.vertex(edge.end())?.point();

        let Some(face_list) = edge_to_faces.get(&edge_id.index()) else {
            continue;
        };
        for &fid in face_list {
            let poly = match face_polygons.get(&fid.index()) {
                Some(p) => p,
                None => continue,
            };
            // For each vertex of the target edge, find the shortest adjacent
            // non-target edge on this face. The radius must not exceed that length.
            for &edge_pt in &[p_start, p_end] {
                let mut min_adj = f64::MAX;
                for (i, pos) in poly.positions.iter().enumerate() {
                    let next_i = (i + 1) % poly.positions.len();
                    let next_pos = poly.positions[next_i];
                    // Skip the target edge itself
                    if target_set.contains(&poly.wire_edge_ids[i].index()) {
                        continue;
                    }
                    // Only check edges sharing the vertex
                    if (*pos - edge_pt).length() < tol.linear
                        || (next_pos - edge_pt).length() < tol.linear
                    {
                        let edge_len = (next_pos - *pos).length();
                        if edge_len < min_adj {
                            min_adj = edge_len;
                        }
                    }
                }
                if radius > min_adj && min_adj < f64::MAX {
                    return Err(crate::OperationsError::InvalidInput {
                        reason: format!(
                            "fillet radius {radius:.6} exceeds adjacent edge length {min_adj:.6}"
                        ),
                    });
                }
            }
        }
    }

    // Phase 2c: Validate radius against adjacent face curvature (analytic surfaces).
    // The rolling ball rolls on the adjacent face; its radius must not meet or
    // exceed the minimum principal radius of curvature of that surface, or the
    // offset surface degenerates (e.g. a cylinder of radius R offset by R
    // collapses to a line, a sphere offset by its own radius collapses to a point).
    for &edge_id in &filtered_edges {
        let edge = topo.edge(edge_id)?;
        let p_start = topo.vertex(edge.start())?.point();
        let p_end = topo.vertex(edge.end())?.point();

        let Some(face_list) = edge_to_faces.get(&edge_id.index()) else {
            continue;
        };
        for &fid in face_list {
            let Some(surf) = face_surfaces.get(&fid.index()) else {
                continue;
            };
            let min_curvature_r: f64 = match surf {
                // Planar faces have infinite curvature radius — no constraint.
                // NURBS curvature is not yet estimated analytically — skip.
                FaceSurface::Plane { .. } | FaceSurface::Nurbs(_) => continue,
                // Cylinder: principal curvature κ₁ = 1/R, κ₂ = 0 → min radius = R.
                FaceSurface::Cylinder(s) => s.radius(),
                // Sphere: κ₁ = κ₂ = 1/R → min radius = R.
                FaceSurface::Sphere(s) => s.radius(),
                // Torus: κ₁ = 1/r (minor cross-section, always present).
                // On the inner equator, the major curvature = 1/(R−r), which
                // can exceed 1/r for fat tori (R < 2r). Use the tighter bound.
                FaceSurface::Torus(s) => {
                    let inner_r = s.major_radius() - s.minor_radius();
                    if inner_r > tol.linear {
                        s.minor_radius().min(inner_r)
                    } else {
                        s.minor_radius()
                    }
                }
                // Cone: circumferential κ₂ = tan(α)/v at slant distance v from apex,
                // where α = half_angle from the radial plane.
                // → min curvature radius = v_min * cos(α) / sin(α).
                FaceSurface::Cone(s) => {
                    let (_, v0) = s.project_point(p_start);
                    let (_, v1) = s.project_point(p_end);
                    let v_min = v0.min(v1).abs().max(tol.linear);
                    let cos_a = s.half_angle().cos();
                    let sin_a = s.half_angle().sin();
                    if sin_a < tol.linear {
                        // Near-flat cone (half_angle ≈ 0): curvature radius → ∞, no constraint.
                        continue;
                    }
                    v_min * cos_a / sin_a
                }
            };
            if radius >= min_curvature_r {
                return Err(crate::OperationsError::InvalidInput {
                    reason: format!(
                        "fillet radius {radius:.6} meets or exceeds minimum surface \
                         curvature radius {min_curvature_r:.6} of adjacent face"
                    ),
                });
            }
        }
    }

    // Phase 2d: Detect adjacent fillet overlap on planar faces.
    // When two target edges share a vertex on a common planar face, the rolling
    // ball on each edge creates a contact setback along the other edge from that
    // vertex.  If the sum of setbacks from both vertices of a target edge equals
    // or exceeds the polygon edge length, the fillet strips would overlap.
    //
    // setback along edge E from vertex V (where adjacent edge B is also target):
    //   setback = R / tan(θ / 2)
    // where θ is the interior polygon angle at V between E and B.
    //
    // Only applies to planar adjacent faces (face_polygons).  Curved faces are
    // handled by Phase 2c (curvature bound).
    for &edge_id in &filtered_edges {
        let Some(poly_entries) = edge_to_poly_pos.get(&edge_id.index()) else {
            continue;
        };
        for &(face_key, i_e) in poly_entries {
            let poly = &face_polygons[&face_key];
            let n = poly.positions.len();
            let next_i = (i_e + 1) % n;
            let prev_i = (i_e + n - 1) % n;
            let next_next_i = (next_i + 1) % n;

            let e_vec = poly.positions[next_i] - poly.positions[i_e];
            let e_len = e_vec.length();
            if e_len < tol.linear {
                continue;
            }

            // Setback from start vertex if the previous polygon edge is a target.
            let setback_start: f64 = 'start: {
                let prev_target = target_set.contains(&poly.wire_edge_ids[prev_i].index());
                if prev_target {
                    // Interior angle at start: between (E forward) and (prev backward from start).
                    let d_e = e_vec * (1.0 / e_len);
                    let d_prev_raw = poly.positions[prev_i] - poly.positions[i_e];
                    let prev_len = d_prev_raw.length();
                    if prev_len < tol.linear {
                        break 'start 0.0;
                    }
                    let d_prev = d_prev_raw * (1.0 / prev_len);
                    let cos_t = d_e.dot(d_prev).clamp(-1.0, 1.0);
                    let theta = cos_t.acos();
                    let half_tan = (theta / 2.0).tan();
                    if half_tan < tol.linear {
                        break 'start 0.0;
                    }
                    radius / half_tan
                } else {
                    0.0
                }
            };

            // Setback from end vertex if the next polygon edge is also a target.
            let setback_end: f64 = 'end: {
                let next_target = target_set.contains(&poly.wire_edge_ids[next_i].index());
                if next_target {
                    // Interior angle at end: between (E backward from end) and (next forward).
                    let d_e_bwd = poly.positions[i_e] - poly.positions[next_i];
                    let d_next_raw = poly.positions[next_next_i] - poly.positions[next_i];
                    let bwd_len = e_len; // same magnitude as e_len
                    let next_len = d_next_raw.length();
                    if next_len < tol.linear {
                        break 'end 0.0;
                    }
                    let d_e_bwd_n = d_e_bwd * (1.0 / bwd_len);
                    let d_next_n = d_next_raw * (1.0 / next_len);
                    let cos_t = d_e_bwd_n.dot(d_next_n).clamp(-1.0, 1.0);
                    let theta = cos_t.acos();
                    let half_tan = (theta / 2.0).tan();
                    if half_tan < tol.linear {
                        break 'end 0.0;
                    }
                    radius / half_tan
                } else {
                    0.0
                }
            };

            // Only reject when setbacks come from BOTH ends (one non-target end is
            // already bounded by Phase 2b; two target-edge ends need this check).
            if setback_start > 0.0 && setback_end > 0.0 {
                let total = setback_start + setback_end;
                if total >= e_len {
                    return Err(crate::OperationsError::InvalidInput {
                        reason: format!(
                            "adjacent fillet strips overlap: combined setback \
                             ({setback_start:.6} + {setback_end:.6} = {total:.6}) \
                             equals or exceeds edge length {e_len:.6}"
                        ),
                    });
                }
            }
        }
    }

    // Phase 2d-b: Overlap detection for non-planar adjacent faces.
    // For non-planar faces (cylinder, cone, sphere, torus, NURBS) there is no
    // polygon data.  Instead of interior polygon angles we use edge tangent
    // angles at shared vertices to compute setback distances.
    for &edge_id in &filtered_edges {
        let edge = topo.edge(edge_id)?;
        let start_vid = edge.start();
        let end_vid = edge.end();
        let p_start = topo.vertex(start_vid)?.point();
        let p_end = topo.vertex(end_vid)?.point();
        let edge_len = (p_end - p_start).length();
        if edge_len < tol.linear {
            continue;
        }

        let Some(face_list) = edge_to_faces.get(&edge_id.index()) else {
            continue;
        };

        for &fid in face_list {
            // Skip planar faces (already handled by Phase 2d).
            if face_polygons.contains_key(&fid.index()) {
                continue;
            }

            // For this non-planar face, find other target edges sharing vertices
            // with the current edge.
            let face = topo.face(fid)?;
            let wire = topo.wire(face.outer_wire())?;
            let edge_curve = edge.curve().clone();

            let mut setback_start = 0.0_f64;
            let mut setback_end = 0.0_f64;

            for oe in wire.edges() {
                let adj_eid = oe.edge();
                if adj_eid.index() == edge_id.index() {
                    continue; // skip self
                }
                if !target_set.contains(&adj_eid.index()) {
                    continue; // only check other target edges
                }

                let adj_edge = topo.edge(adj_eid)?;
                let adj_start_vid = adj_edge.start();
                let adj_end_vid = adj_edge.end();
                let adj_start = topo.vertex(adj_start_vid)?.point();
                let adj_end = topo.vertex(adj_end_vid)?.point();
                let adj_curve = adj_edge.curve().clone();

                // Check if adjacent edge shares start vertex of current edge.
                let shares_start = adj_start_vid == start_vid || adj_end_vid == start_vid;
                if shares_start {
                    let t1 = sample_edge_tangent(&edge_curve, p_start, p_end, 0.0);
                    let adj_t = if adj_start_vid == start_vid {
                        sample_edge_tangent(&adj_curve, adj_start, adj_end, 0.0)
                    } else {
                        sample_edge_tangent(&adj_curve, adj_start, adj_end, 1.0)
                    };
                    if let (Ok(t1n), Ok(t2n)) = (t1.normalize(), adj_t.normalize()) {
                        let cos_t = t1n.dot(t2n).clamp(-1.0, 1.0);
                        let theta = cos_t.acos();
                        let half_tan = (theta / 2.0).tan();
                        if half_tan > tol.linear {
                            setback_start = setback_start.max(radius / half_tan);
                        }
                    }
                }

                // Check if adjacent edge shares end vertex of current edge.
                let shares_end = adj_start_vid == end_vid || adj_end_vid == end_vid;
                if shares_end {
                    let t1 = sample_edge_tangent(&edge_curve, p_start, p_end, 1.0);
                    let adj_t = if adj_start_vid == end_vid {
                        sample_edge_tangent(&adj_curve, adj_start, adj_end, 0.0)
                    } else {
                        sample_edge_tangent(&adj_curve, adj_start, adj_end, 1.0)
                    };
                    if let (Ok(t1n), Ok(t2n)) = (t1.normalize(), adj_t.normalize()) {
                        let cos_t = t1n.dot(t2n).clamp(-1.0, 1.0);
                        let theta = cos_t.acos();
                        let half_tan = (theta / 2.0).tan();
                        if half_tan > tol.linear {
                            setback_end = setback_end.max(radius / half_tan);
                        }
                    }
                }
            }

            if setback_start > 0.0 && setback_end > 0.0 {
                let total = setback_start + setback_end;
                if total >= edge_len {
                    return Err(crate::OperationsError::InvalidInput {
                        reason: format!(
                            "adjacent fillet strips overlap on curved face: combined setback \
                             ({setback_start:.6} + {setback_end:.6} = {total:.6}) \
                             equals or exceeds edge length {edge_len:.6}"
                        ),
                    });
                }
            }
        }
    }

    // G1 chain detection — moved before the contact pre-pass so that G1
    // junction vertices are known when computing canonical contacts.
    // Detect chains of consecutive fillet edges that share a vertex.
    // When two fillet strips meet at a vertex on the same pair of faces,
    // they should share contact points for G1 tangent continuity.
    let mut vertex_fillet_adjacency: HashMap<usize, Vec<(usize, usize, usize)>> = HashMap::new();
    for &edge_id in &filtered_edges {
        let edge = topo.edge(edge_id)?;
        if let Some(faces) = edge_to_faces.get(&edge_id.index()) {
            if faces.len() >= 2 {
                let f1 = faces[0].index();
                let f2 = faces[1].index();
                let (fa, fb) = if f1 < f2 { (f1, f2) } else { (f2, f1) };
                vertex_fillet_adjacency
                    .entry(edge.start().index())
                    .or_default()
                    .push((edge_id.index(), fa, fb));
                vertex_fillet_adjacency
                    .entry(edge.end().index())
                    .or_default()
                    .push((edge_id.index(), fa, fb));
            }
        }
    }
    let mut g1_chain_vertices: HashSet<usize> = HashSet::new();
    for (vi, adj) in &vertex_fillet_adjacency {
        if adj.len() == 2 && adj[0].1 == adj[1].1 && adj[0].2 == adj[1].2 {
            g1_chain_vertices.insert(*vi);
        }
    }

    // Pre-pass: precompute fillet strip endpoint contacts using Phase 4's
    // cross-product method.  Phase 3's face trimming will look up these exact
    // values instead of recomputing them from polygon neighbour directions.
    // This ensures both phases produce bitwise-identical positions, preventing
    // duplicate vertices (and thus boundary edges) in assemble_solid_mixed.
    //
    // Key: (vertex_index, edge_index, face_index) → contact Point3
    let fillet_contact_map: HashMap<(usize, usize, usize), Point3> = {
        let mut map = HashMap::new();
        // For G1 junctions: keep the first edge's contacts (entry().or_insert).
        for &edge_id in &filtered_edges {
            let edge = topo.edge(edge_id)?;
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

            let (Some(surf1), Some(surf2)) = (
                face_surfaces.get(&f1.index()),
                face_surfaces.get(&f2.index()),
            ) else {
                continue;
            };

            let edge_curve = edge.curve().clone();
            let edge_tan_start = sample_edge_tangent(&edge_curve, p_start, p_end, 0.0);
            if edge_tan_start.length() < tol.linear {
                continue;
            }

            // Compute contacts at start (t=0) and end (t=1).
            for &(t, vid) in &[(0.0_f64, edge.start()), (1.0_f64, edge.end())] {
                let p = sample_edge_point(&edge_curve, p_start, p_end, t);
                let tan = sample_edge_tangent(&edge_curve, p_start, p_end, t);
                let local_dir = match tan.normalize() {
                    Ok(d) => d,
                    Err(_) => continue,
                };

                let ln1 = match face_surface_normal_at(surf1, p) {
                    Some(n) => n,
                    None => continue,
                };
                let ln2 = match face_surface_normal_at(surf2, p) {
                    Some(n) => n,
                    None => continue,
                };

                // Cross-product directions — same sign convention as Phase 4.
                let c1 = local_dir.cross(ln1);
                let c2 = local_dir.cross(ln2);
                let ld1 = if c1.dot(ln2) < 0.0 { c1 } else { -c1 };
                let ld2 = if c2.dot(ln1) < 0.0 { c2 } else { -c2 };
                let ld1 = ld1.normalize().unwrap_or(c1);
                let ld2 = ld2.normalize().unwrap_or(c2);

                let contact1 = p + ld1 * radius;
                let contact2 = p + ld2 * radius;

                // At G1 junctions, keep the first edge's contacts.
                if g1_chain_vertices.contains(&vid.index()) {
                    map.entry((vid.index(), edge_id.index(), f1.index()))
                        .or_insert(contact1);
                    map.entry((vid.index(), edge_id.index(), f2.index()))
                        .or_insert(contact2);
                } else {
                    map.insert((vid.index(), edge_id.index(), f1.index()), contact1);
                    map.insert((vid.index(), edge_id.index(), f2.index()), contact2);
                }
            }
        }
        map
    };
    log::debug!("fillet contact map: {} entries", fillet_contact_map.len());

    // Phase 3: Build modified (trimmed) planar faces.
    let mut all_specs: Vec<FaceSpec> = Vec::new();
    let mut fillet_face_indices: Vec<usize> = Vec::new();

    for &face_id in &shell_face_ids {
        // Non-planar faces: either pass through or trim at fillet contact points.
        let Some(poly) = face_polygons.get(&face_id.index()) else {
            let face = topo.face(face_id)?;
            let surface = face.surface().clone();
            let wire = topo.wire(face.outer_wire())?;

            // Check if this non-planar face has any target edges.
            let has_target = wire
                .edges()
                .iter()
                .any(|oe| target_set.contains(&oe.edge().index()));

            if !has_target {
                // No target edges: pass through unchanged.
                let verts = crate::boolean::face_polygon(topo, face_id)?;
                let np_inner = extract_inner_wire_positions(topo, face)?;
                all_specs.push(FaceSpec::Surface {
                    vertices: verts,
                    surface,
                    reversed: false,
                    inner_wires: np_inner,
                });
                continue;
            }

            // Has target edges: build trimmed boundary by offsetting vertices
            // at fillet contact locations along the face boundary directions.
            // Collect per-edge vertex positions and edge IDs from the wire.
            let wire_edges: Vec<_> = wire.edges().to_vec();
            let n_we = wire_edges.len();
            let mut positions = Vec::with_capacity(n_we);
            let mut wire_edge_ids = Vec::with_capacity(n_we);
            let mut vertex_ids_np = Vec::with_capacity(n_we);

            for oe in &wire_edges {
                let edge_data = topo.edge(oe.edge())?;
                let vid = oe.oriented_start(edge_data);
                vertex_ids_np.push(vid);
                positions.push(topo.vertex(vid)?.point());
                wire_edge_ids.push(oe.edge());
            }

            if n_we < 3 {
                // Degenerate non-planar face: pass through unchanged.
                let verts = crate::boolean::face_polygon(topo, face_id)?;
                let np_inner = extract_inner_wire_positions(topo, face)?;
                all_specs.push(FaceSpec::Surface {
                    vertices: verts,
                    surface,
                    reversed: false,
                    inner_wires: np_inner,
                });
                continue;
            }

            let mut trimmed_verts: Vec<Point3> = Vec::with_capacity(n_we * 2);

            for i in 0..n_we {
                let prev_i = if i == 0 { n_we - 1 } else { i - 1 };
                let next_i = (i + 1) % n_we;

                let before_filleted = target_set.contains(&wire_edge_ids[prev_i].index());
                let after_filleted = target_set.contains(&wire_edge_ids[i].index());
                let at_fillet_endpoint =
                    vertex_fillet_edges.contains_key(&vertex_ids_np[i].index());

                let pos = positions[i];
                let prev_pos = positions[prev_i];
                let next_pos = positions[next_i];

                // For fillet-adjacent vertices, use Phase 4's exact contact
                // to ensure the trimmed boundary matches the fillet strip.
                let vi = vertex_ids_np[i].index();
                let fi = face_id.index();
                match (before_filleted, after_filleted, at_fillet_endpoint) {
                    (false, false, false) => {
                        trimmed_verts.push(pos);
                    }
                    // Side face: vertex is at a fillet endpoint but neither
                    // adjacent edge of this face is the filleted edge.
                    // Use the two unique Phase 4 fillet contacts at this vertex,
                    // paired by proximity to boundary offsets.
                    (false, false, true) => {
                        let mut unique_contacts: Vec<Point3> = Vec::new();
                        for (&(vi_k, _, _), &pt) in &fillet_contact_map {
                            if vi_k == vi {
                                let already = unique_contacts
                                    .iter()
                                    .any(|uc| (*uc - pt).length() < tol.linear);
                                if !already {
                                    unique_contacts.push(pt);
                                }
                            }
                        }

                        if unique_contacts.len() >= 2 {
                            // Pair by proximity: assign closer-to-prev first,
                            // force the other for next (prevents both mapping
                            // to the same contact).
                            let approx_prev = if let Ok(d) = (prev_pos - pos).normalize() {
                                pos + d * radius
                            } else {
                                pos
                            };
                            let d0 = (unique_contacts[0] - approx_prev).length();
                            let d1 = (unique_contacts[1] - approx_prev).length();
                            if d0 <= d1 {
                                trimmed_verts.push(unique_contacts[0]);
                                trimmed_verts.push(unique_contacts[1]);
                            } else {
                                trimmed_verts.push(unique_contacts[1]);
                                trimmed_verts.push(unique_contacts[0]);
                            }
                        } else {
                            // Fallback: original boundary offset computation.
                            if let Ok(dir_prev) = (prev_pos - pos).normalize() {
                                trimmed_verts.push(pos + dir_prev * radius);
                            } else {
                                trimmed_verts.push(pos);
                            }
                            if let Ok(dir_next) = (next_pos - pos).normalize() {
                                trimmed_verts.push(pos + dir_next * radius);
                            } else {
                                trimmed_verts.push(pos);
                            }
                        }
                    }
                    (true, false, _) => {
                        // The "before" edge is filleted — use its specific contact.
                        let ei = wire_edge_ids[prev_i].index();
                        if let Some(&pt) = fillet_contact_map.get(&(vi, ei, fi)) {
                            trimmed_verts.push(pt);
                        } else if let Ok(dir) = (next_pos - pos).normalize() {
                            trimmed_verts.push(pos + dir * radius);
                        } else {
                            trimmed_verts.push(pos);
                        }
                    }
                    (false, true, _) => {
                        // The "after" edge is filleted — use its specific contact.
                        let ei = wire_edge_ids[i].index();
                        if let Some(&pt) = fillet_contact_map.get(&(vi, ei, fi)) {
                            trimmed_verts.push(pt);
                        } else if let Ok(dir) = (prev_pos - pos).normalize() {
                            trimmed_verts.push(pos + dir * radius);
                        } else {
                            trimmed_verts.push(pos);
                        }
                    }
                    (true, true, _) => {
                        // dir_prev (along "before" edge) is perpendicular to
                        // the "after" fillet edge → use the "after" edge's contact.
                        let ei_after = wire_edge_ids[i].index();
                        if let Some(&pt) = fillet_contact_map.get(&(vi, ei_after, fi)) {
                            trimmed_verts.push(pt);
                        } else if let Ok(dir_prev) = (prev_pos - pos).normalize() {
                            trimmed_verts.push(pos + dir_prev * radius);
                        } else {
                            trimmed_verts.push(pos);
                        }
                        // dir_next (along "after" edge) is perpendicular to
                        // the "before" fillet edge → use the "before" edge's contact.
                        let ei_before = wire_edge_ids[prev_i].index();
                        if let Some(&pt) = fillet_contact_map.get(&(vi, ei_before, fi)) {
                            trimmed_verts.push(pt);
                        } else if let Ok(dir_next) = (next_pos - pos).normalize() {
                            trimmed_verts.push(pos + dir_next * radius);
                        } else {
                            trimmed_verts.push(pos);
                        }
                    }
                }
            }

            let np_inner = extract_inner_wire_positions(topo, face)?;
            all_specs.push(FaceSpec::Surface {
                vertices: trimmed_verts,
                surface,
                reversed: false,
                inner_wires: np_inner,
            });
            continue;
        };
        let n = poly.positions.len();

        // Skip polygon trimming for degenerate faces (e.g., disc caps with a
        // single closed circular edge where start==end vertex).
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

            // Check if this vertex sits at the endpoint of a filleted edge
            // (even if neither adjacent edge of THIS face is the filleted edge).
            // This handles "side faces" that share a corner vertex with the
            // filleted edge — they need the corner split into two contact points.
            let at_fillet_endpoint = vertex_fillet_edges.contains_key(&poly.vertex_ids[i].index());

            // For fillet-adjacent vertices, use Phase 4's exact contact.
            let vi = poly.vertex_ids[i].index();
            let fi = face_id.index();
            match (before_filleted, after_filleted, at_fillet_endpoint) {
                (false, false, false) => {
                    new_verts.push(pos);
                }
                // Side face: use the two unique Phase 4 fillet contacts,
                // paired by proximity to boundary offsets.
                (false, false, true) => {
                    let mut unique_contacts: Vec<Point3> = Vec::new();
                    for (&(vi_k, _, _), &pt) in &fillet_contact_map {
                        if vi_k == vi {
                            let already = unique_contacts
                                .iter()
                                .any(|uc| (*uc - pt).length() < tol.linear);
                            if !already {
                                unique_contacts.push(pt);
                            }
                        }
                    }

                    if unique_contacts.len() >= 2 {
                        let dir_prev = (prev_pos - pos).normalize()?;
                        let approx_prev = pos + dir_prev * radius;
                        let d0 = (unique_contacts[0] - approx_prev).length();
                        let d1 = (unique_contacts[1] - approx_prev).length();
                        if d0 <= d1 {
                            new_verts.push(unique_contacts[0]);
                            new_verts.push(unique_contacts[1]);
                        } else {
                            new_verts.push(unique_contacts[1]);
                            new_verts.push(unique_contacts[0]);
                        }
                    } else {
                        let dir_prev = (prev_pos - pos).normalize()?;
                        new_verts.push(pos + dir_prev * radius);
                        let dir_next = (next_pos - pos).normalize()?;
                        new_verts.push(pos + dir_next * radius);
                    }
                }
                (true, false, _) => {
                    let ei = poly.wire_edge_ids[prev_i].index();
                    if let Some(&pt) = fillet_contact_map.get(&(vi, ei, fi)) {
                        new_verts.push(pt);
                    } else {
                        let dir = (next_pos - pos).normalize()?;
                        new_verts.push(pos + dir * radius);
                    }
                }
                (false, true, _) => {
                    let ei = poly.wire_edge_ids[i].index();
                    if let Some(&pt) = fillet_contact_map.get(&(vi, ei, fi)) {
                        new_verts.push(pt);
                    } else {
                        let dir = (prev_pos - pos).normalize()?;
                        new_verts.push(pos + dir * radius);
                    }
                }
                (true, true, _) => {
                    // dir_prev (along "before" edge) → perpendicular to
                    // the "after" fillet edge → use "after" edge's contact.
                    let ei_after = poly.wire_edge_ids[i].index();
                    if let Some(&pt) = fillet_contact_map.get(&(vi, ei_after, fi)) {
                        new_verts.push(pt);
                    } else {
                        let dir_prev = (prev_pos - pos).normalize()?;
                        new_verts.push(pos + dir_prev * radius);
                    }
                    // dir_next (along "after" edge) → perpendicular to
                    // the "before" fillet edge → use "before" edge's contact.
                    let ei_before = poly.wire_edge_ids[prev_i].index();
                    if let Some(&pt) = fillet_contact_map.get(&(vi, ei_before, fi)) {
                        new_verts.push(pt);
                    } else {
                        let dir_next = (next_pos - pos).normalize()?;
                        new_verts.push(pos + dir_next * radius);
                    }
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

    // Phase 4: Build NURBS fillet faces for each target edge.
    // Also collect contact points per vertex for vertex blend patches.
    // vertex_contacts maps vertex_index → list of (face_index, contact_point) pairs.
    let mut vertex_contacts: HashMap<usize, Vec<(usize, Point3)>> = HashMap::new();
    // For G1 chain junctions, store the contact points computed by the first
    // edge so the second edge can reuse them exactly.
    let mut g1_contact_cache: HashMap<usize, (Point3, Point3)> = HashMap::new();

    for &edge_id in &filtered_edges {
        let edge = topo.edge(edge_id)?;
        let p_start = topo.vertex(edge.start())?.point();
        let p_end = topo.vertex(edge.end())?.point();

        let Some(face_list) = edge_to_faces.get(&edge_id.index()) else {
            continue; // Edge not in map, skip
        };
        if face_list.len() < 2 {
            continue; // Non-manifold edge, skip
        }
        let f1 = face_list[0];
        let f2 = face_list[1];

        // Get face surfaces — needed for normal evaluation on curved faces.
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

        // Snapshot the edge curve before further borrows.
        let edge_curve = edge.curve().clone();

        // Edge direction at the start (used for cross-section geometry).
        let edge_tan = sample_edge_tangent(&edge_curve, p_start, p_end, 0.0);
        if edge_tan.length() < tol.linear {
            continue;
        }
        let edge_dir = edge_tan.normalize()?;

        // Compute reference inward-pointing directions at the edge start.
        let cross1 = edge_dir.cross(n1_start);
        let cross2 = edge_dir.cross(n2_start);

        let d1_raw = if cross1.dot(n2_start) > 0.0 {
            cross1
        } else {
            -cross1
        };
        let d2_raw = if cross2.dot(n1_start) > 0.0 {
            cross2
        } else {
            -cross2
        };

        let d1_ref = d1_raw.normalize().unwrap_or(d1_raw);
        let d2_ref = d2_raw.normalize().unwrap_or(d2_raw);

        // Half dihedral angle at the start (reference for the whole edge).
        let cos_half = d1_ref.dot(d2_ref).clamp(-1.0, 1.0);
        let half_angle = cos_half.acos() / 2.0;

        if half_angle.abs() < tol.angular || (std::f64::consts::PI - half_angle).abs() < tol.angular
        {
            continue;
        }

        // For curved faces, need more samples even if the edge is straight,
        // because the surface normal varies along the edge.
        let both_planar = matches!(surf1, FaceSurface::Plane { .. })
            && matches!(surf2, FaceSurface::Plane { .. });
        let n_v = if both_planar {
            edge_v_samples(&edge_curve)
        } else {
            edge_v_samples(&edge_curve).max(7)
        };

        // Sample cross-section geometry at each v-station along the edge curve.
        let mut grid: Vec<[Point3; 3]> = Vec::with_capacity(n_v);
        let mut bisector_ref = Vec3::new(0.0, 0.0, 0.0);

        #[allow(clippy::cast_precision_loss)]
        for s in 0..n_v {
            let t = s as f64 / (n_v - 1).max(1) as f64;
            let p = sample_edge_point(&edge_curve, p_start, p_end, t);
            let tan = sample_edge_tangent(&edge_curve, p_start, p_end, t);
            let local_dir = tan.normalize().unwrap_or(edge_dir);

            // Evaluate surface normals at this sample point. For planar faces,
            // these are constant; for curved faces, they vary along the edge.
            let ln1 = face_surface_normal_at(surf1, p).unwrap_or(n1_start);
            let ln2 = face_surface_normal_at(surf2, p).unwrap_or(n2_start);

            // Recompute cross-section directions at this sample
            let c1 = local_dir.cross(ln1);
            let c2 = local_dir.cross(ln2);
            // ld1 points from the edge toward the contact point on face 1,
            // inside the dihedral angle (toward the material). This is
            // OPPOSITE to face 2's outward normal.
            let ld1 = if c1.dot(ln2) < 0.0 { c1 } else { -c1 };
            let ld2 = if c2.dot(ln1) < 0.0 { c2 } else { -c2 };
            let ld1 = ld1.normalize().unwrap_or(d1_ref);
            let ld2 = ld2.normalize().unwrap_or(d2_ref);

            let local_cos = ld1.dot(ld2).clamp(-1.0, 1.0);
            let local_half = local_cos.acos() / 2.0;
            let bisector = (ld1 + ld2).normalize().unwrap_or(d1_ref);

            if s == 0 {
                bisector_ref = bisector;
            }

            let contact1 = p + ld1 * radius;
            let mid_dist = radius / local_half.cos().max(0.01);
            let mid_cp = p + bisector * mid_dist;
            let contact2 = p + ld2 * radius;

            grid.push([contact1, mid_cp, contact2]);
        }

        // G1 chain continuity: at chain junction vertices, snap contact points
        // to match the adjacent fillet strip's endpoints for G1 continuity.
        let start_vi = edge.start().index();
        let end_vi = edge.end().index();
        if g1_chain_vertices.contains(&start_vi) {
            if let Some(&(c1, c2)) = g1_contact_cache.get(&start_vi) {
                // Snap this strip's start to match the previous strip's end.
                grid[0] = [c1, grid[0][1], c2];
            } else {
                // First strip at this junction — cache for the next strip.
                g1_contact_cache.insert(start_vi, (grid[0][0], grid[0][2]));
            }
        }
        if g1_chain_vertices.contains(&end_vi) {
            if let Some(&(c1, c2)) = g1_contact_cache.get(&end_vi) {
                let last = n_v - 1;
                grid[last] = [c1, grid[last][1], c2];
            } else {
                let last = n_v - 1;
                g1_contact_cache.insert(end_vi, (grid[last][0], grid[last][2]));
            }
        }

        // Build the fillet surface from the cross-section grid.
        let contact1_start = grid[0][0];
        let contact2_start = grid[0][2];
        let contact1_end = grid[n_v - 1][0];
        let contact2_end = grid[n_v - 1][2];

        let fillet_surface = if n_v == 2 {
            // Line edge: exact rational quadratic arc × linear.
            let arc_half = half_angle;
            let w_mid = arc_half.cos();
            NurbsSurface::new(
                2,
                1,
                vec![0.0, 0.0, 0.0, 1.0, 1.0, 1.0],
                vec![0.0, 0.0, 1.0, 1.0],
                vec![
                    vec![contact1_start, contact1_end],
                    vec![grid[0][1], grid[1][1]],
                    vec![contact2_start, contact2_end],
                ],
                vec![vec![1.0, 1.0], vec![w_mid, w_mid], vec![1.0, 1.0]],
            )
            .map_err(crate::OperationsError::Math)?
        } else {
            // Curved edge: interpolate through sampled cross-sections.
            let n_arc = 3;
            let transposed: Vec<Vec<Point3>> = (0..n_arc)
                .map(|col| (0..n_v).map(|row| grid[row][col]).collect())
                .collect();
            let degree_u = 2.min(n_arc - 1);
            let degree_v = (n_v - 1).min(3);
            interpolate_surface(&transposed, degree_u, degree_v)
                .map_err(crate::OperationsError::Math)?
        };

        all_specs.push(FaceSpec::Surface {
            vertices: vec![contact1_start, contact2_start, contact2_end, contact1_end],
            surface: FaceSurface::Nurbs(fillet_surface),
            reversed: false,
            inner_wires: vec![],
        });

        // Track which faces need normal reversal.
        let srf_mid_normal = match &all_specs[all_specs.len() - 1] {
            FaceSpec::Surface {
                surface: FaceSurface::Nurbs(srf),
                ..
            } => srf.normal(0.5, 0.5).unwrap_or(bisector_ref),
            _ => bisector_ref,
        };
        if srf_mid_normal.dot(bisector_ref) > 0.0 {
            fillet_face_indices.push(all_specs.len() - 1);
        }

        // Record contact points at each vertex for vertex blend detection.
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
                -blend_normal
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

        // Build a spherical cap NURBS patch instead of a flat triangle.
        // The fillet sphere at a vertex corner is tangent to each adjacent
        // face.  Its center lies at the original vertex offset inward by R
        // along each face normal: center = vertex - R * Σ(face_normals).
        if ordered_points.len() == 3 {
            if let Some(v_pos) = original_vertex {
                // Collect distinct face normals from the contacts at this vertex.
                let mut face_normals: Vec<Vec3> = Vec::new();
                for &(face_idx, _) in contacts {
                    if let Some(poly) = face_polygons.get(&face_idx) {
                        let n = poly.normal;
                        let already = face_normals.iter().any(|existing| {
                            (existing.x() - n.x()).abs() < 1e-10
                                && (existing.y() - n.y()).abs() < 1e-10
                                && (existing.z() - n.z()).abs() < 1e-10
                        });
                        if !already {
                            face_normals.push(n);
                        }
                    }
                }

                // Sphere center: vertex offset inward by R along each face normal.
                let normal_sum = face_normals
                    .iter()
                    .fold(Vec3::new(0.0, 0.0, 0.0), |acc, n| {
                        Vec3::new(acc.x() + n.x(), acc.y() + n.y(), acc.z() + n.z())
                    });

                // Determine whether this vertex corner is convex or concave.
                // For a convex corner the face normals (outward) and edge
                // tangents (pointing away from vertex, i.e. inward) point in
                // opposite directions: normal_sum · avg_tangent < 0.
                // For concave corners they align: dot > 0.
                let is_concave = if let Some(fillet_edges) = vertex_fillet_edges.get(&vi) {
                    if fillet_edges.len() >= 3 {
                        let mut tangent_sum = Vec3::new(0.0, 0.0, 0.0);
                        let mut count = 0;
                        for &eid in fillet_edges {
                            if let Ok(edge) = topo.edge(eid) {
                                let e_start = edge.start();
                                let e_end = edge.end();
                                let curve = edge.curve().clone();
                                let p_s = topo.vertex(e_start)?.point();
                                let p_e = topo.vertex(e_end)?.point();
                                let (t_param, sign) = if e_start.index() == vi {
                                    let (t0, _) = curve.domain_with_endpoints(p_s, p_e);
                                    (t0, 1.0)
                                } else {
                                    let (_, t1) = curve.domain_with_endpoints(p_s, p_e);
                                    (t1, -1.0)
                                };
                                let tan = curve.tangent_with_endpoints(t_param, p_s, p_e);
                                if let Ok(n) = (tan * sign).normalize() {
                                    tangent_sum = Vec3::new(
                                        tangent_sum.x() + n.x(),
                                        tangent_sum.y() + n.y(),
                                        tangent_sum.z() + n.z(),
                                    );
                                    count += 1;
                                }
                            }
                        }
                        count >= 3 && normal_sum.dot(tangent_sum) > 0.0
                    } else {
                        false
                    }
                } else {
                    false
                };

                // For outward-pointing face normals on a convex corner, "inward"
                // means subtracting. For concave corners the offset direction
                // is reversed (we add instead of subtract).
                let offset_sign = if is_concave { 1.0 } else { -1.0 };
                let sphere_center = Point3::new(
                    v_pos.x() + offset_sign * radius * normal_sum.x(),
                    v_pos.y() + offset_sign * radius * normal_sum.y(),
                    v_pos.z() + offset_sign * radius * normal_sum.z(),
                );

                let p0 = ordered_points[0];
                let p1 = ordered_points[1];
                let p2 = ordered_points[2];

                // Helper: compute the tangent-intersection control point and
                // weight for a rational quadratic Bézier circular arc from a to b
                // on the sphere.  The middle CP sits at distance r/cos(θ/2) from
                // center (the tangent intersection), and the weight is cos(θ/2).
                let arc_mid_and_weight = |a: Point3, b: Point3| -> Option<(Point3, f64)> {
                    let va = (a - sphere_center).normalize().ok()?;
                    let vb = (b - sphere_center).normalize().ok()?;
                    let r_actual = (a - sphere_center).length();
                    let sum = va + vb;
                    let len = sum.length();
                    if len < 1e-15 {
                        return None;
                    }
                    let dir = Vec3::new(sum.x() / len, sum.y() / len, sum.z() / len);
                    let cos_half = len / 2.0; // cos(θ/2) for unit vectors
                    let r_ctrl = r_actual / cos_half;
                    let cp = Point3::new(
                        sphere_center.x() + dir.x() * r_ctrl,
                        sphere_center.y() + dir.y() * r_ctrl,
                        sphere_center.z() + dir.z() * r_ctrl,
                    );
                    Some((cp, cos_half))
                };

                // Compute per-edge arc midpoints and weights.
                if let (Some((m01, w01)), Some((m12, w12)), Some((m20, w20))) = (
                    arc_mid_and_weight(p0, p1),
                    arc_mid_and_weight(p1, p2),
                    arc_mid_and_weight(p2, p0),
                ) {
                    // Apex: point on sphere in the average direction of the
                    // three boundary radial vectors.
                    let apex = {
                        let avg = Vec3::new(
                            (p0 - sphere_center).x()
                                + (p1 - sphere_center).x()
                                + (p2 - sphere_center).x(),
                            (p0 - sphere_center).y()
                                + (p1 - sphere_center).y()
                                + (p2 - sphere_center).y(),
                            (p0 - sphere_center).z()
                                + (p1 - sphere_center).z()
                                + (p2 - sphere_center).z(),
                        );
                        let len = avg.length().max(1e-15);
                        let r_actual = (p0 - sphere_center).length();
                        Point3::new(
                            sphere_center.x() + avg.x() / len * r_actual,
                            sphere_center.y() + avg.y() / len * r_actual,
                            sphere_center.z() + avg.z() / len * r_actual,
                        )
                    };

                    // Apex weight: product of the three edge weights gives
                    // a consistent rational patch for the spherical triangle.
                    let w_apex = w01 * w12 * w20;

                    // Build a degree (2,2) patch: 3×3 control points.
                    // u=0: p0→m20→p2, u=0.5: m01→apex→m12, u=1: p1→m12→p2
                    // v=1 boundary degenerates to single point p2.
                    //
                    // Weight grid — per-edge weights on each boundary arc,
                    // apex weight in the interior, 1.0 at corners.
                    // Column 2 (degenerate, all CPs = p2) uses 1.0
                    // for symmetric interior parametrization.
                    let cap_surface = NurbsSurface::new(
                        2,
                        2,
                        vec![0.0, 0.0, 0.0, 1.0, 1.0, 1.0],
                        vec![0.0, 0.0, 0.0, 1.0, 1.0, 1.0],
                        vec![vec![p0, m20, p2], vec![m01, apex, p2], vec![p1, m12, p2]],
                        vec![
                            vec![1.0, w20, 1.0],
                            vec![w01, w_apex, 1.0],
                            vec![1.0, w12, 1.0],
                        ],
                    )
                    .map_err(crate::OperationsError::Math)?;

                    all_specs.push(FaceSpec::Surface {
                        vertices: ordered_points,
                        surface: FaceSurface::Nurbs(cap_surface),
                        reversed: false,
                        inner_wires: vec![],
                    });

                    // Check if we need to flip this face.
                    let cap_norm = match &all_specs[all_specs.len() - 1] {
                        FaceSpec::Surface {
                            surface: FaceSurface::Nurbs(srf),
                            ..
                        } => srf.normal(0.5, 0.5).unwrap_or(blend_normal),
                        _ => blend_normal,
                    };
                    // The cap normal should point away from the original vertex.
                    let to_vertex = v_pos - centroid;
                    if to_vertex.dot(cap_norm) > 0.0 {
                        fillet_face_indices.push(all_specs.len() - 1);
                    }
                    continue;
                }
            }
        }

        // Fallback: flat planar blend for non-triangular or degenerate cases.
        let blend_d = dot_normal_point(blend_normal, ordered_points[0]);
        all_specs.push(FaceSpec::Planar {
            vertices: ordered_points,
            normal: blend_normal,
            d: blend_d,
            inner_wires: vec![],
        });
    }

    // Phase 5c: Remove zero-length edges from face specs.
    // Two fillet contacts can coincide when two fillet strips meet at the
    // same point on a face (e.g., two target edges sharing a vertex on the
    // same face pair).  Remove consecutive duplicate vertices.
    // Only apply to faces where we actually detected fillet contact lookups
    // (indicated by having both (true,true) case AND coincident contacts).
    for spec in &mut all_specs {
        let verts = match spec {
            FaceSpec::Planar { vertices, .. }
            | FaceSpec::Surface { vertices, .. }
            | FaceSpec::CylindricalFace { vertices, .. } => vertices,
        };
        // Only dedup if there are actually zero-length edges (consecutive
        // vertices within tolerance). Count them first.
        if verts.len() > 3 {
            let has_zero_len = verts
                .windows(2)
                .any(|w| (w[0] - w[1]).length() < tol.linear)
                || (verts
                    .first()
                    .zip(verts.last())
                    .is_some_and(|(f, l)| (*f - *l).length() < tol.linear));
            if has_zero_len {
                let mut deduped: Vec<Point3> = Vec::with_capacity(verts.len());
                for (i, &v) in verts.iter().enumerate() {
                    let next = verts[(i + 1) % verts.len()];
                    if (v - next).length() > tol.linear {
                        deduped.push(v);
                    }
                }
                if deduped.len() >= 3 && deduped.len() < verts.len() {
                    *verts = deduped;
                }
            }
        }
    }

    // Phase 5d: Snap passthrough face vertices to original solid positions.
    // Residual precision drift from polygon extraction can produce vertices
    // that are nearly coincident with original positions.
    {
        let mut original_verts: Vec<Point3> = Vec::new();
        for poly in face_polygons.values() {
            for &p in &poly.positions {
                let already = original_verts
                    .iter()
                    .any(|existing| (*existing - p).length() < tol.linear);
                if !already {
                    original_verts.push(p);
                }
            }
        }
        for &fid in &shell_face_ids {
            if face_polygons.contains_key(&fid.index()) {
                continue;
            }
            if let Ok(face) = topo.face(fid) {
                if let Ok(wire) = topo.wire(face.outer_wire()) {
                    for oe in wire.edges() {
                        if let Ok(edge_data) = topo.edge(oe.edge()) {
                            let vid = oe.oriented_start(edge_data);
                            if let Ok(v) = topo.vertex(vid) {
                                let p = v.point();
                                let already = original_verts
                                    .iter()
                                    .any(|existing| (*existing - p).length() < tol.linear);
                                if !already {
                                    original_verts.push(p);
                                }
                            }
                        }
                    }
                }
            }
        }
        // Also collect inner wire vertex positions.
        for poly in face_polygons.values() {
            for iw in &poly.inner_wires {
                for &p in iw {
                    let already = original_verts
                        .iter()
                        .any(|existing| (*existing - p).length() < tol.linear);
                    if !already {
                        original_verts.push(p);
                    }
                }
            }
        }
        let snap_tol = tol.linear * 100.0;
        for spec in &mut all_specs {
            // Snap outer wire vertices.
            let verts = match spec {
                FaceSpec::Planar { vertices, .. }
                | FaceSpec::Surface { vertices, .. }
                | FaceSpec::CylindricalFace { vertices, .. } => vertices,
            };
            for v in verts.iter_mut() {
                if let Some(closest) = original_verts
                    .iter()
                    .filter(|ov| (**ov - *v).length() < snap_tol)
                    .min_by(|a, b| {
                        (**a - *v)
                            .length()
                            .partial_cmp(&(**b - *v).length())
                            .unwrap_or(std::cmp::Ordering::Equal)
                    })
                {
                    *v = *closest;
                }
            }
            // Snap inner wire vertices.
            for iw in spec.inner_wires_mut() {
                for v in iw.iter_mut() {
                    if let Some(closest) = original_verts
                        .iter()
                        .filter(|ov| (**ov - *v).length() < snap_tol)
                        .min_by(|a, b| {
                            (**a - *v)
                                .length()
                                .partial_cmp(&(**b - *v).length())
                                .unwrap_or(std::cmp::Ordering::Equal)
                        })
                    {
                        *v = *closest;
                    }
                }
            }
        }
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

    // Merge co-surface faces that the fillet may have split. This keeps the
    // face count minimal, preventing the downstream boolean from triggering
    // the mesh boolean fallback on moderate-complexity filleted solids.
    let _ = crate::heal::unify_faces(topo, solid_id);

    Ok(solid_id)
}
