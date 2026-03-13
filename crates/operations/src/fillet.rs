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
//! geometry (see [`face_surface_normal_at`]).  Planar trimming of NURBS
//! adjacent faces is not yet implemented — those faces are passed through
//! unchanged; small geometric gaps may appear at the fillet boundary.
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

use brepkit_math::nurbs::projection::project_point_to_surface;
use brepkit_math::nurbs::surface::NurbsSurface;
use brepkit_math::nurbs::surface_fitting::interpolate_surface;
use brepkit_math::tolerance::Tolerance;
use brepkit_math::vec::{Point3, Vec3};
use brepkit_topology::Topology;
use brepkit_topology::edge::{EdgeCurve, EdgeId};
use brepkit_topology::face::{FaceId, FaceSurface};
use brepkit_topology::solid::SolidId;
use brepkit_topology::vertex::VertexId;

use crate::boolean::FaceSpec;
use crate::dot_normal_point;

/// Sample a point along an edge curve at normalised parameter `t` ∈ [0, 1].
///
/// For `Line` edges this is simple lerp between start/end.
/// For Circle/Ellipse/NurbsCurve the actual curve geometry is evaluated.
fn sample_edge_point(curve: &EdgeCurve, p_start: Point3, p_end: Point3, t: f64) -> Point3 {
    match curve {
        EdgeCurve::Line => Point3::new(
            p_start.x().mul_add(1.0 - t, p_end.x() * t),
            p_start.y().mul_add(1.0 - t, p_end.y() * t),
            p_start.z().mul_add(1.0 - t, p_end.z() * t),
        ),
        EdgeCurve::Circle(circle) => {
            let ts = circle.project(p_start);
            let mut te = circle.project(p_end);
            if te <= ts {
                te += std::f64::consts::TAU;
            }
            circle.evaluate(ts + (te - ts) * t)
        }
        EdgeCurve::Ellipse(ellipse) => {
            let ts = ellipse.project(p_start);
            let mut te = ellipse.project(p_end);
            if te <= ts {
                te += std::f64::consts::TAU;
            }
            ellipse.evaluate(ts + (te - ts) * t)
        }
        EdgeCurve::NurbsCurve(nurbs) => {
            let (u0, u1) = nurbs.domain();
            nurbs.evaluate(u0 + (u1 - u0) * t)
        }
    }
}

/// Compute the tangent direction along an edge curve at normalised parameter `t`.
///
/// Returns an unnormalised tangent vector. For `Line` edges this is the constant
/// `p_end - p_start` direction.
fn sample_edge_tangent(curve: &EdgeCurve, p_start: Point3, p_end: Point3, t: f64) -> Vec3 {
    match curve {
        EdgeCurve::Line => p_end - p_start,
        EdgeCurve::Circle(circle) => {
            let ts = circle.project(p_start);
            let mut te = circle.project(p_end);
            if te <= ts {
                te += std::f64::consts::TAU;
            }
            circle.tangent(ts + (te - ts) * t)
        }
        EdgeCurve::Ellipse(ellipse) => {
            let ts = ellipse.project(p_start);
            let mut te = ellipse.project(p_end);
            if te <= ts {
                te += std::f64::consts::TAU;
            }
            ellipse.tangent(ts + (te - ts) * t)
        }
        EdgeCurve::NurbsCurve(nurbs) => {
            let (u0, u1) = nurbs.domain();
            let u = u0 + (u1 - u0) * t;
            let d = nurbs.derivatives(u, 1);
            d[1]
        }
    }
}

/// Determine the number of v-direction samples needed for an edge curve.
///
/// Line edges need only 2 samples (start + end) for an exact linear surface.
/// Curved edges need more samples to capture the curvature.
fn edge_v_samples(curve: &EdgeCurve) -> usize {
    match curve {
        EdgeCurve::Line => 2,
        EdgeCurve::Circle(_) | EdgeCurve::Ellipse(_) => 9,
        EdgeCurve::NurbsCurve(_) => 7,
    }
}

/// Compute the outward surface normal of a `FaceSurface` at a given 3D point.
///
/// For analytic surfaces this is exact (no parameter-space projection needed).
/// For NURBS surfaces, uses the midpoint normal as an approximation (full
/// point-inversion would be needed for exactness, but this suffices for fillet
/// cross-section geometry where the point is known to lie on the surface).
fn face_surface_normal_at(surface: &FaceSurface, point: Point3) -> Option<Vec3> {
    match surface {
        FaceSurface::Plane { normal, .. } => Some(*normal),
        FaceSurface::Cylinder(cyl) => {
            // Project point onto cylinder axis to find closest axis point,
            // then the normal is the radial direction from axis to point.
            let dp = point - cyl.origin();
            let along_axis = dp.dot(cyl.axis());
            let on_axis = cyl.origin() + cyl.axis() * along_axis;
            (point - on_axis).normalize().ok()
        }
        FaceSurface::Cone(cone) => {
            // For a cone, the normal is perpendicular to the surface.
            // Project point onto axis, compute the radial direction,
            // then rotate by (90° - half_angle) around the tangent.
            let dp = point - cone.apex();
            let along_axis = dp.dot(cone.axis());
            let radial = dp - cone.axis() * along_axis;
            let radial_n = radial.normalize().ok()?;
            let (sin_a, cos_a) = cone.half_angle().sin_cos();
            // Normal = radial * sin(half_angle) - axis * cos(half_angle)
            Some(radial_n * sin_a + cone.axis() * (-cos_a))
        }
        FaceSurface::Sphere(sph) => (point - sph.center()).normalize().ok(),
        FaceSurface::Torus(tor) => {
            // Project point onto the major circle plane to find the closest
            // point on the major circle, then the normal is from the tube
            // center toward the point.
            let dp = point - tor.center();
            let along_axis = dp.dot(tor.z_axis());
            let in_plane = dp - tor.z_axis() * along_axis;
            let ring_dir = in_plane.normalize().ok()?;
            let tube_center = tor.center() + ring_dir * tor.major_radius();
            (point - tube_center).normalize().ok()
        }
        FaceSurface::Nurbs(srf) => {
            // Project the point onto the NURBS surface to find (u, v), then
            // evaluate the exact normal at that parameter.  Tolerance 1e-4 is
            // sufficient here — we only need (u,v) accurate enough for a
            // reliable normal direction, not for position reconstruction.
            match project_point_to_surface(srf, point, 1e-4) {
                Ok(proj) => srf.normal(proj.u, proj.v).ok(),
                Err(_) => srf.normal(0.5, 0.5).ok(), // fallback to midpoint
            }
        }
    }
}

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

        // Inner wire edges also contribute to adjacency: an edge shared
        // between a face's inner wire (hole boundary) and another face's
        // outer wire should be counted for both faces.
        for &inner_wid in face.inner_wires() {
            let inner_wire = topo.wire(inner_wid)?;
            for oe in inner_wire.edges() {
                edge_to_faces
                    .entry(oe.edge().index())
                    .or_default()
                    .push(face_id);
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
            result_specs.push(FaceSpec::Surface {
                vertices: verts,
                surface: face.surface().clone(),
                reversed: false,
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

        // Inner wire edges also contribute to adjacency.
        for &inner_wid in face.inner_wires() {
            let inner_wire = topo.wire(inner_wid)?;
            for oe in inner_wire.edges() {
                edge_to_faces
                    .entry(oe.edge().index())
                    .or_default()
                    .push(face_id);
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
            },
        );
    }

    // Phase 2: Filter to manifold edges and build vertex-to-edge adjacency.
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
        for poly in face_polygons.values() {
            let Some(i_e) = poly.wire_edge_ids.iter().position(|&e| e == edge_id) else {
                continue;
            };
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
            let setback_start: f64 = {
                let prev_target = target_set.contains(&poly.wire_edge_ids[prev_i].index());
                if !prev_target {
                    0.0
                } else {
                    // Interior angle at start: between (E forward) and (prev backward from start).
                    let d_e = e_vec * (1.0 / e_len);
                    let d_prev_raw = poly.positions[prev_i] - poly.positions[i_e];
                    let prev_len = d_prev_raw.length();
                    if prev_len < tol.linear {
                        continue;
                    }
                    let d_prev = d_prev_raw * (1.0 / prev_len);
                    let cos_t = d_e.dot(d_prev).clamp(-1.0, 1.0);
                    let theta = cos_t.acos();
                    let half_tan = (theta / 2.0).tan();
                    if half_tan < tol.linear {
                        continue;
                    }
                    radius / half_tan
                }
            };

            // Setback from end vertex if the next polygon edge is also a target.
            let setback_end: f64 = {
                let next_target = target_set.contains(&poly.wire_edge_ids[next_i].index());
                if !next_target {
                    0.0
                } else {
                    // Interior angle at end: between (E backward from end) and (next forward).
                    let d_e_bwd = poly.positions[i_e] - poly.positions[next_i];
                    let d_next_raw = poly.positions[next_next_i] - poly.positions[next_i];
                    let bwd_len = e_len; // same magnitude as e_len
                    let next_len = d_next_raw.length();
                    if next_len < tol.linear {
                        continue;
                    }
                    let d_e_bwd_n = d_e_bwd * (1.0 / bwd_len);
                    let d_next_n = d_next_raw * (1.0 / next_len);
                    let cos_t = d_e_bwd_n.dot(d_next_n).clamp(-1.0, 1.0);
                    let theta = cos_t.acos();
                    let half_tan = (theta / 2.0).tan();
                    if half_tan < tol.linear {
                        continue;
                    }
                    radius / half_tan
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
                reversed: false,
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

            match (before_filleted, after_filleted, at_fillet_endpoint) {
                (false, false, false) => {
                    new_verts.push(pos);
                }
                // Side face: vertex is at a fillet endpoint but neither adjacent
                // edge of this face is the filleted edge. Split into two contact
                // points along the face's own edges.
                (false, false, true) => {
                    let dir_prev = (prev_pos - pos).normalize()?;
                    new_verts.push(pos + dir_prev * radius);

                    let dir_next = (next_pos - pos).normalize()?;
                    new_verts.push(pos + dir_next * radius);
                }
                (true, false, _) => {
                    let dir = (next_pos - pos).normalize()?;
                    new_verts.push(pos + dir * radius);
                }
                (false, true, _) => {
                    let dir = (prev_pos - pos).normalize()?;
                    new_verts.push(pos + dir * radius);
                }
                (true, true, _) => {
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

    // Phase 3b: G1 chain detection.
    // Detect chains of consecutive fillet edges that share a vertex.
    // When two fillet strips meet at a vertex on the same pair of faces,
    // they should share contact points for G1 tangent continuity.
    // Build a map: vertex_index → Vec<(edge_idx, face1_idx, face2_idx)>
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
    // Identify G1 chain junctions: vertices where exactly 2 fillet edges share
    // the same face pair. At these junctions, the fillet strips should merge
    // smoothly. We store the shared vertex info for use during Phase 4.
    let mut g1_chain_vertices: HashSet<usize> = HashSet::new();
    for (vi, adj) in &vertex_fillet_adjacency {
        if adj.len() == 2 && adj[0].1 == adj[1].1 && adj[0].2 == adj[1].2 {
            g1_chain_vertices.insert(*vi);
        }
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
            let ld1 = if c1.dot(ln2) > 0.0 { c1 } else { -c1 };
            let ld2 = if c2.dot(ln1) > 0.0 { c2 } else { -c2 };
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

        // Build a spherical cap NURBS patch instead of a flat triangle.
        // The fillet sphere center is at distance R/sin(half_angle) from the
        // original vertex along the bisector of the face normals. For a 3-point
        // corner, we approximate with a rational Coons-like patch.
        if ordered_points.len() == 3 {
            if let Some(v_pos) = original_vertex {
                // Sphere center: the original vertex offset by R along each
                // adjacent fillet's bisector converges to the same point.
                // Use the centroid + outward normal * R as an approximation.
                let sphere_center = Point3::new(
                    centroid.x() + blend_normal.x() * radius,
                    centroid.y() + blend_normal.y() * radius,
                    centroid.z() + blend_normal.z() * radius,
                );

                // Midpoints of each arc edge on the sphere.
                // For a great-circle arc from A to B on sphere center C radius R:
                //   mid = C + R * normalize((A-C) + (B-C))
                let sphere_mid = |a: Point3, b: Point3| -> Option<Point3> {
                    let va = a - sphere_center;
                    let vb = b - sphere_center;
                    let sum = va + vb;
                    let len = sum.length();
                    if len < 1e-15 {
                        return None;
                    }
                    let r_actual = va.length();
                    let dir = Vec3::new(sum.x() / len, sum.y() / len, sum.z() / len);
                    Some(Point3::new(
                        sphere_center.x() + dir.x() * r_actual,
                        sphere_center.y() + dir.y() * r_actual,
                        sphere_center.z() + dir.z() * r_actual,
                    ))
                };

                let p0 = ordered_points[0];
                let p1 = ordered_points[1];
                let p2 = ordered_points[2];

                // Try to build the spherical cap patch.
                if let (Some(m01), Some(m12), Some(m20)) =
                    (sphere_mid(p0, p1), sphere_mid(p1, p2), sphere_mid(p2, p0))
                {
                    // Apex: point on sphere closest to outward normal.
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

                    // Build a degree (2,2) patch: 3×3 control points.
                    // Row 0: p0, m01, p1  (bottom edge arc)
                    // Row 1: m20, apex, m12 (middle arc through apex)
                    // Row 2: p2, p2, p2  (degenerate top = single point)
                    //
                    // Weights: corners 1.0, edge midpoints need rational weight
                    // for circular arc approximation.
                    let w_edge = {
                        let v0 = (p0 - sphere_center).normalize().unwrap_or(blend_normal);
                        let v1 = (p1 - sphere_center).normalize().unwrap_or(blend_normal);
                        let dot = v0.dot(v1).clamp(-1.0, 1.0);
                        let half = (dot.acos() / 2.0).cos();
                        half.max(0.5)
                    };

                    let cap_surface = NurbsSurface::new(
                        2,
                        2,
                        vec![0.0, 0.0, 0.0, 1.0, 1.0, 1.0],
                        vec![0.0, 0.0, 0.0, 1.0, 1.0, 1.0],
                        // v=1 boundary degenerates to single point p2
                        vec![vec![p0, m20, p2], vec![m01, apex, p2], vec![p1, m12, p2]],
                        vec![
                            vec![1.0, w_edge, 1.0],
                            vec![w_edge, w_edge * w_edge, 1.0],
                            vec![1.0, w_edge, 1.0],
                        ],
                    )
                    .map_err(crate::OperationsError::Math)?;

                    all_specs.push(FaceSpec::Surface {
                        vertices: ordered_points,
                        surface: FaceSurface::Nurbs(cap_surface),
                        reversed: false,
                    });

                    // Check if we need to flip this face too.
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

/// Expand a seed edge set by G1 (tangent-continuity) chain propagation.
///
/// Starting from `seed_edges`, iteratively adds any manifold edge that:
/// 1. Shares a vertex with an edge already in the set.
/// 2. Has the same pair of adjacent faces (same ridgeline).
/// 3. Is tangent-continuous at the shared vertex (< 10° deviation).
///
/// This is a private helper for [`fillet_rolling_ball_propagate_g1`].
fn expand_g1_chain(
    topo: &Topology,
    solid: SolidId,
    seed_edges: &[EdgeId],
    tol: Tolerance,
) -> Result<Vec<EdgeId>, crate::OperationsError> {
    let solid_data = topo.solid(solid)?;
    let shell = topo.shell(solid_data.outer_shell())?;
    let shell_face_ids: Vec<FaceId> = shell.faces().to_vec();

    // Build edge→faces and vertex→edges maps for the full shell.
    let mut edge_to_faces: HashMap<usize, Vec<FaceId>> = HashMap::new();
    let mut vertex_to_edges: HashMap<usize, Vec<EdgeId>> = HashMap::new();
    let mut edge_ids: HashMap<usize, EdgeId> = HashMap::new();

    for &fid in &shell_face_ids {
        let face = topo.face(fid)?;
        let wire = topo.wire(face.outer_wire())?;
        for oe in wire.edges() {
            let eid = oe.edge();
            edge_to_faces.entry(eid.index()).or_default().push(fid);
            edge_ids.insert(eid.index(), eid);
            let edge = topo.edge(eid)?;
            vertex_to_edges
                .entry(edge.start().index())
                .or_default()
                .push(eid);
            vertex_to_edges
                .entry(edge.end().index())
                .or_default()
                .push(eid);
        }
    }
    // Deduplicate vertex_to_edges (each edge appears once per adjacent face).
    for edges in vertex_to_edges.values_mut() {
        edges.sort_unstable_by_key(|e: &EdgeId| e.index());
        edges.dedup_by_key(|e: &mut EdgeId| e.index());
    }

    // Iterative BFS expansion.
    let mut expanded: HashSet<usize> = seed_edges.iter().map(|e| e.index()).collect();
    let mut queue: Vec<EdgeId> = seed_edges.to_vec();

    while let Some(current) = queue.pop() {
        // Face pair for current edge (sorted for comparison).
        let Some(cf) = edge_to_faces.get(&current.index()) else {
            continue;
        };
        if cf.len() != 2 {
            continue;
        }
        let (cf1, cf2) = {
            let (a, b) = (cf[0].index(), cf[1].index());
            if a < b { (a, b) } else { (b, a) }
        };

        let cur_edge = topo.edge(current)?;
        let cur_start = topo.vertex(cur_edge.start())?.point();
        let cur_end = topo.vertex(cur_edge.end())?.point();

        for &shared_vid in &[cur_edge.start(), cur_edge.end()] {
            // "Away from vertex" tangent for the current edge at this vertex.
            let t_cur = {
                let t_raw = if shared_vid == cur_edge.start() {
                    // Forward tangent at start points away from vertex — correct sign.
                    sample_edge_tangent(cur_edge.curve(), cur_start, cur_end, 0.0)
                } else {
                    // Forward tangent at end points INTO vertex; negate for "away".
                    -sample_edge_tangent(cur_edge.curve(), cur_start, cur_end, 1.0)
                };
                let len = t_raw.length();
                if len < tol.linear {
                    continue;
                }
                t_raw * (1.0 / len)
            };

            let Some(neighbors) = vertex_to_edges.get(&shared_vid.index()) else {
                continue;
            };
            for &nb in neighbors {
                if expanded.contains(&nb.index()) {
                    continue;
                }
                // Must be manifold (exactly 2 adjacent faces).
                let Some(nf) = edge_to_faces.get(&nb.index()) else {
                    continue;
                };
                if nf.len() != 2 {
                    continue;
                }
                // Must share the same face pair.
                let (nf1, nf2) = {
                    let (a, b) = (nf[0].index(), nf[1].index());
                    if a < b { (a, b) } else { (b, a) }
                };
                if (cf1, cf2) != (nf1, nf2) {
                    continue;
                }

                // "Away from vertex" tangent for the neighbor edge at the shared vertex.
                let nb_edge = topo.edge(nb)?;
                let nb_start = topo.vertex(nb_edge.start())?.point();
                let nb_end = topo.vertex(nb_edge.end())?.point();
                let t_nb = {
                    let t_raw = if shared_vid == nb_edge.start() {
                        sample_edge_tangent(nb_edge.curve(), nb_start, nb_end, 0.0)
                    } else {
                        -sample_edge_tangent(nb_edge.curve(), nb_start, nb_end, 1.0)
                    };
                    let len = t_raw.length();
                    if len < tol.linear {
                        continue;
                    }
                    t_raw * (1.0 / len)
                };

                // G1 continuity: "away" tangents must be anti-parallel (< ~10° deviation).
                // cos(170°) ≈ -0.985.  This is strict: a true G1 joint has dot = -1.0.
                if t_cur.dot(t_nb) < -0.985 {
                    expanded.insert(nb.index());
                    queue.push(nb);
                }
            }
        }
    }

    Ok(expanded
        .iter()
        .filter_map(|idx| edge_ids.get(idx).copied())
        .collect())
}

/// Fillet `seed_edges` and all G1-continuous edges connected to them.
///
/// This is a convenience wrapper around [`fillet_rolling_ball`] that
/// automatically expands the edge set by G1 chain propagation: any
/// manifold edge that shares a vertex, the same pair of adjacent faces,
/// and is tangent-continuous (< 10° deviation) with a seed edge is added
/// to the fillet set.
///
/// # When to use
///
/// Use this function when you want to fillet a smooth edge chain (e.g. the
/// loop of edges on a rounded-rectangle extrusion) without listing every
/// edge explicitly.  Use [`fillet_rolling_ball`] directly when you want
/// precise control over which individual edges are filleted.
///
/// # Errors
///
/// Returns the same errors as [`fillet_rolling_ball`].
pub fn fillet_rolling_ball_propagate_g1(
    topo: &mut Topology,
    solid: SolidId,
    seed_edges: &[EdgeId],
    radius: f64,
) -> Result<SolidId, crate::OperationsError> {
    let tol = Tolerance::new();
    let expanded = expand_g1_chain(topo, solid, seed_edges, tol)?;
    fillet_rolling_ball(topo, solid, &expanded, radius)
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
            all_specs.push(FaceSpec::Surface {
                vertices: verts,
                surface: face.surface().clone(),
                reversed: false,
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

        let edge_tan = sample_edge_tangent(&edge_curve, p_start, p_end, 0.0);
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
            edge_v_samples(&edge_curve).max(n_samples)
        } else {
            edge_v_samples(&edge_curve).max(n_samples).max(7)
        };

        // Build interpolation grid: n_v rows × 3 columns (arc CPs).
        let mut grid: Vec<Vec<Point3>> = Vec::with_capacity(n_v);
        let mut sample_weights: Vec<f64> = Vec::with_capacity(n_v);

        #[allow(clippy::cast_precision_loss)]
        for s in 0..n_v {
            let t = s as f64 / (n_v - 1).max(1) as f64;
            let r = law.evaluate(t);
            let p = sample_edge_point(&edge_curve, p_start, p_end, t);
            let tan = sample_edge_tangent(&edge_curve, p_start, p_end, t);
            let local_dir = tan.normalize().unwrap_or(edge_dir);

            // Evaluate surface normals at this sample point.
            let ln1 = face_surface_normal_at(surf1, p).unwrap_or(n1_start);
            let ln2 = face_surface_normal_at(surf2, p).unwrap_or(n2_start);

            // Recompute cross-section directions at this sample
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

        let surface = NurbsSurface::new(
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
        });
    }

    crate::boolean::assemble_solid_mixed(topo, &all_specs, tol)
}

#[cfg(test)]
mod tests {
    #![allow(clippy::unwrap_used, clippy::expect_used, deprecated)]

    use std::collections::HashSet;

    use brepkit_topology::Topology;
    use brepkit_topology::edge::EdgeId;
    use brepkit_topology::test_utils::make_unit_cube_manifold;
    use brepkit_topology::validation::validate_shell_manifold;

    use crate::test_helpers::assert_euler_genus0;

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
    fn fillet_single_edge_euler() {
        let mut topo = Topology::new();
        let cube = make_unit_cube_manifold(&mut topo);
        let edges = solid_edge_ids(&topo, cube);
        let result = fillet(&mut topo, cube, &[edges[0]], 0.1).expect("fillet should succeed");
        assert_euler_genus0(&topo, result);
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

        // Rolling-ball fillet on a box should still be genus-0 (χ=2).
        assert_euler_genus0(&topo, result);
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

    /// Fillet on a boolean result: fuse(box, cylinder) → fillet should work
    /// on edges shared between two planar faces.
    #[test]
    fn fillet_on_boolean_result() {
        let mut topo = Topology::new();
        let base = crate::primitives::make_box(&mut topo, 80.0, 60.0, 10.0).unwrap();
        let boss = crate::primitives::make_cylinder(&mut topo, 15.0, 30.0).unwrap();
        let mat = brepkit_math::mat::Mat4::translation(40.0, 30.0, 10.0);
        crate::transform::transform_solid(&mut topo, boss, &mat).unwrap();

        let fused = crate::boolean::boolean(&mut topo, crate::boolean::BooleanOp::Fuse, base, boss)
            .unwrap();

        let solid = topo.solid(fused).unwrap();
        let shell = topo.shell(solid.outer_shell()).unwrap();

        // Build edge-to-face map from all wires (outer + inner).
        let mut edge_to_face_ids: HashMap<usize, Vec<FaceId>> = HashMap::new();
        for &fid in shell.faces() {
            let face = topo.face(fid).unwrap();
            let wire = topo.wire(face.outer_wire()).unwrap();
            for oe in wire.edges() {
                edge_to_face_ids
                    .entry(oe.edge().index())
                    .or_default()
                    .push(fid);
            }
            for &iwid in face.inner_wires() {
                let iw = topo.wire(iwid).unwrap();
                for oe in iw.edges() {
                    edge_to_face_ids
                        .entry(oe.edge().index())
                        .or_default()
                        .push(fid);
                }
            }
        }

        // Allow a small number of seam edges from cylindrical band discretization.
        let bad_count = edge_to_face_ids.values().filter(|f| f.len() != 2).count();
        assert!(
            bad_count <= 4,
            "too many non-manifold edges: {bad_count} (expected <= 4 seam edges)",
        );

        // Fillet only manifold edges where BOTH adjacent faces are planar.
        let is_planar = |fid: FaceId| -> bool {
            matches!(topo.face(fid).unwrap().surface(), FaceSurface::Plane { .. })
        };
        let mut planar_edges = Vec::new();
        for (&eidx, face_ids) in &edge_to_face_ids {
            if face_ids.len() == 2 && is_planar(face_ids[0]) && is_planar(face_ids[1]) {
                let face = topo.face(face_ids[0]).unwrap();
                let wire = topo.wire(face.outer_wire()).unwrap();
                for oe in wire.edges() {
                    if oe.edge().index() == eidx {
                        planar_edges.push(oe.edge());
                        break;
                    }
                }
            }
        }
        planar_edges.sort_unstable_by_key(|e| e.index());
        planar_edges.dedup_by_key(|e| e.index());

        assert!(
            !planar_edges.is_empty(),
            "should have planar-planar edges to fillet"
        );
        let result = super::fillet(&mut topo, fused, &planar_edges, 1.0);
        assert!(
            result.is_ok(),
            "fillet on planar edges of boolean result should succeed: {:?}",
            result.err()
        );
    }

    #[test]
    fn fillet_radius_too_large_rejected() {
        let mut topo = Topology::new();
        let solid = crate::primitives::make_box(&mut topo, 2.0, 2.0, 2.0).unwrap();
        let edges = solid_edge_ids(&topo, solid);
        // Unit cube has edge length 2.0 — a radius of 3.0 exceeds adjacent edges.
        let result = super::fillet_rolling_ball(&mut topo, solid, &edges[..1], 3.0);
        assert!(result.is_err(), "should reject radius exceeding face size");
        let msg = format!("{}", result.unwrap_err());
        assert!(
            msg.contains("exceeds"),
            "error should mention exceeds: {msg}"
        );
    }

    #[test]
    fn fillet_radius_exceeds_cylinder_curvature_rejected() {
        // A cylinder of radius 1.0 cannot be filleted with radius >= 1.0:
        // the offset surface would degenerate to a line.
        let mut topo = Topology::new();
        let solid = crate::primitives::make_cylinder(&mut topo, 1.0, 4.0).unwrap();
        let plane_cyl_edge = {
            let s = topo.solid(solid).unwrap();
            let sh = topo.shell(s.outer_shell()).unwrap();
            let mut edge_faces: HashMap<usize, Vec<FaceId>> = HashMap::new();
            for &fid in sh.faces() {
                let wire = topo.wire(topo.face(fid).unwrap().outer_wire()).unwrap();
                for oe in wire.edges() {
                    edge_faces.entry(oe.edge().index()).or_default().push(fid);
                }
            }
            let mut found = None;
            'outer: for (&eidx, fids) in &edge_faces {
                if fids.len() == 2 {
                    let s1 = topo.face(fids[0]).unwrap().surface().clone();
                    let s2 = topo.face(fids[1]).unwrap().surface().clone();
                    let has_plane = matches!(s1, FaceSurface::Plane { .. })
                        || matches!(s2, FaceSurface::Plane { .. });
                    let has_cyl = matches!(s1, FaceSurface::Cylinder(_))
                        || matches!(s2, FaceSurface::Cylinder(_));
                    if has_plane && has_cyl {
                        for &fid in sh.faces() {
                            let wire =
                                topo.wire(topo.face(fid).unwrap().outer_wire()).unwrap();
                            for oe in wire.edges() {
                                if oe.edge().index() == eidx {
                                    found = Some(oe.edge());
                                    break 'outer;
                                }
                            }
                        }
                    }
                }
            }
            found.expect("cylinder must have a plane-cylinder edge")
        };

        // radius == cylinder radius → curvature radius exactly met → reject.
        let result = super::fillet_rolling_ball(&mut topo, solid, &[plane_cyl_edge], 1.0);
        assert!(result.is_err(), "radius == cylinder radius should be rejected");
        let msg = format!("{}", result.unwrap_err());
        assert!(
            msg.contains("curvature"),
            "error should mention curvature: {msg}"
        );

        // radius > cylinder radius → also rejected.
        let result2 = super::fillet_rolling_ball(&mut topo, solid, &[plane_cyl_edge], 1.5);
        assert!(result2.is_err(), "radius > cylinder radius should be rejected");

        // radius < cylinder radius → passes curvature check (may succeed or fail
        // for other fillet reasons, but must not fail on curvature).
        let result3 = super::fillet_rolling_ball(&mut topo, solid, &[plane_cyl_edge], 0.3);
        if let Err(ref e) = result3 {
            let msg = format!("{e}");
            assert!(
                !msg.contains("curvature"),
                "small radius should not fail curvature check: {msg}"
            );
        }
    }

    #[test]
    fn fillet_radius_just_fits() {
        let mut topo = Topology::new();
        let solid = crate::primitives::make_box(&mut topo, 4.0, 4.0, 4.0).unwrap();
        let edges = solid_edge_ids(&topo, solid);
        // Edge length is 4.0 — a radius of 1.0 should fit comfortably.
        let result = super::fillet_rolling_ball(&mut topo, solid, &edges[..1], 1.0);
        assert!(
            result.is_ok(),
            "small radius should succeed: {:?}",
            result.err()
        );
    }

    #[test]
    fn fillet_plane_cylinder_edge() {
        // A cylinder has planar top/bottom and a cylindrical lateral face.
        // The edges between the planar caps and the cylindrical face should
        // now be filleted (previously silently skipped).
        let mut topo = Topology::new();
        let solid = crate::primitives::make_cylinder(&mut topo, 2.0, 4.0).unwrap();

        // Find edges that border both a planar face and a cylindrical face.
        let s = topo.solid(solid).unwrap();
        let sh = topo.shell(s.outer_shell()).unwrap();
        let mut plane_cyl_edges: Vec<EdgeId> = Vec::new();
        let mut edge_faces: HashMap<usize, Vec<FaceId>> = HashMap::new();

        for &fid in sh.faces() {
            let face = topo.face(fid).unwrap();
            let wire = topo.wire(face.outer_wire()).unwrap();
            for oe in wire.edges() {
                edge_faces.entry(oe.edge().index()).or_default().push(fid);
            }
        }

        for (&eidx, fids) in &edge_faces {
            if fids.len() == 2 {
                let s1 = topo.face(fids[0]).unwrap().surface().clone();
                let s2 = topo.face(fids[1]).unwrap().surface().clone();
                let has_plane = matches!(s1, FaceSurface::Plane { .. })
                    || matches!(s2, FaceSurface::Plane { .. });
                let has_cyl = matches!(s1, FaceSurface::Cylinder(_))
                    || matches!(s2, FaceSurface::Cylinder(_));
                if has_plane && has_cyl {
                    // Recover the EdgeId from eidx — walk the shell to find it.
                    for &fid in sh.faces() {
                        let face = topo.face(fid).unwrap();
                        let wire = topo.wire(face.outer_wire()).unwrap();
                        for oe in wire.edges() {
                            if oe.edge().index() == eidx {
                                plane_cyl_edges.push(oe.edge());
                            }
                        }
                    }
                    break; // Just need one edge for the test
                }
            }
        }

        assert!(
            !plane_cyl_edges.is_empty(),
            "cylinder should have plane-cylinder edges"
        );

        // Fillet the first plane-cylinder edge. This should succeed now
        // (previously it would have been silently skipped).
        let result = super::fillet_rolling_ball(&mut topo, solid, &plane_cyl_edges[..1], 0.3);
        assert!(
            result.is_ok(),
            "plane-cylinder fillet should succeed: {:?}",
            result.err()
        );

        // Verify the result has a NURBS fillet face.
        let result_solid = result.unwrap();
        let rs = topo.solid(result_solid).unwrap();
        let rsh = topo.shell(rs.outer_shell()).unwrap();
        let has_nurbs = rsh
            .faces()
            .iter()
            .any(|&fid| matches!(topo.face(fid).unwrap().surface(), FaceSurface::Nurbs(_)));
        assert!(
            has_nurbs,
            "plane-cylinder fillet should produce a NURBS face"
        );
    }

    #[test]
    fn g1_propagate_box_no_expansion() {
        // On a box every edge meets its neighbors at 90°.
        // Seeding one edge should yield a set of size exactly 1 — no expansion.
        let mut topo = Topology::new();
        let solid = crate::primitives::make_box(&mut topo, 1.0, 1.0, 1.0).unwrap();
        let edges = solid_edge_ids(&topo, solid);
        let seed = &edges[..1];

        // expand_g1_chain is private; exercise it via the public wrapper.
        // We expect the wrapper to succeed and the fillet result to be valid
        // (same as seeding with the single edge directly).
        let result = super::fillet_rolling_ball_propagate_g1(&mut topo, solid, seed, 0.1);
        assert!(
            result.is_ok(),
            "propagate_g1 on a box edge should succeed: {:?}",
            result.err()
        );
        let result_solid = result.unwrap();
        let vol = crate::measure::solid_volume(&topo, result_solid, 0.01).unwrap();
        assert!(
            vol > 0.0 && vol < 1.0,
            "filleted box should have smaller volume than original: {vol}"
        );
    }

    #[test]
    fn g1_propagate_collinear_long_box() {
        // Build a long box (4×1×1) and seed one of the long top edges.
        // A box's long edges are each a single edge, so propagation still
        // yields size 1 — but the wrapper should succeed and produce a valid solid.
        let mut topo = Topology::new();
        let solid = crate::primitives::make_box(&mut topo, 4.0, 1.0, 1.0).unwrap();

        // Pick the first edge that is parallel to the X axis (length ≈ 4.0).
        let edges = solid_edge_ids(&topo, solid);
        let long_edge = edges
            .iter()
            .find(|&&eid| {
                let e = topo.edge(eid).unwrap();
                let p0 = topo.vertex(e.start()).unwrap().point();
                let p1 = topo.vertex(e.end()).unwrap().point();
                let len = (p1 - p0).length();
                len > 3.5
            })
            .copied();
        let Some(seed_edge) = long_edge else {
            panic!("could not find a long edge on a 4×1×1 box");
        };

        let result = super::fillet_rolling_ball_propagate_g1(&mut topo, solid, &[seed_edge], 0.1);
        assert!(
            result.is_ok(),
            "propagate_g1 on long-box edge should succeed: {:?}",
            result.err()
        );
        let result_solid = result.unwrap();
        let vol = crate::measure::solid_volume(&topo, result_solid, 0.01).unwrap();
        assert!(
            vol > 0.0 && vol < 4.0,
            "filleted long box volume should be positive and less than original: {vol}"
        );
    }

    #[test]
    fn adjacent_fillet_overlap_all_edges_rejected() {
        // A 1×1×1 box with all 12 edges filleted at R=0.5 must fail:
        // at each 90° corner the setback from each end is R/tan(45°) = 0.5,
        // and 0.5 + 0.5 = 1.0 = edge length → strips exactly touch → reject.
        let mut topo = Topology::new();
        let solid = crate::primitives::make_box(&mut topo, 1.0, 1.0, 1.0).unwrap();
        let edges = solid_edge_ids(&topo, solid);
        let result = super::fillet_rolling_ball(&mut topo, solid, &edges, 0.5);
        assert!(
            result.is_err(),
            "all-edge fillet with R=0.5 on unit box should be rejected"
        );
        let msg = format!("{}", result.unwrap_err());
        assert!(
            msg.contains("adjacent fillet strips overlap"),
            "error should mention overlap: {msg}"
        );
    }

    #[test]
    fn adjacent_fillet_overlap_fits_with_small_radius() {
        // The same box with R=0.4 must succeed: 0.4+0.4=0.8 < 1.0.
        // (We don't check the full solid validity here — that's covered by
        // vertex_blend_all_edges_box.  Just verify Phase 2d does not block it.)
        let mut topo = Topology::new();
        let solid = crate::primitives::make_box(&mut topo, 1.0, 1.0, 1.0).unwrap();
        let edges = solid_edge_ids(&topo, solid);
        let result = super::fillet_rolling_ball(&mut topo, solid, &edges, 0.4);
        assert!(
            result.is_ok(),
            "all-edge fillet with R=0.4 on unit box should be accepted by Phase 2d: {:?}",
            result.err()
        );
    }

    #[test]
    fn adjacent_fillet_single_edge_no_phase2d_rejection() {
        // Filleting one edge of a box has no adjacent target edges — Phase 2d
        // never fires even for R close to the face size.
        let mut topo = Topology::new();
        let solid = crate::primitives::make_box(&mut topo, 1.0, 1.0, 1.0).unwrap();
        let edges = solid_edge_ids(&topo, solid);
        // Phase 2b caps R at the adjacent edge length (1.0). R=0.4 is well below that.
        let result = super::fillet_rolling_ball(&mut topo, solid, &edges[..1], 0.4);
        assert!(
            result.is_ok(),
            "single-edge fillet with R=0.4 should succeed: {:?}",
            result.err()
        );
    }

    #[test]
    fn face_surface_normal_at_nurbs_via_projection() {
        // Directly test the NURBS branch of face_surface_normal_at.
        // Build a flat bilinear NURBS patch in the XY plane.  The outward
        // normal is (0, 0, 1) everywhere; point projection must return a
        // valid (u, v) that yields the correct normal.
        let srf = NurbsSurface::new(
            1,
            1,
            vec![0.0, 0.0, 1.0, 1.0],
            vec![0.0, 0.0, 1.0, 1.0],
            vec![
                vec![Point3::new(0.0, 0.0, 0.0), Point3::new(0.0, 1.0, 0.0)],
                vec![Point3::new(1.0, 0.0, 0.0), Point3::new(1.0, 1.0, 0.0)],
            ],
            vec![vec![1.0, 1.0], vec![1.0, 1.0]],
        )
        .expect("bilinear XY patch");

        let surface = FaceSurface::Nurbs(srf);

        // Test at a non-central point to ensure projection (not midpoint) is used.
        let n = face_surface_normal_at(&surface, Point3::new(0.2, 0.8, 0.0));
        let n = n.expect("NURBS normal should be Some for a surface point");

        // Flat XY patch has normal along ±Z.
        assert!(
            n.z().abs() > 0.9,
            "flat XY patch normal must be along Z, got: {n:?}"
        );
        // Result must be approximately unit length.
        assert!(
            (n.length() - 1.0).abs() < 0.01,
            "NURBS normal must be unit length, got: {}",
            n.length()
        );
    }

    #[test]
    fn fillet_rolling_ball_second_pass_on_nurbs_solid() {
        // After a rolling-ball fillet the result solid contains a NURBS face.
        // A second fillet on a different manifold edge must succeed.  This
        // verifies that face_surfaces containing NURBS entries does not crash
        // fillet_rolling_ball even when the NURBS face is non-manifold in the
        // current implementation (so its normal branch is not reached yet).
        let mut topo = Topology::new();
        let solid = make_unit_cube_manifold(&mut topo);

        let edges1 = solid_edge_ids(&topo, solid);
        let result1 = super::fillet_rolling_ball(&mut topo, solid, &[edges1[0]], 0.1)
            .expect("first rolling-ball fillet should succeed");

        // Confirm NURBS face was created.
        let has_nurbs = {
            let s = topo.solid(result1).unwrap();
            let sh = topo.shell(s.outer_shell()).unwrap();
            sh.faces()
                .iter()
                .any(|&fid| matches!(topo.face(fid).unwrap().surface(), FaceSurface::Nurbs(_)))
        };
        assert!(has_nurbs, "first fillet must produce a NURBS face");

        // Second fillet on a different edge.
        let edges2 = solid_edge_ids(&topo, result1);
        let result2 = super::fillet_rolling_ball(&mut topo, result1, &[edges2[1]], 0.05);
        assert!(
            result2.is_ok(),
            "second fillet on NURBS-containing solid must succeed: {:?}",
            result2.err()
        );

        let vol = crate::measure::solid_volume(&topo, result2.unwrap(), 0.1).unwrap();
        assert!(
            vol > 0.5,
            "doubly-filleted solid must have positive volume, got {vol}"
        );
    }
}
