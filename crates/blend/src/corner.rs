// Walking engine infrastructure — used progressively as more blend paths are wired up.
#![allow(dead_code)]
//! Vertex blend / corner solver.
//!
//! At vertices where multiple fillet stripes meet, gaps appear that need
//! to be closed with smooth surface patches.  This module classifies
//! each vertex and builds the appropriate corner patch:
//!
//! - **Sphere cap** — 3 stripes with equal radii on mutually orthogonal faces.
//! - **Coons patch** — 3+ stripes in a general (asymmetric) configuration.
//! - **Two-edge** — 2 stripes meeting; a simple triangular fill.
//! - **None** — 0-1 stripes; no corner needed.

use brepkit_math::nurbs::surface::NurbsSurface;
use brepkit_math::surfaces::SphericalSurface;
use brepkit_math::vec::{Point3, Vec3};
use brepkit_topology::Topology;
use brepkit_topology::edge::{Edge, EdgeCurve, EdgeId};
use brepkit_topology::face::{Face, FaceId, FaceSurface};
use brepkit_topology::vertex::{Vertex, VertexId};
use brepkit_topology::wire::{OrientedEdge, Wire};

use crate::BlendError;
use crate::section::CircSection;
use crate::stripe::Stripe;

// ── Types ──────────────────────────────────────────────────────────

/// Classification of a vertex blend.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CornerType {
    /// No corner needed (0-1 stripes at vertex).
    None,
    /// Two stripes meeting — extend/intersect their boundaries.
    TwoEdge,
    /// Spherical cap — 3 edges with equal radii on orthogonal faces.
    SphereCap,
    /// General Coons patch — 3+ edges, asymmetric configuration.
    CoonsPatch,
}

/// Result of building a single corner patch.
pub struct CornerResult {
    /// The face created for the corner patch.
    pub face_id: FaceId,
    /// The surface geometry of the corner patch.
    pub surface: FaceSurface,
    /// New edges created for the corner patch boundary.
    pub new_edges: Vec<EdgeId>,
    /// New vertices created for the corner patch.
    pub new_vertices: Vec<VertexId>,
}

// ── Helpers ────────────────────────────────────────────────────────

/// Tolerance for floating-point comparisons.
const TOL: f64 = 1e-7;

/// Tolerance for angular comparisons (cosine of angle threshold).
const ORTHO_COS_TOL: f64 = 0.1;

/// Return the indices (into `stripes`) of stripes whose spine touches `vertex_id`.
fn stripes_at_vertex(vertex_id: VertexId, stripes: &[Stripe], topo: &Topology) -> Vec<usize> {
    let mut result = Vec::new();
    for (i, stripe) in stripes.iter().enumerate() {
        for &eid in stripe.spine.edges() {
            let Ok(edge) = topo.edge(eid) else {
                continue;
            };
            if edge.start() == vertex_id || edge.end() == vertex_id {
                result.push(i);
                break;
            }
        }
    }
    result
}

/// Get the contact points from a stripe at the given vertex.
///
/// Returns `(p1, p2)` — the contact points on the two adjacent faces.
/// Uses the first section if the vertex is at the spine start, or the last
/// section if the vertex is at the spine end.
fn contact_points_at_vertex(
    vertex_id: VertexId,
    stripe: &Stripe,
    topo: &Topology,
) -> Option<(Point3, Point3)> {
    if stripe.sections.is_empty() {
        return Option::None;
    }

    let edges = stripe.spine.edges();
    if edges.is_empty() {
        return Option::None;
    }

    // Check if vertex is at start of first edge
    let first_edge = topo.edge(edges[0]).ok()?;
    if first_edge.start() == vertex_id || first_edge.end() == vertex_id {
        // Determine if it's the start or end of the spine
        let is_start = first_edge.start() == vertex_id;
        if is_start {
            let sec = stripe.sections.first()?;
            return Some((sec.p1, sec.p2));
        }
    }

    // Check if vertex is at end of last edge
    let last_edge = topo.edge(edges[edges.len() - 1]).ok()?;
    if last_edge.end() == vertex_id || last_edge.start() == vertex_id {
        let is_end = last_edge.end() == vertex_id;
        if is_end {
            let sec = stripe.sections.last()?;
            return Some((sec.p1, sec.p2));
        }
    }

    // Fallback: try first or last based on vertex position proximity
    let vpos = topo.vertex(vertex_id).ok()?.point();
    let first_sec = stripe.sections.first()?;
    let last_sec = stripe.sections.last()?;
    let d_first = (first_sec.center - vpos).length();
    let d_last = (last_sec.center - vpos).length();
    if d_first <= d_last {
        Some((first_sec.p1, first_sec.p2))
    } else {
        Some((last_sec.p1, last_sec.p2))
    }
}

/// Collect all unique contact points from stripes meeting at a vertex.
fn collect_contact_points(
    vertex_id: VertexId,
    stripes: &[Stripe],
    stripe_indices: &[usize],
    topo: &Topology,
) -> Vec<Point3> {
    let mut points = Vec::new();
    for &idx in stripe_indices {
        if let Some((p1, p2)) = contact_points_at_vertex(vertex_id, &stripes[idx], topo) {
            // Add if not duplicate
            if !points.iter().any(|q: &Point3| (*q - p1).length() < TOL) {
                points.push(p1);
            }
            if !points.iter().any(|q: &Point3| (*q - p2).length() < TOL) {
                points.push(p2);
            }
        }
    }
    points
}

/// Get the fillet radius of a stripe at the vertex (from the relevant section).
fn stripe_radius_at_vertex(vertex_id: VertexId, stripe: &Stripe, topo: &Topology) -> Option<f64> {
    contact_section_at_vertex(vertex_id, stripe, topo).map(|s| s.radius)
}

/// Get the section at the vertex end of a stripe.
fn contact_section_at_vertex<'a>(
    vertex_id: VertexId,
    stripe: &'a Stripe,
    topo: &Topology,
) -> Option<&'a CircSection> {
    if stripe.sections.is_empty() {
        return Option::None;
    }

    let edges = stripe.spine.edges();
    if edges.is_empty() {
        return Option::None;
    }

    // Check start of first edge
    if let Ok(first_edge) = topo.edge(edges[0]) {
        if first_edge.start() == vertex_id {
            return stripe.sections.first();
        }
    }

    // Check end of last edge
    if let Ok(last_edge) = topo.edge(edges[edges.len() - 1]) {
        if last_edge.end() == vertex_id {
            return stripe.sections.last();
        }
    }

    // Proximity fallback
    let vpos = topo.vertex(vertex_id).ok()?.point();
    let first = stripe.sections.first()?;
    let last = stripe.sections.last()?;
    if (first.center - vpos).length() <= (last.center - vpos).length() {
        Some(first)
    } else {
        Some(last)
    }
}

/// Build a triangular NURBS face from 3 boundary points.
///
/// Creates a degenerate bilinear patch where one edge collapses to a point,
/// forming a triangle: `p0 - p1 - p2`.
fn make_triangular_nurbs_surface(
    p0: Point3,
    p1: Point3,
    p2: Point3,
) -> Result<NurbsSurface, BlendError> {
    // Bilinear (degree 1x1) patch with a degenerate edge.
    // Row 0: p0, p1  (bottom edge)
    // Row 1: p2, p2  (collapsed top edge = triangle apex)
    let control_points = vec![vec![p0, p1], vec![p2, p2]];
    let weights = vec![vec![1.0, 1.0], vec![1.0, 1.0]];
    let knots_u = vec![0.0, 0.0, 1.0, 1.0];
    let knots_v = vec![0.0, 0.0, 1.0, 1.0];

    Ok(NurbsSurface::new(
        1,
        1,
        knots_u,
        knots_v,
        control_points,
        weights,
    )?)
}

/// Build a quadrilateral NURBS surface (bilinear patch) from 4 corner points.
fn make_quad_nurbs_surface(
    p00: Point3,
    p10: Point3,
    p01: Point3,
    p11: Point3,
) -> Result<NurbsSurface, BlendError> {
    let control_points = vec![vec![p00, p10], vec![p01, p11]];
    let weights = vec![vec![1.0, 1.0], vec![1.0, 1.0]];
    let knots_u = vec![0.0, 0.0, 1.0, 1.0];
    let knots_v = vec![0.0, 0.0, 1.0, 1.0];

    Ok(NurbsSurface::new(
        1,
        1,
        knots_u,
        knots_v,
        control_points,
        weights,
    )?)
}

// ── Classification ─────────────────────────────────────────────────

/// Classify the vertex blend type based on the stripes meeting at this vertex.
#[must_use]
pub fn classify_corner(vertex_id: VertexId, stripes: &[Stripe], topo: &Topology) -> CornerType {
    let indices = stripes_at_vertex(vertex_id, stripes, topo);

    match indices.len() {
        0 | 1 => return CornerType::None,
        2 => return CornerType::TwoEdge,
        _ => {}
    }

    // 3+ stripes: check if sphere cap applies
    if indices.len() == 3 {
        // Check equal radii
        let radii: Vec<f64> = indices
            .iter()
            .filter_map(|&i| stripe_radius_at_vertex(vertex_id, &stripes[i], topo))
            .collect();

        if radii.len() == 3 {
            let r0 = radii[0];
            let all_equal = radii.iter().all(|&r| (r - r0).abs() < TOL);

            if all_equal {
                // For a sphere cap, we need 3 mutually orthogonal face normals
                // from the original faces (not the fillet surfaces).
                // Collect from face1 and face2 of each stripe.
                let mut face_normals = Vec::new();
                for &idx in &indices {
                    let stripe = &stripes[idx];
                    for face_id in [stripe.face1, stripe.face2] {
                        if let Ok(face) = topo.face(face_id) {
                            let n = face.surface().normal(0.0, 0.0);
                            let is_dup = face_normals
                                .iter()
                                .any(|existing: &Vec3| existing.dot(n).abs() > 1.0 - ORTHO_COS_TOL);
                            if !is_dup {
                                face_normals.push(n);
                            }
                        }
                    }
                }

                if face_normals.len() == 3 {
                    // Check pairwise orthogonality
                    let ortho_01 = face_normals[0].dot(face_normals[1]).abs() < ORTHO_COS_TOL;
                    let ortho_02 = face_normals[0].dot(face_normals[2]).abs() < ORTHO_COS_TOL;
                    let ortho_12 = face_normals[1].dot(face_normals[2]).abs() < ORTHO_COS_TOL;

                    if ortho_01 && ortho_02 && ortho_12 {
                        return CornerType::SphereCap;
                    }
                }
            }
        }
    }

    CornerType::CoonsPatch
}

// ── Sphere Cap Builder ─────────────────────────────────────────────

/// Build a sphere-cap corner patch for 3 stripes with equal radii meeting
/// at a vertex where the adjacent faces are mutually orthogonal.
///
/// # Errors
/// Returns `BlendError` if topology lookups or geometry construction fails.
#[allow(clippy::too_many_lines)]
pub fn build_sphere_cap(
    vertex_id: VertexId,
    stripes: &[Stripe],
    radius: f64,
    topo: &mut Topology,
) -> Result<CornerResult, BlendError> {
    let indices = stripes_at_vertex(vertex_id, stripes, topo);

    // Collect face normals from the original faces (not fillet surfaces).
    let mut face_normals: Vec<Vec3> = Vec::new();
    for &idx in &indices {
        let stripe = &stripes[idx];
        for face_id in [stripe.face1, stripe.face2] {
            let face_surf = topo.face(face_id)?.surface().clone();
            let n = face_surf.normal(0.0, 0.0);
            let is_dup = face_normals
                .iter()
                .any(|existing| existing.dot(n).abs() > 1.0 - ORTHO_COS_TOL);
            if !is_dup {
                face_normals.push(n);
            }
        }
    }

    // Sphere center = vertex + R*n1 + R*n2 + R*n3
    // Each face normal contributes an independent offset of R along its
    // direction, giving the correct center for the osculating sphere at
    // a box corner where 3 mutually-orthogonal faces meet.
    let vertex_pos = topo.vertex(vertex_id)?.point();
    let offset: Vec3 = face_normals
        .iter()
        .copied()
        .fold(Vec3::new(0.0, 0.0, 0.0), |acc, n| acc + n * radius);
    let center = vertex_pos + offset;
    let offset_dir = offset.normalize().unwrap_or(Vec3::new(0.0, 0.0, 1.0));

    // Collect contact points — these form the boundary of the spherical cap.
    let contact_pts = collect_contact_points(vertex_id, stripes, &indices, topo);

    if contact_pts.len() < 3 {
        return Err(BlendError::CornerFailure { vertex: vertex_id });
    }

    // Build the spherical surface
    let sphere = SphericalSurface::with_axis(center, radius, offset_dir)?;
    let surface = FaceSurface::Sphere(sphere);

    // Create vertices at the contact points and build edges between them.
    let mut new_vertices = Vec::with_capacity(contact_pts.len());
    let mut new_edges = Vec::with_capacity(contact_pts.len());

    for &pt in &contact_pts {
        let vid = topo.add_vertex(Vertex::new(pt, TOL));
        new_vertices.push(vid);
    }

    // Create edges forming a closed loop through the contact points.
    // Each edge is a line segment for G0 continuity (v1 simplification).
    let n_pts = new_vertices.len();
    for i in 0..n_pts {
        let v_start = new_vertices[i];
        let v_end = new_vertices[(i + 1) % n_pts];
        let eid = topo.add_edge(Edge::new(v_start, v_end, EdgeCurve::Line));
        new_edges.push(eid);
    }

    // Build wire from edges
    let oriented_edges: Vec<OrientedEdge> = new_edges
        .iter()
        .map(|&eid| OrientedEdge::new(eid, true))
        .collect();
    let wire = Wire::new(oriented_edges, true)?;
    let wire_id = topo.add_wire(wire);

    // Create the face
    let face = Face::new(wire_id, Vec::new(), surface.clone());
    let face_id = topo.add_face(face);

    Ok(CornerResult {
        face_id,
        surface,
        new_edges,
        new_vertices,
    })
}

// ── Coons Patch Builder ────────────────────────────────────────────

/// Build a Coons-patch corner for 3+ stripes meeting at a vertex.
///
/// For v1, this is a simple triangular or polygonal NURBS patch that
/// interpolates the contact points with G0 (positional) continuity.
///
/// # Errors
/// Returns `BlendError` if topology lookups or surface construction fails.
pub fn build_coons_patch(
    vertex_id: VertexId,
    stripes: &[Stripe],
    topo: &mut Topology,
) -> Result<CornerResult, BlendError> {
    let indices = stripes_at_vertex(vertex_id, stripes, topo);
    let contact_pts = collect_contact_points(vertex_id, stripes, &indices, topo);

    if contact_pts.len() < 3 {
        return Err(BlendError::CornerFailure { vertex: vertex_id });
    }

    // Build a triangular or quad patch depending on point count.
    if contact_pts.len() == 3 {
        let (surface, new_vertices, new_edges) = build_triangular_patch(&contact_pts, topo)?;
        let oriented_edges: Vec<OrientedEdge> = new_edges
            .iter()
            .map(|&eid| OrientedEdge::new(eid, true))
            .collect();
        let wire = Wire::new(oriented_edges, true)?;
        let wire_id = topo.add_wire(wire);
        let face = Face::new(wire_id, Vec::new(), surface.clone());
        let face_id = topo.add_face(face);
        Ok(CornerResult {
            face_id,
            surface,
            new_edges,
            new_vertices,
        })
    } else if contact_pts.len() == 4 {
        let (surface, new_vertices, new_edges) = build_quad_patch(&contact_pts, topo)?;
        let oriented_edges: Vec<OrientedEdge> = new_edges
            .iter()
            .map(|&eid| OrientedEdge::new(eid, true))
            .collect();
        let wire = Wire::new(oriented_edges, true)?;
        let wire_id = topo.add_wire(wire);
        let face = Face::new(wire_id, Vec::new(), surface.clone());
        let face_id = topo.add_face(face);
        Ok(CornerResult {
            face_id,
            surface,
            new_edges,
            new_vertices,
        })
    } else {
        // N > 4: centroid-fan triangulation.
        build_centroid_fan(&contact_pts, topo, vertex_id)
    }
}

/// Build a triangular NURBS patch from 3 contact points.
fn build_triangular_patch(
    pts: &[Point3],
    topo: &mut Topology,
) -> Result<(FaceSurface, Vec<VertexId>, Vec<EdgeId>), BlendError> {
    let p0 = pts[0];
    let p1 = pts[1];
    let p2 = pts[2];

    let nurbs = make_triangular_nurbs_surface(p0, p1, p2)?;
    let surface = FaceSurface::Nurbs(nurbs);

    let v0 = topo.add_vertex(Vertex::new(p0, TOL));
    let v1 = topo.add_vertex(Vertex::new(p1, TOL));
    let v2 = topo.add_vertex(Vertex::new(p2, TOL));

    let e0 = topo.add_edge(Edge::new(v0, v1, EdgeCurve::Line));
    let e1 = topo.add_edge(Edge::new(v1, v2, EdgeCurve::Line));
    let e2 = topo.add_edge(Edge::new(v2, v0, EdgeCurve::Line));

    Ok((surface, vec![v0, v1, v2], vec![e0, e1, e2]))
}

/// Build a quad NURBS patch from 4 contact points.
fn build_quad_patch(
    pts: &[Point3],
    topo: &mut Topology,
) -> Result<(FaceSurface, Vec<VertexId>, Vec<EdgeId>), BlendError> {
    let p0 = pts[0];
    let p1 = pts[1];
    let p2 = pts[2];
    let p3 = pts[3];

    let nurbs = make_quad_nurbs_surface(p0, p1, p3, p2)?;
    let surface = FaceSurface::Nurbs(nurbs);

    let v0 = topo.add_vertex(Vertex::new(p0, TOL));
    let v1 = topo.add_vertex(Vertex::new(p1, TOL));
    let v2 = topo.add_vertex(Vertex::new(p2, TOL));
    let v3 = topo.add_vertex(Vertex::new(p3, TOL));

    let e0 = topo.add_edge(Edge::new(v0, v1, EdgeCurve::Line));
    let e1 = topo.add_edge(Edge::new(v1, v2, EdgeCurve::Line));
    let e2 = topo.add_edge(Edge::new(v2, v3, EdgeCurve::Line));
    let e3 = topo.add_edge(Edge::new(v3, v0, EdgeCurve::Line));

    Ok((surface, vec![v0, v1, v2, v3], vec![e0, e1, e2, e3]))
}

/// Build a centroid-fan triangulation for N > 4 contact points.
///
/// Computes the centroid, sorts points by angle around it, then builds
/// N triangular NURBS faces fanning from the centroid. Returns the first
/// face and collects all vertices/edges.
#[allow(clippy::too_many_lines)]
fn build_centroid_fan(
    pts: &[Point3],
    topo: &mut Topology,
    vertex_id: VertexId,
) -> Result<CornerResult, BlendError> {
    let n = pts.len();
    if n < 3 {
        return Err(BlendError::CornerFailure { vertex: vertex_id });
    }

    // Compute centroid.
    #[allow(clippy::cast_precision_loss)]
    let inv_n = 1.0 / n as f64;
    let mut cx = 0.0;
    let mut cy = 0.0;
    let mut cz = 0.0;
    for p in pts {
        cx += p.x();
        cy += p.y();
        cz += p.z();
    }
    let centroid = Point3::new(cx * inv_n, cy * inv_n, cz * inv_n);

    // Compute a local frame at the centroid for angle sorting.
    // Normal: average cross products of consecutive edges.
    let d0 = pts[0] - centroid;
    let d1 = pts[1] - centroid;
    let up = d0.cross(d1).normalize().unwrap_or(Vec3::new(0.0, 0.0, 1.0));
    let u_axis = d0.normalize().unwrap_or(Vec3::new(1.0, 0.0, 0.0));
    let v_axis = up
        .cross(u_axis)
        .normalize()
        .unwrap_or(Vec3::new(0.0, 1.0, 0.0));

    // Sort indices by angle.
    let mut indices: Vec<usize> = (0..n).collect();
    indices.sort_by(|&a, &b| {
        let da = pts[a] - centroid;
        let db = pts[b] - centroid;
        let angle_a = da.dot(v_axis).atan2(da.dot(u_axis));
        let angle_b = db.dot(v_axis).atan2(db.dot(u_axis));
        angle_a
            .partial_cmp(&angle_b)
            .unwrap_or(std::cmp::Ordering::Equal)
    });

    // Create centroid vertex.
    let centroid_vid = topo.add_vertex(Vertex::new(centroid, TOL));

    // Create vertices for each sorted contact point.
    let mut vids: Vec<VertexId> = Vec::with_capacity(n);
    for &idx in &indices {
        vids.push(topo.add_vertex(Vertex::new(pts[idx], TOL)));
    }

    // Build N triangular faces: centroid → pts[i] → pts[i+1].
    let mut all_new_vertices = vec![centroid_vid];
    all_new_vertices.extend_from_slice(&vids);
    let mut all_new_edges = Vec::new();

    // Create radial edges (centroid → each vertex).
    let mut radial_edges: Vec<EdgeId> = Vec::with_capacity(n);
    for &vid in &vids {
        let eid = topo.add_edge(Edge::new(centroid_vid, vid, EdgeCurve::Line));
        radial_edges.push(eid);
        all_new_edges.push(eid);
    }

    // Create rim edges (pts[i] → pts[i+1]).
    let mut rim_edges: Vec<EdgeId> = Vec::with_capacity(n);
    for i in 0..n {
        let next = (i + 1) % n;
        let eid = topo.add_edge(Edge::new(vids[i], vids[next], EdgeCurve::Line));
        rim_edges.push(eid);
        all_new_edges.push(eid);
    }

    // Build the first triangle as the "main" face — use the first triangle's
    // surface for the CornerResult. Additional triangles get their own faces
    // but we only return one CornerResult; the caller will get the first face
    // and the topology will contain all faces.
    let mut first_face_id = None;

    for i in 0..n {
        let next = (i + 1) % n;

        let nurbs = make_triangular_nurbs_surface(centroid, pts[indices[i]], pts[indices[next]])?;
        let surface = FaceSurface::Nurbs(nurbs);

        // Wire: radial[i] → rim[i] → radial[next] reversed
        let wire = Wire::new(
            vec![
                OrientedEdge::new(radial_edges[i], true),
                OrientedEdge::new(rim_edges[i], true),
                OrientedEdge::new(radial_edges[next], false),
            ],
            true,
        )?;
        let wire_id = topo.add_wire(wire);
        let face = Face::new(wire_id, Vec::new(), surface);
        let fid = topo.add_face(face);

        if first_face_id.is_none() {
            first_face_id = Some(fid);
        }
    }

    let face_id = first_face_id.ok_or(BlendError::CornerFailure { vertex: vertex_id })?;

    // Return the first triangle's surface as representative.
    let representative_surface = FaceSurface::Nurbs(make_triangular_nurbs_surface(
        centroid,
        pts[indices[0]],
        pts[indices[1 % n]],
    )?);

    Ok(CornerResult {
        face_id,
        surface: representative_surface,
        new_edges: all_new_edges,
        new_vertices: all_new_vertices,
    })
}

// ── Two-Edge Builder ───────────────────────────────────────────────

/// Build a simple triangular fill for 2 stripes meeting at a vertex.
///
/// # Errors
/// Returns `BlendError` if topology lookups fail.
fn build_two_edge_patch(
    vertex_id: VertexId,
    stripes: &[Stripe],
    topo: &mut Topology,
) -> Result<CornerResult, BlendError> {
    let indices = stripes_at_vertex(vertex_id, stripes, topo);
    let contact_pts = collect_contact_points(vertex_id, stripes, &indices, topo);

    // With 2 stripes we expect 3-4 unique contact points (some may merge).
    // Build a triangular patch from the first 3 unique points.
    let pts = if contact_pts.len() >= 3 {
        &contact_pts[..3]
    } else {
        // Degenerate case: not enough unique points
        return Err(BlendError::CornerFailure { vertex: vertex_id });
    };

    let (surface, new_vertices, new_edges) = build_triangular_patch(pts, topo)?;

    let oriented_edges: Vec<OrientedEdge> = new_edges
        .iter()
        .map(|&eid| OrientedEdge::new(eid, true))
        .collect();
    let wire = Wire::new(oriented_edges, true)?;
    let wire_id = topo.add_wire(wire);

    let face = Face::new(wire_id, Vec::new(), surface.clone());
    let face_id = topo.add_face(face);

    Ok(CornerResult {
        face_id,
        surface,
        new_edges,
        new_vertices,
    })
}

// ── Top-Level Entry Point ──────────────────────────────────────────

/// Compute vertex blend patches for all corners where multiple stripes meet.
///
/// Iterates over all vertices of the solid, classifies each, and builds
/// the appropriate corner patch.
///
/// # Errors
/// Returns `BlendError` if topology lookups or patch construction fails.
pub fn compute_corners(
    topo: &mut Topology,
    stripes: &[Stripe],
    solid: brepkit_topology::solid::SolidId,
) -> Result<Vec<CornerResult>, BlendError> {
    use brepkit_topology::explorer::solid_vertices;

    let vertices = solid_vertices(topo, solid)?;
    let mut results = Vec::new();

    for vid in vertices {
        // Classify — read-only borrow of topo is fine here since classify
        // does not mutate.
        let corner_type = classify_corner(vid, stripes, topo);

        match corner_type {
            CornerType::None => {}
            CornerType::TwoEdge => {
                let result = build_two_edge_patch(vid, stripes, topo)?;
                results.push(result);
            }
            CornerType::SphereCap => {
                // All 3 stripes have equal radius — grab from the first one.
                let indices = stripes_at_vertex(vid, stripes, topo);
                let radius = stripe_radius_at_vertex(vid, &stripes[indices[0]], topo)
                    .ok_or(BlendError::CornerFailure { vertex: vid })?;
                let result = build_sphere_cap(vid, stripes, radius, topo)?;
                results.push(result);
            }
            CornerType::CoonsPatch => {
                let result = build_coons_patch(vid, stripes, topo)?;
                results.push(result);
            }
        }
    }

    Ok(results)
}

// ── Tests ──────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    #![allow(clippy::unwrap_used, clippy::expect_used, clippy::panic)]

    use super::*;
    use crate::spine::Spine;
    use brepkit_math::nurbs::curve::NurbsCurve;
    use brepkit_math::vec::{Point3, Vec3};
    use brepkit_topology::edge::{Edge, EdgeCurve};
    use brepkit_topology::face::{Face, FaceSurface};
    use brepkit_topology::shell::Shell;
    use brepkit_topology::solid::Solid;
    use brepkit_topology::vertex::Vertex;
    use brepkit_topology::wire::{OrientedEdge, Wire};

    /// Helper: build a simple box topology with 8 vertices, 12 edges, 6 faces,
    /// and return the corner vertex at the origin along with 3 stripes that
    /// meet there.
    fn setup_box_corner() -> (
        Topology,
        VertexId,
        Vec<Stripe>,
        brepkit_topology::solid::SolidId,
    ) {
        let mut topo = Topology::new();

        // Box vertices at unit cube corners
        let v000 = topo.add_vertex(Vertex::new(Point3::new(0.0, 0.0, 0.0), TOL));
        let v100 = topo.add_vertex(Vertex::new(Point3::new(1.0, 0.0, 0.0), TOL));
        let v010 = topo.add_vertex(Vertex::new(Point3::new(0.0, 1.0, 0.0), TOL));
        let v001 = topo.add_vertex(Vertex::new(Point3::new(0.0, 0.0, 1.0), TOL));
        let v110 = topo.add_vertex(Vertex::new(Point3::new(1.0, 1.0, 0.0), TOL));
        let v101 = topo.add_vertex(Vertex::new(Point3::new(1.0, 0.0, 1.0), TOL));
        let v011 = topo.add_vertex(Vertex::new(Point3::new(0.0, 1.0, 1.0), TOL));
        let v111 = topo.add_vertex(Vertex::new(Point3::new(1.0, 1.0, 1.0), TOL));

        // Edges along X, Y, Z from origin vertex
        let ex = topo.add_edge(Edge::new(v000, v100, EdgeCurve::Line));
        let ey = topo.add_edge(Edge::new(v000, v010, EdgeCurve::Line));
        let ez = topo.add_edge(Edge::new(v000, v001, EdgeCurve::Line));

        // Additional edges for face closure (minimal for test)
        let exy = topo.add_edge(Edge::new(v100, v110, EdgeCurve::Line));
        let eyx = topo.add_edge(Edge::new(v010, v110, EdgeCurve::Line));
        let exz = topo.add_edge(Edge::new(v100, v101, EdgeCurve::Line));
        let ezx = topo.add_edge(Edge::new(v001, v101, EdgeCurve::Line));
        let eyz = topo.add_edge(Edge::new(v010, v011, EdgeCurve::Line));
        let ezy = topo.add_edge(Edge::new(v001, v011, EdgeCurve::Line));

        // Faces adjacent to the origin corner (XY, XZ, YZ planes)
        let face_xy = {
            let w = Wire::new(
                vec![
                    OrientedEdge::new(ex, true),
                    OrientedEdge::new(exy, true),
                    OrientedEdge::new(eyx, false),
                    OrientedEdge::new(ey, false),
                ],
                true,
            )
            .unwrap();
            let wid = topo.add_wire(w);
            let f = Face::new(
                wid,
                Vec::new(),
                FaceSurface::Plane {
                    normal: Vec3::new(0.0, 0.0, -1.0),
                    d: 0.0,
                },
            );
            topo.add_face(f)
        };

        let face_xz = {
            let w = Wire::new(
                vec![
                    OrientedEdge::new(ex, true),
                    OrientedEdge::new(exz, true),
                    OrientedEdge::new(ezx, false),
                    OrientedEdge::new(ez, false),
                ],
                true,
            )
            .unwrap();
            let wid = topo.add_wire(w);
            let f = Face::new(
                wid,
                Vec::new(),
                FaceSurface::Plane {
                    normal: Vec3::new(0.0, -1.0, 0.0),
                    d: 0.0,
                },
            );
            topo.add_face(f)
        };

        let face_yz = {
            let w = Wire::new(
                vec![
                    OrientedEdge::new(ey, true),
                    OrientedEdge::new(eyz, true),
                    OrientedEdge::new(ezy, false),
                    OrientedEdge::new(ez, false),
                ],
                true,
            )
            .unwrap();
            let wid = topo.add_wire(w);
            let f = Face::new(
                wid,
                Vec::new(),
                FaceSurface::Plane {
                    normal: Vec3::new(-1.0, 0.0, 0.0),
                    d: 0.0,
                },
            );
            topo.add_face(f)
        };

        // Opposite faces (for a complete shell)
        let e_top1 = topo.add_edge(Edge::new(v101, v111, EdgeCurve::Line));
        let e_top2 = topo.add_edge(Edge::new(v011, v111, EdgeCurve::Line));
        let face_top = {
            let w = Wire::new(
                vec![
                    OrientedEdge::new(exz, true),
                    OrientedEdge::new(e_top1, true),
                    OrientedEdge::new(e_top2, false),
                    OrientedEdge::new(ezy, false),
                ],
                true,
            )
            .unwrap();
            let wid = topo.add_wire(w);
            let f = Face::new(
                wid,
                Vec::new(),
                FaceSurface::Plane {
                    normal: Vec3::new(0.0, 0.0, 1.0),
                    d: 1.0,
                },
            );
            topo.add_face(f)
        };

        let face_right = {
            let w = Wire::new(
                vec![
                    OrientedEdge::new(exy, true),
                    OrientedEdge::new(e_top1, false),
                    OrientedEdge::new(exz, false),
                    OrientedEdge::new(ex, false),
                ],
                true,
            )
            .unwrap();
            let wid = topo.add_wire(w);
            let f = Face::new(
                wid,
                Vec::new(),
                FaceSurface::Plane {
                    normal: Vec3::new(1.0, 0.0, 0.0),
                    d: 1.0,
                },
            );
            topo.add_face(f)
        };

        let face_back = {
            let w = Wire::new(
                vec![
                    OrientedEdge::new(eyz, true),
                    OrientedEdge::new(e_top2, true),
                    OrientedEdge::new(exy, false),
                    OrientedEdge::new(ey, false),
                ],
                true,
            )
            .unwrap();
            let wid = topo.add_wire(w);
            let f = Face::new(
                wid,
                Vec::new(),
                FaceSurface::Plane {
                    normal: Vec3::new(0.0, 1.0, 0.0),
                    d: 1.0,
                },
            );
            topo.add_face(f)
        };

        // Build shell and solid
        let shell = Shell::new(vec![
            face_xy, face_xz, face_yz, face_top, face_right, face_back,
        ])
        .unwrap();
        let shell_id = topo.add_shell(shell);
        let solid = Solid::new(shell_id, vec![]);
        let solid_id = topo.add_solid(solid);

        // Create 3 stripes along the 3 edges from the origin vertex.
        // Each stripe has a radius and sections with contact points.
        let radius = 0.2;

        // Stripe along X edge (between face_xy and face_xz)
        let spine_x = Spine::from_single_edge(&topo, ex).unwrap();
        let stripe_x = Stripe {
            spine: spine_x,
            surface: FaceSurface::Plane {
                normal: Vec3::new(0.0, 0.0, 1.0),
                d: 0.0,
            },
            pcurve1: brepkit_math::curves2d::Curve2D::Line(
                brepkit_math::curves2d::Line2D::new(
                    brepkit_math::vec::Point2::new(0.0, 0.0),
                    brepkit_math::vec::Vec2::new(1.0, 0.0),
                )
                .unwrap(),
            ),
            pcurve2: brepkit_math::curves2d::Curve2D::Line(
                brepkit_math::curves2d::Line2D::new(
                    brepkit_math::vec::Point2::new(0.0, 0.0),
                    brepkit_math::vec::Vec2::new(1.0, 0.0),
                )
                .unwrap(),
            ),
            contact1: NurbsCurve::new(
                1,
                vec![0.0, 0.0, 1.0, 1.0],
                vec![Point3::new(0.0, 0.0, radius), Point3::new(1.0, 0.0, radius)],
                vec![1.0, 1.0],
            )
            .unwrap(),
            contact2: NurbsCurve::new(
                1,
                vec![0.0, 0.0, 1.0, 1.0],
                vec![Point3::new(0.0, radius, 0.0), Point3::new(1.0, radius, 0.0)],
                vec![1.0, 1.0],
            )
            .unwrap(),
            face1: face_xy,
            face2: face_xz,
            sections: vec![
                CircSection {
                    p1: Point3::new(0.0, 0.0, radius),
                    p2: Point3::new(0.0, radius, 0.0),
                    center: Point3::new(0.0, radius, radius),
                    radius,
                    uv1: (0.0, 0.0),
                    uv2: (0.0, 0.0),
                    t: 0.0,
                },
                CircSection {
                    p1: Point3::new(1.0, 0.0, radius),
                    p2: Point3::new(1.0, radius, 0.0),
                    center: Point3::new(1.0, radius, radius),
                    radius,
                    uv1: (0.0, 0.0),
                    uv2: (0.0, 0.0),
                    t: 1.0,
                },
            ],
        };

        // Stripe along Y edge (between face_xy and face_yz)
        let spine_y = Spine::from_single_edge(&topo, ey).unwrap();
        let stripe_y = Stripe {
            spine: spine_y,
            surface: FaceSurface::Plane {
                normal: Vec3::new(0.0, 0.0, 1.0),
                d: 0.0,
            },
            pcurve1: brepkit_math::curves2d::Curve2D::Line(
                brepkit_math::curves2d::Line2D::new(
                    brepkit_math::vec::Point2::new(0.0, 0.0),
                    brepkit_math::vec::Vec2::new(1.0, 0.0),
                )
                .unwrap(),
            ),
            pcurve2: brepkit_math::curves2d::Curve2D::Line(
                brepkit_math::curves2d::Line2D::new(
                    brepkit_math::vec::Point2::new(0.0, 0.0),
                    brepkit_math::vec::Vec2::new(1.0, 0.0),
                )
                .unwrap(),
            ),
            contact1: NurbsCurve::new(
                1,
                vec![0.0, 0.0, 1.0, 1.0],
                vec![Point3::new(0.0, 0.0, radius), Point3::new(0.0, 1.0, radius)],
                vec![1.0, 1.0],
            )
            .unwrap(),
            contact2: NurbsCurve::new(
                1,
                vec![0.0, 0.0, 1.0, 1.0],
                vec![Point3::new(radius, 0.0, 0.0), Point3::new(radius, 1.0, 0.0)],
                vec![1.0, 1.0],
            )
            .unwrap(),
            face1: face_xy,
            face2: face_yz,
            sections: vec![
                CircSection {
                    p1: Point3::new(0.0, 0.0, radius),
                    p2: Point3::new(radius, 0.0, 0.0),
                    center: Point3::new(radius, 0.0, radius),
                    radius,
                    uv1: (0.0, 0.0),
                    uv2: (0.0, 0.0),
                    t: 0.0,
                },
                CircSection {
                    p1: Point3::new(0.0, 1.0, radius),
                    p2: Point3::new(radius, 1.0, 0.0),
                    center: Point3::new(radius, 1.0, radius),
                    radius,
                    uv1: (0.0, 0.0),
                    uv2: (0.0, 0.0),
                    t: 1.0,
                },
            ],
        };

        // Stripe along Z edge (between face_xz and face_yz)
        let spine_z = Spine::from_single_edge(&topo, ez).unwrap();
        let stripe_z = Stripe {
            spine: spine_z,
            surface: FaceSurface::Plane {
                normal: Vec3::new(0.0, 0.0, 1.0),
                d: 0.0,
            },
            pcurve1: brepkit_math::curves2d::Curve2D::Line(
                brepkit_math::curves2d::Line2D::new(
                    brepkit_math::vec::Point2::new(0.0, 0.0),
                    brepkit_math::vec::Vec2::new(1.0, 0.0),
                )
                .unwrap(),
            ),
            pcurve2: brepkit_math::curves2d::Curve2D::Line(
                brepkit_math::curves2d::Line2D::new(
                    brepkit_math::vec::Point2::new(0.0, 0.0),
                    brepkit_math::vec::Vec2::new(1.0, 0.0),
                )
                .unwrap(),
            ),
            contact1: NurbsCurve::new(
                1,
                vec![0.0, 0.0, 1.0, 1.0],
                vec![Point3::new(0.0, radius, 0.0), Point3::new(0.0, radius, 1.0)],
                vec![1.0, 1.0],
            )
            .unwrap(),
            contact2: NurbsCurve::new(
                1,
                vec![0.0, 0.0, 1.0, 1.0],
                vec![Point3::new(radius, 0.0, 0.0), Point3::new(radius, 0.0, 1.0)],
                vec![1.0, 1.0],
            )
            .unwrap(),
            face1: face_xz,
            face2: face_yz,
            sections: vec![
                CircSection {
                    p1: Point3::new(0.0, radius, 0.0),
                    p2: Point3::new(radius, 0.0, 0.0),
                    center: Point3::new(radius, radius, 0.0),
                    radius,
                    uv1: (0.0, 0.0),
                    uv2: (0.0, 0.0),
                    t: 0.0,
                },
                CircSection {
                    p1: Point3::new(0.0, radius, 1.0),
                    p2: Point3::new(radius, 0.0, 1.0),
                    center: Point3::new(radius, radius, 1.0),
                    radius,
                    uv1: (0.0, 0.0),
                    uv2: (0.0, 0.0),
                    t: 1.0,
                },
            ],
        };

        let stripes = vec![stripe_x, stripe_y, stripe_z];
        (topo, v000, stripes, solid_id)
    }

    #[test]
    fn classify_corner_three_stripes() {
        let (topo, v000, stripes, _solid_id) = setup_box_corner();
        let ct = classify_corner(v000, &stripes, &topo);
        assert_eq!(ct, CornerType::SphereCap);
    }

    #[test]
    fn classify_corner_one_stripe() {
        let (topo, v000, stripes, _solid_id) = setup_box_corner();
        // Only pass the first stripe — vertex has 1 stripe → None
        let ct = classify_corner(v000, &stripes[..1], &topo);
        assert_eq!(ct, CornerType::None);
    }

    #[test]
    fn classify_corner_two_stripes() {
        let (topo, v000, stripes, _solid_id) = setup_box_corner();
        let ct = classify_corner(v000, &stripes[..2], &topo);
        assert_eq!(ct, CornerType::TwoEdge);
    }

    #[test]
    fn sphere_cap_correct_center() {
        let (mut topo, v000, stripes, _solid_id) = setup_box_corner();
        let radius = 0.2;
        let result = build_sphere_cap(v000, &stripes, radius, &mut topo).unwrap();

        // Sphere center = vertex + R*n1 + R*n2 + R*n3
        // Face normals point inward: (0,0,-1), (0,-1,0), (-1,0,0)
        // Center = (0,0,0) + 0.2*(0,0,-1) + 0.2*(0,-1,0) + 0.2*(-1,0,0)
        //        = (-0.2, -0.2, -0.2)
        let expected_center = Point3::new(-radius, -radius, -radius);

        match &result.surface {
            FaceSurface::Sphere(s) => {
                let dist = (s.center() - expected_center).length();
                assert!(
                    dist < 1e-10,
                    "Sphere center mismatch: got {:?}, expected {:?}",
                    s.center(),
                    expected_center
                );
            }
            other => panic!("Expected Sphere surface, got {:?}", other.type_tag()),
        }
    }

    #[test]
    fn coons_patch_through_contact_points() {
        let (mut topo, v000, stripes, _solid_id) = setup_box_corner();
        let result = build_coons_patch(v000, &stripes, &mut topo).unwrap();

        // The patch should pass through the contact points at the vertex.
        // Collect expected contact points.
        let indices = stripes_at_vertex(v000, &stripes, &topo);
        let contact_pts = collect_contact_points(v000, &stripes, &indices, &topo);

        assert!(
            contact_pts.len() >= 3,
            "Expected at least 3 contact points, got {}",
            contact_pts.len()
        );

        // Verify the surface passes through the contact points.
        // For a triangular patch, corners are at (0,0), (1,0), (0,1)/(1,1).
        match &result.surface {
            FaceSurface::Nurbs(n) => {
                // Control points of a degree-1 patch are the corner points themselves.
                let cps = n.control_points();
                let p00 = cps[0][0];
                let p10 = cps[0][1];
                let p01 = cps[1][0]; // degenerate = p11 for triangle

                // Verify each contact point is one of the control points.
                for cp in &contact_pts[..3] {
                    let min_dist = [
                        (p00 - *cp).length(),
                        (p10 - *cp).length(),
                        (p01 - *cp).length(),
                    ]
                    .into_iter()
                    .fold(f64::INFINITY, f64::min);
                    assert!(
                        min_dist < 1e-10,
                        "Contact point {:?} not found among patch control points",
                        cp
                    );
                }
            }
            other => panic!("Expected Nurbs surface, got {:?}", other.type_tag()),
        }
    }
}
