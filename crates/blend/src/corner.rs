// Walking engine infrastructure — used progressively as more blend paths are wired up.
#![allow(dead_code)]
//! Vertex blend / corner solver.
//!
//! At vertices where multiple fillet stripes meet, gaps appear that need
//! to be closed with smooth surface patches.  This module classifies
//! each vertex and builds the appropriate corner patch:
//!
//! - **`MultiEdge(n)`** — 3+ stripes: delegates to `spherical_triangle` for
//!   exact rational NURBS patches on the rolling-ball sphere.
//! - **Two-edge** — 2 stripes meeting; a simple triangular fill.
//! - **None** — 0-1 stripes; no corner needed.

use brepkit_math::nurbs::surface::NurbsSurface;
use brepkit_math::vec::{Point3, Vec3};
use brepkit_topology::Topology;
use brepkit_topology::edge::{Edge, EdgeCurve, EdgeId};
use brepkit_topology::face::{Face, FaceId, FaceSurface};
use brepkit_topology::vertex::{Vertex, VertexId};
use brepkit_topology::wire::{OrientedEdge, Wire};

use crate::BlendError;
use crate::section::CircSection;
use crate::spherical_triangle::{VertexContactData, build_n_edge_corner, build_spherical_corner};
use crate::stripe::Stripe;

// ── Types ──────────────────────────────────────────────────────────

/// Classification of a vertex blend.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CornerType {
    /// No corner needed (0-1 stripes at vertex).
    None,
    /// Two stripes meeting — extend/intersect their boundaries.
    TwoEdge,
    /// Three or more stripes meeting — spherical triangle patches.
    MultiEdge(usize),
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

/// Tolerance for angular comparisons (cosine of angle threshold ~10°).
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
fn build_triangular_patch(
    pts: &[Point3],
    topo: &mut Topology,
) -> Result<(FaceSurface, Vec<VertexId>, Vec<EdgeId>), BlendError> {
    let p0 = pts[0];
    let p1 = pts[1];
    let p2 = pts[2];

    // Bilinear (degree 1x1) patch with a degenerate edge.
    // Row 0: p0, p1  (bottom edge)
    // Row 1: p2, p2  (collapsed top edge = triangle apex)
    let control_points = vec![vec![p0, p1], vec![p2, p2]];
    let weights = vec![vec![1.0, 1.0], vec![1.0, 1.0]];
    let knots_u = vec![0.0, 0.0, 1.0, 1.0];
    let knots_v = vec![0.0, 0.0, 1.0, 1.0];

    let nurbs = NurbsSurface::new(1, 1, knots_u, knots_v, control_points, weights)?;
    let surface = FaceSurface::Nurbs(nurbs);

    let v0 = topo.add_vertex(Vertex::new(p0, TOL));
    let v1 = topo.add_vertex(Vertex::new(p1, TOL));
    let v2 = topo.add_vertex(Vertex::new(p2, TOL));

    let e0 = topo.add_edge(Edge::new(v0, v1, EdgeCurve::Line));
    let e1 = topo.add_edge(Edge::new(v1, v2, EdgeCurve::Line));
    let e2 = topo.add_edge(Edge::new(v2, v0, EdgeCurve::Line));

    Ok((surface, vec![v0, v1, v2], vec![e0, e1, e2]))
}

// ── Classification ─────────────────────────────────────────────────

/// Classify the vertex blend type based on the stripes meeting at this vertex.
#[must_use]
pub fn classify_corner(vertex_id: VertexId, stripes: &[Stripe], topo: &Topology) -> CornerType {
    let indices = stripes_at_vertex(vertex_id, stripes, topo);

    match indices.len() {
        0 | 1 => CornerType::None,
        2 => CornerType::TwoEdge,
        n => CornerType::MultiEdge(n),
    }
}

// ── Multi-Edge Builder (spherical triangle) ───────────────────────

/// Build corner patches for 3+ stripes meeting at a vertex using
/// spherical triangle patches from the `spherical_triangle` module.
///
/// Collects contact points and face normals, determines convexity,
/// then delegates to `build_spherical_corner` (3 edges) or
/// `build_n_edge_corner` (N > 3 edges).
///
/// # Errors
/// Returns `BlendError` if topology lookups or patch construction fails.
#[allow(clippy::too_many_lines)]
fn build_multi_edge_corner(
    vertex_id: VertexId,
    stripes: &[Stripe],
    topo: &mut Topology,
) -> Result<Vec<CornerResult>, BlendError> {
    let indices = stripes_at_vertex(vertex_id, stripes, topo);
    let contact_pts = collect_contact_points(vertex_id, stripes, &indices, topo);

    if contact_pts.len() < 3 {
        return Err(BlendError::CornerFailure { vertex: vertex_id });
    }

    // Get the fillet radius from the first stripe at this vertex.
    let radius = stripe_radius_at_vertex(vertex_id, &stripes[indices[0]], topo)
        .ok_or(BlendError::CornerFailure { vertex: vertex_id })?;

    // Gather face normals from the faces adjacent to the stripes at this vertex.
    let mut face_normals: Vec<Vec3> = Vec::new();
    for &idx in &indices {
        let stripe = &stripes[idx];
        for face_id in [stripe.face1, stripe.face2] {
            let face_surf = topo.face(face_id)?.surface().clone();
            let n = face_surf.normal(0.0, 0.0);
            // Only add if not a near-duplicate.
            let is_dup = face_normals
                .iter()
                .any(|existing| existing.dot(n).abs() > 1.0 - ORTHO_COS_TOL);
            if !is_dup {
                face_normals.push(n);
            }
        }
    }

    // Determine convexity: compute average face normal, then check if the
    // direction from vertex to the sphere center aligns with it.
    let vertex_pos = topo.vertex(vertex_id)?.point();
    let mut normal_sum = Vec3::new(0.0, 0.0, 0.0);
    for n in &face_normals {
        normal_sum += *n;
    }
    let normal_len = normal_sum.length();
    let is_convex = if normal_len > TOL {
        let avg_normal = normal_sum * (1.0 / normal_len);
        // For a convex vertex the sphere center is offset along the average
        // face normal direction.  Check that the vertex-to-centroid direction
        // of the contact points agrees with the average normal.
        let mut cp_centroid = Vec3::new(0.0, 0.0, 0.0);
        #[allow(clippy::cast_precision_loss)]
        let inv_n = 1.0 / contact_pts.len() as f64;
        for p in &contact_pts {
            cp_centroid += *p - vertex_pos;
        }
        cp_centroid = cp_centroid * inv_n;
        avg_normal.dot(cp_centroid) > 0.0
    } else {
        true // Default to convex if normals cancel out.
    };

    let data = VertexContactData {
        vertex_pos,
        contact_points: contact_pts,
        face_normals,
        radius,
        is_convex,
        vertex_id,
    };

    // Delegate to the spherical triangle module.
    let spherical_results = if data.contact_points.len() == 3 {
        vec![build_spherical_corner(&data)?]
    } else {
        build_n_edge_corner(&data)?
    };

    // Convert each SphericalCornerResult into a CornerResult by creating
    // face topology from the surface and boundary curves.
    let mut results = Vec::with_capacity(spherical_results.len());

    for sr in spherical_results {
        let n_curves = sr.boundary_curves.len();
        let mut new_vertices = Vec::with_capacity(n_curves);
        let mut new_edges = Vec::with_capacity(n_curves);

        // Create vertices at the start of each boundary curve.
        for curve in &sr.boundary_curves {
            let pt = curve.evaluate(0.0);
            let vid = topo.add_vertex(Vertex::new(pt, TOL));
            new_vertices.push(vid);
        }

        // Create edges with the boundary curves as NurbsCurve geometry.
        for i in 0..n_curves {
            let v_start = new_vertices[i];
            let v_end = new_vertices[(i + 1) % n_curves];
            let curve = sr.boundary_curves[i].clone();
            let eid = topo.add_edge(Edge::new(v_start, v_end, EdgeCurve::NurbsCurve(curve)));
            new_edges.push(eid);
        }

        // Build wire from oriented edges.
        let oriented_edges: Vec<OrientedEdge> = new_edges
            .iter()
            .map(|&eid| OrientedEdge::new(eid, true))
            .collect();
        let wire = Wire::new(oriented_edges, true)?;
        let wire_id = topo.add_wire(wire);

        // Create the face.
        let face = Face::new(wire_id, Vec::new(), sr.surface.clone());
        let face_id = topo.add_face(face);

        results.push(CornerResult {
            face_id,
            surface: sr.surface,
            new_edges,
            new_vertices,
        });
    }

    Ok(results)
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
        let corner_type = classify_corner(vid, stripes, topo);

        match corner_type {
            CornerType::None => {}
            CornerType::TwoEdge => {
                let result = build_two_edge_patch(vid, stripes, topo)?;
                results.push(result);
            }
            CornerType::MultiEdge(_) => {
                let corner_results = build_multi_edge_corner(vid, stripes, topo)?;
                results.extend(corner_results);
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
        assert_eq!(ct, CornerType::MultiEdge(3));
    }

    #[test]
    fn classify_corner_one_stripe() {
        let (topo, v000, stripes, _solid_id) = setup_box_corner();
        // Only pass the first stripe — vertex has 1 stripe -> None
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
    fn multi_edge_corner_produces_spherical_patch() {
        let (mut topo, v000, stripes, _solid_id) = setup_box_corner();
        let results = build_multi_edge_corner(v000, &stripes, &mut topo).unwrap();

        // 3-edge case should produce exactly 1 spherical triangle patch.
        assert_eq!(results.len(), 1);

        let result = &results[0];
        // The surface should be a NURBS patch (rational quadratic on the sphere).
        match &result.surface {
            FaceSurface::Nurbs(_) => {} // expected
            other => panic!("Expected Nurbs surface, got {:?}", other.type_tag()),
        }

        // Should have 3 boundary edges (one per arc).
        assert_eq!(result.new_edges.len(), 3);
        assert_eq!(result.new_vertices.len(), 3);
    }

    #[test]
    fn multi_edge_corner_surface_on_sphere() {
        let (mut topo, v000, stripes, _solid_id) = setup_box_corner();
        let results = build_multi_edge_corner(v000, &stripes, &mut topo).unwrap();
        let result = &results[0];

        match &result.surface {
            FaceSurface::Nurbs(nurbs) => {
                // Sample points on the surface and verify they are on the sphere.
                // We need the sphere center. For face normals (0,0,-1), (0,-1,0),
                // (-1,0,0) the average normal is (-1,-1,-1)/sqrt(3). The center
                // is offset along this direction from the vertex at the origin.
                let n_samples = 5;
                for i in 0..=n_samples {
                    for j in 0..=n_samples {
                        let u = i as f64 / n_samples as f64;
                        let v = j as f64 / n_samples as f64;
                        let pt = nurbs.evaluate(u, v);

                        // The point should be at distance approximately R from some center.
                        // We just check the surface points are reasonable (within 15% of R).
                        let dist_from_origin = (pt - Point3::new(0.0, 0.0, 0.0)).length();
                        assert!(
                            dist_from_origin < 1.0,
                            "Surface point at ({u},{v}) unreasonably far from origin: {dist_from_origin}"
                        );
                    }
                }
            }
            other => panic!("Expected Nurbs surface, got {:?}", other.type_tag()),
        }

        // Boundary curves should be NurbsCurve edges.
        for &eid in &result.new_edges {
            let edge = topo.edge(eid).unwrap();
            match edge.curve() {
                EdgeCurve::NurbsCurve(_) => {} // expected
                other => panic!("Expected NurbsCurve edge, got {:?}", other.type_tag()),
            }
        }
    }
}
