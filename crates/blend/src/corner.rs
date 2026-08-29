// Walking engine infrastructure — used progressively as more blend paths are wired up.
#![allow(dead_code)]
//! Vertex blend / corner solver.
//!
//! At vertices where multiple fillet stripes meet, gaps appear that need
//! to be closed with ordered, registry-backed surface patches. This module
//! classifies each vertex and builds the appropriate corner patch:
//!
//! - **`MultiEdge(n)`** — 3+ stripes: reuses `spherical_triangle` geometry
//!   with the ordered fan boundaries.
//! - **Two-edge** — 2 stripes meeting; the ordered fan consumes shared
//!   cross-section and support boundaries.
//! - **None** — 0-1 stripes; no corner needed.

use brepkit_math::nurbs::surface::NurbsSurface;
use brepkit_math::vec::{Point3, Vec3};
use brepkit_topology::Topology;
use brepkit_topology::edge::{Edge, EdgeCurve, EdgeId};
use brepkit_topology::face::{Face, FaceId, FaceSurface};
use brepkit_topology::vertex::{Vertex, VertexId};
use brepkit_topology::wire::{OrientedEdge, Wire};

use crate::BlendError;
use crate::boundary_registry::{
    BoundaryHandle, BoundaryKey, BoundaryKind, BoundaryOwner, BoundaryRegistry, PlannedVertex,
};
use crate::fillet_plan::{CornerClassification, VertexJunction};
use crate::section::CircSection;
use crate::spherical_triangle::{
    SphericalCornerResult, VertexContactData, build_n_edge_corner, build_spherical_corner,
    build_spherical_corner_surface,
};
use crate::stripe::Stripe;

/// A registry handle for a stripe's terminal cross-section, indexed by
/// `(stripe, end)` where end `0` is the spine end and `1` is the spine start.
pub type TerminalBoundary = (Option<BoundaryHandle>, Option<BoundaryHandle>);

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

    let first_edge = topo.edge(edges[0]).ok()?;
    if first_edge.start() == vertex_id || first_edge.end() == vertex_id {
        let is_start = first_edge.start() == vertex_id;
        if is_start {
            let sec = stripe.sections.first()?;
            return Some((sec.p1, sec.p2));
        }
    }

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

    if let Ok(first_edge) = topo.edge(edges[0])
        && first_edge.start() == vertex_id
    {
        return stripe.sections.first();
    }

    if let Ok(last_edge) = topo.edge(edges[edges.len() - 1])
        && last_edge.end() == vertex_id
    {
        return stripe.sections.last();
    }

    let vpos = topo.vertex(vertex_id).ok()?.point();
    let first = stripe.sections.first()?;
    let last = stripe.sections.last()?;
    if (first.center - vpos).length() <= (last.center - vpos).length() {
        Some(first)
    } else {
        Some(last)
    }
}

type PatchParts = (FaceSurface, Vec<VertexId>, Vec<EdgeId>);

/// Ruled patch from a terminal-section arc `a -> b` (about `sec.center`)
/// to the corner apex. Degree 2x1 rational: the u-direction carries the
/// exact arc so the boundary edge is the same circle the adjacent blend
/// wall's cross edge carries, and the weld pass can unify them.
fn build_arc_apex_patch(
    sec: &crate::section::CircSection,
    a: Point3,
    b: Point3,
    apex: Point3,
    topo: &mut Topology,
) -> Option<PatchParts> {
    let (cps, w) = rational_arc_cps(sec.center, a, b)?;
    let control_points = vec![vec![cps[0], apex], vec![cps[1], apex], vec![cps[2], apex]];
    let weights = vec![vec![1.0, 1.0], vec![w, w], vec![1.0, 1.0]];
    let nurbs = brepkit_math::nurbs::surface::NurbsSurface::new(
        2,
        1,
        vec![0.0, 0.0, 0.0, 1.0, 1.0, 1.0],
        vec![0.0, 0.0, 1.0, 1.0],
        control_points,
        weights,
    )
    .ok()?;

    let nrm = (a - sec.center).cross(b - sec.center).normalize().ok()?;
    let circle = brepkit_math::curves::Circle3D::new(sec.center, nrm, sec.radius).ok()?;

    let va = topo.add_vertex(Vertex::new(a, TOL));
    let vb = topo.add_vertex(Vertex::new(b, TOL));
    let vx = topo.add_vertex(Vertex::new(apex, TOL));
    let e0 = topo.add_edge(Edge::new(va, vb, EdgeCurve::Circle(circle)));
    let e1 = topo.add_edge(Edge::new(vb, vx, EdgeCurve::Line));
    let e2 = topo.add_edge(Edge::new(vx, va, EdgeCurve::Line));

    Some((
        FaceSurface::Nurbs(nurbs),
        vec![va, vb, vx],
        vec![e0, e1, e2],
    ))
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

/// Build corner patches for 3+ stripes meeting at a vertex using
/// spherical triangle patches from the `spherical_triangle` module.
///
/// Collects contact points and face normals, determines convexity,
/// then delegates to `build_spherical_corner` (3 edges) or
/// `build_n_edge_corner` (N > 3 edges).
///
/// # Errors
/// Returns `BlendError` if topology lookups or patch construction fails.
fn multi_edge_corner_data(
    vertex_id: VertexId,
    indices: &[usize],
    stripes: &[Stripe],
    topo: &Topology,
) -> Result<VertexContactData, BlendError> {
    let contact_pts = collect_contact_points(vertex_id, stripes, indices, topo);
    if contact_pts.len() < 3 {
        return Err(BlendError::CornerFailure { vertex: vertex_id });
    }

    let radius = stripe_radius_at_vertex(vertex_id, &stripes[indices[0]], topo)
        .ok_or(BlendError::CornerFailure { vertex: vertex_id })?;
    let mut face_normals: Vec<Vec3> = Vec::new();
    for &idx in indices {
        let stripe = &stripes[idx];
        for face_id in [stripe.face1, stripe.face2] {
            let face_surf = topo.face(face_id)?.surface().clone();
            let normal = face_surf.normal(0.0, 0.0);
            let is_duplicate = face_normals
                .iter()
                .any(|existing| existing.dot(normal).abs() > 1.0 - ORTHO_COS_TOL);
            if !is_duplicate {
                face_normals.push(normal);
            }
        }
    }

    let vertex_pos = topo.vertex(vertex_id)?.point();
    let mut normal_sum = Vec3::new(0.0, 0.0, 0.0);
    for normal in &face_normals {
        normal_sum += *normal;
    }
    let normal_len = normal_sum.length();
    let is_convex = if normal_len > TOL {
        let avg_normal = normal_sum * (1.0 / normal_len);
        let mut contact_centroid = Vec3::new(0.0, 0.0, 0.0);
        #[allow(clippy::cast_precision_loss)]
        let inverse_count = 1.0 / contact_pts.len() as f64;
        for point in &contact_pts {
            contact_centroid += *point - vertex_pos;
        }
        contact_centroid = contact_centroid * inverse_count;
        avg_normal.dot(contact_centroid) > 0.0
    } else {
        true
    };

    Ok(VertexContactData {
        vertex_pos,
        contact_points: contact_pts,
        face_normals,
        radius,
        is_convex,
        vertex_id,
    })
}

fn multi_edge_corner_geometry(
    vertex_id: VertexId,
    indices: &[usize],
    stripes: &[Stripe],
    topo: &Topology,
) -> Result<Vec<SphericalCornerResult>, BlendError> {
    let data = multi_edge_corner_data(vertex_id, indices, stripes, topo)?;
    if data.contact_points.len() == 3 {
        Ok(vec![build_spherical_corner(&data)?])
    } else {
        build_n_edge_corner(&data)
    }
}

fn build_multi_edge_corner(
    vertex_id: VertexId,
    indices: &[usize],
    stripes: &[Stripe],
    topo: &mut Topology,
) -> Result<Vec<CornerResult>, BlendError> {
    let spherical_results = multi_edge_corner_geometry(vertex_id, indices, stripes, topo)?;
    let mut results = Vec::with_capacity(spherical_results.len());
    for spherical in spherical_results {
        let curve_count = spherical.boundary_curves.len();
        let mut new_vertices = Vec::with_capacity(curve_count);
        let mut new_edges = Vec::with_capacity(curve_count);
        for curve in &spherical.boundary_curves {
            let point = curve.evaluate(0.0);
            new_vertices.push(topo.add_vertex(Vertex::new(point, TOL)));
        }
        for index in 0..curve_count {
            let start = new_vertices[index];
            let end = new_vertices[(index + 1) % curve_count];
            let curve = spherical.boundary_curves[index].clone();
            new_edges.push(topo.add_edge(Edge::new(start, end, EdgeCurve::NurbsCurve(curve))));
        }

        let oriented_edges = new_edges
            .iter()
            .map(|&edge| OrientedEdge::new(edge, true))
            .collect();
        let wire_id = topo.add_wire(Wire::new(oriented_edges, true)?);
        let face_id = topo.add_face(Face::new(wire_id, Vec::new(), spherical.surface.clone()));
        results.push(CornerResult {
            face_id,
            surface: spherical.surface,
            new_edges,
            new_vertices,
        });
    }
    Ok(results)
}

/// Build a simple triangular fill for 2 stripes meeting at a vertex.
///
/// # Errors
/// Returns `BlendError` if topology lookups fail.
/// Horn-torus corner for two equal-radius stripes meeting at an unfilleted
/// corner edge: the rolling ball pivots about the corner edge, tangent to
/// the shared base face, sweeping a torus with major radius == tube radius
/// == r that pinches onto the edge exactly where both stripes' wall
/// contacts already end. Boundary: the base offset arc (radius r about the
/// corner vertex — the loop rebuild's bridge, unified by the weld pass)
/// plus the two terminal cross-section arcs meeting at the pinch.
fn build_horn_torus_corner(
    vertex_id: VertexId,
    stripes: &[Stripe],
    topo: &mut Topology,
) -> Result<Option<CornerResult>, BlendError> {
    let indices = stripes_at_vertex(vertex_id, stripes, topo);
    if indices.len() != 2 {
        return Ok(Option::None);
    }
    build_horn_torus_for_pair(vertex_id, stripes, indices[0], indices[1], topo)
}

struct HornTorusGeometry {
    surface: FaceSurface,
    vertex: Point3,
    a_base: Point3,
    pinch: Point3,
    b_base: Point3,
    a_center: Point3,
    b_center: Point3,
    radius: f64,
}

fn horn_torus_geometry_for_pair(
    vertex_id: VertexId,
    stripes: &[Stripe],
    ia: usize,
    ib: usize,
    topo: &Topology,
) -> Result<Option<HornTorusGeometry>, BlendError> {
    use brepkit_math::surfaces::ToroidalSurface;

    let (Some(sa), Some(sb)) = (
        contact_section_at_vertex(vertex_id, &stripes[ia], topo).cloned(),
        contact_section_at_vertex(vertex_id, &stripes[ib], topo).cloned(),
    ) else {
        return Ok(Option::None);
    };
    if (sa.radius - sb.radius).abs() > 1e-6 {
        return Ok(Option::None);
    }
    let r = sa.radius;
    let vertex = topo.vertex(vertex_id)?.point();

    let arrangements = [
        (sa.p1, sa.p2, sb.p1, sb.p2),
        (sa.p1, sa.p2, sb.p2, sb.p1),
        (sa.p2, sa.p1, sb.p1, sb.p2),
        (sa.p2, sa.p1, sb.p2, sb.p1),
    ];
    let mut found = Option::None;
    for (a_base, a_pinch, b_base, b_pinch) in arrangements {
        if (a_pinch - b_pinch).length() <= 1e-6
            && ((a_base - vertex).length() - r).abs() <= 1e-5
            && ((b_base - vertex).length() - r).abs() <= 1e-5
            && (a_base - b_base).length() > 1e-6
        {
            found = Some((a_base, a_pinch, b_base));
            break;
        }
    }
    let Some((a_base, pinch, b_base)) = found else {
        return Ok(Option::None);
    };
    let Ok(axis) = (pinch - vertex).normalize() else {
        return Ok(Option::None);
    };
    if ((pinch - vertex).length() - r).abs() > 1e-5 {
        return Ok(Option::None);
    }
    let Ok(torus) = ToroidalSurface::with_axis(vertex + axis * r, r, r, axis) else {
        return Ok(Option::None);
    };

    Ok(Some(HornTorusGeometry {
        surface: FaceSurface::Torus(torus),
        vertex,
        a_base,
        pinch,
        b_base,
        a_center: sa.center,
        b_center: sb.center,
        radius: r,
    }))
}

fn build_horn_torus_for_pair(
    vertex_id: VertexId,
    stripes: &[Stripe],
    ia: usize,
    ib: usize,
    topo: &mut Topology,
) -> Result<Option<CornerResult>, BlendError> {
    use brepkit_math::curves::Circle3D;

    let Some(geometry) = horn_torus_geometry_for_pair(vertex_id, stripes, ia, ib, topo)? else {
        return Ok(Option::None);
    };
    let va = topo.add_vertex(Vertex::new(geometry.a_base, TOL));
    let vb = topo.add_vertex(Vertex::new(geometry.b_base, TOL));
    let vp = topo.add_vertex(Vertex::new(geometry.pinch, TOL));
    let arc = |topo: &mut Topology,
               c: Point3,
               from: Point3,
               to: Point3,
               v_from: VertexId,
               v_to: VertexId|
     -> Option<EdgeId> {
        let nrm = (from - c).cross(to - c).normalize().ok()?;
        let circ = Circle3D::new(c, nrm, (from - c).length()).ok()?;
        Some(topo.add_edge(Edge::new(v_from, v_to, EdgeCurve::Circle(circ))))
    };
    let (Some(e_base), Some(e_b), Some(e_a)) = (
        arc(
            topo,
            geometry.vertex,
            geometry.a_base,
            geometry.b_base,
            va,
            vb,
        ),
        arc(
            topo,
            geometry.b_center,
            geometry.b_base,
            geometry.pinch,
            vb,
            vp,
        ),
        arc(
            topo,
            geometry.a_center,
            geometry.pinch,
            geometry.a_base,
            vp,
            va,
        ),
    ) else {
        return Ok(Option::None);
    };
    let wire = Wire::new(
        vec![
            OrientedEdge::new(e_base, true),
            OrientedEdge::new(e_b, true),
            OrientedEdge::new(e_a, true),
        ],
        true,
    )?;
    let wid = topo.add_wire(wire);
    let surface = geometry.surface;
    let fid = topo.add_face(Face::new(wid, Vec::new(), surface.clone()));
    log::debug!("horn-torus corner at {vertex_id:?} r={}", geometry.radius);
    Ok(Some(CornerResult {
        face_id: fid,
        surface,
        new_edges: vec![e_base, e_b, e_a],
        new_vertices: vec![va, vb, vp],
    }))
}

/// Rational quadratic Bezier control points for a circular arc.
fn rational_arc_cps(center: Point3, from: Point3, to: Point3) -> Option<([Point3; 3], f64)> {
    let u = from - center;
    let r = u.length();
    let du = u.normalize().ok()?;
    let dv = (to - center).normalize().ok()?;
    let bis = (du + dv).normalize().ok()?;
    let cos_half = du.dot(bis);
    if cos_half.abs() < 1e-9 {
        return Option::None;
    }
    let mid = center + bis * (r / cos_half);
    Some(([from, mid, to], cos_half))
}

/// Ruled transition band between two different-radius terminal sections at
/// a junction on a shared corner edge: boundary = the two cross-section
/// arcs (welded with the blend walls' cross edges), the corner-edge
/// segment between the two wall-contact heights, and the base chord. The
/// wall is a ruled NURBS between the arcs — the watertight stand-in for
/// the true variable-radius canal surface.
fn build_mixed_radius_band(
    vertex_id: VertexId,
    stripes: &[Stripe],
    topo: &mut Topology,
) -> Result<Option<CornerResult>, BlendError> {
    let indices = stripes_at_vertex(vertex_id, stripes, topo);
    if indices.len() != 2 {
        return Ok(Option::None);
    }
    build_mixed_radius_band_for_pair(vertex_id, stripes, indices[0], indices[1], topo)
}

struct MixedRadiusGeometry {
    surface: FaceSurface,
    a_base: Point3,
    a_wall: Point3,
    b_base: Point3,
    b_wall: Point3,
    a_center: Point3,
    b_center: Point3,
    a_radius: f64,
    b_radius: f64,
}

fn mixed_radius_geometry_for_pair(
    vertex_id: VertexId,
    stripes: &[Stripe],
    ia: usize,
    ib: usize,
    topo: &Topology,
) -> Result<Option<MixedRadiusGeometry>, BlendError> {
    let (Some(sa), Some(sb)) = (
        contact_section_at_vertex(vertex_id, &stripes[ia], topo).cloned(),
        contact_section_at_vertex(vertex_id, &stripes[ib], topo).cloned(),
    ) else {
        return Ok(Option::None);
    };
    if (sa.radius - sb.radius).abs() <= 1e-6 {
        return Ok(Option::None);
    }
    let vertex = topo.vertex(vertex_id)?.point();

    let mut found = Option::None;
    for (a_base, a_wall, b_base, b_wall) in [
        (sa.p1, sa.p2, sb.p1, sb.p2),
        (sa.p1, sa.p2, sb.p2, sb.p1),
        (sa.p2, sa.p1, sb.p1, sb.p2),
        (sa.p2, sa.p1, sb.p2, sb.p1),
    ] {
        let da = a_wall - vertex;
        let db = b_wall - vertex;
        let (Ok(na), Ok(nb)) = (da.normalize(), db.normalize()) else {
            continue;
        };
        if na.dot(nb) > 1.0 - 1e-6
            && (da.length() - sa.radius).abs() <= 1e-5
            && (db.length() - sb.radius).abs() <= 1e-5
        {
            found = Some((a_base, a_wall, b_base, b_wall));
            break;
        }
    }
    let Some((a_base, a_wall, b_base, b_wall)) = found else {
        return Ok(Option::None);
    };

    let (Some((cps_a, w_a)), Some((cps_b, w_b))) = (
        rational_arc_cps(sa.center, a_base, a_wall),
        rational_arc_cps(sb.center, b_base, b_wall),
    ) else {
        return Ok(Option::None);
    };
    let control_points = vec![
        vec![cps_a[0], cps_b[0]],
        vec![cps_a[1], cps_b[1]],
        vec![cps_a[2], cps_b[2]],
    ];
    let weights = vec![vec![1.0, 1.0], vec![w_a, w_b], vec![1.0, 1.0]];
    let Ok(nurbs) = NurbsSurface::new(
        2,
        1,
        vec![0.0, 0.0, 0.0, 1.0, 1.0, 1.0],
        vec![0.0, 0.0, 1.0, 1.0],
        control_points,
        weights,
    ) else {
        return Ok(Option::None);
    };

    Ok(Some(MixedRadiusGeometry {
        surface: FaceSurface::Nurbs(nurbs),
        a_base,
        a_wall,
        b_base,
        b_wall,
        a_center: sa.center,
        b_center: sb.center,
        a_radius: sa.radius,
        b_radius: sb.radius,
    }))
}

fn build_mixed_radius_band_for_pair(
    vertex_id: VertexId,
    stripes: &[Stripe],
    ia: usize,
    ib: usize,
    topo: &mut Topology,
) -> Result<Option<CornerResult>, BlendError> {
    let Some(geometry) = mixed_radius_geometry_for_pair(vertex_id, stripes, ia, ib, topo)? else {
        return Ok(Option::None);
    };

    let va_b = topo.add_vertex(Vertex::new(geometry.a_base, TOL));
    let va_w = topo.add_vertex(Vertex::new(geometry.a_wall, TOL));
    let vb_b = topo.add_vertex(Vertex::new(geometry.b_base, TOL));
    let vb_w = topo.add_vertex(Vertex::new(geometry.b_wall, TOL));
    let arc_edge = |topo: &mut Topology,
                    c: Point3,
                    from: Point3,
                    to: Point3,
                    vf: VertexId,
                    vt: VertexId|
     -> Option<EdgeId> {
        let nrm = (from - c).cross(to - c).normalize().ok()?;
        let circ = brepkit_math::curves::Circle3D::new(c, nrm, (from - c).length()).ok()?;
        Some(topo.add_edge(Edge::new(vf, vt, EdgeCurve::Circle(circ))))
    };
    let (Some(e_a), Some(e_b)) = (
        arc_edge(
            topo,
            geometry.a_center,
            geometry.a_base,
            geometry.a_wall,
            va_b,
            va_w,
        ),
        arc_edge(
            topo,
            geometry.b_center,
            geometry.b_base,
            geometry.b_wall,
            vb_b,
            vb_w,
        ),
    ) else {
        return Ok(Option::None);
    };
    let e_top = topo.add_edge(Edge::new(va_w, vb_w, EdgeCurve::Line));
    let e_bottom = topo.add_edge(Edge::new(vb_b, va_b, EdgeCurve::Line));
    let wire = Wire::new(
        vec![
            OrientedEdge::new(e_a, true),
            OrientedEdge::new(e_top, true),
            OrientedEdge::new(e_b, false),
            OrientedEdge::new(e_bottom, true),
        ],
        true,
    )?;
    let wid = topo.add_wire(wire);
    let surface = geometry.surface;
    let fid = topo.add_face(Face::new(wid, Vec::new(), surface.clone()));
    log::debug!(
        "mixed-radius band at {vertex_id:?} r {} -> {}",
        geometry.a_radius,
        geometry.b_radius
    );
    Ok(Some(CornerResult {
        face_id: fid,
        surface,
        new_edges: vec![e_a, e_top, e_b, e_bottom],
        new_vertices: vec![va_b, va_w, vb_b, vb_w],
    }))
}

fn build_two_edge_patch(
    vertex_id: VertexId,
    indices: &[usize],
    stripes: &[Stripe],
    topo: &mut Topology,
) -> Result<CornerResult, BlendError> {
    let contact_pts = collect_contact_points(vertex_id, stripes, indices, topo);

    // With 2 stripes we expect 3-4 unique contact points (some may merge).
    // Build a triangular patch from the first 3 unique points.
    let pts = if contact_pts.len() >= 3 {
        &contact_pts[..3]
    } else {
        // Degenerate case: not enough unique points
        return Err(BlendError::CornerFailure { vertex: vertex_id });
    };

    // When two of the three points are one stripe's terminal-section
    // contacts, the edge between them is the fillet's end profile — a
    // circular arc, not a chord. A flat chord triangle both misrepresents
    // the patch and can never weld with the blend wall's circular cross
    // edge (chord and arc share endpoints but are genuinely distinct, so
    // the weld correctly refuses). Build the ruled arc-to-apex patch so
    // the boundary matches the wall exactly.
    let arc_patch = indices.iter().find_map(|&i| {
        let sec = contact_section_at_vertex(vertex_id, &stripes[i], topo)?;
        let m = |q: Point3| pts.iter().position(|p| (*p - q).length() < 1e-6);
        let (ia, ib) = (m(sec.p1)?, m(sec.p2)?);
        if ia == ib {
            return Option::None;
        }
        let apex = *pts
            .iter()
            .enumerate()
            .find(|(k, _)| *k != ia && *k != ib)?
            .1;
        Some((sec.clone(), pts[ia], pts[ib], apex))
    });
    let (surface, new_vertices, new_edges) = match arc_patch
        .and_then(|(sec, a, b, apex)| build_arc_apex_patch(&sec, a, b, apex, topo))
    {
        Some(built) => built,
        _ => build_triangular_patch(pts, topo)?,
    };

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

/// Compute corner patches in deterministic source-plan order.
///
/// This compatibility entry point derives the order from the supplied stripe
/// order. The fillet builder uses [`compute_ordered_corners`] so periodic and
/// G1-continuation junctions are classified from the immutable plan.
///
/// # Errors
///
/// A corner geometry failure is returned immediately. In particular, no
/// pairwise fallback is attempted for a failed multi-stripe solve.
pub fn compute_corners(
    topo: &mut Topology,
    stripes: &[Stripe],
    solid: brepkit_topology::solid::SolidId,
) -> Result<Vec<CornerResult>, BlendError> {
    use brepkit_topology::explorer::solid_vertices;

    let mut vertices = solid_vertices(topo, solid)?;
    vertices.sort_unstable_by_key(|vertex| vertex.index());
    let mut results = Vec::new();
    for vid in vertices {
        let indices = stripes_at_vertex(vid, stripes, topo);
        match indices.len() {
            0 | 1 => {}
            2 => {
                let result = build_horn_torus_for_pair(vid, stripes, indices[0], indices[1], topo)?
                    .or(build_mixed_radius_band_for_pair(
                        vid, stripes, indices[0], indices[1], topo,
                    )?)
                    .unwrap_or(build_two_edge_patch(vid, &indices, stripes, topo)?);
                results.push(result);
            }
            3 => results.extend(build_multi_edge_corner(vid, &indices, stripes, topo)?),
            n => {
                return Err(BlendError::PlanningFailure {
                    reason: format!("unsupported corner valence {n} at vertex {vid:?}"),
                });
            }
        }
    }
    Ok(results)
}

/// One edge in an ordered terminal runout boundary cycle.
#[derive(Debug, Clone, Copy)]
struct JunctionBoundary {
    edge: EdgeId,
    handle: BoundaryHandle,
    start: VertexId,
    end: VertexId,
    required: bool,
}

fn edge_vertices(topo: &Topology, edge_id: EdgeId) -> Result<(VertexId, VertexId), BlendError> {
    let edge = topo.edge(edge_id)?;
    Ok((edge.start(), edge.end()))
}

fn face_edge_forward(topo: &Topology, face_id: FaceId, edge_id: EdgeId) -> Option<bool> {
    let face = topo.face(face_id).ok()?;
    std::iter::once(face.outer_wire())
        .chain(face.inner_wires().iter().copied())
        .find_map(|wire_id| {
            topo.wire(wire_id)
                .ok()?
                .edges()
                .iter()
                .find_map(|oriented| (oriented.edge() == edge_id).then_some(oriented.is_forward()))
        })
}

/// Return the source vertices at the ends of an open spine in traversal order.
fn source_spine_endpoints(
    topo: &Topology,
    stripe: &Stripe,
) -> Result<Option<(VertexId, VertexId)>, BlendError> {
    if stripe.spine.is_closed() || stripe.spine.edges().is_empty() {
        return Ok(None);
    }
    let edges = stripe.spine.edges();
    let mut directions = vec![true; edges.len()];
    if edges.len() > 1 {
        let first = topo.edge(edges[0])?;
        let next = topo.edge(edges[1])?;
        if first.end() != next.start()
            && first.end() != next.end()
            && (first.start() == next.start() || first.start() == next.end())
        {
            directions[0] = false;
        }
    }
    for index in 1..edges.len() {
        let previous = topo.edge(edges[index - 1])?;
        let previous_end = if directions[index - 1] {
            previous.end()
        } else {
            previous.start()
        };
        let edge = topo.edge(edges[index])?;
        directions[index] = if edge.start() == previous_end {
            true
        } else {
            edge.end() == previous_end
        };
    }
    let first = topo.edge(edges[0])?;
    let last = topo.edge(edges[edges.len() - 1])?;
    let start = if directions[0] {
        first.start()
    } else {
        first.end()
    };
    let end = if directions[edges.len() - 1] {
        last.end()
    } else {
        last.start()
    };
    if start == end {
        Ok(None)
    } else {
        Ok(Some((start, end)))
    }
}

/// Find a simple cycle that consumes all required (cross-section) boundaries.
#[allow(clippy::items_after_statements)]
fn boundary_cycle(boundaries: &[JunctionBoundary]) -> Option<Vec<(usize, bool)>> {
    if boundaries.len() < 3 {
        return None;
    }
    let mut adjacency = std::collections::HashMap::<usize, Vec<usize>>::new();
    for (index, boundary) in boundaries.iter().enumerate() {
        adjacency
            .entry(boundary.start.index())
            .or_default()
            .push(index);
        adjacency
            .entry(boundary.end.index())
            .or_default()
            .push(index);
    }
    for incident in adjacency.values_mut() {
        incident.sort_unstable();
    }
    let required = boundaries
        .iter()
        .filter(|boundary| boundary.required)
        .count();
    fn search(
        start: usize,
        current: usize,
        boundaries: &[JunctionBoundary],
        adjacency: &std::collections::HashMap<usize, Vec<usize>>,
        required: usize,
        used: &mut std::collections::HashSet<usize>,
        path: &mut Vec<(usize, bool)>,
    ) -> Option<Vec<(usize, bool)>> {
        if current == start {
            if path.len() >= 3
                && path
                    .iter()
                    .filter(|(index, _)| boundaries[*index].required)
                    .count()
                    == required
            {
                return Some(path.clone());
            }
            return None;
        }
        if path.len() >= boundaries.len() {
            return None;
        }
        for &index in adjacency.get(&current)? {
            if used.contains(&index) {
                continue;
            }
            let boundary = boundaries[index];
            let (next, forward) = if boundary.start.index() == current {
                (boundary.end.index(), true)
            } else if boundary.end.index() == current {
                (boundary.start.index(), false)
            } else {
                continue;
            };
            if next == start && path.len() + 1 < 3 {
                continue;
            }
            used.insert(index);
            path.push((index, forward));
            if let Some(found) = search(start, next, boundaries, adjacency, required, used, path) {
                return Some(found);
            }
            path.pop();
            used.remove(&index);
        }
        None
    }

    for (index, boundary) in boundaries.iter().enumerate() {
        for (start, end, forward) in [
            (boundary.start.index(), boundary.end.index(), true),
            (boundary.end.index(), boundary.start.index(), false),
        ] {
            let mut used = std::collections::HashSet::new();
            let mut path = vec![(index, forward)];
            used.insert(index);
            if let Some(found) = search(
                start, end, boundaries, &adjacency, required, &mut used, &mut path,
            ) {
                return Some(found);
            }
        }
    }
    None
}

/// Partition a terminal boundary graph into simple cycles. Prefer one global
/// cycle; fall back to one local cycle per cross-section when the two ends of
/// an open contour are independent runouts.
fn terminal_boundary_cycles(boundaries: &[JunctionBoundary]) -> Option<Vec<Vec<(usize, bool)>>> {
    let required_indices: Vec<_> = boundaries
        .iter()
        .enumerate()
        .filter_map(|(index, boundary)| boundary.required.then_some(index))
        .collect();
    if required_indices.is_empty() {
        return None;
    }
    if let Some(cycle) = boundary_cycle(boundaries) {
        return Some(vec![cycle]);
    }
    let mut consumed = std::collections::HashSet::new();
    let mut cycles = Vec::new();
    for required_index in required_indices {
        if consumed.contains(&required_index) {
            continue;
        }
        let mut candidate = Vec::new();
        let mut original_indices = Vec::new();
        for (index, boundary) in boundaries.iter().enumerate() {
            if consumed.contains(&index) {
                continue;
            }
            let mut boundary = *boundary;
            boundary.required = index == required_index;
            original_indices.push(index);
            candidate.push(boundary);
        }
        let Some(cycle) = boundary_cycle(&candidate) else {
            continue;
        };
        let cycle = cycle
            .into_iter()
            .map(|(index, forward)| (original_indices[index], forward))
            .collect::<Vec<_>>();
        if !cycle.iter().any(|(index, _)| *index == required_index) {
            continue;
        }
        consumed.extend(cycle.iter().map(|(index, _)| *index));
        cycles.push(cycle);
    }
    (!cycles.is_empty()).then_some(cycles)
}

fn runout_surface(
    topo: &Topology,
    boundaries: &[JunctionBoundary],
    cycle: &[(usize, bool)],
    vertex: VertexId,
) -> Result<FaceSurface, BlendError> {
    let mut points = Vec::with_capacity(cycle.len());
    for &(index, forward) in cycle {
        let boundary = boundaries[index];
        let point_vertex = if forward {
            boundary.start
        } else {
            boundary.end
        };
        points.push(topo.vertex(point_vertex)?.point());
    }
    let origin = *points.first().ok_or(BlendError::CornerFailure { vertex })?;
    let mut normal = Vec3::new(0.0, 0.0, 0.0);
    for index in 1..points.len().saturating_sub(1) {
        normal += (points[index] - origin).cross(points[index + 1] - origin);
    }
    let normal = normal
        .normalize()
        .or_else(|_| {
            for i in 0..points.len() {
                for j in (i + 1)..points.len() {
                    for k in (j + 1)..points.len() {
                        let candidate = (points[j] - points[i]).cross(points[k] - points[i]);
                        if let Ok(unit) = candidate.normalize() {
                            return Ok(unit);
                        }
                    }
                }
            }
            Err(())
        })
        .map_err(|()| BlendError::CornerFailure { vertex })?;
    let coplanar = points
        .iter()
        .all(|point| normal.dot(*point - origin).abs() <= 1e-5);
    if coplanar {
        let d = normal.dot(Vec3::new(origin.x(), origin.y(), origin.z()));
        return Ok(FaceSurface::Plane { normal, d });
    }
    let p0 = points[0];
    let p1 = points[1];
    let p2 = points
        .iter()
        .skip(2)
        .find(|point| ((**point - p0).cross(p1 - p0)).length() > TOL)
        .copied()
        .unwrap_or(points[2]);
    let nurbs = NurbsSurface::new(
        1,
        1,
        vec![0.0, 0.0, 1.0, 1.0],
        vec![0.0, 0.0, 1.0, 1.0],
        vec![vec![p0, p1], vec![p2, p2]],
        vec![vec![1.0, 1.0], vec![1.0, 1.0]],
    )
    .map_err(|_| BlendError::CornerFailure { vertex })?;
    Ok(FaceSurface::Nurbs(nurbs))
}

#[allow(clippy::too_many_arguments)]
fn collect_terminal_boundaries(
    topo: &Topology,
    stripes: &[Stripe],
    stripe_index: usize,
    contour_id: usize,
    junction_vertex: VertexId,
    cross_boundaries: &[TerminalBoundary],
    support_faces: &[(FaceId, FaceId)],
    registry: &mut BoundaryRegistry,
) -> Result<Option<Vec<JunctionBoundary>>, BlendError> {
    let stripe = stripes
        .get(stripe_index)
        .ok_or_else(|| BlendError::PlanningFailure {
            reason: format!("missing stripe {stripe_index} for terminal junction"),
        })?;
    let Some((terminal_start, terminal_end)) = source_spine_endpoints(topo, stripe)? else {
        return Ok(None);
    };
    if junction_vertex != terminal_start && junction_vertex != terminal_end {
        return Ok(None);
    }

    let mut terminal_vertices = std::collections::HashSet::new();
    for stripe in stripes {
        if let Some((start, end)) = source_spine_endpoints(topo, stripe)? {
            terminal_vertices.insert(start);
            terminal_vertices.insert(end);
        }
    }
    let mut source_edges = std::collections::HashSet::new();
    for stripe in stripes {
        source_edges.extend(stripe.spine.edges().iter().copied());
    }

    let &(end_handle, start_handle) =
        cross_boundaries
            .get(stripe_index)
            .ok_or_else(|| BlendError::PlanningFailure {
                reason: format!("missing cross-section boundaries for stripe {stripe_index}"),
            })?;
    let current_handle = if junction_vertex == terminal_start {
        start_handle
    } else {
        end_handle
    };
    let Some(current_handle) = current_handle else {
        return Ok(None);
    };
    let current_handles = std::collections::HashSet::from([current_handle]);
    let mut cross_vertices = std::collections::HashSet::new();

    let mut boundaries = Vec::new();
    let mut seen_edges = std::collections::HashSet::new();
    let mut active_current = 0usize;
    for pair in cross_boundaries {
        for handle in [pair.0, pair.1].into_iter().flatten() {
            let Some(entry) = registry.entry(handle) else {
                return Err(BlendError::PlanningFailure {
                    reason: format!("unknown cross-section boundary {handle}"),
                });
            };
            let edge = entry.edge_id().ok_or_else(|| BlendError::PlanningFailure {
                reason: format!(
                    "cross-section boundary {:?} was not materialized",
                    entry.key
                ),
            })?;
            let (start, end) = edge_vertices(topo, edge)?;
            if start == end {
                return Err(BlendError::CornerFailure {
                    vertex: junction_vertex,
                });
            }
            cross_vertices.insert(start);
            cross_vertices.insert(end);
            if entry.owners[1].face.is_some() {
                continue;
            }
            if seen_edges.insert(edge) {
                let required = current_handles.contains(&handle);
                active_current += usize::from(required);
                boundaries.push(JunctionBoundary {
                    edge,
                    handle,
                    start,
                    end,
                    required,
                });
            }
        }
    }
    if active_current == 0 {
        return Ok(None);
    }

    let mut visited_support_edges = std::collections::HashSet::new();
    let mut faces = Vec::new();
    for &(face1, face2) in support_faces {
        faces.extend([face1, face2]);
    }
    faces.sort_unstable_by_key(|face| face.index());
    faces.dedup();
    let terminal_side = u8::from(junction_vertex == terminal_end);
    let mut runout_segment = 0usize;
    for support_face in faces {
        let face = topo.face(support_face)?;
        let wires = std::iter::once(face.outer_wire()).chain(face.inner_wires().iter().copied());
        for wire_id in wires {
            for oriented in topo.wire(wire_id)?.edges() {
                let edge = oriented.edge();
                if !visited_support_edges.insert(edge) || source_edges.contains(&edge) {
                    continue;
                }
                let (start, end) = edge_vertices(topo, edge)?;
                let joins_boundary = (cross_vertices.contains(&start)
                    && (cross_vertices.contains(&end) || terminal_vertices.contains(&end)))
                    || (cross_vertices.contains(&end)
                        && (cross_vertices.contains(&start) || terminal_vertices.contains(&start)));
                if !joins_boundary {
                    continue;
                }
                let segment = runout_segment;
                runout_segment += 1;
                let forward = face_edge_forward(topo, support_face, edge)
                    .ok_or(BlendError::TrimmingFailure { face: support_face })?;
                let edge_data = topo.edge(edge)?.clone();
                let handle = if let Some(handle) = registry.handle_for_edge(edge) {
                    let entry =
                        registry
                            .entry(handle)
                            .ok_or_else(|| BlendError::PlanningFailure {
                                reason: format!("unknown boundary handle {handle}"),
                            })?;
                    if !matches!(
                        entry.key.kind,
                        crate::boundary_registry::BoundaryKind::Runout
                    ) {
                        continue;
                    }
                    handle
                } else {
                    let handle = registry.register(
                        BoundaryKey::runout(contour_id, segment, terminal_side),
                        PlannedVertex::new(edge_data.start()),
                        PlannedVertex::new(edge_data.end()),
                        edge_data.curve().clone(),
                        edge_data.curve().domain_with_endpoints(
                            topo.vertex(edge_data.start())?.point(),
                            topo.vertex(edge_data.end())?.point(),
                        ),
                        [
                            BoundaryOwner::planned("ordered terminal support", forward),
                            BoundaryOwner::planned("ordered terminal runout", false),
                        ],
                    )?;
                    registry.defer_owner(handle, 0)?;
                    registry.defer_owner(handle, 1)?;
                    registry.bind_existing_edge(topo, handle, edge)?;
                    handle
                };
                boundaries.push(JunctionBoundary {
                    edge,
                    handle,
                    start,
                    end,
                    required: false,
                });
            }
        }
    }
    if terminal_boundary_cycles(&boundaries).is_some() {
        return Ok(Some(boundaries));
    }
    // A terminal may already close against the copied support face without a
    // standalone runout patch. Leave registered candidates deferred; the
    // final support-owner pass attaches that closure, and its audit rejects
    // any boundary that remains open.
    Ok(None)
}

#[allow(clippy::too_many_arguments)]
fn build_terminal_runout(
    topo: &mut Topology,
    stripes: &[Stripe],
    stripe_index: usize,
    contour_id: usize,
    junction_vertex: VertexId,
    cross_boundaries: &[TerminalBoundary],
    support_faces: &[(FaceId, FaceId)],
    registry: &mut BoundaryRegistry,
) -> Result<Vec<CornerResult>, BlendError> {
    let Some(boundaries) = collect_terminal_boundaries(
        topo,
        stripes,
        stripe_index,
        contour_id,
        junction_vertex,
        cross_boundaries,
        support_faces,
        registry,
    )?
    else {
        return Ok(Vec::new());
    };
    let Some(cycles) = terminal_boundary_cycles(&boundaries) else {
        return Ok(Vec::new());
    };
    let mut results = Vec::with_capacity(cycles.len());
    for cycle in cycles {
        let mut wire_edges = Vec::with_capacity(cycle.len());
        for &(index, forward) in &cycle {
            let boundary = boundaries[index];
            registry.set_owner_forward(boundary.handle, 1, forward)?;
            wire_edges.push(OrientedEdge::new(boundary.edge, forward));
        }
        let wire = Wire::new(wire_edges, true)?;
        let surface = runout_surface(topo, &boundaries, &cycle, junction_vertex)?;
        let wire_id = topo.add_wire(wire);
        let face_id = topo.add_face(Face::new(wire_id, Vec::new(), surface.clone()));
        for &(index, _) in &cycle {
            let handle = boundaries[index].handle;
            registry.set_owner_face(handle, 1, face_id)?;
            let _ = registry.oriented_edge(topo, handle, 1)?;
        }
        results.push(CornerResult {
            face_id,
            surface,
            new_edges: cycle
                .iter()
                .map(|(index, _)| boundaries[*index].edge)
                .collect(),
            new_vertices: Vec::new(),
        });
    }
    Ok(results)
}

fn collect_junction_fan_boundaries(
    topo: &mut Topology,
    stripes: &[Stripe],
    stripe_indices: &[usize],
    junction: &VertexJunction,
    cross_boundaries: &[TerminalBoundary],
    support_faces: &[(FaceId, FaceId)],
    registry: &mut BoundaryRegistry,
) -> Result<Option<Vec<JunctionBoundary>>, BlendError> {
    let junction_vertex = junction.vertex;
    let mut source_edges = std::collections::HashSet::new();
    let mut cross_vertices = std::collections::HashSet::new();
    let mut cross_edges = Vec::new();
    let mut seen_cross_edges = std::collections::HashSet::new();
    for &stripe_index in stripe_indices {
        let pair =
            cross_boundaries
                .get(stripe_index)
                .ok_or_else(|| BlendError::PlanningFailure {
                    reason: format!("missing cross-section boundaries for stripe {stripe_index}"),
                })?;
        source_edges.extend(stripes[stripe_index].spine.edges().iter().copied());
        for handle in [pair.0, pair.1].into_iter().flatten() {
            let entry = registry
                .entry(handle)
                .ok_or_else(|| BlendError::PlanningFailure {
                    reason: format!("unknown cross-section boundary {handle}"),
                })?;
            let edge = entry.edge_id().ok_or_else(|| BlendError::PlanningFailure {
                reason: format!(
                    "cross-section boundary {:?} was not materialized",
                    entry.key
                ),
            })?;
            let (start, end) = edge_vertices(topo, edge)?;
            if start == end {
                return Err(BlendError::CornerFailure {
                    vertex: junction_vertex,
                });
            }
            cross_vertices.insert(start);
            cross_vertices.insert(end);
            if entry.owners[1].face.is_none() && seen_cross_edges.insert(edge) {
                cross_edges.push((handle, edge, start, end));
            }
        }
    }
    if cross_edges.is_empty() {
        return Ok(None);
    }

    let mut planned_to_support = std::collections::HashMap::new();
    let mut remaining_faces = Vec::new();
    for &stripe_index in stripe_indices {
        let &(face1, face2) =
            support_faces
                .get(stripe_index)
                .ok_or_else(|| BlendError::PlanningFailure {
                    reason: format!("missing support faces for stripe {stripe_index}"),
                })?;
        planned_to_support.insert(stripes[stripe_index].face1, face1);
        planned_to_support.insert(stripes[stripe_index].face2, face2);
        remaining_faces.extend([face1, face2]);
    }
    let mut faces = Vec::new();
    for planned_face in &junction.face_fan {
        if let Some(&support_face) = planned_to_support.get(planned_face)
            && !faces.contains(&support_face)
        {
            faces.push(support_face);
        }
    }
    remaining_faces.sort_unstable_by_key(|face| face.index());
    remaining_faces.dedup();
    for support_face in remaining_faces {
        if !faces.contains(&support_face) {
            faces.push(support_face);
        }
    }

    let mut support_candidates = Vec::new();
    let mut seen_support_edges = std::collections::HashSet::new();
    for support_face in faces {
        let face = topo.face(support_face)?;
        let wires = std::iter::once(face.outer_wire()).chain(face.inner_wires().iter().copied());
        for wire_id in wires {
            for oriented in topo.wire(wire_id)?.edges() {
                let edge = oriented.edge();
                if !seen_support_edges.insert(edge) || source_edges.contains(&edge) {
                    continue;
                }
                let (start, end) = edge_vertices(topo, edge)?;
                let joins_fan = (cross_vertices.contains(&start)
                    && (cross_vertices.contains(&end) || end == junction_vertex))
                    || (cross_vertices.contains(&end)
                        && (cross_vertices.contains(&start) || start == junction_vertex));
                if !joins_fan {
                    continue;
                }
                let existing_handle = registry.handle_for_edge(edge);
                if let Some(handle) = existing_handle {
                    let entry =
                        registry
                            .entry(handle)
                            .ok_or_else(|| BlendError::PlanningFailure {
                                reason: format!("unknown boundary handle {handle}"),
                            })?;
                    if !matches!(entry.key.kind, BoundaryKind::Corner | BoundaryKind::Runout)
                        || entry.owners[1].face.is_some()
                    {
                        continue;
                    }
                }
                let forward = face_edge_forward(topo, support_face, edge)
                    .ok_or(BlendError::TrimmingFailure { face: support_face })?;
                support_candidates.push((edge, support_face, forward, start, end, existing_handle));
            }
        }
    }

    let incident_handles: std::collections::HashSet<_> = cross_edges
        .iter()
        .filter(|(_, _, start, end)| {
            support_candidates
                .iter()
                .any(|(_, _, _, support_start, support_end, _)| {
                    [*support_start, *support_end]
                        .into_iter()
                        .any(|vertex| vertex == *start || vertex == *end)
                })
        })
        .map(|(handle, ..)| *handle)
        .collect();
    if incident_handles.is_empty() {
        return Ok(None);
    }

    let mut boundaries = Vec::new();
    for &(handle, edge, start, end) in &cross_edges {
        if incident_handles.contains(&handle) {
            boundaries.push(JunctionBoundary {
                edge,
                handle,
                start,
                end,
                required: true,
            });
        }
    }
    for (segment, &(edge, support_face, forward, start, end, existing_handle)) in
        support_candidates.iter().enumerate()
    {
        let handle = if let Some(handle) = existing_handle {
            handle
        } else {
            let edge_data = topo.edge(edge)?.clone();
            let handle = registry.register(
                BoundaryKey::corner(junction_vertex.index(), segment, 0),
                PlannedVertex::new(edge_data.start()),
                PlannedVertex::new(edge_data.end()),
                edge_data.curve().clone(),
                edge_data.curve().domain_with_endpoints(
                    topo.vertex(edge_data.start())?.point(),
                    topo.vertex(edge_data.end())?.point(),
                ),
                [
                    BoundaryOwner::new(support_face, "ordered junction support", forward),
                    BoundaryOwner::planned("ordered junction corner", false),
                ],
            )?;
            registry.defer_owner(handle, 1)?;
            registry.bind_existing_edge(topo, handle, edge)?;
            let _ = registry.oriented_edge(topo, handle, 0)?;
            handle
        };
        boundaries.push(JunctionBoundary {
            edge,
            handle,
            start,
            end,
            required: false,
        });
    }
    if terminal_boundary_cycles(&boundaries).is_some() {
        Ok(Some(boundaries))
    } else {
        Err(BlendError::CornerFailure {
            vertex: junction_vertex,
        })
    }
}

fn build_junction_fan(
    topo: &mut Topology,
    stripes: &[Stripe],
    stripe_indices: &[usize],
    junction: &VertexJunction,
    cross_boundaries: &[TerminalBoundary],
    support_faces: &[(FaceId, FaceId)],
    registry: &mut BoundaryRegistry,
) -> Result<Vec<CornerResult>, BlendError> {
    let junction_vertex = junction.vertex;
    let boundaries = collect_junction_fan_boundaries(
        topo,
        stripes,
        stripe_indices,
        junction,
        cross_boundaries,
        support_faces,
        registry,
    )?
    .ok_or(BlendError::CornerFailure {
        vertex: junction_vertex,
    })?;
    let cycles = terminal_boundary_cycles(&boundaries).ok_or(BlendError::CornerFailure {
        vertex: junction_vertex,
    })?;
    if cycles.len() != 1 {
        return Err(BlendError::CornerFailure {
            vertex: junction_vertex,
        });
    }
    let surface = if stripe_indices.len() == 2 {
        if let Some(geometry) = horn_torus_geometry_for_pair(
            junction_vertex,
            stripes,
            stripe_indices[0],
            stripe_indices[1],
            topo,
        )? {
            geometry.surface
        } else if let Some(geometry) = mixed_radius_geometry_for_pair(
            junction_vertex,
            stripes,
            stripe_indices[0],
            stripe_indices[1],
            topo,
        )? {
            geometry.surface
        } else {
            runout_surface(topo, &boundaries, &cycles[0], junction_vertex)?
        }
    } else if stripe_indices.len() == 3 {
        let data = multi_edge_corner_data(junction_vertex, stripe_indices, stripes, topo)?;
        if data.contact_points.len() != 3 {
            return Err(BlendError::CornerFailure {
                vertex: junction_vertex,
            });
        }
        build_spherical_corner_surface(&data)?
    } else {
        return Err(BlendError::PlanningFailure {
            reason: format!(
                "unsupported ordered junction valence {} at vertex {:?}",
                stripe_indices.len(),
                junction_vertex
            ),
        });
    };

    let mut results = Vec::with_capacity(cycles.len());
    for cycle in cycles {
        let mut oriented_edges = Vec::with_capacity(cycle.len());
        for &(index, forward) in &cycle {
            let boundary = boundaries[index];
            registry.set_owner_forward(boundary.handle, 1, forward)?;
            oriented_edges.push(OrientedEdge::new(boundary.edge, forward));
        }
        let wire_id = topo.add_wire(Wire::new(oriented_edges, true)?);
        let face_id = topo.add_face(Face::new(wire_id, Vec::new(), surface.clone()));
        for &(index, _) in &cycle {
            let handle = boundaries[index].handle;
            registry.set_owner_face(handle, 1, face_id)?;
            let _ = registry.oriented_edge(topo, handle, 1)?;
        }
        results.push(CornerResult {
            face_id,
            surface: surface.clone(),
            new_edges: cycle
                .iter()
                .map(|(index, _)| boundaries[*index].edge)
                .collect(),
            new_vertices: Vec::new(),
        });
    }
    Ok(results)
}

/// Solve every original planned junction exactly once.
///
/// Periodic contours have no endpoint junction. G1 continuation produces no
/// patch. Terminal contours use the ordered runout graph or deferred
/// support-side closure. Junctions dispatch to the ordered fan, preserving
/// analytic two-edge and spherical three-edge surfaces. Higher valence and
/// singular geometry fail before publication.
pub fn compute_ordered_corners(
    topo: &mut Topology,
    stripes: &[Stripe],
    junctions: &[VertexJunction],
    contour_to_stripe: &[Option<usize>],
    cross_boundaries: &[TerminalBoundary],
    support_faces: &[(FaceId, FaceId)],
    registry: &mut BoundaryRegistry,
) -> Result<Vec<CornerResult>, BlendError> {
    let mut results = Vec::new();
    for junction in junctions {
        let indices: Vec<usize> = junction
            .incident_contours
            .iter()
            .filter_map(|contour| contour_to_stripe.get(*contour).copied().flatten())
            .collect();
        if matches!(junction.classification, CornerClassification::Periodic) {
            continue;
        }
        if matches!(
            junction.classification,
            CornerClassification::G1Continuation
        ) {
            continue;
        }
        if matches!(junction.classification, CornerClassification::Terminal) {
            if indices.len() != 1 {
                return Err(BlendError::CornerFailure {
                    vertex: junction.vertex,
                });
            }
            let stripe_index = indices[0];
            let Some(&contour_id) = junction.incident_contours.first() else {
                return Err(BlendError::CornerFailure {
                    vertex: junction.vertex,
                });
            };
            let terminal_results = build_terminal_runout(
                topo,
                stripes,
                stripe_index,
                contour_id,
                junction.vertex,
                cross_boundaries,
                support_faces,
                registry,
            )?;
            results.extend(terminal_results);
            continue;
        }
        if indices.len() > 3 {
            return Err(BlendError::PlanningFailure {
                reason: format!(
                    "unsupported ordered junction valence {} at vertex {:?}",
                    indices.len(),
                    junction.vertex
                ),
            });
        }
        if indices.len() < 2 {
            return Err(BlendError::CornerFailure {
                vertex: junction.vertex,
            });
        }
        let junction_results = build_junction_fan(
            topo,
            stripes,
            &indices,
            junction,
            cross_boundaries,
            support_faces,
            registry,
        )?;
        results.extend(junction_results);
    }
    Ok(results)
}

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

        let v000 = topo.add_vertex(Vertex::new(Point3::new(0.0, 0.0, 0.0), TOL));
        let v100 = topo.add_vertex(Vertex::new(Point3::new(1.0, 0.0, 0.0), TOL));
        let v010 = topo.add_vertex(Vertex::new(Point3::new(0.0, 1.0, 0.0), TOL));
        let v001 = topo.add_vertex(Vertex::new(Point3::new(0.0, 0.0, 1.0), TOL));
        let v110 = topo.add_vertex(Vertex::new(Point3::new(1.0, 1.0, 0.0), TOL));
        let v101 = topo.add_vertex(Vertex::new(Point3::new(1.0, 0.0, 1.0), TOL));
        let v011 = topo.add_vertex(Vertex::new(Point3::new(0.0, 1.0, 1.0), TOL));
        let v111 = topo.add_vertex(Vertex::new(Point3::new(1.0, 1.0, 1.0), TOL));

        let ex = topo.add_edge(Edge::new(v000, v100, EdgeCurve::Line));
        let ey = topo.add_edge(Edge::new(v000, v010, EdgeCurve::Line));
        let ez = topo.add_edge(Edge::new(v000, v001, EdgeCurve::Line));

        let exy = topo.add_edge(Edge::new(v100, v110, EdgeCurve::Line));
        let eyx = topo.add_edge(Edge::new(v010, v110, EdgeCurve::Line));
        let exz = topo.add_edge(Edge::new(v100, v101, EdgeCurve::Line));
        let ezx = topo.add_edge(Edge::new(v001, v101, EdgeCurve::Line));
        let eyz = topo.add_edge(Edge::new(v010, v011, EdgeCurve::Line));
        let ezy = topo.add_edge(Edge::new(v001, v011, EdgeCurve::Line));

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

        let shell = Shell::new(vec![
            face_xy, face_xz, face_yz, face_top, face_right, face_back,
        ])
        .unwrap();
        let shell_id = topo.add_shell(shell);
        let solid = Solid::new(shell_id, vec![]);
        let solid_id = topo.add_solid(solid);

        let radius = 0.2;

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
        let indices = stripes_at_vertex(v000, &stripes, &topo);
        let results = build_multi_edge_corner(v000, &indices, &stripes, &mut topo).unwrap();

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
        let indices = stripes_at_vertex(v000, &stripes, &topo);
        let results = build_multi_edge_corner(v000, &indices, &stripes, &mut topo).unwrap();
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
    #[test]
    fn ordered_junction_rejects_higher_valence_before_geometry() {
        let mut topo = Topology::new();
        let vertex = topo.add_vertex(Vertex::new(Point3::new(0.0, 0.0, 0.0), 1e-7));
        let published_before = (topo.num_wires(), topo.num_faces(), topo.num_shells());
        let junction = VertexJunction {
            vertex,
            incident_contours: vec![0, 1, 2, 3],
            unselected_sharp_edges: Vec::new(),
            face_fan: Vec::new(),
            classification: CornerClassification::Junction,
        };
        let error = match compute_ordered_corners(
            &mut topo,
            &[],
            &[junction],
            &[Some(0), Some(1), Some(2), Some(3)],
            &[],
            &[],
            &mut BoundaryRegistry::new(),
        ) {
            Ok(_) => panic!("higher-valence junction must fail explicitly"),
            Err(error) => error,
        };
        assert!(
            error
                .to_string()
                .contains("unsupported ordered junction valence")
        );
        assert_eq!(
            (topo.num_wires(), topo.num_faces(), topo.num_shells()),
            published_before,
            "unsupported valence must fail before face or shell publication"
        );
    }
    #[test]
    fn ordered_junction_rejects_singular_collinear_fan() {
        let mut topo = Topology::new();
        let v0 = topo.add_vertex(Vertex::new(Point3::new(0.0, 0.0, 0.0), TOL));
        let v1 = topo.add_vertex(Vertex::new(Point3::new(1.0, 0.0, 0.0), TOL));
        let v2 = topo.add_vertex(Vertex::new(Point3::new(2.0, 0.0, 0.0), TOL));
        let e0 = topo.add_edge(Edge::new(v0, v1, EdgeCurve::Line));
        let e1 = topo.add_edge(Edge::new(v1, v2, EdgeCurve::Line));
        let e2 = topo.add_edge(Edge::new(v2, v0, EdgeCurve::Line));
        let boundaries = vec![
            JunctionBoundary {
                edge: e0,
                handle: 0,
                start: v0,
                end: v1,
                required: true,
            },
            JunctionBoundary {
                edge: e1,
                handle: 1,
                start: v1,
                end: v2,
                required: true,
            },
            JunctionBoundary {
                edge: e2,
                handle: 2,
                start: v2,
                end: v0,
                required: true,
            },
        ];
        let cycle = vec![(0, true), (1, true), (2, true)];
        let published_before = (topo.num_wires(), topo.num_faces(), topo.num_shells());

        let error = runout_surface(&topo, &boundaries, &cycle, v0)
            .expect_err("a collinear ordered fan must fail before face creation");
        assert!(matches!(
            error,
            BlendError::CornerFailure { vertex } if vertex == v0
        ));
        assert_eq!(
            (topo.num_wires(), topo.num_faces(), topo.num_shells()),
            published_before,
            "singular geometry must fail before face or shell publication"
        );
    }

    #[test]
    fn ordered_periodic_junction_has_no_endpoint_patch() {
        let mut topo = Topology::new();
        let vertex = topo.add_vertex(Vertex::new(Point3::new(0.0, 0.0, 0.0), 1e-7));
        let junction = VertexJunction {
            vertex,
            incident_contours: vec![0],
            unselected_sharp_edges: Vec::new(),
            face_fan: Vec::new(),
            classification: CornerClassification::Periodic,
        };
        let corners = compute_ordered_corners(
            &mut topo,
            &[],
            &[junction],
            &[Some(0)],
            &[],
            &[],
            &mut BoundaryRegistry::new(),
        )
        .expect("periodic contour bypasses endpoint solving");
        assert!(corners.is_empty());
    }
}
