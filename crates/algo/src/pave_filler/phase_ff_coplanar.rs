//! Phase FF-Coplanar: coplanar face splitting.
//!
//! Handles the case where two faces from different solids lie on the same
//! plane and partially overlap. Phase FF skips these because parallel planes
//! have no intersection line. This phase runs after FF and creates section
//! edges by projecting one face's boundary edges into the other face's
//! interior.

use brepkit_math::aabb::Aabb3;
use brepkit_math::tolerance::Tolerance;
use brepkit_math::vec::{Point2, Point3, Vec3};
use brepkit_topology::Topology;
use brepkit_topology::edge::{Edge, EdgeCurve};
use brepkit_topology::face::{FaceId, FaceSurface};
use brepkit_topology::solid::SolidId;
use brepkit_topology::vertex::Vertex;

use crate::ds::{GfaArena, Interference, IntersectionCurveDS, Pave, PaveBlock};
use crate::error::AlgoError;

use super::helpers::find_nearby_pave_vertex;

/// Detect coplanar face pairs between two solids and create section edges
/// for boundary edges of one face that lie inside the other.
///
/// # Errors
///
/// Returns [`AlgoError`] if any topology lookup fails.
#[allow(clippy::too_many_lines)]
pub fn perform(
    topo: &mut Topology,
    solid_a: SolidId,
    solid_b: SolidId,
    tol: Tolerance,
    arena: &mut GfaArena,
) -> Result<(), AlgoError> {
    let faces_a = brepkit_topology::explorer::solid_faces(topo, solid_a)?;
    let faces_b = brepkit_topology::explorer::solid_faces(topo, solid_b)?;

    // Collect plane face data: (FaceId, normal, d)
    let planes_a = collect_plane_faces(topo, &faces_a)?;
    let planes_b = collect_plane_faces(topo, &faces_b)?;

    if planes_a.is_empty() || planes_b.is_empty() {
        return Ok(());
    }

    // Pre-compute AABBs
    let bboxes_a = compute_face_bboxes(topo, &planes_a)?;
    let bboxes_b = compute_face_bboxes(topo, &planes_b)?;

    log::debug!(
        "FF-coplanar: checking {} × {} plane face pairs",
        planes_a.len(),
        planes_b.len()
    );

    // Find coplanar pairs
    for (idx_a, &(fa, na, da)) in planes_a.iter().enumerate() {
        let bbox_a = &bboxes_a[idx_a];

        for (idx_b, &(fb, nb, db)) in planes_b.iter().enumerate() {
            let bbox_b = &bboxes_b[idx_b];

            // Check parallel: |dot(na, nb)| > 1 - angular_tol
            let dot = na.dot(nb);
            if dot.abs() < 1.0 - tol.angular {
                continue;
            }

            // Check coplanar: same plane distance (accounting for normal direction)
            let sign = if dot > 0.0 { 1.0 } else { -1.0 };
            if (da - db * sign).abs() > tol.linear {
                continue;
            }

            // AABB overlap check
            if !bbox_a
                .expanded(tol.linear)
                .intersects(bbox_b.expanded(tol.linear))
            {
                continue;
            }

            // Skip pairs that already have FF section edges
            if has_existing_ff_interference(arena, fa, fb) {
                continue;
            }

            // Process this coplanar pair
            process_coplanar_pair(topo, fa, na, fb, tol, arena)?;
        }
    }

    Ok(())
}

/// Collect `(FaceId, normal, d)` for all plane faces in the list.
fn collect_plane_faces(
    topo: &Topology,
    faces: &[FaceId],
) -> Result<Vec<(FaceId, Vec3, f64)>, AlgoError> {
    let mut result = Vec::new();
    for &fid in faces {
        let face = topo.face(fid)?;
        if let FaceSurface::Plane { normal, d } = face.surface() {
            result.push((fid, *normal, *d));
        }
    }
    Ok(result)
}

/// Compute AABBs for plane faces by sampling boundary edges.
fn compute_face_bboxes(
    topo: &Topology,
    planes: &[(FaceId, Vec3, f64)],
) -> Result<Vec<Aabb3>, AlgoError> {
    let mut bboxes = Vec::with_capacity(planes.len());
    for &(fid, _, _) in planes {
        bboxes.push(compute_face_bbox(topo, fid)?);
    }
    Ok(bboxes)
}

/// Compute AABB for a face by sampling its boundary edges.
fn compute_face_bbox(topo: &Topology, face_id: FaceId) -> Result<Aabb3, AlgoError> {
    let edges = brepkit_topology::explorer::face_edges(topo, face_id)?;
    let mut points = Vec::new();

    for eid in edges {
        let edge = topo.edge(eid)?;
        let start_pos = topo.vertex(edge.start())?.point();
        let end_pos = topo.vertex(edge.end())?.point();
        let (t0, t1) = edge.curve().domain_with_endpoints(start_pos, end_pos);

        let n: usize = 8;
        for i in 0..=n {
            let t = t0 + (t1 - t0) * (i as f64 / n as f64);
            let pt = edge.curve().evaluate_with_endpoints(t, start_pos, end_pos);
            points.push(pt);
        }
    }

    if points.is_empty() {
        Ok(Aabb3 {
            min: Point3::new(0.0, 0.0, 0.0),
            max: Point3::new(0.0, 0.0, 0.0),
        })
    } else {
        Ok(Aabb3::from_points(points))
    }
}

/// Check if a section curve already exists at this position for either face.
///
/// Searches `arena.curves` for any existing intersection curve involving
/// `face_a` or `face_b` whose endpoints match `p_start`/`p_end` within
/// tolerance. This prevents the coplanar phase from creating duplicate
/// section edges that already exist from the regular FF phase.
fn has_existing_section_at(
    arena: &GfaArena,
    face_a: FaceId,
    face_b: FaceId,
    p_start: Point3,
    p_end: Point3,
    tol: Tolerance,
) -> bool {
    for curve in &arena.curves {
        // Must involve at least one of our faces
        if curve.face_a != face_a
            && curve.face_a != face_b
            && curve.face_b != face_a
            && curve.face_b != face_b
        {
            continue;
        }

        // Check AABB overlap
        let edge_min = Point3::new(
            p_start.x().min(p_end.x()),
            p_start.y().min(p_end.y()),
            p_start.z().min(p_end.z()),
        );
        let edge_max = Point3::new(
            p_start.x().max(p_end.x()),
            p_start.y().max(p_end.y()),
            p_start.z().max(p_end.z()),
        );
        let expanded = curve.bbox.expanded(tol.linear);
        if edge_min.x() > expanded.max.x()
            || edge_max.x() < expanded.min.x()
            || edge_min.y() > expanded.max.y()
            || edge_max.y() < expanded.min.y()
            || edge_min.z() > expanded.max.z()
            || edge_max.z() < expanded.min.z()
        {
            continue;
        }

        // Check endpoint match: midpoint of proposed edge must be near the
        // existing curve's midpoint. Use midpoint instead of endpoint to
        // handle reversed-direction curves.
        let mid = Point3::new(
            (p_start.x() + p_end.x()) * 0.5,
            (p_start.y() + p_end.y()) * 0.5,
            (p_start.z() + p_end.z()) * 0.5,
        );
        let curve_mid = Point3::new(
            (curve.bbox.min.x() + curve.bbox.max.x()) * 0.5,
            (curve.bbox.min.y() + curve.bbox.max.y()) * 0.5,
            (curve.bbox.min.z() + curve.bbox.max.z()) * 0.5,
        );
        if (mid - curve_mid).length() < tol.linear * 10.0 {
            return true;
        }
    }
    false
}

/// Check if an FF interference already exists for this face pair.
fn has_existing_ff_interference(arena: &GfaArena, fa: FaceId, fb: FaceId) -> bool {
    arena.interference.ff.iter().any(|interf| {
        matches!(interf,
            Interference::FF { f1, f2, .. } if (*f1 == fa && *f2 == fb) || (*f1 == fb && *f2 == fa)
        )
    })
}

/// Process a single coplanar face pair: project boundary edges of each face
/// into the other and create section edges for edges that lie inside.
#[allow(clippy::too_many_lines)]
fn process_coplanar_pair(
    topo: &mut Topology,
    face_a: FaceId,
    normal: Vec3,
    face_b: FaceId,
    tol: Tolerance,
    arena: &mut GfaArena,
) -> Result<(), AlgoError> {
    // Build a 2D projection frame from the shared plane
    let origin = first_wire_vertex(topo, face_a)?;
    let frame = PlaneFrame2D::new(normal, origin);

    // Collect boundary polygons in 2D for both faces
    let poly_a = face_boundary_polygon_2d(topo, face_a, &frame)?;
    let poly_b = face_boundary_polygon_2d(topo, face_b, &frame)?;

    // Collect boundary edges with their 2D endpoints
    let edges_a = face_boundary_edges_2d(topo, face_a, &frame)?;
    let edges_b = face_boundary_edges_2d(topo, face_b, &frame)?;

    // For each boundary edge of face_b, check if it's inside face_a.
    // Skip edges that already have a section curve from the regular FF phase
    // at the same position (prevents duplicate section edges that create
    // spurious inner wires in the face splitter).
    for &(_, p2d_start, p2d_end, p3d_start, p3d_end) in &edges_b {
        if should_create_section_edge(p2d_start, p2d_end, &poly_a, &edges_a, tol.linear)
            && !has_existing_section_at(arena, face_a, face_b, p3d_start, p3d_end, tol)
        {
            create_section_edge(topo, arena, face_a, face_b, p3d_start, p3d_end, tol)?;
        }
    }

    // For each boundary edge of face_a, check if it's inside face_b
    for &(_, p2d_start, p2d_end, p3d_start, p3d_end) in &edges_a {
        if should_create_section_edge(p2d_start, p2d_end, &poly_b, &edges_b, tol.linear)
            && !has_existing_section_at(arena, face_a, face_b, p3d_start, p3d_end, tol)
        {
            create_section_edge(topo, arena, face_a, face_b, p3d_start, p3d_end, tol)?;
        }
    }

    Ok(())
}

/// Get the first vertex position of a face's outer wire.
fn first_wire_vertex(topo: &Topology, face_id: FaceId) -> Result<Point3, AlgoError> {
    let face = topo.face(face_id)?;
    let wire = topo.wire(face.outer_wire())?;
    if let Some(oe) = wire.edges().first() {
        let edge = topo.edge(oe.edge())?;
        Ok(topo.vertex(edge.start())?.point())
    } else {
        Ok(Point3::new(0.0, 0.0, 0.0))
    }
}

/// Collect the outer wire boundary as a 2D polygon.
fn face_boundary_polygon_2d(
    topo: &Topology,
    face_id: FaceId,
    frame: &PlaneFrame2D,
) -> Result<Vec<Point2>, AlgoError> {
    let face = topo.face(face_id)?;
    let wire = topo.wire(face.outer_wire())?;
    let mut polygon = Vec::new();

    for oe in wire.edges() {
        let edge = topo.edge(oe.edge())?;
        // Use oriented start to respect wire traversal direction.
        let vid = oe.oriented_start(edge);
        let pos = topo.vertex(vid)?.point();
        polygon.push(frame.project(pos));
    }

    Ok(polygon)
}

/// Boundary edge info: `(EdgeId, 2D start, 2D end, 3D start, 3D end)`.
type BoundaryEdge = (
    brepkit_topology::edge::EdgeId,
    Point2,
    Point2,
    Point3,
    Point3,
);

/// Collect boundary edges with 2D and 3D endpoint positions.
///
/// Respects oriented edge direction so start/end match wire traversal.
fn face_boundary_edges_2d(
    topo: &Topology,
    face_id: FaceId,
    frame: &PlaneFrame2D,
) -> Result<Vec<BoundaryEdge>, AlgoError> {
    let face = topo.face(face_id)?;
    let wire = topo.wire(face.outer_wire())?;
    let mut edges = Vec::new();

    for oe in wire.edges() {
        let edge = topo.edge(oe.edge())?;
        let (p3_start, p3_end) = if oe.is_forward() {
            (
                topo.vertex(edge.start())?.point(),
                topo.vertex(edge.end())?.point(),
            )
        } else {
            (
                topo.vertex(edge.end())?.point(),
                topo.vertex(edge.start())?.point(),
            )
        };
        let p2_start = frame.project(p3_start);
        let p2_end = frame.project(p3_end);
        edges.push((oe.edge(), p2_start, p2_end, p3_start, p3_end));
    }

    Ok(edges)
}

/// Determine whether a boundary edge should become a section edge in the
/// target face.
///
/// An edge qualifies if its midpoint is inside the target polygon AND it is
/// not a shared boundary edge (both endpoints on the target boundary).
fn should_create_section_edge(
    p2d_start: Point2,
    p2d_end: Point2,
    target_polygon: &[Point2],
    target_edges: &[BoundaryEdge],
    tol: f64,
) -> bool {
    // Skip only if both endpoints lie on the SAME target boundary edge.
    // This identifies truly shared boundary edges (collinear with a target edge).
    // Don't skip if endpoints are on DIFFERENT target edges — those are
    // crossing edges that enter and exit the target boundary.
    let start_edge_idx = which_boundary_edge(p2d_start, target_edges, tol);
    let end_edge_idx = which_boundary_edge(p2d_end, target_edges, tol);
    if let (Some(si), Some(ei)) = (start_edge_idx, end_edge_idx) {
        if si == ei {
            return false; // Both endpoints on the same target edge — shared boundary
        }
    }

    // Check if the edge midpoint is inside the target polygon
    let mid = Point2::new(
        (p2d_start.x() + p2d_end.x()) * 0.5,
        (p2d_start.y() + p2d_end.y()) * 0.5,
    );
    point_in_polygon_2d(mid, target_polygon)
}

/// Create a section edge and register it in the GFA arena.
#[allow(clippy::unnecessary_wraps)]
fn create_section_edge(
    topo: &mut Topology,
    arena: &mut GfaArena,
    face_a: FaceId,
    face_b: FaceId,
    p3d_start: Point3,
    p3d_end: Point3,
    tol: Tolerance,
) -> Result<(), AlgoError> {
    let edge_length = (p3d_end - p3d_start).length();
    if edge_length < tol.linear {
        // Degenerate edge, skip
        return Ok(());
    }

    // Find or create vertices at the endpoints
    let start_vid = find_or_create_vertex(topo, arena, p3d_start, tol);
    let end_vid = find_or_create_vertex(topo, arena, p3d_end, tol);

    // Create a topology edge (Line between the two points)
    let edge = Edge::new(start_vid, end_vid, EdgeCurve::Line);
    let edge_id = topo.add_edge(edge);

    // EdgeCurve::Line uses normalized parameter space [0, 1].
    let start_pave = Pave::new(start_vid, 0.0);
    let end_pave = Pave::new(end_vid, 1.0);
    let pb = PaveBlock::new(edge_id, start_pave, end_pave);
    let pb_id = arena.pave_blocks.alloc(pb);

    // Register in edge_pave_blocks so ForceInterfEE can detect overlaps
    // between this section PB and boundary-edge PBs with the same
    // endpoints. This creates CommonBlocks → shared split edges →
    // manifold shell connectivity between coplanar sub-faces.
    arena
        .edge_pave_blocks
        .entry(edge_id)
        .or_default()
        .push(pb_id);

    // Create the intersection curve entry
    let bbox = Aabb3 {
        min: Point3::new(
            p3d_start.x().min(p3d_end.x()),
            p3d_start.y().min(p3d_end.y()),
            p3d_start.z().min(p3d_end.z()),
        ),
        max: Point3::new(
            p3d_start.x().max(p3d_end.x()),
            p3d_start.y().max(p3d_end.y()),
            p3d_start.z().max(p3d_end.z()),
        ),
    };

    let curve_index = arena.curves.len();
    arena.curves.push(IntersectionCurveDS {
        curve: EdgeCurve::Line,
        face_a,
        face_b,
        bbox,
        pave_blocks: vec![pb_id],
        t_range: (0.0, 1.0),
    });

    arena.interference.ff.push(Interference::FF {
        f1: face_a,
        f2: face_b,
        curve_index,
    });

    log::debug!(
        "FF-coplanar: faces {face_a:?} and {face_b:?} section edge \
         (curve_index={curve_index}, edge={edge_id:?}, pb={pb_id:?})",
    );

    Ok(())
}

/// Find an existing vertex near the point, or create a new one.
fn find_or_create_vertex(
    topo: &mut Topology,
    arena: &GfaArena,
    point: Point3,
    tol: Tolerance,
) -> brepkit_topology::vertex::VertexId {
    if let Some(vid) = find_nearby_pave_vertex(topo, arena, point, tol) {
        return vid;
    }
    topo.add_vertex(Vertex::new(point, tol.linear))
}

// ---------------------------------------------------------------------------
// 2D geometry helpers
// ---------------------------------------------------------------------------

/// Minimal plane frame for 3D ↔ 2D projection (same logic as
/// `builder::plane_frame::PlaneFrame` but kept local to avoid coupling).
struct PlaneFrame2D {
    origin: Point3,
    u_axis: Vec3,
    v_axis: Vec3,
}

impl PlaneFrame2D {
    fn new(normal: Vec3, origin: Point3) -> Self {
        let seed = if normal.x().abs() < 0.9 {
            Vec3::new(1.0, 0.0, 0.0)
        } else {
            Vec3::new(0.0, 1.0, 0.0)
        };
        let u_raw = normal.cross(seed);
        let u_axis = u_raw.normalize().unwrap_or(Vec3::new(1.0, 0.0, 0.0));
        let v_axis = normal.cross(u_axis);
        Self {
            origin,
            u_axis,
            v_axis,
        }
    }

    fn project(&self, p: Point3) -> Point2 {
        let d = p - self.origin;
        Point2::new(d.dot(self.u_axis), d.dot(self.v_axis))
    }
}

/// Ray-casting point-in-polygon test.
///
/// Returns `true` if `pt` is strictly inside `polygon` (CCW or CW vertex order).
fn point_in_polygon_2d(pt: Point2, polygon: &[Point2]) -> bool {
    if polygon.len() < 3 {
        return false;
    }

    let mut inside = false;
    let n = polygon.len();
    let mut j = n - 1;

    for i in 0..n {
        let pi = polygon[i];
        let pj = polygon[j];

        let yi = pi.y();
        let yj = pj.y();
        let xi = pi.x();
        let xj = pj.x();

        if ((yi > pt.y()) != (yj > pt.y())) && (pt.x() < (xj - xi) * (pt.y() - yi) / (yj - yi) + xi)
        {
            inside = !inside;
        }

        j = i;
    }

    inside
}

/// Return the index of the boundary edge that the point lies on, if any.
fn which_boundary_edge(pt: Point2, edges: &[BoundaryEdge], tol: f64) -> Option<usize> {
    edges
        .iter()
        .position(|&(_, a, b, _, _)| point_on_segment_2d(pt, a, b, tol))
}

/// Check if a 2D point lies on a line segment within tolerance.
fn point_on_segment_2d(pt: Point2, a: Point2, b: Point2, tol: f64) -> bool {
    let ab = Point2::new(b.x() - a.x(), b.y() - a.y());
    let ap = Point2::new(pt.x() - a.x(), pt.y() - a.y());

    let ab_len_sq = ab.x() * ab.x() + ab.y() * ab.y();
    if ab_len_sq < tol * tol {
        // Degenerate segment — just check distance to endpoint
        return ap.x() * ap.x() + ap.y() * ap.y() <= tol * tol;
    }

    // Project pt onto the line through a→b
    let t = (ap.x() * ab.x() + ap.y() * ab.y()) / ab_len_sq;
    if t < -tol || t > 1.0 + tol {
        return false;
    }

    // Distance from pt to the closest point on the segment
    let closest_x = a.x() + t.clamp(0.0, 1.0) * ab.x();
    let closest_y = a.y() + t.clamp(0.0, 1.0) * ab.y();
    let dx = pt.x() - closest_x;
    let dy = pt.y() - closest_y;

    dx * dx + dy * dy <= tol * tol
}

#[cfg(test)]
mod tests {
    #![allow(clippy::unwrap_used, clippy::expect_used)]
    use super::*;

    #[test]
    fn point_in_unit_square() {
        let square = vec![
            Point2::new(0.0, 0.0),
            Point2::new(1.0, 0.0),
            Point2::new(1.0, 1.0),
            Point2::new(0.0, 1.0),
        ];
        assert!(point_in_polygon_2d(Point2::new(0.5, 0.5), &square));
        assert!(!point_in_polygon_2d(Point2::new(2.0, 0.5), &square));
        assert!(!point_in_polygon_2d(Point2::new(-0.1, 0.5), &square));
    }

    #[test]
    fn point_on_segment() {
        let a = Point2::new(0.0, 0.0);
        let b = Point2::new(1.0, 0.0);
        assert!(point_on_segment_2d(Point2::new(0.5, 0.0), a, b, 1e-7));
        assert!(!point_on_segment_2d(Point2::new(0.5, 1.0), a, b, 1e-7));
        assert!(point_on_segment_2d(Point2::new(0.0, 0.0), a, b, 1e-7));
        assert!(point_on_segment_2d(Point2::new(1.0, 0.0), a, b, 1e-7));
    }
}
