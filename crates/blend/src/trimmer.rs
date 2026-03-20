//! Face trimming along contact curves.
//!
//! After computing a fillet or chamfer blend surface, the original adjacent
//! faces must be trimmed along the contact curves — the lines where the blend
//! surface meets the original geometry. This module splits planar faces along
//! straight contact lines, creating new edges, vertices, and wires for the
//! trimmed result.

use brepkit_math::vec::Point3;
use brepkit_topology::Topology;
use brepkit_topology::edge::{Edge, EdgeCurve, EdgeId};
use brepkit_topology::face::{Face, FaceId, FaceSurface};
use brepkit_topology::vertex::{Vertex, VertexId};
use brepkit_topology::wire::{OrientedEdge, Wire};

use crate::BlendError;

/// Which side of the contact curve to keep.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TrimSide {
    /// Keep the left side of the contact curve (relative to its direction).
    Left,
    /// Keep the right side of the contact curve (relative to its direction).
    Right,
}

/// Result of trimming a face along a contact curve.
#[derive(Debug, Clone)]
pub struct TrimResult {
    /// The newly created trimmed face (or the original if untrimmed).
    pub trimmed_face: FaceId,
    /// New edges created during trimming (sub-edges from splits).
    pub new_edges: Vec<EdgeId>,
    /// New vertices created at contact curve / boundary intersections.
    pub new_vertices: Vec<VertexId>,
    /// The edge running along the contact curve between the two boundary
    /// intersection points. `None` when the face was returned untrimmed
    /// (e.g. non-planar surfaces).
    pub contact_edge: Option<EdgeId>,
}

/// Default vertex tolerance used when creating intersection vertices.
const VERTEX_TOL: f64 = 1e-7;

/// Tolerance for the 2D segment intersection parameter test.
const PARAM_TOL: f64 = 1e-10;

/// A point where the contact line crosses a face boundary edge.
struct BoundaryHit {
    /// Index into the wire's oriented-edge list.
    edge_idx: usize,
    /// Parameter along the oriented edge (0..1).
    t: f64,
    /// 3D intersection point.
    point_3d: Point3,
}

/// Trim a face along a contact curve, keeping the side away from the fillet.
///
/// For planar faces with straight contact lines (the plane-plane fillet case),
/// this finds where the contact line intersects the face boundary, splits
/// those boundary edges, and builds a new wire loop for the trimmed face.
///
/// Non-planar faces are returned untrimmed: the original face ID is placed in
/// `trimmed_face` and no new topology is created.
///
/// # Errors
///
/// Returns [`BlendError::TrimmingFailure`] if the contact curve does not
/// produce exactly two boundary intersections, or if topology lookups fail.
/// Returns [`BlendError::Topology`] on arena errors.
#[allow(clippy::too_many_lines)]
pub fn trim_face(
    topo: &mut Topology,
    face_id: FaceId,
    contact_3d: &[Point3],
    contact_uv: &[(f64, f64)],
    keep_side: TrimSide,
) -> Result<TrimResult, BlendError> {
    // ── Guard: only planar faces for v1 ──────────────────────────────
    let face = topo.face(face_id)?;
    if !face.surface().is_planar() {
        log::warn!(
            "trim_face: non-planar surface ({}) on face {face_id:?} — returning untrimmed",
            face.surface().type_tag(),
        );
        return Ok(TrimResult {
            trimmed_face: face_id,
            new_edges: Vec::new(),
            new_vertices: Vec::new(),
            contact_edge: None,
        });
    }

    // ── Snapshot the face data (borrow rules) ────────────────────────
    let surface = face.surface().clone();
    let reversed = face.is_reversed();
    let outer_wire_id = face.outer_wire();

    let outer_wire = topo.wire(outer_wire_id)?;
    let oriented_edges: Vec<OrientedEdge> = outer_wire.edges().to_vec();

    // Need at least 2 UV points to define the contact line.
    if contact_uv.len() < 2 {
        return Err(BlendError::TrimmingFailure { face: face_id });
    }

    let uv_start = contact_uv[0];
    let uv_end = contact_uv[contact_uv.len() - 1];

    // ── Snapshot all edge data ───────────────────────────────────────
    // For each oriented edge, record:
    //   (oriented_edge, start_point, end_point, start_uv, end_uv)
    // where "start" / "end" follow the orientation.
    let mut edge_data: Vec<(OrientedEdge, Point3, Point3)> =
        Vec::with_capacity(oriented_edges.len());
    for &oe in &oriented_edges {
        let edge = topo.edge(oe.edge())?;
        let start_vid = oe.oriented_start(edge);
        let end_vid = oe.oriented_end(edge);
        let start_pt = topo.vertex(start_vid)?.point();
        let end_pt = topo.vertex(end_vid)?.point();
        edge_data.push((oe, start_pt, end_pt));
    }

    // ── Project boundary vertices to 2D using the face plane ─────────
    // For a planar face we use a local 2D coordinate system derived from
    // the first two edge directions.
    let (origin, u_axis, v_axis) = plane_local_frame(&surface, &edge_data, face_id)?;

    let project = |pt: Point3| -> (f64, f64) {
        let d = pt - origin;
        (u_axis.dot(d), v_axis.dot(d))
    };

    // Convert contact line endpoints from the caller-provided UV to our
    // local frame. If the caller's UV already matches the plane's local
    // frame we use them directly; otherwise we re-project the 3D points.
    let (line_a, line_b) = if contact_3d.len() >= 2 {
        (
            project(contact_3d[0]),
            project(contact_3d[contact_3d.len() - 1]),
        )
    } else {
        (uv_start, uv_end)
    };

    // ── Find boundary-edge intersections ─────────────────────────────
    let mut hits: Vec<BoundaryHit> = Vec::new();

    for (idx, &(_, start_pt, end_pt)) in edge_data.iter().enumerate() {
        let a1 = project(start_pt);
        let a2 = project(end_pt);

        if let Some(t) = segment_intersect_2d(a1, a2, line_a, line_b) {
            let pt = Point3::new(
                start_pt.x() + (end_pt.x() - start_pt.x()) * t,
                start_pt.y() + (end_pt.y() - start_pt.y()) * t,
                start_pt.z() + (end_pt.z() - start_pt.z()) * t,
            );
            hits.push(BoundaryHit {
                edge_idx: idx,
                t,
                point_3d: pt,
            });
        }
    }

    // We expect exactly 2 hits for a convex planar face.
    if hits.len() != 2 {
        return Err(BlendError::TrimmingFailure { face: face_id });
    }

    // Order hits so that hit_a comes first in wire traversal order.
    // They are already ordered by edge_idx from the iteration above,
    // but if both hits are on the same edge we order by parameter.
    if hits[0].edge_idx > hits[1].edge_idx
        || (hits[0].edge_idx == hits[1].edge_idx && hits[0].t > hits[1].t)
    {
        hits.swap(0, 1);
    }

    let hit_a = &hits[0];
    let hit_b = &hits[1];

    // ── Create intersection vertices ─────────────────────────────────
    let va_id = topo.add_vertex(Vertex::new(hit_a.point_3d, VERTEX_TOL));
    let vb_id = topo.add_vertex(Vertex::new(hit_b.point_3d, VERTEX_TOL));

    // ── Split boundary edges at intersection points ──────────────────
    // Each hit splits one oriented edge into two sub-edges:
    //   original: oriented from S→E
    //   sub-edge 1: S → V_hit  (forward w.r.t. original orientation)
    //   sub-edge 2: V_hit → E

    // Split edge A
    let (sub_a1, sub_a2) = split_edge_at(topo, &edge_data[hit_a.edge_idx].0, va_id)?;

    // Split edge B
    let (sub_b1, sub_b2) = split_edge_at(topo, &edge_data[hit_b.edge_idx].0, vb_id)?;

    // ── Contact edge ─────────────────────────────────────────────────
    // The contact edge connects the two intersection vertices.
    // Its direction determines which side is "left" vs "right".
    let contact_edge_id = topo.add_edge(Edge::new(va_id, vb_id, EdgeCurve::Line));

    // ── Determine which side to keep ─────────────────────────────────
    // The contact line divides the boundary edges into two chains:
    //   Chain 1: edges from hit_a to hit_b (in wire order)
    //   Chain 2: edges from hit_b to hit_a (wrapping around)
    // "Left" = chain 1 side, "Right" = chain 2 side (relative to
    // the contact direction va→vb and the face normal).

    // Build chain 1: sub_a2, edges between A and B, sub_b1
    // Build chain 2: sub_b2, edges after B wrapping to before A, sub_a1

    let n_edges = edge_data.len();

    let mut chain1: Vec<OrientedEdge> = Vec::new();
    let mut chain2: Vec<OrientedEdge> = Vec::new();

    // Chain 1: from hit_a to hit_b
    chain1.push(sub_a2);
    if hit_a.edge_idx != hit_b.edge_idx {
        // Add full edges between A's edge and B's edge
        for i in (hit_a.edge_idx + 1)..hit_b.edge_idx {
            chain1.push(oriented_edges[i]);
        }
        chain1.push(sub_b1);
    }
    // If same edge, sub_a2 already ends at the split and sub_b1 starts there;
    // we handle this by noting sub_a2's end is va and sub_b1's start is also
    // the original edge's start — but actually if they are on the same edge
    // the sub-edges already cover the range. We need an intermediate edge:
    // from va to vb along the original edge. This case is complex; for v1
    // we only handle the common case of hits on different edges.
    if hit_a.edge_idx == hit_b.edge_idx {
        return Err(BlendError::TrimmingFailure { face: face_id });
    }

    // Chain 2: from hit_b to hit_a (wrapping)
    chain2.push(sub_b2);
    for i in (hit_b.edge_idx + 1)..n_edges {
        chain2.push(oriented_edges[i]);
    }
    for i in 0..hit_a.edge_idx {
        chain2.push(oriented_edges[i]);
    }
    chain2.push(sub_a1);

    // ── Pick the kept chain based on TrimSide ────────────────────────
    // Use the face plane normal and contact direction to determine left/right.
    let face_normal = match &surface {
        FaceSurface::Plane { normal, .. } => {
            if reversed {
                -*normal
            } else {
                *normal
            }
        }
        _ => return Err(BlendError::TrimmingFailure { face: face_id }),
    };

    let contact_dir = hit_b.point_3d - hit_a.point_3d;

    // Take a sample point from chain 1 to determine which side it is on.
    let sample_pt = edge_data
        .get(if hit_a.edge_idx + 1 < hit_b.edge_idx {
            hit_a.edge_idx + 1
        } else {
            hit_a.edge_idx
        })
        .map(|(_, s, _)| *s)
        .ok_or(BlendError::TrimmingFailure { face: face_id })?;

    let to_sample = sample_pt - hit_a.point_3d;
    let cross = contact_dir.cross(to_sample);
    let chain1_is_left = face_normal.dot(cross) > 0.0;

    let (kept_chain, contact_forward) = match keep_side {
        TrimSide::Left => {
            if chain1_is_left {
                (chain1, true) // contact va→vb closes the loop
            } else {
                (chain2, false) // contact vb→va closes the loop
            }
        }
        TrimSide::Right => {
            if chain1_is_left {
                (chain2, false)
            } else {
                (chain1, true)
            }
        }
    };

    // ── Build the trimmed wire ───────────────────────────────────────
    let mut loop_edges = kept_chain;
    loop_edges.push(OrientedEdge::new(contact_edge_id, contact_forward));

    let trimmed_wire =
        Wire::new(loop_edges, true).map_err(|_| BlendError::TrimmingFailure { face: face_id })?;
    let trimmed_wire_id = topo.add_wire(trimmed_wire);

    // ── Build the trimmed face ───────────────────────────────────────
    let mut trimmed_face = Face::new(trimmed_wire_id, Vec::new(), surface);
    if reversed {
        trimmed_face.set_reversed(true);
    }
    let trimmed_face_id = topo.add_face(trimmed_face);

    // ── Collect results ──────────────────────────────────────────────
    let new_edges = vec![sub_a1.edge(), sub_a2.edge(), sub_b1.edge(), sub_b2.edge()];

    Ok(TrimResult {
        trimmed_face: trimmed_face_id,
        new_edges,
        new_vertices: vec![va_id, vb_id],
        contact_edge: Some(contact_edge_id),
    })
}

/// Split an oriented boundary edge at a new vertex, producing two sub-edges.
///
/// Returns `(before, after)` as [`OrientedEdge`] values following the same
/// traversal direction as the input.
fn split_edge_at(
    topo: &mut Topology,
    oe: &OrientedEdge,
    split_vertex: VertexId,
) -> Result<(OrientedEdge, OrientedEdge), BlendError> {
    let edge = topo.edge(oe.edge())?;
    let start_vid = oe.oriented_start(edge);
    let end_vid = oe.oriented_end(edge);
    let curve = edge.curve().clone();

    // Sub-edge 1: original start → split vertex
    // Sub-edge 2: split vertex → original end
    // Both use the same curve type (Line for planar faces in v1).
    let e1_id = topo.add_edge(Edge::new(start_vid, split_vertex, curve.clone()));
    let e2_id = topo.add_edge(Edge::new(split_vertex, end_vid, curve));

    // Both sub-edges are traversed forward in the oriented direction
    // because we constructed them S→V and V→E matching the traversal.
    Ok((
        OrientedEdge::new(e1_id, true),
        OrientedEdge::new(e2_id, true),
    ))
}

/// Compute a local 2D coordinate frame for a planar face.
///
/// Returns `(origin, u_axis, v_axis)` where `origin` is the first vertex
/// position and `u_axis`, `v_axis` span the plane.
fn plane_local_frame(
    surface: &FaceSurface,
    edge_data: &[(OrientedEdge, Point3, Point3)],
    face_id: FaceId,
) -> Result<(Point3, brepkit_math::vec::Vec3, brepkit_math::vec::Vec3), BlendError> {
    use brepkit_math::vec::Vec3;

    let normal = match surface {
        FaceSurface::Plane { normal, .. } => *normal,
        _ => return Err(BlendError::TrimmingFailure { face: face_id }),
    };

    let origin = edge_data
        .first()
        .map(|(_, pt, _)| *pt)
        .ok_or(BlendError::TrimmingFailure { face: face_id })?;

    // U-axis: direction along first edge.
    let first_dir = edge_data[0].2 - edge_data[0].1;
    let u_axis = first_dir.normalize().unwrap_or(Vec3::new(1.0, 0.0, 0.0));

    // V-axis: normal × u to complete a right-handed frame.
    let v_axis = normal
        .cross(u_axis)
        .normalize()
        .unwrap_or(Vec3::new(0.0, 1.0, 0.0));

    Ok((origin, u_axis, v_axis))
}

/// Test if two 2D line segments intersect.
///
/// Segments are `(a1→a2)` and `(b1→b2)`. Returns the parameter `t` on
/// segment `a` (in `[0, 1]`) where the crossing occurs, or `None` if
/// the segments do not cross.
fn segment_intersect_2d(
    a1: (f64, f64),
    a2: (f64, f64),
    b1: (f64, f64),
    b2: (f64, f64),
) -> Option<f64> {
    let dx_a = a2.0 - a1.0;
    let dy_a = a2.1 - a1.1;
    let dx_b = b2.0 - b1.0;
    let dy_b = b2.1 - b1.1;

    let denom = dx_a * dy_b - dy_a * dx_b;

    // Parallel or degenerate segments.
    if denom.abs() < PARAM_TOL {
        return None;
    }

    let dx_ab = b1.0 - a1.0;
    let dy_ab = b1.1 - a1.1;

    let t = (dx_ab * dy_b - dy_ab * dx_b) / denom;
    let u = (dx_ab * dy_a - dy_ab * dx_a) / denom;

    // Both parameters must be in [0, 1] for a proper crossing.
    // Use a small tolerance to catch intersections at edge endpoints.
    let valid_range = -PARAM_TOL..=(1.0 + PARAM_TOL);
    if valid_range.contains(&t) && valid_range.contains(&u) {
        Some(t.clamp(0.0, 1.0))
    } else {
        None
    }
}

// ===========================================================================
// Tests
// ===========================================================================

#[cfg(test)]
mod tests {
    #![allow(clippy::unwrap_used, clippy::expect_used, clippy::panic)]

    use super::*;
    use brepkit_math::vec::{Point3, Vec3};
    use brepkit_topology::Topology;

    /// Helper: create a unit square face on the XY plane (z=0).
    ///
    /// Vertices:
    ///   v0 = (0,0,0), v1 = (1,0,0), v2 = (1,1,0), v3 = (0,1,0)
    ///
    /// Edges: v0→v1, v1→v2, v2→v3, v3→v0 (all Line, forward).
    fn make_square_face(topo: &mut Topology) -> (FaceId, [VertexId; 4], [EdgeId; 4]) {
        let v0 = topo.add_vertex(Vertex::new(Point3::new(0.0, 0.0, 0.0), VERTEX_TOL));
        let v1 = topo.add_vertex(Vertex::new(Point3::new(1.0, 0.0, 0.0), VERTEX_TOL));
        let v2 = topo.add_vertex(Vertex::new(Point3::new(1.0, 1.0, 0.0), VERTEX_TOL));
        let v3 = topo.add_vertex(Vertex::new(Point3::new(0.0, 1.0, 0.0), VERTEX_TOL));

        let e0 = topo.add_edge(Edge::new(v0, v1, EdgeCurve::Line)); // bottom
        let e1 = topo.add_edge(Edge::new(v1, v2, EdgeCurve::Line)); // right
        let e2 = topo.add_edge(Edge::new(v2, v3, EdgeCurve::Line)); // top
        let e3 = topo.add_edge(Edge::new(v3, v0, EdgeCurve::Line)); // left

        let wire = Wire::new(
            vec![
                OrientedEdge::new(e0, true),
                OrientedEdge::new(e1, true),
                OrientedEdge::new(e2, true),
                OrientedEdge::new(e3, true),
            ],
            true,
        )
        .unwrap();
        let wire_id = topo.add_wire(wire);

        let surface = FaceSurface::Plane {
            normal: Vec3::new(0.0, 0.0, 1.0),
            d: 0.0,
        };
        let face = Face::new(wire_id, Vec::new(), surface);
        let face_id = topo.add_face(face);

        (face_id, [v0, v1, v2, v3], [e0, e1, e2, e3])
    }

    #[test]
    fn trim_square_face_with_diagonal() {
        let mut topo = Topology::new();
        let (face_id, _verts, _edges) = make_square_face(&mut topo);

        // Contact line: from bottom edge midpoint (0.5, 0) to top edge midpoint (0.5, 1).
        // This is a vertical line splitting the square in half.
        let contact_3d = vec![Point3::new(0.5, 0.0, 0.0), Point3::new(0.5, 1.0, 0.0)];
        let contact_uv = vec![(0.5, 0.0), (0.5, 1.0)];

        let result = trim_face(&mut topo, face_id, &contact_3d, &contact_uv, TrimSide::Left)
            .expect("trim should succeed");

        // The trimmed face should have a new wire.
        let trimmed_face = topo.face(result.trimmed_face).unwrap();
        let trimmed_wire = topo.wire(trimmed_face.outer_wire()).unwrap();

        // Expect 4 edges: bottom-half, right-full, top-half, contact-edge
        // (for the right side) or bottom-half, contact-edge, top-half, left-full
        // (for the left side).
        assert_eq!(
            trimmed_wire.edges().len(),
            4,
            "trimmed wire should have 4 edges"
        );

        // 2 new vertices at the intersection points.
        assert_eq!(result.new_vertices.len(), 2);

        // 4 new sub-edges from splitting 2 boundary edges.
        assert_eq!(result.new_edges.len(), 4);

        // Verify intersection vertex positions.
        let va = topo.vertex(result.new_vertices[0]).unwrap().point();
        let vb = topo.vertex(result.new_vertices[1]).unwrap().point();
        // One should be at (0.5, 0, 0) and the other at (0.5, 1, 0).
        let pts: Vec<(f64, f64, f64)> = vec![(va.x(), va.y(), va.z()), (vb.x(), vb.y(), vb.z())];
        assert!(
            pts.iter()
                .any(|p| (p.0 - 0.5).abs() < 1e-10 && p.1.abs() < 1e-10),
            "expected intersection at (0.5, 0, 0)"
        );
        assert!(
            pts.iter()
                .any(|p| (p.0 - 0.5).abs() < 1e-10 && (p.1 - 1.0).abs() < 1e-10),
            "expected intersection at (0.5, 1, 0)"
        );
    }

    #[test]
    fn trim_preserves_surface() {
        let mut topo = Topology::new();
        let (face_id, _verts, _edges) = make_square_face(&mut topo);

        let contact_3d = vec![Point3::new(0.5, 0.0, 0.0), Point3::new(0.5, 1.0, 0.0)];
        let contact_uv = vec![(0.5, 0.0), (0.5, 1.0)];

        let result = trim_face(
            &mut topo,
            face_id,
            &contact_3d,
            &contact_uv,
            TrimSide::Right,
        )
        .expect("trim should succeed");

        let original = topo.face(face_id).unwrap();
        let trimmed = topo.face(result.trimmed_face).unwrap();

        // Both should be Plane with the same normal.
        match (original.surface(), trimmed.surface()) {
            (
                FaceSurface::Plane {
                    normal: n1, d: d1, ..
                },
                FaceSurface::Plane {
                    normal: n2, d: d2, ..
                },
            ) => {
                assert!((n1.x() - n2.x()).abs() < 1e-14);
                assert!((n1.y() - n2.y()).abs() < 1e-14);
                assert!((n1.z() - n2.z()).abs() < 1e-14);
                assert!((d1 - d2).abs() < 1e-14);
            }
            _ => panic!("expected both faces to be Plane"),
        }
    }

    #[test]
    fn non_planar_face_returns_untrimmed() {
        use brepkit_math::surfaces::CylindricalSurface;

        let mut topo = Topology::new();

        // Create a minimal face with a cylindrical surface.
        let v0 = topo.add_vertex(Vertex::new(Point3::new(1.0, 0.0, 0.0), VERTEX_TOL));
        let v1 = topo.add_vertex(Vertex::new(Point3::new(0.0, 1.0, 0.0), VERTEX_TOL));
        let e0 = topo.add_edge(Edge::new(v0, v1, EdgeCurve::Line));
        let e1 = topo.add_edge(Edge::new(v1, v0, EdgeCurve::Line));

        let wire = Wire::new(
            vec![OrientedEdge::new(e0, true), OrientedEdge::new(e1, true)],
            true,
        )
        .unwrap();
        let wire_id = topo.add_wire(wire);

        let cyl_surface =
            CylindricalSurface::new(Point3::new(0.0, 0.0, 0.0), Vec3::new(0.0, 0.0, 1.0), 1.0)
                .unwrap();
        let surface = FaceSurface::Cylinder(cyl_surface);
        let face = Face::new(wire_id, Vec::new(), surface);
        let face_id = topo.add_face(face);

        let contact_3d = vec![Point3::new(0.5, 0.0, 0.0), Point3::new(0.5, 1.0, 0.0)];
        let contact_uv = vec![(0.5, 0.0), (0.5, 1.0)];

        let result = trim_face(&mut topo, face_id, &contact_3d, &contact_uv, TrimSide::Left)
            .expect("should return untrimmed result");

        // Untrimmed: same face, no new topology.
        assert_eq!(result.trimmed_face, face_id);
        assert!(result.new_edges.is_empty());
        assert!(result.new_vertices.is_empty());
    }

    #[test]
    fn segment_intersect_2d_crossing() {
        // Two crossing segments.
        let t = segment_intersect_2d((0.0, 0.0), (1.0, 1.0), (0.0, 1.0), (1.0, 0.0));
        assert!(t.is_some());
        let t = t.unwrap();
        assert!((t - 0.5).abs() < 1e-10, "t={t}");
    }

    #[test]
    fn segment_intersect_2d_parallel() {
        // Parallel segments — no intersection.
        let t = segment_intersect_2d((0.0, 0.0), (1.0, 0.0), (0.0, 1.0), (1.0, 1.0));
        assert!(t.is_none());
    }

    #[test]
    fn segment_intersect_2d_no_overlap() {
        // Non-parallel but non-overlapping segments.
        let t = segment_intersect_2d((0.0, 0.0), (1.0, 0.0), (2.0, -1.0), (2.0, 1.0));
        assert!(t.is_none());
    }
}
