//! Face offset: create a new face offset from an existing face by a given
//! distance along its surface normal.
//!
//! For planar faces, this is an exact operation (translate along normal).
//! For NURBS faces, the offset is approximated by sampling the surface
//! normal field and refitting via surface interpolation.

use brepkit_math::nurbs::surface_fitting::interpolate_surface;
use brepkit_math::tolerance::Tolerance;
use brepkit_math::vec::{Point3, Vec3};
use brepkit_topology::Topology;
use brepkit_topology::face::{FaceId, FaceSurface};

use crate::OperationsError;

/// Create a new face that is offset from `face_id` by `distance` along
/// the outward surface normal.
///
/// Positive distance offsets outward (away from the solid interior),
/// negative distance offsets inward.
///
/// For planar faces, the operation is exact. For NURBS faces, the
/// surface is sampled at a grid of `samples × samples` points and
/// re-interpolated.
///
/// # Errors
///
/// Returns an error if:
/// - The face lookup fails
/// - NURBS surface normal computation fails at any sample point
/// - Surface re-interpolation fails
pub fn offset_face(
    topo: &mut Topology,
    face_id: FaceId,
    distance: f64,
    samples: usize,
) -> Result<FaceId, OperationsError> {
    let tol = Tolerance::new();
    if distance.abs() < tol.linear {
        // Zero offset: just copy the face.
        return copy_face(topo, face_id);
    }

    let face = topo.face(face_id)?;
    let surface = face.surface().clone();
    let outer_wire = face.outer_wire();
    let inner_wires: Vec<_> = face.inner_wires().to_vec();

    match surface {
        FaceSurface::Plane { normal, d } => {
            offset_planar_face(topo, outer_wire, &inner_wires, normal, d, distance)
        }
        FaceSurface::Nurbs(ref nurbs) => offset_nurbs_face(topo, face_id, nurbs, distance, samples),
        FaceSurface::Cylinder(_)
        | FaceSurface::Cone(_)
        | FaceSurface::Sphere(_)
        | FaceSurface::Torus(_) => Err(OperationsError::InvalidInput {
            reason: "offset of analytic surface faces is not yet supported".into(),
        }),
    }
}

/// Offset a planar face: shift plane along its normal.
fn offset_planar_face(
    topo: &mut Topology,
    outer_wire: brepkit_topology::wire::WireId,
    inner_wires: &[brepkit_topology::wire::WireId],
    normal: Vec3,
    d: f64,
    distance: f64,
) -> Result<FaceId, OperationsError> {
    // New plane: same normal, shifted d.
    let new_d = d + distance;

    // Offset all vertices in the wires.
    let offset_vec = Vec3::new(
        normal.x() * distance,
        normal.y() * distance,
        normal.z() * distance,
    );

    let new_outer = offset_wire_vertices(topo, outer_wire, offset_vec)?;

    let mut new_inner = Vec::new();
    for &iw in inner_wires {
        let new_iw = offset_wire_vertices(topo, iw, offset_vec)?;
        new_inner.push(new_iw);
    }

    let new_surface = FaceSurface::Plane { normal, d: new_d };
    let face_id = topo.faces.alloc(brepkit_topology::face::Face::new(
        new_outer,
        new_inner,
        new_surface,
    ));
    Ok(face_id)
}

/// Offset a NURBS face by sampling and refitting.
fn offset_nurbs_face(
    topo: &mut Topology,
    face_id: FaceId,
    nurbs: &brepkit_math::nurbs::NurbsSurface,
    distance: f64,
    samples: usize,
) -> Result<FaceId, OperationsError> {
    let n = samples.max(4); // Minimum 4×4 grid.
    #[allow(clippy::cast_precision_loss)]
    let divisor = (n - 1) as f64;

    // Sample the surface and offset each point along the normal.
    let mut offset_grid: Vec<Vec<Point3>> = Vec::with_capacity(n);

    #[allow(clippy::cast_precision_loss)]
    for i in 0..n {
        let u = i as f64 / divisor;
        let mut row = Vec::with_capacity(n);
        for j in 0..n {
            let v = j as f64 / divisor;

            let pt = nurbs.evaluate(u, v);
            let normal = nurbs
                .normal(u, v)
                .map_err(|e| OperationsError::InvalidInput {
                    reason: format!("NURBS normal computation failed at ({u}, {v}): {e}"),
                })?;

            let offset_pt = Point3::new(
                normal.x().mul_add(distance, pt.x()),
                normal.y().mul_add(distance, pt.y()),
                normal.z().mul_add(distance, pt.z()),
            );
            row.push(offset_pt);
        }
        offset_grid.push(row);
    }

    // Fit a new NURBS surface through the offset points.
    let degree = nurbs.degree_u().min(nurbs.degree_v()).min(3);
    let offset_surface = interpolate_surface(&offset_grid, degree, degree).map_err(|e| {
        OperationsError::InvalidInput {
            reason: format!("offset surface interpolation failed: {e}"),
        }
    })?;

    // Copy the wire topology from the original face, offsetting vertices.
    let face = topo.face(face_id)?;
    let outer_wire = face.outer_wire();
    let inner_wires: Vec<_> = face.inner_wires().to_vec();

    // Offset wire vertices along the NURBS normal at their positions.
    let new_outer = offset_wire_along_nurbs(topo, outer_wire, nurbs, distance)?;

    let mut new_inner = Vec::new();
    for &iw in &inner_wires {
        let new_iw = offset_wire_along_nurbs(topo, iw, nurbs, distance)?;
        new_inner.push(new_iw);
    }

    let new_surface = FaceSurface::Nurbs(offset_surface);
    let face_id = topo.faces.alloc(brepkit_topology::face::Face::new(
        new_outer,
        new_inner,
        new_surface,
    ));
    Ok(face_id)
}

/// Copy a face with new IDs.
fn copy_face(topo: &mut Topology, face_id: FaceId) -> Result<FaceId, OperationsError> {
    let face = topo.face(face_id)?;
    let surface = face.surface().clone();
    let outer_wire = face.outer_wire();
    let inner_wires: Vec<_> = face.inner_wires().to_vec();

    let new_outer = copy_wire(topo, outer_wire)?;
    let mut new_inner = Vec::new();
    for &iw in &inner_wires {
        new_inner.push(copy_wire(topo, iw)?);
    }

    let new_face = topo.faces.alloc(brepkit_topology::face::Face::new(
        new_outer, new_inner, surface,
    ));
    Ok(new_face)
}

/// Copy a wire with new vertex and edge IDs.
fn copy_wire(
    topo: &mut Topology,
    wire_id: brepkit_topology::wire::WireId,
) -> Result<brepkit_topology::wire::WireId, OperationsError> {
    use brepkit_topology::edge::Edge;
    use brepkit_topology::edge::EdgeCurve;
    use brepkit_topology::vertex::Vertex;
    use brepkit_topology::wire::{OrientedEdge, Wire};

    let wire = topo.wire(wire_id)?;
    let edges = wire.edges().to_vec();

    // Snapshot edge data before allocating (borrow checker).
    let mut edge_snaps: Vec<(Point3, f64, Point3, f64, EdgeCurve, bool)> = Vec::new();
    for oe in &edges {
        let edge = topo.edge(oe.edge())?;
        let start = topo.vertex(edge.start())?;
        let end = topo.vertex(edge.end())?;
        edge_snaps.push((
            start.point(),
            start.tolerance(),
            end.point(),
            end.tolerance(),
            edge.curve().clone(),
            oe.is_forward(),
        ));
    }

    let mut new_oriented = Vec::new();
    for (start_pt, start_tol, end_pt, end_tol, curve, forward) in edge_snaps {
        let new_start = topo.vertices.alloc(Vertex::new(start_pt, start_tol));
        let new_end = topo.vertices.alloc(Vertex::new(end_pt, end_tol));
        let new_edge = topo.edges.alloc(Edge::new(new_start, new_end, curve));
        new_oriented.push(OrientedEdge::new(new_edge, forward));
    }

    let new_wire = topo.wires.alloc(Wire::new(new_oriented, true)?);
    Ok(new_wire)
}

/// Offset all vertices in a wire by a constant vector.
fn offset_wire_vertices(
    topo: &mut Topology,
    wire_id: brepkit_topology::wire::WireId,
    offset: Vec3,
) -> Result<brepkit_topology::wire::WireId, OperationsError> {
    use brepkit_topology::edge::{Edge, EdgeCurve};
    use brepkit_topology::vertex::Vertex;
    use brepkit_topology::wire::{OrientedEdge, Wire};

    let wire = topo.wire(wire_id)?;
    let edges = wire.edges().to_vec();

    // Snapshot then allocate.
    let mut edge_snaps: Vec<(Point3, Point3, EdgeCurve, bool)> = Vec::new();
    for oe in &edges {
        let edge = topo.edge(oe.edge())?;
        let start_pt = topo.vertex(edge.start())?.point();
        let end_pt = topo.vertex(edge.end())?.point();
        edge_snaps.push((start_pt, end_pt, edge.curve().clone(), oe.is_forward()));
    }

    let mut new_oriented = Vec::new();
    for (start_pt, end_pt, curve, forward) in edge_snaps {
        let new_start = topo.vertices.alloc(Vertex::new(start_pt + offset, 1e-7));
        let new_end = topo.vertices.alloc(Vertex::new(end_pt + offset, 1e-7));
        let new_edge = topo.edges.alloc(Edge::new(new_start, new_end, curve));
        new_oriented.push(OrientedEdge::new(new_edge, forward));
    }

    let new_wire = topo.wires.alloc(Wire::new(new_oriented, true)?);
    Ok(new_wire)
}

/// Offset wire vertices along the NURBS surface normal at their closest
/// parametric position.
fn offset_wire_along_nurbs(
    topo: &mut Topology,
    wire_id: brepkit_topology::wire::WireId,
    nurbs: &brepkit_math::nurbs::NurbsSurface,
    distance: f64,
) -> Result<brepkit_topology::wire::WireId, OperationsError> {
    use brepkit_topology::edge::{Edge, EdgeCurve};
    use brepkit_topology::vertex::Vertex;
    use brepkit_topology::wire::{OrientedEdge, Wire};

    let wire = topo.wire(wire_id)?;
    let edges = wire.edges().to_vec();

    // Snapshot then compute offsets.
    let mut snaps: Vec<(Point3, Point3, bool)> = Vec::new();
    for oe in &edges {
        let edge = topo.edge(oe.edge())?;
        let start_pt = topo.vertex(edge.start())?.point();
        let end_pt = topo.vertex(edge.end())?.point();
        snaps.push((start_pt, end_pt, oe.is_forward()));
    }

    let mut new_oriented = Vec::new();
    for (start_pt, end_pt, forward) in snaps {
        let new_start_pt = offset_point_on_surface(nurbs, start_pt, distance)?;
        let new_end_pt = offset_point_on_surface(nurbs, end_pt, distance)?;

        let new_start = topo.vertices.alloc(Vertex::new(new_start_pt, 1e-7));
        let new_end = topo.vertices.alloc(Vertex::new(new_end_pt, 1e-7));
        let new_edge = topo
            .edges
            .alloc(Edge::new(new_start, new_end, EdgeCurve::Line));
        new_oriented.push(OrientedEdge::new(new_edge, forward));
    }

    let new_wire = topo.wires.alloc(Wire::new(new_oriented, true)?);
    Ok(new_wire)
}

/// Offset a single point along the surface normal at its closest parametric
/// position on the NURBS surface.
fn offset_point_on_surface(
    nurbs: &brepkit_math::nurbs::NurbsSurface,
    point: Point3,
    distance: f64,
) -> Result<Point3, OperationsError> {
    use brepkit_math::nurbs::projection::project_point_to_surface;

    let proj = project_point_to_surface(nurbs, point, 1e-7).map_err(|e| {
        OperationsError::InvalidInput {
            reason: format!("surface projection failed: {e}"),
        }
    })?;
    let u = proj.u;
    let v = proj.v;
    let normal = nurbs
        .normal(u, v)
        .map_err(|e| OperationsError::InvalidInput {
            reason: format!("NURBS normal at ({u}, {v}) failed: {e}"),
        })?;

    Ok(Point3::new(
        normal.x().mul_add(distance, point.x()),
        normal.y().mul_add(distance, point.y()),
        normal.z().mul_add(distance, point.z()),
    ))
}

#[cfg(test)]
mod tests {
    #![allow(clippy::unwrap_used, clippy::expect_used, clippy::panic)]

    use brepkit_topology::Topology;
    use brepkit_topology::test_utils::make_unit_square_face;

    use super::*;

    #[test]
    fn offset_planar_face_outward() {
        let mut topo = Topology::new();
        let face = make_unit_square_face(&mut topo);

        let offset = offset_face(&mut topo, face, 1.0, 10).unwrap();

        // The offset face should have a shifted plane.
        let offset_face = topo.face(offset).unwrap();
        match offset_face.surface() {
            FaceSurface::Plane { normal, d } => {
                // Unit square face is on z=0, normal=(0,0,1), d=0.
                // Offset by 1.0 should give d=1.0.
                assert!((normal.z() - 1.0).abs() < 1e-6);
                assert!((d - 1.0).abs() < 1e-6);
            }
            _ => panic!("expected planar surface"),
        }
    }

    #[test]
    fn offset_planar_face_inward() {
        let mut topo = Topology::new();
        let face = make_unit_square_face(&mut topo);

        let offset = offset_face(&mut topo, face, -0.5, 10).unwrap();

        let offset_face = topo.face(offset).unwrap();
        match offset_face.surface() {
            FaceSurface::Plane { d, .. } => {
                assert!((d - (-0.5)).abs() < 1e-6);
            }
            _ => panic!("expected planar surface"),
        }
    }

    #[test]
    fn offset_zero_returns_copy() {
        let mut topo = Topology::new();
        let face = make_unit_square_face(&mut topo);

        let offset = offset_face(&mut topo, face, 0.0, 10).unwrap();

        // Should return a different face ID (it's a copy).
        assert_ne!(face, offset);

        // But same surface properties.
        let original = topo.face(face).unwrap();
        let copied = topo.face(offset).unwrap();
        match (original.surface(), copied.surface()) {
            (
                FaceSurface::Plane { normal: n1, d: d1 },
                FaceSurface::Plane { normal: n2, d: d2 },
            ) => {
                assert!((n1.x() - n2.x()).abs() < 1e-10);
                assert!((n1.y() - n2.y()).abs() < 1e-10);
                assert!((n1.z() - n2.z()).abs() < 1e-10);
                assert!((d1 - d2).abs() < 1e-10);
            }
            _ => panic!("expected both planar"),
        }
    }

    #[test]
    fn offset_face_preserves_vertex_count() {
        let mut topo = Topology::new();
        let face = make_unit_square_face(&mut topo);

        let offset = offset_face(&mut topo, face, 2.0, 10).unwrap();

        // Count vertices in original vs offset.
        let orig_face = topo.face(face).unwrap();
        let offset_face = topo.face(offset).unwrap();

        let orig_wire = topo.wire(orig_face.outer_wire()).unwrap();
        let off_wire = topo.wire(offset_face.outer_wire()).unwrap();

        assert_eq!(orig_wire.edges().len(), off_wire.edges().len());
    }

    #[test]
    fn offset_vertices_are_shifted() {
        let mut topo = Topology::new();
        let face = make_unit_square_face(&mut topo);

        let offset = offset_face(&mut topo, face, 3.0, 10).unwrap();

        // Get a vertex from the offset face and check it's shifted.
        let off_face = topo.face(offset).unwrap();
        let off_wire = topo.wire(off_face.outer_wire()).unwrap();
        let first_edge = off_wire.edges()[0];
        let edge = topo.edge(first_edge.edge()).unwrap();
        let vert = topo.vertex(edge.start()).unwrap();

        // Original unit square has vertices at z=0.
        // Offset of 3.0 along +Z normal should give z=3.0.
        assert!(
            (vert.point().z() - 3.0).abs() < 1e-6,
            "expected z=3.0, got z={}",
            vert.point().z()
        );
    }

    // ── NURBS face helpers ────────────────────────────────────────────────────

    /// Build a flat bilinear NURBS face on the XY plane (z = `z_height`).
    ///
    /// Degree-1 in both u and v, 2×2 control points, clamped knot vectors.
    /// The wire has 4 line-edges forming a unit square at `z_height`.
    fn make_flat_nurbs_face(topo: &mut Topology, z_height: f64) -> FaceId {
        use brepkit_math::nurbs::NurbsSurface;
        use brepkit_math::vec::Point3 as P;
        use brepkit_topology::edge::{Edge, EdgeCurve};
        use brepkit_topology::face::Face;
        use brepkit_topology::vertex::Vertex;
        use brepkit_topology::wire::{OrientedEdge, Wire};

        // 2×2 control-point grid spanning [0,1]×[0,1] at given z.
        let ctrl = vec![
            vec![P::new(0.0, 0.0, z_height), P::new(1.0, 0.0, z_height)],
            vec![P::new(0.0, 1.0, z_height), P::new(1.0, 1.0, z_height)],
        ];
        let weights = vec![vec![1.0_f64, 1.0], vec![1.0, 1.0]];
        // Clamped knot vector for degree 1, 2 control points: [0,0,1,1]
        let knots = vec![0.0, 0.0, 1.0, 1.0];
        let nurbs = NurbsSurface::new(1, 1, knots.clone(), knots, ctrl, weights).unwrap();

        // Wire: unit square in XY at z_height.
        let tol = 1e-7;
        let v0 = topo
            .vertices
            .alloc(Vertex::new(P::new(0.0, 0.0, z_height), tol));
        let v1 = topo
            .vertices
            .alloc(Vertex::new(P::new(1.0, 0.0, z_height), tol));
        let v2 = topo
            .vertices
            .alloc(Vertex::new(P::new(1.0, 1.0, z_height), tol));
        let v3 = topo
            .vertices
            .alloc(Vertex::new(P::new(0.0, 1.0, z_height), tol));

        let e0 = topo.edges.alloc(Edge::new(v0, v1, EdgeCurve::Line));
        let e1 = topo.edges.alloc(Edge::new(v1, v2, EdgeCurve::Line));
        let e2 = topo.edges.alloc(Edge::new(v2, v3, EdgeCurve::Line));
        let e3 = topo.edges.alloc(Edge::new(v3, v0, EdgeCurve::Line));

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
        let wid = topo.wires.alloc(wire);

        topo.faces
            .alloc(Face::new(wid, vec![], FaceSurface::Nurbs(nurbs)))
    }

    // ── NURBS face offset tests ───────────────────────────────────────────────

    #[test]
    fn offset_nurbs_face_outward_produces_nurbs_surface() {
        let mut topo = Topology::new();
        let face = make_flat_nurbs_face(&mut topo, 0.0);

        let offset_id = offset_face(&mut topo, face, 1.0, 6).unwrap();

        // The result must carry a NURBS surface.
        let off_face = topo.face(offset_id).unwrap();
        assert!(
            matches!(off_face.surface(), FaceSurface::Nurbs(_)),
            "expected NURBS surface after NURBS offset"
        );
    }

    #[test]
    fn offset_nurbs_face_new_id_differs_from_original() {
        let mut topo = Topology::new();
        let face = make_flat_nurbs_face(&mut topo, 0.0);

        let offset_id = offset_face(&mut topo, face, 0.5, 6).unwrap();

        assert_ne!(face, offset_id, "offset should return a new face ID");
    }

    #[test]
    fn offset_nurbs_face_wire_has_same_edge_count() {
        let mut topo = Topology::new();
        let face = make_flat_nurbs_face(&mut topo, 0.0);

        let offset_id = offset_face(&mut topo, face, 1.0, 6).unwrap();

        let orig_wire = topo.wire(topo.face(face).unwrap().outer_wire()).unwrap();
        let off_wire = topo
            .wire(topo.face(offset_id).unwrap().outer_wire())
            .unwrap();
        assert_eq!(
            orig_wire.edges().len(),
            off_wire.edges().len(),
            "offset wire should have the same edge count as the original"
        );
    }

    #[test]
    fn offset_nurbs_face_negative_distance() {
        let mut topo = Topology::new();
        // Place the surface at z=2 so a negative offset moves it toward z=1.
        let face = make_flat_nurbs_face(&mut topo, 0.0);

        let offset_id = offset_face(&mut topo, face, -0.5, 6).unwrap();

        // Result is still a valid NURBS face.
        let off_face = topo.face(offset_id).unwrap();
        assert!(matches!(off_face.surface(), FaceSurface::Nurbs(_)));
    }

    #[test]
    fn offset_nurbs_face_very_small_distance() {
        let mut topo = Topology::new();
        let face = make_flat_nurbs_face(&mut topo, 0.0);

        // A distance well above the linear tolerance (1e-7) but very small.
        let offset_id = offset_face(&mut topo, face, 1e-4, 6).unwrap();

        let off_face = topo.face(offset_id).unwrap();
        assert!(matches!(off_face.surface(), FaceSurface::Nurbs(_)));
    }

    #[test]
    fn offset_nurbs_face_zero_returns_copy() {
        let mut topo = Topology::new();
        let face = make_flat_nurbs_face(&mut topo, 0.0);

        // Exactly zero: should go through the copy_face path, keeping NURBS.
        let copy_id = offset_face(&mut topo, face, 0.0, 6).unwrap();

        assert_ne!(face, copy_id);
        let copy_face = topo.face(copy_id).unwrap();
        assert!(
            matches!(copy_face.surface(), FaceSurface::Nurbs(_)),
            "zero offset of NURBS face should still be NURBS"
        );
    }

    #[test]
    fn offset_analytic_surface_returns_error() {
        use brepkit_math::surfaces::CylindricalSurface;
        use brepkit_math::vec::{Point3 as P, Vec3};
        use brepkit_topology::edge::{Edge, EdgeCurve};
        use brepkit_topology::face::Face;
        use brepkit_topology::vertex::Vertex;
        use brepkit_topology::wire::{OrientedEdge, Wire};

        let mut topo = Topology::new();

        // Build a minimal face with a Cylinder surface (not yet supported by offset).
        let tol = 1e-7;
        let v0 = topo.vertices.alloc(Vertex::new(P::new(0.0, 0.0, 0.0), tol));
        let v1 = topo.vertices.alloc(Vertex::new(P::new(1.0, 0.0, 0.0), tol));
        let v2 = topo.vertices.alloc(Vertex::new(P::new(1.0, 1.0, 0.0), tol));
        let v3 = topo.vertices.alloc(Vertex::new(P::new(0.0, 1.0, 0.0), tol));
        let e0 = topo.edges.alloc(Edge::new(v0, v1, EdgeCurve::Line));
        let e1 = topo.edges.alloc(Edge::new(v1, v2, EdgeCurve::Line));
        let e2 = topo.edges.alloc(Edge::new(v2, v3, EdgeCurve::Line));
        let e3 = topo.edges.alloc(Edge::new(v3, v0, EdgeCurve::Line));
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
        let wid = topo.wires.alloc(wire);
        let cyl =
            CylindricalSurface::new(P::new(0.0, 0.0, 0.0), Vec3::new(0.0, 0.0, 1.0), 1.0).unwrap();
        let face_id = topo
            .faces
            .alloc(Face::new(wid, vec![], FaceSurface::Cylinder(cyl)));

        let result = offset_face(&mut topo, face_id, 1.0, 6);
        assert!(
            result.is_err(),
            "offset of analytic surface should return an error"
        );
    }

    #[test]
    fn offset_nurbs_face_large_distance() {
        let mut topo = Topology::new();
        let face = make_flat_nurbs_face(&mut topo, 0.0);

        // A large positive offset should still succeed (the surface is flat so
        // normals are uniform and well-defined everywhere).
        let offset_id = offset_face(&mut topo, face, 100.0, 8).unwrap();

        let off_face = topo.face(offset_id).unwrap();
        assert!(matches!(off_face.surface(), FaceSurface::Nurbs(_)));
    }

    #[test]
    fn offset_nurbs_face_minimum_samples_clamped() {
        let mut topo = Topology::new();
        let face = make_flat_nurbs_face(&mut topo, 0.0);

        // Passing samples=1 should be clamped to 4 internally and still succeed.
        let offset_id = offset_face(&mut topo, face, 1.0, 1).unwrap();

        let off_face = topo.face(offset_id).unwrap();
        assert!(matches!(off_face.surface(), FaceSurface::Nurbs(_)));
    }
}
