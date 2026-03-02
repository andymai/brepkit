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
            FaceSurface::Nurbs(_) => panic!("expected planar surface"),
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
            FaceSurface::Nurbs(_) => panic!("expected planar surface"),
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
}
