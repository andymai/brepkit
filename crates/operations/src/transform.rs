//! Affine transforms applied to topological shapes.

use std::collections::HashSet;

use brepkit_math::mat::Mat4;
use brepkit_math::nurbs::curve::NurbsCurve;
use brepkit_math::nurbs::surface::NurbsSurface;
use brepkit_math::tolerance::Tolerance;
use brepkit_math::vec::Vec3;
use brepkit_topology::Topology;
use brepkit_topology::edge::{EdgeCurve, EdgeId};
use brepkit_topology::face::{FaceId, FaceSurface};
use brepkit_topology::solid::SolidId;
use brepkit_topology::vertex::VertexId;

/// Apply an affine transform to a solid, modifying vertex positions and
/// face surface geometry in place.
///
/// The transform matrix must be non-degenerate (non-zero determinant).
/// All unique vertices reachable from the solid's shells are transformed,
/// NURBS edge curves and face surfaces have their control points updated,
/// and all planar face normals are updated using the inverse transpose.
///
/// # Errors
///
/// Returns an error if the matrix is degenerate or a referenced entity is missing.
#[allow(clippy::too_many_lines)]
pub fn transform_solid(
    topo: &mut Topology,
    solid: SolidId,
    matrix: &Mat4,
) -> Result<(), crate::OperationsError> {
    let tol = Tolerance::new();
    if tol.approx_eq(matrix.determinant(), 0.0) {
        return Err(crate::OperationsError::InvalidInput {
            reason: "transform matrix is degenerate (zero determinant)".into(),
        });
    }

    // Collect all unique vertex IDs, edge IDs, and face IDs in a read phase.
    let (vertex_ids, edge_ids, face_ids) = collect_solid_entities(topo, solid)?;

    // Mutate phase 1: transform each vertex.
    for vid in vertex_ids {
        let vertex = topo.vertex_mut(vid)?;
        let new_point = matrix.mul_point(vertex.point());
        vertex.set_point(new_point);
    }

    // Mutate phase 2: transform NURBS edge curves.
    // Line edges need no update — their geometry is defined by vertices.
    for eid in edge_ids {
        let edge = topo.edge(eid)?;
        if let EdgeCurve::NurbsCurve(c) = edge.curve() {
            let new_control_points: Vec<_> = c
                .control_points()
                .iter()
                .map(|pt| matrix.mul_point(*pt))
                .collect();
            let new_curve = NurbsCurve::new(
                c.degree(),
                c.knots().to_vec(),
                new_control_points,
                c.weights().to_vec(),
            );
            topo.edge_mut(eid)?
                .set_curve(EdgeCurve::NurbsCurve(new_curve?));
        }
    }

    // Mutate phase 3: transform face surface geometry.
    // For plane normals, use the inverse transpose: n' = (M⁻¹)ᵀ · n
    let normal_matrix = matrix.inverse()?.transpose();

    for fid in face_ids {
        let face = topo.face(fid)?;
        match face.surface() {
            FaceSurface::Plane { normal, .. } => {
                let n = *normal;
                // Transform the normal via the inverse transpose (treating it as
                // a direction, so we use mul_point on a point at (nx, ny, nz)
                // and subtract the translation component).
                let transformed =
                    normal_matrix.mul_point(brepkit_math::vec::Point3::new(n.x(), n.y(), n.z()));
                // Extract direction only (ignore any translation component from
                // the inverse transpose by subtracting the origin transform).
                let origin = normal_matrix.mul_point(brepkit_math::vec::Point3::new(0.0, 0.0, 0.0));
                let raw = Vec3::new(
                    transformed.x() - origin.x(),
                    transformed.y() - origin.y(),
                    transformed.z() - origin.z(),
                );
                let new_normal = raw.normalize()?;

                // Recompute d from a transformed vertex on this face. We use
                // the first vertex of the outer wire.
                let wire = topo.wire(face.outer_wire())?;
                let first_oe = &wire.edges()[0];
                let edge = topo.edge(first_oe.edge())?;
                let ref_vid = if first_oe.is_forward() {
                    edge.start()
                } else {
                    edge.end()
                };
                let ref_point = topo.vertex(ref_vid)?.point();
                let new_d = new_normal.dot(Vec3::new(ref_point.x(), ref_point.y(), ref_point.z()));

                // Now mutate the face.
                let face_mut = topo.face_mut(fid)?;
                face_mut.set_surface(FaceSurface::Plane {
                    normal: new_normal,
                    d: new_d,
                });
            }
            FaceSurface::Nurbs(s) => {
                let new_control_points: Vec<Vec<_>> = s
                    .control_points()
                    .iter()
                    .map(|row| row.iter().map(|pt| matrix.mul_point(*pt)).collect())
                    .collect();
                let new_surface = NurbsSurface::new(
                    s.degree_u(),
                    s.degree_v(),
                    s.knots_u().to_vec(),
                    s.knots_v().to_vec(),
                    new_control_points,
                    s.weights().to_vec(),
                );
                topo.face_mut(fid)?
                    .set_surface(FaceSurface::Nurbs(new_surface?));
            }
            FaceSurface::Cylinder(cyl) => {
                let new_origin = matrix.mul_point(cyl.origin());
                let new_axis = transform_direction(matrix, cyl.axis())?;
                let new_cyl = brepkit_math::surfaces::CylindricalSurface::new(
                    new_origin,
                    new_axis,
                    cyl.radius(),
                )?;
                topo.face_mut(fid)?
                    .set_surface(FaceSurface::Cylinder(new_cyl));
            }
            FaceSurface::Cone(cone) => {
                let new_apex = matrix.mul_point(cone.apex());
                let new_axis = transform_direction(matrix, cone.axis())?;
                let new_cone = brepkit_math::surfaces::ConicalSurface::new(
                    new_apex,
                    new_axis,
                    cone.half_angle(),
                )?;
                topo.face_mut(fid)?.set_surface(FaceSurface::Cone(new_cone));
            }
            FaceSurface::Sphere(sph) => {
                let new_center = matrix.mul_point(sph.center());
                let new_sph =
                    brepkit_math::surfaces::SphericalSurface::new(new_center, sph.radius())?;
                topo.face_mut(fid)?
                    .set_surface(FaceSurface::Sphere(new_sph));
            }
            FaceSurface::Torus(tor) => {
                let new_center = matrix.mul_point(tor.center());
                let new_tor = brepkit_math::surfaces::ToroidalSurface::new(
                    new_center,
                    tor.major_radius(),
                    tor.minor_radius(),
                )?;
                topo.face_mut(fid)?.set_surface(FaceSurface::Torus(new_tor));
            }
        }
    }

    Ok(())
}

/// Transforms a direction vector by applying the matrix and subtracting the
/// translation component, then normalizing.
fn transform_direction(matrix: &Mat4, dir: Vec3) -> Result<Vec3, crate::OperationsError> {
    let origin = matrix.mul_point(brepkit_math::vec::Point3::new(0.0, 0.0, 0.0));
    let tip = matrix.mul_point(brepkit_math::vec::Point3::new(dir.x(), dir.y(), dir.z()));
    let raw = Vec3::new(
        tip.x() - origin.x(),
        tip.y() - origin.y(),
        tip.z() - origin.z(),
    );
    Ok(raw.normalize()?)
}

/// Traverses solid → shells → faces → wires → edges → vertices and
/// returns deduplicated sets of vertex IDs, edge IDs, and face IDs.
#[allow(clippy::type_complexity)]
fn collect_solid_entities(
    topo: &Topology,
    solid: SolidId,
) -> Result<(HashSet<VertexId>, HashSet<EdgeId>, HashSet<FaceId>), crate::OperationsError> {
    let mut vertex_ids = HashSet::new();
    let mut edge_ids = HashSet::new();
    let mut face_ids = HashSet::new();
    let solid_data = topo.solid(solid)?;
    let shell_ids: Vec<_> = std::iter::once(solid_data.outer_shell())
        .chain(solid_data.inner_shells().iter().copied())
        .collect();

    for shell_id in shell_ids {
        let shell = topo.shell(shell_id)?;
        let fids: Vec<_> = shell.faces().to_vec();

        for face_id in fids {
            face_ids.insert(face_id);
            let face = topo.face(face_id)?;
            let wire_ids: Vec<_> = std::iter::once(face.outer_wire())
                .chain(face.inner_wires().iter().copied())
                .collect();

            for wire_id in wire_ids {
                let wire = topo.wire(wire_id)?;
                for oe in wire.edges() {
                    let eid = oe.edge();
                    edge_ids.insert(eid);
                    let edge = topo.edge(eid)?;
                    vertex_ids.insert(edge.start());
                    vertex_ids.insert(edge.end());
                }
            }
        }
    }

    Ok((vertex_ids, edge_ids, face_ids))
}

#[cfg(test)]
mod tests {
    #![allow(clippy::unwrap_used)]

    use brepkit_math::mat::Mat4;
    use brepkit_math::tolerance::Tolerance;
    use brepkit_topology::Topology;
    use brepkit_topology::face::FaceSurface;
    use brepkit_topology::test_utils::make_unit_cube;

    use super::*;

    #[test]
    fn translate_cube() {
        let mut topo = Topology::new();
        let solid = make_unit_cube(&mut topo);
        let matrix = Mat4::translation(1.0, 0.0, 0.0);

        transform_solid(&mut topo, solid, &matrix).unwrap();

        // All vertices should have x shifted by 1.0.
        let tol = Tolerance::new();
        for (_id, v) in topo.vertices.iter() {
            let x = v.point().x();
            assert!(
                tol.approx_eq(x, 1.0) || tol.approx_eq(x, 2.0),
                "unexpected x = {x}"
            );
        }
    }

    #[test]
    fn identity_transform_no_change() {
        let mut topo = Topology::new();
        let solid = make_unit_cube(&mut topo);

        let before: Vec<_> = topo.vertices.iter().map(|(_, v)| v.point()).collect();

        transform_solid(&mut topo, solid, &Mat4::identity()).unwrap();

        let tol = Tolerance::new();
        for (i, (_, v)) in topo.vertices.iter().enumerate() {
            assert!(tol.approx_eq(v.point().x(), before[i].x()));
            assert!(tol.approx_eq(v.point().y(), before[i].y()));
            assert!(tol.approx_eq(v.point().z(), before[i].z()));
        }
    }

    #[test]
    fn degenerate_matrix_error() {
        let mut topo = Topology::new();
        let solid = make_unit_cube(&mut topo);
        let matrix = Mat4::scale(0.0, 1.0, 1.0);

        let result = transform_solid(&mut topo, solid, &matrix);
        assert!(result.is_err());
    }

    /// Rotating a cube 90 degrees around the Z axis should update face normals.
    #[test]
    fn rotation_updates_face_normals() {
        let mut topo = Topology::new();
        let solid = make_unit_cube(&mut topo);

        // 90-degree rotation around Z: +X face normal → +Y, -X → -Y, etc.
        let matrix = Mat4::rotation_z(std::f64::consts::FRAC_PI_2);
        transform_solid(&mut topo, solid, &matrix).unwrap();

        let tol = Tolerance::loose();
        let solid_data = topo.solid(solid).unwrap();
        let shell = topo.shell(solid_data.outer_shell()).unwrap();

        // Collect all plane normals.
        let mut normals: Vec<Vec3> = Vec::new();
        for &fid in shell.faces() {
            let f = topo.face(fid).unwrap();
            if let FaceSurface::Plane { normal, .. } = f.surface() {
                normals.push(*normal);
            }
        }

        // Original cube had normals along ±X, ±Y, ±Z.
        // After 90° Z-rotation: ±X → ±Y, ±Y → ∓X, ±Z unchanged.
        // So we should still have 6 normals, each approximately axis-aligned.
        assert_eq!(normals.len(), 6);

        // Check that we still have a +Z and -Z normal (unchanged by Z rotation).
        let has_pos_z = normals
            .iter()
            .any(|n| tol.approx_eq(n.z(), 1.0) && tol.approx_eq(n.x(), 0.0));
        let has_neg_z = normals
            .iter()
            .any(|n| tol.approx_eq(n.z(), -1.0) && tol.approx_eq(n.x(), 0.0));
        assert!(has_pos_z, "should have +Z normal after Z rotation");
        assert!(has_neg_z, "should have -Z normal after Z rotation");
    }

    /// Revolving a face produces NURBS surfaces; translating the result
    /// should move both vertices and NURBS control points.
    #[test]
    fn transform_nurbs_solid() {
        use brepkit_math::vec::Point3;
        use brepkit_topology::edge::{Edge, EdgeCurve};
        use brepkit_topology::face::Face;
        use brepkit_topology::vertex::Vertex;
        use brepkit_topology::wire::{OrientedEdge, Wire};

        let mut topo = Topology::new();

        // Create a small rectangular face at x=[2,3], z=[0,1], y=0.
        // This is offset from the Y axis so the revolve doesn't self-intersect.
        let tol_val = 1e-10;
        let v0 = topo
            .vertices
            .alloc(Vertex::new(Point3::new(2.0, 0.0, 0.0), tol_val));
        let v1 = topo
            .vertices
            .alloc(Vertex::new(Point3::new(3.0, 0.0, 0.0), tol_val));
        let v2 = topo
            .vertices
            .alloc(Vertex::new(Point3::new(3.0, 0.0, 1.0), tol_val));
        let v3 = topo
            .vertices
            .alloc(Vertex::new(Point3::new(2.0, 0.0, 1.0), tol_val));

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

        let normal = brepkit_math::vec::Vec3::new(0.0, -1.0, 0.0);
        let rect = topo.faces.alloc(Face::new(
            wid,
            vec![],
            FaceSurface::Plane { normal, d: 0.0 },
        ));

        // Revolve 90° around the Y axis to produce NURBS faces.
        let solid = crate::revolve::revolve(
            &mut topo,
            rect,
            Point3::new(0.0, 0.0, 0.0),
            brepkit_math::vec::Vec3::new(0.0, 1.0, 0.0),
            std::f64::consts::FRAC_PI_2,
        )
        .unwrap();

        // Record a NURBS surface control point before the transform.
        let solid_data = topo.solid(solid).unwrap();
        let shell = topo.shell(solid_data.outer_shell()).unwrap();
        let mut original_nurbs_cp = None;
        for &fid in shell.faces() {
            let f = topo.face(fid).unwrap();
            if let FaceSurface::Nurbs(s) = f.surface() {
                original_nurbs_cp = Some(s.control_points()[0][0]);
                break;
            }
        }
        let original_cp = original_nurbs_cp.unwrap();

        // Translate by (10, 0, 0).
        let matrix = Mat4::translation(10.0, 0.0, 0.0);
        transform_solid(&mut topo, solid, &matrix).unwrap();

        // Verify NURBS control points have shifted.
        let tol = Tolerance::new();
        let solid_data = topo.solid(solid).unwrap();
        let shell = topo.shell(solid_data.outer_shell()).unwrap();
        let mut found = false;
        for &fid in shell.faces() {
            let f = topo.face(fid).unwrap();
            if let FaceSurface::Nurbs(s) = f.surface() {
                let cp = s.control_points()[0][0];
                assert!(
                    tol.approx_eq(cp.x(), original_cp.x() + 10.0),
                    "NURBS control point x should shift by 10, got {} (was {})",
                    cp.x(),
                    original_cp.x()
                );
                assert!(
                    tol.approx_eq(cp.y(), original_cp.y()),
                    "NURBS control point y should be unchanged"
                );
                assert!(
                    tol.approx_eq(cp.z(), original_cp.z()),
                    "NURBS control point z should be unchanged"
                );
                found = true;
                break;
            }
        }
        assert!(found, "should still have NURBS faces after transform");
    }
}
