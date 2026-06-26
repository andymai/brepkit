#![allow(clippy::unwrap_used)]

use brepkit_math::mat::Mat4;
use brepkit_math::tolerance::Tolerance;
use brepkit_topology::Topology;
use brepkit_topology::face::FaceSurface;
use brepkit_topology::test_utils::make_unit_cube_non_manifold;

use super::*;

#[test]
fn translate_cube() {
    let mut topo = Topology::new();
    let solid = make_unit_cube_non_manifold(&mut topo);
    let matrix = Mat4::translation(1.0, 0.0, 0.0);

    transform_solid(&mut topo, solid, &matrix).unwrap();

    // All vertices should have x shifted by 1.0.
    let tol = Tolerance::new();
    for (_id, v) in topo.vertices().iter() {
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
    let solid = make_unit_cube_non_manifold(&mut topo);

    let before: Vec<_> = topo.vertices().iter().map(|(_, v)| v.point()).collect();

    transform_solid(&mut topo, solid, &Mat4::identity()).unwrap();

    let tol = Tolerance::new();
    for (i, (_, v)) in topo.vertices().iter().enumerate() {
        assert!(tol.approx_eq(v.point().x(), before[i].x()));
        assert!(tol.approx_eq(v.point().y(), before[i].y()));
        assert!(tol.approx_eq(v.point().z(), before[i].z()));
    }
}

#[test]
fn degenerate_matrix_error() {
    let mut topo = Topology::new();
    let solid = make_unit_cube_non_manifold(&mut topo);
    let matrix = Mat4::scale(0.0, 1.0, 1.0);

    let result = transform_solid(&mut topo, solid, &matrix);
    assert!(result.is_err());
}

/// Rotating a cube 90 degrees around the Z axis should update face normals.
#[test]
fn rotation_updates_face_normals() {
    let mut topo = Topology::new();
    let solid = make_unit_cube_non_manifold(&mut topo);

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

/// Build a minimal solid containing a single face with the given surface.
///
/// The wire is a unit square in XY; only the face surface type varies.
fn make_single_face_solid(
    topo: &mut Topology,
    surface: FaceSurface,
) -> brepkit_topology::solid::SolidId {
    use brepkit_math::vec::Point3;
    use brepkit_topology::edge::{Edge, EdgeCurve};
    use brepkit_topology::face::Face;
    use brepkit_topology::shell::Shell;
    use brepkit_topology::solid::Solid;
    use brepkit_topology::vertex::Vertex;
    use brepkit_topology::wire::{OrientedEdge, Wire};

    let tol = 1e-7;
    let v0 = topo.add_vertex(Vertex::new(Point3::new(0.0, 0.0, 0.0), tol));
    let v1 = topo.add_vertex(Vertex::new(Point3::new(1.0, 0.0, 0.0), tol));
    let v2 = topo.add_vertex(Vertex::new(Point3::new(1.0, 1.0, 0.0), tol));
    let v3 = topo.add_vertex(Vertex::new(Point3::new(0.0, 1.0, 0.0), tol));

    let e0 = topo.add_edge(Edge::new(v0, v1, EdgeCurve::Line));
    let e1 = topo.add_edge(Edge::new(v1, v2, EdgeCurve::Line));
    let e2 = topo.add_edge(Edge::new(v2, v3, EdgeCurve::Line));
    let e3 = topo.add_edge(Edge::new(v3, v0, EdgeCurve::Line));

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
    let wid = topo.add_wire(wire);
    let fid = topo.add_face(Face::new(wid, vec![], surface));
    let shell = Shell::new(vec![fid]).unwrap();
    let shell_id = topo.add_shell(shell);
    topo.add_solid(Solid::new(shell_id, vec![]))
}

#[test]
fn translate_cylinder_face_updates_origin() {
    use brepkit_math::surfaces::CylindricalSurface;
    use brepkit_math::vec::Point3;

    let mut topo = Topology::new();
    let cyl =
        CylindricalSurface::new(Point3::new(0.0, 0.0, 0.0), Vec3::new(0.0, 0.0, 1.0), 2.0).unwrap();
    let solid = make_single_face_solid(&mut topo, FaceSurface::Cylinder(cyl));

    let matrix = Mat4::translation(5.0, 3.0, 1.0);
    transform_solid(&mut topo, solid, &matrix).unwrap();

    // Find the (now-transformed) cylinder face.
    let tol = Tolerance::new();
    let solid_data = topo.solid(solid).unwrap();
    let shell = topo.shell(solid_data.outer_shell()).unwrap();
    let mut found = false;
    for &fid in shell.faces() {
        if let FaceSurface::Cylinder(c) = topo.face(fid).unwrap().surface() {
            assert!(
                tol.approx_eq(c.origin().x(), 5.0),
                "cylinder origin x should be 5.0, got {}",
                c.origin().x()
            );
            assert!(
                tol.approx_eq(c.origin().y(), 3.0),
                "cylinder origin y should be 3.0, got {}",
                c.origin().y()
            );
            assert!(
                tol.approx_eq(c.origin().z(), 1.0),
                "cylinder origin z should be 1.0, got {}",
                c.origin().z()
            );
            // The axis (0,0,1) should be unchanged by a pure translation.
            assert!(
                tol.approx_eq(c.axis().z(), 1.0),
                "cylinder axis z should still be 1.0"
            );
            assert!(
                tol.approx_eq(c.radius(), 2.0),
                "cylinder radius should be unchanged"
            );
            found = true;
        }
    }
    assert!(found, "cylinder face not found after transform");
}

#[test]
fn rotate_cylinder_face_updates_axis() {
    use brepkit_math::surfaces::CylindricalSurface;
    use brepkit_math::vec::Point3;

    let mut topo = Topology::new();
    // Cylinder with axis along +Z.
    let cyl =
        CylindricalSurface::new(Point3::new(0.0, 0.0, 0.0), Vec3::new(0.0, 0.0, 1.0), 1.0).unwrap();
    let solid = make_single_face_solid(&mut topo, FaceSurface::Cylinder(cyl));

    // 90° rotation around Y: Z-axis → X-axis
    let matrix = Mat4::rotation_y(std::f64::consts::FRAC_PI_2);
    transform_solid(&mut topo, solid, &matrix).unwrap();

    let tol = Tolerance::loose();
    let solid_data = topo.solid(solid).unwrap();
    let shell = topo.shell(solid_data.outer_shell()).unwrap();
    let mut found = false;
    for &fid in shell.faces() {
        if let FaceSurface::Cylinder(c) = topo.face(fid).unwrap().surface() {
            // After 90° Y rotation, original Z-axis should point along +X.
            assert!(
                tol.approx_eq(c.axis().x().abs(), 1.0),
                "cylinder axis should be along X after Y rotation, got {:?}",
                c.axis()
            );
            found = true;
        }
    }
    assert!(found, "cylinder face not found after rotation");
}

#[test]
fn translate_cone_face_updates_apex() {
    use brepkit_math::surfaces::ConicalSurface;
    use brepkit_math::vec::Point3;

    let mut topo = Topology::new();
    let cone = ConicalSurface::new(
        Point3::new(0.0, 0.0, 0.0),
        Vec3::new(0.0, 0.0, 1.0),
        std::f64::consts::FRAC_PI_4,
    )
    .unwrap();
    let solid = make_single_face_solid(&mut topo, FaceSurface::Cone(cone));

    let matrix = Mat4::translation(2.0, 4.0, 6.0);
    transform_solid(&mut topo, solid, &matrix).unwrap();

    let tol = Tolerance::new();
    let solid_data = topo.solid(solid).unwrap();
    let shell = topo.shell(solid_data.outer_shell()).unwrap();
    let mut found = false;
    for &fid in shell.faces() {
        if let FaceSurface::Cone(c) = topo.face(fid).unwrap().surface() {
            assert!(
                tol.approx_eq(c.apex().x(), 2.0),
                "cone apex x should be 2.0, got {}",
                c.apex().x()
            );
            assert!(
                tol.approx_eq(c.apex().y(), 4.0),
                "cone apex y should be 4.0"
            );
            assert!(
                tol.approx_eq(c.apex().z(), 6.0),
                "cone apex z should be 6.0"
            );
            // Axis should be unchanged by a translation.
            assert!(
                tol.approx_eq(c.axis().z(), 1.0),
                "cone axis z should still be 1.0"
            );
            found = true;
        }
    }
    assert!(found, "cone face not found after transform");
}

#[test]
fn translate_sphere_face_updates_center() {
    use brepkit_math::surfaces::SphericalSurface;
    use brepkit_math::vec::Point3;

    let mut topo = Topology::new();
    let sphere = SphericalSurface::new(Point3::new(0.0, 0.0, 0.0), 3.0).unwrap();
    let solid = make_single_face_solid(&mut topo, FaceSurface::Sphere(sphere));

    let matrix = Mat4::translation(-1.0, 2.0, 5.0);
    transform_solid(&mut topo, solid, &matrix).unwrap();

    let tol = Tolerance::new();
    let solid_data = topo.solid(solid).unwrap();
    let shell = topo.shell(solid_data.outer_shell()).unwrap();
    let mut found = false;
    for &fid in shell.faces() {
        if let FaceSurface::Sphere(s) = topo.face(fid).unwrap().surface() {
            assert!(
                tol.approx_eq(s.center().x(), -1.0),
                "sphere center x should be -1.0"
            );
            assert!(
                tol.approx_eq(s.center().y(), 2.0),
                "sphere center y should be 2.0"
            );
            assert!(
                tol.approx_eq(s.center().z(), 5.0),
                "sphere center z should be 5.0"
            );
            assert!(
                tol.approx_eq(s.radius(), 3.0),
                "sphere radius should be unchanged"
            );
            found = true;
        }
    }
    assert!(found, "sphere face not found after transform");
}

#[test]
fn translate_torus_face_updates_center() {
    use brepkit_math::surfaces::ToroidalSurface;
    use brepkit_math::vec::Point3;

    let mut topo = Topology::new();
    let torus = ToroidalSurface::new(Point3::new(0.0, 0.0, 0.0), 5.0, 1.5).unwrap();
    let solid = make_single_face_solid(&mut topo, FaceSurface::Torus(torus));

    let matrix = Mat4::translation(10.0, -3.0, 0.5);
    transform_solid(&mut topo, solid, &matrix).unwrap();

    let tol = Tolerance::new();
    let solid_data = topo.solid(solid).unwrap();
    let shell = topo.shell(solid_data.outer_shell()).unwrap();
    let mut found = false;
    for &fid in shell.faces() {
        if let FaceSurface::Torus(t) = topo.face(fid).unwrap().surface() {
            assert!(
                tol.approx_eq(t.center().x(), 10.0),
                "torus center x should be 10.0"
            );
            assert!(
                tol.approx_eq(t.center().y(), -3.0),
                "torus center y should be -3.0"
            );
            assert!(
                tol.approx_eq(t.center().z(), 0.5),
                "torus center z should be 0.5"
            );
            assert!(
                tol.approx_eq(t.major_radius(), 5.0),
                "torus major radius should be unchanged"
            );
            assert!(
                tol.approx_eq(t.minor_radius(), 1.5),
                "torus minor radius should be unchanged"
            );
            found = true;
        }
    }
    assert!(found, "torus face not found after transform");
}

#[test]
fn transform_direction_zero_vector_is_error() {
    // A zero direction vector cannot be normalized and must return an error.
    // This exercises the normalize() error branch in transform_direction.
    let result = super::transform_direction(&Mat4::identity(), Vec3::new(0.0, 0.0, 0.0));
    assert!(
        result.is_err(),
        "transform_direction with zero vector should return an error"
    );
}

#[test]
fn transform_direction_unit_z_identity_unchanged() {
    // Identity matrix should leave a unit direction unchanged.
    let dir = Vec3::new(0.0, 0.0, 1.0);
    let result = super::transform_direction(&Mat4::identity(), dir).unwrap();
    let tol = Tolerance::new();
    assert!(tol.approx_eq(result.z(), 1.0), "z should remain 1.0");
    assert!(tol.approx_eq(result.x(), 0.0), "x should remain 0.0");
    assert!(tol.approx_eq(result.y(), 0.0), "y should remain 0.0");
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

    // Build a NURBS-faced solid by lofting two offset squares: a smooth loft
    // produces genuine NURBS side surfaces (revolve of a polygonal profile is now
    // recognised as analytic cone/cylinder/plane bands, so it no longer yields
    // NURBS walls — see the revolve analytic-surface recognition).
    let square = |topo: &mut Topology, half: f64, z: f64| -> FaceId {
        let tol_val = 1e-10;
        let a = topo.add_vertex(Vertex::new(Point3::new(-half, -half, z), tol_val));
        let b = topo.add_vertex(Vertex::new(Point3::new(half, -half, z), tol_val));
        let c = topo.add_vertex(Vertex::new(Point3::new(half, half, z), tol_val));
        let d = topo.add_vertex(Vertex::new(Point3::new(-half, half, z), tol_val));
        let e0 = topo.add_edge(Edge::new(a, b, EdgeCurve::Line));
        let e1 = topo.add_edge(Edge::new(b, c, EdgeCurve::Line));
        let e2 = topo.add_edge(Edge::new(c, d, EdgeCurve::Line));
        let e3 = topo.add_edge(Edge::new(d, a, EdgeCurve::Line));
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
        let wid = topo.add_wire(wire);
        topo.add_face(Face::new(
            wid,
            vec![],
            FaceSurface::Plane {
                normal: brepkit_math::vec::Vec3::new(0.0, 0.0, 1.0),
                d: z,
            },
        ))
    };
    let p0 = square(&mut topo, 3.0, 0.0);
    let p1 = square(&mut topo, 2.0, 2.0);
    let p2 = square(&mut topo, 3.0, 4.0);
    let solid = crate::loft::loft_smooth(&mut topo, &[p0, p1, p2]).unwrap();

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

#[test]
fn translate_wire() {
    use brepkit_math::vec::Point3;
    use brepkit_topology::builder::make_polygon_wire;

    let mut topo = Topology::new();
    let wire = make_polygon_wire(
        &mut topo,
        &[
            Point3::new(0.0, 0.0, 0.0),
            Point3::new(1.0, 0.0, 0.0),
            Point3::new(1.0, 1.0, 0.0),
        ],
        1e-7,
    )
    .unwrap();

    transform_wire(&mut topo, wire, &Mat4::translation(5.0, 0.0, 0.0)).unwrap();

    // All vertices should have x shifted by 5 (original x values were 0, 1, 1).
    let tol = Tolerance::new();
    let w = topo.wire(wire).unwrap();
    for oe in w.edges() {
        let edge = topo.edge(oe.edge()).unwrap();
        let x = topo.vertex(edge.start()).unwrap().point().x();
        assert!(
            tol.approx_eq(x, 5.0) || tol.approx_eq(x, 6.0),
            "vertex x should be 5.0 or 6.0 after translation, got {x}"
        );
    }
}

#[test]
fn degenerate_matrix_errors_for_wire() {
    use brepkit_math::vec::Point3;
    use brepkit_topology::builder::make_polygon_wire;

    let mut topo = Topology::new();
    let wire = make_polygon_wire(
        &mut topo,
        &[
            Point3::new(0.0, 0.0, 0.0),
            Point3::new(1.0, 0.0, 0.0),
            Point3::new(1.0, 1.0, 0.0),
        ],
        1e-7,
    )
    .unwrap();

    let result = transform_wire(&mut topo, wire, &Mat4::scale(0.0, 1.0, 1.0));
    assert!(result.is_err());
}

#[test]
fn translate_wire_with_circle_edge() {
    use brepkit_math::curves::Circle3D;
    use brepkit_math::vec::{Point3, Vec3};
    use brepkit_topology::edge::{Edge, EdgeCurve};
    use brepkit_topology::vertex::Vertex;
    use brepkit_topology::wire::{OrientedEdge, Wire};

    let mut topo = Topology::new();
    let v = topo.add_vertex(Vertex::new(Point3::new(1.0, 0.0, 0.0), 1e-7));
    let circle = Circle3D::new(Point3::new(0.0, 0.0, 0.0), Vec3::new(0.0, 0.0, 1.0), 1.0).unwrap();
    let edge = topo.add_edge(Edge::new(v, v, EdgeCurve::Circle(circle)));
    let wire = Wire::new(vec![OrientedEdge::new(edge, true)], true).unwrap();
    let wid = topo.add_wire(wire);

    transform_wire(&mut topo, wid, &Mat4::translation(5.0, 0.0, 0.0)).unwrap();

    // Vertex should be shifted.
    let tol = Tolerance::new();
    let pos = topo.vertex(v).unwrap().point();
    assert!(
        tol.approx_eq(pos.x(), 6.0),
        "vertex should be at x=6 after +5 translation, got {}",
        pos.x()
    );

    // Circle center should also be shifted.
    let w = topo.wire(wid).unwrap();
    let e = topo.edge(w.edges()[0].edge()).unwrap();
    assert!(
        matches!(e.curve(), EdgeCurve::Circle(_)),
        "expected Circle edge after transform"
    );
    if let EdgeCurve::Circle(c) = e.curve() {
        assert!(
            tol.approx_eq(c.center().x(), 5.0),
            "circle center should be at x=5, got {}",
            c.center().x()
        );
    }
}
