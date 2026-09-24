#![allow(clippy::unwrap_used, clippy::expect_used, clippy::panic)]

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

/// A box whose faces and edges are all exact NURBS.
fn nurbs_box(topo: &mut Topology) -> SolidId {
    let solid = crate::primitives::make_box(topo, 2.0, 3.0, 4.0).unwrap();
    crate::heal::convert_to_bspline(topo, solid).unwrap();
    solid
}

/// Worst distance from any vertex or edge sample of `solid` to the surface
/// of each face it bounds.
fn worst_boundary_offset(topo: &Topology, solid: SolidId) -> f64 {
    let mut worst: f64 = 0.0;
    for fid in brepkit_topology::explorer::solid_faces(topo, solid).unwrap() {
        let face = topo.face(fid).unwrap();
        let on_surface = |p: brepkit_math::vec::Point3| -> f64 {
            match face.surface() {
                FaceSurface::Plane { normal, d } => {
                    (normal.dot(p - brepkit_math::vec::Point3::new(0.0, 0.0, 0.0)) - d).abs()
                }
                surface => {
                    let (u, v) = surface.project_point(p).unwrap();
                    (surface.evaluate(u, v).unwrap() - p).length()
                }
            }
        };
        let wires = std::iter::once(face.outer_wire()).chain(face.inner_wires().iter().copied());
        for wid in wires {
            for oe in topo.wire(wid).unwrap().edges() {
                let edge = topo.edge(oe.edge()).unwrap();
                let p0 = topo.vertex(edge.start()).unwrap().point();
                let p1 = topo.vertex(edge.end()).unwrap().point();
                let (t0, t1) = edge.curve().domain_with_endpoints(p0, p1);
                for k in 0..=8 {
                    let t = t0 + (t1 - t0) * f64::from(k) / 8.0;
                    worst = worst.max(on_surface(edge.curve().evaluate_with_endpoints(t, p0, p1)));
                }
            }
        }
    }
    worst
}

/// Signed area vector of a face's outer wire, traversed in wire order.
fn outer_wire_area_vector(topo: &Topology, fid: FaceId) -> Vec3 {
    let face = topo.face(fid).unwrap();
    let wire = topo.wire(face.outer_wire()).unwrap();
    let pts: Vec<brepkit_math::vec::Point3> = wire
        .edges()
        .iter()
        .map(|oe| {
            let edge = topo.edge(oe.edge()).unwrap();
            let v = if oe.is_forward() {
                edge.start()
            } else {
                edge.end()
            };
            topo.vertex(v).unwrap().point()
        })
        .collect();
    let origin = pts[0];
    let mut area = Vec3::new(0.0, 0.0, 0.0);
    for w in pts.windows(2).skip(1) {
        area += (w[0] - origin).cross(w[1] - origin) * 0.5;
    }
    area
}

fn mesh_is_watertight_with_volume(topo: &Topology, solid: SolidId) -> f64 {
    let mesh = crate::tessellate::tessellate_solid(topo, solid, 0.01).unwrap();
    assert_eq!(
        crate::tessellate::boundary_edge_count(&mesh),
        0,
        "mesh must be watertight"
    );
    crate::measure::oriented_solid_volume(topo, solid, 0.01).unwrap()
}

/// A rigid motion must move every curved surface WITH its frame: a torus or
/// sphere rebuilt around the world z axis leaves its boundary edges off the
/// surface. The pose also spins each primitive about its own axis, which
/// used to put a rim sample on the cylinder's u seam and twist the band mesh.
#[test]
fn rigid_motion_keeps_curved_surfaces_on_their_boundaries() {
    use crate::primitives::{make_cone, make_cylinder, make_sphere, make_torus};

    type Make = fn(&mut Topology) -> SolidId;
    let pose = Mat4::translation(1.0, 2.0, 3.0)
        * Mat4::rotation_x(0.9)
        * Mat4::rotation_y(-0.4)
        * Mat4::rotation_z(1.3);
    let shapes: [(&str, Make); 4] = [
        ("cylinder", |t| make_cylinder(t, 1.5, 4.0).unwrap()),
        ("cone", |t| make_cone(t, 2.0, 1.0, 3.0).unwrap()),
        ("sphere", |t| make_sphere(t, 2.0, 16).unwrap()),
        ("torus", |t| make_torus(t, 6.0, 1.5, 16).unwrap()),
    ];
    for (name, make) in shapes {
        let mut topo = Topology::new();
        let solid = make(&mut topo);
        let before = mesh_is_watertight_with_volume(&topo, solid);
        // `make_sphere` bounds its hemispheres with a chordal equator, so the
        // sphere starts off its own boundary by the sagitta.
        let offset_before = worst_boundary_offset(&topo, solid);
        transform_solid(&mut topo, solid, &pose).unwrap();
        let offset = worst_boundary_offset(&topo, solid);
        assert!(
            (offset - offset_before).abs() < 1e-9,
            "{name}: boundary moved {offset_before} -> {offset} off its surfaces"
        );
        let after = mesh_is_watertight_with_volume(&topo, solid);
        assert!(
            (after - before).abs() < 1e-9 * before,
            "{name}: mesh volume {before} became {after}"
        );
        assert!(
            crate::validate::validate_solid(&topo, solid)
                .unwrap()
                .is_valid()
        );
    }
}

/// A mirror keeps every outer wire counter-clockwise around its face's
/// outward normal. Without the wire reversal the mirrored solid validates
/// clean but winds every face backwards, which later booleans read as
/// same-direction shared edges.
#[test]
fn mirror_keeps_outer_wires_counter_clockwise() {
    let mut topo = Topology::new();
    let solid = crate::primitives::make_box(&mut topo, 2.0, 3.0, 4.0).unwrap();
    let mirrored = crate::mirror::mirror(
        &mut topo,
        solid,
        brepkit_math::vec::Point3::new(0.0, 0.0, 0.0),
        Vec3::new(1.0, 0.0, 0.0),
    )
    .unwrap();
    for fid in brepkit_topology::explorer::solid_faces(&topo, mirrored).unwrap() {
        let face = topo.face(fid).unwrap();
        let FaceSurface::Plane { normal, .. } = face.surface() else {
            panic!("box faces are planar");
        };
        let outward = if face.is_reversed() {
            -*normal
        } else {
            *normal
        };
        assert!(
            outer_wire_area_vector(&topo, fid).dot(outward) > 0.0,
            "face {fid:?} winds clockwise around its outward normal"
        );
    }

    // Fuse the mirror image back onto an overlapping box: the shared walls
    // must meet with opposite edge senses.
    let other = crate::primitives::make_box(&mut topo, 2.0, 2.0, 2.0).unwrap();
    transform_solid(&mut topo, other, &Mat4::translation(-1.0, 1.0, 1.0)).unwrap();
    let fused =
        crate::boolean::boolean(&mut topo, crate::boolean::BooleanOp::Fuse, mirrored, other)
            .unwrap();
    let report = crate::validate::validate_solid(&topo, fused).unwrap();
    assert!(
        report.is_valid(),
        "fuse of a mirrored box: {:?}",
        report.issues
    );
}

/// A NURBS face's normal is the cross product of its partials, which a
/// mirror reverses; the face flag must flip so the solid stays outward.
#[test]
fn mirror_keeps_nurbs_faces_outward() {
    let mut topo = Topology::new();
    let solid = nurbs_box(&mut topo);
    let mirrored = crate::mirror::mirror(
        &mut topo,
        solid,
        brepkit_math::vec::Point3::new(5.0, 0.0, 0.0),
        Vec3::new(1.0, 1.0, 0.0),
    )
    .unwrap();
    let volume = mesh_is_watertight_with_volume(&topo, mirrored);
    assert!(
        (volume - 24.0).abs() < 1e-9,
        "mirrored NURBS box volume {volume}"
    );
    assert!(
        crate::validate::validate_solid(&topo, mirrored)
            .unwrap()
            .is_valid()
    );
}

/// Stretching a cylinder along its axis keeps it a cylinder; stretching it
/// across the axis makes an elliptic wall, which only NURBS can carry.
#[test]
fn non_uniform_scale_keeps_cylinders_exact() {
    use crate::primitives::make_cylinder;
    let exact = std::f64::consts::PI * 1.5 * 1.5 * 4.0;

    let mut topo = Topology::new();
    let along = make_cylinder(&mut topo, 1.5, 4.0).unwrap();
    transform_solid(&mut topo, along, &Mat4::scale(1.0, 1.0, 3.0)).unwrap();
    let wall = brepkit_topology::explorer::solid_faces(&topo, along)
        .unwrap()
        .into_iter()
        .find_map(|f| match topo.face(f).unwrap().surface() {
            FaceSurface::Cylinder(c) => Some(c.radius()),
            _ => None,
        })
        .expect("an axial stretch keeps the cylinder analytic");
    assert!((wall - 1.5).abs() < 1e-12);
    let v = crate::measure::solid_volume(&topo, along, 0.001).unwrap();
    assert!(
        (v - 3.0 * exact).abs() < 1e-3 * exact,
        "axial stretch volume {v}"
    );

    let across = make_cylinder(&mut topo, 1.5, 4.0).unwrap();
    let squash = Mat4::rotation_z(0.3) * Mat4::scale(2.0, 1.0, 1.0);
    transform_solid(&mut topo, across, &squash).unwrap();
    assert!(worst_boundary_offset(&topo, across) < 1e-9);
    assert!(
        brepkit_topology::explorer::solid_faces(&topo, across)
            .unwrap()
            .iter()
            .all(|&f| !matches!(topo.face(f).unwrap().surface(), FaceSurface::Cylinder(_))),
        "an elliptic wall cannot stay a circular cylinder"
    );
    // The wall's seam lies a quarter turn from the rims' parametric origin
    // (an ellipse starts at its major vertex); the mesh must still close.
    mesh_is_watertight_with_volume(&topo, across);
    let v = crate::measure::solid_volume(&topo, across, 0.01).unwrap();
    assert!(
        (v - 2.0 * exact).abs() < 5e-3 * exact,
        "elliptic cylinder volume {v}"
    );
}

/// The image of a circle under a non-uniform scale is an ellipse whose axes
/// are the image's principal axes, not the images of the circle's own axes.
#[test]
fn non_uniform_scale_of_a_tilted_circle_is_its_exact_ellipse() {
    use brepkit_math::curves::Circle3D;
    use brepkit_math::vec::Point3;
    use brepkit_topology::edge::{Edge, EdgeCurve};
    use brepkit_topology::vertex::Vertex;
    use brepkit_topology::wire::Wire;

    let mut topo = Topology::new();
    let circle = Circle3D::new(
        Point3::new(1.0, -2.0, 0.5),
        Vec3::new(0.3, 0.5, 0.8).normalize().unwrap(),
        2.0,
    )
    .unwrap();
    let start = circle.evaluate(0.0);
    let v = topo.add_vertex(Vertex::new(start, 1e-7));
    let e = topo.add_edge(Edge::new(v, v, EdgeCurve::Circle(circle.clone())));
    let w = topo.add_wire(Wire::new(vec![OrientedEdge::new(e, true)], true).unwrap());

    let m = Mat4::scale(2.0, 1.0, 0.5);
    transform_wire(&mut topo, w, &m).unwrap();
    let EdgeCurve::Ellipse(ellipse) = topo.edge(e).unwrap().curve().clone() else {
        panic!("a squashed circle is an ellipse");
    };
    assert!(ellipse.u_axis().dot(ellipse.v_axis()).abs() < 1e-12);
    for k in 0..32 {
        let p = m.mul_point(circle.evaluate(f64::from(k) * std::f64::consts::TAU / 32.0));
        let d = p - ellipse.center();
        let (x, y) = (d.dot(ellipse.u_axis()), d.dot(ellipse.v_axis()));
        let implicit = (x / ellipse.semi_major()).powi(2) + (y / ellipse.semi_minor()).powi(2);
        assert!(
            (implicit - 1.0).abs() < 1e-12,
            "sample {k} off the ellipse: {implicit}"
        );
        assert!(
            d.dot(ellipse.normal()).abs() < 1e-12,
            "sample {k} off the ellipse plane"
        );
    }
}

/// Stored pcurves stay only where the face's parameterization is carried
/// over: always for NURBS, never for a plane (it has no stored frame).
#[test]
fn transform_drops_pcurves_that_no_longer_fit_their_surface() {
    use brepkit_math::curves2d::{Curve2D, Line2D};
    use brepkit_math::vec::{Point2, Vec2};
    use brepkit_topology::pcurve::PCurve;

    let mut topo = Topology::new();
    let planar = crate::primitives::make_box(&mut topo, 1.0, 1.0, 1.0).unwrap();
    let curved = nurbs_box(&mut topo);
    let first_face = |s: SolidId| brepkit_topology::explorer::solid_faces(&topo, s).unwrap()[0];
    let (plane, nurbs) = (first_face(planar), first_face(curved));
    let edge_of = |f: FaceId| {
        topo.wire(topo.face(f).unwrap().outer_wire())
            .unwrap()
            .edges()[0]
            .edge()
    };
    let (plane_edge, nurbs_edge) = (edge_of(plane), edge_of(nurbs));
    let line = || {
        PCurve::new(
            Curve2D::Line(Line2D::new(Point2::new(0.0, 0.0), Vec2::new(1.0, 0.0)).unwrap()),
            0.0,
            1.0,
        )
    };
    topo.pcurves_mut().set(plane_edge, plane, line());
    topo.pcurves_mut().set(nurbs_edge, nurbs, line());

    transform_solid(&mut topo, planar, &Mat4::rotation_x(0.4)).unwrap();
    transform_solid(&mut topo, curved, &Mat4::rotation_x(0.4)).unwrap();
    assert!(!topo.pcurves().contains(plane_edge, plane));
    assert!(topo.pcurves().contains(nurbs_edge, nurbs));
}
