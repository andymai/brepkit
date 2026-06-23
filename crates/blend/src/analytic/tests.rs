#![allow(clippy::unwrap_used, clippy::expect_used, clippy::panic)]

use super::*;
use brepkit_topology::edge::{Edge, EdgeCurve};
use brepkit_topology::face::Face;
use brepkit_topology::vertex::Vertex;
use brepkit_topology::wire::{OrientedEdge, Wire};

/// Create a spine along a single edge from `a` to `b`, plus two dummy faces.
fn make_spine(topo: &mut Topology, a: Point3, b: Point3) -> (Spine, FaceId, FaceId) {
    let v0 = topo.add_vertex(Vertex::new(a, 1e-7));
    let v1 = topo.add_vertex(Vertex::new(b, 1e-7));
    let eid = topo.add_edge(Edge::new(v0, v1, EdgeCurve::Line));

    let oe = OrientedEdge::new(eid, true);
    let w1 = topo.add_wire(Wire::new(vec![oe], false).unwrap());
    let w2 = topo.add_wire(Wire::new(vec![OrientedEdge::new(eid, true)], false).unwrap());
    let f1 = topo.add_face(Face::new(
        w1,
        vec![],
        FaceSurface::Plane {
            normal: Vec3::new(0.0, 0.0, 1.0),
            d: 0.0,
        },
    ));
    let f2 = topo.add_face(Face::new(
        w2,
        vec![],
        FaceSurface::Plane {
            normal: Vec3::new(0.0, 1.0, 0.0),
            d: 0.0,
        },
    ));

    let spine = Spine::from_single_edge(topo, eid).unwrap();
    (spine, f1, f2)
}

#[test]
fn plane_plane_90_degree_fillet() {
    let mut topo = Topology::new();

    // Two perpendicular planes meeting along X axis at origin
    let n1 = Vec3::new(0.0, 0.0, 1.0); // XY plane (top)
    let n2 = Vec3::new(0.0, 1.0, 0.0); // XZ plane (front)
    let (spine, f1, f2) = make_spine(
        &mut topo,
        Point3::new(0.0, 0.0, 0.0),
        Point3::new(10.0, 0.0, 0.0),
    );

    let radius = 2.0;
    let result = plane_plane_fillet(&spine, &topo, n1, n2, radius, f1, f2).unwrap();

    // The result surface should be a cylinder
    match &result.stripe.surface {
        FaceSurface::Cylinder(cyl) => {
            assert!(
                (cyl.radius() - radius).abs() < 1e-10,
                "Expected radius {radius}, got {}",
                cyl.radius()
            );

            let axis = cyl.axis();
            assert!(
                axis.dot(Vec3::new(1.0, 0.0, 0.0)).abs() > 0.99,
                "Cylinder axis should be along X, got {axis:?}"
            );
        }
        other => panic!("Expected Cylinder surface, got {other:?}"),
    }

    // Contact curves should be lines parallel to X axis
    let c1_start = result.stripe.contact1.evaluate(0.0);
    let c1_end = result.stripe.contact1.evaluate(1.0);
    let c1_dir = (c1_end - c1_start).normalize().unwrap();
    assert!(
        c1_dir.dot(Vec3::new(1.0, 0.0, 0.0)).abs() > 0.99,
        "Contact 1 should be along X"
    );

    // Half-angle for 90 deg is pi/4, so offset = R/sin(pi/4) = R*sqrt(2)
    let half_angle = std::f64::consts::FRAC_PI_4;
    let expected_offset = radius / half_angle.sin();
    let sections = &result.stripe.sections;
    assert_eq!(sections.len(), 2);
    assert!((sections[0].radius - radius).abs() < 1e-10);

    // Center should be offset from origin by R/sin(45deg) along bisector
    let bisector = (n1 + n2).normalize().unwrap();
    let expected_center = Point3::new(0.0, 0.0, 0.0) + bisector * expected_offset;
    let actual_center = sections[0].center;
    assert!(
        (actual_center - expected_center).length() < 1e-10,
        "Expected center at {expected_center:?}, got {actual_center:?}"
    );
}

#[test]
fn plane_plane_60_degree_fillet() {
    let mut topo = Topology::new();

    let n1 = Vec3::new(0.0, 0.0, 1.0);
    // Normal at 60 deg from n1
    let angle = std::f64::consts::FRAC_PI_3;
    let n2 = Vec3::new(0.0, angle.sin(), angle.cos());
    let n2 = n2.normalize().unwrap();

    let (spine, f1, f2) = make_spine(
        &mut topo,
        Point3::new(0.0, 0.0, 0.0),
        Point3::new(5.0, 0.0, 0.0),
    );

    let radius = 1.5;
    let result = plane_plane_fillet(&spine, &topo, n1, n2, radius, f1, f2).unwrap();

    match &result.stripe.surface {
        FaceSurface::Cylinder(cyl) => {
            assert!(
                (cyl.radius() - radius).abs() < 1e-10,
                "Expected radius {radius}, got {}",
                cyl.radius()
            );
        }
        other => panic!("Expected Cylinder surface, got {other:?}"),
    }

    // Verify center offset matches expected geometry
    let cos_angle = n1.dot(n2);
    let half = cos_angle.acos() / 2.0;
    let expected_offset = radius / half.sin();

    let center = result.stripe.sections[0].center;
    let origin = Point3::new(0.0, 0.0, 0.0);
    let actual_offset = (center - origin).length();
    assert!(
        (actual_offset - expected_offset).abs() < 1e-10,
        "Expected offset {expected_offset}, got {actual_offset}"
    );
}

#[test]
fn plane_plane_chamfer_is_flat() {
    let mut topo = Topology::new();

    let n1 = Vec3::new(0.0, 0.0, 1.0);
    let n2 = Vec3::new(0.0, 1.0, 0.0);
    let (spine, f1, f2) = make_spine(
        &mut topo,
        Point3::new(0.0, 0.0, 0.0),
        Point3::new(10.0, 0.0, 0.0),
    );

    let d1 = 3.0;
    let d2 = 2.0;
    let result = plane_plane_chamfer(&spine, &topo, n1, n2, d1, d2, f1, f2).unwrap();

    // The result should be a plane
    match &result.stripe.surface {
        FaceSurface::Plane { normal, d } => {
            // Normal should be perpendicular to the spine direction (X axis)
            let spine_dir = Vec3::new(1.0, 0.0, 0.0);
            assert!(
                normal.dot(spine_dir).abs() < 1e-10,
                "Chamfer normal should be perpendicular to spine, dot={:.6}",
                normal.dot(spine_dir)
            );
            assert!(
                (normal.length() - 1.0).abs() < 1e-10,
                "Normal should be unit length"
            );
            assert!(d.is_finite(), "d should be finite");
        }
        other => panic!("Expected Plane surface for chamfer, got {other:?}"),
    }

    // Contact curves should be lines parallel to X
    let c1_start = result.stripe.contact1.evaluate(0.0);
    let c1_end = result.stripe.contact1.evaluate(1.0);
    let c1_dir = (c1_end - c1_start).normalize().unwrap();
    assert!(
        c1_dir.dot(Vec3::new(1.0, 0.0, 0.0)).abs() > 0.99,
        "Contact 1 should be along X"
    );
}

#[test]
fn non_analytic_returns_none() {
    let mut topo = Topology::new();

    // One NURBS surface — should return None
    let nurbs_surf = FaceSurface::Nurbs(
        brepkit_math::nurbs::surface::NurbsSurface::new(
            1,
            1,
            vec![0.0, 0.0, 1.0, 1.0],
            vec![0.0, 0.0, 1.0, 1.0],
            vec![
                vec![Point3::new(0.0, 0.0, 0.0), Point3::new(1.0, 0.0, 0.0)],
                vec![Point3::new(0.0, 1.0, 0.0), Point3::new(1.0, 1.0, 0.0)],
            ],
            vec![vec![1.0, 1.0], vec![1.0, 1.0]],
        )
        .unwrap(),
    );
    let plane_surf = FaceSurface::Plane {
        normal: Vec3::new(0.0, 0.0, 1.0),
        d: 0.0,
    };

    let (spine, f1, f2) = make_spine(
        &mut topo,
        Point3::new(0.0, 0.0, 0.0),
        Point3::new(1.0, 0.0, 0.0),
    );

    let result = try_analytic_fillet(&nurbs_surf, &plane_surf, &spine, &topo, 1.0, f1, f2).unwrap();
    assert!(result.is_none(), "NURBS-Plane pair should return None");

    let result =
        try_analytic_chamfer(&nurbs_surf, &plane_surf, &spine, &topo, 1.0, 1.0, f1, f2).unwrap();
    assert!(result.is_none(), "NURBS-Plane chamfer should return None");
}

/// Concave plane-cylinder fillet — verifies the analytic helper's
/// concave branch (cylinder face reversed = "hole through plate")
/// emits a torus with `major = r_c − r` (NOT `r_c + r`), positioned
/// to touch the plate inside the spine and the cylinder lateral
/// inside the hole.
///
/// Built via direct topology synthesis since `boolean(Cut)` would
/// tessellate the cylinder lateral and yield a polygonal hole.
#[test]
fn plane_cylinder_fillet_concave_emits_torus_with_smaller_major() {
    use brepkit_math::curves::Circle3D;
    use brepkit_math::surfaces::CylindricalSurface;
    use brepkit_topology::edge::{Edge, EdgeCurve};
    use brepkit_topology::face::Face;
    use brepkit_topology::vertex::Vertex;
    use brepkit_topology::wire::{OrientedEdge, Wire};

    let mut topo = Topology::new();
    let r_c: f64 = 2.0;
    let r_fillet = 0.3;

    // Spine: a closed Circle3D edge of radius r_c around the +z axis,
    // sharing a single vertex (start == end) since the spine wraps.
    let v = topo.add_vertex(Vertex::new(Point3::new(r_c, 0.0, 0.0), 1e-7));
    let circle = Circle3D::new(Point3::new(0.0, 0.0, 0.0), Vec3::new(0.0, 0.0, 1.0), r_c).unwrap();
    let eid = topo.add_edge(Edge::new(v, v, EdgeCurve::Circle(circle)));
    let spine = Spine::from_single_edge(&topo, eid).unwrap();

    // Plate face — non-reversed, normal = +z (top of plate).
    let w1 = topo.add_wire(Wire::new(vec![OrientedEdge::new(eid, true)], true).unwrap());
    let face_plate = topo.add_face(Face::new(
        w1,
        vec![],
        FaceSurface::Plane {
            normal: Vec3::new(0.0, 0.0, 1.0),
            d: 0.0,
        },
    ));

    // Cylinder face — REVERSED, marking it as the wall of a hole
    // (topological outward = -radial, opposite to the geometric +radial
    // returned by ParametricSurface::normal).
    let cyl_surface =
        CylindricalSurface::new(Point3::new(0.0, 0.0, 0.0), Vec3::new(0.0, 0.0, 1.0), r_c).unwrap();
    let w2 = topo.add_wire(Wire::new(vec![OrientedEdge::new(eid, false)], true).unwrap());
    let face_cyl = topo.add_face(Face::new_reversed(
        w2,
        vec![],
        FaceSurface::Cylinder(cyl_surface.clone()),
    ));

    // The fillet dispatcher applies `orient_plane_surface` so the helper
    // sees the inward plane normal. For a plate top with outward = +z
    // and material below, inward = -z.
    let n_p_inward = Vec3::new(0.0, 0.0, -1.0);

    let result = plane_cylinder_fillet(
        n_p_inward,
        0.0,
        &cyl_surface,
        &spine,
        &topo,
        r_fillet,
        face_plate,
        face_cyl,
    )
    .unwrap()
    .expect("concave plane-cylinder fillet should produce a stripe");

    let torus = match result.stripe.surface {
        FaceSurface::Torus(t) => t,
        other => panic!("expected Torus, got {}", other.type_tag()),
    };

    // Concave: major = r_c − r_fillet = 1.7, minor = 0.3.
    assert!(
        (torus.minor_radius() - r_fillet).abs() < 1e-9,
        "torus minor should equal fillet radius {r_fillet}, got {}",
        torus.minor_radius()
    );
    assert!(
        (torus.major_radius() - (r_c - r_fillet)).abs() < 1e-9,
        "torus major should be r_c − r_fillet = {} for concave, got {}",
        r_c - r_fillet,
        torus.major_radius()
    );

    // The torus center sits at `+r` ABOVE the plate (in the empty
    // wedge direction = -n_p_inward = +z), distinguishing the concave
    // case from the convex one (which would have center at `-r`).
    let center = torus.center();
    assert!(
        (center.x()).abs() < 1e-9 && (center.y()).abs() < 1e-9,
        "torus center should be on the cylinder axis"
    );
    assert!(
        (center.z() - r_fillet).abs() < 1e-9,
        "concave torus center should sit at z = +r ({r_fillet}), got {}",
        center.z()
    );
}

/// A bare disc cap whose only boundary is the cylinder rim (no inner wires,
/// all boundary vertices within `r_c` of the axis) is an INWARD rim — the
/// fillet rounds the disc inward, with `major = r_c − r` and the torus centre
/// one radius into the material (z = +r). This is the gh #967 geometry, and
/// the direct analogue of the integration-level
/// `fillet_cylinder_base_circle_produces_torus`. (A genuine post-on-a-plate,
/// where the plate extends past the cylinder, instead keeps `major = r_c + r`;
/// that configuration is not modelled by a bare disc.)
#[test]
fn plane_cylinder_fillet_rim_emits_torus_with_smaller_major() {
    use brepkit_math::curves::Circle3D;
    use brepkit_math::surfaces::CylindricalSurface;
    use brepkit_topology::edge::{Edge, EdgeCurve};
    use brepkit_topology::face::Face;
    use brepkit_topology::vertex::Vertex;
    use brepkit_topology::wire::{OrientedEdge, Wire};

    let mut topo = Topology::new();
    let r_c: f64 = 2.0;
    let r_fillet = 0.3;

    let v = topo.add_vertex(Vertex::new(Point3::new(r_c, 0.0, 0.0), 1e-7));
    let circle = Circle3D::new(Point3::new(0.0, 0.0, 0.0), Vec3::new(0.0, 0.0, 1.0), r_c).unwrap();
    let eid = topo.add_edge(Edge::new(v, v, EdgeCurve::Circle(circle)));
    let spine = Spine::from_single_edge(&topo, eid).unwrap();

    let w1 = topo.add_wire(Wire::new(vec![OrientedEdge::new(eid, true)], true).unwrap());
    let face_plate = topo.add_face(Face::new(
        w1,
        vec![],
        FaceSurface::Plane {
            normal: Vec3::new(0.0, 0.0, -1.0),
            d: 0.0,
        },
    ));

    // NOT reversed — the disc cap's rim borders a solid (post/cylinder body).
    let cyl_surface =
        CylindricalSurface::new(Point3::new(0.0, 0.0, 0.0), Vec3::new(0.0, 0.0, 1.0), r_c).unwrap();
    let w2 = topo.add_wire(Wire::new(vec![OrientedEdge::new(eid, false)], true).unwrap());
    let face_cyl = topo.add_face(Face::new(
        w2,
        vec![],
        FaceSurface::Cylinder(cyl_surface.clone()),
    ));

    // For the cylinder primitive bottom rim the dispatcher gives
    // n_p_inward = +z (after flipping the bottom-cap outward = -z).
    let n_p_inward = Vec3::new(0.0, 0.0, 1.0);

    let result = plane_cylinder_fillet(
        n_p_inward,
        0.0,
        &cyl_surface,
        &spine,
        &topo,
        r_fillet,
        face_plate,
        face_cyl,
    )
    .unwrap()
    .expect("rim plane-cylinder fillet should produce a stripe");

    let torus = match result.stripe.surface {
        FaceSurface::Torus(t) => t,
        other => panic!("expected Torus, got {}", other.type_tag()),
    };

    assert!(
        (torus.major_radius() - (r_c - r_fillet)).abs() < 1e-9,
        "torus major should be r_c − r_fillet = {} for an inward rim, got {}",
        r_c - r_fillet,
        torus.major_radius()
    );
    // Rim case: torus centre at z = +r (one radius into the material).
    let center = torus.center();
    assert!(
        (center.z() - r_fillet).abs() < 1e-9,
        "rim torus centre should sit at z = +r ({r_fillet}), got {}",
        center.z()
    );
}

/// Concave plane-cylinder fillet rejects radii ≥ r_c/2 — past that
/// threshold `major = r_c - r ≤ minor = r` and the construction
/// becomes a self-intersecting spindle torus, which is invalid as a
/// fillet surface. Convex must still accept radii up to `r_c`.
#[test]
fn plane_cylinder_fillet_concave_rejects_spindle_radius() {
    use brepkit_math::curves::Circle3D;
    use brepkit_math::surfaces::CylindricalSurface;
    use brepkit_topology::edge::{Edge, EdgeCurve};
    use brepkit_topology::face::Face;
    use brepkit_topology::vertex::Vertex;
    use brepkit_topology::wire::{OrientedEdge, Wire};

    let r_c: f64 = 2.0;

    let setup = |topo: &mut Topology, reversed: bool| {
        let v = topo.add_vertex(Vertex::new(Point3::new(r_c, 0.0, 0.0), 1e-7));
        let circle =
            Circle3D::new(Point3::new(0.0, 0.0, 0.0), Vec3::new(0.0, 0.0, 1.0), r_c).unwrap();
        let eid = topo.add_edge(Edge::new(v, v, EdgeCurve::Circle(circle)));
        let spine = Spine::from_single_edge(topo, eid).unwrap();
        let w1 = topo.add_wire(Wire::new(vec![OrientedEdge::new(eid, true)], true).unwrap());
        let face_plate = topo.add_face(Face::new(
            w1,
            vec![],
            FaceSurface::Plane {
                normal: Vec3::new(0.0, 0.0, -1.0),
                d: 0.0,
            },
        ));
        let cyl =
            CylindricalSurface::new(Point3::new(0.0, 0.0, 0.0), Vec3::new(0.0, 0.0, 1.0), r_c)
                .unwrap();
        let w2 = topo.add_wire(Wire::new(vec![OrientedEdge::new(eid, false)], true).unwrap());
        let face_cyl = if reversed {
            topo.add_face(Face::new_reversed(
                w2,
                vec![],
                FaceSurface::Cylinder(cyl.clone()),
            ))
        } else {
            topo.add_face(Face::new(w2, vec![], FaceSurface::Cylinder(cyl.clone())))
        };
        (spine, cyl, face_plate, face_cyl)
    };

    // Concave: r > r_c/2 ⇒ would form spindle torus ⇒ reject.
    let mut topo_concave = Topology::new();
    let (spine_concave, cyl_concave, fp_concave, fc_concave) = setup(&mut topo_concave, true);
    let n_p_inward = Vec3::new(0.0, 0.0, -1.0);
    let result = plane_cylinder_fillet(
        n_p_inward,
        0.0,
        &cyl_concave,
        &spine_concave,
        &topo_concave,
        // Just above r_c/2 = 1.0 — would produce major = 0.9 < minor = 1.1.
        1.1,
        fp_concave,
        fc_concave,
    )
    .unwrap();
    assert!(
        result.is_none(),
        "concave fillet must reject r > r_c/2 (spindle-torus regime)"
    );

    // Concave at r exactly r_c/2 is also a degenerate equality (major
    // = minor); rejected.
    let result_eq = plane_cylinder_fillet(
        n_p_inward,
        0.0,
        &cyl_concave,
        &spine_concave,
        &topo_concave,
        1.0,
        fp_concave,
        fc_concave,
    )
    .unwrap();
    assert!(
        result_eq.is_none(),
        "concave fillet must reject r = r_c/2 (degenerate major = minor)"
    );

    // The non-reversed cylinder face here borders a bare disc cap (the rim
    // circle is the plate's only boundary), so it is an INWARD rim, not a
    // post-on-a-plate. The rim shares the concave `major = r_c − r` formula, so
    // it must reject r ≥ r_c/2 (spindle) too…
    let mut topo_rim = Topology::new();
    let (spine_rim, cyl_rim, fp_rim, fc_rim) = setup(&mut topo_rim, false);
    let n_p_inward_rim = Vec3::new(0.0, 0.0, 1.0);
    let result_rim_spindle = plane_cylinder_fillet(
        n_p_inward_rim,
        0.0,
        &cyl_rim,
        &spine_rim,
        &topo_rim,
        1.5,
        fp_rim,
        fc_rim,
    )
    .unwrap();
    assert!(
        result_rim_spindle.is_none(),
        "rim fillet must reject r > r_c/2 (spindle-torus regime)"
    );

    // …and accept a small radius (major = r_c − r > minor = r).
    let result_rim_ok = plane_cylinder_fillet(
        n_p_inward_rim,
        0.0,
        &cyl_rim,
        &spine_rim,
        &topo_rim,
        0.5,
        fp_rim,
        fc_rim,
    )
    .unwrap();
    assert!(
        result_rim_ok.is_some(),
        "rim fillet should accept r < r_c/2"
    );
}

/// Concave plane-cone fillet ("tapered hole through plate") emits a
/// torus with `major = r_p − r·cot(α/2)` (the convex case is
/// `r_p + r·cot(α/2)`). Direct helper test mirroring the
/// plane-cylinder concave coverage.
#[test]
fn plane_cone_fillet_concave_emits_torus_with_smaller_major() {
    use brepkit_math::curves::Circle3D;
    use brepkit_math::surfaces::ConicalSurface;
    use brepkit_topology::edge::{Edge, EdgeCurve};
    use brepkit_topology::face::Face;
    use brepkit_topology::vertex::Vertex;
    use brepkit_topology::wire::{OrientedEdge, Wire};

    let mut topo = Topology::new();
    // Apex 6 units above the plate, half-angle α = atan2(6, 3) so the
    // cone-plate intersection (the spine) lands at radius r_p = 3.
    let alpha = 6.0_f64.atan2(3.0);
    let r_p = 3.0;
    let r_fillet = 0.3;

    let v = topo.add_vertex(Vertex::new(Point3::new(r_p, 0.0, 0.0), 1e-7));
    let circle = Circle3D::new(Point3::new(0.0, 0.0, 0.0), Vec3::new(0.0, 0.0, 1.0), r_p).unwrap();
    let eid = topo.add_edge(Edge::new(v, v, EdgeCurve::Circle(circle)));
    let spine = Spine::from_single_edge(&topo, eid).unwrap();

    // Plate face: outward = +z (top of plate, plate material below).
    let w1 = topo.add_wire(Wire::new(vec![OrientedEdge::new(eid, true)], true).unwrap());
    let face_plate = topo.add_face(Face::new(
        w1,
        vec![],
        FaceSurface::Plane {
            normal: Vec3::new(0.0, 0.0, 1.0),
            d: 0.0,
        },
    ));

    // Cone face REVERSED — wall of a tapered hole, with topological
    // outward pointing into the empty hole.
    let cone_surface =
        ConicalSurface::new(Point3::new(0.0, 0.0, 6.0), Vec3::new(0.0, 0.0, -1.0), alpha).unwrap();
    let w2 = topo.add_wire(Wire::new(vec![OrientedEdge::new(eid, false)], true).unwrap());
    let face_cone = topo.add_face(Face::new_reversed(
        w2,
        vec![],
        FaceSurface::Cone(cone_surface.clone()),
    ));

    // Plate top has outward = +z; after `orient_plane_surface` the
    // helper sees inward = -z.
    let n_p_inward = Vec3::new(0.0, 0.0, -1.0);

    let result = plane_cone_fillet(
        n_p_inward,
        0.0,
        &cone_surface,
        &spine,
        &topo,
        r_fillet,
        face_plate,
        face_cone,
    )
    .unwrap()
    .expect("concave plane-cone fillet should produce a stripe");

    let torus = match result.stripe.surface {
        FaceSurface::Torus(t) => t,
        other => panic!("expected Torus, got {}", other.type_tag()),
    };

    // Expected: major = r_p - r·cot(α/2). For α ≈ 1.107 (atan2(6,3)),
    // cot(α/2) ≈ 1.618. So major ≈ 3 - 0.485 = 2.515.
    let expected_major = r_p - r_fillet * (alpha * 0.5).tan().recip();
    assert!(
        (torus.minor_radius() - r_fillet).abs() < 1e-9,
        "torus minor should equal fillet radius {r_fillet}, got {}",
        torus.minor_radius()
    );
    assert!(
        (torus.major_radius() - expected_major).abs() < 1e-9,
        "concave torus major should be r_p − r·cot(α/2) ≈ {expected_major:.6}, got {}",
        torus.major_radius()
    );

    // Center sits at +r ABOVE the plate (in the empty wedge direction
    // = -n_p_inward = +z), distinguishing concave from convex
    // (which has center at -r below the plate).
    let center = torus.center();
    assert!(
        (center.z() - r_fillet).abs() < 1e-9,
        "concave torus center should sit at z = +r ({r_fillet}), got {}",
        center.z()
    );
}

/// Concave plane-cone fillet rejects radii that would produce a
/// spindle torus (i.e. when `r·(cot(α/2) + 1) ≥ r_p` so
/// `major ≤ minor`). At the cylinder limit α = π/2 this collapses to
/// `r ≥ r_p/2`, matching the plane-cylinder bound.
#[test]
fn plane_cone_fillet_concave_rejects_spindle_radius() {
    use brepkit_math::curves::Circle3D;
    use brepkit_math::surfaces::ConicalSurface;
    use brepkit_topology::edge::{Edge, EdgeCurve};
    use brepkit_topology::face::Face;
    use brepkit_topology::vertex::Vertex;
    use brepkit_topology::wire::{OrientedEdge, Wire};

    let mut topo = Topology::new();
    // Same cone setup as the previous test.
    let alpha = 6.0_f64.atan2(3.0);
    let r_p = 3.0;
    let cot_half = (alpha * 0.5).tan().recip();
    // Max valid concave radius: r_p / (cot(α/2) + 1).
    let r_max = r_p / (cot_half + 1.0);

    let v = topo.add_vertex(Vertex::new(Point3::new(r_p, 0.0, 0.0), 1e-7));
    let circle = Circle3D::new(Point3::new(0.0, 0.0, 0.0), Vec3::new(0.0, 0.0, 1.0), r_p).unwrap();
    let eid = topo.add_edge(Edge::new(v, v, EdgeCurve::Circle(circle)));
    let spine = Spine::from_single_edge(&topo, eid).unwrap();
    let w1 = topo.add_wire(Wire::new(vec![OrientedEdge::new(eid, true)], true).unwrap());
    let face_plate = topo.add_face(Face::new(
        w1,
        vec![],
        FaceSurface::Plane {
            normal: Vec3::new(0.0, 0.0, 1.0),
            d: 0.0,
        },
    ));
    let cone_surface =
        ConicalSurface::new(Point3::new(0.0, 0.0, 6.0), Vec3::new(0.0, 0.0, -1.0), alpha).unwrap();
    let w2 = topo.add_wire(Wire::new(vec![OrientedEdge::new(eid, false)], true).unwrap());
    let face_cone = topo.add_face(Face::new_reversed(
        w2,
        vec![],
        FaceSurface::Cone(cone_surface.clone()),
    ));
    let n_p_inward = Vec3::new(0.0, 0.0, -1.0);

    // Just above r_max → spindle regime → reject.
    let result_spindle = plane_cone_fillet(
        n_p_inward,
        0.0,
        &cone_surface,
        &spine,
        &topo,
        r_max * 1.01,
        face_plate,
        face_cone,
    )
    .unwrap();
    assert!(
        result_spindle.is_none(),
        "concave fillet must reject r > r_p / (cot(α/2)+1) (spindle-torus regime)"
    );

    // Exactly at r_max → horn-torus boundary (major = minor), where the
    // tube touches the axis at a degenerate point — also rejected.
    let result_horn = plane_cone_fillet(
        n_p_inward,
        0.0,
        &cone_surface,
        &spine,
        &topo,
        r_max,
        face_plate,
        face_cone,
    )
    .unwrap();
    assert!(
        result_horn.is_none(),
        "concave fillet must reject r = r_p / (cot(α/2)+1) (horn-torus boundary)"
    );

    // Below r_max — should succeed.
    let result_ok = plane_cone_fillet(
        n_p_inward,
        0.0,
        &cone_surface,
        &spine,
        &topo,
        r_max * 0.5,
        face_plate,
        face_cone,
    )
    .unwrap();
    assert!(
        result_ok.is_some(),
        "concave fillet should accept r below the spindle threshold"
    );
}

/// Concave plane-cylinder chamfer (chamfer at the top rim of a hole
/// through a plate). The chamfer face is a cone whose plate-side
/// contact lands at radial `r_c + d1` (outside the spine, in the
/// surrounding plate material), and whose cylinder-side contact lands
/// at axial `−d2` along the hole wall going into the plate.
#[test]
fn plane_cylinder_chamfer_concave_emits_cone() {
    use brepkit_math::curves::Circle3D;
    use brepkit_math::surfaces::CylindricalSurface;
    use brepkit_topology::edge::{Edge, EdgeCurve};
    use brepkit_topology::face::Face;
    use brepkit_topology::vertex::Vertex;
    use brepkit_topology::wire::{OrientedEdge, Wire};

    let mut topo = Topology::new();
    let r_c: f64 = 2.0;
    let d = 0.4;

    let v = topo.add_vertex(Vertex::new(Point3::new(r_c, 0.0, 0.0), 1e-7));
    let circle = Circle3D::new(Point3::new(0.0, 0.0, 0.0), Vec3::new(0.0, 0.0, 1.0), r_c).unwrap();
    let eid = topo.add_edge(Edge::new(v, v, EdgeCurve::Circle(circle)));
    let spine = Spine::from_single_edge(&topo, eid).unwrap();

    // Plate top face: outward = +z (raw — chamfer dispatcher passes
    // unflipped; it's the plate top of a plate with a hole).
    let w1 = topo.add_wire(Wire::new(vec![OrientedEdge::new(eid, true)], true).unwrap());
    let face_plate = topo.add_face(Face::new(
        w1,
        vec![],
        FaceSurface::Plane {
            normal: Vec3::new(0.0, 0.0, 1.0),
            d: 0.0,
        },
    ));

    // Cylinder face REVERSED — the hole wall, with topological outward
    // pointing into the empty hole.
    let cyl_surface =
        CylindricalSurface::new(Point3::new(0.0, 0.0, 0.0), Vec3::new(0.0, 0.0, 1.0), r_c).unwrap();
    let w2 = topo.add_wire(Wire::new(vec![OrientedEdge::new(eid, false)], true).unwrap());
    let face_cyl = topo.add_face(Face::new_reversed(
        w2,
        vec![],
        FaceSurface::Cylinder(cyl_surface.clone()),
    ));

    let n_p_inward = Vec3::new(0.0, 0.0, 1.0);
    let result = plane_cylinder_chamfer(
        n_p_inward,
        0.0,
        &cyl_surface,
        &spine,
        &topo,
        d,
        d,
        face_plate,
        face_cyl,
    )
    .unwrap()
    .expect("concave plane-cylinder chamfer should produce a stripe");

    let cone_surf = match result.stripe.surface {
        FaceSurface::Cone(c) => c,
        other => panic!("expected Cone, got {}", other.type_tag()),
    };

    // Half-angle for symmetric chamfer is π/4 in either case.
    assert!(
        (cone_surf.half_angle() - std::f64::consts::FRAC_PI_4).abs() < 1e-12,
        "chamfer cone half-angle should be π/4 for symmetric d, got {}",
        cone_surf.half_angle()
    );

    // Plate-side contact at `r_c + d` (in the surrounding plate
    // material at z=0), cylinder-side contact at `-d` (going down
    // into the hole wall).
    // Frame3::from_normal(+z) gives x_axis = (0, 1, 0); chamfer cone's
    // axis is -z (= -apex_dir = -(s·n_p_inward) = -(-z) = +z, but
    // wait — let me re-derive: s = -1 for concave; apex_dir =
    // s · n_p_inward = -n_p_inward = -z; cone_axis = -apex_dir = +z).
    // So cone axis is +z, frame x_axis = (0, 1, 0).
    let want_plate = Point3::new(0.0, r_c + d, 0.0);
    let want_cyl = Point3::new(0.0, r_c, -d);
    let mut closest_plate = f64::INFINITY;
    let mut closest_cyl = f64::INFINITY;
    for i in 0..1440 {
        let v = (f64::from(i) / 1440.0) * std::f64::consts::TAU;
        // Try u=0 across a range of v.
        let p = ParametricSurface::evaluate(&cone_surf, 0.0, v);
        closest_plate = closest_plate.min((p - want_plate).length());
        closest_cyl = closest_cyl.min((p - want_cyl).length());
    }
    // The cone surface at SOME (u=0, v) should pass close to both
    // contact points; sampling along v with sufficient density.
    assert!(
        closest_plate < 1e-3,
        "concave chamfer cone should pass near plate contact at {want_plate:?}; closest = {closest_plate:.6}"
    );
    assert!(
        closest_cyl < 1e-3,
        "concave chamfer cone should pass near cyl contact at {want_cyl:?}; closest = {closest_cyl:.6}"
    );
}

/// Convex plane-sphere fillet: a sphere intersecting a plate from above
/// (post-on-slab configuration). The fillet rounds the convex spine
/// circle and produces a Toroidal blend surface.
///
/// Scenario:
///   - Plate top face at z=0 with raw outward = +z.
///   - After orient_plane_surface (which the dispatcher applies for
///     fillet), n_p_inward = -z (into plate material below).
///   - Sphere center at (0,0,h)=( 0,0,1) above plate, radius R=2,
///     so spine circle is z=0 with r_p = √(R²−h²) = √3.
///   - Fillet radius r=0.3.
///
/// Predicted analytics (using h_signed = -h = -1):
///   - R_t² = r_p² + 2r(R − h_signed) = 3 + 2·0.3·(2−(−1)) = 4.8
///   - Torus center: p_axis − n_p_inward · r = (0,0,0) − (−z)·0.3 = (0,0,+0.3)
///   - Plate contact at radial R_t, z=0
///   - Sphere contact at radial R·R_t/(R+r) ≈ 1.905,
///     z = +r(R − h_signed)/(R + r) ≈ +0.391 (above plate)
#[test]
fn plane_sphere_fillet_convex_emits_torus() {
    use brepkit_math::curves::Circle3D;
    use brepkit_math::surfaces::SphericalSurface;
    use brepkit_topology::edge::{Edge, EdgeCurve};
    use brepkit_topology::face::Face;
    use brepkit_topology::vertex::Vertex;
    use brepkit_topology::wire::{OrientedEdge, Wire};

    let mut topo = Topology::new();
    let big_r: f64 = 2.0;
    let h_real: f64 = 1.0;
    let r_fillet: f64 = 0.3;
    let r_p_sq = big_r * big_r - h_real * h_real;
    let r_p = r_p_sq.sqrt();

    let v = topo.add_vertex(Vertex::new(Point3::new(r_p, 0.0, 0.0), 1e-7));
    let circle = Circle3D::new(Point3::new(0.0, 0.0, 0.0), Vec3::new(0.0, 0.0, 1.0), r_p).unwrap();
    let eid = topo.add_edge(Edge::new(v, v, EdgeCurve::Circle(circle)));
    let spine = Spine::from_single_edge(&topo, eid).unwrap();

    // Plate top face at z=0 with raw outward = +z (away from plate
    // material at z<0). After orient_plane_surface, the dispatcher
    // would pass n_p_inward = -z; we mirror that here.
    let w1 = topo.add_wire(Wire::new(vec![OrientedEdge::new(eid, true)], true).unwrap());
    let face_plate = topo.add_face(Face::new(
        w1,
        vec![],
        FaceSurface::Plane {
            normal: Vec3::new(0.0, 0.0, 1.0),
            d: 0.0,
        },
    ));

    // Sphere centered above plate, NOT reversed (convex post).
    let sphere = SphericalSurface::new(Point3::new(0.0, 0.0, h_real), big_r).unwrap();
    let w2 = topo.add_wire(Wire::new(vec![OrientedEdge::new(eid, false)], true).unwrap());
    let face_sphere = topo.add_face(Face::new(w2, vec![], FaceSurface::Sphere(sphere.clone())));

    let n_p_inward = Vec3::new(0.0, 0.0, -1.0);
    let result = plane_sphere_fillet(
        n_p_inward,
        0.0,
        &sphere,
        &spine,
        &topo,
        r_fillet,
        face_plate,
        face_sphere,
    )
    .unwrap()
    .expect("convex plane-sphere fillet should produce a stripe");

    let torus = match result.stripe.surface {
        FaceSurface::Torus(t) => t,
        other => panic!("expected Torus, got {}", other.type_tag()),
    };

    let expected_major_sq = r_p_sq + 2.0 * r_fillet * (big_r + h_real);
    let expected_major = expected_major_sq.sqrt();
    assert!(
        (torus.major_radius() - expected_major).abs() < 1e-12,
        "torus major should be √(r_p² + 2r(R+h)) = {expected_major}, got {}",
        torus.major_radius()
    );
    assert!(
        (torus.minor_radius() - r_fillet).abs() < 1e-12,
        "torus minor should equal fillet radius {r_fillet}, got {}",
        torus.minor_radius()
    );

    // Torus center at z = +r (above plate, in empty wedge between
    // plate top and upper hemisphere).
    let center = torus.center();
    assert!(
        (center.x()).abs() < 1e-12 && (center.y()).abs() < 1e-12,
        "torus center should be on z-axis, got {center:?}"
    );
    assert!(
        (center.z() - r_fillet).abs() < 1e-12,
        "torus center z should be +r_fillet = {r_fillet}, got {}",
        center.z()
    );

    // Both contacts must lie ON the torus surface — verify via
    // project_point (frame-orientation-agnostic).
    let want_plate = Point3::new(expected_major, 0.0, 0.0);
    let r_plus_r = big_r + r_fillet;
    let want_sphere = Point3::new(
        expected_major * big_r / r_plus_r,
        0.0,
        r_fillet * (big_r + h_real) / r_plus_r,
    );
    let (u_p, v_p) = ParametricSurface::project_point(&torus, want_plate);
    let on_torus_plate = ParametricSurface::evaluate(&torus, u_p, v_p);
    let (u_s, v_s) = ParametricSurface::project_point(&torus, want_sphere);
    let on_torus_sphere = ParametricSurface::evaluate(&torus, u_s, v_s);
    assert!(
        (on_torus_plate - want_plate).length() < 1e-9,
        "plate contact must lie on torus: project→eval gave {on_torus_plate:?}, want {want_plate:?}"
    );
    assert!(
        (on_torus_sphere - want_sphere).length() < 1e-9,
        "sphere contact must lie on torus: project→eval gave {on_torus_sphere:?}, want {want_sphere:?}"
    );

    // Sanity-check: sphere contact point should also lie on the
    // sphere surface itself (distance R from center).
    let sphere_dist = (want_sphere - Point3::new(0.0, 0.0, h_real)).length();
    assert!(
        (sphere_dist - big_r).abs() < 1e-9,
        "sphere contact must lie on sphere: distance from center = {sphere_dist}, want {big_r}"
    );
}

/// Concave plane-sphere fillet: a spherical pocket carved out of a plate
/// top — fillet rounds the rim where plate top meets pocket wall. Sphere
/// face is REVERSED (its topological outward points INTO the pocket air,
/// away from plate material).
///
/// Geometry differs from the convex post-on-slab case in two ways:
/// the rolling ball lives INSIDE the pocket (axially on the +n_p_inward
/// side) and is INTERNALLY tangent to the sphere (`R − r` instead of
/// `R + r`). The unified `signed_offset = −1` factor flips both.
///
/// For sphere center at (0,0,−h)=(0,0,−1), R=2, plate top at z=0 with
/// raw outward +z, n_p_inward = −z, h_signed = +1, r=0.3:
///   - R_t² = r_p² − 2r(R − h) = 3 − 0.6 = 2.4
///   - Torus center at z = −r (below plate, in pocket)
///   - Plate contact at radial R_t < r_p (closer to axis than the spine)
///   - Sphere contact at z = -0.176 (below plate, on the LOWER
///     hemisphere where the pocket face actually exists — confirms
///     internal tangency lands on the right portion of sphere)
#[test]
fn plane_sphere_fillet_concave_emits_torus_with_smaller_major() {
    use brepkit_math::curves::Circle3D;
    use brepkit_math::surfaces::SphericalSurface;
    use brepkit_topology::edge::{Edge, EdgeCurve};
    use brepkit_topology::face::Face;
    use brepkit_topology::vertex::Vertex;
    use brepkit_topology::wire::{OrientedEdge, Wire};

    let mut topo = Topology::new();
    let big_r: f64 = 2.0;
    let h_real: f64 = 1.0;
    let r_fillet: f64 = 0.3;
    let r_p_sq = big_r * big_r - h_real * h_real;
    let r_p = r_p_sq.sqrt();

    let v = topo.add_vertex(Vertex::new(Point3::new(r_p, 0.0, 0.0), 1e-7));
    let circle = Circle3D::new(Point3::new(0.0, 0.0, 0.0), Vec3::new(0.0, 0.0, 1.0), r_p).unwrap();
    let eid = topo.add_edge(Edge::new(v, v, EdgeCurve::Circle(circle)));
    let spine = Spine::from_single_edge(&topo, eid).unwrap();

    let w1 = topo.add_wire(Wire::new(vec![OrientedEdge::new(eid, true)], true).unwrap());
    let face_plate = topo.add_face(Face::new(
        w1,
        vec![],
        FaceSurface::Plane {
            normal: Vec3::new(0.0, 0.0, 1.0),
            d: 0.0,
        },
    ));

    // Sphere centered BELOW plate (pocket); face REVERSED.
    let sphere = SphericalSurface::new(Point3::new(0.0, 0.0, -h_real), big_r).unwrap();
    let w2 = topo.add_wire(Wire::new(vec![OrientedEdge::new(eid, false)], true).unwrap());
    let face_sphere = topo.add_face(Face::new_reversed(
        w2,
        vec![],
        FaceSurface::Sphere(sphere.clone()),
    ));

    let n_p_inward = Vec3::new(0.0, 0.0, -1.0);
    let result = plane_sphere_fillet(
        n_p_inward,
        0.0,
        &sphere,
        &spine,
        &topo,
        r_fillet,
        face_plate,
        face_sphere,
    )
    .unwrap()
    .expect("concave plane-sphere fillet should produce a stripe");

    let torus = match result.stripe.surface {
        FaceSurface::Torus(t) => t,
        other => panic!("expected Torus, got {}", other.type_tag()),
    };

    let expected_major_sq = r_p_sq - 2.0 * r_fillet * (big_r - h_real);
    let expected_major = expected_major_sq.sqrt();
    assert!(
        (torus.major_radius() - expected_major).abs() < 1e-12,
        "concave torus major should be √(r_p² − 2r(R−h)) = {expected_major}, got {}",
        torus.major_radius()
    );
    assert!(
        torus.major_radius() < r_p,
        "concave torus major must be smaller than spine radius (plate contact moves INWARD), got {} vs r_p={r_p}",
        torus.major_radius()
    );
    assert!(
        (torus.minor_radius() - r_fillet).abs() < 1e-12,
        "torus minor should equal fillet radius {r_fillet}, got {}",
        torus.minor_radius()
    );

    // Torus center at z = -r (below plate, inside pocket air).
    let center = torus.center();
    assert!(
        (center.z() - (-r_fillet)).abs() < 1e-12,
        "concave torus center z should be −r_fillet = {}, got {}",
        -r_fillet,
        center.z()
    );

    // Sphere contact must land on the LOWER hemisphere (z<0) — the
    // actual pocket face. If we'd applied the convex external-tangency
    // formula by mistake, contact would end up on the upper hemisphere
    // (z>0) where there's no face.
    let denom = big_r - r_fillet;
    let want_sphere = Point3::new(
        expected_major * big_r / denom,
        0.0,
        -r_fillet * (big_r - h_real) / denom,
    );
    assert!(
        want_sphere.z() < 0.0,
        "concave sphere contact must be on lower hemisphere (z<0), got z={}",
        want_sphere.z()
    );
    let sphere_dist = (want_sphere - Point3::new(0.0, 0.0, -h_real)).length();
    assert!(
        (sphere_dist - big_r).abs() < 1e-9,
        "sphere contact must lie on sphere: distance from center = {sphere_dist}, want {big_r}"
    );

    // Verify both contacts land on the torus surface.
    let want_plate = Point3::new(expected_major, 0.0, 0.0);
    let (u_p, v_p) = ParametricSurface::project_point(&torus, want_plate);
    let on_torus_plate = ParametricSurface::evaluate(&torus, u_p, v_p);
    let (u_s, v_s) = ParametricSurface::project_point(&torus, want_sphere);
    let on_torus_sphere = ParametricSurface::evaluate(&torus, u_s, v_s);
    assert!(
        (on_torus_plate - want_plate).length() < 1e-9,
        "plate contact must lie on torus: project→eval gave {on_torus_plate:?}, want {want_plate:?}"
    );
    assert!(
        (on_torus_sphere - want_sphere).length() < 1e-9,
        "sphere contact must lie on torus: project→eval gave {on_torus_sphere:?}, want {want_sphere:?}"
    );
}

/// Concave plane-sphere fillet rejects radii past the spindle bound,
/// where `major² = r_p² − 2r(R−h)` would shrink below `r²`. Convex
/// must still accept those same radii (its `+2r(R−h)` term grows).
#[test]
fn plane_sphere_fillet_concave_rejects_spindle_radius() {
    use brepkit_math::curves::Circle3D;
    use brepkit_math::surfaces::SphericalSurface;
    use brepkit_topology::edge::{Edge, EdgeCurve};
    use brepkit_topology::face::Face;
    use brepkit_topology::vertex::Vertex;
    use brepkit_topology::wire::{OrientedEdge, Wire};

    let mut topo = Topology::new();
    let big_r: f64 = 2.0;
    let h_real: f64 = 1.0;
    let r_p_sq = big_r * big_r - h_real * h_real;
    let r_p = r_p_sq.sqrt();

    let v = topo.add_vertex(Vertex::new(Point3::new(r_p, 0.0, 0.0), 1e-7));
    let circle = Circle3D::new(Point3::new(0.0, 0.0, 0.0), Vec3::new(0.0, 0.0, 1.0), r_p).unwrap();
    let eid = topo.add_edge(Edge::new(v, v, EdgeCurve::Circle(circle)));
    let spine = Spine::from_single_edge(&topo, eid).unwrap();

    let w1 = topo.add_wire(Wire::new(vec![OrientedEdge::new(eid, true)], true).unwrap());
    let face_plate = topo.add_face(Face::new(
        w1,
        vec![],
        FaceSurface::Plane {
            normal: Vec3::new(0.0, 0.0, 1.0),
            d: 0.0,
        },
    ));

    let sphere = SphericalSurface::new(Point3::new(0.0, 0.0, -h_real), big_r).unwrap();
    let w2 = topo.add_wire(Wire::new(vec![OrientedEdge::new(eid, false)], true).unwrap());
    let face_sphere = topo.add_face(Face::new_reversed(
        w2,
        vec![],
        FaceSurface::Sphere(sphere.clone()),
    ));

    let n_p_inward = Vec3::new(0.0, 0.0, -1.0);

    // Concave spindle threshold: solving r² + 2r(R−h) > r_p² for the
    // positive root gives r > √((R−h)² + r_p²) − (R−h).
    // For R=2, r_p²=3, R−h=1: r > √(1+3)−1 = 1. So r=1.5 must reject.
    let too_big = 1.5;
    let result = plane_sphere_fillet(
        n_p_inward,
        0.0,
        &sphere,
        &spine,
        &topo,
        too_big,
        face_plate,
        face_sphere,
    )
    .unwrap();
    assert!(
        result.is_none(),
        "concave fillet at r={too_big} should reject (spindle / R_t < minor)"
    );

    // But convex at the same r is still fine: R_t² = r_p² + 2r·3 = 3 + 9 = 12, R_t ≈ 3.46 > r.
    // Build a mirror topology with face NOT reversed.
    let mut topo2 = Topology::new();
    let v2 = topo2.add_vertex(Vertex::new(Point3::new(r_p, 0.0, 0.0), 1e-7));
    let circle2 = Circle3D::new(Point3::new(0.0, 0.0, 0.0), Vec3::new(0.0, 0.0, 1.0), r_p).unwrap();
    let eid2 = topo2.add_edge(Edge::new(v2, v2, EdgeCurve::Circle(circle2)));
    let spine2 = Spine::from_single_edge(&topo2, eid2).unwrap();
    let w1b = topo2.add_wire(Wire::new(vec![OrientedEdge::new(eid2, true)], true).unwrap());
    let face_plate2 = topo2.add_face(Face::new(
        w1b,
        vec![],
        FaceSurface::Plane {
            normal: Vec3::new(0.0, 0.0, 1.0),
            d: 0.0,
        },
    ));
    let sphere2 = SphericalSurface::new(Point3::new(0.0, 0.0, h_real), big_r).unwrap();
    let w2b = topo2.add_wire(Wire::new(vec![OrientedEdge::new(eid2, false)], true).unwrap());
    let face_sphere2 = topo2.add_face(Face::new(w2b, vec![], FaceSurface::Sphere(sphere2.clone())));
    let result_convex = plane_sphere_fillet(
        n_p_inward,
        0.0,
        &sphere2,
        &spine2,
        &topo2,
        too_big,
        face_plate2,
        face_sphere2,
    )
    .unwrap();
    assert!(
        result_convex.is_some(),
        "convex fillet at the same r={too_big} should still succeed"
    );
}

/// Convex plane-sphere chamfer: a sphere intersecting a plate from
/// above (post-on-slab). The chamfer cuts the rim with a flat conical
/// slice tangent to plate at radial `r_p+d1` and to the sphere at
/// arc-length `d2` along the meridian toward the apex.
///
/// For sphere center at (0, 0, h)=(0, 0, 1), R=2, plate top at z=0
/// with raw outward +z (chamfer dispatcher uses raw, no orient), and
/// symmetric d1=d2=0.3:
///   - δ = d2/R = 0.15
///   - Sphere contact at radial r_p·cos δ + h·sin δ ≈ √3·0.989 + 1·0.149 ≈ 1.862
///     and z = h(1−cos δ) + r_p·sin δ ≈ 0.011 + √3·0.149 ≈ 0.270
///   - Plate contact at radial r_p+d1 = √3+0.3 ≈ 2.032 (z=0)
///   - Cone half-angle β where tan β = −Δz/Δr = 0.270/0.170 ≈ 1.589
///     (β ≈ 57.8°; not the small-δ Taylor limit which would give
///     atan(r_p/(R−h)) = atan(√3/1) = 60°)
///   - Cone apex at z = (r_p+d1)·tan β ≈ 3.230
#[test]
fn plane_sphere_chamfer_convex_emits_cone() {
    use brepkit_math::curves::Circle3D;
    use brepkit_math::surfaces::SphericalSurface;
    use brepkit_topology::edge::{Edge, EdgeCurve};
    use brepkit_topology::face::Face;
    use brepkit_topology::vertex::Vertex;
    use brepkit_topology::wire::{OrientedEdge, Wire};

    let mut topo = Topology::new();
    let big_r: f64 = 2.0;
    let h_real: f64 = 1.0;
    let d: f64 = 0.3;
    let r_p_sq = big_r * big_r - h_real * h_real;
    let r_p = r_p_sq.sqrt();

    let v = topo.add_vertex(Vertex::new(Point3::new(r_p, 0.0, 0.0), 1e-7));
    let circle = Circle3D::new(Point3::new(0.0, 0.0, 0.0), Vec3::new(0.0, 0.0, 1.0), r_p).unwrap();
    let eid = topo.add_edge(Edge::new(v, v, EdgeCurve::Circle(circle)));
    let spine = Spine::from_single_edge(&topo, eid).unwrap();

    let w1 = topo.add_wire(Wire::new(vec![OrientedEdge::new(eid, true)], true).unwrap());
    // Chamfer convention: dispatcher passes the surface's RAW normal
    // (no orient). Plate slab top has raw outward = +z.
    let face_plate = topo.add_face(Face::new(
        w1,
        vec![],
        FaceSurface::Plane {
            normal: Vec3::new(0.0, 0.0, 1.0),
            d: 0.0,
        },
    ));

    let sphere = SphericalSurface::new(Point3::new(0.0, 0.0, h_real), big_r).unwrap();
    let w2 = topo.add_wire(Wire::new(vec![OrientedEdge::new(eid, false)], true).unwrap());
    let face_sphere = topo.add_face(Face::new(w2, vec![], FaceSurface::Sphere(sphere.clone())));

    let n_p_inward = Vec3::new(0.0, 0.0, 1.0);
    let result = plane_sphere_chamfer(
        n_p_inward,
        0.0,
        &sphere,
        &spine,
        &topo,
        d,
        d,
        face_plate,
        face_sphere,
    )
    .unwrap()
    .expect("convex plane-sphere chamfer should produce a stripe");

    let chamfer_cone = match result.stripe.surface {
        FaceSurface::Cone(c) => c,
        other => panic!("expected Cone, got {}", other.type_tag()),
    };

    // Predicted contacts.
    let delta = d / big_r;
    let (sin_d, cos_d) = delta.sin_cos();
    let sphere_radial_pred = r_p * cos_d + h_real * sin_d;
    let sphere_axial_pred = h_real * (1.0 - cos_d) + r_p * sin_d;

    // Predicted cone half-angle: tan β = −Δz/Δr.
    let delta_r = sphere_radial_pred - (r_p + d);
    let delta_z = sphere_axial_pred;
    assert!(delta_r < 0.0, "Δr should be negative (cone narrows up)");
    let expected_beta = (-delta_z / delta_r).atan();
    assert!(
        (chamfer_cone.half_angle() - expected_beta).abs() < 1e-12,
        "chamfer cone half-angle should be atan(−Δz/Δr) = {expected_beta}, got {}",
        chamfer_cone.half_angle()
    );

    // Apex position: on +z axis at z = (r_p+d) · tan β.
    let expected_apex_z = (r_p + d) * expected_beta.tan();
    let apex = chamfer_cone.apex();
    assert!(
        apex.x().abs() < 1e-12 && apex.y().abs() < 1e-12,
        "apex should be on z-axis, got {apex:?}"
    );
    assert!(
        (apex.z() - expected_apex_z).abs() < 1e-9,
        "apex z = {}, expected {expected_apex_z}",
        apex.z()
    );

    // Cone axis points down (-z) — generator opens away from apex
    // toward the chamfer line.
    let axis = chamfer_cone.axis();
    assert!(
        axis.dot(Vec3::new(0.0, 0.0, 1.0)) < -1.0 + 1e-12,
        "chamfer cone axis should be -z, got {axis:?}"
    );

    // Both contact points must lie on the chamfer cone surface.
    let want_plate = Point3::new(r_p + d, 0.0, 0.0);
    let want_sphere = Point3::new(sphere_radial_pred, 0.0, sphere_axial_pred);
    let (u_p, v_p) = ParametricSurface::project_point(&chamfer_cone, want_plate);
    let on_cone_plate = ParametricSurface::evaluate(&chamfer_cone, u_p, v_p);
    let (u_s, v_s) = ParametricSurface::project_point(&chamfer_cone, want_sphere);
    let on_cone_sphere = ParametricSurface::evaluate(&chamfer_cone, u_s, v_s);
    assert!(
        (on_cone_plate - want_plate).length() < 1e-9,
        "plate contact must lie on chamfer cone: project→eval gave {on_cone_plate:?}, want {want_plate:?}"
    );
    assert!(
        (on_cone_sphere - want_sphere).length() < 1e-9,
        "sphere contact must lie on chamfer cone: project→eval gave {on_cone_sphere:?}, want {want_sphere:?}"
    );

    // Sanity: sphere contact lies on the actual sphere (distance R).
    let sphere_dist = (want_sphere - Point3::new(0.0, 0.0, h_real)).length();
    assert!(
        (sphere_dist - big_r).abs() < 1e-9,
        "sphere contact must lie on sphere: distance = {sphere_dist}, want {big_r}"
    );
}

/// Concave plane-sphere chamfer: a spherical pocket carved out of a
/// plate top — chamfer rounds the rim where plate meets pocket wall.
/// Sphere face is REVERSED. The chamfer surface is a cone with apex
/// BELOW the plate (in pocket air, axis pointing upward through the
/// chamfer back to the plate level).
///
/// For sphere at (0, 0, −h)=(0, 0, −1) (below plate), R=2, plate top
/// raw outward +z, n_p_inward=+z (chamfer convention), face reversed,
/// d=0.3:
///   - δ = d/R = 0.15
///   - sphere_radial = r_p·cos δ + (−1)(−1)·sin δ ≈ 1.862 (same as convex)
///   - sphere_axial  = (−1)(1−cos δ) + (−1)·r_p·sin δ ≈ −0.270 (BELOW plate)
///   - Δr = −0.170, Δz = −0.270
///   - z_apex = −(r_p+d)·Δz/Δr ≈ −3.227 (apex BELOW plate)
///   - chamfer axis = +z (opens upward toward contacts)
///   - cone β = atan(|z_apex|/(r_p+d)) ≈ 57.8° (same magnitude as convex)
#[test]
fn plane_sphere_chamfer_concave_emits_cone() {
    use brepkit_math::curves::Circle3D;
    use brepkit_math::surfaces::SphericalSurface;
    use brepkit_topology::edge::{Edge, EdgeCurve};
    use brepkit_topology::face::Face;
    use brepkit_topology::vertex::Vertex;
    use brepkit_topology::wire::{OrientedEdge, Wire};

    let mut topo = Topology::new();
    let big_r: f64 = 2.0;
    let h_real: f64 = 1.0;
    let d: f64 = 0.3;
    let r_p_sq = big_r * big_r - h_real * h_real;
    let r_p = r_p_sq.sqrt();

    let v = topo.add_vertex(Vertex::new(Point3::new(r_p, 0.0, 0.0), 1e-7));
    let circle = Circle3D::new(Point3::new(0.0, 0.0, 0.0), Vec3::new(0.0, 0.0, 1.0), r_p).unwrap();
    let eid = topo.add_edge(Edge::new(v, v, EdgeCurve::Circle(circle)));
    let spine = Spine::from_single_edge(&topo, eid).unwrap();

    let w1 = topo.add_wire(Wire::new(vec![OrientedEdge::new(eid, true)], true).unwrap());
    let face_plate = topo.add_face(Face::new(
        w1,
        vec![],
        FaceSurface::Plane {
            normal: Vec3::new(0.0, 0.0, 1.0),
            d: 0.0,
        },
    ));

    // Sphere centered BELOW plate (pocket); face REVERSED.
    let sphere = SphericalSurface::new(Point3::new(0.0, 0.0, -h_real), big_r).unwrap();
    let w2 = topo.add_wire(Wire::new(vec![OrientedEdge::new(eid, false)], true).unwrap());
    let face_sphere = topo.add_face(Face::new_reversed(
        w2,
        vec![],
        FaceSurface::Sphere(sphere.clone()),
    ));

    let n_p_inward = Vec3::new(0.0, 0.0, 1.0);
    let result = plane_sphere_chamfer(
        n_p_inward,
        0.0,
        &sphere,
        &spine,
        &topo,
        d,
        d,
        face_plate,
        face_sphere,
    )
    .unwrap()
    .expect("concave plane-sphere chamfer should produce a stripe");

    let chamfer_cone = match result.stripe.surface {
        FaceSurface::Cone(c) => c,
        other => panic!("expected Cone, got {}", other.type_tag()),
    };

    // Predicted contacts (concave: signed_offset = -1, h_signed = -1).
    let h_signed = -h_real;
    let signed_offset = -1.0_f64;
    let delta = d / big_r;
    let (sin_d, cos_d) = delta.sin_cos();
    let sphere_radial_pred = r_p * cos_d + signed_offset * h_signed * sin_d;
    let sphere_axial_pred = h_signed * (1.0 - cos_d) + signed_offset * r_p * sin_d;

    let delta_r = sphere_radial_pred - (r_p + d);
    let delta_z = sphere_axial_pred;
    let expected_z_apex = -(r_p + d) * delta_z / delta_r;
    let expected_beta = (expected_z_apex.abs() / (r_p + d)).atan();
    assert!(
        expected_z_apex < 0.0,
        "concave: z_apex should be negative (apex below plate), got {expected_z_apex}"
    );
    assert!(
        sphere_axial_pred < 0.0,
        "concave: sphere contact must be below plate (z<0), got {sphere_axial_pred}"
    );

    assert!(
        (chamfer_cone.half_angle() - expected_beta).abs() < 1e-12,
        "chamfer cone half-angle should be {expected_beta}, got {}",
        chamfer_cone.half_angle()
    );

    // Apex BELOW plate.
    let apex = chamfer_cone.apex();
    assert!(
        apex.x().abs() < 1e-12 && apex.y().abs() < 1e-12,
        "apex should be on z-axis, got {apex:?}"
    );
    assert!(
        (apex.z() - expected_z_apex).abs() < 1e-9,
        "apex z = {}, expected {expected_z_apex}",
        apex.z()
    );

    // Cone axis points UP (+z) for concave (toward the contacts above
    // the apex) — opposite to convex.
    let axis = chamfer_cone.axis();
    assert!(
        axis.dot(Vec3::new(0.0, 0.0, 1.0)) > 1.0 - 1e-12,
        "concave chamfer cone axis should be +z, got {axis:?}"
    );

    // Both contacts lie on the chamfer cone.
    let want_plate = Point3::new(r_p + d, 0.0, 0.0);
    let want_sphere = Point3::new(sphere_radial_pred, 0.0, sphere_axial_pred);
    let (u_p, v_p) = ParametricSurface::project_point(&chamfer_cone, want_plate);
    let on_cone_plate = ParametricSurface::evaluate(&chamfer_cone, u_p, v_p);
    let (u_s, v_s) = ParametricSurface::project_point(&chamfer_cone, want_sphere);
    let on_cone_sphere = ParametricSurface::evaluate(&chamfer_cone, u_s, v_s);
    assert!(
        (on_cone_plate - want_plate).length() < 1e-9,
        "plate contact must lie on chamfer cone: gave {on_cone_plate:?}, want {want_plate:?}"
    );
    assert!(
        (on_cone_sphere - want_sphere).length() < 1e-9,
        "sphere contact must lie on chamfer cone: gave {on_cone_sphere:?}, want {want_sphere:?}"
    );

    // Sphere contact must lie on the actual sphere face — i.e. on the
    // LOWER hemisphere where the pocket boundary lives. Convex would
    // have placed it on the upper hemisphere, but with signed_offset
    // = -1 the contact lands at z<0 here.
    let sphere_dist = (want_sphere - Point3::new(0.0, 0.0, -h_real)).length();
    assert!(
        (sphere_dist - big_r).abs() < 1e-9,
        "sphere contact must lie on sphere: distance = {sphere_dist}, want {big_r}"
    );
}

/// Sphere-sphere convex fillet: two intersecting spheres meeting along
/// a circular spine, rolling-ball blend traces a torus around the
/// line connecting their centers.
///
/// For sphere1 at origin (R=2), sphere2 at (3, 0, 0) (R=2.5),
/// D=3, both faces NOT reversed:
///   a₀ = (4 − 6.25 + 9) / 6 = 6.75/6 = 1.125
///   r_p² = 4 − 1.265625 = 2.734375, r_p ≈ 1.654
///   For r=0.4:
///     δ = (2 − 2.5)/3 = −1/6
///     a_ball = 1.125 + 0.4·(−1/6) ≈ 1.0583
///     R_t² = (2.4)² − (1.0583)² = 5.76 − 1.12 = 4.64
///     R_t ≈ 2.154
#[test]
fn sphere_sphere_fillet_convex_emits_torus() {
    use brepkit_math::curves::Circle3D;
    use brepkit_math::surfaces::SphericalSurface;
    use brepkit_topology::edge::{Edge, EdgeCurve};
    use brepkit_topology::face::Face;
    use brepkit_topology::vertex::Vertex;
    use brepkit_topology::wire::{OrientedEdge, Wire};

    let mut topo = Topology::new();
    let big_r1: f64 = 2.0;
    let big_r2: f64 = 2.5;
    let big_d: f64 = 3.0;
    let r_fillet: f64 = 0.4;

    // Place spheres along +z (brepkit's `SphericalSurface::new` uses
    // Frame3::from_normal with default z-axis = +z, which our
    // axisymmetry guard requires to be aligned with the C1→C2 line).
    // Sphere 1 at origin, sphere 2 at (0, 0, D); spine in the
    // z = a0 plane with axis +z.
    let a0 = (big_r1 * big_r1 - big_r2 * big_r2 + big_d * big_d) / (2.0 * big_d);
    let r_p_sq = big_r1 * big_r1 - a0 * a0;
    let r_p = r_p_sq.sqrt();

    let s1 = SphericalSurface::new(Point3::new(0.0, 0.0, 0.0), big_r1).unwrap();
    let s2 = SphericalSurface::new(Point3::new(0.0, 0.0, big_d), big_r2).unwrap();
    let spine_circle =
        Circle3D::new(Point3::new(0.0, 0.0, a0), Vec3::new(0.0, 0.0, 1.0), r_p).unwrap();
    let v = topo.add_vertex(Vertex::new(Point3::new(r_p, 0.0, a0), 1e-7));
    let eid = topo.add_edge(Edge::new(v, v, EdgeCurve::Circle(spine_circle)));
    let spine = Spine::from_single_edge(&topo, eid).unwrap();

    let w1 = topo.add_wire(Wire::new(vec![OrientedEdge::new(eid, true)], true).unwrap());
    let face1 = topo.add_face(Face::new(w1, vec![], FaceSurface::Sphere(s1.clone())));
    let w2 = topo.add_wire(Wire::new(vec![OrientedEdge::new(eid, false)], true).unwrap());
    let face2 = topo.add_face(Face::new(w2, vec![], FaceSurface::Sphere(s2.clone())));

    let result = sphere_sphere_fillet(&s1, &s2, &spine, &topo, r_fillet, face1, face2)
        .unwrap()
        .expect("convex sphere-sphere fillet should produce a stripe");

    let torus = match result.stripe.surface {
        FaceSurface::Torus(t) => t,
        other => panic!("expected Torus, got {}", other.type_tag()),
    };

    // Predicted torus parameters.
    let big_delta = (big_r1 - big_r2) / big_d;
    let a_ball = a0 + r_fillet * big_delta;
    let expected_major = ((big_r1 + r_fillet) * (big_r1 + r_fillet) - a_ball * a_ball).sqrt();
    assert!(
        (torus.major_radius() - expected_major).abs() < 1e-12,
        "major should be √((R1+r)²−a_ball²)={expected_major}, got {}",
        torus.major_radius()
    );
    assert!(
        (torus.minor_radius() - r_fillet).abs() < 1e-12,
        "minor should equal fillet radius {r_fillet}, got {}",
        torus.minor_radius()
    );

    // Torus center sits on the C1-C2 axis at the rolling-ball axial
    // position. C1=origin, axis=+z, so center=(0, 0, a_ball).
    let center = torus.center();
    assert!(
        center.x().abs() < 1e-12 && center.y().abs() < 1e-12,
        "torus center should be on +z axis, got {center:?}"
    );
    assert!(
        (center.z() - a_ball).abs() < 1e-12,
        "torus center z should be a_ball={a_ball}, got {}",
        center.z()
    );

    // Predict and verify both 3D contact points lie ON the torus.
    // sphere1 contact axial from C1 = R1·a_ball/(R1+r).
    let s1_axial = big_r1 * a_ball / (big_r1 + r_fillet);
    let s1_radial = big_r1 * expected_major / (big_r1 + r_fillet);
    let want_s1 = Point3::new(s1_radial, 0.0, s1_axial);
    // sphere2 contact axial from C2 = R2·(a_ball − D)/(R2+r).
    // World z = D + that.
    let s2_axial_from_c2 = big_r2 * (a_ball - big_d) / (big_r2 + r_fillet);
    let s2_radial = big_r2 * expected_major / (big_r2 + r_fillet);
    let want_s2 = Point3::new(s2_radial, 0.0, big_d + s2_axial_from_c2);

    let (u_p, v_p) = ParametricSurface::project_point(&torus, want_s1);
    let on_torus_s1 = ParametricSurface::evaluate(&torus, u_p, v_p);
    let (u_q, v_q) = ParametricSurface::project_point(&torus, want_s2);
    let on_torus_s2 = ParametricSurface::evaluate(&torus, u_q, v_q);
    assert!(
        (on_torus_s1 - want_s1).length() < 1e-9,
        "sphere1 contact must lie on torus: gave {on_torus_s1:?}, want {want_s1:?}"
    );
    assert!(
        (on_torus_s2 - want_s2).length() < 1e-9,
        "sphere2 contact must lie on torus: gave {on_torus_s2:?}, want {want_s2:?}"
    );

    // And both contact points lie on their respective spheres.
    let dist_s1 = (want_s1 - Point3::new(0.0, 0.0, 0.0)).length();
    let dist_s2 = (want_s2 - Point3::new(0.0, 0.0, big_d)).length();
    assert!(
        (dist_s1 - big_r1).abs() < 1e-9,
        "sphere1 contact must lie on sphere1: distance={dist_s1}, want R1={big_r1}"
    );
    assert!(
        (dist_s2 - big_r2).abs() < 1e-9,
        "sphere2 contact must lie on sphere2: distance={dist_s2}, want R2={big_r2}"
    );
}

/// Sphere-sphere both-concave fillet: two intersecting spherical
/// cavities (e.g. two overlapping ball-shaped voids carved into a
/// solid). Both faces REVERSED ⇒ rolling ball internally tangent to
/// both spheres; effective radii Q1=R1−r, Q2=R2−r.
///
/// For sphere1 at origin (R=2), sphere2 at (0,0,3) (R=2.5), D=3,
/// both faces REVERSED, r=0.4:
///   Q1 = 1.6, Q2 = 2.1
///   a_ball = (Q1²−Q2²+D²)/(2D) = 7.15/6 ≈ 1.192
///   R_t²   = Q1²−a_ball² = 2.56−1.421 ≈ 1.139
///   R_t    ≈ 1.067 (smaller than convex case where R_t ≈ 2.154,
///                    confirming the internal-tangency reduction)
#[test]
fn sphere_sphere_fillet_both_concave_emits_smaller_torus() {
    use brepkit_math::curves::Circle3D;
    use brepkit_math::surfaces::SphericalSurface;
    use brepkit_topology::edge::{Edge, EdgeCurve};
    use brepkit_topology::face::Face;
    use brepkit_topology::vertex::Vertex;
    use brepkit_topology::wire::{OrientedEdge, Wire};

    let mut topo = Topology::new();
    let big_r1: f64 = 2.0;
    let big_r2: f64 = 2.5;
    let big_d: f64 = 3.0;
    let r_fillet: f64 = 0.4;

    let a0 = (big_r1 * big_r1 - big_r2 * big_r2 + big_d * big_d) / (2.0 * big_d);
    let r_p_sq = big_r1 * big_r1 - a0 * a0;
    let r_p = r_p_sq.sqrt();

    let s1 = SphericalSurface::new(Point3::new(0.0, 0.0, 0.0), big_r1).unwrap();
    let s2 = SphericalSurface::new(Point3::new(0.0, 0.0, big_d), big_r2).unwrap();
    let spine_circle =
        Circle3D::new(Point3::new(0.0, 0.0, a0), Vec3::new(0.0, 0.0, 1.0), r_p).unwrap();
    let v = topo.add_vertex(Vertex::new(Point3::new(r_p, 0.0, a0), 1e-7));
    let eid = topo.add_edge(Edge::new(v, v, EdgeCurve::Circle(spine_circle)));
    let spine = Spine::from_single_edge(&topo, eid).unwrap();

    let w1 = topo.add_wire(Wire::new(vec![OrientedEdge::new(eid, true)], true).unwrap());
    let face1 = topo.add_face(Face::new_reversed(
        w1,
        vec![],
        FaceSurface::Sphere(s1.clone()),
    ));
    let w2 = topo.add_wire(Wire::new(vec![OrientedEdge::new(eid, false)], true).unwrap());
    let face2 = topo.add_face(Face::new_reversed(
        w2,
        vec![],
        FaceSurface::Sphere(s2.clone()),
    ));

    let result = sphere_sphere_fillet(&s1, &s2, &spine, &topo, r_fillet, face1, face2)
        .unwrap()
        .expect("both-concave sphere-sphere fillet should produce a stripe");

    let torus = match result.stripe.surface {
        FaceSurface::Torus(t) => t,
        other => panic!("expected Torus, got {}", other.type_tag()),
    };

    let q1 = big_r1 - r_fillet;
    let q2 = big_r2 - r_fillet;
    let a_ball = (q1 * q1 - q2 * q2 + big_d * big_d) / (2.0 * big_d);
    let expected_major = (q1 * q1 - a_ball * a_ball).sqrt();

    assert!(
        (torus.major_radius() - expected_major).abs() < 1e-12,
        "major should be √(Q1²−a_ball²)={expected_major}, got {}",
        torus.major_radius()
    );
    assert!(
        (torus.minor_radius() - r_fillet).abs() < 1e-12,
        "minor should equal fillet radius {r_fillet}, got {}",
        torus.minor_radius()
    );

    // Crucial check: concave torus is SMALLER than the convex
    // counterpart at the same r — internal vs external tangency.
    // Compute the convex major for reference.
    let q1_conv = big_r1 + r_fillet;
    let q2_conv = big_r2 + r_fillet;
    let a_ball_conv = (q1_conv * q1_conv - q2_conv * q2_conv + big_d * big_d) / (2.0 * big_d);
    let convex_major = (q1_conv * q1_conv - a_ball_conv * a_ball_conv).sqrt();
    assert!(
        torus.major_radius() < convex_major,
        "concave major ({}) must be smaller than convex major ({convex_major}) at same r",
        torus.major_radius()
    );

    // Verify each contact lies on its respective sphere.
    let s1_axial = big_r1 * a_ball / q1;
    let s1_radial = big_r1 * expected_major / q1;
    let want_s1 = Point3::new(s1_radial, 0.0, s1_axial);
    let s2_axial_from_c2 = big_r2 * (a_ball - big_d) / q2;
    let s2_radial = big_r2 * expected_major / q2;
    let want_s2 = Point3::new(s2_radial, 0.0, big_d + s2_axial_from_c2);

    let dist_s1 = (want_s1 - Point3::new(0.0, 0.0, 0.0)).length();
    let dist_s2 = (want_s2 - Point3::new(0.0, 0.0, big_d)).length();
    assert!(
        (dist_s1 - big_r1).abs() < 1e-9,
        "sphere1 contact must lie on sphere1: distance={dist_s1}, want R1={big_r1}"
    );
    assert!(
        (dist_s2 - big_r2).abs() < 1e-9,
        "sphere2 contact must lie on sphere2: distance={dist_s2}, want R2={big_r2}"
    );

    // And both lie on the torus.
    let (u_p, v_p) = ParametricSurface::project_point(&torus, want_s1);
    let on_torus_s1 = ParametricSurface::evaluate(&torus, u_p, v_p);
    let (u_q, v_q) = ParametricSurface::project_point(&torus, want_s2);
    let on_torus_s2 = ParametricSurface::evaluate(&torus, u_q, v_q);
    assert!(
        (on_torus_s1 - want_s1).length() < 1e-9,
        "sphere1 contact must lie on torus: {on_torus_s1:?} vs {want_s1:?}"
    );
    assert!(
        (on_torus_s2 - want_s2).length() < 1e-9,
        "sphere2 contact must lie on torus: {on_torus_s2:?} vs {want_s2:?}"
    );
}

/// Sphere-sphere fillet rejects radii that collapse `Qi = Ri − r` to
/// zero in the concave case (rolling ball would coincide with sphere
/// center). Convex at the same r is still valid.
#[test]
fn sphere_sphere_fillet_concave_rejects_collapsing_q() {
    use brepkit_math::curves::Circle3D;
    use brepkit_math::surfaces::SphericalSurface;
    use brepkit_topology::edge::{Edge, EdgeCurve};
    use brepkit_topology::face::Face;
    use brepkit_topology::vertex::Vertex;
    use brepkit_topology::wire::{OrientedEdge, Wire};

    let mut topo = Topology::new();
    let big_r: f64 = 2.0;
    let big_d: f64 = 3.0;
    // r ≥ R1 ⇒ Q1 ≤ 0 in the concave case.
    let r_too_big = 2.1;

    let a0 = (big_r * big_r - big_r * big_r + big_d * big_d) / (2.0 * big_d);
    let r_p_sq = big_r * big_r - a0 * a0;
    let r_p = r_p_sq.sqrt();

    let s1 = SphericalSurface::new(Point3::new(0.0, 0.0, 0.0), big_r).unwrap();
    let s2 = SphericalSurface::new(Point3::new(0.0, 0.0, big_d), big_r).unwrap();
    let spine_circle =
        Circle3D::new(Point3::new(0.0, 0.0, a0), Vec3::new(0.0, 0.0, 1.0), r_p).unwrap();
    let v = topo.add_vertex(Vertex::new(Point3::new(r_p, 0.0, a0), 1e-7));
    let eid = topo.add_edge(Edge::new(v, v, EdgeCurve::Circle(spine_circle)));
    let spine = Spine::from_single_edge(&topo, eid).unwrap();

    let w1 = topo.add_wire(Wire::new(vec![OrientedEdge::new(eid, true)], true).unwrap());
    let face1 = topo.add_face(Face::new_reversed(
        w1,
        vec![],
        FaceSurface::Sphere(s1.clone()),
    ));
    let w2 = topo.add_wire(Wire::new(vec![OrientedEdge::new(eid, false)], true).unwrap());
    let face2 = topo.add_face(Face::new_reversed(
        w2,
        vec![],
        FaceSurface::Sphere(s2.clone()),
    ));

    let result = sphere_sphere_fillet(&s1, &s2, &spine, &topo, r_too_big, face1, face2).unwrap();
    assert!(
        result.is_none(),
        "concave fillet at r≥R should reject (Qi collapses to ≤ 0)"
    );

    // Convex at the same r is still fine.
    let mut topo2 = Topology::new();
    let v2 = topo2.add_vertex(Vertex::new(Point3::new(r_p, 0.0, a0), 1e-7));
    let circle2 = Circle3D::new(Point3::new(0.0, 0.0, a0), Vec3::new(0.0, 0.0, 1.0), r_p).unwrap();
    let eid2 = topo2.add_edge(Edge::new(v2, v2, EdgeCurve::Circle(circle2)));
    let spine2 = Spine::from_single_edge(&topo2, eid2).unwrap();
    let w1b = topo2.add_wire(Wire::new(vec![OrientedEdge::new(eid2, true)], true).unwrap());
    let face1b = topo2.add_face(Face::new(w1b, vec![], FaceSurface::Sphere(s1.clone())));
    let w2b = topo2.add_wire(Wire::new(vec![OrientedEdge::new(eid2, false)], true).unwrap());
    let face2b = topo2.add_face(Face::new(w2b, vec![], FaceSurface::Sphere(s2.clone())));
    let result_convex =
        sphere_sphere_fillet(&s1, &s2, &spine2, &topo2, r_too_big, face1b, face2b).unwrap();
    assert!(
        result_convex.is_some(),
        "convex fillet at the same r={r_too_big} should still succeed"
    );
}

/// Sphere-sphere mixed-convexity fillet: sphere1 face NOT reversed
/// (convex; ball externally tangent, `Q1 = R1 + r`); sphere2 face
/// REVERSED (concave; ball internally tangent, `Q2 = R2 − r`).
/// Geometrically this is the "post emerging through a spherical
/// cavity" configuration — uncommon but the Q-substitution handles
/// it just like the symmetric cases.
///
/// For R1=2, R2=2.5, D=3, sphere1 NOT reversed, sphere2 REVERSED,
/// r=0.4:
///   Q1 = 2.4, Q2 = 2.1
///   a_ball = (5.76 − 4.41 + 9)/6 = 10.35/6 = 1.725
///   R_t² = 5.76 − 2.976 = 2.784, R_t ≈ 1.668
/// (Sandwiched between the convex-convex R_t ≈ 2.154 and the
/// concave-concave R_t ≈ 1.067, which is what we'd expect.)
#[test]
fn sphere_sphere_fillet_mixed_emits_torus() {
    use brepkit_math::curves::Circle3D;
    use brepkit_math::surfaces::SphericalSurface;
    use brepkit_topology::edge::{Edge, EdgeCurve};
    use brepkit_topology::face::Face;
    use brepkit_topology::vertex::Vertex;
    use brepkit_topology::wire::{OrientedEdge, Wire};

    let mut topo = Topology::new();
    let big_r1: f64 = 2.0;
    let big_r2: f64 = 2.5;
    let big_d: f64 = 3.0;
    let r_fillet: f64 = 0.4;

    let a0 = (big_r1 * big_r1 - big_r2 * big_r2 + big_d * big_d) / (2.0 * big_d);
    let r_p_sq = big_r1 * big_r1 - a0 * a0;
    let r_p = r_p_sq.sqrt();

    let s1 = SphericalSurface::new(Point3::new(0.0, 0.0, 0.0), big_r1).unwrap();
    let s2 = SphericalSurface::new(Point3::new(0.0, 0.0, big_d), big_r2).unwrap();
    let spine_circle =
        Circle3D::new(Point3::new(0.0, 0.0, a0), Vec3::new(0.0, 0.0, 1.0), r_p).unwrap();
    let v = topo.add_vertex(Vertex::new(Point3::new(r_p, 0.0, a0), 1e-7));
    let eid = topo.add_edge(Edge::new(v, v, EdgeCurve::Circle(spine_circle)));
    let spine = Spine::from_single_edge(&topo, eid).unwrap();

    let w1 = topo.add_wire(Wire::new(vec![OrientedEdge::new(eid, true)], true).unwrap());
    // Sphere 1: NOT reversed (convex, external tangency).
    let face1 = topo.add_face(Face::new(w1, vec![], FaceSurface::Sphere(s1.clone())));
    let w2 = topo.add_wire(Wire::new(vec![OrientedEdge::new(eid, false)], true).unwrap());
    // Sphere 2: REVERSED (concave, internal tangency).
    let face2 = topo.add_face(Face::new_reversed(
        w2,
        vec![],
        FaceSurface::Sphere(s2.clone()),
    ));

    let result = sphere_sphere_fillet(&s1, &s2, &spine, &topo, r_fillet, face1, face2)
        .unwrap()
        .expect("mixed sphere-sphere fillet should produce a stripe");

    let torus = match result.stripe.surface {
        FaceSurface::Torus(t) => t,
        other => panic!("expected Torus, got {}", other.type_tag()),
    };

    let q1 = big_r1 + r_fillet; // sphere1 convex
    let q2 = big_r2 - r_fillet; // sphere2 concave
    let a_ball = (q1 * q1 - q2 * q2 + big_d * big_d) / (2.0 * big_d);
    let expected_major = (q1 * q1 - a_ball * a_ball).sqrt();

    assert!(
        (torus.major_radius() - expected_major).abs() < 1e-12,
        "mixed major should be √(Q1²−a_ball²)={expected_major}, got {}",
        torus.major_radius()
    );

    // Check ordering: mixed major must sit BETWEEN convex/convex and
    // concave/concave at the same r — confirms the Q-substitution
    // produces the right interpolation.
    let q1_cc = big_r1 + r_fillet;
    let q2_cc = big_r2 + r_fillet;
    let a_ball_cc = (q1_cc * q1_cc - q2_cc * q2_cc + big_d * big_d) / (2.0 * big_d);
    let convex_convex_major = (q1_cc * q1_cc - a_ball_cc * a_ball_cc).sqrt();
    let q1_kk = big_r1 - r_fillet;
    let q2_kk = big_r2 - r_fillet;
    let a_ball_kk = (q1_kk * q1_kk - q2_kk * q2_kk + big_d * big_d) / (2.0 * big_d);
    let concave_concave_major = (q1_kk * q1_kk - a_ball_kk * a_ball_kk).sqrt();
    assert!(
        torus.major_radius() < convex_convex_major && torus.major_radius() > concave_concave_major,
        "mixed major ({}) should sit between concave-concave ({concave_concave_major}) and convex-convex ({convex_convex_major})",
        torus.major_radius()
    );

    // Both contacts on respective spheres.
    let s1_axial = big_r1 * a_ball / q1;
    let s1_radial = big_r1 * expected_major / q1;
    let want_s1 = Point3::new(s1_radial, 0.0, s1_axial);
    let s2_axial_from_c2 = big_r2 * (a_ball - big_d) / q2;
    let s2_radial = big_r2 * expected_major / q2;
    let want_s2 = Point3::new(s2_radial, 0.0, big_d + s2_axial_from_c2);

    let dist_s1 = (want_s1 - Point3::new(0.0, 0.0, 0.0)).length();
    let dist_s2 = (want_s2 - Point3::new(0.0, 0.0, big_d)).length();
    assert!(
        (dist_s1 - big_r1).abs() < 1e-9,
        "sphere1 contact must lie on sphere1: distance={dist_s1}, want R1={big_r1}"
    );
    assert!(
        (dist_s2 - big_r2).abs() < 1e-9,
        "sphere2 contact must lie on sphere2: distance={dist_s2}, want R2={big_r2}"
    );

    // Both on torus.
    let (u_p, v_p) = ParametricSurface::project_point(&torus, want_s1);
    let on_torus_s1 = ParametricSurface::evaluate(&torus, u_p, v_p);
    let (u_q, v_q) = ParametricSurface::project_point(&torus, want_s2);
    let on_torus_s2 = ParametricSurface::evaluate(&torus, u_q, v_q);
    assert!(
        (on_torus_s1 - want_s1).length() < 1e-9,
        "sphere1 contact must lie on torus: {on_torus_s1:?} vs {want_s1:?}"
    );
    assert!(
        (on_torus_s2 - want_s2).length() < 1e-9,
        "sphere2 contact must lie on torus: {on_torus_s2:?} vs {want_s2:?}"
    );
}

/// Sphere-sphere convex chamfer: two intersecting spheres meeting
/// along a circular spine; chamfer surface is an axisymmetric cone
/// connecting both sphere-side contact circles.
///
/// For sphere1 at origin (R=2), sphere2 at (0, 0, 3) (R=2.5), D=3,
/// both faces NOT reversed, symmetric d=0.4:
///   - δ1 = 0.2, δ2 = 0.16
///   - contact1 at radial r_p·cos δ1 + a₀·sin δ1 ≈ 1.844,
///     z ≈ a₀·cos δ1 − r_p·sin δ1 ≈ 0.774 (z<a₀: below spine)
///   - contact2 at radial r_p·cos δ2 + (D−a₀)·sin δ2 ≈ 1.932,
///     z ≈ D − (D−a₀)·cos δ2 + r_p·sin δ2 ≈ 1.413 (z>a₀: above spine)
///   - Cone apex below both contacts on the +z axis (~z=−12.6)
///   - Cone axis = +z (opens upward toward both contacts)
#[test]
fn sphere_sphere_chamfer_convex_emits_cone() {
    use brepkit_math::curves::Circle3D;
    use brepkit_math::surfaces::SphericalSurface;
    use brepkit_topology::edge::{Edge, EdgeCurve};
    use brepkit_topology::face::Face;
    use brepkit_topology::vertex::Vertex;
    use brepkit_topology::wire::{OrientedEdge, Wire};

    let mut topo = Topology::new();
    let big_r1: f64 = 2.0;
    let big_r2: f64 = 2.5;
    let big_d: f64 = 3.0;
    let d: f64 = 0.4;

    let a0 = (big_r1 * big_r1 - big_r2 * big_r2 + big_d * big_d) / (2.0 * big_d);
    let r_p_sq = big_r1 * big_r1 - a0 * a0;
    let r_p = r_p_sq.sqrt();

    let s1 = SphericalSurface::new(Point3::new(0.0, 0.0, 0.0), big_r1).unwrap();
    let s2 = SphericalSurface::new(Point3::new(0.0, 0.0, big_d), big_r2).unwrap();
    let spine_circle =
        Circle3D::new(Point3::new(0.0, 0.0, a0), Vec3::new(0.0, 0.0, 1.0), r_p).unwrap();
    let v = topo.add_vertex(Vertex::new(Point3::new(r_p, 0.0, a0), 1e-7));
    let eid = topo.add_edge(Edge::new(v, v, EdgeCurve::Circle(spine_circle)));
    let spine = Spine::from_single_edge(&topo, eid).unwrap();

    let w1 = topo.add_wire(Wire::new(vec![OrientedEdge::new(eid, true)], true).unwrap());
    let face1 = topo.add_face(Face::new(w1, vec![], FaceSurface::Sphere(s1.clone())));
    let w2 = topo.add_wire(Wire::new(vec![OrientedEdge::new(eid, false)], true).unwrap());
    let face2 = topo.add_face(Face::new(w2, vec![], FaceSurface::Sphere(s2.clone())));

    let result = sphere_sphere_chamfer(&s1, &s2, &spine, &topo, d, d, face1, face2)
        .unwrap()
        .expect("convex sphere-sphere chamfer should produce a stripe");

    let chamfer_cone = match result.stripe.surface {
        FaceSurface::Cone(c) => c,
        other => panic!("expected Cone, got {}", other.type_tag()),
    };

    // Predicted contacts.
    let delta1 = d / big_r1;
    let delta2 = d / big_r2;
    let (sin1, cos1) = delta1.sin_cos();
    let (sin2, cos2) = delta2.sin_cos();
    let p1_r = r_p * cos1 + a0 * sin1;
    let p1_z = a0 * cos1 - r_p * sin1;
    let p2_r = r_p * cos2 + (big_d - a0) * sin2;
    let p2_z = big_d - (big_d - a0) * cos2 + r_p * sin2;

    // Contact1 below spine, contact2 above spine — characteristic
    // of the convex-convex case (faces extend AWAY from each other).
    assert!(p1_z < a0, "convex contact1 should be below spine z=a0");
    assert!(p2_z > a0, "convex contact2 should be above spine z=a0");

    // Predicted apex from line P1-P2 extrapolation to r=0.
    let dr = p2_r - p1_r;
    let dz = p2_z - p1_z;
    let expected_apex_z = p1_z - p1_r * dz / dr;

    let apex = chamfer_cone.apex();
    assert!(
        apex.x().abs() < 1e-12 && apex.y().abs() < 1e-12,
        "apex should be on z-axis, got {apex:?}"
    );
    assert!(
        (apex.z() - expected_apex_z).abs() < 1e-9,
        "apex z = {}, expected {expected_apex_z}",
        apex.z()
    );

    // Cone axis: contacts are above apex (mid_z > apex_z), so axis = +z.
    let axis = chamfer_cone.axis();
    assert!(
        axis.dot(Vec3::new(0.0, 0.0, 1.0)) > 1.0 - 1e-12,
        "convex chamfer cone axis should be +z, got {axis:?}"
    );

    // Both contacts must lie on the chamfer cone.
    let want_p1 = Point3::new(p1_r, 0.0, p1_z);
    let want_p2 = Point3::new(p2_r, 0.0, p2_z);
    let (u_p, v_p) = ParametricSurface::project_point(&chamfer_cone, want_p1);
    let on_cone_p1 = ParametricSurface::evaluate(&chamfer_cone, u_p, v_p);
    let (u_q, v_q) = ParametricSurface::project_point(&chamfer_cone, want_p2);
    let on_cone_p2 = ParametricSurface::evaluate(&chamfer_cone, u_q, v_q);
    assert!(
        (on_cone_p1 - want_p1).length() < 1e-9,
        "contact1 must lie on chamfer cone: {on_cone_p1:?} vs {want_p1:?}"
    );
    assert!(
        (on_cone_p2 - want_p2).length() < 1e-9,
        "contact2 must lie on chamfer cone: {on_cone_p2:?} vs {want_p2:?}"
    );

    // Both contacts also lie on their respective spheres.
    let dist_s1 = (want_p1 - Point3::new(0.0, 0.0, 0.0)).length();
    let dist_s2 = (want_p2 - Point3::new(0.0, 0.0, big_d)).length();
    assert!(
        (dist_s1 - big_r1).abs() < 1e-9,
        "contact1 must lie on sphere1: distance={dist_s1}, want R1={big_r1}"
    );
    assert!(
        (dist_s2 - big_r2).abs() < 1e-9,
        "contact2 must lie on sphere2: distance={dist_s2}, want R2={big_r2}"
    );
}

/// Sphere-cylinder convex fillet: a sphere primitive fused to a
/// cylinder primitive along their shared axis. The intersection is
/// a pair of circles at axial offset ±h_s = ±√(R_s²−r_c²) from the
/// sphere center; we fillet the +h_s spine.
///
/// For sphere at origin (R=3), cylinder axis +z through origin
/// (r_c=2), both faces NOT reversed, r=0.4:
///   - h_s = √(9−4) = √5 ≈ 2.236
///   - Q_s = 3.4, Q_c = 2.4
///   - a_ball = √(Q_s² − Q_c²) = √(11.56 − 5.76) = √5.8 ≈ 2.408
///   - major = Q_c = 2.4
#[test]
fn sphere_cylinder_fillet_convex_emits_torus() {
    use brepkit_math::curves::Circle3D;
    use brepkit_math::surfaces::{CylindricalSurface, SphericalSurface};
    use brepkit_topology::edge::{Edge, EdgeCurve};
    use brepkit_topology::face::Face;
    use brepkit_topology::vertex::Vertex;
    use brepkit_topology::wire::{OrientedEdge, Wire};

    let mut topo = Topology::new();
    let big_r_s: f64 = 3.0;
    let r_c: f64 = 2.0;
    let r_fillet: f64 = 0.4;
    let h_s = (big_r_s * big_r_s - r_c * r_c).sqrt();

    let sph = SphericalSurface::new(Point3::new(0.0, 0.0, 0.0), big_r_s).unwrap();
    let cyl =
        CylindricalSurface::new(Point3::new(0.0, 0.0, 0.0), Vec3::new(0.0, 0.0, 1.0), r_c).unwrap();

    // Spine: circle at z = +h_s, radius r_c, axis +z.
    let spine_circle =
        Circle3D::new(Point3::new(0.0, 0.0, h_s), Vec3::new(0.0, 0.0, 1.0), r_c).unwrap();
    let v = topo.add_vertex(Vertex::new(Point3::new(r_c, 0.0, h_s), 1e-7));
    let eid = topo.add_edge(Edge::new(v, v, EdgeCurve::Circle(spine_circle)));
    let spine = Spine::from_single_edge(&topo, eid).unwrap();

    let w1 = topo.add_wire(Wire::new(vec![OrientedEdge::new(eid, true)], true).unwrap());
    let face_sphere = topo.add_face(Face::new(w1, vec![], FaceSurface::Sphere(sph.clone())));
    let w2 = topo.add_wire(Wire::new(vec![OrientedEdge::new(eid, false)], true).unwrap());
    let face_cyl = topo.add_face(Face::new(w2, vec![], FaceSurface::Cylinder(cyl.clone())));

    let result = sphere_cylinder_fillet(&sph, &cyl, &spine, &topo, r_fillet, face_sphere, face_cyl)
        .unwrap()
        .expect("convex sphere-cylinder fillet should produce a stripe");

    let torus = match result.stripe.surface {
        FaceSurface::Torus(t) => t,
        other => panic!("expected Torus, got {}", other.type_tag()),
    };

    let q_s = big_r_s + r_fillet;
    let q_c = r_c + r_fillet;
    let expected_major = q_c;
    let expected_a_ball = (q_s * q_s - q_c * q_c).sqrt();

    assert!(
        (torus.major_radius() - expected_major).abs() < 1e-12,
        "major should be Q_c = {expected_major}, got {}",
        torus.major_radius()
    );
    assert!(
        (torus.minor_radius() - r_fillet).abs() < 1e-12,
        "minor should be r = {r_fillet}, got {}",
        torus.minor_radius()
    );

    // Torus center on +z axis at z = a_ball (positive since spine
    // is at +h_s).
    let center = torus.center();
    assert!(
        center.x().abs() < 1e-12 && center.y().abs() < 1e-12,
        "torus center should be on z-axis, got {center:?}"
    );
    assert!(
        (center.z() - expected_a_ball).abs() < 1e-12,
        "torus center z should be a_ball = {expected_a_ball}, got {}",
        center.z()
    );

    // Sphere contact radial = R_s · Q_c / Q_s, axial = R_s · a_ball / Q_s.
    let sph_axial = big_r_s * expected_a_ball / q_s;
    let sph_radial = big_r_s * q_c / q_s;
    let want_sph = Point3::new(sph_radial, 0.0, sph_axial);
    // Cylinder contact at radial r_c, axial a_ball.
    let want_cyl = Point3::new(r_c, 0.0, expected_a_ball);

    // Both lie on torus.
    let (u_p, v_p) = ParametricSurface::project_point(&torus, want_sph);
    let on_torus_sph = ParametricSurface::evaluate(&torus, u_p, v_p);
    let (u_q, v_q) = ParametricSurface::project_point(&torus, want_cyl);
    let on_torus_cyl = ParametricSurface::evaluate(&torus, u_q, v_q);
    assert!(
        (on_torus_sph - want_sph).length() < 1e-9,
        "sphere contact must lie on torus: {on_torus_sph:?} vs {want_sph:?}"
    );
    assert!(
        (on_torus_cyl - want_cyl).length() < 1e-9,
        "cylinder contact must lie on torus: {on_torus_cyl:?} vs {want_cyl:?}"
    );

    // Both lie on their respective surfaces.
    let dist_sph = (want_sph - Point3::new(0.0, 0.0, 0.0)).length();
    assert!(
        (dist_sph - big_r_s).abs() < 1e-9,
        "sphere contact must lie on sphere: distance = {dist_sph}, want R_s = {big_r_s}"
    );
    let dist_cyl_radial = (want_cyl.x().powi(2) + want_cyl.y().powi(2)).sqrt();
    assert!(
        (dist_cyl_radial - r_c).abs() < 1e-9,
        "cylinder contact must lie on cylinder: radial = {dist_cyl_radial}, want r_c = {r_c}"
    );
}

/// Sphere-cylinder convex chamfer: a sphere primitive fused to a
/// cylinder primitive along their shared axis. The chamfer surface
/// is an axisymmetric cone connecting the sphere-side and
/// cylinder-side contacts.
///
/// For sphere at origin (R=3), cylinder axis +z (r_c=2), spine at
/// z=+h_s=+√5≈2.236, both faces NOT reversed, symmetric d=0.4:
///   - δ = 0.4/3 ≈ 0.1333
///   - sphere contact at radial r_c·cos δ − h_s·sin δ ≈ 1.685,
///     z = h_s·cos δ + r_c·sin δ ≈ 2.482
///   - cyl contact at radial r_c=2.0, z = h_s − d ≈ 1.836
///   - Δr ≈ 0.315, Δz ≈ −0.646
///   - apex z = z_sph − r_sph·Δz/Δr ≈ 5.94 (above contacts)
#[test]
fn sphere_cylinder_chamfer_convex_emits_cone() {
    use brepkit_math::curves::Circle3D;
    use brepkit_math::surfaces::{CylindricalSurface, SphericalSurface};
    use brepkit_topology::edge::{Edge, EdgeCurve};
    use brepkit_topology::face::Face;
    use brepkit_topology::vertex::Vertex;
    use brepkit_topology::wire::{OrientedEdge, Wire};

    let mut topo = Topology::new();
    let big_r_s: f64 = 3.0;
    let r_c: f64 = 2.0;
    let d: f64 = 0.4;
    let h_s = (big_r_s * big_r_s - r_c * r_c).sqrt();

    let sph = SphericalSurface::new(Point3::new(0.0, 0.0, 0.0), big_r_s).unwrap();
    let cyl =
        CylindricalSurface::new(Point3::new(0.0, 0.0, 0.0), Vec3::new(0.0, 0.0, 1.0), r_c).unwrap();

    let spine_circle =
        Circle3D::new(Point3::new(0.0, 0.0, h_s), Vec3::new(0.0, 0.0, 1.0), r_c).unwrap();
    let v = topo.add_vertex(Vertex::new(Point3::new(r_c, 0.0, h_s), 1e-7));
    let eid = topo.add_edge(Edge::new(v, v, EdgeCurve::Circle(spine_circle)));
    let spine = Spine::from_single_edge(&topo, eid).unwrap();

    let w1 = topo.add_wire(Wire::new(vec![OrientedEdge::new(eid, true)], true).unwrap());
    let face_sphere = topo.add_face(Face::new(w1, vec![], FaceSurface::Sphere(sph.clone())));
    let w2 = topo.add_wire(Wire::new(vec![OrientedEdge::new(eid, false)], true).unwrap());
    let face_cyl = topo.add_face(Face::new(w2, vec![], FaceSurface::Cylinder(cyl.clone())));

    let result = sphere_cylinder_chamfer(&sph, &cyl, &spine, &topo, d, d, face_sphere, face_cyl)
        .unwrap()
        .expect("convex sphere-cylinder chamfer should produce a stripe");

    let chamfer_cone = match result.stripe.surface {
        FaceSurface::Cone(c) => c,
        other => panic!("expected Cone, got {}", other.type_tag()),
    };

    // Predicted contacts.
    let delta = d / big_r_s;
    let (sin_d, cos_d) = delta.sin_cos();
    let r_sph_pred = r_c * cos_d - h_s * sin_d;
    let z_sph_pred = h_s * cos_d + r_c * sin_d;
    let r_cyl_pred = r_c;
    let z_cyl_pred = h_s - d;

    // Predicted apex from line extrapolation.
    let dr = r_cyl_pred - r_sph_pred;
    let dz = z_cyl_pred - z_sph_pred;
    let expected_apex_z = z_sph_pred - r_sph_pred * dz / dr;

    let apex = chamfer_cone.apex();
    assert!(
        apex.x().abs() < 1e-12 && apex.y().abs() < 1e-12,
        "apex should be on z-axis, got {apex:?}"
    );
    assert!(
        (apex.z() - expected_apex_z).abs() < 1e-9,
        "apex z = {}, expected {expected_apex_z}",
        apex.z()
    );
    assert!(
        expected_apex_z > z_sph_pred,
        "apex should be above the contacts (mid z < apex z), got apex_z={expected_apex_z}"
    );

    // Cone axis: contacts are below apex, so axis points -z.
    let axis = chamfer_cone.axis();
    assert!(
        axis.dot(Vec3::new(0.0, 0.0, 1.0)) < -1.0 + 1e-12,
        "convex chamfer cone axis should be -z (apex above), got {axis:?}"
    );

    // Both contacts on the chamfer cone.
    let want_sph = Point3::new(r_sph_pred, 0.0, z_sph_pred);
    let want_cyl = Point3::new(r_cyl_pred, 0.0, z_cyl_pred);
    let (u_p, v_p) = ParametricSurface::project_point(&chamfer_cone, want_sph);
    let on_cone_sph = ParametricSurface::evaluate(&chamfer_cone, u_p, v_p);
    let (u_q, v_q) = ParametricSurface::project_point(&chamfer_cone, want_cyl);
    let on_cone_cyl = ParametricSurface::evaluate(&chamfer_cone, u_q, v_q);
    assert!(
        (on_cone_sph - want_sph).length() < 1e-9,
        "sphere contact must lie on chamfer cone: {on_cone_sph:?} vs {want_sph:?}"
    );
    assert!(
        (on_cone_cyl - want_cyl).length() < 1e-9,
        "cylinder contact must lie on chamfer cone: {on_cone_cyl:?} vs {want_cyl:?}"
    );

    // Sphere contact lies on sphere; cylinder contact has correct radius.
    let dist_sph = (want_sph - Point3::new(0.0, 0.0, 0.0)).length();
    assert!(
        (dist_sph - big_r_s).abs() < 1e-9,
        "sphere contact must lie on sphere: distance={dist_sph}, want R_s={big_r_s}"
    );
    let dist_cyl_radial = (want_cyl.x().powi(2) + want_cyl.y().powi(2)).sqrt();
    assert!(
        (dist_cyl_radial - r_c).abs() < 1e-9,
        "cylinder contact must have radial r_c: got {dist_cyl_radial}, want {r_c}"
    );
}

/// Sphere-sphere both-concave chamfer: two intersecting spherical
/// cavities. Both `s1 = s2 = −1` flip the meridian arms; each sphere
/// goes TOWARD the other (instead of AWAY in the convex case).
///
/// The implementation already handled per-sphere signed_offsets; this
/// test confirms it works for the symmetric concave case.
///
/// For R1=2, R2=2.5, D=3, both faces REVERSED, symmetric d=0.4:
///   - δ1 = 0.2, δ2 = 0.16, a₀ ≈ 1.125, r_p ≈ 1.654
///   - p1_r = r_p·cos δ1 − a₀·sin δ1 ≈ 1.398
///     (less than convex r_p+a₀_term ≈ 1.844 — going toward axis)
///   - p1_z = a₀·cos δ1 + r_p·sin δ1 ≈ 1.432
///     (ABOVE spine z = a₀, opposite convex which had below)
///   - p2_z = D − (D−a₀)·cos δ2 − r_p·sin δ2 ≈ 0.886
///     (BELOW spine, opposite convex)
#[test]
fn sphere_sphere_chamfer_both_concave_emits_cone() {
    use brepkit_math::curves::Circle3D;
    use brepkit_math::surfaces::SphericalSurface;
    use brepkit_topology::edge::{Edge, EdgeCurve};
    use brepkit_topology::face::Face;
    use brepkit_topology::vertex::Vertex;
    use brepkit_topology::wire::{OrientedEdge, Wire};

    let mut topo = Topology::new();
    let big_r1: f64 = 2.0;
    let big_r2: f64 = 2.5;
    let big_d: f64 = 3.0;
    let d: f64 = 0.4;

    let a0 = (big_r1 * big_r1 - big_r2 * big_r2 + big_d * big_d) / (2.0 * big_d);
    let r_p_sq = big_r1 * big_r1 - a0 * a0;
    let r_p = r_p_sq.sqrt();

    let s1 = SphericalSurface::new(Point3::new(0.0, 0.0, 0.0), big_r1).unwrap();
    let s2 = SphericalSurface::new(Point3::new(0.0, 0.0, big_d), big_r2).unwrap();
    let spine_circle =
        Circle3D::new(Point3::new(0.0, 0.0, a0), Vec3::new(0.0, 0.0, 1.0), r_p).unwrap();
    let v = topo.add_vertex(Vertex::new(Point3::new(r_p, 0.0, a0), 1e-7));
    let eid = topo.add_edge(Edge::new(v, v, EdgeCurve::Circle(spine_circle)));
    let spine = Spine::from_single_edge(&topo, eid).unwrap();

    let w1 = topo.add_wire(Wire::new(vec![OrientedEdge::new(eid, true)], true).unwrap());
    let face1 = topo.add_face(Face::new_reversed(
        w1,
        vec![],
        FaceSurface::Sphere(s1.clone()),
    ));
    let w2 = topo.add_wire(Wire::new(vec![OrientedEdge::new(eid, false)], true).unwrap());
    let face2 = topo.add_face(Face::new_reversed(
        w2,
        vec![],
        FaceSurface::Sphere(s2.clone()),
    ));

    let result = sphere_sphere_chamfer(&s1, &s2, &spine, &topo, d, d, face1, face2)
        .unwrap()
        .expect("both-concave sphere-sphere chamfer should produce a stripe");

    let chamfer_cone = match result.stripe.surface {
        FaceSurface::Cone(c) => c,
        other => panic!("expected Cone, got {}", other.type_tag()),
    };

    // Predicted contacts with s1 = s2 = -1.
    let delta1 = d / big_r1;
    let delta2 = d / big_r2;
    let (sin1, cos1) = delta1.sin_cos();
    let (sin2, cos2) = delta2.sin_cos();
    let p1_r = r_p * cos1 - a0 * sin1; // s1 = -1 ⇒ -a0
    let p1_z = a0 * cos1 + r_p * sin1; // s1 = -1 ⇒ +r_p
    let p2_r = r_p * cos2 - (big_d - a0) * sin2; // s2 = -1
    let p2_z = big_d - (big_d - a0) * cos2 - r_p * sin2; // s2 = -1 ⇒ -r_p

    // Concave-concave: contact1 ABOVE spine (z > a₀), contact2 BELOW
    // spine (z < a₀) — opposite the convex-convex pattern.
    assert!(
        p1_z > a0,
        "concave contact1 should be above spine (z > a0): got {p1_z}"
    );
    assert!(
        p2_z < a0,
        "concave contact2 should be below spine (z < a0): got {p2_z}"
    );

    // Both contacts on chamfer cone.
    let want_p1 = Point3::new(p1_r, 0.0, p1_z);
    let want_p2 = Point3::new(p2_r, 0.0, p2_z);
    let (u_p, v_p) = ParametricSurface::project_point(&chamfer_cone, want_p1);
    let on_cone_p1 = ParametricSurface::evaluate(&chamfer_cone, u_p, v_p);
    let (u_q, v_q) = ParametricSurface::project_point(&chamfer_cone, want_p2);
    let on_cone_p2 = ParametricSurface::evaluate(&chamfer_cone, u_q, v_q);
    assert!(
        (on_cone_p1 - want_p1).length() < 1e-9,
        "concave contact1 must lie on chamfer cone: {on_cone_p1:?} vs {want_p1:?}"
    );
    assert!(
        (on_cone_p2 - want_p2).length() < 1e-9,
        "concave contact2 must lie on chamfer cone: {on_cone_p2:?} vs {want_p2:?}"
    );

    // Both contacts on respective spheres.
    let dist_s1 = (want_p1 - Point3::new(0.0, 0.0, 0.0)).length();
    let dist_s2 = (want_p2 - Point3::new(0.0, 0.0, big_d)).length();
    assert!(
        (dist_s1 - big_r1).abs() < 1e-9,
        "contact1 must lie on sphere1: distance={dist_s1}, want R1={big_r1}"
    );
    assert!(
        (dist_s2 - big_r2).abs() < 1e-9,
        "contact2 must lie on sphere2: distance={dist_s2}, want R2={big_r2}"
    );
}

/// Sphere-sphere mixed-convexity chamfer: covers BOTH (s1=+1, s2=−1)
/// and (s1=−1, s2=+1). Each is geometrically distinct from the
/// symmetric cases (and from each other), with contacts on different
/// cap arms.
///
/// Verifies the chamfer via project_point on the EMITTED
/// `result.stripe.contact{1,2}` curves (not on test-computed
/// formulas, which would be tautological for the sphere-distance
/// check since `r² + (z − Cz)² = R²` holds algebraically for any
/// sign).
#[test]
fn sphere_sphere_chamfer_mixed_emits_cone() {
    use brepkit_math::curves::Circle3D;
    use brepkit_math::surfaces::SphericalSurface;
    use brepkit_topology::edge::{Edge, EdgeCurve};
    use brepkit_topology::face::Face;
    use brepkit_topology::vertex::Vertex;
    use brepkit_topology::wire::{OrientedEdge, Wire};

    let big_r1: f64 = 2.0;
    let big_r2: f64 = 2.5;
    let big_d: f64 = 3.0;
    let d: f64 = 0.4;
    let a0 = (big_r1 * big_r1 - big_r2 * big_r2 + big_d * big_d) / (2.0 * big_d);
    let r_p = (big_r1 * big_r1 - a0 * a0).sqrt();

    // Run both mixed configurations (s1, s2) ∈ {(+,−), (−,+)} via a
    // closure parameterized by which face to reverse. For each:
    //   - emit the chamfer
    //   - extract the actual 3D contact point from the impl's
    //     `contact1`/`contact2` NURBS curves
    //   - assert that the EMITTED contact lies on the corresponding
    //     sphere (this DOES test the implementation, unlike
    //     sampling the test's own formula — see Greptile feedback
    //     on PR #604: r² + (z − Cz)² = R² holds algebraically for
    //     any sign of `s_i`, so a formula-derived contact would
    //     pass tautologically).
    let run_case = |reverse_s1: bool, reverse_s2: bool| {
        let mut topo = Topology::new();
        let s1_surf = SphericalSurface::new(Point3::new(0.0, 0.0, 0.0), big_r1).unwrap();
        let s2_surf = SphericalSurface::new(Point3::new(0.0, 0.0, big_d), big_r2).unwrap();
        let spine_circle =
            Circle3D::new(Point3::new(0.0, 0.0, a0), Vec3::new(0.0, 0.0, 1.0), r_p).unwrap();
        let v = topo.add_vertex(Vertex::new(Point3::new(r_p, 0.0, a0), 1e-7));
        let eid = topo.add_edge(Edge::new(v, v, EdgeCurve::Circle(spine_circle)));
        let spine = Spine::from_single_edge(&topo, eid).unwrap();

        let w1 = topo.add_wire(Wire::new(vec![OrientedEdge::new(eid, true)], true).unwrap());
        let face1 = if reverse_s1 {
            topo.add_face(Face::new_reversed(
                w1,
                vec![],
                FaceSurface::Sphere(s1_surf.clone()),
            ))
        } else {
            topo.add_face(Face::new(w1, vec![], FaceSurface::Sphere(s1_surf.clone())))
        };
        let w2 = topo.add_wire(Wire::new(vec![OrientedEdge::new(eid, false)], true).unwrap());
        let face2 = if reverse_s2 {
            topo.add_face(Face::new_reversed(
                w2,
                vec![],
                FaceSurface::Sphere(s2_surf.clone()),
            ))
        } else {
            topo.add_face(Face::new(w2, vec![], FaceSurface::Sphere(s2_surf.clone())))
        };

        let result = sphere_sphere_chamfer(&s1_surf, &s2_surf, &spine, &topo, d, d, face1, face2)
            .unwrap()
            .expect("mixed sphere-sphere chamfer should produce a stripe");

        // Verify emitted contact endpoints lie on their respective
        // spheres. Sample the EMITTED curve via
        // `evaluate(t_start)` rather than reading control points —
        // rational NURBS arcs have intermediate control points
        // OFF the curve, and even endpoint coverage couples the
        // test to construction details (degree, knot vector).
        let (t1_start, _) = result.stripe.contact1.domain();
        let c1_point = result.stripe.contact1.evaluate(t1_start);
        let dist_s1 = (c1_point - Point3::new(0.0, 0.0, 0.0)).length();
        assert!(
            (dist_s1 - big_r1).abs() < 1e-9,
            "({reverse_s1}, {reverse_s2}): emitted contact1 must lie on sphere1: \
                 distance = {dist_s1}, want R1 = {big_r1}"
        );

        let (t2_start, _) = result.stripe.contact2.domain();
        let c2_point = result.stripe.contact2.evaluate(t2_start);
        let dist_s2 = (c2_point - Point3::new(0.0, 0.0, big_d)).length();
        assert!(
            (dist_s2 - big_r2).abs() < 1e-9,
            "({reverse_s1}, {reverse_s2}): emitted contact2 must lie on sphere2: \
                 distance = {dist_s2}, want R2 = {big_r2}"
        );

        // Emitted surface is a Cone (the chamfer is a cone of
        // revolution about the C1-C2 axis).
        assert!(
            matches!(result.stripe.surface, FaceSurface::Cone(_)),
            "({reverse_s1}, {reverse_s2}): expected Cone, got {}",
            result.stripe.surface.type_tag()
        );

        // Both emitted contact points lie on the chamfer cone via
        // project_point round-trip.
        if let FaceSurface::Cone(ref cone) = result.stripe.surface {
            let (u_p, v_p) = ParametricSurface::project_point(cone, c1_point);
            let on_cone_p1 = ParametricSurface::evaluate(cone, u_p, v_p);
            assert!(
                (on_cone_p1 - c1_point).length() < 1e-9,
                "({reverse_s1}, {reverse_s2}): emitted contact1 must lie on chamfer cone"
            );
            let (u_q, v_q) = ParametricSurface::project_point(cone, c2_point);
            let on_cone_p2 = ParametricSurface::evaluate(cone, u_q, v_q);
            assert!(
                (on_cone_p2 - c2_point).length() < 1e-9,
                "({reverse_s1}, {reverse_s2}): emitted contact2 must lie on chamfer cone"
            );
        }
    };

    run_case(false, true); // (s1=+1, s2=-1)
    run_case(true, false); // (s1=-1, s2=+1)
}

/// Sphere-cylinder both-concave chamfer: spherical cavity + cylindrical
/// hole-tool. Both `s_sph = s_cyl = −1` flip the meridian arms relative
/// to convex.
///
/// For R_s=3, r_c=2, +z spine, both faces REVERSED, d=0.4:
///   - r_sph = 2·cos δ + h_s·sin δ ≈ 2.280 (sphere goes AWAY from
///     spine on the +radial side, opposite the convex case)
///   - z_sph ≈ 1.950 (BELOW spine, opposite z_sph_convex ≈ 2.482)
///   - z_cyl = a_spine + d ≈ 2.636 (ABOVE spine, opposite convex)
///   - Apex z ≈ 7.54 (still above contacts, but at different position)
#[test]
fn sphere_cylinder_chamfer_both_concave_emits_cone() {
    use brepkit_math::curves::Circle3D;
    use brepkit_math::surfaces::{CylindricalSurface, SphericalSurface};
    use brepkit_topology::edge::{Edge, EdgeCurve};
    use brepkit_topology::face::Face;
    use brepkit_topology::vertex::Vertex;
    use brepkit_topology::wire::{OrientedEdge, Wire};

    let mut topo = Topology::new();
    let big_r_s: f64 = 3.0;
    let r_c: f64 = 2.0;
    let d: f64 = 0.4;
    let h_s = (big_r_s * big_r_s - r_c * r_c).sqrt();

    let sph = SphericalSurface::new(Point3::new(0.0, 0.0, 0.0), big_r_s).unwrap();
    let cyl =
        CylindricalSurface::new(Point3::new(0.0, 0.0, 0.0), Vec3::new(0.0, 0.0, 1.0), r_c).unwrap();
    let spine_circle =
        Circle3D::new(Point3::new(0.0, 0.0, h_s), Vec3::new(0.0, 0.0, 1.0), r_c).unwrap();
    let v = topo.add_vertex(Vertex::new(Point3::new(r_c, 0.0, h_s), 1e-7));
    let eid = topo.add_edge(Edge::new(v, v, EdgeCurve::Circle(spine_circle)));
    let spine = Spine::from_single_edge(&topo, eid).unwrap();

    let w1 = topo.add_wire(Wire::new(vec![OrientedEdge::new(eid, true)], true).unwrap());
    let face_sphere = topo.add_face(Face::new_reversed(
        w1,
        vec![],
        FaceSurface::Sphere(sph.clone()),
    ));
    let w2 = topo.add_wire(Wire::new(vec![OrientedEdge::new(eid, false)], true).unwrap());
    let face_cyl = topo.add_face(Face::new_reversed(
        w2,
        vec![],
        FaceSurface::Cylinder(cyl.clone()),
    ));

    let result = sphere_cylinder_chamfer(&sph, &cyl, &spine, &topo, d, d, face_sphere, face_cyl)
        .unwrap()
        .expect("both-concave sphere-cylinder chamfer should produce a stripe");

    let chamfer_cone = match result.stripe.surface {
        FaceSurface::Cone(c) => c,
        other => panic!("expected Cone, got {}", other.type_tag()),
    };

    // Predicted contacts with s_sph = s_cyl = -1.
    let delta = d / big_r_s;
    let (sin_d, cos_d) = delta.sin_cos();
    let r_sph_pred = r_c * cos_d + h_s * sin_d; // s_sph=-1 flips sphere arm
    let z_sph_pred = h_s * cos_d - r_c * sin_d; // s_sph=-1 flips axial offset
    let z_cyl_pred = h_s + d; // a_spine + d (s_cyl=-1 reverses cyl direction)

    // Concave sphere contact moved to OPPOSITE arm: now r > r_c (above spine
    // radially) AND z < spine_z (below spine, toward cyl side).
    assert!(
        r_sph_pred > r_c,
        "concave sphere contact should have r > r_c (opposite convex case): got {r_sph_pred}"
    );
    assert!(
        z_sph_pred < h_s,
        "concave sphere contact should have z < h_s (toward cyl side): got {z_sph_pred} vs h_s={h_s}"
    );
    assert!(
        z_cyl_pred > h_s,
        "concave cyl contact should have z > h_s (opposite the convex direction): got {z_cyl_pred} vs h_s={h_s}"
    );

    // Both contacts on chamfer cone.
    let want_sph = Point3::new(r_sph_pred, 0.0, z_sph_pred);
    let want_cyl = Point3::new(r_c, 0.0, z_cyl_pred);
    let (u_p, v_p) = ParametricSurface::project_point(&chamfer_cone, want_sph);
    let on_cone_sph = ParametricSurface::evaluate(&chamfer_cone, u_p, v_p);
    let (u_q, v_q) = ParametricSurface::project_point(&chamfer_cone, want_cyl);
    let on_cone_cyl = ParametricSurface::evaluate(&chamfer_cone, u_q, v_q);
    assert!(
        (on_cone_sph - want_sph).length() < 1e-9,
        "concave sphere contact must lie on chamfer cone: {on_cone_sph:?} vs {want_sph:?}"
    );
    assert!(
        (on_cone_cyl - want_cyl).length() < 1e-9,
        "concave cyl contact must lie on chamfer cone: {on_cone_cyl:?} vs {want_cyl:?}"
    );

    // Sphere contact at distance R_s from sphere center.
    let dist_sph = (want_sph - Point3::new(0.0, 0.0, 0.0)).length();
    assert!(
        (dist_sph - big_r_s).abs() < 1e-9,
        "sphere contact must lie on sphere: distance={dist_sph}, want R_s={big_r_s}"
    );
}

/// Sphere-cylinder mixed chamfer: sphere convex (s_sph=+1) + cyl
/// concave (s_cyl=−1). Sphere contact lies on the AWAY-from-cyl
/// cap (like convex-convex), cyl contact moves OPPOSITE direction
/// from spine (like both-concave). Apex ends up BELOW the contacts
/// (axis = +z), a distinct topological configuration from the
/// symmetric cases where apex is always above.
#[test]
fn sphere_cylinder_chamfer_mixed_emits_cone() {
    use brepkit_math::curves::Circle3D;
    use brepkit_math::surfaces::{CylindricalSurface, SphericalSurface};
    use brepkit_topology::edge::{Edge, EdgeCurve};
    use brepkit_topology::face::Face;
    use brepkit_topology::vertex::Vertex;
    use brepkit_topology::wire::{OrientedEdge, Wire};

    let mut topo = Topology::new();
    let big_r_s: f64 = 3.0;
    let r_c: f64 = 2.0;
    let d: f64 = 0.4;
    let h_s = (big_r_s * big_r_s - r_c * r_c).sqrt();

    let sph = SphericalSurface::new(Point3::new(0.0, 0.0, 0.0), big_r_s).unwrap();
    let cyl =
        CylindricalSurface::new(Point3::new(0.0, 0.0, 0.0), Vec3::new(0.0, 0.0, 1.0), r_c).unwrap();
    let spine_circle =
        Circle3D::new(Point3::new(0.0, 0.0, h_s), Vec3::new(0.0, 0.0, 1.0), r_c).unwrap();
    let v = topo.add_vertex(Vertex::new(Point3::new(r_c, 0.0, h_s), 1e-7));
    let eid = topo.add_edge(Edge::new(v, v, EdgeCurve::Circle(spine_circle)));
    let spine = Spine::from_single_edge(&topo, eid).unwrap();

    // Sphere convex (NOT reversed), cyl concave (REVERSED).
    let w1 = topo.add_wire(Wire::new(vec![OrientedEdge::new(eid, true)], true).unwrap());
    let face_sphere = topo.add_face(Face::new(w1, vec![], FaceSurface::Sphere(sph.clone())));
    let w2 = topo.add_wire(Wire::new(vec![OrientedEdge::new(eid, false)], true).unwrap());
    let face_cyl = topo.add_face(Face::new_reversed(
        w2,
        vec![],
        FaceSurface::Cylinder(cyl.clone()),
    ));

    let result = sphere_cylinder_chamfer(&sph, &cyl, &spine, &topo, d, d, face_sphere, face_cyl)
        .unwrap()
        .expect("mixed sphere-cylinder chamfer should produce a stripe");

    let chamfer_cone = match result.stripe.surface {
        FaceSurface::Cone(c) => c,
        other => panic!("expected Cone, got {}", other.type_tag()),
    };

    // Predicted contacts (s_sph=+1, s_cyl=-1).
    let delta = d / big_r_s;
    let (sin_d, cos_d) = delta.sin_cos();
    let r_sph_pred = r_c * cos_d - h_s * sin_d; // sphere convex arm
    let z_sph_pred = h_s * cos_d + r_c * sin_d;
    let z_cyl_pred = h_s + d; // cyl concave goes opposite

    let dr = r_c - r_sph_pred;
    let dz = z_cyl_pred - z_sph_pred;
    let expected_apex_z = z_sph_pred - r_sph_pred * dz / dr;

    // Mixed apex BELOW both contacts ⇒ axis = +z.
    assert!(
        expected_apex_z < z_sph_pred && expected_apex_z < z_cyl_pred,
        "mixed apex should be below both contacts, got apex_z={expected_apex_z}, \
             z_sph={z_sph_pred}, z_cyl={z_cyl_pred}"
    );
    let axis = chamfer_cone.axis();
    assert!(
        axis.dot(Vec3::new(0.0, 0.0, 1.0)) > 1.0 - 1e-12,
        "mixed chamfer cone axis should be +z (apex below contacts), got {axis:?}"
    );

    // Apex on z-axis at predicted position.
    let apex = chamfer_cone.apex();
    assert!(
        apex.x().abs() < 1e-12 && apex.y().abs() < 1e-12,
        "apex should be on z-axis, got {apex:?}"
    );
    assert!(
        (apex.z() - expected_apex_z).abs() < 1e-9,
        "apex z = {}, expected {expected_apex_z}",
        apex.z()
    );

    // Both contacts on the chamfer cone.
    let want_sph = Point3::new(r_sph_pred, 0.0, z_sph_pred);
    let want_cyl = Point3::new(r_c, 0.0, z_cyl_pred);
    let (u_p, v_p) = ParametricSurface::project_point(&chamfer_cone, want_sph);
    let on_cone_sph = ParametricSurface::evaluate(&chamfer_cone, u_p, v_p);
    let (u_q, v_q) = ParametricSurface::project_point(&chamfer_cone, want_cyl);
    let on_cone_cyl = ParametricSurface::evaluate(&chamfer_cone, u_q, v_q);
    assert!(
        (on_cone_sph - want_sph).length() < 1e-9,
        "mixed sphere contact must lie on chamfer cone: {on_cone_sph:?} vs {want_sph:?}"
    );
    assert!(
        (on_cone_cyl - want_cyl).length() < 1e-9,
        "mixed cyl contact must lie on chamfer cone: {on_cone_cyl:?} vs {want_cyl:?}"
    );
}

/// `try_analytic_chamfer` with (Cylinder, Sphere) ordering: the
/// dispatcher must swap d1/d2 + face1/face2 then `swap_stripe_sides`
/// so the caller-facing Stripe is consistent with the original
/// surface ordering. Confirms the swap path is wired correctly.
#[test]
fn try_analytic_chamfer_cylinder_sphere_dispatch_swaps_correctly() {
    use brepkit_math::curves::Circle3D;
    use brepkit_math::surfaces::{CylindricalSurface, SphericalSurface};
    use brepkit_topology::edge::{Edge, EdgeCurve};
    use brepkit_topology::face::Face;
    use brepkit_topology::vertex::Vertex;
    use brepkit_topology::wire::{OrientedEdge, Wire};

    let mut topo = Topology::new();
    let big_r_s: f64 = 3.0;
    let r_c: f64 = 2.0;
    let d1_outer: f64 = 0.5; // distance on cylinder (the FIRST surface here)
    let d2_outer: f64 = 0.4; // distance on sphere (the SECOND surface here)
    let h_s = (big_r_s * big_r_s - r_c * r_c).sqrt();

    let sph = SphericalSurface::new(Point3::new(0.0, 0.0, 0.0), big_r_s).unwrap();
    let cyl =
        CylindricalSurface::new(Point3::new(0.0, 0.0, 0.0), Vec3::new(0.0, 0.0, 1.0), r_c).unwrap();

    let spine_circle =
        Circle3D::new(Point3::new(0.0, 0.0, h_s), Vec3::new(0.0, 0.0, 1.0), r_c).unwrap();
    let v = topo.add_vertex(Vertex::new(Point3::new(r_c, 0.0, h_s), 1e-7));
    let eid = topo.add_edge(Edge::new(v, v, EdgeCurve::Circle(spine_circle)));
    let spine = Spine::from_single_edge(&topo, eid).unwrap();

    // Note the ordering: cylinder is FIRST, sphere is SECOND.
    let w_cyl = topo.add_wire(Wire::new(vec![OrientedEdge::new(eid, true)], true).unwrap());
    let face1_cyl = topo.add_face(Face::new(w_cyl, vec![], FaceSurface::Cylinder(cyl.clone())));
    let w_sph = topo.add_wire(Wire::new(vec![OrientedEdge::new(eid, false)], true).unwrap());
    let face2_sph = topo.add_face(Face::new(w_sph, vec![], FaceSurface::Sphere(sph.clone())));

    // Surface order: surf1 = cylinder, surf2 = sphere.
    let surf1 = FaceSurface::Cylinder(cyl);
    let surf2 = FaceSurface::Sphere(sph);
    let result = try_analytic_chamfer(
        &surf1, &surf2, &spine, &topo, d1_outer, d2_outer, face1_cyl, face2_sph,
    )
    .unwrap()
    .expect("dispatcher should produce a stripe for (cyl, sphere) chamfer");

    // After swap, the stripe's face1 should be the cylinder face (the
    // dispatcher's face1) and face2 should be the sphere face. The
    // direct (Sphere, Cylinder) call would have face1 = sphere; the
    // swap_stripe_sides flip restores the original ordering.
    assert_eq!(
        result.stripe.face1, face1_cyl,
        "stripe.face1 should match the dispatcher's first face (cylinder), \
             confirming swap_stripe_sides restored the caller-facing ordering"
    );
    assert_eq!(
        result.stripe.face2, face2_sph,
        "stripe.face2 should match the dispatcher's second face (sphere)"
    );

    // The d1/d2 swap means the cylinder gets `d1_outer` and the
    // sphere gets `d2_outer`. We can verify by predicting the
    // contacts and confirming they match: cyl axial offset from
    // spine = d1_outer (going INTO cyl).
    let z_cyl_pred = h_s - d1_outer;
    // sphere geodesic δ = d2_outer / R_s
    let delta = d2_outer / big_r_s;
    let (sin_d, cos_d) = delta.sin_cos();
    let r_sph_pred = r_c * cos_d - h_s * sin_d;
    let z_sph_pred = h_s * cos_d + r_c * sin_d;

    let chamfer_cone = match result.stripe.surface {
        FaceSurface::Cone(c) => c,
        other => panic!("expected Cone, got {}", other.type_tag()),
    };
    let want_cyl = Point3::new(r_c, 0.0, z_cyl_pred);
    let want_sph = Point3::new(r_sph_pred, 0.0, z_sph_pred);
    let (u_p, v_p) = ParametricSurface::project_point(&chamfer_cone, want_cyl);
    let on_cone_cyl = ParametricSurface::evaluate(&chamfer_cone, u_p, v_p);
    let (u_q, v_q) = ParametricSurface::project_point(&chamfer_cone, want_sph);
    let on_cone_sph = ParametricSurface::evaluate(&chamfer_cone, u_q, v_q);
    assert!(
        (on_cone_cyl - want_cyl).length() < 1e-9,
        "cylinder contact (using dispatcher's d1) must lie on cone: {on_cone_cyl:?} vs {want_cyl:?}"
    );
    assert!(
        (on_cone_sph - want_sph).length() < 1e-9,
        "sphere contact (using dispatcher's d2) must lie on cone: {on_cone_sph:?} vs {want_sph:?}"
    );
}

/// Sphere-cone convex fillet: a sphere centered on the cone axis,
/// fillet around one of the two sphere-cone intersection circles.
///
/// For sphere at origin (R_s=3), cone apex at (0,0,−2) with axis +z
/// and half-angle π/3, both faces NOT reversed, r=0.3:
///   - h_signed = +2 (sphere center is 2 units above apex along axis)
///   - β = π/3, cos β = 0.5, sin β = √3/2
///   - Spine z (from sphere center) = (−4 + √384)/8 ≈ 1.949 (the +z spine)
///   - Spine radial (z+h)·cot β ≈ 2.279
///   - A = r + h·cos β = 0.3 + 1.0 = 1.3
///   - c_root = −A·cos β + sin β·√((R_s+r)²−A²) ≈ 1.977 (matches +z spine sign)
///   - R_t = (r + (c+h)·cos β)/sin β ≈ 2.642
#[test]
fn sphere_cone_fillet_convex_emits_torus() {
    use brepkit_math::curves::Circle3D;
    use brepkit_math::surfaces::{ConicalSurface, SphericalSurface};
    use brepkit_topology::edge::{Edge, EdgeCurve};
    use brepkit_topology::face::Face;
    use brepkit_topology::vertex::Vertex;
    use brepkit_topology::wire::{OrientedEdge, Wire};

    let mut topo = Topology::new();
    let big_r_s: f64 = 3.0;
    let h_signed: f64 = 2.0; // apex 2 units below sphere center
    let beta: f64 = std::f64::consts::PI / 3.0;
    let r_fillet: f64 = 0.3;

    // Spine z (from sphere center) on the +z side.
    let cot_b = beta.cos() / beta.sin();
    let qa = 1.0 / (beta.sin() * beta.sin());
    let qb = 2.0 * h_signed * cot_b * cot_b;
    let qc = h_signed * h_signed * cot_b * cot_b - big_r_s * big_r_s;
    let q_disc = qb * qb - 4.0 * qa * qc;
    let z_spine = (-qb + q_disc.sqrt()) / (2.0 * qa);
    let r_spine = (z_spine + h_signed) * cot_b;

    let sph = SphericalSurface::new(Point3::new(0.0, 0.0, 0.0), big_r_s).unwrap();
    let cone = ConicalSurface::new(
        Point3::new(0.0, 0.0, -h_signed),
        Vec3::new(0.0, 0.0, 1.0),
        beta,
    )
    .unwrap();

    let spine_circle = Circle3D::new(
        Point3::new(0.0, 0.0, z_spine),
        Vec3::new(0.0, 0.0, 1.0),
        r_spine,
    )
    .unwrap();
    let v = topo.add_vertex(Vertex::new(Point3::new(r_spine, 0.0, z_spine), 1e-7));
    let eid = topo.add_edge(Edge::new(v, v, EdgeCurve::Circle(spine_circle)));
    let spine = Spine::from_single_edge(&topo, eid).unwrap();

    let w1 = topo.add_wire(Wire::new(vec![OrientedEdge::new(eid, true)], true).unwrap());
    let face_sphere = topo.add_face(Face::new(w1, vec![], FaceSurface::Sphere(sph.clone())));
    let w2 = topo.add_wire(Wire::new(vec![OrientedEdge::new(eid, false)], true).unwrap());
    let face_cone = topo.add_face(Face::new(w2, vec![], FaceSurface::Cone(cone.clone())));

    let result = sphere_cone_fillet(&sph, &cone, &spine, &topo, r_fillet, face_sphere, face_cone)
        .unwrap()
        .expect("convex sphere-cone fillet should produce a stripe");

    let torus = match result.stripe.surface {
        FaceSurface::Torus(t) => t,
        other => panic!("expected Torus, got {}", other.type_tag()),
    };

    // Predicted torus parameters.
    let big_a = r_fillet + h_signed * beta.cos();
    let disc = (big_r_s + r_fillet) * (big_r_s + r_fillet) - big_a * big_a;
    let expected_z_b = -big_a * beta.cos() + beta.sin() * disc.sqrt();
    let expected_major = (r_fillet + (expected_z_b + h_signed) * beta.cos()) / beta.sin();

    assert!(
        (torus.major_radius() - expected_major).abs() < 1e-9,
        "major should be (r + (c+h)·cos β)/sin β = {expected_major}, got {}",
        torus.major_radius()
    );
    assert!(
        (torus.minor_radius() - r_fillet).abs() < 1e-12,
        "minor should be r = {r_fillet}, got {}",
        torus.minor_radius()
    );

    // Torus center on +z axis at z = expected_z_b.
    let center = torus.center();
    assert!(
        center.x().abs() < 1e-12 && center.y().abs() < 1e-12,
        "torus center should be on z-axis, got {center:?}"
    );
    assert!(
        (center.z() - expected_z_b).abs() < 1e-9,
        "torus center z should be c_root = {expected_z_b}, got {}",
        center.z()
    );

    // Predicted contacts.
    let sph_axial = big_r_s * expected_z_b / (big_r_s + r_fillet);
    let sph_radial = big_r_s * expected_major / (big_r_s + r_fillet);
    let want_sph = Point3::new(sph_radial, 0.0, sph_axial);
    let cone_axial = expected_z_b + r_fillet * beta.cos();
    let cone_radial = expected_major - r_fillet * beta.sin();
    let want_cone = Point3::new(cone_radial, 0.0, cone_axial);

    // Verify both contacts lie on the torus.
    let (u_p, v_p) = ParametricSurface::project_point(&torus, want_sph);
    let on_torus_sph = ParametricSurface::evaluate(&torus, u_p, v_p);
    let (u_q, v_q) = ParametricSurface::project_point(&torus, want_cone);
    let on_torus_cone = ParametricSurface::evaluate(&torus, u_q, v_q);
    assert!(
        (on_torus_sph - want_sph).length() < 1e-9,
        "sphere contact must lie on torus: {on_torus_sph:?} vs {want_sph:?}"
    );
    assert!(
        (on_torus_cone - want_cone).length() < 1e-9,
        "cone contact must lie on torus: {on_torus_cone:?} vs {want_cone:?}"
    );

    // Sphere contact on sphere.
    let dist_sph = (want_sph - Point3::new(0.0, 0.0, 0.0)).length();
    assert!(
        (dist_sph - big_r_s).abs() < 1e-9,
        "sphere contact must lie on sphere: distance={dist_sph}, want R_s={big_r_s}"
    );

    // Cone contact on cone: r = (z + h_signed) · cot β.
    let predicted_cone_radial = (cone_axial + h_signed) * cot_b;
    assert!(
        (cone_radial - predicted_cone_radial).abs() < 1e-9,
        "cone contact must lie on cone surface: predicted radial {predicted_cone_radial}, got {cone_radial}"
    );
}

/// Sphere-cone fillet with cone face REVERSED (sphere convex, cone
/// concave) — sphere fitting into a conical cavity. With s_sph=+1,
/// s_cone=−1 the geometry uses internal cone tangency.
///
/// For R_s=3, h_signed=+2, β=π/3, sphere face NOT reversed, cone
/// face REVERSED, r=0.3:
///   - Q_s = 3.3, A = s_cone·r + h·cos β = −0.3 + 1.0 = 0.7
///   - disc = Q_s² − A² = 10.4, sqrt ≈ 3.225
///   - c_root_a = −A·cos β + sin β·sqrt ≈ 2.443 (closest to +z spine)
///   - R_t = (s_cone·r + (c+h)·cos β)/sin β ≈ 2.219
///   - Compare convex case: R_t was ≈ 2.642, so concave-cone gives
///     SMALLER major (consistent with the rolling ball being inside
///     the cone region instead of outside).
#[test]
fn sphere_cone_fillet_concave_cone_emits_smaller_torus() {
    use brepkit_math::curves::Circle3D;
    use brepkit_math::surfaces::{ConicalSurface, SphericalSurface};
    use brepkit_topology::edge::{Edge, EdgeCurve};
    use brepkit_topology::face::Face;
    use brepkit_topology::vertex::Vertex;
    use brepkit_topology::wire::{OrientedEdge, Wire};

    let mut topo = Topology::new();
    let big_r_s: f64 = 3.0;
    let h_signed: f64 = 2.0;
    let beta: f64 = std::f64::consts::PI / 3.0;
    let r_fillet: f64 = 0.3;

    let cot_b = beta.cos() / beta.sin();
    let qa_q = 1.0 / (beta.sin() * beta.sin());
    let qb_q = 2.0 * h_signed * cot_b * cot_b;
    let qc_q = h_signed * h_signed * cot_b * cot_b - big_r_s * big_r_s;
    let q_disc = qb_q * qb_q - 4.0 * qa_q * qc_q;
    let z_spine = (-qb_q + q_disc.sqrt()) / (2.0 * qa_q);
    let r_spine = (z_spine + h_signed) * cot_b;

    let sph = SphericalSurface::new(Point3::new(0.0, 0.0, 0.0), big_r_s).unwrap();
    let cone = ConicalSurface::new(
        Point3::new(0.0, 0.0, -h_signed),
        Vec3::new(0.0, 0.0, 1.0),
        beta,
    )
    .unwrap();

    let spine_circle = Circle3D::new(
        Point3::new(0.0, 0.0, z_spine),
        Vec3::new(0.0, 0.0, 1.0),
        r_spine,
    )
    .unwrap();
    let v = topo.add_vertex(Vertex::new(Point3::new(r_spine, 0.0, z_spine), 1e-7));
    let eid = topo.add_edge(Edge::new(v, v, EdgeCurve::Circle(spine_circle)));
    let spine = Spine::from_single_edge(&topo, eid).unwrap();

    // Sphere face NOT reversed (convex post); cone face REVERSED
    // (cone-shaped cavity).
    let w1 = topo.add_wire(Wire::new(vec![OrientedEdge::new(eid, true)], true).unwrap());
    let face_sphere = topo.add_face(Face::new(w1, vec![], FaceSurface::Sphere(sph.clone())));
    let w2 = topo.add_wire(Wire::new(vec![OrientedEdge::new(eid, false)], true).unwrap());
    let face_cone = topo.add_face(Face::new_reversed(
        w2,
        vec![],
        FaceSurface::Cone(cone.clone()),
    ));

    let result = sphere_cone_fillet(&sph, &cone, &spine, &topo, r_fillet, face_sphere, face_cone)
        .unwrap()
        .expect("mixed sphere-cone fillet should produce a stripe");

    let torus = match result.stripe.surface {
        FaceSurface::Torus(t) => t,
        other => panic!("expected Torus, got {}", other.type_tag()),
    };

    // Predicted torus parameters with s_sph=+1, s_cone=-1.
    let q_s = big_r_s + r_fillet; // s_sph = +1
    let big_a_pred = -r_fillet + h_signed * beta.cos(); // s_cone = -1
    let disc = q_s * q_s - big_a_pred * big_a_pred;
    let c_root_a = -big_a_pred * beta.cos() + beta.sin() * disc.sqrt();
    let c_root_b = -big_a_pred * beta.cos() - beta.sin() * disc.sqrt();
    let z_b = if (c_root_a - z_spine).abs() <= (c_root_b - z_spine).abs() {
        c_root_a
    } else {
        c_root_b
    };
    let expected_major = (-r_fillet + (z_b + h_signed) * beta.cos()) / beta.sin();

    assert!(
        (torus.major_radius() - expected_major).abs() < 1e-9,
        "concave-cone major should be {expected_major}, got {}",
        torus.major_radius()
    );

    // Sanity: concave-cone major < convex-cone major at same r.
    let big_a_convex = r_fillet + h_signed * beta.cos();
    let disc_convex = (big_r_s + r_fillet) * (big_r_s + r_fillet) - big_a_convex * big_a_convex;
    let c_convex = -big_a_convex * beta.cos() + beta.sin() * disc_convex.sqrt();
    let convex_major = (r_fillet + (c_convex + h_signed) * beta.cos()) / beta.sin();
    assert!(
        torus.major_radius() < convex_major,
        "concave-cone major ({}) should be smaller than convex ({convex_major})",
        torus.major_radius()
    );

    // Sphere contact at distance R_s from sphere center.
    let sph_axial = big_r_s * z_b / q_s;
    let sph_radial = big_r_s * expected_major / q_s;
    let want_sph = Point3::new(sph_radial, 0.0, sph_axial);
    let dist_sph = (want_sph - Point3::new(0.0, 0.0, 0.0)).length();
    assert!(
        (dist_sph - big_r_s).abs() < 1e-9,
        "sphere contact must lie on sphere: distance={dist_sph}, want R_s={big_r_s}"
    );

    // Cone contact at the predicted (axial, radial) — should lie on cone.
    // s_cone = -1 ⇒ axial offset NEGATIVE, radial offset POSITIVE.
    let cone_axial = z_b - r_fillet * beta.cos();
    let cone_radial = expected_major + r_fillet * beta.sin();
    let predicted_cone_radial = (cone_axial + h_signed) * cot_b;
    assert!(
        (cone_radial - predicted_cone_radial).abs() < 1e-9,
        "cone contact must lie on cone: predicted radial {predicted_cone_radial}, got {cone_radial}"
    );

    // Both contacts on the torus.
    let want_cone = Point3::new(cone_radial, 0.0, cone_axial);
    let (u_p, v_p) = ParametricSurface::project_point(&torus, want_sph);
    let on_torus_sph = ParametricSurface::evaluate(&torus, u_p, v_p);
    let (u_q, v_q) = ParametricSurface::project_point(&torus, want_cone);
    let on_torus_cone = ParametricSurface::evaluate(&torus, u_q, v_q);
    assert!(
        (on_torus_sph - want_sph).length() < 1e-9,
        "sphere contact must lie on torus: {on_torus_sph:?} vs {want_sph:?}"
    );
    assert!(
        (on_torus_cone - want_cone).length() < 1e-9,
        "cone contact must lie on torus: {on_torus_cone:?} vs {want_cone:?}"
    );
}

/// Sphere-cone fillet with BOTH faces reversed (sphere cavity inside
/// a cone cavity). s_sph = s_cone = −1 so Q_s = R_s − r AND
/// A = −r + h·cos β; both flips are independent and the test below
/// pins down the (concave-sphere, concave-cone) branch that's
/// distinct from the previously-tested (convex, concave-cone) case.
///
/// For R_s=3, h=2, β=π/3, both faces REVERSED, r=0.3:
///   - Q_s = 2.7, A = 0.7
///   - disc = Q_s² − A² = 6.80, sqrt ≈ 2.608
///   - c_root closer to +z spine (z_spine ≈ 1.949) ≈ 1.908
///   - R_t = (s_cone·r + (c+h)·cos β)/sin β ≈ 1.910
///     (smaller than convex-convex 2.642 AND smaller than
///     concave-cone-only 2.219 — confirms BOTH flips compose)
#[test]
fn sphere_cone_fillet_both_concave_emits_smaller_torus() {
    use brepkit_math::curves::Circle3D;
    use brepkit_math::surfaces::{ConicalSurface, SphericalSurface};
    use brepkit_topology::edge::{Edge, EdgeCurve};
    use brepkit_topology::face::Face;
    use brepkit_topology::vertex::Vertex;
    use brepkit_topology::wire::{OrientedEdge, Wire};

    let mut topo = Topology::new();
    let big_r_s: f64 = 3.0;
    let h_signed: f64 = 2.0;
    let beta: f64 = std::f64::consts::PI / 3.0;
    let r_fillet: f64 = 0.3;

    let cot_b = beta.cos() / beta.sin();
    let qa_q = 1.0 / (beta.sin() * beta.sin());
    let qb_q = 2.0 * h_signed * cot_b * cot_b;
    let qc_q = h_signed * h_signed * cot_b * cot_b - big_r_s * big_r_s;
    let q_disc = qb_q * qb_q - 4.0 * qa_q * qc_q;
    let z_spine = (-qb_q + q_disc.sqrt()) / (2.0 * qa_q);
    let r_spine = (z_spine + h_signed) * cot_b;

    let sph = SphericalSurface::new(Point3::new(0.0, 0.0, 0.0), big_r_s).unwrap();
    let cone = ConicalSurface::new(
        Point3::new(0.0, 0.0, -h_signed),
        Vec3::new(0.0, 0.0, 1.0),
        beta,
    )
    .unwrap();

    let spine_circle = Circle3D::new(
        Point3::new(0.0, 0.0, z_spine),
        Vec3::new(0.0, 0.0, 1.0),
        r_spine,
    )
    .unwrap();
    let v = topo.add_vertex(Vertex::new(Point3::new(r_spine, 0.0, z_spine), 1e-7));
    let eid = topo.add_edge(Edge::new(v, v, EdgeCurve::Circle(spine_circle)));
    let spine = Spine::from_single_edge(&topo, eid).unwrap();

    // Both faces REVERSED (sphere cavity meets cone cavity).
    let w1 = topo.add_wire(Wire::new(vec![OrientedEdge::new(eid, true)], true).unwrap());
    let face_sphere = topo.add_face(Face::new_reversed(
        w1,
        vec![],
        FaceSurface::Sphere(sph.clone()),
    ));
    let w2 = topo.add_wire(Wire::new(vec![OrientedEdge::new(eid, false)], true).unwrap());
    let face_cone = topo.add_face(Face::new_reversed(
        w2,
        vec![],
        FaceSurface::Cone(cone.clone()),
    ));

    let result = sphere_cone_fillet(&sph, &cone, &spine, &topo, r_fillet, face_sphere, face_cone)
        .unwrap()
        .expect("both-concave sphere-cone fillet should produce a stripe");

    let torus = match result.stripe.surface {
        FaceSurface::Torus(t) => t,
        other => panic!("expected Torus, got {}", other.type_tag()),
    };

    // Predicted torus parameters with s_sph = s_cone = -1.
    let q_s = big_r_s - r_fillet;
    let big_a = -r_fillet + h_signed * beta.cos();
    let disc = q_s * q_s - big_a * big_a;
    let c_root_a = -big_a * beta.cos() + beta.sin() * disc.sqrt();
    let c_root_b = -big_a * beta.cos() - beta.sin() * disc.sqrt();
    let z_b = if (c_root_a - z_spine).abs() <= (c_root_b - z_spine).abs() {
        c_root_a
    } else {
        c_root_b
    };
    let expected_major = (-r_fillet + (z_b + h_signed) * beta.cos()) / beta.sin();

    assert!(
        (torus.major_radius() - expected_major).abs() < 1e-9,
        "both-concave major should be {expected_major}, got {}",
        torus.major_radius()
    );

    // Sanity: both-concave major < convex-convex major < ALSO <
    // concave-cone-only — i.e. the two flips compose to give the
    // smallest torus.
    let convex_a = r_fillet + h_signed * beta.cos();
    let convex_disc = (big_r_s + r_fillet) * (big_r_s + r_fillet) - convex_a * convex_a;
    let convex_c = -convex_a * beta.cos() + beta.sin() * convex_disc.sqrt();
    let convex_major = (r_fillet + (convex_c + h_signed) * beta.cos()) / beta.sin();
    let concave_cone_a = -r_fillet + h_signed * beta.cos();
    let concave_cone_disc =
        (big_r_s + r_fillet) * (big_r_s + r_fillet) - concave_cone_a * concave_cone_a;
    let concave_cone_c = -concave_cone_a * beta.cos() + beta.sin() * concave_cone_disc.sqrt();
    let concave_cone_major = (-r_fillet + (concave_cone_c + h_signed) * beta.cos()) / beta.sin();
    assert!(
        torus.major_radius() < concave_cone_major,
        "both-concave major ({}) should be smaller than concave-cone-only ({concave_cone_major})",
        torus.major_radius()
    );
    assert!(
        concave_cone_major < convex_major,
        "concave-cone-only major ({concave_cone_major}) should be smaller than convex ({convex_major})"
    );

    // Sphere contact at distance R_s.
    let sph_axial = big_r_s * z_b / q_s;
    let sph_radial = big_r_s * expected_major / q_s;
    let want_sph = Point3::new(sph_radial, 0.0, sph_axial);
    let dist_sph = (want_sph - Point3::new(0.0, 0.0, 0.0)).length();
    assert!(
        (dist_sph - big_r_s).abs() < 1e-9,
        "sphere contact must lie on sphere: {dist_sph} vs R_s={big_r_s}"
    );

    // Cone contact on cone surface.
    let cone_axial = z_b - r_fillet * beta.cos();
    let cone_radial = expected_major + r_fillet * beta.sin();
    let predicted_cone_radial = (cone_axial + h_signed) * cot_b;
    assert!(
        (cone_radial - predicted_cone_radial).abs() < 1e-9,
        "cone contact must lie on cone: predicted {predicted_cone_radial}, got {cone_radial}"
    );
}

/// Sphere-cone convex chamfer: a sphere centered on the cone axis,
/// chamfer rounding the corner where they meet.
///
/// For sphere at origin (R_s=3), cone apex at (0,0,−2) with axis +z
/// and half-angle π/3, both faces NOT reversed, symmetric d=0.3:
///   - h_signed = +2, β = π/3, cot β = 1/√3
///   - Spine z (from sphere center) ≈ +1.949 (the +z spine)
///   - Spine radial r_spine = (z+h)·cot β ≈ 2.279
///   - Sphere contact: δ=0.1, sphere_arm_sign=−1,
///     r_sph = r_spine·cos δ − spine_z·sin δ ≈ 2.073,
///     z_sph = spine_z·cos δ + r_spine·sin δ ≈ 2.167
///   - Cone contact (toward apex): r=r_spine−d·cos β ≈ 2.129,
///     z=spine_z−d·sin β ≈ 1.689
///   - Δr ≈ +0.056, Δz ≈ −0.478 ⇒ apex z ≈ 19.86 (well above contacts)
#[test]
fn sphere_cone_chamfer_convex_emits_cone() {
    use brepkit_math::curves::Circle3D;
    use brepkit_math::surfaces::{ConicalSurface, SphericalSurface};
    use brepkit_topology::edge::{Edge, EdgeCurve};
    use brepkit_topology::face::Face;
    use brepkit_topology::vertex::Vertex;
    use brepkit_topology::wire::{OrientedEdge, Wire};

    let mut topo = Topology::new();
    let big_r_s: f64 = 3.0;
    let h_signed: f64 = 2.0;
    let beta: f64 = std::f64::consts::PI / 3.0;
    let d: f64 = 0.3;

    // Solve for spine z (on +z side).
    let cot_b = beta.cos() / beta.sin();
    let qa = 1.0 / (beta.sin() * beta.sin());
    let qb = 2.0 * h_signed * cot_b * cot_b;
    let qc = h_signed * h_signed * cot_b * cot_b - big_r_s * big_r_s;
    let q_disc = qb * qb - 4.0 * qa * qc;
    let z_spine = (-qb + q_disc.sqrt()) / (2.0 * qa);
    let r_spine = (z_spine + h_signed) * cot_b;

    let sph = SphericalSurface::new(Point3::new(0.0, 0.0, 0.0), big_r_s).unwrap();
    let cone = ConicalSurface::new(
        Point3::new(0.0, 0.0, -h_signed),
        Vec3::new(0.0, 0.0, 1.0),
        beta,
    )
    .unwrap();

    let spine_circle = Circle3D::new(
        Point3::new(0.0, 0.0, z_spine),
        Vec3::new(0.0, 0.0, 1.0),
        r_spine,
    )
    .unwrap();
    let v = topo.add_vertex(Vertex::new(Point3::new(r_spine, 0.0, z_spine), 1e-7));
    let eid = topo.add_edge(Edge::new(v, v, EdgeCurve::Circle(spine_circle)));
    let spine = Spine::from_single_edge(&topo, eid).unwrap();

    let w1 = topo.add_wire(Wire::new(vec![OrientedEdge::new(eid, true)], true).unwrap());
    let face_sphere = topo.add_face(Face::new(w1, vec![], FaceSurface::Sphere(sph.clone())));
    let w2 = topo.add_wire(Wire::new(vec![OrientedEdge::new(eid, false)], true).unwrap());
    let face_cone = topo.add_face(Face::new(w2, vec![], FaceSurface::Cone(cone.clone())));

    let result = sphere_cone_chamfer(&sph, &cone, &spine, &topo, d, d, face_sphere, face_cone)
        .unwrap()
        .expect("convex sphere-cone chamfer should produce a stripe");

    let chamfer_cone = match result.stripe.surface {
        FaceSurface::Cone(c) => c,
        other => panic!("expected Cone, got {}", other.type_tag()),
    };

    // Predicted contacts.
    let delta = d / big_r_s;
    let (sin_d, cos_d) = delta.sin_cos();
    let sphere_arm_sign = -1.0_f64; // spine_z > 0
    let r_sph_pred = r_spine * cos_d + sphere_arm_sign * z_spine * sin_d;
    let z_sph_pred = z_spine * cos_d - sphere_arm_sign * r_spine * sin_d;
    let r_cone_pred = r_spine - d * beta.cos();
    let z_cone_pred = z_spine - d * beta.sin();

    // Predicted apex.
    let dr = r_cone_pred - r_sph_pred;
    let dz = z_cone_pred - z_sph_pred;
    let expected_apex_z = z_sph_pred - r_sph_pred * dz / dr;
    let mid_z = 0.5 * (z_sph_pred + z_cone_pred);
    let r_avg = 0.5 * (r_sph_pred + r_cone_pred);
    let expected_beta = ((mid_z - expected_apex_z).abs() / r_avg).atan();

    assert!(
        expected_apex_z > z_sph_pred && expected_apex_z > z_cone_pred,
        "apex should be above both contacts, got apex_z={expected_apex_z}"
    );
    assert!(
        (chamfer_cone.half_angle() - expected_beta).abs() < 1e-9,
        "chamfer half-angle should be atan(|z_apex - mid_z| / r_avg) = {expected_beta}, got {}",
        chamfer_cone.half_angle()
    );

    // Apex on +z axis.
    let apex = chamfer_cone.apex();
    assert!(
        apex.x().abs() < 1e-12 && apex.y().abs() < 1e-12,
        "apex should be on z-axis, got {apex:?}"
    );
    assert!(
        (apex.z() - expected_apex_z).abs() < 1e-9,
        "apex z = {}, expected {expected_apex_z}",
        apex.z()
    );

    // Cone axis = -z (apex above contacts, opens downward).
    let axis = chamfer_cone.axis();
    assert!(
        axis.dot(Vec3::new(0.0, 0.0, 1.0)) < -1.0 + 1e-12,
        "convex chamfer cone axis should be -z, got {axis:?}"
    );

    // Both contacts on chamfer cone.
    let want_sph = Point3::new(r_sph_pred, 0.0, z_sph_pred);
    let want_cone = Point3::new(r_cone_pred, 0.0, z_cone_pred);
    let (u_p, v_p) = ParametricSurface::project_point(&chamfer_cone, want_sph);
    let on_cone_sph = ParametricSurface::evaluate(&chamfer_cone, u_p, v_p);
    let (u_q, v_q) = ParametricSurface::project_point(&chamfer_cone, want_cone);
    let on_cone_cone = ParametricSurface::evaluate(&chamfer_cone, u_q, v_q);
    assert!(
        (on_cone_sph - want_sph).length() < 1e-9,
        "sphere contact must lie on chamfer cone: {on_cone_sph:?} vs {want_sph:?}"
    );
    assert!(
        (on_cone_cone - want_cone).length() < 1e-9,
        "cone contact must lie on chamfer cone: {on_cone_cone:?} vs {want_cone:?}"
    );

    // Both contacts on their respective surfaces.
    let dist_sph = (want_sph - Point3::new(0.0, 0.0, 0.0)).length();
    assert!(
        (dist_sph - big_r_s).abs() < 1e-9,
        "sphere contact must lie on sphere: distance={dist_sph}, want R_s={big_r_s}"
    );
    let cone_predicted_radial = (z_cone_pred + h_signed) * cot_b;
    assert!(
        (r_cone_pred - cone_predicted_radial).abs() < 1e-9,
        "cone contact must lie on cone surface: predicted radial {cone_predicted_radial}, got {r_cone_pred}"
    );
}

/// Sphere-cone chamfer with BOTH faces reversed (sphere cavity meets
/// cone cavity at a concave corner). s_sph = s_cone = −1 flip both
/// meridian arms.
///
/// For R_s=3, h=2, β=π/3, both faces REVERSED, d=0.3:
///   - Sphere arm flips: contact moves to OPPOSITE cap (toward cone
///     side) ⇒ z_sph DECREASES below spine_z
///   - Cone arm flips: contact moves AWAY from apex along generator
///     ⇒ r_cone INCREASES, z_cone INCREASES (away from apex)
///   - These flips compose to give a different chamfer cone than
///     the convex case
#[test]
fn sphere_cone_chamfer_both_concave_emits_cone() {
    use brepkit_math::curves::Circle3D;
    use brepkit_math::surfaces::{ConicalSurface, SphericalSurface};
    use brepkit_topology::edge::{Edge, EdgeCurve};
    use brepkit_topology::face::Face;
    use brepkit_topology::vertex::Vertex;
    use brepkit_topology::wire::{OrientedEdge, Wire};

    let mut topo = Topology::new();
    let big_r_s: f64 = 3.0;
    let h_signed: f64 = 2.0;
    let beta: f64 = std::f64::consts::PI / 3.0;
    let d: f64 = 0.3;

    let cot_b = beta.cos() / beta.sin();
    let qa_q = 1.0 / (beta.sin() * beta.sin());
    let qb_q = 2.0 * h_signed * cot_b * cot_b;
    let qc_q = h_signed * h_signed * cot_b * cot_b - big_r_s * big_r_s;
    let q_disc = qb_q * qb_q - 4.0 * qa_q * qc_q;
    let z_spine = (-qb_q + q_disc.sqrt()) / (2.0 * qa_q);
    let r_spine = (z_spine + h_signed) * cot_b;

    let sph = SphericalSurface::new(Point3::new(0.0, 0.0, 0.0), big_r_s).unwrap();
    let cone = ConicalSurface::new(
        Point3::new(0.0, 0.0, -h_signed),
        Vec3::new(0.0, 0.0, 1.0),
        beta,
    )
    .unwrap();

    let spine_circle = Circle3D::new(
        Point3::new(0.0, 0.0, z_spine),
        Vec3::new(0.0, 0.0, 1.0),
        r_spine,
    )
    .unwrap();
    let v = topo.add_vertex(Vertex::new(Point3::new(r_spine, 0.0, z_spine), 1e-7));
    let eid = topo.add_edge(Edge::new(v, v, EdgeCurve::Circle(spine_circle)));
    let spine = Spine::from_single_edge(&topo, eid).unwrap();

    // Both faces REVERSED.
    let w1 = topo.add_wire(Wire::new(vec![OrientedEdge::new(eid, true)], true).unwrap());
    let face_sphere = topo.add_face(Face::new_reversed(
        w1,
        vec![],
        FaceSurface::Sphere(sph.clone()),
    ));
    let w2 = topo.add_wire(Wire::new(vec![OrientedEdge::new(eid, false)], true).unwrap());
    let face_cone = topo.add_face(Face::new_reversed(
        w2,
        vec![],
        FaceSurface::Cone(cone.clone()),
    ));

    let result = sphere_cone_chamfer(&sph, &cone, &spine, &topo, d, d, face_sphere, face_cone)
        .unwrap()
        .expect("both-concave sphere-cone chamfer should produce a stripe");

    let chamfer_cone = match result.stripe.surface {
        FaceSurface::Cone(c) => c,
        other => panic!("expected Cone, got {}", other.type_tag()),
    };

    // Predicted contacts with s_sph = s_cone = -1.
    // sphere_arm_sign = -spine_sign · s_sph = -1·-1 = +1 (flipped from convex).
    let delta = d / big_r_s;
    let (sin_d, cos_d) = delta.sin_cos();
    let sphere_arm_sign = 1.0_f64; // s_sph = -1, spine_sign = +1
    let r_sph_pred = r_spine * cos_d + sphere_arm_sign * z_spine * sin_d;
    let z_sph_pred = z_spine * cos_d - sphere_arm_sign * r_spine * sin_d;
    // Cone arm: s_cone = -1, so go AWAY from apex.
    let r_cone_pred = r_spine + d * beta.cos();
    let z_cone_pred = z_spine + d * beta.sin();

    // Sphere contact moved toward cone side (z DECREASED below spine).
    assert!(
        z_sph_pred < z_spine,
        "concave sphere contact should be below spine z (toward cone): got {z_sph_pred} vs spine {z_spine}"
    );
    // Cone contact moved AWAY from apex (z and r INCREASED).
    assert!(
        r_cone_pred > r_spine && z_cone_pred > z_spine,
        "concave cone contact should be away from apex: got ({r_cone_pred}, {z_cone_pred})"
    );

    // Both contacts on chamfer cone.
    let want_sph = Point3::new(r_sph_pred, 0.0, z_sph_pred);
    let want_cone = Point3::new(r_cone_pred, 0.0, z_cone_pred);
    let (u_p, v_p) = ParametricSurface::project_point(&chamfer_cone, want_sph);
    let on_cone_sph = ParametricSurface::evaluate(&chamfer_cone, u_p, v_p);
    let (u_q, v_q) = ParametricSurface::project_point(&chamfer_cone, want_cone);
    let on_cone_cone = ParametricSurface::evaluate(&chamfer_cone, u_q, v_q);
    assert!(
        (on_cone_sph - want_sph).length() < 1e-9,
        "concave sphere contact must lie on chamfer cone: {on_cone_sph:?} vs {want_sph:?}"
    );
    assert!(
        (on_cone_cone - want_cone).length() < 1e-9,
        "concave cone contact must lie on chamfer cone: {on_cone_cone:?} vs {want_cone:?}"
    );

    // Sphere contact at distance R_s.
    let dist_sph = (want_sph - Point3::new(0.0, 0.0, 0.0)).length();
    assert!(
        (dist_sph - big_r_s).abs() < 1e-9,
        "sphere contact must lie on sphere: {dist_sph} vs R_s={big_r_s}"
    );

    // Cone contact on cone surface.
    let predicted_cone_radial = (z_cone_pred + h_signed) * cot_b;
    assert!(
        (r_cone_pred - predicted_cone_radial).abs() < 1e-9,
        "cone contact must lie on cone surface: predicted radial {predicted_cone_radial}, got {r_cone_pred}"
    );
}

/// Sphere-cone mixed chamfer: sphere convex (s_sph=+1) + cone
/// concave (s_cone=−1). Sphere contact lies on the AWAY-from-cone
/// cap (like convex-convex), but cone contact moves AWAY from
/// apex along generator instead of toward it (like both-concave
/// for the cone arm).
///
/// For R_s=3, h=2, β=π/3, sphere NOT reversed, cone REVERSED, d=0.3:
///   - sphere_arm_sign = -1·+1 = -1 (convex)
///   - r_sph = r_spine·cos δ − spine_z·sin δ ≈ 2.073 (same as convex)
///   - z_sph = spine_z·cos δ + r_spine·sin δ ≈ 2.167 (above spine)
///   - cone goes AWAY from apex: r_cone = r_spine + d·cos β ≈ 2.429,
///     z_cone = spine_z + d·sin β ≈ 2.209 (above spine, toward sphere)
#[test]
fn sphere_cone_chamfer_mixed_emits_cone() {
    use brepkit_math::curves::Circle3D;
    use brepkit_math::surfaces::{ConicalSurface, SphericalSurface};
    use brepkit_topology::edge::{Edge, EdgeCurve};
    use brepkit_topology::face::Face;
    use brepkit_topology::vertex::Vertex;
    use brepkit_topology::wire::{OrientedEdge, Wire};

    let mut topo = Topology::new();
    let big_r_s: f64 = 3.0;
    let h_signed: f64 = 2.0;
    let beta: f64 = std::f64::consts::PI / 3.0;
    let d: f64 = 0.3;

    let cot_b = beta.cos() / beta.sin();
    let qa_q = 1.0 / (beta.sin() * beta.sin());
    let qb_q = 2.0 * h_signed * cot_b * cot_b;
    let qc_q = h_signed * h_signed * cot_b * cot_b - big_r_s * big_r_s;
    let q_disc = qb_q * qb_q - 4.0 * qa_q * qc_q;
    let z_spine = (-qb_q + q_disc.sqrt()) / (2.0 * qa_q);
    let r_spine = (z_spine + h_signed) * cot_b;

    let sph = SphericalSurface::new(Point3::new(0.0, 0.0, 0.0), big_r_s).unwrap();
    let cone = ConicalSurface::new(
        Point3::new(0.0, 0.0, -h_signed),
        Vec3::new(0.0, 0.0, 1.0),
        beta,
    )
    .unwrap();

    let spine_circle = Circle3D::new(
        Point3::new(0.0, 0.0, z_spine),
        Vec3::new(0.0, 0.0, 1.0),
        r_spine,
    )
    .unwrap();
    let v = topo.add_vertex(Vertex::new(Point3::new(r_spine, 0.0, z_spine), 1e-7));
    let eid = topo.add_edge(Edge::new(v, v, EdgeCurve::Circle(spine_circle)));
    let spine = Spine::from_single_edge(&topo, eid).unwrap();

    // Sphere NOT reversed, cone REVERSED.
    let w1 = topo.add_wire(Wire::new(vec![OrientedEdge::new(eid, true)], true).unwrap());
    let face_sphere = topo.add_face(Face::new(w1, vec![], FaceSurface::Sphere(sph.clone())));
    let w2 = topo.add_wire(Wire::new(vec![OrientedEdge::new(eid, false)], true).unwrap());
    let face_cone = topo.add_face(Face::new_reversed(
        w2,
        vec![],
        FaceSurface::Cone(cone.clone()),
    ));

    let result = sphere_cone_chamfer(&sph, &cone, &spine, &topo, d, d, face_sphere, face_cone)
        .unwrap()
        .expect("mixed sphere-cone chamfer should produce a stripe");

    let chamfer_cone = match result.stripe.surface {
        FaceSurface::Cone(c) => c,
        other => panic!("expected Cone, got {}", other.type_tag()),
    };

    // Predicted contacts (s_sph=+1, s_cone=-1).
    let delta = d / big_r_s;
    let (sin_d, cos_d) = delta.sin_cos();
    // sphere_arm_sign = -spine_sign · s_sph = -1 · +1 = -1 (convex sphere).
    let r_sph_pred = r_spine * cos_d - z_spine * sin_d;
    let z_sph_pred = z_spine * cos_d + r_spine * sin_d;
    // s_cone = -1 ⇒ cone goes AWAY from apex.
    let r_cone_pred = r_spine + d * beta.cos();
    let z_cone_pred = z_spine + d * beta.sin();

    // Sphere contact on natural convex arm (above spine).
    assert!(
        z_sph_pred > z_spine,
        "convex sphere contact should be above spine: got {z_sph_pred}"
    );
    // Cone contact moves AWAY from apex (away from convex direction).
    assert!(
        r_cone_pred > r_spine && z_cone_pred > z_spine,
        "concave cone contact should be away from apex: got ({r_cone_pred}, {z_cone_pred})"
    );

    // Both contacts on chamfer cone.
    let want_sph = Point3::new(r_sph_pred, 0.0, z_sph_pred);
    let want_cone = Point3::new(r_cone_pred, 0.0, z_cone_pred);
    let (u_p, v_p) = ParametricSurface::project_point(&chamfer_cone, want_sph);
    let on_cone_sph = ParametricSurface::evaluate(&chamfer_cone, u_p, v_p);
    let (u_q, v_q) = ParametricSurface::project_point(&chamfer_cone, want_cone);
    let on_cone_cone = ParametricSurface::evaluate(&chamfer_cone, u_q, v_q);
    assert!(
        (on_cone_sph - want_sph).length() < 1e-9,
        "sphere contact must lie on chamfer cone: {on_cone_sph:?} vs {want_sph:?}"
    );
    assert!(
        (on_cone_cone - want_cone).length() < 1e-9,
        "cone contact must lie on chamfer cone: {on_cone_cone:?} vs {want_cone:?}"
    );

    // Sphere contact at distance R_s.
    let dist_sph = (want_sph - Point3::new(0.0, 0.0, 0.0)).length();
    assert!(
        (dist_sph - big_r_s).abs() < 1e-9,
        "sphere contact must lie on sphere: {dist_sph} vs R_s={big_r_s}"
    );

    // Cone contact on cone surface.
    let predicted_cone_radial = (z_cone_pred + h_signed) * cot_b;
    assert!(
        (r_cone_pred - predicted_cone_radial).abs() < 1e-9,
        "cone contact must lie on cone: predicted {predicted_cone_radial}, got {r_cone_pred}"
    );
}

/// Cylinder-cylinder convex fillet for two intersecting cylinders
/// with PARALLEL axes. The intersection is two straight lines parallel
/// to the cyl axes, and the rolling-ball blend is an exact cylinder
/// around an axis parallel to those.
///
/// For cyl1 axis = +z through origin (r=2), cyl2 axis = +z at (3, 0, *)
/// (r=2.5), D=3, both faces NOT reversed, r=0.4:
///   - x_spine = (4 − 6.25 + 9)/6 = 1.125
///   - y_spine = ±√(4 − 1.265625) = ±1.654
///   - For r=0.4: Q1=2.4, Q2=2.9
///     x_ball = (Q1²−Q2²+D²)/(2D) = (5.76−8.41+9)/6 ≈ 1.058
///     y_ball = sign·√(Q1²−x_ball²) = √(5.76−1.119) ≈ 2.154
///   - Fillet cylinder axis +z at (1.058, 2.154, *), radius 0.4
#[test]
fn cylinder_cylinder_fillet_parallel_axes_emits_cylinder() {
    use brepkit_math::surfaces::CylindricalSurface;
    use brepkit_topology::edge::{Edge, EdgeCurve};
    use brepkit_topology::face::Face;
    use brepkit_topology::vertex::Vertex;
    use brepkit_topology::wire::{OrientedEdge, Wire};

    let mut topo = Topology::new();
    let r1: f64 = 2.0;
    let r2: f64 = 2.5;
    let big_d: f64 = 3.0;
    let r_fillet: f64 = 0.4;

    let cyl1 =
        CylindricalSurface::new(Point3::new(0.0, 0.0, 0.0), Vec3::new(0.0, 0.0, 1.0), r1).unwrap();
    let cyl2 = CylindricalSurface::new(Point3::new(big_d, 0.0, 0.0), Vec3::new(0.0, 0.0, 1.0), r2)
        .unwrap();

    // Spine: line at (x_spine, +y_spine, z) for z ∈ [0, 4]; segment
    // direction = +z.
    let x_spine = (r1 * r1 - r2 * r2 + big_d * big_d) / (2.0 * big_d);
    let y_spine = (r1 * r1 - x_spine * x_spine).sqrt();
    let z_lo = 0.0_f64;
    let z_hi = 4.0_f64;
    let p_start = Point3::new(x_spine, y_spine, z_lo);
    let p_end = Point3::new(x_spine, y_spine, z_hi);
    let v_start = topo.add_vertex(Vertex::new(p_start, 1e-7));
    let v_end = topo.add_vertex(Vertex::new(p_end, 1e-7));
    let line = brepkit_math::nurbs::curve::NurbsCurve::new(
        1,
        vec![0.0, 0.0, 1.0, 1.0],
        vec![p_start, p_end],
        vec![1.0, 1.0],
    )
    .unwrap();
    let eid = topo.add_edge(Edge::new(v_start, v_end, EdgeCurve::NurbsCurve(line)));
    let spine = Spine::from_single_edge(&topo, eid).unwrap();

    let w1 = topo.add_wire(Wire::new(vec![OrientedEdge::new(eid, true)], false).unwrap());
    let face1 = topo.add_face(Face::new(w1, vec![], FaceSurface::Cylinder(cyl1.clone())));
    let w2 = topo.add_wire(Wire::new(vec![OrientedEdge::new(eid, false)], false).unwrap());
    let face2 = topo.add_face(Face::new(w2, vec![], FaceSurface::Cylinder(cyl2.clone())));

    let result = cylinder_cylinder_fillet(&cyl1, &cyl2, &spine, &topo, r_fillet, face1, face2)
        .unwrap()
        .expect("parallel-axis cyl-cyl fillet should produce a stripe");

    let fillet_cyl = match result.stripe.surface {
        FaceSurface::Cylinder(c) => c,
        other => panic!("expected Cylinder, got {}", other.type_tag()),
    };

    // Predicted ball position.
    let q1 = r1 + r_fillet;
    let q2 = r2 + r_fillet;
    let x_ball = (q1 * q1 - q2 * q2 + big_d * big_d) / (2.0 * big_d);
    let y_ball = (q1 * q1 - x_ball * x_ball).sqrt();

    assert!(
        (fillet_cyl.radius() - r_fillet).abs() < 1e-12,
        "fillet cylinder radius should equal r = {r_fillet}, got {}",
        fillet_cyl.radius()
    );
    // Axis = +z (parallel to original cyls).
    let axis = fillet_cyl.axis();
    assert!(
        axis.dot(Vec3::new(0.0, 0.0, 1.0)) > 1.0 - 1e-12,
        "fillet cylinder axis should be +z, got {axis:?}"
    );
    // Origin at (x_ball, y_ball, z_lo).
    let origin = fillet_cyl.origin();
    assert!(
        (origin.x() - x_ball).abs() < 1e-12 && (origin.y() - y_ball).abs() < 1e-12,
        "fillet cylinder origin should be ({x_ball}, {y_ball}, *), got {origin:?}"
    );

    // Cyl1 contact line at (r1·x_ball/q1, r1·y_ball/q1, z).
    let want_c1 = Point3::new(r1 * x_ball / q1, r1 * y_ball / q1, z_lo);
    let dist_c1_axis = (want_c1.x().powi(2) + want_c1.y().powi(2)).sqrt();
    assert!(
        (dist_c1_axis - r1).abs() < 1e-9,
        "cyl1 contact must lie on cyl1 (radial = r1): got {dist_c1_axis}, want {r1}"
    );
    // Cyl2 contact line at (D + r2·(x_ball−D)/q2, r2·y_ball/q2, z).
    let want_c2 = Point3::new(big_d + r2 * (x_ball - big_d) / q2, r2 * y_ball / q2, z_lo);
    let dist_c2_axis = ((want_c2.x() - big_d).powi(2) + want_c2.y().powi(2)).sqrt();
    assert!(
        (dist_c2_axis - r2).abs() < 1e-9,
        "cyl2 contact must lie on cyl2 (radial from cyl2 axis = r2): got {dist_c2_axis}, want {r2}"
    );

    // Both contacts on the fillet cylinder surface (distance r from
    // ball-line in xy).
    let dist_c1_to_ball = ((want_c1.x() - x_ball).powi(2) + (want_c1.y() - y_ball).powi(2)).sqrt();
    let dist_c2_to_ball = ((want_c2.x() - x_ball).powi(2) + (want_c2.y() - y_ball).powi(2)).sqrt();
    assert!(
        (dist_c1_to_ball - r_fillet).abs() < 1e-9,
        "cyl1 contact must lie on fillet cylinder: distance from ball-line = {dist_c1_to_ball}, want r = {r_fillet}"
    );
    assert!(
        (dist_c2_to_ball - r_fillet).abs() < 1e-9,
        "cyl2 contact must lie on fillet cylinder: distance from ball-line = {dist_c2_to_ball}, want r = {r_fillet}"
    );
}

/// Cylinder-cylinder both-concave fillet: two intersecting cylindrical
/// holes with parallel axes. Both s_i = −1 ⇒ Q_i = r_i − r (internal
/// tangency), so the rolling ball is INSIDE both cylinders.
///
/// For r1=2, r2=2.5, D=3, both faces REVERSED, r=0.4:
///   - Q1 = 1.6, Q2 = 2.1
///   - x_ball = (Q1²−Q2²+D²)/(2D) = (2.56−4.41+9)/6 ≈ 1.192
///   - y_ball = sign·√(Q1²−x_ball²) = √(2.56−1.421) ≈ 1.067
///   - Both contacts internal: cyl1 contact at radial r1·x_ball/Q1 from
///     cyl1 axis (different from convex case)
#[test]
fn cylinder_cylinder_fillet_both_concave_emits_cylinder() {
    use brepkit_math::surfaces::CylindricalSurface;
    use brepkit_topology::edge::{Edge, EdgeCurve};
    use brepkit_topology::face::Face;
    use brepkit_topology::vertex::Vertex;
    use brepkit_topology::wire::{OrientedEdge, Wire};

    let mut topo = Topology::new();
    let r1: f64 = 2.0;
    let r2: f64 = 2.5;
    let big_d: f64 = 3.0;
    let r_fillet: f64 = 0.4;

    let cyl1 =
        CylindricalSurface::new(Point3::new(0.0, 0.0, 0.0), Vec3::new(0.0, 0.0, 1.0), r1).unwrap();
    let cyl2 = CylindricalSurface::new(Point3::new(big_d, 0.0, 0.0), Vec3::new(0.0, 0.0, 1.0), r2)
        .unwrap();

    let x_spine = (r1 * r1 - r2 * r2 + big_d * big_d) / (2.0 * big_d);
    let y_spine = (r1 * r1 - x_spine * x_spine).sqrt();
    let z_lo = 0.0_f64;
    let z_hi = 4.0_f64;
    let p_start = Point3::new(x_spine, y_spine, z_lo);
    let p_end = Point3::new(x_spine, y_spine, z_hi);
    let v_start = topo.add_vertex(Vertex::new(p_start, 1e-7));
    let v_end = topo.add_vertex(Vertex::new(p_end, 1e-7));
    let line = brepkit_math::nurbs::curve::NurbsCurve::new(
        1,
        vec![0.0, 0.0, 1.0, 1.0],
        vec![p_start, p_end],
        vec![1.0, 1.0],
    )
    .unwrap();
    let eid = topo.add_edge(Edge::new(v_start, v_end, EdgeCurve::NurbsCurve(line)));
    let spine = Spine::from_single_edge(&topo, eid).unwrap();

    // Both faces REVERSED.
    let w1 = topo.add_wire(Wire::new(vec![OrientedEdge::new(eid, true)], false).unwrap());
    let face1 = topo.add_face(Face::new_reversed(
        w1,
        vec![],
        FaceSurface::Cylinder(cyl1.clone()),
    ));
    let w2 = topo.add_wire(Wire::new(vec![OrientedEdge::new(eid, false)], false).unwrap());
    let face2 = topo.add_face(Face::new_reversed(
        w2,
        vec![],
        FaceSurface::Cylinder(cyl2.clone()),
    ));

    let result = cylinder_cylinder_fillet(&cyl1, &cyl2, &spine, &topo, r_fillet, face1, face2)
        .unwrap()
        .expect("both-concave cyl-cyl fillet should produce a stripe");

    let fillet_cyl = match result.stripe.surface {
        FaceSurface::Cylinder(c) => c,
        other => panic!("expected Cylinder, got {}", other.type_tag()),
    };

    let q1 = r1 - r_fillet;
    let q2 = r2 - r_fillet;
    let x_ball = (q1 * q1 - q2 * q2 + big_d * big_d) / (2.0 * big_d);
    let y_ball = (q1 * q1 - x_ball * x_ball).sqrt();

    assert!(
        (fillet_cyl.radius() - r_fillet).abs() < 1e-12,
        "fillet radius should be r = {r_fillet}, got {}",
        fillet_cyl.radius()
    );
    let origin = fillet_cyl.origin();
    assert!(
        (origin.x() - x_ball).abs() < 1e-12 && (origin.y() - y_ball).abs() < 1e-12,
        "concave fillet origin should be ({x_ball}, {y_ball}, *), got {origin:?}"
    );

    // Axis parallel to original cyl axes (+z).
    let axis = fillet_cyl.axis();
    assert!(
        axis.dot(Vec3::new(0.0, 0.0, 1.0)) > 1.0 - 1e-12,
        "concave fillet axis should be +z (parallel to original cyls), got {axis:?}"
    );

    // Verify ball is INSIDE both cyls (internal tangency) — read
    // from the EMITTED cylinder origin, not from our own computed
    // x_ball/y_ball (which would be tautologically Q_i < r_i by
    // construction).
    let actual_dist_to_cyl1_axis = (origin.x().powi(2) + origin.y().powi(2)).sqrt();
    let actual_dist_to_cyl2_axis = ((origin.x() - big_d).powi(2) + origin.y().powi(2)).sqrt();
    assert!(
        actual_dist_to_cyl1_axis < r1 - 1e-9,
        "concave: emitted fillet origin must be INSIDE cyl1 (distance {actual_dist_to_cyl1_axis} < r1 = {r1})"
    );
    assert!(
        actual_dist_to_cyl2_axis < r2 - 1e-9,
        "concave: emitted fillet origin must be INSIDE cyl2 (distance {actual_dist_to_cyl2_axis} < r2 = {r2})"
    );

    // Cyl1 and cyl2 contacts lie on their respective cylinder surfaces.
    let want_c1 = Point3::new(r1 * x_ball / q1, r1 * y_ball / q1, z_lo);
    let dist_c1_axis = (want_c1.x().powi(2) + want_c1.y().powi(2)).sqrt();
    assert!(
        (dist_c1_axis - r1).abs() < 1e-9,
        "cyl1 contact must lie on cyl1: got {dist_c1_axis}, want {r1}"
    );
    let want_c2 = Point3::new(big_d + r2 * (x_ball - big_d) / q2, r2 * y_ball / q2, z_lo);
    let dist_c2_axis = ((want_c2.x() - big_d).powi(2) + want_c2.y().powi(2)).sqrt();
    assert!(
        (dist_c2_axis - r2).abs() < 1e-9,
        "cyl2 contact must lie on cyl2: got {dist_c2_axis}, want {r2}"
    );

    // Tangency to the EMITTED fillet cylinder: each contact must be
    // at distance `r_fillet` from the fillet-cyl axis (the ball
    // line) in the perpendicular plane. This catches axis/origin
    // bugs that the previous assertions wouldn't see.
    let dist_c1_to_ball =
        ((want_c1.x() - origin.x()).powi(2) + (want_c1.y() - origin.y()).powi(2)).sqrt();
    let dist_c2_to_ball =
        ((want_c2.x() - origin.x()).powi(2) + (want_c2.y() - origin.y()).powi(2)).sqrt();
    assert!(
        (dist_c1_to_ball - r_fillet).abs() < 1e-9,
        "cyl1 contact must be at distance r from fillet ball-line: got {dist_c1_to_ball}, want {r_fillet}"
    );
    assert!(
        (dist_c2_to_ball - r_fillet).abs() < 1e-9,
        "cyl2 contact must be at distance r from fillet ball-line: got {dist_c2_to_ball}, want {r_fillet}"
    );
}

/// Cylinder-cylinder mixed-convexity fillet: covers BOTH (s1=+1,
/// s2=−1) and (s1=−1, s2=+1) via a parameterized closure.
///
/// For mixed configs, one cylinder is internally tangent to the
/// rolling ball (`Q_i = r_i − r`) and the other externally tangent
/// (`Q_i = r_i + r`). The resulting fillet cylinder has its origin
/// at a position determined by the asymmetric `(Q1², Q2²)` pair —
/// distinct from both convex (both `+ r`) and both-concave
/// (both `− r`) cases.
#[test]
fn cylinder_cylinder_fillet_mixed_emits_cylinder() {
    use brepkit_math::surfaces::CylindricalSurface;
    use brepkit_topology::edge::{Edge, EdgeCurve};
    use brepkit_topology::face::Face;
    use brepkit_topology::vertex::Vertex;
    use brepkit_topology::wire::{OrientedEdge, Wire};

    let r1: f64 = 2.0;
    let r2: f64 = 2.5;
    let big_d: f64 = 3.0;
    let r_fillet: f64 = 0.4;
    let x_spine = (r1 * r1 - r2 * r2 + big_d * big_d) / (2.0 * big_d);
    let y_spine = (r1 * r1 - x_spine * x_spine).sqrt();

    let run_case = |reverse_s1: bool, reverse_s2: bool| {
        let mut topo = Topology::new();
        let cyl1 =
            CylindricalSurface::new(Point3::new(0.0, 0.0, 0.0), Vec3::new(0.0, 0.0, 1.0), r1)
                .unwrap();
        let cyl2 =
            CylindricalSurface::new(Point3::new(big_d, 0.0, 0.0), Vec3::new(0.0, 0.0, 1.0), r2)
                .unwrap();
        let z_lo = 0.0_f64;
        let z_hi = 4.0_f64;
        let p_start = Point3::new(x_spine, y_spine, z_lo);
        let p_end = Point3::new(x_spine, y_spine, z_hi);
        let v_start = topo.add_vertex(Vertex::new(p_start, 1e-7));
        let v_end = topo.add_vertex(Vertex::new(p_end, 1e-7));
        let line = brepkit_math::nurbs::curve::NurbsCurve::new(
            1,
            vec![0.0, 0.0, 1.0, 1.0],
            vec![p_start, p_end],
            vec![1.0, 1.0],
        )
        .unwrap();
        let eid = topo.add_edge(Edge::new(v_start, v_end, EdgeCurve::NurbsCurve(line)));
        let spine = Spine::from_single_edge(&topo, eid).unwrap();

        let w1 = topo.add_wire(Wire::new(vec![OrientedEdge::new(eid, true)], false).unwrap());
        let face1 = if reverse_s1 {
            topo.add_face(Face::new_reversed(
                w1,
                vec![],
                FaceSurface::Cylinder(cyl1.clone()),
            ))
        } else {
            topo.add_face(Face::new(w1, vec![], FaceSurface::Cylinder(cyl1.clone())))
        };
        let w2 = topo.add_wire(Wire::new(vec![OrientedEdge::new(eid, false)], false).unwrap());
        let face2 = if reverse_s2 {
            topo.add_face(Face::new_reversed(
                w2,
                vec![],
                FaceSurface::Cylinder(cyl2.clone()),
            ))
        } else {
            topo.add_face(Face::new(w2, vec![], FaceSurface::Cylinder(cyl2.clone())))
        };

        let result = cylinder_cylinder_fillet(&cyl1, &cyl2, &spine, &topo, r_fillet, face1, face2)
            .unwrap()
            .expect("mixed cyl-cyl fillet should produce a stripe");

        let fillet_cyl = match result.stripe.surface {
            FaceSurface::Cylinder(c) => c,
            other => panic!(
                "({reverse_s1}, {reverse_s2}): expected Cylinder, got {}",
                other.type_tag()
            ),
        };

        // Predicted parameters with per-face Q-substitution.
        let s1_signed = if reverse_s1 { -1.0_f64 } else { 1.0_f64 };
        let s2_signed = if reverse_s2 { -1.0_f64 } else { 1.0_f64 };
        let q1 = r1 + s1_signed * r_fillet;
        let q2 = r2 + s2_signed * r_fillet;
        let x_ball = (q1 * q1 - q2 * q2 + big_d * big_d) / (2.0 * big_d);
        let y_ball = (q1 * q1 - x_ball * x_ball).sqrt();

        assert!(
            (fillet_cyl.radius() - r_fillet).abs() < 1e-12,
            "({reverse_s1}, {reverse_s2}): fillet radius should be r = {r_fillet}, got {}",
            fillet_cyl.radius()
        );
        let origin = fillet_cyl.origin();
        assert!(
            (origin.x() - x_ball).abs() < 1e-12 && (origin.y() - y_ball).abs() < 1e-12,
            "({reverse_s1}, {reverse_s2}): fillet origin should be ({x_ball}, {y_ball}, *), got {origin:?}"
        );

        // Axis parallel to original cyls.
        let axis = fillet_cyl.axis();
        assert!(
            axis.dot(Vec3::new(0.0, 0.0, 1.0)) > 1.0 - 1e-12,
            "({reverse_s1}, {reverse_s2}): fillet axis should be +z, got {axis:?}"
        );

        // Read EMITTED contact endpoints. Each lies on its
        // respective cylinder (radial r_i from cyl_i axis) AND at
        // distance r from the fillet ball-line.
        let (t1_start, _) = result.stripe.contact1.domain();
        let c1_point = result.stripe.contact1.evaluate(t1_start);
        let (t2_start, _) = result.stripe.contact2.domain();
        let c2_point = result.stripe.contact2.evaluate(t2_start);

        let dist_c1_axis = (c1_point.x().powi(2) + c1_point.y().powi(2)).sqrt();
        let dist_c2_axis = ((c2_point.x() - big_d).powi(2) + c2_point.y().powi(2)).sqrt();
        assert!(
            (dist_c1_axis - r1).abs() < 1e-9,
            "({reverse_s1}, {reverse_s2}): c1 must lie on cyl1: {dist_c1_axis} vs r1 = {r1}"
        );
        assert!(
            (dist_c2_axis - r2).abs() < 1e-9,
            "({reverse_s1}, {reverse_s2}): c2 must lie on cyl2: {dist_c2_axis} vs r2 = {r2}"
        );

        let dist_c1_to_ball =
            ((c1_point.x() - x_ball).powi(2) + (c1_point.y() - y_ball).powi(2)).sqrt();
        let dist_c2_to_ball =
            ((c2_point.x() - x_ball).powi(2) + (c2_point.y() - y_ball).powi(2)).sqrt();
        assert!(
            (dist_c1_to_ball - r_fillet).abs() < 1e-9,
            "({reverse_s1}, {reverse_s2}): c1 must be at r from fillet ball-line: \
                 {dist_c1_to_ball} vs r = {r_fillet}"
        );
        assert!(
            (dist_c2_to_ball - r_fillet).abs() < 1e-9,
            "({reverse_s1}, {reverse_s2}): c2 must be at r from fillet ball-line: \
                 {dist_c2_to_ball} vs r = {r_fillet}"
        );
    };

    run_case(false, true); // (s1=+1, s2=-1)
    run_case(true, false); // (s1=-1, s2=+1)
}

/// Cone-cone coaxial convex fillet: two cones sharing the same axis
/// line with different half-angles. Their intersection is a single
/// circle, and the rolling-ball blend is a torus.
///
/// For cone1 apex at origin, β1=π/3 (60°); cone2 apex at (0,0,2),
/// β2=π/4 (45°); both axes +z, both faces NOT reversed, r=0.3:
///   - sin(β1−β2) = sin(π/12) ≈ 0.2588
///   - z_spine = h_2·cos β2·sin β1/sin(β1−β2) ≈ 4.732
///   - r_spine = z_spine·cot β1 ≈ 2.732
///   - z_b ≈ 4.548 (slightly less than z_spine for convex case)
///   - R_t ≈ 2.974 (slightly larger than r_spine — fillet outside both cones)
#[test]
fn cone_cone_coaxial_fillet_convex_emits_torus() {
    use brepkit_math::curves::Circle3D;
    use brepkit_math::surfaces::ConicalSurface;
    use brepkit_topology::edge::{Edge, EdgeCurve};
    use brepkit_topology::face::Face;
    use brepkit_topology::vertex::Vertex;
    use brepkit_topology::wire::{OrientedEdge, Wire};

    let mut topo = Topology::new();
    let beta1: f64 = std::f64::consts::PI / 3.0;
    let beta2: f64 = std::f64::consts::PI / 4.0;
    let h_2: f64 = 2.0;
    let r_fillet: f64 = 0.3;

    // Predicted spine.
    let sin_minus = (beta1 - beta2).sin();
    let z_spine = h_2 * beta2.cos() * beta1.sin() / sin_minus;
    let r_spine = z_spine * (beta1.cos() / beta1.sin());

    let cone1 =
        ConicalSurface::new(Point3::new(0.0, 0.0, 0.0), Vec3::new(0.0, 0.0, 1.0), beta1).unwrap();
    let cone2 =
        ConicalSurface::new(Point3::new(0.0, 0.0, h_2), Vec3::new(0.0, 0.0, 1.0), beta2).unwrap();

    let spine_circle = Circle3D::new(
        Point3::new(0.0, 0.0, z_spine),
        Vec3::new(0.0, 0.0, 1.0),
        r_spine,
    )
    .unwrap();
    let v = topo.add_vertex(Vertex::new(Point3::new(r_spine, 0.0, z_spine), 1e-7));
    let eid = topo.add_edge(Edge::new(v, v, EdgeCurve::Circle(spine_circle)));
    let spine = Spine::from_single_edge(&topo, eid).unwrap();

    let w1 = topo.add_wire(Wire::new(vec![OrientedEdge::new(eid, true)], true).unwrap());
    let face1 = topo.add_face(Face::new(w1, vec![], FaceSurface::Cone(cone1.clone())));
    let w2 = topo.add_wire(Wire::new(vec![OrientedEdge::new(eid, false)], true).unwrap());
    let face2 = topo.add_face(Face::new(w2, vec![], FaceSurface::Cone(cone2.clone())));

    let result = cone_cone_coaxial_fillet(&cone1, &cone2, &spine, &topo, r_fillet, face1, face2)
        .unwrap()
        .expect("convex coaxial cone-cone fillet should produce a stripe");

    let torus = match result.stripe.surface {
        FaceSurface::Torus(t) => t,
        other => panic!("expected Torus, got {}", other.type_tag()),
    };

    // Predicted z_b and R_t (convex-convex: s1=s2=+1).
    let expected_z_b =
        (h_2 * beta2.cos() * beta1.sin() + r_fillet * (beta2.sin() - beta1.sin())) / sin_minus;
    let expected_major = (expected_z_b * (beta1.cos() - beta2.cos()) + h_2 * beta2.cos())
        / (beta1.sin() - beta2.sin());

    assert!(
        (torus.major_radius() - expected_major).abs() < 1e-9,
        "torus major should be {expected_major}, got {}",
        torus.major_radius()
    );
    assert!(
        (torus.minor_radius() - r_fillet).abs() < 1e-12,
        "minor should equal r = {r_fillet}, got {}",
        torus.minor_radius()
    );

    // Major > spine radius (convex fillet outside both cones).
    assert!(
        torus.major_radius() > r_spine,
        "convex fillet major ({}) should be > r_spine ({r_spine})",
        torus.major_radius()
    );

    // Torus center on axis at z = z_b.
    let center = torus.center();
    assert!(
        center.x().abs() < 1e-12 && center.y().abs() < 1e-12,
        "torus center on z-axis, got {center:?}"
    );
    assert!(
        (center.z() - expected_z_b).abs() < 1e-9,
        "torus center z should be {expected_z_b}, got {}",
        center.z()
    );

    // Verify rolling-ball external tangency to BOTH cones:
    //   R_t · sin β_i − (z_b − z_apex_i) · cos β_i = r.
    let tang1 = expected_major * beta1.sin() - expected_z_b * beta1.cos();
    let tang2 = expected_major * beta2.sin() - (expected_z_b - h_2) * beta2.cos();
    assert!(
        (tang1 - r_fillet).abs() < 1e-9,
        "cone1 tangency: {tang1} should equal r = {r_fillet}"
    );
    assert!(
        (tang2 - r_fillet).abs() < 1e-9,
        "cone2 tangency: {tang2} should equal r = {r_fillet}"
    );

    // Both contacts on the torus + on their respective cones.
    let cot_b1 = beta1.cos() / beta1.sin();
    let cot_b2 = beta2.cos() / beta2.sin();
    let c1_axial = expected_z_b + r_fillet * beta1.cos();
    let c1_radial = expected_major - r_fillet * beta1.sin();
    let c2_axial = expected_z_b + r_fillet * beta2.cos();
    let c2_radial = expected_major - r_fillet * beta2.sin();
    let want_c1 = Point3::new(c1_radial, 0.0, c1_axial);
    let want_c2 = Point3::new(c2_radial, 0.0, c2_axial);

    // Cone1: r = (z − z_apex_1) · cot β1 = c1_axial · cot β1.
    let pred_c1_radial = c1_axial * cot_b1;
    assert!(
        (c1_radial - pred_c1_radial).abs() < 1e-9,
        "cone1 contact must lie on cone1 surface: predicted radial {pred_c1_radial}, got {c1_radial}"
    );
    // Cone2: r = (z − h_2) · cot β2.
    let pred_c2_radial = (c2_axial - h_2) * cot_b2;
    assert!(
        (c2_radial - pred_c2_radial).abs() < 1e-9,
        "cone2 contact must lie on cone2 surface: predicted radial {pred_c2_radial}, got {c2_radial}"
    );

    let (u_p, v_p) = ParametricSurface::project_point(&torus, want_c1);
    let on_torus_c1 = ParametricSurface::evaluate(&torus, u_p, v_p);
    let (u_q, v_q) = ParametricSurface::project_point(&torus, want_c2);
    let on_torus_c2 = ParametricSurface::evaluate(&torus, u_q, v_q);
    assert!(
        (on_torus_c1 - want_c1).length() < 1e-9,
        "cone1 contact must lie on torus: {on_torus_c1:?} vs {want_c1:?}"
    );
    assert!(
        (on_torus_c2 - want_c2).length() < 1e-9,
        "cone2 contact must lie on torus: {on_torus_c2:?} vs {want_c2:?}"
    );
}

/// Cone-cone coaxial both-concave fillet: two concave conical
/// cavities sharing an axis. Both s_i = −1 ⇒ rolling ball is
/// internally tangent to both cones.
///
/// The same setup as the convex test (cone1 apex at origin β=π/3;
/// cone2 apex at z=2 β=π/4; shared axis +z; r=0.3) but with both
/// faces REVERSED — exercises the `s_i = −1` branches in both
/// `z_b` and `R_t` formulas, plus the `(s1 − s2) · r` term that
/// vanishes when both signs match.
#[test]
fn cone_cone_coaxial_fillet_both_concave_emits_torus() {
    use brepkit_math::curves::Circle3D;
    use brepkit_math::surfaces::ConicalSurface;
    use brepkit_topology::edge::{Edge, EdgeCurve};
    use brepkit_topology::face::Face;
    use brepkit_topology::vertex::Vertex;
    use brepkit_topology::wire::{OrientedEdge, Wire};

    let mut topo = Topology::new();
    let beta1: f64 = std::f64::consts::PI / 3.0;
    let beta2: f64 = std::f64::consts::PI / 4.0;
    let h_2: f64 = 2.0;
    let r_fillet: f64 = 0.3;

    let sin_minus = (beta1 - beta2).sin();
    let z_spine = h_2 * beta2.cos() * beta1.sin() / sin_minus;
    let r_spine = z_spine * (beta1.cos() / beta1.sin());

    let cone1 =
        ConicalSurface::new(Point3::new(0.0, 0.0, 0.0), Vec3::new(0.0, 0.0, 1.0), beta1).unwrap();
    let cone2 =
        ConicalSurface::new(Point3::new(0.0, 0.0, h_2), Vec3::new(0.0, 0.0, 1.0), beta2).unwrap();

    let spine_circle = Circle3D::new(
        Point3::new(0.0, 0.0, z_spine),
        Vec3::new(0.0, 0.0, 1.0),
        r_spine,
    )
    .unwrap();
    let v = topo.add_vertex(Vertex::new(Point3::new(r_spine, 0.0, z_spine), 1e-7));
    let eid = topo.add_edge(Edge::new(v, v, EdgeCurve::Circle(spine_circle)));
    let spine = Spine::from_single_edge(&topo, eid).unwrap();

    // Both faces REVERSED.
    let w1 = topo.add_wire(Wire::new(vec![OrientedEdge::new(eid, true)], true).unwrap());
    let face1 = topo.add_face(Face::new_reversed(
        w1,
        vec![],
        FaceSurface::Cone(cone1.clone()),
    ));
    let w2 = topo.add_wire(Wire::new(vec![OrientedEdge::new(eid, false)], true).unwrap());
    let face2 = topo.add_face(Face::new_reversed(
        w2,
        vec![],
        FaceSurface::Cone(cone2.clone()),
    ));

    let result = cone_cone_coaxial_fillet(&cone1, &cone2, &spine, &topo, r_fillet, face1, face2)
        .unwrap()
        .expect("both-concave coaxial cone-cone fillet should produce a stripe");

    let torus = match result.stripe.surface {
        FaceSurface::Torus(t) => t,
        other => panic!("expected Torus, got {}", other.type_tag()),
    };

    // Predicted z_b, R_t (s1=s2=-1).
    let s1 = -1.0_f64;
    let s2 = -1.0_f64;
    let expected_z_b = (h_2 * beta2.cos() * beta1.sin()
        + r_fillet * (s1 * beta2.sin() - s2 * beta1.sin()))
        / sin_minus;
    let expected_major =
        (expected_z_b * (beta1.cos() - beta2.cos()) + h_2 * beta2.cos() + (s1 - s2) * r_fillet)
            / (beta1.sin() - beta2.sin());

    assert!(
        (torus.major_radius() - expected_major).abs() < 1e-9,
        "concave torus major should be {expected_major}, got {}",
        torus.major_radius()
    );

    // Concave fillet sits INSIDE both cones (major < r_spine).
    assert!(
        torus.major_radius() < r_spine,
        "concave fillet major ({}) should be < r_spine ({r_spine})",
        torus.major_radius()
    );

    // Internal tangency to both cones:
    //   R_t · sin β_i − (z_b − z_apex_i) · cos β_i = s_i · r = −r.
    let tang1 = expected_major * beta1.sin() - expected_z_b * beta1.cos();
    let tang2 = expected_major * beta2.sin() - (expected_z_b - h_2) * beta2.cos();
    assert!(
        (tang1 + r_fillet).abs() < 1e-9,
        "cone1 internal tangency: {tang1} should equal -r = {}",
        -r_fillet
    );
    assert!(
        (tang2 + r_fillet).abs() < 1e-9,
        "cone2 internal tangency: {tang2} should equal -r = {}",
        -r_fillet
    );

    // Contacts on respective cone surfaces.
    let cot_b1 = beta1.cos() / beta1.sin();
    let cot_b2 = beta2.cos() / beta2.sin();
    let c1_axial = expected_z_b + s1 * r_fillet * beta1.cos();
    let c1_radial = expected_major - s1 * r_fillet * beta1.sin();
    let c2_axial = expected_z_b + s2 * r_fillet * beta2.cos();
    let c2_radial = expected_major - s2 * r_fillet * beta2.sin();
    let pred_c1_radial = c1_axial * cot_b1;
    assert!(
        (c1_radial - pred_c1_radial).abs() < 1e-9,
        "cone1 contact must lie on cone1: predicted {pred_c1_radial}, got {c1_radial}"
    );
    let pred_c2_radial = (c2_axial - h_2) * cot_b2;
    assert!(
        (c2_radial - pred_c2_radial).abs() < 1e-9,
        "cone2 contact must lie on cone2: predicted {pred_c2_radial}, got {c2_radial}"
    );

    // Both contacts on torus.
    let want_c1 = Point3::new(c1_radial, 0.0, c1_axial);
    let want_c2 = Point3::new(c2_radial, 0.0, c2_axial);
    let (u_p, v_p) = ParametricSurface::project_point(&torus, want_c1);
    let on_torus_c1 = ParametricSurface::evaluate(&torus, u_p, v_p);
    let (u_q, v_q) = ParametricSurface::project_point(&torus, want_c2);
    let on_torus_c2 = ParametricSurface::evaluate(&torus, u_q, v_q);
    assert!(
        (on_torus_c1 - want_c1).length() < 1e-9,
        "cone1 contact must lie on torus: {on_torus_c1:?} vs {want_c1:?}"
    );
    assert!(
        (on_torus_c2 - want_c2).length() < 1e-9,
        "cone2 contact must lie on torus: {on_torus_c2:?} vs {want_c2:?}"
    );
}

/// Cone-cone coaxial mixed-convexity fillet: covers BOTH
/// (s1=+1, s2=−1) and (s1=−1, s2=+1) sign combinations.
///
/// In mixed configs the linear system
///   R_t · sin β_i − (z_b − z_apex_i) · cos β_i = s_i · r
/// solves with `s1 ≠ s2`, so the (s1−s2) term in the major-radius
/// formula contributes ±2r and the (s1·sin β2 − s2·sin β1) term
/// adds (sin β1 + sin β2) — both opposite signs from the symmetric
/// cases. The result places the fillet ball *outside* the spine
/// ring (one external + one internal contact) rather than aligned
/// with it.
///
/// Verifies emitted torus major/minor match the closed-form
/// solution and that contacts read from emitted curves via
/// `evaluate(t_start)` are positioned at the predicted offset from
/// the torus center along each cone's outward surface normal:
///
/// ```text
/// c_i = (R_t − s_i · r · sin β_i, z_b + s_i · r · cos β_i)
/// ```
///
/// The position-on-cone check (predicted (r, z) from `s_i`) is
/// stronger than a "lies on cone" tautology — for mixed configs
/// it independently confirms the implementation chose the contact
/// on the correct side of the spine ring (one external + one
/// internal tangency).
#[test]
fn cone_cone_coaxial_fillet_mixed_emits_torus() {
    use brepkit_math::curves::Circle3D;
    use brepkit_math::surfaces::ConicalSurface;
    use brepkit_topology::edge::{Edge, EdgeCurve};
    use brepkit_topology::face::Face;
    use brepkit_topology::vertex::Vertex;
    use brepkit_topology::wire::{OrientedEdge, Wire};

    let beta1: f64 = std::f64::consts::PI / 3.0;
    let beta2: f64 = std::f64::consts::PI / 4.0;
    let h_2: f64 = 2.0;
    let r_fillet: f64 = 0.3;
    let sin_minus = (beta1 - beta2).sin();
    let z_spine = h_2 * beta2.cos() * beta1.sin() / sin_minus;
    let r_spine = z_spine * (beta1.cos() / beta1.sin());

    let run_case = |reverse_s1: bool, reverse_s2: bool| {
        let mut topo = Topology::new();
        let cone1 =
            ConicalSurface::new(Point3::new(0.0, 0.0, 0.0), Vec3::new(0.0, 0.0, 1.0), beta1)
                .unwrap();
        let cone2 =
            ConicalSurface::new(Point3::new(0.0, 0.0, h_2), Vec3::new(0.0, 0.0, 1.0), beta2)
                .unwrap();
        let spine_circle = Circle3D::new(
            Point3::new(0.0, 0.0, z_spine),
            Vec3::new(0.0, 0.0, 1.0),
            r_spine,
        )
        .unwrap();
        let v = topo.add_vertex(Vertex::new(Point3::new(r_spine, 0.0, z_spine), 1e-7));
        let eid = topo.add_edge(Edge::new(v, v, EdgeCurve::Circle(spine_circle)));
        let spine = Spine::from_single_edge(&topo, eid).unwrap();

        let w1 = topo.add_wire(Wire::new(vec![OrientedEdge::new(eid, true)], true).unwrap());
        let face1 = if reverse_s1 {
            topo.add_face(Face::new_reversed(
                w1,
                vec![],
                FaceSurface::Cone(cone1.clone()),
            ))
        } else {
            topo.add_face(Face::new(w1, vec![], FaceSurface::Cone(cone1.clone())))
        };
        let w2 = topo.add_wire(Wire::new(vec![OrientedEdge::new(eid, false)], true).unwrap());
        let face2 = if reverse_s2 {
            topo.add_face(Face::new_reversed(
                w2,
                vec![],
                FaceSurface::Cone(cone2.clone()),
            ))
        } else {
            topo.add_face(Face::new(w2, vec![], FaceSurface::Cone(cone2.clone())))
        };

        let result =
            cone_cone_coaxial_fillet(&cone1, &cone2, &spine, &topo, r_fillet, face1, face2)
                .unwrap()
                .expect("mixed coaxial cone-cone fillet should produce a stripe");

        let torus = match result.stripe.surface {
            FaceSurface::Torus(ref t) => t.clone(),
            ref other => panic!(
                "({reverse_s1}, {reverse_s2}): expected Torus, got {}",
                other.type_tag()
            ),
        };

        let s1 = if reverse_s1 { -1.0_f64 } else { 1.0 };
        let s2 = if reverse_s2 { -1.0_f64 } else { 1.0 };
        let expected_z_b = (h_2 * beta2.cos() * beta1.sin()
            + r_fillet * (s1 * beta2.sin() - s2 * beta1.sin()))
            / sin_minus;
        let expected_major =
            (expected_z_b * (beta1.cos() - beta2.cos()) + h_2 * beta2.cos() + (s1 - s2) * r_fillet)
                / (beta1.sin() - beta2.sin());

        assert!(
            (torus.major_radius() - expected_major).abs() < 1e-9,
            "({reverse_s1}, {reverse_s2}): mixed torus major should be {expected_major}, got {}",
            torus.major_radius()
        );
        assert!(
            (torus.minor_radius() - r_fillet).abs() < 1e-9,
            "({reverse_s1}, {reverse_s2}): minor should be {r_fillet}, got {}",
            torus.minor_radius()
        );

        // Read EMITTED contact endpoints from the stripe, then
        // assert they sit at the predicted (r, z) on each cone:
        //   c_i = (R_t − s_i·r·sin β_i, z_b + s_i·r·cos β_i)
        // Position-on-cone is stronger than "lies on cone" — for
        // mixed configs the contact must be on the *correct side*
        // of the spine ring (s1=+1 retreats toward apex_1; s2=−1
        // extends from apex_2), and asserting predicted (r, z)
        // catches that without becoming tautological with the
        // major/minor checks above.
        let (t1_start, _) = result.stripe.contact1.domain();
        let c1_point = result.stripe.contact1.evaluate(t1_start);
        let (t2_start, _) = result.stripe.contact2.domain();
        let c2_point = result.stripe.contact2.evaluate(t2_start);

        let pred_c1_r = expected_major - s1 * r_fillet * beta1.sin();
        let pred_c1_z = expected_z_b + s1 * r_fillet * beta1.cos();
        let pred_c2_r = expected_major - s2 * r_fillet * beta2.sin();
        let pred_c2_z = expected_z_b + s2 * r_fillet * beta2.cos();

        let c1_radial = (c1_point.x().powi(2) + c1_point.y().powi(2)).sqrt();
        let c2_radial = (c2_point.x().powi(2) + c2_point.y().powi(2)).sqrt();
        assert!(
            (c1_radial - pred_c1_r).abs() < 1e-9 && (c1_point.z() - pred_c1_z).abs() < 1e-9,
            "({reverse_s1}, {reverse_s2}): contact1 should be at \
                 (r={pred_c1_r}, z={pred_c1_z}); got (r={c1_radial}, z={})",
            c1_point.z()
        );
        assert!(
            (c2_radial - pred_c2_r).abs() < 1e-9 && (c2_point.z() - pred_c2_z).abs() < 1e-9,
            "({reverse_s1}, {reverse_s2}): contact2 should be at \
                 (r={pred_c2_r}, z={pred_c2_z}); got (r={c2_radial}, z={})",
            c2_point.z()
        );
    };

    run_case(false, true); // (s1=+1, s2=-1)
    run_case(true, false); // (s1=-1, s2=+1)
}

/// Cylinder-cylinder convex chamfer: parallel-axis cyls with the
/// chamfer surface a planar bevel containing both contact lines
/// (each parallel to the cyl axes).
///
/// For cyl1 r=2 at origin axis +z, cyl2 r=2.5 at (3,0,*) axis +z,
/// D=3, both faces NOT reversed, +y spine, d=0.4 each:
///   - Spine at (1.125, √3, *)
///   - Δθ_1 = +1·d/r1 = 0.2 (CCW on cyl1, AWAY from cyl2)
///   - Δθ_2 = −1·d/r2 = −0.16 (CW on cyl2, AWAY from cyl1)
///   - Chamfer plane through both contact lines
#[test]
fn cylinder_cylinder_chamfer_convex_emits_plane() {
    use brepkit_math::surfaces::CylindricalSurface;
    use brepkit_topology::edge::{Edge, EdgeCurve};
    use brepkit_topology::face::Face;
    use brepkit_topology::vertex::Vertex;
    use brepkit_topology::wire::{OrientedEdge, Wire};

    let mut topo = Topology::new();
    let r1: f64 = 2.0;
    let r2: f64 = 2.5;
    let big_d: f64 = 3.0;
    let d: f64 = 0.4;

    let cyl1 =
        CylindricalSurface::new(Point3::new(0.0, 0.0, 0.0), Vec3::new(0.0, 0.0, 1.0), r1).unwrap();
    let cyl2 = CylindricalSurface::new(Point3::new(big_d, 0.0, 0.0), Vec3::new(0.0, 0.0, 1.0), r2)
        .unwrap();

    let x_spine = (r1 * r1 - r2 * r2 + big_d * big_d) / (2.0 * big_d);
    let y_spine = (r1 * r1 - x_spine * x_spine).sqrt();
    let z_lo = 0.0_f64;
    let z_hi = 4.0_f64;
    let p_start = Point3::new(x_spine, y_spine, z_lo);
    let p_end = Point3::new(x_spine, y_spine, z_hi);
    let v_start = topo.add_vertex(Vertex::new(p_start, 1e-7));
    let v_end = topo.add_vertex(Vertex::new(p_end, 1e-7));
    let line = brepkit_math::nurbs::curve::NurbsCurve::new(
        1,
        vec![0.0, 0.0, 1.0, 1.0],
        vec![p_start, p_end],
        vec![1.0, 1.0],
    )
    .unwrap();
    let eid = topo.add_edge(Edge::new(v_start, v_end, EdgeCurve::NurbsCurve(line)));
    let spine = Spine::from_single_edge(&topo, eid).unwrap();

    let w1 = topo.add_wire(Wire::new(vec![OrientedEdge::new(eid, true)], false).unwrap());
    let face1 = topo.add_face(Face::new(w1, vec![], FaceSurface::Cylinder(cyl1.clone())));
    let w2 = topo.add_wire(Wire::new(vec![OrientedEdge::new(eid, false)], false).unwrap());
    let face2 = topo.add_face(Face::new(w2, vec![], FaceSurface::Cylinder(cyl2.clone())));

    let result = cylinder_cylinder_chamfer(&cyl1, &cyl2, &spine, &topo, d, d, face1, face2)
        .unwrap()
        .expect("convex parallel-axis cyl-cyl chamfer should produce a stripe");

    let (chamfer_normal, chamfer_d) = match result.stripe.surface {
        FaceSurface::Plane { normal, d } => (normal, d),
        other => panic!("expected Plane, got {}", other.type_tag()),
    };

    // Predicted contacts.
    let dtheta1 = d / r1; // y_sign=+1, s1=+1
    let dtheta2 = -d / r2; // y_sign=+1, s2=+1, negation
    let (sin1, cos1) = dtheta1.sin_cos();
    let (sin2, cos2) = dtheta2.sin_cos();
    let c1_x = x_spine * cos1 - y_spine * sin1;
    let c1_y = y_spine * cos1 + x_spine * sin1;
    let c2_local_x = (x_spine - big_d) * cos2 - y_spine * sin2;
    let c2_local_y = y_spine * cos2 + (x_spine - big_d) * sin2;
    let c2_x = c2_local_x + big_d;
    let c2_y = c2_local_y;

    // Contact1 must lie on cyl1 (radial = r1 from cyl1 axis).
    let dist_c1_axis = (c1_x.powi(2) + c1_y.powi(2)).sqrt();
    assert!(
        (dist_c1_axis - r1).abs() < 1e-9,
        "cyl1 contact must lie on cyl1: distance = {dist_c1_axis}, want r1 = {r1}"
    );
    // Contact2 must lie on cyl2.
    let dist_c2_axis = ((c2_x - big_d).powi(2) + c2_y.powi(2)).sqrt();
    assert!(
        (dist_c2_axis - r2).abs() < 1e-9,
        "cyl2 contact must lie on cyl2: distance = {dist_c2_axis}, want r2 = {r2}"
    );

    // Both contact LINES (z varies) must lie on the chamfer plane.
    // For a plane (normal · p = d) and contact at (c_x, c_y, z) for
    // any z, the plane equation must hold ∀z. Since the plane normal
    // is computed as a_cyl.cross(span) and a_cyl = +z, the normal
    // has zero z-component. ⇒ checking with z = z_lo suffices.
    let p1 = Point3::new(c1_x, c1_y, z_lo);
    let p2 = Point3::new(c2_x, c2_y, z_lo);
    let on_plane_1 = chamfer_normal.dot(Vec3::new(p1.x(), p1.y(), p1.z())) - chamfer_d;
    let on_plane_2 = chamfer_normal.dot(Vec3::new(p2.x(), p2.y(), p2.z())) - chamfer_d;
    assert!(
        on_plane_1.abs() < 1e-9,
        "cyl1 contact must lie on chamfer plane: residual {on_plane_1}"
    );
    assert!(
        on_plane_2.abs() < 1e-9,
        "cyl2 contact must lie on chamfer plane: residual {on_plane_2}"
    );

    // Chamfer plane normal must be perpendicular to the cyl axis +z
    // (the contact lines are along +z, so the plane contains them).
    assert!(
        chamfer_normal.z().abs() < 1e-12,
        "chamfer plane normal should be perpendicular to z axis, got {chamfer_normal:?}"
    );
}

/// Cylinder-cylinder both-concave chamfer at the −y spine (exercises
/// BOTH the `s_i = −1` branches AND the `y_sign = −1` branch in the
/// `dtheta_i = y_sign · s_i · d_i / r_i` formulas).
///
/// Setup: same cylinders as the convex test (r1=2, r2=2.5, D=3,
/// d=0.4) but spine at the −y intersection line and both faces
/// REVERSED. This means:
///   - dtheta_1 = (−1)·(−1)·d/r1 = +d/r1 (still CCW on cyl1, but
///     for a different geometric reason — concave going TOWARD
///     cyl2 from the −y spine = +θ direction)
///   - dtheta_2 = −(−1)·(−1)·d/r2 = −d/r2 (CW on cyl2)
///
/// The signs happen to match the convex +y-spine case numerically,
/// but they reach there via a DIFFERENT path through the formulas,
/// exercising both the y_sign and s_i flip branches.
#[test]
fn cylinder_cylinder_chamfer_both_concave_negative_y_emits_plane() {
    use brepkit_math::surfaces::CylindricalSurface;
    use brepkit_topology::edge::{Edge, EdgeCurve};
    use brepkit_topology::face::Face;
    use brepkit_topology::vertex::Vertex;
    use brepkit_topology::wire::{OrientedEdge, Wire};

    let mut topo = Topology::new();
    let r1: f64 = 2.0;
    let r2: f64 = 2.5;
    let big_d: f64 = 3.0;
    let d: f64 = 0.4;

    let cyl1 =
        CylindricalSurface::new(Point3::new(0.0, 0.0, 0.0), Vec3::new(0.0, 0.0, 1.0), r1).unwrap();
    let cyl2 = CylindricalSurface::new(Point3::new(big_d, 0.0, 0.0), Vec3::new(0.0, 0.0, 1.0), r2)
        .unwrap();

    let x_spine = (r1 * r1 - r2 * r2 + big_d * big_d) / (2.0 * big_d);
    let y_spine = -((r1 * r1 - x_spine * x_spine).sqrt()); // NEGATIVE-y spine
    let z_lo = 0.0_f64;
    let z_hi = 4.0_f64;
    let p_start = Point3::new(x_spine, y_spine, z_lo);
    let p_end = Point3::new(x_spine, y_spine, z_hi);
    let v_start = topo.add_vertex(Vertex::new(p_start, 1e-7));
    let v_end = topo.add_vertex(Vertex::new(p_end, 1e-7));
    let line = brepkit_math::nurbs::curve::NurbsCurve::new(
        1,
        vec![0.0, 0.0, 1.0, 1.0],
        vec![p_start, p_end],
        vec![1.0, 1.0],
    )
    .unwrap();
    let eid = topo.add_edge(Edge::new(v_start, v_end, EdgeCurve::NurbsCurve(line)));
    let spine = Spine::from_single_edge(&topo, eid).unwrap();

    // Both faces REVERSED.
    let w1 = topo.add_wire(Wire::new(vec![OrientedEdge::new(eid, true)], false).unwrap());
    let face1 = topo.add_face(Face::new_reversed(
        w1,
        vec![],
        FaceSurface::Cylinder(cyl1.clone()),
    ));
    let w2 = topo.add_wire(Wire::new(vec![OrientedEdge::new(eid, false)], false).unwrap());
    let face2 = topo.add_face(Face::new_reversed(
        w2,
        vec![],
        FaceSurface::Cylinder(cyl2.clone()),
    ));

    let result = cylinder_cylinder_chamfer(&cyl1, &cyl2, &spine, &topo, d, d, face1, face2)
        .unwrap()
        .expect("both-concave −y-spine cyl-cyl chamfer should produce a stripe");

    let (chamfer_normal, chamfer_d) = match result.stripe.surface {
        FaceSurface::Plane { normal, d } => (normal, d),
        other => panic!("expected Plane, got {}", other.type_tag()),
    };

    // Predicted contacts. y_sign = -1, s1 = s2 = -1.
    // dtheta_1 = (-1)·(-1)·d/r1 = +d/r1
    // dtheta_2 = -(-1)·(-1)·d/r2 = -d/r2
    let dtheta1 = d / r1;
    let dtheta2 = -d / r2;
    let (sin1, cos1) = dtheta1.sin_cos();
    let (sin2, cos2) = dtheta2.sin_cos();
    let c1_x = x_spine * cos1 - y_spine * sin1;
    let c1_y = y_spine * cos1 + x_spine * sin1;
    let c2_local_x = (x_spine - big_d) * cos2 - y_spine * sin2;
    let c2_local_y = y_spine * cos2 + (x_spine - big_d) * sin2;
    let c2_x = c2_local_x + big_d;
    let c2_y = c2_local_y;

    // Contacts on respective cylinder surfaces.
    let dist_c1_axis = (c1_x.powi(2) + c1_y.powi(2)).sqrt();
    assert!(
        (dist_c1_axis - r1).abs() < 1e-9,
        "cyl1 contact must lie on cyl1: distance = {dist_c1_axis}, want r1 = {r1}"
    );
    let dist_c2_axis = ((c2_x - big_d).powi(2) + c2_y.powi(2)).sqrt();
    assert!(
        (dist_c2_axis - r2).abs() < 1e-9,
        "cyl2 contact must lie on cyl2: distance = {dist_c2_axis}, want r2 = {r2}"
    );

    // Both contacts on the chamfer plane.
    let p1 = Point3::new(c1_x, c1_y, z_lo);
    let p2 = Point3::new(c2_x, c2_y, z_lo);
    let on_plane_1 = chamfer_normal.dot(Vec3::new(p1.x(), p1.y(), p1.z())) - chamfer_d;
    let on_plane_2 = chamfer_normal.dot(Vec3::new(p2.x(), p2.y(), p2.z())) - chamfer_d;
    assert!(
        on_plane_1.abs() < 1e-9,
        "cyl1 contact must lie on chamfer plane: residual {on_plane_1}"
    );
    assert!(
        on_plane_2.abs() < 1e-9,
        "cyl2 contact must lie on chamfer plane: residual {on_plane_2}"
    );

    // Concave-going-TOWARD geometry: contacts now pull TOWARD the
    // other cyl (rather than away). For y_sign=-1 with concave, both
    // contacts have negative y components less negative than the
    // spine (toward y=0).
    assert!(
        c1_y > y_spine,
        "concave cyl1 contact should pull TOWARD y=0: got {c1_y} vs spine {y_spine}"
    );
    assert!(
        c2_y > y_spine,
        "concave cyl2 contact should pull TOWARD y=0: got {c2_y} vs spine {y_spine}"
    );

    // Chamfer plane perpendicular to z axis.
    assert!(
        chamfer_normal.z().abs() < 1e-12,
        "chamfer plane normal must be perpendicular to z, got {chamfer_normal:?}"
    );
}

/// Cylinder-cylinder mixed-convexity chamfer: covers BOTH (s1=+1,
/// s2=−1) and (s1=−1, s2=+1) at the +y spine.
///
/// In mixed configs the per-cyl angular displacement signs combine
/// asymmetrically:
///   Δθ_1 = y_sign · s1 · d/r1
///   Δθ_2 = −y_sign · s2 · d/r2
/// The product `s1 · s2` differs from the symmetric cases, giving
/// a chord between contacts with a different slope and hence a
/// distinct chamfer plane.
///
/// Verifies emitted contact endpoints via `evaluate(t_start)` and
/// asserts each lies on its cylinder surface AND on the chamfer
/// plane (residual against `n · p − d` < tol).
#[test]
fn cylinder_cylinder_chamfer_mixed_emits_plane() {
    use brepkit_math::surfaces::CylindricalSurface;
    use brepkit_topology::edge::{Edge, EdgeCurve};
    use brepkit_topology::face::Face;
    use brepkit_topology::vertex::Vertex;
    use brepkit_topology::wire::{OrientedEdge, Wire};

    let r1: f64 = 2.0;
    let r2: f64 = 2.5;
    let big_d: f64 = 3.0;
    let d: f64 = 0.4;
    let x_spine = (r1 * r1 - r2 * r2 + big_d * big_d) / (2.0 * big_d);
    let y_spine = (r1 * r1 - x_spine * x_spine).sqrt();

    let run_case = |reverse_s1: bool, reverse_s2: bool| {
        let mut topo = Topology::new();
        let cyl1 =
            CylindricalSurface::new(Point3::new(0.0, 0.0, 0.0), Vec3::new(0.0, 0.0, 1.0), r1)
                .unwrap();
        let cyl2 =
            CylindricalSurface::new(Point3::new(big_d, 0.0, 0.0), Vec3::new(0.0, 0.0, 1.0), r2)
                .unwrap();
        let z_lo = 0.0_f64;
        let z_hi = 4.0_f64;
        let p_start = Point3::new(x_spine, y_spine, z_lo);
        let p_end = Point3::new(x_spine, y_spine, z_hi);
        let v_start = topo.add_vertex(Vertex::new(p_start, 1e-7));
        let v_end = topo.add_vertex(Vertex::new(p_end, 1e-7));
        let line = brepkit_math::nurbs::curve::NurbsCurve::new(
            1,
            vec![0.0, 0.0, 1.0, 1.0],
            vec![p_start, p_end],
            vec![1.0, 1.0],
        )
        .unwrap();
        let eid = topo.add_edge(Edge::new(v_start, v_end, EdgeCurve::NurbsCurve(line)));
        let spine = Spine::from_single_edge(&topo, eid).unwrap();

        let w1 = topo.add_wire(Wire::new(vec![OrientedEdge::new(eid, true)], false).unwrap());
        let face1 = if reverse_s1 {
            topo.add_face(Face::new_reversed(
                w1,
                vec![],
                FaceSurface::Cylinder(cyl1.clone()),
            ))
        } else {
            topo.add_face(Face::new(w1, vec![], FaceSurface::Cylinder(cyl1.clone())))
        };
        let w2 = topo.add_wire(Wire::new(vec![OrientedEdge::new(eid, false)], false).unwrap());
        let face2 = if reverse_s2 {
            topo.add_face(Face::new_reversed(
                w2,
                vec![],
                FaceSurface::Cylinder(cyl2.clone()),
            ))
        } else {
            topo.add_face(Face::new(w2, vec![], FaceSurface::Cylinder(cyl2.clone())))
        };

        let result = cylinder_cylinder_chamfer(&cyl1, &cyl2, &spine, &topo, d, d, face1, face2)
            .unwrap()
            .expect("mixed cyl-cyl chamfer should produce a stripe");

        let (chamfer_normal, chamfer_d) = match result.stripe.surface {
            FaceSurface::Plane { normal, d } => (normal, d),
            ref other => panic!(
                "({reverse_s1}, {reverse_s2}): expected Plane, got {}",
                other.type_tag()
            ),
        };

        // Read EMITTED contact endpoints.
        let (t1_start, _) = result.stripe.contact1.domain();
        let c1_point = result.stripe.contact1.evaluate(t1_start);
        let (t2_start, _) = result.stripe.contact2.domain();
        let c2_point = result.stripe.contact2.evaluate(t2_start);

        // Each lies on its cylinder surface.
        let dist_c1_axis = (c1_point.x().powi(2) + c1_point.y().powi(2)).sqrt();
        let dist_c2_axis = ((c2_point.x() - big_d).powi(2) + c2_point.y().powi(2)).sqrt();
        assert!(
            (dist_c1_axis - r1).abs() < 1e-9,
            "({reverse_s1}, {reverse_s2}): c1 must lie on cyl1: \
                 distance = {dist_c1_axis}, want r1 = {r1}"
        );
        assert!(
            (dist_c2_axis - r2).abs() < 1e-9,
            "({reverse_s1}, {reverse_s2}): c2 must lie on cyl2: \
                 distance = {dist_c2_axis}, want r2 = {r2}"
        );

        // Both contacts on the chamfer plane.
        let on_plane_1 =
            chamfer_normal.dot(Vec3::new(c1_point.x(), c1_point.y(), c1_point.z())) - chamfer_d;
        let on_plane_2 =
            chamfer_normal.dot(Vec3::new(c2_point.x(), c2_point.y(), c2_point.z())) - chamfer_d;
        assert!(
            on_plane_1.abs() < 1e-9,
            "({reverse_s1}, {reverse_s2}): c1 on chamfer plane: residual {on_plane_1}"
        );
        assert!(
            on_plane_2.abs() < 1e-9,
            "({reverse_s1}, {reverse_s2}): c2 on chamfer plane: residual {on_plane_2}"
        );

        // Plane normal perpendicular to z (cyl axes).
        assert!(
            chamfer_normal.z().abs() < 1e-12,
            "({reverse_s1}, {reverse_s2}): plane normal should be perpendicular to z, got {chamfer_normal:?}"
        );
    };

    run_case(false, true); // (s1=+1, s2=-1)
    run_case(true, false); // (s1=-1, s2=+1)
}

/// Cone-cone coaxial convex chamfer: two cones sharing an axis with
/// different half-angles. Chamfer surface is an axisymmetric cone
/// connecting both cone-generator contact circles.
///
/// Same setup as the cone-cone fillet test (cone1 apex at origin
/// β=π/3; cone2 apex at z=2 β=π/4; faces NOT reversed; d=0.3):
///   - z_spine ≈ 4.732, r_spine ≈ 2.732
///   - cone1 contact retreats toward apex1: (r_spine − d·cos β1,
///     z_spine − d·sin β1) ≈ (2.582, 4.472)
///   - cone2 contact extends away from apex2: (r_spine + d·cos β2,
///     z_spine + d·sin β2) ≈ (2.944, 4.944)
///   - chamfer cone apex on axis at line P1−P2 extrapolated to r=0
#[test]
fn cone_cone_coaxial_chamfer_convex_emits_cone() {
    use brepkit_math::curves::Circle3D;
    use brepkit_math::surfaces::ConicalSurface;
    use brepkit_topology::edge::{Edge, EdgeCurve};
    use brepkit_topology::face::Face;
    use brepkit_topology::vertex::Vertex;
    use brepkit_topology::wire::{OrientedEdge, Wire};

    let mut topo = Topology::new();
    let beta1: f64 = std::f64::consts::PI / 3.0;
    let beta2: f64 = std::f64::consts::PI / 4.0;
    let h_2: f64 = 2.0;
    let d: f64 = 0.3;

    let sin_minus = (beta1 - beta2).sin();
    let z_spine = h_2 * beta2.cos() * beta1.sin() / sin_minus;
    let r_spine = z_spine * (beta1.cos() / beta1.sin());

    let cone1 =
        ConicalSurface::new(Point3::new(0.0, 0.0, 0.0), Vec3::new(0.0, 0.0, 1.0), beta1).unwrap();
    let cone2 =
        ConicalSurface::new(Point3::new(0.0, 0.0, h_2), Vec3::new(0.0, 0.0, 1.0), beta2).unwrap();

    let spine_circle = Circle3D::new(
        Point3::new(0.0, 0.0, z_spine),
        Vec3::new(0.0, 0.0, 1.0),
        r_spine,
    )
    .unwrap();
    let v = topo.add_vertex(Vertex::new(Point3::new(r_spine, 0.0, z_spine), 1e-7));
    let eid = topo.add_edge(Edge::new(v, v, EdgeCurve::Circle(spine_circle)));
    let spine = Spine::from_single_edge(&topo, eid).unwrap();

    let w1 = topo.add_wire(Wire::new(vec![OrientedEdge::new(eid, true)], true).unwrap());
    let face1 = topo.add_face(Face::new(w1, vec![], FaceSurface::Cone(cone1.clone())));
    let w2 = topo.add_wire(Wire::new(vec![OrientedEdge::new(eid, false)], true).unwrap());
    let face2 = topo.add_face(Face::new(w2, vec![], FaceSurface::Cone(cone2.clone())));

    let result = cone_cone_coaxial_chamfer(&cone1, &cone2, &spine, &topo, d, d, face1, face2)
        .unwrap()
        .expect("convex coaxial cone-cone chamfer should produce a stripe");

    let chamfer_cone = match result.stripe.surface {
        FaceSurface::Cone(c) => c,
        other => panic!("expected Cone, got {}", other.type_tag()),
    };

    // Predicted contacts (s1=s2=+1).
    let r_c1 = r_spine - d * beta1.cos();
    let z_c1 = z_spine - d * beta1.sin();
    let r_c2 = r_spine + d * beta2.cos();
    let z_c2 = z_spine + d * beta2.sin();

    // Cone1 contact retreats toward apex1 (r and z BOTH decrease).
    assert!(
        r_c1 < r_spine && z_c1 < z_spine,
        "cone1 contact should retreat toward apex1: got ({r_c1}, {z_c1}) vs spine ({r_spine}, {z_spine})"
    );
    // Cone2 contact extends away from apex2 (r and z BOTH increase).
    assert!(
        r_c2 > r_spine && z_c2 > z_spine,
        "cone2 contact should extend away from apex2: got ({r_c2}, {z_c2}) vs spine"
    );

    // Predicted chamfer apex (line extrapolated to r=0).
    let dr = r_c2 - r_c1;
    let dz = z_c2 - z_c1;
    let expected_apex_z = z_c1 - r_c1 * dz / dr;
    let mid_z = 0.5 * (z_c1 + z_c2);
    let r_avg = 0.5 * (r_c1 + r_c2);
    let expected_beta = ((mid_z - expected_apex_z).abs() / r_avg).atan();

    let apex = chamfer_cone.apex();
    assert!(
        apex.x().abs() < 1e-12 && apex.y().abs() < 1e-12,
        "apex should be on z-axis, got {apex:?}"
    );
    assert!(
        (apex.z() - expected_apex_z).abs() < 1e-9,
        "apex z = {}, expected {expected_apex_z}",
        apex.z()
    );
    assert!(
        (chamfer_cone.half_angle() - expected_beta).abs() < 1e-9,
        "chamfer half-angle should be {expected_beta}, got {}",
        chamfer_cone.half_angle()
    );

    // Cone axis: contacts above apex (apex z ≈ 1.105 < contacts z ≈ 4.5–5).
    let axis = chamfer_cone.axis();
    assert!(
        axis.dot(Vec3::new(0.0, 0.0, 1.0)) > 1.0 - 1e-12,
        "convex chamfer cone axis should be +z (contacts above apex), got {axis:?}"
    );

    // Both contacts on chamfer cone.
    let want_c1 = Point3::new(r_c1, 0.0, z_c1);
    let want_c2 = Point3::new(r_c2, 0.0, z_c2);
    let (u_p, v_p) = ParametricSurface::project_point(&chamfer_cone, want_c1);
    let on_cone_c1 = ParametricSurface::evaluate(&chamfer_cone, u_p, v_p);
    let (u_q, v_q) = ParametricSurface::project_point(&chamfer_cone, want_c2);
    let on_cone_c2 = ParametricSurface::evaluate(&chamfer_cone, u_q, v_q);
    assert!(
        (on_cone_c1 - want_c1).length() < 1e-9,
        "cone1 contact must lie on chamfer cone: {on_cone_c1:?} vs {want_c1:?}"
    );
    assert!(
        (on_cone_c2 - want_c2).length() < 1e-9,
        "cone2 contact must lie on chamfer cone: {on_cone_c2:?} vs {want_c2:?}"
    );

    // Both contacts on respective cones.
    let cot_b1 = beta1.cos() / beta1.sin();
    let cot_b2 = beta2.cos() / beta2.sin();
    let pred_r_c1 = z_c1 * cot_b1;
    let pred_r_c2 = (z_c2 - h_2) * cot_b2;
    assert!(
        (r_c1 - pred_r_c1).abs() < 1e-9,
        "cone1 contact must lie on cone1: predicted radial {pred_r_c1}, got {r_c1}"
    );
    assert!(
        (r_c2 - pred_r_c2).abs() < 1e-9,
        "cone2 contact must lie on cone2: predicted radial {pred_r_c2}, got {r_c2}"
    );
}

/// Cone-cone coaxial both-concave chamfer: two cone-shaped cavities
/// sharing an axis. Both s_i = −1 ⇒ each contact moves to the
/// OPPOSITE side along its generator from the convex case.
///
/// Same setup as `cone_cone_coaxial_chamfer_convex_emits_cone` but
/// with both faces REVERSED. Per-face orientation flips compose:
/// cone1 contact extends AWAY from apex1 (instead of retreating
/// toward); cone2 contact retreats TOWARD apex2 (instead of
/// extending away).
#[test]
fn cone_cone_coaxial_chamfer_both_concave_emits_cone() {
    use brepkit_math::curves::Circle3D;
    use brepkit_math::surfaces::ConicalSurface;
    use brepkit_topology::edge::{Edge, EdgeCurve};
    use brepkit_topology::face::Face;
    use brepkit_topology::vertex::Vertex;
    use brepkit_topology::wire::{OrientedEdge, Wire};

    let mut topo = Topology::new();
    let beta1: f64 = std::f64::consts::PI / 3.0;
    let beta2: f64 = std::f64::consts::PI / 4.0;
    let h_2: f64 = 2.0;
    let d: f64 = 0.3;

    let sin_minus = (beta1 - beta2).sin();
    let z_spine = h_2 * beta2.cos() * beta1.sin() / sin_minus;
    let r_spine = z_spine * (beta1.cos() / beta1.sin());

    let cone1 =
        ConicalSurface::new(Point3::new(0.0, 0.0, 0.0), Vec3::new(0.0, 0.0, 1.0), beta1).unwrap();
    let cone2 =
        ConicalSurface::new(Point3::new(0.0, 0.0, h_2), Vec3::new(0.0, 0.0, 1.0), beta2).unwrap();

    let spine_circle = Circle3D::new(
        Point3::new(0.0, 0.0, z_spine),
        Vec3::new(0.0, 0.0, 1.0),
        r_spine,
    )
    .unwrap();
    let v = topo.add_vertex(Vertex::new(Point3::new(r_spine, 0.0, z_spine), 1e-7));
    let eid = topo.add_edge(Edge::new(v, v, EdgeCurve::Circle(spine_circle)));
    let spine = Spine::from_single_edge(&topo, eid).unwrap();

    // Both faces REVERSED.
    let w1 = topo.add_wire(Wire::new(vec![OrientedEdge::new(eid, true)], true).unwrap());
    let face1 = topo.add_face(Face::new_reversed(
        w1,
        vec![],
        FaceSurface::Cone(cone1.clone()),
    ));
    let w2 = topo.add_wire(Wire::new(vec![OrientedEdge::new(eid, false)], true).unwrap());
    let face2 = topo.add_face(Face::new_reversed(
        w2,
        vec![],
        FaceSurface::Cone(cone2.clone()),
    ));

    let result = cone_cone_coaxial_chamfer(&cone1, &cone2, &spine, &topo, d, d, face1, face2)
        .unwrap()
        .expect("both-concave coaxial cone-cone chamfer should produce a stripe");

    let chamfer_cone = match result.stripe.surface {
        FaceSurface::Cone(c) => c,
        other => panic!("expected Cone, got {}", other.type_tag()),
    };

    // Predicted contacts (s1=s2=-1 ⇒ both signs flip).
    let r_c1 = r_spine + d * beta1.cos();
    let z_c1 = z_spine + d * beta1.sin();
    let r_c2 = r_spine - d * beta2.cos();
    let z_c2 = z_spine - d * beta2.sin();

    // Cone1 contact extends AWAY from apex1 (opposite convex).
    assert!(
        r_c1 > r_spine && z_c1 > z_spine,
        "concave cone1 contact should extend away from apex1: got ({r_c1}, {z_c1}) vs spine ({r_spine}, {z_spine})"
    );
    // Cone2 contact retreats TOWARD apex2 (opposite convex).
    assert!(
        r_c2 < r_spine && z_c2 < z_spine,
        "concave cone2 contact should retreat toward apex2: got ({r_c2}, {z_c2}) vs spine"
    );

    // Predicted chamfer apex (line P1-P2 extrapolated to r=0) and
    // half-angle. Two non-coaxial points on a cone determine apex z
    // and half-angle, but axis direction needs an explicit check.
    let dr = r_c2 - r_c1;
    let dz = z_c2 - z_c1;
    let expected_apex_z = z_c1 - r_c1 * dz / dr;
    let mid_z = 0.5 * (z_c1 + z_c2);
    let r_avg = 0.5 * (r_c1 + r_c2);
    let expected_beta = ((mid_z - expected_apex_z).abs() / r_avg).atan();

    let apex = chamfer_cone.apex();
    assert!(
        apex.x().abs() < 1e-12 && apex.y().abs() < 1e-12,
        "concave apex should be on z-axis, got {apex:?}"
    );
    assert!(
        (apex.z() - expected_apex_z).abs() < 1e-9,
        "concave apex z = {}, expected {expected_apex_z}",
        apex.z()
    );
    assert!(
        (chamfer_cone.half_angle() - expected_beta).abs() < 1e-9,
        "concave chamfer half-angle should be {expected_beta}, got {}",
        chamfer_cone.half_angle()
    );

    // Cone axis direction (apex below contacts ⇒ axis = +z, opens
    // upward). project_point would not catch a flipped axis with a
    // mirrored apex, so check the axis direction explicitly.
    let axis = chamfer_cone.axis();
    assert!(
        axis.dot(Vec3::new(0.0, 0.0, 1.0)) > 1.0 - 1e-12,
        "concave chamfer cone axis should be +z (apex below contacts), got {axis:?}"
    );

    // Both contacts on chamfer cone.
    let want_c1 = Point3::new(r_c1, 0.0, z_c1);
    let want_c2 = Point3::new(r_c2, 0.0, z_c2);
    let (u_p, v_p) = ParametricSurface::project_point(&chamfer_cone, want_c1);
    let on_cone_c1 = ParametricSurface::evaluate(&chamfer_cone, u_p, v_p);
    let (u_q, v_q) = ParametricSurface::project_point(&chamfer_cone, want_c2);
    let on_cone_c2 = ParametricSurface::evaluate(&chamfer_cone, u_q, v_q);
    assert!(
        (on_cone_c1 - want_c1).length() < 1e-9,
        "cone1 contact must lie on chamfer cone: {on_cone_c1:?} vs {want_c1:?}"
    );
    assert!(
        (on_cone_c2 - want_c2).length() < 1e-9,
        "cone2 contact must lie on chamfer cone: {on_cone_c2:?} vs {want_c2:?}"
    );

    // Both contacts on respective cone surfaces.
    let cot_b1 = beta1.cos() / beta1.sin();
    let cot_b2 = beta2.cos() / beta2.sin();
    let pred_r_c1 = z_c1 * cot_b1;
    let pred_r_c2 = (z_c2 - h_2) * cot_b2;
    assert!(
        (r_c1 - pred_r_c1).abs() < 1e-9,
        "cone1 contact must lie on cone1: predicted radial {pred_r_c1}, got {r_c1}"
    );
    assert!(
        (r_c2 - pred_r_c2).abs() < 1e-9,
        "cone2 contact must lie on cone2: predicted radial {pred_r_c2}, got {r_c2}"
    );
}

/// Cone-cone coaxial mixed-convexity chamfer: covers BOTH (s1=+1,
/// s2=−1) and (s1=−1, s2=+1). For each, contacts are on different
/// generator arms relative to the symmetric cases.
///
/// Reads emitted contact endpoints via `evaluate(t_start)`. Asserts
/// each emitted contact matches the analytic prediction
/// `(r_spine ∓ s_i·d·cos β_i, z_spine ∓ s_i·d·sin β_i)` (sharper
/// than just "lies on cone surface", which a degenerate impl
/// returning `(r_spine, z_spine)` for both contacts would
/// trivially satisfy via `r = (z − z_apex)·cot β`).
#[test]
fn cone_cone_coaxial_chamfer_mixed_emits_cone() {
    use brepkit_math::curves::Circle3D;
    use brepkit_math::surfaces::ConicalSurface;
    use brepkit_topology::edge::{Edge, EdgeCurve};
    use brepkit_topology::face::Face;
    use brepkit_topology::vertex::Vertex;
    use brepkit_topology::wire::{OrientedEdge, Wire};

    let beta1: f64 = std::f64::consts::PI / 3.0;
    let beta2: f64 = std::f64::consts::PI / 4.0;
    let h_2: f64 = 2.0;
    let d: f64 = 0.3;
    let sin_minus = (beta1 - beta2).sin();
    let z_spine = h_2 * beta2.cos() * beta1.sin() / sin_minus;
    let r_spine = z_spine * (beta1.cos() / beta1.sin());

    let run_case = |reverse_s1: bool, reverse_s2: bool| {
        let mut topo = Topology::new();
        let cone1 =
            ConicalSurface::new(Point3::new(0.0, 0.0, 0.0), Vec3::new(0.0, 0.0, 1.0), beta1)
                .unwrap();
        let cone2 =
            ConicalSurface::new(Point3::new(0.0, 0.0, h_2), Vec3::new(0.0, 0.0, 1.0), beta2)
                .unwrap();
        let spine_circle = Circle3D::new(
            Point3::new(0.0, 0.0, z_spine),
            Vec3::new(0.0, 0.0, 1.0),
            r_spine,
        )
        .unwrap();
        let v = topo.add_vertex(Vertex::new(Point3::new(r_spine, 0.0, z_spine), 1e-7));
        let eid = topo.add_edge(Edge::new(v, v, EdgeCurve::Circle(spine_circle)));
        let spine = Spine::from_single_edge(&topo, eid).unwrap();

        let w1 = topo.add_wire(Wire::new(vec![OrientedEdge::new(eid, true)], true).unwrap());
        let face1 = if reverse_s1 {
            topo.add_face(Face::new_reversed(
                w1,
                vec![],
                FaceSurface::Cone(cone1.clone()),
            ))
        } else {
            topo.add_face(Face::new(w1, vec![], FaceSurface::Cone(cone1.clone())))
        };
        let w2 = topo.add_wire(Wire::new(vec![OrientedEdge::new(eid, false)], true).unwrap());
        let face2 = if reverse_s2 {
            topo.add_face(Face::new_reversed(
                w2,
                vec![],
                FaceSurface::Cone(cone2.clone()),
            ))
        } else {
            topo.add_face(Face::new(w2, vec![], FaceSurface::Cone(cone2.clone())))
        };

        let result = cone_cone_coaxial_chamfer(&cone1, &cone2, &spine, &topo, d, d, face1, face2)
            .unwrap()
            .expect("mixed coaxial cone-cone chamfer should produce a stripe");

        // Sample EMITTED contact curves at start of domain.
        let (t1_start, _) = result.stripe.contact1.domain();
        let c1_point = result.stripe.contact1.evaluate(t1_start);
        let (t2_start, _) = result.stripe.contact2.domain();
        let c2_point = result.stripe.contact2.evaluate(t2_start);

        // Compute analytic predictions: contact_i moves along
        // generator_i from spine by s_i·d, retreating toward apex_i
        // when s_i=+1 and extending away when s_i=−1.
        let s1_signed = if reverse_s1 { -1.0_f64 } else { 1.0_f64 };
        let s2_signed = if reverse_s2 { -1.0_f64 } else { 1.0_f64 };
        let pred_c1_r = r_spine - s1_signed * d * beta1.cos();
        let pred_c1_z = z_spine - s1_signed * d * beta1.sin();
        let pred_c2_r = r_spine + s2_signed * d * beta2.cos();
        let pred_c2_z = z_spine + s2_signed * d * beta2.sin();

        let c1_radial = (c1_point.x().powi(2) + c1_point.y().powi(2)).sqrt();
        let c2_radial = (c2_point.x().powi(2) + c2_point.y().powi(2)).sqrt();
        assert!(
            (c1_radial - pred_c1_r).abs() < 1e-9 && (c1_point.z() - pred_c1_z).abs() < 1e-9,
            "({reverse_s1}, {reverse_s2}): contact1 should be at (r={pred_c1_r}, z={pred_c1_z}); \
                 got (r={c1_radial}, z={})",
            c1_point.z()
        );
        assert!(
            (c2_radial - pred_c2_r).abs() < 1e-9 && (c2_point.z() - pred_c2_z).abs() < 1e-9,
            "({reverse_s1}, {reverse_s2}): contact2 should be at (r={pred_c2_r}, z={pred_c2_z}); \
                 got (r={c2_radial}, z={})",
            c2_point.z()
        );

        // Emitted surface is a Cone with axis = −z (apex ABOVE
        // contacts). Distinct from the symmetric cases which have
        // axis = +z: in mixed configs both contacts retreat
        // /extend along generators with the SAME radial sign
        // (one s_i flips), so the chord between them slopes the
        // OPPOSITE way and the line P1−P2 extrapolates to r=0
        // ABOVE the contacts rather than below.
        let chamfer_cone = match result.stripe.surface {
            FaceSurface::Cone(ref c) => c,
            ref other => panic!(
                "({reverse_s1}, {reverse_s2}): expected Cone, got {}",
                other.type_tag()
            ),
        };
        let axis = chamfer_cone.axis();
        assert!(
            axis.dot(Vec3::new(0.0, 0.0, 1.0)) < -1.0 + 1e-12,
            "({reverse_s1}, {reverse_s2}): chamfer cone axis should be −z (mixed = apex above), got {axis:?}"
        );

        // Both contacts on the chamfer cone via project_point
        // round-trip (the impl chose them on its own surface, so
        // this is a regression-check only).
        let (u_p, v_p) = ParametricSurface::project_point(chamfer_cone, c1_point);
        let on_cone_p1 = ParametricSurface::evaluate(chamfer_cone, u_p, v_p);
        assert!(
            (on_cone_p1 - c1_point).length() < 1e-9,
            "({reverse_s1}, {reverse_s2}): contact1 must lie on chamfer cone"
        );
        let (u_q, v_q) = ParametricSurface::project_point(chamfer_cone, c2_point);
        let on_cone_p2 = ParametricSurface::evaluate(chamfer_cone, u_q, v_q);
        assert!(
            (on_cone_p2 - c2_point).length() < 1e-9,
            "({reverse_s1}, {reverse_s2}): contact2 must lie on chamfer cone"
        );
    };

    run_case(false, true); // (s1=+1, s2=-1)
    run_case(true, false); // (s1=-1, s2=+1)
}

/// Sphere-cylinder both-concave fillet: spherical cavity intersecting
/// a cylindrical hole. Both faces REVERSED ⇒ Q_sph = R_s − r,
/// Q_cyl = r_c − r (internal tangency on both surfaces).
///
/// For R_s=3, r_c=2 (sphere encloses cyl), both faces REVERSED,
/// r=0.4, +z spine:
///   - Q_sph = 2.6, Q_cyl = 1.6
///   - a_ball = √(Q_sph² − Q_cyl²) = √4.2 ≈ 2.049
///   - major = Q_cyl = 1.6 (concave fillet sits INSIDE cyl axially)
///   - Major < r_c (vs convex which has major = r_c + r > r_c) —
///     confirms internal-tangency reduction.
#[test]
fn sphere_cylinder_fillet_both_concave_emits_smaller_torus() {
    use brepkit_math::curves::Circle3D;
    use brepkit_math::surfaces::{CylindricalSurface, SphericalSurface};
    use brepkit_topology::edge::{Edge, EdgeCurve};
    use brepkit_topology::face::Face;
    use brepkit_topology::vertex::Vertex;
    use brepkit_topology::wire::{OrientedEdge, Wire};

    let mut topo = Topology::new();
    let big_r_s: f64 = 3.0;
    let r_c: f64 = 2.0;
    let r_fillet: f64 = 0.4;
    let h_s = (big_r_s * big_r_s - r_c * r_c).sqrt();

    let sph = SphericalSurface::new(Point3::new(0.0, 0.0, 0.0), big_r_s).unwrap();
    let cyl =
        CylindricalSurface::new(Point3::new(0.0, 0.0, 0.0), Vec3::new(0.0, 0.0, 1.0), r_c).unwrap();
    let spine_circle =
        Circle3D::new(Point3::new(0.0, 0.0, h_s), Vec3::new(0.0, 0.0, 1.0), r_c).unwrap();
    let v = topo.add_vertex(Vertex::new(Point3::new(r_c, 0.0, h_s), 1e-7));
    let eid = topo.add_edge(Edge::new(v, v, EdgeCurve::Circle(spine_circle)));
    let spine = Spine::from_single_edge(&topo, eid).unwrap();

    // Both faces REVERSED.
    let w1 = topo.add_wire(Wire::new(vec![OrientedEdge::new(eid, true)], true).unwrap());
    let face_sphere = topo.add_face(Face::new_reversed(
        w1,
        vec![],
        FaceSurface::Sphere(sph.clone()),
    ));
    let w2 = topo.add_wire(Wire::new(vec![OrientedEdge::new(eid, false)], true).unwrap());
    let face_cyl = topo.add_face(Face::new_reversed(
        w2,
        vec![],
        FaceSurface::Cylinder(cyl.clone()),
    ));

    let result = sphere_cylinder_fillet(&sph, &cyl, &spine, &topo, r_fillet, face_sphere, face_cyl)
        .unwrap()
        .expect("both-concave sphere-cylinder fillet should produce a stripe");

    let torus = match result.stripe.surface {
        FaceSurface::Torus(t) => t,
        other => panic!("expected Torus, got {}", other.type_tag()),
    };

    let q_sph = big_r_s - r_fillet;
    let q_cyl = r_c - r_fillet;
    let expected_a_ball = (q_sph * q_sph - q_cyl * q_cyl).sqrt();
    let expected_major = q_cyl;

    assert!(
        (torus.major_radius() - expected_major).abs() < 1e-12,
        "concave major should be Q_cyl = {expected_major}, got {}",
        torus.major_radius()
    );
    assert!(
        torus.major_radius() < r_c,
        "concave major ({}) must be < r_c = {r_c} (vs convex which would be > r_c)",
        torus.major_radius()
    );
    assert!(
        (torus.minor_radius() - r_fillet).abs() < 1e-12,
        "minor should equal r_fillet = {r_fillet}, got {}",
        torus.minor_radius()
    );

    // Torus center on +z axis at z = a_ball (positive since +z spine).
    let center = torus.center();
    assert!(
        center.x().abs() < 1e-12 && center.y().abs() < 1e-12,
        "torus center on z-axis, got {center:?}"
    );
    assert!(
        (center.z() - expected_a_ball).abs() < 1e-12,
        "torus center z should be a_ball = {expected_a_ball}, got {}",
        center.z()
    );

    // Verify ball is INSIDE both surfaces (internal tangency).
    let actual_dist_to_sphere_center =
        (center.x().powi(2) + center.y().powi(2) + center.z().powi(2)).sqrt();
    assert!(
        actual_dist_to_sphere_center < big_r_s - 1e-9,
        "concave: ball must be INSIDE sphere (distance {actual_dist_to_sphere_center} < R_s = {big_r_s})"
    );
    let actual_dist_to_cyl_axis = (center.x().powi(2) + center.y().powi(2)).sqrt();
    assert!(
        actual_dist_to_cyl_axis < r_c - 1e-9,
        "concave: ball must be INSIDE cyl (distance {actual_dist_to_cyl_axis} < r_c = {r_c})"
    );

    // Sphere contact at distance R_s from sphere center.
    let sph_axial = big_r_s * expected_a_ball / q_sph;
    let sph_radial = big_r_s * expected_major / q_sph;
    let want_sph = Point3::new(sph_radial, 0.0, sph_axial);
    let dist_sph = (want_sph - Point3::new(0.0, 0.0, 0.0)).length();
    assert!(
        (dist_sph - big_r_s).abs() < 1e-9,
        "sphere contact must lie on sphere: {dist_sph} vs R_s = {big_r_s}"
    );

    // Cylinder contact at radial r_c.
    let want_cyl = Point3::new(r_c, 0.0, expected_a_ball);
    let cyl_radial = (want_cyl.x().powi(2) + want_cyl.y().powi(2)).sqrt();
    assert!(
        (cyl_radial - r_c).abs() < 1e-9,
        "cyl contact must have radial r_c: got {cyl_radial}, want {r_c}"
    );

    // Both contacts lie on the torus.
    let (u_p, v_p) = ParametricSurface::project_point(&torus, want_sph);
    let on_torus_sph = ParametricSurface::evaluate(&torus, u_p, v_p);
    let (u_q, v_q) = ParametricSurface::project_point(&torus, want_cyl);
    let on_torus_cyl = ParametricSurface::evaluate(&torus, u_q, v_q);
    assert!(
        (on_torus_sph - want_sph).length() < 1e-9,
        "sphere contact on torus: {on_torus_sph:?} vs {want_sph:?}"
    );
    assert!(
        (on_torus_cyl - want_cyl).length() < 1e-9,
        "cyl contact on torus: {on_torus_cyl:?} vs {want_cyl:?}"
    );
}

/// Concave plane-cone chamfer: chamfering the top rim of a tapered hole.
///
/// Geometry: cone primitive (apex above plate at z=h, axis −z,
/// half-angle α) used as a hole tool through a plate at z=0. The
/// resulting solid has plate material at z<0 and the hole wall is the
/// (reversed) cone primitive's lateral face. We chamfer the spine where
/// hole wall meets plate top.
///
/// Detection: `axis_c · n_p_inward = (−z)·(+z) = −1` (antiparallel),
/// so `signed_offset = −1`. The chamfer cone's apex sits BELOW the
/// plate (z<0), its axis is +z, opening upward through the plate into
/// the empty wedge above the chamfer.
///
/// For α = π/4, h = 4 (so r_p = 4), and symmetric d1 = d2 = 1:
///   chamfer half-angle β = π/2 − α/2 = 3π/8 (independent of concave/convex)
///   plate-side contact at radius r_p + d1 = 5, z = 0
///   cone-side contact at radius r_p + d2·cos α ≈ 4.707, z = −d2·sin α ≈ −0.707
#[test]
fn plane_cone_chamfer_concave_emits_chamfer_cone() {
    use brepkit_math::curves::Circle3D;
    use brepkit_math::surfaces::ConicalSurface;
    use brepkit_topology::edge::{Edge, EdgeCurve};
    use brepkit_topology::face::Face;
    use brepkit_topology::vertex::Vertex;
    use brepkit_topology::wire::{OrientedEdge, Wire};

    let mut topo = Topology::new();
    let alpha: f64 = std::f64::consts::FRAC_PI_4;
    let h: f64 = 4.0;
    let r_p: f64 = h * (alpha.cos() / alpha.sin());
    let d: f64 = 1.0;

    let v = topo.add_vertex(Vertex::new(Point3::new(r_p, 0.0, 0.0), 1e-7));
    let circle = Circle3D::new(Point3::new(0.0, 0.0, 0.0), Vec3::new(0.0, 0.0, 1.0), r_p).unwrap();
    let eid = topo.add_edge(Edge::new(v, v, EdgeCurve::Circle(circle)));
    let spine = Spine::from_single_edge(&topo, eid).unwrap();

    let w1 = topo.add_wire(Wire::new(vec![OrientedEdge::new(eid, true)], true).unwrap());
    let face_plate = topo.add_face(Face::new(
        w1,
        vec![],
        FaceSurface::Plane {
            normal: Vec3::new(0.0, 0.0, 1.0),
            d: 0.0,
        },
    ));

    // Cone primitive: apex at (0,0,h), axis = −z, half-angle α. Used
    // here as the wall of a hole, so the FACE is reversed (topological
    // outward points into the empty hole).
    let cone_surf =
        ConicalSurface::new(Point3::new(0.0, 0.0, h), Vec3::new(0.0, 0.0, -1.0), alpha).unwrap();
    let w2 = topo.add_wire(Wire::new(vec![OrientedEdge::new(eid, false)], true).unwrap());
    let face_cone = topo.add_face(Face::new_reversed(
        w2,
        vec![],
        FaceSurface::Cone(cone_surf.clone()),
    ));

    let n_p_inward = Vec3::new(0.0, 0.0, 1.0);
    let result = plane_cone_chamfer(
        n_p_inward, 0.0, &cone_surf, &spine, &topo, d, d, face_plate, face_cone,
    )
    .unwrap()
    .expect("concave plane-cone chamfer should produce a stripe");

    let chamfer_cone = match result.stripe.surface {
        FaceSurface::Cone(c) => c,
        other => panic!("expected Cone, got {}", other.type_tag()),
    };

    // Symmetric chamfer half-angle: β = π/2 − α/2 = 3π/8.
    let expected_beta = std::f64::consts::FRAC_PI_2 - alpha * 0.5;
    assert!(
        (chamfer_cone.half_angle() - expected_beta).abs() < 1e-12,
        "chamfer cone half-angle should be 3π/8 for α=π/4, d1=d2; got {}",
        chamfer_cone.half_angle()
    );

    // Apex BELOW the plate at z = −(r_p+d) · dz/dr where dz = sin α,
    // dr = 1 − cos α (with d=1). For α=π/4: dz=√2/2, dr=1−√2/2,
    // ratio ≈ 2.414, mag ≈ 5·2.414 ≈ 12.07.
    let apex = chamfer_cone.apex();
    let dz = alpha.sin();
    let dr = 1.0 - alpha.cos();
    let expected_apex_z = -(r_p + d) * dz / dr;
    assert!(
        (apex.x()).abs() < 1e-12 && (apex.y()).abs() < 1e-12,
        "apex should lie on z-axis, got {apex:?}"
    );
    assert!(
        (apex.z() - expected_apex_z).abs() < 1e-9,
        "apex z = {}, expected {}",
        apex.z(),
        expected_apex_z
    );

    // Chamfer cone axis = +z (opens upward through the plate).
    let axis = chamfer_cone.axis();
    assert!(
        axis.dot(Vec3::new(0.0, 0.0, 1.0)) > 1.0 - 1e-12,
        "chamfer cone axis should be +z, got {axis:?}"
    );

    // Both contact points must lie on the chamfer cone surface. We
    // project each onto the chamfer cone and verify the resulting
    // surface point matches to high precision; this avoids depending
    // on the exact frame orientation chosen by `Frame3::from_normal`.
    let want_plate = Point3::new(r_p + d, 0.0, 0.0);
    let cone_contact_axial = -d * alpha.sin();
    let cone_contact_radial = r_p + d * alpha.cos();
    let want_cone = Point3::new(cone_contact_radial, 0.0, cone_contact_axial);
    let (u_p, v_p) = ParametricSurface::project_point(&chamfer_cone, want_plate);
    let on_surf_plate = ParametricSurface::evaluate(&chamfer_cone, u_p, v_p);
    let (u_c, v_c) = ParametricSurface::project_point(&chamfer_cone, want_cone);
    let on_surf_cone = ParametricSurface::evaluate(&chamfer_cone, u_c, v_c);
    assert!(
        (on_surf_plate - want_plate).length() < 1e-9,
        "plate contact must lie on chamfer cone: project→eval gave {on_surf_plate:?}, want {want_plate:?}"
    );
    assert!(
        (on_surf_cone - want_cone).length() < 1e-9,
        "cone-side contact must lie on chamfer cone: project→eval gave {on_surf_cone:?}, want {want_cone:?}"
    );
}
