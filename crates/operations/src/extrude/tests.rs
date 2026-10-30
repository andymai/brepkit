#![allow(clippy::unwrap_used, clippy::print_stderr)]

use std::collections::HashMap;

use brepkit_math::tolerance::Tolerance;
use brepkit_topology::Topology;
use brepkit_topology::face::FaceSurface;
use brepkit_topology::test_utils::{make_unit_square_face, make_unit_triangle_face};

use super::*;
use crate::test_helpers::assert_euler_genus0;

#[test]
fn extrude_square_creates_box() {
    let mut topo = Topology::new();
    let face = make_unit_square_face(&mut topo);

    let solid = extrude(&mut topo, face, Vec3::new(0.0, 0.0, 1.0), 1.0).unwrap();

    let solid_data = topo.solid(solid).unwrap();
    let shell = topo.shell(solid_data.outer_shell()).unwrap();

    // 4 sides + top + bottom = 6 faces
    assert_eq!(shell.faces().len(), 6);
    // 4 input + 4 top + 4 vertical = 12 edges (original input edges are reused)
    assert_eq!(topo.edges().len(), 12);
    // 4 input + 4 top = 8 vertices
    assert_eq!(topo.vertices().len(), 8);
}

#[test]
fn extrude_triangle_creates_prism() {
    let mut topo = Topology::new();
    let face = make_unit_triangle_face(&mut topo);

    let solid = extrude(&mut topo, face, Vec3::new(0.0, 0.0, 1.0), 1.0).unwrap();

    let solid_data = topo.solid(solid).unwrap();
    let shell = topo.shell(solid_data.outer_shell()).unwrap();

    // 3 sides + top + bottom = 5 faces
    assert_eq!(shell.faces().len(), 5);
    assert_eq!(topo.edges().len(), 9);
    assert_eq!(topo.vertices().len(), 6);
}

#[test]
fn extrude_rect_with_circular_hole_uses_exact_cylinder() {
    // gh #966: a rectangle with a circular hole, extruded, must remove the
    // exact analytic π·r²·h instead of an inscribed-polygon prism. The hole
    // wall becomes ONE cylinder face, not a faceted ring, and the measured
    // volume matches the analytic value.
    use brepkit_math::vec::Point3;
    use brepkit_topology::builder::{make_circle_edge, make_polygon_wire};
    use brepkit_topology::face::Face;
    use brepkit_topology::wire::{OrientedEdge, Wire};

    let tol = 1e-7;
    let mut topo = Topology::new();

    let outer = make_polygon_wire(
        &mut topo,
        &[
            Point3::new(0.0, 0.0, 0.0),
            Point3::new(20.0, 0.0, 0.0),
            Point3::new(20.0, 20.0, 0.0),
            Point3::new(0.0, 20.0, 0.0),
        ],
        tol,
    )
    .unwrap();
    let hole = make_circle_edge(
        &mut topo,
        Point3::new(10.0, 10.0, 0.0),
        Vec3::new(0.0, 0.0, 1.0),
        3.0,
        tol,
    )
    .unwrap();
    let inner = topo.add_wire(Wire::new(vec![OrientedEdge::new(hole, false)], true).unwrap());
    let face = topo.add_face(Face::new(
        outer,
        vec![inner],
        FaceSurface::Plane {
            normal: Vec3::new(0.0, 0.0, 1.0),
            d: 0.0,
        },
    ));

    let solid = extrude(&mut topo, face, Vec3::new(0.0, 0.0, 1.0), 10.0).unwrap();

    let shell = topo
        .shell(topo.solid(solid).unwrap().outer_shell())
        .unwrap();
    let cyl_count = shell
        .faces()
        .iter()
        .filter(|&&fid| matches!(topo.face(fid).unwrap().surface(), FaceSurface::Cylinder(_)))
        .count();
    assert_eq!(cyl_count, 1, "hole wall must be one exact cylinder face");

    let vol = crate::measure::solid_volume(&topo, solid, 0.01).unwrap();
    let expected = (400.0 - std::f64::consts::PI * 9.0) * 10.0;
    assert!(
        (vol - expected).abs() / expected < 1e-3,
        "expected ~{expected}, got {vol}"
    );
    assert!(
        crate::validate::validate_solid(&topo, solid)
            .unwrap()
            .is_valid()
    );
}

#[test]
fn extrude_zero_direction_error() {
    let mut topo = Topology::new();
    let face = make_unit_square_face(&mut topo);

    let result = extrude(&mut topo, face, Vec3::new(0.0, 0.0, 0.0), 1.0);
    assert!(result.is_err());
}

#[test]
fn extrude_zero_distance_error() {
    let mut topo = Topology::new();
    let face = make_unit_square_face(&mut topo);

    let result = extrude(&mut topo, face, Vec3::new(0.0, 0.0, 1.0), 0.0);
    assert!(result.is_err());
}

/// Verify that extruding a +Z face upward produces a solid where:
/// - The bottom face normal points -Z (outward-downward)
/// - The top face normal points +Z (outward-upward)
/// - All edges are shared by exactly 2 faces (manifold)
#[test]
fn extrude_orientation_correct() {
    let mut topo = Topology::new();
    let face = make_unit_square_face(&mut topo);
    let solid = extrude(&mut topo, face, Vec3::new(0.0, 0.0, 1.0), 1.0).unwrap();

    let tol = Tolerance::new();
    let solid_data = topo.solid(solid).unwrap();
    let shell = topo.shell(solid_data.outer_shell()).unwrap();

    let mut found_bottom = false;
    let mut found_top = false;
    for &fid in shell.faces() {
        let f = topo.face(fid).unwrap();
        if let FaceSurface::Plane { normal, .. } = f.surface() {
            // Bottom: normal ≈ (0, 0, -1)
            if tol.approx_eq(normal.z(), -1.0)
                && tol.approx_eq(normal.x(), 0.0)
                && tol.approx_eq(normal.y(), 0.0)
            {
                found_bottom = true;
            }
            // Top: normal ≈ (0, 0, 1)
            if tol.approx_eq(normal.z(), 1.0)
                && tol.approx_eq(normal.x(), 0.0)
                && tol.approx_eq(normal.y(), 0.0)
            {
                found_top = true;
            }
        }
    }
    assert!(found_bottom, "bottom face should have -Z normal");
    assert!(found_top, "top face should have +Z normal");

    // Verify manifold: every edge used by exactly 2 faces.
    let mut edge_counts: HashMap<usize, usize> = HashMap::new();
    for &fid in shell.faces() {
        let f = topo.face(fid).unwrap();
        let wire = topo.wire(f.outer_wire()).unwrap();
        for oe in wire.edges() {
            *edge_counts.entry(oe.edge().index()).or_insert(0) += 1;
        }
    }
    for (&edge_idx, &count) in &edge_counts {
        assert_eq!(
            count, 2,
            "edge {edge_idx} shared by {count} faces, expected 2"
        );
    }
}

// ── NURBS face extrusion tests ──────────────────────

/// Build a NURBS face: a curved surface on the XY plane.
fn make_nurbs_face(topo: &mut Topology) -> FaceId {
    use brepkit_math::nurbs::surface::NurbsSurface;

    // Bicubic surface with some curvature.
    let cps = vec![
        vec![Point3::new(0.0, 0.0, 0.0), Point3::new(1.0, 0.0, 0.0)],
        vec![Point3::new(0.0, 1.0, 0.5), Point3::new(1.0, 1.0, 0.5)],
    ];
    let weights = vec![vec![1.0, 1.0], vec![1.0, 1.0]];
    let knots = vec![0.0, 0.0, 1.0, 1.0];
    let surface = NurbsSurface::new(1, 1, knots.clone(), knots, cps, weights).unwrap();

    let tol = 1e-7;
    let v0 = topo.add_vertex(Vertex::new(Point3::new(0.0, 0.0, 0.0), tol));
    let v1 = topo.add_vertex(Vertex::new(Point3::new(1.0, 0.0, 0.0), tol));
    let v2 = topo.add_vertex(Vertex::new(Point3::new(1.0, 1.0, 0.5), tol));
    let v3 = topo.add_vertex(Vertex::new(Point3::new(0.0, 1.0, 0.5), tol));

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

    topo.add_face(Face::new(wid, vec![], FaceSurface::Nurbs(surface)))
}

#[test]
fn extrude_nurbs_face_creates_solid() {
    let mut topo = Topology::new();
    let face = make_nurbs_face(&mut topo);

    let solid = extrude(&mut topo, face, Vec3::new(0.0, 0.0, 1.0), 2.0).unwrap();

    let s = topo.solid(solid).unwrap();
    let sh = topo.shell(s.outer_shell()).unwrap();

    // 4 sides + top + bottom = 6 faces
    assert_eq!(
        sh.faces().len(),
        6,
        "extruded NURBS face should have 6 faces"
    );
}

#[test]
fn extrude_nurbs_face_top_is_nurbs() {
    let mut topo = Topology::new();
    let face = make_nurbs_face(&mut topo);

    let solid = extrude(&mut topo, face, Vec3::new(0.0, 0.0, 1.0), 2.0).unwrap();

    let s = topo.solid(solid).unwrap();
    let sh = topo.shell(s.outer_shell()).unwrap();

    // At least 2 faces should be NURBS (top and bottom caps).
    let nurbs_count = sh
        .faces()
        .iter()
        .filter(|&&fid| matches!(topo.face(fid).unwrap().surface(), FaceSurface::Nurbs(_)))
        .count();

    assert!(
        nurbs_count >= 2,
        "extruded NURBS face should have at least 2 NURBS caps, got {nurbs_count}"
    );
}

#[test]
fn extrude_nurbs_face_top_translated() {
    let mut topo = Topology::new();
    let face = make_nurbs_face(&mut topo);

    let solid = extrude(&mut topo, face, Vec3::new(0.0, 0.0, 1.0), 3.0).unwrap();

    let s = topo.solid(solid).unwrap();
    let sh = topo.shell(s.outer_shell()).unwrap();

    // Find the top NURBS face (the one at higher z).
    let mut top_z = f64::MIN;
    for &fid in sh.faces() {
        let f = topo.face(fid).unwrap();
        if let FaceSurface::Nurbs(surface) = f.surface() {
            let pt = surface.evaluate(0.0, 0.0);
            if pt.z() > top_z {
                top_z = pt.z();
            }
        }
    }

    // The top surface should be translated by distance 3.0 along Z.
    // Original (0,0) point is at z=0, so top should be at z=3.0.
    assert!(
        (top_z - 3.0).abs() < 1e-7,
        "top NURBS surface should be at z≈3.0, got z={top_z}"
    );
}

#[test]
fn extrude_nurbs_face_positive_volume() {
    let mut topo = Topology::new();
    let face = make_nurbs_face(&mut topo);

    let solid = extrude(&mut topo, face, Vec3::new(0.0, 0.0, 1.0), 2.0).unwrap();

    // The bilinear patch z = 0.5 y over the unit square, swept 2 along z:
    // every vertical line through the square crosses 2 of the solid.
    let vol = crate::measure::solid_volume(&topo, solid, 0.1).unwrap();
    assert!((vol - 2.0).abs() < 1e-9, "the sweep holds 2, got {vol}");

    assert_euler_genus0(&topo, solid);
}

/// Helper: create a square face with a smaller square hole in the center.
fn make_face_with_hole(topo: &mut Topology) -> FaceId {
    // Outer wire: 2×2 square centered at origin.
    let outer_pts = vec![
        Point3::new(-1.0, -1.0, 0.0),
        Point3::new(1.0, -1.0, 0.0),
        Point3::new(1.0, 1.0, 0.0),
        Point3::new(-1.0, 1.0, 0.0),
    ];
    let outer_wire = brepkit_topology::builder::make_polygon_wire(topo, &outer_pts, 1e-7).unwrap();

    // Inner wire: 0.5×0.5 square hole (CW winding = hole).
    let inner_pts = vec![
        Point3::new(-0.25, -0.25, 0.0),
        Point3::new(-0.25, 0.25, 0.0),
        Point3::new(0.25, 0.25, 0.0),
        Point3::new(0.25, -0.25, 0.0),
    ];
    let inner_wire = brepkit_topology::builder::make_polygon_wire(topo, &inner_pts, 1e-7).unwrap();

    let normal = Vec3::new(0.0, 0.0, 1.0);
    let d = 0.0;
    let face = Face::new(
        outer_wire,
        vec![inner_wire],
        FaceSurface::Plane { normal, d },
    );
    topo.add_face(face)
}

#[test]
fn extrude_face_with_hole_produces_more_faces() {
    let mut topo = Topology::new();
    let face = make_face_with_hole(&mut topo);

    let solid = extrude(&mut topo, face, Vec3::new(0.0, 0.0, 1.0), 1.0).unwrap();

    let solid_data = topo.solid(solid).unwrap();
    let shell = topo.shell(solid_data.outer_shell()).unwrap();

    // Outer: 4 sides + top + bottom = 6
    // Inner: 4 inward-facing sides = 4
    // Total: 10 faces
    assert_eq!(
        shell.faces().len(),
        10,
        "extruded face with hole should have 10 faces (6 outer + 4 inner)"
    );
}

#[test]
fn extrude_face_with_hole_caps_have_inner_wires() {
    let mut topo = Topology::new();
    let face = make_face_with_hole(&mut topo);

    let solid = extrude(&mut topo, face, Vec3::new(0.0, 0.0, 1.0), 1.0).unwrap();

    let solid_data = topo.solid(solid).unwrap();
    let shell = topo.shell(solid_data.outer_shell()).unwrap();

    // The bottom and top faces should have inner wires (holes).
    let faces_with_holes_count = shell
        .faces()
        .iter()
        .filter(|&&fid| !topo.face(fid).unwrap().inner_wires().is_empty())
        .count();

    assert_eq!(
        faces_with_holes_count, 2,
        "bottom and top caps should both have inner wire holes"
    );
}

#[test]
fn extrude_zero_distance_errors() {
    let mut topo = Topology::new();
    let face = make_unit_square_face(&mut topo);
    let result = extrude(&mut topo, face, Vec3::new(0.0, 0.0, 1.0), 0.0);
    assert!(result.is_err(), "zero distance extrusion should error");
}

#[test]
fn extrude_face_with_hole_has_correct_volume() {
    let mut topo_solid = Topology::new();
    let solid_face = make_unit_square_face(&mut topo_solid);
    let solid_box = extrude(&mut topo_solid, solid_face, Vec3::new(0.0, 0.0, 1.0), 1.0).unwrap();

    let mut topo_hollow = Topology::new();
    let hollow_face = make_face_with_hole(&mut topo_hollow);
    let hollow_solid =
        extrude(&mut topo_hollow, hollow_face, Vec3::new(0.0, 0.0, 1.0), 1.0).unwrap();

    let vol_solid = crate::measure::solid_volume(&topo_solid, solid_box, 0.1).unwrap();
    let vol_hollow = crate::measure::solid_volume(&topo_hollow, hollow_solid, 0.1).unwrap();

    // Solid box: 1×1×1 = 1.0 exactly.
    let rel_solid = (vol_solid - 1.0).abs() / 1.0;
    assert!(
        rel_solid < 1e-8,
        "unit box volume should be 1.0, got {vol_solid} (rel_err={rel_solid:.2e})"
    );

    // Hollow: outer 2×2×1 = 4.0, hole 0.5×0.5×1 = 0.25, net = 3.75.
    //
    // Note: the signed-tetrahedra volume method for solids with inner
    // wires (holes) uses `volume_from_direct_face_tessellation`, which
    // relies on correct face winding from `tessellate()`. The hole
    // subtraction accuracy depends on the inner wall face orientations.
    let expected_hollow = 3.75;
    let rel_hollow = (vol_hollow - expected_hollow).abs() / expected_hollow;
    assert!(
        rel_hollow < 0.01,
        "hollow extrusion volume should be {expected_hollow}, got {vol_hollow} \
             (rel_err={rel_hollow:.2e}). If > 1%, inner-wire volume subtraction may be buggy."
    );
}

/// Extrude a square along +Z by distance 5 → volume = 1×1×5 = 5.0.
#[test]
fn extrude_square_volume_exact() {
    let mut topo = Topology::new();
    let face = make_unit_square_face(&mut topo);
    let solid = extrude(&mut topo, face, Vec3::new(0.0, 0.0, 1.0), 5.0).unwrap();

    let vol = crate::measure::solid_volume(&topo, solid, 0.1).unwrap();
    // All-planar: exact to floating-point.
    let rel_err = (vol - 5.0).abs() / 5.0;
    assert!(
        rel_err < 1e-8,
        "extruded unit square by 5 should have volume 5.0, got {vol} (rel_err={rel_err:.2e})"
    );
}

/// Extrude a triangle by 3 → volume = (base_area × height) = (0.5 × 3) = 1.5.
#[test]
fn extrude_triangle_volume_exact() {
    let mut topo = Topology::new();
    let face = make_unit_triangle_face(&mut topo);
    let solid = extrude(&mut topo, face, Vec3::new(0.0, 0.0, 1.0), 3.0).unwrap();

    let vol = crate::measure::solid_volume(&topo, solid, 0.1).unwrap();
    // Unit triangle area = 0.5, height = 3.0 → V = 1.5.
    let expected = 1.5;
    let rel_err = (vol - expected).abs() / expected;
    assert!(
        rel_err < 1e-8,
        "extruded unit triangle by 3 should have volume {expected}, got {vol} (rel_err={rel_err:.2e})"
    );
}

/// Extrude in a non-axis-aligned direction.
///
/// `extrude()` does NOT normalize the direction: offset = direction × distance.
/// With direction=(1,0,1) and distance=2, offset = (2,0,2).
///
/// For a sheared prism, volume = base_area × |offset · face_normal|.
/// Face normal is (0,0,1) (XY plane), so V = 1.0 × |2| = 2.0.
#[test]
fn extrude_oblique_direction_volume() {
    let mut topo = Topology::new();
    let face = make_unit_square_face(&mut topo);
    let solid = extrude(&mut topo, face, Vec3::new(1.0, 0.0, 1.0), 2.0).unwrap();

    let vol = crate::measure::solid_volume(&topo, solid, 0.1).unwrap();
    // offset = (1,0,1)*2 = (2,0,2). Height along Z (face normal) = 2.0.
    // V = base_area × height = 1.0 × 2.0 = 2.0.
    let expected = 2.0;
    let rel_err = (vol - expected).abs() / expected;
    assert!(
        rel_err < 1e-8,
        "oblique extrusion volume should be {expected}, got {vol} (rel_err={rel_err:.2e})"
    );
}

/// Reproduce brepjs compound extrude: 20×20 rectangle with a circular
/// polygon hole (CCW winding, 32 segments, radius 3), extruded by 10.
#[test]
fn extrude_face_with_ccw_circle_hole_volume() {
    let mut topo = Topology::new();

    // Outer wire: 20×20 rectangle (CCW).
    let outer_pts = vec![
        Point3::new(0.0, 0.0, 0.0),
        Point3::new(20.0, 0.0, 0.0),
        Point3::new(20.0, 20.0, 0.0),
        Point3::new(0.0, 20.0, 0.0),
    ];
    let outer_wire =
        brepkit_topology::builder::make_polygon_wire(&mut topo, &outer_pts, 1e-7).unwrap();

    // Inner wire: 32-segment polygon circle at center (10,10), radius 3.
    // CCW winding (standard math convention: cos/sin going counter-clockwise).
    let n_segments = 32;
    let cx = 10.0;
    let cy = 10.0;
    let r = 3.0;
    let inner_pts: Vec<Point3> = (0..n_segments)
        .map(|i| {
            #[allow(clippy::cast_precision_loss)]
            let theta = 2.0 * std::f64::consts::PI * (i as f64) / (n_segments as f64);
            Point3::new(cx + r * theta.cos(), cy + r * theta.sin(), 0.0)
        })
        .collect();
    let inner_wire =
        brepkit_topology::builder::make_polygon_wire(&mut topo, &inner_pts, 1e-7).unwrap();

    let normal = Vec3::new(0.0, 0.0, 1.0);
    let face = Face::new(
        outer_wire,
        vec![inner_wire],
        FaceSurface::Plane { normal, d: 0.0 },
    );
    let face_id = topo.add_face(face);

    let solid = extrude(&mut topo, face_id, Vec3::new(0.0, 0.0, 1.0), 10.0).unwrap();

    let vol = crate::measure::solid_volume(&topo, solid, 0.1).unwrap();

    // Expected: 20*20*10 - polygon_area*10
    // Polygon area of regular 32-gon inscribed in circle r=3:
    // A = n * r^2 * sin(2*pi/n) / 2 = 32 * 9 * sin(pi/16) / 2 ≈ 27.86
    // Expected volume ≈ 4000 - 278.6 = 3721.4
    let polygon_area: f64 = (0..n_segments)
        .map(|i| {
            #[allow(clippy::cast_precision_loss)]
            let theta1 = 2.0 * std::f64::consts::PI * (i as f64) / (n_segments as f64);
            #[allow(clippy::cast_precision_loss)]
            let theta2 = 2.0 * std::f64::consts::PI * ((i + 1) as f64) / (n_segments as f64);
            0.5 * r * r * (theta2 - theta1).sin().abs()
        })
        .sum();
    let expected = 20.0 * 20.0 * 10.0 - polygon_area * 10.0;
    let rel_err = (vol - expected).abs() / expected;
    assert!(
        rel_err < 0.01,
        "CCW circle hole extrusion volume should be ~{expected:.1}, got {vol:.1} \
             (rel_err={rel_err:.2e})"
    );
}

#[test]
fn extrude_analytic_circle_hole_volume_is_exact() {
    use brepkit_math::curves::Circle3D;

    // 20×20 plate with a true (analytic) circular hole, radius 3, extruded by
    // 10. The hole is four quarter-circle arc edges (how a full circle is
    // lifted), so the extruded wall is four CylindricalSurface faces and the
    // volume is exactly box − cylinder = 4000 − π·9·10. Tessellation (and a
    // chord-polygonised cap) mis-count the curved boundary by ~1-2%; exact
    // divergence-theorem integration over the analytic faces must hit it.
    let mut topo = Topology::new();

    let outer_pts = vec![
        Point3::new(0.0, 0.0, 0.0),
        Point3::new(20.0, 0.0, 0.0),
        Point3::new(20.0, 20.0, 0.0),
        Point3::new(0.0, 20.0, 0.0),
    ];
    let outer_wire =
        brepkit_topology::builder::make_polygon_wire(&mut topo, &outer_pts, 1e-7).unwrap();

    let (cx, cy, r) = (10.0, 10.0, 3.0);
    let mk = |topo: &mut Topology, x: f64, y: f64| {
        topo.add_vertex(Vertex::new(Point3::new(x, y, 0.0), 1e-7))
    };
    let p = [
        mk(&mut topo, cx + r, cy),
        mk(&mut topo, cx, cy + r),
        mk(&mut topo, cx - r, cy),
        mk(&mut topo, cx, cy - r),
    ];
    let circ = || {
        Circle3D::with_axes(
            Point3::new(cx, cy, 0.0),
            Vec3::new(0.0, 0.0, 1.0),
            r,
            Vec3::new(1.0, 0.0, 0.0),
            Vec3::new(0.0, 1.0, 0.0),
        )
        .unwrap()
    };
    let arcs: Vec<_> = (0..4)
        .map(|i| {
            let e = topo.add_edge(Edge::new(p[i], p[(i + 1) % 4], EdgeCurve::Circle(circ())));
            OrientedEdge::new(e, true)
        })
        .collect();
    let inner_wire = topo.add_wire(Wire::new(arcs, true).unwrap());

    let face = topo.add_face(Face::new(
        outer_wire,
        vec![inner_wire],
        FaceSurface::Plane {
            normal: Vec3::new(0.0, 0.0, 1.0),
            d: 0.0,
        },
    ));

    let solid = extrude(&mut topo, face, Vec3::new(0.0, 0.0, 1.0), 10.0).unwrap();
    // Pass a deliberately COARSE deflection: solid_volume must internally clamp
    // it fine enough to integrate the curved hole wall accurately. Without that
    // clamp the inscribed mesh under-counts the wall and the volume is ~1-2% off.
    let vol = crate::measure::solid_volume(&topo, solid, 0.5).unwrap();

    let expected = 20.0 * 20.0 * 10.0 - std::f64::consts::PI * r * r * 10.0;
    let rel_err = (vol - expected).abs() / expected;
    assert!(
        rel_err < 1e-3,
        "analytic circular-hole volume should be ~{expected:.4}, got {vol:.4} \
             (rel_err={rel_err:.2e}) — coarse deflection must be clamped fine"
    );
}

#[test]
fn extrude_cw_profile_produces_correct_solid() {
    crate::test_helpers::assert_cw_profile_produces_valid_solid(
        |topo, face| extrude(topo, face, Vec3::new(0.0, 0.0, 1.0), 2.0).unwrap(),
        2.0,
        0.01,
    );
}

/// Translation invariance: signed mesh volume should not change when
/// the solid is translated far from the origin.
#[test]
fn extrude_cw_profile_translation_invariant() {
    use brepkit_topology::test_utils::make_cw_unit_square_face;

    // Build solid at origin
    let mut topo1 = Topology::new();
    let face1 = make_cw_unit_square_face(&mut topo1);
    let solid1 = extrude(&mut topo1, face1, Vec3::new(0.0, 0.0, 1.0), 2.0).unwrap();
    let vol1 = crate::measure::solid_volume(&topo1, solid1, 0.1).unwrap();

    // Build solid translated by (1000, 1000, 1000)
    let mut topo2 = Topology::new();
    let face2 = make_cw_unit_square_face(&mut topo2);
    let solid2 = extrude(&mut topo2, face2, Vec3::new(0.0, 0.0, 1.0), 2.0).unwrap();
    crate::transform::transform_solid(
        &mut topo2,
        solid2,
        &brepkit_math::mat::Mat4::translation(1000.0, 1000.0, 1000.0),
    )
    .unwrap();
    let vol2 = crate::measure::solid_volume(&topo2, solid2, 0.1).unwrap();

    let rel_err = (vol1 - vol2).abs() / vol1.max(1e-12);
    assert!(
        rel_err < 0.01,
        "CW extrusion volumes should match: origin={vol1}, translated={vol2}, \
             rel_err={rel_err:.2e}"
    );
}

/// Extruding a face with a NURBS curve edge that is geometrically a
/// circular arc should produce a `Cylinder` side face (not a ruled
/// NURBS surface). brepjs
/// constructs circles as rational-quadratic NURBS, and extruding those
/// as NURBS leaves a small but persistent volume deficit. Recognizing
/// the curve as a circle and using the exact analytic cylinder surface
/// recovers π·r²·h precisely.
///
/// For a NURBS that is *not* recognized as any analytic curve, the
/// behavior is still to produce a NURBS ruled surface (covered by the
/// rest of the side_face_surface match arm).
#[test]
fn extrude_recognized_circle_nurbs_arc_uses_cylinder_side_face() {
    use brepkit_math::nurbs::curve::NurbsCurve;

    let mut topo = Topology::new();
    let tol = Tolerance::new();

    let v0 = topo.add_vertex(Vertex::new(Point3::new(0.0, 0.0, 0.0), tol.linear));
    let v1 = topo.add_vertex(Vertex::new(Point3::new(1.0, 0.0, 0.0), tol.linear));
    let v2 = topo.add_vertex(Vertex::new(Point3::new(1.0, 1.0, 0.0), tol.linear));
    let v3 = topo.add_vertex(Vertex::new(Point3::new(0.0, 1.0, 0.0), tol.linear));

    let arc_curve = NurbsCurve::new(
        2,
        vec![0.0, 0.0, 0.0, 1.0, 1.0, 1.0],
        vec![
            Point3::new(1.0, 0.0, 0.0),
            Point3::new(1.5, 0.5, 0.0),
            Point3::new(1.0, 1.0, 0.0),
        ],
        vec![1.0, std::f64::consts::FRAC_1_SQRT_2, 1.0],
    )
    .unwrap();

    let e0 = topo.add_edge(Edge::new(v0, v1, EdgeCurve::Line));
    let e1 = topo.add_edge(Edge::new(v1, v2, EdgeCurve::NurbsCurve(arc_curve)));
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
    let wire_id = topo.add_wire(wire);
    let face = topo.add_face(Face::new(
        wire_id,
        vec![],
        FaceSurface::Plane {
            normal: Vec3::new(0.0, 0.0, 1.0),
            d: 0.0,
        },
    ));

    let solid = extrude(&mut topo, face, Vec3::new(0.0, 0.0, 1.0), 2.0).unwrap();

    let s = topo.solid(solid).unwrap();
    let sh = topo.shell(s.outer_shell()).unwrap();

    assert_eq!(sh.faces().len(), 6, "should have 6 faces");

    let nurbs_count = sh
        .faces()
        .iter()
        .filter(|&&fid| matches!(topo.face(fid).unwrap().surface(), FaceSurface::Nurbs(_)))
        .count();
    let cylinder_count = sh
        .faces()
        .iter()
        .filter(|&&fid| matches!(topo.face(fid).unwrap().surface(), FaceSurface::Cylinder(_)))
        .count();
    assert_eq!(
        cylinder_count, 1,
        "rational-quadratic circular arc should produce 1 Cylinder side face"
    );
    assert_eq!(
        nurbs_count, 0,
        "no NURBS side face expected once the arc is recognized as a Circle"
    );
}

/// Extruding a face with a Circle edge should produce a cylindrical side face.
#[test]
fn extrude_circle_edge_produces_cylinder_side_face() {
    use brepkit_math::curves::Circle3D;

    let mut topo = Topology::new();
    let tol = Tolerance::new();

    // Build a face: three line edges + one semicircular arc edge.
    let v0 = topo.add_vertex(Vertex::new(Point3::new(0.0, 0.0, 0.0), tol.linear));
    let v1 = topo.add_vertex(Vertex::new(Point3::new(1.0, 0.0, 0.0), tol.linear));
    let v2 = topo.add_vertex(Vertex::new(Point3::new(1.0, 1.0, 0.0), tol.linear));
    let v3 = topo.add_vertex(Vertex::new(Point3::new(0.0, 1.0, 0.0), tol.linear));

    let circle = Circle3D::with_axes(
        Point3::new(1.0, 0.5, 0.0),
        Vec3::new(0.0, 0.0, 1.0),
        0.5,
        Vec3::new(0.0, -1.0, 0.0),
        Vec3::new(1.0, 0.0, 0.0),
    )
    .unwrap();

    let e0 = topo.add_edge(Edge::new(v0, v1, EdgeCurve::Line));
    let e1 = topo.add_edge(Edge::new(v1, v2, EdgeCurve::Circle(circle)));
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
    let wire_id = topo.add_wire(wire);
    let face = topo.add_face(Face::new(
        wire_id,
        vec![],
        FaceSurface::Plane {
            normal: Vec3::new(0.0, 0.0, 1.0),
            d: 0.0,
        },
    ));

    let solid = extrude(&mut topo, face, Vec3::new(0.0, 0.0, 1.0), 3.0).unwrap();

    let s = topo.solid(solid).unwrap();
    let sh = topo.shell(s.outer_shell()).unwrap();

    let cyl_count = sh
        .faces()
        .iter()
        .filter(|&&fid| matches!(topo.face(fid).unwrap().surface(), FaceSurface::Cylinder(_)))
        .count();
    assert_eq!(
        cyl_count, 1,
        "exactly 1 side face should be a cylinder, got {cyl_count}"
    );
}

/// NURBS arc extrude → volume should match analytic formula.
#[test]
fn nurbs_arc_extrude_volume() {
    use brepkit_math::nurbs::fitting::interpolate;
    use brepkit_math::tolerance::Tolerance;

    let mut topo = Topology::new();
    let tol = Tolerance::new();

    let w = 41.5_f64;
    let d = 41.5_f64;
    let r = 3.75_f64;
    let h = 21.0_f64;
    let hw = w / 2.0;
    let hd = d / 2.0;

    // Quarter circle via NURBS interpolation (same path as brepjs)
    let make_arc_nurbs = |cx: f64, cy: f64, start_angle: f64| {
        let n_pts = 24;
        let mut pts = Vec::new();
        for i in 0..=n_pts {
            let angle = start_angle + std::f64::consts::FRAC_PI_2 * i as f64 / n_pts as f64;
            pts.push(Point3::new(cx + r * angle.cos(), cy + r * angle.sin(), 0.0));
        }
        interpolate(&pts, 3).unwrap()
    };

    let positions = [
        Point3::new(hw - r, -hd, 0.0),
        Point3::new(hw, -hd + r, 0.0),
        Point3::new(hw, hd - r, 0.0),
        Point3::new(hw - r, hd, 0.0),
        Point3::new(-hw + r, hd, 0.0),
        Point3::new(-hw, hd - r, 0.0),
        Point3::new(-hw, -hd + r, 0.0),
        Point3::new(-hw + r, -hd, 0.0),
    ];

    let vids: Vec<_> = positions
        .iter()
        .map(|p| topo.add_vertex(Vertex::new(*p, tol.linear)))
        .collect();

    let arcs = [
        make_arc_nurbs(hw - r, -hd + r, -std::f64::consts::FRAC_PI_2),
        make_arc_nurbs(hw - r, hd - r, 0.0),
        make_arc_nurbs(-hw + r, hd - r, std::f64::consts::FRAC_PI_2),
        make_arc_nurbs(-hw + r, -hd + r, std::f64::consts::PI),
    ];

    let e_bot = topo.add_edge(Edge::new(vids[7], vids[0], EdgeCurve::Line));
    let e_br = topo.add_edge(Edge::new(
        vids[0],
        vids[1],
        EdgeCurve::NurbsCurve(arcs[0].clone()),
    ));
    let e_right = topo.add_edge(Edge::new(vids[1], vids[2], EdgeCurve::Line));
    let e_tr = topo.add_edge(Edge::new(
        vids[2],
        vids[3],
        EdgeCurve::NurbsCurve(arcs[1].clone()),
    ));
    let e_top = topo.add_edge(Edge::new(vids[3], vids[4], EdgeCurve::Line));
    let e_tl = topo.add_edge(Edge::new(
        vids[4],
        vids[5],
        EdgeCurve::NurbsCurve(arcs[2].clone()),
    ));
    let e_left = topo.add_edge(Edge::new(vids[5], vids[6], EdgeCurve::Line));
    let e_bl = topo.add_edge(Edge::new(
        vids[6],
        vids[7],
        EdgeCurve::NurbsCurve(arcs[3].clone()),
    ));

    let wire = Wire::new(
        vec![
            OrientedEdge::new(e_bot, true),
            OrientedEdge::new(e_br, true),
            OrientedEdge::new(e_right, true),
            OrientedEdge::new(e_tr, true),
            OrientedEdge::new(e_top, true),
            OrientedEdge::new(e_tl, true),
            OrientedEdge::new(e_left, true),
            OrientedEdge::new(e_bl, true),
        ],
        true,
    )
    .unwrap();
    let wire_id = topo.add_wire(wire);

    let normal = Vec3::new(0.0, 0.0, 1.0);
    let face = Face::new(wire_id, vec![], FaceSurface::Plane { normal, d: 0.0 });
    let face_id = topo.add_face(face);

    let solid = crate::extrude::extrude(&mut topo, face_id, Vec3::new(0.0, 0.0, 1.0), h).unwrap();

    let sh = topo
        .shell(topo.solid(solid).unwrap().outer_shell())
        .unwrap();
    assert_eq!(sh.faces().len(), 10, "should have 10 faces");

    // Also try Circle3D edges for comparison
    let vol_circle = {
        let mut t2 = Topology::new();
        let tol2 = Tolerance::new();
        let z_axis = Vec3::new(0.0, 0.0, 1.0);
        let vids2: Vec<_> = positions
            .iter()
            .map(|p| t2.add_vertex(Vertex::new(*p, tol2.linear)))
            .collect();
        let mk_line2 = |topo: &mut Topology, s, e| topo.add_edge(Edge::new(s, e, EdgeCurve::Line));
        let mk_arc2 = |topo: &mut Topology, s, e, center: Point3| {
            let circle = brepkit_math::curves::Circle3D::new(center, z_axis, r).unwrap();
            topo.add_edge(Edge::new(s, e, EdgeCurve::Circle(circle)))
        };
        let e2_bot = mk_line2(&mut t2, vids2[7], vids2[0]);
        let e2_br = mk_arc2(
            &mut t2,
            vids2[0],
            vids2[1],
            Point3::new(hw - r, -hd + r, 0.0),
        );
        let e2_right = mk_line2(&mut t2, vids2[1], vids2[2]);
        let e2_tr = mk_arc2(
            &mut t2,
            vids2[2],
            vids2[3],
            Point3::new(hw - r, hd - r, 0.0),
        );
        let e2_top = mk_line2(&mut t2, vids2[3], vids2[4]);
        let e2_tl = mk_arc2(
            &mut t2,
            vids2[4],
            vids2[5],
            Point3::new(-hw + r, hd - r, 0.0),
        );
        let e2_left = mk_line2(&mut t2, vids2[5], vids2[6]);
        let e2_bl = mk_arc2(
            &mut t2,
            vids2[6],
            vids2[7],
            Point3::new(-hw + r, -hd + r, 0.0),
        );
        let wire2 = Wire::new(
            vec![
                OrientedEdge::new(e2_bot, true),
                OrientedEdge::new(e2_br, true),
                OrientedEdge::new(e2_right, true),
                OrientedEdge::new(e2_tr, true),
                OrientedEdge::new(e2_top, true),
                OrientedEdge::new(e2_tl, true),
                OrientedEdge::new(e2_left, true),
                OrientedEdge::new(e2_bl, true),
            ],
            true,
        )
        .unwrap();
        let wid2 = t2.add_wire(wire2);
        let face2 = Face::new(wid2, vec![], FaceSurface::Plane { normal, d: 0.0 });
        let fid2 = t2.add_face(face2);
        let solid2 = crate::extrude::extrude(&mut t2, fid2, Vec3::new(0.0, 0.0, 1.0), h).unwrap();
        crate::measure::solid_volume(&t2, solid2, 0.01).unwrap()
    };
    eprintln!("[nurbs_arc] Circle3D volume: {vol_circle:.2}");

    // Debug: per-face volume contribution via signed tetrahedra
    for &fid in sh.faces() {
        let f = topo.face(fid).unwrap();
        let kind = match f.surface() {
            FaceSurface::Plane { .. } => "Plane",
            FaceSurface::Nurbs(_) => "Nurbs",
            FaceSurface::Cylinder(_) => "Cylinder",
            _ => "Other",
        };
        let w2 = topo.wire(f.outer_wire()).unwrap();
        let mesh = crate::tessellate::tessellate(&topo, fid, 0.01).unwrap();
        let tri_count = mesh.indices.len() / 3;
        let mut face_vol = 0.0_f64;
        for t in 0..tri_count {
            let v0 = mesh.positions[mesh.indices[t * 3] as usize];
            let v1 = mesh.positions[mesh.indices[t * 3 + 1] as usize];
            let v2 = mesh.positions[mesh.indices[t * 3 + 2] as usize];
            let a = Vec3::new(v0.x(), v0.y(), v0.z());
            let b = Vec3::new(v1.x(), v1.y(), v1.z());
            let c = Vec3::new(v2.x(), v2.y(), v2.z());
            face_vol += a.dot(b.cross(c));
        }
        face_vol /= 6.0;
        eprintln!(
            "[nurbs_arc] Face {}: {kind}, {} edges, {} tris, vol_contrib={face_vol:.2}",
            fid.index(),
            w2.edges().len(),
            tri_count
        );
    }

    let vol = crate::measure::solid_volume(&topo, solid, 0.01).unwrap();
    let corner_area = (4.0 - std::f64::consts::PI) * r * r;
    let expected_vol = (w * d - corner_area) * h;
    let pct = (vol - expected_vol).abs() / expected_vol;
    eprintln!(
        "[nurbs_arc] Volume: {vol:.2}, expected: {expected_vol:.2}, err: {:.2}%",
        pct * 100.0
    );
    assert!(
        pct < 0.05,
        "NURBS arc extrude volume error too large: {vol:.2} vs {expected_vol:.2} ({:.2}%)",
        pct * 100.0
    );
}

/// Extruding a two-edge profile of one circular arc plus a closing diameter
/// (a half-disk) must orient the cylindrical side face outward, so the swept
/// volume matches the half-disk × height. This shares the winding-detection
/// path with the half-ellipse case (#869): a two-vertex loop whose endpoints
/// alone give zero signed area.
#[test]
fn extrude_half_circle_face_volume() {
    use brepkit_math::curves::Circle3D;
    use brepkit_math::vec::{Point3, Vec3};

    let mut topo = Topology::new();
    let tol = Tolerance::new();
    let r = 5.0_f64;
    // Half-circle arc (0,0) -> (5,5) apex -> (10,0), center (5,0,0), bulging +y.
    let v0 = topo.add_vertex(Vertex::new(Point3::new(0.0, 0.0, 0.0), tol.linear));
    let v1 = topo.add_vertex(Vertex::new(Point3::new(10.0, 0.0, 0.0), tol.linear));
    let circle = Circle3D::with_axes(
        Point3::new(5.0, 0.0, 0.0),
        Vec3::new(0.0, 0.0, 1.0),
        r,
        Vec3::new(-1.0, 0.0, 0.0),
        Vec3::new(0.0, 1.0, 0.0),
    )
    .unwrap();
    let arc = topo.add_edge(Edge::new(v0, v1, EdgeCurve::Circle(circle)));
    let line = topo.add_edge(Edge::new(v1, v0, EdgeCurve::Line));
    let wire = Wire::new(
        vec![OrientedEdge::new(arc, true), OrientedEdge::new(line, true)],
        true,
    )
    .unwrap();
    let wid = topo.add_wire(wire);
    let face = topo.add_face(Face::new(
        wid,
        vec![],
        FaceSurface::Plane {
            normal: Vec3::new(0.0, 0.0, 1.0),
            d: 0.0,
        },
    ));
    let expected = 0.5 * std::f64::consts::PI * r * r;
    let solid = extrude(&mut topo, face, Vec3::new(0.0, 0.0, 1.0), 1.0).unwrap();
    let vol = crate::measure::solid_volume(&topo, solid, 0.001).unwrap();
    assert!(
        (vol - expected).abs() / expected < 0.01,
        "extruded half-circle volume {vol:.4} != {expected:.4}"
    );
}

#[test]
fn extrude_half_circle_reversed_edge_volume() {
    use brepkit_math::curves::Circle3D;
    use brepkit_math::vec::{Point3, Vec3};

    // Same upper half-disc as `extrude_half_circle_face_volume`, but the arc is
    // used REVERSED in the wire — guards the Circle arm of `EdgeCurve::reversed`
    // (a reversed circle arc would otherwise sweep the complementary lower arc).
    let mut topo = Topology::new();
    let tol = Tolerance::new();
    let r = 5.0_f64;
    let v0 = topo.add_vertex(Vertex::new(Point3::new(0.0, 0.0, 0.0), tol.linear));
    let v1 = topo.add_vertex(Vertex::new(Point3::new(10.0, 0.0, 0.0), tol.linear));
    let circle = Circle3D::with_axes(
        Point3::new(5.0, 0.0, 0.0),
        Vec3::new(0.0, 0.0, 1.0),
        r,
        Vec3::new(-1.0, 0.0, 0.0),
        Vec3::new(0.0, 1.0, 0.0),
    )
    .unwrap();
    // arc stored (0,0)->(10,0) bulging +y; used reversed so the loop walks
    // (10,0)->(5,5)->(0,0)->(10,0): the upper half-disc wound CCW.
    let arc = topo.add_edge(Edge::new(v0, v1, EdgeCurve::Circle(circle)));
    let line = topo.add_edge(Edge::new(v0, v1, EdgeCurve::Line));
    let wire = Wire::new(
        vec![OrientedEdge::new(arc, false), OrientedEdge::new(line, true)],
        true,
    )
    .unwrap();
    let wid = topo.add_wire(wire);
    let face = topo.add_face(Face::new(
        wid,
        vec![],
        FaceSurface::Plane {
            normal: Vec3::new(0.0, 0.0, 1.0),
            d: 0.0,
        },
    ));
    let expected = 0.5 * std::f64::consts::PI * r * r;
    let solid = extrude(&mut topo, face, Vec3::new(0.0, 0.0, 1.0), 1.0).unwrap();
    let vol = crate::measure::solid_volume(&topo, solid, 0.001).unwrap();
    assert!(
        (vol - expected).abs() / expected < 0.01,
        "reversed-edge half-circle volume {vol:.4} != {expected:.4}"
    );
}

#[test]
fn extrude_half_disc_reversed_nurbs_edge_volume() {
    use brepkit_math::nurbs::fitting::interpolate;
    use brepkit_math::vec::{Point3, Vec3};

    // Upper half-disc whose arc boundary is a NURBS (interpolated semicircle),
    // used REVERSED in the wire. Guards the NURBS arm of `EdgeCurve::reversed`:
    // a reversed NURBS edge would otherwise leave the translated top edge's
    // endpoints inconsistent with its curve (the swept side then degenerates).
    let mut topo = Topology::new();
    let tol = Tolerance::new();
    let r = 5.0_f64;
    let cx = 5.0_f64;
    // Upper semicircle (0,0) -> (5,5) -> (10,0): angle PI down to 0.
    let n_pts = 24;
    let pts: Vec<Point3> = (0..=n_pts)
        .map(|i| {
            let angle = std::f64::consts::PI * (1.0 - f64::from(i) / f64::from(n_pts));
            Point3::new(cx + r * angle.cos(), r * angle.sin(), 0.0)
        })
        .collect();
    let nurbs = interpolate(&pts, 3).unwrap();
    let v0 = topo.add_vertex(Vertex::new(Point3::new(0.0, 0.0, 0.0), tol.linear));
    let v1 = topo.add_vertex(Vertex::new(Point3::new(10.0, 0.0, 0.0), tol.linear));
    // arc stored (0,0)->(10,0) bulging +y; used reversed so the loop walks
    // (10,0)->(5,5)->(0,0)->(10,0): the upper half-disc wound CCW.
    let arc = topo.add_edge(Edge::new(v0, v1, EdgeCurve::NurbsCurve(nurbs)));
    let line = topo.add_edge(Edge::new(v0, v1, EdgeCurve::Line));
    let wire = Wire::new(
        vec![OrientedEdge::new(arc, false), OrientedEdge::new(line, true)],
        true,
    )
    .unwrap();
    let wid = topo.add_wire(wire);
    let face = topo.add_face(Face::new(
        wid,
        vec![],
        FaceSurface::Plane {
            normal: Vec3::new(0.0, 0.0, 1.0),
            d: 0.0,
        },
    ));
    // NURBS interpolation of the semicircle carries a small fitting error, so
    // allow 2% (the bug otherwise degenerates the solid, not a 2% shift).
    let expected = 0.5 * std::f64::consts::PI * r * r;
    let solid = extrude(&mut topo, face, Vec3::new(0.0, 0.0, 1.0), 1.0).unwrap();
    let vol = crate::measure::solid_volume(&topo, solid, 0.001).unwrap();
    assert!(
        (vol - expected).abs() / expected < 0.02,
        "reversed-edge half-disc (NURBS) volume {vol:.4} != {expected:.4}"
    );
}

/// Regression for #869: extruding an `EdgeCurve::Ellipse` arc must build the
/// ruled side surface from the edge's TRIMMED arc, not the full ellipse, so
/// the solid volume matches the half-ellipse swept area (not the full ellipse).
#[test]
fn extrude_half_ellipse_face_volume() {
    use brepkit_math::vec::{Point3, Vec3};
    use brepkit_topology::builder::make_ellipse_arc;

    let mut topo = Topology::new();
    let center = Point3::new(5.0, 0.0, 0.0);
    let axis = Vec3::new(0.0, 0.0, -1.0);
    let ref_dir = Vec3::new(0.0, 1.0, 0.0);
    let start = Point3::new(0.0, 0.0, 0.0);
    let end = Point3::new(10.0, 0.0, 0.0);
    // Up half-ellipse arc (0,0) -> (5,8.333) -> (10,0).
    let arc = make_ellipse_arc(
        &mut topo,
        center,
        axis,
        8.333_333_333_333_334,
        5.0,
        ref_dir,
        start,
        end,
        1e-7,
    )
    .unwrap();
    let (v0, v1) = {
        let e = topo.edge(arc).unwrap();
        (e.start(), e.end())
    };
    // Closing diameter line (10,0) -> (0,0).
    let line = topo.add_edge(Edge::new(v1, v0, EdgeCurve::Line));
    let wire = Wire::new(
        vec![OrientedEdge::new(arc, true), OrientedEdge::new(line, true)],
        true,
    )
    .unwrap();
    let wid = topo.add_wire(wire);
    let face = topo.add_face(Face::new(
        wid,
        vec![],
        FaceSurface::Plane {
            normal: Vec3::new(0.0, 0.0, 1.0),
            d: 0.0,
        },
    ));
    // Half-ellipse area = (1/2)*pi*8.333*5 = 65.45; extrude by 1 => vol 65.45.
    let expected = 0.5 * std::f64::consts::PI * 8.333_333_333_333_334 * 5.0;
    let face_area = crate::measure::face_area(&topo, face, 0.001).unwrap();
    // The planar cap already measures correctly; the swept side is the bug.
    assert!(
        (face_area - expected).abs() / expected < 0.01,
        "half-ellipse cap area {face_area:.4} != {expected:.4}"
    );
    let solid = extrude(&mut topo, face, Vec3::new(0.0, 0.0, 1.0), 1.0).unwrap();
    // The swept ellipse-arc side surface is geometrically exact (the rational
    // quadratic ellipse arc is the affine image of a circle arc), so the only
    // residual is tessellation discretization of the curved side; a fine
    // deflection converges the divergence-theorem volume to the true 65.45.
    // (Before the fix the full ellipse was swept, giving 81.09.)
    let vol = crate::measure::solid_volume(&topo, solid, 0.001).unwrap();
    assert!(
        (vol - expected).abs() / expected < 0.01,
        "extruded half-ellipse volume {vol:.4} != {expected:.4}"
    );
}

#[test]
fn extrude_half_ellipse_reversed_edge_volume() {
    use brepkit_math::vec::{Point3, Vec3};
    use brepkit_topology::builder::make_ellipse_arc;

    // Same upper-half-ellipse region as `extrude_half_ellipse_face_volume`, but
    // the arc is traversed REVERSED in the wire. The arc edge's stored start/end
    // then disagree with wire-traversal order. `side_face_surface` derives its
    // swept-side domain from the stored endpoints, and the translated top edge
    // is parameter-reversed to match — so the swept region is the upper arc.
    let mut topo = Topology::new();
    let center = Point3::new(5.0, 0.0, 0.0);
    let axis = Vec3::new(0.0, 0.0, -1.0);
    let ref_dir = Vec3::new(0.0, 1.0, 0.0);
    let start = Point3::new(0.0, 0.0, 0.0);
    let end = Point3::new(10.0, 0.0, 0.0);
    // Up half-ellipse arc, stored (0,0) -> (5,8.333) -> (10,0).
    let arc = make_ellipse_arc(
        &mut topo,
        center,
        axis,
        8.333_333_333_333_334,
        5.0,
        ref_dir,
        start,
        end,
        1e-7,
    )
    .unwrap();
    let (v0, v1) = {
        let e = topo.edge(arc).unwrap();
        (e.start(), e.end())
    };
    // Closing diameter line (0,0) -> (10,0); the arc is used reversed, so the
    // loop walks (10,0) -> (5,8.333) -> (0,0) -> (10,0): the upper region wound CW.
    let line = topo.add_edge(Edge::new(v0, v1, EdgeCurve::Line));
    let wire = Wire::new(
        vec![OrientedEdge::new(arc, false), OrientedEdge::new(line, true)],
        true,
    )
    .unwrap();
    let wid = topo.add_wire(wire);
    let face = topo.add_face(Face::new(
        wid,
        vec![],
        FaceSurface::Plane {
            normal: Vec3::new(0.0, 0.0, 1.0),
            d: 0.0,
        },
    ));
    let expected = 0.5 * std::f64::consts::PI * 8.333_333_333_333_334 * 5.0;
    let solid = extrude(&mut topo, face, Vec3::new(0.0, 0.0, 1.0), 1.0).unwrap();
    let vol = crate::measure::solid_volume(&topo, solid, 0.001).unwrap();
    assert!(
        (vol - expected).abs() / expected < 0.01,
        "reversed-edge half-ellipse volume {vol:.4} != {expected:.4}"
    );
}

#[test]
fn extrude_spline_encoded_profile_recovers_analytic_walls() {
    // A chamfered-rectangle profile whose edges arrive as B-splines (the 2D
    // drawing pipeline ships corner-treated profiles this way): four straight
    // runs and one chamfer segment as degree-1 NURBS, one fillet corner as a
    // rational-quadratic arc. Every side wall must come out Plane/Cylinder.
    use brepkit_geometry::convert::{circle_to_nurbs, line_to_nurbs};
    use brepkit_topology::edge::Edge;
    use brepkit_topology::face::Face;
    use brepkit_topology::vertex::Vertex;
    use brepkit_topology::wire::Wire;

    let mut topo = Topology::new();
    let tol = Tolerance::new().linear;
    // Rectangle [0,10]x[0,6] with a 2mm chamfer at (10,6) and an r=2 fillet
    // arc at (0,6).
    let pts = [
        Point3::new(0.0, 0.0, 0.0),
        Point3::new(10.0, 0.0, 0.0),
        Point3::new(10.0, 4.0, 0.0), // chamfer start
        Point3::new(8.0, 6.0, 0.0),  // chamfer end
        Point3::new(2.0, 6.0, 0.0),  // fillet start
        Point3::new(0.0, 4.0, 0.0),  // fillet end
    ];
    let vids: Vec<_> = pts
        .iter()
        .map(|&p| topo.add_vertex(Vertex::new(p, tol)))
        .collect();
    let nurbs_line = |a: Point3, b: Point3| EdgeCurve::NurbsCurve(line_to_nurbs(a, b).unwrap());
    let fillet_circle = brepkit_math::curves::Circle3D::new(
        Point3::new(2.0, 4.0, 0.0),
        Vec3::new(0.0, 0.0, 1.0),
        2.0,
    )
    .unwrap();
    // CCW arc from (2,6) to (0,4) — parameters via projection, because
    // Circle3D picks its own reference axis.
    let t0 = fillet_circle.project(pts[4]);
    let mut t1 = fillet_circle.project(pts[5]);
    if t1 < t0 {
        t1 += std::f64::consts::TAU;
    }
    let fillet_arc = EdgeCurve::NurbsCurve(circle_to_nurbs(&fillet_circle, t0, t1).unwrap());
    let curves = [
        nurbs_line(pts[0], pts[1]),
        nurbs_line(pts[1], pts[2]),
        nurbs_line(pts[2], pts[3]), // chamfer
        nurbs_line(pts[3], pts[4]),
        fillet_arc,
        nurbs_line(pts[5], pts[0]),
    ];
    let mut oes = Vec::new();
    for (i, c) in curves.into_iter().enumerate() {
        let a = vids[i];
        let b = vids[(i + 1) % vids.len()];
        let e = topo.add_edge(Edge::new(a, b, c));
        oes.push(OrientedEdge::new(e, true));
    }
    let wid = topo.add_wire(Wire::new(oes, true).unwrap());
    let face = topo.add_face(Face::new(
        wid,
        vec![],
        FaceSurface::Plane {
            normal: Vec3::new(0.0, 0.0, 1.0),
            d: 0.0,
        },
    ));
    let solid = extrude(&mut topo, face, Vec3::new(0.0, 0.0, 1.0), 3.0).unwrap();
    let mut nurbs_walls = 0usize;
    let mut cylinders = 0usize;
    for fid in brepkit_topology::explorer::solid_faces(&topo, solid).unwrap() {
        match topo.face(fid).unwrap().surface() {
            FaceSurface::Nurbs(_) => nurbs_walls += 1,
            FaceSurface::Cylinder(_) => cylinders += 1,
            _ => {}
        }
    }
    assert_eq!(
        nurbs_walls, 0,
        "spline-encoded straight/arc profile edges must extrude as analytic walls"
    );
    assert_eq!(
        cylinders, 1,
        "the fillet arc must extrude as a true cylinder"
    );
    // Exact area: 10*6 - chamfer triangle (2*2/2) - fillet corner (4 - pi) = 58 - (4 - pi).
    let expected = (60.0 - 2.0 - (4.0 - std::f64::consts::PI)) * 3.0;
    let vol = crate::measure::solid_volume(&topo, solid, 0.001).unwrap();
    assert!(
        (vol - expected).abs() / expected < 1e-4,
        "analytic prism volume {vol:.9} != exact {expected:.9}"
    );
}

#[test]
fn extrude_reversed_spline_arc_profile_recovers_analytic_walls() {
    // Same chamfered/filleted profile, but the fillet spline is STORED
    // end-to-start relative to the edge vertices (the `rev` branch of the
    // profile normalization): the recovered Circle must still cover the
    // fillet arc, not its complement.
    use brepkit_geometry::convert::{circle_to_nurbs, line_to_nurbs};
    use brepkit_topology::edge::Edge;
    use brepkit_topology::face::Face;
    use brepkit_topology::vertex::Vertex;
    use brepkit_topology::wire::Wire;

    let mut topo = Topology::new();
    let tol = Tolerance::new().linear;
    let pts = [
        Point3::new(0.0, 0.0, 0.0),
        Point3::new(10.0, 0.0, 0.0),
        Point3::new(10.0, 6.0, 0.0),
        Point3::new(2.0, 6.0, 0.0),
        Point3::new(0.0, 4.0, 0.0),
    ];
    let vids: Vec<_> = pts
        .iter()
        .map(|&p| topo.add_vertex(Vertex::new(p, tol)))
        .collect();
    let nurbs_line = |a: Point3, b: Point3| EdgeCurve::NurbsCurve(line_to_nurbs(a, b).unwrap());
    let fillet_circle = brepkit_math::curves::Circle3D::new(
        Point3::new(2.0, 4.0, 0.0),
        Vec3::new(0.0, 0.0, 1.0),
        2.0,
    )
    .unwrap();
    // Spline built from (0,4) to (2,6) — REVERSED against the wire traversal
    // (2,6) -> (0,4). The physical fillet quarter runs CW from (0,4), i.e. a
    // decreasing parameter span.
    let t0 = fillet_circle.project(pts[4]);
    let t1 = t0 - std::f64::consts::FRAC_PI_2;
    let reversed_arc = EdgeCurve::NurbsCurve(circle_to_nurbs(&fillet_circle, t0, t1).unwrap());
    let curves = [
        nurbs_line(pts[0], pts[1]),
        nurbs_line(pts[1], pts[2]),
        nurbs_line(pts[2], pts[3]),
        reversed_arc,
        nurbs_line(pts[4], pts[0]),
    ];
    let mut oes = Vec::new();
    for (i, c) in curves.into_iter().enumerate() {
        let a = vids[i];
        let b = vids[(i + 1) % vids.len()];
        let e = topo.add_edge(Edge::new(a, b, c));
        oes.push(OrientedEdge::new(e, true));
    }
    let wid = topo.add_wire(Wire::new(oes, true).unwrap());
    let face = topo.add_face(Face::new(
        wid,
        vec![],
        FaceSurface::Plane {
            normal: Vec3::new(0.0, 0.0, 1.0),
            d: 0.0,
        },
    ));
    let solid = extrude(&mut topo, face, Vec3::new(0.0, 0.0, 1.0), 3.0).unwrap();
    let mut nurbs_walls = 0usize;
    for fid in brepkit_topology::explorer::solid_faces(&topo, solid).unwrap() {
        if matches!(topo.face(fid).unwrap().surface(), FaceSurface::Nurbs(_)) {
            nurbs_walls += 1;
        }
    }
    assert_eq!(
        nurbs_walls, 0,
        "reversed spline arc must still recover analytic walls"
    );
    // Exact area: 10*6 - fillet corner (4 - pi).
    let expected = (60.0 - (4.0 - std::f64::consts::PI)) * 3.0;
    let vol = crate::measure::solid_volume(&topo, solid, 0.001).unwrap();
    assert!(
        (vol - expected).abs() / expected < 1e-4,
        "reversed-arc prism volume {vol:.9} != exact {expected:.9}"
    );
}

/// A plane section's ellipse starts where it meets its wall's seam, not at
/// its frame's origin; splitting it has to run the turn from that vertex.
#[test]
fn split_closed_edge_runs_the_turn_from_the_vertex() {
    use brepkit_math::curves::Ellipse3D;
    use brepkit_topology::edge::{Edge, EdgeCurve};
    use brepkit_topology::vertex::Vertex;
    use std::f64::consts::FRAC_PI_2;

    let mut topo = Topology::new();
    let ellipse = Ellipse3D::new(
        Point3::new(0.0, 0.0, 0.0),
        Vec3::new(0.0, 0.0, 1.0),
        3.0,
        2.0,
    )
    .unwrap();
    let tol = Tolerance::new().linear;
    let v = topo.add_vertex(Vertex::new(ellipse.evaluate(FRAC_PI_2), tol));
    let edge = topo.add_edge(Edge::new(v, v, EdgeCurve::Ellipse(ellipse.clone())));

    let pieces = split_closed_edge(&mut topo, edge, 4, tol).unwrap();
    let corners: Vec<Point3> = pieces
        .iter()
        .map(|&e| topo.vertex(topo.edge(e).unwrap().start()).unwrap().point())
        .collect();
    for (k, corner) in corners.iter().enumerate() {
        #[allow(clippy::cast_precision_loss)]
        let expected = ellipse.evaluate(FRAC_PI_2 * (1.0 + k as f64));
        assert!(
            (*corner - expected).length() < 1e-9,
            "corner {k} at {corner:?}, expected {expected:?}"
        );
    }
}

/// A 2 x 1 bilinear patch at z = 0.5, its surface facing up or down and its
/// wire wound to match, extruded 1.5 up or down: each is valid, holds 3, and
/// each cap faces away from the solid along its outward normal.
#[test]
fn a_planar_nurbs_face_extrudes_either_way_with_caps_facing_out() {
    use brepkit_math::nurbs::surface::NurbsSurface;
    use brepkit_math::vec::{Point3, Vec3};
    use brepkit_topology::builder::make_polygon_wire;
    use brepkit_topology::face::Face;

    for up in [true, false] {
        for direction in [Vec3::new(0.0, 0.0, 1.0), Vec3::new(0.0, 0.0, -1.0)] {
            let mut topo = Topology::new();
            let at = |x: f64, y: f64| Point3::new(x, y, 0.5);
            // Rows run along x and columns along y, so the surface faces +z;
            // swapping the axes turns it to -z.
            let (rows, corners) = if up {
                (
                    vec![
                        vec![at(0.0, 0.0), at(0.0, 1.0)],
                        vec![at(2.0, 0.0), at(2.0, 1.0)],
                    ],
                    [at(0.0, 0.0), at(2.0, 0.0), at(2.0, 1.0), at(0.0, 1.0)],
                )
            } else {
                (
                    vec![
                        vec![at(0.0, 0.0), at(2.0, 0.0)],
                        vec![at(0.0, 1.0), at(2.0, 1.0)],
                    ],
                    [at(0.0, 0.0), at(0.0, 1.0), at(2.0, 1.0), at(2.0, 0.0)],
                )
            };
            let knots = vec![0.0, 0.0, 1.0, 1.0];
            let patch =
                NurbsSurface::new(1, 1, knots.clone(), knots, rows, vec![vec![1.0; 2]; 2]).unwrap();
            let wire = make_polygon_wire(&mut topo, &corners, 1e-7).unwrap();
            let face = topo.add_face(Face::new(wire, vec![], FaceSurface::Nurbs(patch)));
            let solid = extrude(&mut topo, face, direction, 1.5).unwrap();

            assert!(
                crate::validate::validate_solid(&topo, solid)
                    .unwrap()
                    .is_valid()
            );
            let volume = crate::measure::solid_volume(&topo, solid, 1e-3).unwrap();
            assert!(
                (volume - 3.0).abs() < 1e-9,
                "up {up}, direction {direction:?}: {volume}"
            );
            assert_nurbs_faces_face_out(&topo, solid, &format!("up {up}, direction {direction:?}"));
        }
    }
}

/// A point 0.01 off the middle of each NURBS face of `solid`, along the
/// face's outward normal, lies outside it.
fn assert_nurbs_faces_face_out(
    topo: &Topology,
    solid: brepkit_topology::solid::SolidId,
    label: &str,
) {
    for fid in brepkit_topology::explorer::solid_faces(topo, solid).unwrap() {
        let f = topo.face(fid).unwrap();
        let FaceSurface::Nurbs(s) = f.surface() else {
            continue;
        };
        let n = s.normal(0.5, 0.5).unwrap();
        let outward = if f.is_reversed() { -n } else { n };
        let off = s.evaluate(0.5, 0.5) + outward * 0.01;
        assert_eq!(
            crate::classify::classify_point(topo, solid, off, 1e-3, 1e-7).unwrap(),
            crate::classify::PointClassification::Outside,
            "{label}: face {fid:?} faces in"
        );
    }
}

/// The patch S(u, v) = (6u(1 - u), -1 + 6u² - 4u³, v), cubic in u with
/// control points (0, -1), (2, -1), (2, 1), (0, 1), swept 1.5 along x either
/// way. Its vertices' centroid (0, 0, 0.5) lies nearest the patch at its
/// u = 0 edge, where the face stands edge-on to x; inside, its normal
/// leans along +x. Each solid is valid and has its caps facing out.
#[test]
fn a_curved_nurbs_face_edge_on_at_its_centroid_extrudes_with_caps_facing_out() {
    use brepkit_math::nurbs::curve::NurbsCurve;
    use brepkit_math::nurbs::surface::NurbsSurface;
    use brepkit_math::vec::{Point3, Vec3};
    use brepkit_topology::edge::{Edge, EdgeCurve};
    use brepkit_topology::face::Face;
    use brepkit_topology::vertex::Vertex;
    use brepkit_topology::wire::{OrientedEdge, Wire};

    for direction in [Vec3::new(1.0, 0.0, 0.0), Vec3::new(-1.0, 0.0, 0.0)] {
        let mut topo = Topology::new();
        let at = |x: f64, y: f64, z: f64| Point3::new(x, y, z);
        let bow = |z: f64| {
            [
                at(0.0, -1.0, z),
                at(2.0, -1.0, z),
                at(2.0, 1.0, z),
                at(0.0, 1.0, z),
            ]
        };
        let cubic = vec![0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0];
        let rows: Vec<Vec<Point3>> = (0..4).map(|i| vec![bow(0.0)[i], bow(1.0)[i]]).collect();
        let patch = NurbsSurface::new(
            3,
            1,
            cubic.clone(),
            vec![0.0, 0.0, 1.0, 1.0],
            rows,
            vec![vec![1.0; 2]; 4],
        )
        .unwrap();
        let [a, b, c, d] = [
            at(0.0, -1.0, 0.0),
            at(0.0, 1.0, 0.0),
            at(0.0, 1.0, 1.0),
            at(0.0, -1.0, 1.0),
        ]
        .map(|p| topo.add_vertex(Vertex::new(p, 1e-7)));
        let curve = |z: f64| {
            EdgeCurve::NurbsCurve(
                NurbsCurve::new(3, cubic.clone(), bow(z).to_vec(), vec![1.0; 4]).unwrap(),
            )
        };
        let low = topo.add_edge(Edge::new(a, b, curve(0.0)));
        let right = topo.add_edge(Edge::new(b, c, EdgeCurve::Line));
        let high = topo.add_edge(Edge::new(d, c, curve(1.0)));
        let left = topo.add_edge(Edge::new(d, a, EdgeCurve::Line));
        let wire = Wire::new(
            vec![
                OrientedEdge::new(low, true),
                OrientedEdge::new(right, true),
                OrientedEdge::new(high, false),
                OrientedEdge::new(left, true),
            ],
            true,
        )
        .unwrap();
        let wire = topo.add_wire(wire);
        let face = topo.add_face(Face::new(wire, vec![], FaceSurface::Nurbs(patch)));
        let solid = extrude(&mut topo, face, direction, 1.5).unwrap();

        let label = format!("direction {direction:?}");
        assert!(
            crate::validate::validate_solid(&topo, solid)
                .unwrap()
                .is_valid(),
            "{label}"
        );
        assert_nurbs_faces_face_out(&topo, solid, &label);
    }
}

/// The patch S(u, v) = (u, F(u), v) over the unit square, F of degree 11
/// with F' a multiple of [u (u - 1/4) (u - 1/2) (u - 3/4) (u - 1)]^2 rising
/// from 0 to 1, swept 1 along x. Its x-normal F' vanishes at its vertices'
/// centroid and at every quarter of its edges, yet the sweep is a solid:
/// it is valid and has its caps facing out.
#[test]
fn a_nurbs_face_edge_on_at_its_quarters_extrudes_with_caps_facing_out() {
    use brepkit_math::nurbs::curve::NurbsCurve;
    use brepkit_math::nurbs::surface::NurbsSurface;
    use brepkit_math::vec::{Point3, Vec3};
    use brepkit_topology::edge::{Edge, EdgeCurve};
    use brepkit_topology::face::Face;
    use brepkit_topology::vertex::Vertex;
    use brepkit_topology::wire::{OrientedEdge, Wire};

    let f = [
        0.0, 0.0, 0.0, 5.04, -11.34, 18.3, -17.3, 12.34, -4.04, 1.0, 1.0, 1.0,
    ];
    let row = |z: f64| -> Vec<Point3> {
        (0..12)
            .map(|i| Point3::new(f64::from(i) / 11.0, f[i as usize], z))
            .collect()
    };
    let mut knots = vec![0.0; 12];
    knots.extend([1.0; 12]);
    let mut topo = Topology::new();
    let rows: Vec<Vec<Point3>> = (0..12).map(|i| vec![row(0.0)[i], row(1.0)[i]]).collect();
    let patch = NurbsSurface::new(
        11,
        1,
        knots.clone(),
        vec![0.0, 0.0, 1.0, 1.0],
        rows,
        vec![vec![1.0; 2]; 12],
    )
    .unwrap();
    let at = |x: f64, y: f64, z: f64| Point3::new(x, y, z);
    let [a, b, c, d] = [
        at(0.0, 0.0, 0.0),
        at(1.0, 1.0, 0.0),
        at(1.0, 1.0, 1.0),
        at(0.0, 0.0, 1.0),
    ]
    .map(|p| topo.add_vertex(Vertex::new(p, 1e-7)));
    let curve = |z: f64| {
        EdgeCurve::NurbsCurve(NurbsCurve::new(11, knots.clone(), row(z), vec![1.0; 12]).unwrap())
    };
    let low = topo.add_edge(Edge::new(a, b, curve(0.0)));
    let right = topo.add_edge(Edge::new(b, c, EdgeCurve::Line));
    let high = topo.add_edge(Edge::new(d, c, curve(1.0)));
    let left = topo.add_edge(Edge::new(d, a, EdgeCurve::Line));
    let wire = Wire::new(
        vec![
            OrientedEdge::new(low, true),
            OrientedEdge::new(right, true),
            OrientedEdge::new(high, false),
            OrientedEdge::new(left, true),
        ],
        true,
    )
    .unwrap();
    let wire = topo.add_wire(wire);
    let face = topo.add_face(Face::new(wire, vec![], FaceSurface::Nurbs(patch)));
    let solid = extrude(&mut topo, face, Vec3::new(1.0, 0.0, 0.0), 1.0).unwrap();
    assert!(
        crate::validate::validate_solid(&topo, solid)
            .unwrap()
            .is_valid()
    );
    assert_nurbs_faces_face_out(&topo, solid, "edge-on at its quarters");
}

/// The patch S(u, v) = (u, v, u³/3 + uv² - u) trimmed to the annulus
/// 1.1 ≤ √(u² + v²) ≤ 1.2, its rims lifted exactly as rational sextic
/// quarters, swept 1 along +x. Its x-normal 1 - u² - v² is negative all
/// over the face but +1 at its vertices' centroid, the hole's centre: the
/// cap at the face faces -x and the one at its end +x.
#[test]
fn a_holed_nurbs_face_extrudes_with_caps_facing_out() {
    use brepkit_math::nurbs::curve::NurbsCurve;
    use brepkit_math::nurbs::surface::NurbsSurface;
    use brepkit_math::vec::{Point3, Vec3};
    use brepkit_topology::edge::{Edge, EdgeCurve};
    use brepkit_topology::face::Face;
    use brepkit_topology::vertex::Vertex;
    use brepkit_topology::wire::{OrientedEdge, Wire};

    // Polynomials as coefficients, lowest first.
    let mul = |a: &[f64], b: &[f64]| {
        let mut r = vec![0.0; a.len() + b.len() - 1];
        for (i, x) in a.iter().enumerate() {
            for (j, y) in b.iter().enumerate() {
                r[i + j] += x * y;
            }
        }
        r
    };
    let choose = |n: usize, k: usize| (0..k).fold(1.0, |c, i| c * (n - i) as f64 / (i + 1) as f64);
    let bernstein = |a: &[f64]| -> Vec<f64> {
        (0..=6)
            .map(|i| {
                (0..=i)
                    .map(|k| choose(i, k) / choose(6, k) * a.get(k).copied().unwrap_or(0.0))
                    .sum()
            })
            .collect()
    };
    // The quarter turn `k` of the rim of radius `r`, lifted onto the patch:
    // with c = 1 + s², x = X/c and y = Y/c for (X, Y) = (r (1 - s²), 2 r s)
    // turned k quarters, so x, y and the height x³/3 + x y² - x times c³
    // are sextics over the weight c³.
    let quarter = |r: f64, k: usize| {
        let (a, b) = (vec![r, 0.0, -r], vec![0.0, 2.0 * r]);
        let neg = |p: &[f64]| p.iter().map(|x| -x).collect::<Vec<f64>>();
        let (x, y) = match k {
            0 => (a, b),
            1 => (neg(&b), a),
            2 => (neg(&a), neg(&b)),
            _ => (b, neg(&a)),
        };
        let c2 = mul(&[1.0, 0.0, 1.0], &[1.0, 0.0, 1.0]);
        let c3 = mul(&c2, &[1.0, 0.0, 1.0]);
        let height: Vec<f64> = {
            let cube: Vec<f64> = mul(&mul(&x, &x), &x).iter().map(|v| v / 3.0).collect();
            let cross = mul(&x, &mul(&y, &y));
            let line = mul(&x, &c2);
            (0..7)
                .map(|i| {
                    cube.get(i).copied().unwrap_or(0.0) + cross.get(i).copied().unwrap_or(0.0)
                        - line.get(i).copied().unwrap_or(0.0)
                })
                .collect()
        };
        let w = bernstein(&c3);
        let (xs, ys, zs) = (
            bernstein(&mul(&x, &c2)),
            bernstein(&mul(&y, &c2)),
            bernstein(&height),
        );
        let points = (0..7)
            .map(|i| Point3::new(xs[i] / w[i], ys[i] / w[i], zs[i] / w[i]))
            .collect();
        let mut knots = vec![0.0; 7];
        knots.extend([1.0; 7]);
        NurbsCurve::new(6, knots, points, w).unwrap()
    };

    let mut topo = Topology::new();
    let (z0, z1) = (-611.0 / 375.0, 4147.0 / 1500.0);
    let (y0, y1) = (13.0 / 30.0, 949.0 / 500.0);
    let rows: Vec<Vec<Point3>> = [[z0, z1, z0], [y0, y1, y0], [-y0, -y1, -y0], [-z0, -z1, -z0]]
        .iter()
        .enumerate()
        .map(|(i, zs)| {
            (0..3)
                .map(|j| {
                    Point3::new(
                        2.6f64.mul_add(i as f64 / 3.0, -1.3),
                        1.3f64.mul_add(j as f64, -1.3),
                        zs[j],
                    )
                })
                .collect()
        })
        .collect();
    let patch = NurbsSurface::new(
        3,
        2,
        vec![0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0],
        vec![0.0, 0.0, 0.0, 1.0, 1.0, 1.0],
        rows,
        vec![vec![1.0; 3]; 4],
    )
    .unwrap();
    // A rim as four quarters, counterclockwise about +z or clockwise.
    let rim = |topo: &mut Topology, r: f64, ccw: bool| {
        let curves: Vec<NurbsCurve> = (0..4).map(|k| quarter(r, k)).collect();
        let vertices: Vec<_> = curves
            .iter()
            .map(|c| topo.add_vertex(Vertex::new(c.control_points()[0], 1e-7)))
            .collect();
        let mut edges: Vec<OrientedEdge> = curves
            .into_iter()
            .enumerate()
            .map(|(k, c)| {
                let e = Edge::new(vertices[k], vertices[(k + 1) % 4], EdgeCurve::NurbsCurve(c));
                OrientedEdge::new(topo.add_edge(e), ccw)
            })
            .collect();
        if !ccw {
            edges.reverse();
        }
        topo.add_wire(Wire::new(edges, true).unwrap())
    };
    let outer = rim(&mut topo, 1.2, true);
    let hole = rim(&mut topo, 1.1, false);
    let face = topo.add_face(Face::new(
        outer,
        vec![hole],
        FaceSurface::Nurbs(patch.clone()),
    ));
    let solid = extrude(&mut topo, face, Vec3::new(1.0, 0.0, 0.0), 1.0).unwrap();

    // A point well inside the annulus, at radius 1.15 and 30 degrees.
    let (u, v) = (
        1.15 * 30f64.to_radians().cos(),
        1.15 * 30f64.to_radians().sin(),
    );
    let (s, t) = ((u + 1.3) / 2.6, (v + 1.3) / 2.6);
    assert!(patch.normal(s, t).unwrap().x() < 0.0);
    assert_caps_face_out_at(&topo, solid, &patch, (s, t), Vec3::new(1.0, 0.0, 0.0));
}

/// The caps of `solid`, an extrusion of a face on `patch` by `along`, each
/// face away from the solid at the patch's point `(s, t)` on the face: the
/// cap at the face against `along`, the one at its end along it.
fn assert_caps_face_out_at(
    topo: &Topology,
    solid: brepkit_topology::solid::SolidId,
    patch: &brepkit_math::nurbs::surface::NurbsSurface,
    (s, t): (f64, f64),
    along: brepkit_math::vec::Vec3,
) {
    let n = patch.normal(s, t).unwrap();
    let mut caps = 0;
    for fid in brepkit_topology::explorer::solid_faces(topo, solid).unwrap() {
        let f = topo.face(fid).unwrap();
        let FaceSurface::Nurbs(surface) = f.surface() else {
            continue;
        };
        if surface.degree_u() != patch.degree_u() || surface.degree_v() != patch.degree_v() {
            continue;
        }
        let shift = surface.evaluate(s, t) - patch.evaluate(s, t);
        let at_end = if shift.length() <= 1e-9 {
            false
        } else if (shift - along).length() <= 1e-9 {
            true
        } else {
            continue;
        };
        let facing = if f.is_reversed() { -n } else { n };
        let outward = if at_end { along } else { -along };
        assert!(
            facing.dot(outward) > 0.0,
            "the cap at the end {at_end} faces in"
        );
        caps += 1;
    }
    assert_eq!(caps, 2);
}

/// A flat patch facing +z, trimmed to the ring between a circle of radius
/// 1, one closed edge, and a hole of radius 0.99 in four quarter arcs,
/// extruded 1 along +z: its caps face out. Sixteen points round the whole
/// circle shadow 2.5% less than it, more than the ring holds.
#[test]
fn a_thin_ring_extrudes_with_caps_facing_out_however_its_rims_are_cut() {
    use brepkit_math::curves::Circle3D;
    use brepkit_math::nurbs::surface::NurbsSurface;
    use brepkit_math::vec::{Point3, Vec3};
    use brepkit_topology::edge::{Edge, EdgeCurve};
    use brepkit_topology::face::Face;
    use brepkit_topology::vertex::Vertex;
    use brepkit_topology::wire::{OrientedEdge, Wire};
    let mut topo = Topology::new();
    let at = |x: f64, y: f64| Point3::new(x, y, 0.0);
    let knots = vec![0.0, 0.0, 1.0, 1.0];
    let patch = NurbsSurface::new(
        1,
        1,
        knots.clone(),
        knots,
        vec![
            vec![at(-1.2, -1.2), at(-1.2, 1.2)],
            vec![at(1.2, -1.2), at(1.2, 1.2)],
        ],
        vec![vec![1.0; 2]; 2],
    )
    .unwrap();
    let z = Vec3::new(0.0, 0.0, 1.0);
    let seam = topo.add_vertex(Vertex::new(at(1.0, 0.0), 1e-7));
    let rim = Circle3D::new(at(0.0, 0.0), z, 1.0).unwrap();
    let rim = topo.add_edge(Edge::new(seam, seam, EdgeCurve::Circle(rim)));
    let outer = topo.add_wire(Wire::new(vec![OrientedEdge::new(rim, true)], true).unwrap());
    let quarters: Vec<_> = (0..4)
        .map(|k| {
            let a = std::f64::consts::FRAC_PI_2 * f64::from(k);
            topo.add_vertex(Vertex::new(at(0.99 * a.cos(), 0.99 * a.sin()), 1e-7))
        })
        .collect();
    let hole_circle = Circle3D::new(at(0.0, 0.0), z, 0.99).unwrap();
    let mut hole_edges: Vec<OrientedEdge> = (0..4)
        .map(|k| {
            let e = Edge::new(
                quarters[k],
                quarters[(k + 1) % 4],
                EdgeCurve::Circle(hole_circle.clone()),
            );
            OrientedEdge::new(topo.add_edge(e), false)
        })
        .collect();
    hole_edges.reverse();
    let hole = topo.add_wire(Wire::new(hole_edges, true).unwrap());
    let face = topo.add_face(Face::new(
        outer,
        vec![hole],
        FaceSurface::Nurbs(patch.clone()),
    ));
    let solid = extrude(&mut topo, face, z, 1.0).unwrap();
    let a = 30f64.to_radians();
    let (x, y) = (0.995 * a.cos(), 0.995 * a.sin());
    assert_caps_face_out_at(&topo, solid, &patch, ((x + 1.2) / 2.4, (y + 1.2) / 2.4), z);
}

/// A unit cylinder's NURBS wall, closed round its axis with its seam at
/// angle 0, trimmed to the angles -45 to +45 degrees across the seam and
/// to z = 0 to 1, extruded 1 along +x: its normal leans +x all over the
/// face, and its caps face out.
#[test]
fn a_patch_across_its_surfaces_seam_extrudes_with_caps_facing_out() {
    use brepkit_math::curves::Circle3D;
    use brepkit_math::nurbs::surface::NurbsSurface;
    use brepkit_math::vec::{Point3, Vec3};
    use brepkit_topology::edge::{Edge, EdgeCurve};
    use brepkit_topology::face::Face;
    use brepkit_topology::vertex::Vertex;
    use brepkit_topology::wire::{OrientedEdge, Wire};
    let w = std::f64::consts::FRAC_1_SQRT_2;
    let ring = [
        (1.0, 0.0, 1.0),
        (1.0, 1.0, w),
        (0.0, 1.0, 1.0),
        (-1.0, 1.0, w),
        (-1.0, 0.0, 1.0),
        (-1.0, -1.0, w),
        (0.0, -1.0, 1.0),
        (1.0, -1.0, w),
        (1.0, 0.0, 1.0),
    ];
    let patch = NurbsSurface::new(
        2,
        1,
        vec![
            0.0, 0.0, 0.0, 0.25, 0.25, 0.5, 0.5, 0.75, 0.75, 1.0, 1.0, 1.0,
        ],
        vec![0.0, 0.0, 1.0, 1.0],
        ring.iter()
            .map(|&(x, y, _)| vec![Point3::new(x, y, 0.0), Point3::new(x, y, 1.0)])
            .collect(),
        ring.iter().map(|&(_, _, w)| vec![w, w]).collect(),
    )
    .unwrap();
    let mut topo = Topology::new();
    let at = |deg: f64, z: f64| {
        let a = deg.to_radians();
        Point3::new(a.cos(), a.sin(), z)
    };
    let [a, b, c, d] = [at(-45.0, 0.0), at(45.0, 0.0), at(45.0, 1.0), at(-45.0, 1.0)]
        .map(|p| topo.add_vertex(Vertex::new(p, 1e-7)));
    let arc = |z: f64| {
        EdgeCurve::Circle(
            Circle3D::new(Point3::new(0.0, 0.0, z), Vec3::new(0.0, 0.0, 1.0), 1.0).unwrap(),
        )
    };
    let low = topo.add_edge(Edge::new(a, b, arc(0.0)));
    let right = topo.add_edge(Edge::new(b, c, EdgeCurve::Line));
    let high = topo.add_edge(Edge::new(d, c, arc(1.0)));
    let left = topo.add_edge(Edge::new(d, a, EdgeCurve::Line));
    let wire = Wire::new(
        vec![
            OrientedEdge::new(low, true),
            OrientedEdge::new(right, true),
            OrientedEdge::new(high, false),
            OrientedEdge::new(left, true),
        ],
        true,
    )
    .unwrap();
    let wire = topo.add_wire(wire);
    let face = topo.add_face(Face::new(wire, vec![], FaceSurface::Nurbs(patch.clone())));
    let x = Vec3::new(1.0, 0.0, 0.0);
    let solid = extrude(&mut topo, face, x, 1.0).unwrap();
    let st = FaceSurface::Nurbs(patch.clone())
        .project_point(at(10.0, 0.5))
        .unwrap();
    assert!(patch.normal(st.0, st.1).unwrap().x() > 0.0);
    assert_caps_face_out_at(&topo, solid, &patch, st, x);
}
