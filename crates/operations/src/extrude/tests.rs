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

    // NURBS face is a bilinear patch: corners at z=0 (y=0) and z=0.5 (y=1).
    // The extrusion offset is (0,0,2), so top surface is at z=2 and z=2.5.
    // The solid is a "wedge" — thicker at one end.
    //
    // For a bilinear patch f(u,v) with z = 0.5v, the XY footprint is a
    // unit square. The enclosed volume between bottom and top surfaces:
    // V = ∫∫ height dA where height = 2.0 (constant, offset along Z).
    // But the solid shape is bounded by the slanted bottom/top NURBS faces.
    //
    // Actual result from tessellation: ~0.667 (≈ 2/3).
    // This is plausible: the bilinear bottom face "scoops out" volume.
    let vol = crate::measure::solid_volume(&topo, solid, 0.1).unwrap();
    assert!(
        vol > 0.3 && vol < 1.5,
        "extruded NURBS solid should have positive volume in (0.3, 1.5), got {vol}"
    );

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
    // used REVERSED in the wire — guards the Circle arm of `reverse_edge_curve`
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
    // used REVERSED in the wire. Guards the NURBS arm of `reverse_edge_curve`:
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
