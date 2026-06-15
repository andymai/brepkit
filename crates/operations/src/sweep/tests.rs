#![allow(clippy::unwrap_used)]

use std::collections::HashMap;

use brepkit_math::nurbs::curve::NurbsCurve;
use brepkit_math::tolerance::Tolerance;
use brepkit_math::vec::Point3;
use brepkit_topology::Topology;
use brepkit_topology::face::FaceSurface;
use brepkit_topology::test_utils::make_unit_square_face;

use super::*;

/// Helper: create a straight-line NURBS path from origin along +Z by `length`.
fn straight_z_path(length: f64) -> NurbsCurve {
    NurbsCurve::new(
        1,
        vec![0.0, 0.0, 1.0, 1.0],
        vec![Point3::new(0.0, 0.0, 0.0), Point3::new(0.0, 0.0, length)],
        vec![1.0, 1.0],
    )
    .unwrap()
}

/// Helper: create a quarter-circle NURBS path in the XZ plane.
fn quarter_circle_xz_path(radius: f64) -> NurbsCurve {
    let w = std::f64::consts::FRAC_1_SQRT_2;
    NurbsCurve::new(
        2,
        vec![0.0, 0.0, 0.0, 1.0, 1.0, 1.0],
        vec![
            Point3::new(0.0, 0.0, 0.0),
            Point3::new(radius, 0.0, 0.0),
            Point3::new(radius, 0.0, radius),
        ],
        vec![1.0, w, 1.0],
    )
    .unwrap()
}

/// Helper: a `size`×`size` square profile face centered at the origin in
/// the XY plane (normal +Z).
fn make_square(topo: &mut Topology, size: f64) -> FaceId {
    let hs = size / 2.0;
    let t = 1e-7;
    let v0 = topo.add_vertex(Vertex::new(Point3::new(-hs, -hs, 0.0), t));
    let v1 = topo.add_vertex(Vertex::new(Point3::new(hs, -hs, 0.0), t));
    let v2 = topo.add_vertex(Vertex::new(Point3::new(hs, hs, 0.0), t));
    let v3 = topo.add_vertex(Vertex::new(Point3::new(-hs, hs, 0.0), t));
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
    topo.add_face(Face::new(
        wid,
        vec![],
        FaceSurface::Plane {
            normal: Vec3::new(0.0, 0.0, 1.0),
            d: 0.0,
        },
    ))
}

#[test]
fn multi_section_sweep_line_spine_tapered_volume() {
    let mut topo = Topology::new();
    let big = make_square(&mut topo, 10.0);
    let small = make_square(&mut topo, 6.0);
    let spine = straight_z_path(20.0);

    let solid = multi_section_sweep(&mut topo, &spine, &[(big, 0.0), (small, 1.0)], true).unwrap();
    let vol = crate::measure::solid_volume(&topo, solid, 0.1).unwrap();

    // Frustum between a 10×10 and 6×6 square over height 20 lies strictly
    // between the two straight-prism volumes (6²·20 = 720 and 10²·20 = 2000).
    assert!(
        vol > 720.0 && vol < 2000.0,
        "expected tapered volume, got {vol}"
    );
}

#[test]
fn multi_section_sweep_curved_spine_positive_volume() {
    let mut topo = Topology::new();
    // Three sections so the loft follows the quarter-circle spine; the RMF
    // keeps each profile perpendicular and twist-free.
    let a = make_square(&mut topo, 4.0);
    let b = make_square(&mut topo, 4.0);
    let c = make_square(&mut topo, 4.0);
    let spine = quarter_circle_xz_path(20.0);

    let solid =
        multi_section_sweep(&mut topo, &spine, &[(a, 0.0), (b, 0.5), (c, 1.0)], true).unwrap();
    let vol = crate::measure::solid_volume(&topo, solid, 0.1).unwrap();
    assert!(
        vol > 0.0,
        "curved-spine multi-section sweep volume, got {vol}"
    );
}

#[test]
fn multi_section_sweep_rejects_single_section() {
    let mut topo = Topology::new();
    let only = make_square(&mut topo, 4.0);
    let spine = straight_z_path(10.0);
    assert!(multi_section_sweep(&mut topo, &spine, &[(only, 0.0)], true).is_err());
}

#[test]
fn multi_section_sweep_rejects_out_of_range_param() {
    let mut topo = Topology::new();
    let a = make_square(&mut topo, 4.0);
    let b = make_square(&mut topo, 4.0);
    let spine = straight_z_path(10.0);
    assert!(multi_section_sweep(&mut topo, &spine, &[(a, 0.0), (b, 1.5)], true).is_err());
}

#[test]
fn multi_section_sweep_unsorted_params_match_sorted() {
    // Placement sorts by parameter, so input order must not change the result.
    let mut t1 = Topology::new();
    let (a1, b1) = (make_square(&mut t1, 10.0), make_square(&mut t1, 4.0));
    let s1 = multi_section_sweep(
        &mut t1,
        &straight_z_path(20.0),
        &[(a1, 0.0), (b1, 1.0)],
        true,
    )
    .unwrap();
    let v_sorted = crate::measure::solid_volume(&t1, s1, 0.1).unwrap();

    let mut t2 = Topology::new();
    let (a2, b2) = (make_square(&mut t2, 10.0), make_square(&mut t2, 4.0));
    let s2 = multi_section_sweep(
        &mut t2,
        &straight_z_path(20.0),
        &[(b2, 1.0), (a2, 0.0)],
        true,
    )
    .unwrap();
    let v_unsorted = crate::measure::solid_volume(&t2, s2, 0.1).unwrap();

    assert!(
        (v_sorted - v_unsorted).abs() < 1e-6,
        "{v_sorted} vs {v_unsorted}"
    );
}

#[test]
fn profile_to_frame_matrix_is_proper_rotation() {
    // The placement must be a proper rotation (det +1), never a reflection,
    // or asymmetric profiles would be mirrored.
    let mut topo = Topology::new();
    let face = make_square(&mut topo, 4.0);
    let tangent = Vec3::new(1.0, 1.0, 1.0).normalize().unwrap();
    let up = orthogonalize(Vec3::new(0.0, 0.0, 1.0), tangent);
    let frame = Frame {
        origin: Point3::new(5.0, 6.0, 7.0),
        tangent,
        up,
        right: tangent.cross(up),
    };
    let m = profile_to_frame_matrix(&topo, face, &frame).unwrap();
    let r = &m.0;
    let det = r[0][0] * (r[1][1] * r[2][2] - r[1][2] * r[2][1])
        - r[0][1] * (r[1][0] * r[2][2] - r[1][2] * r[2][0])
        + r[0][2] * (r[1][0] * r[2][1] - r[1][1] * r[2][0]);
    assert!(
        (det - 1.0).abs() < 1e-9,
        "rotation det should be +1, got {det}"
    );
}

/// Helper: a `2*hx`×`2*hy` rectangle profile at the origin in the XY plane.
fn make_rect(topo: &mut Topology, hx: f64, hy: f64) -> FaceId {
    let t = 1e-7;
    let v0 = topo.add_vertex(Vertex::new(Point3::new(-hx, -hy, 0.0), t));
    let v1 = topo.add_vertex(Vertex::new(Point3::new(hx, -hy, 0.0), t));
    let v2 = topo.add_vertex(Vertex::new(Point3::new(hx, hy, 0.0), t));
    let v3 = topo.add_vertex(Vertex::new(Point3::new(-hx, hy, 0.0), t));
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
    topo.add_face(Face::new(
        wid,
        vec![],
        FaceSurface::Plane {
            normal: Vec3::new(0.0, 0.0, 1.0),
            d: 0.0,
        },
    ))
}

/// Helper: a degree-1 guide curve from `(x0,y0,0)` to `(x1,y1,10)`.
fn guide_line(x0: f64, y0: f64, x1: f64, y1: f64) -> NurbsCurve {
    NurbsCurve::new(
        1,
        vec![0.0, 0.0, 1.0, 1.0],
        vec![Point3::new(x0, y0, 0.0), Point3::new(x1, y1, 10.0)],
        vec![1.0, 1.0],
    )
    .unwrap()
}

#[test]
fn sweep_guided_produces_valid_solid() {
    let mut topo = Topology::new();
    let profile = make_square(&mut topo, 4.0);
    let spine = straight_z_path(10.0);
    let aux = guide_line(10.0, 0.0, 10.0, 0.0);
    let solid = sweep_guided(&mut topo, profile, &spine, aux).unwrap();
    let vol = crate::measure::solid_volume(&topo, solid, 0.1).unwrap();
    assert!(vol > 0.0, "guided sweep volume, got {vol}");
}

#[test]
fn sweep_guided_rotating_aux_rolls_profile() {
    // A wide rectangle swept straight stays flat in Y. A guide that rotates
    // from +X to +Y over the sweep rolls the rectangle 90°, sweeping its
    // wide axis into Y — so the Y-extent grows far beyond the flat case.
    let mut t_plain = Topology::new();
    let p_plain = make_rect(&mut t_plain, 8.0, 1.0);
    let s_plain = sweep_with_options(
        &mut t_plain,
        p_plain,
        &straight_z_path(10.0),
        &SweepOptions::default(),
    )
    .unwrap();
    let bb_plain = crate::measure::solid_bounding_box(&t_plain, s_plain).unwrap();
    let y_plain = bb_plain.max.y() - bb_plain.min.y();

    let mut t_guided = Topology::new();
    let p_guided = make_rect(&mut t_guided, 8.0, 1.0);
    let aux = guide_line(30.0, 0.0, 0.0, 30.0);
    let s_guided = sweep_guided(&mut t_guided, p_guided, &straight_z_path(10.0), aux).unwrap();
    let bb_guided = crate::measure::solid_bounding_box(&t_guided, s_guided).unwrap();
    let y_guided = bb_guided.max.y() - bb_guided.min.y();

    assert!(
        y_plain < 4.0,
        "plain sweep keeps the rectangle flat in Y, got {y_plain}"
    );
    assert!(
        y_guided > 8.0,
        "the rotating guide should roll the wide axis into Y, got {y_guided}"
    );
}

#[test]
fn sweep_guided_handles_guide_meeting_spine() {
    // The guide starts coincident with the spine (up undefined at t=0) then
    // diverges — frame continuity must still yield a valid finite solid.
    let mut topo = Topology::new();
    let profile = make_square(&mut topo, 3.0);
    let spine = straight_z_path(10.0);
    let aux = guide_line(0.0, 0.0, 10.0, 0.0);
    let solid = sweep_guided(&mut topo, profile, &spine, aux).unwrap();
    let vol = crate::measure::solid_volume(&topo, solid, 0.1).unwrap();
    assert!(
        vol > 0.0 && vol.is_finite(),
        "guide-meets-spine should still yield a valid solid, got {vol}"
    );
}

#[test]
fn sweep_square_along_line() {
    // Sweeping along a straight line should produce a box-like solid,
    // similar to extrude.
    let mut topo = Topology::new();
    let face = make_unit_square_face(&mut topo);
    let path = straight_z_path(2.0);

    let solid = sweep(&mut topo, face, &path).unwrap();

    let solid_data = topo.solid(solid).unwrap();
    let shell = topo.shell(solid_data.outer_shell()).unwrap();

    // 4 segments (from max(2*2, 4)) × 4 profile edges = 16 side faces
    // + 2 caps = 18 faces total.
    let num_segs = (path.control_points().len() * 2).max(4);
    let expected_faces = num_segs * 4 + 2;
    assert_eq!(shell.faces().len(), expected_faces);

    // Verify all faces are planar.
    for &fid in shell.faces() {
        let f = topo.face(fid).unwrap();
        assert!(
            matches!(f.surface(), FaceSurface::Plane { .. }),
            "all sweep V1 faces should be planar"
        );
    }
}

#[test]
fn sweep_square_along_quarter_circle() {
    let mut topo = Topology::new();
    let face = make_unit_square_face(&mut topo);
    let path = quarter_circle_xz_path(5.0);

    let solid = sweep(&mut topo, face, &path).unwrap();

    let solid_data = topo.solid(solid).unwrap();
    let shell = topo.shell(solid_data.outer_shell()).unwrap();

    // 6 segments (max(3*2, 4)) × 4 edges + 2 caps = 26 faces.
    let num_segs = (path.control_points().len() * 2).max(4);
    let expected_faces = num_segs * 4 + 2;
    assert_eq!(shell.faces().len(), expected_faces);

    // Verify manifold: every edge shared by exactly 2 faces.
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

#[test]
fn sweep_insufficient_control_points_error() {
    let mut topo = Topology::new();
    let face = make_unit_square_face(&mut topo);

    // A path with only 1 control point is invalid.
    let path = NurbsCurve::new(
        0,
        vec![0.0, 1.0],
        vec![Point3::new(0.0, 0.0, 0.0)],
        vec![1.0],
    )
    .unwrap();

    let result = sweep(&mut topo, face, &path);
    assert!(result.is_err());
}

#[test]
fn sweep_zero_path_error() {
    let mut topo = Topology::new();
    let face = make_unit_square_face(&mut topo);

    // A path where start == end (zero length).
    let path = NurbsCurve::new(
        1,
        vec![0.0, 0.0, 1.0, 1.0],
        vec![Point3::new(1.0, 2.0, 3.0), Point3::new(1.0, 2.0, 3.0)],
        vec![1.0, 1.0],
    )
    .unwrap();

    let result = sweep(&mut topo, face, &path);
    assert!(result.is_err());
}

#[test]
fn sweep_and_tessellate_roundtrip() {
    use crate::tessellate::tessellate;

    let mut topo = Topology::new();
    let face = make_unit_square_face(&mut topo);
    let path = quarter_circle_xz_path(5.0);

    let solid = sweep(&mut topo, face, &path).unwrap();

    let solid_data = topo.solid(solid).unwrap();
    let shell = topo.shell(solid_data.outer_shell()).unwrap();
    let tol = Tolerance::new();

    for &fid in shell.faces() {
        let mesh = tessellate(&topo, fid, 0.25).unwrap();
        assert!(!mesh.positions.is_empty());
        assert!(!mesh.indices.is_empty());
        assert_eq!(mesh.positions.len(), mesh.normals.len());

        for normal in &mesh.normals {
            let len = normal.length();
            assert!(
                tol.approx_eq(len, 1.0) || tol.approx_eq(len, 0.0),
                "normal length should be ~1.0, got {len}"
            );
        }
    }
}

#[test]
fn sweep_with_default_options_matches_basic() {
    let mut topo = Topology::new();
    let face = crate::primitives::make_box(&mut topo, 0.5, 0.5, 0.01).unwrap();
    let solid = topo.solid(face).unwrap();
    let shell = topo.shell(solid.outer_shell()).unwrap();
    let profile = shell.faces()[0];

    let path = NurbsCurve::new(
        1,
        vec![0.0, 0.0, 1.0, 1.0],
        vec![Point3::new(0.0, 0.0, 0.0), Point3::new(0.0, 0.0, 5.0)],
        vec![1.0, 1.0],
    )
    .unwrap();

    let options = SweepOptions::default();
    let result = sweep_with_options(&mut topo, profile, &path, &options);
    assert!(result.is_ok());
}

#[test]
fn sweep_with_linear_scale() {
    let mut topo = Topology::new();
    let profile = make_unit_square_face(&mut topo);

    let path = NurbsCurve::new(
        1,
        vec![0.0, 0.0, 1.0, 1.0],
        vec![Point3::new(0.0, 0.0, 0.0), Point3::new(0.0, 0.0, 5.0)],
        vec![1.0, 1.0],
    )
    .unwrap();

    let options = SweepOptions {
        scale_law: Some(Box::new(|t| 0.5f64.mul_add(-t, 1.0))), // taper from 1.0 to 0.5
        segments: 8,
        ..Default::default()
    };

    let result = sweep_with_options(&mut topo, profile, &path, &options).unwrap();

    let vol = crate::measure::solid_volume(&topo, result, 0.5).unwrap();
    assert!(vol > 0.0, "tapered sweep should have positive volume");
}

#[test]
fn sweep_fixed_contact_mode() {
    let mut topo = Topology::new();
    let profile = make_unit_square_face(&mut topo);

    let path = NurbsCurve::new(
        1,
        vec![0.0, 0.0, 1.0, 1.0],
        vec![Point3::new(0.0, 0.0, 0.0), Point3::new(0.0, 0.0, 5.0)],
        vec![1.0, 1.0],
    )
    .unwrap();

    let options = SweepOptions {
        contact_mode: SweepContactMode::Fixed,
        ..Default::default()
    };

    let result = sweep_with_options(&mut topo, profile, &path, &options);
    assert!(result.is_ok());
}

#[test]
fn sweep_constant_normal_mode() {
    let mut topo = Topology::new();
    let profile = make_unit_square_face(&mut topo);

    let path = NurbsCurve::new(
        1,
        vec![0.0, 0.0, 1.0, 1.0],
        vec![Point3::new(0.0, 0.0, 0.0), Point3::new(0.0, 0.0, 5.0)],
        vec![1.0, 1.0],
    )
    .unwrap();

    let options = SweepOptions {
        contact_mode: SweepContactMode::ConstantNormal(Vec3::new(0.0, 1.0, 0.0)),
        ..Default::default()
    };

    let result = sweep_with_options(&mut topo, profile, &path, &options);
    assert!(result.is_ok());
}

// ── Smooth sweep tests ──────────────────────────

#[test]
fn sweep_smooth_produces_nurbs_sides() {
    let mut topo = Topology::new();
    let profile = make_unit_square_face(&mut topo);
    let path = straight_z_path(2.0);

    let solid = sweep_smooth(&mut topo, profile, &path).unwrap();

    let s = topo.solid(solid).unwrap();
    let sh = topo.shell(s.outer_shell()).unwrap();

    // Should have N NURBS sides + 2 planar caps.
    let nurbs_count = sh
        .faces()
        .iter()
        .filter(|&&fid| matches!(topo.face(fid).unwrap().surface(), FaceSurface::Nurbs(_)))
        .count();

    assert!(
        nurbs_count > 0,
        "smooth sweep should produce NURBS side faces"
    );

    // Fewer faces than the basic sweep (N sides vs N*segments sides).
    let profile_edge_count = 4; // square has 4 edges
    let expected_face_count = profile_edge_count + 2; // N sides + 2 caps
    assert_eq!(
        sh.faces().len(),
        expected_face_count,
        "smooth sweep should have {expected_face_count} faces, got {}",
        sh.faces().len()
    );
}

#[test]
fn sweep_smooth_positive_volume() {
    let mut topo = Topology::new();
    let profile = make_unit_square_face(&mut topo);
    let path = straight_z_path(3.0);

    let solid = sweep_smooth(&mut topo, profile, &path).unwrap();

    let vol = crate::measure::solid_volume(&topo, solid, 0.1).unwrap();
    assert!(
        vol > 0.0,
        "smooth sweep should have positive volume, got {vol}"
    );
}

#[test]
fn sweep_smooth_curved_path() {
    let mut topo = Topology::new();
    let profile = make_unit_square_face(&mut topo);
    let path = quarter_circle_xz_path(5.0);

    let solid = sweep_smooth(&mut topo, profile, &path).unwrap();

    let s = topo.solid(solid).unwrap();
    let sh = topo.shell(s.outer_shell()).unwrap();

    assert_eq!(
        sh.faces().len(),
        6,
        "smooth curved sweep should have 6 faces"
    );

    let vol = crate::measure::solid_volume(&topo, solid, 0.1).unwrap();
    assert!(vol > 0.0, "curved smooth sweep should have positive volume");
}

/// Helper: create a closed circular NURBS path (full circle in XZ plane).
///
/// Uses the XZ plane so that a profile in XY sweeps with full 3D extent
/// (the path tangent at t=0 is +Z, giving the profile extent in both
/// right(Y) and up(X) directions relative to the frame).
fn closed_circle_path(radius: f64) -> NurbsCurve {
    // Full circle as a rational quadratic NURBS with 9 control points.
    let w = std::f64::consts::FRAC_1_SQRT_2;
    let r = radius;
    NurbsCurve::new(
        2,
        vec![
            0.0, 0.0, 0.0, 0.25, 0.25, 0.5, 0.5, 0.75, 0.75, 1.0, 1.0, 1.0,
        ],
        vec![
            Point3::new(r, 0.0, 0.0),
            Point3::new(r, 0.0, r),
            Point3::new(0.0, 0.0, r),
            Point3::new(-r, 0.0, r),
            Point3::new(-r, 0.0, 0.0),
            Point3::new(-r, 0.0, -r),
            Point3::new(0.0, 0.0, -r),
            Point3::new(r, 0.0, -r),
            Point3::new(r, 0.0, 0.0),
        ],
        vec![1.0, w, 1.0, w, 1.0, w, 1.0, w, 1.0],
    )
    .unwrap()
}

#[test]
fn sweep_closed_circular_path() {
    // Sweeping a small square profile around a closed circle should
    // produce a torus-like solid with no cap faces.
    let mut topo = Topology::new();
    let profile = make_unit_square_face(&mut topo);
    let path = closed_circle_path(5.0);

    let solid = sweep(&mut topo, profile, &path).unwrap();

    let solid_data = topo.solid(solid).unwrap();
    let shell = topo.shell(solid_data.outer_shell()).unwrap();

    // Closed sweep: no caps, only side faces.
    let num_segs = (path.control_points().len() * 2).max(4);
    let expected_faces = num_segs * 4; // 4 edges per profile × num_segments
    assert_eq!(
        shell.faces().len(),
        expected_faces,
        "closed sweep should have {expected_faces} side faces (no caps)"
    );

    // Verify manifold: every edge shared by exactly 2 faces.
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
            "edge {edge_idx} shared by {count} faces, expected 2 (manifold)"
        );
    }

    // Should have positive volume.
    let vol = crate::measure::solid_volume(&topo, solid, 0.1).unwrap();
    assert!(
        vol > 0.0,
        "closed sweep should have positive volume, got {vol}"
    );
}

/// Helper: create a square face with a smaller square hole (inner wire).
fn make_square_face_with_hole(topo: &mut Topology) -> FaceId {
    use brepkit_topology::edge::{Edge, EdgeCurve};
    use brepkit_topology::face::{Face, FaceSurface};
    use brepkit_topology::vertex::Vertex;
    use brepkit_topology::wire::{OrientedEdge, Wire};

    let lin = Tolerance::new().linear;

    // Outer square: 2x2 centered at origin in XY plane
    let ov0 = topo.add_vertex(Vertex::new(Point3::new(-1.0, -1.0, 0.0), lin));
    let ov1 = topo.add_vertex(Vertex::new(Point3::new(1.0, -1.0, 0.0), lin));
    let ov2 = topo.add_vertex(Vertex::new(Point3::new(1.0, 1.0, 0.0), lin));
    let ov3 = topo.add_vertex(Vertex::new(Point3::new(-1.0, 1.0, 0.0), lin));

    let oe0 = topo.add_edge(Edge::new(ov0, ov1, EdgeCurve::Line));
    let oe1 = topo.add_edge(Edge::new(ov1, ov2, EdgeCurve::Line));
    let oe2 = topo.add_edge(Edge::new(ov2, ov3, EdgeCurve::Line));
    let oe3 = topo.add_edge(Edge::new(ov3, ov0, EdgeCurve::Line));

    let outer_wire = topo.add_wire(
        Wire::new(
            vec![
                OrientedEdge::new(oe0, true),
                OrientedEdge::new(oe1, true),
                OrientedEdge::new(oe2, true),
                OrientedEdge::new(oe3, true),
            ],
            true,
        )
        .unwrap(),
    );

    // Inner square: 0.5x0.5 centered at origin (hole)
    let iv0 = topo.add_vertex(Vertex::new(Point3::new(-0.25, -0.25, 0.0), lin));
    let iv1 = topo.add_vertex(Vertex::new(Point3::new(0.25, -0.25, 0.0), lin));
    let iv2 = topo.add_vertex(Vertex::new(Point3::new(0.25, 0.25, 0.0), lin));
    let iv3 = topo.add_vertex(Vertex::new(Point3::new(-0.25, 0.25, 0.0), lin));

    let ie0 = topo.add_edge(Edge::new(iv0, iv1, EdgeCurve::Line));
    let ie1 = topo.add_edge(Edge::new(iv1, iv2, EdgeCurve::Line));
    let ie2 = topo.add_edge(Edge::new(iv2, iv3, EdgeCurve::Line));
    let ie3 = topo.add_edge(Edge::new(iv3, iv0, EdgeCurve::Line));

    let inner_wire = topo.add_wire(
        Wire::new(
            vec![
                OrientedEdge::new(ie0, true),
                OrientedEdge::new(ie1, true),
                OrientedEdge::new(ie2, true),
                OrientedEdge::new(ie3, true),
            ],
            true,
        )
        .unwrap(),
    );

    topo.add_face(Face::new(
        outer_wire,
        vec![inner_wire],
        FaceSurface::Plane {
            normal: Vec3::new(0.0, 0.0, 1.0),
            d: 0.0,
        },
    ))
}

#[test]
fn sweep_closed_path_with_inner_hole() {
    // Sweeping a profile with inner holes along a closed path should not panic.
    let mut topo = Topology::new();
    let profile = make_square_face_with_hole(&mut topo);
    let path = closed_circle_path(5.0);

    let solid = sweep(&mut topo, profile, &path).unwrap();

    let vol = crate::measure::solid_volume(&topo, solid, 0.1).unwrap();
    assert!(
        vol > 0.0,
        "closed sweep with inner hole should have positive volume, got {vol}"
    );
}

#[test]
fn sweep_smooth_closed_path() {
    // sweep_smooth delegates to sweep for closed paths.
    let mut topo = Topology::new();
    let profile = make_unit_square_face(&mut topo);
    let path = closed_circle_path(5.0);

    let solid = sweep_smooth(&mut topo, profile, &path).unwrap();

    let vol = crate::measure::solid_volume(&topo, solid, 0.1).unwrap();
    assert!(
        vol > 0.0,
        "smooth closed sweep should have positive volume, got {vol}"
    );
}

#[test]
fn sweep_cw_profile_produces_correct_solid() {
    let path = straight_z_path(3.0);
    crate::test_helpers::assert_cw_profile_produces_valid_solid(
        |topo, face| sweep(topo, face, &path).unwrap(),
        3.0,
        0.05,
    );
}

/// Translation invariance for CW-wound sweep.
#[test]
fn sweep_cw_profile_translation_invariant() {
    use brepkit_topology::test_utils::make_cw_unit_square_face;

    let mut topo1 = Topology::new();
    let face1 = make_cw_unit_square_face(&mut topo1);
    let path1 = straight_z_path(3.0);
    let solid1 = sweep(&mut topo1, face1, &path1).unwrap();
    let vol1 = crate::measure::solid_volume(&topo1, solid1, 0.1).unwrap();

    let mut topo2 = Topology::new();
    let face2 = make_cw_unit_square_face(&mut topo2);
    let path2 = straight_z_path(3.0);
    let solid2 = sweep(&mut topo2, face2, &path2).unwrap();
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
        "CW sweep volumes should match: origin={vol1}, translated={vol2}, \
             rel_err={rel_err:.2e}"
    );
}

/// Sweep a CW-wound profile along a NON-PARALLEL axis (X path, XY profile).
/// This exercises the `input_normal` negation fix — without it, the
/// `orthogonalize(input_normal, path_tangent)` up-hint is wrong and
/// the profile is flipped upside-down.
#[test]
fn sweep_cw_profile_nonparallel_axis() {
    use brepkit_topology::edge::{Edge, EdgeCurve};
    use brepkit_topology::face::Face;
    use brepkit_topology::vertex::Vertex;
    use brepkit_topology::wire::{OrientedEdge, Wire};

    let mut topo = Topology::new();
    let tol_val = 1e-7;

    // CW rectangle 1×2 on XY plane: (0,0)→(0,2)→(1,2)→(1,0)
    let v0 = topo.add_vertex(Vertex::new(Point3::new(0.0, 0.0, 0.0), tol_val));
    let v1 = topo.add_vertex(Vertex::new(Point3::new(0.0, 2.0, 0.0), tol_val));
    let v2 = topo.add_vertex(Vertex::new(Point3::new(1.0, 2.0, 0.0), tol_val));
    let v3 = topo.add_vertex(Vertex::new(Point3::new(1.0, 0.0, 0.0), tol_val));

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

    // CW winding → Newell normal = -Z
    let face = topo.add_face(Face::new(
        wid,
        vec![],
        FaceSurface::Plane {
            normal: Vec3::new(0.0, 0.0, -1.0),
            d: 0.0,
        },
    ));

    // Sweep along +Z (profile normal is perpendicular to path → up-hint matters)
    let path = straight_z_path(5.0);
    let solid = sweep(&mut topo, face, &path).unwrap();

    // Expected: 1×2×5 = 10.0
    let vol = crate::measure::solid_volume(&topo, solid, 0.1).unwrap();
    assert!(
        (vol - 10.0).abs() < 0.5,
        "CW 1×2 rectangle swept along Z should produce volume ~10.0, got {vol}"
    );
}

// ── Miter sweep tests ──────────────────────────

/// Helper: create an L-shaped polyline path (two line segments with
/// a 90-degree turn).
fn l_shaped_path() -> NurbsCurve {
    // Degree-1 NURBS: (0,0,0)→(0,0,5)→(5,0,5). Path goes along +Z
    // then turns +X — perpendicular to the XY-plane unit square
    // profile so the swept cross-section has nonzero area.
    // Internal knot at t=0.5 creates a C0 kink.
    NurbsCurve::new(
        1,
        vec![0.0, 0.0, 0.5, 1.0, 1.0],
        vec![
            Point3::new(0.0, 0.0, 0.0),
            Point3::new(0.0, 0.0, 5.0),
            Point3::new(5.0, 0.0, 5.0),
        ],
        vec![1.0, 1.0, 1.0],
    )
    .unwrap()
}

#[test]
fn detect_kinks_l_shaped_path() {
    let path = l_shaped_path();
    let kinks = detect_kinks(&path);
    assert_eq!(kinks.len(), 1, "L-shaped path should have one kink");
    assert!((kinks[0] - 0.5).abs() < 1e-6, "kink should be at t=0.5");
}

#[test]
fn detect_kinks_no_kinks_for_smooth_path() {
    // A smooth cubic NURBS with no internal knot multiplicity.
    let path = quarter_circle_xz_path(5.0);
    let kinks = detect_kinks(&path);
    assert!(kinks.is_empty(), "smooth path should have no kinks");
}

#[test]
fn detect_kinks_straight_line_no_kinks() {
    let path = straight_z_path(5.0);
    let kinks = detect_kinks(&path);
    assert!(kinks.is_empty(), "straight line should have no kinks");
}

#[test]
fn detect_kinks_collinear_polyline_no_kinks() {
    // A polyline with 3 collinear points — no tangent change.
    let path = NurbsCurve::new(
        1,
        vec![0.0, 0.0, 0.5, 1.0, 1.0],
        vec![
            Point3::new(0.0, 0.0, 0.0),
            Point3::new(5.0, 0.0, 0.0),
            Point3::new(10.0, 0.0, 0.0),
        ],
        vec![1.0, 1.0, 1.0],
    )
    .unwrap();
    let kinks = detect_kinks(&path);
    assert!(kinks.is_empty(), "collinear polyline should have no kinks");
}

#[test]
fn sweep_miter_l_shaped_path() {
    let mut topo = Topology::new();
    let profile = make_unit_square_face(&mut topo);
    let path = l_shaped_path();

    let options = SweepOptions {
        corner_mode: SweepCornerMode::Miter,
        ..Default::default()
    };
    let solid = sweep_with_options(&mut topo, profile, &path, &options).unwrap();

    let vol = crate::measure::solid_volume(&topo, solid, 0.1).unwrap();
    assert!(
        vol > 0.0,
        "miter sweep should have positive volume, got {vol}"
    );

    // Verify manifold: every edge shared by exactly 2 faces.
    let solid_data = topo.solid(solid).unwrap();
    let shell = topo.shell(solid_data.outer_shell()).unwrap();

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
            "edge {edge_idx} shared by {count} faces, expected 2 (manifold)"
        );
    }
}

#[test]
fn sweep_miter_l_shaped_volume_correct() {
    // L-shaped path: (0,0,0)→(0,0,5)→(5,0,5) with 1×1 square profile.
    // With miter, the volume is two rectangular prisms joined at a 45-degree
    // miter plane. Each leg has length ~5, profile area ~1, so total is
    // roughly 10 (minus/plus the miter overlap which approximately cancels).
    let mut topo = Topology::new();
    let profile = make_unit_square_face(&mut topo);
    let path = l_shaped_path();

    let options = SweepOptions {
        corner_mode: SweepCornerMode::Miter,
        ..Default::default()
    };
    let solid = sweep_with_options(&mut topo, profile, &path, &options).unwrap();

    let vol = crate::measure::solid_volume(&topo, solid, 0.1).unwrap();

    // The exact volume depends on the miter geometry, but should be
    // in a reasonable range for a 1×1 profile swept along two 5-unit legs.
    assert!(
        vol > 5.0 && vol < 15.0,
        "L-sweep volume should be roughly 10 (two 5-unit legs), got {vol}"
    );
}

#[test]
fn sweep_miter_u_shaped_path() {
    // U-shaped path: 3 segments with 2 kinks, in ZX plane so
    // the XY-plane profile has nonzero cross-section area.
    let path = NurbsCurve::new(
        1,
        vec![0.0, 0.0, 1.0 / 3.0, 2.0 / 3.0, 1.0, 1.0],
        vec![
            Point3::new(0.0, 0.0, 0.0),
            Point3::new(0.0, 0.0, 5.0),
            Point3::new(5.0, 0.0, 5.0),
            Point3::new(5.0, 0.0, 0.0),
        ],
        vec![1.0, 1.0, 1.0, 1.0],
    )
    .unwrap();

    let mut topo = Topology::new();
    let profile = make_unit_square_face(&mut topo);

    let options = SweepOptions {
        corner_mode: SweepCornerMode::Miter,
        ..Default::default()
    };
    let solid = sweep_with_options(&mut topo, profile, &path, &options).unwrap();

    let vol = crate::measure::solid_volume(&topo, solid, 0.1).unwrap();
    assert!(
        vol > 0.0,
        "U-shaped miter sweep should have positive volume, got {vol}"
    );
}

#[test]
fn sweep_miter_fallback_smooth_on_no_kinks() {
    // Smooth path has no kinks — miter mode should fall back to smooth.
    let mut topo = Topology::new();
    let profile = make_unit_square_face(&mut topo);
    let path = straight_z_path(5.0);

    let options = SweepOptions {
        corner_mode: SweepCornerMode::Miter,
        ..Default::default()
    };
    let solid = sweep_with_options(&mut topo, profile, &path, &options).unwrap();

    let vol = crate::measure::solid_volume(&topo, solid, 0.1).unwrap();
    assert!(
        (vol - 5.0).abs() < 1.0,
        "straight-path miter fallback should produce volume ~5.0, got {vol}"
    );
}

// ── densify_path_points (rectLipSweep non-square overshoot regression) ──

/// Sample a rounded-rectangle boundary the way `sweepAlongEdges` does: line
/// edges contribute only endpoints; corner arcs contribute interior samples.
/// This reproduces the under-sampled long-edge condition that made the global
/// interpolating path fit overshoot on non-square spines.
fn sparse_rounded_rect(half_x: f64, half_y: f64, r: f64) -> Vec<Point3> {
    let (cx, cy) = (half_x - r, half_y - r);
    let hp = std::f64::consts::FRAC_PI_2;
    let corners = [
        (cx, -cy, -hp, 0.0),
        (cx, cy, 0.0, hp),
        (-cx, cy, hp, std::f64::consts::PI),
        (-cx, -cy, std::f64::consts::PI, 3.0 * hp),
    ];
    let mut pts: Vec<Point3> = Vec::new();
    let push = |p: Point3, pts: &mut Vec<Point3>| {
        if pts.last().is_none_or(|l: &Point3| (*l - p).length() > 1e-7) {
            pts.push(p);
        }
    };
    for &(ccx, ccy, a0, a1) in &corners {
        for k in 0..=8 {
            let a = a0 + (a1 - a0) * f64::from(k) / 8.0;
            push(
                Point3::new(ccx + r * a.cos(), ccy + r * a.sin(), 0.0),
                &mut pts,
            );
        }
    }
    if let Some(first) = pts.first().copied() {
        push(first, &mut pts);
    }
    pts
}

fn path_x_extent(path: &NurbsCurve) -> (f64, f64) {
    let mut lo = f64::INFINITY;
    let mut hi = f64::NEG_INFINITY;
    for k in 0..=400 {
        let x = path.evaluate(f64::from(k) / 400.0).x();
        lo = lo.min(x);
        hi = hi.max(x);
    }
    (lo, hi)
}

#[test]
fn densify_fixes_nonsquare_spine_overshoot() {
    use brepkit_math::nurbs::fitting::interpolate;

    // 42 x 126 rounded rect → half 21 x 63, r 3.75. Long edges (~118mm) are
    // sampled only at endpoints, so the raw fit overshoots far past x = ±21.
    let pts = sparse_rounded_rect(21.0, 63.0, 3.75);

    let raw = interpolate(&pts, 3).unwrap();
    let (_, raw_hi) = path_x_extent(&raw);
    assert!(
        raw_hi > 50.0,
        "expected the un-densified fit to overshoot (got x_max {raw_hi}); \
         if this no longer overshoots the regression guard is moot"
    );

    let dense = densify_path_points(&pts);
    let fixed = interpolate(&dense, 3).unwrap();
    let (fixed_lo, fixed_hi) = path_x_extent(&fixed);
    assert!(
        fixed_hi < 22.0 && fixed_lo > -22.0,
        "densified fit must stay near the true ±21 bound, got x[{fixed_lo:.2},{fixed_hi:.2}]"
    );
}

#[test]
fn densify_leaves_uniform_polyline_unchanged() {
    // Evenly spaced points: no gap exceeds the median multiple, so no inserts.
    let pts: Vec<Point3> = (0..10)
        .map(|i| Point3::new(f64::from(i), 0.0, 0.0))
        .collect();
    assert_eq!(densify_path_points(&pts).len(), pts.len());
}

#[test]
fn densify_short_input_is_identity() {
    let pts = vec![Point3::new(0.0, 0.0, 0.0), Point3::new(10.0, 0.0, 0.0)];
    assert_eq!(densify_path_points(&pts), pts);
}
