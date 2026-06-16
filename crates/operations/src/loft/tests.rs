#![allow(clippy::unwrap_used, clippy::expect_used)]

use brepkit_math::tolerance::Tolerance;
use brepkit_math::vec::{Point3, Vec3};
use brepkit_topology::Topology;
use brepkit_topology::edge::{Edge, EdgeCurve};
use brepkit_topology::face::{Face, FaceSurface};
use brepkit_topology::vertex::Vertex;
use brepkit_topology::wire::{OrientedEdge, Wire};

use super::*;

/// Helper: make a square face at z=offset with given size.
fn make_square_at(topo: &mut Topology, size: f64, z: f64) -> FaceId {
    let hs = size / 2.0;
    let tol_val = 1e-7;
    let v0 = topo.add_vertex(Vertex::new(Point3::new(-hs, -hs, z), tol_val));
    let v1 = topo.add_vertex(Vertex::new(Point3::new(hs, -hs, z), tol_val));
    let v2 = topo.add_vertex(Vertex::new(Point3::new(hs, hs, z), tol_val));
    let v3 = topo.add_vertex(Vertex::new(Point3::new(-hs, hs, z), tol_val));

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
            d: z,
        },
    ))
}

#[test]
fn loft_two_identical_squares_makes_box() {
    let mut topo = Topology::new();
    let bottom = make_square_at(&mut topo, 1.0, 0.0);
    let top = make_square_at(&mut topo, 1.0, 1.0);

    let solid = loft(&mut topo, &[bottom, top]).unwrap();

    let s = topo.solid(solid).unwrap();
    let sh = topo.shell(s.outer_shell()).unwrap();

    // 2 caps + 4 sides = 6 faces
    assert_eq!(sh.faces().len(), 6, "lofted box should have 6 faces");

    // Volume should be 1.0 (unit cube)
    let vol = crate::measure::solid_volume(&topo, solid, 0.1).unwrap();
    let tol = Tolerance::loose();
    assert!(
        tol.approx_eq(vol, 1.0),
        "lofted box volume should be ~1.0, got {vol}"
    );
}

#[test]
fn loft_tapered_frustum() {
    let mut topo = Topology::new();
    let bottom = make_square_at(&mut topo, 2.0, 0.0);
    let top = make_square_at(&mut topo, 1.0, 3.0);

    let solid = loft(&mut topo, &[bottom, top]).unwrap();

    let s = topo.solid(solid).unwrap();
    let sh = topo.shell(s.outer_shell()).unwrap();

    assert_eq!(sh.faces().len(), 6);

    let vol = crate::measure::solid_volume(&topo, solid, 0.1).unwrap();
    // Frustum of a square pyramid: V = h/3 * (A1 + A2 + sqrt(A1*A2))
    // A1 = 4.0, A2 = 1.0, h = 3.0
    // V = 3/3 * (4 + 1 + 2) = 7.0
    let expected = 7.0;
    assert!(
        (vol - expected).abs() / expected < 0.05,
        "tapered frustum volume should be ~{expected}, got {vol} (error: {:.1}%)",
        (vol - expected).abs() / expected * 100.0
    );
}

#[test]
fn loft_three_profiles() {
    let mut topo = Topology::new();
    let p0 = make_square_at(&mut topo, 2.0, 0.0);
    let p1 = make_square_at(&mut topo, 1.0, 1.5);
    let p2 = make_square_at(&mut topo, 2.0, 3.0);

    let solid = loft(&mut topo, &[p0, p1, p2]).unwrap();

    let s = topo.solid(solid).unwrap();
    let sh = topo.shell(s.outer_shell()).unwrap();

    // 2 caps + 2 sections × 4 edges = 10 faces
    assert_eq!(sh.faces().len(), 10);

    let vol = crate::measure::solid_volume(&topo, solid, 0.1).unwrap();
    assert!(vol > 0.0, "lofted solid should have positive volume");
}

#[test]
fn loft_single_profile_error() {
    let mut topo = Topology::new();
    let p0 = make_square_at(&mut topo, 1.0, 0.0);

    assert!(loft(&mut topo, &[p0]).is_err());
}

#[test]
fn loft_mismatched_vertex_count_error() {
    let mut topo = Topology::new();
    let square = make_square_at(&mut topo, 1.0, 0.0);

    let tol_val = 1e-7;
    let v0 = topo.add_vertex(Vertex::new(Point3::new(0.0, 0.0, 1.0), tol_val));
    let v1 = topo.add_vertex(Vertex::new(Point3::new(1.0, 0.0, 1.0), tol_val));
    let v2 = topo.add_vertex(Vertex::new(Point3::new(0.5, 1.0, 1.0), tol_val));

    let e0 = topo.add_edge(Edge::new(v0, v1, EdgeCurve::Line));
    let e1 = topo.add_edge(Edge::new(v1, v2, EdgeCurve::Line));
    let e2 = topo.add_edge(Edge::new(v2, v0, EdgeCurve::Line));

    let wire = Wire::new(
        vec![
            OrientedEdge::new(e0, true),
            OrientedEdge::new(e1, true),
            OrientedEdge::new(e2, true),
        ],
        true,
    )
    .unwrap();
    let wid = topo.add_wire(wire);
    let tri = topo.add_face(Face::new(
        wid,
        vec![],
        FaceSurface::Plane {
            normal: Vec3::new(0.0, 0.0, 1.0),
            d: 1.0,
        },
    ));

    // Profiles with different vertex counts should succeed via resampling.
    let result = loft(&mut topo, &[square, tri]);
    assert!(
        result.is_ok(),
        "loft with different vertex counts should succeed via resampling"
    );
}

/// Helper: make a CW-wound square face at z=offset with given size.
fn make_cw_square_at(topo: &mut Topology, size: f64, z: f64) -> FaceId {
    let hs = size / 2.0;
    let tol_val = 1e-7;
    // CW order: v0→v3→v2→v1 (reversed from make_square_at)
    let v0 = topo.add_vertex(Vertex::new(Point3::new(-hs, -hs, z), tol_val));
    let v1 = topo.add_vertex(Vertex::new(Point3::new(-hs, hs, z), tol_val));
    let v2 = topo.add_vertex(Vertex::new(Point3::new(hs, hs, z), tol_val));
    let v3 = topo.add_vertex(Vertex::new(Point3::new(hs, -hs, z), tol_val));

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
            d: z,
        },
    ))
}

#[test]
fn loft_cw_profiles_produces_correct_solid() {
    let mut topo = Topology::new();
    let bottom = make_cw_square_at(&mut topo, 1.0, 0.0);
    let top = make_cw_square_at(&mut topo, 1.0, 1.0);

    let solid = loft(&mut topo, &[bottom, top]).unwrap();

    // CW profiles should be auto-corrected to produce a valid solid
    // with positive volume (not inside-out).
    let vol = crate::measure::solid_volume(&topo, solid, 0.1).unwrap();
    assert!(
        vol > 0.0,
        "CW-wound loft should produce positive volume, got {vol}"
    );
}

#[test]
fn loft_smooth_two_profiles_delegates() {
    // With 2 profiles, loft_smooth delegates to basic loft (ruled surfaces).
    let mut topo = Topology::new();
    let p0 = make_square_at(&mut topo, 1.0, 0.0);
    let p1 = make_square_at(&mut topo, 1.0, 1.0);

    let solid = loft_smooth(&mut topo, &[p0, p1]).unwrap();

    let s = topo.solid(solid).unwrap();
    let sh = topo.shell(s.outer_shell()).unwrap();
    assert_eq!(
        sh.faces().len(),
        6,
        "2-profile smooth loft should have 6 faces"
    );
}

#[test]
fn loft_smooth_three_profiles_has_nurbs() {
    let mut topo = Topology::new();
    let p0 = make_square_at(&mut topo, 2.0, 0.0);
    let p1 = make_square_at(&mut topo, 1.0, 1.5);
    let p2 = make_square_at(&mut topo, 2.0, 3.0);

    let solid = loft_smooth(&mut topo, &[p0, p1, p2]).unwrap();

    let s = topo.solid(solid).unwrap();
    let sh = topo.shell(s.outer_shell()).unwrap();

    // 2 caps + 4 NURBS sides = 6 faces (one surface per edge, spanning all profiles)
    assert_eq!(
        sh.faces().len(),
        6,
        "3-profile smooth loft should have 6 faces"
    );

    // Verify at least one NURBS face exists (the side surfaces).
    let has_nurbs = sh.faces().iter().any(|&fid| {
        matches!(
            topo.face(fid).expect("face").surface(),
            FaceSurface::Nurbs(_)
        )
    });
    assert!(has_nurbs, "smooth loft should produce NURBS side faces");
}

#[test]
fn loft_smooth_three_profiles_positive_volume() {
    let mut topo = Topology::new();
    let p0 = make_square_at(&mut topo, 2.0, 0.0);
    let p1 = make_square_at(&mut topo, 1.0, 1.5);
    let p2 = make_square_at(&mut topo, 2.0, 3.0);

    let solid = loft_smooth(&mut topo, &[p0, p1, p2]).unwrap();

    let vol = crate::measure::solid_volume(&topo, solid, 0.1).unwrap();
    assert!(
        vol > 0.0,
        "smooth loft should have positive volume, got {vol}"
    );
}

#[test]
fn loft_smooth_four_profiles() {
    let mut topo = Topology::new();
    let p0 = make_square_at(&mut topo, 2.0, 0.0);
    let p1 = make_square_at(&mut topo, 1.5, 1.0);
    let p2 = make_square_at(&mut topo, 1.0, 2.0);
    let p3 = make_square_at(&mut topo, 1.5, 3.0);

    let solid = loft_smooth(&mut topo, &[p0, p1, p2, p3]).unwrap();

    let s = topo.solid(solid).unwrap();
    let sh = topo.shell(s.outer_shell()).unwrap();
    assert_eq!(
        sh.faces().len(),
        6,
        "4-profile smooth loft should have 6 faces"
    );

    let vol = crate::measure::solid_volume(&topo, solid, 0.1).unwrap();
    assert!(vol > 0.0, "smooth loft should have positive volume");
}

/// Helper: make a planar circle face whose single outer edge is an
/// analytic [`Circle3D`], centered at `(0,0,z)` with axis +Z.
fn make_circle_face_at(topo: &mut Topology, radius: f64, z: f64) -> FaceId {
    let tol_val = 1e-7;
    let axis = Vec3::new(0.0, 0.0, 1.0);
    let center = Point3::new(0.0, 0.0, z);
    let circle = brepkit_math::curves::Circle3D::new(center, axis, radius).unwrap();
    let seam = topo.add_vertex(Vertex::new(circle.evaluate(0.0), tol_val));
    let edge = topo.add_edge(Edge::new(seam, seam, EdgeCurve::Circle(circle)));
    let wire = Wire::new(vec![OrientedEdge::new(edge, true)], true).unwrap();
    let wid = topo.add_wire(wire);
    topo.add_face(Face::new(
        wid,
        vec![],
        FaceSurface::Plane { normal: axis, d: z },
    ))
}

/// Helper: make a planar circle face whose outer edge is stored as a
/// rational NURBS curve that is geometrically a circle (exercises the
/// canonical-recognition branch).
fn make_nurbs_circle_face_at(topo: &mut Topology, radius: f64, z: f64) -> FaceId {
    let tol_val = 1e-7;
    let axis = Vec3::new(0.0, 0.0, 1.0);
    let center = Point3::new(0.0, 0.0, z);
    let circle = brepkit_math::curves::Circle3D::new(center, axis, radius).unwrap();
    let nurbs =
        brepkit_geometry::convert::circle_to_nurbs(&circle, 0.0, 2.0 * std::f64::consts::PI)
            .unwrap();
    let seam = topo.add_vertex(Vertex::new(circle.evaluate(0.0), tol_val));
    let edge = topo.add_edge(Edge::new(seam, seam, EdgeCurve::NurbsCurve(nurbs)));
    let wire = Wire::new(vec![OrientedEdge::new(edge, true)], true).unwrap();
    let wid = topo.add_wire(wire);
    topo.add_face(Face::new(
        wid,
        vec![],
        FaceSurface::Plane { normal: axis, d: z },
    ))
}

fn assert_analytic_frustum_solid(topo: &Topology, solid: SolidId, expected_volume: f64) {
    let s = topo.solid(solid).unwrap();
    let sh = topo.shell(s.outer_shell()).unwrap();
    assert_eq!(
        sh.faces().len(),
        3,
        "analytic loft must emit exactly 3 faces (two caps + one analytic side)"
    );
    let has_curved_side = sh.faces().iter().any(|&fid| {
        matches!(
            topo.face(fid).unwrap().surface(),
            FaceSurface::Cylinder(_) | FaceSurface::Cone(_)
        )
    });
    assert!(
        has_curved_side,
        "analytic loft side face must be cylindrical/conical, not planar"
    );
    let vol = crate::measure::solid_volume(topo, solid, 0.05).unwrap();
    let rel_err = (vol - expected_volume).abs() / expected_volume;
    assert!(
        rel_err < 0.005,
        "analytic loft volume {vol} should be within 0.5% of {expected_volume} (err {:.3}%)",
        rel_err * 100.0
    );
}

#[test]
fn loft_two_circles_volume_within_0_5pct_of_truncated_cone() {
    let h = 20.0;
    let big = 10.0;
    let small = 5.0;
    let expected = std::f64::consts::PI * h / 3.0 * (big * big + big * small + small * small);

    let mut topo = Topology::new();
    let bottom = make_circle_face_at(&mut topo, big, 0.0);
    let top = make_circle_face_at(&mut topo, small, h);
    let solid = loft(&mut topo, &[bottom, top]).unwrap();
    assert_analytic_frustum_solid(&topo, solid, expected);

    let mut topo2 = Topology::new();
    let bottom2 = make_nurbs_circle_face_at(&mut topo2, big, 0.0);
    let top2 = make_nurbs_circle_face_at(&mut topo2, small, h);
    let solid2 = loft(&mut topo2, &[bottom2, top2]).unwrap();
    assert_analytic_frustum_solid(&topo2, solid2, expected);
}

#[test]
fn loft_two_equal_circles_makes_cylinder() {
    let h = 20.0;
    let r = 10.0;
    let expected = std::f64::consts::PI * r * r * h;

    let mut topo = Topology::new();
    let bottom = make_circle_face_at(&mut topo, r, 0.0);
    let top = make_circle_face_at(&mut topo, r, h);
    let solid = loft(&mut topo, &[bottom, top]).unwrap();
    assert_analytic_frustum_solid(&topo, solid, expected);
}

#[test]
fn loft_non_coaxial_circles_falls_back_with_positive_volume() {
    let mut topo = Topology::new();
    let bottom = make_circle_face_at(&mut topo, 10.0, 0.0);
    let tol_val = 1e-7;
    let axis = Vec3::new(0.0, 0.0, 1.0);
    let center = Point3::new(8.0, 0.0, 20.0);
    let circle = brepkit_math::curves::Circle3D::new(center, axis, 5.0).unwrap();
    let seam = topo.add_vertex(Vertex::new(circle.evaluate(0.0), tol_val));
    let edge = topo.add_edge(Edge::new(seam, seam, EdgeCurve::Circle(circle)));
    let wire = Wire::new(vec![OrientedEdge::new(edge, true)], true).unwrap();
    let wid = topo.add_wire(wire);
    let top = topo.add_face(Face::new(
        wid,
        vec![],
        FaceSurface::Plane {
            normal: axis,
            d: 20.0,
        },
    ));

    let solid = loft(&mut topo, &[bottom, top]).unwrap();
    let vol = crate::measure::solid_volume(&topo, solid, 0.1).unwrap();
    assert!(
        vol > 0.0,
        "non-coaxial loft should fall back to a positive-volume solid, got {vol}"
    );
}

#[test]
fn loft_three_coaxial_circles_two_analytic_bands() {
    let r_big = 10.0;
    let r_mid = 5.0;
    let h = 20.0;
    let band = |ra: f64, rb: f64| std::f64::consts::PI * h / 3.0 * (ra * ra + ra * rb + rb * rb);
    let expected = band(r_big, r_mid) + band(r_mid, r_big);

    let mut topo = Topology::new();
    let p0 = make_circle_face_at(&mut topo, r_big, 0.0);
    let p1 = make_circle_face_at(&mut topo, r_mid, h);
    let p2 = make_circle_face_at(&mut topo, r_big, 2.0 * h);

    let solid = loft(&mut topo, &[p0, p1, p2]).unwrap();
    let s = topo.solid(solid).unwrap();
    let sh = topo.shell(s.outer_shell()).unwrap();

    let analytic_sides = sh
        .faces()
        .iter()
        .filter(|&&fid| {
            matches!(
                topo.face(fid).unwrap().surface(),
                FaceSurface::Cylinder(_) | FaceSurface::Cone(_)
            )
        })
        .count();
    assert_eq!(
        analytic_sides, 2,
        "three coaxial circles should emit two analytic frustum bands"
    );

    let vol = crate::measure::solid_volume(&topo, solid, 0.05).unwrap();
    let rel_err = (vol - expected).abs() / expected;
    assert!(
        rel_err < 0.005,
        "three-circle loft volume {vol} should be within 0.5% of {expected} (err {:.3}%)",
        rel_err * 100.0
    );
}

#[test]
fn loft_smooth_surface_passes_through_profiles() {
    let mut topo = Topology::new();
    let p0 = make_square_at(&mut topo, 2.0, 0.0);
    let p1 = make_square_at(&mut topo, 1.0, 2.0);
    let p2 = make_square_at(&mut topo, 2.0, 4.0);

    let solid = loft_smooth(&mut topo, &[p0, p1, p2]).unwrap();

    let s = topo.solid(solid).unwrap();
    let sh = topo.shell(s.outer_shell()).unwrap();

    // Find a NURBS side face and verify it passes through the middle profile.
    for &fid in sh.faces() {
        let face = topo.face(fid).expect("face");
        if let FaceSurface::Nurbs(surface) = face.surface() {
            // At u=0.5 (middle profile), the surface should pass through
            // the middle profile's vertex positions. Evaluate at u=0.5, v=0.
            let mid_pt = surface.evaluate(0.5, 0.0);
            // The middle profile is at z=2.0.
            assert!(
                (mid_pt.z() - 2.0).abs() < 0.5,
                "surface at u=0.5 should be near z=2.0, got z={:.3}",
                mid_pt.z()
            );
            break;
        }
    }
}

/// Rounded-rect profile (4 Line edges + 4 Circle arc corners), CCW.
fn make_rr_arcs(topo: &mut Topology, hw: f64, hd: f64, r: f64, z: f64) -> FaceId {
    use brepkit_math::curves::Circle3D;
    let r = r.min(hw.min(hd));
    let cc = [
        Point3::new(hw - r, -hd + r, z),
        Point3::new(hw - r, hd - r, z),
        Point3::new(-hw + r, hd - r, z),
        Point3::new(-hw + r, -hd + r, z),
    ];
    let ap = [
        (Point3::new(hw - r, -hd, z), Point3::new(hw, -hd + r, z)),
        (Point3::new(hw, hd - r, z), Point3::new(hw - r, hd, z)),
        (Point3::new(-hw + r, hd, z), Point3::new(-hw, hd - r, z)),
        (Point3::new(-hw, -hd + r, z), Point3::new(-hw + r, -hd, z)),
    ];
    let axis = Vec3::new(0.0, 0.0, 1.0);
    let mut v = Vec::new();
    for p in &ap {
        v.push(topo.add_vertex(Vertex::new(p.0, 1e-7)));
        v.push(topo.add_vertex(Vertex::new(p.1, 1e-7)));
    }
    let mut e = Vec::new();
    e.push(topo.add_edge(Edge::new(v[7], v[0], EdgeCurve::Line)));
    for i in 0..4 {
        e.push(topo.add_edge(Edge::new(
            v[2 * i],
            v[2 * i + 1],
            EdgeCurve::Circle(Circle3D::new(cc[i], axis, r).unwrap()),
        )));
        if i < 3 {
            e.push(topo.add_edge(Edge::new(v[2 * i + 1], v[2 * i + 2], EdgeCurve::Line)));
        }
    }
    let wire = Wire::new(
        e.iter().map(|&id| OrientedEdge::new(id, true)).collect(),
        true,
    )
    .unwrap();
    let wid = topo.add_wire(wire);
    topo.add_face(Face::new(
        wid,
        vec![],
        FaceSurface::Plane { normal: axis, d: z },
    ))
}

#[test]
fn loft_rounded_rect_preserves_arc_corners() {
    let mut topo = Topology::new();
    let bottom = make_rr_arcs(&mut topo, 10.0, 10.0, 2.0, 0.0);
    let top = make_rr_arcs(&mut topo, 14.0, 14.0, 3.0, 10.0);

    let solid = loft(&mut topo, &[bottom, top]).unwrap();

    let s = topo.solid(solid).unwrap();
    let sh = topo.shell(s.outer_shell()).unwrap();

    // 2 caps + (4 straight walls + 4 arc corners) = 10 faces; the corners
    // must be NURBS (arcs preserved), not faceted into Line walls.
    let faces = sh.faces();
    assert_eq!(faces.len(), 10, "rounded-rect loft should have 10 faces");
    let nurbs_faces = faces
        .iter()
        .filter(|&&f| matches!(topo.face(f).unwrap().surface(), FaceSurface::Nurbs(_)))
        .count();
    assert_eq!(
        nurbs_faces, 4,
        "the 4 arc corners should be NURBS side faces"
    );

    // Watertight genus-0: every edge shared by exactly 2 faces, euler == 2.
    let manifold = brepkit_topology::validation::validate_shell_manifold(sh, &topo);
    assert!(
        manifold.is_ok(),
        "loft should be a closed manifold: {manifold:?}"
    );
    let (f, e, v) = brepkit_topology::explorer::solid_entity_counts(&topo, solid).unwrap();
    #[allow(clippy::cast_possible_wrap)]
    let euler = (v as i64) - (e as i64) + (f as i64);
    assert_eq!(
        euler, 2,
        "rounded-rect frustum should be genus-0 (F={f} E={e} V={v})"
    );

    // Volume sits between the inscribed and circumscribed box frustums and
    // exceeds the straight-chamfer (octagon) approximation — i.e. the arcs
    // genuinely bulge out. Mean cross-section ≈ rr(12,12,2.5) area.
    let vol = crate::measure::solid_volume(&topo, solid, 0.01).unwrap();
    let rr_area = |hw: f64, hd: f64, r: f64| 4.0 * hw * hd + (std::f64::consts::PI - 4.0) * r * r;
    let approx = 0.5 * (rr_area(10.0, 10.0, 2.0) + rr_area(14.0, 14.0, 3.0)) * 10.0;
    assert!(
        (vol - approx).abs() / approx < 0.02,
        "rounded-rect frustum volume {vol:.1} should be within 2% of ~{approx:.1}"
    );
}

#[test]
fn loft_multi_section_rounded_rect_preserves_arc_corners() {
    // Regression: a multi-section (>2 profile) rounded-rect loft must keep its
    // arc corners curve-preserved (NURBS/Cone), not facet them to a polygon. A
    // faceted multi-section lip/socket is what drove the gridfinity bin fuse to
    // its non-manifold mesh fallback. Before the fix this was gated out
    // (`num_profiles != 2`) and fell back to the all-planar polygon loft.
    let mut topo = Topology::new();
    let p: Vec<FaceId> = [
        (10.0, 10.0, 2.0, 0.0),
        (11.0, 11.0, 2.2, 3.0),
        (12.0, 12.0, 2.5, 6.0),
        (14.0, 14.0, 3.0, 10.0),
    ]
    .iter()
    .map(|&(hw, hd, r, z)| make_rr_arcs(&mut topo, hw, hd, r, z))
    .collect();

    let solid = loft(&mut topo, &p).unwrap();
    let s = topo.solid(solid).unwrap();
    let sh = topo.shell(s.outer_shell()).unwrap();
    let faces = sh.faces();

    // 2 caps + 3 bands * (4 straight walls + 4 arc corners) = 26 faces.
    assert_eq!(
        faces.len(),
        26,
        "4-section rounded-rect loft should have 26 faces"
    );
    let curved = faces
        .iter()
        .filter(|&&f| {
            matches!(
                topo.face(f).unwrap().surface(),
                FaceSurface::Nurbs(_) | FaceSurface::Cone(_) | FaceSurface::Cylinder(_)
            )
        })
        .count();
    assert_eq!(
        curved, 12,
        "3 bands x 4 arc corners = 12 curve-preserved side faces (not faceted)"
    );

    let manifold = brepkit_topology::validation::validate_shell_manifold(sh, &topo);
    assert!(
        manifold.is_ok(),
        "multi-section loft should be a closed manifold: {manifold:?}"
    );
    let (f, e, v) = brepkit_topology::explorer::solid_entity_counts(&topo, solid).unwrap();
    #[allow(clippy::cast_possible_wrap)]
    let euler = (v as i64) - (e as i64) + (f as i64);
    assert_eq!(
        euler, 2,
        "multi-section frustum should be genus-0 (F={f} E={e} V={v})"
    );
}

/// Rounded-rect profile with each 90° corner split into TWO co-circular arcs,
/// mimicking drawn rounded-rects that split corners inconsistently with size.
fn make_rr_arcs_split(topo: &mut Topology, hw: f64, hd: f64, r: f64, z: f64) -> FaceId {
    use brepkit_math::curves::Circle3D;
    let r = r.min(hw.min(hd));
    let cc = [
        Point3::new(hw - r, -hd + r, z),
        Point3::new(hw - r, hd - r, z),
        Point3::new(-hw + r, hd - r, z),
        Point3::new(-hw + r, -hd + r, z),
    ];
    let ap = [
        (Point3::new(hw - r, -hd, z), Point3::new(hw, -hd + r, z)),
        (Point3::new(hw, hd - r, z), Point3::new(hw - r, hd, z)),
        (Point3::new(-hw + r, hd, z), Point3::new(-hw, hd - r, z)),
        (Point3::new(-hw, -hd + r, z), Point3::new(-hw + r, -hd, z)),
    ];
    let axis = Vec3::new(0.0, 0.0, 1.0);
    let mut v = Vec::new();
    for p in &ap {
        v.push(topo.add_vertex(Vertex::new(p.0, 1e-7)));
        v.push(topo.add_vertex(Vertex::new(p.1, 1e-7)));
    }
    let mut e = Vec::new();
    e.push(topo.add_edge(Edge::new(v[7], v[0], EdgeCurve::Line)));
    for i in 0..4 {
        let circle = Circle3D::new(cc[i], axis, r).unwrap();
        // Bisector midpoint splits the 90° corner into two co-circular sub-arcs.
        let bis = ((ap[i].0 - cc[i]) + (ap[i].1 - cc[i])).normalize().unwrap();
        let vmid = topo.add_vertex(Vertex::new(cc[i] + bis * r, 1e-7));
        e.push(topo.add_edge(Edge::new(v[2 * i], vmid, EdgeCurve::Circle(circle.clone()))));
        e.push(topo.add_edge(Edge::new(vmid, v[2 * i + 1], EdgeCurve::Circle(circle))));
        if i < 3 {
            e.push(topo.add_edge(Edge::new(v[2 * i + 1], v[2 * i + 2], EdgeCurve::Line)));
        }
    }
    let wire = Wire::new(
        e.iter().map(|&id| OrientedEdge::new(id, true)).collect(),
        true,
    )
    .unwrap();
    let wid = topo.add_wire(wire);
    topo.add_face(Face::new(
        wid,
        vec![],
        FaceSurface::Plane { normal: axis, d: z },
    ))
}

#[test]
fn loft_merges_split_arc_corners_for_curve_preservation() {
    // One profile with single-arc corners + one whose corners are split into two
    // co-circular sub-arcs (as drawn rounded-rects do, inconsistently). They must
    // still curve-preserve: the arc-merge pass canonicalizes both so their edges
    // align. Before the fix the 8-vs-12 edge-count mismatch forced the faceted
    // polygon loft — the gridfinity lip's faceting.
    let mut topo = Topology::new();
    let single = make_rr_arcs(&mut topo, 12.0, 12.0, 2.5, 0.0); // 8 edges, corner center 9.5
    let split = make_rr_arcs_split(&mut topo, 11.5, 11.5, 2.0, 5.0); // 12 edges, center 9.5
    let solid = loft(&mut topo, &[single, split]).unwrap();
    let sh = topo
        .shell(topo.solid(solid).unwrap().outer_shell())
        .unwrap();
    let curved = sh
        .faces()
        .iter()
        .filter(|&&f| {
            matches!(
                topo.face(f).unwrap().surface(),
                FaceSurface::Nurbs(_) | FaceSurface::Cone(_) | FaceSurface::Cylinder(_)
            )
        })
        .count();
    assert_eq!(
        curved, 4,
        "the 4 corners must stay curve-preserved despite the split-arc profile"
    );
    let manifold = brepkit_topology::validation::validate_shell_manifold(sh, &topo);
    assert!(
        manifold.is_ok(),
        "split-arc loft should be a closed manifold: {manifold:?}"
    );
    let (f, e, v) = brepkit_topology::explorer::solid_entity_counts(&topo, solid).unwrap();
    #[allow(clippy::cast_possible_wrap)]
    let euler = (v as i64) - (e as i64) + (f as i64);
    assert_eq!(
        euler, 2,
        "split-arc frustum should be genus-0 (F={f} E={e} V={v})"
    );
}
