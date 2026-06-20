#![allow(clippy::unwrap_used, clippy::expect_used)]

use super::*;
use crate::builder::FaceClass;
use crate::ds::Rank;
use brepkit_math::tolerance::Tolerance;
use brepkit_math::vec::{Point3, Vec3};
use brepkit_topology::builder::{make_face_from_wire, make_polygon_wire};

/// Build a planar rectangular sub-face on the z=0 plane.
fn rect_sub_face(
    topo: &mut Topology,
    min_x: f64,
    max_x: f64,
    min_y: f64,
    max_y: f64,
    rank: Rank,
    interior: Point3,
) -> SubFace {
    let pts = vec![
        Point3::new(min_x, min_y, 0.0),
        Point3::new(max_x, min_y, 0.0),
        Point3::new(max_x, max_y, 0.0),
        Point3::new(min_x, max_y, 0.0),
    ];
    let wire = make_polygon_wire(topo, &pts, 1e-7).unwrap();
    let face_id = make_face_from_wire(topo, wire).unwrap();
    SubFace {
        face_id,
        classification: FaceClass::Unknown,
        rank,
        interior_point: Some(interior),
    }
}

/// Regression test for issue #696: two same-rank planar faces with one
/// fully contained inside the other should be reported as a
/// within-rank duplicate, not as a cross-rank SD pair.
#[test]
fn detects_within_rank_planar_containment() {
    let mut topo = Topology::new();
    let arena = GfaArena::new();
    let face_ranks: HashMap<FaceId, Rank> = HashMap::new();
    let tol = Tolerance::new();

    // Large outer face (rank A) and small contained face (also rank A).
    // Edge sets differ (different vertex sets), so the edge-set pass
    // skips them — the geometric containment pass catches the dup.
    let large = rect_sub_face(
        &mut topo,
        0.0,
        10.0,
        0.0,
        10.0,
        Rank::A,
        Point3::new(5.0, 5.0, 0.0),
    );
    let small = rect_sub_face(
        &mut topo,
        3.0,
        5.0,
        3.0,
        5.0,
        Rank::A,
        Point3::new(4.0, 4.0, 0.0),
    );
    let sub_faces = vec![large, small];

    let result = detect_same_domain(&topo, &arena, &sub_faces, &face_ranks, tol);

    assert!(result.pairs.is_empty(), "no cross-rank pair expected");
    assert_eq!(
        result.within_rank_dups.len(),
        1,
        "expected exactly one within-rank duplicate"
    );
    let dup = &result.within_rank_dups[0];
    assert_eq!(dup.representative, 0, "large face (idx 0) is the rep");
    assert_eq!(dup.duplicate, 1, "small face (idx 1) is the duplicate");
}

/// Two adjacent non-overlapping coplanar faces should NOT be unioned —
/// regression guard against the over-aggressive interior-only test
/// that broke `fuse_ring_overlapping_shelled_box_height`.
#[test]
fn adjacent_coplanar_faces_not_duplicates() {
    let mut topo = Topology::new();
    let arena = GfaArena::new();
    let face_ranks: HashMap<FaceId, Rank> = HashMap::new();
    let tol = Tolerance::new();

    // Two side-by-side rectangles, sharing one edge but not overlapping.
    let left = rect_sub_face(
        &mut topo,
        0.0,
        5.0,
        0.0,
        10.0,
        Rank::A,
        Point3::new(2.5, 5.0, 0.0),
    );
    let right = rect_sub_face(
        &mut topo,
        5.0,
        10.0,
        0.0,
        10.0,
        Rank::A,
        Point3::new(7.5, 5.0, 0.0),
    );
    let sub_faces = vec![left, right];

    let result = detect_same_domain(&topo, &arena, &sub_faces, &face_ranks, tol);

    assert!(result.pairs.is_empty(), "no cross-rank pair expected");
    assert!(
        result.within_rank_dups.is_empty(),
        "adjacent non-overlapping faces should not be unioned, got {} dup(s)",
        result.within_rank_dups.len()
    );
}

/// Cross-rank geometric containment should set `geometric_overlap=true`,
/// marking the pair as a different-extent overlap so the BOP selector orders
/// it by area. Regression for the P1 review comment on the original PR.
#[test]
fn cross_rank_geometric_containment_marks_overlapping() {
    let mut topo = Topology::new();
    let arena = GfaArena::new();
    let face_ranks: HashMap<FaceId, Rank> = HashMap::new();
    let tol = Tolerance::new();

    // Rank A: large face. Rank B: small face fully inside A's outline,
    // with a different boundary (the edge-set pass misses it).
    let large_a = rect_sub_face(
        &mut topo,
        0.0,
        10.0,
        0.0,
        10.0,
        Rank::A,
        Point3::new(5.0, 5.0, 0.0),
    );
    let small_b = rect_sub_face(
        &mut topo,
        3.0,
        5.0,
        3.0,
        5.0,
        Rank::B,
        Point3::new(4.0, 4.0, 0.0),
    );
    let sub_faces = vec![large_a, small_b];

    let result = detect_same_domain(&topo, &arena, &sub_faces, &face_ranks, tol);

    assert!(
        result.within_rank_dups.is_empty(),
        "cross-rank pair should not be reported as within-rank dup"
    );
    assert_eq!(result.pairs.len(), 1, "expected one cross-rank SD pair");
    assert!(
        result.pairs[0].geometric_overlap,
        "geometric containment must set geometric_overlap=true (different-extent overlap pair)"
    );
}

/// A reversed face's effective normal is opposite its surface normal, so
/// pairing a reversed and an unreversed coplanar face with identical
/// boundaries must report `same_orientation = false`.
#[test]
fn reversed_face_flips_same_orientation() {
    let mut topo = Topology::new();
    let arena = GfaArena::new();
    let face_ranks: HashMap<FaceId, Rank> = HashMap::new();
    let tol = Tolerance::new();

    let face_a = rect_sub_face(
        &mut topo,
        0.0,
        1.0,
        0.0,
        1.0,
        Rank::A,
        Point3::new(0.5, 0.5, 0.0),
    );
    let face_b = rect_sub_face(
        &mut topo,
        0.0,
        1.0,
        0.0,
        1.0,
        Rank::B,
        Point3::new(0.5, 0.5, 0.0),
    );
    topo.face_mut(face_b.face_id).unwrap().set_reversed(true);
    let sub_faces = vec![face_a, face_b];

    let result = detect_same_domain(&topo, &arena, &sub_faces, &face_ranks, tol);

    assert_eq!(result.pairs.len(), 1, "expected one cross-rank SD pair");
    assert!(
        !result.pairs[0].same_orientation,
        "reversed face must flip effective orientation"
    );
}

/// Cross-rank coplanar faces that PARTIALLY overlap — neither fully
/// contained in the other — must still be paired by the geometric pass.
/// Two boxes stacked with a lateral offset share a partially-overlapping
/// coincident planar contact face (a sub-rectangle); the contained-only
/// test misses this, leaving the coincident pieces un-cancelled and the
/// fused result non-manifold. Closes the documented same-domain "detects
/// containment but not overlap" gap.
///
/// Discriminating: without the partial-overlap branch in
/// `planar_faces_overlap`, `pairs` is empty (neither face is contained in
/// the other, so the two containment checks both fail); with it, the
/// intersection-area test pairs them.
#[test]
fn cross_rank_partial_overlap_marks_overlapping() {
    let mut topo = Topology::new();
    let arena = GfaArena::new();
    let face_ranks: HashMap<FaceId, Rank> = HashMap::new();
    let tol = Tolerance::new();

    // A: [0,10]x[0,10] (area 100). B: [3,13]x[0,10] (area 100), shifted +3x.
    // Overlap [3,10]x[0,10] = 70 > 50% of the smaller face. B's x=13 lies
    // outside A and A's x=0 lies outside B, so neither is contained.
    let face_a = rect_sub_face(
        &mut topo,
        0.0,
        10.0,
        0.0,
        10.0,
        Rank::A,
        Point3::new(5.0, 5.0, 0.0),
    );
    let face_b = rect_sub_face(
        &mut topo,
        3.0,
        13.0,
        0.0,
        10.0,
        Rank::B,
        Point3::new(8.0, 5.0, 0.0),
    );
    let sub_faces = vec![face_a, face_b];

    let result = detect_same_domain(&topo, &arena, &sub_faces, &face_ranks, tol);

    assert!(
        result.within_rank_dups.is_empty(),
        "cross-rank pair should not be reported as within-rank dup"
    );
    assert_eq!(
        result.pairs.len(),
        1,
        "partially-overlapping coplanar cross-rank faces must be paired"
    );
    assert!(
        result.pairs[0].geometric_overlap,
        "geometric overlap must set geometric_overlap=true (different-extent overlap pair)"
    );
}

/// The partial-overlap branch is gated at 50% of the smaller face: two
/// coplanar faces sharing only a thin sliver of area must NOT be paired,
/// so a numerical overlap along a shared edge does not annihilate disjoint
/// faces.
#[test]
fn cross_rank_small_overlap_not_paired() {
    let mut topo = Topology::new();
    let arena = GfaArena::new();
    let face_ranks: HashMap<FaceId, Rank> = HashMap::new();
    let tol = Tolerance::new();

    // A: [0,10]x[0,10] (area 100). B: [9,19]x[0,10] (area 100), shifted +9x.
    // Overlap [9,10]x[0,10] = 10 < 50% of the smaller face.
    let face_a = rect_sub_face(
        &mut topo,
        0.0,
        10.0,
        0.0,
        10.0,
        Rank::A,
        Point3::new(5.0, 5.0, 0.0),
    );
    let face_b = rect_sub_face(
        &mut topo,
        9.0,
        19.0,
        0.0,
        10.0,
        Rank::B,
        Point3::new(14.0, 5.0, 0.0),
    );
    let sub_faces = vec![face_a, face_b];

    let result = detect_same_domain(&topo, &arena, &sub_faces, &face_ranks, tol);

    assert!(
        result.pairs.is_empty(),
        "sub-threshold overlap must not pair coplanar faces, got {} pair(s)",
        result.pairs.len()
    );
    assert!(
        result.within_rank_dups.is_empty(),
        "sub-threshold overlap must not produce within-rank dups"
    );
}

#[test]
fn planes_same_domain_same_direction() {
    let tol = Tolerance::new();
    let a = FaceSurface::Plane {
        normal: Vec3::new(0.0, 0.0, 1.0),
        d: 5.0,
    };
    let b = FaceSurface::Plane {
        normal: Vec3::new(0.0, 0.0, 1.0),
        d: 5.0,
    };
    assert_eq!(surfaces_same_domain(&a, &b, tol), Some(true));
}

#[test]
fn planes_same_domain_opposite_direction() {
    let tol = Tolerance::new();
    let a = FaceSurface::Plane {
        normal: Vec3::new(0.0, 0.0, 1.0),
        d: 5.0,
    };
    let b = FaceSurface::Plane {
        normal: Vec3::new(0.0, 0.0, -1.0),
        d: -5.0,
    };
    assert_eq!(surfaces_same_domain(&a, &b, tol), Some(false));
}

#[test]
fn planes_different_distance_not_same_domain() {
    let tol = Tolerance::new();
    let a = FaceSurface::Plane {
        normal: Vec3::new(0.0, 0.0, 1.0),
        d: 5.0,
    };
    let b = FaceSurface::Plane {
        normal: Vec3::new(0.0, 0.0, 1.0),
        d: 10.0,
    };
    assert_eq!(surfaces_same_domain(&a, &b, tol), None);
}

#[test]
fn mixed_surface_types_not_same_domain() {
    let tol = Tolerance::new();
    let a = FaceSurface::Plane {
        normal: Vec3::new(0.0, 0.0, 1.0),
        d: 0.0,
    };
    let b = FaceSurface::Sphere(
        brepkit_math::surfaces::SphericalSurface::new(Point3::new(0.0, 0.0, 0.0), 1.0)
            .expect("valid sphere"),
    );
    assert_eq!(surfaces_same_domain(&a, &b, tol), None);
}

#[test]
fn cones_same_domain_same_direction() {
    let tol = Tolerance::new();
    let a = FaceSurface::Cone(
        brepkit_math::surfaces::ConicalSurface::with_ref_dir(
            Point3::new(0.0, 0.0, 0.0),
            Vec3::new(0.0, 0.0, 1.0),
            std::f64::consts::FRAC_PI_6,
            Vec3::new(1.0, 0.0, 0.0),
        )
        .expect("valid cone"),
    );
    let b = FaceSurface::Cone(
        brepkit_math::surfaces::ConicalSurface::with_ref_dir(
            Point3::new(0.0, 0.0, 0.0),
            Vec3::new(0.0, 0.0, 1.0),
            std::f64::consts::FRAC_PI_6,
            Vec3::new(0.0, 1.0, 0.0),
        )
        .expect("valid cone"),
    );
    assert_eq!(surfaces_same_domain(&a, &b, tol), Some(true));
}

#[test]
fn cones_different_half_angle_not_same_domain() {
    let tol = Tolerance::new();
    let a = FaceSurface::Cone(
        brepkit_math::surfaces::ConicalSurface::with_ref_dir(
            Point3::new(0.0, 0.0, 0.0),
            Vec3::new(0.0, 0.0, 1.0),
            std::f64::consts::FRAC_PI_6,
            Vec3::new(1.0, 0.0, 0.0),
        )
        .expect("valid cone"),
    );
    let b = FaceSurface::Cone(
        brepkit_math::surfaces::ConicalSurface::with_ref_dir(
            Point3::new(0.0, 0.0, 0.0),
            Vec3::new(0.0, 0.0, 1.0),
            std::f64::consts::FRAC_PI_4,
            Vec3::new(1.0, 0.0, 0.0),
        )
        .expect("valid cone"),
    );
    assert_eq!(surfaces_same_domain(&a, &b, tol), None);
}

#[test]
fn torus_same_domain_same_direction_ignores_ref_dir() {
    let tol = Tolerance::new();
    let a = FaceSurface::Torus(
        brepkit_math::surfaces::ToroidalSurface::with_axis(
            Point3::new(0.0, 0.0, 0.0),
            3.0,
            1.0,
            Vec3::new(0.0, 0.0, 1.0),
        )
        .expect("valid torus"),
    );
    // Same surface, but constructed with a different ref direction —
    // x_axis/y_axis differ but z_axis matches, so this is the same surface.
    let b = FaceSurface::Torus(
        brepkit_math::surfaces::ToroidalSurface::with_axis_and_ref_dir(
            Point3::new(0.0, 0.0, 0.0),
            3.0,
            1.0,
            Vec3::new(0.0, 0.0, 1.0),
            Vec3::new(0.0, 1.0, 0.0),
        )
        .expect("valid torus"),
    );
    assert_eq!(surfaces_same_domain(&a, &b, tol), Some(true));
}

#[test]
fn torus_same_domain_opposite_direction() {
    let tol = Tolerance::new();
    let a = FaceSurface::Torus(
        brepkit_math::surfaces::ToroidalSurface::with_axis(
            Point3::new(1.0, 2.0, 3.0),
            5.0,
            1.0,
            Vec3::new(0.0, 0.0, 1.0),
        )
        .expect("valid torus"),
    );
    let b = FaceSurface::Torus(
        brepkit_math::surfaces::ToroidalSurface::with_axis(
            Point3::new(1.0, 2.0, 3.0),
            5.0,
            1.0,
            Vec3::new(0.0, 0.0, -1.0),
        )
        .expect("valid torus"),
    );
    assert_eq!(surfaces_same_domain(&a, &b, tol), Some(false));
}

#[test]
fn torus_different_major_radius_not_same_domain() {
    let tol = Tolerance::new();
    let a = FaceSurface::Torus(
        brepkit_math::surfaces::ToroidalSurface::with_axis(
            Point3::new(0.0, 0.0, 0.0),
            3.0,
            1.0,
            Vec3::new(0.0, 0.0, 1.0),
        )
        .expect("valid torus"),
    );
    let b = FaceSurface::Torus(
        brepkit_math::surfaces::ToroidalSurface::with_axis(
            Point3::new(0.0, 0.0, 0.0),
            4.0,
            1.0,
            Vec3::new(0.0, 0.0, 1.0),
        )
        .expect("valid torus"),
    );
    assert_eq!(surfaces_same_domain(&a, &b, tol), None);
}

#[test]
fn torus_different_minor_radius_not_same_domain() {
    let tol = Tolerance::new();
    let a = FaceSurface::Torus(
        brepkit_math::surfaces::ToroidalSurface::with_axis(
            Point3::new(0.0, 0.0, 0.0),
            3.0,
            1.0,
            Vec3::new(0.0, 0.0, 1.0),
        )
        .expect("valid torus"),
    );
    let b = FaceSurface::Torus(
        brepkit_math::surfaces::ToroidalSurface::with_axis(
            Point3::new(0.0, 0.0, 0.0),
            3.0,
            0.5,
            Vec3::new(0.0, 0.0, 1.0),
        )
        .expect("valid torus"),
    );
    assert_eq!(surfaces_same_domain(&a, &b, tol), None);
}

#[test]
fn torus_different_center_not_same_domain() {
    let tol = Tolerance::new();
    let a = FaceSurface::Torus(
        brepkit_math::surfaces::ToroidalSurface::with_axis(
            Point3::new(0.0, 0.0, 0.0),
            3.0,
            1.0,
            Vec3::new(0.0, 0.0, 1.0),
        )
        .expect("valid torus"),
    );
    let b = FaceSurface::Torus(
        brepkit_math::surfaces::ToroidalSurface::with_axis(
            Point3::new(1.0, 0.0, 0.0),
            3.0,
            1.0,
            Vec3::new(0.0, 0.0, 1.0),
        )
        .expect("valid torus"),
    );
    assert_eq!(surfaces_same_domain(&a, &b, tol), None);
}

#[test]
fn torus_skew_axes_not_same_domain() {
    let tol = Tolerance::new();
    let a = FaceSurface::Torus(
        brepkit_math::surfaces::ToroidalSurface::with_axis(
            Point3::new(0.0, 0.0, 0.0),
            3.0,
            1.0,
            Vec3::new(0.0, 0.0, 1.0),
        )
        .expect("valid torus"),
    );
    let b = FaceSurface::Torus(
        brepkit_math::surfaces::ToroidalSurface::with_axis(
            Point3::new(0.0, 0.0, 0.0),
            3.0,
            1.0,
            Vec3::new(1.0, 0.0, 0.0),
        )
        .expect("valid torus"),
    );
    assert_eq!(surfaces_same_domain(&a, &b, tol), None);
}

#[test]
fn quantize_point_deterministic() {
    let scale = 1.0 / 1e-7; // default tolerance
    let p = Point3::new(1.0, 2.0, 3.0);
    let q1 = quantize_point(p, scale);
    let q2 = quantize_point(p, scale);
    assert_eq!(q1, q2, "quantization should be deterministic");
}

#[test]
fn quantize_nearby_points_collapse() {
    let tol = Tolerance::new();
    let scale = 1.0 / tol.linear;
    let p1 = Point3::new(1.0, 2.0, 3.0);
    let p2 = Point3::new(1.0 + tol.linear * 0.4, 2.0, 3.0);
    let q1 = quantize_point(p1, scale);
    let q2 = quantize_point(p2, scale);
    assert_eq!(q1, q2, "nearby points should collapse to same grid cell");
}

#[test]
fn quantize_distant_points_differ() {
    let tol = Tolerance::new();
    let scale = 1.0 / tol.linear;
    let p1 = Point3::new(1.0, 2.0, 3.0);
    let p2 = Point3::new(1.0 + tol.linear * 2.0, 2.0, 3.0);
    let q1 = quantize_point(p1, scale);
    let q2 = quantize_point(p2, scale);
    assert_ne!(
        q1, q2,
        "points separated by 2x tolerance should be in different cells"
    );
}

#[test]
fn union_find_basic_groups() {
    let mut uf = UnionFind::new(5);
    uf.union(0, 1);
    uf.union(2, 3);
    assert_eq!(uf.find(0), uf.find(1));
    assert_eq!(uf.find(2), uf.find(3));
    assert_ne!(uf.find(0), uf.find(2));
    // Transitive closure
    uf.union(1, 3);
    assert_eq!(uf.find(0), uf.find(3));
}

/// Build an angular patch of an axis-aligned (Z-axis) cylinder centred at
/// `(cx, cy)` with radius `r`, spanning angle `[u0, u1]` and height `[z0, z1]`.
/// The wire is two line seams (at u0 and u1) and two circular arcs (at z0, z1).
#[allow(clippy::too_many_arguments)]
fn cylinder_patch(
    topo: &mut Topology,
    cx: f64,
    cy: f64,
    r: f64,
    u0: f64,
    u1: f64,
    z0: f64,
    z1: f64,
    rank: Rank,
) -> SubFace {
    use brepkit_math::curves::Circle3D;
    use brepkit_math::surfaces::CylindricalSurface;
    use brepkit_topology::edge::{Edge, EdgeCurve};
    use brepkit_topology::face::{Face, FaceSurface};
    use brepkit_topology::vertex::Vertex;
    use brepkit_topology::wire::{OrientedEdge, Wire};

    let center = Point3::new(cx, cy, 0.0);
    let axis = Vec3::new(0.0, 0.0, 1.0);
    let pt = |u: f64, z: f64| Point3::new(cx + r * u.cos(), cy + r * u.sin(), z);

    let v00 = topo.add_vertex(Vertex::new(pt(u0, z0), 1e-7));
    let v10 = topo.add_vertex(Vertex::new(pt(u1, z0), 1e-7));
    let v11 = topo.add_vertex(Vertex::new(pt(u1, z1), 1e-7));
    let v01 = topo.add_vertex(Vertex::new(pt(u0, z1), 1e-7));

    let circ_bot = Circle3D::new(Point3::new(cx, cy, z0), axis, r).unwrap();
    let circ_top = Circle3D::new(Point3::new(cx, cy, z1), axis, r).unwrap();

    // bottom arc v00 -> v10, right seam v10 -> v11, top arc v11 -> v01, left seam v01 -> v00
    let e_bot = topo.add_edge(Edge::new(v00, v10, EdgeCurve::Circle(circ_bot)));
    let e_right = topo.add_edge(Edge::new(v10, v11, EdgeCurve::Line));
    let e_top = topo.add_edge(Edge::new(v11, v01, EdgeCurve::Circle(circ_top)));
    let e_left = topo.add_edge(Edge::new(v01, v00, EdgeCurve::Line));

    let wire = Wire::new(
        vec![
            OrientedEdge::new(e_bot, true),
            OrientedEdge::new(e_right, true),
            OrientedEdge::new(e_top, true),
            OrientedEdge::new(e_left, true),
        ],
        true,
    )
    .unwrap();
    let wid = topo.add_wire(wire);
    let surf = CylindricalSurface::new(center, axis, r).unwrap();
    let face_id = topo.add_face(Face::new(wid, vec![], FaceSurface::Cylinder(surf)));

    // Interior sample at the patch's parametric centre.
    let um = 0.5 * (u0 + u1);
    let zm = 0.5 * (z0 + z1);
    SubFace {
        face_id,
        classification: FaceClass::Unknown,
        rank,
        interior_point: Some(pt(um, zm)),
    }
}

/// Build an angular patch of an axis-aligned (Z-axis) cone with `apex` at
/// `(0, 0, apex_z)`, given `half_angle`, spanning `[u0, u1]` over heights
/// `[z0, z1]` ABOVE the apex. The bottom/top rims are circles at the cone's
/// radius for each height.
#[allow(clippy::too_many_arguments)]
fn cone_patch(
    topo: &mut Topology,
    apex_z: f64,
    half_angle: f64,
    u0: f64,
    u1: f64,
    z0: f64,
    z1: f64,
    rank: Rank,
) -> SubFace {
    use brepkit_math::curves::Circle3D;
    use brepkit_math::surfaces::ConicalSurface;
    use brepkit_topology::edge::{Edge, EdgeCurve};
    use brepkit_topology::face::{Face, FaceSurface};
    use brepkit_topology::vertex::Vertex;
    use brepkit_topology::wire::{OrientedEdge, Wire};

    let apex = Point3::new(0.0, 0.0, apex_z);
    let axis = Vec3::new(0.0, 0.0, 1.0);
    // Radius at height z above the apex: z * tan(half_angle) is wrong for this
    // parameterization; here radius = (axial distance) * cos(a)/sin(a) where the
    // axial distance from apex is (z - apex_z). cos/sin = cot(a).
    let cot = half_angle.cos() / half_angle.sin();
    let radius_at_z = |z: f64| (z - apex_z) * cot;
    let pt = |u: f64, z: f64| {
        let r = radius_at_z(z);
        Point3::new(r * u.cos(), r * u.sin(), z)
    };

    let v00 = topo.add_vertex(Vertex::new(pt(u0, z0), 1e-7));
    let v10 = topo.add_vertex(Vertex::new(pt(u1, z0), 1e-7));
    let v11 = topo.add_vertex(Vertex::new(pt(u1, z1), 1e-7));
    let v01 = topo.add_vertex(Vertex::new(pt(u0, z1), 1e-7));

    let circ_bot = Circle3D::new(Point3::new(0.0, 0.0, z0), axis, radius_at_z(z0)).unwrap();
    let circ_top = Circle3D::new(Point3::new(0.0, 0.0, z1), axis, radius_at_z(z1)).unwrap();

    let e_bot = topo.add_edge(Edge::new(v00, v10, EdgeCurve::Circle(circ_bot)));
    let e_right = topo.add_edge(Edge::new(v10, v11, EdgeCurve::Line));
    let e_top = topo.add_edge(Edge::new(v11, v01, EdgeCurve::Circle(circ_top)));
    let e_left = topo.add_edge(Edge::new(v01, v00, EdgeCurve::Line));

    let wire = Wire::new(
        vec![
            OrientedEdge::new(e_bot, true),
            OrientedEdge::new(e_right, true),
            OrientedEdge::new(e_top, true),
            OrientedEdge::new(e_left, true),
        ],
        true,
    )
    .unwrap();
    let wid = topo.add_wire(wire);
    let surf = ConicalSurface::new(apex, axis, half_angle).unwrap();
    let face_id = topo.add_face(Face::new(wid, vec![], FaceSurface::Cone(surf)));

    let um = 0.5 * (u0 + u1);
    let zm = 0.5 * (z0 + z1);
    SubFace {
        face_id,
        classification: FaceClass::Unknown,
        rank,
        interior_point: Some(pt(um, zm)),
    }
}

/// Two coaxial cone patches with the same apex and half-angle, an EIGHTH
/// contained in a QUARTER over the same height band, pair as a cross-rank SD
/// overlap (the cone analogue of the cylinder corner case) and keep the larger
/// quarter as the representative.
#[test]
fn coaxial_cone_eighth_in_quarter_pairs() {
    use std::f64::consts::{FRAC_PI_2, FRAC_PI_4};
    let mut topo = Topology::new();
    let arena = GfaArena::new();
    let face_ranks: HashMap<FaceId, Rank> = HashMap::new();
    let tol = Tolerance::new();

    let half_angle = std::f64::consts::FRAC_PI_6; // 30°
    // Eighth and quarter on the same cone, same height band z in [2, 4].
    let eighth = cone_patch(
        &mut topo,
        0.0,
        half_angle,
        0.0,
        FRAC_PI_4,
        2.0,
        4.0,
        Rank::A,
    );
    let quarter = cone_patch(
        &mut topo,
        0.0,
        half_angle,
        0.0,
        FRAC_PI_2,
        2.0,
        4.0,
        Rank::B,
    );
    let sub_faces = vec![eighth, quarter];

    let result = detect_same_domain(&topo, &arena, &sub_faces, &face_ranks, tol);
    assert_eq!(
        result.pairs.len(),
        1,
        "coaxial cone eighth-in-quarter must form one cross-rank SD pair"
    );
    assert!(result.pairs[0].geometric_overlap);
    assert_eq!(
        result.pairs[0].representative, 1,
        "the larger quarter cone patch must be the representative"
    );
}

/// Two coaxial cone patches on disjoint height bands must NOT pair (the cone
/// analogue of the stacked-cylinder guard).
#[test]
fn coaxial_cone_disjoint_z_bands_do_not_pair() {
    use std::f64::consts::FRAC_PI_2;
    let mut topo = Topology::new();
    let arena = GfaArena::new();
    let face_ranks: HashMap<FaceId, Rank> = HashMap::new();
    let tol = Tolerance::new();

    let half_angle = std::f64::consts::FRAC_PI_6;
    let lower = cone_patch(
        &mut topo,
        0.0,
        half_angle,
        0.0,
        FRAC_PI_2,
        2.0,
        4.0,
        Rank::A,
    );
    let upper = cone_patch(
        &mut topo,
        0.0,
        half_angle,
        0.0,
        FRAC_PI_2,
        4.0,
        6.0,
        Rank::B,
    );
    let sub_faces = vec![lower, upper];

    let result = detect_same_domain(&topo, &arena, &sub_faces, &face_ranks, tol);
    assert!(
        result.pairs.is_empty() && result.within_rank_dups.is_empty(),
        "stacked cone patches on disjoint bands must not pair (got {} pairs)",
        result.pairs.len()
    );
}

/// Two coaxial same-radius cylinder patches that overlap in (θ, z) — a body's
/// angular EIGHTH contained in a lip's QUARTER over the same height band — are
/// paired as a cross-rank same-domain overlap, with the larger (quarter) kept
/// as the representative. This is the 3×3 stacking-lip corner case: the GFA
/// builder splits both operands at the shared rim (z = 16 here) so the body's
/// upper eighth and the lip's lower quarter share the SAME z-band, making the
/// eighth's (θ, z) rectangle a strict subset of the quarter's.
#[test]
fn coaxial_cylinder_eighth_in_quarter_pairs() {
    use std::f64::consts::FRAC_PI_4;
    let mut topo = Topology::new();
    let arena = GfaArena::new();
    let face_ranks: HashMap<FaceId, Rank> = HashMap::new();
    let tol = Tolerance::new();

    // Body upper eighth: u in [0, π/4], z in [13.3, 16].
    let eighth = cylinder_patch(
        &mut topo,
        0.0,
        0.0,
        3.75,
        0.0,
        FRAC_PI_4,
        13.3,
        16.0,
        Rank::A,
    );
    // Lip lower quarter: u in [0, π/2], z in [13.3, 16] (same band, wider arc).
    let quarter = cylinder_patch(
        &mut topo,
        0.0,
        0.0,
        3.75,
        0.0,
        2.0 * FRAC_PI_4,
        13.3,
        16.0,
        Rank::B,
    );
    let sub_faces = vec![eighth, quarter];

    let result = detect_same_domain(&topo, &arena, &sub_faces, &face_ranks, tol);
    assert_eq!(
        result.pairs.len(),
        1,
        "coaxial eighth-in-quarter must form one cross-rank SD pair"
    );
    let p = &result.pairs[0];
    assert!(
        p.geometric_overlap,
        "an overlap (not edge-set) pair must be flagged as a containment/overlap"
    );
    // The quarter (idx 1, rank B) is larger and must be the representative.
    assert_eq!(
        p.representative, 1,
        "the larger quarter patch must be the kept representative"
    );
}

/// A genuine PARTIAL overlap where the patches are offset in BOTH θ and z, so
/// their shared region is a small fraction of either, must NOT pair — the same
/// area-fraction guard the planar path uses against pairing faces that merely
/// abut along a sliver.
#[test]
fn coaxial_cylinder_thin_partial_overlap_does_not_pair() {
    use std::f64::consts::{FRAC_PI_2, FRAC_PI_4, FRAC_PI_8};
    let mut topo = Topology::new();
    let arena = GfaArena::new();
    let face_ranks: HashMap<FaceId, Rank> = HashMap::new();
    let tol = Tolerance::new();

    // Patch A: u in [0, π/4], z in [13.3, 16].
    let a = cylinder_patch(
        &mut topo,
        0.0,
        0.0,
        3.75,
        0.0,
        FRAC_PI_4,
        13.3,
        16.0,
        Rank::A,
    );
    // Patch B: u in [π/8, π/8+π/2], z in [14.85, 17.55] — offset in both axes,
    // so the shared region u∈[π/8,π/4] × z∈[14.85,16] is ~20% of A.
    let b = cylinder_patch(
        &mut topo,
        0.0,
        0.0,
        3.75,
        FRAC_PI_8,
        FRAC_PI_8 + FRAC_PI_2,
        14.85,
        17.55,
        Rank::B,
    );
    let sub_faces = vec![a, b];

    let result = detect_same_domain(&topo, &arena, &sub_faces, &face_ranks, tol);
    assert!(
        result.pairs.is_empty(),
        "a thin partial overlap (< 50% of the smaller patch) must not pair (got {} pairs)",
        result.pairs.len()
    );
}

/// Two coaxial same-radius cylinder patches on DISJOINT height bands (stacked,
/// touching only at one ring) must NOT be paired — they share a parametric
/// projection only at the touching z, not a 2D area. Guards the axial-reference
/// mis-pairing the (θ, axial) projection must avoid.
#[test]
fn coaxial_cylinder_disjoint_z_bands_do_not_pair() {
    use std::f64::consts::FRAC_PI_2;
    let mut topo = Topology::new();
    let arena = GfaArena::new();
    let face_ranks: HashMap<FaceId, Rank> = HashMap::new();
    let tol = Tolerance::new();

    // Lower patch z in [0, 10] and upper patch z in [10, 20], same angular span.
    let lower = cylinder_patch(
        &mut topo,
        0.0,
        0.0,
        3.75,
        0.0,
        FRAC_PI_2,
        0.0,
        10.0,
        Rank::A,
    );
    let upper = cylinder_patch(
        &mut topo,
        0.0,
        0.0,
        3.75,
        0.0,
        FRAC_PI_2,
        10.0,
        20.0,
        Rank::B,
    );
    let sub_faces = vec![lower, upper];

    let result = detect_same_domain(&topo, &arena, &sub_faces, &face_ranks, tol);
    assert!(
        result.pairs.is_empty() && result.within_rank_dups.is_empty(),
        "stacked patches on disjoint z-bands must not be same-domain (got {} pairs)",
        result.pairs.len()
    );
}

/// Two coaxial same-radius cylinder patches on OPPOSITE angular sides over the
/// same height band must NOT be paired — disjoint in θ, so no genuine 3D
/// overlap. Guards against the seam-wraparound shift aligning opposite sides.
#[test]
fn coaxial_cylinder_opposite_sides_do_not_pair() {
    use std::f64::consts::{FRAC_PI_4, PI};
    let mut topo = Topology::new();
    let arena = GfaArena::new();
    let face_ranks: HashMap<FaceId, Rank> = HashMap::new();
    let tol = Tolerance::new();

    // +X side: u in [0, π/4]; opposite (−X) side: u in [π, π + π/4].
    let near = cylinder_patch(
        &mut topo,
        0.0,
        0.0,
        3.75,
        0.0,
        FRAC_PI_4,
        0.0,
        10.0,
        Rank::A,
    );
    let far = cylinder_patch(
        &mut topo,
        0.0,
        0.0,
        3.75,
        PI,
        PI + FRAC_PI_4,
        0.0,
        10.0,
        Rank::B,
    );
    let sub_faces = vec![near, far];

    let result = detect_same_domain(&topo, &arena, &sub_faces, &face_ranks, tol);
    assert!(
        result.pairs.is_empty(),
        "opposite-side patches must not be same-domain (got {} pairs)",
        result.pairs.len()
    );
}
