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

/// Cross-rank geometric containment should set `b_contained_in_a=true`
/// so `apply_sd_selection` cancels the pair under Cut. Regression for
/// the P1 review comment on the original PR.
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
        result.pairs[0].b_contained_in_a,
        "geometric containment must set b_contained_in_a=true so Cut cancels both"
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
