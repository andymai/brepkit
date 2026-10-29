#![allow(clippy::unwrap_used, clippy::expect_used)]

use crate::nurbs::surface::NurbsSurface;
use crate::vec::{Point3, Vec3};

use super::surface_marching::march_intersection;
use super::surface_marching::{near_existing_segment, second_order_tangent};
use super::surface_seeding::{
    REFINE_CALLS, find_ssi_seeds_grid, find_ssi_seeds_subdivision, refine_ssi_point,
};
use super::*;

/// Create a simple bilinear NURBS surface (flat plane at z=0, from (0,0) to (1,1)).
fn flat_surface() -> NurbsSurface {
    NurbsSurface::new(
        1,
        1,
        vec![0.0, 0.0, 1.0, 1.0],
        vec![0.0, 0.0, 1.0, 1.0],
        vec![
            vec![Point3::new(0.0, 0.0, 0.0), Point3::new(0.0, 1.0, 0.0)],
            vec![Point3::new(1.0, 0.0, 0.0), Point3::new(1.0, 1.0, 0.0)],
        ],
        vec![vec![1.0, 1.0], vec![1.0, 1.0]],
    )
    .unwrap()
}

/// Create a curved surface (saddle shape).
fn saddle_surface() -> NurbsSurface {
    NurbsSurface::new(
        2,
        2,
        vec![0.0, 0.0, 0.0, 1.0, 1.0, 1.0],
        vec![0.0, 0.0, 0.0, 1.0, 1.0, 1.0],
        vec![
            vec![
                Point3::new(0.0, 0.0, 0.0),
                Point3::new(0.0, 0.5, 0.25),
                Point3::new(0.0, 1.0, 0.0),
            ],
            vec![
                Point3::new(0.5, 0.0, -0.25),
                Point3::new(0.5, 0.5, 0.0),
                Point3::new(0.5, 1.0, 0.25),
            ],
            vec![
                Point3::new(1.0, 0.0, 0.0),
                Point3::new(1.0, 0.5, -0.25),
                Point3::new(1.0, 1.0, 0.0),
            ],
        ],
        vec![vec![1.0; 3]; 3],
    )
    .unwrap()
}

// -- Plane-NURBS intersection --

#[test]
fn flat_surface_plane_no_intersection() {
    let surface = flat_surface();
    // Plane at z=1 shouldn't intersect surface at z=0.
    let result = intersect_plane_nurbs(&surface, Vec3::new(0.0, 0.0, 1.0), 1.0, 30).unwrap();

    assert!(result.is_empty(), "no intersection expected");
}

#[test]
fn saddle_surface_plane_intersection() {
    let surface = saddle_surface();
    // Plane at z=0 should intersect the saddle surface.
    let result = intersect_plane_nurbs(&surface, Vec3::new(0.0, 0.0, 1.0), 0.0, 50).unwrap();

    assert!(
        !result.is_empty(),
        "saddle surface should intersect z=0 plane"
    );

    // The intersection curve should have points near z=0.
    for curve in &result {
        for pt in &curve.points {
            assert!(
                pt.point.z().abs() < 1e-4,
                "intersection point should be near z=0, got z={}",
                pt.point.z()
            );
        }
    }
}

// -- Line-NURBS intersection --

#[test]
fn line_flat_surface_intersection() {
    let surface = flat_surface();
    // Vertical ray through (0.5, 0.5) should hit the surface at z=0.
    let result = intersect_line_nurbs(
        &surface,
        Point3::new(0.5, 0.5, 1.0),
        Vec3::new(0.0, 0.0, -1.0),
        20,
    )
    .unwrap();

    assert!(!result.is_empty(), "ray should hit flat surface");

    let pt = &result[0];
    assert!(
        (pt.point.x() - 0.5).abs() < 1e-4,
        "x should be ~0.5, got {}",
        pt.point.x()
    );
    assert!(
        (pt.point.y() - 0.5).abs() < 1e-4,
        "y should be ~0.5, got {}",
        pt.point.y()
    );
    assert!(
        pt.point.z().abs() < 1e-4,
        "z should be ~0.0, got {}",
        pt.point.z()
    );
}

#[test]
fn an_oblique_line_crosses_a_surface_once_on_its_exact_point() {
    let surface = flat_surface();
    // About 12 degrees off the surface: every grid seed near the ray refines
    // to the one crossing at t = 0.4.
    let dir = Vec3::new(1.0, 0.7, -0.25);
    let result = intersect_line_nurbs(&surface, Point3::new(0.2, 0.3, 0.1), dir, 20).unwrap();
    assert_eq!(result.len(), 1, "{result:?}");
    let exact = Point3::new(0.6, 0.58, 0.0);
    assert!(
        (result[0].point - exact).length() < 1e-9,
        "{:?}",
        result[0].point
    );
}

#[test]
fn line_misses_surface() {
    let surface = flat_surface();
    // Ray parallel to the surface should miss.
    let result = intersect_line_nurbs(
        &surface,
        Point3::new(0.5, 0.5, 1.0),
        Vec3::new(1.0, 0.0, 0.0),
        20,
    )
    .unwrap();

    assert!(result.is_empty(), "parallel ray should miss");
}

// -- Intersection point quality --

#[test]
fn refined_points_are_on_plane() {
    let surface = saddle_surface();
    let normal = Vec3::new(0.0, 0.0, 1.0);
    let d = 0.1; // Slightly above z=0.
    let result = intersect_plane_nurbs(&surface, normal, d, 50).unwrap();

    for curve in &result {
        for pt in &curve.points {
            let signed_dist = Vec3::new(pt.point.x(), pt.point.y(), pt.point.z()).dot(normal) - d;
            assert!(
                signed_dist.abs() < 1e-4,
                "point should be on plane, signed_dist={signed_dist}"
            );
        }
    }
}

// -- NURBS-NURBS intersection --

/// Create a flat surface at z=0.5 (overlapping region with `flat_surface` at z=0).
fn flat_surface_offset() -> NurbsSurface {
    NurbsSurface::new(
        1,
        1,
        vec![0.0, 0.0, 1.0, 1.0],
        vec![0.0, 0.0, 1.0, 1.0],
        vec![
            vec![Point3::new(0.0, 0.0, 0.5), Point3::new(0.0, 1.0, 0.5)],
            vec![Point3::new(1.0, 0.0, 0.5), Point3::new(1.0, 1.0, 0.5)],
        ],
        vec![vec![1.0, 1.0], vec![1.0, 1.0]],
    )
    .unwrap()
}

/// Create a tilted flat surface that intersects the flat z=0 surface.
fn tilted_surface() -> NurbsSurface {
    // Surface tilted in the XZ plane: goes from z=-0.5 at x=0 to z=0.5 at x=1.
    NurbsSurface::new(
        1,
        1,
        vec![0.0, 0.0, 1.0, 1.0],
        vec![0.0, 0.0, 1.0, 1.0],
        vec![
            vec![Point3::new(0.0, 0.0, -0.5), Point3::new(0.0, 1.0, -0.5)],
            vec![Point3::new(1.0, 0.0, 0.5), Point3::new(1.0, 1.0, 0.5)],
        ],
        vec![vec![1.0, 1.0], vec![1.0, 1.0]],
    )
    .unwrap()
}

#[test]
fn parallel_surfaces_no_intersection() {
    let s1 = flat_surface();
    let s2 = flat_surface_offset();
    let result = intersect_nurbs_nurbs(&s1, &s2, 15, 0.02).unwrap();
    assert!(result.is_empty(), "parallel surfaces should not intersect");
}

#[test]
fn refine_ssi_basic() {
    let s1 = flat_surface();
    let s2 = tilted_surface();
    // At u1=0.5, v1=0.5 on flat -> (0.5, 0.5, 0)
    // At u2=0.5, v2=0.5 on tilted -> (0.5, 0.5, 0)
    // These should refine to an intersection point.
    let result = refine_ssi_point(&s1, &s2, 0.5, 0.5, 0.5, 0.5, 1e-6);
    assert!(
        result.is_some(),
        "refine should find intersection at (0.5, 0.5)"
    );
}

#[test]
fn seed_finding_basic() {
    let s1 = flat_surface();
    let s2 = tilted_surface();

    // Verify surfaces evaluate correctly.
    let p1 = s1.evaluate(0.5, 0.5);
    let p2 = s2.evaluate(0.5, 0.5);
    let dist = (p1 - p2).length();
    assert!(
        dist < 0.01,
        "flat(0.5,0.5)={p1:?} tilted(0.5,0.5)={p2:?} dist={dist}",
    );

    // Verify refine works from off-center guess.
    let refined = refine_ssi_point(&s1, &s2, 0.5263, 0.5, 0.5263, 0.5, 1e-6);
    assert!(
        refined.is_some(),
        "refine should converge from off-center guess"
    );

    let seeds = find_ssi_seeds_grid(&s1, &s2, 10, 1e-6);
    assert!(
        !seeds.is_empty(),
        "should find seeds between flat and tilted surfaces"
    );
}

#[test]
fn tilted_intersects_flat() {
    let s1 = flat_surface();
    let s2 = tilted_surface();

    // First verify seed finding works.
    let seeds = find_ssi_seeds_grid(&s1, &s2, 10, 1e-6);
    assert!(
        !seeds.is_empty(),
        "should find at least one seed point, got 0"
    );

    let result = intersect_nurbs_nurbs(&s1, &s2, 10, 0.05).unwrap();

    assert!(
        !result.is_empty(),
        "tilted surface should intersect flat surface (seeds: {})",
        seeds.len()
    );

    for curve in &result {
        for pt in &curve.points {
            assert!(
                pt.point.z().abs() < 0.15,
                "point should be near z=0, got z={}",
                pt.point.z()
            );
        }
    }
}

/// A plane crossing that runs between grid columns (one crossing per row,
/// each on an interior horizontal edge) must come back as ONE curve. The
/// scan used to visit every interior cell edge from both adjacent cells, so
/// every crossing refined to an identical duplicate; each point's nearest
/// neighbor was its own twin, the proximity chainer's average-spacing
/// statistic collapsed to ~0, and the threshold degenerated to its floor —
/// below the real row pitch, so the crossing chained as dashes (the kumiko
/// strut end-patch x wedge plane).
#[test]
fn plane_crossing_between_grid_columns_is_one_curve() {
    // tilted_surface: z from -0.5 at x=0 to +0.5 at x=1; plane z=0 cuts it
    // along the vertical UV line u=0.5. At 10 samples the row pitch (0.111)
    // exceeds the 5%-of-diagonal threshold floor (0.05).
    let surface = tilted_surface();
    let curves = intersect_plane_nurbs(&surface, Vec3::new(0.0, 0.0, 1.0), 0.0, 10).unwrap();
    assert_eq!(
        curves.len(),
        1,
        "one transversal plane crossing must chain into one curve, got {}",
        curves.len()
    );
}

/// A marched section that leaves the domain must end ON the exact domain
/// boundary, not at the marcher's 0.1%-of-span interior clamp margin. Two
/// faces sharing a boundary edge each march their own section; if both stop
/// a margin short, the chain ends miss by twice the margin scaled by patch
/// size (~2e-3 on the kumiko strut quads) and no downstream weld can close
/// the junction.
#[test]
fn marched_section_ends_on_exact_domain_boundary() {
    let s1 = flat_surface();
    let s2 = tilted_surface();

    // Intersection: the line x=0.5 on z=0, crossing v (= y) from 0 to 1.
    let result = intersect_nurbs_nurbs(&s1, &s2, 10, 0.05).unwrap();
    assert!(!result.is_empty());

    let mut y_min = f64::MAX;
    let mut y_max = f64::MIN;
    for curve in &result {
        for pt in [curve.points.first(), curve.points.last()]
            .into_iter()
            .flatten()
        {
            y_min = y_min.min(pt.point.y());
            y_max = y_max.max(pt.point.y());
        }
    }
    assert!(
        y_min.abs() < 1e-6,
        "chain end must reach the v=0 boundary exactly, got y_min={y_min:.9}"
    );
    assert!(
        (y_max - 1.0).abs() < 1e-6,
        "chain end must reach the v=1 boundary exactly, got y_max={y_max:.9}"
    );
}

#[test]
fn ssi_points_lie_on_both_surfaces() {
    let s1 = flat_surface();
    let s2 = tilted_surface();
    let result = intersect_nurbs_nurbs(&s1, &s2, 10, 0.02).unwrap();

    for curve in &result {
        for pt in &curve.points {
            // Check point lies on surface 1.
            let p1 = s1.evaluate(pt.param1.0, pt.param1.1);
            let dist1 = (p1 - pt.point).length();
            assert!(dist1 < 0.05, "point should lie on surface 1, dist={dist1}");

            // Check point lies on surface 2.
            let p2 = s2.evaluate(pt.param2.0, pt.param2.1);
            let dist2 = (p2 - pt.point).length();
            assert!(dist2 < 0.05, "point should lie on surface 2, dist={dist2}");
        }
    }
}

/// Create a dome-shaped NURBS surface (quadratic, unit domain).
/// High at center (z=2), low at edges (z=-1), so slicing at z=0
/// produces a closed ring-like intersection.
fn dome_surface() -> NurbsSurface {
    NurbsSurface::new(
        2,
        2,
        vec![0.0, 0.0, 0.0, 1.0, 1.0, 1.0],
        vec![0.0, 0.0, 0.0, 1.0, 1.0, 1.0],
        vec![
            vec![
                Point3::new(0.0, 0.0, -1.0),
                Point3::new(0.0, 0.5, 0.5),
                Point3::new(0.0, 1.0, -1.0),
            ],
            vec![
                Point3::new(0.5, 0.0, 0.5),
                Point3::new(0.5, 0.5, 2.0),
                Point3::new(0.5, 1.0, 0.5),
            ],
            vec![
                Point3::new(1.0, 0.0, -1.0),
                Point3::new(1.0, 0.5, 0.5),
                Point3::new(1.0, 1.0, -1.0),
            ],
        ],
        vec![vec![1.0; 3]; 3],
    )
    .unwrap()
}

/// Create a flat surface at a given z height, mapping [0,1]^2 to the
/// same XY extent [0,1]x[0,1] as the dome.
fn flat_plane_at_z(z: f64) -> NurbsSurface {
    NurbsSurface::new(
        1,
        1,
        vec![0.0, 0.0, 1.0, 1.0],
        vec![0.0, 0.0, 1.0, 1.0],
        vec![
            vec![Point3::new(0.0, 0.0, z), Point3::new(0.0, 1.0, z)],
            vec![Point3::new(1.0, 0.0, z), Point3::new(1.0, 1.0, z)],
        ],
        vec![vec![1.0, 1.0], vec![1.0, 1.0]],
    )
    .unwrap()
}

#[test]
fn ssi_tangential_touch() {
    // Two surfaces that touch tangentially: a dome and a flat plane at the
    // dome's peak height. The normals are parallel at the touch point, so
    // this exercises the singular_tangent_direction fallback.
    let dome = dome_surface();
    // The dome peaks around z=2 at the center. Use a plane slightly below
    // to create a tangential touch region.
    let peak_z = dome.evaluate(0.5, 0.5).z();

    // Place the plane at the peak height -- tangential contact.
    let plane = flat_plane_at_z(peak_z);

    // At the tangent point both normals point in +z, so cross product vanishes.
    // The marching should handle this gracefully via singular_tangent_direction.
    let seed = refine_ssi_point(&dome, &plane, 0.5, 0.5, 0.5, 0.5, 1e-6);
    assert!(
        seed.is_some(),
        "should find a seed at the tangential contact point"
    );

    let seed = seed.unwrap();
    assert!(
        (seed.point.z() - peak_z).abs() < 0.2,
        "seed should be near z={peak_z}, got z={}",
        seed.point.z()
    );

    // March from the tangential point. The key requirement is that this
    // does not panic and handles the singular point.
    let traced = march_intersection(&dome, &plane, &seed, 0.05, 1e-6);

    // At a true tangential touch (single point contact), marching may
    // produce few or no additional points -- that's acceptable. The test
    // ensures we don't crash/panic at the singular point.
    // If the plane is slightly below peak, there may be a small intersection
    // loop.
    for pt in &traced {
        // All traced points should be reasonably close to both surfaces.
        let p1 = dome.evaluate(pt.param1.0, pt.param1.1);
        let p2 = plane.evaluate(pt.param2.0, pt.param2.1);
        let dist1 = (p1 - pt.point).length();
        let dist2 = (p2 - pt.point).length();
        assert!(
            dist1 < 0.5,
            "traced point should be near dome surface, dist={dist1}"
        );
        assert!(
            dist2 < 0.5,
            "traced point should be near plane surface, dist={dist2}"
        );
    }
}

#[test]
fn ssi_closed_loop() {
    // Intersect a dome surface with a horizontal plane.
    // Use a known seed point and march directly to test closed-loop
    // detection without the expensive O(n^4) seed search.
    let dome = dome_surface();
    let plane = flat_plane_at_z(0.0);

    // Find one seed by refining a point we know is on the intersection
    // (from the debug test: the z=0 contour passes through the region
    // around u=0.25 on the dome).
    let seed = refine_ssi_point(&dome, &plane, 0.25, 0.5, 0.25, 0.5, 1e-6)
        .expect("should refine to a seed on the dome-plane intersection");

    // Verify the seed is near z=0.
    assert!(
        seed.point.z().abs() < 0.1,
        "seed should be near z=0, got z={}",
        seed.point.z()
    );

    // March from the seed.
    let traced = march_intersection(&dome, &plane, &seed, 0.05, 1e-6);

    assert!(
        traced.len() >= 5,
        "should trace at least 5 points, got {}",
        traced.len()
    );

    // Check that the curve closes: first and last points should be close.
    let first = &traced[0];
    let last = &traced[traced.len() - 1];
    let gap = (first.point - last.point).length();

    assert!(
        gap < 0.5,
        "expected closed loop (first-last gap < 0.5), got gap={gap:.4}"
    );

    // All points should lie near z=0.
    for pt in &traced {
        assert!(
            pt.point.z().abs() < 0.15,
            "intersection point should be near z=0, got z={}",
            pt.point.z()
        );
    }
}

// -- Subdivision seed finder tests --

#[test]
fn subdivision_finds_seeds() {
    let s1 = flat_surface();
    let s2 = tilted_surface();

    let seeds = find_ssi_seeds_subdivision(&s1, &s2, 1e-6);
    assert!(
        !seeds.is_empty(),
        "subdivision should find seeds between flat and tilted"
    );

    // All seeds should lie on both surfaces
    for seed in &seeds {
        let p1 = s1.evaluate(seed.param1.0, seed.param1.1);
        let p2 = s2.evaluate(seed.param2.0, seed.param2.1);
        assert!(
            (p1 - seed.point).length() < 0.01,
            "seed should lie on surface 1"
        );
        assert!(
            (p2 - seed.point).length() < 0.01,
            "seed should lie on surface 2"
        );
    }
}

// -- Chain building tests --

#[test]
fn chain_separates_branches() {
    // Two clusters of points with a gap between them
    let points = vec![
        IntersectionPoint {
            point: Point3::new(0.0, 0.0, 0.0),
            param1: (0.0, 0.0),
            param2: (0.0, 0.0),
        },
        IntersectionPoint {
            point: Point3::new(0.1, 0.0, 0.0),
            param1: (0.1, 0.0),
            param2: (0.1, 0.0),
        },
        IntersectionPoint {
            point: Point3::new(0.2, 0.0, 0.0),
            param1: (0.2, 0.0),
            param2: (0.2, 0.0),
        },
        // Gap
        IntersectionPoint {
            point: Point3::new(5.0, 0.0, 0.0),
            param1: (0.5, 0.0),
            param2: (0.5, 0.0),
        },
        IntersectionPoint {
            point: Point3::new(5.1, 0.0, 0.0),
            param1: (0.6, 0.0),
            param2: (0.6, 0.0),
        },
    ];

    let chains = chain_intersection_points(&points, 0.5);
    assert_eq!(
        chains.len(),
        2,
        "should separate into 2 branches, got {}",
        chains.len()
    );
}

#[test]
fn chain_detects_single_group() {
    // Points close together: should form 1 chain
    let points: Vec<IntersectionPoint> = (0..5)
        .map(|i| {
            let x = f64::from(i) * 0.1;
            IntersectionPoint {
                point: Point3::new(x, 0.0, 0.0),
                param1: (x, 0.0),
                param2: (x, 0.0),
            }
        })
        .collect();

    let chains = chain_intersection_points(&points, 0.5);
    assert_eq!(chains.len(), 1, "all close points should form 1 chain");
    assert_eq!(chains[0].len(), 5);
}

/// Test second-order tangent analysis with two nearly-tangent surfaces.
#[test]
fn second_order_tangent_finds_direction() {
    // Two surfaces that touch at (0.5, 0.5): one flat, one dome.
    // At the touch point, normals are parallel (both ~+z), so
    // first-order tangent n1 x n2 ~ 0.
    let dome = dome_surface();
    let peak_z = dome.evaluate(0.5, 0.5).z();

    // Place a flat plane at the dome's peak height.
    let plane = flat_plane_at_z(peak_z);

    // Try the second-order analysis.
    let result = second_order_tangent(&dome, &plane, 0.5, 0.5, 0.5, 0.5);

    // The result should be Some (a direction was found) or None
    // (degenerate -- surfaces osculate to second order).
    // For a dome with quadratic curvature vs flat plane, the
    // curvature difference is non-zero, so we should get a direction.
    if let Some(dir) = result {
        // The direction should be a unit vector in the tangent plane.
        let len = dir.length();
        assert!(
            (len - 1.0).abs() < 0.01,
            "tangent direction should be unit length, got {len}"
        );
        // The direction should be roughly in the XY plane (since
        // both surfaces are horizontal at the touch point).
        assert!(
            dir.z().abs() < 0.5,
            "tangent direction should be mostly horizontal, got z={}",
            dir.z()
        );
    }
    // None is also acceptable for this degenerate case -- it means
    // the perturbation fallback will be used.
}

// -- Non-normalized domain tests --

/// Create a bilinear surface over domain [0, 100] x [0, 100].
fn wide_domain_surface(z: f64) -> NurbsSurface {
    NurbsSurface::new(
        1,
        1,
        vec![0.0, 0.0, 100.0, 100.0],
        vec![0.0, 0.0, 100.0, 100.0],
        vec![
            vec![Point3::new(0.0, 0.0, z), Point3::new(0.0, 10.0, z)],
            vec![Point3::new(10.0, 0.0, z), Point3::new(10.0, 10.0, z)],
        ],
        vec![vec![1.0, 1.0], vec![1.0, 1.0]],
    )
    .unwrap()
}

/// Create a tilted surface over domain [0, 100] x [0, 100] that
/// crosses z=0 at x=5.
fn wide_domain_tilted() -> NurbsSurface {
    NurbsSurface::new(
        1,
        1,
        vec![0.0, 0.0, 100.0, 100.0],
        vec![0.0, 0.0, 100.0, 100.0],
        vec![
            vec![Point3::new(0.0, 0.0, -5.0), Point3::new(0.0, 10.0, -5.0)],
            vec![Point3::new(10.0, 0.0, 5.0), Point3::new(10.0, 10.0, 5.0)],
        ],
        vec![vec![1.0, 1.0], vec![1.0, 1.0]],
    )
    .unwrap()
}

#[test]
fn plane_nurbs_wide_domain() {
    // Surface with knot domain [0, 100] -- should still find the
    // intersection with the z=0 plane.
    let tilted = wide_domain_tilted();

    // Verify domain is actually [0, 100].
    let (u_min, u_max) = tilted.domain_u();
    let (v_min, v_max) = tilted.domain_v();
    assert!(u_min.abs() < 1e-10);
    assert!((u_max - 100.0).abs() < 1e-10);
    assert!(v_min.abs() < 1e-10);
    assert!((v_max - 100.0).abs() < 1e-10);

    let result = intersect_plane_nurbs(&tilted, Vec3::new(0.0, 0.0, 1.0), 0.0, 50).unwrap();

    assert!(
        !result.is_empty(),
        "should find intersection on [0,100] domain surface"
    );

    for curve in &result {
        for pt in &curve.points {
            assert!(
                pt.point.z().abs() < 0.2,
                "intersection point should be near z=0, got z={}",
                pt.point.z()
            );
            // x should be near 5.0 (the midpoint where z crosses 0)
            assert!(
                (pt.point.x() - 5.0).abs() < 1.0,
                "x should be near 5.0, got {}",
                pt.point.x()
            );
        }
    }
}

#[test]
fn ssi_wide_domain_surfaces() {
    // Two surfaces with [0, 100] domains that intersect.
    let s1 = wide_domain_surface(0.0);
    let s2 = wide_domain_tilted();

    // Verify domains.
    assert!((s1.domain_u().1 - 100.0).abs() < 1e-10);
    assert!((s2.domain_u().1 - 100.0).abs() < 1e-10);

    let seeds = find_ssi_seeds_grid(&s1, &s2, 15, 1e-6);
    assert!(
        !seeds.is_empty(),
        "should find seeds between wide-domain surfaces"
    );

    let result = intersect_nurbs_nurbs(&s1, &s2, 15, 0.0).unwrap();
    assert!(
        !result.is_empty(),
        "should find SSI on [0,100] domain surfaces"
    );

    for curve in &result {
        for pt in &curve.points {
            assert!(
                pt.point.z().abs() < 0.5,
                "SSI point should be near z=0, got z={}",
                pt.point.z()
            );
        }
    }
}

#[test]
fn line_nurbs_wide_domain() {
    // Ray intersection with a surface having [0, 100] domain.
    let surface = wide_domain_surface(0.0);

    let result = intersect_line_nurbs(
        &surface,
        Point3::new(5.0, 5.0, 1.0),
        Vec3::new(0.0, 0.0, -1.0),
        20,
    )
    .unwrap();

    assert!(!result.is_empty(), "ray should hit wide-domain surface");

    let pt = &result[0];
    assert!(
        (pt.point.x() - 5.0).abs() < 0.5,
        "x should be ~5.0, got {}",
        pt.point.x()
    );
    assert!(
        pt.point.z().abs() < 0.1,
        "z should be ~0.0, got {}",
        pt.point.z()
    );
}

/// Create a half-cylinder-like surface with v-domain [0, 2pi].
fn cylinder_nurbs_surface() -> NurbsSurface {
    use std::f64::consts::PI;
    let tau = 2.0 * PI;
    // Approximate a cylinder of radius 1, height 2, with a degree-2
    // NURBS surface in v (angular) and degree-1 in u (height).
    // Use 9 control points in v for a full circle (rational).
    let r = 1.0;
    let w = std::f64::consts::FRAC_1_SQRT_2; // cos(45 deg)

    // v knots for a full circle: [0,0,0, pi/2,pi/2, pi,pi, 3pi/2,3pi/2, 2pi,2pi,2pi]
    let knots_v = vec![
        0.0,
        0.0,
        0.0,
        PI / 2.0,
        PI / 2.0,
        PI,
        PI,
        3.0 * PI / 2.0,
        3.0 * PI / 2.0,
        tau,
        tau,
        tau,
    ];

    // 9 control points around the circle at z=0 and z=2.
    let circle_cps = [
        (r, 0.0, 1.0),
        (r, r, w),
        (0.0, r, 1.0),
        (-r, r, w),
        (-r, 0.0, 1.0),
        (-r, -r, w),
        (0.0, -r, 1.0),
        (r, -r, w),
        (r, 0.0, 1.0),
    ];

    let cps_bottom: Vec<Point3> = circle_cps
        .iter()
        .map(|&(x, y, _)| Point3::new(x, y, 0.0))
        .collect();
    let cps_top: Vec<Point3> = circle_cps
        .iter()
        .map(|&(x, y, _)| Point3::new(x, y, 2.0))
        .collect();

    let weights_row: Vec<f64> = circle_cps.iter().map(|&(_, _, w_)| w_).collect();

    NurbsSurface::new(
        1,
        2,
        vec![0.0, 0.0, 2.0, 2.0], // u: height [0, 2]
        knots_v,
        vec![cps_bottom, cps_top],
        vec![weights_row.clone(), weights_row],
    )
    .unwrap()
}

#[test]
fn plane_nurbs_cylinder_domain() {
    use std::f64::consts::PI;
    let cylinder = cylinder_nurbs_surface();

    // Verify domain is [0,2] x [0, 2pi].
    let (u_min, u_max) = cylinder.domain_u();
    let (v_min, v_max) = cylinder.domain_v();
    assert!((u_min - 0.0).abs() < 1e-10);
    assert!((u_max - 2.0).abs() < 1e-10);
    assert!((v_min - 0.0).abs() < 1e-10);
    assert!((v_max - 2.0 * PI).abs() < 1e-10);

    // Intersect with a plane at z=1 (horizontal slice through cylinder).
    let result = intersect_plane_nurbs(&cylinder, Vec3::new(0.0, 0.0, 1.0), 1.0, 50).unwrap();

    assert!(
        !result.is_empty(),
        "should find intersection of cylinder with z=1 plane"
    );

    // All intersection points should be near z=1 and at radius ~1.
    for curve in &result {
        for pt in &curve.points {
            assert!(
                (pt.point.z() - 1.0).abs() < 0.2,
                "z should be ~1.0, got {}",
                pt.point.z()
            );
            let r = (pt.point.x().powi(2) + pt.point.y().powi(2)).sqrt();
            assert!((r - 1.0).abs() < 0.2, "radius should be ~1.0, got {r}");
        }
    }
}

/// Verify that the tangential touch test still works with the new
/// second-order analysis integrated into the main SSI pipeline.
#[test]
fn ssi_tangential_with_second_order() {
    let dome = dome_surface();
    let peak_z = dome.evaluate(0.5, 0.5).z();
    let plane = flat_plane_at_z(peak_z - 0.3); // Below peak but not extremely close

    // This should find an intersection loop near the peak.
    // Use a large march step since we only care about correctness, not density.
    let result = intersect_nurbs_nurbs(&dome, &plane, 5, 0.2).unwrap();

    // Near-tangential: may or may not find an intersection (depends
    // on numerical precision), but should NOT crash.
    for curve in &result {
        for pt in &curve.points {
            // All points should be close to the plane height.
            assert!(
                (pt.point.z() - (peak_z - 0.3)).abs() < 0.5,
                "intersection point should be near z={:.2}, got z={:.4}",
                peak_z - 0.3,
                pt.point.z()
            );
        }
    }
}

/// Line curve through the flat surface at z=0: from (-1,-1,0.5) to (2,2,-0.5).
/// Should cross the unit square plane at one point.
#[test]
fn curve_surface_line_through_flat_plane() {
    use crate::nurbs::curve::NurbsCurve;

    let surf = flat_surface(); // z=0 plane, (0..1, 0..1)
    // Straight line from (-1,-1,0.5) to (2,2,-0.5) as degree-1 NURBS.
    let curve = NurbsCurve::new(
        1,
        vec![0.0, 0.0, 1.0, 1.0],
        vec![Point3::new(-1.0, -1.0, 0.5), Point3::new(2.0, 2.0, -0.5)],
        vec![1.0, 1.0],
    )
    .unwrap();

    let hits = intersect_curve_surface(&curve, &surf, 1e-7).unwrap();
    assert_eq!(hits.len(), 1, "expected 1 hit, got {}", hits.len());

    let hit = &hits[0];
    // The line is C(t) = (-1 + 3t, -1 + 3t, 0.5 - t). C(t).z = 0 -> t = 0.5.
    // C(0.5) = (0.5, 0.5, 0.0).
    assert!(
        (hit.point.z()).abs() < 1e-5,
        "z should be ~0, got {}",
        hit.point.z()
    );
    assert!(
        (hit.point.x() - 0.5).abs() < 1e-5,
        "x should be ~0.5, got {}",
        hit.point.x()
    );
    assert!(
        (hit.t - 0.5).abs() < 1e-4,
        "t should be ~0.5, got {}",
        hit.t
    );
}

/// A degree-2 curve (parabola) intersecting a flat plane -- should find 2 points.
#[test]
fn curve_surface_parabola_through_flat_plane() {
    use crate::nurbs::curve::NurbsCurve;

    let surf = flat_surface(); // z=0, (0..1, 0..1)
    // Quadratic curve from (0.2, 0.5, -0.3) through control (0.5, 0.5, 1.0)
    // to (0.8, 0.5, -0.3). The z-component is:
    //   z(t) = (1-t)^2(-0.3) + 2t(1-t)(1.0) + t^2(-0.3)
    //        = -0.3 + 2.6t - 2.6t^2
    // z = 0 at t ~ 0.133 and t ~ 0.867 -- two clear crossings.
    let curve = NurbsCurve::new(
        2,
        vec![0.0, 0.0, 0.0, 1.0, 1.0, 1.0],
        vec![
            Point3::new(0.2, 0.5, -0.3),
            Point3::new(0.5, 0.5, 1.0),
            Point3::new(0.8, 0.5, -0.3),
        ],
        vec![1.0, 1.0, 1.0],
    )
    .unwrap();

    let hits = intersect_curve_surface(&curve, &surf, 1e-7).unwrap();
    assert_eq!(hits.len(), 2, "expected 2 hits, got {}", hits.len());

    // Both hits should be on the z=0 plane.
    for hit in &hits {
        assert!(
            hit.point.z().abs() < 1e-4,
            "z should be ~0, got {}",
            hit.point.z()
        );
    }
    // Parameters should be symmetric around 0.5.
    assert!(hits[0].t < 0.5, "first hit t should be < 0.5");
    assert!(hits[1].t > 0.5, "second hit t should be > 0.5");
}

/// Build a cylinder NURBS surface along z-axis, centered at (cx, cy).
fn cylinder_at(cx: f64, cy: f64, r: f64, z_lo: f64, z_hi: f64) -> NurbsSurface {
    use std::f64::consts::PI;
    let tau = 2.0 * PI;
    let w = std::f64::consts::FRAC_1_SQRT_2;

    let knots_v = vec![
        0.0,
        0.0,
        0.0,
        PI / 2.0,
        PI / 2.0,
        PI,
        PI,
        3.0 * PI / 2.0,
        3.0 * PI / 2.0,
        tau,
        tau,
        tau,
    ];

    let circle_cps = [
        (r, 0.0, 1.0),
        (r, r, w),
        (0.0, r, 1.0),
        (-r, r, w),
        (-r, 0.0, 1.0),
        (-r, -r, w),
        (0.0, -r, 1.0),
        (r, -r, w),
        (r, 0.0, 1.0),
    ];

    let cps_lo: Vec<Point3> = circle_cps
        .iter()
        .map(|&(x, y, _)| Point3::new(cx + x, cy + y, z_lo))
        .collect();
    let cps_hi: Vec<Point3> = circle_cps
        .iter()
        .map(|&(x, y, _)| Point3::new(cx + x, cy + y, z_hi))
        .collect();

    let weights: Vec<f64> = circle_cps.iter().map(|&(_, _, w_)| w_).collect();

    NurbsSurface::new(
        1,
        2,
        vec![z_lo, z_lo, z_hi, z_hi],
        knots_v,
        vec![cps_lo, cps_hi],
        vec![weights.clone(), weights],
    )
    .unwrap()
}

/// Build a cylinder NURBS surface along x-axis, centered at (cy, cz).
fn cylinder_along_x(cy: f64, cz: f64, r: f64, x_lo: f64, x_hi: f64) -> NurbsSurface {
    use std::f64::consts::PI;
    let tau = 2.0 * PI;
    let w = std::f64::consts::FRAC_1_SQRT_2;

    let knots_v = vec![
        0.0,
        0.0,
        0.0,
        PI / 2.0,
        PI / 2.0,
        PI,
        PI,
        3.0 * PI / 2.0,
        3.0 * PI / 2.0,
        tau,
        tau,
        tau,
    ];

    // Circle in YZ plane.
    let circle_cps = [
        (r, 0.0, 1.0),
        (r, r, w),
        (0.0, r, 1.0),
        (-r, r, w),
        (-r, 0.0, 1.0),
        (-r, -r, w),
        (0.0, -r, 1.0),
        (r, -r, w),
        (r, 0.0, 1.0),
    ];

    let cps_lo: Vec<Point3> = circle_cps
        .iter()
        .map(|&(y, z, _)| Point3::new(x_lo, cy + y, cz + z))
        .collect();
    let cps_hi: Vec<Point3> = circle_cps
        .iter()
        .map(|&(y, z, _)| Point3::new(x_hi, cy + y, cz + z))
        .collect();

    let weights: Vec<f64> = circle_cps.iter().map(|&(_, _, w_)| w_).collect();

    NurbsSurface::new(
        1,
        2,
        vec![x_lo, x_lo, x_hi, x_hi],
        knots_v,
        vec![cps_lo, cps_hi],
        vec![weights.clone(), weights],
    )
    .unwrap()
}

#[test]
fn ssi_perpendicular_cylinders_two_loops() {
    // Two perpendicular cylinders of radius 1 centered at the origin:
    // cylinder A along z-axis, cylinder B along x-axis.
    // They produce two distinct closed intersection loops.
    let cyl_z = cylinder_at(0.0, 0.0, 1.0, -2.0, 2.0);
    let cyl_x = cylinder_along_x(0.0, 0.0, 1.0, -2.0, 2.0);

    let result = intersect_nurbs_nurbs(&cyl_z, &cyl_x, 20, 0.0).unwrap();

    // Should find at least 1 curve (ideally 2 for both loops).
    assert!(
        !result.is_empty(),
        "perpendicular cylinders must produce intersection curves"
    );

    // Verify all intersection points lie on both surfaces.
    for curve in &result {
        for pt in &curve.points {
            let on_cyl_z = {
                let x = pt.point.x();
                let y = pt.point.y();
                (x * x + y * y).sqrt()
            };
            let on_cyl_x = {
                let y = pt.point.y();
                let z = pt.point.z();
                (y * y + z * z).sqrt()
            };
            assert!(
                (on_cyl_z - 1.0).abs() < 0.05,
                "point should be on z-cylinder (r={on_cyl_z})"
            );
            assert!(
                (on_cyl_x - 1.0).abs() < 0.05,
                "point should be on x-cylinder (r={on_cyl_x})"
            );
        }
    }
}

#[test]
fn segment_distance_dedup_works() {
    // Verify that near_existing_segment uses segment distance,
    // not just point distance.
    let p0 = IntersectionPoint {
        point: Point3::new(0.0, 0.0, 0.0),
        param1: (0.0, 0.0),
        param2: (0.0, 0.0),
    };
    let p1 = IntersectionPoint {
        point: Point3::new(10.0, 0.0, 0.0),
        param1: (1.0, 0.0),
        param2: (1.0, 0.0),
    };
    let segment = vec![p0, p1];

    // Point near the middle of the segment (y=0.01).
    let near_mid = IntersectionPoint {
        point: Point3::new(5.0, 0.01, 0.0),
        param1: (0.5, 0.0),
        param2: (0.5, 0.0),
    };
    assert!(near_existing_segment(
        std::slice::from_ref(&segment),
        &near_mid,
        0.1
    ));

    // Point far from the segment (y=2.0).
    let far = IntersectionPoint {
        point: Point3::new(5.0, 2.0, 0.0),
        param1: (0.5, 0.0),
        param2: (0.5, 0.0),
    };
    assert!(!near_existing_segment(
        std::slice::from_ref(&segment),
        &far,
        0.1
    ));
}

#[test]
fn dual_surface_validation_passes_for_known_intersection() {
    use crate::nurbs::projection::project_point_to_surface;

    // Two transversely intersecting planar NURBS surfaces: flat (z=0) and
    // tilted (z goes from -0.5 to +0.5 across x). Their intersection is a
    // line at x=0.5 that must lie on both surfaces within tolerance.
    let s1 = flat_surface();
    let s2 = tilted_surface();

    let curves = intersect_nurbs_nurbs(&s1, &s2, 15, 0.02).unwrap();
    assert!(
        !curves.is_empty(),
        "transverse planar surfaces should produce at least one intersection curve"
    );

    let tol = 1e-3;
    for ic in &curves {
        let (t_min, t_max) = ic.curve.domain();
        for i in 0..5 {
            let t = t_min + (t_max - t_min) * i as f64 / 4.0;
            let pt = ic.curve.evaluate(t);

            // Point must be close to surface 1.
            let proj1 = project_point_to_surface(&s1, pt, tol).unwrap();
            assert!(
                proj1.distance < tol,
                "curve point at t={t:.3} deviates {:.2e} from surface 1",
                proj1.distance
            );

            // Point must be close to surface 2.
            let proj2 = project_point_to_surface(&s2, pt, tol).unwrap();
            assert!(
                proj2.distance < tol,
                "curve point at t={t:.3} deviates {:.2e} from surface 2",
                proj2.distance
            );
        }
    }
}

fn flat_at(z: f64) -> NurbsSurface {
    NurbsSurface::new(
        1,
        1,
        vec![0.0, 0.0, 1.0, 1.0],
        vec![0.0, 0.0, 1.0, 1.0],
        vec![
            vec![Point3::new(0.0, 0.0, z), Point3::new(0.0, 1.0, z)],
            vec![Point3::new(1.0, 0.0, z), Point3::new(1.0, 1.0, z)],
        ],
        vec![vec![1.0, 1.0], vec![1.0, 1.0]],
    )
    .unwrap()
}

#[test]
fn grid_seeder_declares_a_grazing_disjoint_pair_empty() {
    // 0.05 apart: every sample pair sits inside the grid's closeness
    // threshold, so before the failure budget this refined all of them.
    let s1 = flat_at(0.0);
    let s2 = flat_at(0.05);
    let before = REFINE_CALLS.with(std::cell::Cell::get);
    assert!(find_ssi_seeds_grid(&s1, &s2, 32, 1e-6).is_empty());
    let refinements = REFINE_CALLS.with(std::cell::Cell::get) - before;
    // At most one refinement per mutual nearest pair (n*n) plus the bounded
    // closest set; the exhaustive pass refined every one of the ~n^4 pairs.
    assert!(
        refinements <= 32 * 32 + 256,
        "grid seeder refined {refinements} pairs on a disjoint pair"
    );
    assert!(
        intersect_nurbs_nurbs(&s1, &s2, 32, 0.01)
            .unwrap()
            .is_empty()
    );
}

#[test]
fn grid_seeder_keeps_searching_after_its_first_seed() {
    // Two separate crossings, one per opposite corner, with the surfaces
    // grazing above z=0 everywhere else: both must be seeded.
    let s1 = flat_at(0.0);
    let s2 = NurbsSurface::new(
        1,
        1,
        vec![0.0, 0.0, 1.0, 1.0],
        vec![0.0, 0.0, 1.0, 1.0],
        vec![
            vec![Point3::new(0.0, 0.0, -0.02), Point3::new(0.0, 1.0, 0.08)],
            vec![Point3::new(1.0, 0.0, 0.08), Point3::new(1.0, 1.0, -0.02)],
        ],
        vec![vec![1.0, 1.0], vec![1.0, 1.0]],
    )
    .unwrap();
    let seeds = find_ssi_seeds_grid(&s1, &s2, 32, 1e-6);
    let near = |x: f64, y: f64| {
        seeds
            .iter()
            .any(|s| (s.point.x() - x).abs() < 0.3 && (s.point.y() - y).abs() < 0.3)
    };
    assert!(
        near(0.0, 0.0),
        "the (0, 0) crossing was not seeded: {seeds:?}"
    );
    assert!(
        near(1.0, 1.0),
        "the (1, 1) crossing was not seeded: {seeds:?}"
    );
}

#[test]
fn grid_seeder_budget_keeps_a_shallow_crossing() {
    // A 2.3 degree tilt crossing z=0 along x=0.5: the closest sample pairs
    // are on the crossing and must seed it before any budget applies.
    let s1 = flat_at(0.0);
    let s2 = NurbsSurface::new(
        1,
        1,
        vec![0.0, 0.0, 1.0, 1.0],
        vec![0.0, 0.0, 1.0, 1.0],
        vec![
            vec![Point3::new(0.0, 0.0, -0.02), Point3::new(0.0, 1.0, -0.02)],
            vec![Point3::new(1.0, 0.0, 0.02), Point3::new(1.0, 1.0, 0.02)],
        ],
        vec![vec![1.0, 1.0], vec![1.0, 1.0]],
    )
    .unwrap();
    let seeds = find_ssi_seeds_grid(&s1, &s2, 32, 1e-6);
    assert!(!seeds.is_empty(), "the crossing must still be seeded");
    for seed in &seeds {
        assert!(
            seed.point.z().abs() < 1e-5,
            "seed off the plane: {:?}",
            seed.point
        );
        assert!(
            (seed.point.x() - 0.5).abs() < 1e-4,
            "seed off the crossing: {:?}",
            seed.point
        );
    }
    let curves = intersect_nurbs_nurbs(&s1, &s2, 32, 0.01).unwrap();
    assert!(!curves.is_empty(), "the crossing must still be traced");
}

#[test]
fn grid_seeder_budget_keeps_a_crossing_confined_to_one_corner() {
    // s2 dips below z=0 only near the (1, 1) corner: almost every sample pair
    // is close but far from the crossing, so the closest-first order and
    // the failure budget must not give up before reaching it.
    let s1 = flat_at(0.0);
    let s2 = NurbsSurface::new(
        1,
        1,
        vec![0.0, 0.0, 1.0, 1.0],
        vec![0.0, 0.0, 1.0, 1.0],
        vec![
            vec![Point3::new(0.0, 0.0, 0.08), Point3::new(0.0, 1.0, 0.08)],
            vec![Point3::new(1.0, 0.0, 0.08), Point3::new(1.0, 1.0, -0.02)],
        ],
        vec![vec![1.0, 1.0], vec![1.0, 1.0]],
    )
    .unwrap();
    let seeds = find_ssi_seeds_grid(&s1, &s2, 32, 1e-6);
    assert!(!seeds.is_empty(), "the corner crossing must be seeded");
    for seed in &seeds {
        assert!(
            seed.point.z().abs() < 1e-5,
            "seed off the plane: {:?}",
            seed.point
        );
        assert!(
            seed.point.x() > 0.7 && seed.point.y() > 0.7,
            "seed away from the corner: {:?}",
            seed.point
        );
    }
}

/// How many times a section's points turn about their centroid in `xy`.
fn xy_winding(points: &[IntersectionPoint]) -> f64 {
    #[allow(clippy::cast_precision_loss)]
    let n = points.len() as f64;
    let (cx, cy) = points.iter().fold((0.0, 0.0), |(x, y), p| {
        (x + p.point.x() / n, y + p.point.y() / n)
    });
    let angle = |p: &IntersectionPoint| (p.point.y() - cy).atan2(p.point.x() - cx);
    let turned: f64 = points
        .windows(2)
        .map(|w| {
            let d = angle(&w[1]) - angle(&w[0]);
            d - (d / std::f64::consts::TAU).round() * std::f64::consts::TAU
        })
        .sum();
    turned.abs() / std::f64::consts::TAU
}

/// A closed section is traced once: a plane through a drilled cylinder cuts
/// one circle, and the forward march closing it onto its seed must not be
/// followed by a backward march tracing it again, nor may a step striding
/// past the seed keep winding it.
#[test]
fn closed_sections_are_traced_once() {
    let hole = intersect_nurbs_nurbs(
        &flat_plane_at_z(0.0),
        &cylinder_at(0.5, 0.5, 0.2, -1.0, 1.0),
        32,
        0.01,
    )
    .unwrap();
    assert_eq!(hole.len(), 1, "one circle");
    let pts = &hole[0].points;
    let length: f64 = pts
        .windows(2)
        .map(|w| (w[1].point - w[0].point).length())
        .sum();
    let circumference = std::f64::consts::TAU * 0.2;
    assert!(
        (length - circumference).abs() < 0.02 * circumference,
        "traced length {length}, circumference {circumference}"
    );
    assert!(
        (xy_winding(pts) - 1.0).abs() < 0.05,
        "winding {}",
        xy_winding(pts)
    );

    let peak = dome_surface().evaluate(0.5, 0.5).z();
    let dome =
        intersect_nurbs_nurbs(&dome_surface(), &flat_plane_at_z(peak - 0.3), 32, 0.01).unwrap();
    assert_eq!(dome.len(), 1, "one loop round the dome");
    let winding = xy_winding(&dome[0].points);
    assert!(
        (winding - 1.0).abs() < 0.05,
        "dome section winds {winding} times"
    );
}

/// An open section whose end lies within the join distance of a closed
/// loop's start stays apart from the loop, whichever of the two was traced
/// first.
#[test]
fn an_open_section_never_absorbs_a_closed_loop() {
    let at = |x: f64, y: f64| IntersectionPoint {
        point: Point3::new(x, y, 0.0),
        param1: (0.0, 0.0),
        param2: (0.0, 0.0),
    };
    let open = vec![at(-1.0, 0.0), at(-0.5, 0.0), at(0.99, 0.0)];
    let loop_: Vec<IntersectionPoint> = (0..=8)
        .map(|k| {
            let a = std::f64::consts::TAU * f64::from(k % 8) / 8.0;
            at(2.0 - a.cos(), a.sin())
        })
        .collect();
    for segments in [vec![open.clone(), loop_.clone()], vec![loop_, open]] {
        let chains = super::chaining::chain_traced_segments(segments, 1e-3, 0.05);
        assert_eq!(chains.len(), 2, "the loop and the open section stay apart");
    }
}

/// The tool's scoop: a cubic profile from the floor lip (y -5.117, z 2.25)
/// up to the back wall (y -18.15, z 15.283), extruded along x over
/// [-44.03, 44.03] (u is x, v runs the profile from the wall down).
fn scoop_surface() -> NurbsSurface {
    let profile = [
        (-18.15, 15.283_333_333_333_331),
        (-21.630_890_664_590_066, 7.697_528_956_612_448),
        (-12.702_471_043_387_55, -1.230_890_664_590_062),
        (-5.116_666_666_666_667, 2.25),
    ];
    let row =
        |x: f64| -> Vec<Point3> { profile.iter().map(|&(y, z)| Point3::new(x, y, z)).collect() };
    NurbsSurface::new(
        1,
        3,
        vec![0.0, 0.0, 1.0, 1.0],
        vec![0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0],
        vec![row(-44.03), row(44.03)],
        vec![vec![1.0; 4]; 2],
    )
    .unwrap()
}

/// A tapered wall leaning out from x 41.03 at z 0 to 44.03 at z 6 cuts the
/// scoop and leaves it through the patch's x = 44.03 edge, where the wall's
/// top edge meets the scoop's end profile. A crossing found on that edge
/// line must stay on it: refined in both parameters it drifted 1e-3 inside,
/// and the section's end missed the corner vertex the edges share.
#[test]
fn plane_section_ends_on_the_patch_edge_it_leaves_through() {
    let surface = scoop_surface();
    let normal = Vec3::new(0.894_427_191, 0.0, -0.447_213_595_5);
    let d = normal.dot(Vec3::new(41.03, 0.0, 0.0));
    let curves = intersect_plane_nurbs(&surface, normal, d, 32).unwrap();
    let ends: Vec<Point3> = curves
        .iter()
        .flat_map(|c| [c.points[0].point, c.points[c.points.len() - 1].point])
        .collect();
    assert!(
        ends.iter().any(|p| (p.x() - 44.03).abs() < 1e-9),
        "no section end on the x = 44.03 edge: {ends:?}"
    );
}

/// The scoop against the corner cylinder at (-41, -17): the section climbs
/// to the scoop's top edge at z 15.283. Steps past that edge were clamped
/// back into the patch, and Newton from the clamped state slid down the
/// curve to points already traced, which the march kept: the chain ran up
/// and down its top half three times and its fitted curve strayed 0.3 off
/// both surfaces.
#[test]
fn a_march_reaching_a_patch_edge_does_not_double_back() {
    let scoop = scoop_surface();
    let cylinder = crate::surfaces::CylindricalSurface::with_ref_dir(
        Point3::new(-41.0, -17.0, 6.0),
        Vec3::new(0.0, 0.0, 1.0),
        3.03,
        Vec3::new(0.0, 1.0, 0.0),
    )
    .unwrap()
    .to_nurbs(0.0, 19.25)
    .unwrap();
    let seed =
        refine_ssi_point(&cylinder, &scoop, 0.3575, 0.3439, 0.007_43, 0.115_58, 1e-6).unwrap();
    let chain = march_intersection(&cylinder, &scoop, &seed, 0.01, 1e-6);
    let z: Vec<f64> = chain.iter().map(|p| p.point.z()).collect();
    assert!(
        z.iter().any(|&z| z > 15.2),
        "the march never reached the top edge: {z:?}"
    );
    let rising = z.windows(2).all(|w| w[1] >= w[0] - 1e-9);
    let falling = z.windows(2).all(|w| w[1] <= w[0] + 1e-9);
    assert!(rising || falling, "the section doubles back: z = {z:?}");
}

/// The scoop against the bin pocket's corner cylinder at (41, -17): the
/// scoop's profile bulges past its top edge's line, so the section swings
/// out round the corner and back. The march steps up to 0.87 along it, and
/// later seeds re-trace it with points up to 0.014 off those chords, which
/// the overlap trim read as a second curve over the bulge.
#[test]
fn a_re_traced_section_comes_back_once() {
    use crate::nurbs::projection::project_point_to_curve;
    let scoop = scoop_surface();
    let cylinder = crate::surfaces::CylindricalSurface::with_ref_dir(
        Point3::new(41.0, -17.0, 0.0),
        Vec3::new(0.0, 0.0, 1.0),
        2.55,
        Vec3::new(1.0, 0.0, 0.0),
    )
    .unwrap()
    .to_nurbs(4.7, 22.55)
    .unwrap();
    let curves = intersect_nurbs_nurbs(&cylinder, &scoop, 32, 0.01).unwrap();
    // The scoop crosses the cylinder once on either side of its axis.
    assert_eq!(curves.len(), 2, "{} sections", curves.len());
    for (i, a) in curves.iter().enumerate() {
        for (j, b) in curves.iter().enumerate() {
            if i == j {
                continue;
            }
            let inner = &b.points[1..b.points.len() - 1];
            let shared = inner
                .iter()
                .filter(|p| {
                    project_point_to_curve(&a.curve, p.point, 1e-9)
                        .is_ok_and(|proj| proj.distance < 1e-4)
                })
                .count();
            assert_eq!(
                shared, 0,
                "curve {j} re-traces curve {i} at {shared} points"
            );
        }
    }
}

/// A trace through three points of a unit circle 60 degrees apart: a point
/// on the arc between two of them, off their chord by its sag, lies on the
/// trace; one as far off on the chord's inner side does not.
#[test]
fn a_chord_takes_its_sag_on_the_side_its_arc_bulges() {
    let on = |deg: f64| {
        let a = deg.to_radians();
        IntersectionPoint {
            point: Point3::new(a.cos(), a.sin(), 0.0),
            param1: (0.0, 0.0),
            param2: (0.0, 0.0),
        }
    };
    let trace = vec![vec![on(-60.0), on(0.0), on(60.0)]];
    assert!(near_existing_segment(&trace, &on(30.0), 0.005));
    let mid = Point3::new(0.75, 0.433_012_701_892_219_3, 0.0);
    let inner = IntersectionPoint {
        point: mid + Vec3::new(-0.866_025_403_784_438_6, -0.5, 0.0) * 0.05,
        param1: (0.0, 0.0),
        param2: (0.0, 0.0),
    };
    assert!(!near_existing_segment(&trace, &inner, 0.005));
}

/// A corner cylinder of radius 2.55 and the two strut facets of a lattice
/// piece that meet at a corner poking 0.001 past it (piece 260 of the
/// mitsukude lattice on dividers).
fn corner_poke() -> (NurbsSurface, NurbsSurface, NurbsSurface) {
    let cylinder = crate::surfaces::CylindricalSurface::with_ref_dir(
        Point3::new(-38.0, 38.0, 1.2),
        Vec3::new(0.0, 0.0, 1.0),
        2.55,
        Vec3::new(0.0, 1.0, 0.0),
    )
    .unwrap()
    .to_nurbs(2.25, 34.65)
    .unwrap();
    let corner = Point3::new(
        -40.550_932_465_140_48,
        38.033_270_787_110_52,
        6.774_488_848_638_517,
    );
    let facet = |cps: [[Point3; 2]; 2]| {
        NurbsSurface::new(
            1,
            1,
            vec![0.0, 0.0, 1.0, 1.0],
            vec![0.0, 0.0, 1.0, 1.0],
            cps.iter().map(|row| row.to_vec()).collect(),
            vec![vec![1.0, 1.0], vec![1.0, 1.0]],
        )
        .unwrap()
    };
    let near = Point3::new(
        -39.528_148_878_129_32,
        37.991_952_164_664_326,
        7.400_752_775_094_245_5,
    );
    let corner_facet = facet([
        [
            Point3::new(
                -40.347_425_683_831_744,
                38.364_140_220_280_99,
                6.507_244_192_471_161,
            ),
            corner,
        ],
        [
            Point3::new(
                -39.374_137_980_938_84,
                38.242_315_874_207_48,
                7.198_525_521_671_666,
            ),
            near,
        ],
    ]);
    let side_facet = facet([
        [
            corner,
            Point3::new(
                -40.708_292_298_294_65,
                37.657_112_719_831_07,
                7.055_014_719_355_846,
            ),
        ],
        [
            near,
            Point3::new(
                -39.646_445_904_048_82,
                37.709_272_227_529_18,
                7.611_577_043_614_155,
            ),
        ],
    ]);
    (cylinder, corner_facet, side_facet)
}

/// Whether `p` lies on the radius-2.55 cylinder of [`corner_poke`].
fn on_corner_cylinder(p: Point3) -> bool {
    ((p.x() + 38.0).hypot(p.y() - 38.0) - 2.55).abs() < 1e-6
}

/// Distance from `p` to the segment `a b`.
fn off_segment(p: Point3, a: Point3, b: Point3) -> f64 {
    let ab = b - a;
    let f = ((p - a).dot(ab) / ab.dot(ab)).clamp(0.0, 1.0);
    (p - (a + ab * f)).length()
}

/// The corner facet meets the cylinder in a section 0.003 long between its
/// two edges at the corner, closer to it than the march keeps from a patch's
/// edges: neither crossing takes a step, and the curve comes back joining
/// them.
#[test]
fn a_section_within_a_patch_corner_comes_back() {
    let (cylinder, facet, _) = corner_poke();
    let curves = intersect_nurbs_nurbs(&cylinder, &facet, 20, 0.01).unwrap();
    assert_eq!(curves.len(), 1, "{} sections", curves.len());
    let curve = &curves[0].curve;
    let (t0, t1) = curve.domain();
    let cps = facet.control_points();
    let ends = [curve.evaluate(t0), curve.evaluate(t1)];
    for (edge, (a, b)) in [(cps[0][0], cps[0][1]), (cps[0][1], cps[1][1])]
        .into_iter()
        .enumerate()
    {
        assert!(
            ends.iter().any(|&e| off_segment(e, a, b) < 1e-6),
            "no end on edge {edge}: {ends:?}"
        );
    }
    for k in 0..=8 {
        let p = curve.evaluate((t1 - t0).mul_add(f64::from(k) / 8.0, t0));
        assert!(on_corner_cylinder(p), "{p:?} off the cylinder");
        let foot = crate::nurbs::projection::project_point_to_surface(&facet, p, 1e-12).unwrap();
        assert!(
            foot.distance < 1e-6,
            "{p:?} {} off the facet",
            foot.distance
        );
    }
}

/// The side facet's section leaves it 0.0011 from the corner, through its
/// edge along the corner facet: the march stalls a margin inside the patch
/// with both of its parameters near their edges, and the end lies on that
/// one edge rather than 0.0004 inside it.
#[test]
fn a_section_leaving_beside_a_patch_corner_ends_on_its_edge() {
    let (cylinder, _, facet) = corner_poke();
    let curves = intersect_nurbs_nurbs(&cylinder, &facet, 20, 0.01).unwrap();
    assert_eq!(curves.len(), 1, "{} sections", curves.len());
    let curve = &curves[0].curve;
    let (t0, t1) = curve.domain();
    let cps = facet.control_points();
    let (a, b) = (cps[0][0], cps[1][0]);
    let end = [curve.evaluate(t0), curve.evaluate(t1)]
        .into_iter()
        .min_by(|p, q| off_segment(*p, a, b).total_cmp(&off_segment(*q, a, b)))
        .unwrap();
    assert!(
        off_segment(end, a, b) < 1e-7,
        "{end:?} stands {} off the edge",
        off_segment(end, a, b)
    );
    assert!(on_corner_cylinder(end), "{end:?} off the cylinder");
}

/// The farthest a section's curve strays from `off`, sampled densely.
fn farthest(curves: &[IntersectionCurve], off: impl Fn(Point3) -> f64) -> f64 {
    let mut far = 0.0_f64;
    for ic in curves {
        let (t0, t1) = ic.curve.domain();
        for i in 0..=2000 {
            let t = t0 + (t1 - t0) * f64::from(i) / 2000.0;
            far = far.max(off(ic.curve.evaluate(t)));
        }
    }
    far
}

/// A tilted plane cuts the rational cylinder in an ellipse; between its
/// marched points the fitted curve stays on the plane and the cylinder.
#[test]
fn a_plane_section_holds_its_tolerance_between_points() {
    let cylinder = cylinder_nurbs_surface();
    let n = Vec3::new(0.3, 0.1, 1.0);
    let d = n.z();
    let curves = intersect_plane_nurbs(&cylinder, n, d, 64).unwrap();
    assert_eq!(curves.len(), 1);
    let far = farthest(&curves, |p| {
        let plane = (n.dot(Vec3::new(p.x(), p.y(), p.z())) - d).abs() / n.length();
        plane.max((p.x().hypot(p.y()) - 1.0).abs())
    });
    assert!(far < 1e-6, "section strays {far:.2e}");
}

/// Crossing cylinders of radii 1 and 0.6 meet in two loops; between their
/// marched points the fitted curves stay on both cylinders.
#[test]
fn a_marched_section_holds_its_tolerance_between_points() {
    let cyl_z = cylinder_at(0.0, 0.0, 1.0, -2.0, 2.0);
    let cyl_x = cylinder_along_x(0.0, 0.0, 0.6, -2.0, 2.0);
    let curves = intersect_nurbs_nurbs(&cyl_z, &cyl_x, 20, 0.0).unwrap();
    assert_eq!(curves.len(), 2);
    let far = farthest(&curves, |p| {
        (p.x().hypot(p.y()) - 1.0)
            .abs()
            .max((p.y().hypot(p.z()) - 0.6).abs())
    });
    assert!(far < 5e-6, "section strays {far:.2e}");
}

/// The plane `v − u = 0.9` crosses the cubic sheet `z = (x − 0.05)³` from
/// one corner edge to the other, through the sheet's inflection: the line
/// between the two grid crossings meets the section at its middle and
/// strays 4.7e-5 at its quarter marks, which the fit must catch.
#[test]
fn a_plane_section_through_an_inflection_holds_its_tolerance() {
    let a = 0.05_f64;
    let z = [
        -a.powi(3),
        a * a - a.powi(3),
        2.0 * a * a - a.powi(3) - a,
        (1.0 - a).powi(3),
    ];
    let cps = (0..4)
        .map(|i| {
            let x = f64::from(i) / 3.0;
            vec![
                Point3::new(x, 0.0, z[i as usize]),
                Point3::new(x, 1.0, z[i as usize]),
            ]
        })
        .collect();
    let sheet = NurbsSurface::new(
        3,
        1,
        vec![0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0],
        vec![0.0, 0.0, 1.0, 1.0],
        cps,
        vec![vec![1.0; 2]; 4],
    )
    .unwrap();
    let n = Vec3::new(-1.0, 1.0, 0.0);
    let curves = intersect_plane_nurbs(&sheet, n, 0.9, 10).unwrap();
    assert_eq!(curves.len(), 1);
    let far = farthest(&curves, |p| {
        let plane = (n.dot(Vec3::new(p.x(), p.y(), p.z())) - 0.9).abs() / n.length();
        plane.max((p.z() - (p.x() - a).powi(3)).abs())
    });
    assert!(far < 1e-6, "section strays {far:.2e}");
}

/// A level plane cuts the dome in a loop inside its patch, which comes back
/// closed: its curve ends where it starts.
#[test]
fn a_plane_section_loop_comes_back_closed() {
    let dome = dome_surface();
    let curves = intersect_plane_nurbs(&dome, Vec3::new(0.0, 0.0, 1.0), 0.0, 32).unwrap();
    assert_eq!(curves.len(), 1);
    let curve = &curves[0].curve;
    let (t0, t1) = curve.domain();
    let gap = (curve.evaluate(t1) - curve.evaluate(t0)).length();
    assert!(gap < 1e-9, "loop left open by {gap:.2e}");
}

/// Planes at heights 1e-8 and 1e-14 cut the paraboloid `z = x² + y²` in
/// loops of radius 1e-4 and 1e-7 round its vertex, far inside one cell of
/// the sampling grid, which no grid edge crosses. Each comes back closed,
/// on the plane and the paraboloid.
#[test]
fn a_plane_section_loop_inside_one_grid_cell_is_found() {
    let row = |x: f64, z: [f64; 3]| {
        vec![
            Point3::new(x, -1.0, z[0]),
            Point3::new(x, 0.0, z[1]),
            Point3::new(x, 1.0, z[2]),
        ]
    };
    let bowl = NurbsSurface::new(
        2,
        2,
        vec![0.0, 0.0, 0.0, 1.0, 1.0, 1.0],
        vec![0.0, 0.0, 0.0, 1.0, 1.0, 1.0],
        vec![
            row(-1.0, [2.0, 0.0, 2.0]),
            row(0.0, [0.0, -2.0, 0.0]),
            row(1.0, [2.0, 0.0, 2.0]),
        ],
        vec![vec![1.0; 3]; 3],
    )
    .unwrap();
    for radius in [1e-4, 1e-7] {
        let height = radius * radius;
        let curves = intersect_plane_nurbs(&bowl, Vec3::new(0.0, 0.0, 1.0), height, 64).unwrap();
        assert_eq!(curves.len(), 1, "radius {radius}");
        let curve = &curves[0].curve;
        let (t0, t1) = curve.domain();
        assert!((curve.evaluate(t1) - curve.evaluate(t0)).length() < 1e-3 * radius);
        let far = farthest(&curves, |p| {
            let rim = (p.x().hypot(p.y()) - radius).abs();
            rim.max((p.z() - height).abs())
        });
        // Evaluating the bowl rounds its height by about 2e-16, which
        // moves where a loop of radius r crosses the plane by 1e-16 / r.
        assert!(
            far < 1e-3 * radius + 1e-16 / radius,
            "radius {radius}: loop strays {far:.2e}"
        );
    }
}

/// The Bézier patch of degree `p` on the unit square whose coordinates are
/// the polynomials with terms `(a, b, c)`, each `c·uᵃvᵇ`.
fn polynomial_patch(p: u32, coords: [&[(u32, u32, f64)]; 3]) -> NurbsSurface {
    let choose = |n: u32, k: u32| {
        if k > n {
            return 0.0;
        }
        (0..k).fold(1.0, |acc, m| acc * f64::from(n - m) / f64::from(m + 1))
    };
    let coefficient = |terms: &[(u32, u32, f64)], i: u32, j: u32| {
        terms
            .iter()
            .map(|&(a, b, c)| c * choose(i, a) / choose(p, a) * choose(j, b) / choose(p, b))
            .sum::<f64>()
    };
    let order = usize::try_from(p).unwrap() + 1;
    let knots = [vec![0.0; order], vec![1.0; order]].concat();
    let control = (0..=p)
        .map(|i| {
            (0..=p)
                .map(|j| {
                    let [x, y, z] = coords.map(|terms| coefficient(terms, i, j));
                    Point3::new(x, y, z)
                })
                .collect()
        })
        .collect();
    NurbsSurface::new(
        order - 1,
        order - 1,
        knots.clone(),
        knots,
        control,
        vec![vec![1.0; order]; order],
    )
    .unwrap()
}

/// A loop inside a cell along the patch's edge, whose least grid distance
/// lies on that edge, is found.
#[test]
fn a_tiny_loop_beside_the_patch_edge_is_found() {
    let a = 0.25_f64 / 63.0;
    let bowl = polynomial_patch(
        2,
        [
            &[(1, 0, 1.0)],
            &[(0, 1, 1.0)],
            &[
                (0, 0, a.mul_add(a, 0.25)),
                (1, 0, -2.0 * a),
                (2, 0, 1.0),
                (0, 1, -1.0),
                (0, 2, 1.0),
            ],
        ],
    );
    let curves = intersect_plane_nurbs(&bowl, Vec3::new(0.0, 0.0, 1.0), 1e-8, 64).unwrap();
    assert_eq!(curves.len(), 1);
    let far = farthest(&curves, |p| {
        let rim = ((p.x() - a).hypot(p.y() - 0.5) - 1e-4).abs();
        rim.max((p.z() - 1e-8).abs())
    });
    assert!(far < 1e-7, "loop strays {far:.2e}");
}

/// A long loop along a valley running across the grid's lines is found
/// from a node far along the valley from its lowest point.
#[test]
fn a_long_tiny_loop_across_the_grid_is_found() {
    // (v − 0.5 − 0.4(u − 0.5))² + 1e-4·(u − 0.5)²
    let valley = polynomial_patch(
        2,
        [
            &[(1, 0, 1.0)],
            &[(0, 1, 1.0)],
            &[
                (0, 0, 0.09 + 0.25e-4),
                (1, 0, 0.24 - 1e-4),
                (2, 0, 0.16 + 1e-4),
                (0, 1, -0.6),
                (0, 2, 1.0),
                (1, 1, -0.8),
            ],
        ],
    );
    let height = (1e-3_f64 / 63.0).powi(2);
    let curves = intersect_plane_nurbs(&valley, Vec3::new(0.0, 0.0, 1.0), height, 64).unwrap();
    assert_eq!(curves.len(), 1);
    // The loop is `q = 1` for `q = (w² + 1e-4·s²) / height`; a point's
    // distance from it is about `|q − 1| / |∇q|`.
    let far = farthest(&curves, |p| {
        let (s, w) = (p.x() - 0.5, p.y() - 0.5 - 0.4 * (p.x() - 0.5));
        let q = (w * w + 1e-4 * s * s) / height;
        let grad = Vec3::new(2e-4 * s - 0.8 * w, 2.0 * w, 0.0) * (1.0 / height);
        ((q - 1.0).abs() / grad.length()).max((p.z() - height).abs())
    });
    assert!(far < 1e-7, "loop strays {far:.2e}");
    for point in &curves[0].points {
        let p = point.point;
        let (s, w) = (p.x() - 0.5, p.y() - 0.5 - 0.4 * (p.x() - 0.5));
        let rim = ((w * w + 1e-4 * s * s) / height).sqrt() - 1.0;
        assert!(
            rim.abs() < 1e-6,
            "a point strays {rim:.2e} of the loop's size"
        );
    }
}

/// A plane just above the flat bottom of `z = x⁴ + y⁴` cuts a loop where
/// the surface runs so nearly along the plane that points twice the loop's
/// size lie within 1e-12 of it; the loop found is the true one.
#[test]
fn a_tiny_loop_on_a_flat_bottom_is_its_true_size() {
    let quartic = polynomial_patch(
        4,
        [
            &[(0, 0, -1.0), (1, 0, 2.0)],
            &[(0, 0, -1.0), (0, 1, 2.0)],
            &[
                (0, 0, 2.0),
                (1, 0, -8.0),
                (2, 0, 24.0),
                (3, 0, -32.0),
                (4, 0, 16.0),
                (0, 1, -8.0),
                (0, 2, 24.0),
                (0, 3, -32.0),
                (0, 4, 16.0),
            ],
        ],
    );
    let curves = intersect_plane_nurbs(&quartic, Vec3::new(0.0, 0.0, 1.0), 1e-14, 10).unwrap();
    assert_eq!(curves.len(), 1);
    // Evaluating the patch rounds its height by about 1e-16, a hundredth
    // of the plane's, which leaves the loop's size good to about 0.25%.
    let far = farthest(&curves, |p| {
        (((p.x().powi(4) + p.y().powi(4)) / 1e-14).powf(0.25) - 1.0).abs()
    });
    assert!(far < 1e-2, "loop strays {far:.2e} of its size");
}

/// Within 1e-12 of a plane just above the flat bottom of `z = x⁴ + y⁴`
/// lie points over twice as far out as the section; refining one of them
/// still finds the section.
#[test]
fn refining_onto_a_nearly_tangent_plane_reaches_the_section() {
    let quartic = polynomial_patch(
        4,
        [
            &[(0, 0, -1.0), (1, 0, 2.0)],
            &[(0, 0, -1.0), (0, 1, 2.0)],
            &[
                (0, 0, 2.0),
                (1, 0, -8.0),
                (2, 0, 24.0),
                (3, 0, -32.0),
                (4, 0, 16.0),
                (0, 1, -8.0),
                (0, 2, 24.0),
                (0, 3, -32.0),
                (0, 4, 16.0),
            ],
        ],
    );
    let guess = 0.5 * (1.0 + 5.6e-4);
    let up = Vec3::new(0.0, 0.0, 1.0);
    let wide = quartic.evaluate(guess, guess);
    assert!((wide.z() - 1e-14).abs() < 1e-12);
    let point =
        plane::refine_plane_surface_point(&quartic, up, 1e-14, (guess, guess), None).unwrap();
    let (x, y) = (point.point.x(), point.point.y());
    let size = ((x.powi(4) + y.powi(4)) / 1e-14).powf(0.25);
    assert!(
        (size - 1.0).abs() < 1e-2,
        "refined to {size:.3} of the section's size"
    );
}

/// The plane `y = 0` cuts the rational cylinder in two lines: `x = 1` on its
/// seam, where both edge columns of the grid lie on the plane, and `x = −1`,
/// along a column of nodes when the grid has one at a half turn. Both come
/// back whole, on the plane and the cylinder.
#[test]
fn a_plane_section_along_the_seam_and_a_column_of_nodes_is_found() {
    let cylinder = cylinder_nurbs_surface();
    for samples in [64, 33] {
        let curves =
            intersect_plane_nurbs(&cylinder, Vec3::new(0.0, 1.0, 0.0), 0.0, samples).unwrap();
        assert_eq!(curves.len(), 2, "{samples} samples");
        let mut sides: Vec<f64> = Vec::new();
        for ic in &curves {
            let (t0, t1) = ic.curve.domain();
            let (a, b) = (ic.curve.evaluate(t0), ic.curve.evaluate(t1));
            assert!(
                (a.z() - b.z()).abs() > 2.0 - 1e-9,
                "{samples} samples: a line cut short"
            );
            let x = a.x();
            let far = farthest(std::slice::from_ref(ic), |p| {
                p.y().abs().max((p.x() - x).abs())
            });
            assert!(far < 1e-9, "{samples} samples: line strays {far:.2e}");
            sides.push(x);
        }
        sides.sort_by(f64::total_cmp);
        assert!((sides[0] + 1.0).abs() < 1e-9 && (sides[1] - 1.0).abs() < 1e-9);
    }
}

/// The plane `z = 0` cuts a rational torus with radii 3 and 1 in its inner
/// and outer equators. The outer lies on the tube's seam, whose last row of
/// control points sits at `sin 2π`, 2.4e-16 below the plane, so that the
/// grid's two edge rows read 0 and −2.4e-16; both circles come back whole.
#[test]
fn a_plane_section_along_a_seam_within_rounding_of_it_is_found() {
    let w = std::f64::consts::FRAC_1_SQRT_2;
    let ring = [
        (1.0, 0.0, 1.0),
        (1.0, 1.0, w),
        (0.0, 1.0, 1.0),
        (-1.0, 1.0, w),
        (-1.0, std::f64::consts::PI.sin(), 1.0),
        (-1.0, -1.0, w),
        (0.0, -1.0, 1.0),
        (1.0, -1.0, w),
        (1.0, std::f64::consts::TAU.sin(), 1.0),
    ];
    let cps = ring
        .iter()
        .map(|&(sx, sy, _)| {
            ring.iter()
                .map(|&(a, b, _)| Point3::new((3.0 + a) * sx, (3.0 + a) * sy, b))
                .collect()
        })
        .collect();
    let weights = ring
        .iter()
        .map(|&(_, _, wu)| ring.iter().map(|&(_, _, wv)| wu * wv).collect())
        .collect();
    let knots = vec![
        0.0, 0.0, 0.0, 0.25, 0.25, 0.5, 0.5, 0.75, 0.75, 1.0, 1.0, 1.0,
    ];
    let torus = NurbsSurface::new(2, 2, knots.clone(), knots, cps, weights).unwrap();
    let curves = intersect_plane_nurbs(&torus, Vec3::new(0.0, 0.0, 1.0), 0.0, 64).unwrap();
    assert_eq!(curves.len(), 2);
    let mut radii: Vec<f64> = Vec::new();
    for ic in &curves {
        let (t0, t1) = ic.curve.domain();
        let start = ic.curve.evaluate(t0);
        assert!((ic.curve.evaluate(t1) - start).length() < 1e-9);
        let radius = start.x().hypot(start.y()).round();
        let far = farthest(std::slice::from_ref(ic), |p| {
            p.z().abs().max((p.x().hypot(p.y()) - radius).abs())
        });
        assert!(far < 1e-7, "circle of radius {radius} strays {far:.2e}");
        radii.push(radius);
    }
    radii.sort_by(f64::total_cmp);
    assert_eq!(radii, [2.0, 4.0]);
}

/// The plane `z = 0` touches `z = (x² + y² − 1)²` along the unit circle and
/// the trough `z = x²` along the line `x = 0`, where neither surface's
/// height crosses the plane but rounds a few 1e-16 either side of it. Each
/// contact comes back whole: the circle closed, the line from edge to edge.
#[test]
fn a_plane_touching_a_surface_along_a_curve_traces_it() {
    let x = [(0, 0, -2.0), (1, 0, 4.0)];
    let y = [(0, 0, -2.0), (0, 1, 4.0)];
    let quartic = polynomial_patch(
        4,
        [
            &x,
            &y,
            &[
                (0, 0, 49.0),
                (0, 1, -224.0),
                (0, 2, 480.0),
                (0, 3, -512.0),
                (0, 4, 256.0),
                (1, 0, -224.0),
                (1, 1, 512.0),
                (1, 2, -512.0),
                (2, 0, 480.0),
                (2, 1, -512.0),
                (2, 2, 512.0),
                (3, 0, -512.0),
                (4, 0, 256.0),
            ],
        ],
    );
    let trough = polynomial_patch(2, [&x, &y, &[(0, 0, 4.0), (1, 0, -16.0), (2, 0, 16.0)]]);
    let up = Vec3::new(0.0, 0.0, 1.0);
    for samples in [64, 63] {
        let circle = intersect_plane_nurbs(&quartic, up, 0.0, samples).unwrap();
        assert_eq!(circle.len(), 1, "{samples} samples");
        let (t0, t1) = circle[0].curve.domain();
        assert!((circle[0].curve.evaluate(t0) - circle[0].curve.evaluate(t1)).length() < 1e-9);
        let far = farthest(&circle, |p| {
            (p.x().hypot(p.y()) - 1.0).abs().max(p.z().abs())
        });
        assert!(far < 1e-6, "{samples} samples: the circle strays {far:.2e}");
        let line = intersect_plane_nurbs(&trough, up, 0.0, samples).unwrap();
        assert_eq!(line.len(), 1, "{samples} samples");
        let (t0, t1) = line[0].curve.domain();
        let (a, b) = (line[0].curve.evaluate(t0), line[0].curve.evaluate(t1));
        assert!((a.y().abs() - 2.0).abs() < 1e-12 && (b.y().abs() - 2.0).abs() < 1e-12);
        let far = farthest(&line, |p| p.x().abs().max(p.z().abs()));
        assert!(far < 1e-9, "{samples} samples: the line strays {far:.2e}");
    }
}

/// The plane `z = 0` touches the paraboloid `z = x² + y²` at its vertex and
/// `z = (x² − 1)² + (y² − 1)²` at `(±1, ±1)`, curving away both ways at
/// each: every touch comes back as a curve that stays on its point, with 64
/// samples and with 33, which puts the touches on grid nodes, and whichever
/// way the plane faces.
#[test]
fn a_plane_touching_a_surface_at_points_returns_them() {
    let x = [(0, 0, -2.0), (1, 0, 4.0)];
    let y = [(0, 0, -2.0), (0, 1, 4.0)];
    let bowl = polynomial_patch(
        2,
        [
            &x,
            &y,
            &[
                (0, 0, 8.0),
                (0, 1, -16.0),
                (0, 2, 16.0),
                (1, 0, -16.0),
                (2, 0, 16.0),
            ],
        ],
    );
    let dimples = polynomial_patch(
        4,
        [
            &x,
            &y,
            &[
                (0, 0, 18.0),
                (0, 1, -96.0),
                (0, 2, 352.0),
                (0, 3, -512.0),
                (0, 4, 256.0),
                (1, 0, -96.0),
                (2, 0, 352.0),
                (3, 0, -512.0),
                (4, 0, 256.0),
            ],
        ],
    );
    let touches = [
        (bowl, vec![(0.0, 0.0)]),
        (
            dimples,
            vec![(-1.0, -1.0), (-1.0, 1.0), (1.0, -1.0), (1.0, 1.0)],
        ),
    ];
    let (up, down) = (Vec3::new(0.0, 0.0, 1.0), Vec3::new(0.0, 0.0, -1.0));
    for ((surface, expected), normal, samples) in touches
        .iter()
        .flat_map(|t| [(t, up, 64), (t, up, 33), (t, down, 64), (t, down, 33)])
    {
        let curves = intersect_plane_nurbs(surface, normal, 0.0, samples).unwrap();
        let mut found: Vec<(f64, f64)> = curves
            .iter()
            .map(|ic| {
                assert_eq!(ic.points.len(), 1);
                let (t0, t1) = ic.curve.domain();
                let p = ic.points[0].point;
                assert!((ic.curve.evaluate(t0) - p).length() < 1e-15);
                assert!((ic.curve.evaluate(t1) - p).length() < 1e-15);
                assert!(p.z().abs() < 1e-14);
                (p.x(), p.y())
            })
            .collect();
        found.sort_by(|a, b| a.0.total_cmp(&b.0).then(a.1.total_cmp(&b.1)));
        assert_eq!(
            found.len(),
            expected.len(),
            "{samples} samples, normal {normal:?}"
        );
        for (f, e) in found.iter().zip(expected) {
            assert!((f.0 - e.0).hypot(f.1 - e.1) < 1e-9, "{f:?} for {e:?}");
        }
    }
}

/// The plane `z = 0` meets the saddle `z = xy` in the two axes, which cross
/// at the origin, inside a cell of the grid. The section comes back as four
/// curves, each from the crossing out to the patch's edge along an axis.
#[test]
fn a_plane_section_crossing_itself_ends_its_curves_at_the_crossing() {
    let saddle = polynomial_patch(
        1,
        [
            &[(0, 0, -1.0), (1, 0, 2.0)],
            &[(0, 0, -1.0), (0, 1, 2.0)],
            &[(0, 0, 1.0), (1, 0, -2.0), (0, 1, -2.0), (1, 1, 4.0)],
        ],
    );
    let curves = intersect_plane_nurbs(&saddle, Vec3::new(0.0, 0.0, 1.0), 0.0, 64).unwrap();
    assert_eq!(curves.len(), 4);
    for ic in &curves {
        let (t0, t1) = ic.curve.domain();
        let (a, b) = (ic.curve.evaluate(t0), ic.curve.evaluate(t1));
        let (centre, rim) = if a.x().hypot(a.y()) < b.x().hypot(b.y()) {
            (a, b)
        } else {
            (b, a)
        };
        assert!(
            centre.x().hypot(centre.y()) < 1e-12,
            "a curve ends {centre:?}"
        );
        assert!((rim.x().abs().max(rim.y().abs()) - 1.0).abs() < 1e-12);
        let far = farthest(std::slice::from_ref(ic), |p| {
            p.x().abs().min(p.y().abs()).max(p.z().abs())
        });
        assert!(far < 1e-7, "a curve strays {far:.2e} off the axes");
    }
}

/// The trough `z = x²` of the test above, with its parameter across the
/// floor running over a millionth where the one along it runs over one,
/// still touches `z = 0` along the whole line: steps along the floor are
/// measured against the domain, not in its parameters.
#[test]
fn a_plane_touching_a_squeezed_surface_along_a_line_traces_it_whole() {
    let trough = polynomial_patch(
        2,
        [
            &[(0, 0, -2.0), (1, 0, 4.0)],
            &[(0, 0, -2.0), (0, 1, 4.0)],
            &[(0, 0, 4.0), (1, 0, -16.0), (2, 0, 16.0)],
        ],
    );
    let squeezed = NurbsSurface::new(
        2,
        2,
        trough.knots_u().iter().map(|k| k * 1e-6).collect(),
        trough.knots_v().to_vec(),
        trough.control_points().to_vec(),
        trough.weights().to_vec(),
    )
    .unwrap();
    let line = intersect_plane_nurbs(&squeezed, Vec3::new(0.0, 0.0, 1.0), 0.0, 64).unwrap();
    assert_eq!(line.len(), 1);
    let (t0, t1) = line[0].curve.domain();
    let (a, b) = (line[0].curve.evaluate(t0), line[0].curve.evaluate(t1));
    assert!(
        (a.y().abs() - 2.0).abs() < 1e-12 && (b.y().abs() - 2.0).abs() < 1e-12,
        "the line runs from {a:?} to {b:?}"
    );
    let far = farthest(&line, |p| p.x().abs().max(p.z().abs()));
    assert!(far < 1e-9, "the line strays {far:.2e}");
}

/// The plane `z = 0` touches `z = (y − x²)²` along the parabola `y = x²`,
/// which leaves the patch at its corners `(0, 0)` and `(1, 1)`, where the
/// traced curve ends: its last step settles along the patch's edge.
#[test]
fn a_plane_touching_a_surface_along_a_curve_traces_it_to_the_patch_edge() {
    let gutter = polynomial_patch(
        4,
        [
            &[(1, 0, 1.0)],
            &[(0, 1, 1.0)],
            &[(0, 2, 1.0), (2, 1, -2.0), (4, 0, 1.0)],
        ],
    );
    let curves = intersect_plane_nurbs(&gutter, Vec3::new(0.0, 0.0, 1.0), 0.0, 64).unwrap();
    assert_eq!(curves.len(), 1);
    let (t0, t1) = curves[0].curve.domain();
    let mut ends = [t0, t1].map(|t| curves[0].curve.evaluate(t));
    ends.sort_by(|p, q| p.x().total_cmp(&q.x()));
    assert!(
        (ends[0] - Point3::new(0.0, 0.0, 0.0)).length() < 1e-9
            && (ends[1] - Point3::new(1.0, 1.0, 0.0)).length() < 1e-9,
        "the curve runs from {:?} to {:?}",
        ends[0],
        ends[1]
    );
    let far = farthest(&curves, |p| (p.y() - p.x() * p.x()).abs().max(p.z().abs()));
    assert!(far < 1e-7, "the curve strays {far:.2e}");
}

/// A sheet 1e12 along `x` from the origin, rising from −0.001 to 0.001
/// across it, crosses `z = 0` along its middle, where the section's points
/// lie: coordinates the plane's normal does not weigh leave the plane's
/// distance unrounded.
#[test]
fn a_plane_section_far_from_the_origin_along_the_plane_is_found() {
    let at = |x: f64, y: f64, z: f64| Point3::new(x, y, z);
    let sheet = NurbsSurface::new(
        1,
        1,
        vec![0.0, 0.0, 1.0, 1.0],
        vec![0.0, 0.0, 1.0, 1.0],
        vec![
            vec![at(1e12, 0.0, -0.001), at(1e12, 1.0, -0.001)],
            vec![at(1e12 + 1.0, 0.0, 0.001), at(1e12 + 1.0, 1.0, 0.001)],
        ],
        vec![vec![1.0; 2]; 2],
    )
    .unwrap();
    let curves = intersect_plane_nurbs(&sheet, Vec3::new(0.0, 0.0, 1.0), 0.0, 32).unwrap();
    assert_eq!(curves.len(), 1);
    for point in &curves[0].points {
        let p = point.point;
        assert!(
            p.z().abs() < 1e-12 && (p.x() - (1e12 + 0.5)).abs() < 1e-3,
            "{p:?}"
        );
    }
}
