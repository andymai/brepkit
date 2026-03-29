//! Surface-surface and curve-surface intersection routines.
//!
//! These are the geometric foundations for boolean operations on NURBS solids.
//!
//! ## Algorithms
//!
//! - **Plane-NURBS**: Sample the NURBS surface on a grid, find sign changes of the
//!   signed distance to the plane, trace zero-crossings via linear interpolation,
//!   then refine with Newton iteration.
//! - **NURBS-NURBS**: Subdivision + marching method in (u1,v1,u2,v2) parameter space.
//! - **Line-surface**: Newton iteration from grid-based seed points.

#![allow(
    clippy::many_single_char_names,
    clippy::similar_names,
    clippy::suboptimal_flops,
    clippy::needless_range_loop,
    clippy::cast_precision_loss,
    clippy::doc_markdown,
    clippy::missing_const_for_fn,
    clippy::manual_let_else
)]

mod chaining;
mod curve_surface;
mod line;
mod plane;
mod surface_marching;
mod surface_seeding;

use crate::nurbs::curve::NurbsCurve;
use crate::vec::Point3;

// Re-export all public items.
pub use chaining::{chain_intersection_points, estimate_chain_threshold};
pub use curve_surface::{CurveSurfaceHit, intersect_curve_surface};
pub use line::intersect_line_nurbs;
pub use plane::intersect_plane_nurbs;
pub use surface_seeding::intersect_nurbs_nurbs;

/// Maximum work-queue entries for the branch-aware SSI marcher.
const MAX_QUEUE_SIZE: usize = 100;

/// Maximum traced curve segments before stopping branch exploration.
const MAX_SEGMENTS: usize = 50;

/// Maximum branch points detected per march direction.
const MAX_BRANCHES_PER_DIRECTION: usize = 10;

/// Maximum iterations for Newton-type solvers.
///
/// 20 iterations is sufficient for quadratic convergence from reasonable seeds
/// (quadratic convergence achieves ~1e-12 in ~6 iterations from a 1e-1 seed).
/// The limit is generous to handle near-singular cases where convergence slows.
const MAX_NEWTON_ITER: usize = 20;

/// A point on an intersection curve, with parameter values on both surfaces.
#[derive(Debug, Clone, Copy)]
pub struct IntersectionPoint {
    /// 3D position of the intersection.
    pub point: Point3,
    /// Parameter on the first surface (u1, v1) or the curve parameter.
    pub param1: (f64, f64),
    /// Parameter on the second surface (u2, v2).
    pub param2: (f64, f64),
}

/// Result of a surface-surface intersection: a list of intersection curves.
#[derive(Debug, Clone)]
pub struct IntersectionCurve {
    /// The 3D intersection curve as a NURBS.
    pub curve: NurbsCurve,
    /// Sampled points along the curve with parameter values.
    pub points: Vec<IntersectionPoint>,
}

#[cfg(test)]
mod tests {
    #![allow(clippy::unwrap_used, clippy::expect_used)]

    use crate::nurbs::surface::NurbsSurface;
    use crate::vec::{Point3, Vec3};

    use super::surface_marching::march_intersection;
    use super::surface_marching::{near_existing_segment, second_order_tangent};
    use super::surface_seeding::{
        find_ssi_seeds_grid, find_ssi_seeds_subdivision, refine_ssi_point,
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
                let signed_dist =
                    Vec3::new(pt.point.x(), pt.point.y(), pt.point.z()).dot(normal) - d;
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
}
