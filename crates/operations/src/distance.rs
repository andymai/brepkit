//! Distance measurement between shapes.
//!
//! Computes minimum distance between solids and point-to-solid distance.
//! Supports planar, NURBS, and analytic (cylinder, cone, sphere, torus) faces
//! with BVH spatial acceleration.

#![allow(
    clippy::many_single_char_names,
    clippy::similar_names,
    clippy::suboptimal_flops,
    clippy::needless_range_loop,
    clippy::cast_precision_loss,
    clippy::doc_markdown,
    clippy::module_name_repetitions,
    clippy::cast_sign_loss,
    clippy::cast_possible_truncation,
    clippy::manual_let_else,
    clippy::needless_pass_by_value,
    clippy::imprecise_flops
)]

use brepkit_math::vec::{Point3, Vec3};
use brepkit_topology::Topology;
use brepkit_topology::face::FaceId;
use brepkit_topology::solid::SolidId;

/// Result of a distance computation.
#[derive(Debug, Clone)]
pub struct DistanceResult {
    /// The minimum distance found.
    pub distance: f64,
    /// The closest point on the first shape.
    pub point_a: Point3,
    /// The closest point on the second shape.
    pub point_b: Point3,
}

/// Compute the minimum distance from a point to a solid.
///
/// Uses BVH over face AABBs for acceleration. Dispatches per face type:
/// planar (point-to-polygon), NURBS (Newton projection), and analytic
/// (closed-form for cylinder/cone/sphere/torus).
///
/// # Errors
///
/// Returns an error if the solid is invalid.
pub fn point_to_solid_distance(
    topo: &Topology,
    point: Point3,
    solid: SolidId,
) -> Result<DistanceResult, crate::OperationsError> {
    // Every shell, each face clipped to its trim.
    let result = brepkit_check::distance::point_to_solid(topo, point, solid)?;
    Ok(DistanceResult {
        distance: result.distance,
        point_a: result.point_a,
        point_b: result.point_b,
    })
}

/// Compute the minimum distance between two solids.
///
/// Checks vertices of each solid against faces of the other, with
/// BVH acceleration. Also checks edge-to-edge distances for the
/// closest vertex pairs.
///
/// # Errors
///
/// Returns an error if either solid is invalid.
pub fn solid_to_solid_distance(
    topo: &Topology,
    solid_a: SolidId,
    solid_b: SolidId,
) -> Result<DistanceResult, crate::OperationsError> {
    // Every shell of both, each face clipped to its trim.
    let result = brepkit_check::distance::solid_to_solid(topo, solid_a, solid_b)?;
    Ok(DistanceResult {
        distance: result.distance,
        point_a: result.point_a,
        point_b: result.point_b,
    })
}

/// Compute the minimum distance from a point to a face.
///
/// # Errors
///
/// Returns an error if the face lookup fails.
pub fn point_to_face(
    topo: &Topology,
    point: Point3,
    face_id: FaceId,
) -> Result<DistanceResult, crate::OperationsError> {
    if let Some((dist, closest)) = point_to_face_distance(topo, point, face_id)? {
        Ok(DistanceResult {
            distance: dist,
            point_a: point,
            point_b: closest,
        })
    } else {
        // Fallback: distance to closest wire vertex
        let face = topo.face(face_id)?;
        let wire = topo.wire(face.outer_wire())?;
        let mut best = f64::INFINITY;
        let mut best_pt = point;
        for oe in wire.edges() {
            let edge = topo.edge(oe.edge())?;
            let vp = topo.vertex(edge.start())?.point();
            let d = (point - vp).length();
            if d < best {
                best = d;
                best_pt = vp;
            }
        }
        Ok(DistanceResult {
            distance: best,
            point_a: point,
            point_b: best_pt,
        })
    }
}

/// Compute the minimum distance from a point to an edge.
///
/// For line edges, uses exact point-to-segment distance.
/// For curved edges, samples the curve and returns the closest sample.
///
/// # Errors
///
/// Returns an error if the edge lookup fails.
#[allow(clippy::cast_precision_loss)]
pub fn point_to_edge(
    topo: &Topology,
    point: Point3,
    edge_id: brepkit_topology::edge::EdgeId,
) -> Result<DistanceResult, crate::OperationsError> {
    let edge = topo.edge(edge_id)?;
    let start = topo.vertex(edge.start())?.point();
    let end = topo.vertex(edge.end())?.point();

    if matches!(edge.curve(), brepkit_topology::edge::EdgeCurve::Line) {
        let closest = closest_point_on_segment(point, start, end);
        let dist = (point - closest).length();
        Ok(DistanceResult {
            distance: dist,
            point_a: point,
            point_b: closest,
        })
    } else {
        let (t0, t1) = match edge.curve() {
            brepkit_topology::edge::EdgeCurve::NurbsCurve(nc) => nc.domain(),
            brepkit_topology::edge::EdgeCurve::Circle(c) => {
                if edge.is_closed() {
                    (0.0, std::f64::consts::TAU)
                } else {
                    // Project start/end vertices to get actual arc parameter range.
                    let mut t0 = c.project(start);
                    let mut t1 = c.project(end);
                    if t0 < 0.0 {
                        t0 += std::f64::consts::TAU;
                    }
                    if t1 <= t0 {
                        t1 += std::f64::consts::TAU;
                    }
                    (t0, t1)
                }
            }
            brepkit_topology::edge::EdgeCurve::Ellipse(e) => {
                if edge.is_closed() {
                    (0.0, std::f64::consts::TAU)
                } else {
                    let mut t0 = e.project(start);
                    let mut t1 = e.project(end);
                    if t0 < 0.0 {
                        t0 += std::f64::consts::TAU;
                    }
                    if t1 <= t0 {
                        t1 += std::f64::consts::TAU;
                    }
                    (t0, t1)
                }
            }
            // Line was handled above (early return via `if` branch).
            brepkit_topology::edge::EdgeCurve::Line => (0.0, 0.0),
        };
        let n_samples = 64;
        let mut best_dist = f64::INFINITY;
        let mut best_pt = start;
        for i in 0..=n_samples {
            let t = t0 + (t1 - t0) * (i as f64) / (n_samples as f64);
            let pt = match edge.curve() {
                brepkit_topology::edge::EdgeCurve::NurbsCurve(nc) => nc.evaluate(t),
                brepkit_topology::edge::EdgeCurve::Circle(c) => c.evaluate(t),
                brepkit_topology::edge::EdgeCurve::Ellipse(e) => e.evaluate(t),
                // Line was handled above.
                brepkit_topology::edge::EdgeCurve::Line => start,
            };
            let d = (point - pt).length();
            if d < best_dist {
                best_dist = d;
                best_pt = pt;
            }
        }
        Ok(DistanceResult {
            distance: best_dist,
            point_a: point,
            point_b: best_pt,
        })
    }
}

/// Closest point on a line segment to a point.
fn closest_point_on_segment(point: Point3, a: Point3, b: Point3) -> Point3 {
    let ab = b - a;
    let len_sq = ab.length_squared();
    if len_sq < 1e-30 {
        return a;
    }
    let ap = point - a;
    let t = ap.dot(ab) / len_sq;
    let t = t.clamp(0.0, 1.0);
    a + ab * t
}

/// Compute the distance from a point to a single face, dispatching by type.
pub(crate) fn point_to_face_distance(
    topo: &Topology,
    point: Point3,
    face_id: FaceId,
) -> Result<Option<(f64, Point3)>, crate::OperationsError> {
    // The face as trimmed, not its whole surface.
    Ok(brepkit_check::distance::point_to_face(
        topo, point, face_id,
    )?)
}

// -- BVH helpers --------------------------------------------------------------

// -- Segment-to-segment distance ----------------------------------------------

// -- Existing helpers (preserved) ---------------------------------------------

/// Point-in-polygon test for 3D (projecting to 2D).
pub(crate) fn point_in_polygon_3d(point: &Point3, polygon: &[Point3], normal: &Vec3) -> bool {
    use brepkit_math::predicates::point_in_polygon;
    use brepkit_math::vec::Point2;

    let ax = normal.x().abs();
    let ay = normal.y().abs();
    let az = normal.z().abs();

    let (proj_pt, proj_poly): (Point2, Vec<Point2>) = if az >= ax && az >= ay {
        (
            Point2::new(point.x(), point.y()),
            polygon.iter().map(|p| Point2::new(p.x(), p.y())).collect(),
        )
    } else if ay >= ax {
        (
            Point2::new(point.x(), point.z()),
            polygon.iter().map(|p| Point2::new(p.x(), p.z())).collect(),
        )
    } else {
        (
            Point2::new(point.y(), point.z()),
            polygon.iter().map(|p| Point2::new(p.y(), p.z())).collect(),
        )
    };

    point_in_polygon(proj_pt, &proj_poly)
}

#[cfg(test)]
mod tests {
    #![allow(clippy::unwrap_used)]

    use brepkit_math::tolerance::Tolerance;
    use brepkit_math::vec::Point3;
    use brepkit_topology::Topology;
    use brepkit_topology::test_utils::make_unit_cube_manifold_at;

    use super::*;

    #[test]
    fn point_inside_cube_distance_is_half() {
        let mut topo = Topology::new();
        let cube = make_unit_cube_manifold_at(&mut topo, 0.0, 0.0, 0.0);

        // Point at center of cube — closest face is 0.5 away.
        let result = point_to_solid_distance(&topo, Point3::new(0.5, 0.5, 0.5), cube).unwrap();
        let tol = Tolerance::loose();
        assert!(
            tol.approx_eq(result.distance, 0.5),
            "center-to-face distance should be ~0.5, got {}",
            result.distance
        );
    }

    #[test]
    fn point_outside_cube_distance() {
        let mut topo = Topology::new();
        let cube = make_unit_cube_manifold_at(&mut topo, 0.0, 0.0, 0.0);

        // Point above the cube.
        let result = point_to_solid_distance(&topo, Point3::new(0.5, 0.5, 3.0), cube).unwrap();
        let tol = Tolerance::loose();
        assert!(
            tol.approx_eq(result.distance, 2.0),
            "point 2 above cube top should be distance ~2.0, got {}",
            result.distance
        );
    }

    #[test]
    fn disjoint_cubes_distance() {
        let mut topo = Topology::new();
        let a = make_unit_cube_manifold_at(&mut topo, 0.0, 0.0, 0.0);
        let b = make_unit_cube_manifold_at(&mut topo, 5.0, 0.0, 0.0);

        let result = solid_to_solid_distance(&topo, a, b).unwrap();
        let tol = Tolerance::loose();
        // Cubes are [0,1] and [5,6], gap is 4.0.
        assert!(
            tol.approx_eq(result.distance, 4.0),
            "disjoint cubes should be ~4.0 apart, got {}",
            result.distance
        );
    }

    #[test]
    fn adjacent_cubes_distance_is_zero() {
        let mut topo = Topology::new();
        let a = make_unit_cube_manifold_at(&mut topo, 0.0, 0.0, 0.0);
        let b = make_unit_cube_manifold_at(&mut topo, 1.0, 0.0, 0.0);

        let result = solid_to_solid_distance(&topo, a, b).unwrap();
        let tol = Tolerance::loose();
        assert!(
            tol.approx_eq(result.distance, 0.0),
            "touching cubes should have distance ~0, got {}",
            result.distance
        );
    }

    #[test]
    fn same_solid_distance_is_zero() {
        let mut topo = Topology::new();
        let a = make_unit_cube_manifold_at(&mut topo, 0.0, 0.0, 0.0);

        let result = solid_to_solid_distance(&topo, a, a).unwrap();
        let tol = Tolerance::loose();
        assert!(
            tol.approx_eq(result.distance, 0.0),
            "distance to self should be 0, got {}",
            result.distance
        );
    }
}
