//! Phase 4: create new edges from face-face intersection curves.
//!
//! After Phase 3 computes intersection curves between adjacent offset faces,
//! this phase creates the corresponding topology: vertices at the intersection
//! line endpoints and edges connecting them.

use brepkit_math::vec::{Point3, Vec3};
use brepkit_topology::Topology;
use brepkit_topology::edge::{Edge, EdgeCurve, EdgeId};
use brepkit_topology::solid::SolidId;
use brepkit_topology::vertex::VertexId;

use crate::data::{OffsetData, find_or_create_vertex};
use crate::error::OffsetError;

/// Create new edges from the intersection curves computed in Phase 3.
///
/// For each `FaceIntersection` with non-empty `curve_points`, this function:
/// 1. Finds or creates vertices at the first and last curve points
///    (deduplicated by tolerance).
/// 2. Creates a `Line` edge between them.
/// 3. Stores the new edge ID in `FaceIntersection::new_edges`.
///
/// # Errors
///
/// Returns [`OffsetError`] if topology operations fail.
#[allow(clippy::unnecessary_wraps)]
pub fn intersect_pcurves_2d(
    topo: &mut Topology,
    _solid: SolidId,
    data: &mut OffsetData,
) -> Result<(), OffsetError> {
    let tol = data.options.tolerance.linear;
    let mut vertex_cache: Vec<(Point3, VertexId)> = Vec::new();

    for intersection in &mut data.intersections {
        if intersection.curve_points.len() < 2 {
            continue;
        }

        if let Some(edge_id) =
            create_edge_from_curve_points(topo, &mut vertex_cache, &intersection.curve_points, tol)
        {
            intersection.new_edges.push(edge_id);
        }
    }

    Ok(())
}

/// Create a topological edge from sampled intersection curve points.
///
/// If the points form a closed circle (all equidistant from centroid),
/// creates a `Circle` edge. Otherwise creates a `Line` edge between
/// the first and last points.
///
/// Returns `None` if the edge would be degenerate.
fn create_edge_from_curve_points(
    topo: &mut Topology,
    vertex_cache: &mut Vec<(Point3, VertexId)>,
    points: &[Point3],
    tol: f64,
) -> Option<EdgeId> {
    if points.len() < 2 {
        return None;
    }

    // Try to fit a circle if we have enough points.
    if points.len() >= 8 {
        if let Some((circle, seam_pt)) = fit_circle_3d(points, tol) {
            let v = find_or_create_vertex(topo, vertex_cache, seam_pt, tol);
            return Some(topo.add_edge(Edge::new(v, v, EdgeCurve::Circle(circle))));
        }
    }

    // Fall back to line edge.
    let p_start = points[0];
    let p_end = points[points.len() - 1];
    let v_start = find_or_create_vertex(topo, vertex_cache, p_start, tol);
    let v_end = find_or_create_vertex(topo, vertex_cache, p_end, tol);
    if v_start == v_end {
        return None;
    }
    Some(topo.add_edge(Edge::new(v_start, v_end, EdgeCurve::Line)))
}

/// Squared distance between two points.
fn dist_sq(a: Point3, b: Point3) -> f64 {
    let dx = a.x() - b.x();
    let dy = a.y() - b.y();
    let dz = a.z() - b.z();
    dx.mul_add(dx, dy.mul_add(dy, dz * dz))
}

/// Fit a `Circle3D` to sampled points if they lie on a circle within tolerance.
///
/// Uses a 3-point circumcircle from well-spaced samples (marching samples
/// are non-uniform, so centroid ≠ center). Validates against all points.
///
/// Returns `(Circle3D, seam_point)` where `seam_point` is the first point
/// projected exactly onto the fitted circle. Returns `None` if points
/// don't form a circle.
#[allow(clippy::too_many_lines)]
fn fit_circle_3d(points: &[Point3], tol: f64) -> Option<(brepkit_math::curves::Circle3D, Point3)> {
    let n = points.len();
    if n < 8 {
        return None;
    }

    // Pick 3 well-spaced points.
    let p0 = points[0];
    let p1 = points[n / 3];
    let p2 = points[2 * n / 3];

    // Compute circumcircle of p0, p1, p2 in 3D.
    // 1. Plane normal from cross product.
    let d1 = Vec3::new(p1.x() - p0.x(), p1.y() - p0.y(), p1.z() - p0.z());
    let d2 = Vec3::new(p2.x() - p0.x(), p2.y() - p0.y(), p2.z() - p0.z());
    let normal = d1.cross(d2);
    let normal_len = normal.length();
    if normal_len < 1e-15 {
        return None; // Collinear
    }
    let normal = Vec3::new(
        normal.x() / normal_len,
        normal.y() / normal_len,
        normal.z() / normal_len,
    );

    // 2. Build local 2D frame in the plane.
    let u_axis = {
        let len = d1.length();
        if len < 1e-15 {
            return None;
        }
        Vec3::new(d1.x() / len, d1.y() / len, d1.z() / len)
    };
    let v_axis = normal.cross(u_axis);

    // 3. Project 3 points to 2D.
    let proj = |p: Point3| -> (f64, f64) {
        let dx = p.x() - p0.x();
        let dy = p.y() - p0.y();
        let dz = p.z() - p0.z();
        let v = Vec3::new(dx, dy, dz);
        (v.dot(u_axis), v.dot(v_axis))
    };
    let (ax, ay) = (0.0, 0.0); // p0 in local coords
    let (bx, by) = proj(p1);
    let (cx_l, cy_l) = proj(p2);

    // 4. Circumcenter in 2D: solve perpendicular bisector intersection.
    let d_val = 2.0 * (ax * (by - cy_l) + bx * (cy_l - ay) + cx_l * (ay - by));
    if d_val.abs() < 1e-15 {
        return None;
    }
    let ax2 = ax.mul_add(ax, ay * ay);
    let bx2 = bx.mul_add(bx, by * by);
    let cx2 = cx_l.mul_add(cx_l, cy_l * cy_l);
    let ux = (ax2 * (by - cy_l) + bx2 * (cy_l - ay) + cx2 * (ay - by)) / d_val;
    let uy = (ax2 * (cx_l - bx) + bx2 * (ax - cx_l) + cx2 * (bx - ax)) / d_val;

    let radius = ((ax - ux).powi(2) + (ay - uy).powi(2)).sqrt();
    if radius < tol {
        return None;
    }

    // 5. Lift circumcenter back to 3D.
    let center = Point3::new(
        p0.x() + ux * u_axis.x() + uy * v_axis.x(),
        p0.y() + ux * u_axis.y() + uy * v_axis.y(),
        p0.z() + ux * u_axis.z() + uy * v_axis.z(),
    );

    // 6. Validate: all points should be within tolerance of the circle.
    let max_dev = points
        .iter()
        .map(|p| (dist_sq(*p, center).sqrt() - radius).abs())
        .fold(0.0_f64, f64::max);
    if max_dev > tol.max(radius * 1e-4) {
        return None; // Deviation exceeds tolerance → not a circle
    }

    let circle = brepkit_math::curves::Circle3D::new(center, normal, radius).ok()?;

    // Seam point: project first point exactly onto the circle.
    let dir = Vec3::new(
        points[0].x() - center.x(),
        points[0].y() - center.y(),
        points[0].z() - center.z(),
    );
    let dir_len = dir.length();
    let seam_pt = if dir_len > 1e-15 {
        Point3::new(
            center.x() + radius * dir.x() / dir_len,
            center.y() + radius * dir.y() / dir_len,
            center.z() + radius * dir.z() / dir_len,
        )
    } else {
        points[0]
    };

    Some((circle, seam_pt))
}

#[cfg(test)]
mod tests {
    #![allow(clippy::unwrap_used, clippy::expect_used)]
    use super::*;
    use crate::data::{OffsetData, OffsetOptions};
    use brepkit_topology::Topology;
    use brepkit_topology::solid::SolidId;

    fn run_phases_1_to_4(topo: &mut Topology, solid: SolidId, distance: f64) -> OffsetData {
        let mut data = OffsetData::new(distance, OffsetOptions::default(), vec![]);
        crate::analyse::analyse_edges(topo, solid, &mut data).unwrap();
        crate::offset::build_offset_faces(topo, solid, &mut data).unwrap();
        crate::inter3d::intersect_faces_3d(topo, solid, &mut data).unwrap();
        intersect_pcurves_2d(topo, solid, &mut data).unwrap();
        data
    }

    #[test]
    fn box_intersections_have_new_edges() {
        let mut topo = Topology::new();
        let solid = brepkit_topology::test_utils::make_unit_cube_manifold(&mut topo);
        let data = run_phases_1_to_4(&mut topo, solid, 0.5);
        for fi in &data.intersections {
            assert!(
                !fi.new_edges.is_empty(),
                "intersection for edge {:?} should have new edges",
                fi.original_edge
            );
        }
    }

    #[test]
    fn box_new_edges_are_valid() {
        let mut topo = Topology::new();
        let solid = brepkit_topology::test_utils::make_unit_cube_manifold(&mut topo);
        let data = run_phases_1_to_4(&mut topo, solid, 0.5);
        for fi in &data.intersections {
            for &eid in &fi.new_edges {
                let edge = topo.edge(eid).unwrap();
                let start = topo.vertex(edge.start()).unwrap().point();
                let end = topo.vertex(edge.end()).unwrap().point();
                let length = ((end.x() - start.x()).powi(2)
                    + (end.y() - start.y()).powi(2)
                    + (end.z() - start.z()).powi(2))
                .sqrt();
                assert!(
                    length > 1e-10,
                    "new edge should have non-zero length, got {length}"
                );
            }
        }
    }

    #[test]
    fn vertices_are_deduplicated_within_tolerance() {
        let mut topo = Topology::new();
        let mut cache = Vec::new();
        let tol = 1e-7;
        let p = brepkit_math::vec::Point3::new(1.0, 2.0, 3.0);
        let v1 = find_or_create_vertex(&mut topo, &mut cache, p, tol);
        let p_near = brepkit_math::vec::Point3::new(1.0, 2.0, 3.0 + 1e-9);
        let v2 = find_or_create_vertex(&mut topo, &mut cache, p_near, tol);
        assert_eq!(v1, v2, "nearby points should reuse the same vertex");

        let p_far = brepkit_math::vec::Point3::new(1.0, 2.0, 4.0);
        let v3 = find_or_create_vertex(&mut topo, &mut cache, p_far, tol);
        assert_ne!(v1, v3, "distant points should get different vertices");
    }

    #[test]
    fn box_creates_12_edges() {
        let mut topo = Topology::new();
        let solid = brepkit_topology::test_utils::make_unit_cube_manifold(&mut topo);
        let data = run_phases_1_to_4(&mut topo, solid, 0.5);
        let total_new_edges: usize = data.intersections.iter().map(|fi| fi.new_edges.len()).sum();
        assert_eq!(
            total_new_edges, 12,
            "box offset should create 12 new edges (one per original edge)"
        );
    }

    #[test]
    fn circle_points_produce_circle_edge() {
        use std::f64::consts::TAU;
        let mut topo = Topology::new();
        let mut cache = Vec::new();
        let tol = 1e-7;

        // 32 points on a circle at z=5, radius=2.5
        let n = 32;
        let radius = 2.5;
        let points: Vec<_> = (0..n)
            .map(|i| {
                let t = TAU * i as f64 / n as f64;
                brepkit_math::vec::Point3::new(radius * t.cos(), radius * t.sin(), 5.0)
            })
            .collect();

        let result = create_edge_from_curve_points(&mut topo, &mut cache, &points, tol);
        assert!(result.is_some(), "should create an edge from circle points");
        let eid = result.unwrap();
        let edge = topo.edge(eid).unwrap();
        assert_eq!(
            edge.start(),
            edge.end(),
            "circle edge should be closed (start == end)"
        );
        assert!(
            matches!(edge.curve(), brepkit_topology::edge::EdgeCurve::Circle(_)),
            "edge should be a Circle, got {:?}",
            edge.curve()
        );
    }
}
