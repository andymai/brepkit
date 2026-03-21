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
/// Returns `(Circle3D, seam_point)` where `seam_point` is the first point
/// projected exactly onto the fitted circle. Returns `None` if points
/// don't form a circle.
fn fit_circle_3d(points: &[Point3], tol: f64) -> Option<(brepkit_math::curves::Circle3D, Point3)> {
    let n = points.len() as f64;
    let cx: f64 = points.iter().map(|p| p.x()).sum::<f64>() / n;
    let cy: f64 = points.iter().map(|p| p.y()).sum::<f64>() / n;
    let cz: f64 = points.iter().map(|p| p.z()).sum::<f64>() / n;
    let center = Point3::new(cx, cy, cz);

    let radius = points
        .iter()
        .map(|p| dist_sq(*p, center).sqrt())
        .sum::<f64>()
        / n;

    if radius < tol {
        return None;
    }

    // Check all points are within tolerance of the circle radius.
    let max_dev = points
        .iter()
        .map(|p| (dist_sq(*p, center).sqrt() - radius).abs())
        .fold(0.0_f64, f64::max);
    if max_dev > tol * 100.0 {
        return None;
    }

    // Normal via Newell's method on centered points.
    let mut nx = 0.0_f64;
    let mut ny = 0.0_f64;
    let mut nz = 0.0_f64;
    for i in 0..points.len() {
        let a = points[i];
        let b = points[(i + 1) % points.len()];
        nx += (a.y() - cy) * (b.z() - cz) - (a.z() - cz) * (b.y() - cy);
        ny += (a.z() - cz) * (b.x() - cx) - (a.x() - cx) * (b.z() - cz);
        nz += (a.x() - cx) * (b.y() - cy) - (a.y() - cy) * (b.x() - cx);
    }
    let nlen = (nx.mul_add(nx, ny.mul_add(ny, nz * nz))).sqrt();
    if nlen < 1e-15 {
        return None;
    }
    let normal = Vec3::new(nx / nlen, ny / nlen, nz / nlen);

    let circle = brepkit_math::curves::Circle3D::new(center, normal, radius).ok()?;

    // Seam point: project first point exactly onto the circle.
    let dir = Vec3::new(points[0].x() - cx, points[0].y() - cy, points[0].z() - cz);
    let dir_len = dir.length();
    let seam_pt = if dir_len > 1e-15 {
        Point3::new(
            cx + radius * dir.x() / dir_len,
            cy + radius * dir.y() / dir_len,
            cz + radius * dir.z() / dir_len,
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
