//! Phase 4: create new edges from face-face intersection curves.
//!
//! After Phase 3 computes intersection curves between adjacent offset faces,
//! this phase creates the corresponding topology: vertices at the intersection
//! line endpoints and edges connecting them.

use brepkit_math::tolerance::Tolerance;
use brepkit_math::vec::Point3;
use brepkit_topology::Topology;
use brepkit_topology::edge::{Edge, EdgeCurve};
use brepkit_topology::solid::SolidId;
use brepkit_topology::vertex::{Vertex, VertexId};

use crate::data::OffsetData;
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

        let p_start = intersection.curve_points[0];
        let p_end = intersection.curve_points[intersection.curve_points.len() - 1];

        let v_start = find_or_create_vertex(topo, &mut vertex_cache, p_start, tol);
        let v_end = find_or_create_vertex(topo, &mut vertex_cache, p_end, tol);

        // Skip degenerate edges where both endpoints coincide.
        if v_start == v_end {
            continue;
        }

        let edge_id = topo.add_edge(Edge::new(v_start, v_end, EdgeCurve::Line));
        intersection.new_edges.push(edge_id);
    }

    Ok(())
}

/// Find an existing vertex within tolerance of `point`, or create a new one.
fn find_or_create_vertex(
    topo: &mut Topology,
    cache: &mut Vec<(Point3, VertexId)>,
    point: Point3,
    tol: f64,
) -> VertexId {
    for &(cached_pt, vid) in cache.iter() {
        let dx = point.x() - cached_pt.x();
        let dy = point.y() - cached_pt.y();
        let dz = point.z() - cached_pt.z();
        if dx * dx + dy * dy + dz * dz <= tol * tol {
            return vid;
        }
    }

    let vid = topo.add_vertex(Vertex::new(point, Tolerance::default().linear));
    cache.push((point, vid));
    vid
}

#[cfg(test)]
mod tests {
    #![allow(clippy::unwrap_used, clippy::expect_used)]
    use super::*;
    use crate::data::{OffsetData, OffsetOptions};
    use brepkit_topology::Topology;

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
        // Each of 12 intersections should have at least 1 new edge.
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
        // Verify that vertex deduplication works by creating intersections
        // with known overlapping endpoints.
        let mut topo = Topology::new();
        let tol = Tolerance::default().linear;
        let mut cache: Vec<(Point3, VertexId)> = Vec::new();

        let p1 = Point3::new(1.0, 2.0, 3.0);
        let p2 = Point3::new(1.0, 2.0, 3.0 + tol * 0.5); // within tolerance
        let p3 = Point3::new(1.0, 2.0, 3.0 + tol * 2.0); // outside tolerance

        let v1 = find_or_create_vertex(&mut topo, &mut cache, p1, tol);
        let v2 = find_or_create_vertex(&mut topo, &mut cache, p2, tol);
        let v3 = find_or_create_vertex(&mut topo, &mut cache, p3, tol);

        assert_eq!(v1, v2, "points within tolerance should share a vertex");
        assert_ne!(
            v1, v3,
            "points outside tolerance should get distinct vertices"
        );
        assert_eq!(cache.len(), 2);
    }

    #[test]
    fn box_creates_12_edges() {
        let mut topo = Topology::new();
        let solid = brepkit_topology::test_utils::make_unit_cube_manifold(&mut topo);
        let data = run_phases_1_to_4(&mut topo, solid, 0.5);
        let total_new_edges: usize = data.intersections.iter().map(|fi| fi.new_edges.len()).sum();
        // 12 intersections, each producing 1 new edge.
        assert_eq!(total_new_edges, 12);
    }
}
