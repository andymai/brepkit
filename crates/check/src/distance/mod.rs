//! Minimum distance and extrema between shapes.

#![allow(
    clippy::many_single_char_names,
    clippy::similar_names,
    clippy::suboptimal_flops
)]

pub(crate) mod analytic;
pub(crate) mod edge;
pub(crate) mod lipschitz;

use std::collections::HashSet;

use brepkit_math::aabb::Aabb3;
use brepkit_math::bvh::Bvh;
use brepkit_math::vec::{Point3, Vec3};
use brepkit_topology::Topology;
use brepkit_topology::face::{FaceId, FaceSurface};
use brepkit_topology::solid::SolidId;

use crate::CheckError;

/// Which topological element supports a closest point.
#[derive(Debug, Clone, Copy)]
pub enum SupportElement {
    /// Closest point is on a face.
    Face(FaceId, f64, f64),
}

/// A single distance solution.
#[derive(Debug, Clone)]
pub struct DistanceResult {
    /// The minimum distance.
    pub distance: f64,
    /// Closest point on shape A (or the query point).
    pub point_a: Point3,
    /// Closest point on shape B.
    pub point_b: Point3,
}

/// Compute the minimum distance from a point to a solid.
///
/// Uses BVH over face AABBs for acceleration. Dispatches per face type:
/// planar (point-to-polygon), analytic (closed-form), NURBS (Newton projection).
///
/// # Errors
///
/// Returns an error if any topology entity is missing.
#[allow(clippy::too_many_lines)]
pub fn point_to_solid(
    topo: &Topology,
    point: Point3,
    solid: SolidId,
) -> Result<DistanceResult, CheckError> {
    let solid_data = topo.solid(solid)?;
    let shell = topo.shell(solid_data.outer_shell())?;
    let face_ids: Vec<FaceId> = shell.faces().to_vec();

    // Build AABBs and BVH.
    let mut face_aabbs: Vec<(usize, Aabb3)> = Vec::with_capacity(face_ids.len());
    for (i, &fid) in face_ids.iter().enumerate() {
        let aabb = crate::util::face_aabb(topo, fid)?;
        face_aabbs.push((i, aabb));
    }
    let bvh = Bvh::build(&face_aabbs);

    let mut best_dist = f64::INFINITY;
    let mut best_point = point;

    // Sort candidates by AABB distance to point for early termination.
    let mut candidates: Vec<usize> = (0..face_aabbs.len()).collect();
    candidates.sort_by(|&a, &b| {
        let da = face_aabbs[a].1.distance_squared_to_point(point);
        let db = face_aabbs[b].1.distance_squared_to_point(point);
        da.partial_cmp(&db).unwrap_or(std::cmp::Ordering::Equal)
    });

    // Use BVH closest hint to potentially reorder.
    if let Some(closest_idx) = bvh.query_closest(point) {
        if let Some(pos) = candidates
            .iter()
            .position(|&i| face_aabbs[i].0 == closest_idx)
        {
            candidates.swap(0, pos);
        }
    }

    for idx in candidates {
        let aabb_dist_sq = face_aabbs[idx].1.distance_squared_to_point(point);
        if aabb_dist_sq > best_dist * best_dist {
            continue; // not break — BVH swap may have reordered candidates
        }
        let face_idx = face_aabbs[idx].0;
        let fid = face_ids[face_idx];
        if let Ok(Some((dist, closest))) = point_to_face(topo, point, fid) {
            if dist < best_dist {
                best_dist = dist;
                best_point = closest;
            }
        }
    }

    Ok(DistanceResult {
        distance: best_dist,
        point_a: point,
        point_b: best_point,
    })
}

/// Compute distance from a point to a single face, dispatching by surface type.
///
/// # Errors
///
/// Returns an error if the face lookup fails.
pub fn point_to_face(
    topo: &Topology,
    point: Point3,
    face_id: FaceId,
) -> Result<Option<(f64, Point3)>, CheckError> {
    let face = topo.face(face_id)?;
    match face.surface() {
        FaceSurface::Plane { normal, d } => {
            let polygon = crate::util::face_polygon(topo, face_id)?;
            Ok(point_to_polygon_distance(point, &polygon, *normal, *d))
        }
        FaceSurface::Cylinder(cyl) => {
            let (dist, closest) = analytic::point_to_cylinder(point, cyl);
            if is_point_in_face_boundary(topo, face_id, closest)? {
                Ok(Some((dist, closest)))
            } else {
                Ok(closest_point_on_wire_edges(topo, face_id, point)?)
            }
        }
        FaceSurface::Cone(cone) => {
            let (dist, closest) = analytic::point_to_cone(point, cone);
            if is_point_in_face_boundary(topo, face_id, closest)? {
                Ok(Some((dist, closest)))
            } else {
                Ok(closest_point_on_wire_edges(topo, face_id, point)?)
            }
        }
        FaceSurface::Sphere(sph) => {
            let (dist, closest) = analytic::point_to_sphere(point, sph);
            if is_point_in_face_boundary(topo, face_id, closest)? {
                Ok(Some((dist, closest)))
            } else {
                Ok(closest_point_on_wire_edges(topo, face_id, point)?)
            }
        }
        FaceSurface::Torus(tor) => {
            let (dist, closest) = analytic::point_to_torus(point, tor);
            if is_point_in_face_boundary(topo, face_id, closest)? {
                Ok(Some((dist, closest)))
            } else {
                Ok(closest_point_on_wire_edges(topo, face_id, point)?)
            }
        }
        FaceSurface::Nurbs(nurbs) => {
            match brepkit_math::nurbs::projection::project_point_to_surface(nurbs, point, 1e-7) {
                Ok(proj) => {
                    if is_point_in_face_boundary(topo, face_id, proj.point)? {
                        Ok(Some((proj.distance, proj.point)))
                    } else {
                        closest_point_on_wire_edges(topo, face_id, point)
                    }
                }
                Err(_) => Ok(None),
            }
        }
    }
}

/// Compute the minimum distance between two solids.
///
/// Checks vertex-to-vertex, vertex-to-face, and edge-to-edge pairs
/// with AABB pruning for acceleration.
///
/// # Errors
///
/// Returns an error if any topology entity is missing.
#[allow(clippy::too_many_lines)]
pub fn solid_to_solid(
    topo: &Topology,
    solid_a: SolidId,
    solid_b: SolidId,
) -> Result<DistanceResult, CheckError> {
    // Collect vertex positions from each solid.
    let verts_a = collect_solid_vertices(topo, solid_a)?;
    let verts_b = collect_solid_vertices(topo, solid_b)?;

    let mut best_dist = f64::INFINITY;
    let mut best_a = Point3::new(0.0, 0.0, 0.0);
    let mut best_b = Point3::new(0.0, 0.0, 0.0);

    // Pass 1: Vertex-vertex (cheap upper bound).
    for &pa in &verts_a {
        for &pb in &verts_b {
            let dist = (pa - pb).length();
            if dist < best_dist {
                best_dist = dist;
                best_a = pa;
                best_b = pb;
            }
        }
    }

    // Pass 2: Vertices of A against faces of B.
    let faces_b = collect_solid_faces(topo, solid_b)?;
    let mut aabbs_b: Vec<(usize, Aabb3)> = Vec::with_capacity(faces_b.len());
    for (i, &fid) in faces_b.iter().enumerate() {
        let aabb = crate::util::face_aabb(topo, fid)?;
        aabbs_b.push((i, aabb));
    }

    for &pa in &verts_a {
        for &(idx, ref aabb) in &aabbs_b {
            if aabb.distance_squared_to_point(pa) > best_dist * best_dist {
                continue;
            }
            if let Ok(Some((dist, closest))) = point_to_face(topo, pa, faces_b[idx]) {
                if dist < best_dist {
                    best_dist = dist;
                    best_a = pa;
                    best_b = closest;
                }
            }
        }
    }

    // Pass 3: Vertices of B against faces of A.
    let faces_a = collect_solid_faces(topo, solid_a)?;
    let mut aabbs_a: Vec<(usize, Aabb3)> = Vec::with_capacity(faces_a.len());
    for (i, &fid) in faces_a.iter().enumerate() {
        let aabb = crate::util::face_aabb(topo, fid)?;
        aabbs_a.push((i, aabb));
    }

    for &pb in &verts_b {
        for &(idx, ref aabb) in &aabbs_a {
            if aabb.distance_squared_to_point(pb) > best_dist * best_dist {
                continue;
            }
            if let Ok(Some((dist, closest))) = point_to_face(topo, pb, faces_a[idx]) {
                if dist < best_dist {
                    best_dist = dist;
                    best_b = pb;
                    best_a = closest;
                }
            }
        }
    }

    // Pass 4: Edge-edge with AABB pruning.
    let edges_a = collect_solid_edge_segments(topo, solid_a)?;
    let edges_b = collect_solid_edge_segments(topo, solid_b)?;

    for &(p0a, p1a) in &edges_a {
        let aabb_a = Aabb3::try_from_points([p0a, p1a].iter().copied())
            .unwrap_or(Aabb3 { min: p0a, max: p0a });
        for &(p0b, p1b) in &edges_b {
            let aabb_b = Aabb3::try_from_points([p0b, p1b].iter().copied())
                .unwrap_or(Aabb3 { min: p0b, max: p0b });
            if aabb_distance(&aabb_a, &aabb_b) > best_dist {
                continue;
            }
            let (dist, ca, cb) = edge::segment_segment_distance(p0a, p1a, p0b, p1b);
            if dist < best_dist {
                best_dist = dist;
                best_a = ca;
                best_b = cb;
            }
        }
    }

    Ok(DistanceResult {
        distance: best_dist,
        point_a: best_a,
        point_b: best_b,
    })
}

/// Collect all unique vertex positions from a solid.
fn collect_solid_vertices(topo: &Topology, solid: SolidId) -> Result<Vec<Point3>, CheckError> {
    let solid_data = topo.solid(solid)?;
    let shell = topo.shell(solid_data.outer_shell())?;
    let mut seen = HashSet::new();
    let mut points = Vec::new();
    for &fid in shell.faces() {
        let face = topo.face(fid)?;
        let mut wire_ids = vec![face.outer_wire()];
        wire_ids.extend(face.inner_wires().iter().copied());
        for wid in wire_ids {
            let wire = topo.wire(wid)?;
            for oe in wire.edges() {
                let edge_data = topo.edge(oe.edge())?;
                for vid in [edge_data.start(), edge_data.end()] {
                    if seen.insert(vid) {
                        points.push(topo.vertex(vid)?.point());
                    }
                }
            }
        }
    }
    Ok(points)
}

/// Collect all face IDs from a solid's outer shell.
fn collect_solid_faces(topo: &Topology, solid: SolidId) -> Result<Vec<FaceId>, CheckError> {
    let solid_data = topo.solid(solid)?;
    let shell = topo.shell(solid_data.outer_shell())?;
    Ok(shell.faces().to_vec())
}

/// Collect edge segments as polylines for edge-edge distance computation.
///
/// Line edges produce a single segment. Curved edges (circle, ellipse, NURBS)
/// are sampled at multiple points to capture the curve geometry.
#[allow(clippy::cast_precision_loss)]
fn collect_solid_edge_segments(
    topo: &Topology,
    solid: SolidId,
) -> Result<Vec<(Point3, Point3)>, CheckError> {
    use brepkit_topology::edge::EdgeCurve;

    let solid_data = topo.solid(solid)?;
    let shell = topo.shell(solid_data.outer_shell())?;
    let mut seen = HashSet::new();
    let mut segments = Vec::new();

    let n_samples = 8usize;

    for &fid in shell.faces() {
        let face = topo.face(fid)?;
        // Iterate outer wire + inner wires (holes)
        let mut wire_ids = vec![face.outer_wire()];
        wire_ids.extend(face.inner_wires().iter().copied());
        for wid in wire_ids {
            let wire = topo.wire(wid)?;
            for oe in wire.edges() {
                let eid = oe.edge();
                if !seen.insert(eid) {
                    continue;
                }
                let edge_data = topo.edge(eid)?;
                let start_pt = topo.vertex(edge_data.start())?.point();
                let end_pt = topo.vertex(edge_data.end())?.point();

                match edge_data.curve() {
                    EdgeCurve::Line => {
                        segments.push((start_pt, end_pt));
                    }
                    EdgeCurve::Circle(c) => {
                        let is_closed = edge_data.start() == edge_data.end();
                        let (t0, t1) = if is_closed {
                            (0.0, std::f64::consts::TAU)
                        } else {
                            let t0 = c.project(start_pt);
                            let mut t1 = c.project(end_pt);
                            if t1 <= t0 {
                                t1 += std::f64::consts::TAU;
                            }
                            (t0, t1)
                        };
                        let mut prev = c.evaluate(t0);
                        for i in 1..=n_samples {
                            let t = t0 + (t1 - t0) * (i as f64) / (n_samples as f64);
                            let curr = c.evaluate(t);
                            segments.push((prev, curr));
                            prev = curr;
                        }
                    }
                    EdgeCurve::Ellipse(e) => {
                        let is_closed = edge_data.start() == edge_data.end();
                        let (t0, t1) = if is_closed {
                            (0.0, std::f64::consts::TAU)
                        } else {
                            let t0 = e.project(start_pt);
                            let mut t1 = e.project(end_pt);
                            if t1 <= t0 {
                                t1 += std::f64::consts::TAU;
                            }
                            (t0, t1)
                        };
                        let mut prev = e.evaluate(t0);
                        for i in 1..=n_samples {
                            let t = t0 + (t1 - t0) * (i as f64) / (n_samples as f64);
                            let curr = e.evaluate(t);
                            segments.push((prev, curr));
                            prev = curr;
                        }
                    }
                    EdgeCurve::NurbsCurve(nc) => {
                        let (t0, t1) = nc.domain();
                        let mut prev = nc.evaluate(t0);
                        for i in 1..=n_samples {
                            let t = t0 + (t1 - t0) * (i as f64) / (n_samples as f64);
                            let curr = nc.evaluate(t);
                            segments.push((prev, curr));
                            prev = curr;
                        }
                    }
                }
            }
        }
    }

    Ok(segments)
}

/// Check if a point lies within the face's boundary polygon.
fn is_point_in_face_boundary(
    topo: &Topology,
    face_id: FaceId,
    point: Point3,
) -> Result<bool, CheckError> {
    let polygon = crate::util::face_polygon(topo, face_id)?;
    if polygon.len() < 3 {
        return Ok(true); // Full-surface face
    }
    let normal = crate::util::polygon_normal(&polygon);
    Ok(crate::util::point_in_polygon_3d(&point, &polygon, &normal))
}

/// Find the closest point on the wire edges of a face to a given point.
///
/// Iterates both the outer wire and inner wires (holes).
fn closest_point_on_wire_edges(
    topo: &Topology,
    face_id: FaceId,
    point: Point3,
) -> Result<Option<(f64, Point3)>, CheckError> {
    let face = topo.face(face_id)?;
    let mut best_dist = f64::INFINITY;
    let mut best_pt = point;

    let mut wire_ids = vec![face.outer_wire()];
    wire_ids.extend(face.inner_wires().iter().copied());

    for wid in wire_ids {
        let wire = topo.wire(wid)?;
        for oe in wire.edges() {
            let edge_data = topo.edge(oe.edge())?;
            let p0 = topo.vertex(edge_data.start())?.point();
            let p1 = topo.vertex(edge_data.end())?.point();
            let (dist, closest) = point_to_segment(point, p0, p1);
            if dist < best_dist {
                best_dist = dist;
                best_pt = closest;
            }
        }
    }
    if best_dist < f64::INFINITY {
        Ok(Some((best_dist, best_pt)))
    } else {
        Ok(None)
    }
}

/// Point-to-polygon distance for planar faces.
///
/// Projects the point onto the plane, checks if inside polygon, otherwise
/// finds the closest point on polygon edges.
fn point_to_polygon_distance(
    point: Point3,
    polygon: &[Point3],
    normal: Vec3,
    d: f64,
) -> Option<(f64, Point3)> {
    if polygon.len() < 3 {
        return None;
    }

    // Project point onto plane.
    let (_, projected) = analytic::point_to_plane(point, normal, d);

    // Check if projected point is inside polygon.
    if crate::util::point_in_polygon_3d(&projected, polygon, &normal) {
        let dist = (point - projected).length();
        return Some((dist, projected));
    }

    // Otherwise, find closest point on polygon edges.
    let mut best_dist = f64::INFINITY;
    let mut best_pt = polygon[0];
    let n = polygon.len();
    for i in 0..n {
        let j = (i + 1) % n;
        let (dist, closest) = point_to_segment(point, polygon[i], polygon[j]);
        if dist < best_dist {
            best_dist = dist;
            best_pt = closest;
        }
    }
    Some((best_dist, best_pt))
}

/// Distance from point to line segment.
fn point_to_segment(point: Point3, a: Point3, b: Point3) -> (f64, Point3) {
    let ab = b - a;
    let ap = point - a;
    let len_sq = ab.length_squared();
    if len_sq < 1e-30 {
        return ((point - a).length(), a);
    }
    let t = (ap.dot(ab) / len_sq).clamp(0.0, 1.0);
    let closest = Point3::new(
        ab.x().mul_add(t, a.x()),
        ab.y().mul_add(t, a.y()),
        ab.z().mul_add(t, a.z()),
    );
    ((point - closest).length(), closest)
}

/// Compute minimum distance between two AABBs.
fn aabb_distance(a: &Aabb3, b: &Aabb3) -> f64 {
    let dx = (a.min.x() - b.max.x()).max(b.min.x() - a.max.x()).max(0.0);
    let dy = (a.min.y() - b.max.y()).max(b.min.y() - a.max.y()).max(0.0);
    let dz = (a.min.z() - b.max.z()).max(b.min.z() - a.max.z()).max(0.0);
    (dx.mul_add(dx, dy.mul_add(dy, dz * dz))).sqrt()
}

#[cfg(test)]
mod tests {
    #![allow(clippy::unwrap_used, clippy::expect_used)]

    use super::*;
    use brepkit_math::surfaces::{CylindricalSurface, SphericalSurface, ToroidalSurface};

    #[test]
    fn point_to_sphere_outside() {
        let sphere = SphericalSurface::new(Point3::new(0.0, 0.0, 0.0), 1.0).unwrap();
        let (dist, closest) = analytic::point_to_sphere(Point3::new(3.0, 0.0, 0.0), &sphere);
        assert!(
            (dist - 2.0).abs() < 1e-10,
            "distance should be 2.0, got {dist}"
        );
        assert!((closest.x() - 1.0).abs() < 1e-10);
        assert!(closest.y().abs() < 1e-10);
        assert!(closest.z().abs() < 1e-10);
    }

    #[test]
    fn point_to_sphere_inside() {
        let sphere = SphericalSurface::new(Point3::new(0.0, 0.0, 0.0), 1.0).unwrap();
        let (dist, closest) = analytic::point_to_sphere(Point3::new(0.5, 0.0, 0.0), &sphere);
        assert!(
            (dist - 0.5).abs() < 1e-10,
            "distance should be 0.5, got {dist}"
        );
        assert!((closest.x() - 1.0).abs() < 1e-10);
    }

    #[test]
    fn point_to_cylinder_outside() {
        let cyl =
            CylindricalSurface::new(Point3::new(0.0, 0.0, 0.0), Vec3::new(0.0, 0.0, 1.0), 1.0)
                .unwrap();
        let (dist, closest) = analytic::point_to_cylinder(Point3::new(2.0, 0.0, 0.0), &cyl);
        assert!(
            (dist - 1.0).abs() < 1e-10,
            "distance should be 1.0, got {dist}"
        );
        assert!((closest.x() - 1.0).abs() < 1e-10);
        assert!(closest.y().abs() < 1e-10);
        assert!(closest.z().abs() < 1e-10);
    }

    #[test]
    fn point_to_plane_above() {
        let normal = Vec3::new(0.0, 0.0, 1.0);
        let d = 0.0;
        let (dist, closest) = analytic::point_to_plane(Point3::new(0.0, 0.0, 5.0), normal, d);
        assert!(
            (dist - 5.0).abs() < 1e-10,
            "distance should be 5.0, got {dist}"
        );
        assert!(closest.x().abs() < 1e-10);
        assert!(closest.y().abs() < 1e-10);
        assert!(closest.z().abs() < 1e-10);
    }

    #[test]
    fn segment_segment_parallel() {
        // Two parallel segments along X, separated by 2.0 in Y.
        let (dist, _, _) = edge::segment_segment_distance(
            Point3::new(0.0, 0.0, 0.0),
            Point3::new(1.0, 0.0, 0.0),
            Point3::new(0.0, 2.0, 0.0),
            Point3::new(1.0, 2.0, 0.0),
        );
        assert!(
            (dist - 2.0).abs() < 1e-10,
            "parallel segment distance should be 2.0, got {dist}"
        );
    }

    #[test]
    fn segment_segment_crossing() {
        // Two segments that cross: one along X, one along Y, both through origin.
        let (dist, _, _) = edge::segment_segment_distance(
            Point3::new(-1.0, 0.0, 0.0),
            Point3::new(1.0, 0.0, 0.0),
            Point3::new(0.0, -1.0, 0.0),
            Point3::new(0.0, 1.0, 0.0),
        );
        assert!(
            dist < 1e-10,
            "crossing segment distance should be ~0, got {dist}"
        );
    }

    #[test]
    fn segment_segment_skew() {
        // Two skew segments separated by 3.0 in Z.
        let (dist, ca, cb) = edge::segment_segment_distance(
            Point3::new(0.0, 0.0, 0.0),
            Point3::new(1.0, 0.0, 0.0),
            Point3::new(0.0, 0.0, 3.0),
            Point3::new(0.0, 1.0, 3.0),
        );
        assert!(
            (dist - 3.0).abs() < 1e-10,
            "skew segment distance should be 3.0, got {dist}"
        );
        assert!(ca.z().abs() < 1e-10);
        assert!((cb.z() - 3.0).abs() < 1e-10);
    }

    #[test]
    fn solid_to_solid_separated() {
        use brepkit_topology::test_utils::make_unit_cube_manifold_at;
        let mut topo = Topology::new();
        // Two unit cubes: one at origin, one at (3, 0, 0). Gap of 2.0 in X.
        let a = make_unit_cube_manifold_at(&mut topo, 0.0, 0.0, 0.0);
        let b = make_unit_cube_manifold_at(&mut topo, 3.0, 0.0, 0.0);
        let result = solid_to_solid(&topo, a, b).unwrap();
        assert!(
            (result.distance - 2.0).abs() < 1e-10,
            "distance should be 2.0, got {}",
            result.distance
        );
    }

    #[test]
    fn point_to_torus_outside() {
        // Torus at origin with major_radius=3, minor_radius=1, Z-axis.
        let torus = ToroidalSurface::new(Point3::new(0.0, 0.0, 0.0), 3.0, 1.0).unwrap();
        // Point at (6, 0, 0): major circle closest is (3,0,0), tube dist = 3, minor_r = 1.
        let (dist, closest) = analytic::point_to_torus(Point3::new(6.0, 0.0, 0.0), &torus);
        assert!(
            (dist - 2.0).abs() < 1e-10,
            "distance should be 2.0, got {dist}"
        );
        assert!((closest.x() - 4.0).abs() < 1e-10);
        assert!(closest.y().abs() < 1e-10);
        assert!(closest.z().abs() < 1e-10);
    }
}
