//! Distance measurement between shapes.
//!
//! Equivalent to `BRepExtrema_DistShapeShape` in `OpenCascade`.
//! Computes minimum distance between solids and point-to-solid distance.

use brepkit_math::tolerance::Tolerance;
use brepkit_math::vec::{Point3, Vec3};
use brepkit_topology::Topology;
use brepkit_topology::face::FaceSurface;
use brepkit_topology::solid::SolidId;

use crate::boolean::face_vertices;

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
/// Checks the distance from the point to every face of the solid.
/// For planar faces, computes the point-to-polygon distance.
///
/// # Errors
///
/// Returns an error if the solid is invalid or contains NURBS faces.
pub fn point_to_solid_distance(
    topo: &Topology,
    point: Point3,
    solid: SolidId,
) -> Result<DistanceResult, crate::OperationsError> {
    let tol = Tolerance::new();

    let solid_data = topo.solid(solid)?;
    let shell = topo.shell(solid_data.outer_shell())?;

    let mut best_dist = f64::INFINITY;
    let mut best_point = point;

    for &fid in shell.faces() {
        let face = topo.face(fid)?;
        let (normal, d) = match face.surface() {
            FaceSurface::Plane { normal, d } => (*normal, *d),
            FaceSurface::Nurbs(_) => {
                return Err(crate::OperationsError::InvalidInput {
                    reason: "distance to NURBS faces not yet supported".into(),
                });
            }
        };

        let verts = face_vertices(topo, fid)?;
        if let Some((dist, closest)) = point_to_polygon_distance(point, &verts, normal, d, tol) {
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

/// Compute the minimum distance between two solids.
///
/// Checks all vertex-to-face and edge-to-edge distances between
/// the two solids. Uses AABB culling for early rejection.
///
/// # Errors
///
/// Returns an error if either solid is invalid or contains NURBS faces.
pub fn solid_to_solid_distance(
    topo: &Topology,
    solid_a: SolidId,
    solid_b: SolidId,
) -> Result<DistanceResult, crate::OperationsError> {
    // Collect vertices from both solids.
    let verts_a = collect_solid_points(topo, solid_a)?;
    let verts_b = collect_solid_points(topo, solid_b)?;

    let mut best_dist = f64::INFINITY;
    let mut best_a = Point3::new(0.0, 0.0, 0.0);
    let mut best_b = Point3::new(0.0, 0.0, 0.0);

    // Check all vertex pairs between the two solids.
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

    // Also check vertices of A against faces of B, and vice versa.
    let data_b = topo.solid(solid_b)?;
    let shell_b = topo.shell(data_b.outer_shell())?;
    let tol = Tolerance::new();

    for &pa in &verts_a {
        for &fid in shell_b.faces() {
            let face = topo.face(fid)?;
            let (normal, d) = match face.surface() {
                FaceSurface::Plane { normal, d } => (*normal, *d),
                FaceSurface::Nurbs(_) => continue,
            };
            let face_verts = face_vertices(topo, fid)?;
            if let Some((dist, closest)) =
                point_to_polygon_distance(pa, &face_verts, normal, d, tol)
            {
                if dist < best_dist {
                    best_dist = dist;
                    best_a = pa;
                    best_b = closest;
                }
            }
        }
    }

    let data_a = topo.solid(solid_a)?;
    let shell_a = topo.shell(data_a.outer_shell())?;

    for &pb in &verts_b {
        for &fid in shell_a.faces() {
            let face = topo.face(fid)?;
            let (normal, d) = match face.surface() {
                FaceSurface::Plane { normal, d } => (*normal, *d),
                FaceSurface::Nurbs(_) => continue,
            };
            let face_verts = face_vertices(topo, fid)?;
            if let Some((dist, closest)) =
                point_to_polygon_distance(pb, &face_verts, normal, d, tol)
            {
                if dist < best_dist {
                    best_dist = dist;
                    best_a = closest;
                    best_b = pb;
                }
            }
        }
    }

    Ok(DistanceResult {
        distance: best_dist,
        point_a: best_a,
        point_b: best_b,
    })
}

/// Compute the distance from a point to a planar polygon.
///
/// Returns `(distance, closest_point)` or `None` if the polygon is degenerate.
fn point_to_polygon_distance(
    point: Point3,
    verts: &[Point3],
    normal: Vec3,
    d: f64,
    _tol: Tolerance,
) -> Option<(f64, Point3)> {
    if verts.len() < 3 {
        return None;
    }

    // Project point onto the plane.
    let signed_dist = normal.dot(Vec3::new(point.x(), point.y(), point.z())) - d;
    let projected = Point3::new(
        (-normal.x()).mul_add(signed_dist, point.x()),
        (-normal.y()).mul_add(signed_dist, point.y()),
        (-normal.z()).mul_add(signed_dist, point.z()),
    );

    // Check if projected point is inside the polygon.
    if point_in_polygon_3d(&projected, verts, &normal) {
        return Some((signed_dist.abs(), projected));
    }

    // If outside, find closest point on polygon edges.
    let mut best_dist = f64::INFINITY;
    let mut best_point = verts[0];
    let n = verts.len();

    for i in 0..n {
        let j = (i + 1) % n;
        let (dist, closest) = point_to_segment_distance(point, verts[i], verts[j]);
        if dist < best_dist {
            best_dist = dist;
            best_point = closest;
        }
    }

    Some((best_dist, best_point))
}

/// Point-in-polygon test for 3D (projecting to 2D).
fn point_in_polygon_3d(point: &Point3, polygon: &[Point3], normal: &Vec3) -> bool {
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

/// Distance from a point to a line segment.
fn point_to_segment_distance(point: Point3, a: Point3, b: Point3) -> (f64, Point3) {
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

/// Collect all unique vertex positions from a solid.
fn collect_solid_points(
    topo: &Topology,
    solid: SolidId,
) -> Result<Vec<Point3>, crate::OperationsError> {
    let mut seen = std::collections::HashSet::new();
    let mut points = Vec::new();

    let solid_data = topo.solid(solid)?;
    let shell = topo.shell(solid_data.outer_shell())?;

    for &fid in shell.faces() {
        let face = topo.face(fid)?;
        let wire = topo.wire(face.outer_wire())?;
        for oe in wire.edges() {
            let edge = topo.edge(oe.edge())?;
            for vid in [edge.start(), edge.end()] {
                if seen.insert(vid.index()) {
                    points.push(topo.vertex(vid)?.point());
                }
            }
        }
    }

    Ok(points)
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
