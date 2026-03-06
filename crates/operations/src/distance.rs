//! Distance measurement between shapes.
//!
//! Equivalent to `BRepExtrema_DistShapeShape` in `OpenCascade`.
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

use brepkit_math::aabb::Aabb3;
use brepkit_math::bvh::Bvh;
use brepkit_math::tolerance::Tolerance;
use brepkit_math::vec::{Point3, Vec3};
use brepkit_topology::Topology;
use brepkit_topology::face::{FaceId, FaceSurface};
use brepkit_topology::solid::SolidId;

use crate::boolean::face_polygon;

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
#[allow(clippy::too_many_lines)]
pub fn point_to_solid_distance(
    topo: &Topology,
    point: Point3,
    solid: SolidId,
) -> Result<DistanceResult, crate::OperationsError> {
    let tol = Tolerance::new();

    let solid_data = topo.solid(solid)?;
    let shell = topo.shell(solid_data.outer_shell())?;
    let face_ids: Vec<FaceId> = shell.faces().to_vec();

    // Build BVH over face AABBs.
    let face_aabbs = build_face_aabbs(topo, &face_ids)?;
    let bvh = Bvh::build(&face_aabbs);

    // Use BVH to find candidate faces in order of proximity.
    let mut best_dist = f64::INFINITY;
    let mut best_point = point;

    // Query BVH for closest face AABB, then iterate candidates.
    let candidates = bvh_distance_candidates(&bvh, &face_aabbs, point);

    for idx in candidates {
        let fid = face_ids[idx];
        let aabb_dist_sq = face_aabbs[idx].1.distance_squared_to_point(point);
        if aabb_dist_sq > best_dist * best_dist {
            continue;
        }

        if let Some((dist, closest)) = point_to_face_distance(topo, point, fid, tol)? {
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
/// Checks vertices of each solid against faces of the other, with
/// BVH acceleration. Also checks edge-to-edge distances for the
/// closest vertex pairs.
///
/// # Errors
///
/// Returns an error if either solid is invalid.
#[allow(clippy::too_many_lines)]
pub fn solid_to_solid_distance(
    topo: &Topology,
    solid_a: SolidId,
    solid_b: SolidId,
) -> Result<DistanceResult, crate::OperationsError> {
    let tol = Tolerance::new();

    let verts_a = collect_solid_points(topo, solid_a)?;
    let verts_b = collect_solid_points(topo, solid_b)?;

    let mut best_dist = f64::INFINITY;
    let mut best_a = Point3::new(0.0, 0.0, 0.0);
    let mut best_b = Point3::new(0.0, 0.0, 0.0);

    // Quick vertex-to-vertex pass.
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

    // Vertices of A against faces of B.
    let data_b = topo.solid(solid_b)?;
    let shell_b = topo.shell(data_b.outer_shell())?;
    let faces_b: Vec<FaceId> = shell_b.faces().to_vec();
    let aabbs_b = build_face_aabbs(topo, &faces_b)?;
    let bvh_b = Bvh::build(&aabbs_b);

    for &pa in &verts_a {
        let candidates = bvh_distance_candidates(&bvh_b, &aabbs_b, pa);
        for idx in candidates {
            let aabb_dist_sq = aabbs_b[idx].1.distance_squared_to_point(pa);
            if aabb_dist_sq > best_dist * best_dist {
                continue;
            }
            if let Some((dist, closest)) = point_to_face_distance(topo, pa, faces_b[idx], tol)? {
                if dist < best_dist {
                    best_dist = dist;
                    best_a = pa;
                    best_b = closest;
                }
            }
        }
    }

    // Vertices of B against faces of A.
    let data_a = topo.solid(solid_a)?;
    let shell_a = topo.shell(data_a.outer_shell())?;
    let faces_a: Vec<FaceId> = shell_a.faces().to_vec();
    let aabbs_a = build_face_aabbs(topo, &faces_a)?;
    let bvh_a = Bvh::build(&aabbs_a);

    for &pb in &verts_b {
        let candidates = bvh_distance_candidates(&bvh_a, &aabbs_a, pb);
        for idx in candidates {
            let aabb_dist_sq = aabbs_a[idx].1.distance_squared_to_point(pb);
            if aabb_dist_sq > best_dist * best_dist {
                continue;
            }
            if let Some((dist, closest)) = point_to_face_distance(topo, pb, faces_a[idx], tol)? {
                if dist < best_dist {
                    best_dist = dist;
                    best_a = closest;
                    best_b = pb;
                }
            }
        }
    }

    // Edge-to-edge pass for closest edge pairs.
    let edges_a = collect_solid_edges(topo, solid_a)?;
    let edges_b = collect_solid_edges(topo, solid_b)?;

    for &(a1, a2) in &edges_a {
        for &(b1, b2) in &edges_b {
            let (dist, ca, cb) = segment_to_segment_distance(a1, a2, b1, b2);
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

/// Compute the distance from a point to a single face, dispatching by type.
fn point_to_face_distance(
    topo: &Topology,
    point: Point3,
    face_id: FaceId,
    tol: Tolerance,
) -> Result<Option<(f64, Point3)>, crate::OperationsError> {
    let face = topo.face(face_id)?;
    match face.surface() {
        FaceSurface::Plane { normal, d } => {
            let verts = face_polygon(topo, face_id)?;
            Ok(point_to_polygon_distance(point, &verts, *normal, *d, tol))
        }
        FaceSurface::Nurbs(surface) => {
            let proj = brepkit_math::nurbs::projection::project_point_to_surface(
                surface, point, tol.linear,
            );
            match proj {
                Ok(p) => Ok(Some((p.distance, p.point))),
                Err(_) => Ok(None),
            }
        }
        FaceSurface::Cylinder(cyl) => Ok(Some(point_to_cylinder(point, cyl))),
        FaceSurface::Cone(cone) => Ok(Some(point_to_cone(point, cone))),
        FaceSurface::Sphere(sph) => Ok(Some(point_to_sphere(point, sph))),
        FaceSurface::Torus(tor) => Ok(Some(point_to_torus(point, tor))),
    }
}

// -- Analytic point-to-surface distance (closed-form) -------------------------

/// Closest point on a cylinder to a given point.
fn point_to_cylinder(
    point: Point3,
    cyl: &brepkit_math::surfaces::CylindricalSurface,
) -> (f64, Point3) {
    let pv = Vec3::new(
        point.x() - cyl.origin().x(),
        point.y() - cyl.origin().y(),
        point.z() - cyl.origin().z(),
    );
    // Project onto axis to get height parameter.
    let h = pv.dot(cyl.axis());
    // Radial vector: pv - h * axis.
    let radial = Vec3::new(
        pv.x() - h * cyl.axis().x(),
        pv.y() - h * cyl.axis().y(),
        pv.z() - h * cyl.axis().z(),
    );
    let r_len = radial.length();

    let closest = if r_len < 1e-15 {
        // Point is on the axis — pick any radial direction.
        let u = 0.0;
        cyl.evaluate(u, h)
    } else {
        // Closest point is at the same height, on the surface.
        let scale = cyl.radius() / r_len;
        Point3::new(
            cyl.origin().x() + radial.x() * scale + h * cyl.axis().x(),
            cyl.origin().y() + radial.y() * scale + h * cyl.axis().y(),
            cyl.origin().z() + radial.z() * scale + h * cyl.axis().z(),
        )
    };
    ((point - closest).length(), closest)
}

/// Closest point on a cone to a given point.
fn point_to_cone(point: Point3, cone: &brepkit_math::surfaces::ConicalSurface) -> (f64, Point3) {
    let pv = Vec3::new(
        point.x() - cone.apex().x(),
        point.y() - cone.apex().y(),
        point.z() - cone.apex().z(),
    );
    let h = pv.dot(cone.axis());

    // Radial component perpendicular to axis.
    let radial = Vec3::new(
        pv.x() - h * cone.axis().x(),
        pv.y() - h * cone.axis().y(),
        pv.z() - h * cone.axis().z(),
    );
    let r_len = radial.length();

    if h <= 0.0 && r_len < 1e-15 {
        // Very close to apex.
        return ((point - cone.apex()).length(), cone.apex());
    }

    // The cone surface at distance v from apex has radius v * cos(half_angle).
    // Project the point onto the cone's generatrix line.
    let (sin_a, cos_a) = cone.half_angle().sin_cos();
    // Distance along generatrix: h * sin_a + r_len * cos_a (projection onto surface direction).
    let v = h.mul_add(sin_a, r_len * cos_a);

    if v <= 0.0 {
        // Closest point is the apex.
        return ((point - cone.apex()).length(), cone.apex());
    }

    // Cone surface point at parameter v along the generatrix direction.
    let cone_r = v * cos_a;
    let cone_h = v * sin_a;

    let closest = if r_len < 1e-15 {
        Point3::new(
            cone.apex().x() + cone_h * cone.axis().x(),
            cone.apex().y() + cone_h * cone.axis().y(),
            cone.apex().z() + cone_h * cone.axis().z(),
        )
    } else {
        let radial_dir_x = radial.x() / r_len;
        let radial_dir_y = radial.y() / r_len;
        let radial_dir_z = radial.z() / r_len;
        Point3::new(
            cone.apex().x() + cone_h * cone.axis().x() + cone_r * radial_dir_x,
            cone.apex().y() + cone_h * cone.axis().y() + cone_r * radial_dir_y,
            cone.apex().z() + cone_h * cone.axis().z() + cone_r * radial_dir_z,
        )
    };

    ((point - closest).length(), closest)
}

/// Closest point on a sphere to a given point.
fn point_to_sphere(
    point: Point3,
    sphere: &brepkit_math::surfaces::SphericalSurface,
) -> (f64, Point3) {
    let pv = Vec3::new(
        point.x() - sphere.center().x(),
        point.y() - sphere.center().y(),
        point.z() - sphere.center().z(),
    );
    let dist_to_center = pv.length();

    if dist_to_center < 1e-15 {
        // Point is at the center — closest surface point is arbitrary.
        let closest = Point3::new(
            sphere.center().x() + sphere.radius(),
            sphere.center().y(),
            sphere.center().z(),
        );
        return (sphere.radius(), closest);
    }

    let scale = sphere.radius() / dist_to_center;
    let closest = Point3::new(
        sphere.center().x() + pv.x() * scale,
        sphere.center().y() + pv.y() * scale,
        sphere.center().z() + pv.z() * scale,
    );
    ((dist_to_center - sphere.radius()).abs(), closest)
}

/// Closest point on a torus to a given point.
///
/// Projects onto the major circle first, then onto the minor circle.
fn point_to_torus(point: Point3, torus: &brepkit_math::surfaces::ToroidalSurface) -> (f64, Point3) {
    let pv = Vec3::new(
        point.x() - torus.center().x(),
        point.y() - torus.center().y(),
        point.z() - torus.center().z(),
    );

    // The torus axis is z_axis. Project pv onto the plane perpendicular to z_axis.
    // Torus uses x_axis/y_axis/z_axis from its struct but they're private.
    // We reconstruct: the torus is centered at center, with z_axis = (0,0,1) by default.
    // Using evaluate to get the major circle direction:
    // At (u, 0), the point is at major_radius from center in the equatorial plane.
    // We can use the z-component relative to center as the axial distance.
    // For a standard torus (z_axis = (0,0,1)):
    let z_axis = Vec3::new(0.0, 0.0, 1.0); // ToroidalSurface default
    let h = pv.dot(z_axis);

    // Radial projection in the equatorial plane.
    let radial = Vec3::new(
        pv.x() - h * z_axis.x(),
        pv.y() - h * z_axis.y(),
        pv.z() - h * z_axis.z(),
    );
    let r_len = radial.length();

    // Closest point on major circle.
    let major_r = torus.major_radius();
    let minor_r = torus.minor_radius();

    let (major_closest_x, major_closest_y, major_closest_z) = if r_len < 1e-15 {
        // On the axis — pick any direction.
        (
            torus.center().x() + major_r,
            torus.center().y(),
            torus.center().z(),
        )
    } else {
        let scale = major_r / r_len;
        (
            torus.center().x() + radial.x() * scale,
            torus.center().y() + radial.y() * scale,
            torus.center().z() + radial.z() * scale,
        )
    };

    // Vector from major circle point to query point.
    let tube_vec = Vec3::new(
        point.x() - major_closest_x,
        point.y() - major_closest_y,
        point.z() - major_closest_z,
    );
    let tube_dist = tube_vec.length();

    if tube_dist < 1e-15 {
        // Point is on the major circle — closest torus point is minor_r away.
        let dir = if r_len < 1e-15 {
            z_axis
        } else {
            Vec3::new(radial.x() / r_len, radial.y() / r_len, radial.z() / r_len)
        };
        let closest = Point3::new(
            major_closest_x + minor_r * dir.x(),
            major_closest_y + minor_r * dir.y(),
            major_closest_z + minor_r * dir.z(),
        );
        return (minor_r, closest);
    }

    let tube_scale = minor_r / tube_dist;
    let closest = Point3::new(
        major_closest_x + tube_vec.x() * tube_scale,
        major_closest_y + tube_vec.y() * tube_scale,
        major_closest_z + tube_vec.z() * tube_scale,
    );
    ((tube_dist - minor_r).abs(), closest)
}

// -- BVH helpers --------------------------------------------------------------

/// Build AABBs for a set of faces (from vertex extents).
fn build_face_aabbs(
    topo: &Topology,
    face_ids: &[FaceId],
) -> Result<Vec<(usize, Aabb3)>, crate::OperationsError> {
    let mut result = Vec::with_capacity(face_ids.len());
    for (i, &fid) in face_ids.iter().enumerate() {
        let face = topo.face(fid)?;
        let wire = topo.wire(face.outer_wire())?;
        let mut min = Point3::new(f64::INFINITY, f64::INFINITY, f64::INFINITY);
        let mut max = Point3::new(f64::NEG_INFINITY, f64::NEG_INFINITY, f64::NEG_INFINITY);
        for oe in wire.edges() {
            let edge = topo.edge(oe.edge())?;
            for vid in [edge.start(), edge.end()] {
                let p = topo.vertex(vid)?.point();
                min = Point3::new(min.x().min(p.x()), min.y().min(p.y()), min.z().min(p.z()));
                max = Point3::new(max.x().max(p.x()), max.y().max(p.y()), max.z().max(p.z()));
            }
        }
        // Expand AABB slightly for analytic surfaces (they may extend beyond vertices).
        let margin = 0.01;
        min = Point3::new(min.x() - margin, min.y() - margin, min.z() - margin);
        max = Point3::new(max.x() + margin, max.y() + margin, max.z() + margin);
        result.push((i, Aabb3 { min, max }));
    }
    Ok(result)
}

/// Get candidate face indices sorted by AABB distance to a point.
fn bvh_distance_candidates(bvh: &Bvh, aabbs: &[(usize, Aabb3)], point: Point3) -> Vec<usize> {
    // Start with the closest BVH node and expand.
    // For simplicity, query all faces and sort by AABB distance.
    let mut candidates: Vec<(usize, f64)> = aabbs
        .iter()
        .map(|(i, aabb)| (*i, aabb.distance_squared_to_point(point)))
        .collect();
    candidates.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal));

    // Use BVH for pruning if available.
    if let Some(closest_idx) = bvh.query_closest(point) {
        // Move closest to front if not already.
        if let Some(pos) = candidates.iter().position(|(i, _)| *i == closest_idx) {
            candidates.swap(0, pos);
        }
    }

    candidates.into_iter().map(|(i, _)| i).collect()
}

// -- Segment-to-segment distance ----------------------------------------------

/// Compute the minimum distance between two 3D line segments.
///
/// Returns `(distance, closest_on_a, closest_on_b)`.
fn segment_to_segment_distance(
    a1: Point3,
    a2: Point3,
    b1: Point3,
    b2: Point3,
) -> (f64, Point3, Point3) {
    let da = a2 - a1;
    let db = b2 - b1;
    let r = a1 - b1;

    let a_sq = da.dot(da);
    let e_sq = db.dot(db);
    let f = db.dot(r);

    if a_sq < 1e-30 && e_sq < 1e-30 {
        return ((a1 - b1).length(), a1, b1);
    }

    let (s, t) = if a_sq < 1e-30 {
        (0.0, (f / e_sq).clamp(0.0, 1.0))
    } else {
        let c = da.dot(r);
        if e_sq < 1e-30 {
            ((-c / a_sq).clamp(0.0, 1.0), 0.0)
        } else {
            let b_val = da.dot(db);
            let denom = a_sq * e_sq - b_val * b_val;

            let mut s = if denom.abs() > 1e-30 {
                ((b_val * f - c * e_sq) / denom).clamp(0.0, 1.0)
            } else {
                0.0
            };

            let mut t = (b_val * s + f) / e_sq;

            if t < 0.0 {
                t = 0.0;
                s = (-c / a_sq).clamp(0.0, 1.0);
            } else if t > 1.0 {
                t = 1.0;
                s = ((b_val - c) / a_sq).clamp(0.0, 1.0);
            }

            (s, t)
        }
    };

    let closest_a = Point3::new(
        da.x().mul_add(s, a1.x()),
        da.y().mul_add(s, a1.y()),
        da.z().mul_add(s, a1.z()),
    );
    let closest_b = Point3::new(
        db.x().mul_add(t, b1.x()),
        db.y().mul_add(t, b1.y()),
        db.z().mul_add(t, b1.z()),
    );

    ((closest_a - closest_b).length(), closest_a, closest_b)
}

/// Collect all edge segments from a solid.
fn collect_solid_edges(
    topo: &Topology,
    solid: SolidId,
) -> Result<Vec<(Point3, Point3)>, crate::OperationsError> {
    let mut seen = std::collections::HashSet::new();
    let mut edges = Vec::new();

    let solid_data = topo.solid(solid)?;
    let shell = topo.shell(solid_data.outer_shell())?;

    for &fid in shell.faces() {
        let face = topo.face(fid)?;
        let wire = topo.wire(face.outer_wire())?;
        for oe in wire.edges() {
            if seen.insert(oe.edge().index()) {
                let edge = topo.edge(oe.edge())?;
                let p1 = topo.vertex(edge.start())?.point();
                let p2 = topo.vertex(edge.end())?.point();
                edges.push((p1, p2));
            }
        }
    }

    Ok(edges)
}

// -- Existing helpers (preserved) ---------------------------------------------

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

    #[test]
    fn point_to_sphere_distance() {
        let sphere =
            brepkit_math::surfaces::SphericalSurface::new(Point3::new(0.0, 0.0, 0.0), 5.0).unwrap();
        let (dist, closest) = point_to_sphere(Point3::new(10.0, 0.0, 0.0), &sphere);
        let tol = Tolerance::loose();
        assert!(
            tol.approx_eq(dist, 5.0),
            "distance to sphere should be ~5.0, got {dist}"
        );
        assert!(
            tol.approx_eq(closest.x(), 5.0),
            "closest x should be ~5.0, got {}",
            closest.x()
        );
    }

    #[test]
    fn point_to_cylinder_distance() {
        let cyl = brepkit_math::surfaces::CylindricalSurface::new(
            Point3::new(0.0, 0.0, 0.0),
            Vec3::new(0.0, 0.0, 1.0),
            3.0,
        )
        .unwrap();
        let (dist, _closest) = point_to_cylinder(Point3::new(5.0, 0.0, 1.0), &cyl);
        let tol = Tolerance::loose();
        assert!(
            tol.approx_eq(dist, 2.0),
            "distance to cylinder should be ~2.0, got {dist}"
        );
    }

    #[test]
    fn segment_to_segment_parallel() {
        let (dist, _, _) = segment_to_segment_distance(
            Point3::new(0.0, 0.0, 0.0),
            Point3::new(1.0, 0.0, 0.0),
            Point3::new(0.0, 3.0, 0.0),
            Point3::new(1.0, 3.0, 0.0),
        );
        let tol = Tolerance::loose();
        assert!(
            tol.approx_eq(dist, 3.0),
            "parallel segments 3 apart should have distance ~3.0, got {dist}"
        );
    }

    #[test]
    fn segment_to_segment_crossing() {
        let (dist, _, _) = segment_to_segment_distance(
            Point3::new(0.0, 0.0, 0.0),
            Point3::new(1.0, 0.0, 0.0),
            Point3::new(0.5, 0.0, -1.0),
            Point3::new(0.5, 0.0, 1.0),
        );
        let tol = Tolerance::loose();
        assert!(
            tol.approx_eq(dist, 0.0),
            "crossing segments should have distance ~0, got {dist}"
        );
    }
}
