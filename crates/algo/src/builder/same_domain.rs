//! Same-domain face detection and merging.
//!
//! When two faces from opposing solids share the same underlying surface
//! (coplanar planes, coincident cylinders, etc.), they are "same-domain"
//! faces. These need special handling:
//!
//! - **CoplanarSame**: normals point the same way — kept in fuse/intersect.
//! - **CoplanarOpposite**: normals point opposite — kept in cut (for B faces).

use std::collections::HashMap;
use std::hash::BuildHasher;

use brepkit_math::tolerance::Tolerance;
use brepkit_topology::Topology;
use brepkit_topology::face::{FaceId, FaceSurface};

use super::{FaceClass, SubFace};
use crate::ds::{GfaArena, Rank};

/// Detect same-domain face pairs and update their classification.
///
/// Scans all sub-faces looking for pairs where both faces lie on the
/// same geometric surface. Updates classification to `CoplanarSame`
/// or `CoplanarOpposite` based on normal alignment.
#[allow(clippy::too_many_lines)]
pub fn detect_same_domain<S: BuildHasher>(
    topo: &Topology,
    _arena: &GfaArena,
    sub_faces: &mut [SubFace],
    _face_ranks: &HashMap<FaceId, Rank, S>,
    tol: Tolerance,
) {
    let n = sub_faces.len();
    let mut same_domain_count = 0u32;

    // Collect face surfaces to avoid repeated lookups.
    let surfaces: Vec<Option<&FaceSurface>> = sub_faces
        .iter()
        .map(|sf| {
            topo.face(sf.face_id)
                .ok()
                .map(brepkit_topology::face::Face::surface)
        })
        .collect();

    // Check every pair of sub-faces from opposing solids.
    for i in 0..n {
        if sub_faces[i].classification != FaceClass::Unknown {
            continue;
        }
        let Some(surf_i) = surfaces[i] else {
            continue;
        };
        let rank_i = sub_faces[i].rank;

        for j in (i + 1)..n {
            if sub_faces[j].classification != FaceClass::Unknown {
                continue;
            }
            // Only compare faces from opposing solids
            if sub_faces[j].rank == rank_i {
                continue;
            }
            let Some(surf_j) = surfaces[j] else {
                continue;
            };

            // Check overlap between face boundary AABBs
            let overlaps =
                face_bboxes_overlap(topo, sub_faces[i].face_id, sub_faces[j].face_id, tol);
            if !overlaps {
                continue;
            }

            if let Some(same_dir) = surfaces_same_domain(surf_i, surf_j, tol) {
                // Verify actual area overlap — faces that only share an edge
                // (e.g., touching boxes) must NOT be classified as same-domain.
                if !faces_have_interior_overlap(
                    topo,
                    sub_faces[i].face_id,
                    sub_faces[j].face_id,
                    tol,
                ) {
                    continue;
                }
                if same_dir {
                    sub_faces[i].classification = FaceClass::CoplanarSame;
                    sub_faces[j].classification = FaceClass::CoplanarSame;
                } else {
                    sub_faces[i].classification = FaceClass::CoplanarOpposite;
                    sub_faces[j].classification = FaceClass::CoplanarOpposite;
                }
                same_domain_count += 1;
                break; // Each face can only be same-domain with one opposing face
            }
        }
    }

    log::debug!("detect_same_domain: {same_domain_count} same-domain pairs found");
}

/// Check if two face bounding boxes overlap (with tolerance expansion).
fn face_bboxes_overlap(topo: &Topology, a: FaceId, b: FaceId, tol: Tolerance) -> bool {
    let bbox = |fid: FaceId| -> Option<brepkit_math::aabb::Aabb3> {
        let face = topo.face(fid).ok()?;
        let wire = topo.wire(face.outer_wire()).ok()?;
        let mut points = Vec::new();
        for oe in wire.edges() {
            let e = topo.edge(oe.edge()).ok()?;
            let sp = topo.vertex(e.start()).ok()?.point();
            let ep = topo.vertex(e.end()).ok()?.point();
            points.push(sp);
            points.push(ep);
            // Curved edges can bulge beyond their endpoints
            if !matches!(e.curve(), brepkit_topology::edge::EdgeCurve::Line) {
                let (t0, t1) = e.curve().domain_with_endpoints(sp, ep);
                let t_mid = 0.5_f64.mul_add(t1 - t0, t0);
                points.push(e.curve().evaluate_with_endpoints(t_mid, sp, ep));
            }
        }
        brepkit_math::aabb::Aabb3::try_from_points(points)
    };
    let Some(ba) = bbox(a) else { return false };
    let Some(bb) = bbox(b) else { return false };
    ba.expanded(tol.linear).intersects(bb.expanded(tol.linear))
}

/// Check whether two coplanar faces overlap in interior area.
///
/// Projects both face boundaries to 2D in the shared plane, then checks
/// if any vertex of face A is inside face B's boundary polygon (or vice
/// versa). Faces that only share an edge (zero area overlap) return false.
///
/// For non-plane surfaces (cylinder, cone, etc.), always returns `true`
/// because surface-level coincidence implies parameter-space overlap.
fn faces_have_interior_overlap(topo: &Topology, a: FaceId, b: FaceId, tol: Tolerance) -> bool {
    use brepkit_math::vec::{Point2, Vec3};

    let face_a = match topo.face(a) {
        Ok(f) => f,
        Err(_) => return true, // Assume overlap if we can't check
    };

    // Only do the polygon check for plane faces — non-plane same-domain
    // faces (cylinders sharing an axis, etc.) genuinely overlap if they
    // share the same surface and overlapping AABBs.
    let normal = match face_a.surface() {
        FaceSurface::Plane { normal, .. } => *normal,
        _ => return true,
    };

    // Build a local 2D frame from the plane normal
    let ref_axis = if normal.x().abs() < 0.9 {
        Vec3::new(1.0, 0.0, 0.0)
    } else {
        Vec3::new(0.0, 1.0, 0.0)
    };
    let u_axis = ref_axis.cross(normal);
    let u_len = u_axis.length();
    if u_len < 1e-12 {
        return true;
    }
    let u_axis = u_axis * (1.0 / u_len);
    let v_axis = normal.cross(u_axis);

    let project = |p: brepkit_math::vec::Point3| -> Point2 {
        Point2::new(
            p.x() * u_axis.x() + p.y() * u_axis.y() + p.z() * u_axis.z(),
            p.x() * v_axis.x() + p.y() * v_axis.y() + p.z() * v_axis.z(),
        )
    };

    let poly_a = face_boundary_projected(topo, a, &project);
    let poly_b = face_boundary_projected(topo, b, &project);

    if poly_a.len() < 3 || poly_b.len() < 3 {
        return true; // Can't check — assume overlap
    }

    // Check polygon centroids first — handles the case where two identical
    // polygons fully overlap (all vertices are ON the boundary of the other,
    // so vertex-in-polygon returns false, but centroids are strictly inside).
    let centroid = |poly: &[brepkit_math::vec::Point2]| -> brepkit_math::vec::Point2 {
        let n = poly.len() as f64;
        let sx: f64 = poly.iter().map(|p| p.x()).sum();
        let sy: f64 = poly.iter().map(|p| p.y()).sum();
        brepkit_math::vec::Point2::new(sx / n, sy / n)
    };
    let ca = centroid(&poly_a);
    if point_strictly_inside_polygon(&ca, &poly_b, tol.linear) {
        return true;
    }
    let cb = centroid(&poly_b);
    if point_strictly_inside_polygon(&cb, &poly_a, tol.linear) {
        return true;
    }

    // Check if any vertex of A is strictly inside B's polygon
    for pt in &poly_a {
        if point_strictly_inside_polygon(pt, &poly_b, tol.linear) {
            return true;
        }
    }
    // Check if any vertex of B is strictly inside A's polygon
    for pt in &poly_b {
        if point_strictly_inside_polygon(pt, &poly_a, tol.linear) {
            return true;
        }
    }

    // Check if any edge of A crosses any edge of B in their interiors
    for i in 0..poly_a.len() {
        let a0 = &poly_a[i];
        let a1 = &poly_a[(i + 1) % poly_a.len()];
        for j in 0..poly_b.len() {
            let b0 = &poly_b[j];
            let b1 = &poly_b[(j + 1) % poly_b.len()];
            if segments_cross_interior(a0, a1, b0, b1, tol.linear) {
                return true;
            }
        }
    }

    false
}

/// Project a face's outer wire boundary vertices to 2D.
fn face_boundary_projected(
    topo: &Topology,
    face_id: FaceId,
    project: &dyn Fn(brepkit_math::vec::Point3) -> brepkit_math::vec::Point2,
) -> Vec<brepkit_math::vec::Point2> {
    let face = match topo.face(face_id) {
        Ok(f) => f,
        Err(_) => return Vec::new(),
    };
    let wire = match topo.wire(face.outer_wire()) {
        Ok(w) => w,
        Err(_) => return Vec::new(),
    };

    let mut pts = Vec::new();
    for oe in wire.edges() {
        let edge = match topo.edge(oe.edge()) {
            Ok(e) => e,
            Err(_) => continue,
        };
        let p = match topo.vertex(oe.oriented_start(edge)) {
            Ok(v) => v.point(),
            Err(_) => continue,
        };
        pts.push(project(p));
    }
    pts
}

/// Check if a point is strictly inside a polygon (not on boundary).
///
/// Uses ray-casting with a boundary distance check to distinguish interior
/// from edge-touching.
fn point_strictly_inside_polygon(
    pt: &brepkit_math::vec::Point2,
    poly: &[brepkit_math::vec::Point2],
    tol: f64,
) -> bool {
    let n = poly.len();

    // First check if point is ON any edge (within tolerance) — boundary, not interior.
    for i in 0..n {
        let a = &poly[i];
        let b = &poly[(i + 1) % n];
        if point_to_segment_dist_2d(pt, a, b) < tol {
            return false;
        }
    }

    // Ray-casting for interior test
    let mut inside = false;
    let px = pt.x();
    let py = pt.y();
    let mut j = n - 1;
    for i in 0..n {
        let yi = poly[i].y();
        let yj = poly[j].y();
        let xi = poly[i].x();
        let xj = poly[j].x();
        if ((yi > py) != (yj > py)) && (px < (xj - xi) * (py - yi) / (yj - yi) + xi) {
            inside = !inside;
        }
        j = i;
    }
    inside
}

/// Distance from a 2D point to a line segment.
fn point_to_segment_dist_2d(
    pt: &brepkit_math::vec::Point2,
    a: &brepkit_math::vec::Point2,
    b: &brepkit_math::vec::Point2,
) -> f64 {
    let dx = b.x() - a.x();
    let dy = b.y() - a.y();
    let len_sq = dx.mul_add(dx, dy * dy);
    if len_sq < 1e-30 {
        return ((pt.x() - a.x()).powi(2) + (pt.y() - a.y()).powi(2)).sqrt();
    }
    let t = ((pt.x() - a.x()) * dx + (pt.y() - a.y()) * dy) / len_sq;
    let t = t.clamp(0.0, 1.0);
    let proj_x = t.mul_add(dx, a.x());
    let proj_y = t.mul_add(dy, a.y());
    ((pt.x() - proj_x).powi(2) + (pt.y() - proj_y).powi(2)).sqrt()
}

/// Check if two 2D line segments cross in their interiors (not at endpoints).
fn segments_cross_interior(
    a0: &brepkit_math::vec::Point2,
    a1: &brepkit_math::vec::Point2,
    b0: &brepkit_math::vec::Point2,
    b1: &brepkit_math::vec::Point2,
    _tol: f64,
) -> bool {
    let d1x = a1.x() - a0.x();
    let d1y = a1.y() - a0.y();
    let d2x = b1.x() - b0.x();
    let d2y = b1.y() - b0.y();
    let det = d1x.mul_add(d2y, -(d1y * d2x));
    if det.abs() < 1e-15 {
        return false; // Parallel
    }
    let dx = b0.x() - a0.x();
    let dy = b0.y() - a0.y();
    let t = dx.mul_add(d2y, -(dy * d2x)) / det;
    let s = dx.mul_add(d1y, -(dy * d1x)) / det;
    // Strictly interior: t and s in (eps, 1-eps)
    let eps = 0.01;
    t > eps && t < 1.0 - eps && s > eps && s < 1.0 - eps
}

/// Check if two surfaces represent the same geometric domain.
///
/// Returns `Some(true)` for same-direction normals (CoplanarSame),
/// `Some(false)` for opposite normals (CoplanarOpposite), or
/// `None` if not the same domain.
fn surfaces_same_domain(a: &FaceSurface, b: &FaceSurface, tol: Tolerance) -> Option<bool> {
    match (a, b) {
        (FaceSurface::Plane { normal: na, d: da }, FaceSurface::Plane { normal: nb, d: db }) => {
            let dot = na.dot(*nb);
            if dot > 1.0 - tol.angular {
                // Same direction — check distance
                if (da - db).abs() < tol.linear {
                    return Some(true);
                }
            } else if dot < -1.0 + tol.angular {
                // Opposite direction — check distance
                if (da + db).abs() < tol.linear {
                    return Some(false);
                }
            }
            None
        }
        (FaceSurface::Cylinder(ca), FaceSurface::Cylinder(cb)) => {
            // Same cylinder: same origin, same axis, same radius
            if (ca.radius() - cb.radius()).abs() > tol.linear {
                return None;
            }
            let axis_dot = ca.axis().dot(cb.axis());
            if axis_dot.abs() < 1.0 - tol.angular {
                return None;
            }
            // Check if origins lie on the same axis line
            let diff = cb.origin() - ca.origin();
            let along_axis = diff.dot(ca.axis());
            let perp_dist = (diff - ca.axis() * along_axis).length();
            if perp_dist > tol.linear {
                return None;
            }
            Some(axis_dot > 0.0)
        }
        (FaceSurface::Sphere(sa), FaceSurface::Sphere(sb)) => {
            if (sa.radius() - sb.radius()).abs() > tol.linear {
                return None;
            }
            let dist = (sa.center() - sb.center()).length();
            if dist > tol.linear {
                return None;
            }
            // Spheres with same center/radius are always same-domain-same
            Some(true)
        }
        (FaceSurface::Cone(ca), FaceSurface::Cone(cb)) => {
            if (ca.half_angle() - cb.half_angle()).abs() > tol.angular {
                return None;
            }
            let axis_dot = ca.axis().dot(cb.axis());
            if axis_dot.abs() < 1.0 - tol.angular {
                return None;
            }
            let dist = (ca.apex() - cb.apex()).length();
            if dist > tol.linear {
                return None;
            }
            Some(axis_dot > 0.0)
        }
        _ => None,
    }
}
