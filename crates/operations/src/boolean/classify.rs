#![allow(dead_code)]
//! Phase 4: Classification — point-in-solid tests for boolean fragment labeling.
//!
//! Delegates analytic classifier construction to `brepkit_algo::classifier`
//! (canonical implementation) via `try_build_analytic_classifier`. This module
//! implements the ray-casting point-in-solid tests, coplanar guards, BVH
//! acceleration, and fragment-level classification as operations-specific
//! wrappers over pre-extracted face data.

use super::intersect::point_along_line;
use super::types::{FaceData, FaceFragment};

use brepkit_algo::FaceClass;
use brepkit_math::aabb::Aabb3;
use brepkit_math::bvh::Bvh;
use brepkit_math::predicates::point_in_polygon;
use brepkit_math::tolerance::Tolerance;
use brepkit_math::vec::{Point2, Point3, Vec3};

use crate::dot_normal_point;

// ---------------------------------------------------------------------------
// Re-exports from algo (canonical implementation)
// ---------------------------------------------------------------------------

pub(super) use brepkit_algo::classifier::try_build_analytic_classifier;

// ---------------------------------------------------------------------------
// BVH construction
// ---------------------------------------------------------------------------

/// Build a BVH over face data for accelerated ray-cast classification.
///
/// Returns `None` when the face count is small enough that linear scan is
/// faster than BVH construction + traversal overhead.
pub(super) fn build_face_bvh(faces: &FaceData) -> Option<Bvh> {
    // Only worth building for ≥32 faces (BVH overhead vs linear scan).
    if faces.len() < 32 {
        return None;
    }
    let aabbs: Vec<(usize, Aabb3)> = faces
        .iter()
        .enumerate()
        .map(|(i, (_, verts, _, _))| {
            let bb = Aabb3::from_points(verts.iter().copied());
            (i, bb)
        })
        .collect();
    Some(Bvh::build(&aabbs))
}

// ---------------------------------------------------------------------------
// Fragment classification
// ---------------------------------------------------------------------------

/// Classify a face fragment relative to the opposite solid.
///
/// When `bvh` is `Some`, uses BVH-accelerated ray queries instead of
/// linearly scanning all faces. This reduces classification from O(F) to
/// O(log F) per fragment.
pub(super) fn classify_fragment(
    frag: &FaceFragment,
    opposite: &FaceData,
    bvh: Option<&Bvh>,
    tol: Tolerance,
) -> FaceClass {
    let centroid = polygon_centroid(&frag.vertices);
    let class = classify_point(centroid, frag.normal, opposite, bvh, tol);
    guard_tangent_coplanar(class, &frag.vertices, frag.normal, opposite, bvh, tol)
}

/// Guard against false coplanar classifications at tangent touch points.
///
/// When a face touches a curved surface tangentially, the centroid may lie
/// exactly on the opposing surface with aligned normals, causing a false
/// Coplanar classification. This function verifies coplanar results by
/// checking whether most fragment vertices are also on the opposing face
/// planes. If fewer than half are, it's a tangent touch — re-classify via
/// ray-casting from a vertex that's clearly off the opposing surface.
pub(super) fn guard_tangent_coplanar(
    class: FaceClass,
    vertices: &[Point3],
    normal: Vec3,
    opposite: &FaceData,
    bvh: Option<&Bvh>,
    tol: Tolerance,
) -> FaceClass {
    if !matches!(class, FaceClass::CoplanarSame | FaceClass::CoplanarOpposite) || vertices.len() < 3
    {
        return class;
    }

    // Check how many vertices are on the same plane as any opposing face.
    // Use plane distance only (not polygon containment) to avoid false
    // negatives from vertices on polygon edges.
    let mut on_plane_count = 0usize;
    for v in vertices {
        let on_any_plane = opposite.iter().any(|(_, _verts, n_opp, d_opp)| {
            let dist = dot_normal_point(*n_opp, *v) - d_opp;
            dist.abs() < tol.linear * 10.0
        });
        if on_any_plane {
            on_plane_count += 1;
        }
    }

    // If most vertices are on an opposing face plane, this is likely a true
    // coplanar situation — keep the original classification. Only override
    // when very few vertices (at most 1) are on any opposing plane, which
    // indicates a tangent touch at a point or line.
    if on_plane_count <= 1 {
        // Find a vertex that's NOT on any opposing face for reliable ray-cast.
        for v in vertices {
            let on_any = opposite.iter().any(|(_, _verts, n_opp, d_opp)| {
                let dist = dot_normal_point(*n_opp, *v) - d_opp;
                dist.abs() < tol.linear * 10.0
            });
            if !on_any {
                return classify_point(*v, normal, opposite, bvh, tol);
            }
        }
        // All vertices near some opposing plane — use centroid ray-cast.
        let centroid = polygon_centroid(vertices);
        return multiray_classify(centroid, normal, opposite, bvh, tol);
    }

    class
}

/// Classify a point (centroid with a ray direction) relative to an opposite solid.
///
/// This is the core classification logic for pre-extracted face data, separated
/// from `FaceFragment` to avoid unnecessary cloning when the centroid/normal
/// are already known.
pub(super) fn classify_point(
    centroid: Point3,
    normal: Vec3,
    opposite: &FaceData,
    bvh: Option<&Bvh>,
    tol: Tolerance,
) -> FaceClass {
    // First check for coplanar faces — must scan candidates only.
    // For coplanar test we need faces near the centroid's plane, so use
    // BVH point-containment if available, otherwise linear scan.
    let coplanar_indices: Vec<usize> = if let Some(bvh) = bvh {
        let probe = Aabb3 {
            min: centroid + Vec3::new(-tol.linear, -tol.linear, -tol.linear),
            max: centroid + Vec3::new(tol.linear, tol.linear, tol.linear),
        };
        bvh.query_overlap(&probe)
    } else {
        (0..opposite.len()).collect()
    };

    for &i in &coplanar_indices {
        let (_, ref verts, n_opp, d_opp) = opposite[i];
        // Skip if the centroid coincides with a vertex of the opposing face.
        // This prevents false coplanar matches when the centroid is at a
        // singular point (e.g. cone apex, sphere pole) that is a vertex of
        // tessellated face data. At such vertices, the centroid lies on the
        // triangle plane (distance = 0) but the face is NOT truly coplanar.
        // Use a tight threshold (10× tolerance) to avoid interfering with
        // legitimate near-touching geometry.
        let near_vertex = verts.iter().any(|v| {
            let dx = centroid.x() - v.x();
            let dy = centroid.y() - v.y();
            let dz = centroid.z() - v.z();
            dx * dx + dy * dy + dz * dz < tol.linear * 10.0 * tol.linear * 10.0
        });
        if near_vertex {
            continue;
        }
        let dist = dot_normal_point(n_opp, centroid) - d_opp;
        if dist.abs() < tol.linear && point_in_face_3d(centroid, verts, &n_opp) {
            let dot = normal.dot(n_opp);
            return if dot > tol.angular {
                FaceClass::CoplanarSame
            } else if dot < -tol.angular {
                FaceClass::CoplanarOpposite
            } else {
                // Normals are nearly perpendicular — treat as non-coplanar.
                // Fall through to ray-cast classification.
                continue;
            };
        }
    }

    multiray_classify(centroid, normal, opposite, bvh, tol)
}

// ---------------------------------------------------------------------------
// Ray-casting helpers
// ---------------------------------------------------------------------------

/// Test a single face against a ray for crossing parity.
///
/// Returns +1 for a front-to-back crossing, -1 for back-to-front, or 0 for
/// no intersection (parallel, behind origin, or outside polygon).
#[inline]
fn ray_face_crossing(
    centroid: Point3,
    ray_dir: Vec3,
    verts: &[Point3],
    n_opp: Vec3,
    d_opp: f64,
    tol: Tolerance,
) -> i32 {
    let denom = n_opp.dot(ray_dir);
    if denom.abs() < tol.angular {
        return 0;
    }
    let numer = d_opp - dot_normal_point(n_opp, centroid);
    let t = numer / denom;
    if t <= tol.linear {
        return 0;
    }
    let hit = point_along_line(&centroid, &ray_dir, t);
    if point_in_face_3d(hit, verts, &n_opp) {
        if denom > 0.0 { -1 } else { 1 }
    } else {
        0
    }
}

/// Multi-ray inside/outside classification via majority vote.
///
/// Casts 3 rays (the given normal + two ~55° rotations) and counts boundary
/// crossings. Returns Inside if 2+ of 3 rays report an odd crossing count.
/// This is the shared implementation used by both `classify_point` and the
/// tangent-coplanar guard's fallback.
fn multiray_classify(
    point: Point3,
    normal: Vec3,
    opposite: &FaceData,
    bvh: Option<&Bvh>,
    tol: Tolerance,
) -> FaceClass {
    let ray_dirs = {
        let perp = if normal.x().abs() < 0.9 {
            Vec3::new(1.0, 0.0, 0.0)
        } else {
            Vec3::new(0.0, 1.0, 0.0)
        };
        let cross_vec = normal.cross(perp);
        let axis_len = cross_vec.length();
        // Numerical-zero guard: if the cross product has near-zero length,
        // the fragment normal is nearly parallel to the chosen perpendicular
        // vector — fall back to using the normal itself for all 3 ray
        // directions (degenerate but safe; the majority vote still works).
        if axis_len < 1e-12 {
            [normal, normal, normal]
        } else {
            let inv = 1.0 / axis_len;
            let axis = Vec3::new(
                cross_vec.x() * inv,
                cross_vec.y() * inv,
                cross_vec.z() * inv,
            );
            let rodrigues = |cos_a: f64, sin_a: f64| -> Vec3 {
                let dot = axis.dot(normal);
                let cross = axis.cross(normal);
                Vec3::new(
                    normal.x().mul_add(
                        cos_a,
                        cross.x().mul_add(sin_a, axis.x() * dot * (1.0 - cos_a)),
                    ),
                    normal.y().mul_add(
                        cos_a,
                        cross.y().mul_add(sin_a, axis.y() * dot * (1.0 - cos_a)),
                    ),
                    normal.z().mul_add(
                        cos_a,
                        cross.z().mul_add(sin_a, axis.z() * dot * (1.0 - cos_a)),
                    ),
                )
            };
            // cos(55°) ≈ 0.574, sin(55°) ≈ 0.819
            [normal, rodrigues(0.574, 0.819), rodrigues(0.574, -0.819)]
        }
    };

    // Reuse a single candidate buffer across all 3 ray queries to avoid
    // per-ray Vec allocation. The buffer grows on the first query and is
    // cleared+reused on subsequent queries.
    let mut inside_votes = 0u8;
    let mut candidates = Vec::new();
    for ray_dir in &ray_dirs {
        let mut crossings = 0i32;
        if let Some(bvh) = bvh {
            bvh.query_ray_into(point, *ray_dir, &mut candidates);
            for &idx in &candidates {
                let (_, ref verts, n_opp, d_opp) = opposite[idx];
                crossings += ray_face_crossing(point, *ray_dir, verts, n_opp, d_opp, tol);
            }
        } else {
            for &(_, ref verts, n_opp, d_opp) in opposite {
                crossings += ray_face_crossing(point, *ray_dir, verts, n_opp, d_opp, tol);
            }
        }
        if crossings != 0 {
            inside_votes += 1;
        }
    }

    if inside_votes >= 2 {
        FaceClass::Inside
    } else {
        FaceClass::Outside
    }
}

// ---------------------------------------------------------------------------
// Geometry helpers
// ---------------------------------------------------------------------------

/// Compute the centroid of a polygon.
///
/// Returns the origin if the polygon is empty (should not happen in
/// practice since fragments are filtered for `len >= 3`).
#[inline]
pub(super) fn polygon_centroid(vertices: &[Point3]) -> Point3 {
    if vertices.is_empty() {
        return Point3::new(0.0, 0.0, 0.0);
    }
    #[allow(clippy::cast_precision_loss)] // polygon vertex counts fit in f64
    let inv_n = 1.0 / vertices.len() as f64;
    let (sx, sy, sz) = vertices.iter().fold((0.0, 0.0, 0.0), |(ax, ay, az), v| {
        (ax + v.x(), ay + v.y(), az + v.z())
    });
    Point3::new(sx * inv_n, sy * inv_n, sz * inv_n)
}

/// Test if a 3D point lies inside a planar face polygon by projecting to 2D.
#[inline]
pub(super) fn point_in_face_3d(point: Point3, polygon: &[Point3], normal: &Vec3) -> bool {
    if polygon.len() < 3 {
        return false;
    }

    // Project to 2D by dropping the coordinate corresponding to the largest
    // normal component. This avoids degenerate projections.
    let ax = normal.x().abs();
    let ay = normal.y().abs();
    let az = normal.z().abs();

    let (project_point, project_polygon): (Point2, Vec<Point2>) = if az >= ax && az >= ay {
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

    point_in_polygon(project_point, &project_polygon)
}
