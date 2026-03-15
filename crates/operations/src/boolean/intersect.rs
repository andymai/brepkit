//! Phase 1-2: Intersection computation for the boolean pipeline.
//!
//! Computes intersection curves between face pairs of two solids, producing
//! `IntersectionSegment`s that drive the face-splitting phase.

use std::collections::HashSet;

use brepkit_math::aabb::Aabb3;
use brepkit_math::bvh::Bvh;
use brepkit_math::obb::Obb3;
use brepkit_math::plane::plane_plane_intersection;
use brepkit_math::tolerance::Tolerance;
use brepkit_math::vec::{Point3, Vec3};
use brepkit_topology::Topology;
use brepkit_topology::face::{FaceId, FaceSurface};
use brepkit_topology::solid::SolidId;

use super::types::{FaceData, FacePairSide, IntersectionSegment};

// ---------------------------------------------------------------------------
// Phase 1a: Analytic intersection segments
// ---------------------------------------------------------------------------

/// Try analytic (closed-form) intersection for plane+analytic face pairs.
///
/// Returns intersection segments and the set of original `(face_a_idx, face_b_idx)`
/// pairs that were handled, so the tessellated path can skip them.
///
/// This is the "fast path" for booleans involving analytic solids: a box-sphere
/// boolean only needs 6 plane-sphere tests (each O(1)) instead of ~5000
/// triangle-triangle tests.
#[allow(clippy::too_many_lines, clippy::type_complexity)]
pub(super) fn compute_analytic_segments(
    topo: &Topology,
    solid_a: SolidId,
    solid_b: SolidId,
    tol: Tolerance,
) -> Result<(Vec<IntersectionSegment>, HashSet<(usize, usize)>), crate::OperationsError> {
    let mut segments = Vec::new();
    let mut handled = HashSet::new();

    // Collect original face IDs + surfaces for both solids.
    let faces_a = collect_original_faces(topo, solid_a)?;
    let faces_b = collect_original_faces(topo, solid_b)?;

    for &(fid_a, ref surf_a) in &faces_a {
        for &(fid_b, ref surf_b) in &faces_b {
            // Try plane (A) + analytic (B).
            if let Some(segs) = try_plane_analytic_pair(fid_a, surf_a, fid_b, surf_b, tol) {
                segments.extend(segs);
                handled.insert((fid_a.index(), fid_b.index()));
                continue;
            }
            // Try plane (B) + analytic (A).
            if let Some(segs) = try_plane_analytic_pair(fid_b, surf_b, fid_a, surf_a, tol) {
                // Note: segments store face_a/face_b in the order the pair was tested.
                // Re-tag with correct face IDs.
                for seg in segs {
                    segments.push(IntersectionSegment {
                        face_a: fid_a,
                        face_b: fid_b,
                        p0: seg.p0,
                        p1: seg.p1,
                    });
                }
                handled.insert((fid_a.index(), fid_b.index()));
            }
        }
    }

    Ok((segments, handled))
}

/// Collect the original `(FaceId, FaceSurface)` pairs for a solid's outer shell.
fn collect_original_faces(
    topo: &Topology,
    solid_id: SolidId,
) -> Result<Vec<(FaceId, FaceSurface)>, crate::OperationsError> {
    let solid = topo.solid(solid_id)?;
    let shell = topo.shell(solid.outer_shell())?;
    let mut result = Vec::with_capacity(shell.faces().len());
    for &fid in shell.faces() {
        let face = topo.face(fid)?;
        result.push((fid, face.surface().clone()));
    }
    Ok(result)
}

/// Try to compute intersection segments for a plane + analytic surface pair.
///
/// Returns `None` if the pair isn't plane + analytic, or if the closed-form
/// intersection fails. The returned segments have `face_a = plane_fid` and
/// `face_b = analytic_fid`.
#[allow(clippy::cast_precision_loss)]
fn try_plane_analytic_pair(
    plane_fid: FaceId,
    plane_surf: &FaceSurface,
    analytic_fid: FaceId,
    analytic_surf: &FaceSurface,
    tol: Tolerance,
) -> Option<Vec<IntersectionSegment>> {
    use brepkit_math::analytic_intersection::sample_plane_analytic;

    // Extract plane normal + d.
    let (normal, d) = match plane_surf {
        FaceSurface::Plane { normal, d } => (*normal, *d),
        _ => return None,
    };

    let analytic = analytic_surf.as_analytic()?;

    // Get sample points without NURBS curve fitting.
    let chains = sample_plane_analytic(analytic, normal, d).ok()?;

    // Convert sampled point chains to consecutive segments.
    let mut segments = Vec::new();
    for chain in &chains {
        if chain.len() < 2 {
            continue;
        }
        for window in chain.windows(2) {
            let p0 = window[0];
            let p1 = window[1];
            // Skip degenerate segments.
            let dx = p1.x() - p0.x();
            let dy = p1.y() - p0.y();
            let dz = p1.z() - p0.z();
            if dx * dx + dy * dy + dz * dz < tol.linear * tol.linear {
                continue;
            }
            segments.push(IntersectionSegment {
                face_a: plane_fid,
                face_b: analytic_fid,
                p0,
                p1,
            });
        }
    }

    if segments.is_empty() {
        None
    } else {
        Some(segments)
    }
}

// ---------------------------------------------------------------------------
// Phase 1b: Tessellated intersection segments
// ---------------------------------------------------------------------------

/// Compute all intersection segments between face pairs of two solids.
///
/// Uses a BVH over solid B's faces for O(n log m) broad-phase filtering
/// instead of brute-force O(n * m).
///
/// Face pairs in `skip_pairs` (original face IDs handled by analytic path)
/// are excluded from tessellated intersection.
pub(super) fn compute_intersection_segments(
    faces_a: &FaceData,
    faces_b: &FaceData,
    tol: Tolerance,
    skip_pairs: &HashSet<(usize, usize)>,
) -> Vec<IntersectionSegment> {
    let mut segments = Vec::new();

    // Build BVH over solid B's faces (AABB broad phase).
    let b_entries: Vec<(usize, Aabb3)> = faces_b
        .iter()
        .enumerate()
        .map(|(i, (_, verts, _, _))| (i, Aabb3::from_points(verts.iter().copied())))
        .collect();
    let bvh = Bvh::build(&b_entries);

    // Pre-compute OBBs for tighter secondary filtering.
    // For planar faces, use normal-guided OBB (near-zero thickness);
    // for others, use PCA-based OBB.
    let obbs_b: Vec<Obb3> = faces_b
        .iter()
        .map(|(_, verts, normal, _)| Obb3::from_points_with_normal(verts.iter().copied(), *normal))
        .collect();

    for &(fid_a, ref verts_a, n_a, d_a) in faces_a {
        let aabb_a = Aabb3::from_points(verts_a.iter().copied());
        let obb_a = Obb3::from_points_with_normal(verts_a.iter().copied(), n_a);
        let candidates = bvh.query_overlap(&aabb_a);

        for &b_idx in &candidates {
            // OBB-OBB secondary filter: reject false positives from loose AABBs.
            if !obb_a.intersects(&obbs_b[b_idx]) {
                continue;
            }

            let (fid_b, ref verts_b, n_b, d_b) = faces_b[b_idx];

            // Skip face pairs already handled by the analytic fast path.
            if skip_pairs.contains(&(fid_a.index(), fid_b.index())) {
                continue;
            }

            let side_a = FacePairSide {
                fid: fid_a,
                verts: verts_a,
                normal: n_a,
                d: d_a,
            };
            let side_b = FacePairSide {
                fid: fid_b,
                verts: verts_b,
                normal: n_b,
                d: d_b,
            };

            segments.extend(intersect_face_pair(&side_a, &side_b, tol));
        }
    }

    segments
}

/// Intersect two planar face polygons. Returns intersection segments for all
/// overlapping intervals (handles concave polygons correctly).
fn intersect_face_pair(
    a: &FacePairSide<'_>,
    b: &FacePairSide<'_>,
    tol: Tolerance,
) -> Vec<IntersectionSegment> {
    // Plane-plane intersection line.
    let Some((line_pt, line_dir)) =
        plane_plane_intersection(a.normal, a.d, b.normal, b.d, tol.linear)
    else {
        return Vec::new();
    };

    // Clip against both polygons (multi-interval for concave support).
    let intervals_a = polygon_clip_intervals(&line_pt, &line_dir, a.verts, &a.normal, tol);
    let intervals_b = polygon_clip_intervals(&line_pt, &line_dir, b.verts, &b.normal, tol);

    // Intersect the two interval lists.
    let overlaps = intersect_interval_lists(&intervals_a, &intervals_b, tol.linear);

    overlaps
        .into_iter()
        .filter(|(lo, hi)| lo.is_finite() && hi.is_finite())
        .map(|(t_min, t_max)| IntersectionSegment {
            face_a: a.fid,
            face_b: b.fid,
            p0: point_along_line(&line_pt, &line_dir, t_min),
            p1: point_along_line(&line_pt, &line_dir, t_max),
        })
        .collect()
}

// ---------------------------------------------------------------------------
// Geometric helpers
// ---------------------------------------------------------------------------

/// Helper: `point + dir * t` as a `Point3`.
pub(super) fn point_along_line(pt: &Point3, dir: &Vec3, t: f64) -> Point3 {
    Point3::new(
        dir.x().mul_add(t, pt.x()),
        dir.y().mul_add(t, pt.y()),
        dir.z().mul_add(t, pt.z()),
    )
}

/// Clip a line against a polygon, returning all inside intervals.
///
/// The line is `P(t) = line_pt + t * line_dir`. The polygon lies on a plane
/// with normal `face_normal`. Returns a sorted list of `(t_enter, t_exit)`
/// intervals where the line is inside the polygon.
///
/// Unlike Cyrus-Beck (which only handles convex polygons), this computes
/// actual finite edge crossings and uses midpoint winding-number tests to
/// support concave polygons.
pub(super) fn polygon_clip_intervals(
    line_pt: &Point3,
    line_dir: &Vec3,
    polygon: &[Point3],
    face_normal: &Vec3,
    tol: Tolerance,
) -> Vec<(f64, f64)> {
    let n = polygon.len();
    if n < 3 {
        return Vec::new();
    }

    // Project everything to 2D by dropping the dominant normal axis.
    let (ax1, ax2) = dominant_projection_axes(face_normal);

    let proj = |p: Point3| -> (f64, f64) {
        (
            p.x() * ax1.x() + p.y() * ax1.y() + p.z() * ax1.z(),
            p.x() * ax2.x() + p.y() * ax2.y() + p.z() * ax2.z(),
        )
    };

    let (lx, ly) = proj(*line_pt);
    let dx = line_dir.x() * ax1.x() + line_dir.y() * ax1.y() + line_dir.z() * ax1.z();
    let dy = line_dir.x() * ax2.x() + line_dir.y() * ax2.y() + line_dir.z() * ax2.z();

    // Collect crossing parameters where the line crosses *finite* polygon edges.
    let mut crossings: Vec<f64> = Vec::with_capacity(n);

    for i in 0..n {
        let j = (i + 1) % n;
        let (ex, ey) = proj(polygon[i]);
        let (fx, fy) = proj(polygon[j]);

        // Edge direction in 2D
        let edx = fx - ex;
        let edy = fy - ey;

        // Solve: line_pt_2d + t * dir_2d = edge_pt + s * edge_dir
        // → t * dx - s * edx = ex - lx
        // → t * dy - s * edy = ey - ly
        let det = dx * (-edy) - dy * (-edx);
        // Numerical-zero guard: 1e-15 catches parallel/near-parallel edges.
        // For unit-scale coordinates, det = |dir|*|edge|*sin(angle); at 1e-15
        // the angle is negligible and the intersection point is numerically
        // undefined.
        if det.abs() < 1e-15 {
            continue; // Parallel
        }

        let rhs_x = ex - lx;
        let rhs_y = ey - ly;
        let t = (rhs_x * (-edy) - rhs_y * (-edx)) / det;
        let s = (dx * rhs_y - dy * rhs_x) / det;

        // The crossing must be within the finite edge: s ∈ [0, 1].
        if s >= -tol.linear && s <= 1.0 + tol.linear {
            crossings.push(t);
        }
    }

    if crossings.is_empty() {
        if point_in_polygon_3d(line_pt, polygon, face_normal) {
            return vec![(f64::NEG_INFINITY, f64::INFINITY)];
        }
        return Vec::new();
    }

    crossings.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    crossings.dedup_by(|a, b| (*a - *b).abs() < tol.linear);

    // Build intervals by testing midpoints between consecutive crossings.
    let mut intervals = Vec::new();

    // Test before first crossing
    let t_before = crossings[0] - 1.0;
    let p_before = point_along_line(line_pt, line_dir, t_before);
    if point_in_polygon_3d(&p_before, polygon, face_normal) {
        intervals.push((f64::NEG_INFINITY, crossings[0]));
    }

    for w in crossings.windows(2) {
        let t_mid = (w[0] + w[1]) * 0.5;
        let p_mid = point_along_line(line_pt, line_dir, t_mid);
        if point_in_polygon_3d(&p_mid, polygon, face_normal) {
            intervals.push((w[0], w[1]));
        }
    }

    // Test after last crossing
    let t_after = crossings[crossings.len() - 1] + 1.0;
    let p_after = point_along_line(line_pt, line_dir, t_after);
    if point_in_polygon_3d(&p_after, polygon, face_normal) {
        intervals.push((crossings[crossings.len() - 1], f64::INFINITY));
    }

    intervals
}

/// Test if a 3D point lies inside a 3D polygon (both on the same plane).
///
/// Projects to 2D using the face normal, then uses winding number.
pub(super) fn point_in_polygon_3d(pt: &Point3, polygon: &[Point3], normal: &Vec3) -> bool {
    // Choose projection axes: drop the component aligned with the dominant normal axis.
    let (ax1, ax2) = dominant_projection_axes(normal);

    let px = pt.x() * ax1.x() + pt.y() * ax1.y() + pt.z() * ax1.z();
    let py = pt.x() * ax2.x() + pt.y() * ax2.y() + pt.z() * ax2.z();

    let mut winding = 0i32;
    let n = polygon.len();
    for i in 0..n {
        let j = (i + 1) % n;
        let yi = polygon[i].x() * ax2.x() + polygon[i].y() * ax2.y() + polygon[i].z() * ax2.z();
        let yj = polygon[j].x() * ax2.x() + polygon[j].y() * ax2.y() + polygon[j].z() * ax2.z();

        if yi <= py {
            if yj > py {
                let xi =
                    polygon[i].x() * ax1.x() + polygon[i].y() * ax1.y() + polygon[i].z() * ax1.z();
                let xj =
                    polygon[j].x() * ax1.x() + polygon[j].y() * ax1.y() + polygon[j].z() * ax1.z();
                if cross_2d(xi - px, yi - py, xj - px, yj - py) > 0.0 {
                    winding += 1;
                }
            }
        } else if yj <= py {
            let xi = polygon[i].x() * ax1.x() + polygon[i].y() * ax1.y() + polygon[i].z() * ax1.z();
            let xj = polygon[j].x() * ax1.x() + polygon[j].y() * ax1.y() + polygon[j].z() * ax1.z();
            if cross_2d(xi - px, yi - py, xj - px, yj - py) < 0.0 {
                winding -= 1;
            }
        }
    }
    winding != 0
}

/// Pick two axes for projecting a 3D polygon to 2D by dropping the dominant
/// normal component.
fn dominant_projection_axes(normal: &Vec3) -> (Vec3, Vec3) {
    let ax = normal.x().abs();
    let ay = normal.y().abs();
    let az = normal.z().abs();
    if az >= ax && az >= ay {
        // Drop Z
        (Vec3::new(1.0, 0.0, 0.0), Vec3::new(0.0, 1.0, 0.0))
    } else if ay >= ax {
        // Drop Y
        (Vec3::new(1.0, 0.0, 0.0), Vec3::new(0.0, 0.0, 1.0))
    } else {
        // Drop X
        (Vec3::new(0.0, 1.0, 0.0), Vec3::new(0.0, 0.0, 1.0))
    }
}

/// 2D cross product (determinant).
fn cross_2d(ax: f64, ay: f64, bx: f64, by: f64) -> f64 {
    ax * by - ay * bx
}

/// Intersect two sorted lists of intervals, returning all overlapping sub-intervals.
pub(super) fn intersect_interval_lists(
    a: &[(f64, f64)],
    b: &[(f64, f64)],
    tol: f64,
) -> Vec<(f64, f64)> {
    let mut result = Vec::new();
    let mut ai = 0;
    let mut bi = 0;
    while ai < a.len() && bi < b.len() {
        let lo = a[ai].0.max(b[bi].0);
        let hi = a[ai].1.min(b[bi].1);
        if hi - lo > tol {
            result.push((lo, hi));
        }
        // Advance the interval that ends first.
        if a[ai].1 < b[bi].1 {
            ai += 1;
        } else {
            bi += 1;
        }
    }
    result
}

/// Cyrus-Beck clipping of a line against a convex polygon (single interval).
///
/// The line is `P(t) = line_pt + t * line_dir`. The polygon lies on a plane
/// with normal `face_normal`. Returns `(t_min, t_max)` of the segment inside
/// the polygon, or `None` if the line doesn't cross the polygon.
///
/// Only correct for convex polygons. For concave polygons, use
/// [`polygon_clip_intervals`] which returns multiple intervals.
pub(super) fn cyrus_beck_clip(
    line_pt: &Point3,
    line_dir: &Vec3,
    polygon: &[Point3],
    face_normal: &Vec3,
    tol: Tolerance,
) -> Option<(f64, f64)> {
    let n = polygon.len();
    if n < 3 {
        return None;
    }

    let mut t_enter = f64::NEG_INFINITY;
    let mut t_exit = f64::INFINITY;

    for i in 0..n {
        let j = (i + 1) % n;
        let edge_vec = polygon[j] - polygon[i];
        let edge_normal = face_normal.cross(edge_vec);

        let w = *line_pt - polygon[i];
        let denom = edge_normal.dot(*line_dir);
        let numer = -edge_normal.dot(w);

        if denom.abs() < tol.angular {
            if edge_normal.dot(w) < 0.0 {
                return None;
            }
            continue;
        }

        let t = numer / denom;
        if denom > 0.0 {
            t_enter = t_enter.max(t);
        } else {
            t_exit = t_exit.min(t);
        }

        if t_enter > t_exit {
            return None;
        }
    }

    if t_enter > t_exit {
        None
    } else {
        Some((t_enter, t_exit))
    }
}
