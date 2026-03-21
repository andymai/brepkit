//! Co-refinement mesh boolean operations on triangle meshes.
//!
//! Implements mesh booleans (fuse, cut, intersect) using the co-refinement
//! approach: compute exact triangle-triangle intersections, insert intersection
//! edges into both meshes, classify sub-triangles by winding number, and
//! assemble the result.
//!
//! This operates directly on [`TriangleMesh`] without requiring topology.

#![allow(clippy::tuple_array_conversions)]

use std::collections::HashMap;

use brepkit_math::aabb::Aabb3;
use brepkit_math::bvh::Bvh;
use brepkit_math::predicates::orient3d;
use brepkit_math::vec::{Point3, Vec3};

use crate::OperationsError;
use crate::boolean::BooleanOp;
use crate::tessellate::TriangleMesh;

// ---------------------------------------------------------------------------
// Public API
// ---------------------------------------------------------------------------

/// Result of a mesh boolean operation.
#[derive(Debug, Clone)]
pub struct MeshBooleanResult {
    /// The resulting triangle mesh.
    pub mesh: TriangleMesh,
}

/// Perform a mesh boolean operation between two triangle meshes.
///
/// Uses co-refinement: compute exact triangle-triangle intersections,
/// insert intersection edges into both meshes via constrained triangulation,
/// classify sub-triangles by winding number, and assemble the result.
///
/// # Errors
/// Returns an error if the operation cannot be completed (e.g. the
/// intersection of disjoint meshes is empty).
#[allow(clippy::too_many_lines)]
pub fn mesh_boolean(
    mesh_a: &TriangleMesh,
    mesh_b: &TriangleMesh,
    op: BooleanOp,
    tolerance: f64,
) -> Result<MeshBooleanResult, OperationsError> {
    // Step 1: BVH broad-phase
    let (bvh_b, _aabbs_b) = build_triangle_bvh(mesh_b);
    let pairs = find_intersecting_pairs(mesh_a, &bvh_b);

    // Step 2: Triangle-triangle intersection
    let intersections = compute_all_intersections(mesh_a, mesh_b, &pairs, tolerance);

    // Step 3: Split meshes by intersection edges
    let split_a = split_mesh_by_intersections(mesh_a, &intersections, true, tolerance);
    let split_b = split_mesh_by_intersections(mesh_b, &intersections, false, tolerance);

    // Step 4: Classify sub-triangles
    let classify_a = classify_triangles(&split_a, mesh_b, tolerance);
    let classify_b = classify_triangles(&split_b, mesh_a, tolerance);

    // Step 5: Assemble result
    let mesh = assemble_result(&split_a, &split_b, &classify_a, &classify_b, op);

    if mesh.positions.is_empty() {
        return Err(OperationsError::InvalidInput {
            reason: "mesh boolean produced empty result".into(),
        });
    }

    Ok(MeshBooleanResult { mesh })
}

// ---------------------------------------------------------------------------
// Step 1: BVH broad-phase
// ---------------------------------------------------------------------------

/// Build a BVH over a mesh's triangles, returning the BVH and per-triangle AABBs.
fn build_triangle_bvh(mesh: &TriangleMesh) -> (Bvh, Vec<Aabb3>) {
    let tri_count = mesh.indices.len() / 3;
    let mut aabbs = Vec::with_capacity(tri_count);
    let mut entries = Vec::with_capacity(tri_count);

    for i in 0..tri_count {
        let (v0, v1, v2) = get_triangle(mesh, i);
        let aabb = Aabb3::from_points([v0, v1, v2]);
        aabbs.push(aabb);
        entries.push((i, aabb));
    }

    let bvh = Bvh::build(&entries);
    (bvh, aabbs)
}

/// Find all potentially intersecting triangle pairs between mesh A and mesh B.
fn find_intersecting_pairs(mesh_a: &TriangleMesh, bvh_b: &Bvh) -> Vec<(usize, usize)> {
    let tri_count_a = mesh_a.indices.len() / 3;
    let mut pairs = Vec::new();

    for i in 0..tri_count_a {
        let (v0, v1, v2) = get_triangle(mesh_a, i);
        let aabb_a = Aabb3::from_points([v0, v1, v2]);
        let candidates = bvh_b.query_overlap(&aabb_a);
        for j in candidates {
            pairs.push((i, j));
        }
    }

    pairs
}

// ---------------------------------------------------------------------------
// Step 2: Triangle-triangle intersection
// ---------------------------------------------------------------------------

/// An intersection segment between two triangles.
#[derive(Debug, Clone)]
struct TriTriIntersection {
    /// Start point of the intersection segment.
    p0: Point3,
    /// End point of the intersection segment.
    p1: Point3,
    /// Index of the triangle in mesh A.
    tri_a: usize,
    /// Index of the triangle in mesh B.
    tri_b: usize,
}

/// Compute all triangle-triangle intersections for the given candidate pairs.
fn compute_all_intersections(
    mesh_a: &TriangleMesh,
    mesh_b: &TriangleMesh,
    pairs: &[(usize, usize)],
    tolerance: f64,
) -> Vec<TriTriIntersection> {
    let mut result = Vec::new();

    for &(tri_a, tri_b) in pairs {
        let (a0, a1, a2) = get_triangle(mesh_a, tri_a);
        let (b0, b1, b2) = get_triangle(mesh_b, tri_b);

        if let Some(mut isect) = intersect_triangles(a0, a1, a2, b0, b1, b2, tolerance) {
            isect.tri_a = tri_a;
            isect.tri_b = tri_b;
            result.push(isect);
        }
    }

    result
}

/// Compute the intersection segment between two triangles using exact predicates.
///
/// Uses Moller's triangle-triangle intersection algorithm with `orient3d`
/// predicates for robustness. Returns `None` if the triangles do not intersect
/// or only touch at a point.
#[allow(clippy::too_many_lines, clippy::similar_names)]
fn intersect_triangles(
    a0: Point3,
    a1: Point3,
    a2: Point3,
    b0: Point3,
    b1: Point3,
    b2: Point3,
    tolerance: f64,
) -> Option<TriTriIntersection> {
    // Classify vertices of B against the plane of A.
    let db0 = orient3d(a0, a1, a2, b0);
    let db1 = orient3d(a0, a1, a2, b1);
    let db2 = orient3d(a0, a1, a2, b2);

    // If all B vertices are on the same side, no intersection.
    if all_same_sign(db0, db1, db2, tolerance) {
        return None;
    }

    // Classify vertices of A against the plane of B.
    let da0 = orient3d(b0, b1, b2, a0);
    let da1 = orient3d(b0, b1, b2, a1);
    let da2 = orient3d(b0, b1, b2, a2);

    // If all A vertices are on the same side, no intersection.
    if all_same_sign(da0, da1, da2, tolerance) {
        return None;
    }

    // Compute the intersection line direction as the cross product of the two
    // triangle normals.
    let na = (a1 - a0).cross(a2 - a0);
    let nb = (b1 - b0).cross(b2 - b0);
    let line_dir = na.cross(nb);

    // Coplanar check: when |na × nb| ≈ 0, the triangles lie in the same plane.
    // The standard Möller algorithm degenerates here — dispatch to 2D overlap.
    let line_len_sq =
        line_dir.x() * line_dir.x() + line_dir.y() * line_dir.y() + line_dir.z() * line_dir.z();
    let na_len_sq = na.x() * na.x() + na.y() * na.y() + na.z() * na.z();
    let nb_len_sq = nb.x() * nb.x() + nb.y() * nb.y() + nb.z() * nb.z();
    // sin²(angle) threshold: tolerance² for angular comparison
    if line_len_sq < (tolerance * tolerance) * na_len_sq.max(nb_len_sq) {
        return intersect_coplanar_triangles(a0, a1, a2, b0, b1, b2, tolerance);
    }

    // Project onto the axis with the largest component for numerical stability.
    let ax = line_dir.x().abs();
    let ay = line_dir.y().abs();
    let az = line_dir.z().abs();

    let project = |p: Point3| -> f64 {
        if ax >= ay && ax >= az {
            p.x()
        } else if ay >= az {
            p.y()
        } else {
            p.z()
        }
    };

    // Compute the intervals of each triangle on the intersection line.
    let (ta_min, ta_max) = triangle_interval(a0, a1, a2, da0, da1, da2, &project)?;
    let (tb_min, tb_max) = triangle_interval(b0, b1, b2, db0, db1, db2, &project)?;

    // Overlap of the two intervals.
    let t_lo = ta_min.max(tb_min);
    let t_hi = ta_max.min(tb_max);

    if t_hi - t_lo < tolerance {
        return None;
    }

    // Reconstruct 3D points from parameter values along the intersection line.
    let tri_data = TriPlaneData {
        v: [a0, a1, a2],
        d: [da0, da1, da2],
    };
    let p0 = point_on_intersection_line(&tri_data, t_lo, &project);
    let p1 = point_on_intersection_line(&tri_data, t_hi, &project);

    Some(TriTriIntersection {
        p0,
        p1,
        tri_a: 0,
        tri_b: 0,
    })
}

/// Handle coplanar triangle-triangle intersection.
///
/// When two triangles lie in the same plane, the standard Möller algorithm
/// degenerates (na × nb = 0). Instead, project both triangles to 2D and
/// find intersection edges of their boundaries. Returns the longest edge
/// of the overlap as the intersection segment, or `None` if the triangles
/// don't overlap.
#[allow(clippy::too_many_lines)]
fn intersect_coplanar_triangles(
    a0: Point3,
    a1: Point3,
    a2: Point3,
    b0: Point3,
    b1: Point3,
    b2: Point3,
    tolerance: f64,
) -> Option<TriTriIntersection> {
    // Choose projection axis: drop the coordinate with the largest normal component.
    let na = (a1 - a0).cross(a2 - a0);
    let nax = na.x().abs();
    let nay = na.y().abs();
    let naz = na.z().abs();

    let to_2d = |p: Point3| -> (f64, f64) {
        if naz >= nax && naz >= nay {
            (p.x(), p.y())
        } else if nay >= nax {
            (p.x(), p.z())
        } else {
            (p.y(), p.z())
        }
    };

    let a2d = [to_2d(a0), to_2d(a1), to_2d(a2)];
    let b2d = [to_2d(b0), to_2d(b1), to_2d(b2)];

    let a3d = [a0, a1, a2];
    let b3d = [b0, b1, b2];

    // Find all edge-edge intersection points between the two triangles.
    let mut hits: Vec<Point3> = Vec::new();

    let a_edges: [(usize, usize); 3] = [(0, 1), (1, 2), (2, 0)];
    let b_edges: [(usize, usize); 3] = [(0, 1), (1, 2), (2, 0)];

    for &(ai, aj) in &a_edges {
        for &(bi, bj) in &b_edges {
            if let Some(pt) = segment_segment_2d(
                a2d[ai], a2d[aj], b2d[bi], b2d[bj], a3d[ai], a3d[aj], b3d[bi], b3d[bj], tolerance,
            ) {
                hits.push(pt);
            }
        }
    }

    // Also check for vertices of A inside B and vice versa.
    for i in 0..3 {
        if point_in_triangle_2d(a2d[i], b2d[0], b2d[1], b2d[2]) {
            hits.push(a3d[i]);
        }
        if point_in_triangle_2d(b2d[i], a2d[0], a2d[1], a2d[2]) {
            hits.push(b3d[i]);
        }
    }

    if hits.len() < 2 {
        return None;
    }

    // Deduplicate and find the two most distant points as the intersection segment.
    let mut best_dist_sq = 0.0_f64;
    let mut best_p0 = hits[0];
    let mut best_p1 = hits[1];

    for i in 0..hits.len() {
        for j in (i + 1)..hits.len() {
            let dx = hits[j].x() - hits[i].x();
            let dy = hits[j].y() - hits[i].y();
            let dz = hits[j].z() - hits[i].z();
            let d2 = dx.mul_add(dx, dy.mul_add(dy, dz * dz));
            if d2 > best_dist_sq {
                best_dist_sq = d2;
                best_p0 = hits[i];
                best_p1 = hits[j];
            }
        }
    }

    if best_dist_sq < tolerance * tolerance {
        return None;
    }

    Some(TriTriIntersection {
        p0: best_p0,
        p1: best_p1,
        tri_a: 0,
        tri_b: 0,
    })
}

/// 2D segment-segment intersection. Returns the 3D intersection point if
/// the two segments intersect in the 2D projection.
#[allow(clippy::too_many_arguments)]
fn segment_segment_2d(
    a0: (f64, f64),
    a1: (f64, f64),
    b0: (f64, f64),
    b1: (f64, f64),
    a0_3d: Point3,
    a1_3d: Point3,
    _b0_3d: Point3,
    _b1_3d: Point3,
    tolerance: f64,
) -> Option<Point3> {
    let dx_a = a1.0 - a0.0;
    let dy_a = a1.1 - a0.1;
    let dx_b = b1.0 - b0.0;
    let dy_b = b1.1 - b0.1;

    let denom = dx_a * dy_b - dy_a * dx_b;
    if denom.abs() < tolerance * tolerance {
        return None; // Parallel or coincident — skip (handled by vertex-in-triangle)
    }

    let dx_ab = b0.0 - a0.0;
    let dy_ab = b0.1 - a0.1;

    let t = (dx_ab * dy_b - dy_ab * dx_b) / denom;
    let u = (dx_ab * dy_a - dy_ab * dx_a) / denom;

    if t >= -tolerance && t <= 1.0 + tolerance && u >= -tolerance && u <= 1.0 + tolerance {
        let t_clamped = t.clamp(0.0, 1.0);
        Some(lerp_point(a0_3d, a1_3d, t_clamped))
    } else {
        None
    }
}

/// Test if a 2D point lies inside a triangle using cross products.
fn point_in_triangle_2d(p: (f64, f64), v0: (f64, f64), v1: (f64, f64), v2: (f64, f64)) -> bool {
    let cross = |a: (f64, f64), b: (f64, f64), c: (f64, f64)| -> f64 {
        (b.0 - a.0) * (c.1 - a.1) - (b.1 - a.1) * (c.0 - a.0)
    };

    let d1 = cross(v0, v1, p);
    let d2 = cross(v1, v2, p);
    let d3 = cross(v2, v0, p);

    let has_neg = (d1 < 0.0) || (d2 < 0.0) || (d3 < 0.0);
    let has_pos = (d1 > 0.0) || (d2 > 0.0) || (d3 > 0.0);

    // Inside the triangle or on its boundary. Handles boundary cases where
    // one or more cross products are zero (e.g., point on edge or vertex).
    !(has_neg && has_pos)
}

/// Check if three signed distances are all on the same side (all positive or all negative).
fn all_same_sign(d0: f64, d1: f64, d2: f64, tolerance: f64) -> bool {
    let pos = d0 > tolerance || d1 > tolerance || d2 > tolerance;
    let neg = d0 < -tolerance || d1 < -tolerance || d2 < -tolerance;
    // All non-negative or all non-positive (accounting for tolerance).
    (d0 >= -tolerance && d1 >= -tolerance && d2 >= -tolerance && pos && !neg)
        || (d0 <= tolerance && d1 <= tolerance && d2 <= tolerance && neg && !pos)
}

/// Compute the parameter interval of a triangle on the intersection line.
///
/// The three distances `d0, d1, d2` are the signed distances of each vertex
/// from the other triangle's plane. The `project` function maps a 3D point
/// to a scalar on the dominant axis of the intersection line.
fn triangle_interval(
    v0: Point3,
    v1: Point3,
    v2: Point3,
    d0: f64,
    d1: f64,
    d2: f64,
    project: &dyn Fn(Point3) -> f64,
) -> Option<(f64, f64)> {
    let p0 = project(v0);
    let p1 = project(v1);
    let p2 = project(v2);

    // Find the lone vertex (the one on the opposite side from the other two).
    // Compute the two intersection points where the triangle crosses the plane.
    let (t0, t1) = if (d0 > 0.0) != (d1 > 0.0) && (d0 > 0.0) != (d2 > 0.0) {
        // v0 is alone
        let ta = interp_param(p0, p1, d0, d1);
        let tb = interp_param(p0, p2, d0, d2);
        (ta, tb)
    } else if (d1 > 0.0) != (d0 > 0.0) && (d1 > 0.0) != (d2 > 0.0) {
        // v1 is alone
        let ta = interp_param(p1, p0, d1, d0);
        let tb = interp_param(p1, p2, d1, d2);
        (ta, tb)
    } else if (d2 > 0.0) != (d0 > 0.0) && (d2 > 0.0) != (d1 > 0.0) {
        // v2 is alone
        let ta = interp_param(p2, p0, d2, d0);
        let tb = interp_param(p2, p1, d2, d1);
        (ta, tb)
    } else {
        // Degenerate: one or more vertices are on the plane.
        // Find the two vertices that straddle or lie on the plane.
        let mut ts = Vec::new();
        if d0.abs() < 1e-15 {
            ts.push(p0);
        }
        if d1.abs() < 1e-15 {
            ts.push(p1);
        }
        if d2.abs() < 1e-15 {
            ts.push(p2);
        }
        // Also check edges that cross the plane.
        if d0 * d1 < 0.0 {
            ts.push(interp_param(p0, p1, d0, d1));
        }
        if d1 * d2 < 0.0 {
            ts.push(interp_param(p1, p2, d1, d2));
        }
        if d0 * d2 < 0.0 {
            ts.push(interp_param(p0, p2, d0, d2));
        }

        if ts.len() < 2 {
            return None;
        }

        let mut lo = ts[0];
        let mut hi = ts[0];
        for &t in &ts[1..] {
            if t < lo {
                lo = t;
            }
            if t > hi {
                hi = t;
            }
        }
        (lo, hi)
    };

    let (lo, hi) = if t0 <= t1 { (t0, t1) } else { (t1, t0) };
    Some((lo, hi))
}

/// Interpolate to find the parameter on the intersection line where an edge
/// crosses the plane.
fn interp_param(p_a: f64, p_b: f64, d_a: f64, d_b: f64) -> f64 {
    let denom = d_a - d_b;
    if denom.abs() < 1e-30 {
        0.5 * (p_a + p_b)
    } else {
        (p_b - p_a).mul_add(d_a / denom, p_a)
    }
}

/// Triangle vertices and their signed distances from a plane.
struct TriPlaneData {
    v: [Point3; 3],
    d: [f64; 3],
}

/// Reconstruct a 3D point on the intersection line given a parameter value
/// along the dominant axis.
///
/// Finds the two edges of the triangle that cross the other triangle's plane
/// and interpolates to find the 3D point whose projection equals `t_target`.
fn point_on_intersection_line(
    tri: &TriPlaneData,
    t_target: f64,
    project: &dyn Fn(Point3) -> f64,
) -> Point3 {
    let [v0, v1, v2] = tri.v;
    let [d0, d1, d2] = tri.d;
    // Collect the 3D points where triangle edges cross the plane.
    let mut crossing_points: Vec<Point3> = Vec::with_capacity(2);
    let mut crossing_params: Vec<f64> = Vec::with_capacity(2);

    let edges = [(v0, v1, d0, d1), (v1, v2, d1, d2), (v0, v2, d0, d2)];

    for &(va, vb, da, db) in &edges {
        if da.abs() < 1e-15 && db.abs() < 1e-15 {
            // Both on the plane: add both endpoints.
            crossing_points.push(va);
            crossing_params.push(project(va));
            crossing_points.push(vb);
            crossing_params.push(project(vb));
        } else if da.abs() < 1e-15 {
            crossing_points.push(va);
            crossing_params.push(project(va));
        } else if db.abs() < 1e-15 {
            crossing_points.push(vb);
            crossing_params.push(project(vb));
        } else if da * db < 0.0 {
            let t = da / (da - db);
            let p = lerp_point(va, vb, t);
            crossing_points.push(p);
            crossing_params.push(project(p));
        }
    }

    if crossing_points.len() < 2 {
        // Fallback: return first crossing point or centroid.
        return crossing_points
            .first()
            .copied()
            .unwrap_or_else(|| triangle_centroid(v0, v1, v2));
    }

    // Interpolate between the two crossing points to hit t_target.
    let t0 = crossing_params[0];
    let t1 = crossing_params[1];
    let denom = t1 - t0;
    if denom.abs() < 1e-30 {
        crossing_points[0]
    } else {
        let s = (t_target - t0) / denom;
        lerp_point(crossing_points[0], crossing_points[1], s)
    }
}

/// Linear interpolation between two points.
fn lerp_point(a: Point3, b: Point3, t: f64) -> Point3 {
    Point3::new(
        (b.x() - a.x()).mul_add(t, a.x()),
        (b.y() - a.y()).mul_add(t, a.y()),
        (b.z() - a.z()).mul_add(t, a.z()),
    )
}

/// Centroid of a triangle.
fn triangle_centroid(v0: Point3, v1: Point3, v2: Point3) -> Point3 {
    Point3::new(
        (v0.x() + v1.x() + v2.x()) / 3.0,
        (v0.y() + v1.y() + v2.y()) / 3.0,
        (v0.z() + v1.z() + v2.z()) / 3.0,
    )
}

// ---------------------------------------------------------------------------
// Step 3: Split meshes by intersection edges
// ---------------------------------------------------------------------------

/// A triangle mesh that has been split by intersection segments.
#[derive(Debug, Clone)]
struct SplitMesh {
    positions: Vec<Point3>,
    normals: Vec<Vec3>,
    triangles: Vec<[u32; 3]>,
}

/// Split a mesh's triangles along intersection segments.
///
/// For each triangle that has intersection segments passing through it,
/// the triangle is subdivided. Triangles with no intersections are passed
/// through unchanged.
#[allow(clippy::too_many_lines)]
fn split_mesh_by_intersections(
    mesh: &TriangleMesh,
    intersections: &[TriTriIntersection],
    is_mesh_a: bool,
    tolerance: f64,
) -> SplitMesh {
    let tri_count = mesh.indices.len() / 3;

    // Group intersection segments by triangle index.
    let mut tri_segments: HashMap<usize, Vec<(Point3, Point3)>> = HashMap::new();
    for isect in intersections {
        let tri_idx = if is_mesh_a { isect.tri_a } else { isect.tri_b };
        tri_segments
            .entry(tri_idx)
            .or_default()
            .push((isect.p0, isect.p1));
    }

    let mut positions = mesh.positions.clone();
    let mut normals = mesh.normals.clone();
    let mut triangles: Vec<[u32; 3]> = Vec::with_capacity(tri_count * 2);

    for i in 0..tri_count {
        let i0 = mesh.indices[i * 3] as usize;
        let i1 = mesh.indices[i * 3 + 1] as usize;
        let i2 = mesh.indices[i * 3 + 2] as usize;

        if let Some(segments) = tri_segments.get(&i) {
            // This triangle needs splitting.
            let v0 = mesh.positions[i0];
            let v1 = mesh.positions[i1];
            let v2 = mesh.positions[i2];
            let n0 = mesh.normals[i0];

            // Collect all unique intersection points on this triangle.
            let mut insert_pts: Vec<Point3> = Vec::new();
            for &(p0, p1) in segments {
                maybe_add_unique(&mut insert_pts, p0, tolerance);
                maybe_add_unique(&mut insert_pts, p1, tolerance);
            }

            // Split the triangle by inserting these points.
            let sub_tris = split_triangle_by_points(v0, v1, v2, &insert_pts, tolerance);

            for (sv0, sv1, sv2) in sub_tris {
                #[allow(clippy::cast_possible_truncation)]
                let base = positions.len() as u32;
                positions.push(sv0);
                positions.push(sv1);
                positions.push(sv2);
                normals.push(n0);
                normals.push(n0);
                normals.push(n0);
                triangles.push([base, base + 1, base + 2]);
            }
        } else {
            // No intersection: keep the triangle as-is.
            #[allow(clippy::cast_possible_truncation)]
            {
                triangles.push([i0 as u32, i1 as u32, i2 as u32]);
            }
        }
    }

    SplitMesh {
        positions,
        normals,
        triangles,
    }
}

/// Add a point to a list if no existing point is within tolerance.
fn maybe_add_unique(pts: &mut Vec<Point3>, p: Point3, tolerance: f64) {
    let tol_sq = tolerance * tolerance;
    for existing in pts.iter() {
        let dx = existing.x() - p.x();
        let dy = existing.y() - p.y();
        let dz = existing.z() - p.z();
        if dx.mul_add(dx, dy.mul_add(dy, dz * dz)) < tol_sq {
            return;
        }
    }
    pts.push(p);
}

/// Split a triangle by inserting points, producing sub-triangles.
///
/// Uses barycentric coordinate classification and simple fan/edge splitting.
/// For each inserted point, the triangle containing it is subdivided.
fn split_triangle_by_points(
    v0: Point3,
    v1: Point3,
    v2: Point3,
    points: &[Point3],
    tolerance: f64,
) -> Vec<(Point3, Point3, Point3)> {
    if points.is_empty() {
        return vec![(v0, v1, v2)];
    }

    // Start with the original triangle and iteratively split.
    let mut tris = vec![(v0, v1, v2)];

    for &pt in points {
        let mut new_tris = Vec::new();
        let mut inserted = false;

        for (tv0, tv1, tv2) in &tris {
            if !inserted {
                if let Some(sub) = try_split_triangle(*tv0, *tv1, *tv2, pt, tolerance) {
                    new_tris.extend(sub);
                    inserted = true;
                    continue;
                }
            }
            new_tris.push((*tv0, *tv1, *tv2));
        }

        tris = new_tris;
    }

    tris
}

/// Try to split a single triangle by inserting a point.
///
/// Returns `None` if the point is outside the triangle or coincident with
/// a vertex. Returns `Some(sub_triangles)` on success.
fn try_split_triangle(
    v0: Point3,
    v1: Point3,
    v2: Point3,
    pt: Point3,
    tolerance: f64,
) -> Option<Vec<(Point3, Point3, Point3)>> {
    let tol_sq = tolerance * tolerance;

    // Check if the point coincides with a vertex.
    if dist_sq(pt, v0) < tol_sq || dist_sq(pt, v1) < tol_sq || dist_sq(pt, v2) < tol_sq {
        return None;
    }

    // Check if the point is on an edge.
    if let Some(edge_idx) = point_on_edge(pt, v0, v1, v2, tolerance) {
        // Split into 2 triangles along the edge containing the point.
        let result = match edge_idx {
            0 => vec![(v0, pt, v2), (pt, v1, v2)], // point on edge v0-v1
            1 => vec![(v1, pt, v0), (pt, v2, v0)], // point on edge v1-v2
            _ => vec![(v2, pt, v1), (pt, v0, v1)], // point on edge v2-v0
        };
        return Some(result);
    }

    // Check if the point is inside the triangle using barycentric coordinates.
    let bary = barycentric(v0, v1, v2, pt);
    if bary.0 < -tolerance || bary.1 < -tolerance || bary.2 < -tolerance {
        return None; // Outside the triangle.
    }

    // Point is inside: split into 3 sub-triangles.
    Some(vec![(v0, v1, pt), (v1, v2, pt), (v2, v0, pt)])
}

/// Check if a point lies on one of the three edges of a triangle.
///
/// Returns the edge index (0: v0-v1, 1: v1-v2, 2: v2-v0) or `None`.
fn point_on_edge(pt: Point3, v0: Point3, v1: Point3, v2: Point3, tolerance: f64) -> Option<usize> {
    let edges = [(v0, v1), (v1, v2), (v2, v0)];
    for (i, &(ea, eb)) in edges.iter().enumerate() {
        let edge = eb - ea;
        let len_sq = edge.length_squared();
        if len_sq < 1e-30 {
            continue;
        }
        let t = (pt - ea).dot(edge) / len_sq;
        if t < -tolerance || t > 1.0 + tolerance {
            continue;
        }
        let closest = Point3::new(
            edge.x().mul_add(t, ea.x()),
            edge.y().mul_add(t, ea.y()),
            edge.z().mul_add(t, ea.z()),
        );
        if dist_sq(pt, closest) < tolerance * tolerance {
            return Some(i);
        }
    }
    None
}

/// Compute barycentric coordinates of a point in a triangle.
fn barycentric(v0: Point3, v1: Point3, v2: Point3, p: Point3) -> (f64, f64, f64) {
    let e0 = v1 - v0;
    let e1 = v2 - v0;
    let ep = p - v0;

    let d00 = e0.dot(e0);
    let d01 = e0.dot(e1);
    let d11 = e1.dot(e1);
    let d20 = ep.dot(e0);
    let d21 = ep.dot(e1);

    let denom = d00.mul_add(d11, -(d01 * d01));
    if denom.abs() < 1e-30 {
        return (-1.0, -1.0, -1.0); // Degenerate triangle.
    }

    let inv = 1.0 / denom;
    let v = d11.mul_add(d20, -(d01 * d21)) * inv;
    let w = d00.mul_add(d21, -(d01 * d20)) * inv;
    let u = 1.0 - v - w;

    (u, v, w)
}

/// Squared distance between two points.
fn dist_sq(a: Point3, b: Point3) -> f64 {
    let d = b - a;
    d.dot(d)
}

// ---------------------------------------------------------------------------
// Step 4: Classification
// ---------------------------------------------------------------------------

/// Classify each triangle in a split mesh as inside or outside the other mesh.
///
/// Uses generalized winding number: a triangle's centroid is inside the other
/// solid if its winding number is approximately 1 (or any non-zero integer).
fn classify_triangles(split: &SplitMesh, other_mesh: &TriangleMesh, _tolerance: f64) -> Vec<bool> {
    split
        .triangles
        .iter()
        .map(|tri| {
            let v0 = split.positions[tri[0] as usize];
            let v1 = split.positions[tri[1] as usize];
            let v2 = split.positions[tri[2] as usize];
            let centroid = triangle_centroid(v0, v1, v2);
            let wn = winding_number_at_point(centroid, other_mesh);
            wn.abs() > 0.5
        })
        .collect()
}

/// Compute the generalized winding number of a point with respect to a
/// triangle mesh.
///
/// Sums the signed solid angles subtended by each triangle as seen from the
/// point, divided by 4pi. For a closed mesh, this returns ~1 for points
/// inside and ~0 for points outside.
///
/// This is a standalone helper that can be reused for other classification tasks.
#[must_use]
pub(crate) fn winding_number_at_point(point: Point3, mesh: &TriangleMesh) -> f64 {
    let tri_count = mesh.indices.len() / 3;
    let mut total_solid_angle = 0.0;

    for i in 0..tri_count {
        let (v0, v1, v2) = get_triangle(mesh, i);

        // Vectors from point to triangle vertices.
        let a = v0 - point;
        let b = v1 - point;
        let c = v2 - point;

        let la = a.length();
        let lb = b.length();
        let lc = c.length();

        // Skip degenerate triangles or if the point is at a vertex.
        if la < 1e-15 || lb < 1e-15 || lc < 1e-15 {
            continue;
        }

        // Van Oosterom & Strackee formula for the signed solid angle of a
        // triangle as seen from a point.
        let numerator = a.dot(b.cross(c));
        let denominator = c
            .dot(a)
            .mul_add(lb, a.dot(b).mul_add(lc, b.dot(c).mul_add(la, la * lb * lc)));

        // atan2 gives the half solid angle; multiply by 2 at the end.
        total_solid_angle += 2.0 * numerator.atan2(denominator);
    }

    total_solid_angle / (4.0 * std::f64::consts::PI)
}

// ---------------------------------------------------------------------------
// Step 5: Assembly
// ---------------------------------------------------------------------------

/// Assemble the result mesh from classified sub-triangles.
///
/// Selection logic:
/// - [`BooleanOp::Fuse`]: `A_outside_B` union `B_outside_A`
/// - [`BooleanOp::Cut`]: `A_outside_B` union `B_inside_A` (with flipped normals)
/// - [`BooleanOp::Intersect`]: `A_inside_B` union `B_inside_A`
fn assemble_result(
    split_a: &SplitMesh,
    split_b: &SplitMesh,
    classify_a: &[bool], // true = inside B
    classify_b: &[bool], // true = inside A
    op: BooleanOp,
) -> TriangleMesh {
    let mut positions = Vec::new();
    let mut normals = Vec::new();
    let mut indices = Vec::new();

    // Process mesh A triangles.
    for (i, tri) in split_a.triangles.iter().enumerate() {
        let inside_b = classify_a.get(i).copied().unwrap_or(false);
        let keep = match op {
            BooleanOp::Fuse | BooleanOp::Cut => !inside_b, // keep outside B
            BooleanOp::Intersect => inside_b,              // keep inside B
        };
        if keep {
            append_triangle(
                &split_a.positions,
                &split_a.normals,
                tri,
                false,
                &mut positions,
                &mut normals,
                &mut indices,
            );
        }
    }

    // Process mesh B triangles.
    for (i, tri) in split_b.triangles.iter().enumerate() {
        let inside_a = classify_b.get(i).copied().unwrap_or(false);
        let (keep, flip) = match op {
            BooleanOp::Fuse => (!inside_a, false),     // keep outside A
            BooleanOp::Cut => (inside_a, true),        // keep inside A, flip normals
            BooleanOp::Intersect => (inside_a, false), // keep inside A
        };
        if keep {
            append_triangle(
                &split_b.positions,
                &split_b.normals,
                tri,
                flip,
                &mut positions,
                &mut normals,
                &mut indices,
            );
        }
    }

    TriangleMesh {
        positions,
        normals,
        indices,
    }
}

/// Append a triangle to the output mesh, optionally flipping its winding and normal.
fn append_triangle(
    src_positions: &[Point3],
    src_normals: &[Vec3],
    tri: &[u32; 3],
    flip: bool,
    positions: &mut Vec<Point3>,
    normals: &mut Vec<Vec3>,
    indices: &mut Vec<u32>,
) {
    #[allow(clippy::cast_possible_truncation)]
    let base = positions.len() as u32;

    if flip {
        // Reverse winding order and negate normals.
        for &idx in tri.iter().rev() {
            let i = idx as usize;
            positions.push(src_positions[i]);
            normals.push(-src_normals[i]);
        }
    } else {
        for &idx in tri {
            let i = idx as usize;
            positions.push(src_positions[i]);
            normals.push(src_normals[i]);
        }
    }

    indices.push(base);
    indices.push(base + 1);
    indices.push(base + 2);
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

/// Extract the three vertices of a triangle from a mesh.
fn get_triangle(mesh: &TriangleMesh, tri_idx: usize) -> (Point3, Point3, Point3) {
    let base = tri_idx * 3;
    let i0 = mesh.indices[base] as usize;
    let i1 = mesh.indices[base + 1] as usize;
    let i2 = mesh.indices[base + 2] as usize;
    (mesh.positions[i0], mesh.positions[i1], mesh.positions[i2])
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    #![allow(clippy::unwrap_used, clippy::expect_used)]

    use super::*;

    /// Create a tetrahedron mesh centered at a point.
    fn tetrahedron_mesh(center: Point3, size: f64) -> TriangleMesh {
        let s = size;
        // Regular tetrahedron vertices.
        let v0 = Point3::new(center.x() + s, center.y() + s, center.z() + s);
        let v1 = Point3::new(center.x() + s, center.y() - s, center.z() - s);
        let v2 = Point3::new(center.x() - s, center.y() + s, center.z() - s);
        let v3 = Point3::new(center.x() - s, center.y() - s, center.z() + s);

        let positions = [v0, v1, v2, v3];

        // Compute face normals for each face.
        let faces = [(0u32, 2, 1), (0, 1, 3), (0, 3, 2), (1, 2, 3)];
        let mut out_positions = Vec::new();
        let mut out_normals = Vec::new();
        let mut indices = Vec::new();

        for &(i0, i1, i2) in &faces {
            let p0 = positions[i0 as usize];
            let p1 = positions[i1 as usize];
            let p2 = positions[i2 as usize];

            let e1 = p1 - p0;
            let e2 = p2 - p0;
            let n = e1.cross(e2).normalize().unwrap_or(Vec3::new(0.0, 0.0, 1.0));

            #[allow(clippy::cast_possible_truncation)]
            let base = out_positions.len() as u32;
            out_positions.push(p0);
            out_positions.push(p1);
            out_positions.push(p2);
            out_normals.push(n);
            out_normals.push(n);
            out_normals.push(n);
            indices.push(base);
            indices.push(base + 1);
            indices.push(base + 2);
        }

        TriangleMesh {
            positions: out_positions,
            normals: out_normals,
            indices,
        }
    }

    /// Create an axis-aligned box mesh centered at a point.
    fn box_mesh(center: Point3, half_size: f64) -> TriangleMesh {
        let s = half_size;
        let cx = center.x();
        let cy = center.y();
        let cz = center.z();

        // 8 corner vertices.
        let verts = [
            Point3::new(cx - s, cy - s, cz - s), // 0
            Point3::new(cx + s, cy - s, cz - s), // 1
            Point3::new(cx + s, cy + s, cz - s), // 2
            Point3::new(cx - s, cy + s, cz - s), // 3
            Point3::new(cx - s, cy - s, cz + s), // 4
            Point3::new(cx + s, cy - s, cz + s), // 5
            Point3::new(cx + s, cy + s, cz + s), // 6
            Point3::new(cx - s, cy + s, cz + s), // 7
        ];

        // 12 triangles (2 per face), with outward-facing normals.
        let face_tris: [(usize, usize, usize, Vec3); 12] = [
            // -Z face
            (0, 3, 2, Vec3::new(0.0, 0.0, -1.0)),
            (0, 2, 1, Vec3::new(0.0, 0.0, -1.0)),
            // +Z face
            (4, 5, 6, Vec3::new(0.0, 0.0, 1.0)),
            (4, 6, 7, Vec3::new(0.0, 0.0, 1.0)),
            // -X face
            (0, 4, 7, Vec3::new(-1.0, 0.0, 0.0)),
            (0, 7, 3, Vec3::new(-1.0, 0.0, 0.0)),
            // +X face
            (1, 2, 6, Vec3::new(1.0, 0.0, 0.0)),
            (1, 6, 5, Vec3::new(1.0, 0.0, 0.0)),
            // -Y face
            (0, 1, 5, Vec3::new(0.0, -1.0, 0.0)),
            (0, 5, 4, Vec3::new(0.0, -1.0, 0.0)),
            // +Y face
            (3, 7, 6, Vec3::new(0.0, 1.0, 0.0)),
            (3, 6, 2, Vec3::new(0.0, 1.0, 0.0)),
        ];

        let mut positions = Vec::new();
        let mut normals = Vec::new();
        let mut indices = Vec::new();

        for &(i0, i1, i2, n) in &face_tris {
            #[allow(clippy::cast_possible_truncation)]
            let base = positions.len() as u32;
            positions.push(verts[i0]);
            positions.push(verts[i1]);
            positions.push(verts[i2]);
            normals.push(n);
            normals.push(n);
            normals.push(n);
            indices.push(base);
            indices.push(base + 1);
            indices.push(base + 2);
        }

        TriangleMesh {
            positions,
            normals,
            indices,
        }
    }

    #[test]
    fn mesh_boolean_disjoint_fuse() {
        let a = tetrahedron_mesh(Point3::new(0.0, 0.0, 0.0), 1.0);
        let b = tetrahedron_mesh(Point3::new(10.0, 0.0, 0.0), 1.0);

        let a_tri_count = a.indices.len() / 3;
        let b_tri_count = b.indices.len() / 3;

        let result = mesh_boolean(&a, &b, BooleanOp::Fuse, 1e-7).unwrap();
        let result_tri_count = result.mesh.indices.len() / 3;

        // Disjoint fuse should contain all triangles from both meshes.
        assert_eq!(
            result_tri_count,
            a_tri_count + b_tri_count,
            "disjoint fuse should combine all triangles: expected {}, got {}",
            a_tri_count + b_tri_count,
            result_tri_count
        );
    }

    #[test]
    fn mesh_boolean_overlapping_intersect() {
        let a = box_mesh(Point3::new(0.0, 0.0, 0.0), 1.0);
        let b = box_mesh(Point3::new(0.5, 0.5, 0.5), 1.0);

        let result = mesh_boolean(&a, &b, BooleanOp::Intersect, 1e-7).unwrap();
        let result_tri_count = result.mesh.indices.len() / 3;

        // The intersection of two overlapping cubes should produce a non-empty result.
        assert!(
            result_tri_count > 0,
            "intersection of overlapping boxes should have triangles, got {}",
            result_tri_count
        );

        // The intersection should have fewer triangles than the union of both inputs
        // (since some faces are discarded).
        let total_input = a.indices.len() / 3 + b.indices.len() / 3;
        assert!(
            result_tri_count < total_input,
            "intersection should have fewer triangles than total input: {} >= {}",
            result_tri_count,
            total_input
        );
    }

    #[test]
    fn mesh_boolean_produces_valid_mesh() {
        let a = box_mesh(Point3::new(0.0, 0.0, 0.0), 1.0);
        let b = box_mesh(Point3::new(0.5, 0.5, 0.5), 1.0);

        let result = mesh_boolean(&a, &b, BooleanOp::Fuse, 1e-7).unwrap();

        // Verify all indices are valid.
        let n_verts = result.mesh.positions.len();
        for &idx in &result.mesh.indices {
            assert!(
                (idx as usize) < n_verts,
                "index {} out of bounds (n_verts = {})",
                idx,
                n_verts
            );
        }

        // Verify indices come in groups of 3.
        assert_eq!(
            result.mesh.indices.len() % 3,
            0,
            "indices should be a multiple of 3"
        );

        // Verify normals match positions count.
        assert_eq!(
            result.mesh.positions.len(),
            result.mesh.normals.len(),
            "positions and normals should have the same count"
        );
    }

    #[test]
    fn winding_number_inside_box() {
        let bx = box_mesh(Point3::new(0.0, 0.0, 0.0), 1.0);

        let inside = winding_number_at_point(Point3::new(0.0, 0.0, 0.0), &bx);
        assert!(
            inside.abs() > 0.5,
            "winding number at center of box should be ~1, got {}",
            inside
        );

        let outside = winding_number_at_point(Point3::new(5.0, 5.0, 5.0), &bx);
        assert!(
            outside.abs() < 0.5,
            "winding number outside box should be ~0, got {}",
            outside
        );
    }

    #[test]
    fn mesh_boolean_disjoint_intersect_is_error() {
        let a = tetrahedron_mesh(Point3::new(0.0, 0.0, 0.0), 1.0);
        let b = tetrahedron_mesh(Point3::new(10.0, 0.0, 0.0), 1.0);

        let result = mesh_boolean(&a, &b, BooleanOp::Intersect, 1e-7);
        assert!(
            result.is_err(),
            "intersection of disjoint meshes should error"
        );
    }
}
