//! Co-refinement mesh boolean operations on triangle meshes.
//!
//! Implements mesh booleans (fuse, cut, intersect) using the co-refinement
//! approach: compute exact triangle-triangle intersections, re-triangulate
//! both meshes so the intersection polylines appear as conforming triangle
//! edges on BOTH sides, classify sub-triangles (inside / outside / lying on
//! the other mesh's surface), and assemble the result.
//!
//! This operates directly on [`TriangleMesh`] without requiring topology.

#![allow(clippy::tuple_array_conversions)]

use brepkit_math::aabb::Aabb3;
use brepkit_math::bvh::Bvh;
use brepkit_math::cdt::Cdt;
use brepkit_math::det_hash::DetHashMap;
use brepkit_math::predicates::orient3d;
use brepkit_math::vec::{Point2, Point3, Vec3};

use crate::OperationsError;
use crate::boolean::BooleanOp;
use crate::tessellate::TriangleMesh;

/// Result of a mesh boolean operation.
#[derive(Debug, Clone)]
pub struct MeshBooleanResult {
    /// The resulting triangle mesh.
    pub mesh: TriangleMesh,
    /// Boundary (one-sided) edge count of the output, measured after welding
    /// vertices by position. Zero for a closed result; nonzero means the
    /// co-refinement could not produce a watertight mesh for these inputs.
    pub boundary_edge_count: usize,
    /// Non-manifold (3+ incidence) edge count of the output, measured after
    /// welding vertices by position. Zero for a 2-manifold result.
    pub non_manifold_edge_count: usize,
}

/// Position-weld grid for the output self-check. Seam vertices are shared
/// verbatim between the two split meshes, so anything below feature scale
/// works; sharing the `mesh_ops` coincident-triangle dedupe grid keeps the
/// self-check and the downstream dedupe measuring on the same weld grid.
const SELF_CHECK_GRID: f64 = crate::tessellate::COINCIDENT_DEDUPE_GRID;

/// Perform a mesh boolean operation between two triangle meshes.
///
/// Uses co-refinement: compute exact triangle-triangle intersections,
/// re-triangulate every intersected triangle with the intersection segments
/// as triangulation constraints (so the seam is a shared, conforming
/// polyline on both meshes), classify sub-triangles by winding number with
/// explicit handling of coincident-surface (exactly touching) triangles,
/// and assemble the result.
///
/// The returned [`MeshBooleanResult`] carries position-welded boundary and
/// non-manifold edge counts of the output so callers can detect a
/// non-watertight result instead of consuming it silently.
///
/// # Errors
/// Returns an error if the operation cannot be completed (e.g. the
/// intersection of disjoint meshes is empty).
pub fn mesh_boolean(
    mesh_a: &TriangleMesh,
    mesh_b: &TriangleMesh,
    op: BooleanOp,
    tolerance: f64,
) -> Result<MeshBooleanResult, OperationsError> {
    // Step 1: BVH broad-phase
    let bvh_a = build_triangle_bvh(mesh_a);
    let bvh_b = build_triangle_bvh(mesh_b);
    let pairs = find_intersecting_pairs(mesh_a, &bvh_b, tolerance);

    // Step 2: Triangle-triangle intersection segments
    let segments = compute_all_intersections(mesh_a, mesh_b, &pairs, tolerance);

    // Step 3: Conforming re-triangulation of both meshes
    let split_a = split_mesh_conforming(mesh_a, &segments, true, tolerance);
    let split_b = split_mesh_conforming(mesh_b, &segments, false, tolerance);

    // Step 4: Classify sub-triangles
    let classify_a = classify_split_triangles(&split_a, mesh_b, &bvh_b, tolerance);
    let classify_b = classify_split_triangles(&split_b, mesh_a, &bvh_a, tolerance);

    // Step 5: Assemble result
    let mesh = assemble_result(&split_a, &split_b, &classify_a, &classify_b, op);

    if mesh.positions.is_empty() {
        return Err(OperationsError::EmptyResult {
            reason: "mesh boolean produced no output vertices".into(),
        });
    }

    let (boundary_edge_count, non_manifold_edge_count) = welded_health(&mesh, SELF_CHECK_GRID);
    Ok(MeshBooleanResult {
        mesh,
        boundary_edge_count,
        non_manifold_edge_count,
    })
}

/// Count boundary and non-manifold edges after welding vertices to a grid.
fn welded_health(mesh: &TriangleMesh, grid: f64) -> (usize, usize) {
    type Q = (i64, i64, i64);
    let s = 1.0 / grid;
    #[allow(clippy::cast_possible_truncation)]
    let q = |p: Point3| -> Q {
        (
            (p.x() * s).round() as i64,
            (p.y() * s).round() as i64,
            (p.z() * s).round() as i64,
        )
    };
    let mut occ: DetHashMap<(Q, Q), u32> = DetHashMap::default();
    for tri in mesh.indices.chunks_exact(3) {
        let a = q(mesh.positions[tri[0] as usize]);
        let b = q(mesh.positions[tri[1] as usize]);
        let c = q(mesh.positions[tri[2] as usize]);
        if a == b || b == c || a == c {
            continue;
        }
        for (p, r) in [(a, b), (b, c), (c, a)] {
            let key = if p <= r { (p, r) } else { (r, p) };
            *occ.entry(key).or_default() += 1;
        }
    }
    let bnd = occ.values().filter(|&&c| c == 1).count();
    let nm = occ.values().filter(|&&c| c > 2).count();
    (bnd, nm)
}

/// Build a BVH over a mesh's triangles.
fn build_triangle_bvh(mesh: &TriangleMesh) -> Bvh {
    let tri_count = mesh.indices.len() / 3;
    let mut entries = Vec::with_capacity(tri_count);
    for i in 0..tri_count {
        let (v0, v1, v2) = get_triangle(mesh, i);
        entries.push((i, Aabb3::from_points([v0, v1, v2])));
    }
    Bvh::build(&entries)
}

/// Find all potentially intersecting triangle pairs between mesh A and mesh B.
fn find_intersecting_pairs(
    mesh_a: &TriangleMesh,
    bvh_b: &Bvh,
    tolerance: f64,
) -> Vec<(usize, usize)> {
    let tri_count_a = mesh_a.indices.len() / 3;
    let mut pairs = Vec::new();

    for i in 0..tri_count_a {
        let (v0, v1, v2) = get_triangle(mesh_a, i);
        let aabb_a = Aabb3::from_points([v0, v1, v2]).expanded(tolerance);
        let candidates = bvh_b.query_overlap(&aabb_a);
        for j in candidates {
            pairs.push((i, j));
        }
    }

    pairs
}

/// An intersection segment between a triangle of mesh A and one of mesh B.
///
/// `apply_a` / `apply_b` say which mesh's split must honor the segment as a
/// triangulation constraint. Transversal intersections apply to both;
/// coplanar-contact co-refinement segments apply to one side only (the
/// segment already runs along an existing edge of the other mesh).
#[derive(Debug, Clone)]
struct IsectSegment {
    p0: Point3,
    p1: Point3,
    tri_a: usize,
    tri_b: usize,
    apply_a: bool,
    apply_b: bool,
}

/// Compute all triangle-triangle intersection segments for the candidate pairs.
fn compute_all_intersections(
    mesh_a: &TriangleMesh,
    mesh_b: &TriangleMesh,
    pairs: &[(usize, usize)],
    tolerance: f64,
) -> Vec<IsectSegment> {
    let mut result = Vec::new();

    for &(tri_a, tri_b) in pairs {
        let (a0, a1, a2) = get_triangle(mesh_a, tri_a);
        let (b0, b1, b2) = get_triangle(mesh_b, tri_b);

        for mut seg in intersect_triangles(a0, a1, a2, b0, b1, b2, tolerance) {
            seg.tri_a = tri_a;
            seg.tri_b = tri_b;
            result.push(seg);
        }
    }

    result
}

/// Compute the intersection segments between two triangles.
///
/// Transversal case: Moller interval overlap on the intersection line of the
/// two planes, using `orient3d` for the side classification. Coplanar case:
/// mutual 2D edge clipping so each mesh conforms to the other's edges inside
/// the shared plane. Returns an empty vec if the triangles do not intersect
/// or only touch at a point.
#[allow(clippy::similar_names)]
fn intersect_triangles(
    a0: Point3,
    a1: Point3,
    a2: Point3,
    b0: Point3,
    b1: Point3,
    b2: Point3,
    tolerance: f64,
) -> Vec<IsectSegment> {
    // Classify vertices of B against the plane of A.
    let db0 = orient3d(a0, a1, a2, b0);
    let db1 = orient3d(a0, a1, a2, b1);
    let db2 = orient3d(a0, a1, a2, b2);

    if all_same_sign(db0, db1, db2, tolerance) {
        return Vec::new();
    }

    // Classify vertices of A against the plane of B.
    let da0 = orient3d(b0, b1, b2, a0);
    let da1 = orient3d(b0, b1, b2, a1);
    let da2 = orient3d(b0, b1, b2, a2);

    if all_same_sign(da0, da1, da2, tolerance) {
        return Vec::new();
    }

    let na = (a1 - a0).cross(a2 - a0);
    let nb = (b1 - b0).cross(b2 - b0);
    let line_dir = na.cross(nb);

    // Coplanar check: when |na × nb| ≈ 0, the triangles lie in the same plane.
    let line_len_sq = line_dir.dot(line_dir);
    let na_len_sq = na.dot(na);
    let nb_len_sq = nb.dot(nb);
    // sin²(angle) threshold: tolerance² for angular comparison
    if line_len_sq < (tolerance * tolerance) * na_len_sq.max(nb_len_sq) {
        return coplanar_corefine_segments([a0, a1, a2], [b0, b1, b2], tolerance);
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

    let Some((ta_min, ta_max)) = triangle_interval(a0, a1, a2, da0, da1, da2, &project) else {
        return Vec::new();
    };
    let Some((tb_min, tb_max)) = triangle_interval(b0, b1, b2, db0, db1, db2, &project) else {
        return Vec::new();
    };

    let t_lo = ta_min.max(tb_min);
    let t_hi = ta_max.min(tb_max);

    if t_hi - t_lo < tolerance {
        return Vec::new();
    }

    let tri_data = TriPlaneData {
        v: [a0, a1, a2],
        d: [da0, da1, da2],
    };
    let p0 = point_on_intersection_line(&tri_data, t_lo, &project);
    let p1 = point_on_intersection_line(&tri_data, t_hi, &project);

    vec![IsectSegment {
        p0,
        p1,
        tri_a: 0,
        tri_b: 0,
        apply_a: true,
        apply_b: true,
    }]
}

/// Co-refinement segments for a coplanar triangle pair.
///
/// Each edge of B clipped to the interior of A becomes a constraint for A's
/// re-triangulation (and vice versa), so both meshes end up conforming to
/// each other's edges inside the shared plane. The clip endpoints are the
/// mutual edge-edge crossing points, which lie on triangle edges of BOTH
/// meshes; the splitter's edge-point propagation carries them onto the
/// neighbors sharing those edges.
fn coplanar_corefine_segments(
    a3d: [Point3; 3],
    b3d: [Point3; 3],
    tolerance: f64,
) -> Vec<IsectSegment> {
    let na = (a3d[1] - a3d[0]).cross(a3d[2] - a3d[0]);
    let nax = na.x().abs();
    let nay = na.y().abs();
    let naz = na.z().abs();

    let to_2d = |p: Point3| -> Point2 {
        if naz >= nax && naz >= nay {
            Point2::new(p.x(), p.y())
        } else if nay >= nax {
            Point2::new(p.x(), p.z())
        } else {
            Point2::new(p.y(), p.z())
        }
    };

    let a2d = [to_2d(a3d[0]), to_2d(a3d[1]), to_2d(a3d[2])];
    let b2d = [to_2d(b3d[0]), to_2d(b3d[1]), to_2d(b3d[2])];

    let mut out = Vec::new();
    // B's edges clipped to A constrain mesh A's triangle.
    clip_edges_into(&b2d, &b3d, &a2d, tolerance, true, &mut out);
    // A's edges clipped to B constrain mesh B's triangle.
    clip_edges_into(&a2d, &a3d, &b2d, tolerance, false, &mut out);
    out
}

/// Clip each edge of `src` against the triangle `clip` (2D), emitting the
/// interior portions as constraint segments for one mesh side.
fn clip_edges_into(
    src2d: &[Point2; 3],
    src3d: &[Point3; 3],
    clip2d: &[Point2; 3],
    tolerance: f64,
    for_mesh_a: bool,
    out: &mut Vec<IsectSegment>,
) {
    for (i, j) in [(0usize, 1usize), (1, 2), (2, 0)] {
        let Some((t0, t1)) = clip_segment_to_triangle_2d(src2d[i], src2d[j], clip2d, tolerance)
        else {
            continue;
        };
        let e3d = src3d[j] - src3d[i];
        let len = e3d.length();
        if (t1 - t0) * len < tolerance * 2.0 {
            continue;
        }
        let p0 = lerp_point(src3d[i], src3d[j], t0);
        let p1 = lerp_point(src3d[i], src3d[j], t1);
        out.push(IsectSegment {
            p0,
            p1,
            tri_a: 0,
            tri_b: 0,
            apply_a: for_mesh_a,
            apply_b: !for_mesh_a,
        });
    }
}

/// Clip the parametric segment `e0 + t*(e1-e0)`, t in [0,1], against a 2D
/// triangle (any winding). Returns the surviving parameter interval, or
/// `None` when the segment misses the triangle.
fn clip_segment_to_triangle_2d(
    e0: Point2,
    e1: Point2,
    tri: &[Point2; 3],
    tolerance: f64,
) -> Option<(f64, f64)> {
    // Orient the triangle CCW so the inward side of each edge is cross >= 0.
    let signed2 = (tri[1].x() - tri[0].x()) * (tri[2].y() - tri[0].y())
        - (tri[1].y() - tri[0].y()) * (tri[2].x() - tri[0].x());
    if signed2.abs() < 1e-30 {
        return None;
    }
    let (o0, o1, o2) = if signed2 > 0.0 {
        (tri[0], tri[1], tri[2])
    } else {
        (tri[0], tri[2], tri[1])
    };

    let mut t_lo = 0.0_f64;
    let mut t_hi = 1.0_f64;
    for (p, q) in [(o0, o1), (o1, o2), (o2, o0)] {
        let ex = q.x() - p.x();
        let ey = q.y() - p.y();
        let elen = ex.hypot(ey);
        if elen < 1e-30 {
            return None;
        }
        // Signed distance of the segment endpoints from this clip edge
        // (positive = inside for a CCW triangle), in true distance units.
        let d0 = (ex * (e0.y() - p.y()) - ey * (e0.x() - p.x())) / elen;
        let d1 = (ex * (e1.y() - p.y()) - ey * (e1.x() - p.x())) / elen;
        let eps = tolerance;
        if d0 < -eps && d1 < -eps {
            return None;
        }
        if d0 >= -eps && d1 >= -eps {
            continue; // fully inside this half-plane
        }
        // Crossing: split at d == 0.
        let t = d0 / (d0 - d1);
        if d0 < -eps {
            t_lo = t_lo.max(t);
        } else {
            t_hi = t_hi.min(t);
        }
    }
    if t_hi <= t_lo {
        return None;
    }
    Some((t_lo, t_hi))
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

/// A triangle mesh that has been split by intersection segments.
#[derive(Debug, Clone)]
struct SplitMesh {
    positions: Vec<Point3>,
    normals: Vec<Vec3>,
    triangles: Vec<[u32; 3]>,
}

/// Where an insertion point sits relative to its host triangle.
enum PointSite {
    Corner,
    Edge(usize),
    Interior,
}

/// Per-host-triangle split input, produced in phase 1 of the splitter.
#[derive(Default)]
struct HostSplit {
    /// Canonical (welded) insertion points.
    pts: Vec<Point3>,
    /// Constraint chains as index pairs into `pts`.
    constraints: Vec<(usize, usize)>,
}

/// Quantized undirected edge key for the cross-triangle edge-point map.
type EdgeKey = ((i64, i64, i64), (i64, i64, i64));

/// The float-to-int cast saturates (Rust `as` semantics), so coordinates
/// beyond ±i64::MAX/S ≈ ±9.2e9 units collapse onto the same key instead of
/// wrapping; models are expected to stay far inside that bound.
fn edge_key(a: Point3, b: Point3) -> EdgeKey {
    const S: f64 = 1.0e9;
    #[allow(clippy::cast_possible_truncation)]
    let q = |p: Point3| -> (i64, i64, i64) {
        (
            (p.x() * S).round() as i64,
            (p.y() * S).round() as i64,
            (p.z() * S).round() as i64,
        )
    };
    let (qa, qb) = (q(a), q(b));
    if qa <= qb { (qa, qb) } else { (qb, qa) }
}

/// Split a mesh's triangles along intersection segments, producing a
/// triangulation that CONFORMS to the segments: every segment appears as a
/// chain of sub-triangle edges, and points landing on a triangle edge are
/// propagated to the neighbor sharing that edge (no T-junctions).
#[allow(clippy::too_many_lines)]
fn split_mesh_conforming(
    mesh: &TriangleMesh,
    segments: &[IsectSegment],
    is_mesh_a: bool,
    tolerance: f64,
) -> SplitMesh {
    let tri_count = mesh.indices.len() / 3;

    // Group constraint segments by host triangle index.
    let mut host_segments: DetHashMap<usize, Vec<(Point3, Point3)>> = DetHashMap::default();
    for seg in segments {
        let applies = if is_mesh_a { seg.apply_a } else { seg.apply_b };
        if !applies {
            continue;
        }
        let host = if is_mesh_a { seg.tri_a } else { seg.tri_b };
        host_segments
            .entry(host)
            .or_default()
            .push((seg.p0, seg.p1));
    }

    // Phase 1: per registered host, weld points, split segments at mutual
    // crossings and collinear interior points, classify each point against
    // the host triangle, and feed on-edge points into the global edge map.
    let mut host_data: DetHashMap<usize, HostSplit> = DetHashMap::default();
    let mut edge_points: DetHashMap<EdgeKey, Vec<Point3>> = DetHashMap::default();

    let mut host_indices: Vec<usize> = host_segments.keys().copied().collect();
    host_indices.sort_unstable();

    for &host in &host_indices {
        let segs = &host_segments[&host];
        let (v0, v1, v2) = get_triangle(mesh, host);

        let mut pts: Vec<Point3> = Vec::new();
        let canon = |p: Point3, pts: &mut Vec<Point3>| -> usize {
            let tol_sq = tolerance * tolerance;
            for (i, q) in pts.iter().enumerate() {
                if dist_sq(*q, p) < tol_sq {
                    return i;
                }
            }
            pts.push(p);
            pts.len() - 1
        };

        // Canonicalize segment endpoints.
        let mut seg_idx: Vec<(usize, usize)> = Vec::with_capacity(segs.len());
        for &(p0, p1) in segs {
            let i0 = canon(p0, &mut pts);
            let i1 = canon(p1, &mut pts);
            if i0 != i1 {
                seg_idx.push((i0, i1));
            }
        }

        // Split segments at mutual transversal crossings. Each round resolves
        // one crossing, so the cap scales with the possible crossing count
        // instead of a fixed budget that dense hosts could silently exhaust.
        let max_rounds = 16 + seg_idx.len() * seg_idx.len();
        let mut changed = true;
        let mut rounds = 0;
        while changed && rounds < max_rounds {
            changed = false;
            rounds += 1;
            'outer: for si in 0..seg_idx.len() {
                for sj in (si + 1)..seg_idx.len() {
                    let (a0, a1) = seg_idx[si];
                    let (b0, b1) = seg_idx[sj];
                    if a0 == b0 || a0 == b1 || a1 == b0 || a1 == b1 {
                        continue;
                    }
                    if let Some(x) =
                        transversal_crossing(pts[a0], pts[a1], pts[b0], pts[b1], tolerance)
                    {
                        let xi = canon(x, &mut pts);
                        seg_idx[si] = (a0, xi);
                        seg_idx.push((xi, a1));
                        seg_idx[sj] = (b0, xi);
                        seg_idx.push((xi, b1));
                        changed = true;
                        break 'outer;
                    }
                }
            }
        }
        if changed {
            log::warn!(
                "mesh boolean: crossing resolution on host triangle {host} exhausted \
                 {max_rounds} rounds; unresolved crossings may force a non-conforming fan split"
            );
        }

        // Split segments at collinear interior points (chained seams from
        // adjacent opposing triangles share endpoints mid-segment).
        let mut chains: Vec<(usize, usize)> = Vec::with_capacity(seg_idx.len());
        for &(i0, i1) in &seg_idx {
            let a = pts[i0];
            let b = pts[i1];
            let ab = b - a;
            let len_sq = ab.dot(ab);
            if len_sq < tolerance * tolerance {
                continue;
            }
            let mut on_seg: Vec<(f64, usize)> = Vec::new();
            for (k, &p) in pts.iter().enumerate() {
                if k == i0 || k == i1 {
                    continue;
                }
                let t = (p - a).dot(ab) / len_sq;
                let margin = tolerance / len_sq.sqrt();
                if t <= margin || t >= 1.0 - margin {
                    continue;
                }
                let foot = lerp_point(a, b, t);
                if dist_sq(p, foot) < tolerance * tolerance {
                    on_seg.push((t, k));
                }
            }
            on_seg.sort_by(|x, y| x.0.total_cmp(&y.0));
            let mut prev = i0;
            for &(_, k) in &on_seg {
                if prev != k {
                    chains.push((prev, k));
                }
                prev = k;
            }
            if prev != i1 {
                chains.push((prev, i1));
            }
        }

        // Classify points; feed on-edge points to the global map.
        let corners = [v0, v1, v2];
        for &p in &pts {
            match classify_point_site(p, v0, v1, v2, tolerance) {
                PointSite::Corner | PointSite::Interior => {}
                PointSite::Edge(e) => {
                    let (ea, eb) = (corners[e], corners[(e + 1) % 3]);
                    let entry = edge_points.entry(edge_key(ea, eb)).or_default();
                    let tol_sq = tolerance * tolerance;
                    if !entry.iter().any(|q| dist_sq(*q, p) < tol_sq) {
                        entry.push(p);
                    }
                }
            }
        }

        host_data.insert(
            host,
            HostSplit {
                pts,
                constraints: chains,
            },
        );
    }

    // Phase 2: re-triangulate every triangle that has constraints or that
    // received points on its edges from a neighbor's split.
    let mut positions = mesh.positions.clone();
    let mut normals = mesh.normals.clone();
    let mut triangles: Vec<[u32; 3]> = Vec::with_capacity(tri_count * 2);

    for i in 0..tri_count {
        let i0 = mesh.indices[i * 3] as usize;
        let i1 = mesh.indices[i * 3 + 1] as usize;
        let i2 = mesh.indices[i * 3 + 2] as usize;
        let v0 = mesh.positions[i0];
        let v1 = mesh.positions[i1];
        let v2 = mesh.positions[i2];

        let host = host_data.get(&i);
        let corners = [v0, v1, v2];
        let mut edge_pts: [Vec<Point3>; 3] = [Vec::new(), Vec::new(), Vec::new()];
        for e in 0..3 {
            let (ea, eb) = (corners[e], corners[(e + 1) % 3]);
            if let Some(list) = edge_points.get(&edge_key(ea, eb)) {
                let dir = eb - ea;
                let len_sq = dir.dot(dir);
                if len_sq < 1e-30 {
                    continue;
                }
                let mut with_t: Vec<(f64, Point3)> = list
                    .iter()
                    .map(|&p| ((p - ea).dot(dir) / len_sq, p))
                    .filter(|&(t, _)| t > 0.0 && t < 1.0)
                    .collect();
                with_t.sort_by(|x, y| x.0.total_cmp(&y.0));
                edge_pts[e] = with_t.into_iter().map(|(_, p)| p).collect();
            }
        }

        let needs_split =
            host.is_some_and(|h| !h.pts.is_empty()) || edge_pts.iter().any(|l| !l.is_empty());
        if !needs_split {
            #[allow(clippy::cast_possible_truncation)]
            {
                triangles.push([i0 as u32, i1 as u32, i2 as u32]);
            }
            continue;
        }

        let sub_tris = retriangulate_conforming(v0, v1, v2, &edge_pts, host, tolerance)
            .unwrap_or_else(|| legacy_fan_split(v0, v1, v2, &edge_pts, host, tolerance));

        let n0 = mesh.normals[i0];
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
    }

    SplitMesh {
        positions,
        normals,
        triangles,
    }
}

/// Transversal crossing point of two 3D segments known to be near-coplanar
/// (both lie in the host triangle's plane). Returns the crossing point when
/// the segments genuinely cross in their interiors.
fn transversal_crossing(
    a0: Point3,
    a1: Point3,
    b0: Point3,
    b1: Point3,
    tolerance: f64,
) -> Option<Point3> {
    let da = a1 - a0;
    let db = b1 - b0;
    let n = da.cross(db);
    let n_len_sq = n.dot(n);
    let la_sq = da.dot(da);
    let lb_sq = db.dot(db);
    if n_len_sq < 1e-12 * la_sq * lb_sq {
        return None; // near-parallel: handled by collinear chaining
    }
    let r = b0 - a0;
    // Solve a0 + t*da = b0 + u*db in the least-squares sense.
    let t = r.cross(db).dot(n) / n_len_sq;
    let u = r.cross(da).dot(n) / n_len_sq;
    let margin_t = tolerance / la_sq.sqrt();
    let margin_u = tolerance / lb_sq.sqrt();
    if t <= margin_t || t >= 1.0 - margin_t || u <= margin_u || u >= 1.0 - margin_u {
        return None;
    }
    let pa = lerp_point(a0, a1, t);
    let pb = lerp_point(b0, b1, u);
    if dist_sq(pa, pb) > tolerance * tolerance * 4.0 {
        return None; // skew, not actually crossing
    }
    Some(pa)
}

/// Classify a point against a host triangle: coincident with a corner, on an
/// edge (0: v0-v1, 1: v1-v2, 2: v2-v0), or interior.
fn classify_point_site(p: Point3, v0: Point3, v1: Point3, v2: Point3, tolerance: f64) -> PointSite {
    let tol_sq = tolerance * tolerance;
    if dist_sq(p, v0) < tol_sq || dist_sq(p, v1) < tol_sq || dist_sq(p, v2) < tol_sq {
        return PointSite::Corner;
    }
    if let Some(e) = point_on_edge(p, v0, v1, v2, tolerance) {
        return PointSite::Edge(e);
    }
    PointSite::Interior
}

/// Conforming re-triangulation of one host triangle via constrained
/// Delaunay: corners + edge points + interior points, with the boundary
/// chains and the intersection segments as constraints.
///
/// Returns `None` when the CDT fails (point location or constraint
/// recovery); the caller falls back to the legacy fan split.
#[allow(clippy::too_many_lines)]
fn retriangulate_conforming(
    v0: Point3,
    v1: Point3,
    v2: Point3,
    edge_pts: &[Vec<Point3>; 3],
    host: Option<&HostSplit>,
    tolerance: f64,
) -> Option<Vec<(Point3, Point3, Point3)>> {
    // Build an orthonormal in-plane frame.
    let e01 = v1 - v0;
    let ng = e01.cross(v2 - v0);
    let ng_len = ng.length();
    let e01_len = e01.length();
    if ng_len < 1e-30 || e01_len < 1e-30 {
        return None;
    }
    let w = ng * (1.0 / ng_len);
    let u = e01 * (1.0 / e01_len);
    let vax = w.cross(u);
    let to2d = |p: Point3| -> Point2 {
        let d = p - v0;
        Point2::new(d.dot(u), d.dot(vax))
    };

    let corners2d = [to2d(v0), to2d(v1), to2d(v2)];
    let mut min = corners2d[0];
    let mut max = corners2d[0];
    for c in &corners2d[1..] {
        min = Point2::new(min.x().min(c.x()), min.y().min(c.y()));
        max = Point2::new(max.x().max(c.x()), max.y().max(c.y()));
    }

    let n_pts = host.map_or(0, |h| h.pts.len());
    let mut cdt = Cdt::with_capacity(
        (min, max),
        3 + n_pts + edge_pts.iter().map(Vec::len).sum::<usize>(),
    );

    // Map from CDT vertex index to 3D position (first writer wins).
    let mut back: DetHashMap<usize, Point3> = DetHashMap::default();
    let insert = |cdt: &mut Cdt, p: Point3, back: &mut DetHashMap<usize, Point3>| {
        let idx = cdt.insert_point(to2d(p)).ok()?;
        back.entry(idx).or_insert(p);
        Some(idx)
    };

    let c0 = insert(&mut cdt, v0, &mut back)?;
    let c1 = insert(&mut cdt, v1, &mut back)?;
    let c2 = insert(&mut cdt, v2, &mut back)?;
    let corner_idx = [c0, c1, c2];

    // Boundary chains: corner -> sorted edge points -> next corner.
    for e in 0..3 {
        let mut prev = corner_idx[e];
        for &p in &edge_pts[e] {
            let idx = insert(&mut cdt, p, &mut back)?;
            if idx != prev {
                cdt.insert_constraint(prev, idx).ok()?;
                prev = idx;
            }
        }
        let last = corner_idx[(e + 1) % 3];
        if prev != last {
            cdt.insert_constraint(prev, last).ok()?;
        }
    }

    // Interior points and intersection-segment constraints.
    if let Some(h) = host {
        let mut pt_idx: Vec<usize> = Vec::with_capacity(h.pts.len());
        for &p in &h.pts {
            pt_idx.push(insert(&mut cdt, p, &mut back)?);
        }
        for &(i0, i1) in &h.constraints {
            let (a, b) = (pt_idx[i0], pt_idx[i1]);
            if a != b {
                cdt.insert_constraint(a, b).ok()?;
            }
        }
    }

    // Extract, orient, and sanity-check the sub-triangles.
    let verts2d = cdt.vertices().to_vec();
    let mut out: Vec<(Point3, Point3, Point3)> = Vec::new();
    let mut area_sum = 0.0_f64;
    for (ia, ib, ic) in cdt.triangles() {
        let (pa, pb, pc) = (verts2d[ia], verts2d[ib], verts2d[ic]);
        let signed2 = (pb.x() - pa.x()) * (pc.y() - pa.y()) - (pb.y() - pa.y()) * (pc.x() - pa.x());
        area_sum += signed2.abs() * 0.5;
        let (qa, qb, qc) = (
            back.get(&ia).copied()?,
            back.get(&ib).copied()?,
            back.get(&ic).copied()?,
        );
        // CCW in the (u, vax) frame maps to a 3D normal along +w (the host
        // triangle's outward geometric normal); flip CW triangles.
        if signed2 >= 0.0 {
            out.push((qa, qb, qc));
        } else {
            out.push((qa, qc, qb));
        }
    }

    // The union of sub-triangles must tile the host triangle exactly.
    let host_area = 0.5 * ng_len;
    if (area_sum - host_area).abs() > (host_area * 1e-6).max(tolerance) {
        return None;
    }

    Some(out)
}

/// Legacy point-insertion fan split, used only when the CDT path fails.
/// Not conforming (segments are not constrained), but never worse than the
/// pre-CDT behavior of this module.
fn legacy_fan_split(
    v0: Point3,
    v1: Point3,
    v2: Point3,
    edge_pts: &[Vec<Point3>; 3],
    host: Option<&HostSplit>,
    tolerance: f64,
) -> Vec<(Point3, Point3, Point3)> {
    let mut points: Vec<Point3> = Vec::new();
    for list in edge_pts {
        for &p in list {
            maybe_add_unique(&mut points, p, tolerance);
        }
    }
    if let Some(h) = host {
        for &p in &h.pts {
            maybe_add_unique(&mut points, p, tolerance);
        }
    }
    split_triangle_by_points(v0, v1, v2, &points, tolerance)
}

/// Add a point to a list if no existing point is within tolerance.
fn maybe_add_unique(pts: &mut Vec<Point3>, p: Point3, tolerance: f64) {
    let tol_sq = tolerance * tolerance;
    for existing in pts.iter() {
        if dist_sq(*existing, p) < tol_sq {
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
            if !inserted && let Some(sub) = try_split_triangle(*tv0, *tv1, *tv2, pt, tolerance) {
                new_tris.extend(sub);
                inserted = true;
                continue;
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

/// Classification of a sub-triangle against the other mesh.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum TriState {
    /// Strictly inside the other solid.
    Inside,
    /// Strictly outside the other solid.
    Outside,
    /// Lying on the other mesh's surface, normals pointing the same way.
    OnSame,
    /// Lying on the other mesh's surface, normals opposed.
    OnOpp,
}

/// Classify each sub-triangle of a split mesh against the other mesh.
///
/// A sub-triangle whose centroid lies on the other mesh's surface (within a
/// small epsilon, with near-parallel normals) is classified `OnSame`/`OnOpp`
/// — the winding number is exactly ½ there and must not be used as an
/// inside/outside coin flip. All other centroids use the generalized winding
/// number.
fn classify_split_triangles(
    split: &SplitMesh,
    other_mesh: &TriangleMesh,
    other_bvh: &Bvh,
    tolerance: f64,
) -> Vec<TriState> {
    // Must not exceed the contact/co-refinement tolerance: faces farther
    // apart than `tolerance` are never co-refined, so classifying them
    // OnSame/OnOpp would drop them in assembly and open the result.
    let eps_on = tolerance.max(1e-9);
    split
        .triangles
        .iter()
        .map(|tri| {
            let v0 = split.positions[tri[0] as usize];
            let v1 = split.positions[tri[1] as usize];
            let v2 = split.positions[tri[2] as usize];
            let centroid = triangle_centroid(v0, v1, v2);
            let host_n = (v1 - v0).cross(v2 - v0);
            let host_n_len = host_n.length();

            if host_n_len > 1e-30
                && let Some((dist, other_n)) =
                    closest_surface_normal(centroid, other_mesh, other_bvh, eps_on)
                && dist < eps_on
            {
                let cos = host_n.dot(other_n) / (host_n_len * other_n.length().max(1e-30));
                if cos > 0.9 {
                    return TriState::OnSame;
                }
                if cos < -0.9 {
                    return TriState::OnOpp;
                }
            }

            let wn = winding_number_at_point(centroid, other_mesh);
            if wn.abs() > 0.5 {
                TriState::Inside
            } else {
                TriState::Outside
            }
        })
        .collect()
}

/// Distance from a point to the nearest triangle of a mesh within `radius`,
/// plus that triangle's geometric (winding-derived, outward) normal.
fn closest_surface_normal(
    p: Point3,
    mesh: &TriangleMesh,
    bvh: &Bvh,
    radius: f64,
) -> Option<(f64, Vec3)> {
    let query = Aabb3::from_points([p]).expanded(radius);
    let candidates = bvh.query_overlap(&query);
    let mut best: Option<(f64, Vec3)> = None;
    for t in candidates {
        let (v0, v1, v2) = get_triangle(mesh, t);
        let n = (v1 - v0).cross(v2 - v0);
        if n.dot(n) < 1e-30 {
            continue;
        }
        let cp = closest_point_on_triangle(p, v0, v1, v2);
        let d = dist_sq(p, cp).sqrt();
        if best.is_none_or(|(bd, _)| d < bd) {
            best = Some((d, n));
        }
    }
    best
}

/// Closest point on a triangle to a point (Ericson, Real-Time Collision
/// Detection, 5.1.5).
#[allow(clippy::many_single_char_names)]
fn closest_point_on_triangle(p: Point3, a: Point3, b: Point3, c: Point3) -> Point3 {
    let ab = b - a;
    let ac = c - a;
    let ap = p - a;

    let d1 = ab.dot(ap);
    let d2 = ac.dot(ap);
    if d1 <= 0.0 && d2 <= 0.0 {
        return a;
    }

    let bp = p - b;
    let d3 = ab.dot(bp);
    let d4 = ac.dot(bp);
    if d3 >= 0.0 && d4 <= d3 {
        return b;
    }

    let vc = d1.mul_add(d4, -(d3 * d2));
    if vc <= 0.0 && d1 >= 0.0 && d3 <= 0.0 {
        let t = d1 / (d1 - d3);
        return lerp_point(a, b, t);
    }

    let cp = p - c;
    let d5 = ab.dot(cp);
    let d6 = ac.dot(cp);
    if d6 >= 0.0 && d5 <= d6 {
        return c;
    }

    let vb = d5.mul_add(d2, -(d1 * d6));
    if vb <= 0.0 && d2 >= 0.0 && d6 <= 0.0 {
        let t = d2 / (d2 - d6);
        return lerp_point(a, c, t);
    }

    let va = d3.mul_add(d6, -(d5 * d4));
    if va <= 0.0 && (d4 - d3) >= 0.0 && (d5 - d6) >= 0.0 {
        let t = (d4 - d3) / ((d4 - d3) + (d5 - d6));
        return lerp_point(b, c, t);
    }

    let denom = 1.0 / (va + vb + vc);
    let v = vb * denom;
    let w = vc * denom;
    Point3::new(
        ac.x().mul_add(w, ab.x().mul_add(v, a.x())),
        ac.y().mul_add(w, ab.y().mul_add(v, a.y())),
        ac.z().mul_add(w, ab.z().mul_add(v, a.z())),
    )
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

/// Assemble the result mesh from classified sub-triangles.
///
/// Selection logic (`OnSame`/`OnOpp` are coincident-surface triangles; mesh
/// A owns the single kept copy of any coincident boundary region, mesh B's
/// coincident triangles are always dropped):
/// - [`BooleanOp::Fuse`]: A outside B or `OnSame`; B strictly outside A
/// - [`BooleanOp::Cut`]: A outside B or `OnOpp`; B strictly inside A (flipped)
/// - [`BooleanOp::Intersect`]: A inside B or `OnSame`; B strictly inside A
fn assemble_result(
    split_a: &SplitMesh,
    split_b: &SplitMesh,
    classify_a: &[TriState],
    classify_b: &[TriState],
    op: BooleanOp,
) -> TriangleMesh {
    let mut positions = Vec::new();
    let mut normals = Vec::new();
    let mut indices = Vec::new();

    for (i, tri) in split_a.triangles.iter().enumerate() {
        let state = classify_a.get(i).copied().unwrap_or(TriState::Outside);
        let keep = match op {
            BooleanOp::Fuse => matches!(state, TriState::Outside | TriState::OnSame),
            BooleanOp::Cut => matches!(state, TriState::Outside | TriState::OnOpp),
            BooleanOp::Intersect => matches!(state, TriState::Inside | TriState::OnSame),
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

    for (i, tri) in split_b.triangles.iter().enumerate() {
        let state = classify_b.get(i).copied().unwrap_or(TriState::Outside);
        let (keep, flip) = match op {
            BooleanOp::Fuse => (state == TriState::Outside, false),
            BooleanOp::Cut => (state == TriState::Inside, true),
            BooleanOp::Intersect => (state == TriState::Inside, false),
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

/// Extract the three vertices of a triangle from a mesh.
fn get_triangle(mesh: &TriangleMesh, tri_idx: usize) -> (Point3, Point3, Point3) {
    let base = tri_idx * 3;
    let i0 = mesh.indices[base] as usize;
    let i1 = mesh.indices[base + 1] as usize;
    let i2 = mesh.indices[base + 2] as usize;
    (mesh.positions[i0], mesh.positions[i1], mesh.positions[i2])
}

#[cfg(test)]
mod tests {
    #![allow(clippy::unwrap_used, clippy::expect_used)]

    use super::*;

    /// Create a tetrahedron mesh centered at a point.
    fn tetrahedron_mesh(center: Point3, size: f64) -> TriangleMesh {
        let s = size;
        let v0 = Point3::new(center.x() + s, center.y() + s, center.z() + s);
        let v1 = Point3::new(center.x() + s, center.y() - s, center.z() - s);
        let v2 = Point3::new(center.x() - s, center.y() + s, center.z() - s);
        let v3 = Point3::new(center.x() - s, center.y() - s, center.z() + s);

        let positions = [v0, v1, v2, v3];

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
        box_mesh_half_extents(center, Vec3::new(half_size, half_size, half_size))
    }

    /// Create an axis-aligned box mesh with per-axis half extents.
    fn box_mesh_half_extents(center: Point3, half: Vec3) -> TriangleMesh {
        let cx = center.x();
        let cy = center.y();
        let cz = center.z();
        let (sx, sy, sz) = (half.x(), half.y(), half.z());

        let verts = [
            Point3::new(cx - sx, cy - sy, cz - sz), // 0
            Point3::new(cx + sx, cy - sy, cz - sz), // 1
            Point3::new(cx + sx, cy + sy, cz - sz), // 2
            Point3::new(cx - sx, cy + sy, cz - sz), // 3
            Point3::new(cx - sx, cy - sy, cz + sz), // 4
            Point3::new(cx + sx, cy - sy, cz + sz), // 5
            Point3::new(cx + sx, cy + sy, cz + sz), // 6
            Point3::new(cx - sx, cy + sy, cz + sz), // 7
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

    /// Signed volume of a triangle mesh via the divergence theorem.
    fn mesh_signed_volume(mesh: &TriangleMesh) -> f64 {
        let mut vol = 0.0;
        for tri in mesh.indices.chunks_exact(3) {
            let a = mesh.positions[tri[0] as usize];
            let b = mesh.positions[tri[1] as usize];
            let c = mesh.positions[tri[2] as usize];
            let va = a - Point3::new(0.0, 0.0, 0.0);
            let vb = b - Point3::new(0.0, 0.0, 0.0);
            let vc = c - Point3::new(0.0, 0.0, 0.0);
            vol += va.dot(vb.cross(vc)) / 6.0;
        }
        vol
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

        // The intersection is a 1.5^3 cube.
        assert_eq!(result.boundary_edge_count, 0, "intersect should be closed");
        assert_eq!(
            result.non_manifold_edge_count, 0,
            "intersect should be manifold"
        );
        let vol = mesh_signed_volume(&result.mesh);
        assert!(
            (vol - 1.5_f64.powi(3)).abs() < 1e-9,
            "intersection volume should be 3.375, got {vol}"
        );
    }

    #[test]
    fn mesh_boolean_overlapping_cut_watertight() {
        let a = box_mesh(Point3::new(0.0, 0.0, 0.0), 1.0);
        let b = box_mesh(Point3::new(0.5, 0.5, 0.5), 1.0);

        let result = mesh_boolean(&a, &b, BooleanOp::Cut, 1e-7).unwrap();
        assert_eq!(result.boundary_edge_count, 0, "cut should be closed");
        assert_eq!(result.non_manifold_edge_count, 0, "cut should be manifold");
        let vol = mesh_signed_volume(&result.mesh);
        assert!(
            (vol - (8.0 - 1.5_f64.powi(3))).abs() < 1e-9,
            "cut volume should be 4.625, got {vol}"
        );
    }

    #[test]
    fn mesh_boolean_coincident_wall_cut() {
        // B occupies the left half of A, sharing three walls exactly:
        // the coincident-contact class that winding-number classification
        // alone gets wrong (winding is exactly 1/2 on the shared walls).
        let a = box_mesh_half_extents(Point3::new(0.0, 0.0, 0.0), Vec3::new(2.0, 1.0, 1.0));
        let b = box_mesh_half_extents(Point3::new(-1.0, 0.0, 0.0), Vec3::new(1.0, 1.0, 1.0));

        let result = mesh_boolean(&a, &b, BooleanOp::Cut, 1e-7).unwrap();
        assert_eq!(
            result.boundary_edge_count, 0,
            "coincident-wall cut should be closed"
        );
        assert_eq!(
            result.non_manifold_edge_count, 0,
            "coincident-wall cut should be manifold"
        );
        let vol = mesh_signed_volume(&result.mesh);
        assert!(
            (vol - 8.0).abs() < 1e-9,
            "remaining half should have volume 8, got {vol}"
        );
    }

    #[test]
    fn mesh_boolean_coincident_wall_fuse() {
        // Two boxes sharing a full wall: fuse must dissolve the shared wall.
        let a = box_mesh_half_extents(Point3::new(-1.0, 0.0, 0.0), Vec3::new(1.0, 1.0, 1.0));
        let b = box_mesh_half_extents(Point3::new(1.0, 0.0, 0.0), Vec3::new(1.0, 1.0, 1.0));

        let result = mesh_boolean(&a, &b, BooleanOp::Fuse, 1e-7).unwrap();
        assert_eq!(
            result.boundary_edge_count, 0,
            "coincident-wall fuse should be closed"
        );
        assert_eq!(
            result.non_manifold_edge_count, 0,
            "coincident-wall fuse should be manifold"
        );
        let vol = mesh_signed_volume(&result.mesh);
        assert!(
            (vol - 16.0).abs() < 1e-9,
            "fused volume should be 16, got {vol}"
        );
    }

    #[test]
    fn mesh_boolean_coplanar_top_stack_fuse() {
        // A small box sitting exactly on top of a bigger one: the contact
        // patch is coincident with opposite normals and must vanish, while
        // the big box's top face must conform to the small footprint.
        let a = box_mesh_half_extents(Point3::new(0.0, 0.0, 0.0), Vec3::new(2.0, 2.0, 1.0));
        let b = box_mesh_half_extents(Point3::new(0.0, 0.0, 1.5), Vec3::new(0.5, 0.5, 0.5));

        let result = mesh_boolean(&a, &b, BooleanOp::Fuse, 1e-7).unwrap();
        assert_eq!(result.boundary_edge_count, 0, "stack fuse should be closed");
        assert_eq!(
            result.non_manifold_edge_count, 0,
            "stack fuse should be manifold"
        );
        let vol = mesh_signed_volume(&result.mesh);
        assert!(
            (vol - 33.0).abs() < 1e-7,
            "stacked volume should be 32 + 1 = 33, got {vol}"
        );
    }

    #[test]
    fn mesh_boolean_near_disjoint_fuse_keeps_facing_walls() {
        // Two boxes separated by a gap wider than the intersection tolerance
        // but inside a naive "coincident" window: the facing walls are never
        // co-refined, so classifying them OnOpp would drop them and open both
        // solids. They must classify Outside and survive the fuse intact.
        let gap = 5e-7;
        let a = box_mesh_half_extents(Point3::new(-1.0, 0.0, 0.0), Vec3::new(1.0, 1.0, 1.0));
        let b = box_mesh_half_extents(Point3::new(1.0 + gap, 0.0, 0.0), Vec3::new(1.0, 1.0, 1.0));

        let result = mesh_boolean(&a, &b, BooleanOp::Fuse, 1e-7).unwrap();
        assert_eq!(
            result.mesh.indices.len() / 3,
            24,
            "near-disjoint fuse must keep all 24 triangles"
        );
        assert_eq!(
            result.boundary_edge_count, 0,
            "near-disjoint fuse should be closed"
        );
        let vol = mesh_signed_volume(&result.mesh);
        assert!(
            (vol - 16.0).abs() < 1e-6,
            "near-disjoint fused volume should be 16, got {vol}"
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
