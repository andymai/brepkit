//! Analytic boolean fast path and supporting helpers.
//!
//! Contains `analytic_boolean` — the exact analytic pipeline that preserves
//! surface types (planes, cylinders, cones, spheres) through boolean
//! operations — plus utility functions shared with the tessellated path
//! and `compound_cut`.

use std::collections::{HashMap, HashSet};

use brepkit_math::aabb::Aabb3;
use brepkit_math::bvh::Bvh;
use brepkit_math::plane::plane_plane_intersection;
use brepkit_math::tolerance::Tolerance;
use brepkit_math::vec::{Point3, Vec3};
use brepkit_topology::Topology;
use brepkit_topology::edge::{Edge, EdgeCurve};
use brepkit_topology::face::{Face, FaceId, FaceSurface};
use brepkit_topology::shell::Shell;
use brepkit_topology::solid::{Solid, SolidId};
use brepkit_topology::vertex::{Vertex, VertexId};
use brepkit_topology::wire::WireId;
use brepkit_topology::wire::{OrientedEdge, Wire};

use super::assembly::{
    quantize, quantize_point, refine_boundary_edges, split_nonmanifold_edges,
    try_shared_boundary_fuse, vertex_merge_resolution,
};
use super::classify::{
    self, build_face_bvh, classify_point, guard_tangent_coplanar, polygon_centroid,
    try_build_analytic_classifier,
};
use super::fragments::{
    build_cone_barrel_wire, build_cylinder_barrel_wire, collect_analytic_vranges,
    create_band_fragments, curve_boundary_crossings, sample_edge_curve, split_cone_at_intersection,
    split_cylinder_at_intersection, split_sphere_at_intersection, tessellate_face_into_fragments,
};
use super::intersect::{
    cyrus_beck_clip, intersect_interval_lists, point_along_line, polygon_clip_intervals,
};
use super::precompute::{
    analytic_face_normal_d, collect_face_data, compute_v_range_hint, face_wire_aabb,
};
use super::split::split_face;
use super::types::{
    AnalyticClassifier, AnalyticFragment, BooleanOp, CLOSED_CURVE_SAMPLES, CurveClassification,
    FaceClass, FaceSnapshot, Source, select_fragment,
};
use super::{timer_elapsed_ms, timer_now};

use super::precompute::face_polygon;

// ---------------------------------------------------------------------------
// Helper functions (shared with mod.rs / compound_cut)
// ---------------------------------------------------------------------------

/// Collect `(FaceId.index(), normal, centroid)` for each face in a solid.
pub(super) fn collect_face_signatures(
    topo: &Topology,
    solid_id: SolidId,
) -> Result<Vec<(usize, Vec3, Point3)>, crate::OperationsError> {
    let solid = topo.solid(solid_id)?;
    let shell = topo.shell(solid.outer_shell())?;
    let mut result = Vec::with_capacity(shell.faces().len());

    for &fid in shell.faces() {
        let face = topo.face(fid)?;
        let verts = face_polygon(topo, fid)?;
        let normal = if let FaceSurface::Plane { normal, .. } = face.surface() {
            *normal
        } else if verts.len() >= 3 {
            let e1 = verts[1] - verts[0];
            let e2 = verts[2] - verts[0];
            e1.cross(e2).normalize().unwrap_or(Vec3::new(0.0, 0.0, 1.0))
        } else {
            Vec3::new(0.0, 0.0, 1.0)
        };

        let centroid = polygon_centroid(&verts);
        result.push((fid.index(), normal, centroid));
    }

    Ok(result)
}

// ── Analytic boolean fast path ─────────────────────────────────────────

/// Compute an AABB for an analytic face that accounts for the surface extent,
/// not just the wire vertices.
///
/// For planar faces the wire vertices are sufficient. For curved faces (sphere,
/// cylinder, etc.) the surface bulges beyond the wire boundary, so we union
/// the vertex-based AABB with the surface's own bounding box.
pub(super) fn surface_aware_aabb(
    surface: &FaceSurface,
    vertices: &[Point3],
    tol: Tolerance,
) -> Aabb3 {
    let wire_bb = Aabb3::from_points(vertices.iter().copied());
    let bb = match surface {
        FaceSurface::Plane { .. } => wire_bb,
        FaceSurface::Sphere(s) => {
            // Full sphere AABB: center +/- radius on all axes.
            // The wire vertices lie on the equator so they don't capture the
            // pole extent. Using the full sphere is conservative but correct.
            let c = s.center();
            let r = s.radius();
            Aabb3 {
                min: Point3::new(c.x() - r, c.y() - r, c.z() - r),
                max: Point3::new(c.x() + r, c.y() + r, c.z() + r),
            }
        }
        FaceSurface::Cylinder(c) => {
            // Cylinder: the surface extends radially from the axis through the
            // wire region. Union the wire BB with radius expansion perpendicular
            // to the axis.
            let r = c.radius();
            let ax = c.axis();
            // Expand perpendicular to axis by radius.
            let dx = r * (1.0 - ax.x() * ax.x()).sqrt();
            let dy = r * (1.0 - ax.y() * ax.y()).sqrt();
            let dz = r * (1.0 - ax.z() * ax.z()).sqrt();
            Aabb3 {
                min: Point3::new(
                    wire_bb.min.x() - dx,
                    wire_bb.min.y() - dy,
                    wire_bb.min.z() - dz,
                ),
                max: Point3::new(
                    wire_bb.max.x() + dx,
                    wire_bb.max.y() + dy,
                    wire_bb.max.z() + dz,
                ),
            }
        }
        FaceSurface::Cone(c) => {
            // Expand the wire BB perpendicular to the cone axis by the
            // cone radius at the wire boundary (same approach as cylinder).
            let ax = c.axis();
            // Max distance from apex along the axis within the wire region.
            let apex = c.apex();
            let max_dist = vertices
                .iter()
                .map(|v| {
                    let d = *v - apex;
                    (d.x() * ax.x() + d.y() * ax.y() + d.z() * ax.z()).abs()
                })
                .fold(0.0_f64, f64::max);
            let r_max = max_dist * c.half_angle().tan().abs();
            let dx = r_max * (1.0 - ax.x() * ax.x()).sqrt();
            let dy = r_max * (1.0 - ax.y() * ax.y()).sqrt();
            let dz = r_max * (1.0 - ax.z() * ax.z()).sqrt();
            Aabb3 {
                min: Point3::new(
                    wire_bb.min.x() - dx,
                    wire_bb.min.y() - dy,
                    wire_bb.min.z() - dz,
                ),
                max: Point3::new(
                    wire_bb.max.x() + dx,
                    wire_bb.max.y() + dy,
                    wire_bb.max.z() + dz,
                ),
            }
        }
        FaceSurface::Torus(t) => {
            // The torus extends R+r in the ring plane and r along the axis.
            let c = t.center();
            let ring_r = t.major_radius() + t.minor_radius();
            let tube_r = t.minor_radius();
            let ax = t.z_axis();
            // Expand by ring_r perpendicular to axis, tube_r along axis.
            let dx = ring_r * (1.0 - ax.x() * ax.x()).sqrt() + tube_r * ax.x().abs();
            let dy = ring_r * (1.0 - ax.y() * ax.y()).sqrt() + tube_r * ax.y().abs();
            let dz = ring_r * (1.0 - ax.z() * ax.z()).sqrt() + tube_r * ax.z().abs();
            Aabb3 {
                min: Point3::new(c.x() - dx, c.y() - dy, c.z() - dz),
                max: Point3::new(c.x() + dx, c.y() + dy, c.z() + dz),
            }
        }
        FaceSurface::Nurbs(_) => {
            // NURBS: use wire vertices (conservative enough).
            wire_bb
        }
    };
    bb.expanded(tol.linear)
}

/// Check if a solid is composed entirely of analytic surfaces (no NURBS).
pub(super) fn is_all_analytic(
    topo: &Topology,
    solid: SolidId,
) -> Result<bool, crate::OperationsError> {
    let s = topo.solid(solid)?;
    let shell = topo.shell(s.outer_shell())?;
    for &fid in shell.faces() {
        let face = topo.face(fid)?;
        if matches!(face.surface(), FaceSurface::Nurbs(_)) {
            return Ok(false);
        }
    }
    Ok(true)
}

/// Check if a solid contains any torus faces.
pub(super) fn has_torus(topo: &Topology, solid: SolidId) -> Result<bool, crate::OperationsError> {
    let s = topo.solid(solid)?;
    let shell = topo.shell(s.outer_shell())?;
    for &fid in shell.faces() {
        let face = topo.face(fid)?;
        if matches!(face.surface(), FaceSurface::Torus(_)) {
            return Ok(true);
        }
    }
    Ok(false)
}

/// Build `edge_curves` for a face polygon by examining the source face's wire edges.
///
/// When the outer wire contains a single closed Circle or Ellipse edge, the
/// polygon vertices all came from sampling that edge. Returns
/// `vec![Some(curve)]` (length 1) to signal a single-closed-curve boundary.
/// Otherwise returns `vec![None; n]` for n polygon vertices.
pub(super) fn edge_curves_from_face(
    topo: &Topology,
    face_id: FaceId,
    n_verts: usize,
) -> Vec<Option<EdgeCurve>> {
    let Ok(face) = topo.face(face_id) else {
        return vec![None; n_verts];
    };
    let Ok(wire) = topo.wire(face.outer_wire()) else {
        return vec![None; n_verts];
    };
    let edges = wire.edges();
    // Single closed Circle or Ellipse edge -> single-curve boundary.
    if edges.len() == 1
        && let Ok(edge) = topo.edge(edges[0].edge())
        && edge.start() == edge.end()
        && matches!(edge.curve(), EdgeCurve::Circle(_) | EdgeCurve::Ellipse(_))
    {
        return vec![Some(edge.curve().clone())];
    }
    vec![None; n_verts]
}

/// Check whether a polygon (given its vertices and face normal) is convex.
///
/// Returns `true` if all cross products of consecutive edge pairs point in the
/// same direction as `face_normal`. Degenerate edges (cross product near zero)
/// are treated as locally convex -- correct for both collinear and coincident
/// vertices.
pub(super) fn is_polygon_convex(verts: &[Point3], face_normal: &Vec3) -> bool {
    let n = verts.len();
    if n < 3 {
        return true;
    }
    for i in 0..n {
        let j = (i + 1) % n;
        let k = (i + 2) % n;
        let e1 = verts[j] - verts[i];
        let e2 = verts[k] - verts[j];
        let cross = e1.cross(e2);
        // If the cross product opposes the face normal, the polygon is concave
        // at this vertex. Skip degenerate (zero-length) edges.
        // 1e-12: degenerate guard on unnormalized cross product (|e1||e2|*sin th).
        // For sub-mm edges (0.1mm), smallest detectable concavity is ~1e-4 rad --
        // well above this threshold.
        if cross.dot(*face_normal) < -1e-12 {
            return false;
        }
    }
    true
}

/// Compute a plane-plane intersection chord clipped to both face polygons.
///
/// Uses `cyrus_beck_clip` (fast, exact for convex polygons) as the primary
/// clipper. Falls back to `polygon_clip_intervals` when a polygon is detected
/// as non-convex, since `cyrus_beck_clip` silently produces wrong results on
/// concave boundaries.
pub(super) fn plane_plane_chord_analytic(
    normal_a: Vec3,
    d_a: f64,
    verts_a: &[Point3],
    normal_b: Vec3,
    d_b: f64,
    verts_b: &[Point3],
    tol: Tolerance,
) -> Option<(Point3, Point3)> {
    let (line_origin, line_dir) =
        plane_plane_intersection(normal_a, d_a, normal_b, d_b, tol.angular)?;

    let convex_a = is_polygon_convex(verts_a, &normal_a);
    let convex_b = is_polygon_convex(verts_b, &normal_b);

    // Fast path: both polygons are convex -- use Cyrus-Beck directly.
    if convex_a && convex_b {
        let t_range_a = cyrus_beck_clip(&line_origin, &line_dir, verts_a, &normal_a, tol);
        let t_range_b = cyrus_beck_clip(&line_origin, &line_dir, verts_b, &normal_b, tol);

        let (t_min_a, t_max_a) = t_range_a?;
        let (t_min_b, t_max_b) = t_range_b?;

        let t_min = t_min_a.max(t_min_b);
        let t_max = t_max_a.min(t_max_b);

        if t_max - t_min < tol.linear {
            return None;
        }

        return Some((
            point_along_line(&line_origin, &line_dir, t_min),
            point_along_line(&line_origin, &line_dir, t_max),
        ));
    }

    // Slow path: at least one polygon is concave. Use polygon_clip_intervals
    // which handles concave boundaries via winding-number interval testing.
    let intervals_a = if convex_a {
        match cyrus_beck_clip(&line_origin, &line_dir, verts_a, &normal_a, tol) {
            Some(iv) => vec![iv],
            None => return None,
        }
    } else {
        polygon_clip_intervals(&line_origin, &line_dir, verts_a, &normal_a, tol)
    };
    let intervals_b = if convex_b {
        match cyrus_beck_clip(&line_origin, &line_dir, verts_b, &normal_b, tol) {
            Some(iv) => vec![iv],
            None => return None,
        }
    } else {
        polygon_clip_intervals(&line_origin, &line_dir, verts_b, &normal_b, tol)
    };

    // Intersect the two interval sets (both already sorted by t).
    let overlaps = intersect_interval_lists(&intervals_a, &intervals_b, tol.linear);

    // Pick the largest finite overlap interval as the chord.
    // polygon_clip_intervals can emit half-infinite sentinel intervals
    // (e.g. (-inf, t] or [t, inf)) for lines fully inside a concave face;
    // these must be skipped to avoid degenerate chord endpoints.
    //
    // For highly concave faces (C-shape, U-shape) the intersection line may
    // produce multiple disjoint finite intervals.  We return only the longest
    // one; secondary intervals are discarded.  This is a known limitation of
    // the single-chord return type -- log a warning in debug builds so the gap
    // is observable without any release overhead.
    #[cfg(debug_assertions)]
    {
        let finite_count = overlaps
            .iter()
            .filter(|(lo, hi)| lo.is_finite() && hi.is_finite())
            .count();
        if finite_count > 1 {
            log::warn!(
                "plane_plane_chord_analytic: {finite_count} finite overlap intervals for \
                 concave face; only largest chord returned — secondary intervals discarded"
            );
        }
    }
    let &(t_min, t_max) = overlaps
        .iter()
        .filter(|(lo, hi)| lo.is_finite() && hi.is_finite())
        .max_by(|(a_lo, a_hi), (b_lo, b_hi)| {
            (a_hi - a_lo)
                .partial_cmp(&(b_hi - b_lo))
                .unwrap_or(std::cmp::Ordering::Equal)
        })?;

    Some((
        point_along_line(&line_origin, &line_dir, t_min),
        point_along_line(&line_origin, &line_dir, t_max),
    ))
}

// ---------------------------------------------------------------------------
// Main analytic boolean pipeline
// ---------------------------------------------------------------------------

/// Perform an analytic boolean preserving exact surface types.
///
/// This is the fast path for solids with only analytic faces. It computes
/// exact plane-analytic intersections and preserves `FaceSurface::Cylinder`,
/// `FaceSurface::Sphere`, etc. in the result.
///
/// Returns `Err` to signal fallback to the tessellated path.
#[allow(clippy::too_many_lines)]
#[allow(clippy::single_match_else)]
pub(super) fn analytic_boolean(
    topo: &mut Topology,
    op: BooleanOp,
    a: SolidId,
    b: SolidId,
    tol: Tolerance,
    deflection: f64,
) -> Result<SolidId, crate::OperationsError> {
    use brepkit_math::analytic_intersection::{
        ExactIntersectionCurve, exact_plane_analytic, intersect_analytic_analytic_bounded,
    };

    let _t_total = timer_now();

    // Collect face info for both solids.
    let solid_a = topo.solid(a)?;
    let shell_a = topo.shell(solid_a.outer_shell())?;
    let face_ids_a: Vec<FaceId> = shell_a.faces().to_vec();

    let solid_b = topo.solid(b)?;
    let shell_b = topo.shell(solid_b.outer_shell())?;
    let face_ids_b: Vec<FaceId> = shell_b.faces().to_vec();

    // ── Pre-snapshot AABB filter ────────────────────────────────────────
    // Compute per-face wire AABBs (cheap: vertex walk + 4-point curve
    // sampling for closed edges). Union them into per-solid overall AABBs.
    // Faces whose wire AABB doesn't overlap the opposing solid's overall
    // AABB cannot intersect any face of that solid — skip the expensive
    // face_polygon() sampling and mark them as passthrough.
    let wire_aabbs_a: Vec<Aabb3> = face_ids_a
        .iter()
        .map(|&fid| face_wire_aabb(topo, fid))
        .collect::<Result<Vec<_>, _>>()?;
    let wire_aabbs_b: Vec<Aabb3> = face_ids_b
        .iter()
        .map(|&fid| face_wire_aabb(topo, fid))
        .collect::<Result<Vec<_>, _>>()?;
    let a_overall_aabb = wire_aabbs_a
        .iter()
        .copied()
        .reduce(Aabb3::union)
        .ok_or_else(|| crate::OperationsError::InvalidInput {
            reason: "solid A has no faces".into(),
        })?;
    let b_overall_aabb = wire_aabbs_b
        .iter()
        .copied()
        .reduce(Aabb3::union)
        .ok_or_else(|| crate::OperationsError::InvalidInput {
            reason: "solid B has no faces".into(),
        })?;

    // ── Shared-boundary Fuse fast path ──────────────────────────────────
    // For Fuse of two all-planar solids: if they share exactly one coplanar
    // face (opposite normals, matching extent), merge by removing the shared
    // face pair and combining remaining faces. O(F) instead of full boolean.
    if op == BooleanOp::Fuse {
        if let Some(result) = try_shared_boundary_fuse(topo, a, b, &face_ids_a, &face_ids_b, tol)? {
            log::debug!("[analytic_boolean] shared-boundary fuse fast path");
            return Ok(result);
        }
    }

    let _t_snap = timer_now();
    let mut snaps_a = Vec::new();
    let mut passthrough_a: Vec<FaceId> = Vec::new();
    for (i, &fid) in face_ids_a.iter().enumerate() {
        if wire_aabbs_a[i].intersects(b_overall_aabb) {
            let face = topo.face(fid)?;
            let surface = face.surface().clone();
            let reversed = face.is_reversed();
            let verts = face_polygon(topo, fid)?;
            let (normal, d) = analytic_face_normal_d(&surface, &verts);
            snaps_a.push(FaceSnapshot {
                id: fid,
                surface,
                vertices: verts,
                normal,
                d,
                reversed,
            });
        } else {
            passthrough_a.push(fid);
        }
    }

    let mut snaps_b = Vec::new();
    let mut passthrough_b: Vec<FaceId> = Vec::new();
    for (i, &fid) in face_ids_b.iter().enumerate() {
        if wire_aabbs_b[i].intersects(a_overall_aabb) {
            let face = topo.face(fid)?;
            let surface = face.surface().clone();
            let reversed = face.is_reversed();
            let verts = face_polygon(topo, fid)?;
            let (normal, d) = analytic_face_normal_d(&surface, &verts);
            snaps_b.push(FaceSnapshot {
                id: fid,
                surface,
                vertices: verts,
                normal,
                d,
                reversed,
            });
        } else {
            passthrough_b.push(fid);
        }
    }

    log::debug!(
        "[boolean] snapshots: {:.3}ms (A={} snap + {} pass, B={} snap + {} pass)",
        timer_elapsed_ms(_t_snap),
        snaps_a.len(),
        passthrough_a.len(),
        snaps_b.len(),
        passthrough_b.len()
    );

    // Compute AABBs for face pairs (surface-aware for non-planar faces).
    let _t_aabb = timer_now();
    let aabbs_a: Vec<Aabb3> = snaps_a
        .iter()
        .map(|s| surface_aware_aabb(&s.surface, &s.vertices, tol))
        .collect();
    let aabbs_b: Vec<Aabb3> = snaps_b
        .iter()
        .map(|s| surface_aware_aabb(&s.surface, &s.vertices, tol))
        .collect();

    // Find intersection curves for each face pair.
    // For each face that gets split, store the intersection points.
    let mut face_intersections_a: HashMap<usize, Vec<(Point3, Point3, Option<EdgeCurve>)>> =
        HashMap::new();
    let mut face_intersections_b: HashMap<usize, Vec<(Point3, Point3, Option<EdgeCurve>)>> =
        HashMap::new();

    // Deferred coplanar edge injection: B's polygon edges to add as chords on
    // A's cap (and vice versa). Only applied AFTER the intersection loop, and
    // only when the cap face ALSO has chord entries from non-coplanar pairs.
    let mut deferred_coplanar_edges_a: HashMap<usize, Vec<(Point3, Point3)>> = HashMap::new();
    let mut deferred_coplanar_edges_b: HashMap<usize, Vec<(Point3, Point3)>> = HashMap::new();

    // Track which faces have non-planar intersection partners (analytic-analytic).
    let mut has_analytic_analytic = false;

    // Track face indices that participate in analytic-analytic intersections
    // (e.g. cylinder-cylinder). These faces need tessellation, not chord splitting.
    let mut analytic_analytic_faces_a: HashSet<usize> = HashSet::new();
    let mut analytic_analytic_faces_b: HashSet<usize> = HashSet::new();

    // V-ranges of intersection curves on cylinder faces (for band-splitting
    // instead of full tessellation). Key: face index, Value: list of (v_min, v_max).
    let mut analytic_intersection_vranges_a: HashMap<usize, Vec<(f64, f64)>> = HashMap::new();
    let mut analytic_intersection_vranges_b: HashMap<usize, Vec<(f64, f64)>> = HashMap::new();

    // Track contained intersection curves (circle/ellipse fully inside a planar face).
    #[allow(clippy::items_after_statements)]
    struct ContainedCurve {
        plane_face_idx: usize,
        plane_source: Source,
        analytic_face_idx: usize,
        edge_curve: EdgeCurve,
    }
    let mut contained_curves: Vec<ContainedCurve> = Vec::new();

    // Build BVH over solid B's face AABBs for O(n log m) broad-phase instead of O(n*m).
    // Only worthwhile when B has enough faces to amortize BVH build cost.
    let analytic_bvh_b = if snaps_b.len() >= 16 {
        let b_bvh_entries: Vec<(usize, Aabb3)> = aabbs_b
            .iter()
            .enumerate()
            .map(|(i, aabb)| (i, *aabb))
            .collect();
        Some(Bvh::build(&b_bvh_entries))
    } else {
        None
    };
    log::debug!(
        "[boolean] aabb+bvh: {:.3}ms (A={}, B={})",
        timer_elapsed_ms(_t_aabb),
        aabbs_a.len(),
        aabbs_b.len()
    );

    let _t_isect = timer_now();
    for (ia, snap_a) in snaps_a.iter().enumerate() {
        // Use BVH for broad-phase when available, otherwise brute-force with AABB check.
        let candidates: Vec<usize> = if let Some(ref bvh) = analytic_bvh_b {
            bvh.query_overlap(&aabbs_a[ia])
        } else {
            (0..snaps_b.len())
                .filter(|&ib| aabbs_a[ia].intersects(aabbs_b[ib]))
                .collect()
        };
        for &ib in &candidates {
            let snap_b = &snaps_b[ib];

            let is_plane_a = matches!(snap_a.surface, FaceSurface::Plane { .. });
            let is_plane_b = matches!(snap_b.surface, FaceSurface::Plane { .. });

            if is_plane_a && is_plane_b {
                // Detect coplanar containment for deferred edge injection.
                {
                    let d_diff = (snap_a.d - snap_b.d).abs();
                    let dot = snap_a.normal.dot(snap_b.normal);
                    if d_diff < tol.linear && dot > 1.0 - tol.angular {
                        let b_in_a = snap_b.vertices.iter().all(|v| {
                            classify::point_in_face_3d(*v, &snap_a.vertices, &snap_a.normal)
                        });
                        let a_in_b = snap_a.vertices.iter().all(|v| {
                            classify::point_in_face_3d(*v, &snap_b.vertices, &snap_b.normal)
                        });
                        if b_in_a && !a_in_b {
                            deferred_coplanar_edges_a
                                .entry(ia)
                                .or_default()
                                .extend(snap_b.vertices.windows(2).map(|w| (w[0], w[1])));
                            let n = snap_b.vertices.len();
                            deferred_coplanar_edges_a
                                .entry(ia)
                                .or_default()
                                .push((snap_b.vertices[n - 1], snap_b.vertices[0]));
                        } else if a_in_b && !b_in_a {
                            deferred_coplanar_edges_b
                                .entry(ib)
                                .or_default()
                                .extend(snap_a.vertices.windows(2).map(|w| (w[0], w[1])));
                            let n = snap_a.vertices.len();
                            deferred_coplanar_edges_b
                                .entry(ib)
                                .or_default()
                                .push((snap_a.vertices[n - 1], snap_a.vertices[0]));
                        }
                    }
                }
                // Plane-plane: use existing logic (chord intersection).
                if let Some(seg) = plane_plane_chord_analytic(
                    snap_a.normal,
                    snap_a.d,
                    &snap_a.vertices,
                    snap_b.normal,
                    snap_b.d,
                    &snap_b.vertices,
                    tol,
                ) {
                    face_intersections_a
                        .entry(ia)
                        .or_default()
                        .push((seg.0, seg.1, None));
                    face_intersections_b
                        .entry(ib)
                        .or_default()
                        .push((seg.0, seg.1, None));
                }
            } else if is_plane_a && !is_plane_b {
                // Plane cuts analytic surface.
                let Some(analytic_surf) = snap_b.surface.as_analytic() else {
                    has_analytic_analytic = true;
                    continue;
                };

                let epa_result = exact_plane_analytic(analytic_surf, snap_a.normal, snap_a.d);
                if let Ok(curves) = epa_result {
                    for curve in curves {
                        let edge_curve = match &curve {
                            ExactIntersectionCurve::Circle(c) => Some(EdgeCurve::Circle(c.clone())),
                            ExactIntersectionCurve::Ellipse(e) => {
                                Some(EdgeCurve::Ellipse(e.clone()))
                            }
                            ExactIntersectionCurve::Points(_) => None,
                        };

                        let classification =
                            curve_boundary_crossings(&curve, &snap_a.vertices, snap_a.normal, tol);
                        match classification {
                            CurveClassification::Crossings(ref samples) => {
                                for pair in samples.windows(2) {
                                    face_intersections_a.entry(ia).or_default().push((
                                        pair[0],
                                        pair[1],
                                        edge_curve.clone(),
                                    ));
                                    face_intersections_b.entry(ib).or_default().push((
                                        pair[0],
                                        pair[1],
                                        edge_curve.clone(),
                                    ));
                                }
                            }
                            CurveClassification::FullyContained => {
                                // Curve lies entirely inside the planar face.
                                // Record it for holed-face / disc / band fragment creation.
                                if let Some(ref ec) = edge_curve {
                                    if face_intersections_a.contains_key(&ia) {
                                        has_analytic_analytic = true;
                                    } else {
                                        contained_curves.push(ContainedCurve {
                                            plane_face_idx: ia,
                                            plane_source: Source::A,
                                            analytic_face_idx: ib,
                                            edge_curve: ec.clone(),
                                        });
                                    }
                                }
                            }
                            CurveClassification::FullyOutside => {}
                        }
                    }
                }
            } else if !is_plane_a && is_plane_b {
                // Analytic surface cut by plane (symmetric case).
                let Some(analytic_surf) = snap_a.surface.as_analytic() else {
                    has_analytic_analytic = true;
                    continue;
                };

                if let Ok(curves) = exact_plane_analytic(analytic_surf, snap_b.normal, snap_b.d) {
                    for curve in curves {
                        let edge_curve = match &curve {
                            ExactIntersectionCurve::Circle(c) => Some(EdgeCurve::Circle(c.clone())),
                            ExactIntersectionCurve::Ellipse(e) => {
                                Some(EdgeCurve::Ellipse(e.clone()))
                            }
                            ExactIntersectionCurve::Points(_) => None,
                        };

                        let classification =
                            curve_boundary_crossings(&curve, &snap_b.vertices, snap_b.normal, tol);
                        match classification {
                            CurveClassification::Crossings(samples) => {
                                for pair in samples.windows(2) {
                                    face_intersections_a.entry(ia).or_default().push((
                                        pair[0],
                                        pair[1],
                                        edge_curve.clone(),
                                    ));
                                    face_intersections_b.entry(ib).or_default().push((
                                        pair[0],
                                        pair[1],
                                        edge_curve.clone(),
                                    ));
                                }
                            }
                            CurveClassification::FullyContained => {
                                // Symmetric: curve inside plane-B face.
                                if let Some(ref ec) = edge_curve {
                                    if face_intersections_b.contains_key(&ib) {
                                        has_analytic_analytic = true;
                                    } else {
                                        contained_curves.push(ContainedCurve {
                                            plane_face_idx: ib,
                                            plane_source: Source::B,
                                            analytic_face_idx: ia,
                                            edge_curve: ec.clone(),
                                        });
                                    }
                                }
                            }
                            CurveClassification::FullyOutside => {}
                        }
                    }
                }
            } else {
                // Analytic-analytic: compute intersection via marching.
                let surf_a_opt = snap_a.surface.as_analytic();
                let surf_b_opt = snap_b.surface.as_analytic();

                if let (Some(surf_a_an), Some(surf_b_an)) = (surf_a_opt, surf_b_opt) {
                    let v_hint_a = compute_v_range_hint(&snap_a.surface, &snap_a.vertices);
                    let v_hint_b = compute_v_range_hint(&snap_b.surface, &snap_b.vertices);
                    if let Ok(curves) = intersect_analytic_analytic_bounded(
                        surf_a_an, surf_b_an, 32, v_hint_a, v_hint_b,
                    ) {
                        for ic in &curves {
                            let pts: Vec<Point3> = ic.points.iter().map(|ip| ip.point).collect();

                            analytic_analytic_faces_a.insert(ia);
                            analytic_analytic_faces_b.insert(ib);

                            for pair in pts.windows(2) {
                                face_intersections_a
                                    .entry(ia)
                                    .or_default()
                                    .push((pair[0], pair[1], None));
                                face_intersections_b
                                    .entry(ib)
                                    .or_default()
                                    .push((pair[0], pair[1], None));
                            }
                        }
                    } else {
                        // Intersection computation failed — mark for full tessellation.
                        analytic_analytic_faces_a.insert(ia);
                        analytic_analytic_faces_b.insert(ib);
                    }
                } else {
                    has_analytic_analytic = true;
                }
            }
        }
    }

    // If we encountered analytic-analytic pairs, fall back.
    if has_analytic_analytic {
        return Err(crate::OperationsError::InvalidInput {
            reason: "analytic-analytic intersection not supported, falling back".into(),
        });
    }

    // ── Compute analytic v-ranges from chord intersection points ────────
    // For cylinder and sphere faces that participate in intersections,
    // compute the v-extent of all chord endpoints on the surface. This
    // enables band/cap-splitting: only the intersection zone gets tessellated,
    // uninvolved regions keep their analytic surface type, preventing face-count
    // explosion in sequential boolean operations.
    collect_analytic_vranges(
        &snaps_a,
        &face_intersections_a,
        &analytic_analytic_faces_a,
        &mut analytic_intersection_vranges_a,
    );
    collect_analytic_vranges(
        &snaps_b,
        &face_intersections_b,
        &analytic_analytic_faces_b,
        &mut analytic_intersection_vranges_b,
    );
    log::debug!(
        "[boolean] intersections: {:.3}ms (isect_a={}, isect_b={})",
        timer_elapsed_ms(_t_isect),
        face_intersections_a.len(),
        face_intersections_b.len()
    );

    // Apply deferred coplanar edge injections, but ONLY for faces that
    // already have chord entries from non-coplanar face pairs (e.g., angled
    // side faces). This ensures that caps with diagonal chords get proper
    // inner-profile chords, while caps without chords (like box-cylinder
    // top caps) remain unaffected.
    for (ia, edges) in deferred_coplanar_edges_a {
        if face_intersections_a.contains_key(&ia) {
            for (p0, p1) in edges {
                face_intersections_a
                    .entry(ia)
                    .or_default()
                    .push((p0, p1, None));
            }
        }
    }
    for (ib, edges) in deferred_coplanar_edges_b {
        if face_intersections_b.contains_key(&ib) {
            for (p0, p1) in edges {
                face_intersections_b
                    .entry(ib)
                    .or_default()
                    .push((p0, p1, None));
            }
        }
    }

    let _t_frag = timer_now();
    // ── Build contained-curve lookup sets ──────────────────────────────

    // Map: plane_face_idx -> list of contained edge curves (keyed by plane source).
    let mut contained_a: HashMap<usize, Vec<EdgeCurve>> = HashMap::new();
    let mut contained_b: HashMap<usize, Vec<EdgeCurve>> = HashMap::new();
    // Map: analytic_face_idx -> list of contained edge curves (keyed by analytic source).
    // Analytic source is the OPPOSITE of plane_source.
    let mut analytic_contained_a: HashMap<usize, Vec<EdgeCurve>> = HashMap::new();
    let mut analytic_contained_b: HashMap<usize, Vec<EdgeCurve>> = HashMap::new();
    for cc in &contained_curves {
        match cc.plane_source {
            Source::A => {
                contained_a
                    .entry(cc.plane_face_idx)
                    .or_default()
                    .push(cc.edge_curve.clone());
                // Analytic face is from B.
                analytic_contained_b
                    .entry(cc.analytic_face_idx)
                    .or_default()
                    .push(cc.edge_curve.clone());
            }
            Source::B => {
                contained_b
                    .entry(cc.plane_face_idx)
                    .or_default()
                    .push(cc.edge_curve.clone());
                // Analytic face is from A.
                analytic_contained_a
                    .entry(cc.analytic_face_idx)
                    .or_default()
                    .push(cc.edge_curve.clone());
            }
        }
    }

    // Pre-classifications: fragment index -> class (bypasses centroid classifier).
    let mut pre_classifications: HashMap<usize, FaceClass> = HashMap::new();
    // Holed-face inner curves: fragment index -> edge curves for inner wire construction.
    let mut holed_face_inner_curves: HashMap<usize, Vec<EdgeCurve>> = HashMap::new();
    // Existing inner wire IDs from source faces (to preserve holes from prior booleans).
    let mut existing_inner_wires: HashMap<usize, Vec<WireId>> = HashMap::new();

    // ── Split faces into fragments ───────────────────────────────────────

    let mut fragments: Vec<AnalyticFragment> = Vec::with_capacity(snaps_a.len() + snaps_b.len());

    // Process solid A faces.
    for (ia, snap) in snaps_a.iter().enumerate() {
        // Analytic faces with intersection v-ranges: band/cap-split instead of
        // full tessellation. Only the intersection region gets tessellated;
        // uninvolved regions keep FaceSurface::Cylinder or FaceSurface::Sphere.
        if let Some(vranges) = analytic_intersection_vranges_a.get(&ia) {
            if matches!(snap.surface, FaceSurface::Sphere(_)) {
                split_sphere_at_intersection(
                    &snap.surface,
                    &snap.vertices,
                    snap.normal,
                    snap.d,
                    Source::A,
                    snap.reversed,
                    vranges,
                    topo,
                    snap.id,
                    deflection,
                    &mut fragments,
                )?;
                continue;
            }
            if matches!(snap.surface, FaceSurface::Cone(_)) {
                split_cone_at_intersection(
                    &snap.surface,
                    &snap.vertices,
                    snap.normal,
                    snap.d,
                    Source::A,
                    snap.reversed,
                    vranges,
                    topo,
                    snap.id,
                    deflection,
                    tol,
                    &mut fragments,
                )?;
                continue;
            }
            split_cylinder_at_intersection(
                &snap.surface,
                &snap.vertices,
                snap.normal,
                snap.d,
                Source::A,
                snap.reversed,
                vranges,
                topo,
                snap.id,
                deflection,
                tol,
                &mut fragments,
            )?;
            continue;
        }
        // Non-cylinder analytic-analytic: full tessellation fallback.
        if analytic_analytic_faces_a.contains(&ia) {
            tessellate_face_into_fragments(topo, snap.id, Source::A, deflection, &mut fragments)?;
            continue;
        }
        if let Some(chords) = face_intersections_a.get(&ia) {
            let chord_pairs: Vec<(Point3, Point3)> =
                chords.iter().map(|&(p0, p1, _)| (p0, p1)).collect();
            let edge_curve_for_face = chords.first().and_then(|c| c.2.clone());

            // Use existing polygon splitting for the face boundary.
            let mut chord_map_local: HashMap<usize, Vec<(Point3, Point3)>> = HashMap::new();
            chord_map_local.insert(snap.id.index(), chord_pairs);

            let planar_frags = split_face(
                snap.id,
                &snap.vertices,
                snap.normal,
                snap.d,
                Source::A,
                &chord_map_local,
                tol,
            );

            for frag in planar_frags {
                let edge_curves = vec![None; frag.vertices.len()];
                fragments.push(AnalyticFragment {
                    vertices: frag.vertices,
                    surface: snap.surface.clone(),
                    normal: frag.normal,
                    d: frag.d,
                    source: Source::A,
                    edge_curves,
                    source_reversed: snap.reversed,
                });
            }

            // For non-planar faces (cylinder, etc.), also create an analytic fragment
            // representing the portion of the analytic surface inside/outside.
            if !matches!(snap.surface, FaceSurface::Plane { .. }) {
                // The analytic face itself becomes a fragment.
                // Its boundary is the intersection curve (circle/ellipse).
                if let Some(ref ec) = edge_curve_for_face {
                    let curve_verts = sample_edge_curve(ec, CLOSED_CURVE_SAMPLES);
                    if curve_verts.len() >= 3 {
                        let avg_normal = snap.normal;
                        let d_val = snap.d;
                        fragments.push(AnalyticFragment {
                            vertices: curve_verts,
                            surface: snap.surface.clone(),
                            normal: avg_normal,
                            d: d_val,
                            source: Source::A,
                            edge_curves: vec![Some(ec.clone())],
                            source_reversed: snap.reversed,
                        });
                    }
                }
            }
        } else if let Some(inner_curves) = contained_a.get(&ia) {
            // Face has no chord intersections but has contained curves.
            // Create holed-face fragment (outer boundary) pre-classified as Outside.
            let holed_idx = fragments.len();
            fragments.push(AnalyticFragment {
                vertices: snap.vertices.clone(),
                surface: snap.surface.clone(),
                normal: snap.normal,
                d: snap.d,
                source: Source::A,
                edge_curves: edge_curves_from_face(topo, snap.id, snap.vertices.len()),
                source_reversed: snap.reversed,
            });
            pre_classifications.insert(holed_idx, FaceClass::Outside);
            holed_face_inner_curves.insert(holed_idx, inner_curves.clone());

            // Preserve existing inner wires (holes from prior booleans).
            let source_face = topo.face(snap.id)?;
            if !source_face.inner_wires().is_empty() {
                existing_inner_wires.insert(holed_idx, source_face.inner_wires().to_vec());
            }

            // Create disc fragment for each contained curve, pre-classified as Inside.
            for ec in inner_curves {
                let curve_verts = sample_edge_curve(ec, CLOSED_CURVE_SAMPLES);
                if curve_verts.len() >= 3 {
                    let disc_idx = fragments.len();
                    fragments.push(AnalyticFragment {
                        vertices: curve_verts,
                        surface: snap.surface.clone(),
                        normal: snap.normal,
                        d: snap.d,
                        source: Source::A,
                        edge_curves: vec![Some(ec.clone())],
                        source_reversed: false, // new disc geometry
                    });
                    pre_classifications.insert(disc_idx, FaceClass::Inside);
                }
            }
        } else if let Some(band_curves) = analytic_contained_a.get(&ia) {
            // Non-planar face with contained curves.
            if matches!(snap.surface, FaceSurface::Sphere(_)) {
                tessellate_face_into_fragments(
                    topo,
                    snap.id,
                    Source::A,
                    deflection,
                    &mut fragments,
                )?;
            } else {
                // Cylinder/cone: create band fragments between contained curves.
                create_band_fragments(
                    &snap.surface,
                    &snap.vertices,
                    snap.normal,
                    snap.d,
                    Source::A,
                    snap.reversed,
                    band_curves,
                    topo,
                    tol,
                    &mut fragments,
                );
            }
        } else {
            // Unsplit face — keep as-is, preserving any existing inner wires.
            let unsplit_idx = fragments.len();
            fragments.push(AnalyticFragment {
                vertices: snap.vertices.clone(),
                surface: snap.surface.clone(),
                normal: snap.normal,
                d: snap.d,
                source: Source::A,
                edge_curves: edge_curves_from_face(topo, snap.id, snap.vertices.len()),
                source_reversed: snap.reversed,
            });
            let source_face = topo.face(snap.id)?;
            if !source_face.inner_wires().is_empty() {
                existing_inner_wires.insert(unsplit_idx, source_face.inner_wires().to_vec());
            }
        }
    }

    // Process solid B faces (same logic).
    for (ib, snap) in snaps_b.iter().enumerate() {
        // Analytic faces with intersection v-ranges: band/cap-split.
        if let Some(vranges) = analytic_intersection_vranges_b.get(&ib) {
            if matches!(snap.surface, FaceSurface::Sphere(_)) {
                split_sphere_at_intersection(
                    &snap.surface,
                    &snap.vertices,
                    snap.normal,
                    snap.d,
                    Source::B,
                    snap.reversed,
                    vranges,
                    topo,
                    snap.id,
                    deflection,
                    &mut fragments,
                )?;
                continue;
            }
            if matches!(snap.surface, FaceSurface::Cone(_)) {
                split_cone_at_intersection(
                    &snap.surface,
                    &snap.vertices,
                    snap.normal,
                    snap.d,
                    Source::B,
                    snap.reversed,
                    vranges,
                    topo,
                    snap.id,
                    deflection,
                    tol,
                    &mut fragments,
                )?;
                continue;
            }
            split_cylinder_at_intersection(
                &snap.surface,
                &snap.vertices,
                snap.normal,
                snap.d,
                Source::B,
                snap.reversed,
                vranges,
                topo,
                snap.id,
                deflection,
                tol,
                &mut fragments,
            )?;
            continue;
        }
        // Non-cylinder analytic-analytic: full tessellation fallback.
        if analytic_analytic_faces_b.contains(&ib) {
            tessellate_face_into_fragments(topo, snap.id, Source::B, deflection, &mut fragments)?;
            continue;
        }
        if let Some(chords) = face_intersections_b.get(&ib) {
            let chord_pairs: Vec<(Point3, Point3)> =
                chords.iter().map(|&(p0, p1, _)| (p0, p1)).collect();
            let edge_curve_for_face = chords.first().and_then(|c| c.2.clone());

            let mut chord_map_local: HashMap<usize, Vec<(Point3, Point3)>> = HashMap::new();
            chord_map_local.insert(snap.id.index(), chord_pairs);

            let planar_frags = split_face(
                snap.id,
                &snap.vertices,
                snap.normal,
                snap.d,
                Source::B,
                &chord_map_local,
                tol,
            );

            for frag in planar_frags {
                let edge_curves = vec![None; frag.vertices.len()];
                fragments.push(AnalyticFragment {
                    vertices: frag.vertices,
                    surface: snap.surface.clone(),
                    normal: frag.normal,
                    d: frag.d,
                    source: Source::B,
                    edge_curves,
                    source_reversed: snap.reversed,
                });
            }

            if !matches!(snap.surface, FaceSurface::Plane { .. }) {
                if let Some(ref ec) = edge_curve_for_face {
                    let curve_verts = sample_edge_curve(ec, CLOSED_CURVE_SAMPLES);
                    if curve_verts.len() >= 3 {
                        fragments.push(AnalyticFragment {
                            vertices: curve_verts,
                            surface: snap.surface.clone(),
                            normal: snap.normal,
                            d: snap.d,
                            source: Source::B,
                            edge_curves: vec![Some(ec.clone())],
                            source_reversed: snap.reversed,
                        });
                    }
                }
            }
        } else if let Some(inner_curves) = contained_b.get(&ib) {
            // Face has contained curves but no chord intersections.
            let holed_idx = fragments.len();
            fragments.push(AnalyticFragment {
                vertices: snap.vertices.clone(),
                surface: snap.surface.clone(),
                normal: snap.normal,
                d: snap.d,
                source: Source::B,
                edge_curves: edge_curves_from_face(topo, snap.id, snap.vertices.len()),
                source_reversed: snap.reversed,
            });
            pre_classifications.insert(holed_idx, FaceClass::Outside);
            holed_face_inner_curves.insert(holed_idx, inner_curves.clone());

            // Preserve existing inner wires (holes from prior booleans).
            let source_face = topo.face(snap.id)?;
            if !source_face.inner_wires().is_empty() {
                existing_inner_wires.insert(holed_idx, source_face.inner_wires().to_vec());
            }

            for ec in inner_curves {
                let curve_verts = sample_edge_curve(ec, CLOSED_CURVE_SAMPLES);
                if curve_verts.len() >= 3 {
                    let disc_idx = fragments.len();
                    fragments.push(AnalyticFragment {
                        vertices: curve_verts,
                        surface: snap.surface.clone(),
                        normal: snap.normal,
                        d: snap.d,
                        source: Source::B,
                        edge_curves: vec![Some(ec.clone())],
                        source_reversed: false, // new disc geometry
                    });
                    pre_classifications.insert(disc_idx, FaceClass::Inside);
                }
            }
        } else if let Some(band_curves) = analytic_contained_b.get(&ib) {
            if matches!(snap.surface, FaceSurface::Sphere(_)) {
                tessellate_face_into_fragments(
                    topo,
                    snap.id,
                    Source::B,
                    deflection,
                    &mut fragments,
                )?;
            } else {
                create_band_fragments(
                    &snap.surface,
                    &snap.vertices,
                    snap.normal,
                    snap.d,
                    Source::B,
                    snap.reversed,
                    band_curves,
                    topo,
                    tol,
                    &mut fragments,
                );
            }
        } else {
            // Unsplit face — keep as-is, preserving any existing inner wires.
            let unsplit_idx = fragments.len();
            fragments.push(AnalyticFragment {
                vertices: snap.vertices.clone(),
                surface: snap.surface.clone(),
                normal: snap.normal,
                d: snap.d,
                source: Source::B,
                edge_curves: edge_curves_from_face(topo, snap.id, snap.vertices.len()),
                source_reversed: snap.reversed,
            });
            let source_face = topo.face(snap.id)?;
            if !source_face.inner_wires().is_empty() {
                existing_inner_wires.insert(unsplit_idx, source_face.inner_wires().to_vec());
            }
        }
    }

    // ── Passthrough face fragments ──────────────────────────────────────
    // Faces whose wire AABB didn't overlap the opposing solid's overall AABB
    // are guaranteed non-overlapping: they can't intersect any face of the
    // opposing solid. Snapshot them now (skipped during intersection phase)
    // and add as pre-classified fragments so they go through assembly for
    // proper edge dedup with adjacent faces.
    let passthrough_keep_a = matches!(op, BooleanOp::Cut | BooleanOp::Fuse);
    let passthrough_keep_b = matches!(op, BooleanOp::Fuse);
    if passthrough_keep_a {
        for &fid in &passthrough_a {
            let face = topo.face(fid)?;
            let surface = face.surface().clone();
            let reversed = face.is_reversed();
            let verts = face_polygon(topo, fid)?;
            let (normal, d) = analytic_face_normal_d(&surface, &verts);
            let pass_idx = fragments.len();
            fragments.push(AnalyticFragment {
                vertices: verts.clone(),
                surface,
                normal,
                d,
                source: Source::A,
                edge_curves: edge_curves_from_face(topo, fid, verts.len()),
                source_reversed: reversed,
            });
            pre_classifications.insert(pass_idx, FaceClass::Outside);
            let source_face = topo.face(fid)?;
            if !source_face.inner_wires().is_empty() {
                existing_inner_wires.insert(pass_idx, source_face.inner_wires().to_vec());
            }
        }
    }
    if passthrough_keep_b {
        for &fid in &passthrough_b {
            let face = topo.face(fid)?;
            let surface = face.surface().clone();
            let reversed = face.is_reversed();
            let verts = face_polygon(topo, fid)?;
            let (normal, d) = analytic_face_normal_d(&surface, &verts);
            let pass_idx = fragments.len();
            fragments.push(AnalyticFragment {
                vertices: verts.clone(),
                surface,
                normal,
                d,
                source: Source::B,
                edge_curves: edge_curves_from_face(topo, fid, verts.len()),
                source_reversed: reversed,
            });
            pre_classifications.insert(pass_idx, FaceClass::Outside);
            let source_face = topo.face(fid)?;
            if !source_face.inner_wires().is_empty() {
                existing_inner_wires.insert(pass_idx, source_face.inner_wires().to_vec());
            }
        }
    }

    log::debug!(
        "[boolean] fragments: {:.3}ms (count={}, passthrough={})",
        timer_elapsed_ms(_t_frag),
        fragments.len(),
        passthrough_a.len() + passthrough_b.len()
    );

    // ── Classification ───────────────────────────────────────────────────
    let _t_class = timer_now();

    // Try analytic classifiers first (O(1) point-in-solid tests).
    // Only build expensive tessellated face data if needed.
    let analytic_cls_a = try_build_analytic_classifier(topo, a);
    let analytic_cls_b = try_build_analytic_classifier(topo, b);

    // Phase 1: classify everything we can with analytic classifiers.
    let mut classes: Vec<Option<FaceClass>> = fragments
        .iter()
        .enumerate()
        .map(|(idx, frag)| {
            if let Some(&class) = pre_classifications.get(&idx) {
                return Some(class);
            }
            let classifier = match frag.source {
                Source::A => analytic_cls_b.as_ref(),
                Source::B => analytic_cls_a.as_ref(),
            };
            let classifier = match classifier {
                Some(c) => c,
                None => return None,
            };
            let centroid = polygon_centroid(&frag.vertices);
            if let Some(class) = classifier.classify(centroid, tol) {
                return Some(class);
            }
            // Centroid is on the boundary of a curved solid (sphere/cylinder/
            // cone). Phase 2 ray-casting from a point ON a curved surface is
            // unreliable (ray may graze the surface), so resolve using vertex
            // majority voting. For box classifiers, defer to Phase 2 — ray-
            // casting from a point on a flat face is reliable.
            if matches!(classifier, AnalyticClassifier::Box { .. }) {
                return None;
            }
            let mut inside = 0u32;
            let mut outside = 0u32;
            for v in &frag.vertices {
                match classifier.classify(*v, tol) {
                    Some(FaceClass::Inside) => inside += 1,
                    Some(FaceClass::Outside) => outside += 1,
                    _ => {}
                }
            }
            if outside > inside && outside > 0 {
                Some(FaceClass::Outside)
            } else if inside > outside && inside > 0 {
                Some(FaceClass::Inside)
            } else {
                None // Truly ambiguous — defer to Phase 2
            }
        })
        .collect();

    // Phase 2a: AABB pre-filter — classify fragments whose centroids fall
    // outside the opposing solid's bounding box as Outside. This avoids
    // building expensive face data + BVH for the majority of fragments
    // in multi-body fuse operations where solids overlap minimally.
    let padded_aabb_a = a_overall_aabb.expanded(tol.linear);
    let padded_aabb_b = b_overall_aabb.expanded(tol.linear);
    for (class, frag) in classes.iter_mut().zip(&fragments) {
        if class.is_some() {
            continue;
        }
        let opposing_aabb = match frag.source {
            Source::A => padded_aabb_b,
            Source::B => padded_aabb_a,
        };
        let centroid = polygon_centroid(&frag.vertices);
        if !opposing_aabb.contains_point(centroid) {
            *class = Some(FaceClass::Outside);
        }
    }

    // Phase 2b: if any fragments are still unclassified, build face data and ray-cast.
    let needs_raycast = classes.iter().any(Option::is_none);
    if needs_raycast {
        let face_data_a = collect_face_data(topo, a, deflection)?;
        let face_data_b = collect_face_data(topo, b, deflection)?;
        let bvh_a = build_face_bvh(&face_data_a);
        let bvh_b = build_face_bvh(&face_data_b);

        for (idx, class) in classes.iter_mut().enumerate() {
            if class.is_some() {
                continue;
            }
            let frag = &fragments[idx];
            let (opposite, bvh) = match frag.source {
                Source::A => (&face_data_b, bvh_b.as_ref()),
                Source::B => (&face_data_a, bvh_a.as_ref()),
            };
            let centroid = polygon_centroid(&frag.vertices);
            let raw = classify_point(centroid, frag.normal, opposite, bvh, tol);
            // Apply tangent guard: verify coplanar classifications
            // using fragment vertices to catch tangent-touch false positives.
            *class = Some(guard_tangent_coplanar(
                raw,
                &frag.vertices,
                frag.normal,
                opposite,
                bvh,
                tol,
            ));
        }
    }

    // All fragments should now be classified.
    let classes: Vec<FaceClass> = classes
        .into_iter()
        .enumerate()
        .map(|(_i, c)| -> Result<FaceClass, crate::OperationsError> {
            c.ok_or_else(|| crate::OperationsError::InvalidInput {
                reason: format!("boolean: fragment {_i} was not classified"),
            })
        })
        .collect::<Result<Vec<_>, _>>()?;

    log::debug!(
        "[boolean] classification: {:.3}ms ({} fragments)",
        timer_elapsed_ms(_t_class),
        classes.len()
    );

    // ── Selection + Assembly ─────────────────────────────────────────────
    let _t_asm = timer_now();

    // Pre-allocate topology arenas based on expected output size.
    let nf = fragments.len();
    topo.reserve(nf.saturating_mul(2), nf.saturating_mul(3), nf, nf, 1, 1);

    let resolution = vertex_merge_resolution(
        fragments.iter().flat_map(|f| f.vertices.iter().copied()),
        tol,
    );
    let mut vertex_map: HashMap<(i64, i64, i64), VertexId> =
        HashMap::with_capacity(fragments.len() * 4);
    let mut edge_map: HashMap<(usize, usize), brepkit_topology::edge::EdgeId> =
        HashMap::with_capacity(fragments.len() * 4);
    let mut face_ids_out = Vec::with_capacity(fragments.len());

    for (idx, (frag, &class)) in fragments.iter().zip(classes.iter()).enumerate() {
        let Some(flip) = select_fragment(frag.source, class, op) else {
            continue;
        };

        let is_nonplanar = !matches!(frag.surface, FaceSurface::Plane { .. });
        let (verts, normal, d_val) = if flip && !is_nonplanar {
            // Planar faces: reverse vertex winding and negate normal/d.
            let rev: Vec<_> = frag.vertices.iter().copied().rev().collect();
            (rev, -frag.normal, -frag.d)
        } else {
            // Non-planar faces: keep original winding; Face::new_reversed
            // handles the flip via the `reversed` flag so tessellation
            // flips triangle winding and normals correctly.
            (frag.vertices.clone(), frag.normal, frag.d)
        };

        let n = verts.len();
        if n < 3 {
            continue;
        }

        // Allocate vertices through shared dedup map.
        let vert_ids: Vec<VertexId> = verts
            .iter()
            .map(|p| {
                let key = quantize_point(*p, resolution);
                *vertex_map
                    .entry(key)
                    .or_insert_with(|| topo.add_vertex(Vertex::new(*p, tol.linear)))
            })
            .collect();

        // Build edges — deduplicate by ordered vertex-index pair so adjacent
        // faces share edge IDs (required for fillet/chamfer adjacency queries).
        //
        // Special handling for closed-curve boundaries and cylinder barrels:
        // instead of creating N line edges per circle/ellipse boundary, create
        // proper closed Circle/Ellipse edges to match canonical B-Rep topology.
        let is_single_closed_curve = frag.edge_curves.len() == 1
            && matches!(
                frag.edge_curves[0],
                Some(EdgeCurve::Circle(_) | EdgeCurve::Ellipse(_))
            );

        // Check single-closed-curve FIRST — disc fragments (cylinder caps) carry
        // FaceSurface::Cylinder but have a single-circle polygon, not the
        // bot[0..n/2]+top_reversed[0..n/2] layout that build_cylinder_barrel_wire expects.
        let wire_id = if is_single_closed_curve {
            // Disc fragment: boundary is a single closed curve (circle/ellipse).
            // Create one closed edge instead of N line edges.
            // Safety: is_single_closed_curve guarantees frag.edge_curves[0] is Some.
            let Some(ec) = frag.edge_curves[0].clone() else {
                unreachable!("is_single_closed_curve guarantees Some")
            };
            let seam_pt = verts[0];
            let vid = *vertex_map
                .entry(quantize_point(seam_pt, resolution))
                .or_insert_with(|| topo.add_vertex(Vertex::new(seam_pt, tol.linear)));
            let eid = *edge_map
                .entry((vid.index(), vid.index()))
                .or_insert_with(|| topo.add_edge(Edge::new(vid, vid, ec)));
            let wire = Wire::new(vec![OrientedEdge::new(eid, true)], true)
                .map_err(crate::OperationsError::Topology)?;
            topo.add_wire(wire)
        } else if let FaceSurface::Cylinder(cyl) = &frag.surface {
            // Cylinder barrel: polygon must have even vertex count and distinct
            // v-levels at verts[0] (bot seam) and verts[last] (top seam).
            // Chord-split fragments don't satisfy this — fall through to generic path.
            let has_band_layout = verts.len() >= 4 && verts.len() % 2 == 0 && {
                let v0 = cyl.axis().dot(verts[0] - cyl.origin());
                let v1 = cyl.axis().dot(verts[verts.len() - 1] - cyl.origin());
                (v0 - v1).abs() > tol.linear
            };
            if has_band_layout {
                build_cylinder_barrel_wire(
                    topo,
                    cyl,
                    &verts,
                    &mut vertex_map,
                    &mut edge_map,
                    resolution,
                    tol,
                )?
            } else {
                // Chord-split or degenerate cylinder fragment — use generic polygon edges.
                let mut oriented_edges = Vec::with_capacity(n);
                for i in 0..n {
                    let j = (i + 1) % n;
                    let vi_idx = vert_ids[i].index();
                    let vj_idx = vert_ids[j].index();
                    let (key_min, key_max) = if vi_idx <= vj_idx {
                        (vi_idx, vj_idx)
                    } else {
                        (vj_idx, vi_idx)
                    };
                    let fwd = vi_idx <= vj_idx;
                    let eid = *edge_map.entry((key_min, key_max)).or_insert_with(|| {
                        let (start, end) = if fwd {
                            (vert_ids[i], vert_ids[j])
                        } else {
                            (vert_ids[j], vert_ids[i])
                        };
                        topo.add_edge(Edge::new(start, end, EdgeCurve::Line))
                    });
                    oriented_edges.push(OrientedEdge::new(eid, fwd));
                }
                let wire =
                    Wire::new(oriented_edges, true).map_err(crate::OperationsError::Topology)?;
                topo.add_wire(wire)
            }
        } else if let FaceSurface::Cone(cone) = &frag.surface {
            // Cone barrel: same band layout as cylinder but with varying radii.
            let has_band_layout = verts.len() >= 4 && verts.len() % 2 == 0 && {
                let (_, v0) = cone.project_point(verts[0]);
                let (_, v1) = cone.project_point(verts[verts.len() - 1]);
                (v0 - v1).abs() > tol.linear
            };
            if has_band_layout {
                build_cone_barrel_wire(
                    topo,
                    cone,
                    &verts,
                    &mut vertex_map,
                    &mut edge_map,
                    resolution,
                    tol,
                )?
            } else {
                // Chord-split or degenerate cone fragment — use generic polygon edges.
                let mut oriented_edges = Vec::with_capacity(n);
                for i in 0..n {
                    let j = (i + 1) % n;
                    let vi_idx = vert_ids[i].index();
                    let vj_idx = vert_ids[j].index();
                    let (key_min, key_max) = if vi_idx <= vj_idx {
                        (vi_idx, vj_idx)
                    } else {
                        (vj_idx, vi_idx)
                    };
                    let fwd = vi_idx <= vj_idx;
                    let eid = *edge_map.entry((key_min, key_max)).or_insert_with(|| {
                        let (start, end) = if fwd {
                            (vert_ids[i], vert_ids[j])
                        } else {
                            (vert_ids[j], vert_ids[i])
                        };
                        topo.add_edge(Edge::new(start, end, EdgeCurve::Line))
                    });
                    oriented_edges.push(OrientedEdge::new(eid, fwd));
                }
                let wire =
                    Wire::new(oriented_edges, true).map_err(crate::OperationsError::Topology)?;
                topo.add_wire(wire)
            }
        } else {
            let mut oriented_edges = Vec::with_capacity(n);
            for i in 0..n {
                let j = (i + 1) % n;
                let vi_idx = vert_ids[i].index();
                let vj_idx = vert_ids[j].index();
                let (key_min, key_max) = if vi_idx <= vj_idx {
                    (vi_idx, vj_idx)
                } else {
                    (vj_idx, vi_idx)
                };
                let is_forward = vi_idx <= vj_idx;

                let edge_id = *edge_map.entry((key_min, key_max)).or_insert_with(|| {
                    let edge_curve = if frag.edge_curves.len() == 1 {
                        frag.edge_curves[0].clone().unwrap_or(EdgeCurve::Line)
                    } else if i < frag.edge_curves.len() {
                        frag.edge_curves[i].clone().unwrap_or(EdgeCurve::Line)
                    } else {
                        EdgeCurve::Line
                    };
                    let (start, end) = if vi_idx <= vj_idx {
                        (vert_ids[i], vert_ids[j])
                    } else {
                        (vert_ids[j], vert_ids[i])
                    };
                    topo.add_edge(Edge::new(start, end, edge_curve))
                });

                oriented_edges.push(OrientedEdge::new(edge_id, is_forward));
            }
            let wire = Wire::new(oriented_edges, true).map_err(crate::OperationsError::Topology)?;
            topo.add_wire(wire)
        };

        // Build inner wires for holed faces (new contained curves + existing holes).
        let mut inner_wire_ids = Vec::new();

        // 1. Carry over existing inner wires from source faces (prior booleans).
        if let Some(existing_wires) = existing_inner_wires.get(&idx) {
            inner_wire_ids.extend_from_slice(existing_wires);
        }

        // 2. Create new inner wires from contained intersection curves.
        //    For Circle/Ellipse curves, create a single closed edge instead
        //    of N line segments — this produces proper B-Rep topology.
        if let Some(inner_curves) = holed_face_inner_curves.get(&idx) {
            for ec in inner_curves {
                let hw_id = if matches!(ec, EdgeCurve::Circle(_) | EdgeCurve::Ellipse(_)) {
                    // Use the seam point (t=0) as the single vertex for the
                    // closed edge. This matches the barrel boundary vertex
                    // so the edge is shared between the hole wire and the
                    // adjacent cylinder barrel face.
                    let seam_pt = sample_edge_curve(ec, CLOSED_CURVE_SAMPLES)[0];
                    let vid = *vertex_map
                        .entry(quantize_point(seam_pt, resolution))
                        .or_insert_with(|| topo.add_vertex(Vertex::new(seam_pt, tol.linear)));
                    let eid = *edge_map
                        .entry((vid.index(), vid.index()))
                        .or_insert_with(|| topo.add_edge(Edge::new(vid, vid, ec.clone())));
                    // Hole wires wind CW (reversed circle). When the face is
                    // flipped, the outer wire is already reversed so the hole
                    // keeps its natural (forward) direction.
                    let hw = Wire::new(vec![OrientedEdge::new(eid, flip)], true)
                        .map_err(crate::OperationsError::Topology)?;
                    topo.add_wire(hw)
                } else {
                    // Non-circle/ellipse: fall back to sampled polygon edges.
                    let mut hole_pts = sample_edge_curve(ec, CLOSED_CURVE_SAMPLES);
                    if !flip {
                        // Reverse for CW winding (hole convention). When the
                        // face is flipped, the outer wire is already reversed
                        // so the hole keeps its natural direction.
                        hole_pts.reverse();
                    }

                    let hole_vert_ids: Vec<VertexId> = hole_pts
                        .iter()
                        .map(|p| {
                            let key = (
                                quantize(p.x(), resolution),
                                quantize(p.y(), resolution),
                                quantize(p.z(), resolution),
                            );
                            *vertex_map
                                .entry(key)
                                .or_insert_with(|| topo.add_vertex(Vertex::new(*p, tol.linear)))
                        })
                        .collect();

                    let hm = hole_vert_ids.len();
                    let mut hole_edges = Vec::with_capacity(hm);
                    for i in 0..hm {
                        let j = (i + 1) % hm;
                        let vi_idx = hole_vert_ids[i].index();
                        let vj_idx = hole_vert_ids[j].index();
                        let is_forward = vi_idx <= vj_idx;
                        let key = if is_forward {
                            (vi_idx, vj_idx)
                        } else {
                            (vj_idx, vi_idx)
                        };

                        let eid = *edge_map.entry(key).or_insert_with(|| {
                            let (start, end) = if is_forward {
                                (hole_vert_ids[i], hole_vert_ids[j])
                            } else {
                                (hole_vert_ids[j], hole_vert_ids[i])
                            };
                            topo.add_edge(Edge::new(start, end, EdgeCurve::Line))
                        });
                        hole_edges.push(OrientedEdge::new(eid, is_forward));
                    }
                    let hw =
                        Wire::new(hole_edges, true).map_err(crate::OperationsError::Topology)?;
                    topo.add_wire(hw)
                };
                inner_wire_ids.push(hw_id);
            }
        }

        let surface = match &frag.surface {
            FaceSurface::Plane { .. } => FaceSurface::Plane { normal, d: d_val },
            other => other.clone(),
        };

        // For non-planar faces, the effective orientation is the XOR of
        // the boolean flip and the source face's original reversed status.
        // This preserves reversal through sequential boolean operations.
        let effective_reversed = if is_nonplanar {
            flip ^ frag.source_reversed
        } else {
            false
        };
        let new_face = if effective_reversed {
            Face::new_reversed(wire_id, inner_wire_ids, surface)
        } else {
            Face::new(wire_id, inner_wire_ids, surface)
        };
        let face = topo.add_face(new_face);
        face_ids_out.push(face);
    }

    log::debug!(
        "[boolean] assembly: {:.3}ms ({} faces)",
        timer_elapsed_ms(_t_asm),
        face_ids_out.len()
    );

    if face_ids_out.is_empty() {
        return Err(crate::OperationsError::InvalidInput {
            reason: "analytic boolean produced no faces".into(),
        });
    }

    // ── Post-assembly edge refinement ──────────────────────────────────
    // Unsplit faces may have long boundary edges that span the same line
    // as multiple shorter edges from adjacent split faces. Refine them
    // so edge sharing works correctly.
    let _t_refine = timer_now();
    let vertex_positions: HashMap<VertexId, Point3> = vertex_map
        .values()
        .filter_map(|&vid| topo.vertex(vid).ok().map(|v| (vid, v.point())))
        .collect();
    refine_boundary_edges(
        topo,
        &mut face_ids_out,
        &mut edge_map,
        tol,
        Some(&vertex_positions),
    )?;
    log::debug!(
        "[boolean] refine_boundary_edges: {:.3}ms ({} faces)",
        timer_elapsed_ms(_t_refine),
        face_ids_out.len()
    );

    // Split non-manifold edges (shared by > 2 faces) into separate copies.
    let _t_nm = timer_now();
    split_nonmanifold_edges(topo, &mut face_ids_out)?;
    log::debug!(
        "[boolean] split_nonmanifold_edges: {:.3}ms ({} faces)",
        timer_elapsed_ms(_t_nm),
        face_ids_out.len()
    );

    let shell = Shell::new(face_ids_out).map_err(crate::OperationsError::Topology)?;
    let shell_id = topo.add_shell(shell);
    log::debug!("[boolean] total: {:.3}ms", timer_elapsed_ms(_t_total));
    Ok(topo.add_solid(Solid::new(shell_id, vec![])))
}
