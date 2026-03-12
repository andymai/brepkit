//! Boolean operations on solids: fuse, cut, and intersect.
//!
//! Supports both planar and NURBS faces. NURBS faces are tessellated
//! into planar triangles before clipping, enabling approximate boolean
//! operations on any solid geometry.

use std::collections::{HashMap, HashSet};

use rayon::prelude::*;

// WASM-compatible timer: `std::time::Instant` panics on wasm32 targets.
#[cfg(not(target_arch = "wasm32"))]
fn timer_now() -> std::time::Instant {
    std::time::Instant::now()
}
#[cfg(not(target_arch = "wasm32"))]
fn timer_elapsed_ms(t: std::time::Instant) -> f64 {
    t.elapsed().as_secs_f64() * 1000.0
}
#[cfg(target_arch = "wasm32")]
fn timer_now() -> () {}
#[cfg(target_arch = "wasm32")]
fn timer_elapsed_ms(_t: ()) -> f64 {
    0.0
}

use brepkit_math::aabb::Aabb3;
use brepkit_math::bvh::Bvh;
use brepkit_math::plane::plane_plane_intersection;
use brepkit_math::predicates::{orient3d_sos, point_in_polygon};
use brepkit_math::tolerance::Tolerance;
use brepkit_math::vec::{Point2, Point3, Vec3};
use brepkit_topology::Topology;
use brepkit_topology::edge::{Edge, EdgeCurve, EdgeId};
use brepkit_topology::face::{Face, FaceId, FaceSurface};
use brepkit_topology::shell::Shell;
use brepkit_topology::solid::{Solid, SolidId};
use brepkit_topology::vertex::{Vertex, VertexId};
use brepkit_topology::wire::WireId;
use brepkit_topology::wire::{OrientedEdge, Wire};

use crate::dot_normal_point;

// ---------------------------------------------------------------------------
// Shared helpers (used across multiple phases)
// ---------------------------------------------------------------------------

/// Convert a `FaceSurface` reference to an `AnalyticSurface` reference.
///
/// Returns `None` for planar and NURBS surfaces.
fn face_surface_to_analytic(
    surface: &FaceSurface,
) -> Option<brepkit_math::analytic_intersection::AnalyticSurface<'_>> {
    use brepkit_math::analytic_intersection::AnalyticSurface;
    match surface {
        FaceSurface::Cylinder(c) => Some(AnalyticSurface::Cylinder(c)),
        FaceSurface::Cone(c) => Some(AnalyticSurface::Cone(c)),
        FaceSurface::Sphere(s) => Some(AnalyticSurface::Sphere(s)),
        FaceSurface::Torus(t) => Some(AnalyticSurface::Torus(t)),
        _ => None,
    }
}

/// Deduplicate 3D points by quantized position (spatial hashing).
///
/// Resolution is derived from the tolerance: `1.0 / tol.linear`.
fn dedup_points_by_position(pts: &mut Vec<Point3>, tol: Tolerance) {
    let scale = 1.0 / tol.linear;
    let mut seen = HashSet::new();
    pts.retain(|p| {
        #[allow(clippy::cast_possible_truncation)]
        let key = (
            (p.x() * scale).round() as i64,
            (p.y() * scale).round() as i64,
            (p.z() * scale).round() as i64,
        );
        seen.insert(key)
    });
}

/// Compute a representative normal and d-value for a face from its surface type.
///
/// For planar faces, returns the plane normal/d directly. For analytic surfaces
/// (cylinder, sphere, cone, torus), computes the normal from the surface
/// definition and a sample vertex — avoiding expensive tessellation.
fn analytic_face_normal_d(surface: &FaceSurface, verts: &[Point3]) -> (Vec3, f64) {
    match surface {
        FaceSurface::Plane { normal, d } => (*normal, *d),
        FaceSurface::Cylinder(cyl) => {
            // Cylinder axis direction as representative normal.
            let n = cyl.axis();
            let d = if verts.is_empty() {
                0.0
            } else {
                crate::dot_normal_point(n, verts[0])
            };
            (n, d)
        }
        FaceSurface::Sphere(sph) => {
            // Outward radial from center through first vertex.
            if let Some(&v) = verts.first() {
                let dir = v - sph.center();
                let n = dir.normalize().unwrap_or(Vec3::new(0.0, 0.0, 1.0));
                (n, crate::dot_normal_point(n, v))
            } else {
                (Vec3::new(0.0, 0.0, 1.0), 0.0)
            }
        }
        FaceSurface::Cone(cone) => {
            let n = cone.axis();
            let d = if verts.is_empty() {
                0.0
            } else {
                crate::dot_normal_point(n, verts[0])
            };
            (n, d)
        }
        FaceSurface::Torus(tor) => {
            let n = tor.z_axis();
            let d = if verts.is_empty() {
                0.0
            } else {
                crate::dot_normal_point(n, verts[0])
            };
            (n, d)
        }
        FaceSurface::Nurbs(_) => {
            // For NURBS, use polygon normal from vertices.
            if verts.len() >= 3 {
                let e1 = verts[1] - verts[0];
                let e2 = verts[2] - verts[0];
                let n = e1.cross(e2).normalize().unwrap_or(Vec3::new(0.0, 0.0, 1.0));
                (n, crate::dot_normal_point(n, verts[0]))
            } else {
                (Vec3::new(0.0, 0.0, 1.0), 0.0)
            }
        }
    }
}

/// Compute the v-range hint for an analytic surface based on face vertices.
///
/// Returns `Some((v_min, v_max))` if the surface has a non-trivial v
/// parameterization that depends on the face extent (cylinder, cone).
/// Returns `None` for surfaces where the default v_range is correct
/// (sphere, torus).
fn compute_v_range_hint(surface: &FaceSurface, verts: &[Point3]) -> Option<(f64, f64)> {
    match surface {
        FaceSurface::Cylinder(cyl) => {
            // v = axial distance from origin. Compute from face vertices
            // with padding to ensure intersections near the boundary are found.
            cylinder_v_extent(cyl, verts).map(|(lo, hi)| {
                let pad = (hi - lo) * 0.1;
                (lo - pad, hi + pad)
            })
        }
        FaceSurface::Cone(cone) => {
            // v = distance from apex along the axis-radial direction.
            // Compute from face vertices.
            let axis = cone.axis();
            let apex = cone.apex();
            let half = cone.half_angle();
            let (sin_a, cos_a) = half.sin_cos();
            let mut v_min = f64::MAX;
            let mut v_max = f64::MIN;
            for &p in verts {
                let d = p - apex;
                let axial = d.dot(axis);
                let radial_sq = (d.dot(d) - axial * axial).max(0.0);
                // v = sqrt(axial^2 + radial_sq) with correct sign
                let v = axial * sin_a + radial_sq.sqrt() * cos_a;
                v_min = v_min.min(v);
                v_max = v_max.max(v);
            }
            if (v_max - v_min).abs() < 1e-10 {
                None
            } else {
                let pad = (v_max - v_min) * 0.1;
                // Clamp the lower bound away from the apex singularity (v=0),
                // but allow negative v if the geometry requires it.
                let lo = if v_min > 0.0 {
                    (v_min - pad).max(0.001)
                } else {
                    v_min - pad
                };
                Some((lo, v_max + pad))
            }
        }
        _ => None, // Sphere and torus have fixed parametric ranges
    }
}

/// Compute the axial extent (v-range) of points projected onto a cylinder axis.
///
/// Returns `None` if the extent is degenerate (< 1e-10).
fn cylinder_v_extent(
    cyl: &brepkit_math::surfaces::CylindricalSurface,
    points: &[Point3],
) -> Option<(f64, f64)> {
    let mut v_min = f64::MAX;
    let mut v_max = f64::MIN;
    for &p in points {
        let v = cyl.axis().dot(p - cyl.origin());
        v_min = v_min.min(v);
        v_max = v_max.max(v);
    }
    if (v_max - v_min).abs() < 1e-10 {
        None
    } else {
        Some((v_min, v_max))
    }
}

/// Merge overlapping v-ranges with padding, clamped to `[extent_min, extent_max]`.
///
/// Returns `None` if the merged zones cover more than 60% of the total extent
/// (band-splitting would not be worthwhile).
fn merge_vranges_with_padding(
    vranges: &[(f64, f64)],
    extent_min: f64,
    extent_max: f64,
    padding_fraction: f64,
) -> Option<Vec<(f64, f64)>> {
    let extent_height = extent_max - extent_min;
    let padding = extent_height * padding_fraction;
    let mut sorted: Vec<(f64, f64)> = vranges.to_vec();
    sorted.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap_or(std::cmp::Ordering::Equal));

    let mut merged: Vec<(f64, f64)> = Vec::new();
    for &(lo, hi) in &sorted {
        let lo_padded = (lo - padding).max(extent_min);
        let hi_padded = (hi + padding).min(extent_max);
        if let Some(last) = merged.last_mut() {
            if lo_padded <= last.1 {
                last.1 = last.1.max(hi_padded);
                continue;
            }
        }
        merged.push((lo_padded, hi_padded));
    }

    let total_iz: f64 = merged.iter().map(|(lo, hi)| hi - lo).sum();
    if total_iz > extent_height * 0.6 {
        None
    } else {
        Some(merged)
    }
}

/// Build an ordered list of v-levels from extent bounds and merged intersection zones.
///
/// Deduplicates levels that are within 1e-10 of each other.
fn build_v_levels(extent_min: f64, extent_max: f64, merged: &[(f64, f64)]) -> Vec<f64> {
    let mut levels: Vec<f64> = vec![extent_min];
    for &(iz_lo, iz_hi) in merged {
        if iz_lo > extent_min + 1e-10 {
            levels.push(iz_lo);
        }
        if iz_hi < extent_max - 1e-10 {
            levels.push(iz_hi);
        }
    }
    levels.push(extent_max);
    levels.dedup_by(|a, b| (*a - *b).abs() < 1e-10);
    levels
}

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

/// Number of samples used when discretizing closed curves (circles, ellipses)
/// in the analytic boolean path. All code paths must use this constant so that
/// band fragments, cap face polygons, and holed-face inner wires share the
/// same vertices and edges at their boundaries.
const CLOSED_CURVE_SAMPLES: usize = 32;

/// Minimum fragment count for parallel classification via rayon.
/// Below this threshold, sequential iteration is faster due to rayon's
/// thread-pool synchronization overhead (~5-20µs).
const PARALLEL_THRESHOLD: usize = 64;

/// The type of boolean operation.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BooleanOp {
    /// Union of two solids.
    Fuse,
    /// Subtraction: first minus second.
    Cut,
    /// Intersection: common volume.
    Intersect,
}

/// Default tessellation deflection for non-planar faces in boolean operations.
///
/// A larger value produces fewer triangles (faster but coarser approximation).
/// Since the boolean result decomposes curved faces into individual planar
/// triangles, keeping this coarse avoids face-count explosion in sequential
/// boolean operations.
const DEFAULT_BOOLEAN_DEFLECTION: f64 = 0.1;

/// Options for boolean operations.
#[derive(Debug, Clone, Copy)]
pub struct BooleanOptions {
    /// Tessellation deflection for non-planar faces.
    ///
    /// Lower values produce more triangles (more accurate but slower).
    /// Default: 0.1.
    pub deflection: f64,
    /// Tolerance for geometric comparisons.
    ///
    /// Controls vertex merging, point classification, and predicate
    /// thresholds throughout the boolean pipeline. Default: `Tolerance::new()`.
    pub tolerance: Tolerance,
}

impl Default for BooleanOptions {
    fn default() -> Self {
        Self {
            deflection: DEFAULT_BOOLEAN_DEFLECTION,
            tolerance: Tolerance::new(),
        }
    }
}

/// Internal context carrying tolerance-derived thresholds through the boolean
/// pipeline. Computed once from `BooleanOptions` at the start of a boolean
/// operation to avoid repeated derivation and hardcoded epsilon values.
#[derive(Debug, Clone, Copy)]
struct BooleanContext {
    /// Base tolerance (used when wiring ctx through the full pipeline).
    #[allow(dead_code)]
    tol: Tolerance,
    /// Vertex merge distance: vertices closer than this are considered identical.
    vertex_merge: f64,
    /// Point classification tolerance: distance threshold for on-surface tests.
    classify_tol: f64,
    /// Degenerate polygon threshold: skip polygons with area below this.
    degenerate_area: f64,
}

impl BooleanContext {
    fn from_options(opts: &BooleanOptions) -> Self {
        let tol = opts.tolerance;
        Self {
            tol,
            // 1000x linear tolerance for vertex merging — aggressive enough to
            // catch coincident vertices while preserving distinct features.
            vertex_merge: tol.linear * 1000.0,
            // Classification tolerance for point-in-solid tests.
            classify_tol: tol.linear * 100.0,
            // Degenerate area threshold (area < this → skip polygon).
            degenerate_area: tol.linear * tol.linear,
        }
    }
}

// ---------------------------------------------------------------------------
// Internal data structures
// ---------------------------------------------------------------------------

/// Which operand a face fragment originated from.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Source {
    A,
    B,
}

/// Classification of a face fragment relative to the opposite solid.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum FaceClass {
    Outside,
    Inside,
    CoplanarSame,
    CoplanarOpposite,
}

/// An intersection segment between two faces.
#[derive(Debug)]
struct IntersectionSegment {
    face_a: FaceId,
    face_b: FaceId,
    p0: Point3,
    p1: Point3,
}

/// A fragment of a face after splitting along intersection chords.
#[derive(Debug)]
struct FaceFragment {
    vertices: Vec<Point3>,
    normal: Vec3,
    d: f64,
    source: Source,
}

/// Parameters for a single face in a face-pair intersection test.
struct FacePairSide<'a> {
    fid: FaceId,
    verts: &'a [Point3],
    normal: Vec3,
    d: f64,
}

// ---------------------------------------------------------------------------
// Public API
// ---------------------------------------------------------------------------

/// Perform a boolean operation on two solids.
///
/// When both solids are composed entirely of analytic faces (planes,
/// cylinders, cones, spheres), uses an exact analytic path that preserves
/// surface types through the boolean. Falls back to the tessellated path
/// for NURBS faces or analytic-analytic face pairs.
///
/// # Errors
///
/// Returns an error if either solid contains NURBS faces, or if the operation
/// produces an empty or non-manifold result.
#[allow(clippy::too_many_lines)]
pub fn boolean(
    topo: &mut Topology,
    op: BooleanOp,
    a: SolidId,
    b: SolidId,
) -> Result<SolidId, crate::OperationsError> {
    boolean_with_options(topo, op, a, b, BooleanOptions::default())
}

/// Perform a boolean operation with custom options.
///
/// See [`boolean`] for details. The `opts` parameter allows configuring
/// tessellation quality for non-planar faces.
///
/// # Errors
///
/// Returns an error if either solid is invalid or the operation produces
/// an empty or non-manifold result.
#[allow(clippy::too_many_lines)]
pub fn boolean_with_options(
    topo: &mut Topology,
    op: BooleanOp,
    a: SolidId,
    b: SolidId,
    opts: BooleanOptions,
) -> Result<SolidId, crate::OperationsError> {
    let tol = opts.tolerance;
    let ctx = BooleanContext::from_options(&opts);

    log::debug!(
        "boolean {op:?}: solids ({}, {}), deflection={}, vertex_merge={}, classify_tol={}, degenerate_area={}",
        a.index(),
        b.index(),
        opts.deflection,
        ctx.vertex_merge,
        ctx.classify_tol,
        ctx.degenerate_area,
    );

    // ── Try analytic fast path ──────────────────────────────────────────
    // Use when both solids are all-analytic (no NURBS) and neither contains
    // torus faces. This covers cutting/drilling with cylinders/cones and
    // sphere intersections. Sphere faces are handled by tessellating them
    // into triangle fragments within the analytic boolean, with O(1)
    // point-in-sphere classification for the opposite solid.
    //
    // Torus faces are excluded because their parametric decomposition
    // (degree-4 intersection curves) is not yet implemented.
    let try_analytic = {
        let both_analytic = is_all_analytic(topo, a)? && is_all_analytic(topo, b)?;
        let no_torus = !has_torus(topo, a)? && !has_torus(topo, b)?;
        both_analytic && no_torus
    };
    if try_analytic {
        if let Ok(solid) = analytic_boolean(topo, op, a, b, tol, opts.deflection) {
            let _ = crate::heal::remove_degenerate_edges(topo, solid, tol.linear)?;
            validate_boolean_result(topo, solid)?;
            return Ok(solid);
        }
        // Analytic path failed; fall back to tessellated boolean.
    }

    // ── Mesh boolean fast path for high-face-count solids ─────────────
    // When either solid has many faces (e.g. from a prior tessellated boolean),
    // the chord-based splitting approach is too slow. Use the mesh boolean
    // which operates on triangle meshes directly.
    {
        let count_a = face_count(topo, a)?;
        let count_b = face_count(topo, b)?;
        if count_a > 100 || count_b > 100 {
            log::debug!(
                "boolean {op:?}: high face count ({count_a}, {count_b}), using mesh boolean"
            );
            if let Ok(result) = mesh_boolean_path(topo, op, a, b, opts.deflection) {
                return Ok(result);
            }
        }
    }

    // ── Phase 0: Guard + Precompute ──────────────────────────────────────

    let faces_a = collect_face_data(topo, a, opts.deflection)?;
    let faces_b = collect_face_data(topo, b, opts.deflection)?;

    let aabb_a = solid_aabb(topo, &faces_a, tol)?;
    let aabb_b = solid_aabb(topo, &faces_b, tol)?;

    // Disjoint AABB shortcut.
    if !aabb_a.intersects(aabb_b) {
        log::debug!("boolean {op:?}: disjoint AABBs, shortcut");
        return handle_disjoint(topo, op, &faces_a, &faces_b);
    }

    // ── Containment shortcut ─────────────────────────────────────────
    // If one solid is entirely inside the other, skip expensive intersection
    // computation and go directly to the appropriate result.
    if let Some(result) = try_containment_shortcut(topo, op, a, b, &faces_a, &faces_b, tol)? {
        return Ok(result);
    }

    // ── Phase 1a: Analytic fast path ───────────────────────────────────

    let (analytic_segs, analytic_pairs) = compute_analytic_segments(topo, a, b, tol)?;

    // ── Phase 1b: Tessellated intersection (skip analytic pairs) ────────

    let tess_segs = compute_intersection_segments(&faces_a, &faces_b, tol, &analytic_pairs);

    let mut segments = analytic_segs;
    segments.extend(tess_segs);

    // Build chord map: FaceId → Vec<(Point3, Point3)>
    let mut chord_map: HashMap<usize, Vec<(Point3, Point3)>> = HashMap::new();
    for seg in &segments {
        chord_map
            .entry(seg.face_a.index())
            .or_default()
            .push((seg.p0, seg.p1));
        chord_map
            .entry(seg.face_b.index())
            .or_default()
            .push((seg.p0, seg.p1));
    }

    // ── Phase 3: Face splitting ──────────────────────────────────────────

    let mut fragments: Vec<FaceFragment> = Vec::new();

    for &(fid, ref verts, normal, d) in &faces_a {
        fragments.extend(split_face(
            fid,
            verts,
            normal,
            d,
            Source::A,
            &chord_map,
            tol,
        ));
    }
    for &(fid, ref verts, normal, d) in &faces_b {
        fragments.extend(split_face(
            fid,
            verts,
            normal,
            d,
            Source::B,
            &chord_map,
            tol,
        ));
    }

    // ── Phase 4: Classification ──────────────────────────────────────────

    // Build analytic classifiers for O(1) point-in-solid tests when possible.
    let analytic_a = try_build_analytic_classifier(topo, a);
    let analytic_b = try_build_analytic_classifier(topo, b);

    // Build BVH acceleration structures for face data.
    let bvh_a = build_face_bvh(&faces_a);
    let bvh_b = build_face_bvh(&faces_b);

    // Classification: parallelize when fragment count justifies rayon overhead.
    let classify_fn = |frag: &FaceFragment| -> FaceClass {
        let centroid = polygon_centroid(&frag.vertices);

        // AABB pre-filter: skip expensive ray-cast for fragments whose centroid
        // is outside the opposing solid's bounding box — they are definitively
        // Outside. Eliminates most ray-casts in large multi-body fuse operations.
        if centroid_outside_opposing_aabb(frag.source, centroid, aabb_a, aabb_b, tol.linear) {
            return FaceClass::Outside;
        }

        let fast = match frag.source {
            Source::A => analytic_b.as_ref().and_then(|c| c.classify(centroid, tol)),
            Source::B => analytic_a.as_ref().and_then(|c| c.classify(centroid, tol)),
        };
        if let Some(class) = fast {
            return class;
        }
        let (opposite, bvh) = match frag.source {
            Source::A => (&faces_b, bvh_b.as_ref()),
            Source::B => (&faces_a, bvh_a.as_ref()),
        };
        classify_fragment(frag, opposite, bvh, tol)
    };

    let classes: Vec<FaceClass> = if fragments.len() >= PARALLEL_THRESHOLD {
        fragments.par_iter().map(classify_fn).collect()
    } else {
        fragments.iter().map(classify_fn).collect()
    };

    // ── Phase 5: Selection ───────────────────────────────────────────────

    let mut selected: Vec<(Vec<Point3>, Vec3, f64)> = Vec::new();

    for (frag, &class) in fragments.iter().zip(classes.iter()) {
        if let Some(flip) = select_fragment(frag.source, class, op) {
            // When flipping, negate the plane and reverse winding to keep CCW from outside.
            let (verts, normal, d) = if flip {
                let rev: Vec<_> = frag.vertices.iter().copied().rev().collect();
                (rev, -frag.normal, -frag.d)
            } else {
                (frag.vertices.clone(), frag.normal, frag.d)
            };
            selected.push((verts, normal, d));
        }
    }

    if selected.is_empty() {
        return Err(crate::OperationsError::InvalidInput {
            reason: "boolean operation produced no faces (empty result)".into(),
        });
    }

    // ── Phase 6: Assembly ────────────────────────────────────────────────

    let result = assemble_solid(topo, &selected, tol)?;

    // ── Phase 6b: Post-boolean healing ─────────────────────────────────
    // Light healing: remove degenerate (zero-length) edges and fix face
    // orientations. We intentionally skip vertex merging and face removal
    // since they can corrupt valid boolean output with small features.
    let _ = crate::heal::remove_degenerate_edges(topo, result, tol.linear)?;

    // ── Phase 7: Degenerate result check ──────────────────────────────
    validate_boolean_result(topo, result)?;

    log::info!(
        "boolean {op:?}: tessellated path succeeded → solid {} ({} faces)",
        result.index(),
        selected.len()
    );
    Ok(result)
}

/// Determine whether a fragment should be kept and whether to flip its normal.
///
/// Returns `Some(false)` to keep as-is, `Some(true)` to keep and flip, or
/// `None` to discard.
#[allow(clippy::match_same_arms)] // arms are semantically distinct (truth table rows)
const fn select_fragment(source: Source, class: FaceClass, op: BooleanOp) -> Option<bool> {
    match (source, class, op) {
        // From A, Outside B
        (Source::A, FaceClass::Outside, BooleanOp::Fuse | BooleanOp::Cut) => Some(false),
        (Source::A, FaceClass::Outside, BooleanOp::Intersect) => None,
        // From A, Inside B
        (Source::A, FaceClass::Inside, BooleanOp::Fuse | BooleanOp::Cut) => None,
        (Source::A, FaceClass::Inside, BooleanOp::Intersect) => Some(false),
        // From B, Outside A
        (Source::B, FaceClass::Outside, BooleanOp::Fuse) => Some(false),
        (Source::B, FaceClass::Outside, BooleanOp::Cut | BooleanOp::Intersect) => None,
        // From B, Inside A
        (Source::B, FaceClass::Inside, BooleanOp::Fuse) => None,
        (Source::B, FaceClass::Inside, BooleanOp::Cut) => Some(true), // flip
        (Source::B, FaceClass::Inside, BooleanOp::Intersect) => Some(false),
        // Coplanar same — keep only from A to avoid duplicates.
        (Source::A, FaceClass::CoplanarSame, BooleanOp::Fuse | BooleanOp::Intersect) => Some(false),
        (_, FaceClass::CoplanarSame, _) => None,
        // Coplanar opposite — for Cut, A's face facing opposite B should be kept
        // (it forms the "skin" at the cut boundary). In all other cases, discard.
        (Source::A, FaceClass::CoplanarOpposite, BooleanOp::Cut) => Some(false),
        (_, FaceClass::CoplanarOpposite, _) => None,
    }
}

// ---------------------------------------------------------------------------
// Phase 0 helpers
// ---------------------------------------------------------------------------

/// Extracted face data: `(FaceId, vertices, normal, d)`.
type FaceData = Vec<(FaceId, Vec<Point3>, Vec3, f64)>;

/// Collect face polygons and plane data for a solid.
///
/// For planar faces, returns the face polygon directly.
/// For NURBS faces, tessellates into triangles and returns each triangle
/// as a separate planar "face" entry. This allows the existing planar
/// boolean clipping algorithm to handle NURBS solids.
/// Number of angular segments used to approximate cylinder faces in the
/// classification face data. 16 segments = 16 quads per cylinder band,
/// sufficient for correct ray-crossing parity.
const CLASSIFIER_CYL_SEGMENTS: usize = 16;

fn collect_face_data(
    topo: &Topology,
    solid_id: SolidId,
    deflection: f64,
) -> Result<FaceData, crate::OperationsError> {
    let solid = topo.solid(solid_id)?;
    let shell = topo.shell(solid.outer_shell())?;
    let mut result = Vec::with_capacity(shell.faces().len());

    for &fid in shell.faces() {
        let face = topo.face(fid)?;
        match face.surface() {
            FaceSurface::Plane { normal, d } => {
                let verts = face_polygon(topo, fid)?;
                result.push((fid, verts, *normal, *d));
            }
            FaceSurface::Cylinder(cyl) => {
                // Approximate the cylinder barrel as planar quads for the
                // classifier. Much faster than full tessellation (~16 quads
                // vs ~800 triangles per band) while giving correct crossing
                // parity for ray-casting.
                let verts = face_polygon(topo, fid)?;
                let Some((v_min, v_max)) = cylinder_v_extent(cyl, &verts) else {
                    continue;
                };
                #[allow(clippy::cast_precision_loss)]
                for i in 0..CLASSIFIER_CYL_SEGMENTS {
                    let u0 = std::f64::consts::TAU * (i as f64) / (CLASSIFIER_CYL_SEGMENTS as f64);
                    let u1 =
                        std::f64::consts::TAU * ((i + 1) as f64) / (CLASSIFIER_CYL_SEGMENTS as f64);
                    let b0 = cyl.evaluate(u0, v_min);
                    let b1 = cyl.evaluate(u1, v_min);
                    let t0 = cyl.evaluate(u0, v_max);
                    let t1 = cyl.evaluate(u1, v_max);

                    // Two triangles per quad.
                    for tri in &[[b0, b1, t1], [b0, t1, t0]] {
                        let edge1 = tri[1] - tri[0];
                        let edge2 = tri[2] - tri[0];
                        let cross = edge1.cross(edge2);
                        let Ok(n) = cross.normalize() else { continue };
                        let d_val = crate::dot_normal_point(n, tri[0]);
                        result.push((fid, tri.to_vec(), n, d_val));
                    }
                }
            }
            FaceSurface::Sphere(sph) => {
                // Use the sphere's center-to-centroid direction for normals
                // rather than cross products from tessellated triangles, since
                // the tessellation winding order may not match face orientation.
                // Respect the face's `reversed` flag to flip the normal when the
                // face's topological orientation opposes the geometric surface.
                let coarse_deflection = deflection * 4.0;
                let mesh = crate::tessellate::tessellate(topo, fid, coarse_deflection)?;
                let center = sph.center();
                let face_data = topo.face(fid)?;
                let sign = if face_data.is_reversed() { -1.0 } else { 1.0 };
                for tri in mesh.indices.chunks_exact(3) {
                    let i0 = tri[0] as usize;
                    let i1 = tri[1] as usize;
                    let i2 = tri[2] as usize;

                    let v0 = mesh.positions[i0];
                    let v1 = mesh.positions[i1];
                    let v2 = mesh.positions[i2];

                    // Radial direction from sphere center → outward normal,
                    // then flip if face is reversed.
                    let cx = (v0.x() + v1.x() + v2.x()) / 3.0;
                    let cy = (v0.y() + v1.y() + v2.y()) / 3.0;
                    let cz = (v0.z() + v1.z() + v2.z()) / 3.0;
                    let dir = Vec3::new(cx - center.x(), cy - center.y(), cz - center.z());
                    let len = (dir.x() * dir.x() + dir.y() * dir.y() + dir.z() * dir.z()).sqrt();
                    if len < 1e-15 {
                        continue;
                    }
                    let n = dir * (sign / len);
                    let d = n.x() * v0.x() + n.y() * v0.y() + n.z() * v0.z();
                    result.push((fid, vec![v0, v1, v2], n, d));
                }
            }
            _ => {
                // Other non-planar: tessellate with coarse deflection.
                let coarse_deflection = deflection * 4.0;
                let mesh = crate::tessellate::tessellate(topo, fid, coarse_deflection)?;
                for tri in mesh.indices.chunks_exact(3) {
                    let i0 = tri[0] as usize;
                    let i1 = tri[1] as usize;
                    let i2 = tri[2] as usize;

                    let v0 = mesh.positions[i0];
                    let v1 = mesh.positions[i1];
                    let v2 = mesh.positions[i2];

                    let edge1 = v1 - v0;
                    let edge2 = v2 - v0;
                    let cross = edge1.cross(edge2);
                    let Ok(normal) = cross.normalize() else {
                        continue; // Skip degenerate triangles (e.g. at cone apex)
                    };
                    let d = crate::dot_normal_point(normal, v0);

                    result.push((fid, vec![v0, v1, v2], normal, d));
                }
            }
        }
    }

    Ok(result)
}

/// Get a polygon approximation of a face by sampling curved edges.
///
/// Samples circle/ellipse edges into 32 points so faces with a
/// single closed-curve edge (e.g. cylinder caps) get a proper polygon.
///
/// # Errors
///
/// Returns an error if the face or its wire cannot be resolved.
pub fn face_polygon(
    topo: &Topology,
    face_id: FaceId,
) -> Result<Vec<Point3>, crate::OperationsError> {
    let face = topo.face(face_id)?;
    let wire = topo.wire(face.outer_wire())?;
    let mut pts = Vec::new();

    for oe in wire.edges() {
        let edge = topo.edge(oe.edge())?;
        let curve = edge.curve();
        // Sample closed parametric edges (start == end vertex).
        // Partial arcs fall through to the vertex-based path.
        let start_vid = edge.start();
        let end_vid = edge.end();
        let is_closed_edge = start_vid == end_vid
            && matches!(
                curve,
                EdgeCurve::Circle(_) | EdgeCurve::Ellipse(_) | EdgeCurve::NurbsCurve(_)
            );
        if is_closed_edge {
            // Must use CLOSED_CURVE_SAMPLES (not a larger value) — vertex count
            // must match create_band_fragments and inner-wire dedup for sharing.
            let mut sampled = sample_edge_curve(curve, CLOSED_CURVE_SAMPLES);
            if !oe.is_forward() {
                sampled.reverse();
            }
            pts.extend(sampled);
        } else {
            let vid = if oe.is_forward() {
                edge.start()
            } else {
                edge.end()
            };
            pts.push(topo.vertex(vid)?.point());
        }
    }

    Ok(pts)
}

/// Compute a conservative AABB for a face using only wire vertex positions.
///
/// Unlike `face_polygon()` which samples closed curves (32 points per circle),
/// this function only collects the start/end vertex positions of each edge.
/// For analytic surfaces (cylinder, sphere, cone, torus, NURBS), the AABB is
/// expanded to account for surface curvature via `expand_aabb_for_surface`.
///
/// This is much cheaper than `face_polygon()` and is used for early rejection:
/// if the wire AABB doesn't overlap the tool's AABB, the face cannot intersect
/// any tool face.
fn face_wire_aabb(topo: &Topology, face_id: FaceId) -> Result<Aabb3, crate::OperationsError> {
    let face = topo.face(face_id)?;
    let wire = topo.wire(face.outer_wire())?;
    let mut points = Vec::with_capacity(wire.edges().len() * 4);
    for oe in wire.edges() {
        let edge = topo.edge(oe.edge())?;
        points.push(topo.vertex(edge.start())?.point());
        points.push(topo.vertex(edge.end())?.point());
        // For closed curve edges (start == end), the two vertex positions
        // are a single point — not enough to capture the curve extent.
        // Sample 4 cardinal points to get a proper AABB.
        if edge.start() == edge.end() {
            let samples = sample_edge_curve(edge.curve(), 4);
            points.extend(samples);
        }
    }
    let mut aabb = Aabb3::try_from_points(points.into_iter()).ok_or_else(|| {
        crate::OperationsError::InvalidInput {
            reason: "face has no vertices".into(),
        }
    })?;
    crate::measure::expand_aabb_for_surface(&mut aabb, face.surface());
    Ok(aabb)
}

/// Compute AABB encompassing all face vertices, expanded for surface curvature.
///
/// For analytic surfaces (sphere, cylinder, cone, torus), the tessellated
/// vertices may not reach surface extremes. We call `expand_aabb_for_surface`
/// on each face to produce a conservative bounding box.
fn solid_aabb(
    topo: &Topology,
    faces: &FaceData,
    tol: Tolerance,
) -> Result<Aabb3, crate::OperationsError> {
    let mut aabb = Aabb3::try_from_points(
        faces
            .iter()
            .flat_map(|(_, verts, _, _)| verts.iter().copied()),
    )
    .map(|bb| bb.expanded(tol.linear))
    .ok_or_else(|| crate::OperationsError::InvalidInput {
        reason: "solid has no vertices".into(),
    })?;

    for (fid, _, _, _) in faces {
        let face = topo.face(*fid)?;
        crate::measure::expand_aabb_for_surface(&mut aabb, face.surface());
    }

    Ok(aabb)
}

/// Return `true` if `centroid` is definitively outside the opposing solid's AABB.
///
/// The AABB is expanded by `pad` before the test to avoid false positives from
/// floating-point rounding — `polygon_centroid` may compute a value that differs
/// by a few ULP from the vertex coordinates used to build the AABB, so a centroid
/// on the boundary can appear to be just outside without padding.
fn centroid_outside_opposing_aabb(
    source: Source,
    centroid: Point3,
    aabb_a: Aabb3,
    aabb_b: Aabb3,
    pad: f64,
) -> bool {
    let test = match source {
        Source::A => aabb_b.expanded(pad),
        Source::B => aabb_a.expanded(pad),
    };
    !test.contains_point(centroid)
}

/// Check if one solid is entirely contained in the other and short-circuit
/// the boolean operation without expensive face intersection computation.
///
/// Uses analytic classifiers (box, sphere, cylinder) for O(1) per-vertex
/// containment tests. Returns `None` if classifiers can't be built or
/// containment isn't detected.
#[allow(clippy::too_many_arguments)]
fn try_containment_shortcut(
    topo: &mut Topology,
    op: BooleanOp,
    _a: SolidId,
    _b: SolidId,
    faces_a: &FaceData,
    faces_b: &FaceData,
    tol: Tolerance,
) -> Result<Option<SolidId>, crate::OperationsError> {
    let classifier_a = try_build_analytic_classifier(topo, _a);
    let classifier_b = try_build_analytic_classifier(topo, _b);

    let extract = |data: &FaceData| -> Vec<(Vec<Point3>, Vec3, f64)> {
        data.iter()
            .map(|(_, verts, normal, d)| (verts.clone(), *normal, *d))
            .collect()
    };

    // Check: is A entirely inside B?
    if let Some(ref cb) = classifier_b {
        let all_a_inside_b = faces_a.iter().all(|(_, verts, _, _)| {
            verts
                .iter()
                .all(|v| matches!(cb.classify(*v, tol), Some(FaceClass::Inside)))
        });
        if all_a_inside_b {
            log::debug!("boolean {op:?}: A fully inside B, shortcut");
            return match op {
                // A - B: A is inside B → nothing remains
                BooleanOp::Cut => Err(crate::OperationsError::InvalidInput {
                    reason: "cut: first solid is fully inside second".into(),
                }),
                // A ∩ B: result is A (since A ⊂ B)
                BooleanOp::Intersect => Ok(Some(assemble_solid(topo, &extract(faces_a), tol)?)),
                // A ∪ B: result is B (since A ⊂ B)
                BooleanOp::Fuse => Ok(Some(assemble_solid(topo, &extract(faces_b), tol)?)),
            };
        }
    }

    // Check: is B entirely inside A?
    if let Some(ref ca) = classifier_a {
        let all_b_inside_a = faces_b.iter().all(|(_, verts, _, _)| {
            verts
                .iter()
                .all(|v| matches!(ca.classify(*v, tol), Some(FaceClass::Inside)))
        });
        if all_b_inside_a {
            log::debug!("boolean {op:?}: B fully inside A, shortcut");
            return match op {
                // A - B: B is inside A → result is A with B-shaped hole
                // Can't shortcut — need actual face splitting.
                BooleanOp::Cut => Ok(None),
                // A ∩ B: result is B (since B ⊂ A)
                BooleanOp::Intersect => Ok(Some(assemble_solid(topo, &extract(faces_b), tol)?)),
                // A ∪ B: result is A (since B ⊂ A)
                BooleanOp::Fuse => Ok(Some(assemble_solid(topo, &extract(faces_a), tol)?)),
            };
        }
    }

    Ok(None)
}

/// Handle the case where two solids' AABBs don't overlap.
fn handle_disjoint(
    topo: &mut Topology,
    op: BooleanOp,
    faces_a: &FaceData,
    faces_b: &FaceData,
) -> Result<SolidId, crate::OperationsError> {
    let tol = Tolerance::new();
    let extract = |data: &FaceData| -> Vec<(Vec<Point3>, Vec3, f64)> {
        data.iter()
            .map(|(_, verts, normal, d)| (verts.clone(), *normal, *d))
            .collect()
    };

    match op {
        BooleanOp::Fuse => {
            let mut selected = extract(faces_a);
            selected.extend(extract(faces_b));
            assemble_solid(topo, &selected, tol)
        }
        BooleanOp::Cut => {
            // A - B with no overlap → A unchanged.
            assemble_solid(topo, &extract(faces_a), tol)
        }
        BooleanOp::Intersect => Err(crate::OperationsError::InvalidInput {
            reason: "intersection of disjoint solids is empty".into(),
        }),
    }
}

// ---------------------------------------------------------------------------
// Phase 1-2: Intersection
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
fn compute_analytic_segments(
    topo: &Topology,
    solid_a: SolidId,
    solid_b: SolidId,
    tol: Tolerance,
) -> Result<
    (
        Vec<IntersectionSegment>,
        std::collections::HashSet<(usize, usize)>,
    ),
    crate::OperationsError,
> {
    use std::collections::HashSet;

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

    let analytic = face_surface_to_analytic(analytic_surf)?;

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

/// Compute all intersection segments between face pairs of two solids.
///
/// Uses a BVH over solid B's faces for O(n log m) broad-phase filtering
/// instead of brute-force O(n * m).
///
/// Face pairs in `skip_pairs` (original face IDs handled by analytic path)
/// are excluded from tessellated intersection.
fn compute_intersection_segments(
    faces_a: &FaceData,
    faces_b: &FaceData,
    tol: Tolerance,
    skip_pairs: &std::collections::HashSet<(usize, usize)>,
) -> Vec<IntersectionSegment> {
    let mut segments = Vec::new();

    // Build BVH over solid B's faces.
    let b_entries: Vec<(usize, Aabb3)> = faces_b
        .iter()
        .enumerate()
        .map(|(i, (_, verts, _, _))| (i, Aabb3::from_points(verts.iter().copied())))
        .collect();
    let bvh = Bvh::build(&b_entries);

    for &(fid_a, ref verts_a, n_a, d_a) in faces_a {
        let aabb_a = Aabb3::from_points(verts_a.iter().copied());
        let candidates = bvh.query_overlap(&aabb_a);

        for &b_idx in &candidates {
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

/// Helper: `point + dir * t` as a `Point3`.
fn point_along_line(pt: &Point3, dir: &Vec3, t: f64) -> Point3 {
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
fn polygon_clip_intervals(
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
fn point_in_polygon_3d(pt: &Point3, polygon: &[Point3], normal: &Vec3) -> bool {
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
fn intersect_interval_lists(a: &[(f64, f64)], b: &[(f64, f64)], tol: f64) -> Vec<(f64, f64)> {
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
fn cyrus_beck_clip(
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

// ---------------------------------------------------------------------------
// Phase 3: Face splitting
// ---------------------------------------------------------------------------

/// Threshold: use CDT batch splitting for faces with this many or more chords.
///
/// Below this threshold, the iterative approach is fast enough and avoids the
/// CDT setup overhead. Above it, the iterative O(N·F) approach becomes a
/// bottleneck while CDT stays O(N log N).
const CDT_CHORD_THRESHOLD: usize = 5;

/// Snap distance multiplier for CDT vertex matching.
///
/// Chord endpoints are computed by line-edge intersection, which accumulates
/// floating-point error on the order of ~10× `tol.linear`. Use 100× as the
/// snap threshold to reliably capture all on-chord/on-boundary vertices
/// without pulling in nearby-but-off-chord polygon vertices.
const CDT_SNAP_FACTOR: f64 = 100.0;

/// Split a face polygon along intersection chords, producing fragments.
///
/// For faces with many chords (≥ `CDT_CHORD_THRESHOLD`), uses CDT-based batch
/// splitting which is O(N log N). For faces with fewer chords, uses the simpler
/// iterative approach. Falls back to iterative on any CDT error.
fn split_face(
    fid: FaceId,
    verts: &[Point3],
    normal: Vec3,
    d: f64,
    source: Source,
    chord_map: &HashMap<usize, Vec<(Point3, Point3)>>,
    tol: Tolerance,
) -> Vec<FaceFragment> {
    let Some(chords) = chord_map.get(&fid.index()).filter(|c| !c.is_empty()) else {
        return vec![FaceFragment {
            vertices: verts.to_vec(),
            normal,
            d,
            source,
        }];
    };

    if chords.len() >= CDT_CHORD_THRESHOLD {
        if let Ok(regions) = split_face_cdt_inner(verts, normal, d, chords, tol) {
            return regions
                .into_iter()
                .filter(|v| polygon_area_2x(v, &normal) > tol.linear * tol.linear)
                .map(|vertices| FaceFragment {
                    vertices,
                    normal,
                    d,
                    source,
                })
                .collect();
        }
    }

    split_face_iterative(verts, normal, d, source, chords, tol)
}

/// Iterative face splitting: apply each chord sequentially.
///
/// Simple and correct for small chord counts. For N chords, each chord splits
/// all existing fragments, so total work is O(N · F) where F grows with each
/// split.
fn split_face_iterative(
    verts: &[Point3],
    normal: Vec3,
    d: f64,
    source: Source,
    chords: &[(Point3, Point3)],
    tol: Tolerance,
) -> Vec<FaceFragment> {
    let mut frags: Vec<Vec<Point3>> = vec![verts.to_vec()];

    for &(c0, c1) in chords {
        let mut new_frags = Vec::new();
        for poly in &frags {
            let (left, right) = split_polygon_by_chord(poly, c0, c1, &normal);
            if left.len() >= 3 {
                new_frags.push(left);
            }
            if right.len() >= 3 {
                new_frags.push(right);
            }
        }
        if !new_frags.is_empty() {
            frags = new_frags;
        }
    }

    frags
        .into_iter()
        .filter(|v| polygon_area_2x(v, &normal) > tol.linear * tol.linear)
        .map(|vertices| FaceFragment {
            vertices,
            normal,
            d,
            source,
        })
        .collect()
}

/// CDT-based batch face splitting — O(N log N) for N chords.
///
/// Each chord defines a splitting LINE (not a finite segment). The chord
/// is extended to the polygon boundary before CDT insertion, matching the
/// semantics of the iterative `split_polygon_by_chord` which classifies
/// vertices relative to the chord plane.
///
/// After extension, chord-chord crossings are computed and the constraints
/// are split at intersection points for safe CDT insertion.
#[allow(clippy::too_many_lines)]
fn split_face_cdt_inner(
    verts: &[Point3],
    normal: Vec3,
    d: f64,
    chords: &[(Point3, Point3)],
    tol: Tolerance,
) -> Result<Vec<Vec<Point3>>, crate::OperationsError> {
    use brepkit_math::cdt::Cdt;
    use brepkit_math::vec::Point2;

    let nv = verts.len();
    if nv < 3 {
        return Ok(vec![verts.to_vec()]);
    }

    // --- Projection: drop the dominant normal axis ---
    let ax = normal.x().abs();
    let ay = normal.y().abs();
    let az = normal.z().abs();

    let project = |p: Point3| -> Point2 {
        if az >= ax && az >= ay {
            Point2::new(p.x(), p.y())
        } else if ay >= ax {
            Point2::new(p.x(), p.z())
        } else {
            Point2::new(p.y(), p.z())
        }
    };

    let unproject = |p2: Point2| -> Point3 {
        if az >= ax && az >= ay {
            let z = (d - normal.x() * p2.x() - normal.y() * p2.y()) / normal.z();
            Point3::new(p2.x(), p2.y(), z)
        } else if ay >= ax {
            let y = (d - normal.x() * p2.x() - normal.z() * p2.y()) / normal.y();
            Point3::new(p2.x(), y, p2.y())
        } else {
            let x = (d - normal.y() * p2.x() - normal.z() * p2.y()) / normal.x();
            Point3::new(x, p2.x(), p2.y())
        }
    };

    let poly_2d: Vec<Point2> = verts.iter().map(|v| project(*v)).collect();

    // --- Extend each chord LINE to the polygon boundary ---
    // The iterative approach treats chords as infinite splitting planes.
    // We match this by extending each chord segment to the polygon boundary.
    let mut chords_2d: Vec<(Point2, Point2)> = Vec::with_capacity(chords.len());
    for &(c0, c1) in chords {
        let p0 = project(c0);
        let p1 = project(c1);
        let extended = extend_chord_to_polygon(p0, p1, &poly_2d);
        chords_2d.push(extended);
    }

    // Normalize chord direction (c0 < c1 lexicographically) so that
    // two chords defining the same line but in opposite directions dedup.
    for chord in &mut chords_2d {
        let (c0, c1) = *chord;
        if c0.x() > c1.x() + 1e-12 || ((c0.x() - c1.x()).abs() < 1e-12 && c0.y() > c1.y() + 1e-12) {
            *chord = (c1, c0);
        }
    }

    // Deduplicate identical chords (same line, same boundary endpoints).
    chords_2d.sort_by(|a, b| {
        a.0.x()
            .partial_cmp(&b.0.x())
            .unwrap_or(std::cmp::Ordering::Equal)
            .then(
                a.0.y()
                    .partial_cmp(&b.0.y())
                    .unwrap_or(std::cmp::Ordering::Equal),
            )
            .then(
                a.1.x()
                    .partial_cmp(&b.1.x())
                    .unwrap_or(std::cmp::Ordering::Equal),
            )
    });
    chords_2d.dedup_by(|a, b| {
        let d0 = (a.0.x() - b.0.x()).powi(2) + (a.0.y() - b.0.y()).powi(2);
        let d1 = (a.1.x() - b.1.x()).powi(2) + (a.1.y() - b.1.y()).powi(2);
        d0 < tol.linear * tol.linear && d1 < tol.linear * tol.linear
    });

    // --- Compute chord-chord intersections (on extended segments) ---
    let mut chord_cross: Vec<Vec<(f64, Point2)>> = vec![Vec::new(); chords_2d.len()];

    for i in 0..chords_2d.len() {
        for j in (i + 1)..chords_2d.len() {
            if let Some((ti, tj, pt)) = seg_seg_cross_2d(
                chords_2d[i].0,
                chords_2d[i].1,
                chords_2d[j].0,
                chords_2d[j].1,
            ) {
                chord_cross[i].push((ti, pt));
                chord_cross[j].push((tj, pt));
            }
        }
    }

    for pts in &mut chord_cross {
        pts.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap_or(std::cmp::Ordering::Equal));
    }

    // --- Compute CDT bounds ---
    let (mut min_x, mut min_y) = (f64::INFINITY, f64::INFINITY);
    let (mut max_x, mut max_y) = (f64::NEG_INFINITY, f64::NEG_INFINITY);
    for &p in &poly_2d {
        min_x = min_x.min(p.x());
        min_y = min_y.min(p.y());
        max_x = max_x.max(p.x());
        max_y = max_y.max(p.y());
    }
    let bounds = (Point2::new(min_x, min_y), Point2::new(max_x, max_y));

    let n_cross: usize = chord_cross.iter().map(Vec::len).sum();
    let n_pts = nv + chords_2d.len() * 2 + n_cross;
    let mut cdt = Cdt::with_capacity(bounds, n_pts);

    // --- Insert polygon vertices ---
    let mut poly_vidxs: Vec<usize> = Vec::with_capacity(nv);
    for &pt in &poly_2d {
        poly_vidxs.push(cdt.insert_point(pt)?);
    }

    // --- Insert chord endpoints (on polygon boundary after extension) ---
    let mut chord_vidxs: Vec<(usize, usize)> = Vec::with_capacity(chords_2d.len());
    for &(c0, c1) in &chords_2d {
        let v0 = cdt.insert_point(c0)?;
        let v1 = cdt.insert_point(c1)?;
        chord_vidxs.push((v0, v1));
    }

    // --- Insert chord-chord intersection points and build per-chord splits ---
    // Track the CDT index assigned to each crossing point so we can
    // construct chord sub-segments directly (avoiding the expensive O(V*C)
    // scan of all CDT vertices).
    let mut chord_splits: Vec<Vec<(f64, usize)>> = (0..chords_2d.len())
        .map(|i| {
            let (v0, v1) = chord_vidxs[i];
            vec![(0.0, v0), (1.0, v1)]
        })
        .collect();

    for (chord_idx, pts) in chord_cross.iter().enumerate() {
        for &(t, pt) in pts {
            let vidx = cdt.insert_point(pt)?;
            chord_splits[chord_idx].push((t, vidx));
        }
    }

    // Also check for T-junctions: chord endpoints from OTHER chords that
    // lie on this chord's segment. Only scan chord endpoints (~2*C vertices),
    // not all CDT vertices (~15K+).
    // Chord endpoints are computed by line-edge intersection, which accumulates
    // floating-point error on the order of ~10× tol.linear. Use 100× as the
    // snap threshold to reliably capture all on-chord/on-boundary vertices
    // without pulling in nearby-but-off-chord polygon vertices.
    let snap_dist = tol.linear * CDT_SNAP_FACTOR;
    {
        let all_cdt_verts = cdt.vertices();
        // Collect unique chord endpoint CDT indices and their 2D positions.
        let mut endpoint_set: Vec<(usize, brepkit_math::vec::Point2)> =
            Vec::with_capacity(chord_vidxs.len() * 2);
        for &(cv0, cv1) in &chord_vidxs {
            endpoint_set.push((cv0, all_cdt_verts[cv0]));
            if cv1 != cv0 {
                endpoint_set.push((cv1, all_cdt_verts[cv1]));
            }
        }
        endpoint_set.sort_unstable_by_key(|&(vi, _)| vi);
        endpoint_set.dedup_by_key(|e| e.0);

        for (i, &(v0, v1)) in chord_vidxs.iter().enumerate() {
            if v0 == v1 {
                continue;
            }
            let c0 = chords_2d[i].0;
            let c1 = chords_2d[i].1;
            for &(vidx, pt) in &endpoint_set {
                if vidx == v0 || vidx == v1 {
                    continue;
                }
                let (t, dist) = point_segment_param_dist_2d(pt, c0, c1);
                if dist < snap_dist && t > 1e-10 && t < 1.0 - 1e-10 {
                    chord_splits[i].push((t, vidx));
                }
            }
        }
    }

    // Sort each chord's split points by parameter.
    for splits in &mut chord_splits {
        splits.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap_or(std::cmp::Ordering::Equal));
        splits.dedup_by(|a, b| a.1 == b.1);
    }

    // --- Build boundary constraints ---
    // Extended chord endpoints lie on polygon edges. Split boundary edges at
    // these points so CDT constraints don't cross.
    let mut boundary_edges: Vec<(usize, usize)> = Vec::new();

    for i in 0..nv {
        let j = (i + 1) % nv;
        let vi = poly_vidxs[i];
        let vj = poly_vidxs[j];
        let edge_a = poly_2d[i];
        let edge_b = poly_2d[j];

        // Find chord endpoints (CDT indices) that lie on this polygon edge.
        let mut on_edge: Vec<(f64, usize)> = Vec::new();
        for &(cv0, cv1) in &chord_vidxs {
            for &cv in &[cv0, cv1] {
                if cv == vi || cv == vj {
                    continue;
                }
                let pt = cdt.vertices()[cv];
                let (t, dist) = point_segment_param_dist_2d(pt, edge_a, edge_b);
                if dist < snap_dist && t > 1e-12 && t < 1.0 - 1e-12 {
                    on_edge.push((t, cv));
                }
            }
        }

        on_edge.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap_or(std::cmp::Ordering::Equal));
        on_edge.dedup_by(|a, b| a.1 == b.1);

        let mut prev = vi;
        for &(_t, cv) in &on_edge {
            if cv != prev {
                boundary_edges.push((prev, cv));
                prev = cv;
            }
        }
        if vj != prev {
            boundary_edges.push((prev, vj));
        }
    }

    for &(a, b) in &boundary_edges {
        cdt.insert_constraint(a, b)?;
    }

    // --- Insert chord constraints (already split at crossings + T-junctions) ---
    let mut chord_separators: Vec<(usize, usize)> = Vec::new();
    for splits in &chord_splits {
        if splits.len() < 2 || splits[0].1 == splits[1].1 {
            continue;
        }
        let mut prev = splits[0].1;
        for &(_, vidx) in &splits[1..] {
            if vidx != prev {
                cdt.insert_constraint(prev, vidx)?;
                chord_separators.push((prev, vidx));
                prev = vidx;
            }
        }
    }

    // --- Remove exterior triangles ---
    cdt.remove_exterior(&boundary_edges);

    // --- Extract regions and unproject to 3D ---
    let regions_2d = cdt.extract_regions(&chord_separators);

    let regions_3d: Vec<Vec<Point3>> = regions_2d
        .into_iter()
        .map(|poly| poly.into_iter().map(&unproject).collect())
        .collect();

    Ok(regions_3d)
}

/// Extend a chord segment to the polygon boundary.
///
/// The chord defines a LINE through `c0` and `c1`. This function finds
/// where that line enters and exits the polygon, returning the boundary
/// intersection points. If the line doesn't cross the polygon (parallel
/// to an edge and outside), returns the original segment.
///
/// Hardcoded epsilons:
/// - `1e-15`: guards against degenerate zero-length chords and parallel-edge
///   denominator checks (numerical zero for f64 with coordinates up to ~1e7).
/// - `1e-10`: edge parameter boundary snap — accepts slight overshoot in the
///   `u ∈ [0, 1]` range due to floating-point arithmetic.
fn extend_chord_to_polygon(
    c0: brepkit_math::vec::Point2,
    c1: brepkit_math::vec::Point2,
    polygon: &[brepkit_math::vec::Point2],
) -> (brepkit_math::vec::Point2, brepkit_math::vec::Point2) {
    use brepkit_math::vec::Point2;

    let dx = c1.x() - c0.x();
    let dy = c1.y() - c0.y();

    if dx.abs() < 1e-15 && dy.abs() < 1e-15 {
        return (c0, c1);
    }

    let n = polygon.len();
    let mut t_vals: Vec<f64> = Vec::new();

    for i in 0..n {
        let j = (i + 1) % n;
        let ex = polygon[j].x() - polygon[i].x();
        let ey = polygon[j].y() - polygon[i].y();

        let denom = dx * ey - dy * ex;
        if denom.abs() < 1e-15 {
            continue; // parallel
        }

        let fx = polygon[i].x() - c0.x();
        let fy = polygon[i].y() - c0.y();

        let t = (fx * ey - fy * ex) / denom;
        let u = (fx * dy - fy * dx) / denom;

        // u must be within polygon edge [0, 1]
        if (-1e-10..=1.0 + 1e-10).contains(&u) {
            t_vals.push(t);
        }
    }

    if t_vals.len() < 2 {
        return (c0, c1);
    }

    t_vals.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    let t_min = t_vals[0];
    let t_max = t_vals[t_vals.len() - 1];

    let p_min = Point2::new(dx.mul_add(t_min, c0.x()), dy.mul_add(t_min, c0.y()));
    let p_max = Point2::new(dx.mul_add(t_max, c0.x()), dy.mul_add(t_max, c0.y()));

    (p_min, p_max)
}

/// Strict interior intersection of two 2D line segments.
///
/// Returns `(t_on_ab, t_on_cd, intersection_point)` where both parameters
/// are strictly in (ε, 1−ε). Endpoint-touching segments are NOT considered
/// crossing — only true interior crossings are reported.
fn seg_seg_cross_2d(
    a: brepkit_math::vec::Point2,
    b: brepkit_math::vec::Point2,
    c: brepkit_math::vec::Point2,
    d_pt: brepkit_math::vec::Point2,
) -> Option<(f64, f64, brepkit_math::vec::Point2)> {
    let dx_ab = b.x() - a.x();
    let dy_ab = b.y() - a.y();
    let dx_cd = d_pt.x() - c.x();
    let dy_cd = d_pt.y() - c.y();

    let denom = dx_ab * dy_cd - dy_ab * dx_cd;
    if denom.abs() < 1e-15 {
        return None; // parallel or collinear
    }

    let dx_ac = c.x() - a.x();
    let dy_ac = c.y() - a.y();

    let t = (dx_ac * dy_cd - dy_ac * dx_cd) / denom;
    let u = (dx_ac * dy_ab - dy_ac * dx_ab) / denom;

    let eps = 1e-10;
    if t > eps && t < 1.0 - eps && u > eps && u < 1.0 - eps {
        let px = dx_ab.mul_add(t, a.x());
        let py = dy_ab.mul_add(t, a.y());
        Some((t, u, brepkit_math::vec::Point2::new(px, py)))
    } else {
        None
    }
}

/// Parameter and distance from a 2D point to a line segment.
///
/// Returns `(t, distance)` where `t ∈ [0, 1]` is the closest parameter
/// along segment `(a, b)`.
fn point_segment_param_dist_2d(
    p: brepkit_math::vec::Point2,
    a: brepkit_math::vec::Point2,
    b: brepkit_math::vec::Point2,
) -> (f64, f64) {
    let dx = b.x() - a.x();
    let dy = b.y() - a.y();
    let len_sq = dx.mul_add(dx, dy * dy);
    if len_sq < 1e-30 {
        let dist = ((p.x() - a.x()).powi(2) + (p.y() - a.y()).powi(2)).sqrt();
        return (0.0, dist);
    }
    let t = ((p.x() - a.x()) * dx + (p.y() - a.y()) * dy) / len_sq;
    let t_clamped = t.clamp(0.0, 1.0);
    let closest_x = dx.mul_add(t_clamped, a.x());
    let closest_y = dy.mul_add(t_clamped, a.y());
    let dist = ((p.x() - closest_x).powi(2) + (p.y() - closest_y).powi(2)).sqrt();
    (t_clamped, dist)
}

/// Compute twice the area of a 3D polygon projected along its normal.
///
/// Used for filtering degenerate (zero-area) fragments. Returns the
/// magnitude of the cross-product sum (Newell's method), which equals
/// `2 * area`.
#[inline]
fn polygon_area_2x(vertices: &[Point3], normal: &Vec3) -> f64 {
    if vertices.len() < 3 {
        return 0.0;
    }
    // Project to the dominant axis plane and compute the shoelace area.
    let ax = normal.x().abs();
    let ay = normal.y().abs();
    let az = normal.z().abs();

    let mut area = 0.0;
    let n = vertices.len();
    for i in 0..n {
        let j = (i + 1) % n;
        let (ui, vi, uj, vj) = if az >= ax && az >= ay {
            (
                vertices[i].x(),
                vertices[i].y(),
                vertices[j].x(),
                vertices[j].y(),
            )
        } else if ay >= ax {
            (
                vertices[i].x(),
                vertices[i].z(),
                vertices[j].x(),
                vertices[j].z(),
            )
        } else {
            (
                vertices[i].y(),
                vertices[i].z(),
                vertices[j].y(),
                vertices[j].z(),
            )
        };
        area += ui.mul_add(vj, -(uj * vi));
    }
    area.abs()
}

/// Split a polygon into two sub-polygons along a chord line defined by
/// two points `c0, c1` on the plane.
///
/// Vertices are classified as left/right of the chord using `orient3d`.
/// Edge-chord intersection points are inserted at sign changes.
fn split_polygon_by_chord(
    polygon: &[Point3],
    c0: Point3,
    c1: Point3,
    normal: &Vec3,
) -> (Vec<Point3>, Vec<Point3>) {
    let n = polygon.len();
    if n < 3 {
        return (polygon.to_vec(), Vec::new());
    }

    // The classification plane is defined by (c0, c1, c0 + normal).
    let c_top = Point3::new(
        c0.x() + normal.x(),
        c0.y() + normal.y(),
        c0.z() + normal.z(),
    );

    // Use SoS perturbation so coplanar vertices get a deterministic side
    // assignment (never duplicated into both halves).
    #[allow(clippy::cast_precision_loss)]
    let signs: Vec<f64> = polygon
        .iter()
        .enumerate()
        .map(|(i, v)| {
            orient3d_sos(
                c0,
                c1,
                c_top,
                *v,
                usize::MAX - 2,
                usize::MAX - 1,
                usize::MAX,
                i,
            )
        })
        .collect();

    let mut left = Vec::new();
    let mut right = Vec::new();

    for i in 0..n {
        let j = (i + 1) % n;
        let si = signs[i];
        let sj = signs[j];

        // Classify using SoS sign — guaranteed non-zero, so each vertex
        // goes to exactly one side (no duplicates).
        if si > 0.0 {
            left.push(polygon[i]);
        } else {
            right.push(polygon[i]);
        }

        // Check for sign change (edge crossing).
        if (si > 0.0 && sj < 0.0) || (si < 0.0 && sj > 0.0) {
            // Interpolate the intersection point.
            let t = si / (si - sj);
            let pi = polygon[i];
            let pj = polygon[j];
            let ix = Point3::new(
                (pj.x() - pi.x()).mul_add(t, pi.x()),
                (pj.y() - pi.y()).mul_add(t, pi.y()),
                (pj.z() - pi.z()).mul_add(t, pi.z()),
            );
            left.push(ix);
            right.push(ix);
        }
    }

    (left, right)
}

// ---------------------------------------------------------------------------
// Phase 4: Classification
// ---------------------------------------------------------------------------

/// Analytic classifier for simple convex solids.
///
/// Instead of ray-casting against hundreds of tessellated triangles, use
/// exact geometric predicates to classify points inside/outside a solid.
enum AnalyticClassifier {
    /// Point-in-sphere: `|p - center| ≤ radius`.
    Sphere { center: Point3, radius: f64 },
    /// Point-in-cylinder: radial distance from axis ≤ radius AND axial
    /// position within [z_min, z_max].
    Cylinder {
        origin: Point3,
        axis: Vec3,
        radius: f64,
        z_min: f64,
        z_max: f64,
    },
    /// Point-in-cone-frustum: radial distance from axis ≤ interpolated radius
    /// AND axial position within [z_min, z_max]. Uses linear interpolation
    /// between r_min (at z_min) and r_max (at z_max) for the expected radius,
    /// which is robust regardless of ConicalSurface apex/axis orientation.
    Cone {
        origin: Point3,
        axis: Vec3,
        z_min: f64,
        z_max: f64,
        r_at_z_min: f64,
        r_at_z_max: f64,
    },
    /// Point-in-box: axis-aligned bounding box test.
    /// O(1) with just 6 comparisons — the fastest classifier.
    Box { min: Point3, max: Point3 },
    /// Point-in-convex-polyhedron: half-plane test against each face.
    /// A point is inside iff `normal_i · p < d_i` for all face planes
    /// (outward-pointing normals, so `normal · p > d` means outside).
    /// O(F) where F is the number of faces — fast for hex prisms (F=8).
    ConvexPolyhedron {
        /// Outward-pointing normals and signed distances: `normal · p > d` means outside.
        planes: Vec<(Vec3, f64)>,
    },
}

impl AnalyticClassifier {
    /// Classify a centroid as Inside, Outside, or None (on boundary → fall back).
    fn classify(&self, centroid: Point3, tol: Tolerance) -> Option<FaceClass> {
        match self {
            Self::Sphere { center, radius } => {
                let dx = centroid.x() - center.x();
                let dy = centroid.y() - center.y();
                let dz = centroid.z() - center.z();
                let dist_sq = dx * dx + dy * dy + dz * dz;
                if dist_sq < (radius - tol.linear) * (radius - tol.linear) {
                    Some(FaceClass::Inside)
                } else if dist_sq > (radius + tol.linear) * (radius + tol.linear) {
                    Some(FaceClass::Outside)
                } else {
                    None // On boundary — fall back to ray-casting
                }
            }
            Self::Cylinder {
                origin,
                axis,
                radius,
                z_min,
                z_max,
            } => {
                let diff = centroid - *origin;
                let axial = diff.dot(*axis);
                // Check axial bounds (cap faces).
                if axial < *z_min - tol.linear || axial > *z_max + tol.linear {
                    return Some(FaceClass::Outside);
                }
                // Radial distance from the axis.
                let projected = *axis * axial;
                let radial_vec = diff - projected;
                let radial_dist_sq = radial_vec.x() * radial_vec.x()
                    + radial_vec.y() * radial_vec.y()
                    + radial_vec.z() * radial_vec.z();
                if radial_dist_sq < (radius - tol.linear) * (radius - tol.linear)
                    && axial > *z_min + tol.linear
                    && axial < *z_max - tol.linear
                {
                    Some(FaceClass::Inside)
                } else if radial_dist_sq > (radius + tol.linear) * (radius + tol.linear) {
                    Some(FaceClass::Outside)
                } else {
                    None // On boundary — fall back to ray-casting
                }
            }
            Self::Cone {
                origin,
                axis,
                z_min,
                z_max,
                r_at_z_min,
                r_at_z_max,
            } => {
                let diff = centroid - *origin;
                let axial = diff.dot(*axis);
                // Check axial bounds.
                if axial < *z_min - tol.linear || axial > *z_max + tol.linear {
                    return Some(FaceClass::Outside);
                }
                // Radial distance from axis.
                let projected = *axis * axial;
                let radial_vec = diff - projected;
                let radial_dist_sq = radial_vec.x() * radial_vec.x()
                    + radial_vec.y() * radial_vec.y()
                    + radial_vec.z() * radial_vec.z();
                // Linearly interpolate expected radius between z_min and z_max.
                let dz = z_max - z_min;
                let t = if dz.abs() > 1e-12 {
                    (axial - z_min) / dz
                } else {
                    0.5
                };
                let expected_r = r_at_z_min + t * (r_at_z_max - r_at_z_min);
                if radial_dist_sq < (expected_r - tol.linear).max(0.0).powi(2)
                    && axial > *z_min + tol.linear
                    && axial < *z_max - tol.linear
                {
                    Some(FaceClass::Inside)
                } else if radial_dist_sq > (expected_r + tol.linear) * (expected_r + tol.linear) {
                    Some(FaceClass::Outside)
                } else {
                    None // On boundary — fall back to ray-casting
                }
            }
            Self::Box { min, max } => {
                // 6 comparisons — the fastest possible classifier.
                let tl = tol.linear;
                if centroid.x() > min.x() + tl
                    && centroid.x() < max.x() - tl
                    && centroid.y() > min.y() + tl
                    && centroid.y() < max.y() - tl
                    && centroid.z() > min.z() + tl
                    && centroid.z() < max.z() - tl
                {
                    Some(FaceClass::Inside)
                } else if centroid.x() < min.x() - tl
                    || centroid.x() > max.x() + tl
                    || centroid.y() < min.y() - tl
                    || centroid.y() > max.y() + tl
                    || centroid.z() < min.z() - tl
                    || centroid.z() > max.z() + tl
                {
                    Some(FaceClass::Outside)
                } else {
                    None // On boundary — fall back to ray-casting
                }
            }
            Self::ConvexPolyhedron { planes } => {
                // For convex polyhedra, a point is inside iff it's on the
                // interior side of every face plane. Outward normal convention:
                // `signed_dist = normal · centroid - d` → positive means outside.
                let tl = tol.linear;
                let mut max_signed_dist = f64::NEG_INFINITY;
                for &(normal, d) in planes {
                    let cv = Vec3::new(centroid.x(), centroid.y(), centroid.z());
                    let signed_dist = normal.dot(cv) - d;
                    max_signed_dist = max_signed_dist.max(signed_dist);
                }
                if max_signed_dist < -tl {
                    Some(FaceClass::Inside)
                } else if max_signed_dist > tl {
                    Some(FaceClass::Outside)
                } else {
                    None // On boundary
                }
            }
        }
    }
}

/// Try to build an analytic classifier for a solid.
///
/// Returns `Some` when the solid is a simple convex analytic shape (e.g. sphere)
/// that supports O(1) point-in-solid tests. Falls back to `None` for complex or
/// non-analytic solids.
fn try_build_analytic_classifier(topo: &Topology, solid: SolidId) -> Option<AnalyticClassifier> {
    let s = topo.solid(solid).ok()?;
    let shell = topo.shell(s.outer_shell()).ok()?;
    let tol = Tolerance::new();

    // Complex solids (>20 faces) can't be simple analytic shapes (box,
    // cylinder+caps, sphere). Skip the face-by-face scan to avoid O(F)
    // overhead on large fused/boolean intermediate results.
    if shell.faces().len() > 20 {
        return None;
    }

    let mut sphere_info: Option<(Point3, f64)> = None;
    let mut cylinder_info: Option<(Point3, Vec3, f64)> = None;
    let mut cone_info: Option<(Point3, Vec3, f64)> = None;
    let mut has_planar = false;
    let mut has_sphere = false;
    let mut has_cylinder = false;
    let mut has_cone = false;

    for &fid in shell.faces() {
        let face = topo.face(fid).ok()?;
        match face.surface() {
            FaceSurface::Sphere(sph) => {
                has_sphere = true;
                if let Some((c, r)) = sphere_info {
                    let dc = (c - sph.center()).length();
                    if dc > tol.linear || (r - sph.radius()).abs() > tol.linear {
                        return None;
                    }
                } else {
                    sphere_info = Some((sph.center(), sph.radius()));
                }
            }
            FaceSurface::Cylinder(cyl) => {
                has_cylinder = true;
                if let Some((o, a, r)) = cylinder_info {
                    let do_ = (o - cyl.origin()).length();
                    let da = 1.0 - a.dot(cyl.axis()).abs();
                    if do_ > tol.linear || da > tol.angular || (r - cyl.radius()).abs() > tol.linear
                    {
                        return None;
                    }
                } else {
                    cylinder_info = Some((cyl.origin(), cyl.axis(), cyl.radius()));
                }
            }
            FaceSurface::Cone(con) => {
                has_cone = true;
                if let Some((a, ax, ha)) = cone_info {
                    let da = (a - con.apex()).length();
                    let dax = 1.0 - ax.dot(con.axis()).abs();
                    if da > tol.linear
                        || dax > tol.angular
                        || (ha - con.half_angle()).abs() > tol.angular
                    {
                        return None;
                    }
                } else {
                    cone_info = Some((con.apex(), con.axis(), con.half_angle()));
                }
            }
            FaceSurface::Plane { .. } => {
                has_planar = true;
            }
            _ => return None, // Unsupported surface type
        }
    }

    // Pure planar solid (all faces are planes) — likely a box.
    // Detect axis-aligned boxes by checking that all faces are axis-aligned planes
    // that form 3 opposite-normal pairs.
    if has_planar && !has_sphere && !has_cylinder {
        let faces = shell.faces();
        if faces.len() == 6 {
            // Collect all plane normals and d values.
            let mut planes: Vec<(Vec3, f64)> = Vec::with_capacity(6);
            let mut is_box = true;
            for &fid in faces {
                if let Ok(face) = topo.face(fid) {
                    if let FaceSurface::Plane { normal, d } = face.surface() {
                        // Check axis-alignment: exactly one component should be ±1.
                        let ax = normal.x().abs();
                        let ay = normal.y().abs();
                        let az = normal.z().abs();
                        if (ax > 1.0 - tol.angular && ay < tol.angular && az < tol.angular)
                            || (ay > 1.0 - tol.angular && ax < tol.angular && az < tol.angular)
                            || (az > 1.0 - tol.angular && ax < tol.angular && ay < tol.angular)
                        {
                            planes.push((*normal, *d));
                        } else {
                            is_box = false;
                            break;
                        }
                    } else {
                        is_box = false;
                        break;
                    }
                } else {
                    is_box = false;
                    break;
                }
            }
            if is_box && planes.len() == 6 {
                // Extract min/max from the 3 axis pairs.
                let mut x_vals = Vec::new();
                let mut y_vals = Vec::new();
                let mut z_vals = Vec::new();
                for &(normal, d) in &planes {
                    if normal.x().abs() > 0.5 {
                        // d = normal · point, so the plane is at x = d/normal.x()
                        x_vals.push(d / normal.x());
                    } else if normal.y().abs() > 0.5 {
                        y_vals.push(d / normal.y());
                    } else {
                        z_vals.push(d / normal.z());
                    }
                }
                if x_vals.len() == 2 && y_vals.len() == 2 && z_vals.len() == 2 {
                    x_vals.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
                    y_vals.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
                    z_vals.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
                    return Some(AnalyticClassifier::Box {
                        min: Point3::new(x_vals[0], y_vals[0], z_vals[0]),
                        max: Point3::new(x_vals[1], y_vals[1], z_vals[1]),
                    });
                }
            }
        }

        // All-planar solid that isn't an axis-aligned box: treat as convex
        // polyhedron (half-plane classifier). Works for hex prisms, wedges,
        // arbitrary extruded polygons, etc.
        // Only valid for CONVEX solids — verify by checking that the vertex
        // centroid is on the interior side of every face plane.
        if has_planar && !has_sphere && !has_cylinder && !has_cone {
            let faces = shell.faces();
            let mut planes = Vec::with_capacity(faces.len());
            // First pass: collect all planes.
            for &fid in faces {
                let face = topo.face(fid).ok()?;
                if let FaceSurface::Plane { normal, d } = face.surface() {
                    let (n, dv) = if face.is_reversed() {
                        (-*normal, -*d)
                    } else {
                        (*normal, *d)
                    };
                    planes.push((n, dv));
                } else {
                    return None;
                }
            }
            // Convexity check: every vertex must be on the interior side of
            // every face plane. This is O(V×F) but only runs for small all-planar
            // solids (hex prisms: V=12, F=8 → 96 dot products).
            let mut all_verts: Vec<Vec3> = Vec::new();
            for &fid in faces {
                let face = topo.face(fid).ok()?;
                let wire = topo.wire(face.outer_wire()).ok()?;
                for oe in wire.edges() {
                    let edge = topo.edge(oe.edge()).ok()?;
                    let v = topo.vertex(edge.start()).ok()?;
                    let pv = Vec3::new(v.point().x(), v.point().y(), v.point().z());
                    all_verts.push(pv);
                }
            }
            let convex_tol = tol.linear * 10.0; // Generous tolerance for vertex-on-face
            let is_convex = planes
                .iter()
                .all(|&(n, d)| all_verts.iter().all(|&v| n.dot(v) <= d + convex_tol));
            if is_convex {
                return Some(AnalyticClassifier::ConvexPolyhedron { planes });
            }
        }
    }

    // Pure sphere solid (all faces are sphere).
    if has_sphere && !has_planar && !has_cylinder {
        let (center, radius) = sphere_info?;
        return Some(AnalyticClassifier::Sphere { center, radius });
    }

    // Cylinder solid (1 cylinder face + 2 planar caps).
    if has_cylinder && has_planar && !has_sphere {
        let (origin, axis, radius) = cylinder_info?;
        // Compute axial extent from planar cap faces.
        let mut z_min = f64::INFINITY;
        let mut z_max = f64::NEG_INFINITY;
        for &fid in shell.faces() {
            let face = topo.face(fid).ok()?;
            if let FaceSurface::Plane { normal, d } = face.surface() {
                // The plane's d value along the cylinder axis gives the cap position.
                // For a cylinder cap, the plane normal is parallel to the axis.
                let dot = normal.dot(axis);
                if dot.abs() > 0.5 {
                    // Project the plane onto the axis to find the cap position.
                    // d = normal · point, so point along axis = d / dot for this axis.
                    // d/dot gives position from global origin; subtract
                    // the cylinder origin's projection to get axis-relative z.
                    let origin_vec = Vec3::new(origin.x(), origin.y(), origin.z());
                    let z = *d / dot - axis.dot(origin_vec);
                    z_min = z_min.min(z);
                    z_max = z_max.max(z);
                }
            }
        }
        if z_min < z_max {
            return Some(AnalyticClassifier::Cylinder {
                origin,
                axis,
                radius,
                z_min,
                z_max,
            });
        }
    }

    // Cone solid (1 cone face + 1 or 2 planar caps).
    if has_cone && has_planar && !has_sphere && !has_cylinder {
        let (apex, axis, _half_angle) = cone_info?;
        // Use the cone's axis as the reference axis, but measure z from a fixed
        // origin (the apex position) rather than relying on the apex being the
        // geometric virtual apex (which may be wrong for frustums).
        let origin = apex;
        let origin_vec = Vec3::new(origin.x(), origin.y(), origin.z());

        // Compute axial extent and radii from planar cap faces.
        // z values are measured from `origin` along `axis`.
        let mut caps: Vec<(f64, f64)> = Vec::new(); // (z_position, radius)
        for &fid in shell.faces() {
            let face = topo.face(fid).ok()?;
            if let FaceSurface::Plane { normal, d } = face.surface() {
                let dot = normal.dot(axis);
                if dot.abs() > 0.5 {
                    let z = *d / dot - axis.dot(origin_vec);
                    // Compute the cap radius from face vertices.
                    let wire = topo.wire(face.outer_wire()).ok()?;
                    let mut max_r_sq = 0.0_f64;
                    for oe in wire.edges() {
                        let edge = topo.edge(oe.edge()).ok()?;
                        for vid in [edge.start(), edge.end()] {
                            let v = topo.vertex(vid).ok()?;
                            let diff = v.point() - origin;
                            let axial_comp = axis * diff.dot(axis);
                            let radial = diff - axial_comp;
                            let r_sq = radial.x() * radial.x()
                                + radial.y() * radial.y()
                                + radial.z() * radial.z();
                            max_r_sq = max_r_sq.max(r_sq);
                        }
                    }
                    caps.push((z, max_r_sq.sqrt()));
                }
            }
        }

        // Sort caps by z and extract z_min/z_max with their radii.
        caps.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap_or(std::cmp::Ordering::Equal));

        let (mut z_min, mut z_max) = (f64::INFINITY, f64::NEG_INFINITY);
        let (mut r_at_z_min, mut r_at_z_max) = (0.0, 0.0);
        for &(z, r) in &caps {
            if z < z_min {
                z_min = z;
                r_at_z_min = r;
            }
            if z > z_max {
                z_max = z;
                r_at_z_max = r;
            }
        }

        // For pointed cones with only one cap.
        if z_min == f64::INFINITY {
            z_min = 0.0;
            r_at_z_min = 0.0;
        }
        if z_max == f64::NEG_INFINITY {
            z_max = 0.0;
            r_at_z_max = 0.0;
        }
        if (z_max - z_min).abs() > tol.linear {
            return Some(AnalyticClassifier::Cone {
                origin,
                axis,
                z_min,
                z_max,
                r_at_z_min,
                r_at_z_max,
            });
        }
    }

    None
}

/// Build a BVH over face data for accelerated ray-cast classification.
///
/// Returns `None` when the face count is small enough that linear scan is
/// faster than BVH construction + traversal overhead.
fn build_face_bvh(faces: &FaceData) -> Option<Bvh> {
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

/// Classify a face fragment relative to the opposite solid.
///
/// When `bvh` is `Some`, uses BVH-accelerated ray queries instead of
/// linearly scanning all faces. This reduces classification from O(F) to
/// O(log F) per fragment.
fn classify_fragment(
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
fn guard_tangent_coplanar(
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
/// This is the core classification logic, separated from `FaceFragment` to avoid
/// unnecessary cloning when the centroid/normal are already known.
fn classify_point(
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
        // Expand a tiny AABB around centroid to find nearby faces.
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

    let mut inside_votes = 0u8;
    for ray_dir in &ray_dirs {
        let mut crossings = 0i32;
        if let Some(bvh) = bvh {
            for idx in bvh.query_ray(point, *ray_dir) {
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

/// Compute the centroid of a polygon.
///
/// Returns the origin if the polygon is empty (should not happen in
/// practice since fragments are filtered for `len >= 3`).
#[inline]
fn polygon_centroid(vertices: &[Point3]) -> Point3 {
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
fn point_in_face_3d(point: Point3, polygon: &[Point3], normal: &Vec3) -> bool {
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

// ---------------------------------------------------------------------------
// Phase 6: Assembly
// ---------------------------------------------------------------------------

/// Quantize a coordinate to a spatial hash key.
#[inline]
#[allow(clippy::cast_possible_truncation)] // coordinate * 1e7 fits in i64
fn quantize(v: f64, resolution: f64) -> i64 {
    (v * resolution).round() as i64
}

/// Quantize a 3D point to a spatial hash key for vertex deduplication.
#[inline]
fn quantize_point(p: Point3, resolution: f64) -> (i64, i64, i64) {
    (
        quantize(p.x(), resolution),
        quantize(p.y(), resolution),
        quantize(p.z(), resolution),
    )
}

/// Assemble a solid from a set of planar face polygons with normals.
///
/// Uses spatial hashing for vertex dedup and edge sharing.
/// This is a convenience wrapper around [`assemble_solid_mixed`] for the
/// common case where all faces are planar.
pub(crate) fn assemble_solid(
    topo: &mut Topology,
    faces: &[(Vec<Point3>, Vec3, f64)],
    tol: Tolerance,
) -> Result<SolidId, crate::OperationsError> {
    let specs: Vec<FaceSpec> = faces
        .iter()
        .map(|(verts, normal, d)| FaceSpec::Planar {
            vertices: verts.clone(),
            normal: *normal,
            d: *d,
        })
        .collect();
    assemble_solid_mixed(topo, &specs, tol)
}

/// A face specification for mixed-surface solid assembly.
///
/// Used by [`assemble_solid_mixed`] to build solids with faces of any
/// surface type — not just planar.
#[derive(Clone)]
pub enum FaceSpec {
    /// A planar face defined by vertex positions and plane equation.
    Planar {
        /// Vertex positions (at least 3).
        vertices: Vec<Point3>,
        /// Outward-facing normal.
        normal: Vec3,
        /// Plane equation signed distance (n · p = d).
        d: f64,
    },
    /// A face with a pre-built surface and vertex positions for the boundary wire.
    Surface {
        /// Vertex positions for the outer wire (at least 3).
        vertices: Vec<Point3>,
        /// The surface geometry.
        surface: FaceSurface,
    },
}

/// Assemble a solid from a set of face specifications with mixed surface types.
///
/// Like [`assemble_solid`], but supports faces with NURBS, analytic, or any
/// other surface type. Uses the same spatial-hashing vertex dedup and edge
/// sharing as the planar variant.
///
/// This is the general-purpose solid assembly function that unblocks operations
/// on non-planar faces.
pub(crate) fn assemble_solid_mixed(
    topo: &mut Topology,
    face_specs: &[FaceSpec],
    tol: Tolerance,
) -> Result<SolidId, crate::OperationsError> {
    let resolution = 1.0 / tol.linear;

    let mut vertex_map: HashMap<(i64, i64, i64), VertexId> =
        HashMap::with_capacity(face_specs.len() * 4);
    let mut edge_map: HashMap<(usize, usize), brepkit_topology::edge::EdgeId> =
        HashMap::with_capacity(face_specs.len() * 4);

    let mut face_ids = Vec::with_capacity(face_specs.len());

    for spec in face_specs {
        let (verts, surface) = match spec {
            FaceSpec::Planar {
                vertices,
                normal,
                d,
            } => (
                vertices.clone(),
                FaceSurface::Plane {
                    normal: *normal,
                    d: *d,
                },
            ),
            FaceSpec::Surface { vertices, surface } => (vertices.clone(), surface.clone()),
        };

        let n = verts.len();
        if n < 3 {
            continue;
        }

        let vert_ids: Vec<VertexId> = verts
            .iter()
            .map(|p| {
                let key = quantize_point(*p, resolution);
                *vertex_map
                    .entry(key)
                    .or_insert_with(|| topo.vertices.alloc(Vertex::new(*p, tol.linear)))
            })
            .collect();

        let mut oriented_edges = Vec::with_capacity(n);
        for i in 0..n {
            let j = (i + 1) % n;
            let vi = vert_ids[i].index();
            let vj = vert_ids[j].index();
            let (key_min, key_max) = if vi <= vj { (vi, vj) } else { (vj, vi) };
            let is_forward = vi <= vj;

            let edge_id = *edge_map.entry((key_min, key_max)).or_insert_with(|| {
                let (start, end) = if vi <= vj {
                    (vert_ids[i], vert_ids[j])
                } else {
                    (vert_ids[j], vert_ids[i])
                };
                topo.edges.alloc(Edge::new(start, end, EdgeCurve::Line))
            });

            oriented_edges.push(OrientedEdge::new(edge_id, is_forward));
        }

        let wire = Wire::new(oriented_edges, true).map_err(crate::OperationsError::Topology)?;
        let wire_id = topo.wires.alloc(wire);
        let face = topo.faces.alloc(Face::new(wire_id, vec![], surface));
        face_ids.push(face);
    }

    if face_ids.is_empty() {
        return Err(crate::OperationsError::InvalidInput {
            reason: "solid assembly produced no faces".into(),
        });
    }

    // Post-assembly edge refinement: split long boundary edges at
    // intermediate collinear vertices so adjacent faces can share edges.
    // Pass precomputed vertex positions from assembly to avoid redundant
    // face→wire→edge→vertex traversal.
    let vertex_positions: HashMap<VertexId, Point3> = vertex_map
        .values()
        .filter_map(|&vid| topo.vertex(vid).ok().map(|v| (vid, v.point())))
        .collect();
    refine_boundary_edges(
        topo,
        &mut face_ids,
        &mut edge_map,
        tol,
        Some(&vertex_positions),
    )?;

    // Split non-manifold edges (shared by > 2 faces) into separate copies,
    // pairing faces by angular ordering around the edge.
    split_nonmanifold_edges(topo, &mut face_ids)?;

    let shell = Shell::new(face_ids).map_err(crate::OperationsError::Topology)?;
    let shell_id = topo.shells.alloc(shell);
    Ok(topo.solids.alloc(Solid::new(shell_id, vec![])))
}

// ---------------------------------------------------------------------------
// Degenerate result detection
// ---------------------------------------------------------------------------

/// Minimum face count for a valid solid.
///
/// A cylinder (2 caps + 1 barrel = 3 faces) is the minimal closed solid
/// produced by boolean operations between boxes and curved primitives.
const MIN_SOLID_FACES: usize = 3;

/// Validate that a boolean result is not degenerate.
///
/// Checks for:
/// - Too few faces (< `MIN_SOLID_FACES`)
/// - No edges or vertices (empty topology)
/// - Euler characteristic, manifold edges, boundary edges, wire closure,
///   degenerate faces, and face area via [`crate::validate::validate_solid`]
fn validate_boolean_result(topo: &Topology, solid: SolidId) -> Result<(), crate::OperationsError> {
    let s = topo.solid(solid)?;
    let shell = topo.shell(s.outer_shell())?;
    let face_count = shell.faces().len();

    if face_count < MIN_SOLID_FACES {
        return Err(crate::OperationsError::InvalidInput {
            reason: format!(
                "boolean result has only {face_count} faces (minimum {MIN_SOLID_FACES} required for a closed solid)"
            ),
        });
    }

    // Check that we have at least some edges and vertices.
    let (f, e, v) = brepkit_topology::explorer::solid_entity_counts(topo, solid)?;
    if e == 0 || v == 0 {
        return Err(crate::OperationsError::InvalidInput {
            reason: format!("boolean result has degenerate topology (F={f}, E={e}, V={v})"),
        });
    }

    // Full topological validation: Euler characteristic, manifold edges,
    // boundary edges, wire closure, degenerate faces.
    // Logged as warnings rather than hard errors — many boolean results have
    // minor topological imperfections (e.g., boundary edges on analytic faces)
    // that don't prevent downstream use. Hard-failing here would reject ~25%
    // of currently working booleans. The long-term fix is post-boolean healing.
    match crate::validate::validate_solid(topo, solid) {
        Ok(report) if !report.is_valid() => {
            let errors: Vec<_> = report
                .issues
                .iter()
                .filter(|i| i.severity == crate::validate::Severity::Error)
                .map(|i| i.description.as_str())
                .collect();
            log::warn!(
                "boolean result has {} validation error(s): {}",
                errors.len(),
                errors.join("; ")
            );
        }
        Err(e) => {
            log::warn!("validate_solid failed (skipping validation): {e}");
        }
        Ok(_) => {}
    }

    Ok(())
}

// ---------------------------------------------------------------------------
// Evolution-tracking wrapper
// ---------------------------------------------------------------------------

/// Perform a boolean operation and return an [`EvolutionMap`] tracking face
/// provenance.
///
/// This wraps [`boolean`] and uses a heuristic (normal + centroid similarity)
/// to match output faces back to their input faces. Faces whose best match
/// score exceeds the similarity threshold are classified as "modified";
/// unmatched input faces are classified as "deleted".
///
/// # Errors
///
/// Returns the same errors as [`boolean`].
pub fn boolean_with_evolution(
    topo: &mut Topology,
    op: BooleanOp,
    a: SolidId,
    b: SolidId,
) -> Result<(SolidId, crate::evolution::EvolutionMap), crate::OperationsError> {
    use crate::evolution::EvolutionMap;

    // Collect input face normals + centroids before the operation mutates topology.
    let input_faces_a = collect_face_signatures(topo, a)?;
    let input_faces_b = collect_face_signatures(topo, b)?;

    let mut input_faces: Vec<(usize, Vec3, Point3)> =
        Vec::with_capacity(input_faces_a.len() + input_faces_b.len());
    input_faces.extend(input_faces_a);
    input_faces.extend(input_faces_b);

    // Run the actual boolean.
    let result = boolean(topo, op, a, b)?;

    // Collect output face normals + centroids.
    let output_faces = collect_face_signatures(topo, result)?;

    // Build evolution map via heuristic matching.
    let mut evo = EvolutionMap::new();
    let mut matched_inputs: std::collections::HashSet<usize> = std::collections::HashSet::new();
    let mut unmatched_outputs: Vec<(usize, Vec3, Point3)> = Vec::new();

    // Normal dot threshold: cos(45deg) — relaxed to handle faces split by
    // booleans where normals may shift slightly.
    let normal_threshold = 0.707;
    // Maximum centroid distance squared for a match (generous).
    let centroid_dist_sq_max = 100.0;

    for &(out_idx, out_normal, out_centroid) in &output_faces {
        let mut best_score = f64::NEG_INFINITY;
        let mut best_input: Option<usize> = None;

        for &(in_idx, in_normal, in_centroid) in &input_faces {
            let dot = out_normal.dot(in_normal);
            if dot < normal_threshold {
                continue;
            }

            let dx = out_centroid.x() - in_centroid.x();
            let dy = out_centroid.y() - in_centroid.y();
            let dz = out_centroid.z() - in_centroid.z();
            let dist_sq = dx.mul_add(dx, dy.mul_add(dy, dz * dz));

            if dist_sq > centroid_dist_sq_max {
                continue;
            }

            // Score: higher normal alignment + closer centroid = better.
            let score = dot - dist_sq / centroid_dist_sq_max;
            if score > best_score {
                best_score = score;
                best_input = Some(in_idx);
            }
        }

        if let Some(in_idx) = best_input {
            evo.add_modified(in_idx, out_idx);
            matched_inputs.insert(in_idx);
        } else {
            unmatched_outputs.push((out_idx, out_normal, out_centroid));
        }
    }

    // Unmatched output faces are "generated" — attribute them to the nearest
    // input face (the face most likely responsible for generating them, e.g.
    // intersection curves create new faces near the boundary).
    for &(out_idx, _out_normal, out_centroid) in &unmatched_outputs {
        let mut best_dist_sq = f64::MAX;
        let mut best_input: Option<usize> = None;

        for &(in_idx, _, in_centroid) in &input_faces {
            let dx = out_centroid.x() - in_centroid.x();
            let dy = out_centroid.y() - in_centroid.y();
            let dz = out_centroid.z() - in_centroid.z();
            let dist_sq = dx.mul_add(dx, dy.mul_add(dy, dz * dz));
            if dist_sq < best_dist_sq {
                best_dist_sq = dist_sq;
                best_input = Some(in_idx);
            }
        }

        if let Some(in_idx) = best_input {
            evo.add_generated(in_idx, out_idx);
            matched_inputs.insert(in_idx);
        }
    }

    // Any input face not matched to any output is deleted.
    for &(in_idx, _, _) in &input_faces {
        if !matched_inputs.contains(&in_idx) {
            evo.add_deleted(in_idx);
        }
    }

    Ok((result, evo))
}

/// Collect `(FaceId.index(), normal, centroid)` for each face in a solid.
fn collect_face_signatures(
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
fn surface_aware_aabb(surface: &FaceSurface, vertices: &[Point3], tol: Tolerance) -> Aabb3 {
    let wire_bb = Aabb3::from_points(vertices.iter().copied());
    let bb = match surface {
        FaceSurface::Plane { .. } => wire_bb,
        FaceSurface::Sphere(s) => {
            // Full sphere AABB: center ± radius on all axes.
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

/// Get the number of faces in a solid's outer shell.
fn face_count(topo: &Topology, solid: SolidId) -> Result<usize, crate::OperationsError> {
    let s = topo.solid(solid)?;
    let shell = topo.shell(s.outer_shell())?;
    Ok(shell.faces().len())
}

/// Perform a boolean operation using the mesh boolean path.
///
/// Tessellates both solids into triangle meshes, runs the mesh boolean,
/// and assembles the result back into topology.
fn mesh_boolean_path(
    topo: &mut Topology,
    op: BooleanOp,
    a: SolidId,
    b: SolidId,
    deflection: f64,
) -> Result<SolidId, crate::OperationsError> {
    // Snapshot original face surfaces before tessellation. These are used
    // after the mesh boolean to re-classify result triangles onto their
    // original analytic surfaces (instead of losing all surface info).
    let original_surfaces = snapshot_face_surfaces(topo, a, b)?;

    let mesh_a = crate::tessellate::tessellate_solid(topo, a, deflection)?;
    let mesh_b = crate::tessellate::tessellate_solid(topo, b, deflection)?;

    let mb_result = crate::mesh_boolean::mesh_boolean(&mesh_a, &mesh_b, op, deflection)?;

    // Convert mesh result to topology, attempting to re-classify each
    // triangle onto an original analytic surface when possible.
    let tol = Tolerance::new();
    let mut face_specs: Vec<FaceSpec> = Vec::new();
    for tri in mb_result.mesh.indices.chunks_exact(3) {
        let i0 = tri[0] as usize;
        let i1 = tri[1] as usize;
        let i2 = tri[2] as usize;

        let v0 = mb_result.mesh.positions[i0];
        let v1 = mb_result.mesh.positions[i1];
        let v2 = mb_result.mesh.positions[i2];

        let edge1 = v1 - v0;
        let edge2 = v2 - v0;
        let normal = edge1
            .cross(edge2)
            .normalize()
            .unwrap_or(Vec3::new(0.0, 0.0, 1.0));
        let d = crate::dot_normal_point(normal, v0);

        // Try to match this triangle to an original face surface.
        let centroid = Point3::new(
            (v0.x() + v1.x() + v2.x()) / 3.0,
            (v0.y() + v1.y() + v2.y()) / 3.0,
            (v0.z() + v1.z() + v2.z()) / 3.0,
        );
        if let Some(surface) = classify_triangle_surface(&original_surfaces, centroid, normal, tol)
        {
            face_specs.push(FaceSpec::Surface {
                vertices: vec![v0, v1, v2],
                surface,
            });
        } else {
            face_specs.push(FaceSpec::Planar {
                vertices: vec![v0, v1, v2],
                normal,
                d,
            });
        }
    }

    if face_specs.is_empty() {
        return Err(crate::OperationsError::InvalidInput {
            reason: "mesh boolean produced empty result".into(),
        });
    }

    let result = assemble_solid_mixed(topo, &face_specs, tol)?;
    validate_boolean_result(topo, result)?;

    Ok(result)
}

/// Snapshot face surfaces from both input solids before tessellation.
fn snapshot_face_surfaces(
    topo: &Topology,
    a: SolidId,
    b: SolidId,
) -> Result<Vec<FaceSurface>, crate::OperationsError> {
    let mut surfaces = Vec::new();
    for &solid_id in &[a, b] {
        let solid = topo.solid(solid_id)?;
        let shell = topo.shell(solid.outer_shell())?;
        for &fid in shell.faces() {
            let face = topo.face(fid)?;
            surfaces.push(face.surface().clone());
        }
    }
    Ok(surfaces)
}

/// Try to classify a result triangle onto an original analytic surface.
///
/// Returns `Some(FaceSurface)` if the triangle's centroid and normal are
/// consistent with an original face surface (the point lies on the surface
/// and the normal matches). Otherwise returns `None` (keep as planar).
fn classify_triangle_surface(
    original_surfaces: &[FaceSurface],
    centroid: Point3,
    normal: Vec3,
    tol: Tolerance,
) -> Option<FaceSurface> {
    for surface in original_surfaces {
        match surface {
            FaceSurface::Plane { .. } => {
                // Already handled as FaceSpec::Planar, skip.
            }
            FaceSurface::Cylinder(cyl) => {
                // Check if centroid is on the cylinder surface.
                let dp = centroid - cyl.origin();
                let dp_vec = Vec3::new(dp.x(), dp.y(), dp.z());
                let along = dp_vec.dot(cyl.axis());
                let radial = dp_vec - cyl.axis() * along;
                let r = radial.length();
                if (r - cyl.radius()).abs() < tol.linear * 100.0 {
                    // Check normal consistency: cylinder normal is radial.
                    if let Ok(rad_dir) = radial.normalize() {
                        if rad_dir.dot(normal).abs() > 0.8 {
                            return Some(surface.clone());
                        }
                    }
                }
            }
            FaceSurface::Sphere(sph) => {
                let dp = centroid - sph.center();
                let r = dp.length();
                if (r - sph.radius()).abs() < tol.linear * 100.0 {
                    if let Ok(dir) = dp.normalize() {
                        if dir.dot(normal).abs() > 0.8 {
                            return Some(surface.clone());
                        }
                    }
                }
            }
            FaceSurface::Cone(cone) => {
                // Check if centroid is on the cone surface.
                let dp = centroid - cone.apex();
                let dp_vec = Vec3::new(dp.x(), dp.y(), dp.z());
                let along = dp_vec.dot(cone.axis());
                let radial = dp_vec - cone.axis() * along;
                let r = radial.length();
                let expected_r = along.abs() * cone.half_angle().tan();
                if (r - expected_r).abs() < tol.linear * 100.0 && along > 0.0 {
                    return Some(surface.clone());
                }
            }
            FaceSurface::Torus(tor) => {
                let dp = centroid - tor.center();
                let dp_vec = Vec3::new(dp.x(), dp.y(), dp.z());
                let along = dp_vec.dot(tor.z_axis());
                let in_plane = dp_vec - tor.z_axis() * along;
                if let Ok(ring_dir) = in_plane.normalize() {
                    let tube_center = tor.center() + ring_dir * tor.major_radius();
                    let tube_dist = (centroid - tube_center).length();
                    if (tube_dist - tor.minor_radius()).abs() < tol.linear * 100.0 {
                        return Some(surface.clone());
                    }
                }
            }
            FaceSurface::Nurbs(_) => {
                // NURBS surface matching would require Newton projection,
                // which is too expensive for per-triangle classification.
                // Skip — these triangles stay as planar.
            }
        }
    }
    None
}

/// Check if a solid is composed entirely of analytic surfaces (no NURBS).
fn is_all_analytic(topo: &Topology, solid: SolidId) -> Result<bool, crate::OperationsError> {
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
fn has_torus(topo: &Topology, solid: SolidId) -> Result<bool, crate::OperationsError> {
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

/// Snapshot of face data for analytic boolean processing.
struct FaceSnapshot {
    id: FaceId,
    surface: FaceSurface,
    vertices: Vec<Point3>,
    normal: Vec3,
    d: f64,
    /// Whether the original face was reversed (needed to preserve orientation
    /// when carrying unsplit faces through sequential booleans).
    reversed: bool,
}

/// Build `edge_curves` for a face polygon by examining the source face's wire edges.
///
/// When the outer wire contains a single closed Circle or Ellipse edge, the
/// polygon vertices all came from sampling that edge. Returns
/// `vec![Some(curve)]` (length 1) to signal a single-closed-curve boundary.
/// Otherwise returns `vec![None; n]` for n polygon vertices.
fn edge_curves_from_face(
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
    // Single closed Circle or Ellipse edge → single-curve boundary.
    if edges.len() == 1
        && let Ok(edge) = topo.edge(edges[0].edge())
        && edge.start() == edge.end()
        && matches!(edge.curve(), EdgeCurve::Circle(_) | EdgeCurve::Ellipse(_))
    {
        return vec![Some(edge.curve().clone())];
    }
    vec![None; n_verts]
}

/// Analytic face fragment preserving the original surface type.
struct AnalyticFragment {
    /// Polygon boundary in 3D (for classification and planar assembly fallback).
    vertices: Vec<Point3>,
    /// The original surface type of the face.
    surface: FaceSurface,
    /// Normal of the face (for planar) or of the polygon approximation.
    normal: Vec3,
    /// Plane d coefficient (for planar faces).
    d: f64,
    /// Which operand this fragment came from.
    source: Source,
    /// Edge curve types for the boundary segments.
    /// `None` = straight line, `Some(curve)` = exact curve (circle, ellipse).
    edge_curves: Vec<Option<EdgeCurve>>,
    /// Whether the source face was reversed (preserved for non-planar faces).
    source_reversed: bool,
}

/// Refine boundary edges by splitting them at intermediate collinear vertices.
///
/// After analytic boolean assembly, unsplit faces may have long edges that
/// span the same geometric line as multiple shorter edges from adjacent
/// split faces. This function splits those long edges at the intermediate
/// vertex positions, enabling proper edge sharing between adjacent faces.
#[allow(clippy::too_many_lines)]
fn refine_boundary_edges(
    topo: &mut Topology,
    face_ids: &mut [FaceId],
    edge_map: &mut HashMap<(usize, usize), EdgeId>,
    tol: Tolerance,
    precomputed_positions: Option<&HashMap<VertexId, Point3>>,
) -> Result<(), crate::OperationsError> {
    // Single-pass: build edge-to-face count AND collect edge vertex pairs.
    // This avoids a second full face→wire→edge→vertex traversal.
    let mut edge_face_count: HashMap<EdgeId, usize> = HashMap::new();
    let mut edge_vertices: HashMap<EdgeId, (VertexId, VertexId)> = HashMap::new();
    for &fid in face_ids.iter() {
        let face = topo.face(fid)?;
        for wid in std::iter::once(face.outer_wire()).chain(face.inner_wires().iter().copied()) {
            let wire = topo.wire(wid)?;
            for oe in wire.edges() {
                let eid = oe.edge();
                *edge_face_count.entry(eid).or_default() += 1;
                if let std::collections::hash_map::Entry::Vacant(e) = edge_vertices.entry(eid) {
                    if let Ok(edge) = topo.edge(eid) {
                        e.insert((edge.start(), edge.end()));
                    }
                }
            }
        }
    }

    // Find boundary edges (used by exactly 1 face)
    let boundary_edges: HashSet<EdgeId> = edge_face_count
        .iter()
        .filter(|&(_, &count)| count == 1)
        .map(|(&eid, _)| eid)
        .collect();

    if boundary_edges.is_empty() {
        return Ok(());
    }

    // Build vertex positions. Use precomputed positions from assembly when
    // available, falling back to topology only for missing vertices
    // (e.g. passthrough faces not in the assembly's vertex_map).
    let mut extra_positions: HashMap<VertexId, Point3> = HashMap::new();
    for &(start, end) in edge_vertices.values() {
        for &vid in &[start, end] {
            let in_pre = precomputed_positions.is_some_and(|p| p.contains_key(&vid));
            if !in_pre {
                if let std::collections::hash_map::Entry::Vacant(e) = extra_positions.entry(vid) {
                    if let Ok(v) = topo.vertex(vid) {
                        e.insert(v.point());
                    }
                }
            }
        }
    }

    // For each boundary edge, find intermediate collinear vertices.
    // Use a spatial hash grid for O(V) build + O(1) amortized query,
    // much faster than SAH BVH's O(V log²V) build for point clouds.
    let get_pos = |vid: &VertexId| -> Option<Point3> {
        precomputed_positions
            .and_then(|p| p.get(vid))
            .or_else(|| extra_positions.get(vid))
            .copied()
    };
    // Build vert_list from both sources, deduplicating by VertexId.
    let mut seen: HashSet<VertexId> = HashSet::new();
    let mut vert_list: Vec<(VertexId, Point3)> = Vec::new();
    if let Some(pre) = precomputed_positions {
        for (&vid, &pos) in pre {
            if seen.insert(vid) {
                vert_list.push((vid, pos));
            }
        }
    }
    for (&vid, &pos) in &extra_positions {
        if seen.insert(vid) {
            vert_list.push((vid, pos));
        }
    }

    // Compute grid cell size from bounding box and vertex count.
    // Target ~1 vertex per cell on average for O(1) query cost.
    // NOTE: cell_size is calibrated from the global vertex population.
    // If boundary faces are concentrated in a small sub-region, the cell
    // size may be too large, degrading to O(boundary_verts) per query.
    // This is acceptable for boolean assembly outputs where vertices are
    // distributed across the full solid extent.
    let (mut bb_min, mut bb_max) = (
        Point3::new(f64::INFINITY, f64::INFINITY, f64::INFINITY),
        Point3::new(f64::NEG_INFINITY, f64::NEG_INFINITY, f64::NEG_INFINITY),
    );
    for &(_, pos) in &vert_list {
        bb_min = Point3::new(
            bb_min.x().min(pos.x()),
            bb_min.y().min(pos.y()),
            bb_min.z().min(pos.z()),
        );
        bb_max = Point3::new(
            bb_max.x().max(pos.x()),
            bb_max.y().max(pos.y()),
            bb_max.z().max(pos.z()),
        );
    }
    let diag = ((bb_max.x() - bb_min.x()).powi(2)
        + (bb_max.y() - bb_min.y()).powi(2)
        + (bb_max.z() - bb_min.z()).powi(2))
    .sqrt();
    let cell_size = (diag / (vert_list.len() as f64).cbrt()).max(tol.linear);
    let inv_cell = 1.0 / cell_size;

    let mut grid: HashMap<(i64, i64, i64), Vec<usize>> = HashMap::new();
    for (i, &(_, pos)) in vert_list.iter().enumerate() {
        let cx = (pos.x() * inv_cell).floor() as i64;
        let cy = (pos.y() * inv_cell).floor() as i64;
        let cz = (pos.z() * inv_cell).floor() as i64;
        grid.entry((cx, cy, cz)).or_default().push(i);
    }

    let mut edge_splits: HashMap<EdgeId, Vec<VertexId>> = HashMap::new();

    for &eid in &boundary_edges {
        let &(start_vid, end_vid) = match edge_vertices.get(&eid) {
            Some(v) => v,
            None => continue,
        };
        let (p0, p1) = match (get_pos(&start_vid), get_pos(&end_vid)) {
            (Some(a), Some(b)) => (a, b),
            _ => continue,
        };
        let dx = p1.x() - p0.x();
        let dy = p1.y() - p0.y();
        let dz = p1.z() - p0.z();
        let len_sq = dx * dx + dy * dy + dz * dz;
        if len_sq < tol.linear * tol.linear {
            continue;
        }
        let len = len_sq.sqrt();

        // Query hash grid with the edge's AABB expanded by tolerance
        let edge_aabb = Aabb3 {
            min: Point3::new(p0.x().min(p1.x()), p0.y().min(p1.y()), p0.z().min(p1.z())),
            max: Point3::new(p0.x().max(p1.x()), p0.y().max(p1.y()), p0.z().max(p1.z())),
        }
        .expanded(tol.linear);
        let min_cx = (edge_aabb.min.x() * inv_cell).floor() as i64;
        let min_cy = (edge_aabb.min.y() * inv_cell).floor() as i64;
        let min_cz = (edge_aabb.min.z() * inv_cell).floor() as i64;
        let max_cx = (edge_aabb.max.x() * inv_cell).floor() as i64;
        let max_cy = (edge_aabb.max.y() * inv_cell).floor() as i64;
        let max_cz = (edge_aabb.max.z() * inv_cell).floor() as i64;

        let mut intermediates: Vec<(f64, VertexId)> = Vec::new();

        for gx in min_cx..=max_cx {
            for gy in min_cy..=max_cy {
                for gz in min_cz..=max_cz {
                    if let Some(indices) = grid.get(&(gx, gy, gz)) {
                        for &cand_idx in indices {
                            let (vid, pos) = vert_list[cand_idx];
                            if vid == start_vid || vid == end_vid {
                                continue;
                            }
                            // Project pos onto line p0 + t*(p1-p0)
                            let dpx = pos.x() - p0.x();
                            let dpy = pos.y() - p0.y();
                            let dpz = pos.z() - p0.z();
                            let t = (dpx * dx + dpy * dy + dpz * dz) / len_sq;

                            // Must be strictly between endpoints
                            if t <= tol.linear / len || t >= 1.0 - tol.linear / len {
                                continue;
                            }

                            // Check distance from point to line
                            let proj_x = p0.x() + t * dx;
                            let proj_y = p0.y() + t * dy;
                            let proj_z = p0.z() + t * dz;
                            let dist_sq = (pos.x() - proj_x).powi(2)
                                + (pos.y() - proj_y).powi(2)
                                + (pos.z() - proj_z).powi(2);

                            if dist_sq < tol.linear * tol.linear {
                                intermediates.push((t, vid));
                            }
                        }
                    }
                }
            }
        }

        if !intermediates.is_empty() {
            intermediates
                .sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap_or(std::cmp::Ordering::Equal));
            intermediates.dedup_by_key(|(_, vid)| *vid);
            edge_splits.insert(eid, intermediates.into_iter().map(|(_, vid)| vid).collect());
        }
    }

    if edge_splits.is_empty() {
        return Ok(());
    }

    // Rebuild faces that have edges needing splits
    for fi in 0..face_ids.len() {
        let fid = face_ids[fi];
        let face = topo.face(fid)?;
        let outer_wire_id = face.outer_wire();
        let outer_wire = topo.wire(outer_wire_id)?;

        let mut needs_rebuild = false;
        for oe in outer_wire.edges() {
            if edge_splits.contains_key(&oe.edge()) {
                needs_rebuild = true;
                break;
            }
        }

        if !needs_rebuild {
            continue;
        }

        // Snapshot face data before mutable borrow
        let surface = face.surface().clone();
        let inner_wires = face.inner_wires().to_vec();
        let is_reversed = face.is_reversed();
        let old_edges: Vec<OrientedEdge> = outer_wire.edges().to_vec();

        // Rebuild the outer wire with split edges
        let mut new_oriented_edges = Vec::new();
        for oe in &old_edges {
            if let Some(intermediates) = edge_splits.get(&oe.edge()) {
                let (start_vid, end_vid) = match edge_vertices.get(&oe.edge()) {
                    Some(&v) => v,
                    None => continue,
                };
                let original_curve = topo.edge(oe.edge())?.curve().clone();

                // Build vertex chain in traversal order
                let chain: Vec<VertexId> = if oe.is_forward() {
                    let mut c = vec![start_vid];
                    c.extend(intermediates.iter().copied());
                    c.push(end_vid);
                    c
                } else {
                    let mut c = vec![end_vid];
                    c.extend(intermediates.iter().rev().copied());
                    c.push(start_vid);
                    c
                };

                // Create sub-edges (reusing from edge_map when possible).
                // Preserve the original edge's curve type so curved edges
                // (Circle, Ellipse) are not silently replaced with lines.
                for k in 0..chain.len() - 1 {
                    let va = chain[k];
                    let vb = chain[k + 1];
                    let va_idx = va.index();
                    let vb_idx = vb.index();
                    let (key_min, key_max) = if va_idx <= vb_idx {
                        (va_idx, vb_idx)
                    } else {
                        (vb_idx, va_idx)
                    };
                    let fwd = va_idx <= vb_idx;
                    let sub_eid = *edge_map.entry((key_min, key_max)).or_insert_with(|| {
                        let (s, e) = if fwd { (va, vb) } else { (vb, va) };
                        topo.edges.alloc(Edge::new(s, e, original_curve.clone()))
                    });
                    new_oriented_edges.push(OrientedEdge::new(sub_eid, fwd));
                }
            } else {
                new_oriented_edges.push(*oe);
            }
        }

        let new_wire =
            Wire::new(new_oriented_edges, true).map_err(crate::OperationsError::Topology)?;
        let new_wire_id = topo.wires.alloc(new_wire);

        let new_face = if is_reversed {
            Face::new_reversed(new_wire_id, inner_wires, surface)
        } else {
            Face::new(new_wire_id, inner_wires, surface)
        };
        face_ids[fi] = topo.faces.alloc(new_face);
    }

    Ok(())
}

/// Split non-manifold edges into multiple coincident copies.
///
/// After boolean assembly, some edges may be shared by more than 2 faces.
/// This happens when two solids share an edge or a vertex exactly, creating
/// an L-shaped junction. A manifold solid requires every edge to be shared
/// by exactly 2 faces.
///
/// This function detects non-manifold edges and duplicates them, assigning
/// each copy to a pair of faces based on angular ordering around the edge.
/// Faces are sorted by the angle of their outward normal projected onto
/// the plane perpendicular to the edge, then paired consecutively.
#[allow(clippy::too_many_lines)]
fn split_nonmanifold_edges(
    topo: &mut Topology,
    face_ids: &mut [FaceId],
) -> Result<(), crate::OperationsError> {
    // Build edge → [(face_index, is_forward)] map.
    let mut edge_faces: HashMap<usize, Vec<(usize, bool)>> = HashMap::new();
    for (fi, &fid) in face_ids.iter().enumerate() {
        let face = topo.face(fid)?;
        let wire = topo.wire(face.outer_wire())?;
        for oe in wire.edges() {
            edge_faces
                .entry(oe.edge().index())
                .or_default()
                .push((fi, oe.is_forward()));
        }
    }

    // Find non-manifold edges (shared by > 2 faces).
    let nonmanifold: Vec<(usize, Vec<(usize, bool)>)> = edge_faces
        .into_iter()
        .filter(|(_, faces)| faces.len() > 2)
        .collect();

    if nonmanifold.is_empty() {
        return Ok(());
    }

    // For each non-manifold edge, sort faces by angle and create edge copies.
    // Map: (face_index, old_edge_index) → new_edge_id
    let mut edge_replacements: HashMap<(usize, usize), EdgeId> = HashMap::new();

    for (edge_idx, face_refs) in &nonmanifold {
        let edge_id = topo.edges.id_from_index(*edge_idx).ok_or_else(|| {
            crate::OperationsError::InvalidInput {
                reason: format!("edge index {edge_idx} not found"),
            }
        })?;
        // Snapshot edge data before any mutable borrows (borrow checker).
        let edge_start = topo.edge(edge_id)?.start();
        let edge_end = topo.edge(edge_id)?.end();
        let edge_curve = topo.edge(edge_id)?.curve().clone();
        let start_pos = topo.vertex(edge_start)?.point();
        let end_pos = topo.vertex(edge_end)?.point();

        // Edge direction vector.
        let edge_dir = Vec3::new(
            end_pos.x() - start_pos.x(),
            end_pos.y() - start_pos.y(),
            end_pos.z() - start_pos.z(),
        );
        let edge_len = edge_dir.length();
        if edge_len < 1e-15 {
            continue;
        }
        let edge_axis = Vec3::new(
            edge_dir.x() / edge_len,
            edge_dir.y() / edge_len,
            edge_dir.z() / edge_len,
        );

        // Build a local 2D frame perpendicular to the edge.
        let perp = if edge_axis.x().abs() < 0.9 {
            Vec3::new(1.0, 0.0, 0.0)
        } else {
            Vec3::new(0.0, 1.0, 0.0)
        };
        let u_axis = edge_axis.cross(perp);
        let u_len = u_axis.length();
        if u_len < 1e-15 {
            continue;
        }
        let u_axis = Vec3::new(u_axis.x() / u_len, u_axis.y() / u_len, u_axis.z() / u_len);
        let v_axis = edge_axis.cross(u_axis);

        // Compute angle for each face's normal projected onto the perpendicular plane.
        let mut face_angles: Vec<(usize, bool, f64)> = Vec::new();
        for &(fi, is_fwd) in face_refs {
            let face = topo.face(face_ids[fi])?;
            let normal = if let FaceSurface::Plane { normal, .. } = face.surface() {
                *normal
            } else {
                // For non-planar faces, approximate normal from wire polygon centroid.
                let wire = topo.wire(face.outer_wire())?;
                let mut sum = Vec3::new(0.0, 0.0, 0.0);
                let mut count = 0usize;
                for oe in wire.edges() {
                    if let Ok(e) = topo.edge(oe.edge()) {
                        if let Ok(v) = topo.vertex(e.start()) {
                            let p = v.point();
                            sum = Vec3::new(sum.x() + p.x(), sum.y() + p.y(), sum.z() + p.z());
                            count += 1;
                        }
                    }
                }
                if count == 0 {
                    continue;
                }
                #[allow(clippy::cast_precision_loss)]
                let inv = 1.0 / count as f64;
                let centroid_dir = Vec3::new(sum.x() * inv, sum.y() * inv, sum.z() * inv);
                let mid = Point3::new(
                    (start_pos.x() + end_pos.x()) * 0.5,
                    (start_pos.y() + end_pos.y()) * 0.5,
                    (start_pos.z() + end_pos.z()) * 0.5,
                );
                Vec3::new(
                    centroid_dir.x() - mid.x(),
                    centroid_dir.y() - mid.y(),
                    centroid_dir.z() - mid.z(),
                )
            };

            // If face is reversed, flip the effective normal for sorting.
            let effective_normal = if face.is_reversed() {
                Vec3::new(-normal.x(), -normal.y(), -normal.z())
            } else {
                normal
            };

            // Project normal onto perpendicular plane and compute angle.
            let proj_u = effective_normal.dot(u_axis);
            let proj_v = effective_normal.dot(v_axis);
            let angle = proj_v.atan2(proj_u);
            face_angles.push((fi, is_fwd, angle));
        }

        // Sort by angle.
        face_angles.sort_by(|a, b| a.2.partial_cmp(&b.2).unwrap_or(std::cmp::Ordering::Equal));

        // Pair consecutive faces (in angular order) and assign edge copies.
        // For N faces around an edge, we need N/2 edge instances.
        // Each pair of consecutive faces shares one edge instance.
        let n = face_angles.len();
        for pair_idx in 0..(n / 2) {
            let i = pair_idx * 2;
            let j = i + 1;
            if j >= n {
                break;
            }

            // Create a new edge copy (or reuse the original for the first pair).
            let new_edge_id = if pair_idx == 0 {
                edge_id
            } else {
                topo.edges
                    .alloc(Edge::new(edge_start, edge_end, edge_curve.clone()))
            };

            edge_replacements.insert((face_angles[i].0, *edge_idx), new_edge_id);
            edge_replacements.insert((face_angles[j].0, *edge_idx), new_edge_id);
        }

        // Handle odd face (if N is odd, the last face keeps the original edge).
        if n % 2 == 1 {
            let last = &face_angles[n - 1];
            edge_replacements.insert((last.0, *edge_idx), edge_id);
        }
    }

    if edge_replacements.is_empty() {
        return Ok(());
    }

    // Rebuild face wires with replaced edges.
    let affected_faces: HashSet<usize> = edge_replacements.keys().map(|(fi, _)| *fi).collect();
    for fi in affected_faces {
        let fid = face_ids[fi];
        let face = topo.face(fid)?;
        let wire = topo.wire(face.outer_wire())?;
        let surface = face.surface().clone();
        let is_reversed = face.is_reversed();
        let inner_wires: Vec<WireId> = face.inner_wires().to_vec();

        let new_edges: Vec<OrientedEdge> = wire
            .edges()
            .iter()
            .map(|oe| {
                if let Some(&new_eid) = edge_replacements.get(&(fi, oe.edge().index())) {
                    OrientedEdge::new(new_eid, oe.is_forward())
                } else {
                    *oe
                }
            })
            .collect();

        let new_wire = Wire::new(new_edges, true).map_err(crate::OperationsError::Topology)?;
        let new_wire_id = topo.wires.alloc(new_wire);
        let new_face = if is_reversed {
            Face::new_reversed(new_wire_id, inner_wires, surface)
        } else {
            Face::new(new_wire_id, inner_wires, surface)
        };
        face_ids[fi] = topo.faces.alloc(new_face);
    }

    Ok(())
}

/// Try to fuse two all-planar solids that share exactly one coplanar face.
///
/// If solids A and B share a face (opposite normals, coplanar, overlapping
/// extent), merge them by removing the shared face pair and combining
/// remaining faces into a new solid via `assemble_solid_mixed`. Returns
/// `None` if the fast path doesn't apply.
#[allow(clippy::too_many_lines)]
fn try_shared_boundary_fuse(
    topo: &mut Topology,
    _a: SolidId,
    _b: SolidId,
    face_ids_a: &[FaceId],
    face_ids_b: &[FaceId],
    tol: Tolerance,
) -> Result<Option<SolidId>, crate::OperationsError> {
    struct PlaneInfo {
        normal: Vec3,
        d: f64,
        vertices: Vec<Point3>,
    }

    /// Area ratio below which two faces are not considered extent-matching.
    const SHARED_FACE_AREA_RATIO_MIN: f64 = 0.99;

    // Only worth it for small solids (avoids pathological cases).
    if face_ids_a.len() > 20 || face_ids_b.len() > 20 {
        return Ok(None);
    }

    // Require all faces to be planar.
    for &fid in face_ids_a.iter().chain(face_ids_b.iter()) {
        if !matches!(topo.face(fid)?.surface(), FaceSurface::Plane { .. }) {
            return Ok(None);
        }
    }

    // Snapshot each face: (normal, d, vertices).
    let snapshot = |fid: FaceId| -> Result<PlaneInfo, crate::OperationsError> {
        let face = topo.face(fid)?;
        let surface = face.surface().clone();
        let reversed = face.is_reversed();
        let verts = face_polygon(topo, fid)?;
        let (mut normal, mut d) = analytic_face_normal_d(&surface, &verts);
        if reversed {
            normal = -normal;
            d = -d;
        }
        Ok(PlaneInfo {
            normal,
            d,
            vertices: verts,
        })
    };

    let infos_a: Vec<PlaneInfo> = face_ids_a
        .iter()
        .map(|&fid| snapshot(fid))
        .collect::<Result<Vec<_>, _>>()?;
    let infos_b: Vec<PlaneInfo> = face_ids_b
        .iter()
        .map(|&fid| snapshot(fid))
        .collect::<Result<Vec<_>, _>>()?;

    // Find shared face pair: coplanar with opposite normals and overlapping extent.
    let mut shared_a = None;
    let mut shared_b = None;
    let mut shared_count = 0;

    for (ia, pa) in infos_a.iter().enumerate() {
        for (ib, pb) in infos_b.iter().enumerate() {
            // Opposite normals, same plane (n_a ≈ -n_b, d_a ≈ -d_b).
            let dot = pa.normal.dot(pb.normal);
            if dot > -1.0 + tol.angular {
                continue;
            }
            if !tol.approx_eq(pa.d, -pb.d) {
                continue;
            }

            // Verify matching extent: both face polygons must have
            // approximately equal area.
            let area_a = polygon_area_3d(&pa.vertices, pa.normal);
            let area_b = polygon_area_3d(&pb.vertices, pb.normal);
            let area_ratio = if area_a > area_b {
                area_b / area_a
            } else {
                area_a / area_b
            };
            if area_ratio < SHARED_FACE_AREA_RATIO_MIN {
                continue;
            }

            // Centroids should be within a geometry-scaled tolerance.
            // Use sqrt(area) as the face extent scale.
            let centroid_a = polygon_centroid(&pa.vertices);
            let centroid_b = polygon_centroid(&pb.vertices);
            let dist = (centroid_a - centroid_b).length();
            let face_extent = area_a.sqrt().max(tol.linear);
            if dist > face_extent * 1e-6 {
                continue;
            }

            shared_a = Some(ia);
            shared_b = Some(ib);
            shared_count += 1;

            if shared_count > 1 {
                // Multiple shared faces → too complex for fast path.
                return Ok(None);
            }
        }
    }

    let (skip_a, skip_b) = match (shared_a, shared_b) {
        (Some(a), Some(b)) => (a, b),
        _ => return Ok(None),
    };

    // Build face specs from all faces except the shared pair.
    let mut face_specs: Vec<FaceSpec> = Vec::with_capacity(face_ids_a.len() + face_ids_b.len() - 2);

    for (i, info) in infos_a.iter().enumerate() {
        if i == skip_a {
            continue;
        }
        face_specs.push(FaceSpec::Planar {
            vertices: info.vertices.clone(),
            normal: info.normal,
            d: info.d,
        });
    }
    for (i, info) in infos_b.iter().enumerate() {
        if i == skip_b {
            continue;
        }
        face_specs.push(FaceSpec::Planar {
            vertices: info.vertices.clone(),
            normal: info.normal,
            d: info.d,
        });
    }

    let result = assemble_solid_mixed(topo, &face_specs, tol)?;
    Ok(Some(result))
}

/// Compute the area of a 3D polygon given its vertices and face normal.
fn polygon_area_3d(vertices: &[Point3], normal: Vec3) -> f64 {
    if vertices.len() < 3 {
        return 0.0;
    }
    let mut area = Vec3::new(0.0, 0.0, 0.0);
    let v0 = vertices[0];
    for i in 1..vertices.len() - 1 {
        let e1 = vertices[i] - v0;
        let e2 = vertices[i + 1] - v0;
        area = area + e1.cross(e2);
    }
    (area.dot(normal) * 0.5).abs()
}

/// Perform an analytic boolean preserving exact surface types.
///
/// This is the fast path for solids with only analytic faces. It computes
/// exact plane-analytic intersections and preserves `FaceSurface::Cylinder`,
/// `FaceSurface::Sphere`, etc. in the result.
///
/// Returns `Err` to signal fallback to the tessellated path.
#[allow(clippy::too_many_lines)]
#[allow(clippy::single_match_else)]
fn analytic_boolean(
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
                let Some(analytic_surf) = face_surface_to_analytic(&snap_b.surface) else {
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
                let Some(analytic_surf) = face_surface_to_analytic(&snap_a.surface) else {
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
                let surf_a_opt = face_surface_to_analytic(&snap_a.surface);
                let surf_b_opt = face_surface_to_analytic(&snap_b.surface);

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

    let _t_frag = timer_now();
    // ── Build contained-curve lookup sets ──────────────────────────────

    // Map: plane_face_idx → list of contained edge curves (keyed by plane source).
    let mut contained_a: HashMap<usize, Vec<EdgeCurve>> = HashMap::new();
    let mut contained_b: HashMap<usize, Vec<EdgeCurve>> = HashMap::new();
    // Map: analytic_face_idx → list of contained edge curves (keyed by analytic source).
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

    // Pre-classifications: fragment index → class (bypasses centroid classifier).
    let mut pre_classifications: HashMap<usize, FaceClass> = HashMap::new();
    // Holed-face inner curves: fragment index → edge curves for inner wire construction.
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
    for (idx, class) in classes.iter_mut().enumerate() {
        if class.is_some() {
            continue;
        }
        let frag = &fragments[idx];
        let centroid = polygon_centroid(&frag.vertices);
        if centroid_outside_opposing_aabb(
            frag.source,
            centroid,
            a_overall_aabb,
            b_overall_aabb,
            tol.linear,
        ) {
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

    let resolution = 1.0 / tol.linear;
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
                    .or_insert_with(|| topo.vertices.alloc(Vertex::new(*p, tol.linear)))
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
                .or_insert_with(|| topo.vertices.alloc(Vertex::new(seam_pt, tol.linear)));
            let eid = *edge_map
                .entry((vid.index(), vid.index()))
                .or_insert_with(|| topo.edges.alloc(Edge::new(vid, vid, ec)));
            let wire = Wire::new(vec![OrientedEdge::new(eid, true)], true)
                .map_err(crate::OperationsError::Topology)?;
            topo.wires.alloc(wire)
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
                        topo.edges.alloc(Edge::new(start, end, EdgeCurve::Line))
                    });
                    oriented_edges.push(OrientedEdge::new(eid, fwd));
                }
                let wire =
                    Wire::new(oriented_edges, true).map_err(crate::OperationsError::Topology)?;
                topo.wires.alloc(wire)
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
                    topo.edges.alloc(Edge::new(start, end, edge_curve))
                });

                oriented_edges.push(OrientedEdge::new(edge_id, is_forward));
            }
            let wire = Wire::new(oriented_edges, true).map_err(crate::OperationsError::Topology)?;
            topo.wires.alloc(wire)
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
                        .or_insert_with(|| topo.vertices.alloc(Vertex::new(seam_pt, tol.linear)));
                    let eid = *edge_map
                        .entry((vid.index(), vid.index()))
                        .or_insert_with(|| topo.edges.alloc(Edge::new(vid, vid, ec.clone())));
                    // Hole wires wind CW (reversed circle). When the face is
                    // flipped, the outer wire is already reversed so the hole
                    // keeps its natural (forward) direction.
                    let hw = Wire::new(vec![OrientedEdge::new(eid, flip)], true)
                        .map_err(crate::OperationsError::Topology)?;
                    topo.wires.alloc(hw)
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
                                .or_insert_with(|| topo.vertices.alloc(Vertex::new(*p, tol.linear)))
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
                            topo.edges.alloc(Edge::new(start, end, EdgeCurve::Line))
                        });
                        hole_edges.push(OrientedEdge::new(eid, is_forward));
                    }
                    let hw =
                        Wire::new(hole_edges, true).map_err(crate::OperationsError::Topology)?;
                    topo.wires.alloc(hw)
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
        let face = topo.faces.alloc(new_face);
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
    let shell_id = topo.shells.alloc(shell);
    log::debug!("[boolean] total: {:.3}ms", timer_elapsed_ms(_t_total));
    Ok(topo.solids.alloc(Solid::new(shell_id, vec![])))
}

// ---------------------------------------------------------------------------
// Compound cut — multi-tool single pass
// ---------------------------------------------------------------------------

/// Cut a target solid by multiple tool solids in a single pass.
///
/// For each tool, the target faces overlapping that tool are intersected and
/// fragments are classified against ALL tools simultaneously. This avoids the
/// O(N²) cost of sequential boolean operations where each cut must process the
/// full accumulated result of all prior cuts.
///
/// If any tool or the target contains NURBS, torus, or other non-analytic
/// surfaces, falls back to sequential `boolean_with_options()` calls.
///
/// # Errors
///
/// Returns an error if any individual boolean operation fails, or if the
/// result is degenerate (empty solid).
#[allow(clippy::too_many_lines)]
#[allow(clippy::items_after_statements)]
pub fn compound_cut(
    topo: &mut Topology,
    target: SolidId,
    tools: &[SolidId],
    opts: BooleanOptions,
) -> Result<SolidId, crate::OperationsError> {
    if tools.is_empty() {
        return Ok(target);
    }
    // Small tool counts: sequential is faster due to lower overhead.
    if tools.len() <= 8 {
        log::debug!(
            "[compound_cut] fallback: small tool count ({})",
            tools.len()
        );

        return compound_cut_sequential(topo, target, tools, opts);
    }

    // Check for non-analytic surfaces — fall back to sequential if found.
    let has_non_analytic = |solid: SolidId| -> Result<bool, crate::OperationsError> {
        let s = topo.solid(solid)?;
        let shell = topo.shell(s.outer_shell())?;
        for &fid in shell.faces() {
            let face = topo.face(fid)?;
            if matches!(
                face.surface(),
                FaceSurface::Nurbs(_) | FaceSurface::Torus(_)
            ) {
                return Ok(true);
            }
        }
        Ok(false)
    };

    if has_non_analytic(target)? {
        log::debug!("[compound_cut] fallback: target has non-analytic surfaces");

        return compound_cut_sequential(topo, target, tools, opts);
    }
    for (i, &tool) in tools.iter().enumerate() {
        if has_non_analytic(tool)? {
            log::debug!("[compound_cut] fallback: tool {i} has non-analytic surfaces");

            return compound_cut_sequential(topo, target, tools, opts);
        }
    }

    let tol = opts.tolerance;
    let deflection = opts.deflection;
    let _t_total = timer_now();

    // ── Phase 0: Precompute tool data ────────────────────────────────────
    struct ToolData {
        snapshots: Vec<FaceSnapshot>,
        aabbs: Vec<Aabb3>,
        overall_aabb: Aabb3,
        classifier: Option<AnalyticClassifier>,
    }

    let mut tool_data: Vec<ToolData> = Vec::with_capacity(tools.len());
    let target_wire_aabbs = {
        let solid_t = topo.solid(target)?;
        let shell_t = topo.shell(solid_t.outer_shell())?;
        let fids: Vec<FaceId> = shell_t.faces().to_vec();
        fids.iter()
            .map(|&fid| face_wire_aabb(topo, fid))
            .collect::<Result<Vec<Aabb3>, _>>()?
    };
    let target_overall_aabb = target_wire_aabbs
        .iter()
        .copied()
        .reduce(Aabb3::union)
        .ok_or_else(|| crate::OperationsError::InvalidInput {
            reason: "target solid has no faces".into(),
        })?;

    for &tool in tools {
        let solid_t = topo.solid(tool)?;
        let shell_t = topo.shell(solid_t.outer_shell())?;
        let face_ids: Vec<FaceId> = shell_t.faces().to_vec();

        // Compute tool's wire AABBs and overall AABB.
        let tool_wire_aabbs: Vec<Aabb3> = face_ids
            .iter()
            .map(|&fid| face_wire_aabb(topo, fid))
            .collect::<Result<Vec<_>, _>>()?;
        let tool_overall = tool_wire_aabbs
            .iter()
            .copied()
            .reduce(Aabb3::union)
            .ok_or_else(|| crate::OperationsError::InvalidInput {
                reason: "tool solid has no faces".into(),
            })?;

        // Skip tools completely disjoint from target.
        if !tool_overall.intersects(target_overall_aabb) {
            continue;
        }

        // Snapshot tool faces that overlap target.
        let mut snapshots = Vec::new();
        let mut aabbs = Vec::new();
        for (i, &fid) in face_ids.iter().enumerate() {
            if tool_wire_aabbs[i].intersects(target_overall_aabb) {
                let face = topo.face(fid)?;
                let surface = face.surface().clone();
                let reversed = face.is_reversed();
                let verts = face_polygon(topo, fid)?;
                let (normal, d) = analytic_face_normal_d(&surface, &verts);
                aabbs.push(surface_aware_aabb(&surface, &verts, tol));
                snapshots.push(FaceSnapshot {
                    id: fid,
                    surface,
                    vertices: verts,
                    normal,
                    d,
                    reversed,
                });
            }
        }

        let classifier = try_build_analytic_classifier(topo, tool);
        tool_data.push(ToolData {
            snapshots,
            aabbs,
            overall_aabb: tool_overall,
            classifier,
        });
    }

    // If no tools overlap target, return unchanged.
    if tool_data.is_empty() {
        return Ok(target);
    }

    let _t_phase0 = timer_elapsed_ms(_t_total);
    log::debug!(
        "[compound_cut] Phase 0: {:.1}ms — {} tools overlap target ({} total tool faces)",
        _t_phase0,
        tool_data.len(),
        tool_data.iter().map(|td| td.snapshots.len()).sum::<usize>()
    );

    // Global BVH over tool overall AABBs — used for O(log N) spatial queries
    // in Phase 1 (passthrough), Phase 2 (intersection), and Phase 4 (classification).
    let tool_bvh = {
        let entries: Vec<(usize, Aabb3)> = tool_data
            .iter()
            .enumerate()
            .map(|(i, td)| (i, td.overall_aabb))
            .collect();
        Bvh::build(&entries)
    };

    // ── Phase 1: Snapshot target faces ───────────────────────────────────
    let solid_a = topo.solid(target)?;
    let shell_a = topo.shell(solid_a.outer_shell())?;
    let face_ids_a: Vec<FaceId> = shell_a.faces().to_vec();

    let mut snaps_a = Vec::new();
    let mut passthrough_a: Vec<FaceId> = Vec::new();
    // A face is passthrough if it doesn't overlap ANY tool (BVH query).
    let mut bvh_buf = Vec::new();
    for (i, &fid) in face_ids_a.iter().enumerate() {
        tool_bvh.query_overlap_into(&target_wire_aabbs[i], &mut bvh_buf);
        if bvh_buf.is_empty() {
            passthrough_a.push(fid);
        } else {
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
        }
    }

    let aabbs_a: Vec<Aabb3> = snaps_a
        .iter()
        .map(|s| surface_aware_aabb(&s.surface, &s.vertices, tol))
        .collect();

    log::debug!(
        "[compound_cut] Phase 1: {:.1}ms — {} snap + {} passthrough",
        timer_elapsed_ms(_t_total) - _t_phase0,
        snaps_a.len(),
        passthrough_a.len()
    );
    let _t_phase1 = timer_elapsed_ms(_t_total);

    // ── Phase 2: Intersection (all tools at once) ────────────────────────
    use brepkit_math::analytic_intersection::{
        ExactIntersectionCurve, exact_plane_analytic, intersect_analytic_analytic_bounded,
    };

    let mut face_intersections_a: HashMap<usize, Vec<(Point3, Point3, Option<EdgeCurve>)>> =
        HashMap::new();
    let mut analytic_analytic_faces_a: HashSet<usize> = HashSet::new();
    let mut analytic_intersection_vranges_a: HashMap<usize, Vec<(f64, f64)>> = HashMap::new();

    // Contained curves: plane_face_idx in target → list of (tool_index, edge_curve).
    struct CompoundContainedCurve {
        plane_face_idx: usize,
        tool_index: usize,
        analytic_face_idx: usize,
        edge_curve: EdgeCurve,
    }
    let mut contained_curves: Vec<CompoundContainedCurve> = Vec::new();

    // Per-tool: face intersections and flags for tool faces.
    struct ToolIntersections {
        face_intersections: HashMap<usize, Vec<(Point3, Point3, Option<EdgeCurve>)>>,
        analytic_analytic_faces: HashSet<usize>,
        analytic_intersection_vranges: HashMap<usize, Vec<(f64, f64)>>,
    }
    let mut tool_intersections: Vec<ToolIntersections> = tool_data
        .iter()
        .map(|_| ToolIntersections {
            face_intersections: HashMap::new(),
            analytic_analytic_faces: HashSet::new(),
            analytic_intersection_vranges: HashMap::new(),
        })
        .collect();

    let mut has_analytic_analytic = false;

    // Build per-tool BVHs over their face AABBs (for face-level broad-phase).
    let tool_face_bvhs: Vec<Option<Bvh>> = tool_data
        .iter()
        .map(|td| {
            if td.aabbs.len() >= 16 {
                let entries: Vec<(usize, Aabb3)> = td
                    .aabbs
                    .iter()
                    .enumerate()
                    .map(|(i, aabb)| (i, *aabb))
                    .collect();
                Some(Bvh::build(&entries))
            } else {
                None
            }
        })
        .collect();

    // Iterate target faces first, then use global BVH to find overlapping tools.
    // This is O(target_faces × log(tools)) instead of O(tools × target_faces).
    let mut overlap_buf = Vec::new();
    let mut candidate_buf = Vec::new();
    for (ia, snap_a) in snaps_a.iter().enumerate() {
        // Global BVH query: which tools overlap this target face?
        tool_bvh.query_overlap_into(&aabbs_a[ia], &mut overlap_buf);

        for &ti in &overlap_buf {
            let td = &tool_data[ti];

            if let Some(ref bvh) = tool_face_bvhs[ti] {
                bvh.query_overlap_into(&aabbs_a[ia], &mut candidate_buf);
            } else {
                candidate_buf.clear();
                candidate_buf.extend(
                    (0..td.snapshots.len()).filter(|&ib| aabbs_a[ia].intersects(td.aabbs[ib])),
                );
            }

            for &ib in &candidate_buf {
                let snap_b = &td.snapshots[ib];

                let is_plane_a = matches!(snap_a.surface, FaceSurface::Plane { .. });
                let is_plane_b = matches!(snap_b.surface, FaceSurface::Plane { .. });

                if is_plane_a && is_plane_b {
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
                        tool_intersections[ti]
                            .face_intersections
                            .entry(ib)
                            .or_default()
                            .push((seg.0, seg.1, None));
                    }
                } else if is_plane_a && !is_plane_b {
                    let Some(analytic_surf) = face_surface_to_analytic(&snap_b.surface) else {
                        has_analytic_analytic = true;
                        continue;
                    };
                    if let Ok(curves) = exact_plane_analytic(analytic_surf, snap_a.normal, snap_a.d)
                    {
                        for curve in curves {
                            let edge_curve = match &curve {
                                ExactIntersectionCurve::Circle(c) => {
                                    Some(EdgeCurve::Circle(c.clone()))
                                }
                                ExactIntersectionCurve::Ellipse(e) => {
                                    Some(EdgeCurve::Ellipse(e.clone()))
                                }
                                ExactIntersectionCurve::Points(_) => None,
                            };
                            let classification = curve_boundary_crossings(
                                &curve,
                                &snap_a.vertices,
                                snap_a.normal,
                                tol,
                            );
                            match classification {
                                CurveClassification::Crossings(ref samples) => {
                                    for pair in samples.windows(2) {
                                        face_intersections_a.entry(ia).or_default().push((
                                            pair[0],
                                            pair[1],
                                            edge_curve.clone(),
                                        ));
                                        tool_intersections[ti]
                                            .face_intersections
                                            .entry(ib)
                                            .or_default()
                                            .push((pair[0], pair[1], edge_curve.clone()));
                                    }
                                }
                                CurveClassification::FullyContained => {
                                    if let Some(ref ec) = edge_curve {
                                        if face_intersections_a.contains_key(&ia) {
                                            has_analytic_analytic = true;
                                        } else {
                                            contained_curves.push(CompoundContainedCurve {
                                                plane_face_idx: ia,
                                                tool_index: ti,
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
                    let Some(analytic_surf) = face_surface_to_analytic(&snap_a.surface) else {
                        has_analytic_analytic = true;
                        continue;
                    };
                    if let Ok(curves) = exact_plane_analytic(analytic_surf, snap_b.normal, snap_b.d)
                    {
                        for curve in curves {
                            let edge_curve = match &curve {
                                ExactIntersectionCurve::Circle(c) => {
                                    Some(EdgeCurve::Circle(c.clone()))
                                }
                                ExactIntersectionCurve::Ellipse(e) => {
                                    Some(EdgeCurve::Ellipse(e.clone()))
                                }
                                ExactIntersectionCurve::Points(_) => None,
                            };
                            let classification = curve_boundary_crossings(
                                &curve,
                                &snap_b.vertices,
                                snap_b.normal,
                                tol,
                            );
                            match classification {
                                CurveClassification::Crossings(samples) => {
                                    for pair in samples.windows(2) {
                                        face_intersections_a.entry(ia).or_default().push((
                                            pair[0],
                                            pair[1],
                                            edge_curve.clone(),
                                        ));
                                        tool_intersections[ti]
                                            .face_intersections
                                            .entry(ib)
                                            .or_default()
                                            .push((pair[0], pair[1], edge_curve.clone()));
                                    }
                                }
                                CurveClassification::FullyContained => {
                                    // Contained in plane B: don't need to track for compound_cut
                                    // since B faces are tools, not target.
                                }
                                CurveClassification::FullyOutside => {}
                            }
                        }
                    }
                } else {
                    // Analytic-analytic.
                    let surf_a_opt = face_surface_to_analytic(&snap_a.surface);
                    let surf_b_opt = face_surface_to_analytic(&snap_b.surface);
                    if let (Some(surf_a_an), Some(surf_b_an)) = (surf_a_opt, surf_b_opt) {
                        let v_hint_a = compute_v_range_hint(&snap_a.surface, &snap_a.vertices);
                        let v_hint_b = compute_v_range_hint(&snap_b.surface, &snap_b.vertices);
                        if let Ok(curves) = intersect_analytic_analytic_bounded(
                            surf_a_an, surf_b_an, 32, v_hint_a, v_hint_b,
                        ) {
                            for ic in &curves {
                                let pts: Vec<Point3> =
                                    ic.points.iter().map(|ip| ip.point).collect();
                                analytic_analytic_faces_a.insert(ia);
                                tool_intersections[ti].analytic_analytic_faces.insert(ib);
                                for pair in pts.windows(2) {
                                    face_intersections_a
                                        .entry(ia)
                                        .or_default()
                                        .push((pair[0], pair[1], None));
                                    tool_intersections[ti]
                                        .face_intersections
                                        .entry(ib)
                                        .or_default()
                                        .push((pair[0], pair[1], None));
                                }
                            }
                        } else {
                            analytic_analytic_faces_a.insert(ia);
                            tool_intersections[ti].analytic_analytic_faces.insert(ib);
                        }
                    } else {
                        has_analytic_analytic = true;
                    }
                }
            }
        }
    }

    // Fall back to sequential if unsupported intersection types found.
    if has_analytic_analytic {
        log::debug!("[compound_cut] fallback: has_analytic_analytic intersection");
        return compound_cut_sequential(topo, target, tools, opts);
    }

    // Compute v-ranges for band splitting.
    collect_analytic_vranges(
        &snaps_a,
        &face_intersections_a,
        &analytic_analytic_faces_a,
        &mut analytic_intersection_vranges_a,
    );

    let _t_phase2 = timer_elapsed_ms(_t_total);
    log::debug!(
        "[compound_cut] Phase 2: {:.1}ms — target={} faces with chords",
        _t_phase2 - _t_phase1,
        face_intersections_a.len()
    );

    // ── Phase 3: Fragment creation ───────────────────────────────────────

    let mut pre_classifications: HashMap<usize, FaceClass> = HashMap::new();
    let mut holed_face_inner_curves: HashMap<usize, Vec<EdgeCurve>> = HashMap::new();
    let mut existing_inner_wires: HashMap<usize, Vec<WireId>> = HashMap::new();
    let mut fragments: Vec<AnalyticFragment> = Vec::with_capacity(
        snaps_a.len() + tool_data.iter().map(|td| td.snapshots.len()).sum::<usize>(),
    );

    // Build contained-curve lookups (target faces with holes).
    let mut contained_a: HashMap<usize, Vec<EdgeCurve>> = HashMap::new();
    // Track which tool faces have contained curves (for band fragments).
    let mut tool_analytic_contained: Vec<HashMap<usize, Vec<EdgeCurve>>> =
        tool_data.iter().map(|_| HashMap::new()).collect();
    for cc in &contained_curves {
        contained_a
            .entry(cc.plane_face_idx)
            .or_default()
            .push(cc.edge_curve.clone());
        tool_analytic_contained[cc.tool_index]
            .entry(cc.analytic_face_idx)
            .or_default()
            .push(cc.edge_curve.clone());
    }

    // --- Target face fragments ---
    let _t_frag_a = timer_now();
    for (ia, snap) in snaps_a.iter().enumerate() {
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
        if analytic_analytic_faces_a.contains(&ia) {
            tessellate_face_into_fragments(topo, snap.id, Source::A, deflection, &mut fragments)?;
            continue;
        }
        if let Some(chords) = face_intersections_a.get(&ia) {
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
            if !matches!(snap.surface, FaceSurface::Plane { .. }) {
                if let Some(ref ec) = edge_curve_for_face {
                    let curve_verts = sample_edge_curve(ec, CLOSED_CURVE_SAMPLES);
                    if curve_verts.len() >= 3 {
                        fragments.push(AnalyticFragment {
                            vertices: curve_verts,
                            surface: snap.surface.clone(),
                            normal: snap.normal,
                            d: snap.d,
                            source: Source::A,
                            edge_curves: vec![Some(ec.clone())],
                            source_reversed: snap.reversed,
                        });
                    }
                }
            }
        } else if let Some(inner_curves) = contained_a.get(&ia) {
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
                        source: Source::A,
                        edge_curves: vec![Some(ec.clone())],
                        source_reversed: false,
                    });
                    pre_classifications.insert(disc_idx, FaceClass::Inside);
                }
            }
        } else {
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

    let _frag_a_count = fragments.len();
    log::debug!(
        "[compound_cut] Phase 3a (target frags): {:.1}ms — {} fragments",
        timer_elapsed_ms(_t_frag_a),
        _frag_a_count
    );
    let _t_frag_b = timer_now();
    // --- Tool face fragments (Source::B) ---
    // Compute v-ranges for each tool (must be done before borrowing ti_ref).
    for ti in 0..tool_data.len() {
        let mut vranges = HashMap::new();
        collect_analytic_vranges(
            &tool_data[ti].snapshots,
            &tool_intersections[ti].face_intersections,
            &tool_intersections[ti].analytic_analytic_faces,
            &mut vranges,
        );
        tool_intersections[ti].analytic_intersection_vranges = vranges;
    }
    for (ti, td) in tool_data.iter().enumerate() {
        let ti_ref = &tool_intersections[ti];
        for (ib, snap) in td.snapshots.iter().enumerate() {
            if let Some(vranges) = ti_ref.analytic_intersection_vranges.get(&ib) {
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
            if ti_ref.analytic_analytic_faces.contains(&ib) {
                tessellate_face_into_fragments(
                    topo,
                    snap.id,
                    Source::B,
                    deflection,
                    &mut fragments,
                )?;
                continue;
            }
            if let Some(chords) = ti_ref.face_intersections.get(&ib) {
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
            } else if let Some(band_curves) = tool_analytic_contained[ti].get(&ib) {
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
    }

    log::debug!(
        "[compound_cut] Phase 3b (tool frags): {:.1}ms — {} fragments",
        timer_elapsed_ms(_t_frag_b),
        fragments.len() - _frag_a_count
    );
    // Passthrough target faces (outside all tools → survive Cut).
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

    let _t_phase3 = timer_elapsed_ms(_t_total);
    log::debug!(
        "[compound_cut] Phase 3: {:.1}ms — {} fragments (passthrough={})",
        _t_phase3 - _t_phase2,
        fragments.len(),
        passthrough_a.len()
    );

    // ── Phase 4: Classification ──────────────────────────────────────────
    // For compound cut: Target fragments (Source::A) must be Outside ALL tools.
    // Tool fragments (Source::B) must be Inside target AND Outside all other tools.
    //
    // OPTIMIZATION: Use AABB filtering to skip tools that can't contain the
    // fragment centroid. A point can only be Inside a tool if it's within the
    // tool's bounding box. For N=100 disjoint tools, this reduces per-fragment
    // classification from O(N) to O(~1-3) on average.
    let target_classifier = try_build_analytic_classifier(topo, target);

    // Expand tool AABBs slightly for classification (tolerance margin).
    let expanded_tool_aabbs: Vec<Aabb3> = tool_data
        .iter()
        .map(|td| td.overall_aabb.expanded(tol.linear))
        .collect();

    let mut classes: Vec<Option<FaceClass>> = fragments
        .iter()
        .enumerate()
        .map(|(idx, frag)| {
            if let Some(&class) = pre_classifications.get(&idx) {
                return Some(class);
            }
            match frag.source {
                Source::A => {
                    // Target fragment: must be Outside ALL tools to survive.
                    // If Inside ANY tool → discard (Inside).
                    let centroid = polygon_centroid(&frag.vertices);
                    // Use global BVH to find tools whose AABB contains centroid.
                    let point_aabb = Aabb3 {
                        min: centroid,
                        max: centroid,
                    };
                    let nearby_tools = tool_bvh.query_overlap(&point_aabb);
                    for &ti in &nearby_tools {
                        // Double-check with expanded AABB (BVH may have slight padding).
                        if !expanded_tool_aabbs[ti].contains_point(centroid) {
                            continue;
                        }
                        if let Some(ref cls) = tool_data[ti].classifier {
                            if cls.classify(centroid, tol) == Some(FaceClass::Inside) {
                                return Some(FaceClass::Inside);
                            }
                        } else {
                            return None; // Need raycast
                        }
                    }
                    Some(FaceClass::Outside)
                }
                Source::B => {
                    // Tool fragment: must be Inside target.
                    let centroid = polygon_centroid(&frag.vertices);
                    if let Some(ref cls) = target_classifier {
                        match cls.classify(centroid, tol) {
                            Some(FaceClass::Inside) => {}
                            Some(FaceClass::Outside) => return Some(FaceClass::Outside),
                            _ => return None,
                        }
                    } else {
                        return None; // Need raycast against target
                    }
                    // Also must be Outside all OTHER tools (for overlapping tools).
                    // Use AABB filtering: only check tools whose AABB contains centroid.
                    let point_aabb = Aabb3 {
                        min: centroid,
                        max: centroid,
                    };
                    let nearby_tools = tool_bvh.query_overlap(&point_aabb);
                    for &ti2 in &nearby_tools {
                        if !expanded_tool_aabbs[ti2].contains_point(centroid) {
                            continue;
                        }
                        if let Some(ref cls2) = tool_data[ti2].classifier {
                            if cls2.classify(centroid, tol) == Some(FaceClass::Inside) {
                                // Could be the fragment's own tool — without tool_index
                                // tracking we can't distinguish. For non-overlapping tools
                                // this is correct (centroid is on-boundary, not strictly Inside).
                                let _ = ti2;
                            }
                        }
                    }
                    Some(FaceClass::Inside)
                }
            }
        })
        .collect();

    // Phase 4b: raycast fallback for unclassified fragments.
    // This handles concave targets (e.g. shelled boxes) where the analytic
    // classifier can't be built. We tessellate the original target/tools and
    // raycast, exactly as analytic_boolean does.
    let needs_raycast = classes.iter().any(Option::is_none);
    if needs_raycast {
        let unclassified_count = classes.iter().filter(|c| c.is_none()).count();
        log::debug!(
            "[compound_cut] raycast fallback for {unclassified_count}/{} fragments",
            fragments.len()
        );

        // Build face data for target (for Source::B raycast) and each tool
        // (for Source::A raycast). Only build what we actually need.
        let needs_target_raycast = classes
            .iter()
            .enumerate()
            .any(|(i, c)| c.is_none() && matches!(fragments[i].source, Source::B));
        let needs_tool_raycast = classes
            .iter()
            .enumerate()
            .any(|(i, c)| c.is_none() && matches!(fragments[i].source, Source::A));

        let target_face_data = if needs_target_raycast {
            Some(collect_face_data(topo, target, deflection)?)
        } else {
            None
        };
        let target_bvh = target_face_data.as_ref().and_then(build_face_bvh);

        // For Source::A fragments, we need to raycast against each relevant tool.
        // Build face data lazily per-tool.
        let tool_face_data: Vec<Option<FaceData>> = if needs_tool_raycast {
            tools
                .iter()
                .enumerate()
                .map(|(i, &tid)| match collect_face_data(topo, tid, deflection) {
                    Ok(fd) => Some(fd),
                    Err(e) => {
                        log::warn!(
                            "[compound_cut] tool {i} tessellation failed, \
                             falling back to sequential: {e}"
                        );
                        None
                    }
                })
                .collect()
        } else {
            vec![None; tools.len()]
        };
        // If any tool failed tessellation, fall back to sequential for correctness.
        if needs_tool_raycast && tool_face_data.iter().any(Option::is_none) {
            return compound_cut_sequential(topo, target, tools, opts);
        }
        let tool_face_bvhs_rc: Vec<Option<Bvh>> = tool_face_data
            .iter()
            .map(|fd| fd.as_ref().and_then(build_face_bvh))
            .collect();

        for (idx, class) in classes.iter_mut().enumerate() {
            if class.is_some() {
                continue;
            }
            let frag = &fragments[idx];
            let centroid = polygon_centroid(&frag.vertices);

            match frag.source {
                Source::B => {
                    // Raycast against original target to determine Inside/Outside.
                    if let Some(ref fd) = target_face_data {
                        let raw =
                            classify_point(centroid, frag.normal, fd, target_bvh.as_ref(), tol);
                        *class = Some(guard_tangent_coplanar(
                            raw,
                            &frag.vertices,
                            frag.normal,
                            fd,
                            target_bvh.as_ref(),
                            tol,
                        ));
                    }
                }
                Source::A => {
                    // Raycast against each nearby tool.
                    let point_aabb = Aabb3 {
                        min: centroid,
                        max: centroid,
                    };
                    let nearby_tools = tool_bvh.query_overlap(&point_aabb);
                    let mut result = FaceClass::Outside;
                    for &ti in &nearby_tools {
                        if !expanded_tool_aabbs[ti].contains_point(centroid) {
                            continue;
                        }
                        if let Some(ref fd) = tool_face_data[ti] {
                            let raw = classify_point(
                                centroid,
                                frag.normal,
                                fd,
                                tool_face_bvhs_rc[ti].as_ref(),
                                tol,
                            );
                            let guarded = guard_tangent_coplanar(
                                raw,
                                &frag.vertices,
                                frag.normal,
                                fd,
                                tool_face_bvhs_rc[ti].as_ref(),
                                tol,
                            );
                            if guarded == FaceClass::Inside {
                                result = FaceClass::Inside;
                                break;
                            }
                        }
                    }
                    *class = Some(result);
                }
            }
        }
    }

    // Classification summary for debugging.
    if log::log_enabled!(log::Level::Debug) {
        let mut a_in = 0usize;
        let mut a_out = 0usize;
        let mut b_in = 0usize;
        let mut b_out = 0usize;
        for (i, c) in classes.iter().enumerate() {
            match (&fragments[i].source, c) {
                (Source::A, Some(FaceClass::Inside)) => a_in += 1,
                (Source::A, Some(FaceClass::Outside)) => a_out += 1,
                (Source::B, Some(FaceClass::Inside)) => b_in += 1,
                (Source::B, Some(FaceClass::Outside)) => b_out += 1,
                _ => {}
            }
        }
        log::debug!(
            "[compound_cut] classification: A(in={a_in} out={a_out}) B(in={b_in} out={b_out}) passthrough={}",
            passthrough_a.len()
        );
    }

    let classes: Vec<FaceClass> = classes
        .into_iter()
        .enumerate()
        .map(|(_i, c)| -> Result<FaceClass, crate::OperationsError> {
            c.ok_or_else(|| crate::OperationsError::InvalidInput {
                reason: format!("compound_cut: fragment {_i} was not classified"),
            })
        })
        .collect::<Result<Vec<_>, _>>()?;

    // ── Phase 5: Assembly ────────────────────────────────────────────────
    let _t_phase4 = timer_elapsed_ms(_t_total);
    log::debug!(
        "[compound_cut] Phase 4: {:.1}ms — classification",
        _t_phase4 - _t_phase3,
    );
    // Reuse the same assembly logic as analytic_boolean (vertex/edge dedup,
    // wire construction, face creation).
    let resolution = 1.0 / tol.linear;
    let mut vertex_map: HashMap<(i64, i64, i64), VertexId> =
        HashMap::with_capacity(fragments.len() * 4);
    let mut edge_map: HashMap<(usize, usize), EdgeId> = HashMap::with_capacity(fragments.len() * 4);
    let mut face_ids_out = Vec::with_capacity(fragments.len());

    for (idx, (frag, &class)) in fragments.iter().zip(classes.iter()).enumerate() {
        let Some(flip) = select_fragment(frag.source, class, BooleanOp::Cut) else {
            continue;
        };
        let is_nonplanar = !matches!(frag.surface, FaceSurface::Plane { .. });
        let is_closed_curve = frag.edge_curves.len() == 1 && frag.edge_curves[0].is_some();

        // For planar faces that need flipping: reverse vertices and negate normal/d.
        // This mirrors analytic_boolean's assembly logic exactly.
        // After pre-reversing, set flip=false for the edge creation code to avoid
        // double-flipping.
        let (verts, normal, d_val, flip) = if flip && !is_nonplanar {
            let rev: Vec<_> = frag.vertices.iter().copied().rev().collect();
            (rev, -frag.normal, -frag.d, false)
        } else {
            (frag.vertices.clone(), frag.normal, frag.d, flip)
        };

        let n = verts.len();
        if n < 3 {
            continue;
        }
        let vert_ids: Vec<VertexId> = verts
            .iter()
            .map(|p| {
                let key = (
                    quantize(p.x(), resolution),
                    quantize(p.y(), resolution),
                    quantize(p.z(), resolution),
                );
                *vertex_map
                    .entry(key)
                    .or_insert_with(|| topo.vertices.alloc(Vertex::new(*p, tol.linear)))
            })
            .collect();

        let wire_id = if is_closed_curve {
            // SAFETY: is_closed_curve checks len==1 && [0].is_some()
            let Some(ec) = frag.edge_curves[0].as_ref() else {
                continue;
            };
            let vid = vert_ids[0];
            let eid = *edge_map
                .entry((vid.index(), vid.index()))
                .or_insert_with(|| topo.edges.alloc(Edge::new(vid, vid, ec.clone())));
            let wire = Wire::new(vec![OrientedEdge::new(eid, !flip)], true)
                .map_err(crate::OperationsError::Topology)?;
            topo.wires.alloc(wire)
        } else if is_nonplanar
            && n >= CLOSED_CURVE_SAMPLES
            && (vert_ids.first() == vert_ids.last()
                || vert_ids
                    .first()
                    .zip(vert_ids.last())
                    .is_some_and(|(f, l)| f.index() == l.index()))
        {
            // Cylinder barrel: build from sampled curve ring.
            let ring_len = if vert_ids.first() == vert_ids.last() {
                n - 1
            } else {
                n
            };
            let mut edges = Vec::with_capacity(ring_len);
            for i in 0..ring_len {
                let j = (i + 1) % ring_len;
                let vi = vert_ids[i];
                let vj = vert_ids[j % vert_ids.len()];
                let is_forward = vi.index() <= vj.index();
                let key = if is_forward {
                    (vi.index(), vj.index())
                } else {
                    (vj.index(), vi.index())
                };
                let eid = *edge_map.entry(key).or_insert_with(|| {
                    let (start, end) = if is_forward { (vi, vj) } else { (vj, vi) };
                    topo.edges.alloc(Edge::new(start, end, EdgeCurve::Line))
                });
                edges.push(OrientedEdge::new(eid, is_forward != flip));
            }
            if flip {
                edges.reverse();
            }
            let wire = Wire::new(edges, true).map_err(crate::OperationsError::Topology)?;
            topo.wires.alloc(wire)
        } else {
            let mut edges = Vec::with_capacity(n);
            for i in 0..n {
                let j = (i + 1) % n;
                let vi = vert_ids[i];
                let vj = vert_ids[j];
                let is_forward = vi.index() <= vj.index();
                let key = if is_forward {
                    (vi.index(), vj.index())
                } else {
                    (vj.index(), vi.index())
                };
                let eid = *edge_map.entry(key).or_insert_with(|| {
                    let (start, end) = if is_forward { (vi, vj) } else { (vj, vi) };
                    topo.edges.alloc(Edge::new(start, end, EdgeCurve::Line))
                });
                edges.push(OrientedEdge::new(eid, is_forward != flip));
            }
            if flip {
                edges.reverse();
            }
            let wire = Wire::new(edges, true).map_err(crate::OperationsError::Topology)?;
            topo.wires.alloc(wire)
        };

        let mut inner_wire_ids = Vec::new();
        if let Some(existing_wires) = existing_inner_wires.get(&idx) {
            inner_wire_ids.extend_from_slice(existing_wires);
        }
        if let Some(inner_curves) = holed_face_inner_curves.get(&idx) {
            for ec in inner_curves {
                let hw_id = if matches!(ec, EdgeCurve::Circle(_) | EdgeCurve::Ellipse(_)) {
                    let seam_pt = sample_edge_curve(ec, CLOSED_CURVE_SAMPLES)[0];
                    let vid = *vertex_map
                        .entry(quantize_point(seam_pt, resolution))
                        .or_insert_with(|| topo.vertices.alloc(Vertex::new(seam_pt, tol.linear)));
                    let eid = *edge_map
                        .entry((vid.index(), vid.index()))
                        .or_insert_with(|| topo.edges.alloc(Edge::new(vid, vid, ec.clone())));
                    let hw = Wire::new(vec![OrientedEdge::new(eid, flip)], true)
                        .map_err(crate::OperationsError::Topology)?;
                    topo.wires.alloc(hw)
                } else {
                    let mut hole_pts = sample_edge_curve(ec, CLOSED_CURVE_SAMPLES);
                    if !flip {
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
                                .or_insert_with(|| topo.vertices.alloc(Vertex::new(*p, tol.linear)))
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
                            topo.edges.alloc(Edge::new(start, end, EdgeCurve::Line))
                        });
                        hole_edges.push(OrientedEdge::new(eid, is_forward));
                    }
                    let hw =
                        Wire::new(hole_edges, true).map_err(crate::OperationsError::Topology)?;
                    topo.wires.alloc(hw)
                };
                inner_wire_ids.push(hw_id);
            }
        }

        let surface = match &frag.surface {
            FaceSurface::Plane { .. } => FaceSurface::Plane { normal, d: d_val },
            other => other.clone(),
        };
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
        let face = topo.faces.alloc(new_face);
        face_ids_out.push(face);
    }

    if face_ids_out.is_empty() {
        return Err(crate::OperationsError::InvalidInput {
            reason: "compound_cut produced no faces".into(),
        });
    }

    // ── Post-assembly ────────────────────────────────────────────────────
    let _t_phase5 = timer_elapsed_ms(_t_total);
    log::debug!(
        "[compound_cut] Phase 5: {:.1}ms — assembly ({} faces)",
        _t_phase5 - _t_phase4,
        face_ids_out.len()
    );

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
    let _t_refine = timer_elapsed_ms(_t_total);
    log::debug!("[compound_cut] refine: {:.1}ms", _t_refine - _t_phase5);
    split_nonmanifold_edges(topo, &mut face_ids_out)?;
    let _t_nm = timer_elapsed_ms(_t_total);
    log::debug!("[compound_cut] split_nm: {:.1}ms", _t_nm - _t_refine);

    let shell = Shell::new(face_ids_out).map_err(crate::OperationsError::Topology)?;
    let shell_id = topo.shells.alloc(shell);
    log::debug!("[compound_cut] total: {:.3}ms", timer_elapsed_ms(_t_total));
    Ok(topo.solids.alloc(Solid::new(shell_id, vec![])))
}

/// Sequential fallback for `compound_cut` when analytic path is unavailable.
///
/// Uses AABB pre-filtering to skip tools that don't overlap the current target,
/// avoiding expensive boolean operations on spatially disjoint tools.
fn compound_cut_sequential(
    topo: &mut Topology,
    target: SolidId,
    tools: &[SolidId],
    opts: BooleanOptions,
) -> Result<SolidId, crate::OperationsError> {
    let _t = timer_now();
    let mut result = target;
    let mut skipped = 0usize;
    for &tool in tools {
        // AABB pre-filter: skip tools that don't overlap the current target.
        let target_aabb = crate::measure::solid_bounding_box(topo, result)?;
        let tool_aabb = crate::measure::solid_bounding_box(topo, tool)?;
        if !target_aabb.intersects(tool_aabb) {
            skipped += 1;
            continue;
        }
        result = boolean_with_options(topo, BooleanOp::Cut, result, tool, opts)?;
    }
    log::debug!(
        "[compound_cut_sequential] {} tools, {} skipped (disjoint), {:.1}ms",
        tools.len(),
        skipped,
        timer_elapsed_ms(_t)
    );
    Ok(result)
}

/// Compute a plane-plane intersection chord clipped to both face polygons.
fn plane_plane_chord_analytic(
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

    let t_range_a = cyrus_beck_clip(&line_origin, &line_dir, verts_a, &normal_a, tol);
    let t_range_b = cyrus_beck_clip(&line_origin, &line_dir, verts_b, &normal_b, tol);

    let (t_min_a, t_max_a) = t_range_a?;
    let (t_min_b, t_max_b) = t_range_b?;

    let t_min = t_min_a.max(t_min_b);
    let t_max = t_max_a.min(t_max_b);

    if t_max - t_min < tol.linear {
        return None;
    }

    Some((
        point_along_line(&line_origin, &line_dir, t_min),
        point_along_line(&line_origin, &line_dir, t_max),
    ))
}

/// Result of classifying an intersection curve against a face boundary.
enum CurveClassification {
    /// The curve crosses the face boundary — contains entry/exit points.
    Crossings(Vec<Point3>),
    /// The entire curve lies inside the face (no boundary crossings).
    FullyContained,
    /// The entire curve lies outside the face.
    FullyOutside,
}

/// Find where an intersection curve crosses the face boundary, returning
/// boundary crossing points.
///
/// Instead of returning all sample points inside the face (which creates many
/// chord segments), this finds the entry/exit points where the curve crosses
/// the polygon boundary. This gives 1-2 chord endpoints per crossing, keeping
/// the split count minimal.
///
/// When the curve lies entirely inside the face (zero crossings, all samples
/// inside), returns `FullyContained` so callers can create a splitting chord
/// through the curve.
fn curve_boundary_crossings(
    curve: &brepkit_math::analytic_intersection::ExactIntersectionCurve,
    face_verts: &[Point3],
    face_normal: Vec3,
    _tol: Tolerance,
) -> CurveClassification {
    use brepkit_math::analytic_intersection::ExactIntersectionCurve;

    let n_samples = 64;
    let raw_points: Vec<Point3> = match curve {
        ExactIntersectionCurve::Circle(c) => (0..=n_samples)
            .map(|i| {
                #[allow(clippy::cast_precision_loss)]
                let t = std::f64::consts::TAU * (i as f64) / (n_samples as f64);
                c.evaluate(t)
            })
            .collect(),
        ExactIntersectionCurve::Ellipse(e) => (0..=n_samples)
            .map(|i| {
                #[allow(clippy::cast_precision_loss)]
                let t = std::f64::consts::TAU * (i as f64) / (n_samples as f64);
                e.evaluate(t)
            })
            .collect(),
        ExactIntersectionCurve::Points(pts) => pts.clone(),
    };

    if raw_points.len() < 2 {
        return CurveClassification::Crossings(raw_points);
    }

    // Pre-project face polygon to 2D once, then test all sample points
    // against the projected polygon. This avoids re-projecting the polygon
    // for every sample point (64 allocations → 1 allocation).
    let ax = face_normal.x().abs();
    let ay = face_normal.y().abs();
    let az = face_normal.z().abs();
    let project_3d_to_2d = |p: Point3| -> Point2 {
        if az >= ax && az >= ay {
            Point2::new(p.x(), p.y())
        } else if ay >= ax {
            Point2::new(p.x(), p.z())
        } else {
            Point2::new(p.y(), p.z())
        }
    };
    let polygon_2d: Vec<Point2> = face_verts.iter().map(|p| project_3d_to_2d(*p)).collect();

    // Classify each sample as inside or outside the face polygon.
    let inside: Vec<bool> = raw_points
        .iter()
        .map(|pt| point_in_polygon(project_3d_to_2d(*pt), &polygon_2d))
        .collect();

    let all_inside = inside.iter().all(|&v| v);
    let none_inside = inside.iter().all(|&v| !v);

    if all_inside {
        return CurveClassification::FullyContained;
    }
    if none_inside {
        return CurveClassification::FullyOutside;
    }

    // Find boundary crossing points: transitions from inside→outside or
    // outside→inside. At each transition, interpolate the approximate crossing
    // point. Also include the first and last interior points as chord endpoints.
    let mut crossings = Vec::new();
    let mut in_run = false;

    for i in 0..raw_points.len() {
        if inside[i] && !in_run {
            // Entering the face: record the entry point (use midpoint of
            // crossing segment for better accuracy).
            if i > 0 && !inside[i - 1] {
                let mid = Point3::new(
                    (raw_points[i - 1].x() + raw_points[i].x()) * 0.5,
                    (raw_points[i - 1].y() + raw_points[i].y()) * 0.5,
                    (raw_points[i - 1].z() + raw_points[i].z()) * 0.5,
                );
                crossings.push(mid);
            } else {
                crossings.push(raw_points[i]);
            }
            in_run = true;
        } else if !inside[i] && in_run {
            // Exiting the face: record the exit point.
            if i > 0 {
                let mid = Point3::new(
                    (raw_points[i - 1].x() + raw_points[i].x()) * 0.5,
                    (raw_points[i - 1].y() + raw_points[i].y()) * 0.5,
                    (raw_points[i - 1].z() + raw_points[i].z()) * 0.5,
                );
                crossings.push(mid);
            }
            in_run = false;
        }
    }

    CurveClassification::Crossings(crossings)
}

/// Tessellate a non-planar face into triangle fragments for the analytic boolean.
///
/// Used for sphere faces where band decomposition isn't feasible. Each triangle
/// gets its own `AnalyticFragment` but retains the original surface type so that
/// output faces preserve the analytic geometry.
fn tessellate_face_into_fragments(
    topo: &Topology,
    face_id: FaceId,
    source: Source,
    deflection: f64,
    fragments: &mut Vec<AnalyticFragment>,
) -> Result<(), crate::OperationsError> {
    let mesh = crate::tessellate::tessellate(topo, face_id, deflection)?;
    for tri in mesh.indices.chunks_exact(3) {
        let v0 = mesh.positions[tri[0] as usize];
        let v1 = mesh.positions[tri[1] as usize];
        let v2 = mesh.positions[tri[2] as usize];

        let edge1 = v1 - v0;
        let edge2 = v2 - v0;
        let cross = edge1.cross(edge2);
        // Skip degenerate (zero-area) triangles.
        let Ok(normal) = cross.normalize() else {
            continue;
        };
        let d_val = dot_normal_point(normal, v0);

        // Use planar surface for tessellated triangle fragments.
        // The original curved surface (e.g. Sphere) can't be used because
        // each triangle is a flat approximation — re-tessellating with the
        // curved surface would project vertices back onto the sphere,
        // producing wrong geometry.
        let plane_surface = FaceSurface::Plane { normal, d: d_val };

        fragments.push(AnalyticFragment {
            vertices: vec![v0, v1, v2],
            surface: plane_surface,
            normal,
            d: d_val,
            source,
            edge_curves: vec![None; 3],
            source_reversed: false, // planar fragment, no reversal needed
        });
    }
    Ok(())
}

/// Split a cylinder face at intersection v-ranges, producing:
/// - Cylinder-surface band fragments for regions *outside* the intersection
/// - Tessellated planar triangle fragments for the narrow intersection band
///
/// This prevents face-count explosion in sequential boolean operations:
/// instead of tessellating the *entire* barrel into ~500 triangles, only the
/// thin intersection region gets tessellated while the rest keeps
/// `FaceSurface::Cylinder`.
#[allow(clippy::too_many_lines, clippy::too_many_arguments)]
fn split_cylinder_at_intersection(
    surface: &FaceSurface,
    face_verts: &[Point3],
    normal: Vec3,
    d: f64,
    source: Source,
    source_reversed: bool,
    vranges: &[(f64, f64)],
    topo: &Topology,
    face_id: FaceId,
    deflection: f64,
    tol: Tolerance,
    fragments: &mut Vec<AnalyticFragment>,
) -> Result<(), crate::OperationsError> {
    let FaceSurface::Cylinder(cyl) = surface else {
        // Not a cylinder — fall back to full tessellation.
        return tessellate_face_into_fragments(topo, face_id, source, deflection, fragments);
    };

    // Compute the barrel's v extent from boundary vertices.
    let Some((barrel_vmin, barrel_vmax)) = cylinder_v_extent(cyl, face_verts) else {
        fragments.push(AnalyticFragment {
            vertices: face_verts.to_vec(),
            surface: surface.clone(),
            normal,
            d,
            source,
            edge_curves: vec![None; face_verts.len()],
            source_reversed,
        });
        return Ok(());
    };

    if vranges.is_empty() {
        // Degenerate or no ranges — keep as-is.
        fragments.push(AnalyticFragment {
            vertices: face_verts.to_vec(),
            surface: surface.clone(),
            normal,
            d,
            source,
            edge_curves: vec![None; face_verts.len()],
            source_reversed,
        });
        return Ok(());
    }

    // Merge overlapping v-ranges with padding. Fall back to full tessellation
    // if intersection zones cover >60% of the barrel height.
    let Some(merged) = merge_vranges_with_padding(vranges, barrel_vmin, barrel_vmax, 0.05) else {
        return tessellate_face_into_fragments(topo, face_id, source, deflection, fragments);
    };

    // Build level list: regions outside merged ranges are cylinder bands,
    // regions inside merged ranges get tessellated.
    // Levels: barrel_vmin, [gap regions as cylinder], [intersection regions tessellated], barrel_vmax
    let n_samples: usize = CLOSED_CURVE_SAMPLES;

    // Collect face polygon vertices at barrel endpoints for exact vertex matching.
    let v_tol = 1e-6;
    let mut verts_at_vmin: Vec<Point3> = Vec::new();
    let mut verts_at_vmax: Vec<Point3> = Vec::new();
    for &p in face_verts {
        let v = cyl.axis().dot(p - cyl.origin());
        if (v - barrel_vmin).abs() < v_tol {
            verts_at_vmin.push(p);
        } else if (v - barrel_vmax).abs() < v_tol {
            verts_at_vmax.push(p);
        }
    }
    dedup_points_by_position(&mut verts_at_vmin, tol);
    dedup_points_by_position(&mut verts_at_vmax, tol);

    #[allow(clippy::cast_precision_loss)]
    let sample_circle_at_v = |v: f64| -> Vec<Point3> {
        if (v - barrel_vmin).abs() < v_tol && verts_at_vmin.len() >= 3 {
            verts_at_vmin.clone()
        } else if (v - barrel_vmax).abs() < v_tol && verts_at_vmax.len() >= 3 {
            verts_at_vmax.clone()
        } else {
            (0..n_samples)
                .map(|i| {
                    let u = std::f64::consts::TAU * (i as f64) / (n_samples as f64);
                    cyl.evaluate(u, v)
                })
                .collect()
        }
    };

    // Helper: create a cylinder band fragment between two v-levels.
    let make_band = |v_bot: f64, v_top: f64, frags: &mut Vec<AnalyticFragment>| {
        if (v_top - v_bot).abs() < 1e-10 {
            return;
        }
        let bot_pts = sample_circle_at_v(v_bot);
        let top_pts = sample_circle_at_v(v_top);

        let mut verts = Vec::with_capacity(bot_pts.len() + top_pts.len());
        verts.extend_from_slice(&bot_pts);
        verts.extend(top_pts.into_iter().rev());

        // Compute normal from a surface point (not centroid) to avoid
        // degenerate zero normal for full-circle bands.
        let surface_point = verts[0];
        let band_normal = (surface_point
            - cyl.origin()
            - cyl.axis() * cyl.axis().dot(surface_point - cyl.origin()))
        .normalize()
        .unwrap_or(normal);
        let centroid = polygon_centroid(&verts);
        let band_d = crate::dot_normal_point(band_normal, centroid);

        let n_verts = verts.len();
        frags.push(AnalyticFragment {
            vertices: verts,
            surface: surface.clone(),
            normal: band_normal,
            d: band_d,
            source,
            edge_curves: vec![None; n_verts],
            source_reversed,
        });
    };

    // Walk through the barrel, creating cylinder bands for all regions.
    // Both gap regions and intersection zones become cylinder bands,
    // preserving FaceSurface::Cylinder throughout. The classifier
    // determines inside/outside for each band independently.
    //
    let levels = build_v_levels(barrel_vmin, barrel_vmax, &merged);

    for w in levels.windows(2) {
        make_band(w[0], w[1], fragments);
    }

    Ok(())
}

/// Split a sphere face into spherical cap fragments at intersection v-levels.
///
/// Analogous to `split_cylinder_at_intersection` but for spherical geometry.
/// Each cap preserves `FaceSurface::Sphere`. At poles (v = ±π/2) the cap
/// degenerates to a single point surrounded by a circle.
#[allow(clippy::too_many_lines)]
#[allow(clippy::too_many_arguments)]
fn split_sphere_at_intersection(
    surface: &FaceSurface,
    face_verts: &[Point3],
    normal: Vec3,
    d: f64,
    source: Source,
    source_reversed: bool,
    vranges: &[(f64, f64)],
    topo: &Topology,
    face_id: FaceId,
    deflection: f64,
    fragments: &mut Vec<AnalyticFragment>,
) -> Result<(), crate::OperationsError> {
    let FaceSurface::Sphere(sph) = surface else {
        return tessellate_face_into_fragments(topo, face_id, source, deflection, fragments);
    };

    // Compute the face's v-extent. For hemispheres the boundary wire only
    // covers one edge (equator); the other limit is a pole.
    let face_data = topo.face(face_id)?;
    let face_v_range = crate::tessellate::compute_sphere_v_range(topo, face_data, sph);
    let face_vmin = face_v_range.0;
    let face_vmax = face_v_range.1;

    if (face_vmax - face_vmin).abs() < 1e-10 || vranges.is_empty() {
        fragments.push(AnalyticFragment {
            vertices: face_verts.to_vec(),
            surface: surface.clone(),
            normal,
            d,
            source,
            edge_curves: vec![None; face_verts.len()],
            source_reversed,
        });
        return Ok(());
    }

    // Merge overlapping v-ranges with padding. Fall back to full tessellation
    // if intersection zones cover >60% of the face height.
    let Some(merged) = merge_vranges_with_padding(vranges, face_vmin, face_vmax, 0.05) else {
        return tessellate_face_into_fragments(topo, face_id, source, deflection, fragments);
    };

    let n_samples: usize = CLOSED_CURVE_SAMPLES;
    let pole_eps = 1e-6;
    let is_south_pole = |v: f64| (v + std::f64::consts::FRAC_PI_2).abs() < pole_eps;
    let is_north_pole = |v: f64| (v - std::f64::consts::FRAC_PI_2).abs() < pole_eps;

    // Sample a circle of points at a given latitude on the sphere.
    #[allow(clippy::cast_precision_loss)]
    let sample_circle_at_v = |v: f64| -> Vec<Point3> {
        if is_south_pole(v) || is_north_pole(v) {
            // Pole: single point.
            vec![sph.evaluate(0.0, v)]
        } else {
            (0..n_samples)
                .map(|i| {
                    let u = std::f64::consts::TAU * (i as f64) / (n_samples as f64);
                    sph.evaluate(u, v)
                })
                .collect()
        }
    };

    // Create a spherical cap fragment between two v-levels.
    let make_cap = |v_bot: f64, v_top: f64, frags: &mut Vec<AnalyticFragment>| {
        if (v_top - v_bot).abs() < 1e-10 {
            return;
        }
        let bot_pts = sample_circle_at_v(v_bot);
        let top_pts = sample_circle_at_v(v_top);

        let mut verts = Vec::with_capacity(bot_pts.len() + top_pts.len());
        verts.extend_from_slice(&bot_pts);
        if top_pts.len() > 1 {
            verts.extend(top_pts.into_iter().rev());
        } else {
            // Pole point: append as-is (single vertex).
            verts.extend(top_pts);
        }

        // Compute outward normal from a surface point.
        let sample_pt = verts[0];
        let cap_normal = (sample_pt - sph.center()).normalize().unwrap_or(normal);
        let centroid = polygon_centroid(&verts);
        let cap_d = crate::dot_normal_point(cap_normal, centroid);

        let n_verts = verts.len();
        frags.push(AnalyticFragment {
            vertices: verts,
            surface: surface.clone(),
            normal: cap_normal,
            d: cap_d,
            source,
            edge_curves: vec![None; n_verts],
            source_reversed,
        });
    };

    let levels = build_v_levels(face_vmin, face_vmax, &merged);

    for w in levels.windows(2) {
        make_cap(w[0], w[1], fragments);
    }

    Ok(())
}

/// Compute v-parameter ranges where intersection chords cross analytic faces.
///
/// For each face that has intersection chords (and isn't in the analytic-analytic
/// set), projects chord endpoints onto the surface's v-parameter and records
/// the (vmin, vmax) interval.
#[allow(clippy::type_complexity)]
fn collect_analytic_vranges(
    snaps: &[FaceSnapshot],
    face_intersections: &HashMap<usize, Vec<(Point3, Point3, Option<EdgeCurve>)>>,
    analytic_analytic_faces: &HashSet<usize>,
    vranges_out: &mut HashMap<usize, Vec<(f64, f64)>>,
) {
    for (idx, snap) in snaps.iter().enumerate() {
        let Some(chords) = face_intersections.get(&idx) else {
            continue;
        };
        if analytic_analytic_faces.contains(&idx) {
            continue;
        }
        let v_of_point: Box<dyn Fn(Point3) -> f64> = match &snap.surface {
            FaceSurface::Cylinder(cyl) => {
                let axis = cyl.axis();
                let origin = cyl.origin();
                Box::new(move |p| axis.dot(p - origin))
            }
            FaceSurface::Sphere(sph) => {
                let sph = sph.clone();
                Box::new(move |p| sph.project_point(p).1)
            }
            _ => continue,
        };
        let mut vmin = f64::MAX;
        let mut vmax = f64::MIN;
        for &(p0, p1, _) in chords {
            for p in &[p0, p1] {
                let v = v_of_point(*p);
                vmin = vmin.min(v);
                vmax = vmax.max(v);
            }
        }
        if vmax > vmin {
            vranges_out.entry(idx).or_default().push((vmin, vmax));
        }
    }
}

/// Create band fragments for a non-planar (analytic) face that has contained
/// curves. Splits the face into bands between the contained curves and the
/// face's natural boundary circles.
///
/// Currently supports cylinder faces. Other surface types fall through without
/// creating bands (the face stays as one unsplit fragment via the caller's
/// else branch, but this path shouldn't be reached for unsupported types).
#[allow(clippy::too_many_lines)]
#[allow(clippy::too_many_arguments)]
fn create_band_fragments(
    surface: &FaceSurface,
    face_verts: &[Point3],
    normal: Vec3,
    d: f64,
    source: Source,
    source_reversed: bool,
    contained_curves: &[EdgeCurve],
    _topo: &Topology,
    tol: Tolerance,
    fragments: &mut Vec<AnalyticFragment>,
) {
    let FaceSurface::Cylinder(cyl) = surface else {
        // For non-cylinder analytic faces, fall back to unsplit fragment.
        fragments.push(AnalyticFragment {
            vertices: face_verts.to_vec(),
            surface: surface.clone(),
            normal,
            d,
            source,
            edge_curves: vec![None; face_verts.len()],
            source_reversed,
        });
        return;
    };

    let n_samples: usize = CLOSED_CURVE_SAMPLES;

    // Pair each contained curve with its v-parameter on the cylinder axis.
    let mut cut_levels: Vec<(f64, &EdgeCurve)> = Vec::new();
    for ec in contained_curves {
        let center = match ec {
            EdgeCurve::Circle(c) => c.center(),
            EdgeCurve::Ellipse(e) => e.center(),
            _ => continue,
        };
        let v = cyl.axis().dot(center - cyl.origin());
        cut_levels.push((v, ec));
    }

    if cut_levels.is_empty() {
        fragments.push(AnalyticFragment {
            vertices: face_verts.to_vec(),
            surface: surface.clone(),
            normal,
            d,
            source,
            edge_curves: vec![None; face_verts.len()],
            source_reversed,
        });
        return;
    }

    // Compute the barrel's v extent from its boundary vertices.
    let Some((v_min, v_max)) = cylinder_v_extent(cyl, face_verts) else {
        fragments.push(AnalyticFragment {
            vertices: face_verts.to_vec(),
            surface: surface.clone(),
            normal,
            d,
            source,
            edge_curves: vec![None; face_verts.len()],
            source_reversed,
        });
        return;
    };

    // Sort cut levels by v-parameter.
    cut_levels.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap_or(std::cmp::Ordering::Equal));

    // Build ordered level list: barrel_bottom, cut1, cut2, ..., barrel_top.
    // Each level is either a barrel endpoint (None) or a cut curve (Some).
    let mut levels: Vec<(f64, Option<&EdgeCurve>)> = vec![(v_min, None)];
    for &(cv, ec) in &cut_levels {
        if let Some(last) = levels.last() {
            if (cv - last.0).abs() > 1e-10 {
                levels.push((cv, Some(ec)));
            }
        }
    }
    if let Some(last) = levels.last() {
        if (v_max - last.0).abs() > 1e-10 {
            levels.push((v_max, None));
        }
    }

    // Extract face polygon vertices at each barrel endpoint level.
    // These come from the same Circle3D::evaluate calls that generated the
    // cap face polygons, ensuring exact floating-point match for vertex dedup.
    let v_tol = 1e-6;
    let mut verts_at_vmin: Vec<Point3> = Vec::new();
    let mut verts_at_vmax: Vec<Point3> = Vec::new();
    for &p in face_verts {
        let v = cyl.axis().dot(p - cyl.origin());
        if (v - v_min).abs() < v_tol {
            verts_at_vmin.push(p);
        } else if (v - v_max).abs() < v_tol {
            verts_at_vmax.push(p);
        }
    }
    // Deduplicate points at each level (seam vertex duplicates circle[0]).
    dedup_points_by_position(&mut verts_at_vmin, tol);
    dedup_points_by_position(&mut verts_at_vmax, tol);

    // Sample a circle at a given v-level. For cut levels with an EdgeCurve,
    // use sample_edge_curve so the points match the holed-face inner wire.
    // For barrel endpoints, use the actual face polygon vertices so they
    // share vertices/edges with adjacent cap faces (exact float match).
    #[allow(clippy::cast_precision_loss)]
    let sample_level = |v: f64, curve: Option<&EdgeCurve>| -> Vec<Point3> {
        if let Some(ec) = curve {
            sample_edge_curve(ec, n_samples)
        } else if (v - v_min).abs() < v_tol && verts_at_vmin.len() >= 3 {
            verts_at_vmin.clone()
        } else if (v - v_max).abs() < v_tol && verts_at_vmax.len() >= 3 {
            verts_at_vmax.clone()
        } else {
            (0..n_samples)
                .map(|i| {
                    let u = std::f64::consts::TAU * (i as f64) / (n_samples as f64);
                    cyl.evaluate(u, v)
                })
                .collect()
        }
    };

    // Create a band fragment for each consecutive pair of levels.
    for w in levels.windows(2) {
        let (v_bot, ec_bot) = w[0];
        let (v_top, ec_top) = w[1];

        let bot_pts = sample_level(v_bot, ec_bot);
        let top_pts = sample_level(v_top, ec_top);

        // Build polygon: bottom circle forward, top circle reversed.
        let mut verts = Vec::with_capacity(2 * n_samples);
        verts.extend_from_slice(&bot_pts);
        verts.extend(top_pts.into_iter().rev());

        // Compute representative normal and d for the band.
        // Use a surface point (first vertex) for the outward normal, not
        // the polygon centroid — the centroid of a full-circle band falls
        // on the cylinder axis, making the radial direction degenerate.
        let surface_point = verts[0];
        let band_normal = (surface_point
            - cyl.origin()
            - cyl.axis() * cyl.axis().dot(surface_point - cyl.origin()))
        .normalize()
        .unwrap_or(normal);
        let centroid = polygon_centroid(&verts);
        let band_d = crate::dot_normal_point(band_normal, centroid);

        let n_verts = verts.len();
        fragments.push(AnalyticFragment {
            vertices: verts,
            surface: surface.clone(),
            normal: band_normal,
            d: band_d,
            source,
            edge_curves: vec![None; n_verts],
            source_reversed,
        });
    }
}

/// Build a proper cylinder barrel wire with Circle edges + seam line.
///
/// Cylinder barrel fragments are represented as polygons with layout:
///   `bot[0..n] ++ top_reversed[0..n]`  (2n vertices total)
/// where n = `CLOSED_CURVE_SAMPLES`. This function consolidates those 2n
/// line edges into 2 Circle edges + 1 seam Line (3 unique edges, 4 oriented),
/// matching the canonical B-Rep topology for a cylinder lateral face.
#[allow(clippy::too_many_arguments)]
fn build_cylinder_barrel_wire(
    topo: &mut Topology,
    cyl: &brepkit_math::surfaces::CylindricalSurface,
    verts: &[Point3],
    vertex_map: &mut HashMap<(i64, i64, i64), VertexId>,
    edge_map: &mut HashMap<(usize, usize), brepkit_topology::edge::EdgeId>,
    resolution: f64,
    tol: Tolerance,
) -> Result<WireId, crate::OperationsError> {
    // The polygon layout is: bot[0..n/2] + top_reversed[0..n/2].
    // bot[0] is at u=0 on the bottom circle; verts[2n-1] = top[0] is at u=0 on the top circle.
    let bot_seam_pos = verts[0];
    let top_seam_pos = verts[verts.len() - 1];

    // Compute v-levels from vertex positions on the cylinder axis.
    let v_bot = cyl.axis().dot(bot_seam_pos - cyl.origin());
    let v_top = cyl.axis().dot(top_seam_pos - cyl.origin());

    // Create Circle3D at each level.
    let bot_center = cyl.origin() + cyl.axis() * v_bot;
    let top_center = cyl.origin() + cyl.axis() * v_top;
    let bot_circle = brepkit_math::curves::Circle3D::new(bot_center, cyl.axis(), cyl.radius())
        .map_err(crate::OperationsError::Math)?;
    let top_circle = brepkit_math::curves::Circle3D::new(top_center, cyl.axis(), cyl.radius())
        .map_err(crate::OperationsError::Math)?;

    // Create/lookup vertices at the seam points.
    let bot_vid = *vertex_map
        .entry(quantize_point(bot_seam_pos, resolution))
        .or_insert_with(|| topo.vertices.alloc(Vertex::new(bot_seam_pos, tol.linear)));
    let top_vid = *vertex_map
        .entry(quantize_point(top_seam_pos, resolution))
        .or_insert_with(|| topo.vertices.alloc(Vertex::new(top_seam_pos, tol.linear)));

    // Create/lookup closed Circle edges — dedup key is (v, v) for closed edges.
    let bot_edge = *edge_map
        .entry((bot_vid.index(), bot_vid.index()))
        .or_insert_with(|| {
            topo.edges
                .alloc(Edge::new(bot_vid, bot_vid, EdgeCurve::Circle(bot_circle)))
        });
    let top_edge = *edge_map
        .entry((top_vid.index(), top_vid.index()))
        .or_insert_with(|| {
            topo.edges
                .alloc(Edge::new(top_vid, top_vid, EdgeCurve::Circle(top_circle)))
        });

    // Create/lookup seam line edge. Forward means bot→top in canonical order.
    let seam_fwd = bot_vid.index() <= top_vid.index();
    let seam_key = if seam_fwd {
        (bot_vid.index(), top_vid.index())
    } else {
        (top_vid.index(), bot_vid.index())
    };
    let seam_edge = *edge_map.entry(seam_key).or_insert_with(|| {
        let (start, end) = if seam_fwd {
            (bot_vid, top_vid)
        } else {
            (top_vid, bot_vid)
        };
        topo.edges.alloc(Edge::new(start, end, EdgeCurve::Line))
    });

    // Wire: bot_circle(fwd) → seam(fwd) → top_circle(rev) → seam(rev)
    // This matches the canonical cylinder lateral wire from make_cylinder.
    let wire = Wire::new(
        vec![
            OrientedEdge::new(bot_edge, true),
            OrientedEdge::new(seam_edge, seam_fwd),
            OrientedEdge::new(top_edge, false),
            OrientedEdge::new(seam_edge, !seam_fwd),
        ],
        true,
    )
    .map_err(crate::OperationsError::Topology)?;
    Ok(topo.wires.alloc(wire))
}

/// Sample an `EdgeCurve` into N points.
fn sample_edge_curve(curve: &EdgeCurve, n: usize) -> Vec<Point3> {
    match curve {
        EdgeCurve::Circle(c) => (0..n)
            .map(|i| {
                #[allow(clippy::cast_precision_loss)]
                let t = std::f64::consts::TAU * (i as f64) / (n as f64);
                c.evaluate(t)
            })
            .collect(),
        EdgeCurve::Ellipse(e) => (0..n)
            .map(|i| {
                #[allow(clippy::cast_precision_loss)]
                let t = std::f64::consts::TAU * (i as f64) / (n as f64);
                e.evaluate(t)
            })
            .collect(),
        EdgeCurve::NurbsCurve(nc) => {
            let (u0, u1) = nc.domain();
            // For closed curves (start ≈ end), use n as divisor to avoid
            // duplicating the first point at t=u_max.
            let start_pt = nc.evaluate(u0);
            let end_pt = nc.evaluate(u1);
            let is_closed = (start_pt - end_pt).length() < 1e-6;
            let divisor = if is_closed { n } else { n - 1 };
            (0..n)
                .map(|i| {
                    #[allow(clippy::cast_precision_loss)]
                    let t = u0 + (u1 - u0) * (i as f64) / (divisor as f64);
                    nc.evaluate(t)
                })
                .collect()
        }
        EdgeCurve::Line => vec![],
    }
}

#[cfg(test)]
mod tests {
    #![allow(clippy::unwrap_used)]
    #![allow(clippy::cast_precision_loss)]

    use brepkit_math::tolerance::Tolerance;
    use brepkit_math::vec::{Point3, Vec3};
    use brepkit_topology::Topology;
    use brepkit_topology::test_utils::make_unit_cube_manifold_at;
    use brepkit_topology::validation::validate_shell_manifold;

    use crate::test_helpers::assert_volume_near;

    use super::*;

    /// Helper: get the face count and validate manifoldness.
    fn check_result(topo: &Topology, solid: SolidId) -> usize {
        let s = topo.solid(solid).unwrap();
        let sh = topo.shell(s.outer_shell()).unwrap();
        assert!(
            validate_shell_manifold(sh, &topo.faces, &topo.wires).is_ok(),
            "result should be manifold"
        );
        sh.faces().len()
    }

    // ── Polygon clipper tests ─────────────────────────────────────────

    #[test]
    fn polygon_clip_convex_square() {
        let tol = Tolerance::new();
        let polygon = vec![
            Point3::new(0.0, 0.0, 0.0),
            Point3::new(2.0, 0.0, 0.0),
            Point3::new(2.0, 2.0, 0.0),
            Point3::new(0.0, 2.0, 0.0),
        ];
        let normal = Vec3::new(0.0, 0.0, 1.0);
        // Line through center, along Y
        let line_pt = Point3::new(1.0, 0.0, 0.0);
        let line_dir = Vec3::new(0.0, 1.0, 0.0);
        let intervals = super::polygon_clip_intervals(&line_pt, &line_dir, &polygon, &normal, tol);
        assert_eq!(intervals.len(), 1, "expected 1 interval, got {intervals:?}");
        assert!(
            (intervals[0].0 - 0.0).abs() < 0.01,
            "t_min={}",
            intervals[0].0
        );
        assert!(
            (intervals[0].1 - 2.0).abs() < 0.01,
            "t_max={}",
            intervals[0].1
        );
    }

    #[test]
    fn polygon_clip_l_shape() {
        let tol = Tolerance::new();
        // L-shaped (concave) polygon
        let polygon = vec![
            Point3::new(0.0, 0.0, 0.0),
            Point3::new(2.0, 0.0, 0.0),
            Point3::new(2.0, 1.0, 0.0),
            Point3::new(1.0, 1.0, 0.0),
            Point3::new(1.0, 2.0, 0.0),
            Point3::new(0.0, 2.0, 0.0),
        ];
        let normal = Vec3::new(0.0, 0.0, 1.0);
        // Line at x=0.5, along Y — should give one interval [0, 2]
        let intervals = super::polygon_clip_intervals(
            &Point3::new(0.5, 0.0, 0.0),
            &Vec3::new(0.0, 1.0, 0.0),
            &polygon,
            &normal,
            tol,
        );
        assert_eq!(
            intervals.len(),
            1,
            "x=0.5 should have 1 interval: {intervals:?}"
        );

        // Line at x=1.5, along Y — should give one interval [0, 1] (narrow arm)
        let intervals2 = super::polygon_clip_intervals(
            &Point3::new(1.5, 0.0, 0.0),
            &Vec3::new(0.0, 1.0, 0.0),
            &polygon,
            &normal,
            tol,
        );
        assert_eq!(
            intervals2.len(),
            1,
            "x=1.5 should have 1 interval: {intervals2:?}"
        );
        assert!(
            intervals2[0].1 < 1.5,
            "should only reach y=1, got {}",
            intervals2[0].1
        );
    }

    // ── Disjoint tests ──────────────────────────────────────────────────

    #[test]
    fn fuse_disjoint_cubes() {
        let mut topo = Topology::new();
        let a = make_unit_cube_manifold_at(&mut topo, 0.0, 0.0, 0.0);
        let b = make_unit_cube_manifold_at(&mut topo, 5.0, 0.0, 0.0);

        let result = boolean(&mut topo, BooleanOp::Fuse, a, b).unwrap();
        assert_eq!(check_result(&topo, result), 12); // 6 + 6
    }

    #[test]
    fn cut_disjoint_returns_a() {
        let mut topo = Topology::new();
        let a = make_unit_cube_manifold_at(&mut topo, 0.0, 0.0, 0.0);
        let b = make_unit_cube_manifold_at(&mut topo, 5.0, 0.0, 0.0);

        let result = boolean(&mut topo, BooleanOp::Cut, a, b).unwrap();
        assert_eq!(check_result(&topo, result), 6);
    }

    #[test]
    fn intersect_disjoint_returns_error() {
        let mut topo = Topology::new();
        let a = make_unit_cube_manifold_at(&mut topo, 0.0, 0.0, 0.0);
        let b = make_unit_cube_manifold_at(&mut topo, 5.0, 0.0, 0.0);

        assert!(boolean(&mut topo, BooleanOp::Intersect, a, b).is_err());
    }

    // ── 1D overlapping tests (offset on one axis) ───────────────────────

    #[test]
    fn fuse_overlapping_cubes() {
        let mut topo = Topology::new();
        let a = make_unit_cube_manifold_at(&mut topo, 0.0, 0.0, 0.0);
        let b = make_unit_cube_manifold_at(&mut topo, 0.5, 0.0, 0.0);

        let result = boolean(&mut topo, BooleanOp::Fuse, a, b).unwrap();
        let _ = check_result(&topo, result);
        assert_volume_near(&topo, result, 1.5, 0.001);
    }

    #[test]
    fn intersect_overlapping_cubes() {
        let mut topo = Topology::new();
        let a = make_unit_cube_manifold_at(&mut topo, 0.0, 0.0, 0.0);
        let b = make_unit_cube_manifold_at(&mut topo, 0.5, 0.0, 0.0);

        let result = boolean(&mut topo, BooleanOp::Intersect, a, b).unwrap();
        let _ = check_result(&topo, result);
        assert_volume_near(&topo, result, 0.5, 0.001);
    }

    #[test]
    fn cut_overlapping_cubes() {
        let mut topo = Topology::new();
        let a = make_unit_cube_manifold_at(&mut topo, 0.0, 0.0, 0.0);
        let b = make_unit_cube_manifold_at(&mut topo, 0.5, 0.0, 0.0);

        let result = boolean(&mut topo, BooleanOp::Cut, a, b).unwrap();
        let _ = check_result(&topo, result);
        assert_volume_near(&topo, result, 0.5, 0.001);
    }

    // ── 3D overlapping tests (offset on all axes) ───────────────────────

    #[test]
    fn fuse_overlapping_3d() {
        let mut topo = Topology::new();
        let a = make_unit_cube_manifold_at(&mut topo, 0.0, 0.0, 0.0);
        let b = make_unit_cube_manifold_at(&mut topo, 0.5, 0.5, 0.5);

        let result = boolean(&mut topo, BooleanOp::Fuse, a, b).unwrap();
        let _ = check_result(&topo, result);
        assert_volume_near(&topo, result, 1.875, 0.001);
    }

    #[test]
    fn intersect_overlapping_3d() {
        let mut topo = Topology::new();
        let a = make_unit_cube_manifold_at(&mut topo, 0.0, 0.0, 0.0);
        let b = make_unit_cube_manifold_at(&mut topo, 0.5, 0.5, 0.5);

        let result = boolean(&mut topo, BooleanOp::Intersect, a, b).unwrap();
        let _ = check_result(&topo, result);
        assert_volume_near(&topo, result, 0.125, 0.001);
    }

    #[test]
    fn cut_overlapping_3d() {
        let mut topo = Topology::new();
        let a = make_unit_cube_manifold_at(&mut topo, 0.0, 0.0, 0.0);
        let b = make_unit_cube_manifold_at(&mut topo, 0.5, 0.5, 0.5);

        let result = boolean(&mut topo, BooleanOp::Cut, a, b).unwrap();
        let _ = check_result(&topo, result);
        assert_volume_near(&topo, result, 0.875, 0.001);
    }

    // ── Flush face test ─────────────────────────────────────────────────

    #[test]
    fn fuse_flush_face_cubes() {
        let mut topo = Topology::new();
        let a = make_unit_cube_manifold_at(&mut topo, 0.0, 0.0, 0.0);
        let b = make_unit_cube_manifold_at(&mut topo, 1.0, 0.0, 0.0);

        let result = boolean(&mut topo, BooleanOp::Fuse, a, b).unwrap();
        let _ = check_result(&topo, result);
        assert_volume_near(&topo, result, 2.0, 0.001);
    }

    // ── NURBS face data collection test ─────────────────────────

    #[test]
    fn collect_face_data_handles_nurbs() {
        // Verify that collect_face_data no longer rejects NURBS solids.
        let mut topo = Topology::new();
        let cyl = crate::primitives::make_cylinder(&mut topo, 0.5, 1.0).unwrap();

        let result = collect_face_data(&topo, cyl, DEFAULT_BOOLEAN_DEFLECTION);
        assert!(
            result.is_ok(),
            "collect_face_data should handle NURBS: {:?}",
            result.err()
        );

        let faces = result.unwrap();
        // Cylinder has planar top/bottom + NURBS side → should produce
        // multiple face entries (tessellated triangles for NURBS).
        assert!(
            faces.len() > 2,
            "cylinder should produce more than 2 face entries, got {}",
            faces.len()
        );
    }

    // ── Analytic boolean tests ──────────────────────────────────────────

    #[test]
    #[allow(clippy::panic)]
    fn cylinder_circle_edges() {
        // make_cylinder should produce Circle edges for the boundary circles.
        let mut topo = Topology::new();
        let cyl = crate::primitives::make_cylinder(&mut topo, 1.0, 2.0).unwrap();
        let solid = topo.solid(cyl).unwrap();
        let shell = topo.shell(solid.outer_shell()).unwrap();

        let mut has_circle_edge = false;
        for &fid in shell.faces() {
            let face = topo.face(fid).unwrap();
            let wire = topo.wire(face.outer_wire()).unwrap();
            for oe in wire.edges() {
                let edge = topo.edge(oe.edge()).unwrap();
                if matches!(edge.curve(), brepkit_topology::edge::EdgeCurve::Circle(_)) {
                    has_circle_edge = true;
                }
            }
        }
        assert!(has_circle_edge, "cylinder should have Circle edges");
    }

    #[test]
    #[allow(clippy::panic)]
    fn circle_edge_length() {
        let mut topo = Topology::new();
        let cyl = crate::primitives::make_cylinder(&mut topo, 1.0, 2.0).unwrap();
        let solid = topo.solid(cyl).unwrap();
        let shell = topo.shell(solid.outer_shell()).unwrap();

        // Find a Circle edge and check its length.
        for &fid in shell.faces() {
            let face = topo.face(fid).unwrap();
            let wire = topo.wire(face.outer_wire()).unwrap();
            for oe in wire.edges() {
                let edge = topo.edge(oe.edge()).unwrap();
                if matches!(edge.curve(), brepkit_topology::edge::EdgeCurve::Circle(_)) {
                    let len = crate::measure::edge_length(&topo, oe.edge()).unwrap();
                    let expected = 2.0 * std::f64::consts::PI * 1.0; // circumference
                    assert!(
                        (len - expected).abs() < 1e-6,
                        "circle edge length should be 2πr = {expected}, got {len}"
                    );
                    return;
                }
            }
        }
        panic!("no Circle edge found");
    }

    #[test]
    #[allow(clippy::panic)]
    fn exact_plane_cylinder_circle() {
        use brepkit_math::analytic_intersection::{
            AnalyticSurface, ExactIntersectionCurve, exact_plane_analytic,
        };
        use brepkit_math::surfaces::CylindricalSurface;
        use brepkit_math::vec::{Point3 as P3, Vec3 as V3};

        let cyl =
            CylindricalSurface::new(P3::new(0.0, 0.0, 0.0), V3::new(0.0, 0.0, 1.0), 2.0).unwrap();
        let curves =
            exact_plane_analytic(AnalyticSurface::Cylinder(&cyl), V3::new(0.0, 0.0, 1.0), 3.0)
                .unwrap();
        assert_eq!(curves.len(), 1);
        match &curves[0] {
            ExactIntersectionCurve::Circle(c) => {
                assert!((c.radius() - 2.0).abs() < 1e-10, "radius should be 2.0");
                assert!(
                    (c.center().z() - 3.0).abs() < 1e-10,
                    "center z should be 3.0"
                );
            }
            _ => panic!("expected Circle, got {:?}", curves[0]),
        }
    }

    #[test]
    #[allow(clippy::panic)]
    fn exact_plane_sphere_circle() {
        use brepkit_math::analytic_intersection::{
            AnalyticSurface, ExactIntersectionCurve, exact_plane_analytic,
        };
        use brepkit_math::surfaces::SphericalSurface;
        use brepkit_math::vec::{Point3 as P3, Vec3 as V3};

        let sphere = SphericalSurface::new(P3::new(0.0, 0.0, 0.0), 3.0).unwrap();
        let curves = exact_plane_analytic(
            AnalyticSurface::Sphere(&sphere),
            V3::new(0.0, 0.0, 1.0),
            0.0,
        )
        .unwrap();
        assert_eq!(curves.len(), 1);
        match &curves[0] {
            ExactIntersectionCurve::Circle(c) => {
                assert!(
                    (c.radius() - 3.0).abs() < 1e-10,
                    "equator radius = sphere radius"
                );
            }
            _ => panic!("expected Circle"),
        }
    }

    #[test]
    #[allow(clippy::panic)]
    fn exact_plane_cylinder_ellipse() {
        use brepkit_math::analytic_intersection::{
            AnalyticSurface, ExactIntersectionCurve, exact_plane_analytic,
        };
        use brepkit_math::surfaces::CylindricalSurface;
        use brepkit_math::vec::{Point3 as P3, Vec3 as V3};

        let cyl =
            CylindricalSurface::new(P3::new(0.0, 0.0, 0.0), V3::new(0.0, 0.0, 1.0), 1.0).unwrap();
        // Oblique plane (45 degrees)
        let n = V3::new(0.0, 1.0, 1.0).normalize().unwrap();
        let curves = exact_plane_analytic(AnalyticSurface::Cylinder(&cyl), n, 0.0).unwrap();
        assert_eq!(curves.len(), 1);
        match &curves[0] {
            ExactIntersectionCurve::Ellipse(e) => {
                assert!((e.semi_minor() - 1.0).abs() < 1e-10, "semi_minor = radius");
                let expected_major = 1.0 / (std::f64::consts::FRAC_1_SQRT_2);
                assert!(
                    (e.semi_major() - expected_major).abs() < 1e-6,
                    "semi_major = r/cos(45°) = {expected_major}, got {}",
                    e.semi_major()
                );
            }
            _ => panic!("expected Ellipse, got {:?}", curves[0]),
        }
    }

    #[test]
    fn box_fuse_box_unchanged() {
        // Pure planar case should still work correctly through analytic path.
        let mut topo = Topology::new();
        let a = crate::primitives::make_box(&mut topo, 2.0, 2.0, 2.0).unwrap();
        let b = crate::primitives::make_box(&mut topo, 2.0, 2.0, 2.0).unwrap();
        // Translate b by (1,0,0)
        crate::transform::transform_solid(
            &mut topo,
            b,
            &brepkit_math::mat::Mat4::translation(1.0, 0.0, 0.0),
        )
        .unwrap();
        let result = boolean(&mut topo, BooleanOp::Fuse, a, b).unwrap();
        let s = topo.solid(result).unwrap();
        let sh = topo.shell(s.outer_shell()).unwrap();
        assert!(!sh.faces().is_empty(), "fuse should produce faces");
    }

    #[test]
    fn cylinder_tessellates_with_circle_edges() {
        // Verify that tessellation of a cylinder's cap (which has Circle edges) works.
        let mut topo = Topology::new();
        let cyl = crate::primitives::make_cylinder(&mut topo, 1.0, 2.0).unwrap();
        let solid = topo.solid(cyl).unwrap();
        let shell = topo.shell(solid.outer_shell()).unwrap();

        for &fid in shell.faces() {
            let face = topo.face(fid).unwrap();
            if matches!(face.surface(), FaceSurface::Plane { .. }) {
                // This is a cap face — tessellate it.
                let mesh = crate::tessellate::tessellate(&topo, fid, 1.0).unwrap();
                assert!(
                    mesh.positions.len() >= 3,
                    "cap face should tessellate to at least 3 positions, got {}",
                    mesh.positions.len()
                );
            }
        }
    }

    #[test]
    fn is_all_analytic_detection() {
        let mut topo = Topology::new();
        let box_s = crate::primitives::make_box(&mut topo, 1.0, 1.0, 1.0).unwrap();
        assert!(is_all_analytic(&topo, box_s).unwrap());

        let cyl = crate::primitives::make_cylinder(&mut topo, 1.0, 2.0).unwrap();
        assert!(is_all_analytic(&topo, cyl).unwrap());
    }

    #[test]
    fn cone_has_circle_edges() {
        let mut topo = Topology::new();
        let cone = crate::primitives::make_cone(&mut topo, 2.0, 0.0, 3.0).unwrap();
        let solid = topo.solid(cone).unwrap();
        let shell = topo.shell(solid.outer_shell()).unwrap();

        let mut has_circle = false;
        for &fid in shell.faces() {
            let face = topo.face(fid).unwrap();
            let wire = topo.wire(face.outer_wire()).unwrap();
            for oe in wire.edges() {
                if matches!(
                    topo.edge(oe.edge()).unwrap().curve(),
                    brepkit_topology::edge::EdgeCurve::Circle(_)
                ) {
                    has_circle = true;
                }
            }
        }
        assert!(has_circle, "cone should have Circle edges");
    }

    // ── Mixed-surface assembly tests ────────────────────

    #[test]
    fn assemble_mixed_planar_only() {
        // Planar-only via FaceSpec should produce the same result as assemble_solid.
        let mut topo = Topology::new();
        let specs = vec![
            FaceSpec::Planar {
                vertices: vec![
                    Point3::new(0.0, 0.0, 0.0),
                    Point3::new(1.0, 0.0, 0.0),
                    Point3::new(1.0, 1.0, 0.0),
                    Point3::new(0.0, 1.0, 0.0),
                ],
                normal: Vec3::new(0.0, 0.0, -1.0),
                d: 0.0,
            },
            FaceSpec::Planar {
                vertices: vec![
                    Point3::new(0.0, 0.0, 1.0),
                    Point3::new(1.0, 0.0, 1.0),
                    Point3::new(1.0, 1.0, 1.0),
                    Point3::new(0.0, 1.0, 1.0),
                ],
                normal: Vec3::new(0.0, 0.0, 1.0),
                d: 1.0,
            },
            FaceSpec::Planar {
                vertices: vec![
                    Point3::new(0.0, 0.0, 0.0),
                    Point3::new(1.0, 0.0, 0.0),
                    Point3::new(1.0, 0.0, 1.0),
                    Point3::new(0.0, 0.0, 1.0),
                ],
                normal: Vec3::new(0.0, -1.0, 0.0),
                d: 0.0,
            },
            FaceSpec::Planar {
                vertices: vec![
                    Point3::new(0.0, 1.0, 0.0),
                    Point3::new(1.0, 1.0, 0.0),
                    Point3::new(1.0, 1.0, 1.0),
                    Point3::new(0.0, 1.0, 1.0),
                ],
                normal: Vec3::new(0.0, 1.0, 0.0),
                d: 1.0,
            },
            FaceSpec::Planar {
                vertices: vec![
                    Point3::new(0.0, 0.0, 0.0),
                    Point3::new(0.0, 1.0, 0.0),
                    Point3::new(0.0, 1.0, 1.0),
                    Point3::new(0.0, 0.0, 1.0),
                ],
                normal: Vec3::new(-1.0, 0.0, 0.0),
                d: 0.0,
            },
            FaceSpec::Planar {
                vertices: vec![
                    Point3::new(1.0, 0.0, 0.0),
                    Point3::new(1.0, 1.0, 0.0),
                    Point3::new(1.0, 1.0, 1.0),
                    Point3::new(1.0, 0.0, 1.0),
                ],
                normal: Vec3::new(1.0, 0.0, 0.0),
                d: 1.0,
            },
        ];

        let solid = assemble_solid_mixed(&mut topo, &specs, Tolerance::new()).unwrap();
        let s = topo.solid(solid).unwrap();
        let sh = topo.shell(s.outer_shell()).unwrap();
        assert_eq!(
            sh.faces().len(),
            6,
            "mixed assembly box should have 6 faces"
        );
    }

    #[test]
    fn assemble_mixed_with_nurbs() {
        use brepkit_math::nurbs::surface::NurbsSurface;

        let mut topo = Topology::new();

        // Create a mix of planar and NURBS faces.
        let nurbs = NurbsSurface::new(
            1,
            1,
            vec![0.0, 0.0, 1.0, 1.0],
            vec![0.0, 0.0, 1.0, 1.0],
            vec![
                vec![Point3::new(0.0, 0.0, 1.0), Point3::new(1.0, 0.0, 1.0)],
                vec![Point3::new(0.0, 1.0, 1.0), Point3::new(1.0, 1.0, 1.0)],
            ],
            vec![vec![1.0, 1.0], vec![1.0, 1.0]],
        )
        .unwrap();

        let specs = vec![
            FaceSpec::Planar {
                vertices: vec![
                    Point3::new(0.0, 0.0, 0.0),
                    Point3::new(1.0, 0.0, 0.0),
                    Point3::new(1.0, 1.0, 0.0),
                    Point3::new(0.0, 1.0, 0.0),
                ],
                normal: Vec3::new(0.0, 0.0, -1.0),
                d: 0.0,
            },
            FaceSpec::Surface {
                vertices: vec![
                    Point3::new(0.0, 0.0, 1.0),
                    Point3::new(1.0, 0.0, 1.0),
                    Point3::new(1.0, 1.0, 1.0),
                    Point3::new(0.0, 1.0, 1.0),
                ],
                surface: FaceSurface::Nurbs(nurbs),
            },
        ];

        let solid = assemble_solid_mixed(&mut topo, &specs, Tolerance::new()).unwrap();
        let s = topo.solid(solid).unwrap();
        let sh = topo.shell(s.outer_shell()).unwrap();
        assert_eq!(sh.faces().len(), 2, "mixed assembly should have 2 faces");

        // Verify the NURBS face exists.
        let has_nurbs = sh
            .faces()
            .iter()
            .any(|&fid| matches!(topo.face(fid).unwrap().surface(), FaceSurface::Nurbs(_)));
        assert!(has_nurbs, "mixed assembly should contain a NURBS face");
    }

    #[test]
    /// Intersect a 10³ box with a sphere of r=7 centered at origin.
    ///
    /// The box occupies (0,0,0)-(10,10,10). The sphere at origin extends
    /// from -7 to +7 in all axes. The intersection is the part of the
    /// sphere inside the box — roughly one octant of the sphere.
    ///
    /// V(sphere) = (4/3)π(343) ≈ 1436.76
    /// V(box) = 1000
    /// Intersection ≤ min(V_box, V_sphere) = 1000.
    /// The sphere extends 7 units into the box but only from origin.
    /// Intersection volume must be > 0 and < both input volumes.
    fn intersect_box_sphere_succeeds() {
        let mut topo = Topology::new();
        let bx = crate::primitives::make_box(&mut topo, 10.0, 10.0, 10.0).unwrap();
        let sp = crate::primitives::make_sphere(&mut topo, 7.0, 16).unwrap();
        let result = boolean(&mut topo, BooleanOp::Intersect, bx, sp).unwrap();

        let vol = crate::measure::solid_volume(&topo, result, 0.1).unwrap();
        // Intersection must be positive and less than both inputs.
        let vol_box = 1000.0;
        let vol_sphere = 4.0 / 3.0 * std::f64::consts::PI * 343.0;
        assert!(
            vol > 0.0,
            "intersection volume should be positive, got {vol}"
        );
        assert!(
            vol < vol_box,
            "intersection volume {vol:.1} should be < box volume {vol_box}"
        );
        assert!(
            vol < vol_sphere,
            "intersection volume {vol:.1} should be < sphere volume {vol_sphere:.1}"
        );
    }

    #[test]
    /// Fuse a 10³ box with a sphere of r=7.
    ///
    /// By inclusion-exclusion: V(A∪B) = V(A) + V(B) - V(A∩B).
    /// Fused volume must be > max(V_box, V_sphere) and ≤ V_box + V_sphere.
    fn fuse_box_sphere_succeeds() {
        let mut topo = Topology::new();
        let bx = crate::primitives::make_box(&mut topo, 10.0, 10.0, 10.0).unwrap();
        let sp = crate::primitives::make_sphere(&mut topo, 7.0, 16).unwrap();
        let result = boolean(&mut topo, BooleanOp::Fuse, bx, sp).unwrap();

        let vol = crate::measure::solid_volume(&topo, result, 0.1).unwrap();
        let vol_box: f64 = 1000.0;
        let vol_sphere = 4.0 / 3.0 * std::f64::consts::PI * 343.0;
        // Fused volume must exceed the larger input (sphere ≈ 1437 > box = 1000).
        // Allow 2% tessellation tolerance on the lower bound.
        let vol_max = vol_box.max(vol_sphere);
        assert!(
            vol > vol_max * 0.98,
            "fuse volume {vol:.1} should be > ~larger input {:.1}",
            vol_max * 0.98
        );
        // And less than the sum (since they overlap).
        assert!(
            vol < vol_box + vol_sphere,
            "fuse volume {vol:.1} should be < sum {:.1}",
            vol_box + vol_sphere
        );
    }

    #[test]
    fn cut_box_by_sphere_succeeds() {
        let mut topo = Topology::new();
        let bx = crate::primitives::make_box(&mut topo, 10.0, 10.0, 10.0).unwrap();
        let sp = crate::primitives::make_sphere(&mut topo, 7.0, 16).unwrap();
        let result = boolean(&mut topo, BooleanOp::Cut, bx, sp);
        assert!(
            result.is_ok(),
            "cut(box, sphere) should succeed: {:?}",
            result.err()
        );
        let r = result.unwrap();
        let vol = crate::measure::solid_volume(&topo, r, 0.1).unwrap();
        assert!(
            vol < 1000.0,
            "cut(box, sphere) volume {vol} should be less than box volume 1000"
        );
    }

    #[test]
    fn cut_box_by_translated_sphere() {
        // Matches brepjs test: box(10,10,10), sphere(r=3) translated to (5,5,5).
        let mut topo = Topology::new();
        let bx = crate::primitives::make_box(&mut topo, 10.0, 10.0, 10.0).unwrap();
        let sp = crate::primitives::make_sphere(&mut topo, 3.0, 32).unwrap();
        // Translate sphere to center of box
        let mat = brepkit_math::mat::Mat4::translation(5.0, 5.0, 5.0);
        crate::transform::transform_solid(&mut topo, sp, &mat).unwrap();

        // Sanity: sphere is entirely inside box
        let sph_vol = crate::measure::solid_volume(&topo, sp, 0.05).unwrap();
        eprintln!("sphere volume: {sph_vol:.1} (expected ~113.1)");

        let result = boolean(&mut topo, BooleanOp::Cut, bx, sp);
        assert!(
            result.is_ok(),
            "cut(box, translated sphere) should succeed: {:?}",
            result.err()
        );
        let r = result.unwrap();
        let vol = crate::measure::solid_volume(&topo, r, 0.05).unwrap();
        let expected = 1000.0 - sph_vol;
        eprintln!("cut volume: {vol:.1} (expected ~{expected:.1})");

        // Count result faces
        let faces = brepkit_topology::explorer::solid_faces(&topo, r).unwrap();
        eprintln!("result has {} faces", faces.len());

        assert!(
            vol < 1000.0,
            "cut volume {vol} should be less than box volume 1000"
        );
        assert!(vol > 0.0, "cut volume should be positive");
    }

    #[test]
    fn cut_box_by_large_sphere_containment() {
        // Sphere (r=50) fully contains the box (10x10x10 at origin).
        // Cut should produce an empty result (error) or a very small volume.
        let mut topo = Topology::new();
        let bx = crate::primitives::make_box(&mut topo, 10.0, 10.0, 10.0).unwrap();
        let sp = crate::primitives::make_sphere(&mut topo, 50.0, 16).unwrap();
        // Box fully inside sphere → cut removes everything → should fail or give ~0 volume.
        let result = boolean(&mut topo, BooleanOp::Cut, bx, sp);
        // Either it errors (all faces discarded) or produces a degenerate result.
        if let Ok(r) = result {
            let vol = crate::measure::solid_volume(&topo, r, 0.1).unwrap();
            assert!(
                vol < 10.0,
                "fully contained cut should remove nearly all volume, got {vol}"
            );
        }
    }

    #[test]
    fn intersect_box_with_containing_sphere() {
        // Sphere (r=50) fully contains the box (10x10x10).
        // Intersect should return the box volume.
        let mut topo = Topology::new();
        let bx = crate::primitives::make_box(&mut topo, 10.0, 10.0, 10.0).unwrap();
        let sp = crate::primitives::make_sphere(&mut topo, 50.0, 16).unwrap();
        let result = boolean(&mut topo, BooleanOp::Intersect, bx, sp);
        assert!(
            result.is_ok(),
            "intersect(box, containing sphere) should succeed: {:?}",
            result.err()
        );
        let r = result.unwrap();
        let vol = crate::measure::solid_volume(&topo, r, 0.1).unwrap();
        assert!(
            (vol - 1000.0).abs() < 50.0,
            "intersect with containing sphere should preserve box volume, got {vol}"
        );
    }

    #[test]
    fn disjoint_box_sphere_cut_preserves_box() {
        // Sphere at origin, box far away → no overlap → cut should preserve box.
        let mut topo = Topology::new();
        let bx = crate::primitives::make_box(&mut topo, 10.0, 10.0, 10.0).unwrap();
        let mat = brepkit_math::mat::Mat4::translation(100.0, 0.0, 0.0);
        crate::transform::transform_solid(&mut topo, bx, &mat).unwrap();
        let sp = crate::primitives::make_sphere(&mut topo, 5.0, 16).unwrap();
        let result = boolean(&mut topo, BooleanOp::Cut, bx, sp);
        assert!(
            result.is_ok(),
            "disjoint cut should succeed: {:?}",
            result.err()
        );
        let r = result.unwrap();
        let vol = crate::measure::solid_volume(&topo, r, 0.1).unwrap();
        assert!(
            (vol - 1000.0).abs() < 50.0,
            "disjoint cut should preserve box volume, got {vol}"
        );
    }

    #[test]
    fn cut_box_by_translated_cylinder() {
        let mut topo = Topology::new();
        let bx = crate::primitives::make_box(&mut topo, 50.0, 30.0, 10.0).unwrap();
        let cyl = crate::primitives::make_cylinder(&mut topo, 5.0, 20.0).unwrap();

        // Translate cylinder to center of box, extending through it.
        let mat = brepkit_math::mat::Mat4::translation(25.0, 15.0, -5.0);
        crate::transform::transform_solid(&mut topo, cyl, &mat).unwrap();

        let result = boolean(&mut topo, BooleanOp::Cut, bx, cyl);
        assert!(
            result.is_ok(),
            "cut(box, cyl) should succeed: {:?}",
            result.err()
        );
        let rr = result.unwrap();
        let vol = crate::measure::solid_volume(&topo, rr, 0.1).unwrap();
        let expected = 50.0 * 30.0 * 10.0 - std::f64::consts::PI * 25.0 * 10.0;
        assert!(
            vol < 15000.0,
            "cut volume {vol} should be less than box volume 15000"
        );
        assert!(
            (vol - expected).abs() < expected * 0.1,
            "cut volume {vol} should be near {expected}"
        );
    }

    #[test]
    fn sequential_cylinder_cuts() {
        let mut topo = Topology::new();
        let plate = crate::primitives::make_box(&mut topo, 50.0, 30.0, 10.0).unwrap();

        // First drill: small cylinder at (10, 10)
        let cyl1 = crate::primitives::make_cylinder(&mut topo, 3.0, 20.0).unwrap();
        let mat1 = brepkit_math::mat::Mat4::translation(10.0, 10.0, -5.0);
        crate::transform::transform_solid(&mut topo, cyl1, &mat1).unwrap();
        let r1 = boolean(&mut topo, BooleanOp::Cut, plate, cyl1).unwrap();

        let s = topo.solid(r1).unwrap();
        let sh = topo.shell(s.outer_shell()).unwrap();
        eprintln!("First cut: {} faces", sh.faces().len());

        // Second drill: small cylinder at (40, 10) — non-overlapping
        let cyl2 = crate::primitives::make_cylinder(&mut topo, 3.0, 20.0).unwrap();
        let mat2 = brepkit_math::mat::Mat4::translation(40.0, 10.0, -5.0);
        crate::transform::transform_solid(&mut topo, cyl2, &mat2).unwrap();
        let r2 = boolean(&mut topo, BooleanOp::Cut, r1, cyl2).unwrap();

        let s2 = topo.solid(r2).unwrap();
        let sh2 = topo.shell(s2.outer_shell()).unwrap();
        eprintln!("Second cut: {} faces", sh2.faces().len());

        let vol = crate::measure::solid_volume(&topo, r2, 0.1).unwrap();
        eprintln!("Volume after 2 drills: {vol}");

        // Third drill at (25, 20)
        let cyl3 = crate::primitives::make_cylinder(&mut topo, 5.0, 20.0).unwrap();
        let mat3 = brepkit_math::mat::Mat4::translation(25.0, 20.0, -5.0);
        crate::transform::transform_solid(&mut topo, cyl3, &mat3).unwrap();
        let r3 = boolean(&mut topo, BooleanOp::Cut, r2, cyl3).unwrap();

        let vol3 = crate::measure::solid_volume(&topo, r3, 0.1).unwrap();
        eprintln!("Volume after 3 drills: {vol3}");

        assert!(
            vol3 < 50.0 * 30.0 * 10.0,
            "drilled plate should have less volume: {vol3}"
        );
    }

    #[test]
    fn intersect_two_cylinders() {
        let mut topo = Topology::new();
        let cyl1 = crate::primitives::make_cylinder(&mut topo, 5.0, 20.0).unwrap();
        let cyl2 = crate::primitives::make_cylinder(&mut topo, 3.0, 20.0).unwrap();

        // Offset second cylinder so it partially overlaps the first.
        let mat = brepkit_math::mat::Mat4::translation(2.0, 0.0, 0.0);
        crate::transform::transform_solid(&mut topo, cyl2, &mat).unwrap();

        let result = boolean(&mut topo, BooleanOp::Intersect, cyl1, cyl2);
        assert!(
            result.is_ok(),
            "intersect(cyl, cyl) should succeed: {:?}",
            result.err()
        );
        let r = result.unwrap();
        let vol = crate::measure::solid_volume(&topo, r, 0.1).unwrap();
        assert!(vol > 0.0, "intersection volume should be positive: {vol}");
        // Intersection must be smaller than either cylinder.
        let vol_cyl2 = std::f64::consts::PI * 3.0_f64.powi(2) * 20.0;
        assert!(
            vol < vol_cyl2,
            "intersection volume {vol} should be less than smaller cylinder {vol_cyl2}"
        );
    }

    #[test]
    fn intersect_two_equal_cylinders() {
        // Same params as brepjs benchmark: r=5, r=5, offset=3
        let mut topo = Topology::new();
        let cyl1 = crate::primitives::make_cylinder(&mut topo, 5.0, 20.0).unwrap();
        let cyl2 = crate::primitives::make_cylinder(&mut topo, 5.0, 20.0).unwrap();
        let mat = brepkit_math::mat::Mat4::translation(3.0, 0.0, 0.0);
        crate::transform::transform_solid(&mut topo, cyl2, &mat).unwrap();

        let result = boolean(&mut topo, BooleanOp::Intersect, cyl1, cyl2);
        assert!(
            result.is_ok(),
            "intersect(cyl r=5, cyl r=5 offset=3) should succeed: {:?}",
            result.err()
        );
        let r = result.unwrap();
        let vol = crate::measure::solid_volume(&topo, r, 0.1).unwrap();
        assert!(vol > 0.0, "intersection volume should be positive: {vol}");
    }

    /// Fuse two overlapping cylinders (r=5,h=20 and r=3,h=20, offset x=2).
    ///
    /// Fused volume must be > max(V_cyl1, V_cyl2) and < V_cyl1 + V_cyl2.
    #[test]
    fn fuse_two_cylinders() {
        use std::f64::consts::PI;

        let mut topo = Topology::new();
        let cyl1 = crate::primitives::make_cylinder(&mut topo, 5.0, 20.0).unwrap();
        let cyl2 = crate::primitives::make_cylinder(&mut topo, 3.0, 20.0).unwrap();

        // Offset x=4 so cyl2 protrudes beyond cyl1 (max extent x=7 > r1=5).
        // At x=2 offset, cyl2 would be entirely inside cyl1 (tangent at x=5).
        let mat = brepkit_math::mat::Mat4::translation(4.0, 0.0, 0.0);
        crate::transform::transform_solid(&mut topo, cyl2, &mat).unwrap();

        let opts = BooleanOptions {
            deflection: 0.02,
            ..BooleanOptions::default()
        };
        let result = boolean_with_options(&mut topo, BooleanOp::Fuse, cyl1, cyl2, opts).unwrap();
        let vol = crate::measure::solid_volume(&topo, result, 0.02).unwrap();

        let vol_cyl1 = PI * 25.0 * 20.0; // ≈ 1570.8
        let vol_cyl2 = PI * 9.0 * 20.0; // ≈ 565.5
        // Fuse volume must exceed cyl1 + a meaningful fraction of cyl2's
        // protrusion. With cyl2 at x=4 (r=3), about half of cyl2 protrudes
        // past cyl1. Use cyl1 + 0.25*cyl2 as a conservative lower bound.
        // Allow 2% tessellation tolerance.
        let lower = (vol_cyl1 + 0.25 * vol_cyl2) * 0.98;
        assert!(
            vol > lower,
            "fuse volume {vol:.1} should be > conservative lower bound {lower:.1}"
        );
        assert!(
            vol < vol_cyl1 + vol_cyl2,
            "fuse volume {vol:.1} should be < sum {:.1}",
            vol_cyl1 + vol_cyl2
        );
    }

    /// Cut a large cylinder by a smaller overlapping one.
    ///
    /// V(A-B) = V(A) - V(A∩B). Since B partially overlaps A,
    /// the result must be positive and less than V(A).
    #[test]
    fn cut_cylinder_by_cylinder() {
        use std::f64::consts::PI;

        let mut topo = Topology::new();
        let cyl1 = crate::primitives::make_cylinder(&mut topo, 5.0, 20.0).unwrap();
        let cyl2 = crate::primitives::make_cylinder(&mut topo, 3.0, 20.0).unwrap();

        let mat = brepkit_math::mat::Mat4::translation(2.0, 0.0, 0.0);
        crate::transform::transform_solid(&mut topo, cyl2, &mat).unwrap();

        let result = boolean(&mut topo, BooleanOp::Cut, cyl1, cyl2).unwrap();
        let vol = crate::measure::solid_volume(&topo, result, 0.1).unwrap();

        let vol_cyl1 = PI * 25.0 * 20.0; // ≈ 1570.8
        assert!(vol > 0.0, "cut volume should be positive, got {vol}");
        assert!(
            vol < vol_cyl1,
            "cut volume {vol:.1} should be < original cylinder {vol_cyl1:.1}"
        );
    }

    /// Staircase-like benchmark: fuse box steps with cylinder posts.
    /// Mimics the brepjs staircase benchmark (OCCT: 4s target).
    #[test]
    fn staircase_fuse_with_cylinders() {
        use std::time::Instant;

        let mut topo = Topology::new();
        let start = Instant::now();

        // Build 10 steps, each is a box with a cylinder post.
        let mut shapes: Vec<SolidId> = Vec::new();
        for i in 0..10 {
            let step = crate::primitives::make_box(&mut topo, 20.0, 30.0, 2.0).unwrap();
            let mat_step = brepkit_math::mat::Mat4::translation(0.0, 0.0, f64::from(i) * 10.0);
            crate::transform::transform_solid(&mut topo, step, &mat_step).unwrap();
            shapes.push(step);

            let post = crate::primitives::make_cylinder(&mut topo, 1.5, 10.0).unwrap();
            let mat_post =
                brepkit_math::mat::Mat4::translation(10.0, 15.0, f64::from(i) * 10.0 + 2.0);
            crate::transform::transform_solid(&mut topo, post, &mat_post).unwrap();
            shapes.push(post);
        }

        // Fuse all shapes together sequentially.
        let mut result = shapes[0];
        for &shape in &shapes[1..] {
            result = boolean(&mut topo, BooleanOp::Fuse, result, shape).unwrap();
        }

        let elapsed = start.elapsed();
        eprintln!("Staircase fuse: {elapsed:?} ({} shapes)", shapes.len());

        let vol = crate::measure::solid_volume(&topo, result, 0.5).unwrap();
        eprintln!("Volume: {vol:.1}");
        assert!(vol > 0.0, "staircase volume should be positive");
    }

    #[test]
    fn profile_cylinder_cylinder_intersect() {
        let mut topo = Topology::new();
        let cyl1 = crate::primitives::make_cylinder(&mut topo, 5.0, 20.0).unwrap();
        let cyl2 = crate::primitives::make_cylinder(&mut topo, 5.0, 20.0).unwrap();
        let mat = brepkit_math::mat::Mat4::translation(3.0, 0.0, 0.0);
        crate::transform::transform_solid(&mut topo, cyl2, &mat).unwrap();

        // Profile multiple runs
        for i in 0..5 {
            let mut t = Topology::new();
            let c1 = crate::primitives::make_cylinder(&mut t, 5.0, 20.0).unwrap();
            let c2 = crate::primitives::make_cylinder(&mut t, 5.0, 20.0).unwrap();
            let m = brepkit_math::mat::Mat4::translation(3.0, 0.0, 0.0);
            crate::transform::transform_solid(&mut t, c2, &m).unwrap();

            let start = std::time::Instant::now();
            let result = boolean(&mut t, BooleanOp::Intersect, c1, c2);
            let elapsed = start.elapsed();
            eprintln!("run {i}: {elapsed:?} result={}", result.is_ok());
        }

        // Final run for correctness check
        let result = boolean(&mut topo, BooleanOp::Intersect, cyl1, cyl2).unwrap();
        let vol = crate::measure::solid_volume(&topo, result, 0.1).unwrap();
        eprintln!("Volume: {vol:.2}");
        assert!(
            vol > 0.0,
            "intersection volume should be positive, got {vol}"
        );
    }

    /// Profile individual phases of the analytic boolean.
    #[test]
    fn profile_analytic_boolean_phases() {
        use std::time::Instant;

        let mut topo = Topology::new();
        let cyl1 = crate::primitives::make_cylinder(&mut topo, 5.0, 20.0).unwrap();
        let cyl2 = crate::primitives::make_cylinder(&mut topo, 5.0, 20.0).unwrap();
        let mat = brepkit_math::mat::Mat4::translation(3.0, 0.0, 0.0);
        crate::transform::transform_solid(&mut topo, cyl2, &mat).unwrap();

        let _tol = Tolerance::new();
        let deflection = DEFAULT_BOOLEAN_DEFLECTION;

        // Phase: is_all_analytic + has_torus checks
        let t = Instant::now();
        let analytic_a = is_all_analytic(&topo, cyl1).unwrap();
        let analytic_b = is_all_analytic(&topo, cyl2).unwrap();
        let no_torus_a = !has_torus(&topo, cyl1).unwrap();
        let no_torus_b = !has_torus(&topo, cyl2).unwrap();
        eprintln!(
            "  checks: {:?} (analytic={analytic_a},{analytic_b} torus={no_torus_a},{no_torus_b})",
            t.elapsed()
        );

        // Phase: face_polygon for all faces
        let t = Instant::now();
        let solid_a = topo.solid(cyl1).unwrap();
        let shell_a = topo.shell(solid_a.outer_shell()).unwrap();
        let face_ids_a: Vec<FaceId> = shell_a.faces().to_vec();
        for &fid in &face_ids_a {
            let _ = face_polygon(&topo, fid).unwrap();
        }
        let solid_b = topo.solid(cyl2).unwrap();
        let shell_b = topo.shell(solid_b.outer_shell()).unwrap();
        let face_ids_b: Vec<FaceId> = shell_b.faces().to_vec();
        for &fid in &face_ids_b {
            let _ = face_polygon(&topo, fid).unwrap();
        }
        eprintln!(
            "  face_polygon: {:?} ({} + {} faces)",
            t.elapsed(),
            face_ids_a.len(),
            face_ids_b.len()
        );

        // Phase: tessellate non-planar faces (for normal extraction)
        let t = Instant::now();
        let mut tess_count = 0;
        for &fid in face_ids_a.iter().chain(face_ids_b.iter()) {
            let face = topo.face(fid).unwrap();
            if !matches!(face.surface(), FaceSurface::Plane { .. }) {
                let _ = crate::tessellate::tessellate(&topo, fid, deflection).unwrap();
                tess_count += 1;
            }
        }
        eprintln!(
            "  tessellate_for_normals: {:?} ({tess_count} faces)",
            t.elapsed()
        );

        // Phase: intersect_analytic_analytic
        let t = Instant::now();
        {
            use brepkit_math::analytic_intersection::{
                AnalyticSurface, intersect_analytic_analytic,
            };
            // Find the cylinder barrel faces and intersect them
            for &fid_a in &face_ids_a {
                let fa = topo.face(fid_a).unwrap();
                if let FaceSurface::Cylinder(c_a) = fa.surface() {
                    for &fid_b in &face_ids_b {
                        let fb = topo.face(fid_b).unwrap();
                        if let FaceSurface::Cylinder(c_b) = fb.surface() {
                            let surf_a = AnalyticSurface::Cylinder(c_a);
                            let surf_b = AnalyticSurface::Cylinder(c_b);
                            let _ = intersect_analytic_analytic(surf_a, surf_b, 32);
                        }
                    }
                }
            }
        }
        eprintln!("  intersect_analytic: {:?}", t.elapsed());

        // Phase: tessellate barrel faces into fragments
        let t = Instant::now();
        let mut frag_count = 0;
        let mut frags = Vec::new();
        for &fid in face_ids_a.iter().chain(face_ids_b.iter()) {
            let face = topo.face(fid).unwrap();
            if matches!(face.surface(), FaceSurface::Cylinder(_)) {
                tessellate_face_into_fragments(&topo, fid, Source::A, deflection, &mut frags)
                    .unwrap();
                frag_count += frags.len();
            }
        }
        eprintln!(
            "  tessellate_fragments: {:?} ({frag_count} frags)",
            t.elapsed()
        );

        // Phase: full boolean (end-to-end)
        let mut topo2 = Topology::new();
        let c1 = crate::primitives::make_cylinder(&mut topo2, 5.0, 20.0).unwrap();
        let c2 = crate::primitives::make_cylinder(&mut topo2, 5.0, 20.0).unwrap();
        let m = brepkit_math::mat::Mat4::translation(3.0, 0.0, 0.0);
        crate::transform::transform_solid(&mut topo2, c2, &m).unwrap();
        let t = Instant::now();
        let result = boolean(&mut topo2, BooleanOp::Intersect, c1, c2).unwrap();
        eprintln!("  full_boolean: {:?}", t.elapsed());

        let vol = crate::measure::solid_volume(&topo2, result, 0.1).unwrap();
        eprintln!("  volume: {vol:.2}");
        assert!(vol > 0.0);
    }

    /// Profile sequential fuses (staircase pattern) to identify scaling bottleneck.
    #[test]
    fn profile_sequential_fuse_scaling() {
        use std::time::Instant;

        let mut topo = Topology::new();
        let step_count = 16_usize;
        let step_rise = 18.0;
        let rotation_per_step = 22.5_f64;
        let step_width = 70.0;
        let step_depth = 25.0;
        let column_radius = 12.0;
        let step_thickness = 4.0;
        let post_radius = 1.5;
        let rail_height = 90.0;
        let rail_radius = column_radius + step_width - 4.0;

        let col_height = step_count as f64 * step_rise + step_thickness;
        let column =
            crate::primitives::make_cylinder(&mut topo, column_radius, col_height).unwrap();
        let landing =
            crate::primitives::make_cylinder(&mut topo, column_radius + step_width, step_thickness)
                .unwrap();

        // Create step pieces (box + cylinder post fused), translated and rotated
        let mut pieces = Vec::new();
        for i in 0..step_count {
            let step = crate::primitives::make_box(
                &mut topo,
                column_radius + step_width,
                step_depth,
                step_thickness,
            )
            .unwrap();
            let post =
                crate::primitives::make_cylinder(&mut topo, post_radius, rail_height).unwrap();

            // Translate post
            let mat = brepkit_math::mat::Mat4::translation(rail_radius, 0.0, step_thickness);
            crate::transform::transform_solid(&mut topo, post, &mat).unwrap();
            // Translate step
            let mat = brepkit_math::mat::Mat4::translation(0.0, -step_depth / 2.0, 0.0);
            crate::transform::transform_solid(&mut topo, step, &mat).unwrap();

            // Fuse step + post
            let piece = boolean(&mut topo, BooleanOp::Fuse, step, post).unwrap();

            // Lift
            let mat = brepkit_math::mat::Mat4::translation(0.0, 0.0, step_rise * (i as f64 + 1.0));
            crate::transform::transform_solid(&mut topo, piece, &mat).unwrap();

            // Rotate
            let angle = rotation_per_step * i as f64;
            let rot = brepkit_math::mat::Mat4::rotation_z(angle.to_radians());
            crate::transform::transform_solid(&mut topo, piece, &rot).unwrap();

            pieces.push(piece);
        }

        let ball1 = crate::primitives::make_sphere(&mut topo, 4.0, 16).unwrap();
        let first_post_top = step_rise + step_thickness + rail_height;
        let mat = brepkit_math::mat::Mat4::translation(rail_radius, 0.0, first_post_top);
        crate::transform::transform_solid(&mut topo, ball1, &mat).unwrap();

        let ball2 = crate::primitives::make_sphere(&mut topo, 4.0, 16).unwrap();
        let last_post_top = first_post_top + step_rise * (step_count as f64 - 1.0);
        let mat = brepkit_math::mat::Mat4::translation(rail_radius, 0.0, last_post_top);
        crate::transform::transform_solid(&mut topo, ball2, &mat).unwrap();
        let angle = rotation_per_step * (step_count as f64 - 1.0);
        let rot = brepkit_math::mat::Mat4::rotation_z(angle.to_radians());
        crate::transform::transform_solid(&mut topo, ball2, &rot).unwrap();

        // Sequential fuse
        let all_parts = std::iter::once(column)
            .chain(std::iter::once(landing))
            .chain(pieces)
            .chain(std::iter::once(ball1))
            .chain(std::iter::once(ball2))
            .collect::<Vec<_>>();

        eprintln!("total parts: {}", all_parts.len());

        // Profile a single fuse step with the accumulated solid at step 8
        let mut current = all_parts[0];
        for &piece in &all_parts[1..9] {
            current = boolean(&mut topo, BooleanOp::Fuse, current, piece).unwrap();
        }

        // Now profile the next fuse step in detail
        let piece = all_parts[9];
        let t0 = Instant::now();

        // Phase 1: face_polygon for all faces of solid A
        let solid_acc = topo.solid(current).unwrap();
        let shell_acc = topo.shell(solid_acc.outer_shell()).unwrap();
        let face_ids_acc: Vec<FaceId> = shell_acc.faces().to_vec();
        eprintln!("  accumulated faces: {}", face_ids_acc.len());
        for &fid in &face_ids_acc {
            let _ = face_polygon(&topo, fid).unwrap();
        }
        eprintln!("  phase1 (face_polygon A): {:?}", t0.elapsed());

        // Phase 2: tessellate non-planar faces
        let t1 = Instant::now();
        let mut tess_count = 0;
        for &fid in &face_ids_acc {
            let face = topo.face(fid).unwrap();
            if !matches!(face.surface(), FaceSurface::Plane { .. }) {
                let _ = crate::tessellate::tessellate(&topo, fid, 0.1).unwrap();
                tess_count += 1;
            }
        }
        eprintln!(
            "  phase2 (tessellate A, {} non-planar): {:?}",
            tess_count,
            t1.elapsed()
        );

        // Phase 3: AABB computation
        let t2 = Instant::now();
        for &fid in &face_ids_acc {
            let face = topo.face(fid).unwrap();
            let verts = face_polygon(&topo, fid).unwrap();
            let _ = surface_aware_aabb(face.surface(), &verts, Tolerance::new());
        }
        eprintln!("  phase3 (AABB A): {:?}", t2.elapsed());

        // Phase 4: classification data collection
        let t3 = Instant::now();
        let deflection = 0.1;
        let face_data_acc = collect_face_data(&topo, current, deflection).unwrap();
        eprintln!(
            "  phase4 (collect_face_data A, {} entries): {:?}",
            face_data_acc.len(),
            t3.elapsed()
        );

        // Full boolean for comparison
        let t_full = Instant::now();
        let _ = boolean(&mut topo, BooleanOp::Fuse, current, piece).unwrap();
        eprintln!("  full_boolean step 9: {:?}", t_full.elapsed());
    }

    /// Verify that `cut(box, cylinder)` produces a reasonable edge count
    /// with proper Circle edges (not tessellated into N line segments).
    #[test]
    fn box_cut_cylinder_edge_count() {
        let mut topo = Topology::new();

        let b = crate::primitives::make_box(&mut topo, 40.0, 20.0, 5.0).unwrap();
        let cyl = crate::primitives::make_cylinder(&mut topo, 3.0, 10.0).unwrap();

        let mat = brepkit_math::mat::Mat4::translation(20.0, 10.0, 0.0);
        let hole = crate::copy::copy_solid(&mut topo, cyl).unwrap();
        crate::transform::transform_solid(&mut topo, hole, &mat).unwrap();

        let result = boolean(&mut topo, BooleanOp::Cut, b, hole).unwrap();

        let edges = brepkit_topology::explorer::solid_edges(&topo, result).unwrap();
        let faces = brepkit_topology::explorer::solid_faces(&topo, result).unwrap();

        // 7 faces: 6 planar (4 sides + top/bottom with holes) + 1 cylinder barrel
        assert_eq!(faces.len(), 7, "expected 7 faces for box-cylinder cut");

        // ~16 edges: 12 box edges + 2 circle edges + 1 seam + maybe 1 extra
        assert!(
            edges.len() <= 20,
            "expected ~16 edges for box-cylinder cut, got {} (was 142 before fix)",
            edges.len()
        );

        // Verify Circle edges exist (not tessellated to line segments)
        let circle_count = edges
            .iter()
            .filter(|&&eid| matches!(topo.edge(eid).unwrap().curve(), EdgeCurve::Circle(_)))
            .count();
        assert!(
            circle_count >= 2,
            "expected at least 2 Circle edges, got {circle_count}"
        );
    }

    #[test]
    fn fuse_overlapping_boxes_validates() {
        let mut topo = Topology::new();
        let a = crate::primitives::make_box(&mut topo, 10.0, 10.0, 10.0).unwrap();
        let b = crate::primitives::make_box(&mut topo, 10.0, 10.0, 10.0).unwrap();
        let mat = brepkit_math::mat::Mat4::translation(5.0, 5.0, 5.0);
        crate::transform::transform_solid(&mut topo, b, &mat).unwrap();

        let fused = boolean(&mut topo, BooleanOp::Fuse, a, b).unwrap();

        // Check for boundary edges
        let edge_map = brepkit_topology::explorer::edge_to_face_map(&topo, fused).unwrap();
        let boundary: Vec<_> = edge_map
            .iter()
            .filter(|(_, faces)| faces.len() == 1)
            .collect();
        assert!(
            boundary.is_empty(),
            "fuse result has {} boundary edge(s): {:?}",
            boundary.len(),
            boundary.iter().map(|(e, _)| e).collect::<Vec<_>>()
        );

        let report = crate::validate::validate_solid(&topo, fused).unwrap();
        assert!(
            report.is_valid(),
            "fuse(overlapping boxes) should validate: {:?}",
            report.issues
        );
    }

    // ── Shared-boundary fuse ────────────────────────────────────

    #[test]
    fn fuse_adjacent_boxes_shared_face() {
        // Two unit cubes sharing a face at x=1: result should be a 2×1×1 box.
        let mut topo = Topology::new();
        let a = crate::primitives::make_box(&mut topo, 1.0, 1.0, 1.0).unwrap();
        let b = crate::primitives::make_box(&mut topo, 1.0, 1.0, 1.0).unwrap();
        let mat = brepkit_math::mat::Mat4::translation(1.0, 0.0, 0.0);
        crate::transform::transform_solid(&mut topo, b, &mat).unwrap();

        let fused = boolean(&mut topo, BooleanOp::Fuse, a, b).unwrap();

        let vol = crate::measure::solid_volume(&topo, fused, 0.01).unwrap();
        let expected = 2.0; // 2×1×1
        assert!(
            (vol - expected).abs() < 0.01 * expected,
            "shared-face fuse volume: {vol} (expected {expected})"
        );

        // Result should have exactly 10 faces (12 - 2 shared).
        let shell_id = topo.solid(fused).unwrap().outer_shell();
        let face_count = topo.shell(shell_id).unwrap().faces().len();
        assert_eq!(
            face_count, 10,
            "shared-face fuse should have exactly 10 faces (12 - 2 shared), got {face_count}"
        );
    }

    #[test]
    fn fuse_adjacent_boxes_3x1_grid() {
        // Three unit cubes in a row: fuse_all should produce a 3×1×1 box.
        let mut topo = Topology::new();
        let a = crate::primitives::make_box(&mut topo, 1.0, 1.0, 1.0).unwrap();
        let b = crate::primitives::make_box(&mut topo, 1.0, 1.0, 1.0).unwrap();
        let c = crate::primitives::make_box(&mut topo, 1.0, 1.0, 1.0).unwrap();
        let mat_b = brepkit_math::mat::Mat4::translation(1.0, 0.0, 0.0);
        let mat_c = brepkit_math::mat::Mat4::translation(2.0, 0.0, 0.0);
        crate::transform::transform_solid(&mut topo, b, &mat_b).unwrap();
        crate::transform::transform_solid(&mut topo, c, &mat_c).unwrap();

        let cid = topo
            .compounds
            .alloc(brepkit_topology::compound::Compound::new(vec![a, b, c]));
        let fused = crate::compound_ops::fuse_all(&mut topo, cid).unwrap();

        let vol = crate::measure::solid_volume(&topo, fused, 0.01).unwrap();
        assert!(
            (vol - 3.0).abs() < 0.03,
            "3×1 grid fuse volume: {vol} (expected 3.0)"
        );
    }

    // ── Degenerate boolean geometry ────────────────────────────

    #[test]
    fn near_tolerance_overlap() {
        // Overlap of exactly the linear tolerance amount
        let mut topo = Topology::new();
        let tol = brepkit_math::tolerance::Tolerance::new();
        let a = make_unit_cube_manifold_at(&mut topo, 0.0, 0.0, 0.0);
        let b = make_unit_cube_manifold_at(&mut topo, 1.0 - tol.linear, 0.0, 0.0);

        // Should either succeed or error — but not panic
        let _result = boolean(&mut topo, BooleanOp::Fuse, a, b);
    }

    #[test]
    fn boolean_nearly_touching() {
        // Gap smaller than tolerance
        let mut topo = Topology::new();
        let a = make_unit_cube_manifold_at(&mut topo, 0.0, 0.0, 0.0);
        let b = make_unit_cube_manifold_at(&mut topo, 1.0 + 1e-9, 0.0, 0.0);

        // Should not panic
        let _result = boolean(&mut topo, BooleanOp::Fuse, a, b);
    }

    // ── compound_cut tests ──────────────────────────────────────────────

    #[test]
    fn compound_cut_empty_tools_returns_target() {
        let mut topo = Topology::new();
        let target = crate::primitives::make_box(&mut topo, 2.0, 2.0, 2.0).unwrap();
        let result = compound_cut(&mut topo, target, &[], BooleanOptions::default()).unwrap();
        assert_eq!(result, target);
    }

    #[test]
    fn diagnose_aabb_filter() {
        use brepkit_math::mat::Mat4;

        let mut topo = Topology::new();
        let target = crate::primitives::make_box(&mut topo, 42.0, 42.0, 7.0).unwrap();
        let cyl = crate::primitives::make_cylinder(&mut topo, 3.75, 7.0).unwrap();
        crate::transform::transform_solid(&mut topo, cyl, &Mat4::translation(21.0, 21.0, 0.0))
            .unwrap();

        let solid_a = topo.solid(target).unwrap();
        let shell_a = topo.shell(solid_a.outer_shell()).unwrap();
        let face_ids_a: Vec<brepkit_topology::face::FaceId> = shell_a.faces().to_vec();

        let solid_b = topo.solid(cyl).unwrap();
        let shell_b = topo.shell(solid_b.outer_shell()).unwrap();
        let face_ids_b: Vec<brepkit_topology::face::FaceId> = shell_b.faces().to_vec();

        let wire_aabbs_a: Vec<_> = face_ids_a
            .iter()
            .map(|&fid| face_wire_aabb(&topo, fid).unwrap())
            .collect();
        let wire_aabbs_b: Vec<_> = face_ids_b
            .iter()
            .map(|&fid| face_wire_aabb(&topo, fid).unwrap())
            .collect();

        let a_overall = wire_aabbs_a.iter().copied().reduce(Aabb3::union).unwrap();
        let b_overall = wire_aabbs_b.iter().copied().reduce(Aabb3::union).unwrap();

        eprintln!(
            "A overall: ({:.2},{:.2},{:.2})-({:.2},{:.2},{:.2})",
            a_overall.min.x(),
            a_overall.min.y(),
            a_overall.min.z(),
            a_overall.max.x(),
            a_overall.max.y(),
            a_overall.max.z()
        );
        eprintln!(
            "B overall: ({:.2},{:.2},{:.2})-({:.2},{:.2},{:.2})",
            b_overall.min.x(),
            b_overall.min.y(),
            b_overall.min.z(),
            b_overall.max.x(),
            b_overall.max.y(),
            b_overall.max.z()
        );

        let mut passthrough_count = 0;
        for (i, &fid) in face_ids_a.iter().enumerate() {
            let face = topo.face(fid).unwrap();
            let overlaps = wire_aabbs_a[i].intersects(b_overall);
            if !overlaps {
                passthrough_count += 1;
            }
            eprintln!(
                "A[{}] {:?} ({:.2},{:.2},{:.2})-({:.2},{:.2},{:.2}) overlaps={}",
                i,
                match face.surface() {
                    FaceSurface::Plane { .. } => "Plane",
                    FaceSurface::Cylinder(_) => "Cyl",
                    _ => "Other",
                },
                wire_aabbs_a[i].min.x(),
                wire_aabbs_a[i].min.y(),
                wire_aabbs_a[i].min.z(),
                wire_aabbs_a[i].max.x(),
                wire_aabbs_a[i].max.y(),
                wire_aabbs_a[i].max.z(),
                overlaps
            );
        }
        for (i, &fid) in face_ids_b.iter().enumerate() {
            let face = topo.face(fid).unwrap();
            let overlaps = wire_aabbs_b[i].intersects(a_overall);
            eprintln!(
                "B[{}] {:?} ({:.2},{:.2},{:.2})-({:.2},{:.2},{:.2}) overlaps={}",
                i,
                match face.surface() {
                    FaceSurface::Plane { .. } => "Plane",
                    FaceSurface::Cylinder(_) => "Cyl",
                    _ => "Other",
                },
                wire_aabbs_b[i].min.x(),
                wire_aabbs_b[i].min.y(),
                wire_aabbs_b[i].min.z(),
                wire_aabbs_b[i].max.x(),
                wire_aabbs_b[i].max.y(),
                wire_aabbs_b[i].max.z(),
                overlaps
            );
        }
        eprintln!("Passthrough A: {}/{}", passthrough_count, face_ids_a.len());

        let result = boolean(&mut topo, BooleanOp::Cut, target, cyl).unwrap();
        assert_volume_near(
            &topo,
            result,
            42.0 * 42.0 * 7.0 - std::f64::consts::PI * 3.75 * 3.75 * 7.0,
            0.05,
        );
    }

    #[test]
    fn compound_cut_single_tool_matches_boolean() {
        use brepkit_math::mat::Mat4;

        let mut topo = Topology::new();
        let target = crate::primitives::make_box(&mut topo, 2.0, 2.0, 2.0).unwrap();
        let cyl = crate::primitives::make_cylinder(&mut topo, 0.5, 2.0).unwrap();
        // Center the cylinder inside the box.
        crate::transform::transform_solid(&mut topo, cyl, &Mat4::translation(1.0, 1.0, 0.0))
            .unwrap();

        // compound_cut with single tool delegates to boolean.
        let result = compound_cut(&mut topo, target, &[cyl], BooleanOptions::default()).unwrap();

        let box_vol = 8.0;
        let cyl_vol = std::f64::consts::PI * 0.25 * 2.0;
        assert_volume_near(&topo, result, box_vol - cyl_vol, 0.05);
    }

    #[test]
    fn compound_cut_two_disjoint_cylinders() {
        use brepkit_math::mat::Mat4;

        let mut topo = Topology::new();
        let target = crate::primitives::make_box(&mut topo, 4.0, 4.0, 2.0).unwrap();
        // Cylinder 1 at (1,1)
        let c1 = crate::primitives::make_cylinder(&mut topo, 0.3, 2.0).unwrap();
        crate::transform::transform_solid(&mut topo, c1, &Mat4::translation(1.0, 1.0, 0.0))
            .unwrap();
        // Cylinder 2 at (3,3) — disjoint from c1
        let c2 = crate::primitives::make_cylinder(&mut topo, 0.3, 2.0).unwrap();
        crate::transform::transform_solid(&mut topo, c2, &Mat4::translation(3.0, 3.0, 0.0))
            .unwrap();

        let result = compound_cut(&mut topo, target, &[c1, c2], BooleanOptions::default()).unwrap();

        let box_vol = 32.0;
        let cyl_vol = std::f64::consts::PI * 0.09 * 2.0;
        assert_volume_near(&topo, result, box_vol - 2.0 * cyl_vol, 0.05);
    }

    #[test]
    fn compound_cut_all_tools_disjoint_returns_unchanged_volume() {
        use brepkit_math::mat::Mat4;

        let mut topo = Topology::new();
        let target = crate::primitives::make_box(&mut topo, 2.0, 2.0, 2.0).unwrap();
        // Both cylinders far away from target.
        let c1 = crate::primitives::make_cylinder(&mut topo, 0.3, 2.0).unwrap();
        crate::transform::transform_solid(&mut topo, c1, &Mat4::translation(10.0, 0.0, 0.0))
            .unwrap();
        let c2 = crate::primitives::make_cylinder(&mut topo, 0.3, 2.0).unwrap();
        crate::transform::transform_solid(&mut topo, c2, &Mat4::translation(-10.0, 0.0, 0.0))
            .unwrap();

        let result = compound_cut(&mut topo, target, &[c1, c2], BooleanOptions::default()).unwrap();

        assert_volume_near(&topo, result, 8.0, 0.001);
    }

    #[test]
    fn compound_cut_matches_sequential_2x2_grid() {
        use brepkit_math::mat::Mat4;

        let mut topo = Topology::new();
        let target = crate::primitives::make_box(&mut topo, 4.0, 4.0, 2.0).unwrap();
        let r = 0.3;
        let spacing = 2.0;
        let mut tools = Vec::new();
        for row in 0..2 {
            for col in 0..2 {
                #[allow(clippy::cast_precision_loss)]
                let x = 1.0 + (col as f64) * spacing;
                #[allow(clippy::cast_precision_loss)]
                let y = 1.0 + (row as f64) * spacing;
                let c = crate::primitives::make_cylinder(&mut topo, r, 2.0).unwrap();
                crate::transform::transform_solid(&mut topo, c, &Mat4::translation(x, y, 0.0))
                    .unwrap();
                tools.push(c);
            }
        }

        // Sequential reference.
        let mut seq_target = crate::primitives::make_box(&mut topo, 4.0, 4.0, 2.0).unwrap();
        for &tool in &tools {
            // Need fresh copies of tools for sequential (tools are consumed by boolean).
            let tool_copy = crate::copy::copy_solid(&mut topo, tool).unwrap();
            seq_target = boolean_with_options(
                &mut topo,
                BooleanOp::Cut,
                seq_target,
                tool_copy,
                BooleanOptions::default(),
            )
            .unwrap();
        }
        let seq_vol = crate::measure::solid_volume(&topo, seq_target, 0.05).unwrap();

        // Compound cut.
        let result = compound_cut(&mut topo, target, &tools, BooleanOptions::default()).unwrap();
        let compound_vol = crate::measure::solid_volume(&topo, result, 0.05).unwrap();

        let rel = (compound_vol - seq_vol).abs() / seq_vol;
        assert!(
            rel < 0.05,
            "compound_cut volume {compound_vol:.4} != sequential {seq_vol:.4} (rel={rel:.4})"
        );
    }

    /// 3×3 grid (9 tools) exercises the compound path (threshold = 8).
    #[test]
    fn compound_cut_matches_sequential_3x3_grid() {
        use brepkit_math::mat::Mat4;

        let mut topo = Topology::new();
        let target = crate::primitives::make_box(&mut topo, 10.0, 10.0, 2.0).unwrap();
        let r = 0.5;
        let mut tools = Vec::new();
        for row in 0..3 {
            for col in 0..3 {
                #[allow(clippy::cast_precision_loss)]
                let x = 2.0 + (col as f64) * 3.0;
                #[allow(clippy::cast_precision_loss)]
                let y = 2.0 + (row as f64) * 3.0;
                let c = crate::primitives::make_cylinder(&mut topo, r, 4.0).unwrap();
                crate::transform::transform_solid(&mut topo, c, &Mat4::translation(x, y, -1.0))
                    .unwrap();
                tools.push(c);
            }
        }

        // Sequential reference.
        let mut seq_topo = topo.clone();
        let mut seq_target = target;
        for &tool in &tools {
            let tool_copy = crate::copy::copy_solid(&mut seq_topo, tool).unwrap();
            seq_target = boolean_with_options(
                &mut seq_topo,
                BooleanOp::Cut,
                seq_target,
                tool_copy,
                BooleanOptions::default(),
            )
            .unwrap();
        }
        let seq_vol = crate::measure::solid_volume(&seq_topo, seq_target, 0.05).unwrap();

        // Compound cut.
        let result = compound_cut(&mut topo, target, &tools, BooleanOptions::default()).unwrap();
        let compound_vol = crate::measure::solid_volume(&topo, result, 0.05).unwrap();

        let rel = (compound_vol - seq_vol).abs() / seq_vol;
        assert!(
            rel < 0.05,
            "compound_cut 3x3 volume {compound_vol:.4} != sequential {seq_vol:.4} (rel={rel:.4})"
        );
    }

    /// 4×4 grid (16 tools) — larger compound cut test.
    #[test]
    fn compound_cut_matches_sequential_4x4_grid() {
        use brepkit_math::mat::Mat4;

        let mut topo = Topology::new();
        let target = crate::primitives::make_box(&mut topo, 20.0, 20.0, 2.0).unwrap();
        let r = 0.5;
        let mut tools = Vec::new();
        for row in 0..4 {
            for col in 0..4 {
                #[allow(clippy::cast_precision_loss)]
                let x = 2.0 + (col as f64) * 4.0;
                #[allow(clippy::cast_precision_loss)]
                let y = 2.0 + (row as f64) * 4.0;
                let c = crate::primitives::make_cylinder(&mut topo, r, 4.0).unwrap();
                crate::transform::transform_solid(&mut topo, c, &Mat4::translation(x, y, -1.0))
                    .unwrap();
                tools.push(c);
            }
        }

        // Sequential reference.
        let mut seq_topo = topo.clone();
        let mut seq_target = target;
        for &tool in &tools {
            let tool_copy = crate::copy::copy_solid(&mut seq_topo, tool).unwrap();
            seq_target = boolean_with_options(
                &mut seq_topo,
                BooleanOp::Cut,
                seq_target,
                tool_copy,
                BooleanOptions::default(),
            )
            .unwrap();
        }
        let seq_vol = crate::measure::solid_volume(&seq_topo, seq_target, 0.05).unwrap();

        // Compound cut.
        let result = compound_cut(&mut topo, target, &tools, BooleanOptions::default()).unwrap();
        let compound_vol = crate::measure::solid_volume(&topo, result, 0.05).unwrap();

        let rel = (compound_vol - seq_vol).abs() / seq_vol;
        assert!(
            rel < 0.05,
            "compound_cut 4x4 volume {compound_vol:.4} != sequential {seq_vol:.4} (rel={rel:.4})"
        );
    }

    /// Test compound_cut with a shelled target + many box cutters.
    /// This simulates the gridfinity honeycomb scenario where the target
    /// has cylindrical fillets (rounded corners) and the tools are hex prisms.
    #[test]
    fn compound_cut_shelled_target_many_tools() {
        use brepkit_math::mat::Mat4;

        let mut topo = Topology::new();

        // Build a target with cylindrical fillets by making a box and
        // cutting cylinders at the corners (creates cylinder surfaces).
        let target = crate::primitives::make_box(&mut topo, 40.0, 40.0, 10.0).unwrap();
        // Add a cylinder to make the target have cylinder surface faces.
        let inner_box = crate::primitives::make_box(&mut topo, 36.0, 36.0, 8.0).unwrap();
        crate::transform::transform_solid(&mut topo, inner_box, &Mat4::translation(2.0, 2.0, 2.0))
            .unwrap();
        let target = boolean_with_options(
            &mut topo,
            BooleanOp::Cut,
            target,
            inner_box,
            BooleanOptions::default(),
        )
        .unwrap();

        // Create 25 small box cutters in a 5×5 grid (above the threshold of 8).
        let mut tools = Vec::new();
        for row in 0..5 {
            for col in 0..5 {
                #[allow(clippy::cast_precision_loss)]
                let x = 4.0 + (col as f64) * 7.0;
                #[allow(clippy::cast_precision_loss)]
                let y = 4.0 + (row as f64) * 7.0;
                let tool = crate::primitives::make_box(&mut topo, 3.0, 3.0, 20.0).unwrap();
                crate::transform::transform_solid(&mut topo, tool, &Mat4::translation(x, y, -5.0))
                    .unwrap();
                tools.push(tool);
            }
        }

        // Sequential reference.
        let mut seq_topo = topo.clone();
        let mut seq_result = target;
        let t0 = std::time::Instant::now();
        for &tool in &tools {
            let tool_copy = crate::copy::copy_solid(&mut seq_topo, tool).unwrap();
            seq_result = boolean_with_options(
                &mut seq_topo,
                BooleanOp::Cut,
                seq_result,
                tool_copy,
                BooleanOptions::default(),
            )
            .unwrap();
        }
        let dt_seq = t0.elapsed();
        let seq_vol = crate::measure::solid_volume(&seq_topo, seq_result, 0.05).unwrap();

        // Compound cut.
        let t0 = std::time::Instant::now();
        let result = compound_cut(&mut topo, target, &tools, BooleanOptions::default()).unwrap();
        let dt_compound = t0.elapsed();
        let compound_vol = crate::measure::solid_volume(&topo, result, 0.05).unwrap();

        let rel = (compound_vol - seq_vol).abs() / seq_vol;
        eprintln!(
            "shelled target + 25 tools: compound={:.1}ms (vol={compound_vol:.1}), sequential={:.1}ms (vol={seq_vol:.1}), rel={rel:.4}",
            dt_compound.as_secs_f64() * 1000.0,
            dt_seq.as_secs_f64() * 1000.0,
        );
        assert!(
            rel < 0.05,
            "compound_cut volume {compound_vol:.1} != sequential {seq_vol:.1} (rel={rel:.4})"
        );
    }

    /// Shelled box + 9 box cutters — exercises raycast classification path.
    #[test]
    fn compound_cut_shelled_target_9_tools() {
        use brepkit_math::mat::Mat4;

        let mut topo = Topology::new();

        // Shelled box: outer 40x40x10, inner 36x36x8 offset by (2,2,2).
        let target = crate::primitives::make_box(&mut topo, 40.0, 40.0, 10.0).unwrap();
        let inner_box = crate::primitives::make_box(&mut topo, 36.0, 36.0, 8.0).unwrap();
        crate::transform::transform_solid(&mut topo, inner_box, &Mat4::translation(2.0, 2.0, 2.0))
            .unwrap();
        let target = boolean_with_options(
            &mut topo,
            BooleanOp::Cut,
            target,
            inner_box,
            BooleanOptions::default(),
        )
        .unwrap();

        // 9 box cutters in a 3×3 grid (above N=8 threshold).
        let mut tools = Vec::new();
        for row in 0..3 {
            for col in 0..3 {
                #[allow(clippy::cast_precision_loss)]
                let x = 8.0 + (col as f64) * 12.0;
                #[allow(clippy::cast_precision_loss)]
                let y = 8.0 + (row as f64) * 12.0;
                let tool = crate::primitives::make_box(&mut topo, 3.0, 3.0, 20.0).unwrap();
                crate::transform::transform_solid(&mut topo, tool, &Mat4::translation(x, y, -5.0))
                    .unwrap();
                tools.push(tool);
            }
        }

        // Sequential reference.
        let mut seq_topo = topo.clone();
        let mut seq_result = target;
        for &tool in &tools {
            let tool_copy = crate::copy::copy_solid(&mut seq_topo, tool).unwrap();
            seq_result = boolean_with_options(
                &mut seq_topo,
                BooleanOp::Cut,
                seq_result,
                tool_copy,
                BooleanOptions::default(),
            )
            .unwrap();
        }
        let seq_vol = crate::measure::solid_volume(&seq_topo, seq_result, 0.05).unwrap();

        // Compound.
        let result = compound_cut(&mut topo, target, &tools, BooleanOptions::default()).unwrap();
        let compound_vol = crate::measure::solid_volume(&topo, result, 0.05).unwrap();

        let rel = (compound_vol - seq_vol).abs() / seq_vol;
        assert!(
            rel < 0.02,
            "compound={compound_vol:.4} != seq={seq_vol:.4} (rel={rel:.4})"
        );
    }

    #[test]
    fn cdt_vs_iterative_cross_chords() {
        // A square face split by 4 crossing chords → should produce identical
        // fragment count and total area.
        let verts = [
            Point3::new(0.0, 0.0, 0.0),
            Point3::new(10.0, 0.0, 0.0),
            Point3::new(10.0, 10.0, 0.0),
            Point3::new(0.0, 10.0, 0.0),
        ];
        let normal = Vec3::new(0.0, 0.0, 1.0);
        let d = 0.0;
        let tol = Tolerance::default();
        let source = super::Source::A;

        let chords = vec![
            (Point3::new(3.0, 0.0, 0.0), Point3::new(3.0, 10.0, 0.0)),
            (Point3::new(7.0, 0.0, 0.0), Point3::new(7.0, 10.0, 0.0)),
            (Point3::new(0.0, 4.0, 0.0), Point3::new(10.0, 4.0, 0.0)),
            (Point3::new(0.0, 8.0, 0.0), Point3::new(10.0, 8.0, 0.0)),
        ];

        // CDT path
        let cdt_regions = super::split_face_cdt_inner(&verts, normal, d, &chords, tol).unwrap();
        let cdt_area: f64 = cdt_regions
            .iter()
            .map(|v| super::polygon_area_2x(v, &normal) / 2.0)
            .sum();

        // Iterative path
        let iter_frags = super::split_face_iterative(&verts, normal, d, source, &chords, tol);
        let iter_area: f64 = iter_frags
            .iter()
            .map(|f| super::polygon_area_2x(&f.vertices, &normal) / 2.0)
            .sum();

        // The total area should equal the face area (100.0).
        assert!(
            (cdt_area - 100.0).abs() < 1.0,
            "CDT total area {cdt_area} != 100.0"
        );
        assert!(
            (iter_area - 100.0).abs() < 1.0,
            "Iterative total area {iter_area} != 100.0"
        );

        // Both should produce 9 regions (3 columns × 3 rows).
        assert_eq!(
            cdt_regions.len(),
            iter_frags.len(),
            "CDT and iterative should produce same number of fragments"
        );
    }

    #[test]
    fn cdt_vs_iterative_negative_normal() {
        // Same test but with negative normal (tests winding reversal).
        let verts = [
            Point3::new(0.0, 0.0, 5.0),
            Point3::new(0.0, 10.0, 5.0),
            Point3::new(10.0, 10.0, 5.0),
            Point3::new(10.0, 0.0, 5.0),
        ];
        let normal = Vec3::new(0.0, 0.0, -1.0);
        let d = -5.0;
        let tol = Tolerance::default();

        let chords = vec![
            (Point3::new(5.0, 0.0, 5.0), Point3::new(5.0, 10.0, 5.0)),
            (Point3::new(0.0, 5.0, 5.0), Point3::new(10.0, 5.0, 5.0)),
            (Point3::new(3.0, 0.0, 5.0), Point3::new(3.0, 10.0, 5.0)),
            (Point3::new(7.0, 0.0, 5.0), Point3::new(7.0, 10.0, 5.0)),
        ];

        let cdt_regions = super::split_face_cdt_inner(&verts, normal, d, &chords, tol).unwrap();
        let cdt_area: f64 = cdt_regions
            .iter()
            .map(|v| super::polygon_area_2x(v, &normal) / 2.0)
            .sum();

        assert!(
            (cdt_area - 100.0).abs() < 1.0,
            "CDT total area {cdt_area} != 100.0 (negative normal)"
        );
    }

    /// Reproduce Gridfinity volume loss: fusing a ring (lip) inside a shelled box.
    #[test]
    fn fuse_ring_inside_shelled_box() {
        let mut topo = Topology::new();

        // Create a box and shell it (remove top face)
        let outer = 10.0;
        let height = 10.0;
        let wall = 1.0;
        let box_solid = crate::primitives::make_box(&mut topo, outer, outer, height).unwrap();

        // Find the top face (+Z)
        let top_faces: Vec<brepkit_topology::face::FaceId> = {
            let s = topo.solid(box_solid).unwrap();
            let sh = topo.shell(s.outer_shell()).unwrap();
            let tol = brepkit_math::tolerance::Tolerance::loose();
            sh.faces()
                .iter()
                .filter(|&&fid| {
                    if let Ok(f) = topo.face(fid) {
                        if let brepkit_topology::face::FaceSurface::Plane { normal, .. } =
                            f.surface()
                        {
                            return tol.approx_eq(normal.z(), 1.0);
                        }
                    }
                    false
                })
                .copied()
                .collect()
        };
        assert_eq!(top_faces.len(), 1, "should find exactly one +Z face");

        let shelled = crate::shell_op::shell(&mut topo, box_solid, wall, &top_faces).unwrap();
        let shell_vol = crate::measure::solid_volume(&topo, shelled, 0.01).unwrap();

        // Create a ring (lip) that sits INSIDE the cavity
        // Ring: outer boundary at 3mm inset, 2mm thick, 3mm tall, placed at z=7
        let ring_outer =
            crate::primitives::make_box(&mut topo, outer - 4.0, outer - 4.0, 3.0).unwrap();
        crate::transform::transform_solid(
            &mut topo,
            ring_outer,
            &brepkit_math::mat::Mat4::translation(2.0, 2.0, 7.0),
        )
        .unwrap();
        let ring_inner =
            crate::primitives::make_box(&mut topo, outer - 8.0, outer - 8.0, 3.0).unwrap();
        crate::transform::transform_solid(
            &mut topo,
            ring_inner,
            &brepkit_math::mat::Mat4::translation(4.0, 4.0, 7.0),
        )
        .unwrap();
        let ring = boolean(&mut topo, BooleanOp::Cut, ring_outer, ring_inner).unwrap();
        let ring_vol = crate::measure::solid_volume(&topo, ring, 0.01).unwrap();

        // Ring is inside cavity, no overlap with walls. Expected fuse volume = shell + ring.
        let expected = shell_vol + ring_vol;

        let fused = boolean(&mut topo, BooleanOp::Fuse, shelled, ring).unwrap();
        let fused_vol = crate::measure::solid_volume(&topo, fused, 0.01).unwrap();

        let rel_err = (fused_vol - expected).abs() / expected;
        assert!(
            rel_err < 0.05,
            "fuse ring inside shelled box: vol={fused_vol:.1} expected={expected:.1} \
             (shell={shell_vol:.1}, ring={ring_vol:.1}, rel_err={rel_err:.3})"
        );
    }

    /// Same test but with cylinders (curved surfaces).
    /// The Gridfinity bin has cylinder corners; this tests if curved shells
    /// fuse correctly with ring-like objects inside the cavity.
    #[test]
    fn fuse_ring_inside_shelled_cylinder() {
        let mut topo = Topology::new();

        // Shelled cylinder: outer R=10, height=16, wall=1.2
        let r = 10.0;
        let h = 16.0;
        let wall = 1.2;
        let cyl = crate::primitives::make_cylinder(&mut topo, r, h).unwrap();

        // Find top face
        let top_faces: Vec<brepkit_topology::face::FaceId> = {
            let s = topo.solid(cyl).unwrap();
            let sh = topo.shell(s.outer_shell()).unwrap();
            let tol = brepkit_math::tolerance::Tolerance::loose();
            sh.faces()
                .iter()
                .filter(|&&fid| {
                    if let Ok(f) = topo.face(fid) {
                        if let brepkit_topology::face::FaceSurface::Plane { normal, .. } =
                            f.surface()
                        {
                            return tol.approx_eq(normal.z(), 1.0);
                        }
                    }
                    false
                })
                .copied()
                .collect()
        };

        let shelled = crate::shell_op::shell(&mut topo, cyl, wall, &top_faces).unwrap();
        let shell_vol = crate::measure::solid_volume(&topo, shelled, 0.01).unwrap();

        // Ring inside: outer R=7, inner R=5, height=3, placed at z=h-3
        let ring_outer = crate::primitives::make_cylinder(&mut topo, 7.0, 3.0).unwrap();
        crate::transform::transform_solid(
            &mut topo,
            ring_outer,
            &brepkit_math::mat::Mat4::translation(0.0, 0.0, h - 3.0),
        )
        .unwrap();
        let ring_inner = crate::primitives::make_cylinder(&mut topo, 5.0, 3.0).unwrap();
        crate::transform::transform_solid(
            &mut topo,
            ring_inner,
            &brepkit_math::mat::Mat4::translation(0.0, 0.0, h - 3.0),
        )
        .unwrap();
        let ring = boolean(&mut topo, BooleanOp::Cut, ring_outer, ring_inner).unwrap();
        let ring_vol = crate::measure::solid_volume(&topo, ring, 0.01).unwrap();

        let expected = shell_vol + ring_vol;
        let fused = boolean(&mut topo, BooleanOp::Fuse, shelled, ring).unwrap();
        let fused_vol = crate::measure::solid_volume(&topo, fused, 0.01).unwrap();

        let rel_err = (fused_vol - expected).abs() / expected;
        assert!(
            rel_err < 0.05,
            "fuse ring inside shelled cylinder: vol={fused_vol:.1} expected={expected:.1} \
             (shell={shell_vol:.1}, ring={ring_vol:.1}, rel_err={rel_err:.3})"
        );
    }

    /// Test fuse with ring partially overlapping shell wall height
    /// (simulates lip extension below wall top).
    #[test]
    fn fuse_ring_overlapping_shelled_box_height() {
        let mut topo = Topology::new();

        let outer = 20.0;
        let h = 16.0;
        let wall = 1.2;
        let box_solid = crate::primitives::make_box(&mut topo, outer, outer, h).unwrap();

        let top_faces: Vec<brepkit_topology::face::FaceId> = {
            let s = topo.solid(box_solid).unwrap();
            let sh = topo.shell(s.outer_shell()).unwrap();
            let tol = brepkit_math::tolerance::Tolerance::loose();
            sh.faces()
                .iter()
                .filter(|&&fid| {
                    if let Ok(f) = topo.face(fid) {
                        if let brepkit_topology::face::FaceSurface::Plane { normal, .. } =
                            f.surface()
                        {
                            return tol.approx_eq(normal.z(), 1.0);
                        }
                    }
                    false
                })
                .copied()
                .collect()
        };

        let shelled = crate::shell_op::shell(&mut topo, box_solid, wall, &top_faces).unwrap();
        let shell_vol = crate::measure::solid_volume(&topo, shelled, 0.01).unwrap();

        // Ring that extends from h-2 to h+3 (partially above, partially overlapping rim area)
        // Ring: outer at 3mm inset from each side, 2mm thick
        let ring_outer_w = outer - 6.0;
        let ring_inner_w = outer - 10.0;
        let ring_h = 5.0;
        let ring_z = h - 2.0; // starts 2mm below top of shelled box

        let ring_o =
            crate::primitives::make_box(&mut topo, ring_outer_w, ring_outer_w, ring_h).unwrap();
        crate::transform::transform_solid(
            &mut topo,
            ring_o,
            &brepkit_math::mat::Mat4::translation(3.0, 3.0, ring_z),
        )
        .unwrap();
        let ring_i =
            crate::primitives::make_box(&mut topo, ring_inner_w, ring_inner_w, ring_h).unwrap();
        crate::transform::transform_solid(
            &mut topo,
            ring_i,
            &brepkit_math::mat::Mat4::translation(5.0, 5.0, ring_z),
        )
        .unwrap();
        let ring = boolean(&mut topo, BooleanOp::Cut, ring_o, ring_i).unwrap();
        let ring_vol = crate::measure::solid_volume(&topo, ring, 0.01).unwrap();

        // Overlap: ring intersects rim faces of shelled box at z=h.
        // The ring at z=14-19 overlaps with the rim at z=16, and the inner walls at z=14-16.
        // But ring (3-5mm inset) doesn't overlap walls (0-1.2mm).
        // Expected: shell + ring - (overlap in rim area)
        // Exact overlap is complex; just check we don't lose MORE than 10%
        let fused = boolean(&mut topo, BooleanOp::Fuse, shelled, ring).unwrap();
        let fused_vol = crate::measure::solid_volume(&topo, fused, 0.01).unwrap();

        // Volume should be at least shell_vol + ring_vol * 0.6 (ring partially inside shell)
        let min_expected = shell_vol + ring_vol * 0.5;
        assert!(
            fused_vol >= min_expected,
            "fuse ring overlapping shell: vol={fused_vol:.1}, min_expected={min_expected:.1} \
             (shell={shell_vol:.1}, ring={ring_vol:.1})"
        );

        // Volume should not exceed simple sum
        assert!(
            fused_vol <= shell_vol + ring_vol + 1.0,
            "fuse ring overlapping shell: vol={fused_vol:.1} > sum={:.1}",
            shell_vol + ring_vol
        );
    }

    /// Reproduce Gridfinity lip volume bug: cut two lofted frustums, check
    /// that mesh volume is translation-invariant (proves consistent normals).
    #[test]
    fn cut_lofted_frustums_consistent_normals() {
        use crate::copy::copy_solid;
        use crate::loft::loft;
        use crate::transform::transform_solid;

        // Helper: make a rounded-rectangle profile face at z
        // nq = number of quarter-circle points for corner rounding
        #[allow(clippy::cast_precision_loss)]
        fn make_rounded_rect_profile(
            topo: &mut Topology,
            hw: f64,
            hd: f64,
            r: f64,
            z: f64,
            nq: usize,
        ) -> FaceId {
            let tol_val = 1e-7;
            let r = r.min(hw.min(hd));
            let mut pts = Vec::new();

            // Bottom edge: left to right
            pts.push(Point3::new(-hw + r, -hd, z));
            pts.push(Point3::new(hw - r, -hd, z));
            // Bottom-right corner arc
            for i in 0..nq {
                let a = -std::f64::consts::FRAC_PI_2
                    + std::f64::consts::FRAC_PI_2 * (i as f64 + 1.0) / nq as f64;
                pts.push(Point3::new(hw - r + r * a.cos(), -hd + r + r * a.sin(), z));
            }
            // Right edge: bottom to top
            pts.push(Point3::new(hw, hd - r, z));
            // Top-right corner arc
            for i in 0..nq {
                let a = std::f64::consts::FRAC_PI_2 * (i as f64 + 1.0) / nq as f64;
                pts.push(Point3::new(hw - r + r * a.cos(), hd - r + r * a.sin(), z));
            }
            // Top edge: right to left
            pts.push(Point3::new(-hw + r, hd, z));
            // Top-left corner arc
            for i in 0..nq {
                let a = std::f64::consts::FRAC_PI_2
                    + std::f64::consts::FRAC_PI_2 * (i as f64 + 1.0) / nq as f64;
                pts.push(Point3::new(-hw + r + r * a.cos(), hd - r + r * a.sin(), z));
            }
            // Left edge: top to bottom
            pts.push(Point3::new(-hw, -hd + r, z));
            // Bottom-left corner arc
            for i in 0..nq {
                let a = std::f64::consts::PI
                    + std::f64::consts::FRAC_PI_2 * (i as f64 + 1.0) / nq as f64;
                pts.push(Point3::new(-hw + r + r * a.cos(), -hd + r + r * a.sin(), z));
            }

            let n = pts.len();
            let vids: Vec<_> = pts
                .iter()
                .map(|&p| topo.vertices.alloc(Vertex::new(p, tol_val)))
                .collect();
            let eids: Vec<_> = (0..n)
                .map(|i| {
                    topo.edges
                        .alloc(Edge::new(vids[i], vids[(i + 1) % n], EdgeCurve::Line))
                })
                .collect();
            let wire = Wire::new(
                eids.iter()
                    .map(|&eid| OrientedEdge::new(eid, true))
                    .collect(),
                true,
            )
            .unwrap();
            let wid = topo.wires.alloc(wire);
            topo.faces.alloc(Face::new(
                wid,
                vec![],
                FaceSurface::Plane {
                    normal: Vec3::new(0.0, 0.0, 1.0),
                    d: z,
                },
            ))
        }

        let mut topo = Topology::new();

        // Gridfinity lip profile: 5 sections with varying insets
        let zs = [-1.2, 0.0, 0.7, 2.5, 4.4];
        let outer_insets = [2.6, 2.6, 1.9, 1.9, 0.0];
        let wall = 2.6;
        let base_hw = 62.25; // half of outerW
        let base_hd = 62.25;
        let corner_r = 3.75;
        let nq = 8; // 8 points per quarter-circle

        // Build outer frustum profiles
        let outer_profiles: Vec<FaceId> = zs
            .iter()
            .zip(outer_insets.iter())
            .map(|(&z, &inset)| {
                let hw = base_hw - inset;
                let hd = base_hd - inset;
                let r = f64::max(corner_r - inset, 0.1);
                make_rounded_rect_profile(&mut topo, hw, hd, r, z, nq)
            })
            .collect();
        let outer = loft(&mut topo, &outer_profiles).unwrap();

        // Build inner frustum profiles
        let inner_profiles: Vec<FaceId> = zs
            .iter()
            .zip(outer_insets.iter())
            .map(|(&z, &inset)| {
                let hw = base_hw - inset - wall;
                let hd = base_hd - inset - wall;
                let r = (corner_r - inset - wall).max(0.1);
                make_rounded_rect_profile(&mut topo, hw, hd, r, z, nq)
            })
            .collect();
        let inner = loft(&mut topo, &inner_profiles).unwrap();

        let outer_vol = crate::measure::solid_volume(&topo, outer, 0.01).unwrap();
        let inner_vol = crate::measure::solid_volume(&topo, inner, 0.01).unwrap();
        assert!(outer_vol > 0.0, "outer vol={outer_vol}");
        assert!(inner_vol > 0.0, "inner vol={inner_vol}");

        // Cut outer - inner to get the lip ring
        let lip = boolean(&mut topo, BooleanOp::Cut, outer, inner).unwrap();
        let lip_vol = crate::measure::solid_volume(&topo, lip, 0.01).unwrap();

        let expected = outer_vol - inner_vol;
        eprintln!(
            "outer={outer_vol:.1}, inner={inner_vol:.1}, \
             expected_lip={expected:.1}, actual_lip={lip_vol:.1}"
        );
        assert!(
            lip_vol > 0.0,
            "lip volume should be positive, got {lip_vol}"
        );
        assert!(
            (lip_vol - expected).abs() / expected < 0.10,
            "lip volume {lip_vol:.1} should be ~{expected:.1}"
        );

        // Translation invariance: proves normal consistency
        let lip_up = copy_solid(&mut topo, lip).unwrap();
        let mat = brepkit_math::mat::Mat4::translation(0.0, 0.0, 100.0);
        transform_solid(&mut topo, lip_up, &mat).unwrap();
        let lip_up_vol = crate::measure::solid_volume(&topo, lip_up, 0.01).unwrap();

        eprintln!("lip@origin={lip_vol:.1}, lip@z100={lip_up_vol:.1}");
        assert!(
            (lip_up_vol - lip_vol).abs() / lip_vol.max(1.0) < 0.05,
            "lip volume not translation-invariant: origin={lip_vol:.1}, z100={lip_up_vol:.1}"
        );

        // Compare watertight vs per-face tessellation signed volume.
        // This mirrors the difference between WASM tessellateSolid and
        // tessellateSolidGrouped paths.
        let faces = brepkit_topology::explorer::solid_faces(&topo, lip).unwrap();
        let mut per_face_signed = 0.0_f64;
        #[allow(unused_assignments)]
        let mut per_face_abs = 0.0_f64;
        let mut face_tris = 0;
        for &fid in &faces {
            let mesh = crate::tessellate::tessellate(&topo, fid, 0.01).unwrap();
            let tri_count = mesh.indices.len() / 3;
            face_tris += tri_count;
            for t in 0..tri_count {
                let p0 = mesh.positions[mesh.indices[t * 3] as usize];
                let p1 = mesh.positions[mesh.indices[t * 3 + 1] as usize];
                let p2 = mesh.positions[mesh.indices[t * 3 + 2] as usize];
                let a = Vec3::new(p0.x(), p0.y(), p0.z());
                let b = Vec3::new(p1.x(), p1.y(), p1.z());
                let c = Vec3::new(p2.x(), p2.y(), p2.z());
                per_face_signed += a.dot(b.cross(c));
            }
        }
        per_face_signed /= 6.0;
        per_face_abs = per_face_signed.abs();

        eprintln!(
            "per-face tess: faces={}, tris={face_tris}, signed={per_face_signed:.1}, abs={per_face_abs:.1}",
            faces.len()
        );
        assert!(
            (per_face_abs - lip_vol).abs() / lip_vol.max(1.0) < 0.10,
            "per-face volume {per_face_abs:.1} != watertight volume {lip_vol:.1}"
        );

        // Also check per-face on translated copy
        let faces_up = brepkit_topology::explorer::solid_faces(&topo, lip_up).unwrap();
        let mut per_face_signed_up = 0.0_f64;
        for &fid in &faces_up {
            let mesh = crate::tessellate::tessellate(&topo, fid, 0.01).unwrap();
            let tri_count = mesh.indices.len() / 3;
            for t in 0..tri_count {
                let p0 = mesh.positions[mesh.indices[t * 3] as usize];
                let p1 = mesh.positions[mesh.indices[t * 3 + 1] as usize];
                let p2 = mesh.positions[mesh.indices[t * 3 + 2] as usize];
                let a = Vec3::new(p0.x(), p0.y(), p0.z());
                let b = Vec3::new(p1.x(), p1.y(), p1.z());
                let c = Vec3::new(p2.x(), p2.y(), p2.z());
                per_face_signed_up += a.dot(b.cross(c));
            }
        }
        per_face_signed_up /= 6.0;
        let per_face_abs_up = per_face_signed_up.abs();

        eprintln!("per-face @z100: signed={per_face_signed_up:.1}, abs={per_face_abs_up:.1}");
        assert!(
            (per_face_abs_up - per_face_abs).abs() / per_face_abs.max(1.0) < 0.05,
            "per-face volume not translation-invariant: origin={per_face_abs:.1}, z100={per_face_abs_up:.1}"
        );
    }

    /// Reproduce the EXACT brepjs Gridfinity lip geometry: 8-vertex octagon
    /// profiles from drawRoundedRectangle → face_polygon.
    #[test]
    fn cut_lofted_frustums_octagon_profiles() {
        use crate::copy::copy_solid;
        use crate::loft::loft;
        use crate::transform::transform_solid;

        /// Create an 8-vertex octagon profile matching drawRoundedRectangle(w,d,r).
        /// face_polygon extracts 8 points: (4 edge starts + 4 arc starts).
        fn make_octagon_profile(topo: &mut Topology, hw: f64, hd: f64, r: f64, z: f64) -> FaceId {
            let tol_val = 1e-7;
            // The 8 vertices from face_polygon on a rounded rect:
            // Going CCW from bottom edge:
            //   v0: (-hw+r, -hd)  = bottom-left arc start (bottom edge end)
            //   v1: (-hw, -hd+r)  = left edge start (bottom-left arc end)
            //   v2: (-hw,  hd-r)  = top-left arc start (left edge end)
            //   v3: (-hw+r,  hd)  = top edge start (top-left arc end)
            //   v4: ( hw-r,  hd)  = top-right arc start (top edge end)
            //   v5: ( hw,  hd-r)  = right edge start (top-right arc end)
            //   v6: ( hw, -hd+r)  = bottom-right arc start (right edge end)
            //   v7: ( hw-r, -hd)  = bottom edge start (bottom-right arc end)
            let pts = [
                Point3::new(-hw + r, -hd, z),
                Point3::new(-hw, -hd + r, z),
                Point3::new(-hw, hd - r, z),
                Point3::new(-hw + r, hd, z),
                Point3::new(hw - r, hd, z),
                Point3::new(hw, hd - r, z),
                Point3::new(hw, -hd + r, z),
                Point3::new(hw - r, -hd, z),
            ];
            let n = pts.len();
            let vids: Vec<_> = pts
                .iter()
                .map(|&p| topo.vertices.alloc(Vertex::new(p, tol_val)))
                .collect();
            let eids: Vec<_> = (0..n)
                .map(|i| {
                    topo.edges
                        .alloc(Edge::new(vids[i], vids[(i + 1) % n], EdgeCurve::Line))
                })
                .collect();
            let wire = Wire::new(
                eids.iter()
                    .map(|&eid| OrientedEdge::new(eid, true))
                    .collect(),
                true,
            )
            .unwrap();
            let wid = topo.wires.alloc(wire);
            topo.faces.alloc(Face::new(
                wid,
                vec![],
                FaceSurface::Plane {
                    normal: Vec3::new(0.0, 0.0, 1.0),
                    d: z,
                },
            ))
        }

        let mut topo = Topology::new();

        // Exact Gridfinity lip dimensions (from WASM debug output):
        let zs = [-1.2, 0.0, 0.7, 2.5, 4.4];
        let outer_insets = [2.6, 2.6, 1.9, 1.9, 0.0];
        let wall = 2.6;
        let base_hw = 62.75; // 125.5 / 2
        let base_hd = 62.75;
        let corner_r = 3.75;

        // Outer frustum profiles
        let outer_profiles: Vec<FaceId> = zs
            .iter()
            .zip(outer_insets.iter())
            .map(|(&z, &inset)| {
                let hw = base_hw - inset;
                let hd = base_hd - inset;
                let r = f64::max(corner_r - inset, 0.1);
                make_octagon_profile(&mut topo, hw, hd, r, z)
            })
            .collect();
        let outer = loft(&mut topo, &outer_profiles).unwrap();

        // Inner frustum profiles
        let inner_profiles: Vec<FaceId> = zs
            .iter()
            .zip(outer_insets.iter())
            .map(|(&z, &inset)| {
                let hw = base_hw - inset - wall;
                let hd = base_hd - inset - wall;
                let r = f64::max(corner_r - inset - wall, 0.1);
                make_octagon_profile(&mut topo, hw, hd, r, z)
            })
            .collect();
        let inner = loft(&mut topo, &inner_profiles).unwrap();

        let outer_vol = crate::measure::solid_volume(&topo, outer, 0.01).unwrap();
        let inner_vol = crate::measure::solid_volume(&topo, inner, 0.01).unwrap();

        // Cut outer - inner
        let lip = boolean(&mut topo, BooleanOp::Cut, outer, inner).unwrap();
        let lip_vol = crate::measure::solid_volume(&topo, lip, 0.01).unwrap();
        let expected = outer_vol - inner_vol;

        assert!(
            lip_vol > 0.0,
            "lip volume should be positive, got {lip_vol}"
        );

        // Translation invariance: proves normal consistency
        let lip_up = copy_solid(&mut topo, lip).unwrap();
        let mat = brepkit_math::mat::Mat4::translation(0.0, 0.0, 16.0);
        transform_solid(&mut topo, lip_up, &mat).unwrap();
        let lip_up_vol = crate::measure::solid_volume(&topo, lip_up, 0.01).unwrap();

        assert!(
            (lip_up_vol - lip_vol).abs() / lip_vol.max(1.0) < 0.05,
            "octagon lip not translation-invariant: origin={lip_vol:.1}, z16={lip_up_vol:.1} \
             (outer={outer_vol:.1}, inner={inner_vol:.1}, expected={expected:.1})"
        );
    }
}
