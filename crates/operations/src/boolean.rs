//! Boolean operations on solids: fuse, cut, and intersect.
//!
//! Supports both planar and NURBS faces. NURBS faces are tessellated
//! into planar triangles before clipping, enabling approximate boolean
//! operations on any solid geometry.

use std::collections::HashMap;

use brepkit_math::aabb::Aabb3;
use brepkit_math::bvh::Bvh;
use brepkit_math::plane::plane_plane_intersection;
use brepkit_math::predicates::{orient3d, point_in_polygon};
use brepkit_math::tolerance::Tolerance;
use brepkit_math::vec::{Point2, Point3, Vec3};
use brepkit_topology::Topology;
use brepkit_topology::edge::{Edge, EdgeCurve};
use brepkit_topology::face::{Face, FaceId, FaceSurface};
use brepkit_topology::shell::Shell;
use brepkit_topology::solid::{Solid, SolidId};
use brepkit_topology::vertex::{Vertex, VertexId};
use brepkit_topology::wire::WireId;
use brepkit_topology::wire::{OrientedEdge, Wire};

use crate::dot_normal_point;

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
}

impl Default for BooleanOptions {
    fn default() -> Self {
        Self {
            deflection: DEFAULT_BOOLEAN_DEFLECTION,
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
    let tol = Tolerance::new();

    log::debug!(
        "boolean {op:?}: solids ({}, {}), deflection={}",
        a.index(),
        b.index(),
        opts.deflection
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

    let aabb_a = solid_aabb(&faces_a, tol)?;
    let aabb_b = solid_aabb(&faces_b, tol)?;

    // Disjoint AABB shortcut.
    if !aabb_a.intersects(aabb_b) {
        log::debug!("boolean {op:?}: disjoint AABBs, shortcut");
        return handle_disjoint(topo, op, &faces_a, &faces_b);
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

    let classes: Vec<FaceClass> = fragments
        .iter()
        .map(|frag| {
            // Try analytic classification first (e.g. point-in-sphere).
            let centroid = polygon_centroid(&frag.vertices);
            let fast = match frag.source {
                Source::A => analytic_b.as_ref().and_then(|c| c.classify(centroid, tol)),
                Source::B => analytic_a.as_ref().and_then(|c| c.classify(centroid, tol)),
            };
            if let Some(class) = fast {
                return class;
            }
            let opposite = match frag.source {
                Source::A => &faces_b,
                Source::B => &faces_a,
            };
            classify_fragment(frag, opposite, tol)
        })
        .collect();

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
        if let FaceSurface::Plane { normal, d } = face.surface() {
            let verts = face_polygon(topo, fid)?;
            result.push((fid, verts, *normal, *d));
        } else {
            // Tessellate non-planar face into triangles and treat each as planar.
            let mesh = crate::tessellate::tessellate(topo, fid, deflection)?;
            for tri in mesh.indices.chunks_exact(3) {
                let i0 = tri[0] as usize;
                let i1 = tri[1] as usize;
                let i2 = tri[2] as usize;

                let v0 = mesh.positions[i0];
                let v1 = mesh.positions[i1];
                let v2 = mesh.positions[i2];

                let edge1 = v1 - v0;
                let edge2 = v2 - v0;
                let normal = edge1
                    .cross(edge2)
                    .normalize()
                    .unwrap_or(Vec3::new(0.0, 0.0, 1.0));
                let d = crate::dot_normal_point(normal, v0);

                result.push((fid, vec![v0, v1, v2], normal, d));
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
            let mut sampled = sample_edge_curve(curve, 128);
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

/// Compute AABB encompassing all face vertices, padded by tolerance.
fn solid_aabb(faces: &FaceData, tol: Tolerance) -> Result<Aabb3, crate::OperationsError> {
    Aabb3::try_from_points(
        faces
            .iter()
            .flat_map(|(_, verts, _, _)| verts.iter().copied()),
    )
    .map(|bb| bb.expanded(tol.linear))
    .ok_or_else(|| crate::OperationsError::InvalidInput {
        reason: "solid has no vertices".into(),
    })
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
    use brepkit_math::analytic_intersection::{AnalyticSurface, sample_plane_analytic};

    // Extract plane normal + d.
    let (normal, d) = match plane_surf {
        FaceSurface::Plane { normal, d } => (*normal, *d),
        _ => return None,
    };

    // Extract analytic surface reference.
    let analytic = match analytic_surf {
        FaceSurface::Cylinder(c) => AnalyticSurface::Cylinder(c),
        FaceSurface::Cone(c) => AnalyticSurface::Cone(c),
        FaceSurface::Sphere(s) => AnalyticSurface::Sphere(s),
        FaceSurface::Torus(t) => AnalyticSurface::Torus(t),
        _ => return None,
    };

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

            if let Some(seg) = intersect_face_pair(&side_a, &side_b, tol) {
                segments.push(seg);
            }
        }
    }

    segments
}

/// Intersect two planar face polygons. Returns an intersection segment if any.
fn intersect_face_pair(
    a: &FacePairSide<'_>,
    b: &FacePairSide<'_>,
    tol: Tolerance,
) -> Option<IntersectionSegment> {
    // Plane-plane intersection line.
    let (line_pt, line_dir) = plane_plane_intersection(a.normal, a.d, b.normal, b.d, tol.linear)?;

    // Cyrus-Beck clip against both polygons.
    let (t_min_a, t_max_a) = cyrus_beck_clip(&line_pt, &line_dir, a.verts, &a.normal, tol)?;
    let (t_min_b, t_max_b) = cyrus_beck_clip(&line_pt, &line_dir, b.verts, &b.normal, tol)?;

    // Intersection of the two parameter intervals.
    let t_min = t_min_a.max(t_min_b);
    let t_max = t_max_a.min(t_max_b);

    if t_max - t_min < tol.linear {
        return None;
    }

    Some(IntersectionSegment {
        face_a: a.fid,
        face_b: b.fid,
        p0: point_along_line(&line_pt, &line_dir, t_min),
        p1: point_along_line(&line_pt, &line_dir, t_max),
    })
}

/// Helper: `point + dir * t` as a `Point3`.
fn point_along_line(pt: &Point3, dir: &Vec3, t: f64) -> Point3 {
    Point3::new(
        dir.x().mul_add(t, pt.x()),
        dir.y().mul_add(t, pt.y()),
        dir.z().mul_add(t, pt.z()),
    )
}

/// Cyrus-Beck clipping of a line against a convex polygon.
///
/// The line is `P(t) = line_pt + t * line_dir`. The polygon lies on a plane
/// with normal `face_normal`. Returns `(t_min, t_max)` of the segment inside
/// the polygon, or `None` if the line doesn't cross the polygon.
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
        // Inward-pointing normal of this edge (within the face plane).
        let edge_normal = face_normal.cross(edge_vec);

        // Vector from polygon vertex to line point.
        let w = *line_pt - polygon[i];

        let denom = edge_normal.dot(*line_dir);
        let numer = -edge_normal.dot(w);

        if denom.abs() < tol.angular {
            // Line parallel to this edge.
            // edge_normal · w > 0 means on the interior side.
            if edge_normal.dot(w) < 0.0 {
                return None; // Outside this edge.
            }
            continue;
        }

        let t = numer / denom;
        if denom > 0.0 {
            // Entering half-space (inward normal convention).
            t_enter = t_enter.max(t);
        } else {
            // Exiting half-space (inward normal convention).
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

/// Split a face polygon along intersection chords, producing fragments.
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
        // No chords — the entire face is a single fragment.
        return vec![FaceFragment {
            vertices: verts.to_vec(),
            normal,
            d,
            source,
        }];
    };

    // Start with the whole polygon as the only fragment.
    let mut frags: Vec<Vec<Point3>> = vec![verts.to_vec()];

    for &(c0, c1) in chords {
        let mut new_frags = Vec::new();
        for poly in &frags {
            let (left, right) = split_polygon_by_chord(poly, c0, c1, &normal, tol);
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

/// Compute twice the area of a 3D polygon projected along its normal.
///
/// Used for filtering degenerate (zero-area) fragments. Returns the
/// magnitude of the cross-product sum (Newell's method), which equals
/// `2 * area`.
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
    tol: Tolerance,
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

    let signs: Vec<f64> = polygon
        .iter()
        .map(|v| orient3d(c0, c1, c_top, *v))
        .collect();

    let mut left = Vec::new();
    let mut right = Vec::new();

    for i in 0..n {
        let j = (i + 1) % n;
        let si = signs[i];
        let sj = signs[j];

        // Classify current vertex.
        if si >= -tol.linear {
            left.push(polygon[i]);
        }
        if si <= tol.linear {
            right.push(polygon[i]);
        }

        // Check for sign change (edge crossing).
        if (si > tol.linear && sj < -tol.linear) || (si < -tol.linear && sj > tol.linear) {
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

    let mut sphere_info: Option<(Point3, f64)> = None;
    for &fid in shell.faces() {
        let face = topo.face(fid).ok()?;
        match face.surface() {
            FaceSurface::Sphere(sph) => {
                if let Some((c, r)) = sphere_info {
                    // All sphere faces must reference the same sphere.
                    let dc = (c - sph.center()).length();
                    if dc > tol.linear || (r - sph.radius()).abs() > tol.linear {
                        return None;
                    }
                } else {
                    sphere_info = Some((sph.center(), sph.radius()));
                }
            }
            _ => return None,
        }
    }

    let (center, radius) = sphere_info?;
    Some(AnalyticClassifier::Sphere { center, radius })
}

/// Classify a face fragment relative to the opposite solid.
fn classify_fragment(frag: &FaceFragment, opposite: &FaceData, tol: Tolerance) -> FaceClass {
    // Compute centroid of the fragment.
    let centroid = polygon_centroid(&frag.vertices);

    // First check for coplanar faces.
    for &(_, ref verts, n_opp, d_opp) in opposite {
        let dist = dot_normal_point(n_opp, centroid) - d_opp;
        if dist.abs() < tol.linear && point_in_face_3d(centroid, verts, &n_opp) {
            let dot = frag.normal.dot(n_opp);
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

    // Ray-cast from centroid along fragment normal.
    let ray_dir = frag.normal;
    let mut crossings = 0i32;

    for &(_, ref verts, n_opp, d_opp) in opposite {
        let denom = n_opp.dot(ray_dir);
        if denom.abs() < tol.angular {
            continue; // Ray parallel to face.
        }
        let numer = d_opp - dot_normal_point(n_opp, centroid);
        let t = numer / denom;
        if t <= tol.linear {
            continue; // Behind or at the ray origin.
        }

        let hit = point_along_line(&centroid, &ray_dir, t);
        if point_in_face_3d(hit, verts, &n_opp) {
            if denom > 0.0 {
                crossings -= 1;
            } else {
                crossings += 1;
            }
        }
    }

    if crossings == 0 {
        FaceClass::Outside
    } else {
        FaceClass::Inside
    }
}

/// Compute the centroid of a polygon.
///
/// Returns the origin if the polygon is empty (should not happen in
/// practice since fragments are filtered for `len >= 3`).
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
#[allow(clippy::cast_possible_truncation)] // coordinate * 1e7 fits in i64
fn quantize(v: f64, resolution: f64) -> i64 {
    (v * resolution).round() as i64
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
    let resolution = 1e7;

    let mut vertex_map: HashMap<(i64, i64, i64), VertexId> = HashMap::new();
    let mut edge_map: HashMap<(usize, usize), brepkit_topology::edge::EdgeId> = HashMap::new();

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
/// - Too few faces (< 4, which can't form a closed solid)
/// - Open shell (boundary edges)
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
        let normal = if let FaceSurface::Plane { normal, .. } = face.surface() {
            *normal
        } else {
            // For non-planar faces, approximate normal from sampled polygon.
            let verts = face_polygon(topo, fid)?;
            if verts.len() >= 3 {
                let e1 = verts[1] - verts[0];
                let e2 = verts[2] - verts[0];
                e1.cross(e2).normalize().unwrap_or(Vec3::new(0.0, 0.0, 1.0))
            } else {
                Vec3::new(0.0, 0.0, 1.0)
            }
        };

        let verts = face_polygon(topo, fid)?;
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
    let mesh_a = crate::tessellate::tessellate_solid(topo, a, deflection)?;
    let mesh_b = crate::tessellate::tessellate_solid(topo, b, deflection)?;

    let mb_result = crate::mesh_boolean::mesh_boolean(&mesh_a, &mesh_b, op, deflection)?;

    // Convert mesh result to topology: each triangle becomes a planar face.
    let tol = Tolerance::new();
    let mut face_data: Vec<(Vec<Point3>, Vec3, f64)> = Vec::new();
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

        face_data.push((vec![v0, v1, v2], normal, d));
    }

    if face_data.is_empty() {
        return Err(crate::OperationsError::InvalidInput {
            reason: "mesh boolean produced empty result".into(),
        });
    }

    let result = assemble_solid(topo, &face_data, tol)?;
    validate_boolean_result(topo, result)?;
    Ok(result)
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
        AnalyticSurface, ExactIntersectionCurve, exact_plane_analytic, intersect_analytic_analytic,
    };

    // Collect face info for both solids.
    let solid_a = topo.solid(a)?;
    let shell_a = topo.shell(solid_a.outer_shell())?;
    let face_ids_a: Vec<FaceId> = shell_a.faces().to_vec();

    let solid_b = topo.solid(b)?;
    let shell_b = topo.shell(solid_b.outer_shell())?;
    let face_ids_b: Vec<FaceId> = shell_b.faces().to_vec();

    let mut snaps_a = Vec::new();
    for &fid in &face_ids_a {
        let face = topo.face(fid)?;
        let surface = face.surface().clone();
        let reversed = face.is_reversed();
        let verts = face_polygon(topo, fid)?;
        let (normal, d) = match &surface {
            FaceSurface::Plane { normal, d } => (*normal, *d),
            _ => {
                // For non-planar faces, compute a representative normal from tessellation.
                let mesh = crate::tessellate::tessellate(topo, fid, deflection)?;
                let avg_normal = if mesh.normals.is_empty() {
                    Vec3::new(0.0, 0.0, 1.0)
                } else {
                    mesh.normals[0]
                };
                let d_val = if mesh.positions.is_empty() {
                    0.0
                } else {
                    dot_normal_point(avg_normal, mesh.positions[0])
                };
                (avg_normal, d_val)
            }
        };
        snaps_a.push(FaceSnapshot {
            id: fid,
            surface,
            vertices: verts,
            normal,
            d,
            reversed,
        });
    }

    let mut snaps_b = Vec::new();
    for &fid in &face_ids_b {
        let face = topo.face(fid)?;
        let surface = face.surface().clone();
        let reversed = face.is_reversed();
        let verts = face_polygon(topo, fid)?;
        let (normal, d) = match &surface {
            FaceSurface::Plane { normal, d } => (*normal, *d),
            _ => {
                let mesh = crate::tessellate::tessellate(topo, fid, deflection)?;
                let avg_normal = if mesh.normals.is_empty() {
                    Vec3::new(0.0, 0.0, 1.0)
                } else {
                    mesh.normals[0]
                };
                let d_val = if mesh.positions.is_empty() {
                    0.0
                } else {
                    dot_normal_point(avg_normal, mesh.positions[0])
                };
                (avg_normal, d_val)
            }
        };
        snaps_b.push(FaceSnapshot {
            id: fid,
            surface,
            vertices: verts,
            normal,
            d,
            reversed,
        });
    }

    // Compute AABBs for face pairs (surface-aware for non-planar faces).
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

    // Track contained intersection curves (circle/ellipse fully inside a planar face).
    #[allow(clippy::items_after_statements)]
    struct ContainedCurve {
        plane_face_idx: usize,
        plane_source: Source,
        analytic_face_idx: usize,
        edge_curve: EdgeCurve,
    }
    let mut contained_curves: Vec<ContainedCurve> = Vec::new();

    for (ia, snap_a) in snaps_a.iter().enumerate() {
        for (ib, snap_b) in snaps_b.iter().enumerate() {
            // Skip non-overlapping AABBs.
            if !aabbs_a[ia].intersects(aabbs_b[ib]) {
                continue;
            }

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
                let analytic_surf = match &snap_b.surface {
                    FaceSurface::Cylinder(c) => AnalyticSurface::Cylinder(c),
                    FaceSurface::Cone(c) => AnalyticSurface::Cone(c),
                    FaceSurface::Sphere(s) => AnalyticSurface::Sphere(s),
                    FaceSurface::Torus(t) => AnalyticSurface::Torus(t),
                    _ => {
                        has_analytic_analytic = true;
                        continue;
                    }
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
                let analytic_surf = match &snap_a.surface {
                    FaceSurface::Cylinder(c) => AnalyticSurface::Cylinder(c),
                    FaceSurface::Cone(c) => AnalyticSurface::Cone(c),
                    FaceSurface::Sphere(s) => AnalyticSurface::Sphere(s),
                    FaceSurface::Torus(t) => AnalyticSurface::Torus(t),
                    _ => {
                        has_analytic_analytic = true;
                        continue;
                    }
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
                let surf_a_opt = match &snap_a.surface {
                    FaceSurface::Cylinder(c) => Some(AnalyticSurface::Cylinder(c)),
                    FaceSurface::Cone(c) => Some(AnalyticSurface::Cone(c)),
                    FaceSurface::Sphere(s) => Some(AnalyticSurface::Sphere(s)),
                    FaceSurface::Torus(t) => Some(AnalyticSurface::Torus(t)),
                    _ => None,
                };
                let surf_b_opt = match &snap_b.surface {
                    FaceSurface::Cylinder(c) => Some(AnalyticSurface::Cylinder(c)),
                    FaceSurface::Cone(c) => Some(AnalyticSurface::Cone(c)),
                    FaceSurface::Sphere(s) => Some(AnalyticSurface::Sphere(s)),
                    FaceSurface::Torus(t) => Some(AnalyticSurface::Torus(t)),
                    _ => None,
                };

                if let (Some(surf_a_an), Some(surf_b_an)) = (surf_a_opt, surf_b_opt) {
                    if let Ok(curves) = intersect_analytic_analytic(surf_a_an, surf_b_an, 32) {
                        for ic in &curves {
                            let pts: Vec<Point3> = ic.points.iter().map(|ip| ip.point).collect();
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

    let mut fragments: Vec<AnalyticFragment> = Vec::new();

    // Process solid A faces.
    for (ia, snap) in snaps_a.iter().enumerate() {
        // Sphere faces: always tessellate into triangle fragments.
        // Chord splitting on the equatorial polygon doesn't produce
        // meaningful sphere surface fragments.
        if matches!(snap.surface, FaceSurface::Sphere(_)) {
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
                    let curve_verts = sample_edge_curve(ec, 32);
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
                edge_curves: vec![None; snap.vertices.len()],
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
                let curve_verts = sample_edge_curve(ec, 32);
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
                edge_curves: vec![None; snap.vertices.len()],
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
        // Sphere faces: always tessellate into triangle fragments.
        if matches!(snap.surface, FaceSurface::Sphere(_)) {
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
                    let curve_verts = sample_edge_curve(ec, 32);
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
                edge_curves: vec![None; snap.vertices.len()],
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
                let curve_verts = sample_edge_curve(ec, 32);
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
                edge_curves: vec![None; snap.vertices.len()],
                source_reversed: snap.reversed,
            });
            let source_face = topo.face(snap.id)?;
            if !source_face.inner_wires().is_empty() {
                existing_inner_wires.insert(unsplit_idx, source_face.inner_wires().to_vec());
            }
        }
    }

    // ── Classification ───────────────────────────────────────────────────

    // Build FaceData for classification (tessellated faces of opposite solid).
    let face_data_a = collect_face_data(topo, a, deflection)?;
    let face_data_b = collect_face_data(topo, b, deflection)?;

    // Analytic classifiers for O(1) point-in-solid tests (e.g. sphere).
    let analytic_cls_a = try_build_analytic_classifier(topo, a);
    let analytic_cls_b = try_build_analytic_classifier(topo, b);

    let classes: Vec<FaceClass> = fragments
        .iter()
        .enumerate()
        .map(|(idx, frag)| {
            // Use pre-classification for contained-curve fragments.
            if let Some(&class) = pre_classifications.get(&idx) {
                return class;
            }
            // Try analytic classification first.
            let centroid = polygon_centroid(&frag.vertices);
            let fast = match frag.source {
                Source::A => analytic_cls_b
                    .as_ref()
                    .and_then(|c| c.classify(centroid, tol)),
                Source::B => analytic_cls_a
                    .as_ref()
                    .and_then(|c| c.classify(centroid, tol)),
            };
            if let Some(class) = fast {
                return class;
            }
            let opposite = match frag.source {
                Source::A => &face_data_b,
                Source::B => &face_data_a,
            };
            let pseudo = FaceFragment {
                vertices: frag.vertices.clone(),
                normal: frag.normal,
                d: frag.d,
                source: frag.source,
            };
            classify_fragment(&pseudo, opposite, tol)
        })
        .collect();

    // ── Selection + Assembly ─────────────────────────────────────────────

    let resolution = 1e7;
    let mut vertex_map: HashMap<(i64, i64, i64), VertexId> = HashMap::new();
    let mut edge_map: HashMap<(usize, usize), brepkit_topology::edge::EdgeId> = HashMap::new();
    let mut face_ids_out = Vec::new();

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

        // Build edges — deduplicate by ordered vertex-index pair so adjacent
        // faces share edge IDs (required for fillet/chamfer adjacency queries).
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
        let wire_id = topo.wires.alloc(wire);

        // Build inner wires for holed faces (new contained curves + existing holes).
        let mut inner_wire_ids = Vec::new();

        // 1. Carry over existing inner wires from source faces (prior booleans).
        if let Some(existing_wires) = existing_inner_wires.get(&idx) {
            inner_wire_ids.extend_from_slice(existing_wires);
        }

        // 2. Create new inner wires from contained intersection curves.
        if let Some(inner_curves) = holed_face_inner_curves.get(&idx) {
            for ec in inner_curves {
                // Sample the curve and REVERSE for CW winding (hole convention).
                let mut hole_pts = sample_edge_curve(ec, 32);
                if flip {
                    // If the face is flipped, don't reverse the hole
                    // (the outer wire is already reversed).
                } else {
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
                    let eid = topo.edges.alloc(Edge::new(
                        hole_vert_ids[i],
                        hole_vert_ids[j],
                        EdgeCurve::Line,
                    ));
                    hole_edges.push(OrientedEdge::new(eid, true));
                }
                let hw = Wire::new(hole_edges, true).map_err(crate::OperationsError::Topology)?;
                inner_wire_ids.push(topo.wires.alloc(hw));
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

    if face_ids_out.is_empty() {
        return Err(crate::OperationsError::InvalidInput {
            reason: "analytic boolean produced no faces".into(),
        });
    }

    let shell = Shell::new(face_ids_out).map_err(crate::OperationsError::Topology)?;
    let shell_id = topo.shells.alloc(shell);
    Ok(topo.solids.alloc(Solid::new(shell_id, vec![])))
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

    // Classify each sample as inside or outside the face polygon.
    let inside: Vec<bool> = raw_points
        .iter()
        .map(|pt| point_in_face_3d(*pt, face_verts, &face_normal))
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

    let n_samples: usize = 32;

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
    let mut v_min = f64::MAX;
    let mut v_max = f64::MIN;
    for &p in face_verts {
        let v = cyl.axis().dot(p - cyl.origin());
        if v < v_min {
            v_min = v;
        }
        if v > v_max {
            v_max = v;
        }
    }

    if (v_max - v_min).abs() < 1e-10 {
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

    // Sample a circle at a given v-level. For cut levels with an EdgeCurve,
    // use sample_edge_curve so the points match the holed-face inner wire.
    // For barrel endpoints, use cylinder parametric evaluation.
    #[allow(clippy::cast_precision_loss)]
    let sample_level = |v: f64, curve: Option<&EdgeCurve>| -> Vec<Point3> {
        if let Some(ec) = curve {
            sample_edge_curve(ec, n_samples)
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
        #[allow(clippy::cast_precision_loss)]
        let centroid = {
            let (sx, sy, sz) = verts.iter().fold((0.0, 0.0, 0.0), |(ax, ay, az), v| {
                (ax + v.x(), ay + v.y(), az + v.z())
            });
            let inv_n = 1.0 / verts.len() as f64;
            Point3::new(sx * inv_n, sy * inv_n, sz * inv_n)
        };
        let band_d = crate::dot_normal_point(band_normal, centroid);

        fragments.push(AnalyticFragment {
            vertices: verts,
            surface: surface.clone(),
            normal: band_normal,
            d: band_d,
            source,
            edge_curves: vec![None; 2 * n_samples],
            source_reversed,
        });
    }
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

    use brepkit_topology::Topology;
    use brepkit_topology::test_utils::make_unit_cube_manifold_at;
    use brepkit_topology::validation::validate_shell_manifold;

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
        assert!(check_result(&topo, result) > 0);
    }

    #[test]
    fn intersect_overlapping_cubes() {
        let mut topo = Topology::new();
        let a = make_unit_cube_manifold_at(&mut topo, 0.0, 0.0, 0.0);
        let b = make_unit_cube_manifold_at(&mut topo, 0.5, 0.0, 0.0);

        let result = boolean(&mut topo, BooleanOp::Intersect, a, b).unwrap();
        assert!(check_result(&topo, result) > 0);
    }

    #[test]
    fn cut_overlapping_cubes() {
        let mut topo = Topology::new();
        let a = make_unit_cube_manifold_at(&mut topo, 0.0, 0.0, 0.0);
        let b = make_unit_cube_manifold_at(&mut topo, 0.5, 0.0, 0.0);

        let result = boolean(&mut topo, BooleanOp::Cut, a, b).unwrap();
        assert!(check_result(&topo, result) > 0);
    }

    // ── 3D overlapping tests (offset on all axes) ───────────────────────

    #[test]
    fn fuse_overlapping_3d() {
        let mut topo = Topology::new();
        let a = make_unit_cube_manifold_at(&mut topo, 0.0, 0.0, 0.0);
        let b = make_unit_cube_manifold_at(&mut topo, 0.5, 0.5, 0.5);

        let result = boolean(&mut topo, BooleanOp::Fuse, a, b).unwrap();
        assert!(check_result(&topo, result) > 0);
    }

    #[test]
    fn intersect_overlapping_3d() {
        let mut topo = Topology::new();
        let a = make_unit_cube_manifold_at(&mut topo, 0.0, 0.0, 0.0);
        let b = make_unit_cube_manifold_at(&mut topo, 0.5, 0.5, 0.5);

        let result = boolean(&mut topo, BooleanOp::Intersect, a, b).unwrap();
        assert!(check_result(&topo, result) > 0);
    }

    #[test]
    fn cut_overlapping_3d() {
        let mut topo = Topology::new();
        let a = make_unit_cube_manifold_at(&mut topo, 0.0, 0.0, 0.0);
        let b = make_unit_cube_manifold_at(&mut topo, 0.5, 0.5, 0.5);

        let result = boolean(&mut topo, BooleanOp::Cut, a, b).unwrap();
        assert!(check_result(&topo, result) > 0);
    }

    // ── Flush face test ─────────────────────────────────────────────────

    #[test]
    fn fuse_flush_face_cubes() {
        let mut topo = Topology::new();
        let a = make_unit_cube_manifold_at(&mut topo, 0.0, 0.0, 0.0);
        let b = make_unit_cube_manifold_at(&mut topo, 1.0, 0.0, 0.0);

        let result = boolean(&mut topo, BooleanOp::Fuse, a, b).unwrap();
        assert!(check_result(&topo, result) > 0);
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
    fn intersect_box_sphere_succeeds() {
        let mut topo = Topology::new();
        let bx = crate::primitives::make_box(&mut topo, 10.0, 10.0, 10.0).unwrap();
        let sp = crate::primitives::make_sphere(&mut topo, 7.0, 16).unwrap();
        let result = boolean(&mut topo, BooleanOp::Intersect, bx, sp);
        assert!(
            result.is_ok(),
            "intersect(box, sphere) should succeed: {:?}",
            result.err()
        );
    }

    #[test]
    fn fuse_box_sphere_succeeds() {
        let mut topo = Topology::new();
        let bx = crate::primitives::make_box(&mut topo, 10.0, 10.0, 10.0).unwrap();
        let sp = crate::primitives::make_sphere(&mut topo, 7.0, 16).unwrap();
        let result = boolean(&mut topo, BooleanOp::Fuse, bx, sp);
        assert!(
            result.is_ok(),
            "fuse(box, sphere) should succeed: {:?}",
            result.err()
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
}
