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
    let tol = Tolerance::new();

    // ── Phase 0: Guard + Precompute ──────────────────────────────────────

    let faces_a = collect_face_data(topo, a)?;
    let faces_b = collect_face_data(topo, b)?;

    let aabb_a = solid_aabb(&faces_a, tol)?;
    let aabb_b = solid_aabb(&faces_b, tol)?;

    // Disjoint AABB shortcut.
    if !aabb_a.intersects(aabb_b) {
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

    let classes: Vec<FaceClass> = fragments
        .iter()
        .map(|frag| {
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

    assemble_solid(topo, &selected, tol)
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
        // Coplanar opposite
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
) -> Result<FaceData, crate::OperationsError> {
    let solid = topo.solid(solid_id)?;
    let shell = topo.shell(solid.outer_shell())?;
    let mut result = Vec::with_capacity(shell.faces().len());

    for &fid in shell.faces() {
        let face = topo.face(fid)?;
        if let FaceSurface::Plane { normal, d } = face.surface() {
            let verts = face_vertices(topo, fid)?;
            result.push((fid, verts, *normal, *d));
        } else {
            // Tessellate non-planar face into triangles and treat each as planar.
            let mesh = crate::tessellate::tessellate(topo, fid, 1.0)?;
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

/// Get the ordered vertices of a face by traversing its outer wire.
pub(crate) fn face_vertices(
    topo: &Topology,
    face_id: FaceId,
) -> Result<Vec<Point3>, crate::OperationsError> {
    let face = topo.face(face_id)?;
    let wire = topo.wire(face.outer_wire())?;
    let mut verts = Vec::with_capacity(wire.edges().len());

    for oe in wire.edges() {
        let edge = topo.edge(oe.edge())?;
        let vid = if oe.is_forward() {
            edge.start()
        } else {
            edge.end()
        };
        verts.push(topo.vertex(vid)?.point());
    }

    Ok(verts)
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

/// Classify a face fragment relative to the opposite solid.
fn classify_fragment(frag: &FaceFragment, opposite: &FaceData, tol: Tolerance) -> FaceClass {
    // Compute centroid of the fragment.
    let centroid = polygon_centroid(&frag.vertices);

    // First check for coplanar faces.
    for &(_, ref verts, n_opp, d_opp) in opposite {
        let dist = dot_normal_point(n_opp, centroid) - d_opp;
        if dist.abs() < tol.linear && point_in_face_3d(centroid, verts, &n_opp) {
            let dot = frag.normal.dot(n_opp);
            return if dot > 0.0 {
                FaceClass::CoplanarSame
            } else {
                FaceClass::CoplanarOpposite
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

/// Assemble a solid from a set of face polygons with normals.
///
/// Uses spatial hashing for vertex dedup and edge sharing.
pub(crate) fn assemble_solid(
    topo: &mut Topology,
    faces: &[(Vec<Point3>, Vec3, f64)],
    tol: Tolerance,
) -> Result<SolidId, crate::OperationsError> {
    let resolution = 1e7;

    let mut vertex_map: HashMap<(i64, i64, i64), VertexId> = HashMap::new();
    let mut edge_map: HashMap<(usize, usize), brepkit_topology::edge::EdgeId> = HashMap::new();

    let mut face_ids = Vec::with_capacity(faces.len());

    for (verts, normal, d) in faces {
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
        let face = topo.faces.alloc(Face::new(
            wire_id,
            vec![],
            FaceSurface::Plane {
                normal: *normal,
                d: *d,
            },
        ));
        face_ids.push(face);
    }

    if face_ids.is_empty() {
        return Err(crate::OperationsError::InvalidInput {
            reason: "boolean operation produced no faces".into(),
        });
    }

    let shell = Shell::new(face_ids).map_err(crate::OperationsError::Topology)?;
    let shell_id = topo.shells.alloc(shell);
    let solid = topo.solids.alloc(Solid::new(shell_id, vec![]));

    Ok(solid)
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

    let mut input_faces: Vec<(usize, Vec3, Point3)> = Vec::with_capacity(
        input_faces_a.len() + input_faces_b.len(),
    );
    input_faces.extend(input_faces_a);
    input_faces.extend(input_faces_b);

    // Run the actual boolean.
    let result = boolean(topo, op, a, b)?;

    // Collect output face normals + centroids.
    let output_faces = collect_face_signatures(topo, result)?;

    // Build evolution map via heuristic matching.
    let mut evo = EvolutionMap::new();
    let mut matched_inputs: std::collections::HashSet<usize> = std::collections::HashSet::new();

    // Normal dot threshold: cos(30deg) — faces with normals diverging more
    // than ~30 degrees are not considered matches.
    let normal_threshold = 0.866;
    // Maximum centroid distance squared for a match (generous, scaled to unit).
    let centroid_dist_sq_max = 10.0;

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
            // Normalize distance contribution to [0, 1] range.
            let score = dot - dist_sq / centroid_dist_sq_max;
            if score > best_score {
                best_score = score;
                best_input = Some(in_idx);
            }
        }

        if let Some(in_idx) = best_input {
            evo.add_modified(in_idx, out_idx);
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
            // For non-planar faces, approximate normal from first triangle.
            let verts = face_vertices(topo, fid)?;
            if verts.len() >= 3 {
                let e1 = verts[1] - verts[0];
                let e2 = verts[2] - verts[0];
                e1.cross(e2)
                    .normalize()
                    .unwrap_or(Vec3::new(0.0, 0.0, 1.0))
            } else {
                Vec3::new(0.0, 0.0, 1.0)
            }
        };

        let verts = face_vertices(topo, fid)?;
        let centroid = polygon_centroid(&verts);
        result.push((fid.index(), normal, centroid));
    }

    Ok(result)
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

        let result = collect_face_data(&topo, cyl);
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
}
