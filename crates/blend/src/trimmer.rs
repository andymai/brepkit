// Walking engine infrastructure — used progressively as more blend paths are wired up.
#![allow(dead_code)]
//! Face trimming along contact curves.
//!
//! After computing a fillet or chamfer blend surface, the original adjacent
//! faces must be trimmed along the contact curves — the lines where the blend
//! surface meets the original geometry. This module splits planar faces along
//! straight contact lines, creating new edges, vertices, and wires for the
//! trimmed result.

use brepkit_math::vec::Point3;
use brepkit_topology::Topology;
use brepkit_topology::edge::{Edge, EdgeCurve, EdgeId};
use brepkit_topology::face::{Face, FaceId, FaceSurface};
use brepkit_topology::vertex::{Vertex, VertexId};
use brepkit_topology::wire::{OrientedEdge, Wire, WireId};

use crate::BlendError;

/// Which side of the contact curve to keep.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TrimSide {
    /// Keep the left side of the contact curve (relative to its direction).
    Left,
    /// Keep the right side of the contact curve (relative to its direction).
    Right,
}

/// How to choose the kept side of the contact curve.
///
/// `Side` is interpreted against the trimmer's internal hit order, which
/// follows the face's wire traversal; callers outside the trimmer cannot
/// predict that frame, so `AwayFrom` names a 3D point (typically a spine
/// point on the edge being blended) and the trimmer keeps the chain on the
/// opposite side, resolved in its own frame.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum TrimKeep {
    /// Keep an explicit side in the trimmer's own frame.
    Side(TrimSide),
    /// Keep the side of the contact curve away from this point.
    AwayFrom(Point3),
}

/// Result of trimming a face along a contact curve.
#[derive(Debug, Clone)]
pub struct TrimResult {
    /// The newly created trimmed face (or the original if untrimmed).
    pub trimmed_face: FaceId,
    /// New edges created during trimming (sub-edges from splits).
    pub new_edges: Vec<EdgeId>,
    /// New vertices created at contact curve / boundary intersections.
    pub new_vertices: Vec<VertexId>,
    /// The edge running along the contact curve between the two boundary
    /// intersection points. `None` when the face was returned untrimmed
    /// (e.g. non-planar surfaces).
    pub contact_edge: Option<EdgeId>,
}

/// Default vertex tolerance used when creating intersection vertices.
const VERTEX_TOL: f64 = 1e-7;

/// Tolerance for the 2D segment intersection parameter test.
const PARAM_TOL: f64 = 1e-10;

/// A point where the contact line crosses a face boundary edge.
struct BoundaryHit {
    /// Index into the wire's oriented-edge list.
    edge_idx: usize,
    /// Parameter along the oriented edge (0..1).
    t: f64,
    /// 3D intersection point.
    point_3d: Point3,
}

/// Trim a face along a contact curve, keeping the side away from the fillet.
///
/// For planar faces with straight contact lines (the plane-plane fillet case),
/// this finds where the contact line intersects the face boundary, splits
/// those boundary edges, and builds a new wire loop for the trimmed face.
///
/// Non-planar faces are returned untrimmed: the original face ID is placed in
/// `trimmed_face` and no new topology is created.
///
/// # Errors
///
/// Returns [`BlendError::TrimmingFailure`] if the contact curve does not
/// produce exactly two boundary intersections, or if topology lookups fail.
/// Returns [`BlendError::Topology`] on arena errors.
#[allow(clippy::too_many_lines)]
pub fn trim_face(
    topo: &mut Topology,
    face_id: FaceId,
    contact_3d: &[Point3],
    contact_uv: &[(f64, f64)],
    keep: TrimKeep,
) -> Result<TrimResult, BlendError> {
    let face = topo.face(face_id)?;
    if !face.surface().is_planar() {
        log::warn!(
            "trim_face: non-planar surface ({}) on face {face_id:?} — returning untrimmed",
            face.surface().type_tag(),
        );
        return Ok(TrimResult {
            trimmed_face: face_id,
            new_edges: Vec::new(),
            new_vertices: Vec::new(),
            contact_edge: None,
        });
    }

    let surface = face.surface().clone();
    let reversed = face.is_reversed();
    let outer_wire_id = face.outer_wire();

    let outer_wire = topo.wire(outer_wire_id)?;
    let oriented_edges: Vec<OrientedEdge> = outer_wire.edges().to_vec();

    if contact_uv.len() < 2 {
        {
            if std::env::var("BK_TRIM_TRACE").is_ok() {
                log::warn!("TRIM-FAIL site1 face={face_id:?}");
            }
            return Err(BlendError::TrimmingFailure { face: face_id });
        }
    }

    let uv_start = contact_uv[0];
    let uv_end = contact_uv[contact_uv.len() - 1];

    // For each oriented edge, record (oriented_edge, start_point, end_point)
    // where "start" / "end" follow the orientation.
    let mut edge_data: Vec<(OrientedEdge, Point3, Point3)> =
        Vec::with_capacity(oriented_edges.len());
    for &oe in &oriented_edges {
        let edge = topo.edge(oe.edge())?;
        let start_vid = oe.oriented_start(edge);
        let end_vid = oe.oriented_end(edge);
        let start_pt = topo.vertex(start_vid)?.point();
        let end_pt = topo.vertex(end_vid)?.point();
        edge_data.push((oe, start_pt, end_pt));
    }

    // For a planar face we use a local 2D coordinate system derived from
    // the first two edge directions.
    let (origin, u_axis, v_axis) = plane_local_frame(&surface, &edge_data, face_id)?;

    let project = |pt: Point3| -> (f64, f64) {
        let d = pt - origin;
        (u_axis.dot(d), v_axis.dot(d))
    };

    // Convert contact line endpoints from the caller-provided UV to our
    // local frame. If the caller's UV already matches the plane's local
    // frame we use them directly; otherwise we re-project the 3D points.
    let (line_a, line_b) = if contact_3d.len() >= 2 {
        (
            project(contact_3d[0]),
            project(contact_3d[contact_3d.len() - 1]),
        )
    } else {
        (uv_start, uv_end)
    };

    let mut hits: Vec<BoundaryHit> = Vec::new();

    for (idx, &(_, start_pt, end_pt)) in edge_data.iter().enumerate() {
        let a1 = project(start_pt);
        let a2 = project(end_pt);

        if let Some(t) = line_segment_intersect_2d(a1, a2, line_a, line_b) {
            let pt = Point3::new(
                start_pt.x() + (end_pt.x() - start_pt.x()) * t,
                start_pt.y() + (end_pt.y() - start_pt.y()) * t,
                start_pt.z() + (end_pt.z() - start_pt.z()) * t,
            );
            hits.push(BoundaryHit {
                edge_idx: idx,
                t,
                point_3d: pt,
            });
        }
    }

    // A contact endpoint landing exactly on an existing boundary VERTEX
    // (a previous stripe's propagated split) registers on both incident
    // chords — one geometric crossing counted twice. Merge
    // position-coincident hits before judging the count.
    hits.sort_by(|a, b| {
        (a.edge_idx, a.t)
            .partial_cmp(&(b.edge_idx, b.t))
            .unwrap_or(std::cmp::Ordering::Equal)
    });
    hits.dedup_by(|b, a| (b.point_3d - a.point_3d).length() < 1e-6);
    if hits.len() > 1 && (hits[0].point_3d - hits[hits.len() - 1].point_3d).length() < 1e-6 {
        hits.pop();
    }

    // We expect exactly 2 hits for a convex planar face.
    if hits.len() != 2 {
        if std::env::var("BK_TRIM_TRACE").is_ok() {
            log::warn!(
                "TRIM-FAIL site2 face={face_id:?} hits={} contact ({line_a:?})->({line_b:?})",
                hits.len()
            );
            for h in &hits {
                log::warn!(
                    "TRIM-FAIL   hit edge_idx={} t={:.4} p={:?}",
                    h.edge_idx,
                    h.t,
                    h.point_3d
                );
            }
        }
        return Err(BlendError::TrimmingFailure { face: face_id });
    }

    // Order hits so that hit_a comes first in wire traversal order.
    // They are already ordered by edge_idx from the iteration above,
    // but if both hits are on the same edge we order by parameter.
    if hits[0].edge_idx > hits[1].edge_idx
        || (hits[0].edge_idx == hits[1].edge_idx && hits[0].t > hits[1].t)
    {
        hits.swap(0, 1);
    }

    let hit_a = &hits[0];
    let hit_b = &hits[1];

    // Both hits on one edge would need an intermediate va→vb sub-edge along
    // the original curve; v1 only handles hits on different edges. The
    // EdgeId comparison also rejects two hits on distinct wire positions of
    // a repeated (seam-style) edge: the second split_edge_at would re-split
    // the original edge that the first propagate_split already rewrote out
    // of every wire, leaving the second sub-edge pair orphaned. Bail before
    // the splits so no wires are mutated on the failure path.
    if hit_a.edge_idx == hit_b.edge_idx
        || edge_data[hit_a.edge_idx].0.edge() == edge_data[hit_b.edge_idx].0.edge()
    {
        {
            if std::env::var("BK_TRIM_TRACE").is_ok() {
                log::warn!("TRIM-FAIL site3 face={face_id:?}");
            }
            return Err(BlendError::TrimmingFailure { face: face_id });
        }
    }

    // Each hit either splits its oriented edge into two sub-edges, or —
    // when it lands on an existing boundary vertex (a previous stripe's
    // propagated split) — reuses that vertex outright: splitting at t=0/1
    // would mint a duplicate vertex and a zero-length sub-edge.
    let ends_a = resolve_hit_ends(topo, edge_data[hit_a.edge_idx].0, hit_a)?;
    let ends_b = resolve_hit_ends(topo, edge_data[hit_b.edge_idx].0, hit_b)?;
    let (va_id, vb_id) = (ends_a.vertex, ends_b.vertex);
    let (sub_a1, sub_a2) = (ends_a.pre, ends_a.post);
    let (sub_b1, sub_b2) = (ends_b.pre, ends_b.post);

    // The contact edge connects the two intersection vertices.
    // Its direction determines which side is "left" vs "right".
    let contact_edge_id = topo.add_edge(Edge::new(va_id, vb_id, EdgeCurve::Line));

    // The contact line divides the boundary edges into two chains:
    //   Chain 1: edges from hit_a to hit_b (in wire order)
    //   Chain 2: edges from hit_b to hit_a (wrapping around)
    // "Left" = chain 1 side, "Right" = chain 2 side (relative to
    // the contact direction va→vb and the face normal).

    let n_edges = edge_data.len();

    let mut chain1: Vec<OrientedEdge> = Vec::new();
    let mut chain2: Vec<OrientedEdge> = Vec::new();

    if let Some(oe) = sub_a2 {
        chain1.push(oe);
    }
    for i in (hit_a.edge_idx + 1)..hit_b.edge_idx {
        chain1.push(oriented_edges[i]);
    }
    if let Some(oe) = sub_b1 {
        chain1.push(oe);
    }

    if let Some(oe) = sub_b2 {
        chain2.push(oe);
    }
    for i in (hit_b.edge_idx + 1)..n_edges {
        chain2.push(oriented_edges[i]);
    }
    for i in 0..hit_a.edge_idx {
        chain2.push(oriented_edges[i]);
    }
    if let Some(oe) = sub_a1 {
        chain2.push(oe);
    }

    // Use the face plane normal and contact direction to determine left/right.
    let face_normal = match &surface {
        FaceSurface::Plane { normal, .. } => {
            if reversed {
                -*normal
            } else {
                *normal
            }
        }
        FaceSurface::Cylinder(_)
        | FaceSurface::Cone(_)
        | FaceSurface::Sphere(_)
        | FaceSurface::Torus(_)
        | FaceSurface::Nurbs(_) => return Err(BlendError::TrimmingFailure { face: face_id }),
    };

    let contact_dir = hit_b.point_3d - hit_a.point_3d;

    // Take a sample point from chain 1 to determine which side it is on.
    let sample_pt = edge_data
        .get(if hit_a.edge_idx + 1 < hit_b.edge_idx {
            hit_a.edge_idx + 1
        } else {
            hit_a.edge_idx
        })
        .map(|(_, s, _)| *s)
        .ok_or(BlendError::TrimmingFailure { face: face_id })?;

    let to_sample = sample_pt - hit_a.point_3d;
    let cross = contact_dir.cross(to_sample);
    let chain1_is_left = face_normal.dot(cross) > 0.0;

    let keep_side = match keep {
        TrimKeep::Side(side) => side,
        TrimKeep::AwayFrom(p) => {
            let p_is_left = face_normal.dot(contact_dir.cross(p - hit_a.point_3d)) > 0.0;
            if p_is_left {
                TrimSide::Right
            } else {
                TrimSide::Left
            }
        }
    };

    // chain1 runs va→…→vb, so the contact edge (va→vb) closes it REVERSED;
    // chain2 runs vb→…→va and closes with the contact edge forward.
    let (kept_chain, contact_forward) = match keep_side {
        TrimSide::Left => {
            if chain1_is_left {
                (chain1, false)
            } else {
                (chain2, true)
            }
        }
        TrimSide::Right => {
            if chain1_is_left {
                (chain2, true)
            } else {
                (chain1, false)
            }
        }
    };

    let mut loop_edges = kept_chain;
    loop_edges.push(OrientedEdge::new(contact_edge_id, contact_forward));

    let trimmed_wire =
        Wire::new(loop_edges, true).map_err(|_| BlendError::TrimmingFailure { face: face_id })?;
    let trimmed_wire_id = topo.add_wire(trimmed_wire);

    let mut trimmed_face = Face::new(trimmed_wire_id, Vec::new(), surface);
    if reversed {
        trimmed_face.set_reversed(true);
    }
    let trimmed_face_id = topo.add_face(trimmed_face);

    let mut new_edges = ends_a.minted_edges;
    new_edges.extend(ends_b.minted_edges);
    let mut new_vertices = Vec::new();
    new_vertices.extend(ends_a.minted_vertex);
    new_vertices.extend(ends_b.minted_vertex);

    Ok(TrimResult {
        trimmed_face: trimmed_face_id,
        new_edges,
        new_vertices,
        contact_edge: Some(contact_edge_id),
    })
}

/// How one boundary hit resolves: the vertex at the crossing plus the
/// boundary pieces before/after it in wire order (`None` when the hit sits
/// on the edge's own endpoint and no split is needed).
struct HitEnds {
    vertex: VertexId,
    pre: Option<OrientedEdge>,
    post: Option<OrientedEdge>,
    minted_vertex: Option<VertexId>,
    minted_edges: Vec<EdgeId>,
}

fn resolve_hit_ends(
    topo: &mut Topology,
    oe: OrientedEdge,
    hit: &BoundaryHit,
) -> Result<HitEnds, BlendError> {
    let edge = topo.edge(oe.edge())?;
    let (sv, ev) = (oe.oriented_start(edge), oe.oriented_end(edge));
    let sp = topo.vertex(sv)?.point();
    let ep = topo.vertex(ev)?.point();
    if (hit.point_3d - sp).length() < 1e-6 {
        return Ok(HitEnds {
            vertex: sv,
            pre: None,
            post: Some(oe),
            minted_vertex: None,
            minted_edges: Vec::new(),
        });
    }
    if (hit.point_3d - ep).length() < 1e-6 {
        return Ok(HitEnds {
            vertex: ev,
            pre: Some(oe),
            post: None,
            minted_vertex: None,
            minted_edges: Vec::new(),
        });
    }
    let v = topo.add_vertex(Vertex::new(hit.point_3d, VERTEX_TOL));
    let (s1, s2) = split_edge_at(topo, &oe, v)?;
    Ok(HitEnds {
        vertex: v,
        pre: Some(s1),
        post: Some(s2),
        minted_vertex: Some(v),
        minted_edges: vec![s1.edge(), s2.edge()],
    })
}

/// Split an oriented boundary edge at a new vertex, producing two sub-edges.
///
/// Returns `(before, after)` as [`OrientedEdge`] values following the same
/// traversal direction as the input.
fn split_edge_at(
    topo: &mut Topology,
    oe: &OrientedEdge,
    split_vertex: VertexId,
) -> Result<(OrientedEdge, OrientedEdge), BlendError> {
    let edge = topo.edge(oe.edge())?;
    let start_vid = oe.oriented_start(edge);
    let end_vid = oe.oriented_end(edge);
    let curve = edge.curve().clone();

    let e1_id = topo.add_edge(Edge::new(start_vid, split_vertex, curve.clone()));
    let e2_id = topo.add_edge(Edge::new(split_vertex, end_vid, curve));

    propagate_split(topo, oe.edge(), oe.is_forward(), e1_id, e2_id)?;

    // Both sub-edges are traversed forward in the oriented direction
    // because we constructed them S→V and V→E matching the traversal.
    Ok((
        OrientedEdge::new(e1_id, true),
        OrientedEdge::new(e2_id, true),
    ))
}

/// Rewrite every wire referencing the split edge to use its two sub-edges.
///
/// A boundary edge crossed by a contact curve is usually shared with a
/// neighbor face that is not itself trimmed (a cap or rim face). Rebuilding
/// only the trimmed face's wire would leave that neighbor referencing the
/// old unsplit edge: the kept sub-edge and the stale edge each end up used
/// by a single face, opening the shell along the shared span.
///
/// `split_forward` is the traversal direction the sub-edges were built in:
/// `e1` runs oriented-start→vertex and `e2` vertex→oriented-end for an
/// occurrence with that orientation; an opposite occurrence traverses
/// `e2` then `e1`, both reversed.
fn propagate_split(
    topo: &mut Topology,
    old_edge: EdgeId,
    split_forward: bool,
    e1: EdgeId,
    e2: EdgeId,
) -> Result<(), BlendError> {
    let mut updates: Vec<(WireId, Vec<OrientedEdge>, bool)> = Vec::new();
    for (wid, wire) in topo.wires().iter() {
        if !wire.edges().iter().any(|oe| oe.edge() == old_edge) {
            continue;
        }
        let mut new_edges: Vec<OrientedEdge> = Vec::with_capacity(wire.edges().len() + 1);
        for oe in wire.edges() {
            if oe.edge() == old_edge {
                if oe.is_forward() == split_forward {
                    new_edges.push(OrientedEdge::new(e1, true));
                    new_edges.push(OrientedEdge::new(e2, true));
                } else {
                    new_edges.push(OrientedEdge::new(e2, false));
                    new_edges.push(OrientedEdge::new(e1, false));
                }
            } else {
                new_edges.push(*oe);
            }
        }
        updates.push((wid, new_edges, wire.is_closed()));
    }
    for (wid, edges, closed) in updates {
        *topo.wire_mut(wid)? = Wire::new(edges, closed)?;
    }
    // Drop registry pcurves keyed by the now-unreferenced edge so per-face
    // enumeration (pcurves_for_face) cannot pick up a stale full-span entry.
    // The sub-edges deliberately get none: downstream consumers regenerate
    // lazily (boolean assembly) or fall back to direct surface projection
    // (tessellation), matching every other edge the blend engine creates.
    let stale_faces: Vec<FaceId> = topo
        .pcurves()
        .pcurves_for_edge(old_edge)
        .into_iter()
        .map(|(fid, _)| fid)
        .collect();
    for fid in stale_faces {
        topo.pcurves_mut().remove(old_edge, fid);
    }
    Ok(())
}

/// Compute a local 2D coordinate frame for a planar face.
///
/// Returns `(origin, u_axis, v_axis)` where `origin` is the first vertex
/// position and `u_axis`, `v_axis` span the plane.
fn plane_local_frame(
    surface: &FaceSurface,
    edge_data: &[(OrientedEdge, Point3, Point3)],
    face_id: FaceId,
) -> Result<(Point3, brepkit_math::vec::Vec3, brepkit_math::vec::Vec3), BlendError> {
    use brepkit_math::vec::Vec3;

    let normal = match surface {
        FaceSurface::Plane { normal, .. } => *normal,
        FaceSurface::Cylinder(_)
        | FaceSurface::Cone(_)
        | FaceSurface::Sphere(_)
        | FaceSurface::Torus(_)
        | FaceSurface::Nurbs(_) => return Err(BlendError::TrimmingFailure { face: face_id }),
    };

    let origin = edge_data
        .first()
        .map(|(_, pt, _)| *pt)
        .ok_or(BlendError::TrimmingFailure { face: face_id })?;

    // U-axis: direction along first edge.
    let first_dir = edge_data[0].2 - edge_data[0].1;
    let u_axis = first_dir.normalize().unwrap_or(Vec3::new(1.0, 0.0, 0.0));

    // V-axis: normal × u to complete a right-handed frame.
    let v_axis = normal
        .cross(u_axis)
        .normalize()
        .unwrap_or(Vec3::new(0.0, 1.0, 0.0));

    Ok((origin, u_axis, v_axis))
}

/// Test if two 2D line segments intersect (both constrained to `[0,1]`).
///
/// Used only in tests; production code uses [`line_segment_intersect_2d`].
#[cfg(test)]
fn segment_intersect_2d(
    a1: (f64, f64),
    a2: (f64, f64),
    b1: (f64, f64),
    b2: (f64, f64),
) -> Option<f64> {
    let dx_a = a2.0 - a1.0;
    let dy_a = a2.1 - a1.1;
    let dx_b = b2.0 - b1.0;
    let dy_b = b2.1 - b1.1;

    let denom = dx_a * dy_b - dy_a * dx_b;

    // Parallel or degenerate segments.
    if denom.abs() < PARAM_TOL {
        return None;
    }

    let dx_ab = b1.0 - a1.0;
    let dy_ab = b1.1 - a1.1;

    let t = (dx_ab * dy_b - dy_ab * dx_b) / denom;
    let u = (dx_ab * dy_a - dy_ab * dx_a) / denom;

    // Both parameters must be in [0, 1] for a proper crossing.
    // Use a small tolerance to catch intersections at edge endpoints.
    let valid_range = -PARAM_TOL..=(1.0 + PARAM_TOL);
    if valid_range.contains(&t) && valid_range.contains(&u) {
        Some(t.clamp(0.0, 1.0))
    } else {
        None
    }
}

/// Intersect an infinite line through `(b1→b2)` with segment `(a1→a2)`.
///
/// The segment parameter `t` on `a` is constrained to `[0, 1]`, but the
/// line parameter `u` on `b` is unconstrained. This is needed for contact
/// line trimming where the contact line extends beyond the face boundary.
///
/// Returns `Some(t)` on segment `a` if the crossing exists.
fn line_segment_intersect_2d(
    a1: (f64, f64),
    a2: (f64, f64),
    b1: (f64, f64),
    b2: (f64, f64),
) -> Option<f64> {
    let dx_a = a2.0 - a1.0;
    let dy_a = a2.1 - a1.1;
    let dx_b = b2.0 - b1.0;
    let dy_b = b2.1 - b1.1;

    let denom = dx_a * dy_b - dy_a * dx_b;

    // Parallel or degenerate.
    if denom.abs() < PARAM_TOL {
        return None;
    }

    let dx_ab = b1.0 - a1.0;
    let dy_ab = b1.1 - a1.1;

    let t = (dx_ab * dy_b - dy_ab * dx_b) / denom;
    // u is unconstrained — the contact line extends infinitely.

    // Only constrain t (the segment parameter).
    let valid_range = -PARAM_TOL..=(1.0 + PARAM_TOL);
    if valid_range.contains(&t) {
        Some(t.clamp(0.0, 1.0))
    } else {
        None
    }
}

// ===========================================================================
// General trimmer (planar + non-planar)
// ===========================================================================

/// Trim a face along a 3D contact curve, handling both planar and non-planar
/// surfaces.
///
/// For planar faces, delegates to [`trim_face`] using the plane's local frame
/// to compute UV coordinates. For non-planar surfaces, projects the contact
/// curve and boundary edges to the surface's UV space and performs 2D
/// intersection there.
///
/// Falls back to returning the face untrimmed if UV projection fails.
///
/// # Errors
///
/// Returns [`BlendError::TrimmingFailure`] on topology or intersection errors.
#[allow(clippy::too_many_lines)]
pub fn trim_face_general(
    topo: &mut Topology,
    face_id: FaceId,
    contact_3d: &[Point3],
    keep: TrimKeep,
) -> Result<TrimResult, BlendError> {
    if contact_3d.len() < 2 {
        {
            if std::env::var("BK_TRIM_TRACE").is_ok() {
                log::warn!("TRIM-FAIL site4 face={face_id:?}");
            }
            return Err(BlendError::TrimmingFailure { face: face_id });
        }
    }

    let face = topo.face(face_id)?;
    let surface = face.surface().clone();

    // Planar path: construct UV from plane frame and delegate
    if let FaceSurface::Plane { normal, d } = &surface {
        let arbitrary = if normal.x().abs() < 0.9 {
            brepkit_math::vec::Vec3::new(1.0, 0.0, 0.0)
        } else {
            brepkit_math::vec::Vec3::new(0.0, 1.0, 0.0)
        };
        let u_axis = normal.cross(arbitrary);
        let u_len = u_axis.length();
        if u_len < 1e-12 {
            return Ok(untrimmed_result(face_id));
        }
        let u_axis = u_axis * (1.0 / u_len);
        let v_axis = normal.cross(u_axis);

        // Origin: any point on the plane
        let origin = *normal * *d;

        let contact_uv: Vec<(f64, f64)> = contact_3d
            .iter()
            .map(|p| {
                let rel = brepkit_math::vec::Vec3::new(
                    p.x() - origin.x(),
                    p.y() - origin.y(),
                    p.z() - origin.z(),
                );
                (rel.dot(u_axis), rel.dot(v_axis))
            })
            .collect();

        return trim_face(topo, face_id, contact_3d, &contact_uv, keep);
    }

    // Non-planar path: project to UV space
    let uv_start = surface.project_point(contact_3d[0]);
    let uv_end = surface.project_point(contact_3d[contact_3d.len() - 1]);

    let (Some(mut uv_s), Some(mut uv_e)) = (uv_start, uv_end) else {
        log::warn!(
            "trim_face_general: UV projection failed for non-planar face {face_id:?}, returning untrimmed"
        );
        return Ok(untrimmed_result(face_id));
    };

    let face = topo.face(face_id)?;
    let reversed = face.is_reversed();
    let outer_wire_id = face.outer_wire();
    let outer_wire = topo.wire(outer_wire_id)?;
    let oriented_edges: Vec<OrientedEdge> = outer_wire.edges().to_vec();

    #[allow(clippy::type_complexity)]
    let mut edge_data_uv: Vec<(OrientedEdge, Point3, Point3, (f64, f64), (f64, f64))> =
        Vec::with_capacity(oriented_edges.len());
    for &oe in &oriented_edges {
        let edge = topo.edge(oe.edge())?;
        let (s3, e3) = if oe.is_forward() {
            (
                topo.vertex(edge.start())?.point(),
                topo.vertex(edge.end())?.point(),
            )
        } else {
            (
                topo.vertex(edge.end())?.point(),
                topo.vertex(edge.start())?.point(),
            )
        };
        let s_uv = surface.project_point(s3);
        let e_uv = surface.project_point(e3);
        let (Some(s_uv), Some(e_uv)) = (s_uv, e_uv) else {
            log::warn!(
                "trim_face_general: boundary UV projection failed for face {face_id:?}, returning untrimmed"
            );
            return Ok(untrimmed_result(face_id));
        };
        edge_data_uv.push((oe, s3, e3, s_uv, e_uv));
    }

    // Periodic surfaces project u into [0, TAU), so a face crossing the
    // parametric seam gets boundary chords that jump a full period and
    // sweep spuriously across the contact line (3 hits instead of 2, and
    // the trim bails). Unwrap u sequentially along the wire so consecutive
    // chords stay continuous, then align the contact endpoints to the same
    // branch.
    if matches!(
        &surface,
        FaceSurface::Cylinder(_) | FaceSurface::Cone(_) | FaceSurface::Sphere(_)
    ) {
        let period = std::f64::consts::TAU;
        let align = |u: f64, reference: f64| u - period * ((u - reference) / period).round();
        let mut prev_u = edge_data_uv[0].3.0;
        for (_, _, _, s_uv, e_uv) in &mut edge_data_uv {
            s_uv.0 = align(s_uv.0, prev_u);
            e_uv.0 = align(e_uv.0, s_uv.0);
            prev_u = e_uv.0;
        }
        let mid_u = edge_data_uv
            .iter()
            .map(|(_, _, _, s_uv, _)| s_uv.0)
            .sum::<f64>()
            / edge_data_uv.len() as f64;
        uv_s.0 = align(uv_s.0, mid_u);
        uv_e.0 = align(uv_e.0, uv_s.0);
    }

    let mut hits: Vec<BoundaryHit> = Vec::new();
    for (edge_idx, (_oe, s3, e3, s_uv, e_uv)) in edge_data_uv.iter().enumerate() {
        if let Some(t) = line_segment_intersect_2d(*s_uv, *e_uv, uv_s, uv_e) {
            let point_3d = *s3 + (*e3 - *s3) * t;
            hits.push(BoundaryHit {
                edge_idx,
                t,
                point_3d,
            });
        }
    }

    // A contact endpoint landing exactly on an existing boundary VERTEX
    // (a previous stripe's propagated split) registers on both incident
    // chords — one geometric crossing counted twice. Merge
    // position-coincident hits before judging the count.
    hits.sort_by(|a, b| {
        (a.edge_idx, a.t)
            .partial_cmp(&(b.edge_idx, b.t))
            .unwrap_or(std::cmp::Ordering::Equal)
    });
    hits.dedup_by(|b, a| (b.point_3d - a.point_3d).length() < 1e-6);
    if hits.len() > 1 && (hits[0].point_3d - hits[hits.len() - 1].point_3d).length() < 1e-6 {
        hits.pop();
    }

    if hits.len() != 2 {
        if std::env::var("BK_TRIM_TRACE").is_ok() {
            log::warn!("TRIM-TRACE face={face_id:?} contact ({uv_s:?})->({uv_e:?})");
            for (i, (_, _, _, s_uv, e_uv)) in edge_data_uv.iter().enumerate() {
                let hit = hits.iter().find(|h| h.edge_idx == i).map(|h| h.t);
                log::warn!("TRIM-TRACE   chord[{i}] ({s_uv:?})->({e_uv:?}) hit={hit:?}");
            }
        }
        log::warn!(
            "trim_face_general: expected 2 boundary hits, got {} for face {face_id:?}, returning untrimmed",
            hits.len()
        );
        return Ok(untrimmed_result(face_id));
    }

    // Sort hits by edge index (to process in wire order)
    hits.sort_by_key(|h| (h.edge_idx, (h.t * 1e10) as i64));

    let hit_a = &hits[0];
    let hit_b = &hits[1];

    // Build the trimmed wire loop: walk the boundary, replacing split edges,
    // inserting the contact edge at the appropriate point.
    let idx_a = hit_a.edge_idx;
    let idx_b = hit_b.edge_idx;

    let oe_a = edge_data_uv[idx_a].0;
    let oe_b = edge_data_uv[idx_b].0;

    // Same wire position, or two positions of a repeated (seam-style) edge:
    // the second split would re-split the edge the first propagate_split
    // already rewrote out of every wire. Bail before any mutation.
    if idx_a == idx_b || oe_a.edge() == oe_b.edge() {
        {
            if std::env::var("BK_TRIM_TRACE").is_ok() {
                log::warn!("TRIM-FAIL site5 face={face_id:?}");
            }
            return Err(BlendError::TrimmingFailure { face: face_id });
        }
    }

    let ends_a = resolve_hit_ends(topo, oe_a, hit_a)?;
    let ends_b = resolve_hit_ends(topo, oe_b, hit_b)?;
    let (va, vb) = (ends_a.vertex, ends_b.vertex);

    let contact_eid = topo.add_edge(Edge::new(va, vb, EdgeCurve::Line));

    // Build "left" side wire: edges from idx_a..idx_b + contact edge.
    // Split sub-edges are created in traversal order, so they use
    // forward=true; endpoint hits contribute the whole original edge on
    // one side and nothing on the other. Existing boundary edges keep
    // their original orientation.
    let mut left_edges: Vec<OrientedEdge> = Vec::new();
    if let Some(oe) = ends_a.post {
        left_edges.push(oe);
    }
    for i in (idx_a + 1)..idx_b {
        left_edges.push(oriented_edges[i]);
    }
    if let Some(oe) = ends_b.pre {
        left_edges.push(oe);
    }
    left_edges.push(OrientedEdge::new(contact_eid, false));

    let mut right_edges: Vec<OrientedEdge> = Vec::new();
    if let Some(oe) = ends_b.post {
        right_edges.push(oe);
    }
    let n = oriented_edges.len();
    for i in 1..(n - (idx_b - idx_a)) {
        let idx = (idx_b + i) % n;
        right_edges.push(oriented_edges[idx]);
    }
    if let Some(oe) = ends_a.pre {
        right_edges.push(oe);
    }
    right_edges.push(OrientedEdge::new(contact_eid, true));

    let keep_side = match keep {
        TrimKeep::Side(side) => side,
        TrimKeep::AwayFrom(p) => {
            // The side test is local to hit_a, so a curved surface's normal
            // AT the hit serves the same role the plane normal does.
            let raw_normal = match &surface {
                FaceSurface::Plane { normal, .. } => *normal,
                _ => match surface.project_point(hit_a.point_3d) {
                    Some((u, v)) => surface.normal(u, v),
                    None => return Err(BlendError::TrimmingFailure { face: face_id }),
                },
            };
            let face_normal = if reversed { -raw_normal } else { raw_normal };
            let contact_dir = hit_b.point_3d - hit_a.point_3d;
            let left_sample = (idx_a..idx_b).rev().find_map(|i| {
                let oe = oriented_edges[i];
                let e = topo.edge(oe.edge()).ok()?;
                let vid = if oe.is_forward() { e.end() } else { e.start() };
                let q = topo.vertex(vid).ok()?.point();
                let side = face_normal.dot(contact_dir.cross(q - hit_a.point_3d));
                (side.abs() > 1e-12).then_some(side > 0.0)
            });
            let Some(left_chain_is_left) = left_sample else {
                {
                    if std::env::var("BK_TRIM_TRACE").is_ok() {
                        log::warn!("TRIM-FAIL site6 face={face_id:?}");
                    }
                    return Err(BlendError::TrimmingFailure { face: face_id });
                }
            };
            let p_is_left = face_normal.dot(contact_dir.cross(p - hit_a.point_3d)) > 0.0;
            if p_is_left == left_chain_is_left {
                TrimSide::Right
            } else {
                TrimSide::Left
            }
        }
    };
    let (keep_edges, _contact_forward) = match keep_side {
        TrimSide::Left => (left_edges, false),
        TrimSide::Right => (right_edges, true),
    };

    if keep_edges.is_empty() {
        return Ok(untrimmed_result(face_id));
    }

    let new_wire = Wire::new(keep_edges, true)?;
    let new_wire_id = topo.add_wire(new_wire);

    // Preserve inner wires from the original face
    let face = topo.face(face_id)?;
    let inner_wires = face.inner_wires().to_vec();
    let mut new_face = Face::new(new_wire_id, inner_wires, surface);
    new_face.set_reversed(reversed);
    let new_face_id = topo.add_face(new_face);

    Ok(TrimResult {
        trimmed_face: new_face_id,
        new_edges: {
            let mut v = ends_a.minted_edges;
            v.extend(ends_b.minted_edges);
            v
        },
        new_vertices: ends_a
            .minted_vertex
            .into_iter()
            .chain(ends_b.minted_vertex)
            .collect(),
        contact_edge: Some(contact_eid),
    })
}

/// Create an untrimmed result (face returned as-is).
fn untrimmed_result(face_id: FaceId) -> TrimResult {
    TrimResult {
        trimmed_face: face_id,
        new_edges: Vec::new(),
        new_vertices: Vec::new(),
        contact_edge: None,
    }
}

// ===========================================================================
// Tests
// ===========================================================================

#[cfg(test)]
mod tests {
    #![allow(clippy::unwrap_used, clippy::expect_used, clippy::panic)]

    use super::*;
    use brepkit_math::vec::{Point3, Vec3};
    use brepkit_topology::Topology;

    /// Helper: create a unit square face on the XY plane (z=0).
    ///
    /// Vertices:
    ///   v0 = (0,0,0), v1 = (1,0,0), v2 = (1,1,0), v3 = (0,1,0)
    ///
    /// Edges: v0→v1, v1→v2, v2→v3, v3→v0 (all Line, forward).
    fn make_square_face(topo: &mut Topology) -> (FaceId, [VertexId; 4], [EdgeId; 4]) {
        let v0 = topo.add_vertex(Vertex::new(Point3::new(0.0, 0.0, 0.0), VERTEX_TOL));
        let v1 = topo.add_vertex(Vertex::new(Point3::new(1.0, 0.0, 0.0), VERTEX_TOL));
        let v2 = topo.add_vertex(Vertex::new(Point3::new(1.0, 1.0, 0.0), VERTEX_TOL));
        let v3 = topo.add_vertex(Vertex::new(Point3::new(0.0, 1.0, 0.0), VERTEX_TOL));

        let e0 = topo.add_edge(Edge::new(v0, v1, EdgeCurve::Line)); // bottom
        let e1 = topo.add_edge(Edge::new(v1, v2, EdgeCurve::Line)); // right
        let e2 = topo.add_edge(Edge::new(v2, v3, EdgeCurve::Line)); // top
        let e3 = topo.add_edge(Edge::new(v3, v0, EdgeCurve::Line)); // left

        let wire = Wire::new(
            vec![
                OrientedEdge::new(e0, true),
                OrientedEdge::new(e1, true),
                OrientedEdge::new(e2, true),
                OrientedEdge::new(e3, true),
            ],
            true,
        )
        .unwrap();
        let wire_id = topo.add_wire(wire);

        let surface = FaceSurface::Plane {
            normal: Vec3::new(0.0, 0.0, 1.0),
            d: 0.0,
        };
        let face = Face::new(wire_id, Vec::new(), surface);
        let face_id = topo.add_face(face);

        (face_id, [v0, v1, v2, v3], [e0, e1, e2, e3])
    }

    #[test]
    fn trim_through_a_presplit_boundary_vertex_succeeds() {
        // A previous stripe's propagated split leaves the boundary already
        // split exactly where this stripe's contact line meets it. The hit
        // registers on BOTH incident chords (t=1 on one, t=0 on the next)
        // — one geometric crossing counted twice — and the 2-hit gate used
        // to bail the whole trim.
        let mut topo = Topology::new();
        let v0 = topo.add_vertex(Vertex::new(Point3::new(0.0, 0.0, 0.0), VERTEX_TOL));
        let vm = topo.add_vertex(Vertex::new(Point3::new(0.5, 0.0, 0.0), VERTEX_TOL));
        let v1 = topo.add_vertex(Vertex::new(Point3::new(1.0, 0.0, 0.0), VERTEX_TOL));
        let v2 = topo.add_vertex(Vertex::new(Point3::new(1.0, 1.0, 0.0), VERTEX_TOL));
        let v3 = topo.add_vertex(Vertex::new(Point3::new(0.0, 1.0, 0.0), VERTEX_TOL));
        let e0a = topo.add_edge(Edge::new(v0, vm, EdgeCurve::Line));
        let e0b = topo.add_edge(Edge::new(vm, v1, EdgeCurve::Line));
        let e1 = topo.add_edge(Edge::new(v1, v2, EdgeCurve::Line));
        let e2 = topo.add_edge(Edge::new(v2, v3, EdgeCurve::Line));
        let e3 = topo.add_edge(Edge::new(v3, v0, EdgeCurve::Line));
        let wire = Wire::new(
            vec![
                OrientedEdge::new(e0a, true),
                OrientedEdge::new(e0b, true),
                OrientedEdge::new(e1, true),
                OrientedEdge::new(e2, true),
                OrientedEdge::new(e3, true),
            ],
            true,
        )
        .unwrap();
        let wire_id = topo.add_wire(wire);
        let face_id = topo.add_face(Face::new(
            wire_id,
            Vec::new(),
            FaceSurface::Plane {
                normal: Vec3::new(0.0, 0.0, 1.0),
                d: 0.0,
            },
        ));

        let contact_3d = vec![Point3::new(0.5, 0.0, 0.0), Point3::new(0.5, 1.0, 0.0)];
        let contact_uv = vec![(0.5, 0.0), (0.5, 1.0)];
        let result = trim_face(
            &mut topo,
            face_id,
            &contact_3d,
            &contact_uv,
            TrimKeep::Side(TrimSide::Left),
        )
        .expect("trim through a pre-split vertex should succeed");
        assert_ne!(result.trimmed_face, face_id, "face must actually trim");
        assert!(result.contact_edge.is_some());
    }

    #[test]
    fn trim_square_face_with_diagonal() {
        let mut topo = Topology::new();
        let (face_id, _verts, _edges) = make_square_face(&mut topo);

        // Contact line: from bottom edge midpoint (0.5, 0) to top edge midpoint (0.5, 1).
        // This is a vertical line splitting the square in half.
        let contact_3d = vec![Point3::new(0.5, 0.0, 0.0), Point3::new(0.5, 1.0, 0.0)];
        let contact_uv = vec![(0.5, 0.0), (0.5, 1.0)];

        let result = trim_face(
            &mut topo,
            face_id,
            &contact_3d,
            &contact_uv,
            TrimKeep::Side(TrimSide::Left),
        )
        .expect("trim should succeed");

        // The trimmed face should have a new wire.
        let trimmed_face = topo.face(result.trimmed_face).unwrap();
        let trimmed_wire = topo.wire(trimmed_face.outer_wire()).unwrap();

        // Expect 4 edges: bottom-half, right-full, top-half, contact-edge
        // (for the right side) or bottom-half, contact-edge, top-half, left-full
        // (for the left side).
        assert_eq!(
            trimmed_wire.edges().len(),
            4,
            "trimmed wire should have 4 edges"
        );

        // 2 new vertices at the intersection points.
        assert_eq!(result.new_vertices.len(), 2);

        // 4 new sub-edges from splitting 2 boundary edges.
        assert_eq!(result.new_edges.len(), 4);

        // Verify intersection vertex positions.
        let va = topo.vertex(result.new_vertices[0]).unwrap().point();
        let vb = topo.vertex(result.new_vertices[1]).unwrap().point();
        // One should be at (0.5, 0, 0) and the other at (0.5, 1, 0).
        let pts: Vec<(f64, f64, f64)> = vec![(va.x(), va.y(), va.z()), (vb.x(), vb.y(), vb.z())];
        assert!(
            pts.iter()
                .any(|p| (p.0 - 0.5).abs() < 1e-10 && p.1.abs() < 1e-10),
            "expected intersection at (0.5, 0, 0)"
        );
        assert!(
            pts.iter()
                .any(|p| (p.0 - 0.5).abs() < 1e-10 && (p.1 - 1.0).abs() < 1e-10),
            "expected intersection at (0.5, 1, 0)"
        );
    }

    /// Attach a neighbor square below the shared bottom edge `e0` of
    /// [`make_square_face`], in the y=0 plane. The neighbor traverses `e0`
    /// reversed (manifold convention).
    fn attach_neighbor_below(
        topo: &mut Topology,
        v0: VertexId,
        v1: VertexId,
        e0: EdgeId,
    ) -> FaceId {
        let v4 = topo.add_vertex(Vertex::new(Point3::new(1.0, 0.0, -1.0), VERTEX_TOL));
        let v5 = topo.add_vertex(Vertex::new(Point3::new(0.0, 0.0, -1.0), VERTEX_TOL));
        let e5 = topo.add_edge(Edge::new(v0, v5, EdgeCurve::Line));
        let e6 = topo.add_edge(Edge::new(v5, v4, EdgeCurve::Line));
        let e7 = topo.add_edge(Edge::new(v4, v1, EdgeCurve::Line));
        let wire = Wire::new(
            vec![
                OrientedEdge::new(e0, false),
                OrientedEdge::new(e5, true),
                OrientedEdge::new(e6, true),
                OrientedEdge::new(e7, true),
            ],
            true,
        )
        .unwrap();
        let wire_id = topo.add_wire(wire);
        let surface = FaceSurface::Plane {
            normal: Vec3::new(0.0, -1.0, 0.0),
            d: 0.0,
        };
        topo.add_face(Face::new(wire_id, Vec::new(), surface))
    }

    /// Verify each oriented edge's end vertex matches the next one's start.
    fn assert_wire_connected(topo: &Topology, face_id: FaceId) {
        let wire = topo.wire(topo.face(face_id).unwrap().outer_wire()).unwrap();
        let oes = wire.edges();
        for i in 0..oes.len() {
            let cur = topo.edge(oes[i].edge()).unwrap();
            let next_oe = oes[(i + 1) % oes.len()];
            let next = topo.edge(next_oe.edge()).unwrap();
            assert_eq!(
                oes[i].oriented_end(cur),
                next_oe.oriented_start(next),
                "wire of face {face_id:?} is disconnected at position {i}"
            );
        }
    }

    #[test]
    fn split_propagates_into_neighbor_wire() {
        let mut topo = Topology::new();
        let (face_id, verts, edges) = make_square_face(&mut topo);
        let neighbor = attach_neighbor_below(&mut topo, verts[0], verts[1], edges[0]);

        // Vertical contact line at x = 0.5 splits e0 (shared with the
        // neighbor) and e2 (unshared).
        let contact_3d = vec![Point3::new(0.5, 0.0, 0.0), Point3::new(0.5, 1.0, 0.0)];
        let contact_uv = vec![(0.5, 0.0), (0.5, 1.0)];
        let result = trim_face(
            &mut topo,
            face_id,
            &contact_3d,
            &contact_uv,
            TrimKeep::Side(TrimSide::Left),
        )
        .expect("trim should succeed");

        // The neighbor must no longer reference the stale unsplit edge.
        let neighbor_wire = topo
            .wire(topo.face(neighbor).unwrap().outer_wire())
            .unwrap();
        assert!(
            neighbor_wire.edges().iter().all(|oe| oe.edge() != edges[0]),
            "neighbor still references the split edge {:?}",
            edges[0]
        );
        assert_eq!(
            neighbor_wire.edges().len(),
            5,
            "neighbor wire should gain one edge from the split"
        );
        assert_wire_connected(&topo, neighbor);
        assert_wire_connected(&topo, result.trimmed_face);

        // The neighbor's replacement must traverse v1 -> split vertex -> v0
        // (it referenced e0 reversed), passing through the split point.
        let split_v = result
            .new_vertices
            .iter()
            .copied()
            .find(|&vid| {
                let p = topo.vertex(vid).unwrap().point();
                p.y().abs() < 1e-9
            })
            .expect("split vertex on the shared edge");
        assert!(
            neighbor_wire.edges().iter().any(|oe| {
                let e = topo.edge(oe.edge()).unwrap();
                oe.oriented_start(e) == split_v
            }),
            "neighbor wire should pass through the split vertex"
        );

        // Exactly one sub-edge of the shared split is used by both the
        // trimmed face and the neighbor (the kept side); the other is used
        // by the neighbor alone.
        let trimmed_wire = topo
            .wire(topo.face(result.trimmed_face).unwrap().outer_wire())
            .unwrap();
        let shared_subs: Vec<EdgeId> = neighbor_wire
            .edges()
            .iter()
            .map(OrientedEdge::edge)
            .filter(|eid| trimmed_wire.edges().iter().any(|toe| toe.edge() == *eid))
            .collect();
        assert_eq!(
            shared_subs.len(),
            1,
            "exactly one sub-edge should be shared between the trimmed face \
             and the neighbor, got {shared_subs:?}"
        );
    }

    #[test]
    fn seam_style_repeated_edge_bails_without_mutation() {
        let mut topo = Topology::new();

        // Slit face: one edge traversed forward then reversed, so the same
        // EdgeId occupies two wire positions — the seam configuration. A
        // crossing contact line hits both positions.
        let v0 = topo.add_vertex(Vertex::new(Point3::new(0.0, 0.0, 0.0), VERTEX_TOL));
        let v1 = topo.add_vertex(Vertex::new(Point3::new(1.0, 0.0, 0.0), VERTEX_TOL));
        let e_seam = topo.add_edge(Edge::new(v0, v1, EdgeCurve::Line));
        let wire = Wire::new(
            vec![
                OrientedEdge::new(e_seam, true),
                OrientedEdge::new(e_seam, false),
            ],
            true,
        )
        .unwrap();
        let wire_id = topo.add_wire(wire);
        let surface = FaceSurface::Plane {
            normal: Vec3::new(0.0, 0.0, 1.0),
            d: 0.0,
        };
        let face_id = topo.add_face(Face::new(wire_id, Vec::new(), surface));

        let n_vertices = topo.num_vertices();
        let n_edges = topo.num_edges();

        let contact_3d = vec![Point3::new(0.5, -1.0, 0.0), Point3::new(0.5, 1.0, 0.0)];
        let contact_uv = vec![(0.5, -1.0), (0.5, 1.0)];
        let result = trim_face(
            &mut topo,
            face_id,
            &contact_3d,
            &contact_uv,
            TrimKeep::Side(TrimSide::Left),
        );
        assert!(
            matches!(result, Err(BlendError::TrimmingFailure { .. })),
            "two hits on one repeated edge must be rejected"
        );

        // The failure path must not have mutated anything: no minted
        // vertices/edges and the wire still references the seam edge twice.
        assert_eq!(topo.num_vertices(), n_vertices);
        assert_eq!(topo.num_edges(), n_edges);
        let wire = topo.wire(wire_id).unwrap();
        assert_eq!(wire.edges().len(), 2);
        assert!(wire.edges().iter().all(|oe| oe.edge() == e_seam));
    }

    #[test]
    fn propagate_split_drops_stale_pcurve_entries() {
        use brepkit_math::curves2d::{Curve2D, Line2D};
        use brepkit_math::vec::{Point2, Vec2};
        use brepkit_topology::pcurve::PCurve;

        let mut topo = Topology::new();
        let (face_id, verts, edges) = make_square_face(&mut topo);
        let neighbor = attach_neighbor_below(&mut topo, verts[0], verts[1], edges[0]);

        let line = Line2D::new(Point2::new(0.0, 0.0), Vec2::new(1.0, 0.0)).unwrap();
        topo.pcurves_mut().set(
            edges[0],
            neighbor,
            PCurve::new(Curve2D::Line(line), 0.0, 1.0),
        );

        let contact_3d = vec![Point3::new(0.5, 0.0, 0.0), Point3::new(0.5, 1.0, 0.0)];
        let contact_uv = vec![(0.5, 0.0), (0.5, 1.0)];
        trim_face(
            &mut topo,
            face_id,
            &contact_3d,
            &contact_uv,
            TrimKeep::Side(TrimSide::Left),
        )
        .expect("trim should succeed");

        assert!(
            !topo.pcurves().contains(edges[0], neighbor),
            "stale pcurve entry for the replaced edge must be removed"
        );
    }

    #[test]
    fn trim_preserves_surface() {
        let mut topo = Topology::new();
        let (face_id, _verts, _edges) = make_square_face(&mut topo);

        let contact_3d = vec![Point3::new(0.5, 0.0, 0.0), Point3::new(0.5, 1.0, 0.0)];
        let contact_uv = vec![(0.5, 0.0), (0.5, 1.0)];

        let result = trim_face(
            &mut topo,
            face_id,
            &contact_3d,
            &contact_uv,
            TrimKeep::Side(TrimSide::Right),
        )
        .expect("trim should succeed");

        let original = topo.face(face_id).unwrap();
        let trimmed = topo.face(result.trimmed_face).unwrap();

        // Both should be Plane with the same normal.
        match (original.surface(), trimmed.surface()) {
            (
                FaceSurface::Plane {
                    normal: n1, d: d1, ..
                },
                FaceSurface::Plane {
                    normal: n2, d: d2, ..
                },
            ) => {
                assert!((n1.x() - n2.x()).abs() < 1e-14);
                assert!((n1.y() - n2.y()).abs() < 1e-14);
                assert!((n1.z() - n2.z()).abs() < 1e-14);
                assert!((d1 - d2).abs() < 1e-14);
            }
            _ => panic!("expected both faces to be Plane"),
        }
    }

    #[test]
    fn non_planar_face_returns_untrimmed() {
        use brepkit_math::surfaces::CylindricalSurface;

        let mut topo = Topology::new();

        // Create a minimal face with a cylindrical surface.
        let v0 = topo.add_vertex(Vertex::new(Point3::new(1.0, 0.0, 0.0), VERTEX_TOL));
        let v1 = topo.add_vertex(Vertex::new(Point3::new(0.0, 1.0, 0.0), VERTEX_TOL));
        let e0 = topo.add_edge(Edge::new(v0, v1, EdgeCurve::Line));
        let e1 = topo.add_edge(Edge::new(v1, v0, EdgeCurve::Line));

        let wire = Wire::new(
            vec![OrientedEdge::new(e0, true), OrientedEdge::new(e1, true)],
            true,
        )
        .unwrap();
        let wire_id = topo.add_wire(wire);

        let cyl_surface =
            CylindricalSurface::new(Point3::new(0.0, 0.0, 0.0), Vec3::new(0.0, 0.0, 1.0), 1.0)
                .unwrap();
        let surface = FaceSurface::Cylinder(cyl_surface);
        let face = Face::new(wire_id, Vec::new(), surface);
        let face_id = topo.add_face(face);

        let contact_3d = vec![Point3::new(0.5, 0.0, 0.0), Point3::new(0.5, 1.0, 0.0)];
        let contact_uv = vec![(0.5, 0.0), (0.5, 1.0)];

        let result = trim_face(
            &mut topo,
            face_id,
            &contact_3d,
            &contact_uv,
            TrimKeep::Side(TrimSide::Left),
        )
        .expect("should return untrimmed result");

        // Untrimmed: same face, no new topology.
        assert_eq!(result.trimmed_face, face_id);
        assert!(result.new_edges.is_empty());
        assert!(result.new_vertices.is_empty());
    }

    #[test]
    fn segment_intersect_2d_crossing() {
        // Two crossing segments.
        let t = segment_intersect_2d((0.0, 0.0), (1.0, 1.0), (0.0, 1.0), (1.0, 0.0));
        assert!(t.is_some());
        let t = t.unwrap();
        assert!((t - 0.5).abs() < 1e-10, "t={t}");
    }

    #[test]
    fn segment_intersect_2d_parallel() {
        // Parallel segments — no intersection.
        let t = segment_intersect_2d((0.0, 0.0), (1.0, 0.0), (0.0, 1.0), (1.0, 1.0));
        assert!(t.is_none());
    }

    #[test]
    fn segment_intersect_2d_no_overlap() {
        // Non-parallel but non-overlapping segments.
        let t = segment_intersect_2d((0.0, 0.0), (1.0, 0.0), (2.0, -1.0), (2.0, 1.0));
        assert!(t.is_none());
    }
}
