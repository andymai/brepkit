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
/// Number of carrier-curve samples used to locate UV boundary crossings.
const BATCH_CURVE_SAMPLES: usize = 64;
fn surface_u_period(surface: &FaceSurface) -> Option<f64> {
    match surface {
        FaceSurface::Cylinder(_)
        | FaceSurface::Cone(_)
        | FaceSurface::Sphere(_)
        | FaceSurface::Torus(_) => Some(std::f64::consts::TAU),
        FaceSurface::Nurbs(surface) if surface.is_periodic_u() => {
            let (u0, u1) = surface.domain_u();
            (u1 > u0).then_some(u1 - u0)
        }
        _ => None,
    }
}

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
#[allow(clippy::redundant_pub_crate)]
pub(crate) fn split_edge_at(
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

/// Split an oriented boundary edge at a vertex, assigning each sub-edge its
/// properly TRIMMED sub-curve (for curved edges, where re-anchoring
/// endpoints alone would leave both halves spanning the full stored curve).
///
/// `left`/`right` are the stored-direction sub-curves from `curve_split`.
#[allow(clippy::redundant_pub_crate)]
pub(crate) fn split_edge_at_with_curves(
    topo: &mut Topology,
    oe: &OrientedEdge,
    split_vertex: VertexId,
    left: brepkit_math::nurbs::curve::NurbsCurve,
    right: brepkit_math::nurbs::curve::NurbsCurve,
) -> Result<(OrientedEdge, OrientedEdge), BlendError> {
    let edge = topo.edge(oe.edge())?;
    let (s_v, e_v) = (edge.start(), edge.end());
    let e1_id = topo.add_edge(Edge::new(s_v, split_vertex, EdgeCurve::NurbsCurve(left)));
    let e2_id = topo.add_edge(Edge::new(split_vertex, e_v, EdgeCurve::NurbsCurve(right)));
    propagate_split(topo, oe.edge(), true, e1_id, e2_id)?;
    if oe.is_forward() {
        Ok((
            OrientedEdge::new(e1_id, true),
            OrientedEdge::new(e2_id, true),
        ))
    } else {
        Ok((
            OrientedEdge::new(e2_id, false),
            OrientedEdge::new(e1_id, false),
        ))
    }
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

/// Trim a face along a 3D contact curve, handling planar and parametric
/// surfaces.
///
/// Planar faces use their local frame; every other supported face uses its
/// native UV parameterization. Projection and arrangement failures are
/// reported to the caller: a fillet must never publish an untrimmed
/// non-planar support face.
///
/// # Errors
/// Returns [`BlendError::TrimmingFailure`] on topology, projection, or
/// intersection errors.
///
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

    // Non-planar path: project to UV space. NURBS projection has a total
    // API with a midpoint fallback, so validate every projection by its
    // surface round trip before accepting it.
    let project = |point: Point3| -> Result<(f64, f64), BlendError> {
        let uv = surface
            .project_point(point)
            .ok_or(BlendError::TrimmingFailure { face: face_id })?;
        let evaluated = surface
            .evaluate(uv.0, uv.1)
            .ok_or(BlendError::TrimmingFailure { face: face_id })?;
        if !uv.0.is_finite() || !uv.1.is_finite() || (evaluated - point).length() > 1e-5 {
            return Err(BlendError::TrimmingFailure { face: face_id });
        }
        Ok(uv)
    };
    let (mut uv_s, mut uv_e) = (
        project(contact_3d[0])?,
        project(contact_3d[contact_3d.len() - 1])?,
    );

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
        let s_uv = project(s3)?;
        let e_uv = project(e3)?;
        edge_data_uv.push((oe, s3, e3, s_uv, e_uv));
    }

    // Analytic surfaces wrap their angular parameter at the seam. Unwrap U
    // along the source wire so a seam-crossing face remains one UV chart.
    if let Some(period) = surface_u_period(&surface) {
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
            "trim_face_general: expected 2 boundary hits, got {} for face {face_id:?}",
            hits.len()
        );
        return Err(BlendError::TrimmingFailure { face: face_id });
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

/// One restriction consumed atomically by the parametric UV batch builder.
#[derive(Debug, Clone)]
pub struct ParametricRestriction {
    /// Sampled points of the contact curve, in traversal order.
    pub contact_3d: Vec<Point3>,
    /// Side of the restriction retained in the result.
    pub keep: TrimKeep,
    /// Exact contact geometry, when available.
    pub curve: Option<EdgeCurve>,
    /// Exact contact pcurve on the support face, when available.
    pub pcurve: Option<brepkit_topology::pcurve::PCurve>,
    /// Ordered source boundary edges replaced by this contact.
    ///
    /// An empty list selects the general UV-arrangement path. Fillet contours
    /// populate this list so a support wire can be reconstructed from its
    /// immutable source boundary without inferring junction ownership from
    /// geometric intersections.
    pub source_edges: Vec<EdgeId>,
}

impl ParametricRestriction {
    /// Construct a line restriction from sampled contact points.
    #[must_use]
    pub fn new(contact_3d: Vec<Point3>, keep: TrimKeep) -> Self {
        Self {
            contact_3d,
            keep,
            curve: None,
            pcurve: None,
            source_edges: Vec::new(),
        }
    }
}

/// Legacy name retained for the BK4 planar API; it is the same data contract,
/// not a separate trimming algorithm.
pub type PlanarRestriction = ParametricRestriction;

/// Result of one atomic parametric support-face reconstruction.
#[derive(Debug, Clone)]
pub struct BatchTrimResult {
    /// Replacement for the requested source face.
    pub trimmed_face: FaceId,
    /// Contact edges in restriction order.
    pub contact_edges: Vec<EdgeId>,
    /// Edges copied from source boundaries and contact edges.
    pub new_edges: Vec<EdgeId>,
    /// Vertices minted at arrangement intersections.
    pub new_vertices: Vec<VertexId>,
    /// Copy-on-write replacements for other incident result faces.
    pub incident_replacements: Vec<(FaceId, FaceId)>,
}

#[derive(Clone)]
struct BatchSegment {
    a: usize,
    b: usize,
    edge: EdgeId,
}
/// Rebuild one parametric face from all restrictions in one UV arrangement.
///
/// The source wires are read-only. Boundary edges are split once, and any
/// source neighbour that needs one of those pieces receives a cloned result
/// face. A projection/arrangement failure is returned to the caller.
type BatchWireEdges = Vec<(OrientedEdge, Vec<(f64, usize)>)>;

fn batch_node_for_point(
    topo: &mut Topology,
    nodes: &mut Vec<((f64, f64), Point3, VertexId)>,
    node_for_vertex: &mut std::collections::HashMap<VertexId, usize>,
    q: (f64, f64),
    point: Point3,
    vertex: Option<VertexId>,
) -> usize {
    if let Some(vtx) = vertex
        && let Some(&n) = node_for_vertex.get(&vtx)
    {
        return n;
    }
    if let Some((n, _)) = nodes
        .iter()
        .enumerate()
        .find(|(_, (old, _, _))| (old.0 - q.0).hypot(old.1 - q.1) < 1e-7)
    {
        if let Some(vtx) = vertex {
            node_for_vertex.insert(vtx, n);
        }
        return n;
    }
    let vtx = vertex.unwrap_or_else(|| topo.add_vertex(Vertex::new(point, VERTEX_TOL)));
    let n = nodes.len();
    nodes.push((q, point, vtx));
    if let Some(old) = vertex {
        node_for_vertex.insert(old, n);
    }
    n
}
/// Preserve an edge carrier's native parameter span when creating an
/// arrangement piece. NURBS carriers are exactly knot-split; analytic
/// carriers retain their geometry and derive the trimmed domain from their
/// endpoint vertices.
fn edge_curve_for_span(
    curve: &EdgeCurve,
    curve_t0: f64,
    curve_t1: f64,
) -> Result<EdgeCurve, BlendError> {
    let EdgeCurve::NurbsCurve(source) = curve else {
        return Ok(curve.clone());
    };
    let (d0, d1) = source.domain();
    let eps = (d1 - d0).abs() * 1e-10;
    if (curve_t0 - d0).abs() <= eps && (curve_t1 - d1).abs() <= eps {
        return Ok(curve.clone());
    }
    let (lo, hi) = (curve_t0.min(curve_t1), curve_t0.max(curve_t1));
    if lo < d0 - eps || hi > d1 + eps || (hi - lo) <= eps {
        return Err(BlendError::Math(
            brepkit_math::MathError::ParameterOutOfRange {
                value: lo,
                min: d0,
                max: d1,
            },
        ));
    }
    let mut span = source.clone();
    if lo > d0 + eps {
        span = brepkit_math::nurbs::knot_ops::curve_split(&span, lo)?.1;
    }
    if hi < d1 - eps {
        span = brepkit_math::nurbs::knot_ops::curve_split(&span, hi)?.0;
    }
    if curve_t0 <= curve_t1 {
        return Ok(EdgeCurve::NurbsCurve(span));
    }
    let (a, b) = span.domain();
    let knots = span.knots().iter().rev().map(|&u| a + b - u).collect();
    let control_points = span.control_points().iter().rev().copied().collect();
    let weights = span.weights().iter().rev().copied().collect();
    Ok(EdgeCurve::NurbsCurve(
        brepkit_math::nurbs::curve::NurbsCurve::new(span.degree(), knots, control_points, weights)?,
    ))
}

fn mapped_surface_uv(
    surface: &FaceSurface,
    point: Point3,
    face_id: FaceId,
) -> Result<(f64, f64), BlendError> {
    if let FaceSurface::Plane { normal, d } = surface {
        let reference = if normal.x().abs() < 0.9 {
            brepkit_math::vec::Vec3::new(1.0, 0.0, 0.0)
        } else {
            brepkit_math::vec::Vec3::new(0.0, 1.0, 0.0)
        };
        let u = normal
            .cross(reference)
            .normalize()
            .map_err(|_| BlendError::TrimmingFailure { face: face_id })?;
        let v = normal.cross(u);
        let origin = Point3::new(normal.x() * d, normal.y() * d, normal.z() * d);
        let offset = point - origin;
        let residual = normal.dot(brepkit_math::vec::Vec3::new(
            point.x(),
            point.y(),
            point.z(),
        )) - d;
        if residual.abs() > 1e-5 {
            return Err(BlendError::TrimmingFailure { face: face_id });
        }
        return Ok((u.dot(offset), v.dot(offset)));
    }
    let projected = surface
        .project_point(point)
        .ok_or(BlendError::TrimmingFailure { face: face_id })?;
    let evaluated = surface
        .evaluate(projected.0, projected.1)
        .ok_or(BlendError::TrimmingFailure { face: face_id })?;
    if (evaluated - point).length() > 1e-4 {
        return Err(BlendError::TrimmingFailure { face: face_id });
    }
    Ok(projected)
}

fn mapped_vertex_for_point(
    topo: &mut Topology,
    known: &mut Vec<(Point3, VertexId)>,
    new_vertices: &mut Vec<VertexId>,
    point: Point3,
) -> VertexId {
    if let Some((_, vertex)) = known
        .iter()
        .find(|(candidate, _)| (*candidate - point).length() <= VERTEX_TOL)
    {
        return *vertex;
    }
    let vertex = topo.add_vertex(Vertex::new(point, VERTEX_TOL));
    known.push((point, vertex));
    new_vertices.push(vertex);
    vertex
}

fn mapped_connector_geometry(
    surface: &FaceSurface,
    start: Point3,
    end: Point3,
    face_id: FaceId,
) -> Result<(EdgeCurve, brepkit_topology::pcurve::PCurve), BlendError> {
    let start_uv = mapped_surface_uv(surface, start, face_id)?;
    let mut end_uv = mapped_surface_uv(surface, end, face_id)?;
    if let Some(period) = surface_u_period(surface) {
        end_uv.0 -= period * ((end_uv.0 - start_uv.0) / period).round();
    }
    if surface.is_planar() {
        let pcurve = brepkit_math::curves2d::NurbsCurve2D::from_line(
            brepkit_math::vec::Point2::new(start_uv.0, start_uv.1),
            brepkit_math::vec::Point2::new(end_uv.0, end_uv.1),
        )?;
        return Ok((
            EdgeCurve::Line,
            brepkit_topology::pcurve::PCurve::new(
                brepkit_math::curves2d::Curve2D::Nurbs(pcurve),
                0.0,
                1.0,
            ),
        ));
    }

    // A connector is a short surface path between adjacent contact ends. Use
    // the same piecewise-linear parameterization in UV and 3D. The dense
    // sampling keeps each 3D chord within the support-surface tolerance while
    // avoiding a second projection/fitting algorithm for analytic and NURBS
    // supports.
    let connector_samples = BATCH_CURVE_SAMPLES * 4;
    let mut points = Vec::with_capacity(connector_samples + 1);
    let mut points_uv = Vec::with_capacity(connector_samples + 1);
    for index in 0..=connector_samples {
        let fraction = index as f64 / connector_samples as f64;
        let u = start_uv.0 + (end_uv.0 - start_uv.0) * fraction;
        let v = start_uv.1 + (end_uv.1 - start_uv.1) * fraction;
        let point = if index == 0 {
            start
        } else if index == connector_samples {
            end
        } else {
            surface
                .evaluate(u, v)
                .ok_or(BlendError::TrimmingFailure { face: face_id })?
        };
        points.push(point);
        points_uv.push(brepkit_math::vec::Point2::new(u, v));
    }
    let mut knots = Vec::with_capacity(connector_samples + 3);
    knots.extend([0.0, 0.0]);
    knots.extend((1..connector_samples).map(|index| index as f64 / connector_samples as f64));
    knots.extend([1.0, 1.0]);
    let weights = vec![1.0; points.len()];
    let curve =
        brepkit_math::nurbs::curve::NurbsCurve::new(1, knots.clone(), points, weights.clone())?;
    let pcurve = brepkit_math::curves2d::NurbsCurve2D::new(1, knots, points_uv, weights)?;
    Ok((
        EdgeCurve::NurbsCurve(curve),
        brepkit_topology::pcurve::PCurve::new(
            brepkit_math::curves2d::Curve2D::Nurbs(pcurve),
            0.0,
            1.0,
        ),
    ))
}

fn mapped_planar_corner_geometry(
    surface: &FaceSurface,
    source_vertices: &[(Point3, VertexId)],
    start: Point3,
    end: Point3,
    face_id: FaceId,
) -> Result<Option<(EdgeCurve, brepkit_topology::pcurve::PCurve)>, BlendError> {
    let FaceSurface::Plane { normal, .. } = surface else {
        return Ok(None);
    };
    let mut best: Option<(Point3, f64, brepkit_math::vec::Vec3)> = None;
    for &(center, _) in source_vertices {
        let from_center = start - center;
        let to_center = end - center;
        let start_radius = from_center.length();
        let end_radius = to_center.length();
        if start_radius <= 1e-6
            || (start_radius - end_radius).abs() > 1e-5
            || from_center.cross(to_center).length() <= 1e-8
        {
            continue;
        }
        let Ok(circle_normal) = from_center.cross(to_center).normalize() else {
            continue;
        };
        if circle_normal.dot(*normal).abs() < 1.0 - 1e-6 {
            continue;
        }
        if best.is_none_or(|(_, radius, _)| start_radius < radius) {
            best = Some((center, start_radius, circle_normal));
        }
    }
    let Some((center, radius, circle_normal)) = best else {
        return Ok(None);
    };

    let circle = brepkit_math::curves::Circle3D::new(center, circle_normal, radius)
        .map_err(|_| BlendError::TrimmingFailure { face: face_id })?;
    let center_uv = mapped_surface_uv(surface, center, face_id)?;
    let start_uv = mapped_surface_uv(surface, start, face_id)?;
    let end_uv = mapped_surface_uv(surface, end, face_id)?;
    let circle_2d = brepkit_math::curves2d::Circle2D::new(
        brepkit_math::vec::Point2::new(center_uv.0, center_uv.1),
        radius,
    )?;
    let start_parameter = circle_2d.project(brepkit_math::vec::Point2::new(start_uv.0, start_uv.1));
    let end_parameter = circle_2d.project(brepkit_math::vec::Point2::new(end_uv.0, end_uv.1));
    let parameter_end = if circle_normal.dot(*normal) >= 0.0 {
        start_parameter + (end_parameter - start_parameter).rem_euclid(std::f64::consts::TAU)
    } else {
        start_parameter - (start_parameter - end_parameter).rem_euclid(std::f64::consts::TAU)
    };
    Ok(Some((
        EdgeCurve::Circle(circle),
        brepkit_topology::pcurve::PCurve::new(
            brepkit_math::curves2d::Curve2D::Circle(circle_2d),
            start_parameter,
            parameter_end,
        ),
    )))
}

fn mapped_contact_forward(
    topo: &Topology,
    face_id: FaceId,
    contact_edge: EdgeId,
    source_run: &[OrientedEdge],
) -> Result<bool, BlendError> {
    let source_first = source_run
        .first()
        .ok_or(BlendError::TrimmingFailure { face: face_id })?;
    let source_last = source_run
        .last()
        .ok_or(BlendError::TrimmingFailure { face: face_id })?;
    let first_edge = topo.edge(source_first.edge())?;
    let last_edge = topo.edge(source_last.edge())?;
    let source_start = topo
        .vertex(source_first.oriented_start(first_edge))?
        .point();
    let source_end = topo.vertex(source_last.oriented_end(last_edge))?.point();
    let contact = topo.edge(contact_edge)?;
    let contact_start = topo.vertex(contact.start())?.point();
    let contact_end = topo.vertex(contact.end())?.point();
    if contact.start() != contact.end() {
        let forward_distance =
            (contact_start - source_start).length() + (contact_end - source_end).length();
        let reverse_distance =
            (contact_end - source_start).length() + (contact_start - source_end).length();
        return Ok(forward_distance <= reverse_distance);
    }

    let stored_start = topo.vertex(first_edge.start())?.point();
    let stored_end = topo.vertex(first_edge.end())?.point();
    let (source_t0, source_t1) = first_edge
        .curve()
        .domain_with_endpoints(stored_start, stored_end);
    let source_parameter = if source_first.is_forward() {
        source_t0
    } else {
        source_t1
    };
    let mut source_tangent =
        first_edge
            .curve()
            .tangent_with_endpoints(source_parameter, stored_start, stored_end);
    if !source_first.is_forward() {
        source_tangent = -source_tangent;
    }
    let (contact_t0, _) = contact
        .curve()
        .domain_with_endpoints(contact_start, contact_end);
    let contact_tangent =
        contact
            .curve()
            .tangent_with_endpoints(contact_t0, contact_start, contact_end);
    Ok(contact_tangent.dot(source_tangent) >= 0.0)
}

#[allow(clippy::too_many_lines)]
fn rebuild_mapped_parametric_face(
    topo: &mut Topology,
    face_id: FaceId,
    restrictions: &[ParametricRestriction],
) -> Result<Option<BatchTrimResult>, BlendError> {
    if restrictions
        .iter()
        .any(|restriction| restriction.source_edges.is_empty())
    {
        return Ok(None);
    }

    let face = topo.face(face_id)?.clone();
    let surface = face.surface().clone();
    let wire_ids: Vec<WireId> = std::iter::once(face.outer_wire())
        .chain(face.inner_wires().iter().copied())
        .collect();
    let mut restriction_for_edge = std::collections::HashMap::<EdgeId, usize>::new();
    for (restriction_index, restriction) in restrictions.iter().enumerate() {
        for &source_edge in &restriction.source_edges {
            if let Some(previous) = restriction_for_edge.insert(source_edge, restriction_index)
                && previous != restriction_index
            {
                return Err(BlendError::TrimmingFailure { face: face_id });
            }
        }
    }

    let mut face_edges = std::collections::HashSet::new();
    let mut known_vertices = Vec::<(Point3, VertexId)>::new();
    let mut known_vertex_ids = std::collections::HashSet::new();
    for &wire_id in &wire_ids {
        let wire = topo.wire(wire_id)?;
        if !wire.is_closed() {
            return Err(BlendError::TrimmingFailure { face: face_id });
        }
        for oriented in wire.edges() {
            face_edges.insert(oriented.edge());
            let edge = topo.edge(oriented.edge())?;
            for vertex in [edge.start(), edge.end()] {
                if known_vertex_ids.insert(vertex) {
                    known_vertices.push((topo.vertex(vertex)?.point(), vertex));
                }
            }
        }
    }
    let source_vertices = known_vertices.clone();
    if restriction_for_edge
        .keys()
        .any(|edge| !face_edges.contains(edge))
    {
        return Err(BlendError::TrimmingFailure { face: face_id });
    }

    let mut new_vertices = Vec::new();
    let mut contact_edges = Vec::with_capacity(restrictions.len());
    for restriction in restrictions {
        if restriction.contact_3d.len() < 2
            || restriction
                .contact_3d
                .iter()
                .copied()
                .any(|point| mapped_surface_uv(&surface, point, face_id).is_err())
        {
            return Err(BlendError::TrimmingFailure { face: face_id });
        }
        let start = restriction.contact_3d[0];
        let end = restriction.contact_3d[restriction.contact_3d.len() - 1];
        let pcurve_closed = restriction.pcurve.as_ref().is_some_and(|pcurve| {
            let a = pcurve.evaluate(pcurve.t_start());
            let b = pcurve.evaluate(pcurve.t_end());
            (a.0[0] - b.0[0]).hypot(a.0[1] - b.0[1]) <= PARAM_TOL
        });
        let closed = (start - end).length() <= VERTEX_TOL || pcurve_closed;
        if closed && (start - end).length() > 1e-4 {
            return Err(BlendError::TrimmingFailure { face: face_id });
        }
        let start_vertex =
            mapped_vertex_for_point(topo, &mut known_vertices, &mut new_vertices, start);
        let end_vertex = if closed {
            start_vertex
        } else {
            mapped_vertex_for_point(topo, &mut known_vertices, &mut new_vertices, end)
        };
        let curve = restriction.curve.clone().unwrap_or(EdgeCurve::Line);
        contact_edges.push(topo.add_edge(Edge::new(start_vertex, end_vertex, curve)));
    }

    let mut emitted = vec![0_usize; restrictions.len()];
    let mut retained_source_edges = std::collections::HashSet::new();
    // Periodic supports wrap onto themselves: the seam is one geometric
    // segment traversed twice (forward and reverse) in the face wire, exactly
    // as the source solid stores it. Distinct fresh edges for the two
    // traversals would leave each with a single face use and trip the
    // postassembly incidence audit. Reuse the first seam connector for its
    let mut seam_edges: std::collections::HashMap<(u64, u64, u64, u64, u64, u64), EdgeId> =
        std::collections::HashMap::new();
    let period = surface_u_period(&surface);
    let mut result_wires = Vec::with_capacity(wire_ids.len());
    let mut new_edges = contact_edges.clone();
    let mut connector_pcurves = Vec::new();
    for &wire_id in &wire_ids {
        let source_wire = topo.wire(wire_id)?.clone();
        let source_edges = source_wire.edges();
        let owners: Vec<Option<usize>> = source_edges
            .iter()
            .map(|oriented| restriction_for_edge.get(&oriented.edge()).copied())
            .collect();
        if owners.iter().all(Option::is_none) {
            retained_source_edges.extend(source_edges.iter().map(OrientedEdge::edge));
            result_wires.push(wire_id);
            continue;
        }

        let transition = (0..owners.len())
            .find(|&index| owners[index] != owners[(index + owners.len() - 1) % owners.len()]);
        let mut pieces = Vec::<(OrientedEdge, bool)>::new();
        if let Some(start_index) = transition {
            let mut consumed = 0;
            while consumed < source_edges.len() {
                let index = (start_index + consumed) % source_edges.len();
                if let Some(restriction_index) = owners[index] {
                    let mut source_run = Vec::new();
                    while consumed < source_edges.len() {
                        let run_index = (start_index + consumed) % source_edges.len();
                        if owners[run_index] != Some(restriction_index) {
                            break;
                        }
                        source_run.push(source_edges[run_index]);
                        consumed += 1;
                    }
                    emitted[restriction_index] += 1;
                    pieces.push((
                        OrientedEdge::new(
                            contact_edges[restriction_index],
                            mapped_contact_forward(
                                topo,
                                face_id,
                                contact_edges[restriction_index],
                                &source_run,
                            )?,
                        ),
                        true,
                    ));
                } else {
                    pieces.push((source_edges[index], false));
                    retained_source_edges.insert(source_edges[index].edge());
                    consumed += 1;
                }
            }
        } else {
            let restriction_index =
                owners[0].ok_or(BlendError::TrimmingFailure { face: face_id })?;
            emitted[restriction_index] += 1;
            pieces.push((
                OrientedEdge::new(
                    contact_edges[restriction_index],
                    mapped_contact_forward(
                        topo,
                        face_id,
                        contact_edges[restriction_index],
                        source_edges,
                    )?,
                ),
                true,
            ));
        }

        let mut reconstructed = Vec::with_capacity(pieces.len() * 2);
        for index in 0..pieces.len() {
            let current = pieces[index];
            let next = pieces[(index + 1) % pieces.len()];
            reconstructed.push(current.0);
            let current_edge = topo.edge(current.0.edge())?;
            let next_edge = topo.edge(next.0.edge())?;
            let current_end = current.0.oriented_end(current_edge);
            let next_start = next.0.oriented_start(next_edge);
            if current_end == next_start {
                continue;
            }
            let start = topo.vertex(current_end)?.point();
            let end = topo.vertex(next_start)?.point();
            if (start - end).length() <= VERTEX_TOL {
                return Err(BlendError::TrimmingFailure { face: face_id });
            }
            let (curve, pcurve) = if current.1 && next.1 {
                if let Some(geometry) =
                    mapped_planar_corner_geometry(&surface, &source_vertices, start, end, face_id)?
                {
                    geometry
                } else {
                    mapped_connector_geometry(&surface, start, end, face_id)?
                }
            } else {
                mapped_connector_geometry(&surface, start, end, face_id)?
            };
            if let Some(_period_value) = period {
                // Periodic supports wrap onto themselves: the seam is one
                // geometric segment traversed twice (forward and reverse) in
                // the face wire, exactly as the source solid stores it.
                // Distinct fresh edges for the two traversals would each get
                // a single face use and trip the incidence audit.
                let key = {
                    let sw = (start.x(), start.y(), start.z()) > (end.x(), end.y(), end.z());
                    let (a, b) = if sw { (end, start) } else { (start, end) };
                    (
                        a.x().to_bits(),
                        a.y().to_bits(),
                        a.z().to_bits(),
                        b.x().to_bits(),
                        b.y().to_bits(),
                        b.z().to_bits(),
                    )
                };
                if let Some(&existing) = seam_edges.get(&key) {
                    // Reverse twin of an already-emitted seam segment: reuse
                    // the same edge entity so both wire traversals share one
                    // use pair, matching the source solid's representation.
                    let existing_edge = topo.edge(existing)?;
                    let forward = existing_edge.start() == current_end;
                    reconstructed.push(OrientedEdge::new(existing, forward));
                    continue;
                }
                let connector = topo.add_edge(Edge::new(current_end, next_start, curve));
                seam_edges.insert(key, connector);
                reconstructed.push(OrientedEdge::new(connector, true));
                connector_pcurves.push((connector, pcurve));
                new_edges.push(connector);
                continue;
            }
            let connector = topo.add_edge(Edge::new(current_end, next_start, curve));
            reconstructed.push(OrientedEdge::new(connector, true));
            connector_pcurves.push((connector, pcurve));
            new_edges.push(connector);
        }
        result_wires.push(topo.add_wire(Wire::new(reconstructed, true)?));
    }
    if emitted.iter().any(|&count| count != 1) {
        return Err(BlendError::TrimmingFailure { face: face_id });
    }

    let mut result_face = Face::new(result_wires[0], result_wires[1..].to_vec(), surface.clone());
    result_face.set_reversed(face.is_reversed());
    let trimmed_face = topo.add_face(result_face);
    for source_edge in retained_source_edges {
        if let Some(pcurve) = topo.pcurves().get(source_edge, face_id).cloned() {
            topo.pcurves_mut().set(source_edge, trimmed_face, pcurve);
        }
    }
    for (restriction_index, &contact_edge) in contact_edges.iter().enumerate() {
        let pcurve = if let Some(pcurve) = restrictions[restriction_index].pcurve.clone() {
            pcurve
        } else {
            let contact = topo.edge(contact_edge)?;
            if contact.start() == contact.end() {
                return Err(BlendError::TrimmingFailure { face: face_id });
            }
            let start = topo.vertex(contact.start())?.point();
            let end = topo.vertex(contact.end())?.point();
            mapped_connector_geometry(&surface, start, end, face_id)?.1
        };
        topo.pcurves_mut().set(contact_edge, trimmed_face, pcurve);
    }
    for (connector, pcurve) in connector_pcurves {
        topo.pcurves_mut().set(connector, trimmed_face, pcurve);
    }

    Ok(Some(BatchTrimResult {
        trimmed_face,
        contact_edges,
        new_edges,
        new_vertices,
        incident_replacements: Vec::new(),
    }))
}

fn trim_parametric_face_batch_in_place(
    topo: &mut Topology,
    face_id: FaceId,
    restrictions: &[ParametricRestriction],
) -> Result<BatchTrimResult, BlendError> {
    if restrictions.is_empty() {
        return Err(BlendError::TrimmingFailure { face: face_id });
    }
    if let Some(result) = rebuild_mapped_parametric_face(topo, face_id, restrictions)? {
        return Ok(result);
    }
    let source_face_ids: Vec<FaceId> = topo.faces().iter().map(|(id, _)| id).collect();
    let face = topo.face(face_id)?.clone();
    let surface = face.surface().clone();
    let reversed = face.is_reversed();
    let (plane_normal, plane_d) = match &surface {
        FaceSurface::Plane { normal, d } => (*normal, *d),
        _ => (brepkit_math::vec::Vec3::new(0.0, 0.0, 1.0), 0.0),
    };
    let arbitrary = if plane_normal.x().abs() < 0.9 {
        brepkit_math::vec::Vec3::new(1.0, 0.0, 0.0)
    } else {
        brepkit_math::vec::Vec3::new(0.0, 1.0, 0.0)
    };
    let (u, v, origin) = if surface.is_planar() {
        let u = plane_normal
            .cross(arbitrary)
            .normalize()
            .map_err(|_| BlendError::TrimmingFailure { face: face_id })?;
        (u, plane_normal.cross(u), plane_normal * plane_d)
    } else {
        (
            brepkit_math::vec::Vec3::new(1.0, 0.0, 0.0),
            brepkit_math::vec::Vec3::new(0.0, 1.0, 0.0),
            brepkit_math::vec::Vec3::new(0.0, 0.0, 0.0),
        )
    };
    let uv = |p: Point3| {
        if surface.is_planar() {
            let q = brepkit_math::vec::Vec3::new(
                p.x() - origin.x(),
                p.y() - origin.y(),
                p.z() - origin.z(),
            );
            (u.dot(q), v.dot(q))
        } else {
            surface
                .project_point(p)
                .map(|q| (q.0, q.1))
                .unwrap_or((f64::NAN, f64::NAN))
        }
    };
    let projection_valid = |p: Point3| {
        if surface.is_planar() {
            let q = brepkit_math::vec::Vec3::new(p.x(), p.y(), p.z());
            (plane_normal.dot(q) - plane_d).abs() <= 1e-5
        } else {
            let q = uv(p);
            q.0.is_finite()
                && q.1.is_finite()
                && surface
                    .evaluate(q.0, q.1)
                    .is_some_and(|evaluated| (evaluated - p).length() <= 1e-5)
        }
    };

    let wire_ids: Vec<WireId> = std::iter::once(face.outer_wire())
        .chain(face.inner_wires().iter().copied())
        .collect();
    let mut nodes: Vec<((f64, f64), Point3, VertexId)> = Vec::new();
    let mut node_for_vertex = std::collections::HashMap::<VertexId, usize>::new();
    let mut original: Vec<BatchWireEdges> = Vec::new();
    for &wid in &wire_ids {
        let oriented_edges = topo.wire(wid)?.edges().to_vec();
        let mut edges = Vec::new();
        for oe in oriented_edges {
            let edge = topo.edge(oe.edge())?;
            let sv = oe.oriented_start(edge);
            let ev = oe.oriented_end(edge);
            let sp = topo.vertex(sv)?.point();
            let ep = topo.vertex(ev)?.point();
            if !projection_valid(sp) || !projection_valid(ep) {
                return Err(BlendError::TrimmingFailure { face: face_id });
            }
            let a = match node_for_vertex.get(&sv).copied() {
                Some(n) => n,
                None => batch_node_for_point(
                    &mut *topo,
                    &mut nodes,
                    &mut node_for_vertex,
                    uv(sp),
                    sp,
                    Some(sv),
                ),
            };
            let b = match node_for_vertex.get(&ev).copied() {
                Some(n) => n,
                None => batch_node_for_point(
                    &mut *topo,
                    &mut nodes,
                    &mut node_for_vertex,
                    uv(ep),
                    ep,
                    Some(ev),
                ),
            };
            edges.push((oe, vec![(0.0, a), (1.0, b)]));
        }
        original.push(edges);
    }
    let source_closed_wires = original
        .iter()
        .enumerate()
        .map(
            |(index, edges)| -> Result<Option<(WireId, bool, Point3)>, BlendError> {
                if edges.len() != 1 {
                    return Ok(None);
                }
                let oriented = edges[0].0;
                let edge = topo.edge(oriented.edge())?;
                if edge.start() != edge.end() {
                    return Ok(None);
                }
                Ok(Some((
                    wire_ids[index],
                    oriented.is_forward(),
                    topo.vertex(edge.start())?.point(),
                )))
            },
        )
        .collect::<Result<Vec<_>, _>>()?;

    let mut restriction_lines = Vec::with_capacity(restrictions.len());
    let mut restriction_paths: Vec<Vec<((f64, f64), Point3)>> =
        Vec::with_capacity(restrictions.len());
    let mut closed_restrictions = Vec::with_capacity(restrictions.len());
    for restriction in restrictions {
        if restriction.contact_3d.len() < 2
            || restriction
                .contact_3d
                .iter()
                .copied()
                .any(|p| !projection_valid(p))
        {
            return Err(BlendError::TrimmingFailure { face: face_id });
        }
        let mut path = Vec::with_capacity(BATCH_CURVE_SAMPLES + 1);
        if let Some(pcurve) = restriction.pcurve.as_ref() {
            let count = BATCH_CURVE_SAMPLES;
            let t0 = pcurve.t_start();
            let dt = pcurve.t_end() - t0;
            let supplied_count = restriction.contact_3d.len() - 1;
            if !surface.is_planar() {
                for (index, point) in restriction.contact_3d.iter().enumerate() {
                    let fraction = index as f64 / supplied_count as f64;
                    let projected = pcurve.evaluate(t0 + dt * fraction);
                    let evaluated = surface
                        .evaluate(projected.0[0], projected.0[1])
                        .ok_or(BlendError::TrimmingFailure { face: face_id })?;
                    if (evaluated - *point).length() > 1e-4 {
                        return Err(BlendError::TrimmingFailure { face: face_id });
                    }
                }
            }
            for index in 0..=count {
                let fraction = index as f64 / count as f64;
                let mut point = if let Some(curve) = restriction.curve.as_ref() {
                    let (curve_start, curve_end) = curve.domain_with_endpoints(
                        restriction.contact_3d[0],
                        restriction.contact_3d[restriction.contact_3d.len() - 1],
                    );
                    let parameter = curve_start + (curve_end - curve_start) * fraction;
                    curve.evaluate_with_endpoints(
                        parameter,
                        restriction.contact_3d[0],
                        restriction.contact_3d[restriction.contact_3d.len() - 1],
                    )
                } else if index == 0 {
                    restriction.contact_3d[0]
                } else if index == count {
                    restriction.contact_3d[restriction.contact_3d.len() - 1]
                } else {
                    let a = restriction.contact_3d.len() - 1;
                    let position = fraction * a as f64;
                    let lower = position.floor() as usize;
                    let alpha = position - lower as f64;
                    restriction.contact_3d[lower]
                        + (restriction.contact_3d[lower + 1] - restriction.contact_3d[lower])
                            * alpha
                };
                let projected = pcurve.evaluate(t0 + dt * fraction);
                let q = (projected.0[0], projected.0[1]);
                if !surface.is_planar() {
                    let evaluated = surface
                        .evaluate(q.0, q.1)
                        .ok_or(BlendError::TrimmingFailure { face: face_id })?;
                    point = evaluated;
                }
                path.push((q, point));
            }
        } else {
            path.extend(
                restriction
                    .contact_3d
                    .iter()
                    .copied()
                    .map(|point| (uv(point), point)),
            );
        }
        let (a, b) = (path[0].0, path[path.len() - 1].0);
        let closed = (a.0 - b.0).hypot(a.1 - b.1) < PARAM_TOL;
        let na = batch_node_for_point(
            &mut *topo,
            &mut nodes,
            &mut node_for_vertex,
            a,
            path[0].1,
            None,
        );
        let nb = if closed {
            na
        } else {
            batch_node_for_point(
                &mut *topo,
                &mut nodes,
                &mut node_for_vertex,
                b,
                path[path.len() - 1].1,
                None,
            )
        };
        closed_restrictions.push(closed);
        restriction_lines.push((a, b, na, nb));
        restriction_paths.push(path);
    }
    // Unwrap the angular U coordinate across periodic seams. This preserves
    // one logical seam identity while keeping the arrangement in a continuous
    // UV chart.
    if let Some(period) = surface_u_period(&surface) {
        let align = |u: f64, reference: f64| u - period * ((u - reference) / period).round();
        let mut previous = None;
        for edges in &mut original {
            for (_, splits) in edges {
                for (index, node) in [splits[0].1, splits[1].1].into_iter().enumerate() {
                    let reference = previous.unwrap_or(nodes[node].0.0);
                    let aligned = align(nodes[node].0.0, reference);
                    nodes[node].0.0 = aligned;
                    if index == 1 {
                        previous = Some(aligned);
                    }
                }
            }
        }
        let reference = previous.unwrap_or(0.0);
        for path in &mut restriction_paths {
            let mut path_reference = reference;
            for (q, _) in path {
                q.0 = align(q.0, path_reference);
                path_reference = q.0;
            }
        }
        for (index, (a, b, na, nb)) in restriction_lines.iter_mut().enumerate() {
            let path = &restriction_paths[index];
            *a = path[0].0;
            *b = path[path.len() - 1].0;
            nodes[*na].0 = *a;
            nodes[*nb].0 = *b;
            for (node_id, target) in [(*na, path[0]), (*nb, path[path.len() - 1])] {
                if let Some(existing) = nodes.iter().position(|(q, point, _)| {
                    (q.0 - target.0.0).hypot(q.1 - target.0.1) < 1e-7
                        && (*point - target.1).length() < 1e-7
                }) {
                    if node_id == *na {
                        *na = existing;
                    } else {
                        *nb = existing;
                    }
                }
            }
        }
    }
    let arrangement_anchor = if surface_u_period(&surface).is_some() {
        original
            .first()
            .and_then(|edges| edges.first())
            .map(|(_, splits)| nodes[splits[0].1].0.0)
    } else {
        None
    };

    // Collect every boundary split before allocating any result wire. Curved
    // boundary edges are sampled in their native parameter domain instead of
    // intersecting their endpoint chord. The minted split point is evaluated
    // on the carrier curve, so the resulting Circle/NURBS sub-edge retains
    // the exact span inferred by `domain_with_endpoints`.
    for edges in &mut original {
        for (oe, splits) in edges {
            let edge = topo.edge(oe.edge())?.clone();
            let stored_start = topo.vertex(edge.start())?.point();
            let stored_end = topo.vertex(edge.end())?.point();
            let (d0, d1) = edge.curve().domain_with_endpoints(stored_start, stored_end);
            let mut previous: Option<((f64, f64), f64)> = None;
            let period = surface_u_period(&surface);
            for sample_index in 0..BATCH_CURVE_SAMPLES {
                let f0 = sample_index as f64 / BATCH_CURVE_SAMPLES as f64;
                let f1 = (sample_index + 1) as f64 / BATCH_CURVE_SAMPLES as f64;
                let native_f0 = if oe.is_forward() { f0 } else { 1.0 - f0 };
                let native_f1 = if oe.is_forward() { f1 } else { 1.0 - f1 };
                let t0 = d0 + (d1 - d0) * native_f0;
                let t1 = d0 + (d1 - d0) * native_f1;
                let p0 = edge
                    .curve()
                    .evaluate_with_endpoints(t0, stored_start, stored_end);
                let p1 = edge
                    .curve()
                    .evaluate_with_endpoints(t1, stored_start, stored_end);
                if !projection_valid(p0) || !projection_valid(p1) {
                    return Err(BlendError::TrimmingFailure { face: face_id });
                }
                let mut q0 = uv(p0);
                let mut q1 = uv(p1);
                if let Some(period) = period {
                    let align = |value: f64, reference: f64| {
                        value - period * ((value - reference) / period).round()
                    };
                    q0.0 = align(q0.0, nodes[splits[0].1].0.0);
                    q1.0 = align(q1.0, q0.0);
                }
                for path in &restriction_paths {
                    for pair in path.windows(2) {
                        let (line_a, line_b) = (pair[0].0, pair[1].0);
                        let Some(alpha) = line_segment_intersect_2d(q0, q1, line_a, line_b) else {
                            continue;
                        };
                        let sample_f = f0 + (f1 - f0) * alpha;
                        let native_f = if oe.is_forward() {
                            sample_f
                        } else {
                            1.0 - sample_f
                        };
                        let curve_t = d0 + (d1 - d0) * native_f;
                        let point =
                            edge.curve()
                                .evaluate_with_endpoints(curve_t, stored_start, stored_end);
                        let oriented_t = sample_f;
                        let mut point_uv = uv(point);
                        if let Some(period) = period {
                            point_uv.0 -= period * ((point_uv.0 - q0.0) / period).round();
                        }
                        if previous.is_some_and(|(old_uv, old_t)| {
                            (old_uv.0 - point_uv.0).hypot(old_uv.1 - point_uv.1) < 1e-8
                                || (old_t - oriented_t).abs() < 1e-8
                        }) {
                            continue;
                        }
                        let n = batch_node_for_point(
                            &mut *topo,
                            &mut nodes,
                            &mut node_for_vertex,
                            point_uv,
                            point,
                            None,
                        );
                        if !splits
                            .iter()
                            .any(|(old_t, _)| (*old_t - oriented_t).abs() < 1e-8)
                        {
                            splits.push((oriented_t, n));
                            previous = Some((point_uv, oriented_t));
                        }
                    }
                }
            }
            splits.sort_by(|x, y| x.0.partial_cmp(&y.0).unwrap_or(std::cmp::Ordering::Equal));
        }
    }

    let mut segments = Vec::<BatchSegment>::new();
    let mut split_edges = Vec::<(EdgeId, Vec<EdgeId>)>::new();
    let mut split_spans = Vec::<(EdgeId, Vec<(EdgeId, f64, f64)>)>::new();
    let mut new_edges = Vec::new();
    for (wire_index, edges) in original.iter().enumerate() {
        for (oe, splits) in edges {
            let old = topo.edge(oe.edge())?.clone();
            let stored_start = topo.vertex(old.start())?.point();
            let stored_end = topo.vertex(old.end())?.point();
            let (d0, d1) = old.curve().domain_with_endpoints(stored_start, stored_end);
            let mut pieces = Vec::new();
            let mut piece_spans = Vec::new();
            for pair in splits.windows(2) {
                let (t0, n0) = pair[0];
                let (t1, n1) = pair[1];
                if source_closed_wires[wire_index].is_some() && n0 == n1 {
                    continue;
                }
                if (t1 - t0).abs() < PARAM_TOL {
                    continue;
                }
                let (start, end, forward) = if oe.is_forward() {
                    (n0, n1, true)
                } else {
                    (n1, n0, false)
                };
                let (curve_t0, curve_t1) = if oe.is_forward() {
                    (d0 + (d1 - d0) * t0, d0 + (d1 - d0) * t1)
                } else {
                    (d0 + (d1 - d0) * (1.0 - t0), d0 + (d1 - d0) * (1.0 - t1))
                };
                let piece_curve = edge_curve_for_span(old.curve(), curve_t0, curve_t1)?;
                let edge_id = topo.add_edge(Edge::new(nodes[start].2, nodes[end].2, piece_curve));
                pieces.push(edge_id);
                let (stored_t0, stored_t1) = if oe.is_forward() { (t0, t1) } else { (t1, t0) };
                piece_spans.push((edge_id, stored_t0, stored_t1));
                new_edges.push(edge_id);
                segments.push(BatchSegment {
                    a: start,
                    b: end,
                    edge: edge_id,
                });
                // Preserve traversal orientation in the segment graph.
                if !forward {
                    if let Some(last) = segments.last_mut() {
                        std::mem::swap(&mut last.a, &mut last.b);
                    } else {
                        return Err(BlendError::TrimmingFailure { face: face_id });
                    }
                }
            }
            if pieces.len() > 1 || splits.len() > 2 {
                split_edges.push((oe.edge(), pieces));
                split_spans.push((oe.edge(), piece_spans));
            }
        }
    }

    let mut contact_edges = Vec::with_capacity(restrictions.len());
    for (index, restriction) in restrictions.iter().enumerate() {
        let (_, _, start, end) = restriction_lines[index];
        let curve = restriction.curve.clone().unwrap_or(EdgeCurve::Line);
        let edge = topo.add_edge(Edge::new(nodes[start].2, nodes[end].2, curve));
        contact_edges.push(edge);
        new_edges.push(edge);
        if !closed_restrictions[index] {
            segments.push(BatchSegment {
                a: start,
                b: end,
                edge,
            });
        }
    }

    // Half-edge walk: predecessor of the reverse edge keeps the cell on left.
    let mut outgoing = vec![Vec::<usize>::new(); nodes.len()];
    let mut half = Vec::<(usize, usize, usize)>::new();
    for (segment_index, segment) in segments.iter().enumerate() {
        let i = half.len();
        half.push((segment.a, segment.b, segment_index));
        half.push((segment.b, segment.a, segment_index));
        outgoing[segment.a].push(i);
        outgoing[segment.b].push(i + 1);
    }
    for (node, list) in outgoing.iter_mut().enumerate() {
        list.sort_by(|a, b| {
            let aa = (nodes[half[*a].1].0.1 - nodes[node].0.1)
                .atan2(nodes[half[*a].1].0.0 - nodes[node].0.0);
            let bb = (nodes[half[*b].1].0.1 - nodes[node].0.1)
                .atan2(nodes[half[*b].1].0.0 - nodes[node].0.0);
            aa.partial_cmp(&bb).unwrap_or(std::cmp::Ordering::Equal)
        });
    }
    let mut next = vec![usize::MAX; half.len()];
    for (i, &(_, to, _)) in half.iter().enumerate() {
        let reverse = i ^ 1;
        let list = &outgoing[to];
        let pos = list
            .iter()
            .position(|&j| j == reverse)
            .ok_or(BlendError::TrimmingFailure { face: face_id })?;
        next[i] = list[(pos + list.len() - 1) % list.len()];
    }
    let mut seen = vec![false; half.len()];
    let mut cycles = Vec::<Vec<usize>>::new();
    for start in 0..half.len() {
        if seen[start] {
            continue;
        }
        let mut cycle = Vec::new();
        let mut current = start;
        while !seen[current] {
            seen[current] = true;
            cycle.push(current);
            current = next[current];
            if cycle.len() > half.len() {
                return Err(BlendError::TrimmingFailure { face: face_id });
            }
        }
        if current == start && cycle.len() >= 3 {
            cycles.push(cycle);
        }
    }

    let loop_uv = original
        .iter()
        .map(|edges| -> Result<Vec<(f64, f64)>, BlendError> {
            let mut polygon = Vec::new();
            let period = surface_u_period(&surface);
            let mut previous_u: Option<f64> = arrangement_anchor;
            for (oriented, _) in edges {
                let edge = topo.edge(oriented.edge())?;
                let stored_start = topo.vertex(edge.start())?.point();
                let stored_end = topo.vertex(edge.end())?.point();
                let (domain_start, domain_end) =
                    edge.curve().domain_with_endpoints(stored_start, stored_end);
                for sample_index in 0..BATCH_CURVE_SAMPLES {
                    let fraction = sample_index as f64 / BATCH_CURVE_SAMPLES as f64;
                    let native_fraction = if oriented.is_forward() {
                        fraction
                    } else {
                        1.0 - fraction
                    };
                    let parameter = domain_start + (domain_end - domain_start) * native_fraction;
                    let point =
                        edge.curve()
                            .evaluate_with_endpoints(parameter, stored_start, stored_end);
                    let mut projected = uv(point);
                    if !projected.0.is_finite() || !projected.1.is_finite() {
                        return Err(BlendError::TrimmingFailure { face: face_id });
                    }
                    if let (Some(period), Some(reference)) = (period, previous_u) {
                        projected.0 -= period * ((projected.0 - reference) / period).round();
                    }
                    previous_u = Some(projected.0);
                    polygon.push(projected);
                }
            }
            Ok(polygon)
        })
        .collect::<Result<Vec<_>, _>>()?;
    let point_in = |p: (f64, f64), polygon: &[(f64, f64)]| {
        let mut inside = false;
        for i in 0..polygon.len() {
            let a = polygon[i];
            let b = polygon[(i + 1) % polygon.len()];
            if (a.1 > p.1) != (b.1 > p.1) && p.0 < (b.0 - a.0) * (p.1 - a.1) / (b.1 - a.1) + a.0 {
                inside = !inside;
            }
        }
        inside
    };
    let point_in_restriction = |p: (f64, f64), path: &[((f64, f64), Point3)]| {
        let mut inside = false;
        for i in 0..path.len() {
            let a = path[i].0;
            let b = path[(i + 1) % path.len()].0;
            if (a.1 > p.1) != (b.1 > p.1) && p.0 < (b.0 - a.0) * (p.1 - a.1) / (b.1 - a.1) + a.0 {
                inside = !inside;
            }
        }
        inside
    };
    let mut kept = Vec::<(Vec<usize>, f64)>::new();
    for cycle in cycles {
        let a = nodes[half[cycle[0]].0].0;
        let b = nodes[half[cycle[0]].1].0;
        let dx = b.0 - a.0;
        let dy = b.1 - a.1;
        let scale = 1e-7_f64.max(dx.hypot(dy) * 1e-7);
        let sample = (
            (a.0 + b.0) * 0.5 - dy / dx.hypot(dy) * scale,
            (a.1 + b.1) * 0.5 + dx / dx.hypot(dy) * scale,
        );
        let material = point_in(sample, &loop_uv[0])
            && loop_uv[1..].iter().all(|hole| !point_in(sample, hole));
        if !material {
            continue;
        }
        let valid = restrictions.iter().enumerate().all(|(i, restriction)| {
            if closed_restrictions[i] {
                let path = &restriction_paths[i];
                let sample_inside = point_in_restriction(sample, path);
                return match restriction.keep {
                    TrimKeep::AwayFrom(point) => {
                        let point_inside = point_in_restriction(uv(point), path);
                        sample_inside != point_inside
                    }
                    TrimKeep::Side(side) => {
                        let signed_area = path
                            .windows(2)
                            .map(|pair| pair[0].0.0 * pair[1].0.1 - pair[0].0.1 * pair[1].0.0)
                            .sum::<f64>()
                            * 0.5;
                        let interior_is_left = signed_area > 0.0;
                        let keep_left = matches!(side, TrimSide::Left);
                        sample_inside == (interior_is_left == keep_left)
                    }
                };
            }
            let contains_contact = cycle.iter().any(|&half_edge| {
                segments.iter().any(|segment| {
                    segment.edge == contact_edges[i]
                        && ((segment.a == half[half_edge].0 && segment.b == half[half_edge].1)
                            || (segment.a == half[half_edge].1 && segment.b == half[half_edge].0))
                })
            });
            if !contains_contact {
                return true;
            }
            let (line_start, line_end, _, _) = restriction_lines[i];
            let side = (line_end.0 - line_start.0) * (sample.1 - line_start.1)
                - (line_end.1 - line_start.1) * (sample.0 - line_start.0);
            if side.abs() < 1e-8 {
                return true;
            }
            match restriction.keep {
                TrimKeep::AwayFrom(point) => {
                    let projected = uv(point);
                    let point_side = (line_end.0 - line_start.0) * (projected.1 - line_start.1)
                        - (line_end.1 - line_start.1) * (projected.0 - line_start.0);
                    side * point_side <= 0.0
                }
                TrimKeep::Side(TrimSide::Left) => {
                    if reversed {
                        side < 0.0
                    } else {
                        side > 0.0
                    }
                }
                TrimKeep::Side(TrimSide::Right) => {
                    if reversed {
                        side > 0.0
                    } else {
                        side < 0.0
                    }
                }
            }
        });
        if valid {
            let area = cycle
                .iter()
                .map(|&h| {
                    let p = nodes[half[h].0].0;
                    let q = nodes[half[h].1].0;
                    p.0 * q.1 - p.1 * q.0
                })
                .sum::<f64>()
                * 0.5;
            if area.abs() > 1e-12 {
                kept.push((cycle, area));
            }
        }
    }
    let outer_index = kept
        .iter()
        .enumerate()
        .max_by(|(_, a), (_, b)| {
            a.1.abs()
                .partial_cmp(&b.1.abs())
                .unwrap_or(std::cmp::Ordering::Equal)
        })
        .map(|(index, _)| index);
    let make_wire = |topo: &mut Topology, cycle: &[usize]| -> Result<WireId, BlendError> {
        let edges = cycle
            .iter()
            .map(|&half_edge| {
                let segment = &segments[half[half_edge].2];
                Ok(OrientedEdge::new(segment.edge, half_edge % 2 == 0))
            })
            .collect::<Result<Vec<_>, BlendError>>()?;
        Ok(topo.add_wire(Wire::new(edges, true)?))
    };
    let mut outer_wire = outer_index
        .map(|index| make_wire(topo, &kept[index].0))
        .transpose()?;
    let mut inner_wires = Vec::new();
    for (index, (cycle, _)) in kept.iter().enumerate() {
        if Some(index) != outer_index {
            inner_wires.push(make_wire(topo, cycle)?);
        }
    }

    let mut source_closed_replacements = vec![None; source_closed_wires.len()];
    let mut unmatched_closed_wires = Vec::new();
    for (restriction_index, &closed) in closed_restrictions.iter().enumerate() {
        if !closed {
            continue;
        }
        let mut matched_source = None;
        let mut best_distance = f64::INFINITY;
        if let TrimKeep::AwayFrom(point) = restrictions[restriction_index].keep {
            for (source_index, source) in source_closed_wires.iter().enumerate() {
                let Some((_, _, seam)) = source else {
                    continue;
                };
                if source_closed_replacements[source_index].is_some() {
                    continue;
                }
                let distance = (*seam - point).length();
                if distance < best_distance {
                    best_distance = distance;
                    matched_source = Some(source_index);
                }
            }
            if best_distance > 1e-5 {
                matched_source = None;
            }
        }
        let forward = matched_source
            .and_then(|index| source_closed_wires[index].as_ref().map(|source| source.1))
            .unwrap_or(true);
        let wire = topo.add_wire(Wire::new(
            vec![OrientedEdge::new(contact_edges[restriction_index], forward)],
            true,
        )?);
        if let Some(source_index) = matched_source {
            source_closed_replacements[source_index] = Some(wire);
        } else {
            unmatched_closed_wires.push((restriction_index, wire));
        }
    }

    for (source_index, source) in source_closed_wires.iter().enumerate() {
        let Some((source_wire, _, _)) = source else {
            continue;
        };
        let replacement = source_closed_replacements[source_index];
        let was_split = original[source_index]
            .iter()
            .any(|(_, splits)| splits.len() > 2);
        if source_index == 0 {
            if let Some(wire) = replacement {
                outer_wire = Some(wire);
            } else if !was_split {
                outer_wire = Some(*source_wire);
            }
        } else if let Some(wire) = replacement {
            inner_wires.push(wire);
        } else if !was_split {
            inner_wires.push(*source_wire);
        }
    }

    let mut unmatched_outer = false;
    for (restriction_index, wire) in unmatched_closed_wires {
        let path = &restriction_paths[restriction_index];
        let keeps_inside = match restrictions[restriction_index].keep {
            TrimKeep::AwayFrom(point) => !point_in_restriction(uv(point), path),
            TrimKeep::Side(side) => {
                let signed_area = path
                    .windows(2)
                    .map(|pair| pair[0].0.0 * pair[1].0.1 - pair[0].0.1 * pair[1].0.0)
                    .sum::<f64>()
                    * 0.5;
                (signed_area > 0.0) == matches!(side, TrimSide::Left)
            }
        };
        if keeps_inside {
            if unmatched_outer {
                return Err(BlendError::TrimmingFailure { face: face_id });
            }
            outer_wire = Some(wire);
            unmatched_outer = true;
        } else {
            inner_wires.push(wire);
        }
    }
    let outer_wire = outer_wire.ok_or(BlendError::TrimmingFailure { face: face_id })?;
    let mut result_face = Face::new(outer_wire, inner_wires, surface.clone());
    result_face.set_reversed(reversed);
    let trimmed_face = topo.add_face(result_face);
    for (index, &edge) in contact_edges.iter().enumerate() {
        if surface.is_planar() && restrictions[index].pcurve.is_none() {
            continue;
        }
        let pcurve = if let Some(pcurve) = restrictions[index].pcurve.clone() {
            pcurve
        } else {
            let (a, b, _, _) = restriction_lines[index];
            let uv_curve = brepkit_math::curves2d::NurbsCurve2D::from_line(
                brepkit_math::vec::Point2::new(a.0, a.1),
                brepkit_math::vec::Point2::new(b.0, b.1),
            )
            .map_err(|_| BlendError::TrimmingFailure { face: face_id })?;
            brepkit_topology::pcurve::PCurve::new(
                brepkit_math::curves2d::Curve2D::Nurbs(uv_curve),
                0.0,
                1.0,
            )
        };
        topo.pcurves_mut().set(edge, trimmed_face, pcurve);
    }
    let copy_pcurves =
        |topo: &mut Topology, source_face: FaceId, target_face: FaceId| -> Result<(), BlendError> {
            let source = topo.face(source_face)?.clone();
            let source_wires: Vec<WireId> = std::iter::once(source.outer_wire())
                .chain(source.inner_wires().iter().copied())
                .collect();
            let mut curves = Vec::new();
            for wire_id in source_wires {
                for oriented in topo.wire(wire_id)?.edges() {
                    if let Some(pcurve) = topo.pcurves().get(oriented.edge(), source_face) {
                        curves.push((oriented.edge(), pcurve.clone()));
                    }
                }
            }
            for (old_edge, pcurve) in curves {
                if let Some((_, spans)) = split_spans.iter().find(|(edge, _)| *edge == old_edge) {
                    let start = pcurve.t_start();
                    let delta = pcurve.t_end() - start;
                    for &(new_edge, t0, t1) in spans {
                        topo.pcurves_mut().set(
                            new_edge,
                            target_face,
                            brepkit_topology::pcurve::PCurve::new(
                                pcurve.curve().clone(),
                                start + delta * t0,
                                start + delta * t1,
                            ),
                        );
                    }
                } else {
                    topo.pcurves_mut().set(old_edge, target_face, pcurve);
                }
            }
            Ok(())
        };
    copy_pcurves(topo, face_id, trimmed_face)?;

    // Clone only source neighbours that reference a copied boundary edge.
    let mut incident_replacements = Vec::new();
    for source_face in source_face_ids {
        if source_face == face_id {
            continue;
        }
        let source = topo.face(source_face)?.clone();
        let mut changed = false;
        let mut clone_wire = |topo: &mut Topology, wire_id: WireId| -> Result<WireId, BlendError> {
            let wire = topo.wire(wire_id)?;
            let mut out = Vec::new();
            for oe in wire.edges() {
                if let Some((_, pieces)) = split_edges.iter().find(|(old, _)| *old == oe.edge()) {
                    changed = true;
                    if oe.is_forward() {
                        out.extend(pieces.iter().map(|&e| OrientedEdge::new(e, true)));
                    } else {
                        out.extend(pieces.iter().rev().map(|&e| OrientedEdge::new(e, false)));
                    }
                } else {
                    out.push(*oe);
                }
            }
            Ok(topo.add_wire(Wire::new(out, wire.is_closed())?))
        };
        let outer = clone_wire(topo, source.outer_wire())?;
        let inners = source
            .inner_wires()
            .iter()
            .copied()
            .map(|w| clone_wire(topo, w))
            .collect::<Result<Vec<_>, _>>()?;
        if changed {
            let mut replacement = Face::new(outer, inners, source.surface().clone());
            replacement.set_reversed(source.is_reversed());
            let replacement_id = topo.add_face(replacement);
            copy_pcurves(topo, source_face, replacement_id)?;
            incident_replacements.push((source_face, replacement_id));
        }
    }
    let source_vertex_nodes: std::collections::HashSet<usize> =
        node_for_vertex.values().copied().collect();
    let new_vertices = nodes
        .iter()
        .enumerate()
        .filter_map(|(index, (_, _, vertex))| {
            (!source_vertex_nodes.contains(&index)).then_some(*vertex)
        })
        .collect();
    Ok(BatchTrimResult {
        trimmed_face,
        contact_edges,
        new_edges,
        new_vertices,
        incident_replacements,
    })
}
/// Rebuild a planar or curved support face from all restrictions at once.
///
/// Work is performed on a copy and committed only after the complete
/// arrangement succeeds. This keeps explicit projection/intersection
/// failures non-destructive to the source topology.
pub fn trim_parametric_face_batch(
    topo: &mut Topology,
    face_id: FaceId,
    restrictions: &[ParametricRestriction],
) -> Result<BatchTrimResult, BlendError> {
    let mut working = topo.clone();
    let result = trim_parametric_face_batch_in_place(&mut working, face_id, restrictions)?;
    *topo = working;
    Ok(result)
}

/// Backwards-compatible planar entry point. The implementation is the same
/// transactional parametric arrangement, but this API rejects non-planar
/// surfaces so existing BK4 callers retain their contract.
pub fn trim_planar_face_batch(
    topo: &mut Topology,
    face_id: FaceId,
    restrictions: &[PlanarRestriction],
) -> Result<BatchTrimResult, BlendError> {
    if !topo.face(face_id)?.surface().is_planar() {
        return Err(BlendError::TrimmingFailure { face: face_id });
    }
    trim_parametric_face_batch(topo, face_id, restrictions)
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
    fn batch_rebuilds_one_and_two_restrictions_atomically() {
        let mut topo = Topology::new();
        let (face_id, _, _) = make_square_face(&mut topo);
        let source_wire = topo.face(face_id).unwrap().outer_wire();
        let before = topo.wire(source_wire).unwrap().clone();
        let mut left = PlanarRestriction::new(
            vec![Point3::new(0.25, 0.0, 0.0), Point3::new(0.25, 1.0, 0.0)],
            TrimKeep::Side(TrimSide::Right),
        );
        left.curve = Some(EdgeCurve::Line);
        let mut right = PlanarRestriction::new(
            vec![Point3::new(0.75, 0.0, 0.0), Point3::new(0.75, 1.0, 0.0)],
            TrimKeep::Side(TrimSide::Left),
        );
        right.curve = Some(EdgeCurve::Line);
        let result = trim_planar_face_batch(&mut topo, face_id, &[left, right]).unwrap();
        assert_ne!(result.trimmed_face, face_id);
        let wire_edges = |wire: &Wire| {
            wire.edges()
                .iter()
                .map(|oe| (oe.edge(), oe.is_forward()))
                .collect::<Vec<_>>()
        };
        assert_eq!(
            wire_edges(topo.wire(source_wire).unwrap()),
            wire_edges(&before),
            "source wire was rewritten"
        );
        assert_eq!(
            topo.wire(topo.face(result.trimmed_face).unwrap().outer_wire())
                .unwrap()
                .edges()
                .len(),
            4
        );
    }
    #[test]
    fn curved_projection_failure_is_explicit_and_source_stable() {
        use brepkit_math::surfaces::CylindricalSurface;

        let mut topo = Topology::new();
        let v0 = topo.add_vertex(Vertex::new(Point3::new(1.0, 0.0, 0.0), VERTEX_TOL));
        let v1 = topo.add_vertex(Vertex::new(Point3::new(1.0, 0.0, 1.0), VERTEX_TOL));
        let e0 = topo.add_edge(Edge::new(v0, v1, EdgeCurve::Line));
        let e1 = topo.add_edge(Edge::new(v1, v0, EdgeCurve::Line));
        let wire = topo.add_wire(
            Wire::new(
                vec![OrientedEdge::new(e0, true), OrientedEdge::new(e1, true)],
                true,
            )
            .unwrap(),
        );
        let surface = FaceSurface::Cylinder(
            CylindricalSurface::new(Point3::new(0.0, 0.0, 0.0), Vec3::new(0.0, 0.0, 1.0), 1.0)
                .unwrap(),
        );
        let face = topo.add_face(Face::new(wire, Vec::new(), surface));
        let counts = (topo.num_vertices(), topo.num_edges(), topo.num_faces());
        let restriction = ParametricRestriction::new(
            vec![Point3::new(2.0, 0.0, 0.0), Point3::new(2.0, 0.0, 1.0)],
            TrimKeep::Side(TrimSide::Left),
        );
        assert!(
            matches!(
                trim_parametric_face_batch(&mut topo, face, &[restriction]),
                Err(BlendError::TrimmingFailure { face: failed }) if failed == face
            ),
            "off-surface projection must be an explicit failure"
        );
        assert_eq!(
            counts,
            (topo.num_vertices(), topo.num_edges(), topo.num_faces()),
            "forced curved projection failure must not mutate source topology"
        );
    }

    #[test]
    fn cylinder_uv_batch_rebuilds_shared_restriction() {
        use brepkit_math::surfaces::CylindricalSurface;

        let mut topo = Topology::new();
        let surface =
            CylindricalSurface::new(Point3::new(0.0, 0.0, 0.0), Vec3::new(0.0, 0.0, 1.0), 1.0)
                .unwrap();
        let point = |u: f64, v: f64| surface.evaluate(u, v);
        let p00 = point(0.0, 0.0);
        let p10 = point(std::f64::consts::FRAC_PI_2, 0.0);
        let p11 = point(std::f64::consts::FRAC_PI_2, 1.0);
        let p01 = point(0.0, 1.0);
        let v00 = topo.add_vertex(Vertex::new(p00, VERTEX_TOL));
        let v10 = topo.add_vertex(Vertex::new(p10, VERTEX_TOL));
        let v11 = topo.add_vertex(Vertex::new(p11, VERTEX_TOL));
        let v01 = topo.add_vertex(Vertex::new(p01, VERTEX_TOL));
        let e0 = topo.add_edge(Edge::new(
            v00,
            v10,
            EdgeCurve::Circle(
                brepkit_math::curves::Circle3D::new(
                    Point3::new(0.0, 0.0, 0.0),
                    Vec3::new(0.0, 0.0, 1.0),
                    1.0,
                )
                .unwrap(),
            ),
        ));
        let e1 = topo.add_edge(Edge::new(v10, v11, EdgeCurve::Line));
        let e2 = topo.add_edge(Edge::new(
            v01,
            v11,
            EdgeCurve::Circle(
                brepkit_math::curves::Circle3D::new(
                    Point3::new(0.0, 0.0, 1.0),
                    Vec3::new(0.0, 0.0, 1.0),
                    1.0,
                )
                .unwrap(),
            ),
        ));
        let e3 = topo.add_edge(Edge::new(v00, v01, EdgeCurve::Line));
        let wire = topo.add_wire(
            Wire::new(
                vec![
                    OrientedEdge::new(e0, true),
                    OrientedEdge::new(e1, true),
                    OrientedEdge::new(e2, false),
                    OrientedEdge::new(e3, false),
                ],
                true,
            )
            .unwrap(),
        );
        let face = topo.add_face(Face::new(
            wire,
            Vec::new(),
            FaceSurface::Cylinder(surface.clone()),
        ));
        let mut restriction = ParametricRestriction::new(
            vec![
                point(std::f64::consts::FRAC_PI_4, 0.0),
                point(std::f64::consts::FRAC_PI_4, 1.0),
            ],
            TrimKeep::Side(TrimSide::Right),
        );
        restriction.curve = Some(EdgeCurve::Line);
        let result = trim_parametric_face_batch(&mut topo, face, &[restriction]).unwrap();
        assert_ne!(result.trimmed_face, face);
        assert_eq!(result.contact_edges.len(), 1);
        assert!(
            topo.pcurves()
                .get(result.contact_edges[0], result.trimmed_face)
                .is_some()
        );
    }
    fn add_parametric_quad(
        topo: &mut Topology,
        surface: FaceSurface,
        points: [Point3; 4],
        curves: [EdgeCurve; 4],
    ) -> FaceId {
        let vertices = points.map(|point| topo.add_vertex(Vertex::new(point, VERTEX_TOL)));
        let edges = [
            topo.add_edge(Edge::new(vertices[0], vertices[1], curves[0].clone())),
            topo.add_edge(Edge::new(vertices[1], vertices[2], curves[1].clone())),
            topo.add_edge(Edge::new(vertices[3], vertices[2], curves[2].clone())),
            topo.add_edge(Edge::new(vertices[0], vertices[3], curves[3].clone())),
        ];
        let wire = topo.add_wire(
            Wire::new(
                [
                    OrientedEdge::new(edges[0], true),
                    OrientedEdge::new(edges[1], true),
                    OrientedEdge::new(edges[2], false),
                    OrientedEdge::new(edges[3], false),
                ]
                .to_vec(),
                true,
            )
            .unwrap(),
        );
        topo.add_face(Face::new(wire, Vec::new(), surface))
    }

    #[test]
    fn cone_support_face_uses_native_uv_batch() {
        use brepkit_math::curves::Circle3D;
        use brepkit_math::surfaces::ConicalSurface;

        let mut topo = Topology::new();
        let cone = ConicalSurface::new(Point3::new(0.0, 0.0, 0.0), Vec3::new(0.0, 0.0, 1.0), 0.35)
            .unwrap();
        let point = |u: f64, v: f64| cone.evaluate(u, v);
        let circle = |v: f64| {
            EdgeCurve::Circle(
                Circle3D::new(
                    cone.apex() + cone.axis() * (cone.half_angle().sin() * v),
                    cone.axis(),
                    cone.radius_at(v),
                )
                .unwrap(),
            )
        };
        let face = add_parametric_quad(
            &mut topo,
            FaceSurface::Cone(cone.clone()),
            [
                point(0.2, 1.0),
                point(1.4, 1.0),
                point(1.4, 2.0),
                point(0.2, 2.0),
            ],
            [circle(1.0), EdgeCurve::Line, circle(2.0), EdgeCurve::Line],
        );
        let result = trim_parametric_face_batch(
            &mut topo,
            face,
            &[ParametricRestriction::new(
                vec![point(0.8, 1.0), point(0.8, 2.0)],
                TrimKeep::Side(TrimSide::Right),
            )],
        );
        assert!(result.is_ok(), "native cone UV support must rebuild");
    }

    #[test]
    fn nurbs_support_face_rebuilds_and_preserves_surface() {
        use brepkit_math::nurbs::surface::NurbsSurface;
        let mut topo = Topology::new();
        let surface = NurbsSurface::new(
            1,
            1,
            vec![0.0, 0.0, 1.0, 1.0],
            vec![0.0, 0.0, 1.0, 1.0],
            vec![
                vec![Point3::new(0.0, 0.0, 0.0), Point3::new(0.0, 1.0, 0.0)],
                vec![Point3::new(1.0, 0.0, 0.0), Point3::new(1.0, 1.0, 0.0)],
            ],
            vec![vec![1.0, 1.0], vec![1.0, 1.0]],
        )
        .unwrap();
        let face = add_parametric_quad(
            &mut topo,
            FaceSurface::Nurbs(surface),
            [
                Point3::new(0.0, 0.0, 0.0),
                Point3::new(1.0, 0.0, 0.0),
                Point3::new(1.0, 1.0, 0.0),
                Point3::new(0.0, 1.0, 0.0),
            ],
            [
                EdgeCurve::Line,
                EdgeCurve::Line,
                EdgeCurve::Line,
                EdgeCurve::Line,
            ],
        );
        let result = trim_parametric_face_batch(
            &mut topo,
            face,
            &[ParametricRestriction::new(
                vec![Point3::new(0.5, 0.0, 0.0), Point3::new(0.5, 1.0, 0.0)],
                TrimKeep::Side(TrimSide::Right),
            )],
        );
        assert!(
            result.is_ok(),
            "NURBS support must use its parameterization"
        );
        let trimmed = result.unwrap().trimmed_face;
        assert!(matches!(
            topo.face(trimmed).unwrap().surface(),
            FaceSurface::Nurbs(_)
        ));
    }

    #[test]
    fn torus_periodic_seam_batch_rebuilds_curved_patch() {
        use brepkit_math::curves::Circle3D;
        use brepkit_math::surfaces::ToroidalSurface;
        let mut topo = Topology::new();
        let torus = ToroidalSurface::new(Point3::new(0.0, 0.0, 0.0), 3.0, 1.0).unwrap();
        let point = |u: f64, v: f64| torus.evaluate(u, v);
        let circle_u = |v: f64| {
            EdgeCurve::Circle(
                Circle3D::new_with_ref(
                    Point3::new(0.0, 0.0, v.sin()),
                    Vec3::new(0.0, 0.0, 1.0),
                    3.0 + v.cos(),
                    Vec3::new(1.0, 0.0, 0.0),
                )
                .unwrap(),
            )
        };
        let circle_v = |u: f64| {
            let radial = Vec3::new(u.cos(), u.sin(), 0.0);
            let tangent = Vec3::new(-u.sin(), u.cos(), 0.0);
            EdgeCurve::Circle(
                Circle3D::new_with_ref(
                    Point3::new(3.0 * u.cos(), 3.0 * u.sin(), 0.0),
                    -tangent,
                    1.0,
                    radial,
                )
                .unwrap(),
            )
        };
        let u0 = std::f64::consts::TAU - 0.5;
        let u1 = std::f64::consts::TAU + 0.5;
        let v0 = 0.4;
        let v1 = 1.4;
        let face = add_parametric_quad(
            &mut topo,
            FaceSurface::Torus(torus.clone()),
            [point(u0, v0), point(u1, v0), point(u1, v1), point(u0, v1)],
            [circle_u(v0), circle_v(u1), circle_u(v1), circle_v(u0)],
        );
        let mut restriction = ParametricRestriction::new(
            vec![point(6.1, v0), point(6.1, v1)],
            TrimKeep::Side(TrimSide::Right),
        );
        let pcurve = brepkit_math::curves2d::Curve2D::Nurbs(
            brepkit_math::curves2d::NurbsCurve2D::new(
                2,
                vec![0.0, 0.0, 0.0, 1.0, 1.0, 1.0],
                vec![
                    brepkit_math::vec::Point2::new(6.1, v0),
                    brepkit_math::vec::Point2::new(6.25, 0.9),
                    brepkit_math::vec::Point2::new(6.1, v1),
                ],
                vec![1.0, 1.0, 1.0],
            )
            .unwrap(),
        );
        restriction.pcurve = Some(brepkit_topology::pcurve::PCurve::new(pcurve, 0.0, 1.0));
        let result = trim_parametric_face_batch(&mut topo, face, &[restriction]);
        assert!(
            result.is_ok(),
            "periodic torus patch must rebuild: {result:?}"
        );
        let result = result.unwrap();
        assert_ne!(result.trimmed_face, face);
        assert_eq!(result.contact_edges.len(), 1);
    }

    #[test]
    fn batch_rebuilds_outer_and_inner_wires() {
        let mut topo = Topology::new();
        let (face_id, _, _) = make_square_face(&mut topo);
        let hole_points = [
            Point3::new(0.25, 0.25, 0.0),
            Point3::new(0.75, 0.25, 0.0),
            Point3::new(0.75, 0.75, 0.0),
            Point3::new(0.25, 0.75, 0.0),
        ];
        let hole_vertices =
            hole_points.map(|point| topo.add_vertex(Vertex::new(point, VERTEX_TOL)));
        let hole_edges = [
            topo.add_edge(Edge::new(
                hole_vertices[0],
                hole_vertices[1],
                EdgeCurve::Line,
            )),
            topo.add_edge(Edge::new(
                hole_vertices[1],
                hole_vertices[2],
                EdgeCurve::Line,
            )),
            topo.add_edge(Edge::new(
                hole_vertices[2],
                hole_vertices[3],
                EdgeCurve::Line,
            )),
            topo.add_edge(Edge::new(
                hole_vertices[3],
                hole_vertices[0],
                EdgeCurve::Line,
            )),
        ];
        let hole_wire = topo.add_wire(
            Wire::new(
                hole_edges
                    .map(|edge| OrientedEdge::new(edge, true))
                    .to_vec(),
                true,
            )
            .unwrap(),
        );
        let outer_wire = topo.face(face_id).unwrap().outer_wire();
        let surface = topo.face(face_id).unwrap().surface().clone();
        let replacement = topo.add_face(Face::new(outer_wire, vec![hole_wire], surface));
        let restriction = ParametricRestriction::new(
            vec![Point3::new(0.5, 0.0, 0.0), Point3::new(0.5, 1.0, 0.0)],
            TrimKeep::Side(TrimSide::Right),
        );
        let result = trim_parametric_face_batch(&mut topo, replacement, &[restriction]).unwrap();
        let trimmed = topo.face(result.trimmed_face).unwrap();
        assert_eq!(trimmed.inner_wires().len(), 1);
        assert!(topo.wire(trimmed.inner_wires()[0]).unwrap().is_closed());
    }

    #[test]
    fn batch_rebuilds_incident_faces_copy_on_write() {
        use brepkit_math::curves2d::{Curve2D, Line2D};
        use brepkit_math::vec::{Point2, Vec2};
        use brepkit_topology::pcurve::PCurve;

        let mut topo = Topology::new();
        let (face_id, verts, edges) = make_square_face(&mut topo);
        let neighbor = attach_neighbor_below(&mut topo, verts[0], verts[1], edges[0]);
        let pcurve = PCurve::new(
            Curve2D::Line(Line2D::new(Point2::new(0.0, 0.0), Vec2::new(1.0, 0.0)).unwrap()),
            0.0,
            1.0,
        );
        topo.pcurves_mut().set(edges[0], face_id, pcurve.clone());
        topo.pcurves_mut().set(edges[0], neighbor, pcurve);
        let source_wire = topo.face(face_id).unwrap().outer_wire();
        let neighbor_wire = topo.face(neighbor).unwrap().outer_wire();
        let edge_signature = |wire: &Wire| {
            wire.edges()
                .iter()
                .map(|oe| (oe.edge(), oe.is_forward()))
                .collect::<Vec<_>>()
        };
        let source_before = edge_signature(topo.wire(source_wire).unwrap());
        let neighbor_before = edge_signature(topo.wire(neighbor_wire).unwrap());
        let restriction = PlanarRestriction::new(
            vec![Point3::new(0.5, 0.0, 0.0), Point3::new(0.5, 1.0, 0.0)],
            TrimKeep::Side(TrimSide::Right),
        );

        let result = trim_planar_face_batch(&mut topo, face_id, &[restriction]).unwrap();

        assert_eq!(
            edge_signature(topo.wire(source_wire).unwrap()),
            source_before,
            "source wire must remain unchanged",
        );
        assert_eq!(
            edge_signature(topo.wire(neighbor_wire).unwrap()),
            neighbor_before,
            "incident source wire must remain unchanged",
        );
        let replacement = result
            .incident_replacements
            .iter()
            .find(|&&(source, _)| source == neighbor)
            .map(|&(_, replacement)| replacement)
            .expect("split incident face must receive a COW replacement");
        let replacement_wire = topo
            .wire(topo.face(replacement).unwrap().outer_wire())
            .unwrap();
        assert!(
            replacement_wire
                .edges()
                .iter()
                .any(|oe| topo.pcurves().get(oe.edge(), replacement).is_some()),
            "COW replacement must preserve source pcurves on split pieces",
        );
        let trimmed = result.trimmed_face;
        let trimmed_wire = topo.wire(topo.face(trimmed).unwrap().outer_wire()).unwrap();
        assert!(
            trimmed_wire
                .edges()
                .iter()
                .any(|oe| topo.pcurves().get(oe.edge(), trimmed).is_some()),
            "trimmed face must preserve source pcurves on split pieces",
        );
        assert!(
            replacement_wire
                .edges()
                .iter()
                .all(|oe| oe.edge() != edges[0])
        );
        assert_eq!(replacement_wire.edges().len(), 5);
    }
    #[test]
    fn batch_split_preserves_nurbs_parameter_range() {
        let mut topo = Topology::new();
        let (face_id, _verts, edges) = make_square_face(&mut topo);
        let curve = brepkit_math::nurbs::curve::NurbsCurve::new(
            1,
            vec![0.0, 0.0, 1.0, 1.0],
            vec![Point3::new(0.0, 0.0, 0.0), Point3::new(1.0, 0.0, 0.0)],
            vec![1.0, 1.0],
        )
        .unwrap();
        topo.edge_mut(edges[0])
            .unwrap()
            .set_curve(EdgeCurve::NurbsCurve(curve));
        let restriction = PlanarRestriction::new(
            vec![Point3::new(0.5, 0.0, 0.0), Point3::new(0.5, 1.0, 0.0)],
            TrimKeep::Side(TrimSide::Right),
        );
        let result = trim_planar_face_batch(&mut topo, face_id, &[restriction]).unwrap();
        let wire = topo
            .wire(topo.face(result.trimmed_face).unwrap().outer_wire())
            .unwrap();
        let mut found = false;
        for oriented in wire.edges() {
            let edge = topo.edge(oriented.edge()).unwrap();
            if !matches!(edge.curve(), EdgeCurve::NurbsCurve(_)) {
                continue;
            }
            let start = topo.vertex(edge.start()).unwrap().point();
            let end = topo.vertex(edge.end()).unwrap().point();
            if (start.x() - 0.5).abs() < 1e-7 || (end.x() - 0.5).abs() < 1e-7 {
                let (a, b) = edge.curve().domain_with_endpoints(start, end);
                assert!(
                    (a - 0.5).abs() < 1e-7 || (b - 0.5).abs() < 1e-7,
                    "split NURBS span must end at 0.5, got ({a}, {b})"
                );
                found = true;
            }
        }
        assert!(found, "batch result must retain a split NURBS sub-edge");
    }
    #[test]
    fn batch_split_preserves_circle_parameter_range() {
        let mut topo = Topology::new();
        let (face_id, _verts, edges) = make_square_face(&mut topo);
        let circle = brepkit_math::curves::Circle3D::new(
            Point3::new(1.0, 0.5, 0.0),
            Vec3::new(0.0, 0.0, 1.0),
            0.5,
        )
        .unwrap();
        topo.edge_mut(edges[1])
            .unwrap()
            .set_curve(EdgeCurve::Circle(circle));
        let restriction = PlanarRestriction::new(
            vec![Point3::new(0.0, 0.5, 0.0), Point3::new(1.5, 0.5, 0.0)],
            TrimKeep::Side(TrimSide::Right),
        );
        let result = trim_planar_face_batch(&mut topo, face_id, &[restriction]).unwrap();
        let split_vertex = result
            .new_vertices
            .iter()
            .copied()
            .find(|&vid| {
                let p = topo.vertex(vid).unwrap().point();
                (p.x() - 1.5).abs() < 1e-7 && (p.y() - 0.5).abs() < 1e-7
            })
            .expect("boundary crossing must be evaluated on the circle, not its chord");
        let wire = topo
            .wire(topo.face(result.trimmed_face).unwrap().outer_wire())
            .unwrap();
        let mut found = false;
        for oriented in wire.edges() {
            let edge = topo.edge(oriented.edge()).unwrap();
            if !matches!(edge.curve(), EdgeCurve::Circle(_)) {
                continue;
            }
            let start = topo.vertex(edge.start()).unwrap().point();
            let end = topo.vertex(edge.end()).unwrap().point();
            if edge.start() != split_vertex && edge.end() != split_vertex {
                continue;
            }
            let (a, b) = edge.curve().domain_with_endpoints(start, end);
            assert!(
                (b - a).abs() < std::f64::consts::TAU - 1e-7,
                "split circle edge must retain a sub-range, got ({a}, {b})"
            );
            found = true;
        }
        assert!(found, "batch result must retain a split circle sub-edge");
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
