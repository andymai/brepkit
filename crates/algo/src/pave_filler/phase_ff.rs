//! Phase FF: Face-face intersection detection.
//!
//! For each (face_a, face_b) pair across solids, computes intersection
//! curves. Results are stored as `IntersectionCurveDS` entries in the
//! GFA arena, with FF interferences referencing them by index.
//!
//! Each raw curve also gets a pave block spanning its full parameter
//! range, with topology vertices and an edge created at the endpoints.

use brepkit_math::aabb::Aabb3;
use brepkit_math::analytic_intersection;
use brepkit_math::nurbs::intersection as nurbs_isect;
use brepkit_math::tolerance::Tolerance;
use brepkit_math::traits::ParametricCurve;
use brepkit_math::vec::{Point3, Vec3};
use brepkit_topology::Topology;
use brepkit_topology::edge::{Edge, EdgeCurve};
use brepkit_topology::face::{FaceId, FaceSurface};
use brepkit_topology::solid::SolidId;
use brepkit_topology::vertex::Vertex;

use crate::ds::{GfaArena, Interference, IntersectionCurveDS, Pave, PaveBlock};
use crate::error::AlgoError;

/// Default number of samples for NURBS intersection.
const NURBS_SAMPLES: usize = 32;

/// Default march step for NURBS-NURBS intersection.
const NURBS_MARCH_STEP: f64 = 0.01;

/// Detect face-face intersections between the two solids.
///
/// For each face pair (one from each solid), computes intersection
/// curves using surface-type-specific algorithms. Plane-plane line
/// curves are trimmed to the mutual overlap of the two faces; other
/// raw curves are stored untrimmed (boundary trimming is a later phase).
///
/// Creates topology vertices and edges for each intersection curve
/// endpoint, and a pave block spanning the full parameter range.
///
/// # Errors
///
/// Returns [`AlgoError`] if any topology lookup or intersection computation fails.
#[allow(clippy::too_many_lines)]
pub fn perform(
    topo: &mut Topology,
    solid_a: SolidId,
    solid_b: SolidId,
    tol: Tolerance,
    arena: &mut GfaArena,
) -> Result<(), AlgoError> {
    let faces_a = brepkit_topology::explorer::solid_faces(topo, solid_a)?;
    let faces_b = brepkit_topology::explorer::solid_faces(topo, solid_b)?;

    // Pre-compute face AABBs for rejection
    let bboxes_a = compute_face_bboxes(topo, &faces_a, tol)?;
    let bboxes_b = compute_face_bboxes(topo, &faces_b, tol)?;

    // Collect all surface data upfront so we don't borrow topo immutably
    // while mutating it later.
    let surfs_a: Vec<FaceSurface> = faces_a
        .iter()
        .map(|&fa| topo.face(fa).map(|f| f.surface().clone()))
        .collect::<Result<_, _>>()?;
    let surfs_b: Vec<FaceSurface> = faces_b
        .iter()
        .map(|&fb| topo.face(fb).map(|f| f.surface().clone()))
        .collect::<Result<_, _>>()?;

    // Pre-compute v-parameter ranges for analytic surfaces (used by AA intersection)
    let v_ranges_a: Vec<Option<(f64, f64)>> = faces_a
        .iter()
        .zip(surfs_a.iter())
        .map(|(&fid, surf)| face_v_range(topo, fid, surf))
        .collect();
    let v_ranges_b: Vec<Option<(f64, f64)>> = faces_b
        .iter()
        .zip(surfs_b.iter())
        .map(|(&fid, surf)| face_v_range(topo, fid, surf))
        .collect();

    // Registry of endpoint vertices created for the EXACT faceted-ramp arcs
    // (see `trim_ellipse_to_boundary_crossings`). Adjacent treads share a
    // boundary line, so their arcs end at a bit-identical crossing point;
    // snapping the second arc's endpoint to the first's vertex chains the arcs
    // into one continuous split curve. Keyed by a fine quantization so only
    // genuinely-coincident crossings (well under the linear tolerance) merge.
    let mut exact_arc_vertices: std::collections::HashMap<
        (i64, i64, i64),
        brepkit_topology::vertex::VertexId,
    > = std::collections::HashMap::new();

    for (idx_a, &fa) in faces_a.iter().enumerate() {
        let bbox_a = &bboxes_a[idx_a];
        let surf_a = &surfs_a[idx_a];

        for (idx_b, &fb) in faces_b.iter().enumerate() {
            let bbox_b = &bboxes_b[idx_b];

            // AABB rejection
            if !bbox_a
                .expanded(tol.linear)
                .intersects(bbox_b.expanded(tol.linear))
            {
                continue;
            }

            let surf_b = &surfs_b[idx_b];

            let v_range_a = v_ranges_a[idx_a];
            let v_range_b = v_ranges_b[idx_b];
            let raw_curves =
                compute_raw_curves(surf_a, surf_b, bbox_a, bbox_b, v_range_a, v_range_b)?;

            // For plane-plane Line curves with all-straight-edge faces, trim
            // each curve to the mutual overlap of the two faces' clipped
            // ranges. Without this the section curve spans the union of both
            // face extents, producing over-long chords that cross face
            // boundaries mid-edge and corrupt the downstream face partition.
            // When a face's polygon is built but the line lies outside it the
            // overlap is empty and the curve is dropped; when a polygon can't
            // be built (or is non-convex) the raw curve is kept conservatively.
            let raw_curves: Vec<RawCurve> = if matches!(
                surf_a,
                FaceSurface::Plane { .. } | FaceSurface::Cylinder(_) | FaceSurface::Cone(_)
            ) && matches!(
                surf_b,
                FaceSurface::Plane { .. } | FaceSurface::Cylinder(_) | FaceSurface::Cone(_)
            ) {
                raw_curves
                    .into_iter()
                    .filter_map(|raw| {
                        if !matches!(raw.curve, EdgeCurve::Line) {
                            return Some(raw);
                        }
                        // A banded partner (cylinder/cone) admits an EXACT
                        // v-window clip: the section line lies ON the surface,
                        // and `project_point`'s v is affine along a straight
                        // line there, so the band limits map to exact line
                        // fractions. Without this, a plane×band Line section
                        // keeps its full span (the polygon clip below is
                        // Indeterminate for the band side), crosses the band
                        // overlong, and the plane-side splitter receives a
                        // dangling silhouette chain (the snap-slot wall).
                        let band = |surf: &FaceSurface,
                                    v_range: Option<(f64, f64)>|
                         -> Option<(f64, f64)> {
                            let (mut v0, mut v1) = v_range?;
                            if v1 < v0 {
                                std::mem::swap(&mut v0, &mut v1);
                            }
                            let (vs, ve) = match surf {
                                FaceSurface::Cylinder(c) => {
                                    (c.project_point(raw.p_start).1, c.project_point(raw.p_end).1)
                                }
                                FaceSurface::Cone(c) => {
                                    (c.project_point(raw.p_start).1, c.project_point(raw.p_end).1)
                                }
                                _ => return None,
                            };
                            let dv = ve - vs;
                            if dv.abs() < 1e-12 {
                                return None;
                            }
                            let f0 = (v0 - vs) / dv;
                            let f1 = (v1 - vs) / dv;
                            Some(if f0 <= f1 { (f0, f1) } else { (f1, f0) })
                        };
                        let mut lo = 0.0_f64;
                        let mut hi = 1.0_f64;
                        if let Some((b0, b1)) = band(surf_a, v_range_a) {
                            lo = lo.max(b0);
                            hi = hi.min(b1);
                        }
                        if let Some((b0, b1)) = band(surf_b, v_range_b) {
                            lo = lo.max(b0);
                            hi = hi.min(b1);
                        }
                        if hi - lo <= 0.0 {
                            return None;
                        }
                        let both_planes = matches!(surf_a, FaceSurface::Plane { .. })
                            && matches!(surf_b, FaceSurface::Plane { .. });
                        if !both_planes {
                            // Mixed pair: the exact band trim is the only clip
                            // applied — the plane-polygon clip below is
                            // calibrated for plane×plane sections, and running
                            // it on a banded pair's line disturbed the
                            // seam-anchored cylinder band splitting.
                            if lo > 0.0 || hi < 1.0 {
                                return trim_raw_line(&raw, lo, hi, tol);
                            }
                            return Some(raw);
                        }
                        if lo > 0.0 || hi < 1.0 {
                            let trimmed = trim_raw_line(&raw, lo, hi, tol)?;
                            return clip_trimmed_line_to_planes(topo, fa, fb, trimmed, tol);
                        }
                        let clip_a = clip_line_to_face(topo, fa, &raw);
                        let clip_b = clip_line_to_face(topo, fb, &raw);
                        match (clip_a, clip_b) {
                            // A face's polygon was built but the line lies
                            // entirely outside it: the mutual overlap is
                            // empty, so the section curve does not belong on
                            // this pair — drop it.
                            (FaceClip::Empty, _) | (_, FaceClip::Empty) => None,
                            // Both clips produced an interval: trim to the
                            // mutual overlap.
                            (FaceClip::Range(a), FaceClip::Range(b)) => {
                                let f0 = a.0.max(b.0);
                                let f1 = a.1.min(b.1);
                                trim_raw_line(&raw, f0, f1, tol)
                            }
                            // One face produced an interval, the other could
                            // not build a usable polygon (degenerate wire,
                            // non-line edges such as rounded-rect corner
                            // arcs, or a non-convex outline). The single
                            // interval is still a superset of the mutual
                            // overlap, so trim to it — keeping the raw curve
                            // here produced over-long chords that crossed
                            // the partner's arc sections mid-edge.
                            (FaceClip::Range(r), FaceClip::Indeterminate)
                            | (FaceClip::Indeterminate, FaceClip::Range(r)) => {
                                trim_raw_line(&raw, r.0, r.1, tol)
                            }
                            // Neither face could build a usable polygon.
                            // Conservatively keep the raw curve and leave
                            // trimming to a later phase.
                            (FaceClip::Indeterminate, FaceClip::Indeterminate) => Some(raw),
                        }
                    })
                    .collect()
            } else {
                raw_curves
            };

            // Exact faceted-ramp × cylinder/cone arc assembly. A thin planar
            // tread meeting a corner cylinder yields a closed ellipse whose
            // in-both arc is a sub-millimetre sliver: the 16-sample AABB
            // pre-filter below and the uniform-t restriction both drop it (no
            // sample lands in the band), so the cylinder never splits. When the
            // pair is {planar tread} × {cylinder/cone}, instead trim the
            // ellipse to the EXACT crossings of the tread's boundary lines with
            // the surface — shared between adjacent treads, so the arcs chain.
            // These exact arcs bypass the generic filters below.
            let (exact_arcs, raw_curves): (Vec<RawCurve>, Vec<RawCurve>) = {
                let ext_a = FaceExtent::new(topo, fa, surf_a, v_range_a, tol);
                let ext_b = FaceExtent::new(topo, fb, surf_b, v_range_b, tol);
                let mut exact = Vec::new();
                let mut rest = Vec::new();
                for raw in raw_curves {
                    if let (Some(ea), Some(eb)) = (&ext_a, &ext_b)
                        && let Some(arcs) = trim_ellipse_to_boundary_crossings(
                            topo, fa, fb, surf_a, surf_b, &raw, ea, eb,
                        )
                    {
                        exact.extend(arcs);
                    } else {
                        rest.push(raw);
                    }
                }
                (exact, rest)
            };

            // Raw curves come from UNTRIMMED surface-surface intersection, so
            // a curve can lie entirely beyond both faces' trimmed extents
            // (e.g. tangency curvelets where a cone grazes a narrower
            // cylinder, or a full cap circle paired with a smaller distant
            // cap). Such curves fragment faces with spurious holes and bogus
            // sub-faces downstream. Keep a curve only if at least one sample
            // lies inside both faces' inflated AABBs. A fixed sample count
            // misses an in-both span much shorter than the curve (a marched
            // plane×cone conic spans the unbounded cone, so a small face's
            // true crossing can be a ~2mm sliver of a ~30mm curve); before
            // declaring a miss, refine with a density scaled to the smaller
            // face AABB dimension — the same escalation the in-both
            // restriction below applies to grazes.
            let bb_a = bbox_a.expanded(tol.linear * 10.0);
            let bb_b = bbox_b.expanded(tol.linear * 10.0);
            let raw_curves: Vec<RawCurve> = raw_curves
                .into_iter()
                .filter(|raw| {
                    const N: usize = 16;
                    let sample = |i: usize, n: usize| -> Point3 {
                        #[allow(clippy::cast_precision_loss)]
                        let f = i as f64 / n as f64;
                        // Line t_range is absolute arc length, not a
                        // normalized [0,1] span — sample by endpoint lerp.
                        if matches!(raw.curve, EdgeCurve::Line) {
                            raw.p_start + (raw.p_end - raw.p_start) * f
                        } else {
                            let t = raw.t_range.0 + (raw.t_range.1 - raw.t_range.0) * f;
                            raw.curve.evaluate_with_endpoints(t, raw.p_start, raw.p_end)
                        }
                    };
                    let in_both =
                        |p: Point3| -> bool { bb_a.contains_point(p) && bb_b.contains_point(p) };
                    if (0..=N).map(|i| sample(i, N)).any(in_both) {
                        return true;
                    }
                    // Straight lines are exactly represented by their
                    // endpoints; a uniform scan cannot under-sample them at
                    // this granularity in practice, and refining every far
                    // pair would be pure cost.
                    if matches!(raw.curve, EdgeCurve::Line) {
                        return false;
                    }
                    let approx_len: f64 = (0..N)
                        .map(|i| (sample(i + 1, N) - sample(i, N)).length())
                        .sum();
                    // Smallest POSITIVE extent: a planar face's bbox is flat
                    // along its normal, and that zero span says nothing about
                    // how finely the in-both region must be sampled.
                    let min_dim = |bb: &Aabb3| -> f64 {
                        let e = bb.max - bb.min;
                        [e.x(), e.y(), e.z()]
                            .into_iter()
                            .filter(|&s| s > tol.linear * 1e2)
                            .fold(f64::INFINITY, f64::min)
                    };
                    let dim_of = |a: &Aabb3, b: &Aabb3| -> f64 {
                        let d = min_dim(a).min(min_dim(b));
                        if d.is_finite() { d } else { tol.linear * 1e2 }
                    };
                    let dim = dim_of(&bb_a, &bb_b);
                    #[allow(clippy::cast_possible_truncation, clippy::cast_sign_loss)]
                    let n_fine = ((8.0 * approx_len / dim).ceil() as usize).clamp(N, 1024);
                    n_fine > N && (0..=n_fine).map(|i| sample(i, n_fine)).any(in_both)
                })
                .collect();

            // Restrict surface-surface intersection curves to the region that
            // lies inside BOTH faces. `compute_raw_curves` works on the
            // unbounded surfaces; the analytic-analytic marcher already bounds
            // its curves, but the plane-analytic and algebraic paths return
            // the full surface-surface curve. A cylinder/cone or
            // cylinder/tilted-plane ellipse that meets the other face only
            // along a shared cap then reaches far past it and slits the
            // partner face's wire. Keep only the in-both span (curves only).
            let raw_curves = restrict_curves_to_faces(
                topo, fa, fb, surf_a, surf_b, v_range_a, v_range_b, raw_curves, tol,
            );
            // Emit the EXACT faceted-ramp arcs with registry-aware endpoint
            // resolution: each arc's endpoints are bit-identical to the shared
            // boundary-line crossing of the adjacent tread's arc, so consult
            // the registry (and pre-existing paves/boundary vertices) and snap.
            for raw in &exact_arcs {
                emit_exact_arc(topo, arena, fa, fb, raw, tol, &mut exact_arc_vertices);
            }

            for raw in raw_curves {
                let mut raw = raw;
                // Closed Circle3D sections — produced by plane-sphere
                // intersections — get split at face-boundary crossings so
                // downstream face splitters see open arcs they can match
                // up. Without this, the closed curve is dropped by
                // `split_noseam_face_direct` and the spherical sub-face is
                // lost entirely.
                //
                // Note: the split here is structurally correct (the FF
                // arcs now have proper endpoints on the analytic face's
                // boundary), but `split_noseam_face_direct` still can't
                // form a single closed cap loop when 2+ arcs cross on a
                // sphere hemisphere (e.g. the great circles from x=0 and
                // y=0 planes meeting at the pole). The hemisphere falls
                // back to "unsplit" → the spherical sub-face is missing
                // from the GFA output → the boolean pipeline retries with
                // mesh boolean. Closing that gap requires generalising
                // the face splitter for multi-arc sphere hemispheres —
                // see `crates/algo/src/builder/face_splitter/mod.rs:138`
                // dispatch and `special_cases.rs::split_noseam_face_direct`.
                if let EdgeCurve::Circle(circle) = &raw.curve {
                    let is_closed = (raw.p_start - raw.p_end).length() < tol.linear;
                    if is_closed {
                        let crossings = closed_circle_boundary_crossings(topo, fa, fb, circle, tol);
                        if crossings.len() >= 2 {
                            emit_split_circle_arcs(
                                topo, arena, fa, fb, &raw, circle, &crossings, tol,
                            );
                            continue;
                        }
                    }
                }

                // Create topology vertices at the curve endpoints.
                // For closed curves (Circle/Ellipse), start and end are the same
                // 3D point — reuse one vertex for correct seam topology.
                //
                // Snap to existing vertices (from input face boundaries or
                // earlier intersection curves) when within tolerance. This is
                // the PutPavesOnCurve equivalent: it ensures intersection curve
                // endpoints share vertices with face boundaries, so the face
                // splitter produces sub-faces with consistent vertex identity.
                let is_closed = (raw.p_start - raw.p_end).length() < tol.linear;

                // For closed curves the start/end point is at an arbitrary
                // curve parameter. If an existing boundary vertex of either
                // face already lies on the curve (e.g. the seam vertex of a
                // coincident cap circle), adopt it as the seam and shift the
                // parameter range to start there — so the section edge and
                // the existing boundary edge share endpoint identity and can
                // be linked into a CommonBlock by `link_existing`.
                //
                // Only safe when the circle stays inside (or on) both faces:
                // a circle that properly crosses a face boundary must keep
                // the legacy fresh-seam path so the downstream splitter can
                // trim it to the in-face arcs.
                let adopted_seam = match &raw.curve {
                    EdgeCurve::Circle(c)
                        if is_closed
                            && !closed_circle_crosses_face_boundaries(topo, fa, fb, c, tol) =>
                    {
                        find_boundary_vertex_on_curve(topo, fa, fb, &raw.curve, tol)
                    }
                    _ => None,
                };
                if let Some((_, t_seam, p_seam)) = adopted_seam {
                    let span = raw.t_range.1 - raw.t_range.0;
                    raw.t_range = (t_seam, t_seam + span);
                    raw.p_start = p_seam;
                    raw.p_end = p_seam;
                }

                let start_vid = adopted_seam.map(|(vid, _, _)| vid).unwrap_or_else(|| {
                    super::helpers::find_nearby_pave_vertex(topo, arena, raw.p_start, tol)
                        .or_else(|| find_nearby_face_vertex(topo, fa, raw.p_start, tol))
                        .or_else(|| find_nearby_face_vertex(topo, fb, raw.p_start, tol))
                        .unwrap_or_else(|| topo.add_vertex(Vertex::new(raw.p_start, tol.linear)))
                });
                let end_vid = if is_closed {
                    start_vid
                } else {
                    super::helpers::find_nearby_pave_vertex(topo, arena, raw.p_end, tol)
                        .or_else(|| find_nearby_face_vertex(topo, fa, raw.p_end, tol))
                        .or_else(|| find_nearby_face_vertex(topo, fb, raw.p_end, tol))
                        .unwrap_or_else(|| topo.add_vertex(Vertex::new(raw.p_end, tol.linear)))
                };

                let edge = Edge::new(start_vid, end_vid, raw.curve.clone());
                let edge_id = topo.add_edge(edge);

                let start_pave = Pave::new(start_vid, raw.t_range.0);
                let end_pave = Pave::new(end_vid, raw.t_range.1);
                let pb = PaveBlock::new(edge_id, start_pave, end_pave);
                let pb_id = arena.pave_blocks.alloc(pb);

                let curve_index = arena.curves.len();
                arena.curves.push(IntersectionCurveDS {
                    curve: raw.curve,
                    face_a: fa,
                    face_b: fb,
                    bbox: raw.bbox,
                    pave_blocks: vec![pb_id],
                    t_range: raw.t_range,
                });

                arena.interference.ff.push(Interference::FF {
                    f1: fa,
                    f2: fb,
                    curve_index,
                });

                log::debug!(
                    "FF: faces {fa:?} and {fb:?} intersect (curve_index={curve_index}, \
                     edge={edge_id:?}, pb={pb_id:?})",
                );
            }
        }
    }

    Ok(())
}

/// Quantize a point to a fine grid for the exact-arc vertex registry. The
/// step (1e-9) is far below the linear tolerance, so only crossings that are
/// numerically the same point (the same boundary-line × surface root computed
/// from the same edge) collapse to one key.
fn exact_arc_key(p: Point3) -> (i64, i64, i64) {
    let s = 1.0e9;
    #[allow(clippy::cast_possible_truncation)]
    (
        (p.x() * s).round() as i64,
        (p.y() * s).round() as i64,
        (p.z() * s).round() as i64,
    )
}

/// Emit one EXACT faceted-ramp arc as an `IntersectionCurveDS` + edge + pave
/// block. Endpoints resolve through the shared-crossing registry first so the
/// adjacent tread's arc, which ends at the same boundary-line crossing, reuses
/// the same vertex and the arcs chain into a continuous split curve.
fn emit_exact_arc(
    topo: &mut Topology,
    arena: &mut GfaArena,
    fa: FaceId,
    fb: FaceId,
    raw: &RawCurve,
    tol: Tolerance,
    registry: &mut std::collections::HashMap<(i64, i64, i64), brepkit_topology::vertex::VertexId>,
) {
    let resolve = |topo: &mut Topology,
                   arena: &GfaArena,
                   registry: &mut std::collections::HashMap<
        (i64, i64, i64),
        brepkit_topology::vertex::VertexId,
    >,
                   p: Point3|
     -> brepkit_topology::vertex::VertexId {
        let key = exact_arc_key(p);
        if let Some(&vid) = registry.get(&key) {
            return vid;
        }
        let vid = super::helpers::find_nearby_pave_vertex(topo, arena, p, tol)
            .or_else(|| find_nearby_face_vertex(topo, fa, p, tol))
            .or_else(|| find_nearby_face_vertex(topo, fb, p, tol))
            .unwrap_or_else(|| topo.add_vertex(Vertex::new(p, tol.linear)));
        registry.insert(key, vid);
        vid
    };

    let start_vid = resolve(topo, arena, registry, raw.p_start);
    let end_vid = resolve(topo, arena, registry, raw.p_end);

    let edge = Edge::new(start_vid, end_vid, raw.curve.clone());
    let edge_id = topo.add_edge(edge);

    let start_pave = Pave::new(start_vid, raw.t_range.0);
    let end_pave = Pave::new(end_vid, raw.t_range.1);
    let pb = PaveBlock::new(edge_id, start_pave, end_pave);
    let pb_id = arena.pave_blocks.alloc(pb);

    let curve_index = arena.curves.len();
    arena.curves.push(IntersectionCurveDS {
        curve: raw.curve.clone(),
        face_a: fa,
        face_b: fb,
        bbox: raw.bbox,
        pave_blocks: vec![pb_id],
        t_range: raw.t_range,
    });

    arena.interference.ff.push(Interference::FF {
        f1: fa,
        f2: fb,
        curve_index,
    });
}

/// A face's trimmed extent, used to test whether a 3D point lies inside the
/// face (so surface-surface intersection curves can be restricted to the
/// region inside both faces — the reference's "true boundary" restriction).
enum FaceExtent {
    /// Planar face: 2D outer boundary polygon in a plane frame (arc edges
    /// sampled), plus any inner-wire (hole) polygons subtracted from it.
    Plane {
        frame: crate::builder::plane_frame::PlaneFrame,
        poly: Vec<brepkit_math::vec::Point2>,
        holes: Vec<Vec<brepkit_math::vec::Point2>>,
        margin: f64,
    },
    /// Analytic lateral face (cylinder/cone/sphere/torus): bound by the
    /// axial `v` parameter range of the face and, for a partial-arc patch
    /// (e.g. a rounded-rect corner quarter-cylinder), the excluded angular
    /// `u` gap so a point on the off-patch side of the full revolution is
    /// rejected.
    Analytic {
        surface: FaceSurface,
        v0: f64,
        v1: f64,
        margin: f64,
        u_gap: Option<(f64, f64)>,
    },
}

impl FaceExtent {
    fn new(
        topo: &Topology,
        face_id: FaceId,
        surface: &FaceSurface,
        v_range: Option<(f64, f64)>,
        tol: Tolerance,
    ) -> Option<Self> {
        if let FaceSurface::Plane { normal, .. } = surface {
            let face = topo.face(face_id).ok()?;
            let outer = topo.wire(face.outer_wire()).ok()?;
            let first = outer.edges().first()?;
            let origin = topo
                .vertex(topo.edge(first.edge()).ok()?.start())
                .ok()?
                .point();
            let frame =
                crate::builder::plane_frame::PlaneFrame::from_normal_and_point(*normal, origin);
            // Project a wire's boundary into the plane frame, sampling each arc
            // edge (endpoints included) so curved corners aren't chord-cut.
            let wire_poly = |wid| -> Option<Vec<brepkit_math::vec::Point2>> {
                let wire = topo.wire(wid).ok()?;
                let mut poly = Vec::new();
                for oe in wire.edges() {
                    let edge = topo.edge(oe.edge()).ok()?;
                    let s3 = topo.vertex(edge.start()).ok()?.point();
                    let e3 = topo.vertex(edge.end()).ok()?.point();
                    match edge.curve() {
                        EdgeCurve::Line => {
                            poly.push(frame.project(if oe.is_forward() { s3 } else { e3 }));
                        }
                        curve => {
                            let (t0, t1) = curve.domain_with_endpoints(s3, e3);
                            for i in 0..=16 {
                                #[allow(clippy::cast_precision_loss)]
                                let f = i as f64 / 16.0;
                                let t = if oe.is_forward() {
                                    t0 + (t1 - t0) * f
                                } else {
                                    t1 + (t0 - t1) * f
                                };
                                poly.push(frame.project(curve.evaluate_with_endpoints(t, s3, e3)));
                            }
                        }
                    }
                }
                Some(poly)
            };
            let poly = wire_poly(face.outer_wire())?;
            if poly.len() < 3 {
                return None;
            }
            let holes: Vec<_> = face
                .inner_wires()
                .iter()
                .filter_map(|&iw| wire_poly(iw))
                .filter(|h| h.len() >= 3)
                .collect();
            // Margin keeps boundary-coincident points (the shared cap) inside;
            // scaled to the SMALLER in-plane extent so a thin band (a scoop
            // staircase tread is full-width but ~0.1 mm tall) keeps a tight
            // margin instead of one scaled to the wide diagonal — a diagonal
            // margin lets a section overshoot the thin band onto the cylinder
            // seam, so the per-tread arcs never stop at the tread boundary.
            let bb = brepkit_math::aabb::Aabb3::from_points(
                poly.iter()
                    .map(|p| brepkit_math::vec::Point3::new(p.x(), p.y(), 0.0)),
            );
            let extent = bb.max - bb.min;
            let smaller = extent.x().abs().min(extent.y().abs());
            Some(Self::Plane {
                frame,
                poly,
                holes,
                margin: (smaller * 0.01).max(tol.linear),
            })
        } else {
            // A whole, untrimmed torus has no `v_range` (its boundary is the
            // degenerate fundamental-polygon seam), yet it IS a real extent: the
            // full tube v ∈ [0, 2π], full revolution u (no gap). Treating it as
            // full extent lets `restrict_curves_to_faces` run the in-both clip,
            // so a plane×torus oval is trimmed to the partner box face's region
            // instead of bailing unclipped. Mirrors the whole-torus AABB gate
            // (`face_boundary_all_degenerate` → `t.aabb()`).
            let (v0, v1) = match v_range {
                Some(r) => r,
                None => {
                    if matches!(surface, FaceSurface::Torus(_))
                        && face_boundary_all_degenerate(topo, face_id, tol).unwrap_or(false)
                    {
                        (0.0, std::f64::consts::TAU)
                    } else {
                        return None;
                    }
                }
            };
            let margin = (v1 - v0).abs() * 0.01 + tol.linear;
            // For a partial-arc lateral face (rounded-rect corner = a 90°
            // quarter-cylinder), record the angular gap the face does NOT
            // cover so `contains` rejects a point that projects onto the
            // off-patch side of the full revolution. Without this, a
            // near-tangent section curve "inside" the v-band but on the far
            // half of the cylinder is wrongly kept, wrapping the trimmed arc
            // onto the wrong side of the wedge.
            let u_gap = face_circumferential_u_gap(topo, face_id, surface);
            Some(Self::Analytic {
                surface: surface.clone(),
                v0,
                v1,
                margin,
                u_gap,
            })
        }
    }

    /// Smallest in-face dimension, used to scale the graze-refinement sample
    /// density in `restrict_curves_to_faces`. Plane: the boundary polygon
    /// bbox's smaller side. Analytic: the `v` span (axial length for
    /// cylinders/cones — a coarse but sufficient proxy; the result only
    /// scales a clamped sample count).
    fn min_dimension(&self) -> f64 {
        match self {
            Self::Plane { poly, .. } => {
                let (mut min_x, mut max_x) = (f64::INFINITY, f64::NEG_INFINITY);
                let (mut min_y, mut max_y) = (f64::INFINITY, f64::NEG_INFINITY);
                for p in poly {
                    min_x = min_x.min(p.x());
                    max_x = max_x.max(p.x());
                    min_y = min_y.min(p.y());
                    max_y = max_y.max(p.y());
                }
                (max_x - min_x).min(max_y - min_y).abs()
            }
            Self::Analytic { v0, v1, .. } => (v1 - v0).abs(),
        }
    }

    /// Like `contains`, but requires the point to sit INSIDE the true face
    /// window by at least `depth` (no boundary margin credit). Distinguishes a
    /// section that genuinely crosses the window interior from one that only
    /// rides the margin band along a boundary (a tangency graze).
    fn contains_strict(&self, p: Point3, depth: f64) -> bool {
        match self {
            Self::Plane {
                frame, poly, holes, ..
            } => {
                let uv = frame.project(p);
                crate::builder::classify_2d::point_in_polygon_2d(uv, poly)
                    && point_to_polygon_dist(uv, poly) > depth
                    && !holes.iter().any(|h| {
                        crate::builder::classify_2d::point_in_polygon_2d(uv, h)
                            || point_to_polygon_dist(uv, h) <= depth
                    })
            }
            Self::Analytic {
                surface,
                v0,
                v1,
                u_gap,
                ..
            } => surface.project_point(p).is_some_and(|(u, v)| {
                // Unlike `contains` (conservative keep on projection failure),
                // the STRICT gate fails closed: an unprojectable point cannot
                // certify a genuine interior crossing.
                let in_v = v >= *v0 + depth && v <= *v1 - depth;
                let in_u = u_gap.is_none_or(|gap| !crate::classifier::u_in_gap(u, gap));
                in_v && in_u
            }),
        }
    }

    fn contains(&self, p: Point3) -> bool {
        match self {
            Self::Plane {
                frame,
                poly,
                holes,
                margin,
            } => {
                let uv = frame.project(p);
                let in_outer = crate::builder::classify_2d::point_in_polygon_2d(uv, poly)
                    || point_to_polygon_dist(uv, poly) <= *margin;
                if !in_outer {
                    return false;
                }
                // A point strictly inside a hole (beyond the boundary margin)
                // is not on the trimmed face.
                !holes.iter().any(|h| {
                    crate::builder::classify_2d::point_in_polygon_2d(uv, h)
                        && point_to_polygon_dist(uv, h) > *margin
                })
            }
            Self::Analytic {
                surface,
                v0,
                v1,
                margin,
                u_gap,
            } => surface.project_point(p).is_none_or(|(u, v)| {
                let in_v = v >= *v0 - *margin && v <= *v1 + *margin;
                let in_u = u_gap.is_none_or(|gap| !crate::classifier::u_in_gap(u, gap));
                in_v && in_u
            }),
        }
    }
}

/// Compute the excluded angular `u` gap of a cylinder/cone lateral face from
/// its outer-wire boundary samples. Returns `None` for a full revolution (no
/// gap) or non-cylindrical/conical surfaces.
///
/// `project_point` returns `(u, v)` where `u` is the circumferential angle,
/// so the largest gap between sorted boundary `u`-samples is the arc the face
/// does not cover (e.g. the 270° the rounded-rect corner quarter omits).
fn face_circumferential_u_gap(
    topo: &Topology,
    face_id: FaceId,
    surface: &FaceSurface,
) -> Option<(f64, f64)> {
    // Sample densely so a FULL-revolution rim circle (one closed edge spanning
    // the whole period) leaves only a tiny per-step gap, well under the
    // `largest_u_gap` threshold — otherwise a 16-step pass (~0.39 rad/step)
    // spuriously reports a gap on a full cylinder wall, wrongly excluding genuine
    // on-surface points near that u. A real partial-arc face (a rounded-rect
    // corner) still shows its large angular gap.
    const N_GAP: usize = 64;
    if !matches!(surface, FaceSurface::Cylinder(_) | FaceSurface::Cone(_)) {
        return None;
    }
    let face = topo.face(face_id).ok()?;
    let wire = topo.wire(face.outer_wire()).ok()?;
    let mut u_samples = Vec::new();
    for oe in wire.edges() {
        let edge = topo.edge(oe.edge()).ok()?;
        let sp = topo.vertex(edge.start()).ok()?.point();
        let ep = topo.vertex(edge.end()).ok()?.point();
        let (t0, t1) = edge.curve().domain_with_endpoints(sp, ep);
        for i in 0..=N_GAP {
            #[allow(clippy::cast_precision_loss)]
            let f = i as f64 / N_GAP as f64;
            let t = t0 + (t1 - t0) * f;
            let pt = edge.curve().evaluate_with_endpoints(t, sp, ep);
            if let Some((u, _)) = surface.project_point(pt) {
                u_samples.push(u);
            }
        }
    }
    crate::classifier::largest_u_gap(&u_samples)
}

/// Minimum distance from a 2D point to a closed polygon's edges.
fn point_to_polygon_dist(p: brepkit_math::vec::Point2, poly: &[brepkit_math::vec::Point2]) -> f64 {
    let n = poly.len();
    let mut best = f64::MAX;
    for i in 0..n {
        let (a, b) = (poly[i], poly[(i + 1) % n]);
        let (abx, aby) = (b.x() - a.x(), b.y() - a.y());
        let (apx, apy) = (p.x() - a.x(), p.y() - a.y());
        let len2 = abx * abx + aby * aby;
        let t = if len2 > 1e-20 {
            ((apx * abx + apy * aby) / len2).clamp(0.0, 1.0)
        } else {
            0.0
        };
        let (cx, cy) = (a.x() + abx * t, a.y() + aby * t);
        best = best.min(((p.x() - cx).powi(2) + (p.y() - cy).powi(2)).sqrt());
    }
    best
}

/// Restrict surface-surface intersection curves to the region inside BOTH
/// faces. `compute_raw_curves` works on the unbounded surfaces, so a
/// plane-analytic or algebraic curve (e.g. a cylinder/tilted-plane ellipse)
/// can reach far past the faces' shared region and slit the partner face's
/// wire. A non-Line curve whose longest contiguous in-both run spans fewer than
/// two sample segments (at most two consecutive in-both samples out of `N+1`)
/// only grazes the mutual extent at a tangency/point — it never splits either
/// face, so it is dropped. Curves with a real in-both span are kept whole (the
/// downstream splitter trims them to the boundary). Lines are left to the
/// downstream `clip_line_to_face`. Conservative: if either face's extent cannot
/// be built, the curves are returned unchanged.
#[allow(clippy::too_many_arguments, clippy::items_after_statements)]
fn restrict_curves_to_faces(
    topo: &Topology,
    fa: FaceId,
    fb: FaceId,
    surf_a: &FaceSurface,
    surf_b: &FaceSurface,
    v_range_a: Option<(f64, f64)>,
    v_range_b: Option<(f64, f64)>,
    raw_curves: Vec<RawCurve>,
    tol: Tolerance,
) -> Vec<RawCurve> {
    let (Some(ext_a), Some(ext_b)) = (
        FaceExtent::new(topo, fa, surf_a, v_range_a, tol),
        FaceExtent::new(topo, fb, surf_b, v_range_b, tol),
    ) else {
        // Conservative: if either face's extent can't be built, don't restrict.
        return raw_curves;
    };

    const N: usize = 24;
    let mut out = Vec::with_capacity(raw_curves.len());
    for raw in raw_curves {
        // Lines are clipped downstream by `clip_line_to_face`; only the
        // unbounded analytic curves (ellipse/circle/marched NURBS) need the
        // mutual-extent test here.
        if matches!(raw.curve, EdgeCurve::Line) {
            out.push(raw);
            continue;
        }
        // Torus × box-plane: a plane×torus oval is clipped to its EXACT in-box
        // arc at the box-edge∩torus crossings (shared with the adjacent box
        // wall, so the notch is watertight). Gated to a closed NURBS oval whose
        // partner is a planar face with straight (Line) boundary edges; defers
        // (None) otherwise, so all other sections keep the sample-clip below.
        if let Some(arcs) =
            trim_torus_oval_to_box_face(topo, fa, fb, surf_a, surf_b, &raw, &ext_a, &ext_b, tol)
        {
            out.extend(arcs);
            continue;
        }

        // OPEN marched-NURBS conic (plane×cone hyperbola/parabola from the
        // `Points` fit): clip it to EXACT crossings with the plane face's
        // straight boundary edges. The generic sample-clip below keeps open
        // curves whole "for the downstream splitter to trim" — but the
        // splitter never clips an open curved section to a plane face's
        // boundary, so a conic spanning the whole cone extent leaves the face
        // unsplit (the dovetail tongue-relief family: tip/flank faces never
        // partition, the whole face classifies by one interior point, and the
        // cut collapses to an open hole shell). Exact endpoints matter: they
        // land ON boundary edges within tolerance so
        // `split_boundary_edges_at_3d_points` anchors them, and the same
        // crossing points chain with the adjacent faces' sections (a point on
        // the shared edge and on the cone lies on BOTH faces' conics).
        if let Some(pieces) =
            trim_open_curve_to_plane_face_lines(topo, fa, surf_a, surf_b, &raw, &ext_a, &ext_b, tol)
        {
            out.extend(pieces);
            continue;
        }
        if let Some(pieces) =
            trim_open_curve_to_plane_face_lines(topo, fb, surf_b, surf_a, &raw, &ext_b, &ext_a, tol)
        {
            out.extend(pieces);
            continue;
        }

        let pt = |i: usize| -> Point3 {
            #[allow(clippy::cast_precision_loss)]
            let f = i as f64 / N as f64;
            let t = raw.t_range.0 + (raw.t_range.1 - raw.t_range.0) * f;
            raw.curve.evaluate_with_endpoints(t, raw.p_start, raw.p_end)
        };
        let inb: Vec<bool> = (0..=N)
            .map(|i| {
                let p = pt(i);
                ext_a.contains(p) && ext_b.contains(p)
            })
            .collect();
        // Longest contiguous in-both run. A closed curve (sample N coincides
        // with sample 0) may have its in-both arc wrap across the seam, so the
        // search extends past N for closed curves (`b1` may exceed N, mapped
        // back via the curve's periodic parameterization).
        let closed = (raw.p_start - raw.p_end).length() < 1e-7;
        let (b0, b1) = longest_inboth_run(&inb, closed);
        // An in-both run spanning fewer than two segments (b1-b0 < 2, i.e. at
        // most two consecutive in-both samples) is a tangency/grazing point —
        // such a curve never splits either face, so drop it.
        //
        // The coarse test cannot distinguish a graze from a REAL crossing much
        // shorter than the curve: a socket-mouth corner circle crossing a 2 mm
        // dovetail tongue face subtends ~8° of the circle — under one 24-sample
        // segment — yet properly splits the tongue face (dropping it gaps the
        // section chain at the corner, the whole top face classifies by a
        // single interior point, and the cut collapses to an open hole shell).
        // Before declaring a graze, refine with a sample density scaled to the
        // SMALLER face extent. A true point tangency stays sub-segment at any
        // resolution and is still dropped.
        if b1 - b0 < 2 {
            let approx_len: f64 = (0..N).map(|i| (pt(i + 1) - pt(i)).length()).sum();
            let min_dim = ext_a
                .min_dimension()
                .min(ext_b.min_dimension())
                .max(tol.linear * 1e3);
            #[allow(clippy::cast_possible_truncation, clippy::cast_sign_loss)]
            let n_fine = ((8.0 * approx_len / min_dim).ceil() as usize).clamp(N, 1024);
            if n_fine <= N {
                out.extend(rescue_corner_crossing(
                    topo, fa, fb, &raw, &ext_a, &ext_b, &inb, tol,
                ));
                continue;
            }
            let ptf = |i: usize| -> Point3 {
                #[allow(clippy::cast_precision_loss)]
                let f = i as f64 / n_fine as f64;
                let t = raw.t_range.0 + (raw.t_range.1 - raw.t_range.0) * f;
                raw.curve.evaluate_with_endpoints(t, raw.p_start, raw.p_end)
            };
            let inb_fine: Vec<bool> = (0..=n_fine)
                .map(|i| {
                    let p = ptf(i);
                    ext_a.contains(p) && ext_b.contains(p)
                })
                .collect();
            let (f0, f1) = longest_inboth_run(&inb_fine, closed);
            if f1 - f0 < 2 {
                out.extend(rescue_corner_crossing(
                    topo, fa, fb, &raw, &ext_a, &ext_b, &inb_fine, tol,
                ));
                continue;
            }
            if closed && f1 - f0 < n_fine && !matches!(raw.curve, EdgeCurve::Circle(_)) {
                out.extend(trim_closed_curve_to_inboth_arc(&raw, f0, f1, n_fine));
                continue;
            }
            out.push(raw);
            continue;
        }
        // A closed ellipse/NURBS loop whose in-both run covers the WHOLE curve
        // (`b1 - b0 == N`) is a genuinely fully-shared seam — e.g. the closed
        // ellipse where two equal perpendicular cylinders cross, every sample of
        // which lies on both lateral faces. Keep it WHOLE: the downstream
        // splitter routes a closed interior loop to
        // `split_face_with_internal_loops` (carving cap + band-with-hole). Only
        // a PARTIAL in-both run (`b1 - b0 < N`, a real out-of-face arc) is
        // trimmed to its in-both arc below: keeping a partial whole would leave a
        // spurious closed self-loop section edge (the splitter only trims OPEN
        // curves and only adopts closed CIRCLES via seam adoption), so a full
        // ellipse from an inner tapered wall meeting an outer corner (the
        // gridfinity lip knife-edge) would survive as a degenerate loop and
        // over-connect the rim. Circles are left whole (seam adoption handles
        // them); open curves are left whole (the splitter clips them).
        if closed && b1 - b0 < N && !matches!(raw.curve, EdgeCurve::Circle(_)) {
            out.extend(trim_closed_curve_to_inboth_arc(&raw, b0, b1, N));
            continue;
        }
        out.push(raw);
    }
    out
}

/// Trim a plane×torus oval to its EXACT in-box arc at the box-edge∩torus
/// crossings (the analogue of box∩sphere's `emit_split_circle_arcs`, for a
/// marched torus oval instead of a `Circle`).
///
/// Gated: fires only when one face is a `Torus`, the partner is a `Plane` whose
/// outer wire is all straight `Line` edges (a box wall), and `raw` is a CLOSED
/// non-`Line` oval. Returns:
/// - `Some(arc)` — the single in-box arc, with its endpoints SNAPPED to the
///   exact box-edge∩torus crossings so the adjacent box wall (trimmed against
///   the same edge) shares those vertices and the notch is watertight.
/// - `Some(vec![raw])` — the oval lies wholly inside the box face (no boundary
///   crossing): keep it whole.
/// - `None` — not this case; the caller's sample-clip handles it.
///
/// The kept arc is the one whose midpoint is inside the box-face polygon (the
/// outer arc that wraps most of the tube), not the inner fragment a contiguous
/// sample run would wrongly keep on a mirror-asymmetric marcher parameterisation.
#[allow(clippy::too_many_arguments)]
fn trim_torus_oval_to_box_face(
    topo: &Topology,
    fa: FaceId,
    fb: FaceId,
    surf_a: &FaceSurface,
    surf_b: &FaceSurface,
    raw: &RawCurve,
    ext_a: &FaceExtent,
    ext_b: &FaceExtent,
    tol: Tolerance,
) -> Option<Vec<RawCurve>> {
    // Identify the torus face/surface and the box (plane) face; the oval must be
    // a closed non-Line curve.
    if matches!(raw.curve, EdgeCurve::Line) {
        return None;
    }
    if (raw.p_start - raw.p_end).length() > 1e-6 {
        return None; // open curve — not a closed oval
    }
    let (torus, plane_face, plane_ext) = match (surf_a, surf_b) {
        (FaceSurface::Torus(t), FaceSurface::Plane { .. }) => (t, fb, ext_b),
        (FaceSurface::Plane { .. }, FaceSurface::Torus(t)) => (t, fa, ext_a),
        _ => return None,
    };
    let _ = ext_a;
    let _ = ext_b;

    // The box face's straight boundary edges (its rectangle sides).
    let face = topo.face(plane_face).ok()?;
    let wire = topo.wire(face.outer_wire()).ok()?;
    let mut box_edges: Vec<(Point3, Point3)> = Vec::new();
    for oe in wire.edges() {
        let e = topo.edge(oe.edge()).ok()?;
        if !matches!(e.curve(), EdgeCurve::Line) {
            return None; // not a straight-edged wall — defer
        }
        let s = topo.vertex(e.start()).ok()?.point();
        let en = topo.vertex(e.end()).ok()?.point();
        box_edges.push((s, en));
    }
    if box_edges.len() < 3 {
        return None;
    }

    // Exact box-edge ∩ torus crossings that lie ON the edge segment AND ON the
    // oval. The crossing point is EXACT; `on_oval_tol` only has to confirm the
    // crossing belongs to THIS oval (vs a different oval branch on the same
    // plane, which is ≳1 mm away) — so it must exceed the MARCHED oval's
    // approximation error (~0.1 mm), well below the inter-branch separation.
    let on_oval_tol = 0.3_f64;
    let dedup_tol = tol.linear * 100.0;
    let mut crossings: Vec<Point3> = Vec::new();
    for &(s, en) in &box_edges {
        let dir = en - s;
        let len = dir.length();
        if len < tol.linear {
            continue;
        }
        for t in analytic_intersection::intersect_line_torus(torus, s, dir) {
            if !(-1e-6..=1.0 + 1e-6).contains(&t) {
                continue; // off the edge segment
            }
            let p = s + dir * t;
            // Must lie on the oval (the marched NURBS), else it's a crossing of
            // a DIFFERENT oval branch on this plane.
            let mut min_d = f64::MAX;
            for i in 0..=128 {
                let tt = raw.t_range.0 + (raw.t_range.1 - raw.t_range.0) * f64::from(i) / 128.0;
                min_d = min_d.min(
                    (raw.curve
                        .evaluate_with_endpoints(tt, raw.p_start, raw.p_end)
                        - p)
                        .length(),
                );
            }
            let on_oval = min_d < on_oval_tol;
            if on_oval && !crossings.iter().any(|c| (*c - p).length() < dedup_tol) {
                crossings.push(p);
            }
        }
    }

    // No boundary crossing: the oval is wholly inside (or outside) the box face.
    if crossings.is_empty() {
        // Inside → keep whole; outside → the caller's sample-clip drops it.
        let mid = raw
            .curve
            .evaluate_with_endpoints(0.5, raw.p_start, raw.p_end);
        return if plane_ext.contains(mid) {
            Some(vec![raw.clone()])
        } else {
            None
        };
    }
    // A single tangential crossing never splits the oval into a kept arc.
    if crossings.len() < 2 {
        return None;
    }

    // Find each crossing's parameter on the closed oval (densest sample).
    let n_dense = 1024usize;
    let oval_at = |frac: f64| -> Point3 {
        let tt = raw.t_range.0 + (raw.t_range.1 - raw.t_range.0) * frac;
        raw.curve
            .evaluate_with_endpoints(tt, raw.p_start, raw.p_end)
    };
    let frac_of = |p: Point3| -> f64 {
        let mut best_f = 0.0;
        let mut best_d = f64::MAX;
        for i in 0..=n_dense {
            #[allow(clippy::cast_precision_loss)]
            let f = i as f64 / n_dense as f64;
            let d = (oval_at(f) - p).length();
            if d < best_d {
                best_d = d;
                best_f = f;
            }
        }
        best_f
    };
    let mut marks: Vec<(f64, Point3)> = crossings.iter().map(|&p| (frac_of(p), p)).collect();
    marks.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap_or(std::cmp::Ordering::Equal));
    if marks.len() < 2 {
        return None;
    }

    // For each arc between consecutive crossings (wrapping the closed oval), the
    // kept arc has its interior midpoint inside the box face. Re-sample that arc
    // (walking the closed oval the correct way) and fit a FRESH NURBS whose
    // endpoints are the EXACT box-edge crossings — so the adjacent box wall
    // shares those vertices. Re-sampling avoids out-of-domain clamped-NURBS
    // evaluation on a wrapping arc.
    let sample_arc = |f0: f64, f1: f64| -> Vec<Point3> {
        // Walk f0 -> f1 forward on the closed oval (wrapping past 1.0).
        let span = if f1 >= f0 { f1 - f0 } else { f1 + 1.0 - f0 };
        let steps = 48usize;
        (0..=steps)
            .map(|i| {
                #[allow(clippy::cast_precision_loss)]
                let f = (f0 + span * (i as f64 / steps as f64)).rem_euclid(1.0);
                oval_at(f)
            })
            .collect()
    };
    // Collect ALL in-box arcs (the oval may cross the box face's boundary into
    // MORE than one in-box interval — e.g. a plane cut that enters the box face
    // twice — and each is a valid section; keeping only the longest would drop
    // the others and leave incomplete walls). Each arc spans between two
    // consecutive boundary crossings whose mid-arc point is inside the box face.
    let mut arc_point_sets: Vec<Vec<Point3>> = Vec::new();
    for i in 0..marks.len() {
        let (f0, p0) = marks[i];
        let (f1, p1) = marks[(i + 1) % marks.len()];
        let mut pts = sample_arc(f0, f1);
        if pts.len() < 4 {
            continue;
        }
        if !plane_ext.contains(pts[pts.len() / 2]) {
            continue;
        }
        let last = pts.len() - 1;
        pts[0] = p0;
        pts[last] = p1;
        arc_point_sets.push(pts);
    }
    if arc_point_sets.is_empty() {
        return None;
    }

    // Emit each kept in-box arc ALWAYS SPLIT at its midpoint into two sub-arcs
    // with a distinct middle vertex. Each kept arc shares BOTH its endpoints (the
    // box-edge∩torus crossings) with the partner box-wall edge that closes the
    // same notch-wall lens — a straight box Line for the inner (x=6) wall, the
    // partner arc for the y-walls. The endpoint-pair-keyed `merge_duplicate_edges`
    // (unchanged, gridfinity-load-bearing) would collapse any two co-endpoint
    // edges, degenerating the wall into a one-edge face. The midpoint vertex
    // breaks that: no two edges of the lens then share BOTH endpoints. The split
    // arc is emitted as ONE shared FF section, so BOTH consumers — the kept
    // toroidal band and the box-wall sub-face — see the same midpoint vertex and
    // stay watertight. Geometry is unchanged (the two halves retrace the arc).
    let fit = |seg: &[Point3]| -> Option<RawCurve> {
        let curve = brepkit_math::nurbs::fitting::interpolate(seg, 3.min(seg.len() - 1)).ok()?;
        let dom = curve.domain();
        let bbox = Aabb3::try_from_points(seg.iter().copied())?;
        Some(RawCurve {
            curve: EdgeCurve::NurbsCurve(curve),
            bbox,
            t_range: dom,
            p_start: seg[0],
            p_end: seg[seg.len() - 1],
        })
    };
    let mut out_arcs = Vec::new();
    for pts in &arc_point_sets {
        if pts.len() >= 7 {
            let mid = pts.len() / 2;
            if let (Some(a0), Some(a1)) = (fit(&pts[..=mid]), fit(&pts[mid..])) {
                out_arcs.push(a0);
                out_arcs.push(a1);
            } else if let Some(a) = fit(pts) {
                out_arcs.push(a);
            }
        } else if let Some(a) = fit(pts) {
            out_arcs.push(a);
        }
    }
    if out_arcs.is_empty() {
        return None;
    }
    Some(out_arcs)
}

/// Rescue a real-but-tiny corner crossing from the graze-drop.
///
/// The extent-scaled refinement above assumes the smallest real crossing spans
/// a fraction of the smaller face's extent — but an open section curve that
/// clips just a CORNER of the mutual face window (entering through one
/// boundary and exiting through an adjacent one, e.g. a parallel-axis
/// cone×cylinder branch leaving through the cone patch's angular window edge)
/// has an in-both run arbitrarily smaller than either face. Dropping it gaps
/// the section chain exactly where the neighbouring faces' sections terminate,
/// so the wire builder backtracks into a zero-area slit (the lite magnet-pad
/// fuse family).
///
/// Bisects the in/out transitions around the sampled in-both run to the exact
/// window crossings and emits the trimmed sub-span. Two gates keep true
/// tangency grazes dropped: the piece midpoint must sit strictly INSIDE both
/// faces' windows (no boundary-margin credit — a graze only ever rides the
/// margin band), and the piece must have real arc length.
#[allow(clippy::too_many_arguments)]
fn rescue_corner_crossing(
    topo: &Topology,
    fa: FaceId,
    fb: FaceId,
    raw: &RawCurve,
    ext_a: &FaceExtent,
    ext_b: &FaceExtent,
    inb: &[bool],
    tol: Tolerance,
) -> Option<RawCurve> {
    // Open curves only: a closed near-tangent loop keeps the historical drop
    // (the seam-adoption / internal-loop paths own closed sections).
    if (raw.p_start - raw.p_end).length() < 1e-7 {
        return None;
    }
    let n = inb.len() - 1;
    let (b0, b1) = longest_inboth_run(inb, false);
    if !inb[b0] {
        return None; // no in-both sample at all
    }
    let span = raw.t_range.1 - raw.t_range.0;
    #[allow(clippy::cast_precision_loss)]
    let t_at = |i: usize| raw.t_range.0 + span * (i as f64) / (n as f64);
    let point_at = |t: f64| raw.curve.evaluate_with_endpoints(t, raw.p_start, raw.p_end);
    let inside = |t: f64| {
        let p = point_at(t);
        ext_a.contains(p) && ext_b.contains(p)
    };
    let mut t_lo = t_at(b0);
    if b0 > 0 {
        let (mut out_t, mut in_t) = (t_at(b0 - 1), t_lo);
        for _ in 0..48 {
            let mid = 0.5 * (out_t + in_t);
            if inside(mid) {
                in_t = mid;
            } else {
                out_t = mid;
            }
        }
        t_lo = in_t;
    }
    let mut t_hi = t_at(b1);
    if b1 < n {
        let (mut in_t, mut out_t) = (t_hi, t_at(b1 + 1));
        for _ in 0..48 {
            let mid = 0.5 * (in_t + out_t);
            if inside(mid) {
                in_t = mid;
            } else {
                out_t = mid;
            }
        }
        t_hi = in_t;
    }
    if t_hi <= t_lo {
        return None;
    }
    let depth = tol.linear * 100.0;
    let mid = point_at(0.5 * (t_lo + t_hi));
    if !(ext_a.contains_strict(mid, depth) && ext_b.contains_strict(mid, depth)) {
        return None;
    }
    let mut len = 0.0;
    let mut prev = point_at(t_lo);
    for k in 1..=16 {
        #[allow(clippy::cast_precision_loss)]
        let p = point_at(t_lo + (t_hi - t_lo) * f64::from(k) / 16.0);
        len += (p - prev).length();
        prev = p;
    }
    if len < depth {
        return None;
    }
    // Snap each endpoint onto the nearest face-boundary curve within the weld
    // band. The trimmed curve is fitted geometry (~1e-6 off the exact
    // surfaces), while the boundary splitters gate candidates at the exact
    // 1e-7 tolerance — the recurring fit-error-vs-exact-gate class. Adopting
    // the boundary FOOT gives both faces the identical on-boundary junction.
    let surf_of = |fid: FaceId| topo.face(fid).ok().map(|f| f.surface().clone());
    let snap = |p: Point3| -> Point3 {
        const NS: usize = 64;
        let weld = tol.linear * 100.0;
        // (distance, foot, foot's curve param, owning face, edge)
        let mut best: Option<(f64, Point3, f64, FaceId, brepkit_topology::edge::EdgeId)> = None;
        for fid in [fa, fb] {
            let Ok(face) = topo.face(fid) else { continue };
            // Inner (hole) wires are legitimate exit boundaries too: a corner
            // crossing leaving through a bore rim must share its exact vertex.
            let wire_ids: Vec<_> = std::iter::once(face.outer_wire())
                .chain(face.inner_wires().iter().copied())
                .collect();
            for oe in wire_ids
                .iter()
                .filter_map(|&wid| topo.wire(wid).ok())
                .flat_map(|w| w.edges().to_vec())
            {
                let Ok(edge) = topo.edge(oe.edge()) else {
                    continue;
                };
                let (Ok(sv), Ok(ev)) = (topo.vertex(edge.start()), topo.vertex(edge.end())) else {
                    continue;
                };
                let (sp, ep) = (sv.point(), ev.point());
                let (d0, d1) = edge.curve().domain_with_endpoints(sp, ep);
                let mut best_k = 0;
                let mut best_d = f64::MAX;
                for k in 0..=NS {
                    #[allow(clippy::cast_precision_loss)]
                    let t = d0 + (d1 - d0) * (k as f64) / (NS as f64);
                    let d = (edge.curve().evaluate_with_endpoints(t, sp, ep) - p).length();
                    if d < best_d {
                        best_d = d;
                        best_k = k;
                    }
                }
                #[allow(clippy::cast_precision_loss)]
                let mut lo = d0 + (d1 - d0) * (best_k.saturating_sub(1) as f64) / (NS as f64);
                #[allow(clippy::cast_precision_loss)]
                let mut hi = d0 + (d1 - d0) * ((best_k + 1).min(NS) as f64) / (NS as f64);
                for _ in 0..48 {
                    let m1 = lo + (hi - lo) / 3.0;
                    let m2 = hi - (hi - lo) / 3.0;
                    let f1 = (edge.curve().evaluate_with_endpoints(m1, sp, ep) - p).length();
                    let f2 = (edge.curve().evaluate_with_endpoints(m2, sp, ep) - p).length();
                    if f1 < f2 {
                        hi = m2;
                    } else {
                        lo = m1;
                    }
                }
                let tm = 0.5 * (lo + hi);
                let foot = edge.curve().evaluate_with_endpoints(tm, sp, ep);
                let d = (foot - p).length();
                if d < weld && best.as_ref().is_none_or(|b| d < b.0) {
                    best = Some((d, foot, tm, fid, oe.edge()));
                }
            }
        }
        let Some((_, foot, tm, owner_fid, eid)) = best else {
            return p;
        };
        // The foot lies ON the boundary curve but is displaced along it by the
        // section's fit error. The true junction is where that boundary curve
        // meets the PARTNER face's surface — refine to it so every section
        // ending here (this one, and the partner-pair sections computed
        // independently) mints the SAME vertex.
        let other = if owner_fid == fa { fb } else { fa };
        let (Some(other_surf), Ok(edge)) = (surf_of(other), topo.edge(eid)) else {
            return foot;
        };
        let (Ok(sv), Ok(ev)) = (topo.vertex(edge.start()), topo.vertex(edge.end())) else {
            return foot;
        };
        let (sp, ep) = (sv.point(), ev.point());
        let dist_to_surf = |q: Point3| -> f64 {
            other_surf
                .project_point(q)
                .and_then(|(u, v)| other_surf.evaluate(u, v))
                .map_or(f64::MAX, |s| (s - q).length())
        };
        let (d0, d1) = edge.curve().domain_with_endpoints(sp, ep);
        let half = (d1 - d0).abs().max(1e-9) * 0.01;
        // Clamp the refine window to the edge's own span so the search never
        // evaluates (and snaps to) a point past the boundary arc's ends.
        let (span_lo, span_hi) = (d0.min(d1), d0.max(d1));
        let (mut lo, mut hi) = ((tm - half).max(span_lo), (tm + half).min(span_hi));
        for _ in 0..64 {
            let m1 = lo + (hi - lo) / 3.0;
            let m2 = hi - (hi - lo) / 3.0;
            let f1 = dist_to_surf(edge.curve().evaluate_with_endpoints(m1, sp, ep));
            let f2 = dist_to_surf(edge.curve().evaluate_with_endpoints(m2, sp, ep));
            if f1 < f2 {
                hi = m2;
            } else {
                lo = m1;
            }
        }
        let tj = 0.5 * (lo + hi);
        let junction = edge.curve().evaluate_with_endpoints(tj, sp, ep);
        if dist_to_surf(junction) <= tol.linear * 10.0 && (junction - foot).length() <= weld {
            junction
        } else {
            foot
        }
    };
    let p_start = snap(point_at(t_lo));
    let p_end = snap(point_at(t_hi));
    Some(RawCurve {
        curve: raw.curve.clone(),
        bbox: raw.bbox,
        t_range: (t_lo, t_hi),
        p_start,
        p_end,
    })
}

/// Trim a CLOSED section curve (Ellipse or NURBS) to its in-both arc
/// `[b0, b1]` of `N` samples, where `b1` may exceed `N` for a run that WRAPS the
/// periodic seam (the closed curve's sample `N` coincides with sample `0`).
///
/// - **Periodic curve (Ellipse):** emit ONE open arc with the unwrapped
///   parameters `[t0, t1]` (where `t1` may be past the domain end). The curve's
///   `cos/sin` evaluation handles out-of-domain parameters, so a single arc
///   spans the wrapping run correctly.
/// - **NURBS curve:** a clamped NURBS evaluates an out-of-domain parameter to a
///   garbage point, so a wrapping run cannot be one arc. Emit TWO in-domain
///   arcs — `[t0, domain_end]` and `[domain_start, t1 − span]` — which meet at
///   the seam (where the closed curve's endpoints coincide), preserving BOTH
///   pieces of the wrapping run instead of dropping the head. A non-wrapping
///   NURBS run (`b1 ≤ N`) is a single in-domain arc.
fn trim_closed_curve_to_inboth_arc(
    raw: &RawCurve,
    b0: usize,
    b1: usize,
    n: usize,
) -> Vec<RawCurve> {
    #[allow(clippy::cast_precision_loss)]
    let frac = |i: usize| i as f64 / n as f64;
    let span = raw.t_range.1 - raw.t_range.0;
    let t0 = raw.t_range.0 + span * frac(b0);
    let t1 = raw.t_range.0 + span * frac(b1);
    let point_at = |t: f64| raw.curve.evaluate_with_endpoints(t, raw.p_start, raw.p_end);

    let one_arc = |ta: f64, tb: f64| RawCurve {
        curve: raw.curve.clone(),
        bbox: raw.bbox,
        t_range: (ta, tb),
        p_start: point_at(ta),
        p_end: point_at(tb),
    };

    let wraps = b1 > n; // the in-both run crosses the periodic seam.
    if matches!(raw.curve, EdgeCurve::NurbsCurve(_)) && wraps {
        // Split at the domain end / start so neither arc leaves the domain.
        let t1_wrapped = t1 - span; // = raw.t_range.0 + span * frac(b1 - n)
        vec![
            one_arc(t0, raw.t_range.1),
            one_arc(raw.t_range.0, t1_wrapped),
        ]
    } else {
        // Periodic curve (any run) or non-wrapping NURBS: a single arc. For a
        // periodic curve `t1` may exceed the domain — that is intentional and
        // evaluates correctly.
        vec![one_arc(t0, t1)]
    }
}

/// Trim a closed `Ellipse` section (the intersection of a thin planar tread
/// with a cylinder/cone lateral face) to its in-both arc(s) using the EXACT
/// points where the planar face's straight boundary edges cross the analytic
/// surface, rather than uniform-t sampling.
///
/// Returns `Some(arcs)` only when the pair is exactly {planar face with
/// straight boundary edges} × {cylinder or cone}, the curve is a closed
/// ellipse, and at least one in-both arc with a real angular span is found.
/// Returns `None` for any other configuration so the caller falls back to the
/// uniform-t restriction. The exact crossings are SHARED between treads that
/// share a boundary line, so consecutive arcs chain through one vertex (the
/// downstream `find_nearby_pave_vertex` snaps them at `tol.linear`).
#[allow(clippy::too_many_arguments)]
fn trim_ellipse_to_boundary_crossings(
    topo: &Topology,
    fa: FaceId,
    fb: FaceId,
    surf_a: &FaceSurface,
    surf_b: &FaceSurface,
    raw: &RawCurve,
    ext_a: &FaceExtent,
    ext_b: &FaceExtent,
) -> Option<Vec<RawCurve>> {
    use brepkit_math::curves::{Circle3D, Ellipse3D};

    // The raw plane×analytic section: a tilted plane yields an Ellipse, a
    // plane perpendicular to the axis yields an exact Circle. Both share the
    // same parameterization surface (project/evaluate by angle).
    enum SecCurve<'a> {
        Ell(&'a Ellipse3D),
        Circ(&'a Circle3D),
    }
    impl SecCurve<'_> {
        fn project(&self, p: Point3) -> f64 {
            match self {
                SecCurve::Ell(e) => e.project(p),
                SecCurve::Circ(c) => c.project(p),
            }
        }
        fn evaluate(&self, t: f64) -> Point3 {
            match self {
                SecCurve::Ell(e) => e.evaluate(t),
                SecCurve::Circ(c) => c.evaluate(t),
            }
        }
        fn edge_curve(&self) -> EdgeCurve {
            match self {
                SecCurve::Ell(e) => EdgeCurve::Ellipse((*e).clone()),
                SecCurve::Circ(c) => EdgeCurve::Circle((*c).clone()),
            }
        }
    }
    let sec = match &raw.curve {
        EdgeCurve::Ellipse(e) => SecCurve::Ell(e),
        EdgeCurve::Circle(c) => SecCurve::Circ(c),
        _ => return None,
    };
    // Must be a closed full section (the raw plane×analytic intersection).
    if (raw.p_start - raw.p_end).length() > 1e-7 {
        return None;
    }

    // Identify the planar face and the analytic (cylinder/cone) face.
    let (plane_face, plane_surf, analytic_face, analytic_surf) = match (surf_a, surf_b) {
        (FaceSurface::Plane { .. }, FaceSurface::Cylinder(_) | FaceSurface::Cone(_)) => {
            (fa, surf_a, fb, surf_b)
        }
        (FaceSurface::Cylinder(_) | FaceSurface::Cone(_), FaceSurface::Plane { .. }) => {
            (fb, surf_b, fa, surf_a)
        }
        _ => return None,
    };
    let FaceSurface::Plane {
        normal: plane_n,
        d: plane_d,
    } = plane_surf
    else {
        return None;
    };

    // Collect exact crossings of the planar face's straight boundary edges
    // with the analytic surface. Each crossing is a point lying on BOTH the
    // plane (it is a plane-boundary edge) and the analytic surface, hence on
    // the section ellipse.
    let face = topo.face(plane_face).ok()?;
    let mut crossings: Vec<Point3> = Vec::new();
    let push_crossing = |p: Point3, crossings: &mut Vec<Point3>| {
        // Keep only points actually on the section curve (rejects a cone's
        // far-nappe root or a seam-line crossing that misses the ellipse).
        let foot = sec.evaluate(sec.project(p));
        if (foot - p).length() > 1e-6 {
            return;
        }
        // Dedup tolerance: the SAME geometric crossing reached two ways (an
        // exact seam-line × plane intersection, and the line-cylinder quadratic
        // root for a tread boundary that meets the seam) can disagree by a
        // little over 1e-6 at these coordinates. A tighter threshold leaves
        // both, spawning a near-degenerate sliver arc whose drifted endpoint
        // then fails the downstream 1e-7 boundary split and dangles. Treads
        // are ~1e-4 apart, so 1e-5 collapses the duplicate without merging
        // genuinely-distinct crossings. Seam crossings are pushed first, so the
        // exact-on-seam point is the one kept.
        if !crossings.iter().any(|q| (*q - p).length() < 1e-5) {
            crossings.push(p);
        }
    };
    // Split at the analytic FACE's seam boundary FIRST: where the ellipse
    // crosses a seam (the quarter-cylinder's straight u-boundary edge), the
    // arc must terminate so it connects to that seam edge and the part beyond
    // the seam (outside the partial face) is excluded. Each seam line lies in
    // the cylinder surface at constant u; it crosses the tread plane at one
    // point. Collected first so the EXACT-on-seam point wins the dedup over a
    // tread-boundary crossing that coincides with the seam but carries ~1e-6
    // of line-cylinder rounding (else the seam-boundary split, which uses the
    // kernel's 1e-7 tolerance, misses it and the chain end dangles).
    if let Ok(aface) = topo.face(analytic_face) {
        for oe in topo.wire(aface.outer_wire()).ok()?.edges() {
            let Ok(edge) = topo.edge(oe.edge()) else {
                continue;
            };
            if !matches!(edge.curve(), EdgeCurve::Line) {
                continue;
            }
            let Ok(sv) = topo.vertex(edge.start()) else {
                continue;
            };
            let Ok(ev) = topo.vertex(edge.end()) else {
                continue;
            };
            if let Some(p) = line_segment_plane_crossing(sv.point(), ev.point(), *plane_n, *plane_d)
            {
                push_crossing(p, &mut crossings);
            }
        }
    }

    for oe in topo.wire(face.outer_wire()).ok()?.edges() {
        let edge = topo.edge(oe.edge()).ok()?;
        if !matches!(edge.curve(), EdgeCurve::Line) {
            // A non-straight boundary edge means this is not a faceted-ramp
            // tread; bail to the generic path rather than guess.
            return None;
        }
        let sp = topo.vertex(edge.start()).ok()?.point();
        let ep = topo.vertex(edge.end()).ok()?.point();
        for p in line_segment_surface_crossings(sp, ep, analytic_surf) {
            push_crossing(p, &mut crossings);
        }
    }

    // Need at least two crossings to bound an arc.
    if crossings.len() < 2 {
        return None;
    }

    // Map each crossing to its angular parameter, sort by angle.
    let mut t_pts: Vec<(f64, Point3)> = crossings
        .into_iter()
        .map(|p| (sec.project(p).rem_euclid(std::f64::consts::TAU), p))
        .collect();
    t_pts.sort_by(|a, b| a.0.total_cmp(&b.0));
    // Drop near-duplicate parameters (a crossing hit by two adjacent edges).
    t_pts.dedup_by(|a, b| (a.0 - b.0).abs() < 1e-9);
    if t_pts.len() < 2 {
        return None;
    }

    // Walk consecutive crossing pairs (including the wrap-around segment).
    // Emit an arc for each interval whose midpoint lies inside BOTH faces.
    let mut arcs = Vec::new();
    let m = t_pts.len();
    for i in 0..m {
        let (t0, p0) = t_pts[i];
        let next = (i + 1) % m;
        let (mut t1, p1) = t_pts[next];
        if t1 <= t0 {
            t1 += std::f64::consts::TAU;
        }
        let t_mid = 0.5 * (t0 + t1);
        let mid = sec.evaluate(t_mid);
        if !(ext_a.contains(mid) && ext_b.contains(mid)) {
            continue;
        }
        // Skip a degenerate sliver (the two crossings coincide angularly).
        if (t1 - t0) < 1e-6 {
            continue;
        }
        arcs.push(RawCurve {
            curve: sec.edge_curve(),
            bbox: raw.bbox,
            t_range: (t0, t1),
            p_start: p0,
            p_end: p1,
        });
    }

    if arcs.is_empty() { None } else { Some(arcs) }
}

/// Crossing of a line SEGMENT `[sp, ep]` with the plane `normal·p = d`.
/// Returns the point for the root parameter `s ∈ [0, 1]`, or `None` if the
/// segment is parallel to the plane or crosses outside `[0, 1]`.
fn line_segment_plane_crossing(sp: Point3, ep: Point3, normal: Vec3, d: f64) -> Option<Point3> {
    let dir = ep - sp;
    let denom = dir.dot(normal);
    if denom.abs() < 1e-15 {
        return None;
    }
    let sp_dot = sp.x() * normal.x() + sp.y() * normal.y() + sp.z() * normal.z();
    let s = (d - sp_dot) / denom;
    if !(-1e-9..=1.0 + 1e-9).contains(&s) {
        return None;
    }
    Some(sp + dir * s.clamp(0.0, 1.0))
}

/// Exact crossings of a line SEGMENT `[sp, ep]` with an analytic surface
/// (cylinder or cone lateral). Returns the 3D crossing points whose parameter
/// lies within the segment. Other surface types return an empty vec (the
/// caller treats "no crossings" as "fall back to uniform-t").
fn line_segment_surface_crossings(sp: Point3, ep: Point3, surface: &FaceSurface) -> Vec<Point3> {
    match surface {
        FaceSurface::Cylinder(cyl) => line_segment_cylinder_crossings(sp, ep, cyl),
        FaceSurface::Cone(cone) => line_segment_cone_crossings(sp, ep, cone),
        _ => Vec::new(),
    }
}

/// Solve the quadratic for where the segment `[sp, ep]` crosses an infinite
/// cylinder of `radius` about `axis` through `origin`. Returns crossing points
/// for roots `s ∈ [0, 1]`.
fn line_segment_cylinder_crossings(
    sp: Point3,
    ep: Point3,
    cyl: &brepkit_math::surfaces::CylindricalSurface,
) -> Vec<Point3> {
    let axis = cyl.axis();
    let o = cyl.origin();
    let r = cyl.radius();
    let d = ep - sp;
    let w = sp - o;
    // Component of each vector perpendicular to the axis.
    let perp = |v: Vec3| -> Vec3 { v - axis * axis.dot(v) };
    let dp = perp(d);
    let wp = perp(w);
    let a = dp.dot(dp);
    let b = 2.0 * wp.dot(dp);
    let c = wp.dot(wp) - r * r;
    solve_segment_quadratic(a, b, c, sp, d)
}

/// Crossings of the segment `[sp, ep]` with an infinite cone. With the
/// surface convention `P = apex + v(radial·cos a + axis·sin a)`, a point at
/// perpendicular radial distance `ρ` and axial distance `t = axis·(P-apex)`
/// lies on the cone when `ρ = t·cot(a)`, i.e. `ρ²·tan²(a) = t²`. Solved as a
/// quadratic in the segment parameter `s`.
fn line_segment_cone_crossings(
    sp: Point3,
    ep: Point3,
    cone: &brepkit_math::surfaces::ConicalSurface,
) -> Vec<Point3> {
    let apex = cone.apex();
    let axis = cone.axis();
    let tan_a = cone.half_angle().tan();
    let k = tan_a * tan_a;
    let d = ep - sp;
    let w = sp - apex;
    let perp = |v: Vec3| -> Vec3 { v - axis * axis.dot(v) };
    let dp = perp(d);
    let wp = perp(w);
    let da = axis.dot(d);
    let wa = axis.dot(w);
    // k·|perp(w + s d)|^2 = (axis·(w + s d))^2
    let a = k * dp.dot(dp) - da * da;
    let b = 2.0 * (k * wp.dot(dp) - da * wa);
    let c = k * wp.dot(wp) - wa * wa;
    solve_segment_quadratic(a, b, c, sp, d)
}

/// Solve `a s^2 + b s + c = 0` for `s ∈ [0, 1]` and return the 3D points
/// `sp + s·d`. Handles the near-linear (`a ≈ 0`) and no-real-root cases.
fn solve_segment_quadratic(a: f64, b: f64, c: f64, sp: Point3, d: Vec3) -> Vec<Point3> {
    let mut pts = Vec::new();
    let mut push_s = |s: f64| {
        if (-1e-9..=1.0 + 1e-9).contains(&s) {
            let sc = s.clamp(0.0, 1.0);
            pts.push(sp + d * sc);
        }
    };
    if a.abs() < 1e-15 {
        if b.abs() >= 1e-15 {
            push_s(-c / b);
        }
        return pts;
    }
    let disc = b * b - 4.0 * a * c;
    if disc < 0.0 {
        return pts;
    }
    let sq = disc.sqrt();
    push_s((-b - sq) / (2.0 * a));
    push_s((-b + sq) / (2.0 * a));
    pts
}

/// Clip an OPEN marched-NURBS section to its in-face span(s) at EXACT
/// crossings with a plane face's straight boundary edges.
///
/// Gated (returns `None`, deferring to the generic sample-clip, otherwise):
/// - `raw` is an open `NurbsCurve` (the plane×cone hyperbola/parabola fit;
///   lines are clipped by `clip_line_to_face`, circles/ellipses have exact
///   paths of their own),
/// - `plane_face`'s extent is planar and ALL its boundary edges (outer and
///   inner wires) are straight `Line` edges — the polygon then IS the exact
///   boundary,
/// - every sampled curve point lies in the face's plane (the section of this
///   pair genuinely lives on the plane face).
///
/// Each boundary crossing is found on the sampled 2D polyline, then refined by
/// bisecting the signed side-of-boundary-line function in curve parameter —
/// machine-precision endpoints ON the boundary edge, so
/// `split_boundary_edges_at_3d_points` anchors them and adjacent faces'
/// sections (which cross the same shared edges at the same 3D points) chain
/// into a closed loop. Kept spans are the intervals between consecutive
/// crossings whose midpoints lie inside the face polygon (outside its holes)
/// AND inside the partner face's extent.
#[allow(clippy::too_many_arguments, clippy::too_many_lines)]
fn trim_open_curve_to_plane_face_lines(
    topo: &Topology,
    plane_face: FaceId,
    plane_surf: &FaceSurface,
    other_surf: &FaceSurface,
    raw: &RawCurve,
    ext_plane: &FaceExtent,
    ext_other: &FaceExtent,
    tol: Tolerance,
) -> Option<Vec<RawCurve>> {
    use crate::builder::classify_2d::point_in_polygon_2d;
    use brepkit_math::vec::Point2;

    // Gated to CONE partners: the plane x cone conic (the `Points`-fit
    // hyperbola/parabola) is the configuration whose whole-curve sections
    // leave small plane faces unsplit (the dovetail tongue relief). Marched
    // sections against NURBS/cylinder partners stay on the generic path —
    // the honeycomb wall-cut weave is calibrated against those staying whole
    // (clipping them regressed its over-share pin).
    if !matches!(other_surf, FaceSurface::Cone(_)) {
        return None;
    }
    if !matches!(raw.curve, EdgeCurve::NurbsCurve(_)) {
        return None;
    }
    if (raw.p_start - raw.p_end).length() < 1e-7 {
        return None;
    }
    if !matches!(plane_surf, FaceSurface::Plane { .. }) {
        return None;
    }
    let FaceExtent::Plane {
        frame, poly, holes, ..
    } = ext_plane
    else {
        return None;
    };
    // Exact crossings are only computed against STRAIGHT boundary edges, so
    // the plane face's straight edges must dominate where the conic exits.
    // Curved boundary edges (a prior cut's rim arcs and conics — the second
    // relief cut of a compound-relieved tongue) are tolerated as long as the
    // kept pieces never need a crossing against them: their chords still
    // enter the sampled polygon for containment, and a piece straying into a
    // curved edge's chord/arc ambiguity is rejected below by the dense
    // in-polygon sample check (declining to the generic path as before).
    let face = topo.face(plane_face).ok()?;
    let mut has_curved_boundary = false;
    for wid in std::iter::once(face.outer_wire()).chain(face.inner_wires().iter().copied()) {
        let wire = topo.wire(wid).ok()?;
        for oe in wire.edges() {
            if !matches!(topo.edge(oe.edge()).ok()?.curve(), EdgeCurve::Line) {
                has_curved_boundary = true;
            }
        }
    }

    let n_samples = 64usize;
    let eval_at =
        |t: f64| -> Point3 { raw.curve.evaluate_with_endpoints(t, raw.p_start, raw.p_end) };
    // The section must lie in the plane (frame round-trip). Sampled check.
    let sample_t = |i: usize| -> f64 {
        #[allow(clippy::cast_precision_loss)]
        let f = i as f64 / n_samples as f64;
        raw.t_range.0 + (raw.t_range.1 - raw.t_range.0) * f
    };
    for i in 0..=n_samples {
        let p = eval_at(sample_t(i));
        let uv = frame.project(p);
        if (frame.evaluate(uv.x(), uv.y()) - p).length() > tol.linear * 10.0 {
            return None;
        }
    }

    // Signed side of a boundary segment's carrier line at curve parameter t.
    let side = |t: f64, a: Point2, b: Point2| -> f64 {
        let uv = frame.project(eval_at(t));
        (b.x() - a.x()).mul_add(uv.y() - a.y(), -((b.y() - a.y()) * (uv.x() - a.x())))
    };

    // Collect refined crossing parameters against every boundary segment
    // (outer polygon + hole polygons).
    let mut crossings: Vec<f64> = Vec::new();
    let mut scan_polygon = |ring: &[Point2]| {
        let m = ring.len();
        for j in 0..m {
            let (a, b) = (ring[j], ring[(j + 1) % m]);
            let seg_len = (b.x() - a.x()).hypot(b.y() - a.y());
            if seg_len < tol.linear {
                continue;
            }
            // A sample landing numerically ON the carrier line zeroes BOTH
            // adjacent products, so a pure sign-change test would skip the
            // crossing entirely; record such a sample as the crossing itself.
            let on_eps = seg_len * 1e-12;
            for i in 0..n_samples {
                let (t_lo, t_hi) = (sample_t(i), sample_t(i + 1));
                let (s_lo, s_hi) = (side(t_lo, a, b), side(t_hi, a, b));
                let t_star = if s_lo.abs() <= on_eps {
                    t_lo
                } else if s_hi.abs() <= on_eps {
                    t_hi
                } else if s_lo * s_hi > 0.0 {
                    continue;
                } else {
                    // Bisect the side function to the carrier-line crossing.
                    let (mut lo, mut hi, mut sl) = (t_lo, t_hi, s_lo);
                    for _ in 0..60 {
                        let mid = f64::midpoint(lo, hi);
                        let sm = side(mid, a, b);
                        if sm * sl < 0.0 {
                            hi = mid;
                        } else {
                            lo = mid;
                            sl = sm;
                        }
                    }
                    f64::midpoint(lo, hi)
                };
                let uv = frame.project(eval_at(t_star));
                // On the SEGMENT (not just its carrier line), with endpoint
                // slack — a crossing at a face corner belongs to both
                // adjacent segments.
                let w = ((uv.x() - a.x()) * (b.x() - a.x()) + (uv.y() - a.y()) * (b.y() - a.y()))
                    / (seg_len * seg_len);
                if !(-1e-6..=1.0 + 1e-6).contains(&w) {
                    continue;
                }
                if !crossings.iter().any(|&c| (c - t_star).abs() < 1e-9) {
                    crossings.push(t_star);
                }
            }
        }
    };
    scan_polygon(poly);
    for h in holes {
        scan_polygon(h);
    }

    // Crossings of the PARTNER (cone) face's angular window edges. A conic
    // through a rounded corner spans past the quarter face's seam ruling onto
    // the neighbouring wall — the out-of-window half is a section of the
    // unbounded cone only, and threading it splits a phantom lens off the
    // plane face (the dovetail tongue tip). Bisect the RAW wrapped angular
    // offset to the gap edge (NOT `u_in_gap`, whose 1e-6 rad margin would
    // land the endpoint microns off the seam and mint a fresh vertex instead
    // of snapping to the EF pave vertex the seam ruling already owns).
    if let FaceExtent::Analytic {
        surface: other_surface,
        u_gap: Some(gap),
        ..
    } = ext_other
    {
        let wrap = |d: f64| -> f64 {
            let w = d.rem_euclid(std::f64::consts::TAU);
            if w > std::f64::consts::PI {
                w - std::f64::consts::TAU
            } else {
                w
            }
        };
        let angle_to = |t: f64, g: f64| -> Option<f64> {
            other_surface
                .project_point(eval_at(t))
                .map(|(u, _)| wrap(u - g))
        };
        let ang_eps = 1e-12_f64;
        for &g in &[gap.0, gap.1] {
            for i in 0..n_samples {
                let (t_lo, t_hi) = (sample_t(i), sample_t(i + 1));
                let (Some(s_lo), Some(s_hi)) = (angle_to(t_lo, g), angle_to(t_hi, g)) else {
                    continue;
                };
                // Only local edge crossings: a sign flip π away is the wrap
                // seam of the offset function, not a window-edge crossing. A
                // sample numerically ON the window edge zeroes both adjacent
                // products, defeating the sign-change test — record it as the
                // crossing itself (its companion must still be local).
                let t_star = if s_lo.abs() <= ang_eps && s_hi.abs() <= 1.0 {
                    t_lo
                } else if s_hi.abs() <= ang_eps && s_lo.abs() <= 1.0 {
                    t_hi
                } else if s_lo * s_hi > 0.0 || s_lo.abs() > 1.0 || s_hi.abs() > 1.0 {
                    continue;
                } else {
                    let (mut lo, mut hi, mut sl) = (t_lo, t_hi, s_lo);
                    for _ in 0..60 {
                        let mid = f64::midpoint(lo, hi);
                        let Some(sm) = angle_to(mid, g) else { break };
                        if sm * sl < 0.0 {
                            hi = mid;
                        } else {
                            lo = mid;
                            sl = sm;
                        }
                    }
                    f64::midpoint(lo, hi)
                };
                if !crossings.iter().any(|&c| (c - t_star).abs() < 1e-9) {
                    crossings.push(t_star);
                }
            }
        }
    }

    // Crossings of the partner face's axial `v`-range rims. The conic is
    // clipped to the plane face's boundary above and to the cone's angular
    // window, but a flank conic can also EXIT through the patch's axial
    // extent (the rim circle at v0/v1) between those — the overshoot then
    // dangles past the rim, the splitter's pendant filter removes the whole
    // section chain, and the cone cap never splits out (the A1-corner
    // doubled-dovetail nub). Bisect v(t) to the exact rim crossing so the
    // kept piece ends ON the rim. The destructure matches any Analytic
    // extent, but this function's entry gate already restricted the partner
    // surface to a Cone, whose `v` is axial and non-periodic.
    if let FaceExtent::Analytic {
        surface: other_surface,
        v0,
        v1,
        ..
    } = ext_other
    {
        let v_of =
            |t: f64| -> Option<f64> { other_surface.project_point(eval_at(t)).map(|(_, v)| v) };
        let v_eps = 1e-12_f64;
        for &vb in &[*v0, *v1] {
            for i in 0..n_samples {
                let (t_lo, t_hi) = (sample_t(i), sample_t(i + 1));
                let (Some(w_lo), Some(w_hi)) = (v_of(t_lo), v_of(t_hi)) else {
                    continue;
                };
                let (s_lo, s_hi) = (w_lo - vb, w_hi - vb);
                let t_star = if s_lo.abs() <= v_eps {
                    t_lo
                } else if s_hi.abs() <= v_eps {
                    t_hi
                } else if s_lo * s_hi > 0.0 {
                    continue;
                } else {
                    let (mut lo, mut hi, mut sl) = (t_lo, t_hi, s_lo);
                    for _ in 0..60 {
                        let mid = f64::midpoint(lo, hi);
                        let Some(sm) = v_of(mid).map(|w| w - vb) else {
                            break;
                        };
                        if sm * sl < 0.0 {
                            hi = mid;
                        } else {
                            lo = mid;
                            sl = sm;
                        }
                    }
                    f64::midpoint(lo, hi)
                };
                if !crossings.iter().any(|&c| (c - t_star).abs() < 1e-9) {
                    crossings.push(t_star);
                }
            }
        }
    }

    // Whole curve inside (or outside) the face: no crossings — keep or drop
    // by a single interior test; deferring (None) would hand the generic
    // sample-clip a curve this path has already proven in-plane, so decide
    // here for consistency.
    let inside_face = |uv: Point2| -> bool {
        point_in_polygon_2d(uv, poly) && !holes.iter().any(|h| point_in_polygon_2d(uv, h))
    };
    let mut ts: Vec<f64> = Vec::with_capacity(crossings.len() + 2);
    ts.push(raw.t_range.0);
    crossings.sort_by(|x, y| x.partial_cmp(y).unwrap_or(std::cmp::Ordering::Equal));
    ts.extend(crossings);
    ts.push(raw.t_range.1);
    ts.dedup_by(|x, y| (*x - *y).abs() < 1e-9);

    let mut pieces = Vec::new();
    for w in ts.windows(2) {
        let (t0, t1) = (w[0], w[1]);
        let span_len = {
            let (p0, p1) = (eval_at(t0), eval_at(t1));
            (p1 - p0).length()
        };
        if span_len < tol.linear {
            continue;
        }
        let t_mid = f64::midpoint(t0, t1);
        let p_mid = eval_at(t_mid);
        if !inside_face(frame.project(p_mid)) || !ext_other.contains(p_mid) {
            continue;
        }
        let (p0, p1) = (eval_at(t0), eval_at(t1));
        let sub_pts: Vec<Point3> = (0..=8)
            .map(|k| eval_at(t0 + (t1 - t0) * (f64::from(k) / 8.0)))
            .collect();
        // With curved boundary edges present, a kept piece must stay strictly
        // inside the sampled polygon along its whole span: exact crossings
        // were only computed against straight segments, so a piece straying
        // out mid-span would have needed a crossing against a curved edge —
        // where the sampled chord is not the real boundary. Decline the whole
        // call (the pre-relaxation behavior) rather than emit it.
        if has_curved_boundary
            && sub_pts[1..8]
                .iter()
                .any(|p| !inside_face(frame.project(*p)))
        {
            return None;
        }
        let bbox = Aabb3::try_from_points(sub_pts)?;
        // Trim the stored NURBS geometry to the kept span. Downstream
        // consumers normalize over `domain_with_endpoints`, which for a NURBS
        // is the FULL knot domain regardless of endpoints — an untrimmed
        // piece would pcurve-fit, UV-project, and tessellate the WHOLE
        // marched conic (garbage UV endpoints on the cone face, a corrupt
        // refit pcurve on the plane face). `curve_split` preserves the
        // parameterization, so `t_range` stays `(t0, t1)` and the trimmed
        // curve's knot domain IS that span. A failed split (numerical, near a
        // knot) must NOT fall back to the untrimmed curve — that is exactly
        // the corrupt state described above — so defer the WHOLE curve to the
        // generic sample-clip instead.
        let piece_curve = match &raw.curve {
            EdgeCurve::NurbsCurve(n) => EdgeCurve::NurbsCurve(trim_nurbs_to_span(n, t0, t1)?),
            other => other.clone(),
        };
        pieces.push(RawCurve {
            curve: piece_curve,
            bbox: bbox.expanded(tol.linear),
            t_range: (t0, t1),
            p_start: p0,
            p_end: p1,
        });
    }
    Some(pieces)
}

/// Extract the `[t0, t1]` sub-curve of a NURBS curve, preserving the original
/// parameterization (the result's knot domain is exactly `[t0, t1]`).
fn trim_nurbs_to_span(
    n: &brepkit_math::nurbs::curve::NurbsCurve,
    t0: f64,
    t1: f64,
) -> Option<brepkit_math::nurbs::curve::NurbsCurve> {
    use brepkit_math::nurbs::knot_ops::curve_split;
    use brepkit_math::traits::ParametricCurve;
    let (d0, d1) = ParametricCurve::domain(n);
    let eps = (d1 - d0).abs() * 1e-9;
    let mut cur = n.clone();
    if t1 < d1 - eps {
        cur = curve_split(&cur, t1).ok()?.0;
    }
    if t0 > d0 + eps {
        cur = curve_split(&cur, t0).ok()?.1;
    }
    Some(cur)
}

/// Longest contiguous run of `true` in `inb` (samples `0..=N`). For a closed
/// curve (sample `N` == sample `0`) the run may wrap across the seam, so the
/// search walks the circular sequence and the returned `b1` may exceed `N`
/// (the caller maps it back through the curve's periodic parameterization). A
/// run covering every distinct sample returns the whole span `(0, N)`.
fn longest_inboth_run(inb: &[bool], closed: bool) -> (usize, usize) {
    let n = inb.len();
    if !closed || n < 2 {
        let (mut b0, mut b1) = (0usize, 0usize);
        let mut start: Option<usize> = None;
        for (i, &v) in inb.iter().enumerate() {
            if v {
                let s = *start.get_or_insert(i);
                if i - s > b1 - b0 {
                    b0 = s;
                    b1 = i;
                }
            } else {
                start = None;
            }
        }
        return (b0, b1);
    }
    // Closed: distinct samples are 0..m (sample m duplicates sample 0).
    let m = n - 1;
    let (mut b0, mut b1) = (0usize, 0usize);
    let mut start: Option<usize> = None;
    for k in 0..2 * m {
        if inb[k % m] {
            let s = *start.get_or_insert(k);
            if k - s >= m {
                return (0, m); // whole curve in-both
            }
            if k - s > b1 - b0 {
                b0 = s;
                b1 = k;
            }
        } else {
            start = None;
        }
    }
    (b0, b1)
}

/// Compute AABB for a face by sampling its boundary edges.
fn compute_face_bbox(topo: &Topology, face_id: FaceId, tol: Tolerance) -> Result<Aabb3, AlgoError> {
    let edges = brepkit_topology::explorer::face_edges(topo, face_id)?;
    let mut points = Vec::new();

    for eid in edges {
        let edge = topo.edge(eid)?;
        let start_pos = topo.vertex(edge.start())?.point();
        let end_pos = topo.vertex(edge.end())?.point();
        let (t0, t1) = edge.curve().domain_with_endpoints(start_pos, end_pos);

        let n: usize = 8;
        for i in 0..=n {
            let t = t0 + (t1 - t0) * (i as f64 / n as f64);
            let pt = edge.curve().evaluate_with_endpoints(t, start_pos, end_pos);
            points.push(pt);
        }
    }

    // A sphere or torus face bulges beyond its boundary edges (a hemisphere's
    // only boundary is its equatorial circle; a full torus's are two degenerate
    // seam points), so the boundary-sampled bbox underestimates the true extent
    // and the broad-phase would wrongly reject genuinely intersecting pairs.
    // Recover the missing extent from the surface, kept as tight as possible so
    // the box stays a sound superset without admitting unrelated geometry.
    let surface_bbox = match topo.face(face_id)?.surface() {
        // Bound to the hemisphere the face occupies (pole side from the boundary
        // winding) — the full-sphere box would let one hemisphere admit the
        // other's sections. Ambiguous winding falls back to the full sphere.
        FaceSurface::Sphere(s) => Some(match sphere_region_axis(topo, face_id, s.center(), tol) {
            Some(axis) => s.aabb_region(axis),
            None => s.aabb(),
        }),
        // Only a full, untrimmed torus needs the surface box — its boundary is
        // the degenerate fundamental-polygon seam, which collapses to a point. A
        // trimmed torus patch is bounded by its real edges; widening it to the
        // whole torus would admit geometry on omitted angular bands.
        FaceSurface::Torus(t) if face_boundary_all_degenerate(topo, face_id, tol)? => {
            Some(t.aabb())
        }
        _ => None,
    };

    Ok(match (points.is_empty(), surface_bbox) {
        (false, Some(sb)) => Aabb3::from_points(points).union(sb),
        (false, None) => Aabb3::from_points(points),
        (true, Some(sb)) => sb,
        // Degenerate face with no edges and no surface box -- zero-volume box.
        (true, None) => Aabb3 {
            min: Point3::new(0.0, 0.0, 0.0),
            max: Point3::new(0.0, 0.0, 0.0),
        },
    })
}

/// Pole-side axis of a spherical face, from its boundary winding: the summed
/// `(midpoint − center) × chord` over the outer wire points from the sphere
/// center into the face's hemisphere (negated for a reversed face). Returns
/// `None` when the wire is degenerate or near-planar through the center, so the
/// side is ambiguous. Shared by the broad-phase AABB and the section in-both
/// filter so the two stay consistent.
fn sphere_region_axis(
    topo: &Topology,
    face_id: FaceId,
    center: Point3,
    tol: Tolerance,
) -> Option<Vec3> {
    let face = topo.face(face_id).ok()?;
    let wire = topo.wire(face.outer_wire()).ok()?;
    // Sample the outer wire into an oriented closed polyline. Endpoint-only
    // sampling fails for a boundary built from a single closed `Circle` edge
    // (start == end, so every chord is zero); sampling along each edge recovers
    // the loop shape so the winding below still yields the pole-side axis.
    let mut pts: Vec<Point3> = Vec::new();
    for oe in wire.edges() {
        let Ok(edge) = topo.edge(oe.edge()) else {
            continue;
        };
        let (Ok(sv), Ok(ev)) = (topo.vertex(edge.start()), topo.vertex(edge.end())) else {
            continue;
        };
        let (sp, ep) = (sv.point(), ev.point());
        let (t0, t1) = edge.curve().domain_with_endpoints(sp, ep);
        // Sample in the edge's natural direction (omit the endpoint — it is the
        // next edge's start), then flip for a reversed orientation.
        let n = 8;
        let mut edge_pts: Vec<Point3> = (0..n)
            .map(|i| {
                let t = t0 + (t1 - t0) * (f64::from(i) / f64::from(n));
                edge.curve().evaluate_with_endpoints(t, sp, ep)
            })
            .collect();
        if !oe.is_forward() {
            edge_pts.reverse();
        }
        pts.append(&mut edge_pts);
    }
    if pts.len() < 3 {
        return None;
    }
    // Summed (midpoint − center) × chord around the closed loop ≈ 2·(area
    // vector): its direction is the face's outward pole axis (negated for a
    // reversed face). The cross products have units of length^2, so the
    // degeneracy threshold is derived from the input magnitudes: `scale` sums
    // each term's bound (|mid − center| · |chord|); a near-planar-through-center
    // loop (ambiguous side) leaves `axis` small relative to it.
    let mut axis = Vec3::new(0.0, 0.0, 0.0);
    let mut scale = 0.0;
    let count = pts.len();
    for i in 0..count {
        let a = pts[i];
        let b = pts[(i + 1) % count];
        let mid = a + (b - a) * 0.5;
        let radial = mid - center;
        let chord = b - a;
        scale += radial.length() * chord.length();
        axis += radial.cross(chord);
    }
    if face.is_reversed() {
        axis = axis * -1.0;
    }
    let len = axis.length();
    if scale < tol.linear * tol.linear || len < scale * tol.linear {
        return None;
    }
    Some(axis * (1.0 / len))
}

/// True when every edge of the face has zero spatial extent (a degenerate seam
/// point) — the signature of a full, untrimmed torus, whose boundary is the
/// doubly-periodic fundamental-polygon seam built from `Line(v0, v0)` edges.
///
/// Extent (not just endpoint coincidence) is the test: a closed `Circle` or
/// closed NURBS edge also has `start == end`, yet it spans a real loop and
/// bounds a *trimmed* patch — those must return `false` so the patch is not
/// over-widened to the whole torus. Each edge is sampled along its curve and
/// must stay within `tol` of its start point.
fn face_boundary_all_degenerate(
    topo: &Topology,
    face_id: FaceId,
    tol: Tolerance,
) -> Result<bool, AlgoError> {
    let edges = brepkit_topology::explorer::face_edges(topo, face_id)?;
    if edges.is_empty() {
        return Ok(false);
    }
    for eid in edges {
        let edge = topo.edge(eid)?;
        let sp = topo.vertex(edge.start())?.point();
        let ep = topo.vertex(edge.end())?.point();
        let (t0, t1) = edge.curve().domain_with_endpoints(sp, ep);
        for frac in [0.0, 0.25, 0.5, 0.75, 1.0] {
            let t = t0 + (t1 - t0) * frac;
            let p = edge.curve().evaluate_with_endpoints(t, sp, ep);
            if (p - sp).length() > tol.linear {
                return Ok(false);
            }
        }
    }
    Ok(true)
}

/// Compute AABBs for a list of faces.
fn compute_face_bboxes(
    topo: &Topology,
    faces: &[FaceId],
    tol: Tolerance,
) -> Result<Vec<Aabb3>, AlgoError> {
    let mut bboxes = Vec::with_capacity(faces.len());
    for &fid in faces {
        bboxes.push(compute_face_bbox(topo, fid, tol)?);
    }
    Ok(bboxes)
}

/// Compute the v-parameter range of a face by projecting boundary vertices.
/// Returns `None` for planes (which have no UV parameterization) or if projection fails.
fn face_v_range(topo: &Topology, face_id: FaceId, surface: &FaceSurface) -> Option<(f64, f64)> {
    let face = topo.face(face_id).ok()?;
    let wire = topo.wire(face.outer_wire()).ok()?;
    let mut v_min = f64::MAX;
    let mut v_max = f64::MIN;
    for oe in wire.edges() {
        let edge = topo.edge(oe.edge()).ok()?;
        let sp = topo.vertex(edge.start()).ok()?.point();
        let ep = topo.vertex(edge.end()).ok()?.point();
        let (t0, t1) = edge.curve().domain_with_endpoints(sp, ep);
        // Sample 5 points to capture v-extremes on curved/closed edges
        for frac in [0.0, 0.25, 0.5, 0.75, 1.0] {
            let t = t0 + (t1 - t0) * frac;
            let pt = edge.curve().evaluate_with_endpoints(t, sp, ep);
            if let Some((_, v)) = surface.project_point(pt) {
                v_min = v_min.min(v);
                v_max = v_max.max(v);
            }
        }
    }
    if v_min < v_max {
        Some((v_min, v_max))
    } else {
        None
    }
}

/// Intermediate intersection result before face IDs are assigned.
#[derive(Clone)]
struct RawCurve {
    /// The 3D curve geometry.
    curve: EdgeCurve,
    /// Bounding box of the curve.
    bbox: Aabb3,
    /// Parameter range on the curve.
    t_range: (f64, f64),
    /// 3D position at the start of the parameter range.
    p_start: Point3,
    /// 3D position at the end of the parameter range.
    p_end: Point3,
}

/// Compute raw intersection curves between two surfaces.
///
/// Dispatches by surface type pair. Raw curves are returned without
/// trimming to face boundaries.
#[allow(clippy::too_many_lines)]
fn compute_raw_curves(
    surf_a: &FaceSurface,
    surf_b: &FaceSurface,
    bbox_a: &Aabb3,
    bbox_b: &Aabb3,
    v_range_a: Option<(f64, f64)>,
    v_range_b: Option<(f64, f64)>,
) -> Result<Vec<RawCurve>, AlgoError> {
    match (surf_a, surf_b) {
        (FaceSurface::Plane { normal: na, d: da }, FaceSurface::Plane { normal: nb, d: db }) => {
            plane_plane_intersection(*na, *da, *nb, *db, bbox_a, bbox_b)
        }

        (FaceSurface::Plane { normal, d }, FaceSurface::Cylinder(cyl))
            if normal.dot(cyl.axis()).abs() < 1e-9 =>
        {
            // Plane parallel to the cylinder axis: the intersection is 0 or 2
            // straight lines along the axis (not a circle/ellipse). The exact
            // analytic path samples the base circle and only keeps points
            // that happen to land on the plane, so it returns nothing for a
            // plane that grazes the lateral surface (e.g. a wall's top edge
            // plane cutting a rounded notch corner). Solve the two contact
            // angles directly and emit axis-parallel lines, bbox-trimmed.
            plane_cylinder_parallel_lines(*normal, *d, cyl, bbox_a, bbox_b)
        }

        (FaceSurface::Cylinder(cyl), FaceSurface::Plane { normal, d })
            if normal.dot(cyl.axis()).abs() < 1e-9 =>
        {
            plane_cylinder_parallel_lines(*normal, *d, cyl, bbox_a, bbox_b)
        }

        (FaceSurface::Plane { normal, d }, other) if other.as_analytic().is_some() => {
            if let Some(analytic) = other.as_analytic() {
                plane_analytic_intersection(*normal, *d, &analytic)
            } else {
                Ok(Vec::new())
            }
        }

        (other, FaceSurface::Plane { normal, d }) if other.as_analytic().is_some() => {
            if let Some(analytic) = other.as_analytic() {
                plane_analytic_intersection(*normal, *d, &analytic)
            } else {
                Ok(Vec::new())
            }
        }

        (FaceSurface::Cone(c1), FaceSurface::Cone(c2)) => {
            // Coaxial cones meet at a single circle that coincides with their
            // shared cap rim. Emit it as an exact Circle so the closed-circle
            // handling (seam adoption + `link_existing`) treats it as the
            // existing shared boundary edge instead of adopting a fresh,
            // redundant section edge. Non-coaxial cones (None) fall through to
            // the general marcher.
            match analytic_intersection::exact_cone_cone(c1, c2)? {
                Some(exacts) => {
                    let mut results = Vec::new();
                    for exact in exacts {
                        if let analytic_intersection::ExactIntersectionCurve::Circle(circle) = exact
                        {
                            let bbox = circle_bbox(&circle);
                            let domain = (0.0, std::f64::consts::TAU);
                            let p_start = ParametricCurve::evaluate(&circle, domain.0);
                            let p_end = ParametricCurve::evaluate(&circle, domain.1);
                            results.push(RawCurve {
                                curve: EdgeCurve::Circle(circle),
                                bbox,
                                t_range: domain,
                                p_start,
                                p_end,
                            });
                        }
                    }
                    Ok(results)
                }
                None => {
                    if let (Some(aa), Some(ab)) = (surf_a.as_analytic(), surf_b.as_analytic()) {
                        analytic_analytic_intersection(&aa, &ab, v_range_a, v_range_b)
                    } else {
                        Ok(Vec::new())
                    }
                }
            }
        }

        (FaceSurface::Cone(cone), FaceSurface::Cylinder(cyl))
        | (FaceSurface::Cylinder(cyl), FaceSurface::Cone(cone)) => {
            // A coaxial cone and cylinder meet at the single circle where the
            // cone's radius equals the cylinder's — the gridfinity lip's top
            // knife edge (inner tapered corner = cone, outer corner =
            // cylinder, concentric, matching at Z_PEAK). Emit it as an exact
            // Circle so seam adoption links it to the shared cap rim, instead
            // of letting the marcher fragment the near-tangent contact into
            // degenerate micro-arcs (the 98-free-edge corruption). Non-coaxial
            // (None) falls through to the general marcher.
            match analytic_intersection::exact_cone_cylinder(cone, cyl)? {
                Some(exacts) => {
                    let mut results = Vec::new();
                    for exact in exacts {
                        if let analytic_intersection::ExactIntersectionCurve::Circle(circle) = exact
                        {
                            let bbox = circle_bbox(&circle);
                            let domain = (0.0, std::f64::consts::TAU);
                            let p_start = ParametricCurve::evaluate(&circle, domain.0);
                            let p_end = ParametricCurve::evaluate(&circle, domain.1);
                            results.push(RawCurve {
                                curve: EdgeCurve::Circle(circle),
                                bbox,
                                t_range: domain,
                                p_start,
                                p_end,
                            });
                        }
                    }
                    Ok(results)
                }
                None => {
                    if let (Some(aa), Some(ab)) = (surf_a.as_analytic(), surf_b.as_analytic()) {
                        analytic_analytic_intersection(&aa, &ab, v_range_a, v_range_b)
                    } else {
                        Ok(Vec::new())
                    }
                }
            }
        }

        (FaceSurface::Sphere(sphere), FaceSurface::Cylinder(cyl))
        | (FaceSurface::Cylinder(cyl), FaceSurface::Sphere(sphere)) => {
            // A coaxial sphere and cylinder meet at one or two circles (the
            // tunnel rims of a sphere-with-bore). Emit them as exact Circles
            // so the closed-circle FF split carves each hemisphere into a
            // band-with-hole + cap; the marcher would return NURBS fragments
            // the splitter cannot recognise, dropping the spherical faces.
            // Non-coaxial (None) falls through to the general marcher.
            match analytic_intersection::exact_sphere_cylinder(sphere, cyl)? {
                Some(exacts) => {
                    let mut results = Vec::new();
                    for exact in exacts {
                        if let analytic_intersection::ExactIntersectionCurve::Circle(circle) = exact
                        {
                            let bbox = circle_bbox(&circle);
                            let domain = (0.0, std::f64::consts::TAU);
                            let p_start = ParametricCurve::evaluate(&circle, domain.0);
                            let p_end = ParametricCurve::evaluate(&circle, domain.1);
                            results.push(RawCurve {
                                curve: EdgeCurve::Circle(circle),
                                bbox,
                                t_range: domain,
                                p_start,
                                p_end,
                            });
                        }
                    }
                    Ok(results)
                }
                None => {
                    if let (Some(aa), Some(ab)) = (surf_a.as_analytic(), surf_b.as_analytic()) {
                        analytic_analytic_intersection(&aa, &ab, v_range_a, v_range_b)
                    } else {
                        Ok(Vec::new())
                    }
                }
            }
        }

        (a, b) if a.as_analytic().is_some() && b.as_analytic().is_some() => {
            if let (Some(aa), Some(ab)) = (a.as_analytic(), b.as_analytic()) {
                analytic_analytic_intersection(&aa, &ab, v_range_a, v_range_b)
            } else {
                Ok(Vec::new())
            }
        }

        (FaceSurface::Plane { normal, d }, FaceSurface::Nurbs(nurbs))
        | (FaceSurface::Nurbs(nurbs), FaceSurface::Plane { normal, d }) => {
            plane_nurbs_intersection(*normal, *d, nurbs)
        }

        (analytic_surf, FaceSurface::Nurbs(nurbs)) if analytic_surf.as_analytic().is_some() => {
            // Deferred to later phases -- analytic-NURBS is complex
            let _ = nurbs;
            Ok(Vec::new())
        }
        (FaceSurface::Nurbs(nurbs), analytic_surf) if analytic_surf.as_analytic().is_some() => {
            let _ = nurbs;
            Ok(Vec::new())
        }

        (FaceSurface::Nurbs(na), FaceSurface::Nurbs(nb)) => nurbs_nurbs_intersection(na, nb),

        // Fallback: unsupported pair
        _ => Ok(Vec::new()),
    }
}

/// Plane-plane intersection: direction = cross product of normals.
#[allow(clippy::unnecessary_wraps)]
fn plane_plane_intersection(
    na: Vec3,
    da: f64,
    nb: Vec3,
    db: f64,
    bbox_a: &Aabb3,
    bbox_b: &Aabb3,
) -> Result<Vec<RawCurve>, AlgoError> {
    let dir = na.cross(nb);
    let dir_len = dir.length();

    if dir_len < 1e-12 {
        // Planes are parallel or coplanar -- no line intersection.
        // Coplanar case is handled separately by the builder.
        return Ok(Vec::new());
    }

    let dir = dir * (1.0 / dir_len);

    // Find a point on the intersection line.
    let point = find_plane_plane_point(na, da, nb, db, dir);

    // Trim parameter range to the combined face AABBs.
    let t_range = trim_t_range_to_aabb(point, dir, bbox_a, bbox_b);
    let p0 = point + dir * t_range.0;
    let p1 = point + dir * t_range.1;

    let bbox = Aabb3 {
        min: Point3::new(p0.x().min(p1.x()), p0.y().min(p1.y()), p0.z().min(p1.z())),
        max: Point3::new(p0.x().max(p1.x()), p0.y().max(p1.y()), p0.z().max(p1.z())),
    };

    Ok(vec![RawCurve {
        curve: EdgeCurve::Line,
        bbox,
        t_range,
        p_start: p0,
        p_end: p1,
    }])
}

/// Trim a line's parameter range to the combined extent of two AABBs.
///
/// Projects the eight corners of the union of `bbox_a` and `bbox_b` onto
/// the line `origin + t * dir` and returns the (min, max) parameter range.
/// `dir` must be unit-length.
fn trim_t_range_to_aabb(origin: Point3, dir: Vec3, bbox_a: &Aabb3, bbox_b: &Aabb3) -> (f64, f64) {
    let cmin = Point3::new(
        bbox_a.min.x().min(bbox_b.min.x()),
        bbox_a.min.y().min(bbox_b.min.y()),
        bbox_a.min.z().min(bbox_b.min.z()),
    );
    let cmax = Point3::new(
        bbox_a.max.x().max(bbox_b.max.x()),
        bbox_a.max.y().max(bbox_b.max.y()),
        bbox_a.max.z().max(bbox_b.max.z()),
    );

    let mut t_min = f64::MAX;
    let mut t_max = f64::MIN;
    for &x in &[cmin.x(), cmax.x()] {
        for &y in &[cmin.y(), cmax.y()] {
            for &z in &[cmin.z(), cmax.z()] {
                let corner = Point3::new(x, y, z);
                let t = (corner - origin).dot(dir);
                t_min = t_min.min(t);
                t_max = t_max.max(t);
            }
        }
    }

    (t_min, t_max)
}

/// Find a point on the plane-plane intersection line.
///
/// The point lies in the plane spanned by the two normals and satisfies
/// both plane equations.
fn find_plane_plane_point(na: Vec3, da: f64, nb: Vec3, db: f64, dir: Vec3) -> Point3 {
    // P = (da * (nb x dir) + db * (dir x na)) / dot(dir, na x nb)
    let na_cross_nb = na.cross(nb);
    let denom = dir.dot(na_cross_nb);

    if denom.abs() < 1e-15 {
        // Degenerate -- return origin as fallback
        return Point3::new(0.0, 0.0, 0.0);
    }

    let nb_cross_dir = nb.cross(dir);
    let dir_cross_na = dir.cross(na);

    Point3::new(
        (da * nb_cross_dir.x() + db * dir_cross_na.x()) / denom,
        (da * nb_cross_dir.y() + db * dir_cross_na.y()) / denom,
        (da * nb_cross_dir.z() + db * dir_cross_na.z()) / denom,
    )
}

/// Plane-analytic surface intersection using exact curves.
fn plane_analytic_intersection(
    normal: Vec3,
    d: f64,
    analytic: &analytic_intersection::AnalyticSurface<'_>,
) -> Result<Vec<RawCurve>, AlgoError> {
    let exact_curves = analytic_intersection::exact_plane_analytic(*analytic, normal, d)?;

    let mut results = Vec::new();
    for exact in exact_curves {
        match exact {
            analytic_intersection::ExactIntersectionCurve::Circle(circle) => {
                let bbox = circle_bbox(&circle);
                let domain = (0.0, std::f64::consts::TAU);
                let p_start = ParametricCurve::evaluate(&circle, domain.0);
                let p_end = ParametricCurve::evaluate(&circle, domain.1);
                results.push(RawCurve {
                    curve: EdgeCurve::Circle(circle),
                    bbox,
                    t_range: domain,
                    p_start,
                    p_end,
                });
            }
            analytic_intersection::ExactIntersectionCurve::Ellipse(ellipse) => {
                let bbox = ellipse_bbox(&ellipse);
                let domain = (0.0, std::f64::consts::TAU);
                let p_start = ParametricCurve::evaluate(&ellipse, domain.0);
                let p_end = ParametricCurve::evaluate(&ellipse, domain.1);
                results.push(RawCurve {
                    curve: EdgeCurve::Ellipse(ellipse),
                    bbox,
                    t_range: domain,
                    p_start,
                    p_end,
                });
            }
            analytic_intersection::ExactIntersectionCurve::Points(pts) => {
                if pts.len() < 2 {
                    continue;
                }
                // Fit a degree-3 NURBS curve through the sampled points
                let nurbs = brepkit_math::nurbs::fitting::interpolate(&pts, 3)
                    .map_err(|e| AlgoError::IntersectionFailed(format!("NURBS fit failed: {e}")))?;
                let t_range = nurbs.domain();
                let bbox = Aabb3::try_from_points(pts.iter().copied()).ok_or_else(|| {
                    AlgoError::IntersectionFailed("empty points for NURBS fit".into())
                })?;
                let end_pt = pts[pts.len() - 1];
                results.push(RawCurve {
                    curve: EdgeCurve::NurbsCurve(nurbs),
                    bbox,
                    t_range,
                    p_start: pts[0],
                    p_end: end_pt,
                });
            }
        }
    }

    Ok(results)
}

/// Intersect a plane parallel to a cylinder's axis with the cylinder.
///
/// When the plane normal is perpendicular to the cylinder axis the contact is
/// 0 or 2 straight lines running along the axis (not a circle/ellipse). Solve
/// the contact angle(s) `u` from `r·(cos u·(n·X) + sin u·(n·Y)) = d − n·O` and
/// emit each as an axis-parallel `Line` raw curve, with its parameter range
/// trimmed to the two faces' combined AABBs (mirrors `plane_plane_intersection`).
///
/// Returns `Result` to match the other `compute_raw_curves` arms (uniform
/// dispatch); it never actually fails.
#[allow(clippy::unnecessary_wraps)]
fn plane_cylinder_parallel_lines(
    normal: Vec3,
    d: f64,
    cyl: &brepkit_math::surfaces::CylindricalSurface,
    bbox_a: &Aabb3,
    bbox_b: &Aabb3,
) -> Result<Vec<RawCurve>, AlgoError> {
    let axis = cyl.axis();
    let r = cyl.radius();
    let origin = cyl.origin();
    let x = cyl.x_axis();
    let y = cyl.y_axis();

    // n·P(u,v) = n·O + v·(n·axis) + r·(cos u·(n·X) + sin u·(n·Y)); n·axis ≈ 0.
    let a = normal.dot(x);
    let b = normal.dot(y);
    let n_dot_o = normal.x() * origin.x() + normal.y() * origin.y() + normal.z() * origin.z();
    let amp = a.hypot(b);
    if amp < 1e-12 {
        return Ok(Vec::new());
    }
    // R·cos(u − φ) = C, φ = atan2(b, a), C = (d − n·O) / r.
    let c = (d - n_dot_o) / r;
    let ratio = c / amp;
    // |ratio| > 1: the plane misses the cylinder; |ratio| ≈ 1: tangent (single
    // grazing line that never splits a face — skip).
    if ratio.abs() > 1.0 - 1e-9 {
        return Ok(Vec::new());
    }
    let phi = b.atan2(a);
    let delta = ratio.clamp(-1.0, 1.0).acos();
    let dir = {
        let len = axis.length();
        if len < 1e-12 {
            return Ok(Vec::new());
        }
        axis * (1.0 / len)
    };

    let mut results = Vec::new();
    for u in [phi + delta, phi - delta] {
        // A point on the contact line at v = 0.
        let base = cyl.evaluate(u, 0.0);
        let t_range = trim_t_range_to_aabb(base, dir, bbox_a, bbox_b);
        if (t_range.1 - t_range.0).abs() < 1e-9 {
            continue;
        }
        let p0 = base + dir * t_range.0;
        let p1 = base + dir * t_range.1;
        let bbox = Aabb3 {
            min: Point3::new(p0.x().min(p1.x()), p0.y().min(p1.y()), p0.z().min(p1.z())),
            max: Point3::new(p0.x().max(p1.x()), p0.y().max(p1.y()), p0.z().max(p1.z())),
        };
        results.push(RawCurve {
            curve: EdgeCurve::Line,
            bbox,
            t_range,
            p_start: p0,
            p_end: p1,
        });
    }
    Ok(results)
}

/// Analytic-analytic surface intersection using marching.
fn analytic_analytic_intersection(
    a: &analytic_intersection::AnalyticSurface<'_>,
    b: &analytic_intersection::AnalyticSurface<'_>,
    v_range_a: Option<(f64, f64)>,
    v_range_b: Option<(f64, f64)>,
) -> Result<Vec<RawCurve>, AlgoError> {
    let isect_curves = analytic_intersection::intersect_analytic_analytic_bounded(
        *a, *b, 32, v_range_a, v_range_b,
    )?;

    let mut results = Vec::new();
    for ic in isect_curves {
        let domain = ic.curve.domain();
        let bbox = nurbs_curve_bbox(&ic.curve);
        let p_start = ParametricCurve::evaluate(&ic.curve, domain.0);
        let p_end = ParametricCurve::evaluate(&ic.curve, domain.1);
        results.push(RawCurve {
            curve: EdgeCurve::NurbsCurve(ic.curve),
            bbox,
            t_range: domain,
            p_start,
            p_end,
        });
    }

    Ok(results)
}

/// Plane-NURBS intersection.
fn plane_nurbs_intersection(
    normal: Vec3,
    d: f64,
    nurbs: &brepkit_math::nurbs::surface::NurbsSurface,
) -> Result<Vec<RawCurve>, AlgoError> {
    let isect_curves = nurbs_isect::intersect_plane_nurbs(nurbs, normal, d, NURBS_SAMPLES)?;

    let mut results = Vec::new();
    for ic in isect_curves {
        let domain = ic.curve.domain();
        let bbox = nurbs_curve_bbox(&ic.curve);
        let p_start = ParametricCurve::evaluate(&ic.curve, domain.0);
        let p_end = ParametricCurve::evaluate(&ic.curve, domain.1);
        results.push(RawCurve {
            curve: EdgeCurve::NurbsCurve(ic.curve),
            bbox,
            t_range: domain,
            p_start,
            p_end,
        });
    }

    Ok(results)
}

/// NURBS-NURBS intersection.
fn nurbs_nurbs_intersection(
    na: &brepkit_math::nurbs::surface::NurbsSurface,
    nb: &brepkit_math::nurbs::surface::NurbsSurface,
) -> Result<Vec<RawCurve>, AlgoError> {
    let isect_curves = nurbs_isect::intersect_nurbs_nurbs(na, nb, NURBS_SAMPLES, NURBS_MARCH_STEP)?;

    let mut results = Vec::new();
    for ic in isect_curves {
        let domain = ic.curve.domain();
        let bbox = nurbs_curve_bbox(&ic.curve);
        let p_start = ParametricCurve::evaluate(&ic.curve, domain.0);
        let p_end = ParametricCurve::evaluate(&ic.curve, domain.1);
        results.push(RawCurve {
            curve: EdgeCurve::NurbsCurve(ic.curve),
            bbox,
            t_range: domain,
            p_start,
            p_end,
        });
    }

    Ok(results)
}

/// Compute AABB for a circle.
fn circle_bbox(circle: &brepkit_math::curves::Circle3D) -> Aabb3 {
    let n = 16;
    let points: Vec<Point3> = (0..=n)
        .map(|i| {
            let t = std::f64::consts::TAU * (i as f64 / n as f64);
            ParametricCurve::evaluate(circle, t)
        })
        .collect();
    Aabb3::from_points(points)
}

/// Compute AABB for an ellipse.
fn ellipse_bbox(ellipse: &brepkit_math::curves::Ellipse3D) -> Aabb3 {
    let n = 16;
    let points: Vec<Point3> = (0..=n)
        .map(|i| {
            let t = std::f64::consts::TAU * (i as f64 / n as f64);
            ParametricCurve::evaluate(ellipse, t)
        })
        .collect();
    Aabb3::from_points(points)
}

/// Find an existing vertex on a face's boundary within tolerance of a point.
///
/// Iterates the face's outer wire vertices and returns the first one
/// within `tol.linear` of `point`. This implements the "PutPavesOnCurve"
/// vertex snapping: intersection curve endpoints at face boundaries reuse
/// the face's existing boundary vertices instead of creating duplicates.
fn find_nearby_face_vertex(
    topo: &Topology,
    face_id: FaceId,
    point: Point3,
    tol: Tolerance,
) -> Option<brepkit_topology::vertex::VertexId> {
    let face = topo.face(face_id).ok()?;
    let wire = topo.wire(face.outer_wire()).ok()?;
    for oe in wire.edges() {
        let edge = topo.edge(oe.edge()).ok()?;
        for &vid in &[edge.start(), edge.end()] {
            let vpt = topo.vertex(vid).ok()?.point();
            if (vpt - point).length() < tol.linear {
                return Some(vid);
            }
        }
    }
    None
}

/// Find an existing boundary vertex of either face that lies on a closed
/// section curve, returning the vertex, its curve parameter, and its 3D
/// position.
///
/// Scans the outer and inner wires of both faces and returns the first
/// vertex whose distance to the curve is within `tol.linear`. Only
/// analytic closed curves (Circle, Ellipse) are supported.
fn find_boundary_vertex_on_curve(
    topo: &Topology,
    face_a: FaceId,
    face_b: FaceId,
    curve: &EdgeCurve,
    tol: Tolerance,
) -> Option<(brepkit_topology::vertex::VertexId, f64, Point3)> {
    let project_eval = |p: Point3| -> Option<(f64, Point3)> {
        match curve {
            EdgeCurve::Circle(c) => {
                let t = c.project(p);
                Some((t, c.evaluate(t)))
            }
            EdgeCurve::Ellipse(e) => {
                let t = e.project(p);
                Some((t, e.evaluate(t)))
            }
            EdgeCurve::Line | EdgeCurve::NurbsCurve(_) => None,
        }
    };

    for fid in [face_a, face_b] {
        let Ok(face) = topo.face(fid) else {
            continue;
        };
        let wires: Vec<brepkit_topology::wire::WireId> = std::iter::once(face.outer_wire())
            .chain(face.inner_wires().iter().copied())
            .collect();
        for wid in wires {
            let Ok(wire) = topo.wire(wid) else {
                continue;
            };
            for oe in wire.edges() {
                let Ok(edge) = topo.edge(oe.edge()) else {
                    continue;
                };
                for vid in [edge.start(), edge.end()] {
                    let Ok(v) = topo.vertex(vid) else {
                        continue;
                    };
                    let p = v.point();
                    let (t, foot) = project_eval(p)?;
                    if (foot - p).length() < tol.linear {
                        return Some((vid, t, p));
                    }
                }
            }
        }
    }
    None
}

/// Check whether a closed section circle properly crosses any boundary
/// edge (outer or inner wire) of either face.
///
/// Line boundary edges count via segment intersection, ignoring hits at
/// the segment endpoints (a seam line legitimately starts on the circle).
/// Circle boundary edges count via the coplanar circle-circle relation:
/// two distinct coplanar circles cross when the in-plane center distance
/// lies strictly between `|r1 - r2|` and `r1 + r2`. Other curve types are
/// not checked.
fn closed_circle_crosses_face_boundaries(
    topo: &Topology,
    face_a: FaceId,
    face_b: FaceId,
    circle: &brepkit_math::curves::Circle3D,
    tol: Tolerance,
) -> bool {
    for fid in [face_a, face_b] {
        let Ok(face) = topo.face(fid) else {
            continue;
        };
        let wires: Vec<brepkit_topology::wire::WireId> = std::iter::once(face.outer_wire())
            .chain(face.inner_wires().iter().copied())
            .collect();
        for wid in wires {
            let Ok(wire) = topo.wire(wid) else {
                continue;
            };
            for oe in wire.edges() {
                let Ok(edge) = topo.edge(oe.edge()) else {
                    continue;
                };
                match edge.curve() {
                    EdgeCurve::Line => {
                        let (Ok(sv), Ok(ev)) = (topo.vertex(edge.start()), topo.vertex(edge.end()))
                        else {
                            continue;
                        };
                        let (sp, ep) = (sv.point(), ev.point());
                        for (p, _) in circle.intersect_segment(sp, ep, tol.linear) {
                            let at_endpoint = (p - sp).length() < tol.linear * 10.0
                                || (p - ep).length() < tol.linear * 10.0;
                            if !at_endpoint {
                                return true;
                            }
                        }
                    }
                    EdgeCurve::Circle(b) => {
                        let coplanar = circle.normal().dot(b.normal()).abs() > 1.0 - 1e-9
                            && (b.center() - circle.center()).dot(circle.normal()).abs()
                                < tol.linear;
                        if !coplanar {
                            continue;
                        }
                        let d = (b.center() - circle.center()).length();
                        let (r1, r2) = (circle.radius(), b.radius());
                        if d > (r1 - r2).abs() + tol.linear && d < r1 + r2 - tol.linear {
                            return true;
                        }
                    }
                    EdgeCurve::Ellipse(_) | EdgeCurve::NurbsCurve(_) => {}
                }
            }
        }
    }
    false
}

/// Find where a closed `Circle3D` section crosses the outer-wire
/// boundaries of the face pair.
///
/// Returns crossings as `(t_on_circle, point_3d)` sorted by `t`, with
/// duplicates removed. Only `Line`-curve boundary edges are considered.
///
/// Hits are collected per face. A face whose boundary yields more than
/// 4 hits is treated as circle-coincident (a chord polygon inscribed in
/// the circle — every chord endpoint lies on the circle, e.g. the
/// equator polygon of a faceted sphere whose great circle IS the
/// section): its hits describe the boundary itself rather than
/// entry/exit points, so they are excluded and only the other face's
/// crossings are used.
///
/// When the pair contains a sphere face, the surviving hits from BOTH
/// faces are unioned so the arcs are trimmed to the mutual overlap of
/// the two face regions (a sphere face has no seam structure, so its
/// noseam splitter needs arcs whose endpoints all land on the mutual
/// region boundary). For other analytic pairs (cylinder/cone laterals)
/// only the non-plane face's crossings are used — those surfaces keep
/// full closed section circles for the periodic band splitter, and
/// their seam-line hit must not combine with plane-boundary hits.
/// Whether a closed section circle crosses any LINE boundary edge of a plane
/// face at a point interior to that edge (not at a shared endpoint).
///
/// Used to distinguish a prism-corner section arc that exits a plane face's
/// boundary (e.g. a notch corner straddling a wall's top edge) from a
/// periodic-band section circle that stays inside the plane face.
fn circle_exits_plane_boundary(
    topo: &Topology,
    plane_face: FaceId,
    circle: &brepkit_math::curves::Circle3D,
    tol: Tolerance,
) -> bool {
    let Ok(face) = topo.face(plane_face) else {
        return false;
    };
    let wires: Vec<brepkit_topology::wire::WireId> = std::iter::once(face.outer_wire())
        .chain(face.inner_wires().iter().copied())
        .collect();
    for wid in wires {
        let Ok(wire) = topo.wire(wid) else {
            continue;
        };
        for oe in wire.edges() {
            let Ok(edge) = topo.edge(oe.edge()) else {
                continue;
            };
            if !matches!(edge.curve(), EdgeCurve::Line) {
                continue;
            }
            let (Ok(sv), Ok(ev)) = (topo.vertex(edge.start()), topo.vertex(edge.end())) else {
                continue;
            };
            let (sp, ep) = (sv.point(), ev.point());
            for (p, _) in circle.intersect_segment(sp, ep, tol.linear) {
                let at_endpoint =
                    (p - sp).length() < tol.linear * 10.0 || (p - ep).length() < tol.linear * 10.0;
                if !at_endpoint {
                    return true;
                }
            }
        }
    }
    false
}

fn closed_circle_boundary_crossings(
    topo: &Topology,
    face_a: FaceId,
    face_b: FaceId,
    circle: &brepkit_math::curves::Circle3D,
    tol: Tolerance,
) -> Vec<(f64, Point3)> {
    // Hits carry the boundary edge they came from when that edge is an ARC
    // (`Some(edge_id)`); line-edge hits carry `None`. Adjacent same-arc hit
    // pairs get a midpoint inserted below — the kept span between them would
    // otherwise share BOTH endpoints with the boundary arc (a co-endpoint
    // lens), which `merge_duplicate_edges` folds into one edge, collapsing
    // the lens region to a zero-area slit. The midpoint split is the
    // sanctioned splitter-side resolution (never make the shared merge
    // smarter).
    let face_hits = |fid: FaceId| -> Vec<(f64, Point3, Option<brepkit_topology::edge::EdgeId>)> {
        let mut hits: Vec<(f64, Point3, Option<brepkit_topology::edge::EdgeId>)> = Vec::new();
        let Ok(face) = topo.face(fid) else {
            return hits;
        };
        let Ok(wire) = topo.wire(face.outer_wire()) else {
            return hits;
        };
        for oe in wire.edges() {
            let Ok(edge) = topo.edge(oe.edge()) else {
                continue;
            };
            let Ok(sv) = topo.vertex(edge.start()) else {
                continue;
            };
            let Ok(ev) = topo.vertex(edge.end()) else {
                continue;
            };
            let mut edge_hits: Vec<(f64, Point3, Option<brepkit_topology::edge::EdgeId>)> =
                Vec::new();
            match edge.curve() {
                EdgeCurve::Line => {
                    for (p, t) in circle.intersect_segment(sv.point(), ev.point(), tol.linear) {
                        edge_hits.push((t, p, None));
                    }
                }
                // Arc boundary edges (a plane face rimmed by a cone/cylinder
                // corner): a section circle crossing them was invisible to the
                // Line-only scan, leaving an ODD crossing set — the arcs then
                // desynchronize `emit_split_circle_arcs`' cyclic pairing and
                // whole in-face spans vanish (the lite magnet-pad fuse).
                EdgeCurve::Circle(bc) => {
                    const NS: usize = 128;
                    let full = (sv.point() - ev.point()).length() < tol.linear;
                    let cand = circle.intersect_circle(bc, tol.linear);
                    if cand.is_empty() {
                        continue;
                    }
                    // Filter to the edge's actual arc by proximity to its
                    // sampled polyline (skipped for a full-circle edge). The
                    // band covers the sampling sagitta plus the fit weld.
                    let mut samples: Vec<Point3> = Vec::new();
                    if !full {
                        let (t0, t1) = edge.curve().domain_with_endpoints(sv.point(), ev.point());
                        for i in 0..=NS {
                            #[allow(clippy::cast_precision_loss)]
                            let t = t0 + (t1 - t0) * (i as f64) / (NS as f64);
                            samples.push(edge.curve().evaluate_with_endpoints(
                                t,
                                sv.point(),
                                ev.point(),
                            ));
                        }
                    }
                    let band = {
                        let step = if samples.len() > 1 {
                            (samples[1] - samples[0]).length()
                        } else {
                            0.0
                        };
                        (step * step / (8.0 * bc.radius().max(tol.linear)))
                            .mul_add(2.0, tol.linear * 100.0)
                    };
                    for (p, t) in cand {
                        let on_arc = full
                            || samples.windows(2).any(|w| {
                                let d = w[1] - w[0];
                                let l2 = d.length_squared();
                                let f = if l2 > 1e-20 {
                                    ((p - w[0]).dot(d) / l2).clamp(0.0, 1.0)
                                } else {
                                    0.0
                                };
                                (p - (w[0] + d * f)).length() <= band
                            });
                        if on_arc {
                            edge_hits.push((t, p, Some(oe.edge())));
                        }
                    }
                }
                _ => continue,
            }
            for (t, p, src) in edge_hits {
                let dup = hits
                    .iter()
                    .any(|(_, q, _)| (*q - p).length() < tol.linear * 10.0);
                if !dup {
                    hits.push((t, p, src));
                }
            }
        }
        hits
    };

    let surface_of = |fid: FaceId| topo.face(fid).ok().map(|f| f.surface().clone());
    let surf_a = surface_of(face_a);
    let surf_b = surface_of(face_b);
    let is_plane = |s: &Option<FaceSurface>| matches!(s, Some(FaceSurface::Plane { .. }));
    let is_sphere = |s: &Option<FaceSurface>| matches!(s, Some(FaceSurface::Sphere(_)));

    let pair_has_sphere = is_sphere(&surf_a) || is_sphere(&surf_b);
    let faces_to_check: Vec<FaceId> = if pair_has_sphere {
        vec![face_a, face_b]
    } else {
        match (is_plane(&surf_a), is_plane(&surf_b)) {
            // Plane x lateral-analytic (cylinder/cone): the analytic face's
            // boundary alone splits a section circle that stays inside the
            // plane face (the periodic band case — its seam-line hits drive
            // the band splitter). But when the circle is a prism CORNER
            // arc that crosses the plane face's own boundary (e.g. a
            // rounded-rect notch whose top corner straddles a wall's top
            // edge), the analytic boundary splits it only at the corner's
            // angular limits, leaving an arc that bulges past the plane
            // boundary — `emit_split_circle_arcs` then rejects it by its
            // midpoint and the corner section is lost. Add the plane face's
            // crossings too, but only when the circle genuinely exits the
            // plane boundary, so the in-plane band case is unaffected (it
            // yields no plane-boundary hits).
            (true, false) => {
                if circle_exits_plane_boundary(topo, face_a, circle, tol) {
                    vec![face_a, face_b]
                } else {
                    vec![face_b]
                }
            }
            (false, true) => {
                if circle_exits_plane_boundary(topo, face_b, circle, tol) {
                    vec![face_a, face_b]
                } else {
                    vec![face_a]
                }
            }
            _ => vec![face_a, face_b],
        }
    };

    let mut hits: Vec<(f64, Point3, Option<brepkit_topology::edge::EdgeId>)> = Vec::new();
    for &fid in &faces_to_check {
        // A sphere hemisphere's boundary is a polygon inscribed in the seam
        // (equator) circle. A section circle that genuinely crosses the seam
        // does so at points that lie ON the seam circle but OUTSIDE the
        // inscribed chords (by the polygon sagitta, which is ~5e-2 for 24
        // facets ≫ tol), so the chord-based `face_hits` misses them entirely.
        // Compute those crossings analytically against the seam *plane*
        // instead — exact and independent of the boundary's facet count.
        if matches!(surface_of(fid), Some(FaceSurface::Sphere(_))) {
            let seam = sphere_seam_plane_crossings(topo, fid, circle, tol);
            // Only take the analytic-seam path (and skip the chord-based
            // `face_hits`) when the section actually crosses this face's seam —
            // i.e. a hemisphere whose faceted equator's chords miss the true
            // crossings by the sagitta. A section that doesn't cross the seam (a
            // latitude-circle, or a non-hemisphere sphere face) falls through to
            // the normal `face_hits` path below.
            if !seam.is_empty() {
                for (t, p) in seam {
                    let dup = hits
                        .iter()
                        .any(|(_, q, _)| (*q - p).length() < tol.linear * 10.0);
                    if !dup {
                        hits.push((t, p, None));
                    }
                }
                continue;
            }
        }
        let fh = face_hits(fid);
        // The boundary is coincident with the section circle when its
        // segments are chords of an *inscribed* polygon: every vertex lands
        // on the circle, so the hits are the polygon's vertices, evenly
        // distributed around the full turn. A bare `len > 4` count misses a
        // 4-segment inscribed polygon (square equator → exactly 4 hits), so
        // test even angular distribution instead of relying on the count.
        let fh_plain: Vec<(f64, Point3)> = fh.iter().map(|&(t, p, _)| (t, p)).collect();
        if hits_are_inscribed_polygon(&fh_plain) {
            log::debug!(
                "closed_circle_boundary_crossings: {fid:?} has {} hits evenly distributed on \
                 the circle — boundary coincident with circle, excluding its hits",
                fh.len()
            );
            continue;
        }
        for (t, p, src) in fh {
            let dup = hits
                .iter()
                .any(|(_, q, _)| (*q - p).length() < tol.linear * 10.0);
            if !dup {
                hits.push((t, p, src));
            }
        }
    }

    hits.sort_by(|a, b| a.0.total_cmp(&b.0));

    // Midpoint-split spans whose BOTH ends were minted by the same boundary
    // ARC edge: the span between them shares both endpoints with that arc (a
    // co-endpoint lens) and `merge_duplicate_edges` would fold the pair,
    // collapsing the lens region into a zero-area slit face. Inserting the
    // angular midpoint keeps the section as two sub-arcs — no shared qpair.
    let n = hits.len();
    if n >= 2 {
        let mut mids: Vec<(f64, Point3)> = Vec::new();
        for i in 0..n {
            let (t0, _, s0) = hits[i];
            let (t1, _, s1) = hits[(i + 1) % n];
            if let (Some(a), Some(b)) = (s0, s1)
                && a == b
            {
                let mut t1u = t1;
                if t1u <= t0 {
                    t1u += std::f64::consts::TAU;
                }
                let tm = 0.5 * (t0 + t1u);
                mids.push((tm.rem_euclid(std::f64::consts::TAU), circle.evaluate(tm)));
            }
        }
        for (t, p) in mids {
            hits.push((t, p, None));
        }
        hits.sort_by(|a, b| a.0.total_cmp(&b.0));
    }

    log::trace!(
        "closed_circle_boundary_crossings: face_a={face_a:?} face_b={face_b:?} hits={}",
        hits.len()
    );

    hits.into_iter().map(|(t, p, _)| (t, p)).collect()
}

/// Crossings of a section `circle` with a sphere face's seam (boundary) plane.
///
/// A sphere hemisphere produced by the primitive builder is bounded by a
/// polygon inscribed in its seam circle. Testing a section circle against
/// those chords misses the true seam crossings by the polygon sagitta, so this
/// computes the crossings analytically: fit the seam plane to the boundary
/// vertices (independent of facet count and sphere orientation), then solve for
/// the circle parameters `t` where the circle pierces that plane.
///
/// Returns the `(t, point)` crossings (0, 1 for a tangent, or 2) where the
/// circle meets the seam plane and the crossing point lies on the sphere.
fn sphere_seam_plane_crossings(
    topo: &Topology,
    fid: FaceId,
    circle: &brepkit_math::curves::Circle3D,
    tol: Tolerance,
) -> Vec<(f64, Point3)> {
    let Ok(face) = topo.face(fid) else {
        return Vec::new();
    };
    let FaceSurface::Sphere(sphere) = face.surface() else {
        return Vec::new();
    };
    let Ok(wire) = topo.wire(face.outer_wire()) else {
        return Vec::new();
    };

    // Seam-plane normal + a point on it, from the boundary polygon (Newell's
    // method), so the result is independent of facet count and orientation.
    let verts: Vec<Point3> = wire
        .edges()
        .iter()
        .filter_map(|oe| {
            let edge = topo.edge(oe.edge()).ok()?;
            let start = if oe.is_forward() {
                edge.start()
            } else {
                edge.end()
            };
            topo.vertex(start)
                .ok()
                .map(brepkit_topology::vertex::Vertex::point)
        })
        .collect();
    if verts.len() < 3 {
        return Vec::new();
    }
    let mut normal = Vec3::new(0.0, 0.0, 0.0);
    let mut centroid = Vec3::new(0.0, 0.0, 0.0);
    let n = verts.len();
    for i in 0..n {
        let a = verts[i];
        let b = verts[(i + 1) % n];
        normal += Vec3::new(
            (a.y() - b.y()) * (a.z() + b.z()),
            (a.z() - b.z()) * (a.x() + b.x()),
            (a.x() - b.x()) * (a.y() + b.y()),
        );
        centroid += Vec3::new(a.x(), a.y(), a.z());
    }
    let Ok(plane_n) = normal.normalize() else {
        return Vec::new();
    };
    #[allow(clippy::cast_precision_loss)]
    let inv_n = 1.0 / n as f64;
    let plane_pt = Point3::new(
        centroid.x() * inv_n,
        centroid.y() * inv_n,
        centroid.z() * inv_n,
    );

    // Solve A·cos t + B·sin t = -D for the circle parameter t, where the circle
    // is C(t) = center + r(u·cos t + v·sin t) and the plane is
    // (P - plane_pt)·plane_n = 0.
    let cc = circle.center();
    let r = circle.radius();
    let a = r * circle.u_axis().dot(plane_n);
    let b = r * circle.v_axis().dot(plane_n);
    let d = (cc - plane_pt).dot(plane_n);
    let amp = (a * a + b * b).sqrt();
    if amp < tol.linear {
        // Circle lies in (or parallel to) the seam plane — not a transversal
        // crossing; defer to the chord-based path / interior treatment.
        return Vec::new();
    }
    let rhs = -d / amp;
    if rhs.abs() > 1.0 + 1e-9 {
        return Vec::new();
    }
    let rhs = rhs.clamp(-1.0, 1.0);
    let phase = b.atan2(a);
    let alpha = rhs.acos();
    let mut out: Vec<(f64, Point3)> = Vec::new();
    for &t in &[phase + alpha, phase - alpha] {
        let p = circle.evaluate(t);
        let on_sphere =
            ((p - sphere.center()).length() - sphere.radius()).abs() < tol.linear * 100.0;
        if !on_sphere {
            continue;
        }
        if !out
            .iter()
            .any(|(_, q)| (*q - p).length() < tol.linear * 10.0)
        {
            out.push((t.rem_euclid(std::f64::consts::TAU), p));
        }
    }
    out
}

/// Whether boundary/circle hits describe an inscribed polygon (the boundary
/// is coincident with the section circle) rather than a small set of
/// entry/exit crossings.
///
/// Hits store the circle parameter `t` (an angle). An inscribed polygon's
/// vertices spread evenly around the full circle, so the sorted angular gaps
/// between consecutive hits (including the wrap-around gap) are all
/// approximately equal and there are at least three of them. Two entry/exit
/// crossings, or a few unevenly clustered hits, fail this test.
fn hits_are_inscribed_polygon(hits: &[(f64, Point3)]) -> bool {
    use std::f64::consts::TAU;
    if hits.len() < 3 {
        return false;
    }
    let mut angles: Vec<f64> = hits.iter().map(|(t, _)| t.rem_euclid(TAU)).collect();
    angles.sort_by(f64::total_cmp);
    #[allow(clippy::cast_precision_loss)]
    let expected = TAU / angles.len() as f64;
    // Relative slack so coarse polygons (few segments) and fine ones (many)
    // are both accepted; a clustered entry/exit pattern has a dominant gap
    // far from `expected` and fails.
    let slack = expected * 0.25;
    for i in 0..angles.len() {
        let next = if i + 1 == angles.len() {
            angles[0] + TAU
        } else {
            angles[i + 1]
        };
        if (next - angles[i] - expected).abs() > slack {
            return false;
        }
    }
    true
}

/// Emit N arc edges + `IntersectionCurveDS` entries for a closed-circle
/// section split at `crossings` (≥ 2 entries, sorted by circle parameter
/// `t`).
///
/// Each arc becomes its own `IntersectionCurveDS` so that the downstream
/// `build_section_edges` path sees N separate (open) section sources
/// instead of one closed curve. `build_section_edges` reconstructs each
/// section's start/end from the curve's `t_range`, so the per-arc t-range
/// here is what carries the split through.
///
/// Arcs whose midpoints fall outside the AABB of either face are dropped
/// — those portions of the original closed curve are not geometrically
/// "on" the face pair, so emitting them as sections would confuse the
/// face splitter.
#[allow(clippy::too_many_arguments)]
fn emit_split_circle_arcs(
    topo: &mut Topology,
    arena: &mut GfaArena,
    face_a: FaceId,
    face_b: FaceId,
    raw: &RawCurve,
    circle: &brepkit_math::curves::Circle3D,
    crossings: &[(f64, Point3)],
    tol: Tolerance,
) {
    // Compute AABBs for each face. For analytic faces the outer wire
    // alone doesn't enclose the face region (e.g. a sphere hemisphere has
    // its outer wire on the equator plane while the surface extends to
    // the pole) — so we union the wire AABB with the surface's AABB to
    // get a usable bounding region.
    let face_aabb = |fid: FaceId| -> Option<Aabb3> {
        let face = topo.face(fid).ok()?;
        let wire = topo.wire(face.outer_wire()).ok()?;
        let mut min = Point3::new(f64::INFINITY, f64::INFINITY, f64::INFINITY);
        let mut max = Point3::new(f64::NEG_INFINITY, f64::NEG_INFINITY, f64::NEG_INFINITY);
        let mut any = false;
        for oe in wire.edges() {
            if let Ok(edge) = topo.edge(oe.edge()) {
                let (Ok(sv), Ok(ev)) = (topo.vertex(edge.start()), topo.vertex(edge.end())) else {
                    continue;
                };
                let (sp, ep) = (sv.point(), ev.point());
                let (t0, t1) = edge.curve().domain_with_endpoints(sp, ep);
                // Sample along the curve, not just endpoints: a cylinder/cone
                // lateral face's circular edges are closed (start == end at the
                // seam vertex), so endpoint-only bounds collapse to a line at
                // the seam and the face's radial extent is lost — which then
                // wrongly rejects every in-face section-arc midpoint.
                let n = 8;
                for i in 0..=n {
                    let frac = f64::from(i) / f64::from(n);
                    let p = edge
                        .curve()
                        .evaluate_with_endpoints(t0 + (t1 - t0) * frac, sp, ep);
                    min = Point3::new(min.x().min(p.x()), min.y().min(p.y()), min.z().min(p.z()));
                    max = Point3::new(max.x().max(p.x()), max.y().max(p.y()), max.z().max(p.z()));
                    any = true;
                }
            }
        }
        if !any {
            return None;
        }
        // For analytic surfaces with finite extent, union the wire AABB
        // with the surface AABB. This expands a "degenerate-in-Z hemisphere
        // wire" (z=0 only) up to the actual hemisphere extent.
        if let FaceSurface::Sphere(sphere) = face.surface() {
            let r = sphere.radius();
            let c = sphere.center();
            min = Point3::new(
                min.x().min(c.x() - r),
                min.y().min(c.y() - r),
                min.z().min(c.z() - r),
            );
            max = Point3::new(
                max.x().max(c.x() + r),
                max.y().max(c.y() + r),
                max.z().max(c.z() + r),
            );
        }
        Some(Aabb3 { min, max }.expanded(tol.linear * 10.0))
    };
    let bbox_a = face_aabb(face_a);
    let bbox_b = face_aabb(face_b);

    // A sphere face's wire+surface AABB covers the whole ball, so the AABB
    // filter alone cannot tell its hemisphere from its twin's. Derive the
    // hemisphere axis from the boundary wire orientation (region lies to
    // the left of the wire under the outward surface normal): the sum of
    // normal x edge-direction over the wire points into the face's region.
    let sphere_side = |fid: FaceId| -> Option<(Point3, Vec3)> {
        let face = topo.face(fid).ok()?;
        let FaceSurface::Sphere(s) = face.surface() else {
            return None;
        };
        let center = s.center();
        sphere_region_axis(topo, fid, center, tol).map(|axis| (center, axis))
    };
    let side_a = sphere_side(face_a);
    let side_b = sphere_side(face_b);
    let side_eps = tol.linear * 10.0;
    let in_both = |p: Point3| -> bool {
        let a_ok = bbox_a.as_ref().is_none_or(|b| b.contains_point(p));
        let b_ok = bbox_b.as_ref().is_none_or(|b| b.contains_point(p));
        let side_ok = |side: &Option<(Point3, Vec3)>| {
            side.as_ref()
                .is_none_or(|(c, axis)| (p - *c).dot(*axis) >= -side_eps)
        };
        a_ok && b_ok && side_ok(&side_a) && side_ok(&side_b)
    };

    // Pass 1: determine which arcs survive the AABB filter, *before*
    // allocating any topology vertices. Otherwise dropping an arc could
    // leave its crossing vertices orphaned in the topology (no edges
    // reference them) — visible to any downstream pass that iterates all
    // vertices. We also split each surviving arc into sub-arcs of span
    // ≤ π so the resulting `EdgeCurve::Circle` is unambiguous: several
    // downstream consumers (tessellation's `shorter_arc_range` in
    // `tessellate/mod.rs`, wire sampling in `topology/builder.rs`)
    // interpret an open circle edge as the *shorter* arc between its
    // endpoints, so a span > π would be flipped to the complementary
    // arc and break face splitting/classification.
    //
    // Each survivor records the inserted crossing points along with
    // their (t, 3D) values; pass 2 then materialises just those
    // vertices (in order) and builds the edges.
    //
    // Each point is `(t, 3D point, original-crossing-index-or-None)`.
    // Endpoints carry `Some(crossing_index)` so we can dedup with
    // adjacent arcs sharing the same crossing vertex; midpoints
    // (inserted to keep span ≤ π) carry `None` and always get a fresh
    // vertex allocation.
    let n = crossings.len();
    let mut survivors: Vec<Vec<(f64, Point3, Option<usize>)>> = Vec::new();
    for i in 0..n {
        let (t0, _p0) = crossings[i];
        // Wrap to next crossing; for the last arc that means going back to
        // the first crossing through the seam (t0 → t1 + 2π).
        let next_i = (i + 1) % n;
        let (mut t1, _p1) = crossings[next_i];
        if t1 <= t0 {
            t1 += std::f64::consts::TAU;
        }

        let t_mid = (t0 + t1) * 0.5;
        let mid_3d = circle.evaluate(t_mid);
        if !in_both(mid_3d) {
            log::debug!(
                "FF: drop closed-circle arc {i}/{n} (midpoint {mid_3d:?} outside face pair)"
            );
            continue;
        }

        // Split into sub-arcs each spanning STRICTLY less than π by inserting
        // midpoint vertices at evenly spaced t-values. A span of exactly π is a
        // diametric semicircle: its two endpoints cannot distinguish it from its
        // complement, so the downstream "shorter arc" interpretation is
        // ambiguous AND the assembler's endpoint-keyed edge merge would collapse
        // two distinct semicircles (e.g. the north/south halves of a sphere's
        // section circle) into one edge. Dividing by a hair under π forces a
        // midpoint vertex for any span ≥ π, giving each piece distinct
        // endpoints.
        let arc_span = t1 - t0;
        let n_sub = (arc_span / (std::f64::consts::PI * 0.999)).ceil().max(1.0) as usize;
        let step = arc_span / n_sub as f64;
        let mut points: Vec<(f64, Point3, Option<usize>)> = Vec::with_capacity(n_sub + 1);
        points.push((t0, crossings[i].1, Some(i)));
        for k in 1..n_sub {
            let tk = t0 + step * k as f64;
            points.push((tk, circle.evaluate(tk), None));
        }
        points.push((t1, crossings[next_i].1, Some(next_i)));

        survivors.push(points);
    }

    // Pass 2: allocate vertices only for crossings/midpoints used by
    // surviving arcs, then materialise edges + pave blocks.
    let mut crossing_vids: Vec<Option<brepkit_topology::vertex::VertexId>> = vec![None; n];
    let resolve_crossing =
        |topo: &mut Topology, arena: &GfaArena, p: Point3| -> brepkit_topology::vertex::VertexId {
            super::helpers::find_nearby_pave_vertex(topo, arena, p, tol)
                .or_else(|| find_nearby_face_vertex(topo, face_a, p, tol))
                .or_else(|| find_nearby_face_vertex(topo, face_b, p, tol))
                .unwrap_or_else(|| topo.add_vertex(Vertex::new(p, tol.linear)))
        };

    // A geometrically identical arc may already exist from an earlier pair
    // sharing a face (e.g. both hemispheres of a faceted sphere paired with
    // the same coplanar box face yield the same equator arc). Re-emitting it
    // would hand the shared face two coincident sections and corrupt its
    // split. First pair wins; the later pair contributes nothing new to the
    // shared face anyway (the circle lies along its twin's region boundary).
    let close_pts = |a: Point3, b: Point3| (a - b).length() < tol.linear * 10.0;
    let arc_exists = |arena: &GfaArena, p_s: Point3, p_e: Point3, p_m: Point3| -> bool {
        arena.curves.iter().any(|c| {
            let EdgeCurve::Circle(existing) = &c.curve else {
                return false;
            };
            let shares_face = c.face_a == face_a
                || c.face_a == face_b
                || c.face_b == face_a
                || c.face_b == face_b;
            if !shares_face
                || (existing.center() - circle.center()).length() > tol.linear * 10.0
                || (existing.radius() - circle.radius()).abs() > tol.linear * 10.0
            {
                return false;
            }
            let s = existing.evaluate(c.t_range.0);
            let e = existing.evaluate(c.t_range.1);
            let m = existing.evaluate(0.5 * (c.t_range.0 + c.t_range.1));
            close_pts(m, p_m)
                && ((close_pts(s, p_s) && close_pts(e, p_e))
                    || (close_pts(s, p_e) && close_pts(e, p_s)))
        })
    };

    let mut emitted = 0_usize;
    let num_survivors = survivors.len();
    for (a_idx, arc_points) in survivors.into_iter().enumerate() {
        // Walk consecutive (t, point) pairs to emit one edge per sub-arc.
        for w in arc_points.windows(2) {
            let (t_s, p_s, idx_s) = w[0];
            let (t_e, p_e, idx_e) = w[1];

            let p_m = circle.evaluate(0.5 * (t_s + t_e));
            if arc_exists(arena, p_s, p_e, p_m) {
                log::debug!(
                    "FF: skip duplicate closed-circle arc t=[{t_s:.4},{t_e:.4}] \
                     for {face_a:?}/{face_b:?}"
                );
                continue;
            }

            let start_vid = match idx_s {
                Some(ci) => {
                    *crossing_vids[ci].get_or_insert_with(|| resolve_crossing(topo, arena, p_s))
                }
                None => topo.add_vertex(Vertex::new(p_s, tol.linear)),
            };
            let end_vid = match idx_e {
                Some(ci) => {
                    *crossing_vids[ci].get_or_insert_with(|| resolve_crossing(topo, arena, p_e))
                }
                None => topo.add_vertex(Vertex::new(p_e, tol.linear)),
            };

            let edge = Edge::new(start_vid, end_vid, EdgeCurve::Circle(circle.clone()));
            let edge_id = topo.add_edge(edge);

            let start_pave = Pave::new(start_vid, t_s);
            let end_pave = Pave::new(end_vid, t_e);
            let pb = PaveBlock::new(edge_id, start_pave, end_pave);
            let pb_id = arena.pave_blocks.alloc(pb);

            let curve_index = arena.curves.len();
            arena.curves.push(IntersectionCurveDS {
                curve: EdgeCurve::Circle(circle.clone()),
                face_a,
                face_b,
                // The full-circle bbox is a safe over-approximation for any arc.
                bbox: raw.bbox,
                pave_blocks: vec![pb_id],
                t_range: (t_s, t_e),
            });

            arena.interference.ff.push(Interference::FF {
                f1: face_a,
                f2: face_b,
                curve_index,
            });
            emitted += 1;

            log::debug!(
                "FF: split closed circle arc {a_idx}/{num_survivors}: \
                 faces {face_a:?}/{face_b:?} t=[{t_s:.4},{t_e:.4}] \
                 edge={edge_id:?} pb={pb_id:?}"
            );
        }
    }

    log::debug!("FF: emitted {emitted}/{n} arcs after AABB filter for {face_a:?}/{face_b:?}");
}

/// Compute AABB for a NURBS curve.
fn nurbs_curve_bbox(curve: &brepkit_math::nurbs::curve::NurbsCurve) -> Aabb3 {
    let (t0, t1) = curve.domain();
    let n: usize = 32;
    let points: Vec<Point3> = (0..=n)
        .map(|i| {
            let t = t0 + (t1 - t0) * (i as f64 / n as f64);
            ParametricCurve::evaluate(curve, t)
        })
        .collect();
    Aabb3::from_points(points)
}

// ── FF curve boundary filtering ──────────────────────────────────────

/// Apply the plane-side polygon clips to a band-trimmed Line raw curve.
/// Mirrors the plane×plane combination arms for a curve that already went
/// through the exact band-window trim.
fn clip_trimmed_line_to_planes(
    topo: &Topology,
    fa: FaceId,
    fb: FaceId,
    raw: RawCurve,
    tol: Tolerance,
) -> Option<RawCurve> {
    let clip_a = clip_line_to_face(topo, fa, &raw);
    let clip_b = clip_line_to_face(topo, fb, &raw);
    match (clip_a, clip_b) {
        (FaceClip::Empty, _) | (_, FaceClip::Empty) => None,
        (FaceClip::Range(a), FaceClip::Range(b)) => {
            trim_raw_line(&raw, a.0.max(b.0), a.1.min(b.1), tol)
        }
        (FaceClip::Range(r), FaceClip::Indeterminate)
        | (FaceClip::Indeterminate, FaceClip::Range(r)) => trim_raw_line(&raw, r.0, r.1, tol),
        (FaceClip::Indeterminate, FaceClip::Indeterminate) => Some(raw),
    }
}

/// Shrink a `Line` raw curve to the fractional sub-range `[f0, f1]` of its
/// current extent, recomputing endpoints, parameter range, and bbox.
///
/// Returns `None` when the trimmed segment is shorter than `tol.linear`
/// (touching or disjoint clip ranges — no real overlap).
fn trim_raw_line(raw: &RawCurve, f0: f64, f1: f64, tol: Tolerance) -> Option<RawCurve> {
    let span = raw.t_range.1 - raw.t_range.0;
    let t0 = raw.t_range.0 + f0 * span;
    let t1 = raw.t_range.0 + f1 * span;
    let seg = raw.p_end - raw.p_start;
    if (f1 - f0) * seg.length() < tol.linear {
        return None;
    }
    let p0 = raw.p_start + seg * f0;
    let p1 = raw.p_start + seg * f1;
    let bbox = Aabb3 {
        min: Point3::new(p0.x().min(p1.x()), p0.y().min(p1.y()), p0.z().min(p1.z())),
        max: Point3::new(p0.x().max(p1.x()), p0.y().max(p1.y()), p0.z().max(p1.z())),
    };
    Some(RawCurve {
        curve: raw.curve.clone(),
        bbox,
        t_range: (t0, t1),
        p_start: p0,
        p_end: p1,
    })
}

/// Outcome of clipping a Line raw curve to a planar face's outer polygon.
///
/// The two "no overlap" causes are kept distinct: `Empty` means a valid
/// convex polygon was built and the line provably lies outside it (the
/// section should be dropped), while `Indeterminate` means the polygon
/// could not be built or is non-convex (the caller should conservatively
/// keep the untrimmed curve, since the Cyrus-Beck clip is only correct
/// for convex polygons).
enum FaceClip {
    /// Fractional `[t_min, t_max]` sub-range of the raw line inside the face.
    Range((f64, f64)),
    /// Convex polygon built; line lies entirely outside it.
    Empty,
    /// Polygon could not be built or is non-convex — keep raw curve.
    Indeterminate,
}

/// Clip a Line curve to a planar face's boundary polygon.
fn clip_line_to_face(topo: &Topology, face_id: FaceId, raw: &RawCurve) -> FaceClip {
    let Ok(face) = topo.face(face_id) else {
        return FaceClip::Indeterminate;
    };
    let FaceSurface::Plane { normal, .. } = face.surface() else {
        return FaceClip::Indeterminate;
    };
    let Ok(wire) = topo.wire(face.outer_wire()) else {
        return FaceClip::Indeterminate;
    };
    if !wire.edges().iter().all(|oe| {
        topo.edge(oe.edge())
            .is_ok_and(|e| matches!(e.curve(), EdgeCurve::Line))
    }) {
        return FaceClip::Indeterminate;
    }

    // Chain edges by shared vertex IDs rather than trusting stored
    // orientation flags — wires from external builders may carry
    // inconsistent `is_forward` flags, and a mis-ordered polygon makes
    // the clip silently truncate the range.
    let mut remaining: Vec<(
        brepkit_topology::vertex::VertexId,
        brepkit_topology::vertex::VertexId,
    )> = Vec::new();
    for oe in wire.edges() {
        let Ok(edge) = topo.edge(oe.edge()) else {
            return FaceClip::Indeterminate;
        };
        remaining.push((edge.start(), edge.end()));
    }
    if remaining.len() < 3 {
        return FaceClip::Indeterminate;
    }
    let (first_start, first_end) = remaining.swap_remove(0);
    let mut vert_ids = vec![first_start, first_end];
    while !remaining.is_empty() {
        let Some(&cur) = vert_ids.last() else {
            return FaceClip::Indeterminate;
        };
        let Some(pos) = remaining.iter().position(|&(s, e)| s == cur || e == cur) else {
            return FaceClip::Indeterminate;
        };
        let (s, e) = remaining.swap_remove(pos);
        vert_ids.push(if s == cur { e } else { s });
    }
    if vert_ids.first() == vert_ids.last() {
        vert_ids.pop();
    }
    let mut verts = Vec::with_capacity(vert_ids.len());
    for vid in vert_ids {
        let Ok(v) = topo.vertex(vid) else {
            return FaceClip::Indeterminate;
        };
        verts.push(v.point());
    }
    if verts.len() < 3 {
        return FaceClip::Indeterminate;
    }

    let frame =
        super::super::builder::plane_frame::PlaneFrame::from_normal_and_point(*normal, verts[0]);
    let poly: Vec<(f64, f64)> = verts
        .iter()
        .map(|v| {
            let uv = frame.project(*v);
            (uv.x(), uv.y())
        })
        .collect();
    let s = frame.project(raw.p_start);
    let e = frame.project(raw.p_end);
    // Cyrus-Beck is only correct for convex polygons. For a non-convex
    // outline (e.g. a faceted scoop-ramp side face, whose profile is a
    // staircase, or a notched cavity floor) use the general crossing-based
    // clip so the section is still trimmed to the face's true extent. Leaving
    // it untrimmed lets a perpendicular plane×plane section span the union of
    // both faces' bounding boxes and cross a rounded-rect corner arc mid-edge,
    // which forces the downstream planar arrangement to bail.
    if !polygon_is_convex(&poly) {
        return match clip_line_to_polygon_general((s.x(), s.y()), (e.x(), e.y()), &poly) {
            Some(range) => FaceClip::Range(range),
            None => FaceClip::Empty,
        };
    }
    match clip_line_to_polygon((s.x(), s.y()), (e.x(), e.y()), &poly) {
        Some(range) => FaceClip::Range(range),
        None => FaceClip::Empty,
    }
}

/// Test whether a simple polygon is convex via a signed-cross-product
/// sweep: all consecutive edge turns must share the same sign.
///
/// Collinear vertices (zero cross product) are tolerated. Degenerate
/// polygons (< 3 vertices) are reported non-convex.
fn polygon_is_convex(polygon: &[(f64, f64)]) -> bool {
    let n = polygon.len();
    if n < 3 {
        return false;
    }
    let mut sign: i32 = 0;
    for i in 0..n {
        let a = polygon[i];
        let b = polygon[(i + 1) % n];
        let c = polygon[(i + 2) % n];
        let cross = (b.0 - a.0) * (c.1 - b.1) - (b.1 - a.1) * (c.0 - b.0);
        if cross.abs() < 1e-12 {
            continue;
        }
        let s = if cross > 0.0 { 1 } else { -1 };
        if sign == 0 {
            sign = s;
        } else if s != sign {
            return false;
        }
    }
    true
}

/// Cyrus-Beck line-polygon clipping. Handles CCW and CW winding.
///
/// Only correct for convex polygons — callers must gate non-convex
/// outlines (see [`polygon_is_convex`]).
fn clip_line_to_polygon(
    start: (f64, f64),
    end: (f64, f64),
    polygon: &[(f64, f64)],
) -> Option<(f64, f64)> {
    let n = polygon.len();
    if n < 3 {
        return None;
    }
    let dx = end.0 - start.0;
    let dy = end.1 - start.1;
    let area2: f64 = (0..n)
        .map(|i| {
            let j = (i + 1) % n;
            polygon[i].0 * polygon[j].1 - polygon[j].0 * polygon[i].1
        })
        .sum();
    let sign = if area2 >= 0.0 { 1.0 } else { -1.0 };
    let d_len = dx.hypot(dy);
    if d_len < 1e-12 {
        return None;
    }
    let mut t_min = 0.0_f64;
    let mut t_max = 1.0_f64;
    for i in 0..n {
        let j = (i + 1) % n;
        let ex = polygon[j].0 - polygon[i].0;
        let ey = polygon[j].1 - polygon[i].1;
        let nx = -ey * sign;
        let ny = ex * sign;
        let denom = nx * dx + ny * dy;
        let num = nx * (start.0 - polygon[i].0) + ny * (start.1 - polygon[i].1);
        // Parallelism must be judged relative to |n|·|d|: `denom` is an
        // unnormalized dot product, so an absolute epsilon misreads a section
        // line COLLINEAR with a polygon edge (a coplanar partner face meeting
        // the clip face exactly along that edge — e.g. a lofted wall's top
        // chord lying in the partner cap's plane) as a genuine crossing. The
        // ratio −num/denom of two roundoff residues then clips the span to a
        // garbage sliver or empties it, and which of the two happens varies
        // per edge — nondeterministic partial section emission.
        let n_len = nx.hypot(ny);
        if n_len < 1e-12 {
            continue;
        }
        if denom.abs() < n_len * d_len * 1e-9 {
            // A near-parallel segment can still drift across the edge by up
            // to d_len·1e-9 over its length, so dropping on the start point
            // alone would discard a segment that genuinely enters the face —
            // reject only when BOTH endpoints sit outside the band
            // (num + denom is the end point's signed offset).
            if num < -n_len * 1e-9 && num + denom < -n_len * 1e-9 {
                return None;
            }
            continue;
        }
        let t = -num / denom;
        if denom > 0.0 {
            t_min = t_min.max(t);
        } else {
            t_max = t_max.min(t);
        }
        if t_min > t_max + 1e-6 {
            return None;
        }
    }
    if t_max - t_min < 1e-6 {
        return None;
    }
    Some((t_min.max(0.0), t_max.min(1.0)))
}

/// Clip a line segment `start`→`end` to an arbitrary (possibly non-convex)
/// simple polygon, returning the fractional `[t_min, t_max]` range that bounds
/// the in-polygon portion(s), or `None` when the segment never enters the
/// polygon.
///
/// The intersection of a line with a non-convex polygon can be several disjoint
/// intervals; this returns their convex hull (first entry to last exit). That
/// is exactly what section trimming needs — a conservative single span that no
/// longer over-reaches past the face, while still covering every in-face part.
/// Crossings are found at each polygon edge; the parametric midpoints between
/// consecutive crossings are classified by a point-in-polygon test so that
/// grazing/collinear touches do not spuriously open an interval.
fn clip_line_to_polygon_general(
    start: (f64, f64),
    end: (f64, f64),
    polygon: &[(f64, f64)],
) -> Option<(f64, f64)> {
    use brepkit_math::predicates::point_in_polygon;
    use brepkit_math::vec::Point2;

    let n = polygon.len();
    if n < 3 {
        return None;
    }
    let dx = end.0 - start.0;
    let dy = end.1 - start.1;
    if dx.hypot(dy) < 1e-12 {
        return None;
    }

    // Collect t-parameters along the segment where it crosses a polygon edge.
    let mut ts: Vec<f64> = vec![0.0, 1.0];
    for i in 0..n {
        let j = (i + 1) % n;
        let (ax, ay) = polygon[i];
        let (bx, by) = polygon[j];
        let ex = bx - ax;
        let ey = by - ay;
        // Solve start + t*d = a + u*e for t (segment param) and u (edge param).
        let denom = dx * ey - dy * ex;
        if denom.abs() < 1e-15 {
            continue; // parallel
        }
        let t = ((ax - start.0) * ey - (ay - start.1) * ex) / denom;
        let u = ((ax - start.0) * dy - (ay - start.1) * dx) / denom;
        if (-1e-9..=1.0 + 1e-9).contains(&u) {
            ts.push(t.clamp(0.0, 1.0));
        }
    }
    ts.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    ts.dedup_by(|a, b| (*a - *b).abs() < 1e-9);

    let poly_pts: Vec<Point2> = polygon.iter().map(|&(x, y)| Point2::new(x, y)).collect();
    let mut lo = f64::MAX;
    let mut hi = f64::MIN;
    for w in ts.windows(2) {
        let (ta, tb) = (w[0], w[1]);
        if tb - ta < 1e-9 {
            continue;
        }
        let tm = 0.5 * (ta + tb);
        let mid = Point2::new(start.0 + dx * tm, start.1 + dy * tm);
        if point_in_polygon(mid, &poly_pts) {
            lo = lo.min(ta);
            hi = hi.max(tb);
        }
    }
    if hi - lo < 1e-6 {
        return None;
    }
    Some((lo.max(0.0), hi.min(1.0)))
}

#[cfg(test)]
mod tests {
    #![allow(clippy::unwrap_used, clippy::expect_used)]
    use super::*;

    #[test]
    fn longest_run_open_middle() {
        // Open curve: longest contiguous in-both run, no wrap-around.
        let inb = [false, true, true, true, false, true, false];
        assert_eq!(longest_inboth_run(&inb, false), (1, 3));
    }

    #[test]
    fn longest_run_closed_wraps_seam() {
        // Closed curve (sample N duplicates sample 0): in-both at samples 4, 0,
        // 1 — the longest run wraps the seam, so b1 extends past the distinct
        // sample count (the caller maps it back via periodic parameterization).
        let inb = [true, true, false, false, true, true];
        assert_eq!(longest_inboth_run(&inb, true), (4, 6));
    }

    #[test]
    fn longest_run_closed_whole() {
        // A closed curve entirely in-both returns the whole span (0, m).
        let inb = [true, true, true, true, true];
        assert_eq!(longest_inboth_run(&inb, true), (0, 4));
    }

    #[test]
    fn trim_partial_ellipse_to_single_open_arc() {
        // A closed Ellipse whose in-both run is a genuine NON-wrapping partial
        // (b0..b1 strictly inside [0, N]) trims to ONE open arc with those exact
        // parameters — NOT kept whole. Domain [0, 2π], N=24, run = samples 6..18.
        use brepkit_math::curves::Ellipse3D;
        let e = Ellipse3D::new(
            Point3::new(0.0, 0.0, 0.0),
            Vec3::new(0.0, 0.0, 1.0),
            4.0,
            3.0,
        )
        .unwrap();
        let raw = RawCurve {
            curve: EdgeCurve::Ellipse(e),
            bbox: Aabb3 {
                min: Point3::new(-4.0, -3.0, 0.0),
                max: Point3::new(4.0, 3.0, 0.0),
            },
            t_range: (0.0, std::f64::consts::TAU),
            p_start: Point3::new(0.0, 0.0, 0.0),
            p_end: Point3::new(0.0, 0.0, 0.0),
        };
        let out = trim_closed_curve_to_inboth_arc(&raw, 6, 18, 24);
        assert_eq!(out.len(), 1, "non-wrapping partial → one open arc");
        let tau = std::f64::consts::TAU;
        assert!((out[0].t_range.0 - tau * 6.0 / 24.0).abs() < 1e-9);
        assert!((out[0].t_range.1 - tau * 18.0 / 24.0).abs() < 1e-9);
        // The arc is OPEN (endpoints differ), so the splitter trims it rather
        // than treating it as a closed internal loop.
        assert!((out[0].p_start - out[0].p_end).length() > 1e-6);
    }

    #[test]
    fn trim_wrapping_nurbs_preserves_both_pieces() {
        // A closed NURBS whose in-both run WRAPS the seam (b1 > N) must emit TWO
        // in-domain arcs (head [t0, domain_end] + tail [domain_start, t1−span]),
        // NOT a clamp-dropped single arc, since a clamped NURBS can't evaluate
        // past its domain. Run = samples 22..28 (i.e. 22,23,0,1,2,3,4 wrapping),
        // N=24, domain [0, 1].
        use brepkit_math::nurbs::fitting::interpolate;
        // A closed NURBS through 8 points around a circle (start == end).
        let mut pts: Vec<Point3> = (0..8)
            .map(|k| {
                let a = std::f64::consts::TAU * f64::from(k) / 8.0;
                Point3::new(3.0 * a.cos(), 3.0 * a.sin(), 0.0)
            })
            .collect();
        pts.push(pts[0]);
        let nurbs = interpolate(&pts, 3).unwrap();
        let (d0, d1) = nurbs.domain();
        let raw = RawCurve {
            curve: EdgeCurve::NurbsCurve(nurbs),
            bbox: Aabb3 {
                min: Point3::new(-3.0, -3.0, 0.0),
                max: Point3::new(3.0, 3.0, 0.0),
            },
            t_range: (d0, d1),
            p_start: pts[0],
            p_end: pts[0],
        };
        let out = trim_closed_curve_to_inboth_arc(&raw, 22, 28, 24);
        assert_eq!(out.len(), 2, "wrapping NURBS run → two in-domain arcs");
        // Both arcs stay within the curve's domain (no out-of-domain garbage).
        for arc in &out {
            assert!(arc.t_range.0 >= d0 - 1e-9 && arc.t_range.0 <= d1 + 1e-9);
            assert!(arc.t_range.1 >= d0 - 1e-9 && arc.t_range.1 <= d1 + 1e-9);
        }
        // Head ends at the domain end; tail starts at the domain start — they
        // join at the seam where the closed curve's endpoints coincide.
        assert!((out[0].t_range.1 - d1).abs() < 1e-9);
        assert!((out[1].t_range.0 - d0).abs() < 1e-9);
    }

    #[test]
    fn sphere_region_axis_closed_circle_boundary() {
        // A sphere face bounded by a single closed `Circle` edge has start ==
        // end, so endpoint-only winding gives a zero chord and no axis. Sampling
        // along the edge must recover the circle's plane normal as the pole axis
        // (otherwise the broad-phase falls back to the loose full-sphere box).
        use brepkit_math::curves::Circle3D;
        use brepkit_math::surfaces::SphericalSurface;
        use brepkit_topology::face::Face;
        use brepkit_topology::wire::{OrientedEdge, Wire};
        let mut topo = Topology::default();
        let v0 = topo.add_vertex(Vertex::new(Point3::new(6.0, 0.0, 0.0), 1e-7));
        let circle =
            Circle3D::new(Point3::new(0.0, 0.0, 0.0), Vec3::new(0.0, 0.0, 1.0), 6.0).unwrap();
        let edge = topo.add_edge(Edge::new(v0, v0, EdgeCurve::Circle(circle)));
        let wire = topo.add_wire(Wire::new(vec![OrientedEdge::new(edge, true)], true).unwrap());
        let sphere = SphericalSurface::new(Point3::new(0.0, 0.0, 0.0), 6.0).unwrap();
        let face = topo.add_face(Face::new(wire, vec![], FaceSurface::Sphere(sphere)));
        let axis = sphere_region_axis(
            &topo,
            face,
            Point3::new(0.0, 0.0, 0.0),
            Tolerance::default(),
        )
        .expect("closed-circle boundary should yield a pole axis");
        // The equatorial circle's plane normal is ±z; sampling recovers it.
        assert!(axis.z().abs() > 0.99, "axis not aligned with z: {axis:?}");
        assert!(axis.x().abs() < 0.05 && axis.y().abs() < 0.05);
    }

    #[test]
    fn closed_circle_torus_boundary_is_not_full_torus() {
        // A torus patch bounded by a closed `Circle` edge has coincident
        // endpoints but real spatial extent — it must NOT be treated as a full
        // (untrimmed) torus, which would over-widen its AABB to the whole torus.
        use brepkit_math::curves::Circle3D;
        use brepkit_math::surfaces::ToroidalSurface;
        use brepkit_topology::face::Face;
        use brepkit_topology::wire::{OrientedEdge, Wire};
        let mut topo = Topology::default();
        let v0 = topo.add_vertex(Vertex::new(Point3::new(13.0, 0.0, 0.0), 1e-7));
        // A tube cross-section circle at u=0: centered on the tube center
        // (10,0,0), in the x-z plane, radius = minor radius 3.
        let circle =
            Circle3D::new(Point3::new(10.0, 0.0, 0.0), Vec3::new(0.0, 1.0, 0.0), 3.0).unwrap();
        let edge = topo.add_edge(Edge::new(v0, v0, EdgeCurve::Circle(circle)));
        let wire = topo.add_wire(Wire::new(vec![OrientedEdge::new(edge, true)], true).unwrap());
        let torus = ToroidalSurface::new(Point3::new(0.0, 0.0, 0.0), 10.0, 3.0).unwrap();
        let face = topo.add_face(Face::new(wire, vec![], FaceSurface::Torus(torus)));
        assert!(!face_boundary_all_degenerate(&topo, face, Tolerance::default()).unwrap());
    }

    #[test]
    fn point_seam_boundary_is_full_torus() {
        // Degenerate `Line(v0, v0)` seam edges (zero extent) are the full-torus
        // signature and must return true.
        use brepkit_math::surfaces::ToroidalSurface;
        use brepkit_topology::face::Face;
        use brepkit_topology::wire::{OrientedEdge, Wire};
        let mut topo = Topology::default();
        let v0 = topo.add_vertex(Vertex::new(Point3::new(13.0, 0.0, 0.0), 1e-7));
        let e0 = topo.add_edge(Edge::new(v0, v0, EdgeCurve::Line));
        let e1 = topo.add_edge(Edge::new(v0, v0, EdgeCurve::Line));
        let wire = topo.add_wire(
            Wire::new(
                vec![OrientedEdge::new(e0, true), OrientedEdge::new(e1, true)],
                true,
            )
            .unwrap(),
        );
        let torus = ToroidalSurface::new(Point3::new(0.0, 0.0, 0.0), 10.0, 3.0).unwrap();
        let face = topo.add_face(Face::new(wire, vec![], FaceSurface::Torus(torus)));
        assert!(face_boundary_all_degenerate(&topo, face, Tolerance::default()).unwrap());
    }

    #[test]
    fn clip_inside_square() {
        let poly = vec![(0.0, 0.0), (1.0, 0.0), (1.0, 1.0), (0.0, 1.0)];
        let r = clip_line_to_polygon((0.2, 0.5), (0.8, 0.5), &poly).unwrap();
        assert!((r.0).abs() < 1e-6 && (r.1 - 1.0).abs() < 1e-6);
    }

    #[test]
    fn clip_crossing() {
        let poly = vec![(0.0, 0.0), (1.0, 0.0), (1.0, 1.0), (0.0, 1.0)];
        let r = clip_line_to_polygon((-1.0, 0.5), (2.0, 0.5), &poly).unwrap();
        assert!((r.0 - 1.0 / 3.0).abs() < 1e-6);
        assert!((r.1 - 2.0 / 3.0).abs() < 1e-6);
    }

    #[test]
    fn clip_outside() {
        let poly = vec![(0.0, 0.0), (1.0, 0.0), (1.0, 1.0), (0.0, 1.0)];
        assert!(clip_line_to_polygon((2.0, 0.5), (3.0, 0.5), &poly).is_none());
    }

    #[test]
    fn clip_collinear_with_edge_keeps_full_span() {
        // Roundoff-scale residues off the bottom edge of a scale-100 square:
        // the natural scale of the unnormalized denom is |n|·|d| ≈ 8000, so
        // an absolute parallel epsilon reads this as a genuine crossing and
        // clips the span to the ratio of two residues (t_max = 0.5 here).
        let poly = vec![(0.0, 0.0), (100.0, 0.0), (100.0, 100.0), (0.0, 100.0)];
        let r = clip_line_to_polygon((10.0, 1e-13), (90.0, -1e-13), &poly).unwrap();
        assert!(r.0.abs() < 1e-9 && (r.1 - 1.0).abs() < 1e-9);
    }

    #[test]
    fn clip_near_parallel_entering_segment_is_kept() {
        // Within the parallel band (sin(angle) < 1e-9) a long segment can
        // still drift across the edge: start 2e-8 outside, end 2e-8 inside.
        // Rejecting on the start point alone would drop it entirely.
        let poly = vec![(0.0, 0.0), (100.0, 0.0), (100.0, 100.0), (0.0, 100.0)];
        let r = clip_line_to_polygon((10.0, -2e-8), (90.0, 2e-8), &poly).unwrap();
        assert!((r.1 - 1.0).abs() < 1e-9);
    }

    #[test]
    fn clip_parallel_outside_edge_is_dropped() {
        let poly = vec![(0.0, 0.0), (100.0, 0.0), (100.0, 100.0), (0.0, 100.0)];
        assert!(clip_line_to_polygon((10.0, -0.5), (90.0, -0.5), &poly).is_none());
    }

    #[test]
    fn clip_zero_length_segment_is_dropped() {
        let poly = vec![(0.0, 0.0), (1.0, 0.0), (1.0, 1.0), (0.0, 1.0)];
        assert!(clip_line_to_polygon((0.5, 0.5), (0.5, 0.5), &poly).is_none());
    }

    #[test]
    fn clip_cw_polygon() {
        let poly = vec![(0.0, 1.0), (1.0, 1.0), (1.0, 0.0), (0.0, 0.0)];
        let r = clip_line_to_polygon((-1.0, 0.5), (2.0, 0.5), &poly).unwrap();
        assert!((r.0 - 1.0 / 3.0).abs() < 1e-6);
        assert!((r.1 - 2.0 / 3.0).abs() < 1e-6);
    }

    #[test]
    fn clip_outside_means_drop() {
        // A line provably outside a built (convex) polygon yields `None`,
        // which the FF trim path maps to `FaceClip::Empty` → drop the curve.
        let poly = vec![(0.0, 0.0), (1.0, 0.0), (1.0, 1.0), (0.0, 1.0)];
        assert!(clip_line_to_polygon((2.0, 0.5), (3.0, 0.5), &poly).is_none());
    }

    #[test]
    fn convex_square_is_convex() {
        let poly = vec![(0.0, 0.0), (1.0, 0.0), (1.0, 1.0), (0.0, 1.0)];
        assert!(polygon_is_convex(&poly));
    }

    #[test]
    fn convex_check_tolerates_collinear() {
        let poly = vec![(0.0, 0.0), (0.5, 0.0), (1.0, 0.0), (1.0, 1.0), (0.0, 1.0)];
        assert!(polygon_is_convex(&poly));
    }

    #[test]
    fn arrow_polygon_is_non_convex() {
        // A concave "arrowhead": the reflex vertex flips the turn sign.
        let poly = vec![(0.0, 0.0), (2.0, 1.0), (0.0, 2.0), (1.0, 1.0)];
        assert!(!polygon_is_convex(&poly));
    }

    #[test]
    fn general_clip_trims_to_nonconvex_extent() {
        // An L-shaped (non-convex) polygon: the staircase profile of a scoop
        // side face. A horizontal section line spanning well past the polygon
        // must be trimmed to the polygon's actual x-extent, not the line's full
        // length — this is what stops a perpendicular plane×plane section from
        // over-reaching across a rounded-rect corner arc.
        // L: (0,0)-(2,0)-(2,1)-(1,1)-(1,2)-(0,2)
        let poly = vec![
            (0.0, 0.0),
            (2.0, 0.0),
            (2.0, 1.0),
            (1.0, 1.0),
            (1.0, 2.0),
            (0.0, 2.0),
        ];
        // Line at y=0.5 (in the wide lower arm): inside for x in [0,2].
        let r = clip_line_to_polygon_general((-5.0, 0.5), (5.0, 0.5), &poly).unwrap();
        let x0 = -5.0 + 10.0 * r.0;
        let x1 = -5.0 + 10.0 * r.1;
        assert!((x0 - 0.0).abs() < 1e-6, "x0={x0}");
        assert!((x1 - 2.0).abs() < 1e-6, "x1={x1}");

        // Line at y=1.5 (in the narrow upper arm): inside only for x in [0,1].
        let r = clip_line_to_polygon_general((-5.0, 1.5), (5.0, 1.5), &poly).unwrap();
        let x0 = -5.0 + 10.0 * r.0;
        let x1 = -5.0 + 10.0 * r.1;
        assert!((x0 - 0.0).abs() < 1e-6, "x0={x0}");
        assert!((x1 - 1.0).abs() < 1e-6, "x1={x1}");

        // A line entirely outside returns None.
        assert!(clip_line_to_polygon_general((-5.0, 3.0), (5.0, 3.0), &poly).is_none());
    }

    fn hit_at(angle: f64) -> (f64, Point3) {
        (angle, Point3::new(angle.cos(), angle.sin(), 0.0))
    }

    #[test]
    fn inscribed_square_equator_is_coincident() {
        // A 4-segment polygon inscribed in the circle (square equator)
        // yields exactly 4 evenly distributed hits — the bare `len > 4`
        // count missed this, treating the boundary as entry/exit crossings.
        use std::f64::consts::FRAC_PI_2;
        let hits: Vec<_> = (0..4).map(|k| hit_at(k as f64 * FRAC_PI_2)).collect();
        assert!(hits_are_inscribed_polygon(&hits));
    }

    #[test]
    fn inscribed_hexagon_is_coincident() {
        use std::f64::consts::FRAC_PI_3;
        let hits: Vec<_> = (0..6).map(|k| hit_at(k as f64 * FRAC_PI_3)).collect();
        assert!(hits_are_inscribed_polygon(&hits));
    }

    #[test]
    fn entry_exit_pair_is_not_coincident() {
        // Two crossings (genuine entry/exit) must not be mistaken for an
        // inscribed boundary.
        let hits = vec![hit_at(0.3), hit_at(2.9)];
        assert!(!hits_are_inscribed_polygon(&hits));
    }

    #[test]
    fn clustered_hits_are_not_coincident() {
        // Four hits clustered on one side leave a dominant wrap-around gap,
        // far from the even spacing of an inscribed polygon.
        let hits = vec![hit_at(0.1), hit_at(0.2), hit_at(0.3), hit_at(0.4)];
        assert!(!hits_are_inscribed_polygon(&hits));
    }
}
